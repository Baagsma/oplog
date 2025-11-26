"""
Real integration tests for oplog with llama.cpp LLM and reranker servers.

Requires:
- LLM server at localhost:8080 (Qwen 3 30B)
- Reranker server at localhost:8090 (BGE-reranker v2)

Run with: python -m pytest tests/test_integration.py -v -s
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel

import oplog
from oplog import db, export

# Persistent database for inspection
DB_PATH = Path(__file__).parent.parent / "integration_traces.db"
EXPORT_DIR = Path(__file__).parent.parent / "exports"


# =============================================================================
# LLM and Reranker Client Helpers
# =============================================================================

LLM_HOST = "localhost"
LLM_PORT = 8080
RERANK_HOST = "localhost"
RERANK_PORT = 8090


def llm_generate(
    messages: List[Dict[str, str]],
    json_schema: Optional[Dict] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Call the LLM server and return the response with timing info."""
    url = f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions"

    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["<|im_end|>"],
    }
    if json_schema:
        data["json_schema"] = json_schema

    start = time.time()
    response = requests.post(url, json=data)
    latency_ms = (time.time() - start) * 1000

    result = response.json()

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    return {
        "content": content,
        "parsed": json.loads(content) if json_schema else content,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def rerank(query: str, documents: List[str], top_k: int = 3) -> tuple:
    """Call the reranker server and return ranked results."""
    url = f"http://{RERANK_HOST}:{RERANK_PORT}/v1/rerank"

    start = time.time()
    response = requests.post(
        url,
        json={"query": query, "documents": documents},
    )
    latency_ms = (time.time() - start) * 1000

    result = response.json()

    rankings = []
    for r in sorted(
        result["results"], key=lambda x: x["relevance_score"], reverse=True
    )[:top_k]:
        rankings.append(
            {
                "index": r["index"],
                "document": documents[r["index"]],
                "score": r["relevance_score"],
            }
        )

    return rankings, latency_ms


# =============================================================================
# Integration Tests
# =============================================================================


def test_sentiment_classification():
    """Test sentiment classification with structured output and oplog capture."""
    oplog.configure("integration_tests", f"sqlite:///{DB_PATH}")

    texts = [
        "I absolutely love this product! Best purchase ever!",
        "This is terrible. Complete waste of money.",
        "It's okay, nothing special but works fine.",
    ]

    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["sentiment", "confidence", "reasoning"],
    }

    with oplog.run(task="sentiment_classification", batch_size=len(texts)) as r:
        for text in texts:
            messages = [
                {
                    "role": "system",
                    "content": "You are a sentiment classifier. Analyze the sentiment of the given text and respond with JSON containing sentiment (positive/negative/neutral), confidence (0-1), and brief reasoning.",
                },
                {"role": "user", "content": text},
            ]

            response = llm_generate(messages, json_schema=schema)

            oplog.op("classify_sentiment").model("qwen3-30b").input(
                text=text, system_prompt=messages[0]["content"]
            ).output(**response["parsed"]).meta(
                latency_ms=response["latency_ms"],
                prompt_tokens=response["prompt_tokens"],
                completion_tokens=response["completion_tokens"],
            ).save()

            print(f"\n[{response['parsed']['sentiment']}] {text[:50]}...")
            print(f"  Confidence: {response['parsed']['confidence']}")
            print(f"  Reasoning: {response['parsed']['reasoning']}")

    # Verify logging
    records = db.query(run_id=r.id)
    assert len(records) == 3

    for record in records:
        assert record.outputs["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= record.outputs["confidence"] <= 1
        assert record.meta["latency_ms"] > 0


def test_entity_extraction():
    """Test entity extraction with structured output."""
    oplog.configure("integration_tests", f"sqlite:///{DB_PATH}")

    text = "John Smith works at Google in Mountain View. He met Sarah Johnson from Microsoft last Tuesday."

    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["PERSON", "ORG", "LOCATION", "DATE"],
                        },
                        "value": {"type": "string"},
                    },
                    "required": ["type", "value"],
                },
            },
        },
        "required": ["entities"],
    }

    messages = [
        {
            "role": "system",
            "content": "Extract named entities from the text. Return JSON with an 'entities' array containing objects with 'type' (PERSON/ORG/LOCATION/DATE) and 'value' fields.",
        },
        {"role": "user", "content": text},
    ]

    response = llm_generate(messages, json_schema=schema)

    with oplog.run(task="entity_extraction") as r:
        oplog.op("extract_entities").model("qwen3-30b").input(text=text).output(
            **response["parsed"]
        ).meta(
            latency_ms=response["latency_ms"],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
        ).save()

    print(f"\nExtracted entities from: {text}")
    for entity in response["parsed"]["entities"]:
        print(f"  [{entity['type']}] {entity['value']}")

    # Verify
    records = db.query(run_id=r.id)
    assert len(records) == 1
    assert len(records[0].outputs["entities"]) > 0


def test_reranking_tool_selection():
    """Test reranking for tool selection with oplog capture."""
    oplog.configure("integration_tests", f"sqlite:///{DB_PATH}")

    tools = [
        "Calculator: Perform mathematical calculations like addition, multiplication, etc.",
        "Weather API: Get current weather and forecasts for any location.",
        "Web Search: Search the internet for information on any topic.",
        "Calendar: Manage events, schedule meetings, and check availability.",
        "Code Executor: Run Python code and return the results.",
        "Translation: Translate text between different languages.",
    ]

    queries = [
        "What's 15 * 23 + 47?",
        "Will it rain in Amsterdam tomorrow?",
        "How do I make a for loop in Python?",
    ]

    with oplog.run(task="tool_selection") as r:
        for query in queries:
            rankings, latency_ms = rerank(query, tools, top_k=3)

            oplog.op("rerank_tools").model("bge-reranker-v2-m3").input(
                query=query, tools=tools
            ).output(rankings=rankings, selected_tool=rankings[0]["document"]).meta(
                latency_ms=latency_ms, top_k=3
            ).save()

            print(f"\nQuery: {query}")
            print(
                f"  Selected: {rankings[0]['document'][:50]}... (score: {rankings[0]['score']:.3f})"
            )

    # Verify
    records = db.query(run_id=r.id)
    assert len(records) == 3


def test_multi_step_qa_chain():
    """Test a multi-step Q&A chain: rerank context -> generate answer."""
    oplog.configure("integration_tests", f"sqlite:///{DB_PATH}")

    # Simulated knowledge base chunks
    knowledge_chunks = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.",
        "The Great Wall of China is over 21,000 kilometers long and was built over many centuries.",
        "Mount Everest is the highest mountain on Earth at 8,849 meters above sea level.",
        "The Amazon River is the largest river by volume and flows through South America.",
        "Tokyo is the capital of Japan and has a population of over 13 million people.",
    ]

    question = "How tall is the Eiffel Tower and when was it built?"

    with oplog.run(pipeline="rag_qa", question=question) as r:
        # Step 1: Rerank to find relevant context
        rankings, rerank_latency = rerank(question, knowledge_chunks, top_k=2)

        oplog.op("retrieve_context").model("bge-reranker-v2-m3").input(
            query=question, chunks=knowledge_chunks
        ).output(
            rankings=rankings,
            top_context=rankings[0]["document"],
        ).meta(
            latency_ms=rerank_latency
        ).save()

        print(f"\nQuestion: {question}")
        print(f"Retrieved context: {rankings[0]['document']}")

        # Step 2: Generate answer using retrieved context
        context = "\n".join([r["document"] for r in rankings])

        messages = [
            {
                "role": "system",
                "content": "Answer the question based only on the provided context. Be concise.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        response = llm_generate(messages, temperature=0.0)

        oplog.op("generate_answer").model("qwen3-30b").input(
            question=question, context=context
        ).output(answer=response["content"]).meta(
            latency_ms=response["latency_ms"],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
        ).save()

        print(f"Answer: {response['content']}")

    # Verify the chain
    records = db.query(run_id=r.id)
    records = sorted(records, key=lambda x: x.seq)

    assert len(records) == 2
    assert records[0].operation == "retrieve_context"
    assert records[1].operation == "generate_answer"
    assert records[0].meta["pipeline"] == "rag_qa"
    assert records[1].meta["pipeline"] == "rag_qa"


def test_translation_with_flagging():
    """Test translation with automatic flagging of low-quality results."""
    oplog.configure("integration_tests", f"sqlite:///{DB_PATH}")

    schema = {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["translation", "confidence"],
    }

    texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Supercalifragilisticexpialidocious",  # Challenging word
    ]

    low_confidence_ids = []

    with oplog.run(task="en_to_es_translation") as r:
        for text in texts:
            messages = [
                {
                    "role": "system",
                    "content": "Translate the text to Spanish. Return JSON with 'translation' and 'confidence' (0-1 based on translation quality/certainty).",
                },
                {"role": "user", "content": text},
            ]

            response = llm_generate(messages, json_schema=schema)

            op_id = (
                oplog.op("translate")
                .model("qwen3-30b")
                .input(text=text, source_lang="en", target_lang="es")
                .output(**response["parsed"])
                .meta(latency_ms=response["latency_ms"])
                .save()
            )

            print(f"\n{text}")
            print(
                f"  → {response['parsed']['translation']} (conf: {response['parsed']['confidence']})"
            )

            # Flag low-confidence translations for human review
            if response["parsed"]["confidence"] < 0.8:
                low_confidence_ids.append(op_id)

    # Flag for review
    if low_confidence_ids:
        db.flag(reason="low_confidence_translation", ids=low_confidence_ids)
        print(f"\nFlagged {len(low_confidence_ids)} translations for review")

    # Verify
    records = db.query(run_id=r.id)
    assert len(records) == 3


def test_export_training_data():
    """Test exporting logged operations for training data."""
    oplog.configure("integration_tests", f"sqlite:///{DB_PATH}")

    schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["tech", "sports", "politics", "entertainment"],
            },
        },
        "required": ["category"],
    }

    # Generate some classification data
    headlines = [
        "Apple announces new iPhone with revolutionary AI features",
        "Lakers win championship in overtime thriller",
        "Senate passes new infrastructure bill",
        "Marvel releases trailer for upcoming superhero film",
    ]

    with oplog.run(task="news_classification") as r:
        for headline in headlines:
            messages = [
                {
                    "role": "system",
                    "content": "Classify the news headline into one category: tech, sports, politics, or entertainment.",
                },
                {"role": "user", "content": headline},
            ]

            response = llm_generate(messages, json_schema=schema)

            oplog.op("classify_news").model("qwen3-30b").input(
                headline=headline
            ).output(**response["parsed"]).meta(
                latency_ms=response["latency_ms"]
            ).save()

    # Query and export
    records = db.query(run_id=r.id)

    # Export to JSONL
    EXPORT_DIR.mkdir(exist_ok=True)
    jsonl_path = EXPORT_DIR / "training.jsonl"
    export.to_jsonl(records, jsonl_path)

    print(f"\nExported {len(records)} records to {jsonl_path}")
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            print(
                f"  {data['inputs']['headline'][:40]}... → {data['outputs']['category']}"
            )

    # Export to DataFrame
    df = export.to_dataframe(records)
    print(f"\nDataFrame shape: {df.shape}")
    print(df[["inputs", "outputs"]].to_string())

    assert len(records) == 4


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
