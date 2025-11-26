"""
Practical test scenarios for oplog - inspired by PBQA usage patterns.

These scenarios demonstrate how oplog captures traces from typical ML/NLP operations.
Run with: python -m pytest tests/test_scenarios.py -v
"""

import json
import tempfile
from pathlib import Path

import oplog
from oplog import db, export


# =============================================================================
# Scenario 1: Simple Structured Generation
# Simulates an LLM call that extracts structured data from text
# =============================================================================


def test_structured_extraction():
    """Log a structured extraction operation (e.g., entity extraction, classification)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "extraction.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        # Simulate extracting structured data from user input
        user_input = "My name is John and I work at Acme Corp as a software engineer."
        extracted = {
            "name": "John",
            "company": "Acme Corp",
            "role": "software engineer",
        }

        op_id = (
            oplog.op("extract_entities")
            .model("llama-3.2-3b")
            .input(text=user_input)
            .output(**extracted)
            .meta(schema="PersonInfo", temperature=0.0)
            .save()
        )

        # Verify we can query it back
        results = db.query(operation="extract_entities")
        assert len(results) == 1
        assert results[0].inputs["text"] == user_input
        assert results[0].outputs["name"] == "John"
        assert results[0].meta["schema"] == "PersonInfo"


# =============================================================================
# Scenario 2: Two-Step Tool Use Chain
# Simulates PBQA's weather example: extract query -> call tool -> generate answer
# =============================================================================


def test_tool_use_chain():
    """Log a multi-step tool use chain within a run context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "tooluse.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        user_query = "What's the weather like in Amsterdam tomorrow?"

        with oplog.run(pipeline="weather_agent") as r:
            # Step 1: Extract structured weather query from natural language
            weather_query = {
                "latitude": 52.37,
                "longitude": 4.89,
                "time": "2025-01-16 12:00",
            }
            oplog.op("extract_weather_query").model("llama-3.2-3b").input(
                query=user_query
            ).output(**weather_query).meta(pattern="weather").save()

            # Step 2: (Tool call happens here - not logged, it's deterministic)
            tool_result = {
                "temperature": "8 C",
                "precipitation_probability": "45%",
                "cloud_cover": "60%",
            }

            # Step 3: Generate natural language answer from tool result
            answer = {
                "thought": "The weather shows mild temperature with moderate cloud cover and some chance of rain.",
                "answer": "Tomorrow in Amsterdam will be cool at around 8°C with partly cloudy skies. There's about a 45% chance of rain, so you might want to bring an umbrella.",
            }
            oplog.op("generate_answer").model("llama-3.2-3b").input(
                query=user_query, tool_result=tool_result
            ).output(**answer).meta(pattern="thinkandanswer").save()

        # Verify the run captured both operations in sequence
        results = db.query(run_id=r.id)
        assert len(results) == 2

        # Sort by seq to ensure consistent ordering
        results = sorted(results, key=lambda r: r.seq)
        assert results[0].seq == 0
        assert results[0].operation == "extract_weather_query"
        assert results[1].seq == 1
        assert results[1].operation == "generate_answer"

        # Both should have the run-level metadata
        assert results[0].meta["pipeline"] == "weather_agent"
        assert results[1].meta["pipeline"] == "weather_agent"


# =============================================================================
# Scenario 3: Conversation History
# Simulates logging a multi-turn conversation
# =============================================================================


def test_conversation_logging():
    """Log multiple turns of a conversation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "conversation.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        conversation = [
            ("Hey there!", "Hello! How can I help you today?"),
            ("What's 1 + 1?", "1 + 1 equals 2."),
            ("What's the capital of France?", "The capital of France is Paris."),
        ]

        with oplog.run(conversation_id="chat_001", user="alice") as r:
            for user_msg, assistant_msg in conversation:
                oplog.op("chat").model("llama-3.2-3b").input(message=user_msg).output(
                    reply=assistant_msg
                ).save()

        # Verify all turns were logged
        results = db.query(run_id=r.id)
        assert len(results) == 3

        # Sort by seq and verify sequence numbers
        results = sorted(results, key=lambda r: r.seq)
        for i, result in enumerate(results):
            assert result.seq == i

        # Verify run metadata propagated
        assert all(r.meta["user"] == "alice" for r in results)


# =============================================================================
# Scenario 4: Reranking for Tool Selection
# Simulates using a reranking model to select the best tool
# =============================================================================


def test_reranking_tool_selection():
    """Log reranking operations for tool selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "rerank.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        user_query = "What's 10 + 20?"
        tools = [
            "Search tool: Find information on the web",
            "Weather tool: Get weather forecasts",
            "Python REPL: Execute Python code for calculations",
            "Calendar tool: Manage events and schedules",
        ]

        # Simulate reranking result - Python REPL should be selected
        rerank_result = [
            {"index": 2, "tool": tools[2], "score": 0.95},
            {"index": 0, "tool": tools[0], "score": 0.3},
            {"index": 3, "tool": tools[3], "score": 0.1},
            {"index": 1, "tool": tools[1], "score": 0.05},
        ]

        op_id = (
            oplog.op("rerank_tools")
            .model("bge-reranker-v2")
            .input(query=user_query, documents=tools)
            .output(rankings=rerank_result)
            .meta(top_k=1, selected_tool="Python REPL")
            .save()
        )

        results = db.query(operation="rerank_tools")
        assert len(results) == 1
        # The top-ranked tool should be Python REPL
        assert "Python REPL" in results[0].outputs["rankings"][0]["tool"]
        assert results[0].outputs["rankings"][0]["score"] == 0.95


# =============================================================================
# Scenario 5: Classification with Flagging
# Simulates classifying items and flagging interesting ones for review
# =============================================================================


def test_classification_with_flagging():
    """Log classification operations and flag edge cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "classification.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        samples = [
            ("I love this product!", "positive", 0.98),
            ("This is the worst experience ever.", "negative", 0.95),
            ("It's okay I guess.", "neutral", 0.52),  # Low confidence - flag!
            ("The quality is meh but shipping was fast.", "mixed", 0.61),  # Ambiguous
        ]

        low_confidence_ids = []

        with oplog.run(task="sentiment_analysis", batch_id="batch_001"):
            for text, label, confidence in samples:
                op_id = (
                    oplog.op("classify_sentiment")
                    .model("distilbert-sentiment")
                    .input(text=text)
                    .output(label=label, confidence=confidence)
                    .save()
                )

                # Flag low-confidence predictions for human review
                if confidence < 0.7:
                    low_confidence_ids.append(op_id)

        # Flag the low-confidence ones
        db.flag(reason="low_confidence", ids=low_confidence_ids)

        # Query flagged items
        flagged = db.query(flagged_for="low_confidence")
        assert len(flagged) == 2
        assert all(r.outputs["confidence"] < 0.7 for r in flagged)


# =============================================================================
# Scenario 6: Export for Training
# Demonstrates exporting logged operations for model training
# =============================================================================


def test_export_for_training():
    """Export logged operations in various formats for training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "training.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        # Log some training-worthy examples
        examples = [
            ("Translate: Hello", "Hola"),
            ("Translate: Goodbye", "Adiós"),
            ("Translate: Thank you", "Gracias"),
        ]

        for input_text, output_text in examples:
            oplog.op("translate").model("nllb-200").input(prompt=input_text).output(
                translation=output_text
            ).meta(source_lang="en", target_lang="es").save()

        # Query the records
        records = db.query(operation="translate")

        # Export to JSONL
        jsonl_path = Path(tmpdir) / "training.jsonl"
        export.to_jsonl(records, jsonl_path)

        # Verify JSONL export
        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

        first = json.loads(lines[0])
        assert "inputs" in first
        assert "outputs" in first
        assert first["operation"] == "translate"

        # Export to DataFrame
        df = export.to_dataframe(records)
        assert len(df) == 3
        assert "inputs" in df.columns
        assert "outputs" in df.columns


# =============================================================================
# Scenario 7: A/B Testing Different Models
# Log operations from different model variants for comparison
# =============================================================================


def test_ab_model_comparison():
    """Log operations from different models for A/B comparison."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "ab_test.db"
        oplog.configure("test_project", f"sqlite:///{db_path}")

        prompt = "Explain quantum computing in simple terms."

        # Model A response
        with oplog.run(experiment="explain_quantum", variant="A"):
            oplog.op("generate").model("llama-3.2-3b").input(prompt=prompt).output(
                text="Quantum computing uses quantum bits that can be 0 and 1 at the same time..."
            ).meta(tokens=45, latency_ms=120).save()

        # Model B response
        with oplog.run(experiment="explain_quantum", variant="B"):
            oplog.op("generate").model("llama-3.2-8b").input(prompt=prompt).output(
                text="Imagine a coin spinning in the air - that's like a quantum bit..."
            ).meta(tokens=52, latency_ms=280).save()

        # Query by model for comparison
        model_a = db.query(model="llama-3.2-3b")
        model_b = db.query(model="llama-3.2-8b")

        assert len(model_a) == 1
        assert len(model_b) == 1
        assert model_a[0].meta["variant"] == "A"
        assert model_b[0].meta["variant"] == "B"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
