# oplogger

A minimal library for capturing ML/NLP operation traces for later training data export.

## Installation

```bash
pip install oplogger

# With PostgreSQL support
pip install oplogger[postgres]

# With export features (pandas, HuggingFace datasets)
pip install oplogger[export]
```

> **Note**: The package is installed as `oplogger` but imported as `oplog`.

## Quick Start

```python
from oplog import configure, op, run, db, export

# Configure once at startup
configure(project="my_project", backend="sqlite:///traces.db")

# Log standalone operations
op("classify") \
    .model("setfit-intent") \
    .input(text="hello world") \
    .output(label="greeting", score=0.95) \
    .save()

# Log grouped operations within a run (with run-level metadata for A/B testing)
with run(strategy="rerank_v2", experiment="exp_042") as r:
    op("retrieve") \
        .model("bge-m3") \
        .input(query="capital of France?", k=10) \
        .output(candidates=["Paris is the capital..."]) \
        .save()

    op("rerank") \
        .model("bge-reranker-base") \
        .input(query="capital of France?", candidates=[...]) \
        .output(ranked=["Paris is the capital..."], scores=[0.94]) \
        .meta(latency_ms=42) \
        .save()
    # Both ops get meta={"strategy": "rerank_v2", "experiment": "exp_042", ...}

# Flag for training
db.flag(run_id=r.id, reason="training", note="clean example")

# Query and export
records = db.query(operation="rerank", flagged_for="training")
export.to_jsonl(records, "training_data.jsonl")
```

## API Reference

### Configuration

```python
configure(project="name", backend="sqlite:///traces.db")
```

**Backend formats:**
- SQLite: `sqlite:///path/to/traces.db` (auto-creates file and parent directories)
- PostgreSQL: `postgresql://user:pass@host:port/dbname`

### Operations

```python
op("operation_type")        # Start building an operation
    .model("model-name")    # Model identifier
    .input(**kwargs)        # Input data (JSON)
    .output(**kwargs)       # Output data (JSON)
    .meta(**kwargs)         # Metadata (latency, tokens, etc.)
    .tags("tag1", "tag2")   # Categorical tags
    .save()                 # Persist and return operation ID
```

### Runs

```python
with run() as r:            # Auto-generated run ID
    op(...).save()          # seq=0
    op(...).save()          # seq=1
    print(r.id)             # Access run ID

with run("custom-id"):      # Explicit run ID
    ...

# Run-level metadata (propagates to all operations in the run)
with run(strategy="methodA", experiment_id="exp123") as r:
    op("test").save()                      # meta={"strategy": "methodA", "experiment_id": "exp123"}
    op("test").meta(latency_ms=42).save()  # meta includes both run + op metadata
```

Run metadata is merged with operation metadata. Operation-level values override run-level on conflicts.

### Database Operations

```python
# Query
records = db.query(
    operation="rerank",     # Filter by operation type
    model="model-name",     # Filter by model
    run_id="...",           # Filter by run
    flagged_for="training", # Filter by flag
    tags=["tag1", "tag2"],  # Filter by tags (AND logic)
    limit=100,              # Pagination
    offset=0,
)

# Flag
db.flag(ids=[...], reason="training", note="optional note")
db.flag(run_id="...", reason="review")

# Unflag
db.unflag(ids=[...])
db.unflag(run_id="...")
```

### Export

```python
# JSONL
export.to_jsonl(records, "output.jsonl")

# CSV
export.to_csv(records, "output.csv")

# pandas DataFrame
df = export.to_dataframe(records)

# HuggingFace Dataset
dataset = export.to_dataset(records)

# Field selection (dot notation for nested fields)
export.to_jsonl(records, "output.jsonl", fields=["inputs.query", "outputs.score"])
```

## Multi-Tracer Usage

For multiple projects or explicit control:

```python
from oplog import Tracer

tracer = Tracer(project="my_project", backend="sqlite:///traces.db")

tracer.op("classify").input(...).save()

with tracer.run() as r:
    tracer.op("rerank").input(...).save()
```

## License

MIT