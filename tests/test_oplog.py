"""Minimal smoke tests for oplog."""

import json
import tempfile
from pathlib import Path

import pytest

from oplog import configure, op, run, db, export


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield f"sqlite:///{db_path}"
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def configured_oplog(temp_db):
    """Configure oplog with a temporary database."""
    configure(project="test_project", backend=temp_db)
    yield


class TestBasicOperations:
    """Test basic op() and save() functionality."""

    def test_standalone_operation(self, configured_oplog):
        """Test logging a standalone operation."""
        op_id = (
            op("classify")
            .model("test-model")
            .input(text="hello world")
            .output(label="greeting", score=0.95)
            .meta(latency_ms=42)
            .tags("test", "greeting")
            .save()
        )

        assert op_id is not None
        assert len(op_id) == 26  # ULID length

        # Query it back
        records = db.query(operation="classify")
        assert len(records) == 1
        assert records[0].id == op_id
        assert records[0].operation == "classify"
        assert records[0].model == "test-model"
        assert records[0].inputs == {"text": "hello world"}
        assert records[0].outputs == {"label": "greeting", "score": 0.95}
        assert records[0].meta == {"latency_ms": 42}
        assert records[0].tags == ["test", "greeting"]

    def test_operation_with_run(self, configured_oplog):
        """Test logging operations within a run context."""
        with run() as r:
            op_id1 = op("retrieve").input(query="test").output(results=[]).save()
            op_id2 = op("rerank").input(query="test").output(ranked=[]).save()

        assert r.id is not None

        # Query by run_id
        records = db.query(run_id=r.id)
        assert len(records) == 2

        # Check sequence numbers (records come back in desc order by created_at)
        seqs = sorted([rec.seq for rec in records])
        assert seqs == [0, 1]

    def test_explicit_run_id(self, configured_oplog):
        """Test using an explicit run ID."""
        with run("my-custom-run-id") as r:
            op("test").save()

        assert r.id == "my-custom-run-id"

        records = db.query(run_id="my-custom-run-id")
        assert len(records) == 1


class TestFlagging:
    """Test flag/unflag functionality."""

    def test_flag_by_id(self, configured_oplog):
        """Test flagging specific operations by ID."""
        op_id = op("test").save()

        count = db.flag(ids=[op_id], reason="training", note="good example")
        assert count == 1

        records = db.query(flagged_for="training")
        assert len(records) == 1
        assert records[0].flagged_for == "training"
        assert records[0].flag_note == "good example"
        assert records[0].flagged_at is not None

    def test_flag_by_run_id(self, configured_oplog):
        """Test flagging all operations in a run."""
        with run() as r:
            op("op1").save()
            op("op2").save()
            op("op3").save()

        count = db.flag(run_id=r.id, reason="review")
        assert count == 3

        records = db.query(run_id=r.id)
        for rec in records:
            assert rec.flagged_for == "review"

    def test_unflag(self, configured_oplog):
        """Test removing flags from operations."""
        op_id = op("test").save()
        db.flag(ids=[op_id], reason="training")

        count = db.unflag(ids=[op_id])
        assert count == 1

        # Re-query to check unflagging worked
        record = db.query(operation="test")[0]
        assert record.flagged_for is None


class TestExport:
    """Test export functionality."""

    def test_to_jsonl(self, configured_oplog, tmp_path):
        """Test JSONL export."""
        op("test").input(foo="bar").output(result=42).save()

        records = db.query(operation="test")
        output_file = tmp_path / "output.jsonl"

        export.to_jsonl(records, output_file)

        with output_file.open() as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["inputs"] == {"foo": "bar"}
            assert data["outputs"] == {"result": 42}

    def test_to_jsonl_with_fields(self, configured_oplog, tmp_path):
        """Test JSONL export with field selection."""
        op("test").input(query="hello", extra="ignored").output(answer="world").save()

        records = db.query(operation="test")
        output_file = tmp_path / "output.jsonl"

        export.to_jsonl(records, output_file, fields=["inputs.query", "outputs.answer"])

        with output_file.open() as f:
            data = json.loads(f.readline())
            assert data == {"inputs.query": "hello", "outputs.answer": "world"}

    def test_to_csv(self, configured_oplog, tmp_path):
        """Test CSV export."""
        op("test").input(text="hello").save()

        records = db.query(operation="test")
        output_file = tmp_path / "output.csv"

        export.to_csv(records, output_file)

        content = output_file.read_text()
        assert "id" in content
        assert "operation" in content
        assert "test" in content

    def test_to_dataframe(self, configured_oplog):
        """Test DataFrame export."""
        pytest.importorskip("pandas")

        op("test").input(x=1).save()
        op("test").input(x=2).save()

        records = db.query(operation="test")
        df = export.to_dataframe(records)

        assert len(df) == 2
        assert "inputs" in df.columns


class TestQuery:
    """Test query functionality."""

    def test_query_by_operation(self, configured_oplog):
        """Test filtering by operation type."""
        op("classify").save()
        op("rerank").save()
        op("classify").save()

        records = db.query(operation="classify")
        assert len(records) == 2

    def test_query_by_model(self, configured_oplog):
        """Test filtering by model."""
        op("test").model("model-a").save()
        op("test").model("model-b").save()
        op("test").model("model-a").save()

        records = db.query(model="model-a")
        assert len(records) == 2

    def test_query_with_limit_offset(self, configured_oplog):
        """Test pagination."""
        for i in range(5):
            op("test").meta(index=i).save()

        records = db.query(limit=2)
        assert len(records) == 2

        records = db.query(limit=2, offset=2)
        assert len(records) == 2

    def test_query_by_tags(self, configured_oplog):
        """Test filtering by tags."""
        op("test").tags("alpha", "beta").save()
        op("test").tags("alpha").save()
        op("test").tags("gamma").save()

        # Single tag
        records = db.query(tags=["alpha"])
        assert len(records) == 2

        # Multiple tags (AND logic)
        records = db.query(tags=["alpha", "beta"])
        assert len(records) == 1