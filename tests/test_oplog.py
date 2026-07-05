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

    def test_run_level_metadata(self, configured_oplog):
        """Run-level metadata lives on the run row, not on operations."""
        with run(strategy="methodA", experiment_id="exp123") as r:
            op("test").save()
            op("test").meta(latency_ms=42).save()

        records = db.query(run_id=r.id)
        assert len(records) == 2

        # Run meta is NOT duplicated onto operations
        for rec in records:
            assert rec.meta is None or "strategy" not in rec.meta

        # It lives once, on the run row
        run_row = db.get_run(r.id)
        assert run_row is not None
        assert run_row.meta["strategy"] == "methodA"
        assert run_row.meta["experiment_id"] == "exp123"

        # Operation-level meta stays on the operation
        latency_record = [x for x in records if x.meta and x.meta.get("latency_ms")][0]
        assert latency_record.meta["latency_ms"] == 42

    def test_run_row_created_on_entry(self, configured_oplog):
        """The run row exists as soon as the run starts — a partial run is
        never lost."""
        with run(stage="early") as r:
            row = db.get_run(r.id)
            assert row is not None
            assert row.meta["stage"] == "early"

    def test_run_add_meta_writes_through(self, configured_oplog):
        """add_meta() appends data discovered mid-run and persists it."""
        with run(kind="turn") as r:
            op("step").save()
            r.add_meta(outcome="success")
            row = db.get_run(r.id)
            assert row.meta == {"kind": "turn", "outcome": "success"}


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
        """Flagging a run flags the run row itself, not its operations."""
        with run() as r:
            op("op1").save()
            op("op2").save()
            op("op3").save()

        count = db.flag(run_id=r.id, reason="review", note="odd resolution")
        assert count == 1

        run_row = db.get_run(r.id)
        assert run_row.flagged_for == "review"
        assert run_row.flag_note == "odd resolution"
        assert run_row.flagged_at is not None

        # Operations stay unflagged — no duplication
        for rec in db.query(run_id=r.id):
            assert rec.flagged_for is None

        # And run queries can filter on the flag
        flagged = db.runs(flagged_for="review")
        assert [x.id for x in flagged] == [r.id]

        assert db.unflag(run_id=r.id) == 1
        assert db.get_run(r.id).flagged_for is None

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


class TestSQLiteAutoCreate:
    """Test SQLite database auto-creation."""

    def test_creates_db_in_new_directory(self, tmp_path):
        """Test that SQLite creates db file and parent directories."""
        # Create a path with nested directories that don't exist
        db_path = tmp_path / "nested" / "dirs" / "traces.db"
        assert not db_path.parent.exists()

        configure(project="test", backend=f"sqlite:///{db_path}")
        op("test").save()

        # Directory and file should now exist
        assert db_path.parent.exists()
        assert db_path.exists()