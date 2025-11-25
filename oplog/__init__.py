"""oplog - ML/NLP operation trace capture library.

A minimal, project-agnostic library for capturing ML/NLP operation traces
for later training data export.

Example usage:

    from oplog import configure, op, run, db, export

    # Configure once at startup
    configure(project="my_project", backend="sqlite:///traces.db")

    # Log standalone operations
    op("classify").model("setfit-intent").input(text="hello").output(label="greeting").save()

    # Log grouped operations within a run
    with run() as r:
        op("retrieve").input(query="...").output(candidates=[...]).save()
        op("rerank").input(query="...").output(ranked=[...]).save()

    # Query and flag
    records = db.query(operation="rerank")
    db.flag(run_id=r.id, reason="training")

    # Export
    export.to_jsonl(records, "output.jsonl")
"""

from oplog.config import configure, Tracer, op, run
from oplog.models import Operation, Record
from oplog import db
from oplog import export

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "configure",
    "Tracer",
    # Operation building
    "op",
    "run",
    # Models
    "Operation",
    "Record",
    # Submodules
    "db",
    "export",
]