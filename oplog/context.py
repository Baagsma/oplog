"""Thread-local run context management for oplog."""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from oplog.utils import generate_ulid


# Thread-local storage for run context
_local = threading.local()


class RunContext:
    """Context for grouping related operations within a run."""

    def __init__(self, run_id: str, meta: Optional[Dict[str, Any]] = None):
        """Initialize a run context.

        Args:
            run_id: The unique identifier for this run.
            meta: Optional metadata to attach to all operations in this run.
        """
        self._id = run_id
        self._seq = 0
        self._meta: Dict[str, Any] = meta or {}

    @property
    def id(self) -> str:
        """Get the run ID."""
        return self._id

    @property
    def seq(self) -> int:
        """Get the current sequence number."""
        return self._seq

    def get_meta(self) -> Dict[str, Any]:
        """Get the run-level metadata.

        Returns:
            A copy of the run metadata dictionary.
        """
        return self._meta.copy()

    def next_seq(self) -> int:
        """Get the next sequence number and increment the counter.

        Returns:
            The current sequence number (before incrementing).
        """
        current = self._seq
        self._seq += 1
        return current


def get_current_run() -> Optional[RunContext]:
    """Get the current run context, if any.

    Returns:
        The current RunContext, or None if not inside a run.
    """
    return getattr(_local, "run_context", None)


@contextmanager
def run_context(
    run_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Iterator[RunContext]:
    """Context manager for grouping related operations.

    Args:
        run_id: Optional explicit run ID. If not provided, a ULID is generated.
        meta: Optional metadata to attach to all operations in this run.

    Yields:
        The RunContext for this run.

    Note:
        Nested runs are not supported. Inner run() calls will shadow outer ones.
    """
    if run_id is None:
        run_id = generate_ulid()

    ctx = RunContext(run_id, meta=meta)
    _local.run_context = ctx
    try:
        yield ctx
    finally:
        _local.run_context = None