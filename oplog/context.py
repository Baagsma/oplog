"""Thread-local run context management for oplog."""

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from oplog.utils import generate_ulid


# Thread-local storage for run context
_local = threading.local()


class RunContext:
    """Context for grouping related operations within a run."""

    def __init__(self, run_id: str):
        """Initialize a run context.

        Args:
            run_id: The unique identifier for this run.
        """
        self._id = run_id
        self._seq = 0

    @property
    def id(self) -> str:
        """Get the run ID."""
        return self._id

    @property
    def seq(self) -> int:
        """Get the current sequence number."""
        return self._seq

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
def run_context(run_id: Optional[str] = None) -> Iterator[RunContext]:
    """Context manager for grouping related operations.

    Args:
        run_id: Optional explicit run ID. If not provided, a ULID is generated.

    Yields:
        The RunContext for this run.

    Note:
        Nested runs are not supported. Inner run() calls will shadow outer ones.
    """
    if run_id is None:
        run_id = generate_ulid()

    ctx = RunContext(run_id)
    _local.run_context = ctx
    try:
        yield ctx
    finally:
        _local.run_context = None