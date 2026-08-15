"""Run context management for oplog.

The ambient run lives in a contextvars.ContextVar, not threading.local:
contextvars follow execution across asyncio tasks AND into worker threads
(asyncio.to_thread copies the context), so operations recorded by a
blocking call that an async caller moved off the event loop still attach
to the caller's run. threading.local silently orphaned exactly those ops
(run_id=None) the moment a caller introduced to_thread.
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from oplog.utils import generate_ulid


_run_context: "contextvars.ContextVar[Optional[RunContext]]" = contextvars.ContextVar(
    "oplog_run_context", default=None
)


class RunContext:
    """Context for grouping related operations within a run."""

    def __init__(self, run_id: str, meta: Optional[Dict[str, Any]] = None):
        """Initialize a run context.

        Args:
            run_id: The unique identifier for this run.
            meta: Optional run-level metadata (stored on the run row).
        """
        self._id = run_id
        self._seq = 0
        self._meta: Dict[str, Any] = meta or {}
        # Set by the tracer once the run row is persisted; lets add_meta()
        # write through so a crash never loses accreted run data.
        self._persist_fn = None

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

    def add_meta(self, **meta: Any) -> "RunContext":
        """Merge additional metadata onto this run and persist immediately.

        The run row is created when the run starts; data that only becomes
        known mid-run (a classification, an outcome summary) is appended
        here rather than serialized post-hoc, so partial runs keep whatever
        was known before an interruption.
        """
        self._meta.update(meta)
        if self._persist_fn is not None:
            self._persist_fn(self)
        return self


def get_current_run() -> Optional[RunContext]:
    """Get the current run context, if any.

    Returns:
        The current RunContext, or None if not inside a run.
    """
    return _run_context.get()


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
    token = _run_context.set(ctx)
    try:
        yield ctx
    finally:
        _run_context.reset(token)