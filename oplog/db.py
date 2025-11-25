"""Database operations for querying and flagging operations."""

from datetime import datetime
from typing import List, Optional

from oplog.config import get_tracer
from oplog.models import Record


def query(
    project: Optional[str] = None,
    operation: Optional[str] = None,
    model: Optional[str] = None,
    run_id: Optional[str] = None,
    flagged_for: Optional[str] = None,
    tags: Optional[List[str]] = None,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[Record]:
    """Query stored operations.

    Args:
        project: Filter by project namespace.
        operation: Filter by operation type.
        model: Filter by model name.
        run_id: Filter by run ID.
        flagged_for: Filter by flag reason.
        tags: Filter by tags (AND logic - must have all specified tags).
        after: Operations created after this time.
        before: Operations created before this time.
        limit: Maximum records to return.
        offset: Skip this many records (for pagination).

    Returns:
        List of matching Record objects.

    Raises:
        RuntimeError: If oplog is not configured.
    """
    tracer = get_tracer()
    return tracer.backend.query(
        project=project,
        operation=operation,
        model=model,
        run_id=run_id,
        flagged_for=flagged_for,
        tags=tags,
        after=after,
        before=before,
        limit=limit,
        offset=offset,
    )


def flag(
    reason: str,
    ids: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    note: Optional[str] = None,
) -> int:
    """Flag operations for later processing.

    Args:
        reason: Flag category (e.g., 'training', 'review', 'bug', 'exclude').
        ids: Specific operation IDs to flag.
        run_id: Flag all operations in this run.
        note: Human-readable note.

    Returns:
        Count of flagged operations.

    Raises:
        ValueError: If neither ids nor run_id is provided.
        RuntimeError: If oplog is not configured.

    Note:
        Only one of ids or run_id should be provided.
    """
    tracer = get_tracer()
    return tracer.backend.flag(
        reason=reason,
        ids=ids,
        run_id=run_id,
        note=note,
    )


def unflag(
    ids: Optional[List[str]] = None,
    run_id: Optional[str] = None,
) -> int:
    """Remove flags from operations.

    Args:
        ids: Specific operation IDs to unflag.
        run_id: Unflag all operations in this run.

    Returns:
        Count of unflagged operations.

    Raises:
        ValueError: If neither ids nor run_id is provided.
        RuntimeError: If oplog is not configured.

    Note:
        Only one of ids or run_id should be provided.
    """
    tracer = get_tracer()
    return tracer.backend.unflag(
        ids=ids,
        run_id=run_id,
    )