"""Backend protocol and implementations for oplog."""

from datetime import datetime
from typing import List, Optional, Protocol, runtime_checkable

from oplog.models import Operation


@runtime_checkable
class Backend(Protocol):
    """Protocol defining the backend interface for operation storage."""

    def save(self, operation: Operation) -> str:
        """Persist an operation. Returns the operation ID."""
        ...

    def query(
        self,
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
    ) -> List[Operation]:
        """Query operations with filters."""
        ...

    def flag(
        self,
        reason: str,
        ids: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> int:
        """Flag operations. Returns count of affected rows."""
        ...

    def unflag(
        self,
        ids: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """Remove flags. Returns count of affected rows."""
        ...

    def init_schema(self) -> None:
        """Create tables if they don't exist."""
        ...