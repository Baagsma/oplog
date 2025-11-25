"""Fluent builder for operation traces."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from oplog.context import get_current_run
from oplog.models import Operation
from oplog.utils import generate_ulid

if TYPE_CHECKING:
    from oplog.backends import Backend


class OpBuilder:
    """Fluent builder for creating operation traces.

    All methods return self for chaining (except save()).
    """

    def __init__(
        self,
        operation: str,
        project: str,
        backend: "Backend",
    ):
        """Initialize an operation builder.

        Args:
            operation: The operation type (e.g., 'rerank', 'nli', 'classify').
            project: The project namespace.
            backend: The backend to save to.
        """
        self._operation = operation
        self._project = project
        self._backend = backend
        self._model_name: Optional[str] = None
        self._inputs: Dict[str, Any] = {}
        self._outputs: Dict[str, Any] = {}
        self._meta: Dict[str, Any] = {}
        self._tags: List[str] = []

    def model(self, name: str) -> "OpBuilder":
        """Set the model identifier used for this operation.

        Args:
            name: Model identifier (e.g., 'bge-reranker-base').

        Returns:
            self for chaining.
        """
        self._model_name = name
        return self

    def input(self, **kwargs: Any) -> "OpBuilder":
        """Add input data. Can be called multiple times; kwargs are merged.

        Args:
            **kwargs: Input key-value pairs (stored as JSON).

        Returns:
            self for chaining.
        """
        self._inputs.update(kwargs)
        return self

    def output(self, **kwargs: Any) -> "OpBuilder":
        """Add output data. Can be called multiple times; kwargs are merged.

        Args:
            **kwargs: Output key-value pairs (stored as JSON).

        Returns:
            self for chaining.
        """
        self._outputs.update(kwargs)
        return self

    def meta(self, **kwargs: Any) -> "OpBuilder":
        """Add metadata. Can be called multiple times; kwargs are merged.

        Args:
            **kwargs: Metadata key-value pairs (e.g., latency_ms, tokens).

        Returns:
            self for chaining.
        """
        self._meta.update(kwargs)
        return self

    def tags(self, *tags: str) -> "OpBuilder":
        """Add categorical tags for filtering.

        Args:
            *tags: Tag strings to add.

        Returns:
            self for chaining.
        """
        self._tags.extend(tags)
        return self

    def save(self) -> str:
        """Persist the operation and return its ID.

        Returns:
            The operation ID (ULID).
        """
        run_ctx = get_current_run()

        # Merge run-level metadata with operation-level metadata
        # Operation metadata overrides run metadata on conflicts
        merged_meta = {}
        if run_ctx:
            merged_meta.update(run_ctx.get_meta())
        if self._meta:
            merged_meta.update(self._meta)

        operation = Operation(
            id=generate_ulid(),
            project=self._project,
            run_id=run_ctx.id if run_ctx else None,
            seq=run_ctx.next_seq() if run_ctx else None,
            operation=self._operation,
            model=self._model_name,
            inputs=self._inputs or None,
            outputs=self._outputs or None,
            meta=merged_meta or None,
            tags=self._tags,
            created_at=datetime.now(timezone.utc),
        )

        self._backend.save(operation)
        return operation.id