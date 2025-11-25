"""Global configuration and Tracer class for oplog."""

from contextlib import contextmanager
from typing import Iterator, Optional, Union

from oplog.backends import Backend
from oplog.backends.sql import SQLBackend
from oplog.builder import OpBuilder
from oplog.context import RunContext, run_context


# Global default tracer
_default_tracer: Optional["Tracer"] = None


class Tracer:
    """Tracer for capturing ML/NLP operation traces.

    Use this class when you need multiple tracers or explicit control.
    For simple cases, use the global configure() and op()/run() functions.
    """

    def __init__(
        self,
        project: str,
        backend: Union[str, Backend],
    ):
        """Initialize a tracer.

        Args:
            project: Project namespace. All operations are tagged with this.
            backend: Either a connection string or a Backend instance.
                Connection string formats:
                - SQLite: sqlite:///path/to/traces.db
                - PostgreSQL: postgresql://user:pass@host:port/dbname
        """
        self._project = project

        if isinstance(backend, str):
            self._backend = SQLBackend(backend)
        else:
            self._backend = backend

    @property
    def project(self) -> str:
        """Get the project namespace."""
        return self._project

    @property
    def backend(self) -> Backend:
        """Get the backend instance."""
        return self._backend

    def op(self, operation: str) -> OpBuilder:
        """Start building an operation trace.

        Args:
            operation: The operation type (e.g., 'rerank', 'nli', 'classify').

        Returns:
            An OpBuilder for fluent configuration.
        """
        return OpBuilder(
            operation=operation,
            project=self._project,
            backend=self._backend,
        )

    @contextmanager
    def run(self, run_id: Optional[str] = None) -> Iterator[RunContext]:
        """Context manager for grouping related operations.

        Args:
            run_id: Optional explicit run ID. If not provided, a ULID is generated.

        Yields:
            The RunContext for this run.
        """
        with run_context(run_id) as ctx:
            yield ctx


def configure(
    project: str,
    backend: Union[str, Backend],
) -> Tracer:
    """Configure the global default tracer.

    Must be called before any operations are logged using the global op()/run().

    Args:
        project: Project namespace. All operations are tagged with this.
        backend: Either a connection string or a Backend instance.

    Returns:
        The configured Tracer instance.
    """
    global _default_tracer
    _default_tracer = Tracer(project=project, backend=backend)
    return _default_tracer


def get_tracer() -> Tracer:
    """Get the global default tracer.

    Raises:
        RuntimeError: If configure() has not been called.
    """
    if _default_tracer is None:
        raise RuntimeError(
            "oplog not configured. Call configure() first."
        )
    return _default_tracer


def op(operation: str) -> OpBuilder:
    """Start building an operation trace using the global tracer.

    Args:
        operation: The operation type (e.g., 'rerank', 'nli', 'classify').

    Returns:
        An OpBuilder for fluent configuration.

    Raises:
        RuntimeError: If configure() has not been called.
    """
    return get_tracer().op(operation)


@contextmanager
def run(run_id: Optional[str] = None) -> Iterator[RunContext]:
    """Context manager for grouping related operations.

    Uses the global tracer.

    Args:
        run_id: Optional explicit run ID. If not provided, a ULID is generated.

    Yields:
        The RunContext for this run.

    Raises:
        RuntimeError: If configure() has not been called.
    """
    # Validate tracer is configured
    get_tracer()
    with run_context(run_id) as ctx:
        yield ctx