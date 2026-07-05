"""SQLAlchemy-based backend implementation."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from oplog.models import Operation, Run


class SQLBackend:
    """SQLAlchemy Core backend supporting SQLite and PostgreSQL."""

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """Initialize the SQL backend.

        Args:
            connection_string: Database connection string (sqlite:/// or postgresql://)
            pool_size: Connection pool size (ignored for SQLite)
            max_overflow: Max overflow connections (ignored for SQLite)
        """
        self._connection_string = connection_string
        self._is_sqlite = connection_string.startswith("sqlite")

        # For SQLite, ensure parent directories exist
        if self._is_sqlite:
            # Parse the path from sqlite:///path/to/db.db
            db_path = connection_string.replace("sqlite:///", "")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create engine with appropriate settings
        if self._is_sqlite:
            self._engine: Engine = create_engine(
                connection_string,
                connect_args={"check_same_thread": False},
            )
        else:
            self._engine = create_engine(
                connection_string,
                pool_size=pool_size,
                max_overflow=max_overflow,
            )

        # Define metadata and tables
        self._metadata = MetaData()
        self._runs = Table(
            "runs",
            self._metadata,
            Column("id", String(255), primary_key=True),
            Column("project", String(255), nullable=False),
            Column("meta", JSON, nullable=True),
            Column("tags", JSON, default="[]"),
            Column("flagged_for", String(255), nullable=True),
            Column("flag_note", Text, nullable=True),
            Column("created_at", DateTime, nullable=False),
            Column("flagged_at", DateTime, nullable=True),
        )
        self._operations = Table(
            "operations",
            self._metadata,
            Column("id", String(26), primary_key=True),
            Column("project", String(255), nullable=False),
            Column("run_id", String(255), nullable=True),
            Column("seq", Integer, nullable=True),
            Column("operation", String(255), nullable=False),
            Column("model", String(255), nullable=True),
            Column("inputs", JSON, nullable=True),
            Column("outputs", JSON, nullable=True),
            Column("meta", JSON, nullable=True),
            Column("tags", JSON, default="[]"),
            Column("flagged_for", String(255), nullable=True),
            Column("flag_note", Text, nullable=True),
            Column("created_at", DateTime, nullable=False),
            Column("flagged_at", DateTime, nullable=True),
        )

        # Initialize schema on creation
        self.init_schema()

    def init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._metadata.create_all(self._engine)

        # Create indexes (SQLAlchemy will skip if they exist)
        indexes = [
            Index("idx_operations_project", self._operations.c.project),
            Index("idx_operations_operation", self._operations.c.operation),
            Index("idx_operations_run_id", self._operations.c.run_id),
            Index("idx_operations_created", self._operations.c.created_at),
            Index("idx_runs_project", self._runs.c.project),
            Index("idx_runs_created", self._runs.c.created_at),
        ]

        for idx in indexes:
            try:
                idx.create(self._engine, checkfirst=True)
            except Exception:
                pass  # Index may already exist

        # Partial index for flagged operations (PostgreSQL only)
        if not self._is_sqlite:
            try:
                flagged_idx = Index(
                    "idx_operations_flagged",
                    self._operations.c.flagged_for,
                    postgresql_where=self._operations.c.flagged_for.isnot(None),
                )
                flagged_idx.create(self._engine, checkfirst=True)
            except Exception:
                pass

    def save(self, operation: Operation) -> str:
        """Persist an operation. Returns the operation ID."""
        data = self._operation_to_row(operation)

        with self._engine.begin() as conn:
            conn.execute(insert(self._operations).values(**data))

        return operation.id

    def save_run(self, run: Run) -> str:
        """Persist a run row. Upserts: an existing id has its meta/tags
        refreshed (callers may legitimately reuse a run id across entries,
        e.g. one id per thread re-entered per iteration)."""
        with self._engine.begin() as conn:
            existing = conn.execute(
                select(self._runs.c.id).where(self._runs.c.id == run.id)
            ).fetchone()
            if existing:
                values: Dict[str, Any] = {}
                if run.meta:
                    values["meta"] = run.meta
                if run.tags:
                    values["tags"] = run.tags
                if values:
                    conn.execute(
                        update(self._runs).where(self._runs.c.id == run.id).values(**values)
                    )
            else:
                conn.execute(
                    insert(self._runs).values(
                        id=run.id,
                        project=run.project,
                        meta=run.meta,
                        tags=run.tags,
                        flagged_for=run.flagged_for,
                        flag_note=run.flag_note,
                        created_at=run.created_at,
                        flagged_at=run.flagged_at,
                    )
                )
        return run.id

    def get_run(self, run_id: str) -> Optional[Run]:
        """Fetch a single run row by id."""
        with self._engine.connect() as conn:
            row = conn.execute(
                select(self._runs).where(self._runs.c.id == run_id)
            ).fetchone()
        return self._row_to_run(row._mapping) if row else None

    def query_runs(
        self,
        project: Optional[str] = None,
        flagged_for: Optional[str] = None,
        tags: Optional[List[str]] = None,
        after: Optional[datetime] = None,
        before: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Run]:
        """Query run rows with filters."""
        stmt = select(self._runs)

        if project is not None:
            stmt = stmt.where(self._runs.c.project == project)
        if flagged_for is not None:
            stmt = stmt.where(self._runs.c.flagged_for == flagged_for)
        if after is not None:
            stmt = stmt.where(self._runs.c.created_at > after)
        if before is not None:
            stmt = stmt.where(self._runs.c.created_at < before)
        if tags:
            for tag in tags:
                if self._is_sqlite:
                    stmt = stmt.where(self._runs.c.tags.contains(f'"{tag}"'))
                else:
                    stmt = stmt.where(self._runs.c.tags.contains([tag]))

        stmt = stmt.order_by(self._runs.c.created_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [self._row_to_run(row._mapping) for row in rows]

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
        stmt = select(self._operations)

        # Apply filters
        if project is not None:
            stmt = stmt.where(self._operations.c.project == project)
        if operation is not None:
            stmt = stmt.where(self._operations.c.operation == operation)
        if model is not None:
            stmt = stmt.where(self._operations.c.model == model)
        if run_id is not None:
            stmt = stmt.where(self._operations.c.run_id == run_id)
        if flagged_for is not None:
            stmt = stmt.where(self._operations.c.flagged_for == flagged_for)
        if after is not None:
            stmt = stmt.where(self._operations.c.created_at > after)
        if before is not None:
            stmt = stmt.where(self._operations.c.created_at < before)

        # Tag filtering (AND logic - must have all specified tags)
        if tags:
            for tag in tags:
                if self._is_sqlite:
                    # SQLite JSON contains check
                    stmt = stmt.where(
                        self._operations.c.tags.contains(f'"{tag}"')
                    )
                else:
                    # PostgreSQL JSON contains check
                    stmt = stmt.where(
                        self._operations.c.tags.contains([tag])
                    )

        # Order by created_at descending (most recent first)
        stmt = stmt.order_by(self._operations.c.created_at.desc())

        # Pagination
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)

        with self._engine.connect() as conn:
            result = conn.execute(stmt)
            rows = result.fetchall()

        return [self._row_to_operation(row._mapping) for row in rows]

    def flag(
        self,
        reason: str,
        ids: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> int:
        """Flag operations (by ids) or a run (by run_id — flags the run row
        itself, not its operations). Returns count of affected rows."""
        if ids is None and run_id is None:
            raise ValueError("Either ids or run_id must be provided")

        now = datetime.now(timezone.utc)
        values = {
            "flagged_for": reason,
            "flagged_at": now,
        }
        if note is not None:
            values["flag_note"] = note

        if run_id is not None:
            stmt = update(self._runs).values(**values).where(self._runs.c.id == run_id)
        else:
            stmt = update(self._operations).values(**values).where(
                self._operations.c.id.in_(ids)
            )

        with self._engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount

    def unflag(
        self,
        ids: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> int:
        """Remove flags from operations (by ids) or from a run row (by
        run_id). Returns count of affected rows."""
        if ids is None and run_id is None:
            raise ValueError("Either ids or run_id must be provided")

        values = {
            "flagged_for": None,
            "flag_note": None,
            "flagged_at": None,
        }

        if run_id is not None:
            stmt = update(self._runs).values(**values).where(self._runs.c.id == run_id)
        else:
            stmt = update(self._operations).values(**values).where(
                self._operations.c.id.in_(ids)
            )

        with self._engine.begin() as conn:
            result = conn.execute(stmt)
            return result.rowcount

    def _operation_to_row(self, op: Operation) -> Dict[str, Any]:
        """Convert an Operation to a database row dict."""
        return {
            "id": op.id,
            "project": op.project,
            "run_id": op.run_id,
            "seq": op.seq,
            "operation": op.operation,
            "model": op.model,
            "inputs": op.inputs,
            "outputs": op.outputs,
            "meta": op.meta,
            "tags": op.tags,
            "flagged_for": op.flagged_for,
            "flag_note": op.flag_note,
            "created_at": op.created_at,
            "flagged_at": op.flagged_at,
        }

    def _row_to_run(self, row: Dict[str, Any]) -> Run:
        """Convert a database row to a Run."""
        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)

        return Run(
            id=row["id"],
            project=row["project"],
            meta=row["meta"],
            tags=tags or [],
            flagged_for=row["flagged_for"],
            flag_note=row["flag_note"],
            created_at=row["created_at"],
            flagged_at=row["flagged_at"],
        )

    def _row_to_operation(self, row: Dict[str, Any]) -> Operation:
        """Convert a database row to an Operation."""
        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)

        return Operation(
            id=row["id"],
            project=row["project"],
            run_id=row["run_id"],
            seq=row["seq"],
            operation=row["operation"],
            model=row["model"],
            inputs=row["inputs"],
            outputs=row["outputs"],
            meta=row["meta"],
            tags=tags or [],
            flagged_for=row["flagged_for"],
            flag_note=row["flag_note"],
            created_at=row["created_at"],
            flagged_at=row["flagged_at"],
        )