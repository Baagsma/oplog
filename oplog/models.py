"""Pydantic models for oplog."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Operation(BaseModel):
    """Represents a captured ML/NLP operation trace."""

    # Identity
    id: str
    project: str

    # Run grouping (optional)
    run_id: Optional[str] = None
    seq: Optional[int] = None

    # Operation data
    operation: str
    model: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    # Organization
    tags: List[str] = Field(default_factory=list)
    flagged_for: Optional[str] = None
    flag_note: Optional[str] = None

    # Timestamps
    created_at: datetime
    flagged_at: Optional[datetime] = None


# Type alias for query results
Record = Operation