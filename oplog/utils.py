"""Utility functions for oplog."""

from ulid import ULID


def generate_ulid() -> str:
    """Generate a new ULID string."""
    return str(ULID())