"""Export format converters for oplog records."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from oplog.models import Record


def _get_nested_value(obj: Any, path: str) -> Any:
    """Get a nested value using dot notation.

    Args:
        obj: The object to extract from.
        path: Dot-separated path (e.g., 'inputs.query').

    Returns:
        The value at the path, or None if not found.
    """
    parts = path.split(".")
    current = obj

    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None

    return current


def _record_to_dict(record: Record, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convert a record to a dictionary, optionally selecting specific fields.

    Args:
        record: The record to convert.
        fields: Optional list of fields to include (supports dot notation).

    Returns:
        Dictionary representation of the record.
    """
    if fields is None:
        # Return full record as dict
        return record.model_dump(mode="json")

    # Select specific fields
    result = {}
    for field in fields:
        value = _get_nested_value(record, field)
        # Use the full path as the key (flattened)
        result[field] = value

    return result


def to_jsonl(
    records: List[Record],
    path: Union[str, Path],
    fields: Optional[List[str]] = None,
) -> None:
    """Export records to JSONL format (one JSON object per line).

    Args:
        records: List of records to export.
        path: Output file path.
        fields: Optional list of fields to include (supports dot notation).
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            data = _record_to_dict(record, fields)
            f.write(json.dumps(data, default=str) + "\n")


def to_csv(
    records: List[Record],
    path: Union[str, Path],
    fields: Optional[List[str]] = None,
) -> None:
    """Export records to CSV format (JSON columns are serialized as strings).

    Args:
        records: List of records to export.
        path: Output file path.
        fields: Optional list of fields to include (supports dot notation).
    """
    if not records:
        # Create empty file with no headers
        Path(path).touch()
        return

    path = Path(path)

    # Get all data first to determine columns
    rows = [_record_to_dict(record, fields) for record in records]

    # Determine column headers from first row
    if fields:
        headers = fields
    else:
        headers = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Serialize complex values to JSON strings
            serialized_row = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    serialized_row[key] = json.dumps(value, default=str)
                else:
                    serialized_row[key] = value
            writer.writerow(serialized_row)


def to_dataframe(
    records: List[Record],
    fields: Optional[List[str]] = None,
) -> "pandas.DataFrame":
    """Export records to a pandas DataFrame.

    Args:
        records: List of records to export.
        fields: Optional list of fields to include (supports dot notation).

    Returns:
        pandas DataFrame with the records.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install oplog[export]"
        )

    rows = [_record_to_dict(record, fields) for record in records]
    return pd.DataFrame(rows)


def to_dataset(
    records: List[Record],
    fields: Optional[List[str]] = None,
) -> "datasets.Dataset":
    """Export records to a HuggingFace Dataset.

    Args:
        records: List of records to export.
        fields: Optional list of fields to include (supports dot notation).

    Returns:
        HuggingFace Dataset with the records.

    Raises:
        ImportError: If datasets is not installed.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets is required for to_dataset(). "
            "Install it with: pip install oplog[export]"
        )

    rows = [_record_to_dict(record, fields) for record in records]
    return Dataset.from_list(rows)