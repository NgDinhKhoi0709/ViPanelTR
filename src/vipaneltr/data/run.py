"""One facade for loading, parsing, and representing Open-ViTabQA tables."""

from __future__ import annotations

from typing import Any

from .loader import DatasetLoader
from .representation import FlattenedTable, create_representation


def prepare_table(table_data: dict[str, Any] | str, *, representation: str = "flattened") -> FlattenedTable:
    """Return the requested prompt representation for one table record or HTML table."""
    if representation != "flattened":
        raise ValueError(f"Unsupported representation: {representation}")
    if isinstance(table_data, str):
        table_data = {"table_html": table_data}
    return create_representation(table_data, format=representation)


def prepare_table_by_id(loader: DatasetLoader, table_id: str, *, representation: str = "flattened") -> FlattenedTable:
    """Load a table by ID and return its prompt representation."""
    table = loader.get_table(table_id)
    if table is None:
        raise KeyError(f"Unknown table_id: {table_id}")
    return prepare_table(table, representation=representation)
