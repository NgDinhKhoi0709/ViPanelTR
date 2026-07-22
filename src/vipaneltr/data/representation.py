"""
Table representation for LLM input (Flatten V1 string).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .parser import HTMLTableParser


class TableRepresentation(ABC):
    """Abstract base class for table representations."""

    @abstractmethod
    def to_string(self) -> str:
        """Convert to string representation for LLM input."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        pass


@dataclass
class FlattenedTable(TableRepresentation):
    """
    Flattened table representation using the Open-ViTabQA Flatten V1 format.

    Flatten V1 serializes each table row as a pipe-separated line. Header
    cells are marked in place with ``<header>``.

    Merged cells are duplicated by HTMLTableParser before flattening.
    """

    table_id: str
    title: str
    domain: str
    rows: List[List[str]]
    table_type: List[str] = field(default_factory=list)
    header_mask: List[List[bool]] = field(default_factory=list)

    @staticmethod
    def _format_header(value: str) -> str:
        """Attach <header> tag to a header value."""
        text = str(value or "").strip()
        if text:
            return f"{text} <header>"
        return "<header>"

    @staticmethod
    def _normalize_rows(rows: List[List[str]]) -> List[List[str]]:
        """Pad rows to rectangular shape for safe index access."""
        if not rows:
            return []

        max_cols = max((len(row) for row in rows), default=0)
        if max_cols == 0:
            return [list(row) for row in rows]

        normalized: List[List[str]] = []
        for row in rows:
            out_row = [str(cell) for cell in row]
            if len(out_row) < max_cols:
                out_row.extend([""] * (max_cols - len(out_row)))
            normalized.append(out_row)
        return normalized

    @staticmethod
    def _normalize_header_mask(
        header_mask: List[List[bool]],
        rows: List[List[str]],
    ) -> List[List[bool]]:
        """Pad header flags to match normalized rows."""
        if not rows:
            return []

        max_cols = max((len(row) for row in rows), default=0)
        normalized: List[List[bool]] = []
        for i, row in enumerate(rows):
            mask_row = header_mask[i] if i < len(header_mask) else []
            out_row = [bool(value) for value in mask_row[:max_cols]]
            if len(out_row) < max_cols:
                out_row.extend([False] * (max_cols - len(out_row)))
            normalized.append(out_row)
        return normalized

    def _to_flatten_v1_lines(self) -> List[str]:
        """Convert rows into row-wise Flatten V1 lines with header tags."""
        table_rows = self._normalize_rows(self.rows)
        if not table_rows:
            return []

        header_mask = self._normalize_header_mask(self.header_mask, table_rows)
        lines: List[str] = []

        for i, row in enumerate(table_rows):
            out_cells: List[str] = []
            for j, value in enumerate(row):
                if header_mask[i][j]:
                    out_cells.append(self._format_header(value))
                else:
                    out_cells.append(str(value or "").strip())
            lines.append("|".join(out_cells))

        return lines

    def to_string(self) -> str:
        """
        Convert to Flatten V1 string format for LLM.

        Output format:
        cell<header>|cell<header>|value...
        """
        return "\n".join(self._to_flatten_v1_lines())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_id": self.table_id,
            "title": self.title,
            "domain": self.domain,
            "rows": self.rows,
            "header_mask": self.header_mask,
            "table_type": self.table_type,
            "flatten_v1": self._to_flatten_v1_lines(),
            "format": "flattened",
        }

    @classmethod
    def from_table_data(cls, table_data: Dict[str, Any]) -> "FlattenedTable":
        """
        Create from Open-ViTabQA table data.

        Args:
            table_data: Table dictionary from table.json

        Returns:
            FlattenedTable instance
        """
        html = str(table_data.get("table_html") or "").strip()

        parsed = HTMLTableParser().parse(html)
        cells = parsed.headers + parsed.rows
        rows = [[cell.value for cell in row] for row in cells]
        header_mask = [[cell.is_header for cell in row] for row in cells]

        return cls(
            table_id=table_data.get("table_id", ""),
            title=table_data.get("table_title", ""),
            domain=table_data.get("table_domain", ""),
            rows=rows,
            table_type=table_data.get("table_type", []),
            header_mask=header_mask,
        )


def create_representation(
    table_data: Dict[str, Any],
    format: Optional[str] = None,
) -> FlattenedTable:
    """Build Flatten V1 representation from a table record."""
    return FlattenedTable.from_table_data(table_data)
