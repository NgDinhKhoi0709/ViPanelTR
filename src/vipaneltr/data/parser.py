"""
HTML table parser for Open-ViTabQA.

Parses HTML tables and extracts structure including merged cells (colspan/rowspan).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag


@dataclass
class Cell:
    """Represents a table cell."""
    value: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    merged_from: Optional[Tuple[int, int]] = None  # (row, col) of source cell if propagated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "row": self.row,
            "col": self.col,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "is_header": self.is_header,
            "merged_from": self.merged_from,
        }


@dataclass
class ParsedTable:
    """Represents a parsed HTML table."""
    headers: List[List[Cell]] = field(default_factory=list)  # Multi-level headers
    rows: List[List[Cell]] = field(default_factory=list)  # Data rows
    merged_regions: List[Dict[str, Any]] = field(default_factory=list)
    num_cols: int = 0
    num_rows: int = 0
    has_merged_headers: bool = False
    has_merged_values: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "headers": [[c.to_dict() for c in row] for row in self.headers],
            "rows": [[c.to_dict() for c in row] for row in self.rows],
            "merged_regions": self.merged_regions,
            "num_cols": self.num_cols,
            "num_rows": self.num_rows,
            "has_merged_headers": self.has_merged_headers,
            "has_merged_values": self.has_merged_values,
        }


class HTMLTableParser:
    """
    Parser for HTML tables with merged cell support.
    
    Handles:
    - colspan/rowspan attributes
    - Multi-level headers
    - Value propagation for merged cells
    """
    
    def __init__(self):
        self._reset()
    
    def _reset(self):
        """Reset parser state."""
        self._grid: List[List[Optional[Cell]]] = []
        self._num_cols = 0
        self._num_rows = 0
    
    def parse(self, html: str) -> ParsedTable:
        """
        Parse an HTML table string.
        
        Args:
            html: HTML string containing a table
            
        Returns:
            ParsedTable object with parsed structure
        """
        self._reset()
        
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        
        if not table:
            return ParsedTable()
        
        # First pass: determine grid size
        rows = table.find_all("tr")
        self._num_rows = len(rows)
        
        # Calculate max columns considering colspan
        max_cols = 0
        for row in rows:
            cols = 0
            for cell in row.find_all(["th", "td"]):
                colspan = int(cell.get("colspan", 1))
                cols += colspan
            max_cols = max(max_cols, cols)
        
        self._num_cols = max_cols
        
        # Initialize grid
        self._grid = [[None for _ in range(self._num_cols)] for _ in range(self._num_rows)]
        
        # Second pass: fill grid with cells
        merged_regions = []
        has_merged_headers = False
        has_merged_values = False
        
        for row_idx, row in enumerate(rows):
            col_idx = 0
            
            for cell in row.find_all(["th", "td"]):
                # Find next available column
                while col_idx < self._num_cols and self._grid[row_idx][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= self._num_cols:
                    break
                
                # Parse cell attributes
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))
                is_header = cell.name == "th"
                value = self._clean_text(cell.get_text())
                
                # Check for merged cells
                if rowspan > 1 or colspan > 1:
                    if is_header:
                        has_merged_headers = True
                    else:
                        has_merged_values = True
                    
                    merged_regions.append({
                        "start": (row_idx, col_idx),
                        "end": (row_idx + rowspan - 1, col_idx + colspan - 1),
                        "value": value,
                        "is_header": is_header,
                    })
                
                # Create primary cell
                primary_cell = Cell(
                    value=value,
                    row=row_idx,
                    col=col_idx,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=is_header,
                )
                
                # Fill grid for span area
                for r in range(row_idx, min(row_idx + rowspan, self._num_rows)):
                    for c in range(col_idx, min(col_idx + colspan, self._num_cols)):
                        if r == row_idx and c == col_idx:
                            self._grid[r][c] = primary_cell
                        else:
                            # Propagated cell
                            self._grid[r][c] = Cell(
                                value=value,
                                row=r,
                                col=c,
                                rowspan=1,
                                colspan=1,
                                is_header=is_header,
                                merged_from=(row_idx, col_idx),
                            )
                
                col_idx += colspan
        
        # Separate headers and data rows
        headers = []
        data_rows = []
        
        for row_idx, row in enumerate(self._grid):
            # Fill None cells with empty Cell
            filled_row = []
            for col_idx, cell in enumerate(row):
                if cell is None:
                    filled_row.append(Cell(value="", row=row_idx, col=col_idx))
                else:
                    filled_row.append(cell)
            
            # Determine if header row (majority th cells)
            header_count = sum(1 for c in filled_row if c.is_header)
            if header_count > len(filled_row) // 2:
                headers.append(filled_row)
            else:
                data_rows.append(filled_row)
        
        return ParsedTable(
            headers=headers,
            rows=data_rows,
            merged_regions=merged_regions,
            num_cols=self._num_cols,
            num_rows=self._num_rows,
            has_merged_headers=has_merged_headers,
            has_merged_values=has_merged_values,
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean cell text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def parse_to_rows(self, html: str) -> List[List[str]]:
        """
        Parse HTML table to simple 2D list (like table_dict format).
        
        Args:
            html: HTML string
            
        Returns:
            List of rows, each row is list of cell values
        """
        parsed = self.parse(html)
        
        result = []
        
        # Add header rows
        for header_row in parsed.headers:
            result.append([cell.value for cell in header_row])
        
        # Add data rows
        for data_row in parsed.rows:
            result.append([cell.value for cell in data_row])
        
        return result


def parse_html_table(html: str) -> ParsedTable:
    """
    Convenience function to parse HTML table.
    
    Args:
        html: HTML string containing a table
        
    Returns:
        ParsedTable object
    """
    parser = HTMLTableParser()
    return parser.parse(html)
