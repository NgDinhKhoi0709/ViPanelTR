"""Public table preprocessing API."""

from .loader import DatasetLoader
from .parser import HTMLTableParser, ParsedTable, parse_html_table
from .representation import FlattenedTable, TableRepresentation, create_representation
from .run import prepare_table, prepare_table_by_id
from .validation import DatasetValidationError, DatasetValidationReport, validate_dataset

__all__ = [
    "DatasetLoader",
    "FlattenedTable",
    "HTMLTableParser",
    "ParsedTable",
    "TableRepresentation",
    "create_representation",
    "parse_html_table",
    "prepare_table",
    "prepare_table_by_id",
    "DatasetValidationError",
    "DatasetValidationReport",
    "validate_dataset",
]
