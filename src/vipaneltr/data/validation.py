from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when an Open-ViTabQA dataset snapshot violates its contract."""


@dataclass(frozen=True)
class DatasetValidationReport:
    table_count: int
    split_counts: dict[str, int]


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON in {path.name}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(f"Expected a JSON object in {path.name}")
    return value


def validate_dataset(path: str | Path) -> DatasetValidationReport:
    root = Path(path)
    required = ("table.json", "qas_train.json", "qas_dev.json", "qas_test.json")
    for name in required:
        if not (root / name).is_file():
            raise DatasetValidationError(f"Missing required dataset file: {name}")

    table_payload = _load_object(root / "table.json")
    tables = table_payload.get("table")
    if not isinstance(tables, list):
        raise DatasetValidationError("table.json must contain a top-level 'table' list")

    table_ids: set[str] = set()
    for index, table in enumerate(tables):
        if not isinstance(table, dict) or not isinstance(table.get("table_id"), str):
            raise DatasetValidationError(f"table.json entry {index} must contain a string table_id")
        table_id = table["table_id"]
        if table_id in table_ids:
            raise DatasetValidationError(f"duplicate table_id '{table_id}'")
        table_ids.add(table_id)

    seen_qa_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        filename = f"qas_{split}.json"
        payload = _load_object(root / filename)
        qas = payload.get("qas")
        if not isinstance(qas, list):
            raise DatasetValidationError(f"{filename} must contain a top-level 'qas' list")
        split_counts[split] = len(qas)
        for index, qa in enumerate(qas):
            if not isinstance(qa, dict):
                raise DatasetValidationError(f"{filename} entry {index} must be an object")
            qa_id = qa.get("qa_id")
            table_id = qa.get("table_id")
            if not isinstance(qa_id, str):
                raise DatasetValidationError(f"{filename} entry {index} must contain a string qa_id")
            if qa_id in seen_qa_ids:
                raise DatasetValidationError(f"duplicate qa_id '{qa_id}'")
            seen_qa_ids.add(qa_id)
            if table_id not in table_ids:
                raise DatasetValidationError(
                    f"{filename} qa_id '{qa_id}' references unknown table_id '{table_id}'"
                )

    return DatasetValidationReport(table_count=len(tables), split_counts=split_counts)
