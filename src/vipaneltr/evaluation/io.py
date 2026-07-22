"""File loading and qa_id alignment for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import AlignedSample, AlignmentCoverage
from .exceptions import EvaluationDataError


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("qas", "table", "tables", "predictions"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise EvaluationDataError(
        f"{source} must contain a JSON list, JSONL records, or a supported record list"
    )


def load_qas_records(path: str | Path) -> list[dict[str, Any]]:
    return load_json_records(path)


def extract_candidates(record: Mapping[str, Any]) -> list[str]:
    raw = record.get("prediction", record.get("predictions", record.get("answer", "")))
    if isinstance(raw, (list, tuple)):
        return [str(value).strip() for value in raw if value is not None] or [""]
    return [str(raw).strip()]


def align_records(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> tuple[list[AlignedSample], AlignmentCoverage]:
    prediction_by_id = {str(item["qa_id"]): item for item in predictions if item.get("qa_id") is not None}
    reference_by_id = {str(item["qa_id"]): item for item in references if item.get("qa_id") is not None}
    samples = []
    for qa_id in sorted(prediction_by_id.keys() & reference_by_id.keys()):
        prediction, reference = prediction_by_id[qa_id], reference_by_id[qa_id]
        samples.append(AlignedSample(qa_id=qa_id, prediction=extract_candidates(prediction), reference=str(reference.get("answer", "")), hints=[str(value) for value in reference.get("hints", []) or []], table_id=reference.get("table_id"), metadata=dict(reference)))
    return samples, AlignmentCoverage(evaluated_ids=[sample.qa_id for sample in samples], missing_predictions=sorted(reference_by_id.keys() - prediction_by_id.keys()), extra_predictions=sorted(prediction_by_id.keys() - reference_by_id.keys()))
