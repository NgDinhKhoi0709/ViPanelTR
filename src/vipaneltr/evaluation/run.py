"""Single public orchestration entry point for Open-ViTabQA evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import exact_match, f1, meteor, rouge1
from .answerability_f1 import evaluate_answerability
from .cost import summarize_cost
from .io import align_records, load_json_records, load_qas_records
from .metrics_by_table_type import evaluate_by_table_type
from .rouge1_by_hint import evaluate_by_hint

DEFAULT_METRICS = ("f1", "em", "rouge1", "meteor")


def _load_table_map(path: str | Path) -> dict[str, dict[str, Any]]:
    records = load_json_records(path)
    return {str(record["table_id"]): record for record in records if record.get("table_id") is not None}


def _core_metrics(samples: list, names: set[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    if "f1" in names:
        result["f1"] = f1.aggregate(f1.score_sample(sample) for sample in samples)
    if "em" in names:
        result["em"] = exact_match.aggregate(exact_match.score_sample(sample) for sample in samples)
    if "rouge1" in names:
        result["rouge1"] = rouge1.aggregate(rouge1.score_sample(sample) for sample in samples)
    if "meteor" in names:
        result["meteor"] = meteor.aggregate(meteor.score_sample(sample) for sample in samples)
    return result


def evaluate_files(prediction_path: str | Path, qas_path: str | Path, *, tables_path: str | Path | None = None, output_path: str | Path | None = None, metrics: list[str] | tuple[str, ...] | None = None, fail_on_metric_error: bool = False) -> dict[str, object]:
    """Evaluate one prediction file and optionally write one stable JSON report."""
    requested = set(metrics or DEFAULT_METRICS)
    predictions = load_json_records(prediction_path)
    references = load_qas_records(qas_path)
    samples, coverage = align_records(predictions, references)
    report: dict[str, object] = {
        "inputs": {"predictions": str(prediction_path), "qas": str(qas_path)},
        "coverage": {"evaluated_ids": coverage.evaluated_ids, "missing_predictions": coverage.missing_predictions, "extra_predictions": coverage.extra_predictions},
        "metrics": {},
        "analyses": {},
        "metric_errors": {},
    }
    try:
        report["metrics"] = _core_metrics(samples, requested)
        if "answerability_f1" in requested:
            report["analyses"]["answerability_f1"] = evaluate_answerability(samples)
        if "rouge1_by_hint" in requested:
            report["analyses"]["rouge1_by_hint"] = evaluate_by_hint(samples)
        if "cost" in requested:
            report["cost"] = summarize_cost(predictions)
        if "metrics_by_table_type" in requested:
            if tables_path is None:
                raise ValueError("tables_path is required for metrics_by_table_type")
            report["analyses"]["metrics_by_table_type"] = evaluate_by_table_type(samples, _load_table_map(tables_path))
    except Exception as error:
        if fail_on_metric_error:
            raise
        report["metric_errors"] = {"evaluation": str(error)}
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
