"""Aggregate the token and USD usage fields emitted by POMA."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .exceptions import EvaluationDataError

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd")


def _number(record: Mapping[str, Any], field: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationDataError(f"Invalid {field}: {record.get(field)!r}") from error
    if value < 0:
        raise EvaluationDataError(f"{field} must be non-negative")
    return value


def _summary(records: list[Mapping[str, Any]]) -> dict[str, float | int]:
    count = len(records)
    totals = {field: sum(_number(record, field) for record in records) for field in USAGE_FIELDS}
    return {
        "records": count,
        "prompt_tokens": int(totals["prompt_tokens"]),
        "completion_tokens": int(totals["completion_tokens"]),
        "total_tokens": int(totals["total_tokens"]),
        "total_cost_usd": round(totals["cost_usd"], 6),
        "average_prompt_tokens": totals["prompt_tokens"] / count if count else 0.0,
        "average_completion_tokens": totals["completion_tokens"] / count if count else 0.0,
        "average_total_tokens": totals["total_tokens"] / count if count else 0.0,
        "average_cost_usd": round(totals["cost_usd"] / count, 6) if count else 0.0,
    }


def summarize_cost(records: Iterable[Mapping[str, Any]]) -> dict[str, object]:
    usable, missing, by_stage = [], 0, defaultdict(list)
    for record in records:
        if not all(field in record and record[field] is not None for field in USAGE_FIELDS):
            missing += 1
            continue
        usable.append(record)
        if str(record.get("stage", "")).strip():
            by_stage[str(record["stage"])].append(record)
    result: dict[str, object] = {**_summary(usable), "missing_usage_records": missing}
    if by_stage:
        result["by_stage"] = {stage: _summary(stage_records) for stage, stage_records in sorted(by_stage.items())}
    return result
