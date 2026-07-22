"""Core-metric breakdown for normal, merged-header, and merged-value tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from . import exact_match, f1, meteor, rouge1
from .contracts import AlignedSample

GROUPS = ("normal", "header_merge", "value_merge")


def classify_table(table: Mapping[str, object]) -> str:
    types = table.get("table_type", [])
    values = {str(value).strip().lower() for value in (types if isinstance(types, list) else [types])}
    if bool(table.get("header_merge")) or "contain_merged_header" in values or "header_merge" in values:
        return "header_merge"
    if bool(table.get("value_merge")) or "contain_merged_value" in values or "value_merge" in values:
        return "value_merge"
    return "normal"


def _metrics(samples: list[AlignedSample]) -> dict[str, dict[str, float | int]]:
    return {"f1": f1.aggregate(f1.score_sample(sample) for sample in samples), "em": exact_match.aggregate(exact_match.score_sample(sample) for sample in samples), "rouge1": rouge1.aggregate(rouge1.score_sample(sample) for sample in samples), "meteor": meteor.aggregate(meteor.score_sample(sample) for sample in samples)}


def evaluate_by_table_type(samples: Iterable[AlignedSample], table_by_id: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    groups = {group: [] for group in GROUPS}
    missing = []
    for sample in samples:
        table = table_by_id.get(sample.table_id or "")
        if table is None:
            missing.append(sample.qa_id)
            continue
        groups[classify_table(table)].append(sample)
    return {"groups": {group: {"count": len(items), "metrics": _metrics(items)} for group, items in groups.items()}, "missing_table_metadata": missing}
