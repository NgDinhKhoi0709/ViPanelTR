"""Typed contracts shared by evaluation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AlignedSample:
    qa_id: str
    prediction: list[str]
    reference: str
    hints: list[str] = field(default_factory=list)
    table_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricScore:
    value: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    best_candidate: str = ""
    best_candidate_index: int = 0


@dataclass(frozen=True)
class AlignmentCoverage:
    evaluated_ids: list[str]
    missing_predictions: list[str]
    extra_predictions: list[str]

