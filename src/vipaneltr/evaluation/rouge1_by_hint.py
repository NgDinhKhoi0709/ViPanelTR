"""ROUGE-1 breakdown by normalized hint/question type."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .contracts import AlignedSample
from .normalization import normalize_text
from .rouge1 import aggregate, score_sample


def infer_hint_type(hints: list[str]) -> str:
    return normalize_text(hints[0]) if hints else "unknown"


def evaluate_by_hint(samples: Iterable[AlignedSample], *, use_vietnamese_tokenization: bool = False) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list] = defaultdict(list)
    for sample in samples:
        grouped[infer_hint_type(sample.hints)].append(score_sample(sample, use_vietnamese_tokenization=use_vietnamese_tokenization))
    return {hint: aggregate(scores) for hint, scores in sorted(grouped.items())}
