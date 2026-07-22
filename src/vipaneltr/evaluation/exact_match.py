"""Exact-match metric for Table QA answers."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import AlignedSample, MetricScore
from .normalization import exact_text_match, is_unanswerable_reference, prediction_is_unanswerable


def score_sample(sample: AlignedSample, *, use_vietnamese_tokenization: bool = False) -> MetricScore:
    candidates = sample.prediction or [""]
    if is_unanswerable_reference(sample.reference):
        index = next((i for i, candidate in enumerate(candidates) if prediction_is_unanswerable([candidate], use_vietnamese_tokenization=use_vietnamese_tokenization)), 0)
        value = 1.0 if prediction_is_unanswerable(candidates, use_vietnamese_tokenization=use_vietnamese_tokenization) else 0.0
        return MetricScore(value=value, best_candidate=candidates[index], best_candidate_index=index)
    index = next((i for i, candidate in enumerate(candidates) if exact_text_match(candidate, sample.reference, hints=sample.hints, use_vietnamese_tokenization=use_vietnamese_tokenization)), 0)
    value = 1.0 if any(exact_text_match(candidate, sample.reference, hints=sample.hints, use_vietnamese_tokenization=use_vietnamese_tokenization) for candidate in candidates) else 0.0
    return MetricScore(value=value, best_candidate=candidates[index], best_candidate_index=index)


def aggregate(scores: Iterable[MetricScore]) -> dict[str, float | int]:
    values = list(scores)
    return {"count": len(values), "value": sum(score.value for score in values) / len(values) if values else 0.0}
