"""Dependency-free METEOR-style unigram score for Table QA answers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .contracts import AlignedSample, MetricScore
from .normalization import is_unanswerable_reference, normalize_text, prediction_is_unanswerable


def _score(prediction: str, reference: str, *, use_vietnamese_tokenization: bool) -> MetricScore:
    predicted = normalize_text(prediction, use_vietnamese_tokenization=use_vietnamese_tokenization).split()
    expected = normalize_text(reference, use_vietnamese_tokenization=use_vietnamese_tokenization).split()
    if not predicted and not expected:
        return MetricScore(value=1.0, precision=1.0, recall=1.0, f1=1.0)
    if not predicted or not expected:
        return MetricScore(value=0.0, precision=0.0, recall=0.0, f1=0.0)
    matches = sum((Counter(predicted) & Counter(expected)).values())
    precision, recall = matches / len(predicted), matches / len(expected)
    value = (10 * precision * recall) / (recall + 9 * precision) if precision and recall else 0.0
    return MetricScore(value=value, precision=precision, recall=recall, f1=value)


def score_sample(sample: AlignedSample, *, use_vietnamese_tokenization: bool = False) -> MetricScore:
    candidates = sample.prediction or [""]
    if is_unanswerable_reference(sample.reference):
        index = next((i for i, candidate in enumerate(candidates) if prediction_is_unanswerable([candidate], use_vietnamese_tokenization=use_vietnamese_tokenization)), 0)
        value = 1.0 if prediction_is_unanswerable(candidates, use_vietnamese_tokenization=use_vietnamese_tokenization) else 0.0
        return MetricScore(value=value, precision=value, recall=value, f1=value, best_candidate=candidates[index], best_candidate_index=index)
    index, best = max(enumerate((_score(candidate, sample.reference, use_vietnamese_tokenization=use_vietnamese_tokenization) for candidate in candidates)), key=lambda item: item[1].value)
    return MetricScore(value=best.value, precision=best.precision, recall=best.recall, f1=best.f1, best_candidate=candidates[index], best_candidate_index=index)


def aggregate(scores: Iterable[MetricScore]) -> dict[str, float | int]:
    values = list(scores)
    return {"count": len(values), "value": sum(score.value for score in values) / len(values) if values else 0.0}
