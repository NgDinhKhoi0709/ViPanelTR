"""ROUGE-1 metric for Table QA answers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .contracts import AlignedSample, MetricScore
from .normalization import is_unanswerable_reference, normalize_text, prediction_is_unanswerable


def _tokens(value: str, *, use_vietnamese_tokenization: bool) -> list[str]:
    return normalize_text(value, use_vietnamese_tokenization=use_vietnamese_tokenization).split()


def _score(prediction: str, reference: str, *, use_vietnamese_tokenization: bool) -> MetricScore:
    predicted, expected = Counter(_tokens(prediction, use_vietnamese_tokenization=use_vietnamese_tokenization)), Counter(_tokens(reference, use_vietnamese_tokenization=use_vietnamese_tokenization))
    if not predicted and not expected:
        return MetricScore(value=1.0, precision=1.0, recall=1.0, f1=1.0)
    if not predicted or not expected:
        return MetricScore(value=0.0, precision=0.0, recall=0.0, f1=0.0)
    overlap = sum((predicted & expected).values())
    precision, recall = overlap / sum(predicted.values()), overlap / sum(expected.values())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricScore(value=f1, precision=precision, recall=recall, f1=f1)


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
    count = len(values)
    return {"count": count, "precision": sum(score.precision or 0.0 for score in values) / count if count else 0.0, "recall": sum(score.recall or 0.0 for score in values) / count if count else 0.0, "f1": sum(score.f1 or 0.0 for score in values) / count if count else 0.0}
