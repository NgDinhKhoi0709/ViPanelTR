"""Character-set F1 metric for Table QA answers."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import AlignedSample, MetricScore
from .normalization import is_unanswerable_reference, prediction_is_unanswerable, normalize_text


def _score(prediction: str, reference: str, *, use_vietnamese_tokenization: bool) -> MetricScore:
    pred_chars = {char for char in normalize_text(prediction, use_vietnamese_tokenization=use_vietnamese_tokenization) if not char.isspace()}
    ref_chars = {char for char in normalize_text(reference, use_vietnamese_tokenization=use_vietnamese_tokenization) if not char.isspace()}
    if not pred_chars and not ref_chars:
        return MetricScore(value=1.0, precision=1.0, recall=1.0, f1=1.0)
    if not pred_chars or not ref_chars:
        return MetricScore(value=0.0, precision=0.0, recall=0.0, f1=0.0)
    common = len(pred_chars & ref_chars)
    precision, recall = common / len(pred_chars), common / len(ref_chars)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricScore(value=f1, precision=precision, recall=recall, f1=f1)


def score_sample(sample: AlignedSample, *, use_vietnamese_tokenization: bool = False) -> MetricScore:
    if is_unanswerable_reference(sample.reference):
        value = 1.0 if prediction_is_unanswerable(sample.prediction, use_vietnamese_tokenization=use_vietnamese_tokenization) else 0.0
        index = next((index for index, candidate in enumerate(sample.prediction) if prediction_is_unanswerable([candidate], use_vietnamese_tokenization=use_vietnamese_tokenization)), 0)
        return MetricScore(value=value, precision=value, recall=value, f1=value, best_candidate=sample.prediction[index], best_candidate_index=index)
    candidates = sample.prediction or [""]
    best_index, best = max(enumerate((_score(candidate, sample.reference, use_vietnamese_tokenization=use_vietnamese_tokenization) for candidate in candidates)), key=lambda item: item[1].value)
    return MetricScore(value=best.value, precision=best.precision, recall=best.recall, f1=best.f1, best_candidate=candidates[best_index], best_candidate_index=best_index)


def aggregate(scores: Iterable[MetricScore]) -> dict[str, float | int]:
    values = list(scores)
    count = len(values)
    return {"count": count, "precision": sum(score.precision or 0.0 for score in values) / count if count else 0.0, "recall": sum(score.recall or 0.0 for score in values) / count if count else 0.0, "f1": sum(score.f1 or 0.0 for score in values) / count if count else 0.0}
