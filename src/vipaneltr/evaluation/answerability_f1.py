"""Answerable versus unanswerable classification report."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import AlignedSample
from .normalization import is_unanswerable_reference, prediction_is_unanswerable


def _class_score(true_positive: int, false_positive: int, false_negative: int, support: int) -> dict[str, float | int]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": support}


def evaluate_answerability(samples: Iterable[AlignedSample]) -> dict[str, object]:
    pairs = [(is_unanswerable_reference(sample.reference), prediction_is_unanswerable(sample.prediction)) for sample in samples]
    tp = sum(gold and predicted for gold, predicted in pairs)
    tn = sum(not gold and not predicted for gold, predicted in pairs)
    fp = sum(not gold and predicted for gold, predicted in pairs)
    fn = sum(gold and not predicted for gold, predicted in pairs)
    answerable = _class_score(tn, fn, fp, tn + fp)
    unanswerable = _class_score(tp, fp, fn, tp + fn)
    total = len(pairs)
    return {"accuracy": (tp + tn) / total if total else 0.0, "macro_f1": (answerable["f1"] + unanswerable["f1"]) / 2, "per_class": {"Answerable": answerable, "Unanswerable": unanswerable}, "confusion": {"gold_answerable_pred_answerable": tn, "gold_answerable_pred_unanswerable": fp, "gold_unanswerable_pred_answerable": fn, "gold_unanswerable_pred_unanswerable": tp}, "total": total}
