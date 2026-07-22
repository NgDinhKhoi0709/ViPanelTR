"""
Evaluator for PanelTR-ViTabQA.
Delegates metric calculations to the unified ViPanelTR evaluation package.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .answerability_f1 import evaluate_answerability
from .io import align_records
from .rouge1_by_hint import evaluate_by_hint
from .run import _core_metrics


class EvaluationResult:
    """Wrapper for backward compatibility with PanelTR EvaluationResult format."""

    def __init__(self, metrics: Dict[str, Any], answerability: Dict[str, Any], hint: Dict[str, Any], coverage: Any):
        self.f1_score = metrics.get("f1", {}).get("f1", 0.0)
        self.exact_match = metrics.get("em", {}).get("value", 0.0)
        self.rouge1_f1 = metrics.get("rouge1", {}).get("f1", 0.0)
        self.meteor_score = metrics.get("meteor", {}).get("value", 0.0)
        if self.meteor_score == 0.0:
            self.meteor_score = metrics.get("meteor", {}).get("meteor", 0.0)

        self.f1_by_answerability = answerability
        self.rouge1_by_hint = hint
        
        self.total_samples = len(coverage.evaluated_ids) if coverage else 0
        
        self.sample_results = []
        self.answerable_correct = 0
        self.unanswerable_correct = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f1_score": self.f1_score,
            "exact_match": self.exact_match,
            "rouge1_f1": self.rouge1_f1,
            "meteor_score": self.meteor_score,

            "f1_by_answerability": self.f1_by_answerability,
            "rouge1_by_hint": self.rouge1_by_hint,
            "total_samples": self.total_samples,
            "answerable_correct": self.answerable_correct,
            "unanswerable_correct": self.unanswerable_correct,
        }


class Evaluator:
    """Compatibility wrapper around the unified evaluation modules."""

    def __init__(
        self,
        metrics: List[str] = None,
        nli_checkpoint: Optional[str] = None,
        bert_model_type: str = "vinai/phobert-large",
        bert_num_layers: int = 17,
        bert_lang: str = "vi",
        use_vietnamese_tokenizer: bool = True,
        fail_on_metric_error: bool = False,
    ):
        cleaned_metrics = []
        if metrics:
            for m in metrics:
                if m not in ("bertscore", "vinli", "bif"):
                    cleaned_metrics.append(m)
        else:
            cleaned_metrics = ["f1", "em", "rouge1", "meteor"]
            
        self.cleaned_metrics = cleaned_metrics
        self.fail_on_metric_error = fail_on_metric_error
        
        self.metric_error_counts = {}
        self.metric_error_messages = {}

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        max_workers: int = 1,
    ) -> EvaluationResult:
        try:
            samples, coverage = align_records(predictions, references)
            metrics = _core_metrics(samples, set(self.cleaned_metrics))
            answerability = evaluate_answerability(samples)
            hint = evaluate_by_hint(samples)
            
            return EvaluationResult(metrics, answerability, hint, coverage)
        except Exception as e:
            if self.fail_on_metric_error:
                raise
            self.metric_error_messages["evaluation"] = str(e)
            return EvaluationResult({}, {}, {}, None)

    def cleanup_models(self) -> None:
        pass


def evaluate_jsonl(
    predictions_path: str,
    references_path: str,
    metrics: List[str] = None,
    nli_checkpoint: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate predictions from JSONL file."""
    import json

    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    with open(references_path, "r", encoding="utf-8") as f:
        references = json.load(f)

    evaluator = Evaluator(metrics=metrics)
    return evaluator.evaluate(predictions, references)
