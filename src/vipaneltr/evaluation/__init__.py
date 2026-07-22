"""Public evaluation API for Open-ViTabQA."""

from .contracts import AlignedSample, AlignmentCoverage, MetricScore
from .evaluator import EvaluationResult, Evaluator
from .run import evaluate_files

__all__ = [
    "AlignedSample",
    "AlignmentCoverage",
    "EvaluationResult",
    "Evaluator",
    "MetricScore",
    "evaluate_files",
]
