#!/usr/bin/env python3
"""Answerability classification analysis using F1-score.

Rule:
- Ground-truth "Null" => Unanswerable, otherwise Answerable.
- Prediction "Null" => Unanswerable, otherwise Answerable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

LABELS = ["Answerable", "Unanswerable"]
NULL_MARKERS = {"", "null", "khong the tra loi", "không thể trả lời"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute answerability classification F1 (Answerable vs Unanswerable)."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to prediction file (dict-with-predictions JSON or list JSON).",
    )
    parser.add_argument(
        "--qas",
        default="dataset/qas_test.json",
        help="Path to qas JSON (contains ground-truth answers).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save JSON report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    # Normalize a common Vietnamese fallback text without requiring extra libs.
    text = text.replace("ô", "o").replace("đ", "d")
    return text


def to_answerability_label(value: Any) -> str:
    return "Unanswerable" if normalize_text(value) in NULL_MARKERS else "Answerable"


def build_gt_label_map(qas_data: Dict[str, Any]) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for qa in qas_data.get("qas", []):
        qa_id = str(qa.get("qa_id", "")).strip()
        gt_answer = qa.get("answer", "")
        label_map[qa_id] = to_answerability_label(gt_answer)
    return label_map


def extract_pred_items(results_data: Any) -> List[Dict[str, Any]]:
    if isinstance(results_data, list):
        return [x for x in results_data if isinstance(x, dict)]
    if isinstance(results_data, dict):
        preds = results_data.get("predictions", [])
        return [x for x in preds if isinstance(x, dict)]
    raise ValueError("Unsupported results format. Expect list or dict JSON.")


def get_pred_text(item: Dict[str, Any]) -> str:
    if isinstance(item.get("formatted_answer"), list) and item["formatted_answer"]:
        return str(item["formatted_answer"][0])
    if "pred_answer" in item:
        return str(item.get("pred_answer", ""))
    return str(item.get("response_text", ""))


def compute_binary_f1(
    pred_labels: Dict[str, str],
    gt_labels: Dict[str, str],
) -> Dict[str, Any]:
    counts_by_label: Dict[str, int] = {label: 0 for label in LABELS}
    for label in gt_labels.values():
        if label in counts_by_label:
            counts_by_label[label] += 1

    confusion = {
        "tp": {label: 0 for label in LABELS},
        "fp": {label: 0 for label in LABELS},
        "fn": {label: 0 for label in LABELS},
    }

    evaluated = 0
    for qa_id, gt in gt_labels.items():
        pred = pred_labels.get(qa_id)
        if pred is None:
            continue
        evaluated += 1
        for label in LABELS:
            if pred == label and gt == label:
                confusion["tp"][label] += 1
            elif pred == label and gt != label:
                confusion["fp"][label] += 1
            elif pred != label and gt == label:
                confusion["fn"][label] += 1

    def calc(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    per_type: Dict[str, Dict[str, float]] = {}
    for label in LABELS:
        m = calc(confusion["tp"][label], confusion["fp"][label], confusion["fn"][label])
        m["count"] = float(counts_by_label[label])
        per_type[label] = m

    macro_f1 = sum(per_type[label]["f1"] for label in LABELS) / len(LABELS)
    weighted_f1 = (
        sum(per_type[label]["f1"] * counts_by_label[label] for label in LABELS) / evaluated
        if evaluated
        else 0.0
    )
    accuracy = (
        sum(1 for qa_id, gt in gt_labels.items() if pred_labels.get(qa_id) == gt) / evaluated
        if evaluated
        else 0.0
    )

    return {
        "evaluated_samples": evaluated,
        "per_type": per_type,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\nAnswerability classification (F1-score)")
    print(f"{'Type':<16}{'#':>8}{'F1':>12}{'Precision':>12}{'Recall':>12}")
    for label in LABELS:
        row = report["per_type"][label]
        print(
            f"{label:<16}{int(row['count']):>8}{row['f1'] * 100:>11.2f}"
            f"{row['precision'] * 100:>12.2f}{row['recall'] * 100:>12.2f}"
        )
    print(f"\nEvaluated: {report['evaluated_samples']}")
    print(f"Macro-F1: {report['macro_f1'] * 100:.2f}")
    print(f"Weighted-F1: {report['weighted_f1'] * 100:.2f}")
    print(f"Accuracy: {report['accuracy'] * 100:.2f}")


def main() -> None:
    args = parse_args()
    results_data = load_json(Path(args.results))
    qas_data = load_json(Path(args.qas))

    gt_labels = build_gt_label_map(qas_data)
    pred_labels: Dict[str, str] = {}
    for item in extract_pred_items(results_data):
        qa_id = str(item.get("qa_id", "")).strip()
        if not qa_id:
            continue
        pred_text = get_pred_text(item)
        pred_labels[qa_id] = to_answerability_label(pred_text)

    report = compute_binary_f1(pred_labels, gt_labels)
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
