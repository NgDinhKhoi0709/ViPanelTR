#!/usr/bin/env python3
"""Compute evaluation metrics grouped by table_type (multi-label)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from vipaneltr.evaluation.evaluator import EvaluationResult, Evaluator

logging.basicConfig(level=logging.WARNING)

CORE_METRICS = ["f1", "em", "rouge1", "meteor"]
BREAKDOWN_METRICS = ["f1_by_answerability", "rouge1_by_hint"]

NULL_MARKERS = {"", "null", "khong the tra loi", "không thể trả lời"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate results.json by table_type using evaluator-compatible metrics."
    )
    parser.add_argument(
        "--results",
        default="outputs/gpt-4o-mini_full/results.json",
        help="Path to results.json",
    )
    parser.add_argument(
        "--tables",
        default="dataset/table.json",
        help="Path to table metadata JSON",
    )
    parser.add_argument(
        "--qas",
        default="dataset/qas_test.json",
        help="Path to qas JSON with answers and hints",
    )
    parser.add_argument(
        "--output",
        default="outputs/gpt-4o-mini_full/metrics_by_table_type.json",
        help="Output report JSON path",
    )
    # nli-checkpoint argument removed
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max workers passed to Evaluator.evaluate()",
    )
    # force-disable-bert argument removed
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: Any) -> str:
    return str(text).strip().lower()


def is_null_like(text: Any) -> bool:
    normalized = normalize_text(text)
    normalized = normalized.replace("ô", "o").replace("đ", "d")
    return normalized in NULL_MARKERS


def detect_metrics(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    notes: List[str] = []
    metrics = CORE_METRICS + BREAKDOWN_METRICS

    missing_core_deps: List[str] = []
    if importlib.util.find_spec("pyvi") is None:
        missing_core_deps.append("pyvi")
    if importlib.util.find_spec("rouge_score") is None:
        missing_core_deps.append("rouge_score")
    if importlib.util.find_spec("nltk") is None:
        missing_core_deps.append("nltk")
    if missing_core_deps:
        notes.append(
            "core metric dependencies missing ({deps}); evaluator may fallback for rouge1/meteor".format(
                deps=", ".join(missing_core_deps)
            )
        )

    # bertscore has been removed from the system

    # vinli detection removed

    return metrics, notes


def build_table_type_map(tables_data: Dict[str, Any]) -> Dict[str, List[str]]:
    table_items = tables_data.get("table", [])
    table_type_map: Dict[str, List[str]] = {}

    for row in table_items:
        table_id = str(row.get("table_id", "")).strip()
        raw_types = row.get("table_type", [])
        if isinstance(raw_types, str):
            types = [raw_types.strip()] if raw_types.strip() else []
        elif isinstance(raw_types, list):
            types = [str(v).strip() for v in raw_types if str(v).strip()]
        else:
            types = []

        table_type_map[table_id] = sorted(set(types or ["unknown"]))
    return table_type_map


def build_reference_map(qas_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    refs: Dict[str, Dict[str, Any]] = {}
    for qa in qas_data.get("qas", []):
        qa_id = str(qa.get("qa_id", "")).strip()
        answer = str(qa.get("answer", ""))
        refs[qa_id] = {
            "qa_id": qa_id,
            "table_id": str(qa.get("table_id", "")).strip(),
            "answer": answer,
            "answerable": not is_null_like(answer),
            "hints": qa.get("hints", []),
        }
    return refs


def build_predictions(results_data: Any) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    if isinstance(results_data, list):
        items = results_data
    elif isinstance(results_data, dict):
        items = results_data.get("predictions", [])
    else:
        raise ValueError("Unsupported results format. Expect list or dict JSON.")

    for item in items:
        qa_id = str(item.get("qa_id", "")).strip()
        table_id = str(item.get("table_id", "")).strip()
        formatted = item.get("formatted_answer")
        if isinstance(formatted, list) and formatted:
            answer: Any = [str(v) for v in formatted]
            first = answer[0]
        else:
            answer = str(item.get("pred_answer", item.get("response_text", "")))
            first = answer

        preds.append(
            {
                "qa_id": qa_id,
                "table_id": table_id,
                "answer": answer,
                "answerable": bool(item.get("answerable", not is_null_like(first))),
            }
        )
    return preds


def extract_compact_metrics(result: EvaluationResult) -> Dict[str, Any]:
    return {
        "f1_score": result.f1_score,
        "exact_match": result.exact_match,
        "rouge1_f1": result.rouge1_f1,
        "meteor_score": result.meteor_score,
        "f1_by_answerability": result.f1_by_answerability,
        "rouge1_by_hint": result.rouge1_by_hint,
        "total_samples": result.total_samples,
        "answerable_correct": result.answerable_correct,
        "unanswerable_correct": result.unanswerable_correct,
    }


def extract_minimal_metrics(result: EvaluationResult) -> Dict[str, Any]:
    return {
        "f1": result.f1_score,
        "em": result.exact_match,
        "r1": result.rouge1_f1,
        "met": result.meteor_score,
    }


def evaluate_subset(
    evaluator: Evaluator,
    preds: Iterable[Dict[str, Any]],
    refs_by_id: Dict[str, Dict[str, Any]],
    max_workers: int,
) -> Tuple[EvaluationResult, int]:
    pred_list = list(preds)
    aligned_preds: List[Dict[str, Any]] = []
    aligned_refs: List[Dict[str, Any]] = []
    missing_ref = 0

    for p in pred_list:
        qa_id = p["qa_id"]
        ref = refs_by_id.get(qa_id)
        if ref is None:
            missing_ref += 1
            continue
        aligned_preds.append(p)
        aligned_refs.append(ref)

    result = evaluator.evaluate(aligned_preds, aligned_refs, max_workers=max_workers)
    return result, missing_ref


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    tables_path = Path(args.tables)
    qas_path = Path(args.qas)
    output_path = Path(args.output)

    results_data = load_json(results_path)
    tables_data = load_json(tables_path)
    qas_data = load_json(qas_path)

    table_type_map = build_table_type_map(tables_data)
    refs_by_id = build_reference_map(qas_data)
    preds = build_predictions(results_data)
    total_predictions = len(preds)

    metrics, metric_notes = detect_metrics(args)
    evaluator = Evaluator(
        metrics=metrics,
        nli_checkpoint=args.nli_checkpoint,
        use_vietnamese_tokenizer=True,
    )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_table_metadata = 0
    for p in preds:
        table_id = p.get("table_id", "")
        table_types = table_type_map.get(table_id)
        if not table_types:
            table_types = ["unknown"]
            missing_table_metadata += 1
        for table_type in table_types:
            grouped[table_type].append(p)

    overall_result, overall_missing_ref = evaluate_subset(
        evaluator=evaluator,
        preds=preds,
        refs_by_id=refs_by_id,
        max_workers=args.max_workers,
    )

    by_table_type: Dict[str, Any] = {}
    for table_type in sorted(grouped.keys()):
        subset = grouped[table_type]
        subset_result, subset_missing_ref = evaluate_subset(
            evaluator=evaluator,
            preds=subset,
            refs_by_id=refs_by_id,
            max_workers=args.max_workers,
        )
        unique_tables = len({p["table_id"] for p in subset})
        by_table_type[table_type] = {
            "num_samples": len(subset),
            "unique_tables": unique_tables,
            "missing_reference": subset_missing_ref,
            "metrics": extract_minimal_metrics(subset_result),
        }

    minimal_report: Dict[str, Dict[str, Any]] = {}
    for table_type in ["contain_merged_header", "contain_merged_value", "normal"]:
        if table_type in by_table_type:
            minimal_report[table_type] = by_table_type[table_type]["metrics"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(minimal_report, f, ensure_ascii=False, indent=2)

    print(f"Saved report: {output_path}")
    print(f"Metrics used: {', '.join(metrics)}")
    print("Per type quick view:")
    for table_type, data in minimal_report.items():
        metric_f1 = data["f1"]
        metric_em = data["em"]
        print(
            f"- {table_type}: f1={metric_f1:.4f}, em={metric_em:.4f}"
        )


if __name__ == "__main__":
    main()
