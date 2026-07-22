"""Create stratified QA subset with distribution close to source set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified QA subset.")
    parser.add_argument("--input", type=str, default="dataset/qas_test.json", help="Input qas JSON path.")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/qas_test_subset400_stratified.json",
        help="Output subset JSON path.",
    )
    parser.add_argument("--size", type=int, default=400, help="Subset size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--stats-output",
        type=str,
        default="outputs/qwen3-8b_full_ablation_400/subset/subset_stats.json",
        help="Path to save subset statistics JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_question_type(hints: List[str]) -> str:
    if not hints:
        return "Unknown"
    return str(hints[0]).strip() or "Unknown"


def analyze_distribution(qas_list: List[Dict[str, Any]]) -> Dict[str, int]:
    type_counter = Counter()
    for qa in qas_list:
        type_counter[get_question_type(qa.get("hints", []))] += 1
    return dict(type_counter)


def stratified_sample(qas_list: List[Dict[str, Any]], target_size: int, seed: int) -> List[Dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if target_size > len(qas_list):
        raise ValueError(f"target_size={target_size} exceeds dataset size={len(qas_list)}")

    rng = random.Random(seed)

    type_to_qas: Dict[str, List[Dict[str, Any]]] = {}
    for qa in qas_list:
        q_type = get_question_type(qa.get("hints", []))
        type_to_qas.setdefault(q_type, []).append(qa)

    total = len(qas_list)
    type_samples: Dict[str, int] = {}
    allocated = 0
    for q_type, bucket in sorted(type_to_qas.items(), key=lambda x: len(x[1]), reverse=True):
        ideal_count = (len(bucket) / total) * target_size
        count = min(round(ideal_count), len(bucket))
        if count == 0 and allocated < target_size:
            count = 1
        type_samples[q_type] = count
        allocated += count

    diff = target_size - allocated
    while diff != 0:
        adjustable = sorted(
            type_to_qas.keys(),
            key=lambda t: len(type_to_qas[t]) - type_samples.get(t, 0),
            reverse=True,
        )
        changed = False
        for q_type in adjustable:
            current = type_samples.get(q_type, 0)
            capacity = len(type_to_qas[q_type])
            if diff > 0 and current < capacity:
                type_samples[q_type] = current + 1
                diff -= 1
                changed = True
            elif diff < 0 and current > 0:
                type_samples[q_type] = current - 1
                diff += 1
                changed = True
            if diff == 0:
                break
        if not changed:
            break

    subset: List[Dict[str, Any]] = []
    for q_type, count in type_samples.items():
        if count > 0:
            subset.extend(rng.sample(type_to_qas[q_type], count))
    rng.shuffle(subset)
    return subset[:target_size]


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_output_path = Path(args.stats_output)

    data = load_json(input_path)
    qas_list = data.get("qas", [])
    if not isinstance(qas_list, list) or not qas_list:
        raise ValueError("Input JSON must contain non-empty 'qas' list.")

    original_dist = analyze_distribution(qas_list)
    subset = stratified_sample(qas_list, target_size=args.size, seed=args.seed)
    subset_dist = analyze_distribution(subset)

    save_json(output_path, {"qas": subset})

    stats: Dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "seed": args.seed,
        "target_size": args.size,
        "actual_size": len(subset),
        "original_total": len(qas_list),
        "distribution": [],
    }

    all_types = sorted(set(list(original_dist.keys()) + list(subset_dist.keys())))
    for q_type in all_types:
        orig_count = original_dist.get(q_type, 0)
        sub_count = subset_dist.get(q_type, 0)
        orig_pct = (orig_count / len(qas_list)) * 100.0 if qas_list else 0.0
        sub_pct = (sub_count / len(subset)) * 100.0 if subset else 0.0
        stats["distribution"].append(
            {
                "question_type": q_type,
                "original_count": orig_count,
                "subset_count": sub_count,
                "original_pct": orig_pct,
                "subset_pct": sub_pct,
                "delta_pct": sub_pct - orig_pct,
            }
        )

    save_json(stats_output_path, stats)

    print(f"Saved subset: {output_path} ({len(subset)} samples)")
    print(f"Saved stats:  {stats_output_path}")


if __name__ == "__main__":
    main()
