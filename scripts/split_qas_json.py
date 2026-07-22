#!/usr/bin/env python3
"""
Split a ViTabQA-style JSON dataset into N contiguous parts.

Input schema:
  { "qas": [ {...}, {...}, ... ] }

Each output keeps the same schema so concatenating parts in order reproduces the original.

Example:
  python scripts/split_qas_json.py --input dataset/qas_test.json --parts 5 --out-dir dataset/qas_test_parts --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root, got {type(data).__name__}")
    return data


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_contiguous_chunks(total: int, parts: int) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) half-open slices covering [0, total) contiguously.
    Chunks are as even as possible; early chunks get the remainder (+1).
    """
    if parts <= 0:
        raise ValueError("--parts must be >= 1")
    if total < 0:
        raise ValueError("total must be >= 0")
    if parts > total and total != 0:
        raise ValueError(f"--parts ({parts}) cannot exceed number of items ({total})")
    if total == 0:
        return [(0, 0)] if parts == 1 else []

    base = total // parts
    rem = total % parts

    chunks: List[Tuple[int, int]] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunks.append((start, end))
        start = end
    return chunks


def qa_ids(qas: Sequence[Dict[str, Any]]) -> List[Any]:
    return [qa.get("qa_id") for qa in qas]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Split a {qas:[...]} JSON dataset into N contiguous parts (preserve order). "
            "Each output keeps the same schema so concatenating parts in order reproduces the original."
        ),
        epilog=(
            "Example:\n"
            "  python scripts/split_qas_json.py --input dataset/qas_test.json --parts 5 "
            "--out-dir dataset/qas_test_parts --verify\n"
            "\n"
            "Output files are named like:\n"
            "  <stem>.part01of05.json, <stem>.part02of05.json, ...\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="input JSON path (schema: {qas:[...]})")
    p.add_argument("--parts", required=True, type=int, help="number of contiguous parts to split into")
    p.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <input_parent>/<input_stem>_parts)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="verify that concatenating outputs reproduces original qa_id order/length",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file does not exist: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_path.parent / f"{in_path.stem}_parts")

    data = load_json(in_path)
    qas = data.get("qas")
    if not isinstance(qas, list):
        raise SystemExit('Invalid input schema: expected key "qas" containing a JSON array')

    total = len(qas)
    parts = int(args.parts)
    if total == 0:
        raise SystemExit("Input has 0 items in qas; nothing to split")
    if parts < 1:
        raise SystemExit("--parts must be >= 1")
    if parts > total:
        raise SystemExit(f"--parts ({parts}) cannot exceed number of items ({total})")

    chunks = compute_contiguous_chunks(total=total, parts=parts)
    width = max(2, len(str(parts)))

    written_paths: List[Path] = []
    written_qas_lists: List[List[Dict[str, Any]]] = []

    print(f"Input: {in_path}")
    print(f"Total QAs: {total}")
    print(f"Parts: {parts}")
    print(f"Out dir: {out_dir}")

    for idx, (start, end) in enumerate(chunks, start=1):
        part_qas = qas[start:end]
        out_name = f"{in_path.stem}.part{idx:0{width}d}of{parts:0{width}d}.json"
        out_path = out_dir / out_name
        save_json(out_path, {"qas": part_qas})
        written_paths.append(out_path)
        written_qas_lists.append(part_qas)
        print(f"- Wrote {out_path}  (items={len(part_qas)}; slice={start}:{end})")

    if args.verify:
        original_ids = qa_ids(qas)
        recombined: List[Dict[str, Any]] = []
        for part_list in written_qas_lists:
            recombined.extend(part_list)
        recombined_ids = qa_ids(recombined)

        if len(recombined) != len(qas):
            raise SystemExit(
                f"VERIFY FAILED: recombined length {len(recombined)} != original length {len(qas)}"
            )
        if recombined_ids != original_ids:
            # Find first mismatch index for a helpful error
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(recombined_ids, original_ids)) if a != b),
                None,
            )
            raise SystemExit(f"VERIFY FAILED: qa_id order mismatch at index {mismatch}")
        print("VERIFY OK: recombined qa_id order and length match original")


if __name__ == "__main__":
    main()

