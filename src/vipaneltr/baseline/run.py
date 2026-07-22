from __future__ import annotations

import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .llm_client import GenConfig, LLMZeroShotClient
from .prompts import (
    PROMPT_VERSION,
    build_tableqa_prompt,
)

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


_FILE_LOCK = threading.Lock()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)



def _append_jsonl_record(path: Path, record: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with _FILE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    records.append({"_line": line_no, "_raw": s, "_error": "not_a_json_object"})
            except Exception as e:
                records.append({"_line": line_no, "_raw": s, "_error": f"json_decode_error: {e}"})
    return records


def _load_existing_qa_ids(path: Path) -> set[str]:
    qa_ids: set[str] = set()
    for rec in _load_jsonl_records(path):
        if not isinstance(rec, dict):
            continue
        qa_id = rec.get("qa_id")
        if qa_id is not None:
            qa_ids.add(str(qa_id))
    return qa_ids


def _write_pretty_json_from_jsonl(jsonl_path: Path) -> Path:
    json_path = jsonl_path.with_suffix(".json")
    records = _load_jsonl_records(jsonl_path)
    _ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return json_path


def _calculate_batch_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if isinstance(r, dict) and not r.get("error")]
    count = len(valid)
    if not count:
        return {
            "count": 0,
            "total_prompt_tokens": 0,
            "average_prompt_tokens": 0.0,
            "total_completion_tokens": 0,
            "average_completion_tokens": 0.0,
            "total_tokens": 0,
            "average_total_tokens": 0.0,
            "total_cost_usd": 0.0,
            "average_cost_usd": 0.0,
        }
    total_prompt = sum(int(r.get("prompt_tokens", 0) or 0) for r in valid)
    total_completion = sum(int(r.get("completion_tokens", 0) or 0) for r in valid)
    total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in valid)
    total_cost = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in valid)
    return {
        "count": count,
        "total_prompt_tokens": total_prompt,
        "average_prompt_tokens": round(total_prompt / count, 1),
        "total_completion_tokens": total_completion,
        "average_completion_tokens": round(total_completion / count, 1),
        "total_tokens": total_tokens,
        "average_total_tokens": round(total_tokens / count, 1),
        "total_cost_usd": round(total_cost, 6),
        "average_cost_usd": round(total_cost / count, 6),
    }


def _write_meta_json(jsonl_path: Path) -> Path:
    meta_path = jsonl_path.with_name(f"{jsonl_path.stem}_meta.json")
    records = _load_jsonl_records(jsonl_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"count": len(records), "statistics": _calculate_batch_stats(records)}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return meta_path


def process_one_qa(
    qa: Dict[str, Any],
    table_idx: Dict[str, Any],
    models: Sequence[str],
    outputs: Dict[str, Path],
    client: LLMZeroShotClient,
    cfg: GenConfig,
    sleep_s: float,
    table_repr_format: str = "paneltr_flattened",
) -> str:
    qa_id = str(qa.get("qa_id") or "")
    table_id = str(qa.get("table_id") or "")
    question = str(qa.get("question") or "").strip()
    groundtruth = qa.get("answer")

    table = table_idx.get(table_id)

    table_str = ""
    if table and isinstance(table, dict):
        try:
            from ..data import create_representation
            table_repr = create_representation(table)
            table_str = table_repr.to_string()
        except Exception as e:
            raise RuntimeError(f"Table processing failed: {e!r}") from e

    prompt = build_tableqa_prompt(
        question=question,
        table_str=table_str,
        answer_language="vi",
    )
    prompt_version = PROMPT_VERSION

    from .utils.llm_retry import call_llm_with_retry

    for model in models:
        usage_cursor = client.usage_cursor()
        resp_text, success = call_llm_with_retry(
            llm_client=client,
            model=model,
            prompt=prompt,
            cfg=cfg,
            max_retries=4,
            caller_name="zeroshot"
        )
        usage = client.usage_since(usage_cursor)

        rec: Dict[str, Any] = {
            "qa_id": qa_id,
            "table_id": table_id,
            "question": question,
            "groundtruth": ("" if groundtruth is None else groundtruth),
            "response_text": resp_text,
            "prompt_version": prompt_version,
            "success": success,
            **usage,
        }
        _append_jsonl_record(outputs[model], rec)

        if sleep_s > 0:
            time.sleep(float(sleep_s))

    return qa_id


def run_batch_zeroshot(
    *,
    qas_path: str | Path,
    tables_path: str | Path,
    models: Sequence[str],
    output_dir: str | Path,
    limit: Optional[int] = None,
    sleep_s: float = 0.0,
    gen_config: Optional[GenConfig] = None,
    max_workers: int = 8,
    table_repr_format: str = "paneltr_flattened",
    output_id: Optional[str] = None,
    openrouter_provider: Optional[Sequence[str]] = None,
    skip_existing_qas: bool = False,
) -> Dict[str, Path]:
    from ..data.loader import DatasetLoader
    tables_path_abs = Path(tables_path).resolve()
    qas_path_abs = Path(qas_path).resolve()
    loader = DatasetLoader(dataset_dir=str(tables_path_abs.parent))
    qas = loader.load_qas_path(str(qas_path_abs))
    table_idx = loader.load_tables_path(str(tables_path_abs))
    if limit is not None:
        qas = qas[: int(limit)]

    out_dir = Path(output_dir)
    _ensure_dir(out_dir)

    client = LLMZeroShotClient()
    cfg = gen_config or GenConfig()
    if openrouter_provider:
        cfg.openrouter_provider = {"only": [str(p).strip() for p in openrouter_provider if str(p).strip()]}

    outputs: Dict[str, Path] = {}
    stem = str(output_id).strip() if output_id is not None else Path(qas_path).stem
    if not stem:
        stem = Path(qas_path).stem
    for model in models:
        safe_model = model.replace("/", "_").replace(":", "_")
        out_path = out_dir / f"{safe_model}_{stem}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not skip_existing_qas:
            out_path.write_text("", encoding="utf-8")
        elif not out_path.exists():
            out_path.write_text("", encoding="utf-8")
        outputs[model] = out_path

    if skip_existing_qas and qas:
        existing_sets = [_load_existing_qa_ids(outputs[m]) for m in models]
        existing_all = set.intersection(*existing_sets) if existing_sets else set()
        if existing_all:
            original_count = len(qas)
            qas = [qa for qa in qas if str(qa.get("qa_id") or "") not in existing_all]
            skipped_count = original_count - len(qas)
            if skipped_count > 0:
                print(f"[skip-existing-qas] skipped {skipped_count} QA(s); pending {len(qas)} QA(s).")

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(qas), desc="LLM-ZeroShot", unit="qa")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_qa,
                qa,
                table_idx,
                models,
                outputs,
                client,
                cfg,
                sleep_s,
                table_repr_format,
            ): qa
            for qa in qas
        }
        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            processed_count += 1
            try:
                qa_id = future.result()
                if pbar is not None:
                    pbar.set_postfix_str(f"qa_id={qa_id}")
                    pbar.update(1)
                elif processed_count % 5 == 0 or processed_count == len(qas):
                    print(f"[{processed_count}/{len(qas)}] processed qa_id={qa_id}")
            except Exception as e:
                if pbar is not None:
                    tqdm.write(f"QA Failed: {e!r}")
                    pbar.update(1)
                else:
                    print(f"QA Failed: {e!r}")

    if pbar is not None:
        pbar.close()

    for _, jsonl_path in outputs.items():
        try:
            _write_pretty_json_from_jsonl(jsonl_path)
            _write_meta_json(jsonl_path)
        except Exception as e:
            print(f"[warn] failed to write pretty JSON for {jsonl_path}: {e!r}", file=sys.stderr)

    return outputs
