"""
Command-line interface for PanelTR-ViTabQA.

Commands: prepare-data, infer, evaluate, and baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import Config, load_config, save_config
from .baseline.llm_client import GenConfig
from .baseline.run import run_batch_zeroshot
from .data.loader import DatasetLoader
from .data.validation import DatasetValidationError, validate_dataset
from .evaluation.evaluator import Evaluator
from .paths import dataset_dir as default_dataset_dir
from .paths import outputs_dir as default_outputs_dir
from .system.core.answer_formatter import AnswerFormatterOutput
from .system.core.investigation import MultiAgentInvestigationOutput
from .system.core.orchestrator import PanelTROrchestrator
from .system.core.peer_review import PeerReviewOutput
from .system.core.self_review import SelfReviewOutput
from .utils.logging import setup_logger, get_logger
from .utils.trace import StructuredOutputSaver
from .utils.artifacts import ArtifactManager


def parse_model_arg(model_str: str) -> tuple[str, str]:
    """Parse model argument in format 'provider/model_name'."""
    if "/" in model_str:
        provider, name = model_str.split("/", 1)
        return provider, name
    return "openai", model_str


PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


def require_provider_credentials(models: List[str]) -> int:
    """Report missing provider variables without exposing credential values."""
    missing: List[str] = []
    for model in models:
        provider, _ = parse_model_arg(model)
        key = PROVIDER_ENV_KEYS.get(provider.lower())
        if key and not os.environ.get(key) and key not in missing:
            missing.append(key)
    for key in missing:
        print(f"Missing required environment variable: {key}", file=sys.stderr)
    return 2 if missing else 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    try:
        report = validate_dataset(args.dataset_dir)
    except DatasetValidationError as exc:
        print(f"Dataset invalid: {exc}", file=sys.stderr)
        return 2
    counts = ", ".join(f"{name}={count}" for name, count in report.split_counts.items())
    print(f"Dataset valid: tables={report.table_count}, {counts}")
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    credential_status = require_provider_credentials(args.models)
    if credential_status:
        return credential_status
    config = GenConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    outputs = run_batch_zeroshot(
        qas_path=args.qas,
        tables_path=args.tables,
        models=args.models,
        output_dir=args.output_dir,
        limit=args.limit,
        sleep_s=args.sleep_s,
        gen_config=config,
        max_workers=args.max_workers,
        table_repr_format=args.table_repr,
        output_id=args.output_id,
        openrouter_provider=args.provider,
        skip_existing_qas=args.skip_existing_qas,
    )
    print("Wrote outputs:")
    for model, path in outputs.items():
        print(f"  {model}: {path}")
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    """Run inference command with full pipeline execution and artifact persistence."""
    logger = get_logger(__name__)

    # Load config
    config = load_config(args.config) if args.config else Config()

    # Override config with CLI args
    if args.model:
        provider, name = parse_model_arg(args.model)
        config.model.provider = provider
        config.model.name = name
    # OpenRouter provider routing via CLI, e.g. --provider deepinfra/bf16 nebius/fp8
    if args.provider:
        setattr(config.model, "openrouter_provider", {"only": args.provider})

    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir

    credential_status = require_provider_credentials(
        [f"{config.model.provider}/{config.model.name}"]
    )
    if credential_status:
        return credential_status

    # Override table representation
    if getattr(args, "table_repr", None):
        config.table.representation = str(args.table_repr).strip().lower()

    # Override threading config
    if args.threads is not None:
        config.threading.max_workers = args.threads
        if args.threads <= 1:
            config.threading.enabled = False

    # Setup logging
    setup_logger(
        level=config.logging.level,
        output_dir=config.logging.output_dir if config.logging.save_traces else None,
    )

    logger.info(f"Starting inference on QAs file: {args.qas}")
    logger.info(f"Model: {config.model.provider}/{config.model.name}")

    # Determine run_id
    run_id = args.run_id
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load dataset
    loader = DatasetLoader(config.dataset_dir)
    qas = loader.load_qas_path(args.qas)
    tables = loader.load_tables()

    # Limit samples if specified
    if args.n and args.n < len(qas):
        qas = qas[: args.n]
        logger.info(f"Limited to {args.n} samples")

    logger.info(f"Loaded {len(qas)} QA pairs")

    # Initialize components
    model_name = f"{config.model.provider}/{config.model.name}"
    artifact_mgr = ArtifactManager(
        args.output_dir,
        run_id,
        model_name=model_name,
        overwrite=getattr(args, 'overwrite', False),
    )
    output_saver = StructuredOutputSaver(args.output_dir, run_id=run_id)
    output_saver.set_config(config)

    # If enabled, reuse previous results.json entries for skipped QAs.
    existing_results_by_qa_id: Dict[str, Dict[str, Any]] = {}
    if getattr(args, "skip_existing_qas", False):
        results_path = Path(args.output_dir) / run_id / "results.json"
        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                predictions = []
                if isinstance(data, dict):
                    predictions = data.get("predictions", []) or []
                elif isinstance(data, list):
                    predictions = data
                if isinstance(predictions, list):
                    for item in predictions:
                        if isinstance(item, dict) and item.get("qa_id"):
                            existing_results_by_qa_id[str(item["qa_id"])] = item
            except Exception as e:
                logger.warning(f"Failed to load existing results.json for reuse: {e}")

    # Create orchestrator lazily so skip-only runs do not require API keys.
    orchestrator_lock = threading.Lock()
    orchestrator: Optional[PanelTROrchestrator] = None

    def get_orchestrator() -> PanelTROrchestrator:
        nonlocal orchestrator
        if orchestrator is not None:
            return orchestrator
        with orchestrator_lock:
            if orchestrator is None:
                orchestrator = PanelTROrchestrator(config)
        return orchestrator

    # ------------------------------------------------------------------
    # Prepare tasks
    # ------------------------------------------------------------------
    tasks: List[Tuple[Dict, Dict]] = []
    for qa in qas:
        table = tables.get(qa["table_id"])
        if table is None:
            logger.warning(f"Table not found: {qa['table_id']}")
            continue
        tasks.append((qa, table))

    def reconstruct_skipped_result(qa: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct a result from saved artifacts without calling the LLM.

        This is used when --skip-existing-qas is enabled and the QA directory already exists
        but results.json doesn't contain an entry for this qa_id (e.g., previous run crashed).
        """
        qa_id = str(qa.get("qa_id", ""))
        table_id = str(qa.get("table_id", ""))
        question = str(qa.get("question", ""))
        groundtruth = qa.get("answer", "")

        try:
            inv_output: Optional[MultiAgentInvestigationOutput] = None
            sr_output: Optional[Dict[str, SelfReviewOutput]] = None
            pr_output: Optional[PeerReviewOutput] = None
            af_output: Optional[AnswerFormatterOutput] = None

            if artifact_mgr.has_complete_phase(qa_id, phase=1):
                phase1_data = artifact_mgr.load_phase_artifacts(qa_id, phase=1)
                inv_output = ArtifactManager.reconstruct_investigation(phase1_data)

            if artifact_mgr.has_complete_phase(qa_id, phase=2):
                phase2_data = artifact_mgr.load_phase_artifacts(qa_id, phase=2)
                if phase2_data:
                    sr_output = {
                        agent_name: ArtifactManager.reconstruct_self_review(agent_data)
                        for agent_name, agent_data in phase2_data.items()
                    }

            if artifact_mgr.has_phase3_artifact(qa_id):
                phase3_artifact = artifact_mgr.load_phase3_artifact(qa_id)
                if phase3_artifact and isinstance(phase3_artifact.get("data"), dict):
                    pr_output = ArtifactManager.reconstruct_peer_review(phase3_artifact["data"])

            if artifact_mgr.has_phase4_artifact(qa_id):
                phase4_artifact = artifact_mgr.load_phase4_artifact(qa_id)
                if phase4_artifact and isinstance(phase4_artifact.get("data"), dict):
                    af_output = ArtifactManager.reconstruct_answer_formatter(phase4_artifact["data"])

            # Decide pred_answer and metadata without LLM calls.
            dissenting_opinions: List[Dict[str, Any]] = []
            if pr_output:
                pred_answer = pr_output.final_answer
                answerable = pr_output.answerable
                confidence = pr_output.confidence
                final_rationale = pr_output.final_rationale
                dissenting_opinions = [d.to_dict() for d in pr_output.dissenting_opinions]
            elif sr_output:
                sr_answer, sr_vote_count = PanelTROrchestrator._get_majority_self_review_answer(sr_output)
                answerable = PanelTROrchestrator._get_majority_self_review_answerable(sr_output)
                total_sr = max(len(sr_output), 1)
                pred_answer = sr_answer
                confidence = sr_vote_count / total_sr
                final_rationale = (
                    "Self-review completed without peer-review "
                    f"(reconstructed, {sr_vote_count}/{total_sr} persona agreement)"
                )
            elif af_output and af_output.original_answer:
                pred_answer = af_output.original_answer
                answerable = True
                confidence = 0.0
                final_rationale = "Reconstructed from phase-4 artifact only"
            elif inv_output:
                majority_answer, vote_count = inv_output.get_majority_answer()
                answerable, _ = inv_output.get_answerability_consensus()
                total = max(len(inv_output.persona_results), 1)
                pred_answer = majority_answer
                confidence = vote_count / total
                final_rationale = f"Majority vote ({vote_count}/{total} agents) (reconstructed)"
            else:
                pred_answer = "Null"
                answerable = False
                confidence = 0.0
                final_rationale = "Reconstruction failed: missing phase artifacts"

            formatted_answer: List[str]
            if af_output and af_output.success and af_output.formatted_answer:
                formatted_answer = af_output.formatted_answer
            else:
                formatted_answer = [pred_answer] if pred_answer else []

            if not answerable:
                pred_answer = "Null"
                formatted_answer = ["Null"]

            return {
                "qa_id": qa_id,
                "table_id": table_id,
                "question": question,
                "pred_answer": pred_answer,
                "formatted_answer": formatted_answer,
                "groundtruth": groundtruth,
                "answerable": answerable,
                "confidence": confidence,
                "final_rationale": final_rationale,
                "dissenting_opinions": dissenting_opinions,
                "trace": {},
            }
        except Exception as e:
            return {
                "qa_id": qa_id,
                "table_id": table_id,
                "question": question,
                "pred_answer": "Null",
                "formatted_answer": ["Null"],
                "groundtruth": groundtruth,
                "answerable": False,
                "confidence": 0.0,
                "final_rationale": f"Reconstruction error: {e}",
                "dissenting_opinions": [],
                "trace": {},
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Processing function (runs per QA)
    # ------------------------------------------------------------------
    def process_single(
        qa: Dict[str, Any], table: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single QA through the full pipeline."""
        qa_id = qa["qa_id"]
        table_id = qa["table_id"]
        question = qa["question"]
        hints = qa.get("hints", [])

        # Skip fully if QA folder already exists (no phases will run).
        if getattr(args, "skip_existing_qas", False):
            qa_dir = artifact_mgr.output_dir / str(qa_id)
            if qa_dir.exists() and qa_dir.is_dir():
                existing = existing_results_by_qa_id.get(str(qa_id))
                if isinstance(existing, dict):
                    out = dict(existing)
                    # Ensure required fields exist and stay consistent with current input QA.
                    out["qa_id"] = str(qa_id)
                    out["table_id"] = str(table_id)
                    out["question"] = str(question)
                    out["groundtruth"] = qa.get("answer", out.get("groundtruth", ""))
                    out.setdefault("formatted_answer", [out.get("pred_answer", "")])
                    out.setdefault("answerable", True)
                    out.setdefault("confidence", 0.0)
                    out.setdefault("final_rationale", "")
                    out.setdefault("dissenting_opinions", [])
                    out.setdefault("trace", {})
                    return out
                return reconstruct_skipped_result(qa)

        # In parallel mode, stagger API calls; sequential mode sleeps in the loop
        if getattr(args, "sleep", 0) > 0 and config.threading.enabled and config.threading.max_workers > 1:
            time.sleep(args.sleep)

        try:
            # --- Table parsing (cache-aware) ---
            cached_table = artifact_mgr.load_parsed_table(table_id)
            if cached_table:
                table_str = cached_table["table_str"]
                has_merged_cells = cached_table["has_merged_cells"]
            else:
                orc = get_orchestrator()
                table_str, has_merged_cells, repr_dict = orc.prepare_table(table)
                artifact_mgr.save_parsed_table(
                    table_id, table_str, repr_dict, has_merged_cells
                )

            # Ensure orchestrator exists for any phase execution / finalize.
            orc = get_orchestrator()

            # --- Phase 1: Investigation ---
            inv_output: Optional[MultiAgentInvestigationOutput] = None

            inv_output = orc.run_phase1(question, table_str, hints, qa_id=qa_id)
            for agent_name, agent_result in inv_output.persona_results.items():
                artifact_mgr.save_agent_artifact(
                    qa_id=qa_id,
                    phase=1,
                    agent_name=agent_name,
                    data=agent_result.to_dict(),
                    question=question,
                )

            unanswerable_count = sum(
                1
                for result in inv_output.persona_results.values()
                if result.answerable_assessment == "unanswerable"
            )
            if unanswerable_count >= config.paneltr.unanswerable_threshold:
                total_agents = max(len(inv_output.persona_results), 1)
                return {
                    "qa_id": qa_id,
                    "table_id": table_id,
                    "question": question,
                    "pred_answer": "Null",
                    "formatted_answer": ["Null"],
                    "groundtruth": qa.get("answer", ""),
                    "answerable": False,
                    "confidence": 0.0,
                    "final_rationale": (
                        "Phase 1 unanswerable threshold reached "
                        f"({unanswerable_count}/{total_agents})"
                    ),
                    "dissenting_opinions": [],
                    "trace": {},
                    **orc.llm_client.get_usage(qa_id),
                }

            # --- Phase 2: Self-Review (optional) ---
            sr_output: Optional[Dict[str, SelfReviewOutput]] = None

            if config.paneltr.enable_self_review:
                sr_output = orc.run_phase2(question, table_str, inv_output, qa_id=qa_id)
                if sr_output is not None:
                    for agent_name, persona_sr_output in sr_output.items():
                        artifact_mgr.save_agent_artifact(
                            qa_id=qa_id,
                            phase=2,
                            agent_name=agent_name,
                            data=persona_sr_output.to_dict(),
                            question=question,
                        )

            # --- Phase 3: Peer-Review ---
            pr_output: Optional[PeerReviewOutput] = None

            if config.paneltr.enable_peer_review:
                pr_output = orc.run_phase3(
                    question, table_str, inv_output, sr_output, hints, has_merged_cells, qa_id=qa_id
                )
                # Save Phase 3 artifacts (detailed + backward-compat)
                pr_dict = pr_output.to_dict()

                # 1. Save presentations (random order + content)
                artifact_mgr.save_phase3_presentations(
                    qa_id=qa_id,
                    presentation_order=pr_output.presentation_order,
                    presentations=[p.to_dict() for p in pr_output.presentations],
                )

                # 2. Save consensus checks
                artifact_mgr.save_phase3_consensus_checks(
                    qa_id=qa_id,
                    consensus_checks=[c.to_dict() for c in pr_output.consensus_checks],
                    semantic_consensus_checks=[sc.to_dict() for sc in pr_output.semantic_consensus_checks],
                )

                # 3. Save deliberation rounds
                for round_idx, round_results in enumerate(pr_output.deliberation_rounds, start=1):
                    artifact_mgr.save_phase3_deliberation_round(
                        qa_id=qa_id,
                        round_num=round_idx,
                        deliberations=[d.to_dict() for d in round_results],
                    )

                # 4. Save voting (if used)
                if pr_output.voting_result is not None:
                    artifact_mgr.save_phase3_voting(
                        qa_id=qa_id,
                        voting_result=pr_output.voting_result.to_dict(),
                    )

                # 5. Save sigma_final (final decision summary)
                artifact_mgr.save_phase3_sigma_final(
                    qa_id=qa_id,
                    final_answer=pr_output.final_answer,
                    answerable=pr_output.answerable,
                    confidence=pr_output.confidence,
                    decision_method=pr_output.decision_method,
                    rationale=pr_output.final_rationale,
                )

                # 6. Save full peer_review.json (backward compatibility)
                artifact_mgr.save_phase3_artifact(
                    qa_id=qa_id,
                    data=pr_dict,
                    question=question,
                )

            # --- Phase 4: Answer Formatter ---
            af_output: Optional[AnswerFormatterOutput] = None
            
            # Get pred_answer from previous phases
            if pr_output:
                current_pred_answer = pr_output.final_answer
            elif sr_output:
                current_pred_answer, _ = PanelTROrchestrator._get_majority_self_review_answer(sr_output)
            else:
                current_pred_answer, _ = inv_output.get_majority_answer()
            
            majority_question_type, _ = inv_output.get_majority_question_type()
            formatter_question_type = [majority_question_type] if majority_question_type else []
            af_output = orc.run_phase4(
                question=question,
                pred_answer=current_pred_answer,
                question_type=formatter_question_type,
                qa_id=qa_id,
            )
            # Save Phase 4 artifact
            artifact_mgr.save_phase4_artifact(
                qa_id=qa_id,
                data=af_output.to_dict(),
                question=question,
            )

            # --- Finalize ---
            result = orc.finalize_result(
                qa_id=qa_id,
                table_id=table_id,
                question=question,
                investigation_output=inv_output,
                self_review_output=sr_output,
                peer_review_output=pr_output,
                answer_formatter_output=af_output,
            )
            result["groundtruth"] = qa.get("answer", "")
            return result

        except Exception as e:
            logger.error(f"Error processing {qa_id}: {e}")
            active_orc = locals().get("orc")
            usage = active_orc.llm_client.get_usage(qa_id) if active_orc else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
            return {
                "qa_id": qa_id,
                "table_id": qa.get("table_id", ""),
                "question": qa.get("question", ""),
                "pred_answer": "Null",
                "formatted_answer": ["Null"],
                "groundtruth": qa.get("answer", ""),
                "answerable": False,
                "confidence": 0.0,
                "final_rationale": f"Error: {str(e)}",
                "dissenting_opinions": [],
                "trace": {},
                "error": str(e),
                **usage,
            }

    # ------------------------------------------------------------------
    # Execute (parallel or sequential)
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []

    if config.threading.enabled and config.threading.max_workers > 1:
        logger.info(f"Running with {config.threading.max_workers} threads")

        with ThreadPoolExecutor(max_workers=config.threading.max_workers) as executor:
            futures = {
                executor.submit(process_single, qa, table): qa["qa_id"]
                for qa, table in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing"
            ):
                result = future.result()
                results.append(result)
                output_saver.add_result(result)
    else:
        logger.info("Running in sequential mode")
        for qa, table in tqdm(tasks, desc="Processing"):
            result = process_single(qa, table)
            results.append(result)
            output_saver.add_result(result)
            if getattr(args, "sleep", 0) > 0:
                time.sleep(args.sleep)

    # ------------------------------------------------------------------
    # Save all artifacts
    # ------------------------------------------------------------------
    # Note: Per-agent artifacts are saved immediately during processing
    # Only need to save final results here

    # Save structured output (results.json, meta.json, etc.)
    save_raw = args.save_raw or config.logging.save_raw_responses
    saved_files = output_saver.save_all(save_raw=save_raw)

    logger.info(f"Saved {len(results)} predictions:")
    for file_type, file_path in saved_files.items():
        logger.info(f"  - {file_type}: {file_path}")
    logger.info(f"  - artifacts: {artifact_mgr.output_dir}/{{qa_id}}/phase-{{1,2,3,4}}/")

    stats = output_saver._compute_stats()
    print("\n" + "=" * 60)
    print("TOKEN CONSUMPTION")
    print(
        f"  Total:   {stats.get('total_tokens', 0)} "
        f"(Prompt: {stats.get('total_prompt_tokens', 0)}, "
        f"Completion: {stats.get('total_completion_tokens', 0)})"
    )
    print(
        f"  Average: {stats.get('average_total_tokens', 0):.1f} "
        f"(Prompt: {stats.get('average_prompt_tokens', 0):.1f}, "
        f"Completion: {stats.get('average_completion_tokens', 0):.1f}) per question"
    )
    print("COST")
    print(f"  Total:   ${stats.get('total_cost_usd', 0.0):.6f}")
    print(f"  Average: ${stats.get('average_cost_usd', 0.0):.6f}/question")
    print("=" * 60)

    # Save config for reproducibility
    if config.logging.save_traces:
        config_path = Path(saved_files["meta"]).parent / "config.yaml"
        save_config(config, str(config_path))
        logger.info(f"  - config: {config_path}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation command."""
    logger = get_logger(__name__)

    if args.output_dir:
        args.output = str(Path(args.output_dir) / "evaluation.json")
    
    # Load config
    config = load_config(args.config) if args.config else Config()
    
    if args.fail_on_metric_error:
        config.evaluation.fail_on_metric_error = True
    
    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir
    
    # Override threading config
    if args.threads is not None:
        config.threading.max_workers = args.threads
        if args.threads <= 1:
            config.threading.enabled = False
    
    # Setup logging
    setup_logger(level=config.logging.level)
    
    logger.info(f"Evaluating predictions: {args.pred}")
    
    # Load predictions - support JSON (object/array) and JSONL formats
    predictions: List[Dict[str, Any]] = []
    pred_path = Path(args.pred)
    
    with open(pred_path, "r", encoding="utf-8") as f:
        content = f.read().lstrip("\ufeff").strip()
        if not content:
            predictions = []
        elif content.startswith("{"):
            # JSON object (new structured output)
            data = json.loads(content)
            if isinstance(data, dict) and "predictions" in data:
                predictions = data["predictions"]
            else:
                predictions = [data]
        elif content.startswith("["):
            # JSON array (common export format)
            data = json.loads(content)
            predictions = data
        else:
            # JSONL (legacy): one JSON object per line
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
    
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Log dataset size and expected performance
    if len(predictions) > 1000:
        logger.info(f"Large dataset ({len(predictions)} samples). Multithreading is highly recommended for better performance.")
    elif len(predictions) > 100:
        logger.info(f"Medium dataset ({len(predictions)} samples). Multithreading can provide moderate speedup.")
    else:
        logger.info(f"Small dataset ({len(predictions)} samples). Multithreading benefits may be minimal.")
    
    # Load ground truth
    loader = DatasetLoader(config.dataset_dir)
    qas = loader.load_qas_path(args.qas)
    qa_question_map = {qa.get("qa_id"): qa.get("question", "") for qa in qas}

    # Parse metrics
    metrics = [m.strip() for m in (args.metrics or "").split(",") if m.strip()]
    if not metrics:
        metrics = list(config.evaluation.metrics)

    # Normalize predictions to evaluator format
    # Prefer formatted_answer (list of variants) for multi-candidate evaluation
    normalized_predictions: List[Dict[str, Any]] = []
    # Strings that should be treated as "no answer" for answerability logic.
    # We still keep the original string in outputs for debugging.
    nullish_strings = {"", "null", "none", "nan", "n/a", "na"}

    def _as_answer_candidates(val: Any) -> List[str]:
        if val is None:
            return [""]
        if isinstance(val, list):
            cands = []
            for x in val:
                if x is None:
                    cands.append("")
                else:
                    cands.append(str(x))
            return cands or [""]
        return [str(val)]

    def _strip_candidate(s: str) -> str:
        return str(s).strip()

    def _is_effectively_empty(s: str) -> bool:
        s2 = str(s).strip()
        return (s2 == "") or (s2.lower() in nullish_strings)

    for p in predictions:
        fa = p.get("formatted_answer")
        if isinstance(fa, list) and fa:
            answer = [_strip_candidate(x) for x in _as_answer_candidates(fa)]
        elif isinstance(fa, str) and fa:
            answer = [_strip_candidate(fa)]  # backward compat: wrap single string
        else:
            # Common field variants across generators
            raw = (
                p.get("answer")
                if p.get("answer") is not None
                else p.get("pred_answer")
                if p.get("pred_answer") is not None
                else p.get("response_text")
                if p.get("response_text") is not None
                else p.get("response")
                if p.get("response") is not None
                else p.get("output_text")
                if p.get("output_text") is not None
                else ""
            )
            answer = [_strip_candidate(x) for x in _as_answer_candidates(raw)]

        pred_answerable = p.get("answerable")
        if not isinstance(pred_answerable, bool):
            pred_answerable = any(not _is_effectively_empty(a) for a in answer)
        normalized_predictions.append(
            {
                "qa_id": p.get("qa_id"),
                "answer": answer,
                "answerable": pred_answerable,
            }
        )

    # Normalize references to evaluator format
    references: List[Dict[str, Any]] = []
    for qa in qas:
        ref_answer = qa.get("answer", "")
        ref_answerable = qa.get("answerable")
        if ref_answerable is None:
            ref_answerable = str(ref_answer).strip().lower() not in {"null", ""}
        references.append(
            {
                "qa_id": qa.get("qa_id"),
                "answer": ref_answer,
                "answerable": ref_answerable,
                "hints": qa.get("hints", []),
            }
        )

    evaluator = Evaluator(
        metrics=metrics,
        use_vietnamese_tokenizer=config.evaluation.use_vietnamese_tokenizer,
        fail_on_metric_error=config.evaluation.fail_on_metric_error,
    )
    
    # Determine max_workers for evaluation
    max_workers = 1
    if config.threading.enabled and config.threading.max_workers > 1:
        max_workers = config.threading.max_workers
        logger.info(f"Running evaluation with {max_workers} threads")
        
        pass
    else:
        logger.info("Running evaluation in sequential mode")
        if config.threading.max_workers > 1:
            logger.info(f"Threading disabled despite max_workers={config.threading.max_workers}. Use --threads > 1 to enable.")
    
    import time
    start_time = time.time()
    
    eval_result = evaluator.evaluate(
        predictions=normalized_predictions,
        references=references,
        max_workers=max_workers,
    )
    
    evaluation_time = time.time() - start_time
    
    # Log performance results
    samples_per_second = len(predictions) / evaluation_time if evaluation_time > 0 else 0
    logger.info(f"Evaluation completed in {evaluation_time:.1f}s ({samples_per_second:.2f} samples/sec)")
    
    if max_workers > 1:
        estimated_sequential_time = evaluation_time * max_workers
        logger.info(f"Estimated speedup: {estimated_sequential_time/evaluation_time:.1f}x faster than sequential")
    
    # Clean up pre-loaded models to free memory (optional)
    evaluator.cleanup_models()
    
    if evaluator.metric_error_counts:
        for metric_name, error_count in evaluator.metric_error_counts.items():
            first_message = evaluator.metric_error_messages.get(metric_name, "unknown error")
            logger.warning(
                "Metric '%s' encountered %d runtime error(s); first error: %s",
                metric_name,
                error_count,
                first_message,
            )

    results = {
        "overall": eval_result.to_dict(),
        "samples": eval_result.sample_results,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for metric, value in results["overall"].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"{metric}:")
            for group_name, group_values in value.items():
                if isinstance(group_values, dict):
                    formatted_parts = []
                    for key, group_value in group_values.items():
                        if isinstance(group_value, float):
                            formatted_parts.append(f"{key}={group_value:.4f}")
                        else:
                            formatted_parts.append(f"{key}={group_value}")
                    print(f"  - {group_name}: " + ", ".join(formatted_parts))
                else:
                    print(f"  - {group_name}: {group_values}")
        else:
            print(f"{metric}: {value}")
    
    print("=" * 60)
    
    # Save detailed results if output specified
    if args.output:
        def normalize_filename_token(value: str) -> str:
            token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
            return token or "unknown"

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved detailed results to {output_path}")

        samples = eval_result.sample_results
        output_stem = output_path.stem
        output_suffix = output_path.suffix or ".json"

        if eval_result.f1_by_answerability is not None:
            f1_by_answerability_samples: Dict[str, List[Dict[str, Any]]] = {}
            for sample in samples:
                group = sample.get("answerability_group")
                score = sample.get("f1_for_answerability")
                if group is None or score is None:
                    continue
                f1_by_answerability_samples.setdefault(group, []).append(
                    {
                        "qa_id": sample.get("qa_id"),
                        "accuracy": score,
                        "question": qa_question_map.get(sample.get("qa_id"), ""),
                        "answer": sample.get("pred_answer"),
                        "groundtruth": sample.get("ref_answer"),
                    }
                )

            for group_name, group_samples in f1_by_answerability_samples.items():
                group_token = normalize_filename_token(group_name)
                group_path = output_path.parent / (
                    f"{output_stem}_f1_by_answerability_{group_token}{output_suffix}"
                )
                with open(group_path, "w", encoding="utf-8") as f:
                    json.dump(group_samples, f, ensure_ascii=False, indent=2)
                logger.info("Saved f1_by_answerability breakdown file: %s", group_path)

        if eval_result.rouge1_by_hint is not None:
            rouge1_by_hint_samples: Dict[str, List[Dict[str, Any]]] = {}
            for sample in samples:
                question_type = sample.get("question_type")
                score = sample.get("rouge1_for_hint")
                if question_type is None or score is None:
                    continue
                rouge1_by_hint_samples.setdefault(question_type, []).append(
                    {
                        "qa_id": sample.get("qa_id"),
                        "accuracy": score,
                        "question": qa_question_map.get(sample.get("qa_id"), ""),
                        "answer": sample.get("pred_answer"),
                        "groundtruth": sample.get("ref_answer"),
                    }
                )

            for question_type, grouped_samples in rouge1_by_hint_samples.items():
                type_token = normalize_filename_token(question_type)
                type_path = output_path.parent / (
                    f"{output_stem}_rouge1_by_hint_{type_token}{output_suffix}"
                )
                with open(type_path, "w", encoding="utf-8") as f:
                    json.dump(grouped_samples, f, ensure_ascii=False, indent=2)
                logger.info("Saved rouge1_by_hint breakdown file: %s", type_path)
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public ViPanelTR command-line parser."""
    parser = argparse.ArgumentParser(
        prog="vipaneltr",
        description="ViPanelTR multi-agent system for Vietnamese Table QA",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prepare_parser = subparsers.add_parser(
        "prepare-data", help="Validate the bundled or selected dataset"
    )
    prepare_parser.add_argument(
        "--dataset-dir", type=str, default=str(default_dataset_dir())
    )
    prepare_parser.set_defaults(handler=cmd_prepare_data)

    infer_parser = subparsers.add_parser("infer", help="Run multi-agent inference")
    infer_parser.add_argument("--qas", type=str, required=True,
                              help="Path to a qas_*.json file")
    infer_parser.add_argument("--n", type=int, default=None,
                              help="Number of samples to process (default: all)")
    infer_parser.add_argument("--model", type=str, default=None,
                              help="Model to use (format: provider/model_name)")
    infer_parser.add_argument(
        "--provider",
        type=str,
        nargs="+",
        default=None,
        help=(
            "OpenRouter provider routing list (mapped to provider.only), "
            "e.g. --provider deepinfra/bf16 nebius/fp8"
        ),
    )
    infer_parser.add_argument("--output-dir", type=str, default=str(default_outputs_dir()),
                              help="Output directory for structured results")
    infer_parser.add_argument("--run-id", type=str, default=None,
                              help="Run identifier (default: timestamp)")
    infer_parser.add_argument(
        "--table-repr",
        dest="table_repr",
        choices=["flattened", "structured"],
        default=None,
        help=(
            "Table representation injected into prompts. "
            "If omitted, uses value from config.table.representation."
        ),
    )
    infer_parser.add_argument("--save-raw", action="store_true",
                              help="Save raw LLM responses to separate file")
    infer_parser.add_argument("--config", type=str, default=None,
                              help="Path to config YAML file")
    infer_parser.add_argument("--dataset-dir", type=str, default=None,
                              help="Path to dataset directory")
    infer_parser.add_argument("--threads", type=int, default=None,
                              help="Number of parallel threads (default: from config)")
    infer_parser.add_argument("--overwrite", action="store_true",
                              help="Overwrite existing artifacts (default: skip existing)")
    infer_parser.add_argument(
        "--skip-existing-qas",
        action="store_true",
        help=(
            "Skip any QA entirely if outputs/<run_id>/<qa_id>/ already exists "
            "(reuse existing results/artifacts; no phases will run for that QA)."
        ),
    )
    infer_parser.add_argument(
        "--sleep",
        type=float,
        default=0,
        help="Seconds to sleep between questions (rate limiting)",
    )
    infer_parser.set_defaults(handler=cmd_infer)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    eval_parser.add_argument("--qas", type=str, required=True,
                             help="Path to QAs JSON file for ground truth")
    eval_parser.add_argument("--preds", "--pred", dest="pred", type=str, required=True,
                             help="Path to predictions JSONL file")
    eval_parser.add_argument("--metrics", type=str, default="f1,em,rouge1,meteor,f1_by_answerability,rouge1_by_hint",
                             help=(
                                 "Comma-separated metrics "
                                 "(f1,em,rouge1,meteor,"
                                 "f1_by_answerability,rouge1_by_hint)"
                             ))
    output_group = eval_parser.add_mutually_exclusive_group()
    output_group.add_argument("--output-dir", type=str, default=None,
                              help="Directory that receives evaluation.json")
    output_group.add_argument("--output", type=str, default=None,
                              help="Exact path for the detailed results JSON")
    eval_parser.add_argument("--config", type=str, default=None,
                             help="Path to config YAML file")
    # nli-checkpoint argument removed
    eval_parser.add_argument("--dataset-dir", type=str, default=None,
                             help="Path to dataset directory")
    eval_parser.add_argument(
        "--fail-on-metric-error",
        action="store_true",
        help="Fail immediately instead of silently falling back when a metric runtime error occurs",
    )
    eval_parser.add_argument("--threads", type=int, default=None,
                            help="Number of parallel threads (default: from config)")
    eval_parser.set_defaults(handler=cmd_evaluate)

    baseline_parser = subparsers.add_parser(
        "baseline", help="Run the zero-shot comparison baseline"
    )
    baseline_parser.add_argument(
        "--qas",
        default=str(default_dataset_dir() / "qas_test.json"),
    )
    baseline_parser.add_argument(
        "--tables",
        default=str(default_dataset_dir() / "table.json"),
    )
    baseline_parser.add_argument("--model", "--models", dest="models", nargs="+",
                                 default=["openai/gpt-4o-mini"])
    baseline_parser.add_argument("--id", dest="output_id", default=None)
    baseline_parser.add_argument("--table-repr", dest="table_repr",
                                 choices=["paneltr_flattened"], default="paneltr_flattened")
    baseline_parser.add_argument("--limit", "--n", dest="limit", type=int, default=None)
    baseline_parser.add_argument("--output-dir", dest="output_dir",
                                 default=str(default_outputs_dir() / "zeroshot"))
    baseline_parser.add_argument("--temperature", type=float, default=0.0)
    baseline_parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    baseline_parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=10000)
    baseline_parser.add_argument("--timeout", type=int, default=60)
    baseline_parser.add_argument("--sleep", "--sleep-s", dest="sleep_s", type=float, default=0.0)
    baseline_parser.add_argument("--threads", "--max-workers", dest="max_workers", type=int, default=4)
    baseline_parser.add_argument("--provider", nargs="+", default=None)
    baseline_parser.add_argument("--skip-existing-qas", action="store_true")
    baseline_parser.set_defaults(handler=cmd_run_baseline)

    return parser


def main(argv: List[str] | None = None) -> int:
    """Run the public CLI and return a process status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
