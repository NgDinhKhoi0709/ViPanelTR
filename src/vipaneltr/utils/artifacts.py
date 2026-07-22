"""
Artifact manager for PanelTR-ViTabQA.

Handles saving/loading per-question/per-agent artifacts for resume, debug, and replay.

Artifact structure:
    {output_dir}/{run_id}/
    ├── parsed_tables/
    │   ├── {table_id}.json
    │   └── {table_id}.md
    ├── {qa_id}/
    │   ├── phase-1/
    │   │   ├── logician.json
    │   │   ├── calculator.json
    │   │   ├── verifier.json
    │   │   ├── structuralist.json
    │   │   └── synthesizer.json
    │   ├── phase-2/
    │   │   ├── logician.json
    │   │   ├── calculator.json
    │   │   ├── verifier.json
    │   │   ├── structuralist.json
    │   │   └── synthesizer.json
    │   ├── phase-3/
    │   │   ├── presentations.json         # Random order + content per agent
    │   │   ├── deliberation_round_1.json  # REVISE/MAINTAIN per agent
    │   │   ├── consensus_checks.json      # Results at each step
    │   │   ├── voting.json                # If majority voting used
    │   │   ├── sigma_final.json           # Final answer + rationale
    │   │   └── peer_review.json           # Full data (backward compat)
    │   └── phase-4/
    │       └── answer_formatter.json      # Formatted answer
    ├── results.json
    └── meta.json
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging import get_logger


logger = get_logger(__name__)

# 5 agents (no Explainer)
AGENT_NAMES = ["logician", "calculator", "verifier", "structuralist", "synthesizer"]


class ArtifactManager:
    """
    Manages per-question/per-agent artifacts for PanelTR pipeline.

    Thread-safe: multiple workers can call save/load concurrently.
    Supports resume by checking file existence before running agents.
    """

    def __init__(
        self,
        output_dir: str,
        run_id: str,
        model_name: str = "unknown",
        overwrite: bool = False,
    ):
        """
        Initialize artifact manager.

        Args:
            output_dir: Base output directory
            run_id: Run identifier
            model_name: Model name for metadata
            overwrite: If True, overwrite existing artifacts; if False, skip existing
        """
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.model_name = model_name
        self.overwrite = overwrite

        # Parsed table cache directory
        self.tables_dir = self.output_dir / "parsed_tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._table_cache: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _get_question_dir(self, qa_id: str) -> Path:
        """Get directory for a specific question."""
        return self.output_dir / qa_id

    def _get_phase_dir(self, qa_id: str, phase: int) -> Path:
        """Get directory for a specific phase of a question."""
        return self._get_question_dir(qa_id) / f"phase-{phase}"

    def _get_agent_file(self, qa_id: str, phase: int, agent_name: str) -> Path:
        """Get file path for a specific agent's output."""
        return self._get_phase_dir(qa_id, phase) / f"{agent_name.lower()}.json"

    # ------------------------------------------------------------------
    # Parsed table caching
    # ------------------------------------------------------------------

    def save_parsed_table(
        self,
        table_id: str,
        table_str: str,
        repr_dict: Dict[str, Any],
        has_merged_cells: bool,
    ) -> None:
        """
        Persist a parsed table to disk and in-memory cache.

        Saves two files per table:
        - {table_id}.json — structured representation + metadata
        - {table_id}.md — raw markdown/text for direct prompt injection
        """
        entry = {
            "table_id": table_id,
            "table_str": table_str,
            "has_merged_cells": has_merged_cells,
            "representation": repr_dict,
            "saved_at": datetime.now().isoformat(),
        }

        with self._lock:
            self._table_cache[table_id] = entry

        # JSON artifact
        json_path = self.tables_dir / f"{table_id}.json"
        self._write_json(json_path, entry)

        # Markdown artifact (plain text for LLM prompt)
        md_path = self.tables_dir / f"{table_id}.md"
        md_path.write_text(table_str, encoding="utf-8")

    def load_parsed_table(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a cached parsed table.

        Checks in-memory cache first, then disk.

        Returns:
            Dict with keys: table_str, has_merged_cells, representation
            or None if not found.
        """
        with self._lock:
            if table_id in self._table_cache:
                return self._table_cache[table_id]

        json_path = self.tables_dir / f"{table_id}.json"
        if json_path.exists():
            data = self._read_json(json_path)
            with self._lock:
                self._table_cache[table_id] = data
            return data

        return None

    # ------------------------------------------------------------------
    # Per-agent artifact management
    # ------------------------------------------------------------------

    def has_agent_artifact(self, qa_id: str, phase: int, agent_name: str) -> bool:
        """Check if an agent's artifact already exists (for resume)."""
        path = self._get_agent_file(qa_id, phase, agent_name)
        return path.exists()

    def should_run_agent(self, qa_id: str, phase: int, agent_name: str) -> bool:
        """
        Determine if an agent should run.

        Returns True if:
        - overwrite is True, OR
        - artifact does not exist
        """
        if self.overwrite:
            return True
        return not self.has_agent_artifact(qa_id, phase, agent_name)

    def get_missing_agents(self, qa_id: str, phase: int) -> List[str]:
        """
        Get list of agents that need to run for a phase.

        Args:
            qa_id: Question ID
            phase: Phase number (1 or 2)

        Returns:
            List of agent names that don't have artifacts yet
        """
        if self.overwrite:
            return AGENT_NAMES.copy()

        missing = []
        for agent in AGENT_NAMES:
            if not self.has_agent_artifact(qa_id, phase, agent):
                missing.append(agent)
        return missing

    def save_agent_artifact(
        self,
        qa_id: str,
        phase: int,
        agent_name: str,
        data: Dict[str, Any],
        question: str = "",
    ) -> str:
        """
        Save a single agent's output for a specific question and phase.

        Args:
            qa_id: Question ID
            phase: Phase number (1 or 2)
            agent_name: Agent name (e.g., "logician")
            data: Agent output data
            question: Original question text (for metadata)

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, phase)
        phase_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "metadata": {
                "question_id": qa_id,
                "phase": phase,
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "run_id": self.run_id,
            },
            "question": question,
            "data": data,
        }

        file_path = self._get_agent_file(qa_id, phase, agent_name)
        self._write_json(file_path, artifact)
        logger.debug(f"Saved artifact: {file_path}")
        return str(file_path)

    def load_agent_artifact(
        self, qa_id: str, phase: int, agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a single agent's artifact.

        Args:
            qa_id: Question ID
            phase: Phase number
            agent_name: Agent name

        Returns:
            Artifact dict with 'metadata', 'question', 'data' keys, or None
        """
        file_path = self._get_agent_file(qa_id, phase, agent_name)
        if not file_path.exists():
            return None
        return self._read_json(file_path)

    def load_phase_artifacts(
        self, qa_id: str, phase: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load all agent artifacts for a specific phase.

        Args:
            qa_id: Question ID
            phase: Phase number

        Returns:
            Dict mapping agent_name -> artifact data
        """
        results = {}
        for agent in AGENT_NAMES:
            artifact = self.load_agent_artifact(qa_id, phase, agent)
            if artifact is not None:
                results[agent] = artifact.get("data", {})
        return results

    def has_complete_phase(self, qa_id: str, phase: int) -> bool:
        """Check if all 5 agents have artifacts for a phase."""
        for agent in AGENT_NAMES:
            if not self.has_agent_artifact(qa_id, phase, agent):
                return False
        return True

    # ------------------------------------------------------------------
    # Reconstruction helpers — rebuild dataclasses from saved dicts
    # ------------------------------------------------------------------

    @staticmethod
    def reconstruct_investigation(phase1_data: Dict[str, Dict[str, Any]]):
        """
        Rebuild MultiAgentInvestigationOutput from phase-1 artifacts.

        Args:
            phase1_data: Dict mapping agent_name -> agent data

        Returns:
            MultiAgentInvestigationOutput
        """
        from ..core.investigation import (
            MultiAgentInvestigationOutput,
            PersonaInvestigationOutput,
        )

        output = MultiAgentInvestigationOutput()

        for persona_name, pdata in phase1_data.items():
            p = PersonaInvestigationOutput(persona_name=persona_name)

            # Analysis fields
            analysis = pdata.get("analysis", {})
            p.complexity = analysis.get("complexity", "basic")
            p.question_type = analysis.get("question_type", "retrieval")
            p.key_observations = analysis.get("key_observations", [])
            p.potential_risks = analysis.get("potential_risks", [])
            p.answerable_assessment = analysis.get("answerable_assessment", "answerable")

            # Solution fields
            solution = pdata.get("solution", {})
            p.solution_plan = solution.get("solution_plan", [])
            p.draft_answer = solution.get("draft_answer", "")
            p.evidence_cells = solution.get("evidence_cells", [])

            # Backward compatibility for old artifacts that still include verification block.
            verification = pdata.get("verification", {})
            if (not p.draft_answer) and isinstance(verification, dict):
                p.draft_answer = verification.get("refined_answer", "")
            if not p.evidence_cells and isinstance(verification, dict):
                p.evidence_cells = verification.get("refined_evidence", [])

            p.confidence = pdata.get("confidence", "medium")
            p.total_latency_ms = pdata.get("total_latency_ms", 0.0)

            # Raw outputs not needed for downstream phases
            p.analyze_output = None
            p.solve_output = None

            output.persona_results[persona_name] = p

        return output

    @staticmethod
    def reconstruct_self_review(data: Dict[str, Any]):
        """
        Rebuild SelfReviewOutput from a saved dict.

        Raw AgentOutput fields are set to None.
        """
        from ..core.self_review import SelfReviewOutput, SelfReviewRound

        output = SelfReviewOutput()
        output.final_verdict = data.get("final_verdict", "validated")
        output.final_answer = data.get("final_answer", "")
        output.final_evidence = data.get("final_evidence", [])
        output.final_confidence = data.get("final_confidence", "medium")
        output.num_rounds = data.get("num_rounds", 0)

        for rdata in data.get("rounds", []):
            issues_found = rdata.get("issues_found")
            if issues_found is None:
                issues_found = rdata.get("issues", [])
            confidence = rdata.get("confidence")
            if confidence is None:
                confidence = rdata.get("confidence_after_review", "medium")
            r = SelfReviewRound(
                round_num=rdata.get("round", 0),
                verdict=rdata.get("verdict", "uncertain"),
                issues_found=issues_found,
                revised_answer=rdata.get("revised_answer"),
                revised_evidence=rdata.get("revised_evidence", []),
                confidence=confidence,
                raw_output=None,
            )
            output.rounds.append(r)

        return output

    @staticmethod
    def reconstruct_peer_review(data: Dict[str, Any]):
        """
        Rebuild PeerReviewOutput from a saved dict.

        Raw AgentOutput fields are set to None (not needed for downstream).
        """
        from ..core.peer_review import (
            PeerReviewOutput,
            DissentingOpinion,
            PresentationResult,
            DeliberationResult,
            ConsensusCheck,
            SemanticConsensusCheck,
            VotingResult,
        )

        output = PeerReviewOutput()
        output.final_answer = data.get("final_answer", "")
        output.answerable = data.get("answerable", True)
        output.confidence = data.get("confidence", 0.0)
        output.consensus_reached = data.get("consensus_reached", False)
        output.final_rationale = data.get("final_rationale", "")
        output.decision_method = data.get("decision_method", "")
        output.vote_summary = data.get("vote_summary", {})
        output.total_latency_ms = data.get("total_latency_ms", 0.0)
        output.presentation_order = data.get("presentation_order", [])

        # Reconstruct presentations
        for p in data.get("presentations", []):
            output.presentations.append(
                PresentationResult(
                    persona=p.get("persona", ""),
                    answer=p.get("answer", ""),
                    answerable=p.get("answerable", True),
                    confidence=p.get("confidence", 0.5),
                    reasoning=p.get("reasoning", ""),
                    key_evidence=p.get("key_evidence", []),
                    raw_output=None,
                    adjusted_from_observation=p.get("adjusted_from_observation", False),
                )
            )

        # Reconstruct consensus checks
        for c in data.get("consensus_checks", []):
            output.consensus_checks.append(
                ConsensusCheck(
                    step=c.get("step", ""),
                    is_unanimous=c.get("is_unanimous", False),
                    answers=c.get("answers", {}),
                    unique_answers=c.get("unique_answers", []),
                    timestamp=c.get("timestamp", ""),
                )
            )

        # Reconstruct semantic consensus checks
        for sc in data.get("semantic_consensus_checks", []):
            output.semantic_consensus_checks.append(
                SemanticConsensusCheck(
                    step=sc.get("step", ""),
                    is_semantically_unanimous=sc.get("is_semantically_unanimous", False),
                    semantic_groups=sc.get("semantic_groups", []),
                    canonical_answer=sc.get("canonical_answer", ""),
                    original_answers=sc.get("original_answers", {}),
                    reasoning=sc.get("reasoning", ""),
                    timestamp=sc.get("timestamp", ""),
                )
            )

        # Reconstruct deliberation rounds
        for round_data in data.get("deliberation_rounds", []):
            round_results = []
            for d in round_data:
                round_results.append(
                    DeliberationResult(
                        persona=d.get("persona", ""),
                        stance=d.get("stance", "maintain"),
                        answer=d.get("answer", ""),
                        answerable=d.get("answerable", True),
                        confidence=d.get("confidence", 0.5),
                        justification=d.get("justification", ""),
                        raw_output=None,
                    )
                )
            output.deliberation_rounds.append(round_results)

        # Reconstruct voting result
        vr_data = data.get("voting_result")
        if vr_data:
            output.voting_result = VotingResult(
                answer_counts=vr_data.get("answer_counts", {}),
                winner=vr_data.get("winner", ""),
                winner_count=vr_data.get("winner_count", 0),
                total_votes=vr_data.get("total_votes", 0),
                is_tie=vr_data.get("is_tie", False),
                tie_break_method=vr_data.get("tie_break_method"),
                minority_opinions=vr_data.get("minority_opinions", []),
            )

        # Reconstruct dissenting opinions
        for d in data.get("dissenting_opinions", []):
            output.dissenting_opinions.append(
                DissentingOpinion(
                    persona=d.get("persona", ""),
                    dissent_answer=d.get("dissent_answer", ""),
                    dissent_reason=d.get("reason", ""),
                    critical_evidence=d.get("critical_evidence", []),
                    weight=d.get("weight", "minor"),
                )
            )

        return output

    # ------------------------------------------------------------------
    # Phase 3 artifact (single file, not per-agent)
    # ------------------------------------------------------------------

    def save_phase3_artifact(
        self, qa_id: str, data: Dict[str, Any], question: str = ""
    ) -> str:
        """
        Save Phase 3 (Peer-Review) artifact.

        Args:
            qa_id: Question ID
            data: PeerReviewOutput.to_dict()
            question: Original question text

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 3)
        phase_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "metadata": {
                "question_id": qa_id,
                "phase": 3,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "run_id": self.run_id,
            },
            "question": question,
            "data": data,
        }

        file_path = phase_dir / "peer_review.json"
        self._write_json(file_path, artifact)
        logger.debug(f"Saved phase 3 artifact: {file_path}")
        return str(file_path)

    def load_phase3_artifact(self, qa_id: str) -> Optional[Dict[str, Any]]:
        """
        Load Phase 3 artifact.

        Returns:
            Artifact dict with 'metadata', 'question', 'data' keys, or None
        """
        file_path = self._get_phase_dir(qa_id, 3) / "peer_review.json"
        if not file_path.exists():
            return None
        return self._read_json(file_path)

    def has_phase3_artifact(self, qa_id: str) -> bool:
        """Check if Phase 3 artifact exists."""
        return (self._get_phase_dir(qa_id, 3) / "peer_review.json").exists()

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def get_all_question_ids(self) -> List[str]:
        """Get list of all question IDs that have artifacts."""
        qa_ids = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name != "parsed_tables":
                qa_ids.append(item.name)
        return sorted(qa_ids)

    def get_phase_summary(self, qa_id: str) -> Dict[str, Any]:
        """
        Get summary of artifact status for a question.

        Returns:
            Dict with phase completion status and agent counts
        """
        summary = {
            "qa_id": qa_id,
            "phase_1": {
                "complete": self.has_complete_phase(qa_id, 1),
                "agents": [],
            },
            "phase_2": {
                "complete": self.has_complete_phase(qa_id, 2),
                "agents": [],
            },
            "phase_3": {
                "complete": self.has_phase3_artifact(qa_id),
            },
        }

        for agent in AGENT_NAMES:
            if self.has_agent_artifact(qa_id, 1, agent):
                summary["phase_1"]["agents"].append(agent)
            if self.has_agent_artifact(qa_id, 2, agent):
                summary["phase_2"]["agents"].append(agent)

        return summary

    # ------------------------------------------------------------------
    # Phase 3 detailed artifacts (PanelTR Standard Flow)
    # ------------------------------------------------------------------

    def save_phase3_presentations(
        self,
        qa_id: str,
        presentation_order: List[str],
        presentations: List[Dict[str, Any]],
    ) -> str:
        """
        Save presentations.json for Phase 3.

        Args:
            qa_id: Question ID
            presentation_order: List of persona names in random order
            presentations: List of PresentationResult.to_dict()

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 3)
        phase_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "presentation_order": presentation_order,
            "presentations": presentations,
            "timestamp": datetime.now().isoformat(),
        }

        file_path = phase_dir / "presentations.json"
        self._write_json(file_path, data)
        logger.debug(f"Saved phase 3 presentations: {file_path}")
        return str(file_path)

    def save_phase3_deliberation_round(
        self,
        qa_id: str,
        round_num: int,
        deliberations: List[Dict[str, Any]],
    ) -> str:
        """
        Save deliberation_round_{t}.json for Phase 3.

        Args:
            qa_id: Question ID
            round_num: Deliberation round number (1..t_max)
            deliberations: List of DeliberationResult.to_dict()

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 3)
        phase_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "round": round_num,
            "deliberations": deliberations,
            "timestamp": datetime.now().isoformat(),
        }

        file_path = phase_dir / f"deliberation_round_{round_num}.json"
        self._write_json(file_path, data)
        logger.debug(f"Saved phase 3 deliberation round {round_num}: {file_path}")
        return str(file_path)

    def save_phase3_consensus_checks(
        self,
        qa_id: str,
        consensus_checks: List[Dict[str, Any]],
        semantic_consensus_checks: List[Dict[str, Any]] = None,
    ) -> str:
        """
        Save consensus_checks.json for Phase 3.

        Args:
            qa_id: Question ID
            consensus_checks: List of ConsensusCheck.to_dict()
            semantic_consensus_checks: List of SemanticConsensusCheck.to_dict()

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 3)
        phase_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "checks": consensus_checks,
            "semantic_checks": semantic_consensus_checks or [],
            "timestamp": datetime.now().isoformat(),
        }

        file_path = phase_dir / "consensus_checks.json"
        self._write_json(file_path, data)
        logger.debug(f"Saved phase 3 consensus checks: {file_path}")
        return str(file_path)

    def save_phase3_voting(
        self,
        qa_id: str,
        voting_result: Dict[str, Any],
    ) -> str:
        """
        Save voting.json for Phase 3 (when majority voting is used).

        Args:
            qa_id: Question ID
            voting_result: VotingResult.to_dict()

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 3)
        phase_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "voting": voting_result,
            "timestamp": datetime.now().isoformat(),
        }

        file_path = phase_dir / "voting.json"
        self._write_json(file_path, data)
        logger.debug(f"Saved phase 3 voting: {file_path}")
        return str(file_path)

    def save_phase3_sigma_final(
        self,
        qa_id: str,
        final_answer: str,
        answerable: bool,
        confidence: float,
        decision_method: str,
        rationale: str,
    ) -> str:
        """
        Save sigma_final.json for Phase 3 (final decision summary).

        Args:
            qa_id: Question ID
            final_answer: The final answer
            answerable: Whether question is answerable
            confidence: Confidence score
            decision_method: "unanimous_consensus" or "majority_voting"
            rationale: Explanation of how decision was reached

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 3)
        phase_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "sigma_final": {
                "final_answer": final_answer,
                "answerable": answerable,
                "confidence": confidence,
                "decision_method": decision_method,
                "rationale": rationale,
            },
            "timestamp": datetime.now().isoformat(),
        }

        file_path = phase_dir / "sigma_final.json"
        self._write_json(file_path, data)
        logger.debug(f"Saved phase 3 sigma_final: {file_path}")
        return str(file_path)

    # ------------------------------------------------------------------
    # Phase 4 artifact (Answer Formatter)
    # ------------------------------------------------------------------

    def save_phase4_artifact(
        self, qa_id: str, data: Dict[str, Any], question: str = ""
    ) -> str:
        """
        Save Phase 4 (Answer Formatter) artifact.

        Args:
            qa_id: Question ID
            data: AnswerFormatterOutput.to_dict()
            question: Original question text

        Returns:
            Path to saved file
        """
        phase_dir = self._get_phase_dir(qa_id, 4)
        phase_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "metadata": {
                "question_id": qa_id,
                "phase": 4,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "run_id": self.run_id,
            },
            "question": question,
            "data": data,
        }

        file_path = phase_dir / "answer_formatter.json"
        self._write_json(file_path, artifact)
        logger.debug(f"Saved phase 4 artifact: {file_path}")
        return str(file_path)

    def load_phase4_artifact(self, qa_id: str) -> Optional[Dict[str, Any]]:
        """
        Load Phase 4 artifact.

        Returns:
            Artifact dict with 'metadata', 'question', 'data' keys, or None
        """
        file_path = self._get_phase_dir(qa_id, 4) / "answer_formatter.json"
        if not file_path.exists():
            return None
        return self._read_json(file_path)

    def has_phase4_artifact(self, qa_id: str) -> bool:
        """Check if Phase 4 artifact exists."""
        return (self._get_phase_dir(qa_id, 4) / "answer_formatter.json").exists()

    @staticmethod
    def reconstruct_answer_formatter(data: Dict[str, Any]):
        """
        Rebuild AnswerFormatterOutput from a saved dict.
        """
        from ..core.answer_formatter import AnswerFormatterOutput

        # Backward-compat: old runs stored formatted_answer as a string
        fa = data.get("formatted_answer", [])
        if isinstance(fa, list):
            formatted_answer = fa
        elif isinstance(fa, str) and fa:
            formatted_answer = [fa]
        else:
            formatted_answer = []

        return AnswerFormatterOutput(
            original_answer=data.get("original_answer", ""),
            question=data.get("question", ""),
            question_type=data.get("question_type", data.get("hints", [])),
            formatted_answer=formatted_answer,
            success=data.get("success", True),
            error=data.get("error"),
            latency_ms=data.get("latency_ms", 0.0),
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file with UTF-8 encoding."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        """Read JSON file with UTF-8 encoding."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
