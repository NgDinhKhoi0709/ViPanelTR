from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...config import Config
from ..agents.llm_client import create_llm_client, LLMClient
from ...data.representation import TableRepresentation, create_representation
from .investigation import (
    InvestigationPhase,
    MultiAgentInvestigationOutput,
    PersonaInvestigationOutput,
)
from .self_review import SelfReviewPhase, SelfReviewOutput
from .peer_review import PeerReviewPhase, PeerReviewOutput, DissentingOpinion
from .answer_formatter import AnswerFormatterPhase, AnswerFormatterOutput
from ...utils.logging import get_logger
from ...utils.trace import TraceBuilder


@dataclass
class PanelTRResult:
    """Final result from PanelTR orchestration."""
    
    # Required output fields
    qa_id: str = ""
    table_id: str = ""
    question: str = ""
    pred_answer: str = ""
    formatted_answer: List[str] = field(default_factory=list)  # Formatted answer variants from Phase 4
    answerable: bool = True
    
    # Optional fields
    final_rationale: str = ""
    confidence: float = 0.0
    
    # Dissenting opinions
    dissenting_opinions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trace
    trace: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "qa_id": self.qa_id,
            "table_id": self.table_id,
            "question": self.question,
            "pred_answer": self.pred_answer,
            "formatted_answer": self.formatted_answer,
            "answerable": self.answerable,
            "final_rationale": self.final_rationale,
            "confidence": self.confidence,
            "dissenting_opinions": self.dissenting_opinions,
            "trace": self.trace,
        }


class PanelTROrchestrator:
    """
    Main orchestrator for PanelTR multi-agent system.
    
    Key features:
    1. All 5 agents conduct individual A→S investigation in parallel
    2. Self-Review is optional and per-agent
    3. Peer-Review reuses investigation results
    4. Dissenting opinions are preserved, not suppressed
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │ Phase 1: Individual Investigation (5 agents parallel)        │
    │   Each agent: A (Analyze) → S (Solve)                        │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Phase 2: Self-Review (optional, per agent)                   │
    │   Each agent reviews own findings                            │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Phase 3: Peer-Review (Panel Discussion)                      │
    │   Round 0: Present investigation findings                    │
    │   Rounds 1-N: Deliberate - can revise OR dissent            │
    │   Final: Synthesizer decides, acknowledges dissents          │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Config):
        """
        Initialize orchestrator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize LLM client
        self.llm_client = create_llm_client(config.model)
        
        # Initialize Investigation phase (6-agent parallel)
        self.investigation = InvestigationPhase(
            llm_client=self.llm_client,
            enabled_personas=config.personas.enabled,
            parallel_execution=config.threading.parallel_personas,
        )
        
        # Initialize Self-Review phase (optional)
        self.self_review = SelfReviewPhase(
            self.llm_client, 
            tmax_self=config.paneltr.tmax_self
        )
        self.enable_self_review = config.paneltr.enable_self_review
        
        # Initialize Peer-Review phase
        self.peer_review = PeerReviewPhase(
            llm_client=self.llm_client,
            tmax_peer=config.paneltr.tmax_peer,
            consensus_threshold=config.paneltr.consensus_threshold,
            enabled_personas=config.personas.enabled,
            parallel_personas=config.threading.parallel_personas,
            enable_semantic_consensus=config.paneltr.enable_semantic_consensus,
        )
        self.enable_peer_review = config.paneltr.enable_peer_review
        
        # Initialize Answer Formatter phase (Phase 4)
        self.answer_formatter = AnswerFormatterPhase(
            llm_client=self.llm_client,
        )
        
        # Trace builder
        self.trace_builder = TraceBuilder(config)
    
    def run(
        self,
        question: str,
        table: Dict[str, Any],
        qa_id: str = "",
        hints: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full PanelTR pipeline.
        
        Args:
            question: The question to answer
            table: Table data from dataset
            qa_id: QA identifier for tracing
            hints: Optional question type hints
            
        Returns:
            Dictionary with prediction and trace
        """
        start_time = time.time()
        
        result = PanelTRResult(
            qa_id=qa_id,
            table_id=table.get("table_id", ""),
            question=question,
        )
        
        hints = hints or []
        
        # Initialize trace
        self.trace_builder.start_trace(qa_id)
        
        try:
            # Create table representation
            table_repr = create_representation(
                table, 
                format=self.config.table.representation
            )
            table_str = table_repr.to_string()
            
            # Check for merged cells
            has_merged_cells = self._has_merged_cells(table)
            
            # === PHASE 1: Individual Investigation (5 agents) ===
            self.logger.debug(f"[{qa_id}] Starting 5-Agent Investigation phase")
            
            investigation_output = self.investigation.run(
                question=question,
                table_repr=table_str,
                hints=hints,
                qa_id=qa_id,
            )
            
            # Add investigation to trace
            self.trace_builder.add_investigation(investigation_output)
            
            # Also add multi-agent investigation details to trace
            self.trace_builder._raw_responses["multi_agent_investigation"] = {
                "persona_count": len(investigation_output.persona_results),
                "personas": list(investigation_output.persona_results.keys()),
                "answers": investigation_output.get_all_answers(),
            }

            unanswerable_count = sum(
                1
                for result in investigation_output.persona_results.values()
                if result.answerable_assessment == "unanswerable"
            )
            if unanswerable_count >= self.config.paneltr.unanswerable_threshold:
                total_agents = max(len(investigation_output.persona_results), 1)
                result.pred_answer = "Null"
                result.formatted_answer = ["Null"]
                result.answerable = False
                result.confidence = 0.0
                result.final_rationale = (
                    "Phase 1 unanswerable threshold reached "
                    f"({unanswerable_count}/{total_agents})"
                )
                result.trace = self.trace_builder.build()
                result.total_latency_ms = (time.time() - start_time) * 1000
                return result.to_dict()
            
            # === PHASE 2: Self-Review (Optional) ===
            self_review_output: Optional[Dict[str, SelfReviewOutput]] = None
            
            if self.enable_self_review:
                self.logger.debug(f"[{qa_id}] Starting Self-Review phase")
                
                self_review_output = self.run_phase2(
                    question=question,
                    table_str=table_str,
                    investigation_output=investigation_output,
                    qa_id=qa_id,
                )
                
                self.trace_builder.add_self_review(self_review_output)
            
            # === PHASE 3: Peer-Review ===
            peer_review_output = None
            
            if self.enable_peer_review:
                self.logger.debug(f"[{qa_id}] Starting Peer-Review phase")
                
                peer_review_output = self.peer_review.run(
                    question=question,
                    table_repr=table_str,
                    investigation_output=investigation_output,
                    self_review_output=self_review_output,
                    hints=hints,
                    has_merged_cells=has_merged_cells,
                    qa_id=qa_id,
                )
                
                self.trace_builder.add_peer_review(peer_review_output)
            
            # === Finalize Result ===
            if peer_review_output:
                # Use peer review result
                result.pred_answer = peer_review_output.final_answer
                result.answerable = peer_review_output.answerable
                result.confidence = peer_review_output.confidence
                result.final_rationale = peer_review_output.final_rationale
                result.dissenting_opinions = [
                    d.to_dict() for d in peer_review_output.dissenting_opinions
                ]
            elif self_review_output:
                # Use aggregated self-review result from per-persona outputs
                sr_answer, sr_vote_count = self._get_majority_self_review_answer(self_review_output)
                sr_answerable = self._get_majority_self_review_answerable(self_review_output)
                total_sr = max(len(self_review_output), 1)
                result.pred_answer = sr_answer
                result.answerable = sr_answerable
                result.confidence = sr_vote_count / total_sr
                result.final_rationale = (
                    f"Self-review completed without peer-review "
                    f"({sr_vote_count}/{total_sr} persona agreement)"
                )
            else:
                # Use investigation majority vote
                majority_answer, vote_count = investigation_output.get_majority_answer()
                answerable, _ = investigation_output.get_answerability_consensus()
                
                result.pred_answer = majority_answer
                result.answerable = answerable
                result.confidence = vote_count / max(len(investigation_output.persona_results), 1)
                result.final_rationale = f"Majority vote ({vote_count}/{len(investigation_output.persona_results)} agents)"
            
            # Handle unanswerable
            if not result.answerable:
                result.pred_answer = "Null"
            
        except Exception as e:
            self.logger.error(f"[{qa_id}] Error in pipeline: {e}")
            result.pred_answer = "Null"
            result.answerable = False
            result.confidence = 0.0
            result.final_rationale = f"Error: {str(e)}"
            self.trace_builder.add_error(str(e))
        
        # Build final trace
        result.trace = self.trace_builder.build()
        result.total_latency_ms = (time.time() - start_time) * 1000
        
        return result.to_dict()
    
    def _has_merged_cells(self, table: Dict[str, Any]) -> bool:
        """Check if table has merged cells."""
        # Check for spans in cells
        for row in table.get("data", []):
            for cell in row:
                if isinstance(cell, dict):
                    if cell.get("rowspan", 1) > 1 or cell.get("colspan", 1) > 1:
                        return True
        return False
    
    def run_investigation_only(
        self,
        question: str,
        table: Dict[str, Any],
        qa_id: str = "",
        hints: List[str] = None,
    ) -> MultiAgentInvestigationOutput:
        """
        Run only the Investigation phase.
        
        Useful for debugging or step-by-step execution.
        
        Args:
            question: The question to answer
            table: Table data from dataset
            qa_id: QA identifier
            hints: Optional hints
            
        Returns:
            MultiAgentInvestigationOutput with all persona results
        """
        self.logger.debug(f"[{qa_id}] Running Investigation only")
        
        table_repr = create_representation(
            table, 
            format=self.config.table.representation
        )
        table_str = table_repr.to_string()
        
        return self.investigation.run(
            question=question,
            table_repr=table_str,
            hints=hints or [],
            qa_id=qa_id,
        )
    
    def run_peer_review_only(
        self,
        question: str,
        table: Dict[str, Any],
        investigation_output: MultiAgentInvestigationOutput,
        self_review_output: Dict[str, SelfReviewOutput],
        qa_id: str = "",
        hints: List[str] = None,
    ) -> PeerReviewOutput:
        """
        Run only the Peer-Review phase.
        
        Requires investigation output from a previous step.
        
        Args:
            question: The question
            table: Table data
            investigation_output: Output from Investigation phase
            qa_id: QA identifier
            hints: Optional hints
            
        Returns:
            PeerReviewOutput with final decision
        """
        self.logger.debug(f"[{qa_id}] Running Peer-Review only")
        
        table_repr = create_representation(
            table, 
            format=self.config.table.representation
        )
        table_str = table_repr.to_string()
        
        has_merged_cells = self._has_merged_cells(table)
        
        return self.peer_review.run(
            question=question,
            table_repr=table_str,
            investigation_output=investigation_output,
            self_review_output=self_review_output,
            hints=hints or [],
            has_merged_cells=has_merged_cells,
            qa_id=qa_id,
        )

    # ------------------------------------------------------------------
    # Atomic phase methods — for phased CLI execution with artifact I/O
    # ------------------------------------------------------------------

    def prepare_table(
        self, table: Dict[str, Any]
    ) -> tuple[str, bool, Dict[str, Any]]:
        """
        Parse and represent a table, returning reusable artifacts.

        Separates table parsing from phase execution so the result
        can be cached to disk and reused across phases.

        Args:
            table: Raw table dict from dataset

        Returns:
            (table_str, has_merged_cells, repr_dict)
        """
        table_repr = create_representation(
            table,
            format=self.config.table.representation,
        )
        table_str = table_repr.to_string()
        repr_dict = table_repr.to_dict()
        has_merged_cells = self._has_merged_cells(table)
        return table_str, has_merged_cells, repr_dict

    def run_phase1(
        self,
        question: str,
        table_str: str,
        hints: List[str] = None,
        qa_id: str = "",
    ) -> MultiAgentInvestigationOutput:
        """
        Run Phase 1: Individual Investigation (5 agents parallel A→S).

        Args:
            question: The question to answer
            table_str: Pre-parsed table string for prompt injection
            hints: Optional question type hints

        Returns:
            MultiAgentInvestigationOutput with all persona results
        """
        return self.investigation.run(
            question=question,
            table_repr=table_str,
            hints=hints or [],
            qa_id=qa_id,
        )

    def run_phase2(
        self,
        question: str,
        table_str: str,
        investigation_output: MultiAgentInvestigationOutput,
        qa_id: str = "",
    ) -> Optional[Dict[str, SelfReviewOutput]]:
        """
        Run Phase 2: Self-Review (optional).

        Returns None if self-review is disabled.

        Args:
            question: The question
            table_str: Pre-parsed table string
            investigation_output: Output from Phase 1

        Returns:
            Dict[persona_name, SelfReviewOutput] or None
        """
        if not self.enable_self_review:
            return None

        persona_items = list(investigation_output.persona_results.items())
        sr_outputs: Dict[str, SelfReviewOutput] = {}

        def run_single_self_review(
            persona_name: str,
            persona_result: PersonaInvestigationOutput,
        ) -> tuple[str, SelfReviewOutput]:
            return persona_name, self.self_review.run(
                question=question,
                table_repr=table_str,
                current_answer=persona_result.get_final_answer(),
                evidence=persona_result.get_final_evidence(),
                qa_id=qa_id,
            )

        if self.config.threading.parallel_personas and len(persona_items) > 1:
            with ThreadPoolExecutor(max_workers=len(persona_items)) as executor:
                futures = {
                    executor.submit(run_single_self_review, persona_name, persona_result): persona_name
                    for persona_name, persona_result in persona_items
                }
                for future in as_completed(futures):
                    persona_name, self_review_output = future.result()
                    sr_outputs[persona_name] = self_review_output
        else:
            for persona_name, persona_result in persona_items:
                persona_name, self_review_output = run_single_self_review(
                    persona_name,
                    persona_result,
                )
                sr_outputs[persona_name] = self_review_output

        return sr_outputs

    def run_phase3(
        self,
        question: str,
        table_str: str,
        investigation_output: MultiAgentInvestigationOutput,
        self_review_output: Optional[Dict[str, SelfReviewOutput]] = None,
        hints: List[str] = None,
        has_merged_cells: bool = False,
        qa_id: str = "",
    ) -> PeerReviewOutput:
        """
        Run Phase 3: Peer-Review (Panel Discussion).

        Args:
            question: The question
            table_str: Pre-parsed table string
            investigation_output: Output from Phase 1
            hints: Optional question type hints
            has_merged_cells: Whether table has merged cells

        Returns:
            PeerReviewOutput with final decision and dissenting opinions
        """
        return self.peer_review.run(
            question=question,
            table_repr=table_str,
            investigation_output=investigation_output,
            self_review_output=self_review_output,
            hints=hints or [],
            has_merged_cells=has_merged_cells,
            qa_id=qa_id,
        )

    def run_phase4(
        self,
        question: str,
        pred_answer: str,
        question_type: List[str] = None,
        qa_id: str = "",
    ) -> AnswerFormatterOutput:
        """
        Phase 4: Answer Formatter.
        
        Formats the final answer to be concise and match groundtruth style.
        
        Args:
            question: The question
            pred_answer: Predicted answer from Phase 3
            question_type: Question type from Phase 1
            
        Returns:
            AnswerFormatterOutput with formatted_answer
        """
        return self.answer_formatter.run(
            question=question,
            pred_answer=pred_answer,
            question_type=question_type or [],
            qa_id=qa_id,
        )

    def finalize_result(
        self,
        qa_id: str,
        table_id: str,
        question: str,
        investigation_output: MultiAgentInvestigationOutput,
        self_review_output: Optional[Dict[str, SelfReviewOutput]] = None,
        peer_review_output: Optional[PeerReviewOutput] = None,
        answer_formatter_output: Optional[AnswerFormatterOutput] = None,
    ) -> Dict[str, Any]:
        """
        Build final PanelTRResult from phase outputs.

        Applies the decision priority: peer-review > self-review > majority vote.

        Args:
            qa_id: QA identifier
            table_id: Table identifier
            question: The question
            investigation_output: Output from Phase 1
            self_review_output: Output from Phase 2 (or None)
            peer_review_output: Output from Phase 3 (or None)
            answer_formatter_output: Output from Phase 4 (or None)

        Returns:
            Result dictionary ready for output
        """
        result = PanelTRResult(
            qa_id=qa_id,
            table_id=table_id,
            question=question,
        )

        if peer_review_output:
            result.pred_answer = peer_review_output.final_answer
            result.answerable = peer_review_output.answerable
            result.confidence = peer_review_output.confidence
            result.final_rationale = peer_review_output.final_rationale
            result.dissenting_opinions = [
                d.to_dict() for d in peer_review_output.dissenting_opinions
            ]
        elif self_review_output:
            sr_answer, sr_vote_count = self._get_majority_self_review_answer(self_review_output)
            sr_answerable = self._get_majority_self_review_answerable(self_review_output)
            total_sr = max(len(self_review_output), 1)
            result.pred_answer = sr_answer
            result.answerable = sr_answerable
            result.confidence = sr_vote_count / total_sr
            result.final_rationale = (
                f"Self-review completed without peer-review "
                f"({sr_vote_count}/{total_sr} persona agreement)"
            )
        else:
            majority_answer, vote_count = investigation_output.get_majority_answer()
            answerable, _ = investigation_output.get_answerability_consensus()
            result.pred_answer = majority_answer
            result.answerable = answerable
            total = max(len(investigation_output.persona_results), 1)
            result.confidence = vote_count / total
            result.final_rationale = (
                f"Majority vote ({vote_count}/{total} agents)"
            )

        # Apply formatted answer if available
        if answer_formatter_output and answer_formatter_output.success:
            result.formatted_answer = answer_formatter_output.formatted_answer
        else:
            result.formatted_answer = [result.pred_answer]  # Fallback

        if not result.answerable:
            result.pred_answer = "Null"
            result.formatted_answer = ["Null"]

        payload = result.to_dict()
        payload.update(self.llm_client.get_usage(qa_id))
        return payload

    @staticmethod
    def _get_majority_self_review_answer(
        self_review_outputs: Dict[str, SelfReviewOutput],
    ) -> tuple[str, int]:
        answer_counts: Dict[str, int] = {}
        originals: Dict[str, str] = {}
        for sr in self_review_outputs.values():
            answer = getattr(sr, "final_answer", "")
            s = str(answer).strip()
            if not s:
                continue
            normalized = s.lower()
            answer_counts[normalized] = answer_counts.get(normalized, 0) + 1
            originals.setdefault(normalized, s)
        if not answer_counts:
            return ("Null", 0)
        winner_norm, winner_count = max(answer_counts.items(), key=lambda x: x[1])
        return (originals.get(winner_norm, winner_norm), winner_count)

    @staticmethod
    def _get_majority_self_review_answerable(
        self_review_outputs: Dict[str, SelfReviewOutput],
    ) -> bool:
        if not self_review_outputs:
            return True
        votes = [
            str(getattr(sr, "final_verdict", "")).strip().lower() == "validated"
            for sr in self_review_outputs.values()
        ]
        return sum(votes) > len(votes) / 2
