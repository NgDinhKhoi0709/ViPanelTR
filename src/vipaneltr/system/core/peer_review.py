"""
Peer-Review Phase implementation - PanelTR Standard Flow.

Phase 3 of PanelTR:
1. Individual Presentation (random order, sequential observation)
2. Consensus Check (5/5 identical → early exit)
3. Collective Deliberation (MAINTAIN/REVISE/SUPPORT/DISSENT)
4. Majority Voting with tie-break rules
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..agents.base_agent import AgentOutput
from ..agents.llm_client import LLMClient
from ..prompts.prompt_router import (
    get_peer_review_prompts,
    get_semantic_consensus_prompt,

)
from ..prompts.persona_lenses import (
    get_persona_lens,
    get_all_persona_names,
    format_prompt_with_lens,
)
from ...utils.json_parser import parse_json_response
from ...utils.llm_retry import call_llm_with_retry

from .investigation import MultiAgentInvestigationOutput, PersonaInvestigationOutput
from .self_review import SelfReviewOutput


@dataclass
class DissentingOpinion:
    """Represents a dissenting opinion from a persona."""
    persona: str
    dissent_answer: str
    dissent_reason: str
    critical_evidence: List[Dict[str, Any]] = field(default_factory=list)
    weight: str = "minor"  # "minor" or "significant"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona": self.persona,
            "dissent_answer": self.dissent_answer,
            "reason": self.dissent_reason,
            "critical_evidence": self.critical_evidence,
            "weight": self.weight,
        }


@dataclass
class PresentationResult:
    """Result of individual presentation."""
    persona: str
    answer: str
    answerable: bool
    confidence: float
    reasoning: str
    key_evidence: List[Dict[str, Any]]
    raw_output: Optional[AgentOutput] = None
    adjusted_from_observation: bool = False  # True if adjusted based on prior presentations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona": self.persona,
            "answer": self.answer,
            "answerable": self.answerable,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_evidence": self.key_evidence,
            "adjusted_from_observation": self.adjusted_from_observation,
            "raw_output": self.raw_output.to_dict() if self.raw_output else None,
        }


@dataclass
class DeliberationResult:
    """Result of deliberation for one agent."""
    persona: str
    stance: str  # "maintain", "revise", "support", or "dissent"
    answer: str
    answerable: bool
    confidence: float
    justification: str
    raw_output: Optional[AgentOutput] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona": self.persona,
            "stance": self.stance,
            "answer": self.answer,
            "answerable": self.answerable,
            "confidence": self.confidence,
            "justification": self.justification,
            "raw_output": self.raw_output.to_dict() if self.raw_output else None,
        }


@dataclass
class ConsensusCheck:
    """Result of consensus check."""
    step: str  # "after_presentation" or "after_deliberation_round_N"
    is_unanimous: bool
    answers: Dict[str, str]  # persona -> answer
    unique_answers: List[str]
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "is_unanimous": self.is_unanimous,
            "answers": self.answers,
            "unique_answers": self.unique_answers,
            "timestamp": self.timestamp,
        }


@dataclass
class SemanticConsensusCheck:
    """Result of semantic consensus check using LLM."""
    step: str  # "after_presentation" or "after_deliberation_round_N"
    is_semantically_unanimous: bool
    semantic_groups: List[Dict[str, Any]]  # [{"answers": [...], "canonical_answer": "..."}]
    canonical_answer: str  # The chosen representative answer
    original_answers: Dict[str, str]  # persona -> answer
    reasoning: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "is_semantically_unanimous": self.is_semantically_unanimous,
            "semantic_groups": self.semantic_groups,
            "canonical_answer": self.canonical_answer,
            "original_answers": self.original_answers,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


@dataclass
class VotingResult:
    """Result of majority voting."""
    answer_counts: Dict[str, int]
    winner: str
    winner_count: int
    total_votes: int
    is_tie: bool
    tie_break_method: Optional[str] = None  # "confidence" or "synthesizer_fallback"
    minority_opinions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_counts": self.answer_counts,
            "winner": self.winner,
            "winner_count": self.winner_count,
            "total_votes": self.total_votes,
            "is_tie": self.is_tie,
            "tie_break_method": self.tie_break_method,
            "minority_opinions": self.minority_opinions,
        }


@dataclass
class PeerReviewOutput:
    """Output from Peer-Review phase."""
    # Final decision
    final_answer: str = ""
    answerable: bool = True
    confidence: float = 0.0
    consensus_reached: bool = False
    final_rationale: str = ""
    decision_method: str = ""  # "unanimous_consensus" or "majority_voting"
    
    # Presentation results (in random order)
    presentation_order: List[str] = field(default_factory=list)
    presentations: List[PresentationResult] = field(default_factory=list)
    
    # Consensus checks
    consensus_checks: List[ConsensusCheck] = field(default_factory=list)
    
    # Semantic consensus checks
    semantic_consensus_checks: List[SemanticConsensusCheck] = field(default_factory=list)
    
    # Deliberation rounds
    deliberation_rounds: List[List[DeliberationResult]] = field(default_factory=list)
    
    # Voting (if used)
    voting_result: Optional[VotingResult] = None
    
    # Dissenting opinions
    dissenting_opinions: List[DissentingOpinion] = field(default_factory=list)
    
    # Timing
    total_latency_ms: float = 0.0
    
    # Legacy compatibility fields
    vote_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "answerable": self.answerable,
            "confidence": self.confidence,
            "consensus_reached": self.consensus_reached,
            "final_rationale": self.final_rationale,
            "decision_method": self.decision_method,
            "presentation_order": self.presentation_order,
            "presentations": [p.to_dict() for p in self.presentations],
            "consensus_checks": [c.to_dict() for c in self.consensus_checks],
            "semantic_consensus_checks": [sc.to_dict() for sc in self.semantic_consensus_checks],
            "deliberation_rounds": [
                [d.to_dict() for d in round_results]
                for round_results in self.deliberation_rounds
            ],
            "voting_result": self.voting_result.to_dict() if self.voting_result else None,
            "dissenting_opinions": [d.to_dict() for d in self.dissenting_opinions],
            "vote_summary": self.vote_summary,
            "total_latency_ms": self.total_latency_ms,
        }


class PeerReviewPhase:
    """
    Peer-Review phase of PanelTR (Standard Flow).
    
    Flow:
    1. Individual Presentation (random order, sequential - later agents observe earlier)
    2. Consensus Check (5/5 identical → sigma_final, exit)
    3. Collective Deliberation (t=1..t_max, MAINTAIN/REVISE/SUPPORT/DISSENT)
    4. After each round: Consensus Check
    5. If no consensus after t_max: Majority Voting with tie-break
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tmax_peer: int = 1,
        consensus_threshold: float = 1.0,
        enabled_personas: List[str] = None,
        parallel_personas: bool = True,
        enable_semantic_consensus: bool = True,
    ):
        """
        Initialize Peer-Review phase.
        
        Args:
            llm_client: LLM client for generating responses
            tmax_peer: Maximum deliberation rounds (paper default=1)
            consensus_threshold: Threshold for consensus (1.0 = 5/5 unanimous)
            enabled_personas: List of enabled persona names
            parallel_personas: Run personas in parallel (ignored for presentation, sequential required)
            enable_semantic_consensus: Enable semantic consensus check via LLM when strict consensus fails
        """
        self.llm_client = llm_client
        self.tmax_peer = tmax_peer
        self.consensus_threshold = consensus_threshold
        self.parallel_personas = parallel_personas
        self.enable_semantic_consensus = enable_semantic_consensus
        
        if enabled_personas is None:
            enabled_personas = get_all_persona_names()
        
        self.enabled_personas = [
            name for name in enabled_personas
            if name.lower() in get_all_persona_names()
        ]
    
    def run(
        self,
        question: str,
        table_repr: str,
        investigation_output: MultiAgentInvestigationOutput,
        self_review_output: Optional[Dict[str, SelfReviewOutput]] = None,
        hints: List[str] = None,
        has_merged_cells: bool = False,
        qa_id: str = "",
    ) -> PeerReviewOutput:
        """Run Peer-Review phase following PanelTR standard flow."""
        start_time = time.time()
        output = PeerReviewOutput()
        hints = hints or []

        effective_self_review_output: Dict[str, SelfReviewOutput] = {}
        if isinstance(self_review_output, dict):
            effective_self_review_output.update(self_review_output)
        fallback_sr = self._build_fallback_self_review_output(investigation_output)
        for persona_name in self.enabled_personas:
            if persona_name not in effective_self_review_output and persona_name in fallback_sr:
                effective_self_review_output[persona_name] = fallback_sr[persona_name]
        
        # ============================================================
        # 1. INDIVIDUAL PRESENTATION (Random Order, Sequential)
        # ============================================================
        # Randomize presentation order
        presentation_order = self.enabled_personas.copy()
        random.shuffle(presentation_order)
        output.presentation_order = presentation_order
        
        # Sequential presentation - later agents observe earlier ones
        prior_presentations: List[PresentationResult] = []
        for persona_name in presentation_order:
            inv_result = investigation_output.get_persona_result(persona_name)
            if inv_result is None:
                continue
            
            presentation = self._run_single_presentation(
                persona_name=persona_name,
                question=question,
                table_repr=table_repr,
                investigation_result=inv_result,
                self_review_output=effective_self_review_output,
                prior_presentations=prior_presentations,
                qa_id=qa_id,
            )
            output.presentations.append(presentation)
            prior_presentations.append(presentation)
        
        # ============================================================
        # 2. CONSENSUS CHECK (After Presentations)
        # ============================================================
        consensus_check = self._check_strict_consensus(
            output.presentations, step="after_presentation"
        )
        output.consensus_checks.append(consensus_check)
        
        if consensus_check.is_unanimous:
            # 5/5 identical → conclude immediately
            output.consensus_reached = True
            output.decision_method = "unanimous_consensus"
            output.final_answer = consensus_check.unique_answers[0]
            output.answerable = self._get_majority_answerable(output.presentations)
            output.confidence = self._get_average_confidence(output.presentations)
            output.final_rationale = "Unanimous consensus reached after individual presentations."
            output.vote_summary = {
                "method": "unanimous_consensus",
                "answer": output.final_answer,
                "vote_count": len(output.presentations),
            }
            output.total_latency_ms = (time.time() - start_time) * 1000
            return output
        
        # ============================================================
        # 2b. SEMANTIC CONSENSUS CHECK (After Presentations)
        # ============================================================
        if self.enable_semantic_consensus and len(consensus_check.unique_answers) > 1:
            semantic_check = self._check_semantic_consensus(
                answers=consensus_check.answers,
                question=question,
                step="after_presentation",
                qa_id=qa_id,
            )
            output.semantic_consensus_checks.append(semantic_check)
            
            if semantic_check.is_semantically_unanimous and semantic_check.canonical_answer:
                output.consensus_reached = True
                output.decision_method = "semantic_consensus"
                output.final_answer = semantic_check.canonical_answer
                output.answerable = self._get_majority_answerable(output.presentations)
                output.confidence = self._get_average_confidence(output.presentations)
                output.final_rationale = (
                    f"Semantic consensus reached after presentations. "
                    f"All answers are semantically equivalent. "
                    f"Reason: {semantic_check.reasoning}"
                )
                output.vote_summary = {
                    "method": "semantic_consensus",
                    "answer": output.final_answer,
                    "semantic_groups": semantic_check.semantic_groups,
                    "vote_count": len(output.presentations),
                }
                output.total_latency_ms = (time.time() - start_time) * 1000
                return output
        
        # ============================================================
        # 3. COLLECTIVE DELIBERATION (t = 1..t_max)
        # ============================================================
        current_answers: Dict[str, Union[PresentationResult, DeliberationResult]] = {
            p.persona: p for p in output.presentations
        }
        
        for round_num in range(1, self.tmax_peer + 1):
            round_results: List[DeliberationResult] = []

            round_input_answers = dict(current_answers)

            def run_deliberation(persona_name: str) -> DeliberationResult:
                return self._run_single_deliberation(
                    persona_name=persona_name,
                    question=question,
                    table_repr=table_repr,
                    current_answers=round_input_answers,
                    round_num=round_num,
                    qa_id=qa_id,
                )

            if self.parallel_personas and len(self.enabled_personas) > 1:
                completed_by_persona: Dict[str, DeliberationResult] = {}
                with ThreadPoolExecutor(max_workers=len(self.enabled_personas)) as executor:
                    futures = {
                        executor.submit(run_deliberation, persona_name): persona_name
                        for persona_name in self.enabled_personas
                    }
                    for future in as_completed(futures):
                        persona_name = futures[future]
                        completed_by_persona[persona_name] = future.result()

                for persona_name in self.enabled_personas:
                    delib_result = completed_by_persona[persona_name]
                    round_results.append(delib_result)
                    current_answers[persona_name] = delib_result
            else:
                for persona_name in self.enabled_personas:
                    delib_result = run_deliberation(persona_name)
                    round_results.append(delib_result)
                    current_answers[persona_name] = delib_result
            
            output.deliberation_rounds.append(round_results)
            
            # Consensus check after deliberation
            consensus_check = self._check_strict_consensus_from_deliberation(
                round_results, step=f"after_deliberation_round_{round_num}"
            )
            output.consensus_checks.append(consensus_check)
            
            if consensus_check.is_unanimous:
                output.consensus_reached = True
                output.decision_method = "unanimous_consensus"
                output.final_answer = consensus_check.unique_answers[0]
                output.answerable = self._get_majority_answerable_delib(round_results)
                output.confidence = self._get_average_confidence_delib(round_results)
                output.final_rationale = f"Unanimous consensus reached after deliberation round {round_num}."
                output.vote_summary = {
                    "method": "unanimous_consensus",
                    "answer": output.final_answer,
                    "round": round_num,
                    "vote_count": len(round_results),
                }
                output.total_latency_ms = (time.time() - start_time) * 1000
                return output
            
            # Semantic consensus check after deliberation
            if self.enable_semantic_consensus and len(consensus_check.unique_answers) > 1:
                semantic_check = self._check_semantic_consensus(
                    answers=consensus_check.answers,
                    question=question,
                    step=f"after_deliberation_round_{round_num}",
                    qa_id=qa_id,
                )
                output.semantic_consensus_checks.append(semantic_check)
                
                if semantic_check.is_semantically_unanimous and semantic_check.canonical_answer:
                    output.consensus_reached = True
                    output.decision_method = "semantic_consensus"
                    output.final_answer = semantic_check.canonical_answer
                    output.answerable = self._get_majority_answerable_delib(round_results)
                    output.confidence = self._get_average_confidence_delib(round_results)
                    output.final_rationale = (
                        f"Semantic consensus reached after deliberation round {round_num}. "
                        f"All answers are semantically equivalent. "
                        f"Reason: {semantic_check.reasoning}"
                    )
                    output.vote_summary = {
                        "method": "semantic_consensus",
                        "answer": output.final_answer,
                        "semantic_groups": semantic_check.semantic_groups,
                        "round": round_num,
                        "vote_count": len(round_results),
                    }
                    output.total_latency_ms = (time.time() - start_time) * 1000
                    return output
        
        # ============================================================
        # 4. MAJORITY VOTING (No consensus after t_max)
        # ============================================================
        final_round = output.deliberation_rounds[-1] if output.deliberation_rounds else None
        if final_round:
            voting_result = self._majority_voting(final_round)
        else:
            voting_result = self._majority_voting_from_presentations(output.presentations)
        
        output.voting_result = voting_result
        output.decision_method = "majority_voting"
        output.final_answer = voting_result.winner
        output.consensus_reached = False
        output.confidence = voting_result.winner_count / voting_result.total_votes
        output.final_rationale = (
            f"No unanimous consensus after {self.tmax_peer} deliberation rounds. "
            f"Decided by majority voting: {voting_result.winner_count}/{voting_result.total_votes} votes."
        )
        output.vote_summary = {
            "method": "majority_voting",
            "answer_counts": voting_result.answer_counts,
            "winner": voting_result.winner,
            "winner_count": voting_result.winner_count,
            "total_votes": voting_result.total_votes,
            "is_tie": voting_result.is_tie,
            "tie_break_method": voting_result.tie_break_method,
        }
        
        # Extract minority opinions as dissenting
        for minority in voting_result.minority_opinions:
            output.dissenting_opinions.append(DissentingOpinion(
                persona=minority["persona"],
                dissent_answer=minority["answer"],
                dissent_reason=minority.get("reasoning", ""),
                weight="minor",
            ))
        
        # Get answerable from majority
        if final_round:
            output.answerable = self._get_majority_answerable_delib(final_round)
        else:
            output.answerable = self._get_majority_answerable(output.presentations)
        
        output.total_latency_ms = (time.time() - start_time) * 1000
        return output
    
    # ----------------------------------------------------------------
    # Presentation
    # ----------------------------------------------------------------
    
    def _run_single_presentation(
        self,
        persona_name: str,
        question: str,
        table_repr: str,
        investigation_result: PersonaInvestigationOutput,
        self_review_output: Optional[Dict[str, SelfReviewOutput]],
        prior_presentations: List[PresentationResult],
        qa_id: str,
    ) -> PresentationResult:
        """Run presentation for a single persona with observation of prior presentations."""
        start_time = time.time()
        lens = get_persona_lens(persona_name)
        persona_sr = None
        if isinstance(self_review_output, dict):
            persona_sr = self_review_output.get(persona_name)

        if persona_sr is not None:
            forced_answer = (persona_sr.final_answer or "").strip()
            forced_verdict = (persona_sr.final_verdict or "").strip().lower()
            forced_confidence = self._normalize_confidence(
                getattr(persona_sr, "final_confidence", "medium")
            )
            forced_evidence = (
                persona_sr.final_evidence
                if isinstance(persona_sr.final_evidence, list)
                else []
            )
            source_note = "Phase 2 self-review"
        else:
            forced_answer = str(investigation_result.get_final_answer() or "").strip()
            forced_confidence = self._normalize_confidence(
                getattr(investigation_result, "confidence", "medium")
            )
            forced_evidence = (
                investigation_result.get_final_evidence()
                if isinstance(investigation_result.get_final_evidence(), list)
                else []
            )
            if not forced_answer and investigation_result.answerable_assessment == "unanswerable":
                forced_answer = "Null"
            forced_verdict = (
                "uncertain"
                if investigation_result.answerable_assessment == "unanswerable"
                else "validated"
            )
            source_note = "Phase 1 fallback (Phase 2 disabled/missing)"

        forced_answerable = bool(
            forced_verdict == "validated"
            and forced_answer
            and forced_answer.strip().lower() not in {"null", "none", "nan", "n/a"}
        )
        
        try:
            # Format prior presentations for context
            prior_context = ""
            if prior_presentations:
                prior_context = "\n\n### Ý kiến đã trình bày trước đó:\n"
                for pp in prior_presentations:
                    prior_context += f"- **{pp.persona}**: {pp.answer} (confidence: {pp.confidence})\n"
                prior_context += "\nBạn có thể giữ nguyên hoặc điều chỉnh quan điểm dựa trên thông tin trên."
            
            inv_result_str = json.dumps(
                investigation_result.to_dict(),
                ensure_ascii=False,
                indent=2
            )

            forced_ctx = (
                "\n\n### OVERRIDE INPUT (BẮT BUỘC)\n"
                f"- source: {source_note}\n"
                f"- proposed_answer MUST be EXACTLY: {forced_answer}\n"
                f"- final_verdict: {forced_verdict}\n"
                f"- final_confidence: {forced_confidence}\n"
                f"- final_evidence(JSON): {json.dumps(forced_evidence, ensure_ascii=False)}\n"
                "- Lưu ý: nếu bạn trả proposed_answer khác, hệ thống sẽ override lại theo input này.\n"
            )
            
            present_prompt, _, _ = get_peer_review_prompts(self.llm_client)
            prompt = format_prompt_with_lens(
                present_prompt,
                lens,
                question=question,
                table=table_repr,
                investigation_result=inv_result_str + forced_ctx + prior_context,
            )

            parsed, response, _ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["proposed_answer"],
                caller_name=f"{lens.name}_Present",
                qa_id=qa_id,
            )

            # Force Phase 2 answerability/confidence/answer + evidence consistency
            parsed = parsed or {}
            parsed["proposed_answer"] = forced_answer
            parsed["answerable"] = forced_answerable
            parsed["confidence"] = forced_confidence

            phase2_key_evidence = self._phase2_evidence_to_key_evidence(forced_evidence)
            if phase2_key_evidence:
                key_evidence = parsed.get("key_evidence", [])
                if not self._key_evidence_is_subset_of_phase2(key_evidence, forced_evidence):
                    parsed["key_evidence"] = phase2_key_evidence
            
            latency = (time.time() - start_time) * 1000
            
            agent_output = AgentOutput(
                data=parsed,
                raw_response=response,
                agent_name=f"{lens.name}_Present",
                success=True,
                latency_ms=latency,
            )
            
            return PresentationResult(
                persona=persona_name,
                answer=forced_answer,
                answerable=forced_answerable,
                confidence=forced_confidence,
                reasoning=parsed.get("reasoning", parsed.get("position_summary", "")),
                key_evidence=parsed.get("key_evidence", phase2_key_evidence),
                raw_output=agent_output,
                adjusted_from_observation=len(prior_presentations) > 0,
            )
            
        except Exception as e:
            phase2_key_evidence = self._phase2_evidence_to_key_evidence(forced_evidence)
            return PresentationResult(
                persona=persona_name,
                answer=forced_answer,
                answerable=forced_answerable,
                confidence=forced_confidence,
                reasoning=f"Error during presentation: {e}",
                key_evidence=phase2_key_evidence,
                adjusted_from_observation=False,
            )

    def _build_fallback_self_review_output(
        self,
        investigation_output: MultiAgentInvestigationOutput,
    ) -> Dict[str, SelfReviewOutput]:
        """Create synthetic self-review outputs from phase-1 investigation."""
        fallback: Dict[str, SelfReviewOutput] = {}
        for persona_name in self.enabled_personas:
            inv_result = investigation_output.get_persona_result(persona_name)
            if inv_result is None:
                continue
            answer = str(inv_result.get_final_answer() or "").strip()
            is_unanswerable = inv_result.answerable_assessment == "unanswerable"
            if not answer and is_unanswerable:
                answer = "Null"
            fallback[persona_name] = SelfReviewOutput(
                final_verdict="uncertain" if is_unanswerable else "validated",
                final_answer=answer,
                final_evidence=inv_result.get_final_evidence() or [],
                final_confidence=str(getattr(inv_result, "confidence", "medium") or "medium"),
                rounds=[],
                num_rounds=0,
            )
        return fallback

    @staticmethod
    def _phase2_evidence_to_key_evidence(
        phase2_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert Phase 2 evidence format to Phase 3 key_evidence format.
        Phase 2 uses {row,col,value,role}; Phase 3 uses {row,col,value,importance}.
        """
        if not isinstance(phase2_evidence, list):
            return []
        out: List[Dict[str, Any]] = []
        for e in phase2_evidence:
            if not isinstance(e, dict):
                continue
            if e.get("row") is None or e.get("col") is None:
                continue
            out.append(
                {
                    "row": e.get("row"),
                    "col": e.get("col"),
                    "value": e.get("value", ""),
                    "importance": "critical",
                }
            )
        return out[:3]

    @staticmethod
    def _key_evidence_is_subset_of_phase2(
        key_evidence: Any,
        phase2_evidence: Any,
    ) -> bool:
        if not isinstance(key_evidence, list) or not isinstance(phase2_evidence, list):
            return False
        if not key_evidence:
            return True

        def _norm_val(v: Any) -> str:
            return str(v).strip().lower()

        phase2_set = set()
        for e in phase2_evidence:
            if not isinstance(e, dict):
                continue
            phase2_set.add((e.get("row"), e.get("col"), _norm_val(e.get("value"))))

        for ke in key_evidence:
            if not isinstance(ke, dict):
                return False
            tup = (ke.get("row"), ke.get("col"), _norm_val(ke.get("value")))
            if tup not in phase2_set:
                return False
        return True
    
    # ----------------------------------------------------------------
    # Deliberation
    # ----------------------------------------------------------------
    
    def _run_single_deliberation(
        self,
        persona_name: str,
        question: str,
        table_repr: str,
        current_answers: Dict[str, Union[PresentationResult, DeliberationResult]],
        round_num: int,
        qa_id: str,
    ) -> DeliberationResult:
        """Run deliberation for a single persona (REVISE or MAINTAIN)."""
        start_time = time.time()
        lens = get_persona_lens(persona_name)
        
        try:
            # Get current position
            my_position = current_answers.get(persona_name)
            if isinstance(my_position, PresentationResult):
                current_position = {
                    "answer": my_position.answer,
                    "confidence": my_position.confidence,
                    "reasoning": my_position.reasoning,
                }
            elif isinstance(my_position, DeliberationResult):
                current_position = {
                    "answer": my_position.answer,
                    "confidence": my_position.confidence,
                    "reasoning": my_position.justification,
                }
            else:
                current_position = {"answer": "", "confidence": 0.5, "reasoning": ""}
            
            # Build colleagues' opinions
            colleagues = []
            for name, pos in current_answers.items():
                if name.lower() == persona_name.lower():
                    continue
                if isinstance(pos, PresentationResult):
                    colleagues.append({
                        "persona": name,
                        "answer": pos.answer,
                        "confidence": pos.confidence,
                        "reasoning": pos.reasoning,
                    })
                elif isinstance(pos, DeliberationResult):
                    colleagues.append({
                        "persona": name,
                        "answer": pos.answer,
                        "confidence": pos.confidence,
                        "stance": pos.stance,
                        "reasoning": pos.justification,
                    })
            
            _, deliberate_prompt, _ = get_peer_review_prompts(self.llm_client)
            agent_name = f"{lens.name}_Deliberate_R{round_num}"
            prompt = format_prompt_with_lens(
                deliberate_prompt,
                lens,
                question=question,
                table=table_repr,
                current_position=json.dumps(current_position, ensure_ascii=False, indent=2),
                colleagues_opinions=json.dumps(colleagues, ensure_ascii=False, indent=2),
                discussion_history=f"Deliberation Round {round_num}",
            )

            parsed, response, ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["stance", "updated_answer"],
                caller_name=agent_name,
                qa_id=qa_id,
            )

            latency = (time.time() - start_time) * 1000

            agent_output = AgentOutput(
                data=parsed,
                raw_response=response,
                agent_name=agent_name,
                success=ok,
                latency_ms=latency,
            )
            
            stance = parsed.get("stance", "maintain").lower()
            if stance not in ("revise", "maintain", "support", "dissent"):
                stance = "maintain"
            
            return DeliberationResult(
                persona=persona_name,
                stance=stance,
                answer=self._normalize_answer(
                    parsed.get("updated_answer", current_position.get("answer", ""))
                ),
                answerable=parsed.get("answerable", True),
                confidence=self._normalize_confidence(parsed.get("confidence", 0.5)),
                justification=parsed.get("stance_justification", parsed.get("final_reasoning", "")),
                raw_output=agent_output,
            )
            
        except Exception as e:
            # Fallback: maintain previous position
            prev = current_answers.get(persona_name)
            if isinstance(prev, PresentationResult):
                return DeliberationResult(
                    persona=persona_name,
                    stance="maintain",
                    answer=prev.answer,
                    answerable=prev.answerable,
                    confidence=prev.confidence,
                    justification=f"Maintained due to error: {e}",
                )
            elif isinstance(prev, DeliberationResult):
                return DeliberationResult(
                    persona=persona_name,
                    stance="maintain",
                    answer=prev.answer,
                    answerable=prev.answerable,
                    confidence=prev.confidence,
                    justification=f"Maintained due to error: {e}",
                )
            else:
                return DeliberationResult(
                    persona=persona_name,
                    stance="maintain",
                    answer="",
                    answerable=True,
                    confidence=0.5,
                    justification=f"Error: {e}",
                )
    
    # ----------------------------------------------------------------
    # Consensus Check
    # ----------------------------------------------------------------
    
    def _check_strict_consensus(
        self,
        presentations: List[PresentationResult],
        step: str,
    ) -> ConsensusCheck:
        """Check if all 5 agents have identical answers."""
        answers = {p.persona: self._normalize_answer(p.answer) for p in presentations}
        unique_answers = list(set(answers.values()))
        
        return ConsensusCheck(
            step=step,
            is_unanimous=(len(unique_answers) == 1 and len(answers) == len(self.enabled_personas)),
            answers=answers,
            unique_answers=unique_answers,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    
    def _check_strict_consensus_from_deliberation(
        self,
        deliberations: List[DeliberationResult],
        step: str,
    ) -> ConsensusCheck:
        """Check consensus from deliberation results."""
        answers = {d.persona: self._normalize_answer(d.answer) for d in deliberations}
        unique_answers = list(set(answers.values()))
        
        return ConsensusCheck(
            step=step,
            is_unanimous=(len(unique_answers) == 1 and len(answers) == len(self.enabled_personas)),
            answers=answers,
            unique_answers=unique_answers,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
    
    def _check_semantic_consensus(
        self,
        answers: Dict[str, str],
        question: str,
        step: str,
        qa_id: str,
    ) -> SemanticConsensusCheck:
        """
        Check if answers are semantically equivalent using LLM.
        
        Called when strict consensus fails (unique_answers > 1). Uses an LLM
        to determine if the different answer strings convey the same meaning.
        Selects the canonical answer based on majority vote within
        the semantic group.
        
        Args:
            answers: Dict of persona -> answer (normalized)
            question: Original question text
            step: Step identifier (e.g., "after_presentation")
            
        Returns:
            SemanticConsensusCheck with grouping and canonical answer
        """
        unique_answers = list(set(answers.values()))
        
        # Count how many personas support each unique answer
        answer_counts = Counter(answers.values())
        vote_counts_str = "\n".join(
            f"- \"{ans}\": {cnt} persona(s)" 
            for ans, cnt in answer_counts.most_common()
        )
        candidate_str = "\n".join(f"- \"{ans}\"" for ans in unique_answers)
        
        semantic_prompt = get_semantic_consensus_prompt(self.llm_client)
        prompt = semantic_prompt.format(
            question=question,
            candidate_answers=candidate_str,
            answer_vote_counts=vote_counts_str,
        )
        
        try:
            parsed, raw_response, ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["all_equivalent"],
                caller_name="SemanticConsensus",
                temperature=0.0,
                qa_id=qa_id,
            )

            if not ok or parsed is None:
                return SemanticConsensusCheck(
                    step=step,
                    is_semantically_unanimous=False,
                    semantic_groups=[{"answers": [a], "canonical_answer": a} for a in unique_answers],
                    canonical_answer="",
                    original_answers=dict(answers),
                    reasoning="Failed to parse LLM response for semantic consensus after retries.",
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )

            all_equivalent = parsed.get("all_equivalent", False)
            semantic_groups = parsed.get("semantic_groups", [])
            reasoning = parsed.get("reasoning", "")
            
            # Determine canonical answer
            canonical_answer = ""
            if all_equivalent and len(semantic_groups) == 1:
                group = semantic_groups[0]
                # Use the canonical_answer suggested by LLM, but verify it's
                # actually the majority-voted one within the group
                group_answers = group.get("answers", unique_answers)
                
                # Find the answer with most persona votes
                best_answer = ""
                best_count = 0
                for ans in group_answers:
                    # Match against normalized answers
                    cnt = sum(
                        1 for v in answers.values()
                        if v == ans or v == self._normalize_answer(ans)
                    )
                    if cnt > best_count:
                        best_count = cnt
                        best_answer = ans
                
                canonical_answer = best_answer if best_answer else group.get("canonical_answer", unique_answers[0])
            elif semantic_groups:
                # Multiple groups — find the largest group
                largest_group = max(semantic_groups, key=lambda g: sum(
                    answer_counts.get(a, 0) for a in g.get("answers", [])
                ))
                canonical_answer = largest_group.get("canonical_answer", "")
            
            return SemanticConsensusCheck(
                step=step,
                is_semantically_unanimous=bool(all_equivalent),
                semantic_groups=semantic_groups,
                canonical_answer=canonical_answer,
                original_answers=dict(answers),
                reasoning=reasoning,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            
        except Exception as e:
            # On any error, fall back to non-equivalent
            return SemanticConsensusCheck(
                step=step,
                is_semantically_unanimous=False,
                semantic_groups=[{"answers": [a], "canonical_answer": a} for a in unique_answers],
                canonical_answer="",
                original_answers=dict(answers),
                reasoning=f"Error during semantic consensus check: {str(e)}",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
    
    # ----------------------------------------------------------------
    # Majority Voting
    # ----------------------------------------------------------------
    
    def _majority_voting(self, deliberations: List[DeliberationResult]) -> VotingResult:
        """Run majority voting with tie-break rules."""
        # Count votes
        answer_counts = Counter(self._normalize_answer(d.answer) for d in deliberations)
        
        if not answer_counts:
            return VotingResult(
                answer_counts={},
                winner="",
                winner_count=0,
                total_votes=0,
                is_tie=False,
            )
        
        # Find winner(s)
        max_count = max(answer_counts.values())
        winners = [ans for ans, cnt in answer_counts.items() if cnt == max_count]
        
        is_tie = len(winners) > 1
        winner = winners[0]
        tie_break_method = None
        
        if is_tie:
            # Tie-break 1: Highest total confidence
            confidence_sums = {}
            for ans in winners:
                confidence_sums[ans] = sum(
                    self._normalize_confidence(d.confidence) for d in deliberations
                    if self._normalize_answer(d.answer) == ans
                )
            
            max_conf = max(confidence_sums.values())
            conf_winners = [ans for ans, conf in confidence_sums.items() if conf == max_conf]
            
            if len(conf_winners) == 1:
                winner = conf_winners[0]
                tie_break_method = "confidence"
            else:
                # Tie-break 2: Synthesizer's answer
                synth_delib = next(
                    (d for d in deliberations if d.persona.lower() == "synthesizer"), 
                    None
                )
                if synth_delib and self._normalize_answer(synth_delib.answer) in conf_winners:
                    winner = self._normalize_answer(synth_delib.answer)
                    tie_break_method = "synthesizer_fallback"
                else:
                    winner = conf_winners[0]  # Deterministic fallback
                    tie_break_method = "first_in_list"
        
        # Collect minority opinions
        minority_opinions = []
        for d in deliberations:
            if self._normalize_answer(d.answer) != winner:
                minority_opinions.append({
                    "persona": d.persona,
                    "answer": d.answer,
                    "reasoning": d.justification,
                })
        
        return VotingResult(
            answer_counts=dict(answer_counts),
            winner=winner,
            winner_count=max_count,
            total_votes=len(deliberations),
            is_tie=is_tie,
            tie_break_method=tie_break_method,
            minority_opinions=minority_opinions,
        )
    
    def _majority_voting_from_presentations(
        self, presentations: List[PresentationResult]
    ) -> VotingResult:
        """Run majority voting from presentations (fallback)."""
        answer_counts = Counter(self._normalize_answer(p.answer) for p in presentations)
        
        if not answer_counts:
            return VotingResult(
                answer_counts={},
                winner="",
                winner_count=0,
                total_votes=0,
                is_tie=False,
            )
        
        max_count = max(answer_counts.values())
        winners = [ans for ans, cnt in answer_counts.items() if cnt == max_count]
        
        is_tie = len(winners) > 1
        winner = winners[0]
        tie_break_method = None
        
        if is_tie:
            # Tie-break by confidence
            confidence_sums = {}
            for ans in winners:
                confidence_sums[ans] = sum(
                    self._normalize_confidence(p.confidence) for p in presentations
                    if self._normalize_answer(p.answer) == ans
                )
            max_conf = max(confidence_sums.values())
            conf_winners = [ans for ans, conf in confidence_sums.items() if conf == max_conf]
            winner = conf_winners[0]
            tie_break_method = "confidence" if len(conf_winners) == 1 else "first_in_list"
        
        minority_opinions = []
        for p in presentations:
            if self._normalize_answer(p.answer) != winner:
                minority_opinions.append({
                    "persona": p.persona,
                    "answer": p.answer,
                    "reasoning": p.reasoning,
                })
        
        return VotingResult(
            answer_counts=dict(answer_counts),
            winner=winner,
            winner_count=max_count,
            total_votes=len(presentations),
            is_tie=is_tie,
            tie_break_method=tie_break_method,
            minority_opinions=minority_opinions,
        )
    
    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    
    @staticmethod
    def _normalize_confidence(value) -> float:
        """Convert confidence to float, handling string labels and numeric strings from LLM."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Try parsing as number first (e.g. "0.9", "0.5")
            try:
                return float(value)
            except ValueError:
                pass
            mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
            return mapping.get(value.strip().lower(), 0.5)
        return 0.5

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        return answer.strip().lower()
    
    def _get_majority_answerable(self, presentations: List[PresentationResult]) -> bool:
        """Get majority answerable from presentations."""
        if not presentations:
            return True
        votes = [p.answerable for p in presentations]
        return sum(votes) > len(votes) / 2
    
    def _get_majority_answerable_delib(self, deliberations: List[DeliberationResult]) -> bool:
        """Get majority answerable from deliberations."""
        if not deliberations:
            return True
        votes = [d.answerable for d in deliberations]
        return sum(votes) > len(votes) / 2
    
    def _get_average_confidence(self, presentations: List[PresentationResult]) -> float:
        """Get average confidence from presentations."""
        if not presentations:
            return 0.5
        return sum(self._normalize_confidence(p.confidence) for p in presentations) / len(presentations)
    
    def _get_average_confidence_delib(self, deliberations: List[DeliberationResult]) -> float:
        """Get average confidence from deliberations."""
        if not deliberations:
            return 0.5
        return sum(self._normalize_confidence(d.confidence) for d in deliberations) / len(deliberations)
