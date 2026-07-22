"""
Self-Review Phase implementation.

Phase 2 of PanelTR: Self-verification loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents.base_agent import AgentOutput
from ..agents.llm_client import LLMClient
from ..prompts.self_review import format_review_history
from ..prompts.investigation import format_evidence  # now in investigation.py (was atomic_functions.py)
from ..prompts.prompt_router import get_self_review_prompt
from ...utils.json_parser import parse_json_response
from ...utils.llm_retry import call_llm_with_retry, _stringify_answer


@dataclass
class SelfReviewRound:
    """Single round of self-review."""
    round_num: int
    verdict: str  # "validated" or "uncertain"
    issues_found: List[str] = field(default_factory=list)
    revised_answer: Optional[str] = None
    revised_evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: str = "medium"
    raw_output: Optional[AgentOutput] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "round": self.round_num,
            "verdict": self.verdict,
            "issues_found": self.issues_found,
            "revised_answer": self.revised_answer,
            "revised_evidence": self.revised_evidence,
            "confidence": self.confidence,
        }


@dataclass
class SelfReviewOutput:
    """Output from Self-Review phase."""
    
    # Final results
    final_verdict: str = "validated"
    final_answer: str = ""
    final_evidence: List[Dict[str, Any]] = field(default_factory=list)
    final_confidence: str = "medium"
    
    # Review history
    rounds: List[SelfReviewRound] = field(default_factory=list)
    num_rounds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tracing."""
        return {
            "final_verdict": self.final_verdict,
            "final_answer": self.final_answer,
            "final_evidence": self.final_evidence,
            "final_confidence": self.final_confidence,
            "num_rounds": self.num_rounds,
            "rounds": [r.to_dict() for r in self.rounds],
        }


class SelfReviewPhase:
    """
    Self-Review phase of PanelTR.
    
    Iteratively reviews and refines the answer until validated
    or maximum iterations reached.
    """
    
    def __init__(self, llm_client: LLMClient, tmax_self: int = 2):
        """
        Initialize Self-Review phase.
        
        Args:
            llm_client: LLM client for generating responses
            tmax_self: Maximum number of self-review iterations
        """
        self.llm_client = llm_client
        self.tmax_self = tmax_self
    
    def run(
        self,
        question: str,
        table_repr: str,
        current_answer: str,
        evidence: List[Dict[str, Any]],
        qa_id: str = "",
    ) -> SelfReviewOutput:
        """
        Run the Self-Review phase.
        
        Args:
            question: The question being answered
            table_repr: String representation of the table
            current_answer: Current answer from Investigation phase
            evidence: Evidence cells from Investigation phase
            
        Returns:
            SelfReviewOutput with final verdict and answer
        """
        output = SelfReviewOutput()
        output.final_answer = current_answer
        output.final_evidence = evidence
        
        review_history = []
        
        for round_num in range(1, self.tmax_self + 1):
            # Run review
            round_result = self._run_review(
                question=question,
                table_repr=table_repr,
                current_answer=output.final_answer,
                evidence=output.final_evidence,
                review_history=review_history,
                qa_id=qa_id,
            )
            
            # Create round record
            review_round = SelfReviewRound(
                round_num=round_num,
                verdict=round_result.get("verdict", "uncertain"),
                issues_found=round_result.get("issues_found", []),
                revised_answer=round_result.get("revised_answer"),
                revised_evidence=round_result.get("revised_evidence", []),
                confidence=round_result.get("confidence_after_review", "medium"),
                raw_output=round_result.get("_raw_output"),
            )
            
            output.rounds.append(review_round)
            review_history.append(review_round.to_dict())
            
            # Check verdict
            if review_round.verdict == "validated":
                output.final_verdict = "validated"
                output.final_confidence = review_round.confidence
                break
            
            # Update answer for next round
            if review_round.revised_answer:
                output.final_answer = _stringify_answer(review_round.revised_answer)
            if review_round.revised_evidence:
                output.final_evidence = review_round.revised_evidence
        
        output.num_rounds = len(output.rounds)
        
        # If never validated, keep last state
        if output.final_verdict != "validated":
            output.final_verdict = "uncertain"
        
        return output
    
    def _run_review(
        self,
        question: str,
        table_repr: str,
        current_answer: str,
        evidence: List[Dict[str, Any]],
        review_history: List[Dict[str, Any]],
        qa_id: str,
    ) -> Dict[str, Any]:
        """Run a single self-review iteration."""
        import time

        start_time = time.time()

        try:
            template = get_self_review_prompt(self.llm_client)
            prompt = template.format(
                question=question,
                table=table_repr,
                current_answer=current_answer,
                evidence=format_evidence(evidence),
                review_history=format_review_history(review_history),
            )

            parsed, response, ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["verdict"],
                caller_name="SelfReviewer",
                qa_id=qa_id,
            )

            # Normalise revised_answer from list to string
            ra = parsed.get("revised_answer")
            if ra is not None:
                parsed["revised_answer"] = _stringify_answer(ra)

            latency = (time.time() - start_time) * 1000

            parsed["_raw_output"] = AgentOutput(
                data=parsed,
                raw_response=response,
                agent_name="SelfReviewer",
                success=ok,
                latency_ms=latency,
            )

            return parsed

        except Exception as e:
            latency = (time.time() - start_time) * 1000

            return {
                "verdict": "uncertain",
                "issues_found": [f"Error during review: {str(e)}"],
                "_raw_output": AgentOutput(
                    data={},
                    raw_response="",
                    agent_name="SelfReviewer",
                    success=False,
                    error=str(e),
                    latency_ms=latency,
                ),
            }
