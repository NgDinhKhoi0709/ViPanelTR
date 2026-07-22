"""
Investigation Phase implementation.

Phase 1 of PanelTR: All 5 agents conduct individual A→S investigation in parallel.
Each agent uses their unique "lens" to analyze and solve.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents.base_agent import AgentOutput
from ..agents.llm_client import LLMClient
from ..prompts.prompt_router import get_investigation_prompts
from ..prompts.persona_lenses import (
    PersonaLens,
    get_persona_lens,
    get_all_persona_names,
    format_prompt_with_lens,
)
from ...utils.json_parser import parse_json_response
from ...utils.llm_retry import call_llm_with_retry, _stringify_answer


@dataclass
class PersonaInvestigationOutput:
    """Output from a single persona's A→S investigation."""
    
    persona_name: str = ""
    
    # Analysis results (A)
    complexity: str = "basic"
    question_type: str = "what"
    key_observations: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    answerable_assessment: str = "answerable"
    
    # Solution results (S)
    solution_plan: List[str] = field(default_factory=list)
    draft_answer: str = ""
    evidence_cells: List[Dict[str, Any]] = field(default_factory=list)
    
    # Confidence
    confidence: str = "medium"
    
    # Raw outputs for tracing
    analyze_output: Optional[AgentOutput] = None
    solve_output: Optional[AgentOutput] = None
    # Timing
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tracing."""
        return {
            "persona": self.persona_name,
            "analysis": {
                "complexity": self.complexity,
                "question_type": self.question_type,
                "key_observations": self.key_observations,
                "potential_risks": self.potential_risks,
                "answerable_assessment": self.answerable_assessment,
            },
            "solution": {
                "solution_plan": self.solution_plan,
                "draft_answer": self.draft_answer,
                "evidence_cells": self.evidence_cells,
            },
            "confidence": self.confidence,
            "total_latency_ms": self.total_latency_ms,
        }
    
    def get_final_answer(self) -> str:
        """Get the final answer from Phase 1 (draft answer from Solve step)."""
        return self.draft_answer
    
    def get_final_evidence(self) -> List[Dict[str, Any]]:
        """Get the final evidence from Phase 1 (evidence from Solve step)."""
        return self.evidence_cells


@dataclass
class MultiAgentInvestigationOutput:
    """
    Output from all 5 agents' individual investigations.
    
    Contains the result from each persona's A→S pipeline.
    """
    
    # Results from each persona
    persona_results: Dict[str, PersonaInvestigationOutput] = field(default_factory=dict)
    
    # Aggregated metadata
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tracing."""
        return {
            "persona_count": len(self.persona_results),
            "personas": {
                name: result.to_dict()
                for name, result in self.persona_results.items()
            },
            "total_latency_ms": self.total_latency_ms,
        }
    
    def get_persona_result(self, persona_name: str) -> Optional[PersonaInvestigationOutput]:
        """Get result from a specific persona."""
        return self.persona_results.get(persona_name.lower())
    
    def get_all_answers(self) -> Dict[str, str]:
        """Get final answers from all personas."""
        return {
            name: result.get_final_answer()
            for name, result in self.persona_results.items()
        }

    def get_majority_question_type(self) -> tuple[str, int]:
        """Get the question type with most votes."""
        question_types: List[str] = []
        for r in self.persona_results.values():
            if r.question_type is None:
                continue
            qt = str(r.question_type).strip()
            if qt:
                question_types.append(qt)
        counts: Dict[str, int] = {}
        for question_type in question_types:
            normalized = question_type.lower()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1

        if not counts:
            return ("what", 0)

        winner_key, winner_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0]

        for result in self.persona_results.values():
            if result.question_type is None:
                continue
            qt = str(result.question_type).strip()
            if qt.lower() == winner_key:
                return (qt, winner_count)

        return (winner_key, winner_count)
    
    def get_majority_answer(self) -> tuple[str, int]:
        """Get the answer with most votes."""
        answers = self.get_all_answers()
        answer_counts: Dict[str, int] = {}
        originals: Dict[str, str] = {}
        for answer in answers.values():
            if answer is None:
                continue
            s = str(answer).strip()
            if not s:
                continue
            normalized = s.lower()
            answer_counts[normalized] = answer_counts.get(normalized, 0) + 1
            originals.setdefault(normalized, s)
        
        if not answer_counts:
            return ("Null", 0)
        
        winner_norm, winner_count = max(answer_counts.items(), key=lambda x: x[1])
        # Return first-seen original casing/format (always a string)
        return (originals.get(winner_norm, winner_norm), winner_count)
    
    def get_answerability_consensus(self) -> tuple[bool, float]:
        """
        Get answerability consensus.
        
        Returns:
            (majority_vote, agreement_ratio)
        """
        votes = [
            str(r.answerable_assessment).strip().lower() != "unanswerable"
            for r in self.persona_results.values()
        ]
        if not votes:
            return (True, 0.0)
        
        answerable_count = sum(1 for v in votes if v)
        ratio = answerable_count / len(votes)
        majority = ratio >= 0.5
        return (majority, ratio)


class InvestigationPhase:
    """
    Investigation phase of PanelTR.
    
    All 5 agents conduct individual A→S investigation:
    1. A (Analyze): Each agent analyzes from their unique perspective
    2. S (Solve): Each agent creates solution using their methodology
    
    Agents run in parallel by default for efficiency.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        enabled_personas: List[str] = None,
        parallel_execution: bool = True,
    ):
        """
        Initialize Investigation phase.
        
        Args:
            llm_client: LLM client for generating responses
            enabled_personas: List of persona names to use (default: all 5)
            parallel_execution: Run agents in parallel
        """
        self.llm_client = llm_client
        self.parallel_execution = parallel_execution
        
        # Initialize enabled personas
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
        hints: List[str] = None,
        qa_id: str = "",
    ) -> MultiAgentInvestigationOutput:
        """
        Run the Investigation phase with all enabled personas.
        
        Args:
            question: The question to answer
            table_repr: String representation of the table
            hints: Optional hints about question type
            
        Returns:
            MultiAgentInvestigationOutput with all persona results
        """
        start_time = time.time()
        hints = hints or []
        hints_str = ", ".join(hints) if hints else "Không có gợi ý"
        
        output = MultiAgentInvestigationOutput()
        
        if self.parallel_execution and len(self.enabled_personas) > 1:
            # === Parallel execution ===
            def run_persona_investigation(persona_name: str) -> tuple[str, PersonaInvestigationOutput]:
                result = self._run_single_persona(
                    persona_name=persona_name,
                    question=question,
                    table_repr=table_repr,
                    hints_str=hints_str,
                    qa_id=qa_id,
                )
                return persona_name, result
            
            with ThreadPoolExecutor(max_workers=len(self.enabled_personas)) as executor:
                futures = {
                    executor.submit(run_persona_investigation, name): name
                    for name in self.enabled_personas
                }
                
                for future in as_completed(futures):
                    persona_name, result = future.result()
                    output.persona_results[persona_name] = result
        else:
            # === Sequential execution ===
            for persona_name in self.enabled_personas:
                result = self._run_single_persona(
                    persona_name=persona_name,
                    question=question,
                    table_repr=table_repr,
                    hints_str=hints_str,
                    qa_id=qa_id,
                )
                output.persona_results[persona_name] = result
        
        output.total_latency_ms = (time.time() - start_time) * 1000
        return output
    
    def _run_single_persona(
        self,
        persona_name: str,
        question: str,
        table_repr: str,
        hints_str: str,
        qa_id: str,
    ) -> PersonaInvestigationOutput:
        """
        Run A→S investigation for a single persona.
        
        Args:
            persona_name: Name of the persona
            question: The question
            table_repr: Table representation
            hints_str: Formatted hints string
            
        Returns:
            PersonaInvestigationOutput with this persona's findings
        """
        start_time = time.time()
        lens = get_persona_lens(persona_name)
        
        output = PersonaInvestigationOutput(persona_name=persona_name)
        
        # Step 1: Analyze (A)
        analyze_result = self._run_analyze(
            lens=lens,
            question=question,
            table_repr=table_repr,
            hints_str=hints_str,
            qa_id=qa_id,
        )
        output.analyze_output = analyze_result
        
        if analyze_result.success:
            output.complexity = analyze_result.get("complexity", "basic")
            qt = analyze_result.get("question_type", "what")
            output.question_type = "" if qt is None else str(qt)
            output.key_observations = analyze_result.get("key_observations", [])
            output.potential_risks = analyze_result.get("potential_risks", [])
            output.answerable_assessment = analyze_result.get("answerable_assessment", "answerable")
        
        # Step 2: Solve (S)
        solve_result = self._run_solve(
            lens=lens,
            question=question,
            table_repr=table_repr,
            analysis=analyze_result.data if analyze_result.success else {},
            qa_id=qa_id,
        )
        output.solve_output = solve_result
        
        if solve_result.success:
            output.solution_plan = solve_result.get("solution_plan", [])
            draft = solve_result.get("draft_answer", "")
            output.draft_answer = _stringify_answer(draft)
            output.evidence_cells = solve_result.get("evidence_cells", [])
            output.confidence = solve_result.get("confidence", "medium")
        
        output.total_latency_ms = (time.time() - start_time) * 1000
        return output
    
    def _run_analyze(
        self,
        lens: PersonaLens,
        question: str,
        table_repr: str,
        hints_str: str,
        qa_id: str,
    ) -> AgentOutput:
        """Run the Analyze (A) step for a persona."""
        start_time = time.time()
        agent_name = f"{lens.name}_Analyzer"

        try:
            analyze_prompt, _ = get_investigation_prompts(self.llm_client)
            prompt = format_prompt_with_lens(
                analyze_prompt,
                lens,
                question=question,
                table=table_repr,
                hints=hints_str,
            )

            parsed, response, ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["answerable_assessment"],
                caller_name=agent_name,
                qa_id=qa_id,
            )

            latency = (time.time() - start_time) * 1000

            return AgentOutput(
                data=parsed,
                raw_response=response,
                agent_name=agent_name,
                success=ok,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return AgentOutput(
                data={},
                raw_response="",
                agent_name=agent_name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
    
    def _run_solve(
        self,
        lens: PersonaLens,
        question: str,
        table_repr: str,
        analysis: Dict[str, Any],
        qa_id: str,
    ) -> AgentOutput:
        """Run the Solve (S) step for a persona."""
        start_time = time.time()
        agent_name = f"{lens.name}_Solver"

        try:
            _, solve_prompt = get_investigation_prompts(self.llm_client)
            prompt = format_prompt_with_lens(
                solve_prompt,
                lens,
                question=question,
                table=table_repr,
                analysis=json.dumps(analysis, ensure_ascii=False, indent=2),
            )

            parsed, response, ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["draft_answer"],
                caller_name=agent_name,
                qa_id=qa_id,
            )

            latency = (time.time() - start_time) * 1000

            return AgentOutput(
                data=parsed,
                raw_response=response,
                agent_name=agent_name,
                success=ok,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return AgentOutput(
                data={},
                raw_response="",
                agent_name=agent_name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
    
