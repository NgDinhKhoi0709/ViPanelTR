"""
Answer Formatter Phase implementation.

Phase 4 of PanelTR: Format the final answer to be concise and match groundtruth style.
Runs after all 3 main phases are complete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents.base_agent import AgentOutput
from ..agents.llm_client import LLMClient
from ..prompts.investigation import FORMAT_ANSWER_PROMPT
from ...utils.json_parser import parse_json_response
from ...utils.llm_retry import call_llm_with_retry
from ...utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class AnswerFormatterOutput:
    """Output from Answer Formatter phase."""
    
    # Input
    original_answer: str = ""
    question: str = ""
    question_type: List[str] = field(default_factory=list)
    
    # Output (list of candidate variants for Exact Match)
    formatted_answer: List[str] = field(default_factory=list)
    
    # Metadata
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tracing."""
        return {
            "original_answer": self.original_answer,
            "question": self.question,
            "question_type": self.question_type,
            "formatted_answer": self.formatted_answer,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class AnswerFormatterPhase:
    """
    Answer Formatter phase of PanelTR.
    
    Takes the final pred_answer from Phase 3 (Peer-Review) and formats it
    to be concise and match groundtruth style.
    
    Key features:
    - Extracts core answer from verbose responses
    - Removes explanatory phrases
    - Normalizes formatting (whitespace, punctuation)
    - Preserves Vietnamese diacritics
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize Answer Formatter phase.
        
        Args:
            llm_client: LLM client for generating responses
        """
        self.llm_client = llm_client
    
    def run(
        self,
        question: str,
        pred_answer: str,
        question_type: List[str] = None,
        qa_id: str = "",
    ) -> AnswerFormatterOutput:
        """
        Run the Answer Formatter phase.
        
        Args:
            question: The original question
            pred_answer: The predicted answer from Phase 3
            question_type: Question type from Phase 1 (optional)
            
        Returns:
            AnswerFormatterOutput with formatted_answer
        """
        start_time = time.time()
        question_type = question_type or []

        pred_answer_str = "" if pred_answer is None else str(pred_answer)
        
        output = AnswerFormatterOutput(
            original_answer=pred_answer_str,
            question=question,
            question_type=question_type,
        )
        
        # Handle special cases
        if not pred_answer_str or pred_answer_str.strip().lower() == "null":
            output.formatted_answer = ["Null"]
            output.latency_ms = (time.time() - start_time) * 1000
            return output
        
        try:
            question_type_str = ", ".join(question_type) if question_type else "Không có question_type"
            prompt = FORMAT_ANSWER_PROMPT.format(
                question=question,
                pred_answer=pred_answer_str,
                question_type=question_type_str,
            )

            parsed, response, _ok = call_llm_with_retry(
                self.llm_client,
                prompt,
                required_fields=["formatted_answer"],
                caller_name="AnswerFormatter",
                qa_id=qa_id,
            )
            output.raw_response = response
            
            formatted = parsed.get("formatted_answer", [])
            # Normalise to list of strings
            if isinstance(formatted, str):
                candidates = [formatted] if formatted.strip() else []
            elif isinstance(formatted, list):
                candidates = [str(item).strip() for item in formatted if item]
            else:
                candidates = []

            # Clean each candidate: remove trailing period for short answers
            cleaned: List[str] = []
            for c in candidates:
                if c.endswith(".") and len(c.split()) <= 5:
                    c = c[:-1]
                if c:
                    cleaned.append(c)

            if cleaned:
                output.formatted_answer = cleaned
            else:
                # Fallback to original if parsing failed
                output.formatted_answer = [pred_answer_str]
                logger.warning(f"AnswerFormatter returned empty, using original answer")
            
            output.success = True
            
        except Exception as e:
            logger.error(f"AnswerFormatter error: {e}")
            output.formatted_answer = [pred_answer_str]  # Fallback to original
            output.success = False
            output.error = str(e)
        
        output.latency_ms = (time.time() - start_time) * 1000
        return output
