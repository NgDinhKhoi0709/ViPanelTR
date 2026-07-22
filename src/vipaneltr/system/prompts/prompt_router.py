"""
Prompt router for returning prompt templates.
(Simplified since versions were removed)
"""

from __future__ import annotations

from typing import Tuple

from .investigation import (
    ANALYZE_PROMPT,
    SOLVE_PROMPT,
    PRESENT_PROMPT,
    DELIBERATE_PROMPT,
    SYNTHESIZE_PROMPT,
)
from .self_review import SELF_REVIEW_PROMPT
from .semantic_consensus import SEMANTIC_CONSENSUS_PROMPT


def get_investigation_prompts(llm_client) -> Tuple[str, str]:
    """Return (analyze, solve) prompt templates."""
    return (ANALYZE_PROMPT, SOLVE_PROMPT)


def get_peer_review_prompts(llm_client) -> Tuple[str, str, str]:
    """Return (present, deliberate, synthesize) prompt templates."""
    return (PRESENT_PROMPT, DELIBERATE_PROMPT, SYNTHESIZE_PROMPT)


def get_self_review_prompt(llm_client) -> str:
    return SELF_REVIEW_PROMPT


def get_semantic_consensus_prompt(llm_client) -> str:
    return SEMANTIC_CONSENSUS_PROMPT
