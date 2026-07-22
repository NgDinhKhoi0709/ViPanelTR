"""
Prompts module for PanelTR-ViTabQA.

Contains prompt templates:
- investigation.py: Atomic prompts (Analyze, Solve, Verify, Present, Deliberate, Synthesize, Format)
- persona_lenses.py: 5 persona lens definitions
- self_review.py: Self-review prompt
- semantic_consensus.py: Semantic consensus check prompt
"""

from .investigation import (
    ANALYZE_PROMPT,
    SOLVE_PROMPT,
    VERIFY_PROMPT,
    PRESENT_PROMPT,
    DELIBERATE_PROMPT,
    SYNTHESIZE_PROMPT,
    FORMAT_ANSWER_PROMPT,
    format_hints,
    format_evidence,
    format_solution_plan,
)
from .semantic_consensus import SEMANTIC_CONSENSUS_PROMPT
from .persona_lenses import (
    PersonaLens,
    get_persona_lens,
    get_all_persona_names,
    format_prompt_with_lens,
    PERSONA_LENSES,
    LOGICIAN,
    CALCULATOR,
    VERIFIER,
    STRUCTURALIST,
    SYNTHESIZER,
)
from . import self_review

__all__ = [
    # Atomic prompts
    "ANALYZE_PROMPT",
    "SOLVE_PROMPT",
    "VERIFY_PROMPT",
    "PRESENT_PROMPT",
    "DELIBERATE_PROMPT",
    "SYNTHESIZE_PROMPT",
    "FORMAT_ANSWER_PROMPT",
    "SEMANTIC_CONSENSUS_PROMPT",
    # Helpers
    "format_hints",
    "format_evidence",
    "format_solution_plan",
    # Persona lenses
    "PersonaLens",
    "get_persona_lens",
    "get_all_persona_names",
    "format_prompt_with_lens",
    "PERSONA_LENSES",
    "LOGICIAN",
    "CALCULATOR",
    "VERIFIER",
    "STRUCTURALIST",
    "SYNTHESIZER",
    # Self-review
    "self_review",
]
