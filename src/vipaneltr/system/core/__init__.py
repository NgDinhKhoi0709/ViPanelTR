"""
Core module for PanelTR-ViTabQA.

Contains the main orchestration logic and phase implementations.

Architecture:
- All 5 agents conduct individual A→S investigation
- Self-Review is optional per-agent
- Peer-Review reuses investigation, allows real dissent
"""

from .orchestrator import PanelTROrchestrator, PanelTRResult
from .investigation import (
    InvestigationPhase,
    MultiAgentInvestigationOutput,
    PersonaInvestigationOutput,
)
from .peer_review import (
    PeerReviewPhase,
    PeerReviewOutput,
    DissentingOpinion,
)
from .self_review import SelfReviewPhase, SelfReviewOutput

__all__ = [
    # Main orchestrator
    "PanelTROrchestrator",
    "PanelTRResult",
    # Investigation
    "InvestigationPhase",
    "MultiAgentInvestigationOutput",
    "PersonaInvestigationOutput",
    # Self-Review
    "SelfReviewPhase",
    "SelfReviewOutput",
    # Peer-Review
    "PeerReviewPhase",
    "PeerReviewOutput",
    "DissentingOpinion",
]
