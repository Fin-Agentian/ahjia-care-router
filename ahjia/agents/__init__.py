"""Layer 2: the six care agents (P:140-149).

Each agent turns Observations (and the clock) into ActionRequests. Safety
requests (fall, voice SOS, night exit, inactivity, stranger at night) go to the
deterministic guardrails; companion requests go to the Care Router.
"""

from __future__ import annotations

from ahjia.agents.base import Agent, AgentContext
from ahjia.agents.companion import CompanionAgent, CompanionConfig
from ahjia.agents.fall import FallAgent
from ahjia.agents.inactivity import InactivityAgent
from ahjia.agents.night_exit import NightExitAgent
from ahjia.agents.visitor import VisitorAgent, VisitorConfig
from ahjia.agents.voice_sos import VoiceSosAgent

__all__ = ["Agent", "AgentContext", "make_agents"]


def make_agents(ctx: AgentContext, face_confirm_s: float = 1.5) -> tuple[Agent, ...]:
    """The six agents in safety-first priority order (P:222): fall and voice SOS
    first, then night exit and inactivity, then visitor, then companion last."""
    return (
        FallAgent(ctx),
        VoiceSosAgent(ctx),
        NightExitAgent(ctx),
        InactivityAgent(ctx),
        VisitorAgent(ctx, VisitorConfig(face_confirm_s=face_confirm_s)),
        CompanionAgent(ctx, CompanionConfig(face_confirm_s=face_confirm_s)),
    )
