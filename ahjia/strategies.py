"""The three strategies compared in the KPIs (DESIGN.md "Strategies", P:423-431).

| Strategy | Role          | Definition |
|----------|---------------|------------|
| S0       | cost ceiling  | every camera at full rate; every perception model on every frame;
|          |               | LLM always loaded and answers every speech segment |
| S1       | real baseline | fixed frame rate; rule-based agents; fixed GenAI hours |
| S2       | our method    | frame rate from occupancy and risk; zone cascade; safety-first QoS;
|          |               | GenAI window from the preload rule |

All three use the same six agents and the same deterministic guardrails
(ahjia/guardrails.py); a strategy can change frame rates, which models run on a
frame, NPU scheduling and the LLM, but never an alert.

Choices made for S1 where P leaves a number open (fair to S1, documented):
fixed rate = 10 FPS (P's "person present" rate, P:218); fixed GenAI hours =
07:00-22:00 (the complement of P's night, P:226). A3 still raises the camera
to full rate in S1 and S0, because A3 is a guardrail action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ahjia.events import STRATEGIES, StrategyName
from ahjia.residency import Profile

FpsMode = Literal["full", "fixed", "adaptive"]
GenAIMode = Literal["always", "fixed_hours", "preload_rule"]


@dataclass(frozen=True)
class StrategyPolicy:
    name: StrategyName
    role: str
    definition: str
    fps_mode: FpsMode
    pipeline: Profile
    genai_mode: GenAIMode
    qos: bool
    llm_answers_all_speech: bool
    fixed_fps: float = 10.0
    genai_hours: tuple[str, str] = ("07:00:00", "22:00:00")


POLICIES: dict[str, StrategyPolicy] = {
    "S0": StrategyPolicy(
        name="S0", role="cost ceiling",
        definition="every camera at full rate; every perception model on every frame; "
                   "LLM always loaded and answers every speech segment",
        fps_mode="full", pipeline="all_models", genai_mode="always", qos=False, llm_answers_all_speech=True,
    ),
    "S1": StrategyPolicy(
        name="S1", role="real baseline",
        definition="fixed frame rate (10 FPS); rule-based agents; fixed GenAI hours (07:00-22:00)",
        fps_mode="fixed", pipeline="fixed", genai_mode="fixed_hours", qos=False, llm_answers_all_speech=False,
    ),
    "S2": StrategyPolicy(
        name="S2", role="our method (Care Router)",
        definition="frame rate from occupancy and risk; zone cascade; safety-first QoS; "
                   "GenAI window from the preload rule",
        fps_mode="adaptive", pipeline="cascade", genai_mode="preload_rule", qos=True, llm_answers_all_speech=False,
    ),
}


def get_policy(name: str) -> StrategyPolicy:
    if name not in POLICIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGIES}")
    return POLICIES[name]


def parse_strategies(text: str) -> tuple[StrategyName, ...]:
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    for name in names:
        get_policy(name)
    return names  # type: ignore[return-value]
