"""Shared data contract for the AhJia Care Router prototype.

Every layer talks through these immutable records:

    Layer 1 (perception)            -> Observation
    Layer 2 (care agents)           -> ActionRequest
    Layer 3 (Care Router, guardrails) -> Decision, HudState

Changing a field here is a contract change. Workers must ask the leader first.
Rules behind the names are in DESIGN.md and the proposal v4 (P:112-252).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

Posture = Literal["stand", "sit", "lie", "unknown"]
Identity = Literal["elder", "family", "caregiver", "stranger", "unknown"]
Zone = Literal[
    "bed", "bedside", "sofa", "floor", "bathroom", "bathroom_door",
    "hallway", "front_door", "living", "kitchen", "other",
]
SpeechSource = Literal["elder", "family", "caregiver", "tv", "other"]
AgentName = Literal["fall", "inactivity", "night_exit", "visitor", "voice_sos", "companion"]
ActionCode = Literal["A0", "A1", "A2", "A3", "A4", "A5"]
AlertState = Literal["none", "pre_alert", "escalated", "cleared_by_elder", "cleared_by_app"]
GenAIState = Literal["off", "loading", "npu_window", "host_cpu"]
StrategyName = Literal["S0", "S1", "S2"]

ACTIONS: dict[str, str] = {
    "A0": "no-op (log only)",
    "A1": "template voice",
    "A2": "wake LLM (companion)",
    "A3": "look closer (full frame rate, full-resolution pose)",
    "A4": "pre-alert family (event text only)",
    "A5": "escalate (call family or emergency contact)",
}
AGENTS: tuple[str, ...] = ("fall", "inactivity", "night_exit", "visitor", "voice_sos", "companion")
STRATEGIES: tuple[str, ...] = ("S0", "S1", "S2")

# Lying in these zones is normal rest, not a fall (P:144, P:245).
REST_ZONES: frozenset[str] = frozenset({"bed", "sofa"})
# High-risk zones get 15 FPS when someone is there; the hallway only at night (P:218).
HIGH_RISK_ZONES: frozenset[str] = frozenset({"bathroom_door", "bedside"})
NIGHT_HIGH_RISK_ZONES: frozenset[str] = frozenset({"hallway"})
NIGHT_START = "22:00:00"  # GenAI is off by default from 22:00 to 07:00 (P:226)
NIGHT_END = "07:00:00"


@dataclass(frozen=True)
class PersonObs:
    """One person seen by one camera at one moment.

    bbox and keypoints are normalized to 0..1 of the frame.
    keypoints holds the 17 COCO points as (x, y, confidence).
    identity is simulated in Stage I (no face recognition runs).
    """

    track_id: int
    identity: Identity = "unknown"
    id_conf: float = 0.0
    posture: Posture = "unknown"
    zone: Zone = "other"
    bbox: tuple[float, float, float, float] | None = None
    keypoints: tuple[tuple[float, float, float], ...] | None = None


@dataclass(frozen=True)
class SpeechObs:
    """One speech segment after VAD and ASR (simulated in Stage I).

    directed is True when the speaker talks to AhJia: calls its name,
    faces the device, or is alone with it (P:197).
    """

    text: str
    source: SpeechSource = "elder"
    directed: bool = False
    lang: str = "en"


@dataclass(frozen=True)
class Observation:
    """Layer 1 output.

    camera is None for audio-only observations.
    persons is the full list of people the camera sees now; an empty tuple
    means the camera sees nobody.
    app_confirm is True when family or a caregiver confirms in the app,
    which is one of the two ways to clear an alert (P:250).
    """

    t: float
    camera: str | None = None
    persons: tuple[PersonObs, ...] = ()
    speech: SpeechObs | None = None
    app_confirm: bool = False


@dataclass(frozen=True)
class ActionRequest:
    """Layer 2 output: what one care agent asks for.

    safety=True means the request goes to the deterministic guardrails,
    never to a learned or tunable policy.
    """

    t: float
    agent: AgentName
    actions: tuple[ActionCode, ...]
    reason: str
    camera: str | None = None
    safety: bool = False


@dataclass(frozen=True)
class Decision:
    """Layer 3 output: what AhJia does, as written to the decision log.

    latency_s is the simulated event-to-response time, or None for A0.
    fps holds (camera, frames per second) pairs after this decision.
    simulated is True when timing or power comes from the Hailo-10H simulator.
    """

    t: float
    clock: str
    strategy: StrategyName
    agent: AgentName | None
    actions: tuple[ActionCode, ...]
    reason: str
    latency_s: float | None
    alert: AlertState
    genai: GenAIState
    fps: tuple[tuple[str, float], ...] = ()
    simulated: bool = True


@dataclass(frozen=True)
class HudState:
    """The 7 HUD lines shown on the demo video (P:470-482)."""

    agent: str = "-"
    decision: str = "-"
    reason: str = "-"
    latency: str = "-"
    cameras: str = "-"
    genai: str = "-"
    alert: str = "none"

    def lines(self) -> tuple[str, ...]:
        return (
            f"Agent    : {self.agent}",
            f"Decision : {self.decision}",
            f"Reason   : {self.reason}",
            f"Latency  : {self.latency}",
            f"Cameras  : {self.cameras}",
            f"GenAI    : {self.genai}",
            f"Alert    : {self.alert}",
        )


@dataclass(frozen=True)
class RunConfig:
    """Settings for one run of the engine (demo or replay).

    start_clock is the wall-clock time at t = 0, as "HH:MM:SS".
    time_scale shortens long agent thresholds (30 min, 2 h) in demos;
    1.0 keeps the real values from P:140-149.
    """

    start_clock: str
    cameras: tuple[str, ...]
    elder_alone: bool = True
    spec_path: str = "specs/hailo10h.yaml"
    time_scale: float = 1.0
    usual_wake_clock: str = "06:30:00"


class Engine(Protocol):
    """Layer 3 engine. Core implements it in ahjia/engine.py (make_engine)."""

    strategy: StrategyName

    def observe(self, obs: Observation) -> tuple[Decision, ...]: ...

    def advance(self, t: float) -> tuple[Decision, ...]: ...

    def hud(self) -> HudState: ...

    def camera_fps(self, camera: str) -> float: ...


def decision_to_json(decision: Decision) -> str:
    """One decision log line (JSONL)."""
    record = asdict(decision)
    record["fps"] = dict(decision.fps)
    record["actions"] = list(decision.actions)
    return json.dumps(record, ensure_ascii=False)
