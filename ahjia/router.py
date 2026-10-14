"""Care Router v4 (layer 3, P:150-241): frame rate per camera, GenAI window,
companion routing. Safety decisions are NOT made here; they belong to the
guardrails. The router only decides where NPU time goes.

Frame rate per camera (P:216-222)
    empty room 1-2 FPS (motion gate + person detection; 2 FPS for cameras that
    watch a high-risk spot or the night hallway / front door), person present
    10 FPS, person in a high-risk zone or time (bathroom door, bedside, hallway
    at night) 15 FPS, A3 = the camera's full rate with full-resolution pose.
    When the NPU is short, empty cameras drop first, then low-risk occupied
    cameras; high-risk and A3 cameras are never lowered; the companion (LLM)
    is served last by the QoS scheduler (residency.allocate).

GenAI window (P:224-241)
    Night 22:00-07:00: vision only, night talk gets template voice (A1).
    Daytime: preload the LLM when p_L x S_L x w_L > rho x C_risk. The
    estimators below are simple, transparent design assumptions (not
    measurements) that Stage II replaces with a learned predictor (P:240).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from ahjia.events import HIGH_RISK_ZONES, NIGHT_HIGH_RISK_ZONES, ActionCode, ActionRequest, GenAIState, PersonObs
from ahjia.residency import EPS, HailoSpec, frame_cost_s
from ahjia.strategies import StrategyPolicy

CameraRisk = Literal["empty", "occupied", "high_risk"]
CameraState = Literal["empty", "occupied", "high_risk", "a3"]
WATCH_CAMERAS: frozenset[str] = frozenset({"bathroom_door", "bedside", "bedroom"})
NIGHT_WATCH_CAMERAS: frozenset[str] = frozenset({"hallway", "front_door"})
FOLLOWUP_PREFIX = "follow-up"


@dataclass(frozen=True)
class FrameRatePlan:
    empty_fps: float = 1.0
    empty_watch_fps: float = 2.0
    occupied_fps: float = 10.0
    high_risk_fps: float = 15.0
    degraded_fps: float = 5.0  # low-risk occupied camera when the NPU is short (assumption)


def camera_risk(persons: tuple[PersonObs, ...], night: bool, camera: str | None = None) -> CameraRisk:
    """A camera is high risk when someone is in a high-risk zone, or when the
    camera itself watches one (e.g. a person on the hallway camera at night,
    even if the foot point falls in the generic "floor" zone)."""
    if not persons:
        return "empty"
    risky = HIGH_RISK_ZONES | (NIGHT_HIGH_RISK_ZONES if night else frozenset())
    if camera in risky or any(p.zone in risky for p in persons):
        return "high_risk"
    return "occupied"


def _watches_risk(camera: str, night: bool) -> bool:
    return camera in WATCH_CAMERAS or (night and camera in NIGHT_WATCH_CAMERAS)


def _adaptive_fps(camera: str, risk: CameraRisk, night: bool, rates: FrameRatePlan) -> float:
    if risk == "high_risk":
        return rates.high_risk_fps
    if risk == "occupied":
        return rates.occupied_fps
    return rates.empty_watch_fps if _watches_risk(camera, night) else rates.empty_fps


def plan_fps(policy: StrategyPolicy, cameras: tuple[str, ...], views: Mapping[str, tuple[PersonObs, ...]],
             night: bool, a3_cameras: frozenset[str], full_fps: float,
             rates: FrameRatePlan | None = None) -> dict[str, float]:
    """Frame rate the strategy wants for each camera (before the NPU budget)."""
    rates = rates or FrameRatePlan()
    plan: dict[str, float] = {}
    for cam in cameras:
        if cam in a3_cameras or policy.fps_mode == "full":
            plan[cam] = float(full_fps)
        elif policy.fps_mode == "fixed":
            plan[cam] = policy.fixed_fps
        else:
            plan[cam] = _adaptive_fps(cam, camera_risk(views.get(cam, ()), night, cam), night, rates)
    return plan


def degrade_for_budget(plan: Mapping[str, float], risks: Mapping[str, str],
                       load_of: Callable[[str, float], float], budget: float,
                       rates: FrameRatePlan | None = None) -> dict[str, float]:
    """S2 QoS: lower empty cameras first, then low-risk occupied ones (P:222)."""
    rates = rates or FrameRatePlan()
    current = dict(plan)
    steps = (("empty", rates.empty_fps), ("occupied", rates.degraded_fps))
    for level, fps in steps:
        if sum(load_of(c, f) for c, f in current.items()) <= budget + EPS:
            break
        current = {c: (min(f, fps) if risks.get(c) == level else f) for c, f in current.items()}
    return current


def camera_load(policy: StrategyPolicy, spec: HailoSpec, state: CameraState, full_fps: float,
                rates: FrameRatePlan | None = None) -> float:
    """NPU seconds per second one camera needs in a given state (for KPI 2)."""
    rates = rates or FrameRatePlan()
    if state == "a3" or policy.fps_mode == "full":
        fps = float(full_fps)
    elif policy.fps_mode == "fixed":
        fps = policy.fixed_fps
    else:
        fps = {"empty": rates.empty_watch_fps, "occupied": rates.occupied_fps,
               "high_risk": rates.high_risk_fps}[state]
    persons = 0 if state == "empty" else 1
    return fps * frame_cost_s(spec, policy.pipeline, persons, full_res_pose=state == "a3")


# ---------------------------------------------------------------- GenAI window

@dataclass(frozen=True)
class PreloadParams:
    """Transparent estimators for the preload rule (design assumptions)."""

    p_nobody: float = 0.02        # P(talk in the next window) when the elder is not in view
    p_elder: float = 0.2          # elder in view in the daytime
    p_visitor: float = 0.6        # family or caregiver at home
    p_recent_talk: float = 0.3    # added when someone talked to AhJia in the last recent_talk_s
    recent_talk_s: float = 600.0
    w_l: float = 1.0              # companion weight (care mode "balanced")
    rho_base: float = 0.02        # P(high-risk event in the window), quiet daytime
    rho_high_risk_zone: float = 0.3
    rho_alert: float = 1.0
    c_risk: float = 30.0          # cost of GenAI interfering with safety, in seconds (the 30 s rule)


@dataclass(frozen=True)
class WindowContext:
    t: float
    night: bool
    elder_visible: bool
    visitor_present: bool
    someone_in_high_risk: bool
    alert_active: bool
    last_directed_t: float | None


@dataclass(frozen=True)
class PreloadTerms:
    p_l: float
    s_l: float
    w_l: float
    rho: float
    c_risk: float

    @property
    def lhs(self) -> float:
        return self.p_l * self.s_l * self.w_l

    @property
    def rhs(self) -> float:
        return self.rho * self.c_risk

    @property
    def open(self) -> bool:
        return self.lhs > self.rhs


def preload_terms(ctx: WindowContext, s_l: float, params: PreloadParams | None = None) -> PreloadTerms:
    """p_L x S_L x w_L > rho x C_risk (P:233-238). At night the window is closed."""
    params = params or PreloadParams()
    if ctx.visitor_present:
        p_l = params.p_visitor
    else:
        p_l = params.p_elder if ctx.elder_visible else params.p_nobody
    if ctx.last_directed_t is not None and ctx.t - ctx.last_directed_t <= params.recent_talk_s:
        p_l += params.p_recent_talk
    rho = params.rho_base
    if ctx.someone_in_high_risk:
        rho = max(rho, params.rho_high_risk_zone)
    if ctx.alert_active:
        rho = max(rho, params.rho_alert)
    if ctx.night:
        p_l = 0.0
    return PreloadTerms(p_l=min(p_l, 0.95), s_l=s_l, w_l=params.w_l, rho=rho, c_risk=params.c_risk)


# ---------------------------------------------------------------- companion routing

@dataclass(frozen=True)
class CompanionRoute:
    actions: tuple[ActionCode, ...]
    note: str = ""
    needs_llm: bool = False
    load_now: bool = False


def route_companion(policy: StrategyPolicy, req: ActionRequest, *, night: bool, genai: GenAIState,
                    in_hours: bool, risk_high: bool) -> CompanionRoute:
    """A2 vs A1 vs A0 for one companion request. Never touches an alert."""
    follow_up = req.reason.startswith(FOLLOWUP_PREFIX)
    no_llm = CompanionRoute(("A0",), "template already said") if follow_up else None
    if policy.llm_answers_all_speech:
        return CompanionRoute(("A2",), "S0 answers every segment", needs_llm=True)
    if req.actions == ("A0",):
        return CompanionRoute(("A0",))
    if policy.genai_mode == "fixed_hours":
        if in_hours:
            return CompanionRoute(("A2",), "fixed GenAI hours", needs_llm=True)
        return no_llm or CompanionRoute(("A1",), "outside GenAI hours: template voice")
    if night:
        return no_llm or CompanionRoute(("A1",), "night: template voice")
    if genai in ("npu_window", "loading"):
        return CompanionRoute(("A2",), "GenAI window", needs_llm=True)
    if risk_high:
        return no_llm or CompanionRoute(("A1",), "risk high: no LLM load, template voice")
    return CompanionRoute(("A2",), "talking to AhJia opens the window", needs_llm=True, load_now=True)
