"""Layer 3 engine: Observations -> agents -> guardrails / Care Router -> Decisions + HUD.

make_engine(strategy, config) returns a CareEngine that implements the Engine
protocol of ahjia/events.py. The engine is deterministic (no randomness).

Per call:
  observe(obs)  first runs advance(obs.t) (timers: 30 s rule, 60 s check-in,
                30 min / 2 h rules, LLM load and decode ends, night boundary),
                then updates the camera view, the guardrails (posture, elder
                speech, app confirm) and the six agents, routes their requests
                (safety -> guardrails, companion -> router), re-plans frame
                rates and the GenAI window, and returns the Decisions.
  advance(t)    only moves the clock.
  hud()         the 7 HUD lines (P:470-482).

The elder's self-clear (I'm OK + standing >= 5 s) trusts only the freshest view
of the elder (the camera that saw her last), and only while that view is newer
than max(2 s, 3 frames at the camera's rate). While an alert is on, an "I'm OK"
is pending and that view shows her standing, next_deadline() asks for a frame
from that camera before the view goes stale, because the replay skips
unchanged frames.

Every latency is simulated with ahjia/residency.py and specs/hailo10h.yaml;
Decision.latency_s is the time from the event to the first response (voice,
text, call or LLM reply), measured from ActionRequest.t (for a greeting that is
the first frame that saw the visitor).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ahjia.agents import make_agents
from ahjia.agents.base import EPS, AgentContext, Deadline, find_elder, in_clock_window, parse_clock
from ahjia.events import (
    NIGHT_END,
    NIGHT_START,
    ActionCode,
    ActionRequest,
    AgentName,
    AlertState,
    Decision,
    HudState,
    Observation,
    PersonObs,
    RunConfig,
    StrategyName,
)
from ahjia.guardrails import GuardrailOutput, Guardrails
from ahjia.residency import Hailo10HSim, MeterTotals, frame_cost_s, load_spec, md1_wait_s, whisper_tokens
from ahjia.router import (
    FrameRatePlan,
    PreloadParams,
    WindowContext,
    camera_risk,
    degrade_for_budget,
    plan_fps,
    preload_terms,
    route_companion,
)
from ahjia.strategies import get_policy

HUD_HOLD_S = 3.0
STALE_MIN_S = 2.0      # a view of the elder older than max(2 s, 3 frames) does not count
STALE_FRAMES = 3.0
AGENT_LABELS = {"fall": "fall", "voice_sos": "voice SOS", "night_exit": "night exit",
                "inactivity": "inactivity", "visitor": "visitor", "companion": "companion"}
ACTION_LABELS = {"A0": "A0 log only", "A1": "A1 template voice", "A2": "A2 wake LLM",
                 "A3": "A3 look closer", "A4": "A4 pre-alert family", "A5": "A5 escalate"}
ALERT_LABELS = {"none": "none", "pre_alert": "pre-alert", "escalated": "escalated",
                "cleared_by_elder": "cleared by the elder", "cleared_by_app": "cleared in app"}
VISITORS = frozenset({"family", "caregiver"})


@dataclass(frozen=True)
class Response:
    """One response action with its simulated timing (used by the KPIs)."""

    t_event: float
    t_decision: float
    t_response: float
    action: ActionCode
    agent: AgentName | None
    safety: bool
    kind: str
    genai_wait_s: float = 0.0


@dataclass(frozen=True)
class _Source:
    kind: Literal["frame", "speech", "time"]
    camera: str | None = None
    asr_s: float = 0.0
    asr_wait_s: float = 0.0


@dataclass(frozen=True)
class _Pending:
    t: float
    t_event: float
    agent: AgentName | None
    actions: tuple[ActionCode, ...]
    reason: str
    camera: str | None
    alert: AlertState
    responses: tuple[Response, ...] = ()
    safety: bool = False


def decision_text(actions: tuple[str, ...]) -> str:
    return ACTION_LABELS.get(actions[0], actions[0]) if len(actions) == 1 else "+".join(actions)


class CareEngine:
    """Care Router engine for one strategy (S0, S1 or S2)."""

    def __init__(self, strategy: StrategyName, config: RunConfig) -> None:
        self.strategy: StrategyName = strategy
        self.policy = get_policy(strategy)
        self.config = config
        self.spec = load_spec(config.spec_path)
        self.ctx = AgentContext.from_run_config(config)
        self.rates = FrameRatePlan()
        self.preload = PreloadParams()
        self._v = self.spec.value
        self._agents = make_agents(self.ctx, face_confirm_s=self._v("pipeline.face_confirm_s"))
        self._guard = Guardrails()
        self._sim = Hailo10HSim(self.spec, qos=self.policy.qos)
        self._t, self._last_step, self._warm = 0.0, float("-inf"), True
        self._views: dict[str, tuple] = {cam: () for cam in config.cameras}
        self._frame_t: dict[str, float] = {}
        self._face_until: dict[tuple[str, int], float] = {}
        self._plan: dict[str, float] = {}
        self._decisions: list[Decision] = []
        self._responses: list[Response] = []
        self._last_directed_t: float | None = None
        self._hud_last: tuple[Decision, str | None, bool] | None = None
        self._hud_hold_until = float("-inf")
        self._refresh(0.0)

    # ------------------------------------------------------------ Engine protocol
    def observe(self, obs: Observation) -> tuple[Decision, ...]:
        decisions = list(self.advance(obs.t))
        pending: list[_Pending] = []
        source = _Source("time")
        if obs.camera is not None:
            source = _Source("frame", camera=obs.camera)
            pending += self._on_frame(obs)
        if obs.speech is not None:
            source = self._on_speech_input(obs)
            if obs.speech.source == "elder":
                pending += self._guard_all(self._guard.on_elder_posture(obs.t, *self._elder_standing()), source)
                pending += self._guard_all(self._guard.on_elder_speech(obs.t, obs.speech.text), source)
        if obs.app_confirm:
            pending += self._guard_all(self._guard.on_app_confirm(obs.t), _Source("time"))
        requests = [r for agent in self._agents for r in agent.on_observation(obs)]
        pending += self._handle(requests, source, obs.t)
        self._refresh(obs.t)
        return tuple(decisions + self._emit(pending))

    def advance(self, t: float) -> tuple[Decision, ...]:
        if t < self._t - EPS:
            raise ValueError(f"time went backwards: {t} < {self._t}")
        decisions: list[Decision] = []
        while True:
            due = [d for d in self._time_deadlines() if self._last_step + EPS < d <= t + EPS]
            if not due:
                break
            decisions += self._step_to(max(min(due), self._t))
        if t > self._last_step + EPS:
            decisions += self._step_to(t)
        return tuple(decisions)

    def hud(self) -> HudState:
        alert = ALERT_LABELS[self._guard.alert]
        if self._hud_last is None:
            return HudState(cameras=self._hud_cameras(None), genai=self._hud_genai(), alert=alert)
        d, camera, _ = self._hud_last
        latency = f"{d.latency_s:.1f} s (sim)" if d.latency_s is not None else "-"
        return HudState(agent=AGENT_LABELS.get(d.agent or "", "-"), decision=decision_text(d.actions),
                        reason=d.reason, latency=latency, cameras=self._hud_cameras(camera),
                        genai=self._hud_genai(), alert=alert)

    def camera_fps(self, camera: str) -> float:
        """Frame rate the router wants for this camera (before NPU saturation)."""
        return self._plan.get(camera, 0.0)

    # ------------------------------------------------------------ extras used by replay / KPIs
    def processed_fps(self, camera: str) -> float:
        """Frames per second the NPU really processes (frames are dropped when overloaded)."""
        return self.camera_fps(camera) * self._sim.vision_service()

    def next_deadline(self) -> Deadline | None:
        found: list[Deadline] = [(t, None) for t in self._time_deadlines() if t > self._t + EPS]
        for source in (*self._agents, self._guard):
            d = source.next_deadline()
            if d is not None and d[1] is not None:
                found.append(d)
        keep_alive = self._keep_alive()
        if keep_alive is not None:
            found.append(keep_alive)
        return min(found, key=lambda d: d[0]) if found else None

    def _keep_alive(self) -> Deadline | None:
        """A frame from the elder's camera before her standing view goes stale, only while an
        alert is on and an "I'm OK" is pending (only then can standing clear the alert)."""
        state = self._guard.state
        if not state.active or state.ok_said_t is None:
            return None
        view = self._elder_view()
        if view is None or view[2].posture != "stand":
            return None
        t_seen, cam, _ = view
        return max(t_seen + self._stale_after(cam) / 2.0, self._t), cam

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def responses(self) -> tuple[Response, ...]:
        return tuple(self._responses)

    @property
    def sim(self) -> Hailo10HSim:
        return self._sim

    def totals(self, t_end: float) -> MeterTotals:
        return self._sim.totals(t_end)

    # ------------------------------------------------------------ time
    def _move_to(self, t: float) -> None:
        if t > self._t:
            self._t = t
        if self._t > 0:
            self._warm = False
        self._sim.advance(self._t)

    def _time_deadlines(self) -> list[float]:
        times = [d[0] for src in (*self._agents, self._guard) if (d := src.next_deadline()) and d[1] is None]
        event = self._sim.next_event(self._t)
        times += [event] if event is not None else []
        times += [v for v in self._face_until.values() if v > self._t + EPS]
        clocks = {parse_clock(NIGHT_START), parse_clock(NIGHT_END), *map(parse_clock, self.policy.genai_hours)}
        times += [self.ctx.next_clock_t(self._t, c) for c in clocks]
        return times

    def _step_to(self, t: float) -> list[Decision]:
        self._move_to(t)
        self._last_step = t
        requests = [r for agent in self._agents for r in agent.on_time(t)]
        pending = self._handle(requests, _Source("time"), t)
        pending += self._guard_all(self._guard.advance(t), _Source("time"))
        self._face_until = {k: v for k, v in self._face_until.items() if v > t + EPS}
        self._refresh(t)
        return self._emit(pending)

    # ------------------------------------------------------------ inputs
    def _on_frame(self, obs: Observation) -> list[_Pending]:
        cam = obs.camera
        old = {p.track_id for p in self._views.get(cam, ())}
        now = {p.track_id for p in obs.persons}
        self._views = {**self._views, cam: obs.persons}
        self._frame_t = {**self._frame_t, cam: obs.t}
        face = {k: v for k, v in self._face_until.items() if k[0] != cam or k[1] in now}
        confirm_s = self._v("pipeline.face_confirm_s")
        face.update({(cam, tid): obs.t + confirm_s for tid in now - old})
        self._face_until = face
        standing, where = self._elder_standing()
        return self._guard_all(self._guard.on_elder_posture(obs.t, standing, where), _Source("frame", camera=cam))

    def _elder_view(self) -> tuple[float, str, PersonObs] | None:
        """(frame time, camera, elder) from the camera that saw the elder most recently.
        On a tie, a view where she is not standing wins: the clear needs certainty."""
        views = [(self._frame_t[cam], cam, elder) for cam, persons in self._views.items()
                 if cam in self._frame_t and (elder := find_elder(persons, self.ctx.elder_alone)) is not None]
        return max(views, key=lambda v: (v[0], v[2].posture != "stand")) if views else None

    def _stale_after(self, cam: str) -> float:
        fps = self.processed_fps(cam)
        return max(STALE_MIN_S, STALE_FRAMES / fps) if fps > 0 else STALE_MIN_S

    def _elder_standing(self) -> tuple[bool | None, str | None]:
        """(is the elder standing, camera) from her freshest view; (None, None) when no camera
        sees her or that view is stale. The clear rule needs posture "stand" (P:249)."""
        view = self._elder_view()
        if view is None:
            return None, None
        t_seen, cam, elder = view
        if self._t - t_seen > self._stale_after(cam) + EPS:
            return None, None
        return elder.posture == "stand", cam

    def _on_speech_input(self, obs: Observation) -> _Source:
        speech = obs.speech
        wait = self._sim.genai_wait(obs.t, whisper_tokens(self.spec, speech.text))
        asr = self._sim.start_whisper(obs.t, speech.text)
        if speech.directed:
            self._last_directed_t = obs.t
        return _Source("speech", asr_s=asr, asr_wait_s=wait)

    # ------------------------------------------------------------ routing
    def _handle(self, requests: Iterable[ActionRequest], source: _Source, t: float) -> list[_Pending]:
        pending: list[_Pending] = []
        for req in requests:
            if req.safety:
                pending += self._guard_all(self._guard.on_request(req), source)
            elif req.agent == "companion":
                pending.append(self._companion(req, source, t))
            else:
                pending.append(_Pending(t, req.t, req.agent, req.actions, req.reason, req.camera, self._guard.alert))
        return pending

    def _processing(self, source: _Source, t: float) -> tuple[float, float]:
        """(time from the input to a decision, part of it spent waiting for the LLM)."""
        router = self._v("pipeline.router_decision_s")
        if source.kind == "speech":
            return source.asr_s + router, source.asr_wait_s
        if source.kind == "frame" and source.camera is not None:
            cost = self._frame_cost(source.camera)
            wait = self._sim.genai_wait(t)
            return cost + md1_wait_s(cost, self._sim.utilization()) + wait + router, wait
        return router, 0.0

    def _guard_all(self, outputs: Iterable[GuardrailOutput], source: _Source) -> list[_Pending]:
        return [self._guard_one(out, source) for out in outputs]

    def _guard_one(self, out: GuardrailOutput, source: _Source) -> _Pending:
        proc, wait = self._processing(source, out.t)
        offsets = {"A1": self._v("pipeline.tts_first_audio_s"), "A4": self._v("pipeline.notify_s"),
                   "A5": self._v("pipeline.call_dial_s"), "A3": 0.0}
        safety = out.kind in ("trigger", "escalate", "checkin", "notify")
        responses = tuple(
            Response(out.t_event, out.t, out.t + proc + offsets[a], a, out.agent, safety, out.kind, wait)
            for a in out.actions if a in offsets)
        return _Pending(out.t, out.t_event, out.agent, out.actions, out.reason, out.camera, out.alert,
                        responses, safety=safety)

    def _companion(self, req: ActionRequest, source: _Source, t: float) -> _Pending:
        route = route_companion(self.policy, req, night=self.ctx.is_night(t), genai=self._sim.genai_state(t),
                                in_hours=self._in_hours(t), risk_high=self._risk_high())
        reason = req.reason if route.actions == req.actions or not route.note else f"{req.reason} · {route.note}"
        ready = t + (source.asr_s if source.kind == "speech" else 0.0) + self._v("pipeline.router_decision_s")
        responses: tuple[Response, ...] = ()
        if route.needs_llm:
            if not (self._sim.llm_resident or self._sim.loading(t)):
                self._sim.start_load(t, warm=self._warm)
            job = self._sim.start_llm_job(ready)
            responses = (Response(req.t, t, job.first_audio, "A2", "companion", False, "companion"),)
        elif route.actions == ("A1",):
            responses = (Response(req.t, t, ready + self._v("pipeline.tts_first_audio_s"), "A1", "companion",
                                  False, "companion"),)
        return _Pending(t, req.t, "companion", route.actions, reason, req.camera, self._guard.alert, responses)

    # ------------------------------------------------------------ planning
    def _in_hours(self, t: float) -> bool:
        start, end = (parse_clock(c) for c in self.policy.genai_hours)
        return in_clock_window(self.ctx.clock_s(t), start, end)

    def _risk_high(self) -> bool:
        night = self.ctx.is_night(self._t)
        return self._guard.active or any(camera_risk(p, night, c) == "high_risk" for c, p in self._views.items())

    def _window(self, t: float) -> WindowContext:
        persons = [p for group in self._views.values() for p in group]
        return WindowContext(
            t=t, night=self.ctx.is_night(t),
            elder_visible=any(find_elder(g, self.ctx.elder_alone) for g in self._views.values()),
            visitor_present=any(p.identity in VISITORS for p in persons),
            someone_in_high_risk=any(camera_risk(g, self.ctx.is_night(t), c) == "high_risk"
                                     for c, g in self._views.items()),
            alert_active=self._guard.active, last_directed_t=self._last_directed_t)

    def _genai_target(self, t: float) -> bool:
        mode = self.policy.genai_mode
        if mode == "always":
            return True
        if mode == "fixed_hours":
            return self._in_hours(t)
        if self.ctx.is_night(t):
            return False
        if self._sim.llm_resident or self._sim.loading(t):
            return True  # once loaded in the daytime it stays; QoS keeps it off the safety path
        return preload_terms(self._window(t), self._v("models.llm.load_s"), self.preload).open

    def _a3_cameras(self) -> frozenset[str]:
        state = self._guard.state
        return frozenset({state.camera}) if state.active and state.camera else frozenset()

    def _frame_cost(self, cam: str) -> float:
        persons = self._views.get(cam, ())
        face = any(self._face_until.get((cam, p.track_id), -1.0) > self._t + EPS for p in persons)
        face = face or (bool(persons) and cam == "front_door" and self.ctx.is_night(self._t))
        return frame_cost_s(self.spec, self.policy.pipeline, len(persons),
                            full_res_pose=cam in self._a3_cameras(), face_needed=face)

    def _refresh(self, t: float) -> None:
        want = self._genai_target(t)
        if want and not (self._sim.llm_resident or self._sim.loading(t)):
            self._sim.start_load(t, warm=self._warm)
        elif not want and (self._sim.llm_resident or self._sim.loading(t)):
            self._sim.unload(t)
        night, a3 = self.ctx.is_night(t), self._a3_cameras()
        plan = plan_fps(self.policy, self.config.cameras, self._views, night, a3,
                        self._v("device.camera_full_fps"), self.rates)
        if self.policy.qos:
            risks = {c: "high_risk" if c in a3 else camera_risk(self._views.get(c, ()), night, c) for c in plan}
            budget = self._v("device.npu_capacity") * self._v("device.max_utilization")
            plan = degrade_for_budget(plan, risks, lambda c, f: f * self._frame_cost(c), budget, self.rates)
        self._plan = plan
        self._sim.set_vision_demand(t, sum(f * self._frame_cost(c) for c, f in plan.items()))

    # ------------------------------------------------------------ output
    def _emit(self, pending: list[_Pending]) -> list[Decision]:
        fps = tuple(self._plan.items())
        out: list[Decision] = []
        for p in pending:
            times = [r.t_response for r in p.responses if r.action != "A3"]
            latency = round(min(times) - p.t_event, 3) if times else None
            d = Decision(t=p.t, clock=self.ctx.clock(p.t), strategy=self.strategy, agent=p.agent,
                         actions=p.actions, reason=p.reason, latency_s=latency, alert=p.alert,
                         genai=self._sim.genai_state(self._t), fps=fps, simulated=True)
            self._decisions.append(d)
            self._responses.extend(p.responses)
            self._show(d, p.camera, p.safety)
            out.append(d)
        return out

    def _show(self, d: Decision, camera: str | None, safety: bool) -> None:
        held = d.t < self._hud_hold_until - EPS
        shown_safety = self._hud_last is not None and self._hud_last[2]
        if held and (d.actions == ("A0",) or (shown_safety and not safety)):
            return
        self._hud_last = (d, camera, safety)
        if d.actions != ("A0",):
            self._hud_hold_until = d.t + HUD_HOLD_S

    def _hud_cameras(self, camera: str | None) -> str:
        if not self._plan:
            return "-"
        focus = camera if camera in self._plan else max(self._plan, key=lambda c: self._plan[c])
        return f"{len(self._plan)} on · {focus} {self._plan[focus]:g} fps"

    def _hud_genai(self) -> str:
        night = self.ctx.is_night(self._t)
        state = self._sim.genai_state(self._t)
        if state == "loading":
            return f"loading LLM ({self._v('models.llm.load_s'):.1f} s, sim)"
        if state == "npu_window":
            return "LLM loaded (night)" if night else "LLM loaded (day)"
        return "off (night)" if night else "off (vision only)"


def make_engine(strategy: StrategyName, config: RunConfig) -> CareEngine:
    """Build the engine for one strategy (Engine protocol in ahjia/events.py)."""
    return CareEngine(strategy, config)
