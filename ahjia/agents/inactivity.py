"""Inactivity agent (P:145).

Three rules, each asks for A1 (a caring template voice). The guardrails then
wait 60 s for an answer before sending A4 (P:244):

* in the bathroom >= 30 min. "In the bathroom" means seen in the bathroom zone,
  or last seen at the bathroom door and then out of every camera's view
  (bathrooms usually have no camera);
* no movement >= 2 h in the daytime (07:00-22:00) while the elder is in view.
  Movement = a posture or zone change, appearing on a camera, or elder speech;
* not up 60 min after the usual wake time: at usual_wake + 60 min the elder has
  not been seen upright since usual_wake - 60 min.

All three thresholds are multiplied by RunConfig.time_scale (1.0 = P's values).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ahjia.agents.base import (
    EPS,
    UPRIGHT,
    AgentContext,
    Deadline,
    earliest,
    find_elder,
    parse_clock,
)
from ahjia.events import NIGHT_END, ActionRequest, Observation

BATHROOM_ZONES = frozenset({"bathroom", "bathroom_door"})


@dataclass(frozen=True)
class InactivityConfig:
    bathroom_s: float = 30 * 60
    still_day_s: float = 2 * 3600
    late_wake_s: float = 60 * 60
    wake_window_before_s: float = 60 * 60


@dataclass(frozen=True)
class _State:
    visible_on: frozenset[str] = frozenset()
    zone: str | None = None
    posture: str | None = None
    last_seen_t: float | None = None
    last_move_t: float = 0.0
    bathroom_since: float | None = None
    upright_seen_t: float | None = None
    fired_bathroom: bool = False
    fired_still: bool = False
    next_wake_check: float | None = None


class InactivityAgent:
    name = "inactivity"

    def __init__(self, ctx: AgentContext, config: InactivityConfig | None = None) -> None:
        self._ctx = ctx
        self._config = config or InactivityConfig()
        check_clock = ctx.usual_wake_s + ctx.scaled(self._config.late_wake_s)
        self._state = _State(next_wake_check=ctx.next_clock_t(-EPS, check_clock))

    # ------------------------------------------------------------ observations
    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]:
        if obs.speech is not None and obs.speech.source == "elder":
            self._state = self._moved(self._state, obs.t)
        if obs.camera is not None:
            self._state = self._see(self._state, obs)
        return ()

    def _moved(self, s: _State, t: float) -> _State:
        return replace(s, last_move_t=t, fired_still=False)

    def _see(self, s: _State, obs: Observation) -> _State:
        elder = find_elder(obs.persons, self._ctx.elder_alone)
        if elder is None:
            visible = s.visible_on - {obs.camera}
            s = replace(s, visible_on=visible)
            if not visible and s.zone in BATHROOM_ZONES and s.bathroom_since is None:
                # First frame that no longer sees her (frames with no change are skipped in replay).
                s = replace(s, bathroom_since=obs.t)
            return s
        appeared = not s.visible_on
        changed = appeared or elder.zone != s.zone or elder.posture != s.posture
        s = replace(s, visible_on=s.visible_on | {obs.camera}, zone=elder.zone,
                    posture=elder.posture, last_seen_t=obs.t)
        if elder.posture in UPRIGHT:
            s = replace(s, upright_seen_t=obs.t)
        if elder.zone == "bathroom":
            s = replace(s, bathroom_since=s.bathroom_since if s.bathroom_since is not None else obs.t)
        else:
            s = replace(s, bathroom_since=None, fired_bathroom=False)
        return self._moved(s, obs.t) if changed else s

    # ------------------------------------------------------------ clock rules
    def on_time(self, t: float) -> tuple[ActionRequest, ...]:
        requests: list[ActionRequest] = []
        s = self._state
        if self._bathroom_due(s, t):
            s = replace(s, fired_bathroom=True)
            requests.append(self._request(t, "in the bathroom ≥ 30 min"))
        if self._still_due(s, t):
            s = replace(s, fired_still=True)
            requests.append(self._request(t, "no movement ≥ 2 h (daytime)"))
        if s.next_wake_check is not None and t >= s.next_wake_check - EPS:
            if not self._was_up(s, s.next_wake_check):
                requests.append(self._request(t, "not up 60 min after the usual wake time"))
            check_clock = self._ctx.usual_wake_s + self._ctx.scaled(self._config.late_wake_s)
            s = replace(s, next_wake_check=self._ctx.next_clock_t(t, check_clock))
        self._state = s
        return tuple(requests)

    def _request(self, t: float, reason: str) -> ActionRequest:
        if self._ctx.time_scale != 1.0:
            reason = f"{reason} (time_scale {self._ctx.time_scale:g})"
        return ActionRequest(t=t, agent="inactivity", actions=("A1",), reason=reason, safety=True)

    def _bathroom_due(self, s: _State, t: float) -> bool:
        return (s.bathroom_since is not None and not s.fired_bathroom
                and t - s.bathroom_since >= self._ctx.scaled(self._config.bathroom_s) - EPS)

    def _still_due(self, s: _State, t: float) -> bool:
        return (bool(s.visible_on) and not s.fired_still and not self._ctx.is_night(t)
                and t - s.last_move_t >= self._ctx.scaled(self._config.still_day_s) - EPS)

    def _was_up(self, s: _State, check_t: float) -> bool:
        if s.visible_on and s.posture in UPRIGHT:
            return True
        window = self._ctx.scaled(self._config.late_wake_s + self._config.wake_window_before_s)
        return s.upright_seen_t is not None and s.upright_seen_t >= check_t - window - EPS

    def next_deadline(self) -> Deadline | None:
        s = self._state
        bathroom = None
        if s.bathroom_since is not None and not s.fired_bathroom:
            bathroom = (s.bathroom_since + self._ctx.scaled(self._config.bathroom_s), None)
        still = None
        if s.visible_on and not s.fired_still:
            due = s.last_move_t + self._ctx.scaled(self._config.still_day_s)
            if self._ctx.is_night(due):
                due = self._ctx.next_clock_t(due, parse_clock(NIGHT_END))
            still = (due, None)
        wake = (s.next_wake_check, None) if s.next_wake_check is not None else None
        return earliest(bathroom, still, wake)
