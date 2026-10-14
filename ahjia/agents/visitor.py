"""Visitor and caregiver agent (P:147).

Face ID sorts people into family, registered caregiver and stranger (face ID is
simulated in Stage I: the identity comes from the scenario or camera config).
An identity counts once it stayed the same for face_confirm_s (1.5 s, P:489).

* caregiver: log arrival and leaving times (A0) - the service record home-care
  providers need;
* stranger at night (22:00-07:00) -> A4 (safety, to the guardrails);
  stranger in the day -> A0 log;
* family: no request here; the companion agent greets them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ahjia.agents.base import EPS, AgentContext, Confirmed, Deadline, IdentityTracker
from ahjia.events import ActionRequest, Observation


@dataclass(frozen=True)
class VisitorConfig:
    face_confirm_s: float = 1.5
    caregiver_leave_grace_s: float = 300.0


@dataclass(frozen=True)
class _State:
    tracker: IdentityTracker
    caregiver_on_site: bool = False
    caregiver_visible_on: frozenset[str] = frozenset()
    caregiver_gone_since: float | None = None


class VisitorAgent:
    name = "visitor"

    def __init__(self, ctx: AgentContext, config: VisitorConfig | None = None) -> None:
        self._ctx = ctx
        self._config = config or VisitorConfig()
        self._state = _State(tracker=IdentityTracker(confirm_s=self._config.face_confirm_s))

    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]:
        if obs.camera is None:
            return ()
        tracker, confirmed = self._state.tracker.update(obs)
        self._state = self._track_caregiver(replace(self._state, tracker=tracker), obs)
        requests: list[ActionRequest] = []
        for face in confirmed:
            request = self._on_confirmed(face, obs.t)
            if request is not None:
                requests.append(request)
        return tuple(requests)

    def _track_caregiver(self, s: _State, obs: Observation) -> _State:
        present = any(p.identity == "caregiver" for p in obs.persons)
        visible = (s.caregiver_visible_on | {obs.camera}) if present else (s.caregiver_visible_on - {obs.camera})
        gone_since = None if visible else (s.caregiver_gone_since if s.caregiver_gone_since is not None else obs.t)
        return replace(s, caregiver_visible_on=visible, caregiver_gone_since=gone_since)

    def _on_confirmed(self, face: Confirmed, t: float) -> ActionRequest | None:
        if face.identity == "caregiver" and not self._state.caregiver_on_site:
            self._state = replace(self._state, caregiver_on_site=True)
            return ActionRequest(t=t, agent="visitor", actions=("A0",), camera=face.camera,
                                 reason=f"caregiver arrived {self._ctx.clock(face.first_seen)}")
        if face.identity != "stranger":
            return None
        if self._ctx.is_night(t):
            return ActionRequest(t=t, agent="visitor", actions=("A4",), camera=face.camera,
                                 reason=f"stranger at night ({face.camera})", safety=True)
        return ActionRequest(t=t, agent="visitor", actions=("A0",), camera=face.camera,
                             reason=f"stranger in view ({face.camera})")

    def on_time(self, t: float) -> tuple[ActionRequest, ...]:
        s = self._state
        if not s.caregiver_on_site or s.caregiver_gone_since is None:
            return ()
        if t < s.caregiver_gone_since + self._config.caregiver_leave_grace_s - EPS:
            return ()
        self._state = replace(s, caregiver_on_site=False)
        return (ActionRequest(t=t, agent="visitor", actions=("A0",),
                              reason=f"caregiver left {self._ctx.clock(s.caregiver_gone_since)}"),)

    def next_deadline(self) -> Deadline | None:
        s = self._state
        leave = None
        if s.caregiver_on_site and s.caregiver_gone_since is not None:
            leave = (s.caregiver_gone_since + self._config.caregiver_leave_grace_s, None)
        faces = s.tracker.next_deadline()
        found = [d for d in (leave, faces) if d is not None]
        return min(found, key=lambda d: d[0]) if found else None
