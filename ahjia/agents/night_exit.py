"""Night exit / wandering agent (P:146).

Trigger: the elder is in the front-door zone between 00:00 and 06:00 -> A1 + A4
(the guardrails add A3 because the request has a camera). One request per
visit: the agent re-arms when the elder leaves the zone.
"""

from __future__ import annotations

from dataclasses import dataclass

from ahjia.agents.base import AgentContext, Deadline, find_elder, in_clock_window, parse_clock
from ahjia.events import ActionRequest, Observation


@dataclass(frozen=True)
class NightExitConfig:
    start: str = "00:00:00"
    end: str = "06:00:00"
    zone: str = "front_door"


class NightExitAgent:
    name = "night_exit"

    def __init__(self, ctx: AgentContext, config: NightExitConfig | None = None) -> None:
        self._ctx = ctx
        self._config = config or NightExitConfig()
        self._at_door: frozenset[str] = frozenset()

    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]:
        if obs.camera is None:
            return ()
        elder = find_elder(obs.persons, self._ctx.elder_alone)
        at_door = elder is not None and elder.zone == self._config.zone
        was_at_door = obs.camera in self._at_door
        self._at_door = (self._at_door | {obs.camera}) if at_door else (self._at_door - {obs.camera})
        if not at_door or was_at_door or not self._in_window(obs.t):
            return ()
        reason = f"at the front door at {self._ctx.clock(obs.t)} ({self._config.start[:5]}-{self._config.end[:5]})"
        return (ActionRequest(t=obs.t, agent="night_exit", actions=("A1", "A4"), reason=reason,
                              camera=obs.camera, safety=True),)

    def _in_window(self, t: float) -> bool:
        return in_clock_window(self._ctx.clock_s(t), parse_clock(self._config.start),
                               parse_clock(self._config.end))

    def on_time(self, t: float) -> tuple[ActionRequest, ...]:
        return ()

    def next_deadline(self) -> Deadline | None:
        return None
