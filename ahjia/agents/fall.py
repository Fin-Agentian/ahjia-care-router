"""Fall agent (P:144).

Trigger: the posture of a track goes from upright (stand or sit) to lying, and
the person then stays lying outside the bed or sofa zone (REST_ZONES) for at
least 2 s. Lying in bed or on the sofa is rest, not a fall. The request goes to
the guardrails (safety=True), which answer with A1 + A4 + A3.

Known Stage I limits: a person who is first seen already on the floor, or who
rolls out of bed, does not trigger (P asks for an upright -> ground transition);
a track that leaves the view resets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from ahjia.agents.base import EPS, UPRIGHT, AgentContext, Deadline
from ahjia.events import REST_ZONES, ActionRequest, Observation, PersonObs

REASON = "lying outside bed/sofa ≥ 2 s"


@dataclass(frozen=True)
class FallConfig:
    confirm_s: float = 2.0
    require_transition: bool = True


@dataclass(frozen=True)
class _Track:
    was_upright: bool = False
    lie_since: float | None = None
    fired: bool = False


def _step(track: _Track, person: PersonObs, t: float, config: FallConfig) -> _Track:
    if person.posture in UPRIGHT:
        return _Track(was_upright=True)
    if person.posture != "lie":
        return track  # unknown posture: keep what we know
    if person.zone in REST_ZONES:
        return replace(track, lie_since=None)
    if track.lie_since is None and (track.was_upright or not config.require_transition):
        return replace(track, lie_since=t)
    return track


@dataclass(frozen=True)
class _FallState:
    tracks: Mapping[tuple[str, int], _Track] = field(default_factory=dict)


class FallAgent:
    name = "fall"

    def __init__(self, ctx: AgentContext, config: FallConfig | None = None) -> None:
        self._ctx = ctx
        self._config = config or FallConfig()
        self._state = _FallState()

    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]:
        if obs.camera is None:
            return ()
        tracks = {k: v for k, v in self._state.tracks.items() if k[0] != obs.camera}
        requests: list[ActionRequest] = []
        for person in obs.persons:
            key = (obs.camera, person.track_id)
            track = _step(self._state.tracks.get(key, _Track()), person, obs.t, self._config)
            if self._due(track, obs.t):
                track = replace(track, fired=True)
                requests.append(ActionRequest(
                    t=obs.t, agent="fall", actions=("A1", "A4", "A3"), reason=REASON,
                    camera=obs.camera, safety=True,
                ))
            tracks[key] = track
        self._state = _FallState(tracks=tracks)
        return tuple(requests)

    def _due(self, track: _Track, t: float) -> bool:
        return (track.lie_since is not None and not track.fired
                and t - track.lie_since >= self._config.confirm_s - EPS)

    def on_time(self, t: float) -> tuple[ActionRequest, ...]:
        return ()  # the 2 s rule needs a frame that still shows the person lying

    def next_deadline(self) -> Deadline | None:
        due = [(tr.lie_since + self._config.confirm_s, key[0])
               for key, tr in self._state.tracks.items()
               if tr.lie_since is not None and not tr.fired]
        return min(due, key=lambda d: d[0]) if due else None
