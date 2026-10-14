"""Shared helpers for the six care agents (layer 2, P:140-149).

Every agent follows the same small interface:

    on_observation(obs) -> tuple[ActionRequest, ...]   # a camera frame or a speech segment
    on_time(t)          -> tuple[ActionRequest, ...]   # clock-driven rules (30 min, 2 h, reminders)
    next_deadline()     -> (t, camera | None) | None   # when a rule could fire next

A deadline with a camera needs a new frame from that camera (for example the
"lying >= 2 s" rule); a deadline without a camera only needs the clock.
Agent state lives in frozen dataclasses that are replaced, never mutated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

from ahjia.events import (
    NIGHT_END,
    NIGHT_START,
    ActionRequest,
    Observation,
    PersonObs,
    RunConfig,
)

DAY_S = 86400
EPS = 1e-9
# Sitting counts as upright: the fall rule looks for upright -> ground (P:144).
UPRIGHT: frozenset[str] = frozenset({"stand", "sit"})

Deadline = tuple[float, str | None]


def parse_clock(text: str) -> int:
    """'HH:MM:SS' -> seconds after midnight."""
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        raise ValueError(f"bad clock value: {text!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_clock(seconds: float) -> str:
    """Seconds after midnight (any day) -> 'HH:MM:SS'."""
    total = int(seconds + EPS) % DAY_S
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def in_clock_window(clock_s: float, start_s: float, end_s: float) -> bool:
    """True when clock_s is in [start, end); windows may wrap past midnight."""
    clock_s %= DAY_S
    if start_s <= end_s:
        return start_s <= clock_s < end_s
    return clock_s >= start_s or clock_s < end_s


@dataclass(frozen=True)
class AgentContext:
    """Run-wide facts every agent needs: wall clock, time scale, routine."""

    start_clock_s: int
    time_scale: float = 1.0
    usual_wake_s: int = 6 * 3600 + 1800
    elder_alone: bool = True

    @classmethod
    def from_run_config(cls, config: RunConfig) -> AgentContext:
        if config.time_scale <= 0:
            raise ValueError("time_scale must be > 0")
        return cls(
            start_clock_s=parse_clock(config.start_clock),
            time_scale=config.time_scale,
            usual_wake_s=parse_clock(config.usual_wake_clock),
            elder_alone=config.elder_alone,
        )

    def clock_s(self, t: float) -> float:
        return (self.start_clock_s + t) % DAY_S

    def clock(self, t: float) -> str:
        return format_clock(self.start_clock_s + t)

    def day_index(self, t: float) -> int:
        return int((self.start_clock_s + t + EPS) // DAY_S)

    def is_night(self, t: float) -> bool:
        """Night is 22:00-07:00: GenAI off by default (P:226)."""
        return in_clock_window(self.clock_s(t), parse_clock(NIGHT_START), parse_clock(NIGHT_END))

    def scaled(self, seconds: float) -> float:
        """Long thresholds (30 min, 2 h, 60 min) shrink with time_scale; 1.0 keeps P's values."""
        return seconds * self.time_scale

    def next_clock_t(self, t: float, clock_s: float) -> float:
        """First run time strictly after t whose wall clock equals clock_s."""
        delta = (clock_s - self.clock_s(t)) % DAY_S
        return t + (delta if delta > EPS else DAY_S)


def normalize_text(text: str) -> str:
    return " ".join(text.replace("’", "'").replace("‘", "'").lower().split())


def _is_ascii(word: str) -> bool:
    return all(ord(ch) < 128 for ch in word)


def find_keyword(text: str, keywords: Iterable[str]) -> str | None:
    """First keyword found in text.

    English keywords match whole words (so "helpful" is not "help");
    Chinese keywords match as substrings (Whisper output has no spaces).
    """
    norm = normalize_text(text)
    for keyword in keywords:
        key = normalize_text(keyword)
        if not key:
            continue
        if _is_ascii(key):
            if re.search(rf"(?<![a-z']){re.escape(key)}(?![a-z])", norm):
                return keyword
        elif key in norm:
            return keyword
    return None


def find_elder(persons: Iterable[PersonObs], elder_alone: bool) -> PersonObs | None:
    """The elder in a frame. When the elder lives alone, a single unknown face counts."""
    persons = tuple(persons)
    for p in persons:
        if p.identity == "elder":
            return p
    if elder_alone and len(persons) == 1 and persons[0].identity == "unknown":
        return persons[0]
    return None


def earliest(*deadlines: Deadline | None) -> Deadline | None:
    found = [d for d in deadlines if d is not None]
    return min(found, key=lambda d: d[0]) if found else None


class Agent(Protocol):
    name: str

    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]: ...

    def on_time(self, t: float) -> tuple[ActionRequest, ...]: ...

    def next_deadline(self) -> Deadline | None: ...


@dataclass(frozen=True)
class Confirmed:
    """A face whose identity stayed the same for confirm_s seconds (simulated ArcFace, P:489)."""

    camera: str
    track_id: int
    identity: str
    first_seen: float
    t: float


@dataclass(frozen=True)
class _Seen:
    identity: str
    first_seen: float
    confirmed: bool = False


@dataclass(frozen=True)
class IdentityTracker:
    """Confirms an identity after it is seen continuously for confirm_s seconds."""

    confirm_s: float = 1.5
    seen: Mapping[tuple[str, int], _Seen] = field(default_factory=dict)

    def update(self, obs: Observation) -> tuple[IdentityTracker, tuple[Confirmed, ...]]:
        if obs.camera is None:
            return self, ()
        kept = {k: v for k, v in self.seen.items() if k[0] != obs.camera}
        confirmed: list[Confirmed] = []
        for p in obs.persons:
            if p.identity == "unknown":
                continue
            key = (obs.camera, p.track_id)
            prev = self.seen.get(key)
            entry = prev if prev and prev.identity == p.identity else _Seen(p.identity, obs.t)
            if not entry.confirmed and obs.t - entry.first_seen >= self.confirm_s - EPS:
                entry = replace(entry, confirmed=True)
                confirmed.append(Confirmed(obs.camera, p.track_id, p.identity, entry.first_seen, obs.t))
            kept[key] = entry
        return replace(self, seen=kept), tuple(confirmed)

    def next_deadline(self) -> Deadline | None:
        pending = [(v.first_seen + self.confirm_s, k[0]) for k, v in self.seen.items() if not v.confirmed]
        return min(pending, key=lambda d: d[0]) if pending else None

    def confirmed_present(self, identities: frozenset[str]) -> bool:
        return any(v.confirmed and v.identity in identities for v in self.seen.values())
