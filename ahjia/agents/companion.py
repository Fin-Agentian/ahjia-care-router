"""Companion and daily-report agent (P:149).

It asks for A2 (wake the LLM) when:

* a family member comes in (first time today, after the 1.5 s face check);
* someone talks to AhJia (SpeechObs.directed, simulated in Stage I);
* the elder says a soft SOS word (dizzy, chest pain): a follow-up question,
  after the guardrails already sent A1 + A4;
* a reminder time or the daily summary time comes.

Speech that is not directed at AhJia (TV, people talking to each other) gets
an A0 request, so the decision log shows why the LLM stayed asleep. The A2 vs
A0 choice rests only on SpeechObs.directed (the talking-to-AhJia features of
P:197); the speaker source label is used for the log text only.
The Care Router decides whether an A2 can really run (GenAI window, night ->
template voice, strategy). This agent never makes a safety decision: all its
requests have safety=False.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ahjia.agents.base import EPS, AgentContext, Deadline, IdentityTracker, earliest, parse_clock
from ahjia.agents.voice_sos import VoiceSosConfig, classify_sos
from ahjia.events import ActionRequest, Observation, SpeechObs


@dataclass(frozen=True)
class CompanionConfig:
    reminders: tuple[str, ...] = ("09:00:00", "18:00:00")
    summary_clock: str | None = "21:00:00"
    face_confirm_s: float = 1.5
    sos: VoiceSosConfig = field(default_factory=VoiceSosConfig)


@dataclass(frozen=True)
class _State:
    tracker: IdentityTracker
    greeted_days: frozenset[int] = frozenset()
    clock_events: tuple[tuple[float, str, str], ...] = ()  # (due t, clock, label)


class CompanionAgent:
    name = "companion"

    def __init__(self, ctx: AgentContext, config: CompanionConfig | None = None) -> None:
        self._ctx = ctx
        self._config = config or CompanionConfig()
        events = [(clock, f"medication reminder {clock[:5]}") for clock in self._config.reminders]
        if self._config.summary_clock:
            events.append((self._config.summary_clock, "daily summary for the family app"))
        scheduled = tuple(sorted((ctx.next_clock_t(-EPS, parse_clock(c)), c, label) for c, label in events))
        self._state = _State(tracker=IdentityTracker(self._config.face_confirm_s), clock_events=scheduled)

    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]:
        if obs.speech is not None:
            return self._on_speech(obs.t, obs.speech)
        if obs.camera is None:
            return ()
        tracker, confirmed = self._state.tracker.update(obs)
        self._state = replace(self._state, tracker=tracker)
        requests: list[ActionRequest] = []
        for face in confirmed:
            day = self._ctx.day_index(face.first_seen)
            if face.identity != "family" or day in self._state.greeted_days:
                continue
            self._state = replace(self._state, greeted_days=self._state.greeted_days | {day})
            requests.append(ActionRequest(t=face.first_seen, agent="companion", actions=("A2",),
                                          reason="family · first today", camera=face.camera))
        return tuple(requests)

    def _on_speech(self, t: float, speech: SpeechObs) -> tuple[ActionRequest, ...]:
        sos = classify_sos(speech.text, self._config.sos)
        if sos is not None and sos[1]:
            return (self._a0(t, f"immediate SOS '{sos[0]}' \u00b7 safety path"),)
        if sos is not None:
            return (ActionRequest(t=t, agent="companion", actions=("A2",),
                                  reason=f"follow-up after SOS word '{sos[0]}'"),)
        if speech.directed:
            return (ActionRequest(t=t, agent="companion", actions=("A2",),
                                  reason=f"{speech.source} talks to AhJia"),)
        return (self._a0(t, f"{speech.source} talking · not directed"),)

    @staticmethod
    def _a0(t: float, reason: str) -> ActionRequest:
        return ActionRequest(t=t, agent="companion", actions=("A0",), reason=reason)

    def on_time(self, t: float) -> tuple[ActionRequest, ...]:
        requests: list[ActionRequest] = []
        events: list[tuple[float, str, str]] = []
        for due, clock, label in self._state.clock_events:
            if due <= t + EPS:
                requests.append(ActionRequest(t=due, agent="companion", actions=("A2",), reason=label))
                due = self._ctx.next_clock_t(t, parse_clock(clock))
            events.append((due, clock, label))
        self._state = replace(self._state, clock_events=tuple(sorted(events)))
        return tuple(requests)

    def next_deadline(self) -> Deadline | None:
        clock = (self._state.clock_events[0][0], None) if self._state.clock_events else None
        return earliest(self._state.tracker.next_deadline(), clock)
