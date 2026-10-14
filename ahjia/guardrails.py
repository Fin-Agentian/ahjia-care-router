"""Deterministic safety guardrails (DESIGN.md "Guardrails", P:243-251).

One alert state machine, shared unchanged by S0, S1 and S2. No learned or
tunable policy is involved, and nothing here waits for a generative model.

Triggers (fall, SOS keyword, night exit) -> A1 + A4 at once, plus A3 when the
request comes from a camera; the alert becomes "pre_alert".

* 救命 / 起不來 / "help" / "can't get up" (the voice agent marks them with A5),
  from any voice and at any time, also during an alert -> A1 + A4 + A5 at once,
  alert "escalated". While the alert is escalated and the last call (A5) was
  placed less than ESCALATE_COOLDOWN_S (60 s) ago, an immediate word is only
  logged ("already escalated · call placed N s ago"): repeated "help" places
  one call, not a burst. It still voids an earlier "I'm OK".
* Any other trigger -> A5 when the elder gives no answer for 30 s (P:247).
  Elder speech after the trigger counts as an answer and stops the timer,
  unless it still contains an SOS keyword: an SOS word is a new alarm, never an
  answer (and never an "I'm OK", even next to "I'm OK"). A symptom word right
  after a simple negation ("not dizzy", 不暈, 沒有胸口痛) does not count, so
  "I'm OK, I'm not dizzy anymore" is an answer and an "I'm OK" (heuristic in
  agents/voice_sos.py; Stage II uses real language understanding). The trigger
  utterance itself does not count. An answer does NOT clear the alert: it stays
  until one of the two clear paths.
* A trigger during an alert is a new event: it re-issues A1 (+ A3 for a camera
  trigger) + A4 with its own reason (the family text names the new symptom) and
  arms the 30 s rule again. If the timer is still running, the earlier deadline
  is kept (no new event may postpone A5). Only a repeated trigger from an agent
  whose timer is still running is merely logged (for example a second camera
  that sees the same fall); every SOS word re-issues. While the alert is
  escalated, a non-immediate trigger always re-issues A1 (+ A3) + A4 (an
  updated family text); it arms no new 30 s timer while the last call is less
  than ESCALATE_COOLDOWN_S old (that call covers it), and arms one again after
  that, so a fall an hour after an unanswered call can call again.

Known limitation: negation is never applied to the immediate words, so
"I don't need help" still places a call (safety first).
* Inactivity is not an instant trigger: A1 first (a check-in); no answer in
  60 s -> A4 and "pre_alert"; then the same 30 s rule applies. An app
  confirmation cancels a pending check-in.
* A stranger at night (visitor agent) -> A4 only; no automatic A5 and no A3
  (P:147 asks for A4 only), so no camera goes to full rate.

Only two ways to clear an alert (P:248-250):
  (a) the elder says 我沒事 / "I'm OK" and then STANDS (posture "stand") for
      >= 5 s -> "cleared_by_elder"; the family gets "cleared by the elder".
      P:249 says 直立 (upright); for the clear we read it as standing, as the
      deck and video say. Sitting does not count, and sitting down during the
      5 s resets the timer. (The fall trigger still treats sitting as upright.)
  (b) family or a caregiver confirms in the app -> "cleared_by_app".
There is no other method that lowers or clears an alert.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ahjia.agents.base import EPS, Deadline, find_keyword
from ahjia.agents.voice_sos import VoiceSosConfig, classify_sos
from ahjia.events import ActionCode, ActionRequest, AgentName, AlertState

ESCALATE_NO_ANSWER_S = 30.0
CHECKIN_NO_ANSWER_S = 60.0
STANDING_CLEAR_S = 5.0
ESCALATE_COOLDOWN_S = 60.0
OK_KEYWORDS: tuple[str, ...] = ("我沒事", "i'm ok", "i am ok", "i'm okay", "i am okay", "im ok")
TRIGGER_AGENTS: frozenset[str] = frozenset({"fall", "voice_sos", "night_exit"})

OutputKind = Literal["trigger", "escalate", "checkin", "notify", "clear", "log"]


def is_ok_phrase(text: str) -> bool:
    return find_keyword(text, OK_KEYWORDS) is not None


def is_sos_phrase(text: str) -> bool:
    """True when the voice SOS agent hears a keyword in text (a negated symptom word does not
    count): a new alarm, never an answer."""
    return classify_sos(text, VoiceSosConfig()) is not None


@dataclass(frozen=True)
class GuardrailOutput:
    """One guardrail decision. t is when it is taken; t_event is the event it answers."""

    t: float
    t_event: float
    agent: AgentName | None
    actions: tuple[ActionCode, ...]
    reason: str
    alert: AlertState
    kind: OutputKind
    camera: str | None = None


@dataclass(frozen=True)
class GuardrailState:
    """camera is the camera under A3 (look closer); None when no camera trigger asked for A3.
    armed_by holds the agents whose triggers armed the running 30 s timer.
    called_t is when the last call (A5) was placed."""

    alert: AlertState = "none"
    agent: AgentName | None = None
    camera: str | None = None
    trigger_t: float | None = None
    escalate_at: float | None = None
    armed_by: frozenset[str] = frozenset()
    checkin_t: float | None = None
    checkin_deadline: float | None = None
    checkin_reason: str = ""
    ok_said_t: float | None = None
    called_t: float | None = None
    standing_since: float | None = None
    standing_camera: str | None = None

    @property
    def active(self) -> bool:
        return self.alert in ("pre_alert", "escalated")


Step = tuple[GuardrailState, tuple[GuardrailOutput, ...]]


def _out(s: GuardrailState, t: float, actions: tuple[ActionCode, ...], reason: str, kind: OutputKind,
         agent: AgentName | None = None, camera: str | None = None, t_event: float | None = None) -> GuardrailOutput:
    return GuardrailOutput(t=t, t_event=t if t_event is None else t_event, agent=agent or s.agent,
                           actions=actions, reason=reason, alert=s.alert, kind=kind,
                           camera=camera if camera is not None else s.camera)


def apply_request(s: GuardrailState, req: ActionRequest) -> Step:
    if not req.safety:
        raise ValueError("guardrails only take safety requests; companion requests go to the router")
    if req.agent == "inactivity":
        return _checkin(s, req)
    if req.agent == "visitor":
        return _notify(s, req)
    if req.agent not in TRIGGER_AGENTS:
        raise ValueError(f"unknown safety agent: {req.agent}")
    return _trigger(s, req)


def _log(s: GuardrailState, req: ActionRequest, why: str) -> Step:
    return s, (_out(s, req.t, ("A0",), f"{why}: {req.reason}", "log", agent=req.agent, camera=req.camera),)


def _trigger(s: GuardrailState, req: ActionRequest) -> Step:
    if "A5" in req.actions:
        return _escalate_now(s, req)
    timer_running = s.active and s.escalate_at is not None
    if timer_running and req.agent in s.armed_by and req.agent != "voice_sos":
        return _log(s, req, "same alert, 30 s rule already running")
    return _alert(s, req, timer_running)


def _look_camera(s: GuardrailState, req: ActionRequest) -> str | None:
    """A3 moves to the newest camera trigger; a voice trigger keeps the current one."""
    if req.camera is not None:
        return req.camera
    return s.camera if s.active else None


def _new_event(s: GuardrailState, req: ActionRequest, **changes: object) -> GuardrailState:
    """Fields every trigger sets: a new event voids an earlier "I'm OK" and any check-in."""
    return replace(s, agent=req.agent, camera=_look_camera(s, req), trigger_t=req.t, checkin_t=None,
                   checkin_deadline=None, ok_said_t=None, **changes)


def _call_recent(s: GuardrailState, t: float) -> bool:
    """Escalated and the last call (A5) was placed less than ESCALATE_COOLDOWN_S before t."""
    return s.alert == "escalated" and s.called_t is not None and t - s.called_t < ESCALATE_COOLDOWN_S - EPS


def _escalate_now(s: GuardrailState, req: ActionRequest) -> Step:
    if _call_recent(s, req.t):
        s = replace(s, ok_said_t=None)  # a call is in progress; a new "help" still voids an "I'm OK"
        return _log(s, req, f"already escalated · call placed {round(req.t - s.called_t, 1):g} s ago")
    actions: tuple[ActionCode, ...] = ("A1", "A4") + (("A3",) if req.camera else ()) + ("A5",)
    s = _new_event(s, req, alert="escalated", escalate_at=None, armed_by=frozenset(), called_t=req.t)
    return s, (_out(s, req.t, actions, req.reason, "trigger", agent=req.agent, camera=req.camera),)


def _alert(s: GuardrailState, req: ActionRequest, timer_running: bool) -> Step:
    """A1 + A4 (+ A3) and the 30 s rule, for a first trigger or a new event during an alert.
    Escalated with a recent call: the family text is updated but no timer is armed."""
    actions: tuple[ActionCode, ...] = ("A1", "A4") + (("A3",) if req.camera else ())
    if _call_recent(s, req.t):
        s = _new_event(s, req, escalate_at=None, armed_by=frozenset())
        return s, (_out(s, req.t, actions, req.reason, "trigger", agent=req.agent, camera=req.camera),)
    deadline = req.t + ESCALATE_NO_ANSWER_S
    escalate_at = min(s.escalate_at, deadline) if timer_running and s.escalate_at is not None else deadline
    armed = (s.armed_by if timer_running else frozenset()) | {req.agent}
    alert: AlertState = "escalated" if s.alert == "escalated" else "pre_alert"
    s = _new_event(s, req, alert=alert, escalate_at=escalate_at, armed_by=armed)
    return s, (_out(s, req.t, actions, req.reason, "trigger", agent=req.agent, camera=req.camera),)


def _checkin(s: GuardrailState, req: ActionRequest) -> Step:
    if s.active:
        return _log(s, req, "already in pre-alert")
    if s.checkin_deadline is not None:
        return _log(s, req, "check-in already running")
    s = replace(s, checkin_t=req.t, checkin_deadline=req.t + CHECKIN_NO_ANSWER_S, checkin_reason=req.reason)
    return s, (_out(s, req.t, ("A1",), req.reason, "checkin", agent="inactivity", camera=req.camera),)


def _notify(s: GuardrailState, req: ActionRequest) -> Step:
    """A4 only (P:147): no 30 s timer and no A3, so the state keeps no camera to look closer at."""
    if s.active:
        return _log(s, req, "already in pre-alert")
    s = replace(s, alert="pre_alert", agent=req.agent, camera=None, trigger_t=req.t, escalate_at=None,
                armed_by=frozenset(), checkin_t=None, checkin_deadline=None, ok_said_t=None)
    return s, (_out(s, req.t, ("A4",), req.reason, "notify", agent=req.agent, camera=req.camera),)


def apply_elder_speech(s: GuardrailState, t: float, text: str) -> Step:
    if is_sos_phrase(text):
        return s, ()  # a new alarm for the voice SOS agent, never an answer or an "I'm OK"
    # A negated symptom ("I'm not dizzy anymore") is not an SOS word: plain answer, or "I'm OK" below.
    outputs: list[GuardrailOutput] = []
    if s.checkin_deadline is not None and s.checkin_t is not None and t > s.checkin_t + EPS:
        s = replace(s, checkin_t=None, checkin_deadline=None)
        outputs.append(_out(s, t, ("A0",), "elder answered the check-in", "log", agent="inactivity"))
    if not s.active or s.trigger_t is None or t <= s.trigger_t + EPS:
        return s, tuple(outputs)
    if s.escalate_at is not None:
        s = replace(s, escalate_at=None, armed_by=frozenset())
        outputs.append(_out(s, t, ("A0",), "elder answered (30 s rule stopped; alert stays)", "log"))
    if is_ok_phrase(text):
        s = replace(s, ok_said_t=t)
    s, cleared = _maybe_clear(s, t)
    return s, tuple(outputs) + cleared


def apply_elder_posture(s: GuardrailState, t: float, standing: bool | None, camera: str | None) -> Step:
    """standing: True = posture "stand"; False = sitting or lying; None = elder not in view."""
    if standing:
        since = s.standing_since if s.standing_since is not None else t
        s = replace(s, standing_since=since, standing_camera=camera)
    else:
        s = replace(s, standing_since=None, standing_camera=None)
    return _maybe_clear(s, t)


def _clear_time(s: GuardrailState) -> float | None:
    if not s.active or s.ok_said_t is None or s.standing_since is None:
        return None
    return max(s.ok_said_t, s.standing_since) + STANDING_CLEAR_S


def _maybe_clear(s: GuardrailState, t: float) -> Step:
    due = _clear_time(s)
    if due is None or t < due - EPS:
        return s, ()
    s = replace(s, alert="cleared_by_elder", escalate_at=None, armed_by=frozenset(), ok_said_t=None)
    return s, (_out(s, t, ("A4",), "cleared by the elder (I'm OK + standing ≥ 5 s)", "clear"),)


def apply_app_confirm(s: GuardrailState, t: float) -> Step:
    if s.active:
        s = replace(s, alert="cleared_by_app", escalate_at=None, armed_by=frozenset(), ok_said_t=None,
                    checkin_t=None, checkin_deadline=None)
        return s, (_out(s, t, ("A0",), "confirmed in the app by family or caregiver", "clear"),)
    if s.checkin_deadline is not None:
        s = replace(s, checkin_t=None, checkin_deadline=None)
        return s, (_out(s, t, ("A0",), "confirmed in the app by family or caregiver (check-in cancelled)",
                        "log", agent="inactivity"),)
    return s, ()


def apply_time(s: GuardrailState, t: float) -> Step:
    outputs: list[GuardrailOutput] = []
    if s.checkin_deadline is not None and t >= s.checkin_deadline - EPS:
        due = s.checkin_deadline
        s = replace(s, alert="pre_alert", trigger_t=due, escalate_at=due + ESCALATE_NO_ANSWER_S,
                    armed_by=frozenset({"inactivity"}), agent="inactivity", camera=None, checkin_t=None,
                    checkin_deadline=None, ok_said_t=None)
        outputs.append(_out(s, due, ("A4",), f"no answer 60 s after check-in: {s.checkin_reason}", "trigger"))
    if s.escalate_at is not None and t >= s.escalate_at - EPS:
        due = s.escalate_at
        s = replace(s, alert="escalated", escalate_at=None, armed_by=frozenset(), called_t=due)
        outputs.append(_out(s, due, ("A5",), "no answer for 30 s", "escalate"))
    return s, tuple(outputs)


def next_deadline(s: GuardrailState) -> Deadline | None:
    found: list[Deadline] = []
    if s.checkin_deadline is not None:
        found.append((s.checkin_deadline, None))
    if s.escalate_at is not None:
        found.append((s.escalate_at, None))
    clear = _clear_time(s)
    if clear is not None:
        found.append((clear, s.standing_camera))
    return min(found, key=lambda d: d[0]) if found else None


class Guardrails:
    """Stateful wrapper around the pure transition functions above."""

    def __init__(self) -> None:
        self._state = GuardrailState()

    @property
    def state(self) -> GuardrailState:
        return self._state

    @property
    def alert(self) -> AlertState:
        return self._state.alert

    @property
    def active(self) -> bool:
        return self._state.active

    def _run(self, step: Step) -> tuple[GuardrailOutput, ...]:
        self._state, outputs = step
        return outputs

    def on_request(self, req: ActionRequest) -> tuple[GuardrailOutput, ...]:
        return self._run(apply_request(self._state, req))

    def on_elder_speech(self, t: float, text: str) -> tuple[GuardrailOutput, ...]:
        return self._run(apply_elder_speech(self._state, t, text))

    def on_elder_posture(self, t: float, standing: bool | None, camera: str | None) -> tuple[GuardrailOutput, ...]:
        return self._run(apply_elder_posture(self._state, t, standing, camera))

    def on_app_confirm(self, t: float) -> tuple[GuardrailOutput, ...]:
        return self._run(apply_app_confirm(self._state, t))

    def advance(self, t: float) -> tuple[GuardrailOutput, ...]:
        return self._run(apply_time(self._state, t))

    def next_deadline(self) -> Deadline | None:
        return next_deadline(self._state)
