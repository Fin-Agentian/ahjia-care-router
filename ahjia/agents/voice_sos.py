"""Voice SOS agent (P:148, P:245-247).

Keyword spotting on the (simulated) Whisper transcript.

* Immediate words (救命, 起不來, "help", "can't get up") -> A1 + A4 + A5 at once.
* Other SOS words (好暈 / 暈, 胸口痛, "dizzy", "chest pain") -> A1 + A4, then the
  guardrails' 30 s no-answer rule.
* A symptom (other) word right after a simple negation does not count, so an
  answer such as "I'm OK, I'm not dizzy anymore" or 我沒事，已經不暈了 is not a new
  alarm. This is a small, explicit heuristic, not language understanding
  (Stage II): English "not", "no", "no longer", "no more", "isn't", "not
  feeling" directly before the word; Chinese 不, 沒有, 不會, 已經不 directly
  before it. Every other occurrence still counts ("I'm not dizzy but my chest
  pain is bad" is a chest-pain alarm).
* Negation is NEVER applied to the immediate words: "I don't need help" still
  calls (safety first; a known limitation).

P's scenario 2 says "我有點暈" and Whisper catches "暈" (P:501), so the bare
character 暈 is in the default list next to 好暈.
The keywords trigger whatever SpeechObs.source says: the real system cannot
tell a TV voice from the elder yet, so a TV voice saying "help" is treated as
real (telling them apart is Stage II work).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ahjia.agents.base import AgentContext, Deadline, find_keyword, normalize_text
from ahjia.events import ActionRequest, Observation

IMMEDIATE_KEYWORDS: tuple[str, ...] = ("救命", "起不來", "help", "can't get up", "cannot get up")
OTHER_KEYWORDS: tuple[str, ...] = ("好暈", "暈", "胸口痛", "dizzy", "chest pain")
NEGATIONS_EN: tuple[str, ...] = ("not", "no", "no longer", "no more", "isn't", "not feeling")
NEGATIONS_ZH: tuple[str, ...] = ("不", "沒有", "不會", "已經不")
_NEGATED_EN = re.compile(rf"(?<![a-z'])(?:{'|'.join(map(re.escape, NEGATIONS_EN))}) $")


@dataclass(frozen=True)
class VoiceSosConfig:
    immediate: tuple[str, ...] = IMMEDIATE_KEYWORDS
    other: tuple[str, ...] = OTHER_KEYWORDS


def _negated(before: str, english: bool) -> bool:
    """True when the text right before a symptom word ends with a simple negation."""
    if english:
        return _NEGATED_EN.search(before) is not None
    return before.rstrip().endswith(NEGATIONS_ZH)


def find_symptom(text: str, keywords: tuple[str, ...]) -> str | None:
    """First symptom keyword with an occurrence that is not negated (same matching as find_keyword)."""
    norm = normalize_text(text)
    for keyword in keywords:
        key = normalize_text(keyword)
        if not key:
            continue
        english = key.isascii()
        pattern = rf"(?<![a-z']){re.escape(key)}(?![a-z])" if english else re.escape(key)
        if any(not _negated(norm[:m.start()], english) for m in re.finditer(pattern, norm)):
            return keyword
    return None


def classify_sos(text: str, config: VoiceSosConfig) -> tuple[str, bool] | None:
    """(keyword, immediate) for an SOS utterance, or None. Immediate words ignore negation."""
    keyword = find_keyword(text, config.immediate)
    if keyword is not None:
        return keyword, True
    keyword = find_symptom(text, config.other)
    return (keyword, False) if keyword is not None else None


class VoiceSosAgent:
    name = "voice_sos"

    def __init__(self, ctx: AgentContext, config: VoiceSosConfig | None = None) -> None:
        self._ctx = ctx
        self._config = config or VoiceSosConfig()

    @property
    def config(self) -> VoiceSosConfig:
        return self._config

    def on_observation(self, obs: Observation) -> tuple[ActionRequest, ...]:
        speech = obs.speech
        if speech is None:
            return ()
        found = classify_sos(speech.text, self._config)
        if found is None:
            return ()
        keyword, immediate = found
        if immediate:
            return (ActionRequest(t=obs.t, agent="voice_sos", actions=("A1", "A4", "A5"),
                                  reason=f"heard '{keyword}' (immediate A5)", safety=True),)
        return (ActionRequest(t=obs.t, agent="voice_sos", actions=("A1", "A4"),
                              reason=f"heard '{keyword}' (30 s rule)", safety=True),)

    def on_time(self, t: float) -> tuple[ActionRequest, ...]:
        return ()

    def next_deadline(self) -> Deadline | None:
        return None
