"""Strategies S0 / S1 / S2 (P:423-431) and the shared guardrails."""

import pytest

from ahjia.engine import make_engine
from ahjia.events import Observation, PersonObs, RunConfig, SpeechObs
from ahjia.strategies import POLICIES, get_policy, parse_strategies

pytestmark = pytest.mark.unit


def test_three_strategies_match_the_proposal():
    s0, s1, s2 = (get_policy(n) for n in ("S0", "S1", "S2"))
    assert (s0.fps_mode, s0.pipeline, s0.genai_mode, s0.qos, s0.llm_answers_all_speech) == \
        ("full", "all_models", "always", False, True)
    assert (s1.fps_mode, s1.pipeline, s1.genai_mode, s1.qos) == ("fixed", "fixed", "fixed_hours", False)
    assert (s2.fps_mode, s2.pipeline, s2.genai_mode, s2.qos) == ("adaptive", "cascade", "preload_rule", True)
    assert set(POLICIES) == {"S0", "S1", "S2"}


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        get_policy("S9")
    with pytest.raises(ValueError):
        parse_strategies("S0,S7")
    assert parse_strategies(" S2 ,S0") == ("S2", "S0")


def _night_fall_script():
    stand = PersonObs(track_id=1, identity="elder", posture="stand", zone="hallway")
    lie = PersonObs(track_id=1, identity="elder", posture="lie", zone="floor")
    return [
        Observation(t=0.0, camera="hallway", persons=(stand,)),
        Observation(t=12.0, camera="hallway", persons=(lie,)),
        Observation(t=14.0, camera="hallway", persons=(lie,)),
        Observation(t=16.0, speech=SpeechObs(text="I'm OK", source="elder", directed=True)),
        Observation(t=19.0, camera="hallway", persons=(lie,)),
        Observation(t=25.0, speech=SpeechObs(text="I can't get up", source="elder")),
        Observation(t=40.0, app_confirm=True),
    ]


def test_guardrails_are_identical_in_every_strategy():
    config = RunConfig(start_clock="02:10:00", cameras=("living", "hallway", "bathroom_door", "front_door"))
    traces = {}
    for name in ("S0", "S1", "S2"):
        engine = make_engine(name, config)
        safety = []
        for obs in _night_fall_script():
            for d in engine.observe(obs):
                if d.agent != "companion":
                    safety.append((d.t, d.agent, d.actions, d.alert))
        safety.extend((d.t, d.agent, d.actions, d.alert) for d in engine.advance(60.0))
        traces[name] = safety
    assert traces["S0"] == traces["S1"] == traces["S2"]
    alerts = [a for *_, a in traces["S2"]]
    assert "pre_alert" in alerts and "escalated" in alerts and alerts[-1] == "cleared_by_app"
