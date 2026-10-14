"""Contract tests for ahjia/events.py and the CLI parser (leader-owned)."""

import dataclasses
import json

import pytest

from ahjia.cli import build_parser
from ahjia.events import (
    ACTIONS,
    AGENTS,
    REST_ZONES,
    STRATEGIES,
    ActionRequest,
    Decision,
    HudState,
    Observation,
    PersonObs,
    RunConfig,
    SpeechObs,
    decision_to_json,
)

pytestmark = pytest.mark.unit


def test_six_agents_and_six_actions_match_the_proposal():
    assert AGENTS == ("fall", "inactivity", "night_exit", "visitor", "voice_sos", "companion")
    assert tuple(ACTIONS) == ("A0", "A1", "A2", "A3", "A4", "A5")
    assert STRATEGIES == ("S0", "S1", "S2")
    assert REST_ZONES == {"bed", "sofa"}


def test_records_are_immutable():
    person = PersonObs(track_id=1, identity="elder", posture="lie", zone="floor")
    obs = Observation(t=1.0, camera="hallway", persons=(person,))
    request = ActionRequest(t=1.0, agent="fall", actions=("A1", "A4", "A3"), reason="lying", safety=True)
    for record in (person, obs, request, SpeechObs(text="help")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.t = 2.0  # type: ignore[misc]


def test_hud_has_seven_lines_in_the_proposal_order():
    lines = HudState(agent="fall", decision="A1+A4+A3", alert="pre_alert").lines()
    assert len(lines) == 7
    labels = [line.split(":")[0].strip() for line in lines]
    assert labels == ["Agent", "Decision", "Reason", "Latency", "Cameras", "GenAI", "Alert"]
    assert lines[6].endswith("pre_alert")


def test_decision_to_json_is_one_flat_line():
    decision = Decision(
        t=14.0, clock="02:10:14", strategy="S2", agent="fall", actions=("A1", "A4", "A3"),
        reason="lying outside bed/sofa >= 2 s", latency_s=1.6, alert="pre_alert", genai="off",
        fps=(("hallway", 30.0), ("living", 1.5)),
    )
    line = decision_to_json(decision)
    assert "\n" not in line
    record = json.loads(line)
    assert record["fps"] == {"hallway": 30.0, "living": 1.5}
    assert record["actions"] == ["A1", "A4", "A3"]
    assert record["simulated"] is True


def test_run_config_defaults_keep_real_thresholds():
    config = RunConfig(start_clock="02:10:00", cameras=("hallway",))
    assert config.time_scale == 1.0
    assert config.elder_alone is True


def test_cli_parses_replay_and_demo():
    parser = build_parser()
    replay = parser.parse_args(["replay", "--all"])
    assert replay.command == "replay" and replay.all and replay.strategies == "S0,S1,S2"
    demo = parser.parse_args(["demo", "--video", "a.mp4", "--camera", "c.yaml"])
    assert demo.command == "demo" and demo.fps == 10.0 and demo.speech is None
    bench = parser.parse_args(["workbench"])
    assert bench.command == "workbench" and bench.out == "outputs/workbench"
    with pytest.raises(SystemExit):
        parser.parse_args(["replay"])  # needs --scenario or --all
