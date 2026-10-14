"""Replay: scenario files -> frame-rate sampling -> engine -> decision logs (DESIGN.md "Scenario file")."""

import argparse
import json
from pathlib import Path

import pytest

from ahjia.replay import (
    SCENARIO_DIR,
    find_scenarios,
    load_scenario,
    main,
    max_busy_cameras,
    run_scenario,
)

pytestmark = pytest.mark.unit

ALL_FILES = sorted(SCENARIO_DIR.glob("*.yaml")) + sorted((SCENARIO_DIR / "scripted").glob("*.yaml"))


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.stem)
def test_every_scenario_file_loads(path: Path):
    scn = load_scenario(path)
    assert scn.name == path.stem and scn.cameras and scn.duration_s > 0
    assert all(0 <= e.t <= scn.duration_s for e in scn.events)
    assert all(e.camera is None or e.camera in scn.cameras for e in scn.events)


def test_demo_scenarios_and_the_24h_day_are_present():
    names = {p.stem for p in SCENARIO_DIR.glob("*.yaml")}
    assert {"family_arrives", "dizzy", "night_fall", "day24h"} <= names
    day = load_scenario(SCENARIO_DIR / "day24h.yaml")
    assert day.synthetic and day.duration_s == 86400
    assert sum(1 for e in day.events if e.speech is not None and e.speech.source == "tv") > 100  # repeat expanded


def test_clock_times_and_repeat_are_expanded(tmp_path: Path):
    path = tmp_path / "mini.yaml"
    path.write_text(
        "name: mini\ntitle: t\nsource: s\nstart_clock: '10:00:00'\ncameras: [living]\nduration_s: 120\n"
        "labels: [{t: '10:00:30', kind: benign, what: x}]\n"
        "events:\n  - {t: '10:00:10', speech: {text: hi, source: tv}, repeat: {every_s: 20, until: 70}}\n",
        encoding="utf-8")
    scn = load_scenario(path)
    assert [e.t for e in scn.events] == [10.0, 30.0, 50.0, 70.0]
    assert scn.labels[0].t == 30.0 and scn.labels[0].window_s == 10.0


@pytest.mark.parametrize("text, message", [
    ("cameras: [living]\nduration_s: 10\nevents: [{t: 1, camera: kitchen, persons: []}]\n", "camera"),
    ("cameras: [living]\nduration_s: 10\nevents: [{t: 1, speech: {text: hi}}]\nlabels: [{t: 1, kind: odd, what: x}]\n", "kind"),
    ("cameras: [living]\nduration_s: 10\nevents: [{t: 99, app_confirm: true}]\n", "duration"),
])
def test_bad_scenarios_are_rejected(tmp_path: Path, text: str, message: str):
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\ntitle: t\nsource: s\nstart_clock: '10:00:00'\n" + text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_scenario(path)


def test_find_scenarios():
    assert [p.stem for p in find_scenarios("night_fall", all_=False)] == ["night_fall"]
    assert [p.stem for p in find_scenarios("tv_help", all_=False)] == ["tv_help"]
    everything = {p.stem for p in find_scenarios(None, all_=True)}
    assert {"day24h", "night_fall", "tv_help"} <= everything
    with pytest.raises(FileNotFoundError):
        find_scenarios("nope", all_=False)


def test_max_busy_cameras_counts_people_after_simultaneous_moves():
    day = load_scenario(SCENARIO_DIR / "day24h.yaml")
    assert max_busy_cameras(day) == 2  # caregiver and elder in different rooms
    assert max_busy_cameras(load_scenario(SCENARIO_DIR / "night_fall.yaml")) == 1


def _alerts(result):
    return [d.alert for d in result.decisions]


def test_night_fall_replay_escalates_in_every_strategy():
    scn = load_scenario(SCENARIO_DIR / "night_fall.yaml")
    for strategy in ("S0", "S1", "S2"):
        result = run_scenario(scn, strategy)
        fall = [d for d in result.decisions if d.agent == "fall"]
        assert fall and fall[0].actions == ("A1", "A4", "A3") and 14.0 <= fall[0].t <= 14.2
        a5 = [d for d in result.decisions if "A5" in d.actions]
        assert a5 and a5[0].t == pytest.approx(19.0)
        assert result.totals.duration_s == pytest.approx(60.0)


def test_frame_rate_changes_detection_time():
    scn = load_scenario(SCENARIO_DIR / "scripted" / "night_front_door.yaml")
    times = {}
    for strategy in ("S1", "S2"):
        result = run_scenario(scn, strategy)
        times[strategy] = [d.t for d in result.decisions if d.agent == "night_exit"][0]
    assert times["S1"] == pytest.approx(6.0)        # 10 FPS grid hits t = 6.0
    assert 6.0 <= times["S2"] <= 6.5                # empty front door watched at 2 FPS at night


@pytest.mark.parametrize("name, check", [
    ("sofa_lie_not_fall", lambda alerts, ds: "pre_alert" not in alerts),
    ("ok_but_lying", lambda alerts, ds: alerts[-1] == "pre_alert" and "escalated" not in alerts),
    ("ok_sits_up", lambda alerts, ds: alerts[-1] == "pre_alert" and "escalated" not in alerts),
    ("ok_and_upright", lambda alerts, ds: "cleared_by_elder" in alerts and "escalated" not in alerts
     and [d.t for d in ds if d.alert == "cleared_by_elder"][0] == pytest.approx(29.0, abs=0.1)),
    ("app_confirm", lambda alerts, ds: alerts[-1] == "cleared_by_app" and "escalated" not in alerts),
    ("dizzy_en", lambda alerts, ds: [d.t for d in ds if d.actions == ("A5",)] == [pytest.approx(34.0)]),
    ("night_front_door", lambda alerts, ds: any({"A1", "A4"} <= set(d.actions) for d in ds)),
    ("tv_help", lambda alerts, ds: "escalated" in alerts and any(d.actions == ("A1", "A4", "A5") for d in ds)),
    ("stranger_night", lambda alerts, ds: any(d.actions == ("A4",) and d.agent == "visitor" for d in ds)),
    ("help_zh", lambda alerts, ds: any(d.actions == ("A1", "A4", "A5") for d in ds)),
    ("bathroom_long", lambda alerts, ds: [d.actions for d in ds if d.agent == "inactivity"][:3]
     == [("A1",), ("A4",), ("A5",)]),
])
def test_scripted_cases_follow_the_rules_in_every_strategy(name, check):
    scn = load_scenario(SCENARIO_DIR / "scripted" / f"{name}.yaml")
    for strategy in ("S0", "S1", "S2"):
        result = run_scenario(scn, strategy)
        assert check(_alerts(result), result.decisions), (name, strategy)


def test_main_writes_logs_and_report(tmp_path: Path):
    args = argparse.Namespace(scenario="night_fall", all=False, strategies="S0,S2", out=str(tmp_path))
    assert main(args) == 0
    log = tmp_path / "replay" / "night_fall" / "S2.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines and all(json.loads(line)["strategy"] == "S2" for line in lines)
    assert (tmp_path / "replay" / "night_fall" / "S0.jsonl").exists()
    for name in ("report.md", "kpi.csv", "kpi.png", "kpi.json"):
        assert (tmp_path / name).stat().st_size > 0
    responses = [json.loads(line) for line in
                 (tmp_path / "replay" / "night_fall" / "S2.responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert responses and {"t_event", "agent", "action", "t_done", "latency_s"} <= set(responses[0])
    assert {r["action"] for r in responses} == {"A1", "A4", "A5"}
    first = responses[0]
    assert first["latency_s"] == pytest.approx(first["t_done"] - first["t_event"], abs=1e-3)


def test_main_reports_unknown_scenarios(tmp_path: Path):
    args = argparse.Namespace(scenario="nope", all=False, strategies="S2", out=str(tmp_path))
    assert main(args) == 2
