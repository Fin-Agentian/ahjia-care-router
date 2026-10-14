"""Workbench: replay outputs + scenario files -> one self-contained workbench.html (DESIGN.md "Workbench")."""

import argparse
import json
import re
from pathlib import Path

import pytest

from ahjia.replay import SCENARIO_DIR, load_scenario
from ahjia.workbench import data as wd
from ahjia.workbench import export as wx

pytestmark = pytest.mark.unit

MINI = (
    "name: mini\ntitle: Mini\nsource: s\nstart_clock: '02:00:00'\ncameras: [hallway, living]\nduration_s: 30\n"
    "labels: [{t: 5, kind: high_risk, what: fall}]\n"
    "events:\n"
    "  - {t: 0, camera: hallway, persons: [{track_id: 1, identity: elder, posture: stand, zone: hallway}]}\n"
    "  - {t: 5, camera: hallway, persons: [{track_id: 1, identity: elder, posture: lie, zone: floor}]}\n"
    "  - {t: 9, speech: {text: \"I can't get up\", source: elder}}\n"
    "  - {t: 20, app_confirm: true}\n"
)


@pytest.fixture()
def mini(tmp_path: Path):
    path = tmp_path / "mini.yaml"
    path.write_text(MINI, encoding="utf-8")
    return load_scenario(path)


# ---------------------------------------------------------------- world state

def test_world_json_keeps_changes_per_camera_and_speech(mini):
    world = wd.world_json(mini)
    assert [c[0] for c in world["cams"]["hallway"]] == [0.0, 5.0]
    assert world["cams"]["living"] == []
    assert world["cams"]["hallway"][1][1] == [{"id": 1, "who": "elder", "pose": "lie", "zone": "floor"}]
    assert world["speech"] == [{"t": 9.0, "text": "I can't get up", "source": "elder", "directed": False,
                                "lang": "en"}]
    assert world["app_confirm"] == [20.0]


@pytest.mark.parametrize("t, pose, speaking, confirm", [
    (0.0, "stand", False, False),
    (4.99, "stand", False, False),
    (5.0, "lie", False, False),
    (9.5, "lie", True, False),
    (14.0, "lie", False, False),
    (21.0, "lie", False, True),
])
def test_world_state_at(mini, t, pose, speaking, confirm):
    state = wd.world_state_at(wd.world_json(mini), t)
    assert state["cams"]["hallway"][0]["pose"] == pose
    assert state["cams"]["living"] == []
    assert bool(state["speech"]) is speaking
    assert state["app_confirm"] is confirm


# ---------------------------------------------------------------- branches

def test_branches_map_to_scripted_files():
    branches = wd.build_branches({p.stem for p in (SCENARIO_DIR / "scripted").glob("*.yaml")})
    mapping = {b["label"]: b["scenario"] for b in branches}
    assert mapping == {
        "…she says I'm OK and stands up": "ok_and_upright",
        "…she says I'm OK but stays down": "ok_but_lying",
        "…family confirms in the app": "app_confirm",
        "…the TV shouts help": "tv_help",
        "…nobody answers (30 s rule)": "dizzy_en",
    }


def test_branch_with_missing_file_is_dropped():
    branches = wd.build_branches({"tv_help", "dizzy_en"})
    assert [b["scenario"] for b in branches] == ["tv_help", "dizzy_en"]


# ---------------------------------------------------------------- KPIs and responses

CSV = ("scenario,group,strategy,duration_s,avg_power_w,cameras_supported,voice_p50,misses,false_alarms\n"
       "day24h,day,S0,86400.0000,2.4890,1,0.7571,0,1\n"
       "day24h,day,S2,86400.0000,0.7134,8,,0,1\n")


def test_kpis_fall_back_to_csv(tmp_path: Path):
    (tmp_path / "kpi.csv").write_text(CSV, encoding="utf-8")
    spec = tmp_path / "spec.yaml"
    spec.write_text("a: {value: 1, kind: official, source: x}\nb: {c: {value: 2, kind: assumption, source: y}}\n",
                    encoding="utf-8")
    kpis = wd.read_kpis(tmp_path, spec_path=spec)
    assert kpis["source"] == "kpi.csv" and kpis["simulated"] is True
    assert kpis["spec_kinds"] == {"official": 1, "estimate": 0, "assumption": 1}
    s0, s2 = kpis["rows"]
    assert s0["avg_power_w"] == pytest.approx(2.489) and s0["cameras_supported"] == 1
    assert s2["voice_p50"] is None and s2["scenario"] == "day24h" and s2["strategy"] == "S2"


def test_kpis_prefer_json(tmp_path: Path):
    (tmp_path / "kpi.csv").write_text(CSV, encoding="utf-8")
    payload = {"simulated": True, "spec_kinds": {"official": 9, "estimate": 4, "assumption": 22},
               "rows": [{"scenario": "x", "group": "demo", "strategy": "S1", "avg_power_w": 1.0}]}
    (tmp_path / "kpi.json").write_text(json.dumps(payload), encoding="utf-8")
    kpis = wd.read_kpis(tmp_path)
    assert kpis["source"] == "kpi.json" and kpis["spec_kinds"]["assumption"] == 22
    assert kpis["rows"] == [{**payload["rows"][0], "delayed_safety_basis": "computed"}]


def test_kpis_missing_files(tmp_path: Path):
    kpis = wd.read_kpis(tmp_path, spec_path=tmp_path / "none.yaml")
    assert kpis["rows"] == [] and kpis["source"] is None
    assert kpis["spec_kinds"] == {"official": 0, "estimate": 0, "assumption": 0}


def test_scripted_summary_pools_counts():
    rows = [{"group": "scripted", "strategy": s, "misses": m, "high_risk_labels": 2, "delayed_safety": d,
             "false_alarms": f} for s, m, d, f in (("S0", 0, 1, 1), ("S0", 0, 0, 0), ("S2", 0, 0, 1))]
    rows.append({"group": "day", "strategy": "S0", "misses": 9, "high_risk_labels": 9, "delayed_safety": 9,
                 "false_alarms": 9})
    out = wd.scripted_summary(rows)
    assert out["S0"] == {"cases": 2, "misses": 0, "high_risk_labels": 4, "delayed_safety": 1, "false_alarms": 1,
                         "delayed_safety_assumed": False}
    assert out["S2"]["cases"] == 1 and "S1" not in out
    assert out["S2"]["delayed_safety_assumed"] is True  # no basis field: S2 falls back to the assumption


# ---------------------------------------------------------------- honesty label: S2 delayed = 0 is an assumption

def test_delayed_safety_basis_uses_the_field_when_present():
    assert wd.delayed_safety_basis({"strategy": "S2", "delayed_safety_basis": "computed"}) == "computed"
    assert wd.delayed_safety_basis({"strategy": "S0", "delayed_safety_basis": "assumption: qos_wait_s"}) \
        == "assumption: qos_wait_s"


@pytest.mark.parametrize("strategy, expected", [("S0", "computed"), ("S1", "computed"),
                                                ("S2", "assumption: qos_wait_s")])
def test_delayed_safety_basis_falls_back_to_the_strategy(strategy, expected):
    assert wd.delayed_safety_basis({"strategy": strategy}) == expected


def test_read_kpis_adds_basis_to_every_row_and_keeps_given_ones(tmp_path: Path):
    payload = {"rows": [{"scenario": "x", "group": "demo", "strategy": "S2"},
                        {"scenario": "x", "group": "demo", "strategy": "S1"},
                        {"scenario": "y", "group": "demo", "strategy": "S2", "delayed_safety_basis": "computed"}]}
    (tmp_path / "kpi.json").write_text(json.dumps(payload), encoding="utf-8")
    bases = [r["delayed_safety_basis"] for r in wd.read_kpis(tmp_path, spec_path=tmp_path / "n.yaml")["rows"]]
    assert bases == ["assumption: qos_wait_s", "computed", "computed"]


def test_scripted_summary_is_assumed_only_when_a_pooled_row_is():
    rows = [{"group": "scripted", "strategy": "S2", "delayed_safety_basis": "computed"},
            {"group": "scripted", "strategy": "S1", "delayed_safety_basis": "computed"},
            {"group": "scripted", "strategy": "S1", "delayed_safety_basis": "assumption: qos_wait_s"}]
    out = wd.scripted_summary(rows)
    assert out["S2"]["delayed_safety_assumed"] is False and out["S1"]["delayed_safety_assumed"] is True


def test_read_jsonl(tmp_path: Path):
    path = tmp_path / "S2.responses.jsonl"
    path.write_text('{"t_event": 1, "action": "A1", "t_done": 1.5, "latency_s": 0.5}\n\n', encoding="utf-8")
    assert wd.read_jsonl(path) == [{"t_event": 1, "action": "A1", "t_done": 1.5, "latency_s": 0.5}]
    assert wd.read_jsonl(tmp_path / "missing.jsonl") is None


# ---------------------------------------------------------------- full build

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("wb")
    code = wx.main(argparse.Namespace(out=str(out)))
    html = (out / "workbench.html").read_text(encoding="utf-8")
    return code, out, html


def _data(html: str) -> dict:
    match = re.search(r'<script type="application/json" id="ahjia-data">(.*?)</script>', html, re.S)
    assert match
    return json.loads(match.group(1))


def test_main_writes_workbench_html(built):
    code, out, html = built
    assert code == 0
    size = (out / "workbench.html").stat().st_size
    assert 0 < size < 3_000_000


def test_data_structure(built):
    data = _data(built[2])
    names = {s["name"] for s in data["scenarios"]}
    assert {"family_arrives", "dizzy", "night_fall", "day24h"} <= names
    assert [s["name"] for s in data["scenarios"] if s["main"]] == ["family_arrives", "dizzy", "night_fall", "day24h"]
    assert [s["name"] for s in data["strategies"]] == ["S0", "S1", "S2"]
    assert len(data["branches"]) == 5 and all(b["scenario"] in names for b in data["branches"])
    nf = next(s for s in data["scenarios"] if s["name"] == "night_fall")
    assert nf["friendly"] == "Night fall" and nf["start_clock"] == "02:10:00"
    run = nf["runs"]["S2"]
    fall = next(d for d in run["decisions"] if d["agent"] == "fall")
    assert fall["actions"] == ["A1", "A4", "A3"] and fall["alert"] == "pre_alert"
    assert any(r["action"] == "A5" for r in run["responses"])
    assert run["timeline"][0]["t"] == 0.0 and run["timeline"][0]["fps"]["hallway"] > run["timeline"][0]["fps"]["living"]
    assert len(run["timeline"][0]["hud"]) == 7
    assert data["kpis"]["rows"] and data["kpis"]["simulated"] is True
    assert set(data["kpis"]["scripted"]) == {"S0", "S1", "S2"}
    assert data["meta"]["rules"]["escalate_no_answer_s"] == 30.0


def test_html_is_self_contained(built):
    html = built[2]
    assert not re.search(r'<script[^>]+src=', html, re.I)
    assert not re.search(r'<link[^>]+href=', html, re.I)
    assert not re.search(r'url\(\s*["\']?https?:', html, re.I)
    assert not re.search(r'(src|href)\s*=\s*["\']https?:', html, re.I)
    assert "window.ahjiaSeek" in html
    assert "Hailo-10H timing and power simulated" in html


def test_render_html_escapes_script_end(tmp_path: Path):
    html = wx.render_html({"x": "</script><b>"})
    assert "</script><b>" not in html
    assert _data(html) == {"x": "</script><b>"}


def test_main_fails_cleanly_when_replay_breaks(tmp_path: Path, monkeypatch):
    def boom(_out):
        raise RuntimeError("no replay")
    monkeypatch.setattr(wx, "run_replay", boom)
    assert wx.main(argparse.Namespace(out=str(tmp_path))) == 1
    assert not (tmp_path / "workbench.html").exists()


def test_public_repo_link_only(built):
    html = built[2]
    assert "github.com/Fin-Agentian/ahjia-care-router" in html
    assert "ASUS_UGen_Hackathon" not in html


def test_clear_rule_says_standing(built):
    data = _data(built[2])
    rule = next(b["rule"] for b in data["branches"] if b["scenario"] == "ok_and_upright")
    assert "stands 5 s" in rule
    assert "stays upright" not in built[2]


def test_future_chips_mode_is_supported(built):
    html = built[2]
    assert 'Q.get("future")' in html and '"hide"' in html and '"dim"' in html


# ---------------------------------------------------------------- honesty label on the page

FOOTNOTE = "S2 follows from the safety-first QoS assumption (qos_wait_s = 0); Stage II measures it."


def test_page_marks_s2_delayed_count_and_shows_the_footnote(built):
    html = built[2]
    assert "* " + FOOTNOTE in html
    assert 'id="kpi-card"' in html and 'id="scripted-card"' in html
    assert 'ASSUMED_MARK = " *"' in html  # appended to the S2 value in the KPI table, scripted table and bar label
    rows = _data(html)["kpis"]["rows"]
    assert rows and all(r["delayed_safety_basis"] for r in rows)
    assert {r["delayed_safety_basis"] for r in rows if r["strategy"] == "S2"} <= {"computed", "assumption: qos_wait_s"}
    assert all(r["delayed_safety_basis"] == "computed" for r in rows if r["strategy"] != "S2")
    scripted = _data(html)["kpis"]["scripted"]
    assert scripted["S0"]["delayed_safety_assumed"] is False


# ---------------------------------------------------------------- one source for the clear time (finding #5)

def test_rules_come_from_the_guardrail_constants(monkeypatch):
    from ahjia import guardrails

    monkeypatch.setattr(guardrails, "STANDING_CLEAR_S", 7.5)
    monkeypatch.setattr(guardrails, "ESCALATE_NO_ANSWER_S", 41.0)
    monkeypatch.setattr(guardrails, "CHECKIN_NO_ANSWER_S", 83.0)
    rules = wx._rules()
    assert rules["standing_clear_s"] == 7.5
    assert rules["escalate_no_answer_s"] == 41.0 and rules["checkin_no_answer_s"] == 83.0


def test_default_clear_time_is_the_real_constant(built):
    from ahjia import guardrails

    assert _data(built[2])["meta"]["rules"]["standing_clear_s"] == guardrails.STANDING_CLEAR_S
    assert "UPRIGHT_CLEAR_S" not in built[2] and "upright_clear_s" not in built[2]


def test_changing_the_constant_changes_exported_value_and_rule_text(monkeypatch, tmp_path: Path):
    from ahjia import guardrails

    monkeypatch.setattr(guardrails, "STANDING_CLEAR_S", 7.5)
    monkeypatch.setattr(wx, "_scenario_entry",
                        lambda path, replay_out, main_names, branch_labels: {"name": path.stem, "main": False})
    data = wx.build_data(tmp_path)
    assert data["meta"]["rules"]["standing_clear_s"] == 7.5
    rule = next(b["rule"] for b in data["branches"] if b["scenario"] == "ok_and_upright")
    assert "stands 7.5 s" in rule


def test_page_has_no_typed_clear_time():
    template = (wx.HERE / "template.html").read_text(encoding="utf-8")
    assert 'id="rule-5">5<' not in template
    assert "META.rules.standing_clear_s" in (wx.HERE / "app.js").read_text(encoding="utf-8")
