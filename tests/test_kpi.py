"""KPIs (DESIGN.md "KPIs", P:433-443) and the report writer."""

import csv
import json
from pathlib import Path

import pytest

from ahjia.kpi import aggregate, compute_kpis, percentile
from ahjia.replay import SCENARIO_DIR, load_scenario, run_scenario
from ahjia.report import write_report
from ahjia.residency import load_spec

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def night_rows():
    scn = load_scenario(SCENARIO_DIR / "night_fall.yaml")
    return {s: compute_kpis(run_scenario(scn, s)) for s in ("S0", "S1", "S2")}


def test_percentile_helper():
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_night_fall_kpis(night_rows):
    s2 = night_rows["S2"]
    assert s2.misses == 0 and s2.false_alarms == 0 and s2.high_risk_labels == 1
    assert s2.voice_p50 is not None and 0.0 < s2.voice_p50 <= 2.0
    assert s2.text_p50 is not None and s2.text_p50 <= 5.0
    assert s2.a5_immediate_s is not None and s2.a5_immediate_s < 4.0
    assert s2.delayed_safety == 0 and s2.llm_wakes == 0
    assert night_rows["S0"].llm_wakes == 1  # S0 answers "I can't get up" with the LLM too
    assert s2.cameras_supported > night_rows["S1"].cameras_supported >= night_rows["S0"].cameras_supported
    assert s2.avg_power_w < night_rows["S1"].avg_power_w < night_rows["S0"].avg_power_w + 1e-9


def test_a_tv_help_counts_as_a_family_false_alarm():
    scn = load_scenario(SCENARIO_DIR / "scripted" / "tv_help.yaml")
    for strategy in ("S0", "S1", "S2"):
        row = compute_kpis(run_scenario(scn, strategy))
        assert row.false_alarms == 1 and row.misses == 0, strategy


def test_cameras_kpi_is_the_lower_of_the_npu_and_host_bounds(night_rows):
    spec = load_spec()
    for row in night_rows.values():
        assert row.cameras_host == int(spec.value("host.max_streams"))
        assert row.cameras_supported == min(row.cameras_npu, row.cameras_host)
    assert spec.param("host.max_streams").kind == "assumption"


def test_added_cameras_carry_the_average_demand_not_empty_rooms():
    day = load_scenario(SCENARIO_DIR / "day24h.yaml")
    row = compute_kpis(run_scenario(day, "S2"))
    assert 1 <= row.cameras_npu < 21  # 21 was the old "extra cameras are empty rooms" bound


def test_misses_and_false_alarms_against_labels():
    scn = load_scenario(SCENARIO_DIR / "scripted" / "sofa_lie_not_fall.yaml")
    row = compute_kpis(run_scenario(scn, "S2"))
    assert row.misses == 0 and row.false_alarms == 0 and row.high_risk_labels == 0
    dizzy = compute_kpis(run_scenario(load_scenario(SCENARIO_DIR / "dizzy.yaml"), "S2"))
    assert dizzy.a5_no_answer_s is not None and 30.0 <= dizzy.a5_no_answer_s <= 34.0


def test_directed_speech_reply_and_greeting():
    scn = load_scenario(SCENARIO_DIR / "family_arrives.yaml")
    rows = {s: compute_kpis(run_scenario(scn, s)) for s in ("S0", "S1", "S2")}
    assert rows["S2"].directed == 1 and rows["S2"].reply_rate == pytest.approx(1.0)
    assert rows["S2"].greeting_s is not None and rows["S2"].talk_p50 is not None
    assert rows["S0"].llm_wakes > rows["S2"].llm_wakes  # TV and phone talk wake the LLM in S0 only


def test_delayed_safety_says_whether_it_is_computed_or_assumed(night_rows):
    # Review finding 4: with QoS (S2) the wait is the spec's assumption, not a simulation result.
    assert night_rows["S0"].delayed_safety_basis == "computed"
    assert night_rows["S1"].delayed_safety_basis == "computed"
    assert night_rows["S2"].delayed_safety_basis == "assumption: qos_wait_s"


def test_aggregate_pools_samples(night_rows):
    pooled = aggregate([night_rows["S2"], night_rows["S2"]], scenario="scripted set")
    assert pooled.scenario == "scripted set" and pooled.strategy == "S2"
    assert pooled.delayed_safety_basis == "assumption: qos_wait_s"
    assert pooled.high_risk_labels == 2 and pooled.misses == 0
    assert pooled.avg_power_w is None
    with pytest.raises(ValueError):
        aggregate([night_rows["S0"], night_rows["S2"]], scenario="mixed")


def test_report_files(tmp_path: Path, night_rows):
    rows = list(night_rows.values())
    write_report(rows, tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "simulated" in text.lower() and "S2" in text and "night_fall" in text
    assert "A TV voice saying 'help' is treated as real; telling a TV from the elder is Stage II work." in text
    assert "estimate (NPU + host decode assumption)" in text
    data = json.loads((tmp_path / "kpi.json").read_text(encoding="utf-8"))
    assert data["simulated"] is True and data["spec_kinds"]["official"] > 0
    assert {r["strategy"] for r in data["rows"]} == {"S0", "S1", "S2"}
    assert {"scenario", "avg_power_w", "cameras_supported", "false_alarms"} <= set(data["rows"][0])
    with (tmp_path / "kpi.csv").open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert {r["strategy"] for r in records} == {"S0", "S1", "S2"}
    for rows_ in (records, data["rows"]):
        basis = {r["strategy"]: r["delayed_safety_basis"] for r in rows_}
        assert basis == {"S0": "computed", "S1": "computed", "S2": "assumption: qos_wait_s"}
    assert "S2: follows from the safety-first QoS assumption (qos_wait_s = 0); Stage II measures it." in text
    s2_line = next(line for line in text.splitlines() if line.startswith("| night_fall | S2 |"))
    assert "0 †" in s2_line
    s1_line = next(line for line in text.splitlines() if line.startswith("| night_fall | S1 |"))
    assert "†" not in s1_line
    assert (tmp_path / "kpi.png").stat().st_size > 1000
