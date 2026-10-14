"""Hailo-10H residency, time-sharing and power simulator (ahjia/residency.py)."""

from pathlib import Path

import pytest

from ahjia.residency import (
    KINDS,
    Demand,
    Hailo10HSim,
    NpuMeter,
    allocate,
    cameras_supported,
    frame_cost_s,
    load_spec,
    md1_wait_s,
    power_w,
    whisper_tokens,
)

pytestmark = pytest.mark.unit

SPEC = load_spec()


def test_every_spec_value_has_a_kind_and_a_source():
    assert SPEC.params
    for key, param in SPEC.params.items():
        assert param.kind in KINDS, key
        assert param.source.strip(), key


def test_spec_holds_the_official_anchors_from_the_proposal():
    assert SPEC.value("models.llm.load_s") == pytest.approx(5.17)
    assert SPEC.param("models.llm.load_s").kind == "official"
    assert SPEC.value("power_w.llm_decode") == pytest.approx(2.1)
    assert SPEC.param("power_w.llm_decode").kind == "official"
    assert SPEC.value("power_w.vision") == pytest.approx(2.5)
    assert SPEC.param("models.pose.fps").kind == "estimate"
    assert SPEC.name("models.llm") == "Qwen2.5-1.5B-Instruct"


def test_spec_without_kind_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("power_w:\n  idle:\n    value: 0.5\n    source: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        load_spec(bad)
    worse = tmp_path / "worse.yaml"
    worse.write_text("power_w:\n  idle:\n    value: 0.5\n    kind: guess\n    source: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_spec(worse)


def test_relative_spec_path_resolves_from_the_package():
    assert load_spec("specs/hailo10h.yaml").value("models.llm.tps") == pytest.approx(7.35)


def test_frame_cost_per_profile():
    det, pose = 1 / 181, 1 / 100
    face = 1 / 520 + 1 / 5191
    assert frame_cost_s(SPEC, "cascade", 0) == pytest.approx(det)
    assert frame_cost_s(SPEC, "cascade", 2) == pytest.approx(det + 2 * 0.25 * pose)
    assert frame_cost_s(SPEC, "cascade", 1, full_res_pose=True) == pytest.approx(det + pose)
    assert frame_cost_s(SPEC, "cascade", 1, face_needed=True) == pytest.approx(det + 0.25 * pose + face)
    assert frame_cost_s(SPEC, "fixed", 0) == pytest.approx(pose)
    assert frame_cost_s(SPEC, "fixed", 1) == pytest.approx(pose + face)
    assert frame_cost_s(SPEC, "all_models", 0) == pytest.approx(det + pose + face)


def test_qos_serves_vision_before_the_llm():
    shares = allocate(Demand(vision=0.9, llm=1.0), qos=True, spec=SPEC)
    assert shares.vision == pytest.approx(0.9) and shares.vision_service == pytest.approx(1.0)
    assert shares.llm == pytest.approx(0.1)


def test_without_qos_everything_shares_in_proportion():
    shares = allocate(Demand(vision=1.0, llm=1.0), qos=False, spec=SPEC)
    assert shares.vision == pytest.approx(0.5) and shares.vision_service == pytest.approx(0.5)
    assert shares.llm == pytest.approx(0.5)


def test_without_qos_an_llm_load_blocks_the_npu():
    shares = allocate(Demand(vision=0.2, loading=True), qos=False, spec=SPEC)
    assert shares.load == pytest.approx(1.0) and shares.vision == 0.0
    with_qos = allocate(Demand(vision=0.2, loading=True), qos=True, spec=SPEC)
    assert with_qos.vision == pytest.approx(0.2) and with_qos.load == pytest.approx(0.8)


def test_power_is_time_share_times_state_power():
    assert power_w(allocate(Demand(), qos=True, spec=SPEC), SPEC) == pytest.approx(0.5)
    assert power_w(allocate(Demand(vision=1.0), qos=True, spec=SPEC), SPEC) == pytest.approx(2.5)
    mixed = power_w(allocate(Demand(vision=0.2, llm=1.0), qos=True, spec=SPEC), SPEC)
    assert mixed == pytest.approx(0.2 * 2.5 + 0.8 * 2.1)


def test_meter_integrates_piecewise_constant_demand():
    meter = NpuMeter(SPEC, qos=True)
    meter.set_demand(0.0, Demand(vision=1.0))
    meter.set_demand(10.0, Demand())
    totals = meter.totals(20.0)
    assert totals.duration_s == pytest.approx(20.0)
    assert totals.energy_j == pytest.approx(10 * 2.5 + 10 * 0.5)
    assert totals.avg_power_w == pytest.approx(1.5)
    assert totals.state_s["vision"] == pytest.approx(10.0)
    assert totals.peak_vision_demand == pytest.approx(1.0)


def test_md1_wait_grows_with_load():
    assert md1_wait_s(0.01, 0.0) == 0.0
    assert md1_wait_s(0.01, 0.8) == pytest.approx(0.02)
    assert md1_wait_s(0.01, 5.0) == pytest.approx(md1_wait_s(0.01, 0.95))


def test_whisper_tokens_for_english_and_chinese():
    assert whisper_tokens(SPEC, "I can't get up") == pytest.approx(4 * 1.3 + 4)
    assert whisper_tokens(SPEC, "我有點暈") == pytest.approx(4 + 4)


def test_llm_load_takes_the_official_load_time():
    sim = Hailo10HSim(SPEC, qos=True)
    end = sim.start_load(10.0)
    assert end == pytest.approx(15.17)
    assert sim.genai_state(12.0) == "loading"
    sim.advance(16.0)
    assert sim.genai_state(16.0) == "npu_window" and sim.llm_resident
    sim.unload(20.0)
    assert sim.genai_state(20.0) == "off"


def test_warm_load_is_instant():
    sim = Hailo10HSim(SPEC, qos=False)
    assert sim.start_load(0.0, warm=True) == 0.0
    assert sim.llm_resident


def test_llm_job_timing_and_queue():
    sim = Hailo10HSim(SPEC, qos=True)
    sim.start_load(0.0, warm=True)
    job = sim.start_llm_job(1.0)
    assert job.start == pytest.approx(1.0)
    assert job.first_audio == pytest.approx(1.0 + 0.37 + 10 / 7.35 + 0.5)
    assert job.end == pytest.approx(1.0 + 0.37 + 40 / 7.35)
    second = sim.start_llm_job(2.0)
    assert second.start == pytest.approx(job.end)
    assert sim.llm_jobs == 2


def test_llm_job_waits_for_a_running_load():
    sim = Hailo10HSim(SPEC, qos=True)
    sim.start_load(0.0)
    job = sim.start_llm_job(1.0)
    assert job.start == pytest.approx(5.17)


def test_llm_job_without_a_model_is_an_error():
    with pytest.raises(RuntimeError):
        Hailo10HSim(SPEC, qos=True).start_llm_job(0.0)


def test_genai_wait_without_qos_and_with_qos():
    no_qos = Hailo10HSim(SPEC, qos=False)
    no_qos.start_load(0.0)
    assert no_qos.genai_wait(2.0) == pytest.approx(3.17)
    no_qos.advance(6.0)
    no_qos.start_llm_job(6.0)
    no_qos.advance(6.5)
    assert no_qos.genai_wait(6.5) == pytest.approx(1 / 7.35)
    assert no_qos.genai_wait(6.5, whisper_tokens=10) == pytest.approx(10 / 7.35)
    qos = Hailo10HSim(SPEC, qos=True)
    qos.start_load(0.0)
    assert qos.genai_wait(2.0) == 0.0


def test_vision_service_drops_when_overloaded_without_qos():
    sim = Hailo10HSim(SPEC, qos=False)
    sim.set_vision_demand(0.0, 2.0)
    assert sim.vision_service() == pytest.approx(0.5)
    totals = sim.totals(10.0)
    assert totals.avg_power_w == pytest.approx(2.5)
    assert totals.overload_s == pytest.approx(10.0)


def test_next_event_reports_load_and_decode_ends():
    sim = Hailo10HSim(SPEC, qos=True)
    sim.start_load(0.0)
    assert sim.next_event(0.0) == pytest.approx(5.17)
    sim.advance(5.17)
    job = sim.start_llm_job(6.0)
    assert sim.next_event(5.17) == pytest.approx(job.start)
    sim.advance(job.start)
    assert sim.next_event(job.start) == pytest.approx(job.end)


def test_whisper_job_latency_and_busy_time():
    sim = Hailo10HSim(SPEC, qos=True)
    latency = sim.start_whisper(1.0, "I can't get up")
    assert latency == pytest.approx(0.3 + (4 * 1.3 + 4) / 25.32)
    assert sim.genai_state(1.0) == "off"


def test_cameras_supported_counts_cameras_until_the_budget_is_used():
    assert cameras_supported(0.8, a3_load=0.3, busy_load=0.1, extra_load=0.05, busy_cameras=2) == 10
    assert cameras_supported(0.8, a3_load=0.53, busy_load=0.53, extra_load=0.53, busy_cameras=1) == 1
    assert cameras_supported(0.1, a3_load=0.5, busy_load=0.1, extra_load=0.1, busy_cameras=1) == 0
    assert cameras_supported(0.8, a3_load=0.1, busy_load=0.1, extra_load=0.0, busy_cameras=1) == 64


def test_meter_integrates_vision_demand_even_when_saturated():
    meter = NpuMeter(SPEC, qos=False)
    meter.set_demand(0.0, Demand(vision=2.0))
    totals = meter.totals(10.0)
    assert totals.vision_demand_s == pytest.approx(20.0)
    assert totals.state_s["vision"] == pytest.approx(10.0)


def test_spec_has_a_host_decode_cap():
    assert SPEC.value("host.max_streams") >= 1
    assert SPEC.param("host.max_streams").kind == "assumption"
