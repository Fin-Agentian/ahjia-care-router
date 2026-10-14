"""Care Router v4: frame rate per camera, GenAI window, companion routing (P:216-241)."""

import pytest

from ahjia.events import ActionRequest, PersonObs
from ahjia.residency import load_spec
from ahjia.router import (
    FrameRatePlan,
    PreloadParams,
    WindowContext,
    camera_load,
    camera_risk,
    degrade_for_budget,
    plan_fps,
    preload_terms,
    route_companion,
)
from ahjia.strategies import get_policy

pytestmark = pytest.mark.unit

SPEC = load_spec()
CAMS = ("living", "hallway", "bathroom_door", "front_door")


def p(zone: str = "living", posture: str = "stand", identity: str = "elder") -> PersonObs:
    return PersonObs(track_id=1, identity=identity, posture=posture, zone=zone)


def views(**kw):
    return {cam: kw.get(cam, ()) for cam in CAMS}


def test_camera_risk_levels():
    assert camera_risk((), night=False) == "empty"
    assert camera_risk((p("living"),), night=False) == "occupied"
    assert camera_risk((p("bathroom_door"),), night=False) == "high_risk"
    assert camera_risk((p("bedside"),), night=False) == "high_risk"
    assert camera_risk((p("hallway"),), night=False) == "occupied"
    assert camera_risk((p("hallway"),), night=True) == "high_risk"


def test_s2_frame_rate_follows_occupancy_and_risk():
    plan = plan_fps(get_policy("S2"), CAMS, views(living=(p("sofa", "sit"),)), night=False,
                    a3_cameras=frozenset(), full_fps=30)
    assert plan == {"living": 10.0, "hallway": 1.0, "bathroom_door": 2.0, "front_door": 1.0}
    night = plan_fps(get_policy("S2"), CAMS, views(hallway=(p("hallway"),)), night=True,
                     a3_cameras=frozenset(), full_fps=30)
    assert night["hallway"] == 15.0 and night["front_door"] == 2.0 and night["living"] == 1.0


def test_a3_puts_the_camera_at_full_rate_in_every_strategy():
    for name in ("S0", "S1", "S2"):
        plan = plan_fps(get_policy(name), CAMS, views(hallway=(p("floor", "lie"),)), night=True,
                        a3_cameras=frozenset({"hallway"}), full_fps=30)
        assert plan["hallway"] == 30.0


def test_s0_is_full_rate_and_s1_is_fixed():
    s0 = plan_fps(get_policy("S0"), CAMS, views(), night=False, a3_cameras=frozenset(), full_fps=30)
    s1 = plan_fps(get_policy("S1"), CAMS, views(living=(p(),)), night=False, a3_cameras=frozenset(), full_fps=30)
    assert set(s0.values()) == {30.0}
    assert set(s1.values()) == {10.0}


def test_degrade_lowers_empty_then_low_risk_cameras_and_never_high_risk():
    plan = {"a": 15.0, "b": 10.0, "c": 2.0}
    risks = {"a": "high_risk", "b": "occupied", "c": "empty"}
    rates = FrameRatePlan()
    first = degrade_for_budget(plan, risks, lambda cam, fps: fps * 0.01, budget=0.265, rates=rates)
    assert first == {"a": 15.0, "b": 10.0, "c": 1.0}
    second = degrade_for_budget(plan, risks, lambda cam, fps: fps * 0.01, budget=0.215, rates=rates)
    assert second == {"a": 15.0, "b": 5.0, "c": 1.0}
    third = degrade_for_budget(plan, risks, lambda cam, fps: fps * 0.01, budget=0.01, rates=rates)
    assert third["a"] == 15.0
    assert degrade_for_budget(plan, risks, lambda cam, fps: 0.0, budget=1.0, rates=rates) == plan


def window(**kw) -> WindowContext:
    base = dict(t=0.0, night=False, elder_visible=True, visitor_present=False,
                someone_in_high_risk=False, alert_active=False, last_directed_t=None)
    base.update(kw)
    return WindowContext(**base)


def test_preload_rule_opens_in_a_quiet_daytime():
    terms = preload_terms(window(), s_l=5.17, params=PreloadParams())
    assert terms.lhs == pytest.approx(terms.p_l * 5.17 * terms.w_l)
    assert terms.rhs == pytest.approx(terms.rho * terms.c_risk)
    assert terms.open


@pytest.mark.parametrize("kw", [
    {"someone_in_high_risk": True}, {"alert_active": True}, {"night": True}, {"elder_visible": False},
])
def test_preload_rule_stays_closed_when_risky_at_night_or_nobody_around(kw):
    assert not preload_terms(window(**kw), s_l=5.17, params=PreloadParams()).open


def test_family_or_recent_talk_raises_the_talk_probability():
    quiet = preload_terms(window(), s_l=5.17, params=PreloadParams())
    family = preload_terms(window(visitor_present=True), s_l=5.17, params=PreloadParams())
    talk = preload_terms(window(t=100.0, last_directed_t=50.0), s_l=5.17, params=PreloadParams())
    assert family.p_l > quiet.p_l and talk.p_l > quiet.p_l


def req(actions=("A2",), reason="elder talks to AhJia") -> ActionRequest:
    return ActionRequest(t=1.0, agent="companion", actions=actions, reason=reason)


def test_s0_answers_every_speech_segment_with_the_llm():
    route = route_companion(get_policy("S0"), req(("A0",), "tv speech · not directed"),
                            night=True, genai="npu_window", in_hours=False, risk_high=False)
    assert route.actions == ("A2",)


def test_not_directed_speech_stays_a0_in_s1_and_s2():
    for name in ("S1", "S2"):
        route = route_companion(get_policy(name), req(("A0",), "tv speech · not directed"),
                                night=False, genai="npu_window", in_hours=True, risk_high=False)
        assert route.actions == ("A0",) and not route.needs_llm


def test_s1_uses_fixed_hours():
    s1 = get_policy("S1")
    assert route_companion(s1, req(), night=False, genai="npu_window", in_hours=True, risk_high=True).actions == ("A2",)
    assert route_companion(s1, req(), night=True, genai="off", in_hours=False, risk_high=False).actions == ("A1",)


def test_s2_night_talk_is_template_voice():
    route = route_companion(get_policy("S2"), req(), night=True, genai="off", in_hours=False, risk_high=False)
    assert route.actions == ("A1",) and "night" in route.note


def test_s2_loads_on_demand_only_when_risk_is_low():
    s2 = get_policy("S2")
    loaded = route_companion(s2, req(), night=False, genai="npu_window", in_hours=True, risk_high=True)
    assert loaded.actions == ("A2",) and not loaded.load_now
    on_demand = route_companion(s2, req(), night=False, genai="off", in_hours=True, risk_high=False)
    assert on_demand.actions == ("A2",) and on_demand.load_now
    risky = route_companion(s2, req(), night=False, genai="off", in_hours=True, risk_high=True)
    assert risky.actions == ("A1",) and not risky.load_now


def test_follow_up_without_an_llm_is_a0_because_the_template_was_already_said():
    follow = req(reason="follow-up after SOS word '暈'")
    route = route_companion(get_policy("S2"), follow, night=False, genai="off", in_hours=True, risk_high=True)
    assert route.actions == ("A0",)
    s1 = route_companion(get_policy("S1"), follow, night=True, genai="off", in_hours=False, risk_high=True)
    assert s1.actions == ("A0",)


def test_camera_load_orders_strategies_by_cost():
    s0 = camera_load(get_policy("S0"), SPEC, "empty", full_fps=30)
    s1 = camera_load(get_policy("S1"), SPEC, "empty", full_fps=30)
    s2 = camera_load(get_policy("S2"), SPEC, "empty", full_fps=30)
    assert s0 > s1 > s2 > 0
    assert camera_load(get_policy("S2"), SPEC, "a3", full_fps=30) > camera_load(get_policy("S2"), SPEC, "high_risk", full_fps=30)


def test_a_person_on_the_hallway_camera_at_night_is_high_risk_whatever_the_zone():
    assert camera_risk((p("floor", "lie"),), night=True, camera="hallway") == "high_risk"
    assert camera_risk((p("floor", "lie"),), night=False, camera="hallway") == "occupied"
