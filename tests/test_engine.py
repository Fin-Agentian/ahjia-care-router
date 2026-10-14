"""Engine: perception -> agents -> Care Router + guardrails -> decisions and HUD."""

import pytest

from ahjia.engine import CareEngine, make_engine
from ahjia.events import Engine, Observation, PersonObs, RunConfig, SpeechObs
from ahjia.residency import load_spec

pytestmark = pytest.mark.unit

CAMS = ("living", "hallway", "bathroom_door", "front_door")


def config(clock: str = "02:10:00", **kw) -> RunConfig:
    return RunConfig(start_clock=clock, cameras=CAMS, **kw)


def elder(posture: str = "stand", zone: str = "hallway", track: int = 1, identity: str = "elder") -> PersonObs:
    return PersonObs(track_id=track, identity=identity, posture=posture, zone=zone)


def frame(t: float, camera: str, *persons: PersonObs) -> Observation:
    return Observation(t=t, camera=camera, persons=tuple(persons))


def said(t: float, text: str, source: str = "elder", directed: bool = False) -> Observation:
    return Observation(t=t, speech=SpeechObs(text=text, source=source, directed=directed))


def night_fall(engine: Engine) -> list:
    out = []
    out += engine.observe(frame(0.0, "hallway", elder()))
    out += engine.observe(frame(12.0, "hallway", elder("lie", "floor")))
    out += engine.observe(frame(14.0, "hallway", elder("lie", "floor")))
    return out


def test_make_engine_implements_the_protocol():
    engine = make_engine("S2", config())
    assert isinstance(engine, CareEngine) and engine.strategy == "S2"
    for method in ("observe", "advance", "hud", "camera_fps"):
        assert callable(getattr(engine, method))
    with pytest.raises(ValueError):
        make_engine("S5", config())


def test_night_fall_scenario_matches_the_proposal_hud():
    engine = make_engine("S2", config())
    engine.observe(frame(0.0, "hallway", elder()))
    assert engine.camera_fps("hallway") == 15.0  # night hallway with a person (P:218)
    assert engine.camera_fps("living") == 1.0
    decisions = night_fall(engine)
    fall = [d for d in decisions if d.agent == "fall"][0]
    assert fall.actions == ("A1", "A4", "A3") and fall.alert == "pre_alert" and fall.genai == "off"
    assert fall.clock == "02:10:14" and fall.simulated
    assert 0.0 < fall.latency_s <= 2.0
    assert dict(fall.fps)["hallway"] == 30.0
    lines = engine.hud().lines()
    assert len(lines) == 7
    text = "\n".join(lines)
    for expected in ("fall", "A1+A4+A3", "lying outside bed/sofa", "s (sim)", "4 on", "hallway 30 fps",
                     "off (night)", "pre-alert"):
        assert expected in text
    sos = engine.observe(said(19.0, "I can't get up"))
    a5 = [d for d in sos if "A5" in d.actions][0]
    assert a5.agent == "voice_sos" and a5.alert == "escalated"
    assert "escalated" in engine.hud().alert


def test_engine_is_deterministic():
    runs = []
    for _ in range(2):
        engine = make_engine("S1", config())
        runs.append(night_fall(engine) + list(engine.advance(60.0)))
    assert runs[0] == runs[1]


def test_time_cannot_go_backwards():
    engine = make_engine("S2", config())
    engine.advance(10.0)
    with pytest.raises(ValueError):
        engine.advance(5.0)


def test_30_s_rule_fires_from_advance():
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    first = engine.observe(said(5.0, "我有點暈"))
    assert [d.actions for d in first if d.agent == "voice_sos"] == [("A1", "A4")]
    later = engine.advance(60.0)
    a5 = [d for d in later if d.actions == ("A5",)]
    assert a5 and a5[0].t == pytest.approx(35.0) and a5[0].alert == "escalated"


def test_dizzy_in_the_daytime_gets_an_llm_follow_up_in_s2():
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    assert engine.hud().genai == "LLM loaded (day)"  # warm start: the window was already open
    out = engine.observe(said(5.0, "我有點暈"))
    follow = [d for d in out if d.agent == "companion"]
    assert follow and follow[0].actions == ("A2",) and follow[0].genai == "npu_window"


def test_tv_speech_wakes_the_llm_only_in_s0():
    for name, expected in (("S0", ("A2",)), ("S1", ("A0",)), ("S2", ("A0",))):
        engine = make_engine(name, config("14:05:00"))
        engine.observe(frame(0.0, "living", elder("sit", "sofa")))
        out = engine.observe(said(3.0, "And now the weather.", source="tv"))
        assert [d.actions for d in out] == [expected], name
        assert engine.hud().alert == "none"


def test_family_greeting_latency_in_the_daytime_window():
    engine = make_engine("S2", config("14:05:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    engine.observe(frame(5.0, "living", elder("sit", "sofa"), elder("stand", "living", 2, "family")))
    out = engine.observe(frame(6.5, "living", elder("sit", "sofa"), elder("stand", "living", 2, "family")))
    greet = [d for d in out if d.agent == "companion"][0]
    assert greet.actions == ("A2",) and greet.reason.startswith("family")
    assert 3.0 <= greet.latency_s <= 4.5  # measured from entry; P:490 estimates 3.8 s
    hud = engine.hud()
    assert hud.agent == "companion" and hud.decision == "A2 wake LLM" and "living 10 fps" in hud.cameras


def test_night_talk_is_template_voice_in_s2():
    engine = make_engine("S2", config("23:30:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    out = engine.observe(said(2.0, "AhJia, what time is it?", directed=True))
    assert out[0].actions == ("A1",) and out[0].genai == "off"
    assert out[0].latency_s < 2.0


def test_s2_loads_the_llm_on_demand_when_the_window_is_closed():
    engine = make_engine("S2", config("15:00:00"))
    out = engine.observe(said(2.0, "AhJia, are you there?", directed=True))  # nobody in view: window closed
    assert out[0].actions == ("A2",) and out[0].genai == "loading"
    assert out[0].latency_s > 5.0  # paid the 5.17 s official load time


def test_app_confirm_clears_and_family_or_tv_cannot():
    engine = make_engine("S2", config())
    night_fall(engine)
    engine.observe(said(15.0, "She is OK, I'm OK", source="family"))
    engine.observe(said(16.0, "I'm OK", source="tv"))
    assert engine.hud().alert == "pre-alert"
    out = engine.observe(Observation(t=20.0, app_confirm=True))
    assert out[-1].alert == "cleared_by_app" and engine.hud().alert == "cleared in app"
    assert engine.camera_fps("hallway") == 15.0  # A3 ends with the alert


def test_im_ok_and_upright_clears_by_the_elder():
    engine = make_engine("S2", config())
    night_fall(engine)
    engine.observe(said(16.0, "I'm OK", directed=True))
    engine.observe(frame(18.0, "hallway", elder("stand", "hallway")))
    assert engine.next_deadline() == (pytest.approx(19.0), "hallway")  # keep-alive frame before 2 s stale
    out = engine.observe(frame(23.0, "hallway", elder("stand", "hallway")))
    cleared = [d for d in out if d.alert == "cleared_by_elder"]
    assert cleared and "cleared by the elder" in cleared[0].reason


def test_time_scale_shortens_the_bathroom_rule():
    engine = make_engine("S2", config("10:00:00", time_scale=0.01))
    engine.observe(frame(0.0, "bathroom_door", elder("stand", "bathroom_door")))
    engine.observe(frame(2.0, "bathroom_door"))
    out = engine.advance(30.0)
    checkin = [d for d in out if d.agent == "inactivity"]
    assert checkin and checkin[0].actions == ("A1",) and checkin[0].t == pytest.approx(20.0)
    later = engine.advance(120.0)  # A4 at 80 s (60 s check-in), A5 at 110 s (30 s rule)
    assert [d.actions for d in later if d.agent == "inactivity"][:2] == [("A4",), ("A5",)]


def test_s0_overloads_the_npu_and_drops_frames():
    engine = make_engine("S0", config("14:00:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    assert engine.camera_fps("living") == 30.0
    assert engine.processed_fps("living") < 30.0
    s2 = make_engine("S2", config("14:00:00"))
    s2.observe(frame(0.0, "living", elder("sit", "sofa")))
    assert s2.processed_fps("living") == pytest.approx(10.0)


def test_s1_safety_wait_is_computed_and_s2_wait_is_the_qos_assumption():
    """S1 (no QoS): the wait is computed from LLM decoding. S2 (safety-first QoS): the wait is not
    computed; it IS the spec's assumption contention.qos_wait_s. Stage II measures it."""
    spec = load_spec()
    assert spec.param("contention.qos_wait_s").kind == "assumption"
    delays = {}
    for name in ("S1", "S2"):
        engine = make_engine(name, config("15:00:00"))
        engine.observe(frame(0.0, "living", elder("stand", "living")))
        engine.observe(said(1.0, "AhJia, tell me a story", directed=True))
        engine.observe(frame(3.0, "living", elder("lie", "floor")))
        engine.observe(frame(5.0, "living", elder("lie", "floor")))
        waits = [r.genai_wait_s for r in engine.responses if r.safety]
        delays[name] = max(waits)
    assert delays["S1"] > 0.0
    assert delays["S2"] == spec.value("contention.qos_wait_s")


def test_responses_and_totals_are_exposed_for_kpis():
    engine = make_engine("S2", config())
    night_fall(engine)
    engine.advance(60.0)
    actions = {r.action for r in engine.responses}
    assert {"A1", "A4"} <= actions
    totals = engine.totals(60.0)
    assert totals.duration_s == pytest.approx(60.0) and 0.4 < totals.avg_power_w < 2.6


def test_visitor_logs_are_decisions_without_latency():
    engine = make_engine("S2", config("10:00:00"))
    engine.observe(frame(0.0, "front_door", elder("stand", "front_door", 5, "caregiver")))
    out = engine.observe(frame(1.5, "front_door", elder("stand", "front_door", 5, "caregiver")))
    log = [d for d in out if d.agent == "visitor"][0]
    assert log.actions == ("A0",) and log.latency_s is None


def test_a_tv_help_is_treated_as_real_in_every_strategy():
    for name in ("S0", "S1", "S2"):
        engine = make_engine(name, config("20:00:00"))
        engine.observe(frame(0.0, "living", elder("sit", "sofa")))
        out = engine.observe(said(5.0, "Help! Somebody help!", source="tv"))
        sos = [d for d in out if d.agent == "voice_sos"]
        assert sos and sos[0].actions == ("A1", "A4", "A5") and sos[0].alert == "escalated", name


def test_sitting_up_does_not_clear_in_the_engine():
    engine = make_engine("S2", config())
    night_fall(engine)  # fall decision at 14 s
    engine.observe(said(16.0, "I'm OK", directed=True))
    engine.observe(frame(18.0, "hallway", elder("sit", "floor")))
    assert engine.next_deadline()[1] is None  # no clear pending: sitting is not standing
    out = engine.advance(60.0)
    assert not any("A5" in d.actions for d in out)  # she answered at 16 s
    assert engine.hud().alert == "pre-alert"


def _reasons(decisions) -> list[str]:
    return [d.reason for d in decisions]


def test_a_second_sos_word_re_alerts_and_still_escalates_after_30_s():
    # Review finding 1: "dizzy" then "chest pain" stopped the 30 s rule and never reached A5.
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    engine.observe(said(5.0, "I feel dizzy"))
    second = engine.observe(said(20.0, "my chest pain is bad"))
    assert not any("answered" in r or "already" in r for r in _reasons(second))
    sos = [d for d in second if d.agent == "voice_sos"]
    assert sos and sos[0].actions == ("A1", "A4") and "chest pain" in sos[0].reason
    a5 = [d for d in engine.advance(60.0) if d.actions == ("A5",)]
    assert a5 and a5[0].t == pytest.approx(35.0) and a5[0].alert == "escalated"


def test_a_fall_after_an_answered_sos_alerts_again_and_escalates():
    # Review finding 2: after an answer, a later fall gave only A0.
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("stand", "living")))
    engine.observe(said(5.0, "I feel dizzy"))
    engine.observe(said(10.0, "a little, I will sit down"))
    engine.observe(frame(70.0, "living", elder("lie", "floor")))
    out = engine.observe(frame(72.0, "living", elder("lie", "floor")))
    fall = [d for d in out if d.agent == "fall"]
    assert fall and fall[0].actions == ("A1", "A4", "A3")
    a5 = [d for d in engine.advance(120.0) if d.actions == ("A5",)]
    assert a5 and a5[0].t == pytest.approx(102.0)


def test_stranger_at_night_is_a4_only_and_keeps_the_frame_rate():
    # Review finding 7: an A4-only notice must not raise the front door to full rate.
    engine = make_engine("S2", config())
    engine.observe(frame(0.0, "front_door", elder("stand", "front_door", 7, "stranger")))
    before = engine.camera_fps("front_door")
    out = engine.observe(frame(1.5, "front_door", elder("stand", "front_door", 7, "stranger")))
    notice = [d for d in out if d.agent == "visitor"]
    assert notice and notice[0].actions == ("A4",)
    assert engine.camera_fps("front_door") == before < 30.0


def test_a_fall_after_a_stranger_notice_alerts_again():
    engine = make_engine("S2", config())
    engine.observe(frame(0.0, "front_door", elder("stand", "front_door", 7, "stranger")))
    engine.observe(frame(0.0, "hallway", elder()))
    engine.observe(frame(1.5, "front_door", elder("stand", "front_door", 7, "stranger")))
    engine.observe(frame(10.0, "hallway", elder("lie", "floor")))
    out = engine.observe(frame(12.0, "hallway", elder("lie", "floor")))
    fall = [d for d in out if d.agent == "fall"]
    assert fall and fall[0].actions == ("A1", "A4", "A3")
    assert engine.camera_fps("hallway") == 30.0
    assert any(d.actions == ("A5",) for d in engine.advance(60.0))


def test_a_stale_stand_on_another_camera_cannot_clear():
    # Review finding 3: living saw her standing at t=0; she falls in the hallway and says I'm OK.
    engine = make_engine("S2", config())
    engine.observe(frame(0.0, "living", elder("stand", "living")))
    engine.observe(frame(5.0, "hallway", elder("stand", "hallway")))
    engine.observe(frame(10.0, "hallway", elder("lie", "floor")))
    engine.observe(frame(12.0, "hallway", elder("lie", "floor")))
    engine.observe(said(14.0, "I'm OK", directed=True))
    out = []
    for k in range(15, 31):
        out += engine.observe(frame(float(k), "hallway", elder("lie", "floor")))
    assert not any(d.alert == "cleared_by_elder" for d in out)
    assert engine.hud().alert == "pre-alert"


def test_a_view_older_than_2_s_does_not_count_as_standing():
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("stand", "living")))
    engine.observe(said(1.0, "I feel dizzy"))
    engine.observe(said(2.0, "I'm OK", directed=True))
    out = []
    for t in (3.0, 6.0, 8.0):  # only the empty hallway sends frames; the living view goes stale
        out += engine.observe(frame(t, "hallway"))
    out += engine.observe(said(9.0, "thank you", directed=True))
    assert not any(d.alert == "cleared_by_elder" for d in out)


def test_the_freshest_view_of_the_elder_wins():
    engine = make_engine("S2", config())
    engine.observe(frame(0.0, "hallway", elder("lie", "floor")))
    engine.observe(said(1.0, "I feel dizzy"))
    engine.observe(said(2.0, "I'm OK", directed=True))
    out = []
    for t in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        out += engine.observe(frame(t, "living", elder("stand", "living")))
        out += engine.observe(frame(t + 0.5, "hallway", elder("sit", "floor")))
    assert not any(d.alert == "cleared_by_elder" for d in out)


def test_an_ok_that_negates_the_symptom_clears_and_never_calls():
    # Re-review HIGH: dizzy at 5; standing, "I'm OK, I'm not dizzy anymore" at 10 became a new SOS -> A5 at 35.
    engine = make_engine("S2", config("14:00:00"))
    out = list(engine.observe(frame(0.0, "living", elder("stand", "living"))))
    out += engine.observe(said(5.0, "I feel dizzy"))
    for t in (7.0, 9.0):
        out += engine.observe(frame(t, "living", elder("stand", "living")))
    reply = engine.observe(said(10.0, "I'm OK, I'm not dizzy anymore", directed=True))
    assert not any("A1" in d.actions for d in reply)  # no new SOS trigger
    assert any("answered" in d.reason for d in reply)
    for t in (11.0, 12.0, 13.0, 14.0, 15.0, 16.0):
        out += engine.observe(frame(t, "living", elder("stand", "living")))
    out += engine.advance(60.0)
    assert [d.t for d in out if d.alert == "cleared_by_elder"] == [pytest.approx(15.0)]
    assert not any("A5" in d.actions for d in out)


def test_the_ok_not_dizzy_scenario_clears_without_a_call_in_every_strategy():
    from ahjia.replay import SCENARIO_DIR, load_scenario, run_scenario

    scn = load_scenario(SCENARIO_DIR / "scripted" / "ok_not_dizzy.yaml")
    for strategy in ("S0", "S1", "S2"):
        ds = run_scenario(scn, strategy).decisions
        assert [d.t for d in ds if "A1" in d.actions and d.agent == "voice_sos"] == [pytest.approx(5.0)], strategy
        assert not any("A5" in d.actions for d in ds), strategy
        cleared = [d.t for d in ds if d.alert == "cleared_by_elder"]
        assert cleared and 15.0 <= cleared[0] <= 16.0, strategy


def test_repeated_help_while_escalated_places_one_call():
    # Re-review MEDIUM: help at 19, 20, 21, 22 and a TV "help me" at 23 gave five A1+A4+A5 in 5 s.
    engine = make_engine("S2", config())
    out = night_fall(engine)
    for t in (19.0, 20.0, 21.0, 22.0):
        out += engine.observe(said(t, "help"))
    out += engine.observe(said(23.0, "help me", source="tv"))
    calls = [d for d in out if "A5" in d.actions]
    assert [d.t for d in calls] == [pytest.approx(19.0)]
    assert sum("already escalated" in d.reason for d in out) == 4


def test_a_symptom_after_an_immediate_call_does_not_call_again():
    # Re-review LOW 4: help at 5 then dizzy at 10 dialled A5 again at 40 s.
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("sit", "sofa")))
    out = list(engine.observe(said(5.0, "help")))
    second = engine.observe(said(10.0, "I feel dizzy"))
    assert [d.actions for d in second if d.agent == "voice_sos"] == [("A1", "A4")]
    out += second + engine.advance(100.0)
    assert [d.t for d in out if "A5" in d.actions] == [pytest.approx(5.0)]


def test_no_keep_alive_frames_while_no_im_ok_is_pending():
    # Re-review LOW 3: keep-alive frames were asked for whenever an alert was on and she stood.
    engine = make_engine("S2", config("10:20:00"))
    engine.observe(frame(0.0, "living", elder("stand", "living")))
    engine.observe(said(1.0, "I feel dizzy"))
    engine.observe(frame(2.0, "living", elder("stand", "living")))
    assert engine.next_deadline()[1] is None
    engine.observe(said(3.0, "I'm OK", directed=True))
    assert engine.next_deadline() == (pytest.approx(3.0), "living")
