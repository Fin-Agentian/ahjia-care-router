"""Guardrails: deterministic alert / escalate / clear rules (DESIGN.md "Guardrails", P:243-251)."""

import pytest

from ahjia.events import ActionRequest
from ahjia.guardrails import ESCALATE_COOLDOWN_S, Guardrails, is_ok_phrase

pytestmark = pytest.mark.unit


def fall(t: float = 14.0, camera: str = "hallway") -> ActionRequest:
    return ActionRequest(t=t, agent="fall", actions=("A1", "A4", "A3"),
                         reason="lying outside bed/sofa >= 2 s", camera=camera, safety=True)


def sos(t: float, immediate: bool, reason: str = "heard a keyword") -> ActionRequest:
    actions = ("A1", "A4", "A5") if immediate else ("A1", "A4")
    return ActionRequest(t=t, agent="voice_sos", actions=actions, reason=reason, safety=True)


def inactivity(t: float) -> ActionRequest:
    return ActionRequest(t=t, agent="inactivity", actions=("A1",), reason="in the bathroom >= 30 min",
                         safety=True)


def night_exit(t: float) -> ActionRequest:
    return ActionRequest(t=t, agent="night_exit", actions=("A1", "A4"), reason="at the front door at night",
                         camera="front_door", safety=True)


def stranger(t: float) -> ActionRequest:
    return ActionRequest(t=t, agent="visitor", actions=("A4",), reason="stranger at night",
                         camera="front_door", safety=True)


def test_fall_trigger_gives_pre_alert_with_voice_text_and_look_closer():
    g = Guardrails()
    out = g.on_request(fall())
    assert out[0].actions == ("A1", "A4", "A3")
    assert g.alert == "pre_alert" and g.active


def test_voice_trigger_without_camera_has_no_look_closer():
    g = Guardrails()
    out = g.on_request(sos(5.0, immediate=False))
    assert out[0].actions == ("A1", "A4")


def test_immediate_sos_escalates_at_once():
    g = Guardrails()
    out = g.on_request(sos(5.0, immediate=True))
    assert out[0].actions == ("A1", "A4", "A5")
    assert g.alert == "escalated"


def test_other_sos_escalates_after_30_s_without_answer():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    assert g.advance(34.9) == ()
    out = g.advance(36.0)
    assert out[0].actions == ("A5",) and out[0].t == pytest.approx(35.0)
    assert g.alert == "escalated"
    assert g.advance(100.0) == ()


def test_an_answer_stops_the_30_s_rule_but_does_not_clear():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.on_elder_speech(10.0, "a little")
    assert g.advance(60.0) == ()
    assert g.alert == "pre_alert"


def test_the_triggering_utterance_is_not_an_answer():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.on_elder_speech(5.0, "我有點暈")
    assert g.advance(35.0)[0].actions == ("A5",)


def test_immediate_sos_during_pre_alert_gives_voice_text_and_call_at_once():
    g = Guardrails()
    g.on_request(fall(14.0))
    out = g.on_request(sos(19.0, immediate=True))
    assert out[0].actions == ("A1", "A4", "A5")
    assert g.alert == "escalated"
    assert g.state.camera == "hallway"  # A3 stays on the fall camera


def test_immediate_sos_after_escalation_calls_again_once_the_cooldown_has_passed():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.advance(35.0)  # the 30 s rule places the call at 35 s
    out = g.on_request(sos(50.0, immediate=True, reason="heard 'help'"))
    assert out[0].actions == ("A0",) and "already escalated · call placed 15 s ago" in out[0].reason
    out = g.on_request(sos(35.0 + ESCALATE_COOLDOWN_S, immediate=True, reason="heard 'help'"))
    assert out[0].actions == ("A1", "A4", "A5") and g.alert == "escalated"


def test_repeated_help_while_escalated_calls_only_once():
    # Re-review MEDIUM: help at 19, 20, 21, 22 and a TV "help me" at 23 gave five calls in 5 s.
    g = Guardrails()
    g.on_request(fall(14.0))
    first = g.on_request(sos(19.0, immediate=True, reason="heard 'help'"))
    assert first[0].actions == ("A1", "A4", "A5")
    repeats = [g.on_request(sos(t, immediate=True, reason="heard 'help'"))[0] for t in (20.0, 21.0, 22.0, 23.0)]
    assert all(o.actions == ("A0",) and o.kind == "log" for o in repeats)
    assert "already escalated · call placed 4 s ago" in repeats[-1].reason
    assert g.alert == "escalated" and g.state.camera == "hallway"
    later = g.on_request(sos(19.0 + ESCALATE_COOLDOWN_S + 1.0, immediate=True, reason="heard 'help'"))
    assert later[0].actions == ("A1", "A4", "A5")


def test_help_during_the_cooldown_still_voids_an_earlier_im_ok():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=True))
    g.on_elder_speech(8.0, "I'm OK")
    assert g.on_request(sos(9.0, immediate=True))[0].actions == ("A0",)
    g.on_elder_posture(10.0, standing=True, camera="living")
    assert g.on_elder_posture(16.0, standing=True, camera="living") == ()
    assert g.alert == "escalated"


def test_a_symptom_after_an_immediate_call_updates_the_family_but_arms_no_second_call():
    # Re-review LOW 4: help at 5 then dizzy at 10 used to dial A5 again at 40 s.
    g = Guardrails()
    g.on_request(sos(5.0, immediate=True))
    out = g.on_request(sos(10.0, immediate=False, reason="heard 'dizzy'"))
    assert out[0].actions == ("A1", "A4") and "dizzy" in out[0].reason and out[0].alert == "escalated"
    assert g.next_deadline() is None
    assert g.advance(100.0) == ()


def test_repeated_trigger_from_the_same_agent_while_its_timer_runs_is_only_logged():
    g = Guardrails()
    g.on_request(fall(14.0))
    out = g.on_request(fall(16.0, camera="living"))  # a second camera sees the same fall
    assert out[0].actions == ("A0",)
    assert g.alert == "pre_alert" and g.next_deadline() == (pytest.approx(44.0), None)


def test_a_second_sos_word_is_not_an_answer_and_re_alerts():
    # Review finding 1: "dizzy" then "chest pain" used to stop the 30 s rule.
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False, reason="heard 'dizzy'"))
    assert g.on_elder_speech(20.0, "my chest pain is bad") == ()
    out = g.on_request(sos(20.0, immediate=False, reason="heard 'chest pain'"))
    assert out[0].actions == ("A1", "A4") and out[0].kind == "trigger"
    assert "chest pain" in out[0].reason
    assert g.next_deadline() == (pytest.approx(35.0), None)  # the first 30 s window still counts
    a5 = g.advance(40.0)
    assert a5[0].actions == ("A5",) and a5[0].t == pytest.approx(35.0) and g.alert == "escalated"


def test_an_ok_that_negates_the_symptom_is_an_answer_and_an_im_ok_not_a_new_sos():
    # Re-review HIGH: dizzy at 5; standing, "I'm OK, I'm not dizzy anymore" at 10 re-armed the timer.
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False, reason="heard 'dizzy'"))
    g.on_elder_posture(9.0, standing=True, camera="living")
    out = g.on_elder_speech(10.0, "I'm OK, I'm not dizzy anymore")
    assert [o.actions for o in out] == [("A0",)] and "answered" in out[0].reason
    assert g.state.ok_said_t == pytest.approx(10.0) and g.next_deadline() == (pytest.approx(15.0), "living")
    cleared = g.on_elder_posture(15.0, standing=True, camera="living")
    assert cleared[0].alert == "cleared_by_elder"
    assert g.advance(40.0) == ()


def test_a_negated_symptom_without_an_ok_phrase_is_a_plain_answer():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False, reason="heard 'dizzy'"))
    out = g.on_elder_speech(10.0, "I'm not dizzy anymore")
    assert [o.actions for o in out] == [("A0",)] and "answered" in out[0].reason
    assert g.state.ok_said_t is None and g.next_deadline() is None
    assert g.advance(60.0) == () and g.alert == "pre_alert"


def test_chinese_ok_with_a_negated_symptom_and_standing_5_s_clears():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False, reason="heard '暈'"))
    g.on_elder_speech(10.0, "我沒事，已經不暈了")
    g.on_elder_posture(11.0, standing=True, camera="living")
    assert g.on_elder_posture(16.0, standing=True, camera="living")[0].alert == "cleared_by_elder"


def test_an_ok_with_a_remaining_symptom_is_a_new_sos_never_an_im_ok():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False, reason="heard 'dizzy'"))
    assert g.on_elder_speech(10.0, "I'm OK, I'm not dizzy, but my chest pain is bad") == ()
    assert g.state.ok_said_t is None and g.next_deadline() == (pytest.approx(35.0), None)


def test_sos_speech_never_answers_the_check_in():
    g = Guardrails()
    g.on_request(inactivity(100.0))
    assert g.on_elder_speech(120.0, "I feel dizzy") == ()
    assert g.state.checkin_deadline == pytest.approx(160.0)


def test_sos_word_after_an_answer_re_arms_the_30_s_rule():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.on_elder_speech(10.0, "a little")
    out = g.on_request(sos(40.0, immediate=False, reason="heard 'chest pain'"))
    assert out[0].actions == ("A1", "A4")
    assert g.advance(70.0)[0].actions == ("A5",)


def test_a_fall_after_an_answer_alerts_again_and_re_arms_the_30_s_rule():
    # Review finding 2: dizzy -> answer -> fall used to give only A0.
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.on_elder_speech(10.0, "a little")
    out = g.on_request(fall(72.0))
    assert out[0].actions == ("A1", "A4", "A3") and g.state.camera == "hallway"
    assert g.advance(101.9) == ()
    a5 = g.advance(102.0)
    assert a5[0].actions == ("A5",) and g.alert == "escalated"


def test_night_exit_then_answer_then_fall_alerts_again():
    g = Guardrails()
    g.on_request(night_exit(6.0))
    g.on_elder_speech(10.0, "I just wanted some air")
    out = g.on_request(fall(40.0))
    assert out[0].actions == ("A1", "A4", "A3")
    assert g.advance(70.0)[0].actions == ("A5",)


def test_a_fall_after_a_stranger_notice_alerts_and_arms_the_30_s_rule():
    g = Guardrails()
    g.on_request(stranger(3.0))
    out = g.on_request(fall(20.0))
    assert out[0].actions == ("A1", "A4", "A3")
    assert g.advance(50.0)[0].actions == ("A5",)


def test_a_trigger_from_another_agent_alerts_again_but_keeps_the_earlier_deadline():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    out = g.on_request(fall(20.0))
    assert out[0].actions == ("A1", "A4", "A3")
    assert g.next_deadline() == (pytest.approx(35.0), None)


def test_the_same_agent_alerts_again_once_its_timer_stopped():
    g = Guardrails()
    g.on_request(fall(14.0))
    g.on_elder_speech(16.0, "ouch")
    out = g.on_request(fall(30.0, camera="living"))
    assert out[0].actions == ("A1", "A4", "A3")
    assert g.next_deadline() == (pytest.approx(60.0), None)


def test_a_new_trigger_after_escalation_alerts_again_but_the_call_is_already_placed():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.advance(35.0)
    out = g.on_request(fall(40.0))
    assert out[0].actions == ("A1", "A4", "A3") and g.alert == "escalated"
    assert g.state.camera == "hallway" and g.next_deadline() is None
    assert g.advance(70.0) == ()


def test_a_fall_30_s_after_the_call_arms_no_second_call():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.advance(35.0)  # the call is placed at 35 s
    out = g.on_request(fall(65.0))
    assert out[0].actions == ("A1", "A4", "A3") and g.alert == "escalated"
    assert g.next_deadline() is None and g.advance(200.0) == ()


def test_a_fall_2_min_after_the_call_calls_again_after_30_s_without_an_answer():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.advance(35.0)  # the call is placed at 35 s
    out = g.on_request(fall(155.0))
    assert out[0].actions == ("A1", "A4", "A3") and g.alert == "escalated"
    assert g.next_deadline() == (pytest.approx(185.0), None)
    assert g.advance(184.9) == ()
    a5 = g.advance(186.0)
    assert a5[0].actions == ("A5",) and a5[0].t == pytest.approx(185.0) and g.alert == "escalated"


def test_an_im_ok_before_a_new_trigger_does_not_clear_it():
    g = Guardrails()
    g.on_request(fall(7.0))
    g.on_elder_speech(10.0, "I'm OK")
    g.on_request(night_exit(12.0))
    g.on_elder_posture(13.0, standing=True, camera="front_door")
    assert g.on_elder_posture(18.0, standing=True, camera="front_door") == ()
    assert g.alert == "pre_alert"


def test_im_ok_while_lying_does_not_clear():
    g = Guardrails()
    g.on_request(fall(14.0))
    g.on_elder_posture(15.0, standing=False, camera="hallway")
    g.on_elder_speech(20.0, "I'm OK")
    g.on_elder_posture(40.0, standing=False, camera="hallway")
    assert g.alert == "pre_alert"
    assert g.advance(120.0) == ()  # she answered, so no 30 s escalation (P:247)
    assert g.alert == "pre_alert"


def test_im_ok_and_upright_5_s_clears_by_the_elder():
    g = Guardrails()
    g.on_request(fall(14.0))
    g.on_elder_speech(20.0, "我沒事")
    g.on_elder_posture(22.0, standing=True, camera="hallway")
    assert g.on_elder_posture(26.9, standing=True, camera="hallway") == ()
    out = g.on_elder_posture(27.0, standing=True, camera="hallway")
    assert g.alert == "cleared_by_elder" and not g.active
    assert out[0].actions == ("A4",) and "cleared by the elder" in out[0].reason


def test_upright_must_be_continuous():
    g = Guardrails()
    g.on_request(fall(14.0))
    g.on_elder_speech(20.0, "I'm OK")
    g.on_elder_posture(22.0, standing=True, camera="hallway")
    g.on_elder_posture(24.0, standing=False, camera="hallway")
    g.on_elder_posture(25.0, standing=True, camera="hallway")
    assert g.on_elder_posture(29.9, standing=True, camera="hallway") == ()
    assert g.on_elder_posture(30.0, standing=True, camera="hallway")[0].alert == "cleared_by_elder"


def test_upright_time_counts_only_after_the_im_ok():
    g = Guardrails()
    g.on_elder_posture(1.0, standing=True, camera="living")
    g.on_request(sos(5.0, immediate=False))
    g.on_elder_speech(20.0, "I'm okay")
    assert g.on_elder_posture(21.0, standing=True, camera="living") == ()
    assert g.next_deadline() == (pytest.approx(25.0), "living")
    assert g.on_elder_posture(25.0, standing=True, camera="living")[0].alert == "cleared_by_elder"


def test_elder_out_of_view_resets_the_upright_timer():
    g = Guardrails()
    g.on_request(fall(14.0))
    g.on_elder_speech(20.0, "I'm OK")
    g.on_elder_posture(21.0, standing=True, camera="hallway")
    g.on_elder_posture(23.0, standing=None, camera=None)
    assert g.on_elder_posture(26.5, standing=True, camera="hallway") == ()
    assert g.alert == "pre_alert"


def test_app_confirm_clears():
    g = Guardrails()
    g.on_request(fall(14.0))
    out = g.on_app_confirm(20.0)
    assert g.alert == "cleared_by_app"
    assert out[0].actions == ("A0",)
    assert g.advance(100.0) == ()


def test_app_confirm_without_alert_does_nothing():
    assert Guardrails().on_app_confirm(3.0) == ()


def test_nothing_but_the_two_paths_clears():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=False))
    g.on_elder_posture(6.0, standing=True, camera="living")
    g.on_elder_posture(100.0, standing=True, camera="living")
    g.on_elder_speech(7.0, "thank you")
    assert g.active
    with pytest.raises(ValueError):
        g.on_request(ActionRequest(t=8.0, agent="companion", actions=("A0",), reason="tv"))


def test_inactivity_is_a1_first_then_a4_after_60_s_then_30_s_rule():
    g = Guardrails()
    out = g.on_request(inactivity(100.0))
    assert out[0].actions == ("A1",) and g.alert == "none"
    assert g.advance(159.9) == ()
    pre = g.advance(160.0)
    assert pre[0].actions == ("A4",) and g.alert == "pre_alert"
    assert g.advance(190.5)[0].actions == ("A5",)


def test_inactivity_answer_cancels_the_check_in():
    g = Guardrails()
    g.on_request(inactivity(100.0))
    g.on_elder_speech(120.0, "I'm fine, just slow today")
    assert g.advance(300.0) == ()
    assert g.alert == "none"


def test_stranger_at_night_is_notify_only():
    g = Guardrails()
    req = ActionRequest(t=3.0, agent="visitor", actions=("A4",), reason="stranger at night",
                        camera="front_door", safety=True)
    out = g.on_request(req)
    assert out[0].actions == ("A4",) and g.alert == "pre_alert"
    assert out[0].camera == "front_door"
    assert g.state.camera is None  # A4 only: no A3, so no camera goes to full rate
    assert g.advance(500.0) == ()


def test_app_confirm_cancels_a_pending_check_in():
    g = Guardrails()
    g.on_request(inactivity(100.0))
    out = g.on_app_confirm(120.0)
    assert out and out[0].actions == ("A0",) and g.alert == "none"
    assert g.next_deadline() is None
    assert g.advance(300.0) == ()


def test_new_case_after_clear():
    g = Guardrails()
    g.on_request(fall(14.0))
    g.on_app_confirm(20.0)
    out = g.on_request(fall(60.0))
    assert out[0].actions == ("A1", "A4", "A3") and g.alert == "pre_alert"


def test_next_deadline_reports_the_30_s_rule():
    g = Guardrails()
    assert g.next_deadline() is None
    g.on_request(sos(5.0, immediate=False))
    assert g.next_deadline() == (pytest.approx(35.0), None)


@pytest.mark.parametrize("text, expected", [
    ("我沒事", True), ("I'm OK", True), ("i’m okay", True), ("I am OK now", True),
    ("I'm not OK", False), ("help", False),
])
def test_ok_phrase(text, expected):
    assert is_ok_phrase(text) is expected


def test_sitting_up_after_im_ok_neither_clears_nor_escalates():
    g = Guardrails()
    g.on_request(fall(7.0))
    g.on_elder_speech(12.0, "I'm OK")
    g.on_elder_posture(18.0, standing=False, camera="hallway")  # sits up on the floor
    assert g.on_elder_posture(30.0, standing=False, camera="hallway") == ()
    assert g.advance(60.0) == ()  # answered: no A5
    assert g.alert == "pre_alert"  # not standing: no clear


def test_sitting_down_during_the_5_s_resets_the_timer():
    g = Guardrails()
    g.on_request(fall(7.0))
    g.on_elder_speech(10.0, "I'm OK")
    g.on_elder_posture(11.0, standing=True, camera="hallway")
    g.on_elder_posture(14.0, standing=False, camera="hallway")
    g.on_elder_posture(15.0, standing=True, camera="hallway")
    assert g.on_elder_posture(19.9, standing=True, camera="hallway") == ()
    assert g.on_elder_posture(20.0, standing=True, camera="hallway")[0].alert == "cleared_by_elder"


def test_a_clear_after_escalation_is_still_possible():
    g = Guardrails()
    g.on_request(sos(5.0, immediate=True))
    g.on_elder_speech(8.0, "I'm OK")
    g.on_elder_posture(9.0, standing=True, camera="living")
    assert g.on_elder_posture(14.0, standing=True, camera="living")[0].alert == "cleared_by_elder"
