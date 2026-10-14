"""Six care agents (DESIGN.md "Six care agents", P:140-149)."""

import pytest

from ahjia.agents import make_agents
from ahjia.agents.base import AgentContext, IdentityTracker, find_keyword, format_clock, parse_clock
from ahjia.agents.companion import CompanionAgent, CompanionConfig
from ahjia.agents.fall import FallAgent
from ahjia.agents.inactivity import InactivityAgent
from ahjia.agents.night_exit import NightExitAgent
from ahjia.agents.visitor import VisitorAgent
from ahjia.agents.voice_sos import VoiceSosAgent, VoiceSosConfig, classify_sos
from ahjia.events import AGENTS, Observation, PersonObs, RunConfig, SpeechObs

pytestmark = pytest.mark.unit


def ctx(clock: str = "14:00:00", time_scale: float = 1.0, wake: str = "06:30:00") -> AgentContext:
    config = RunConfig(start_clock=clock, cameras=("living", "hallway", "front_door"),
                       time_scale=time_scale, usual_wake_clock=wake)
    return AgentContext.from_run_config(config)


def person(track: int = 1, identity: str = "elder", posture: str = "stand", zone: str = "living") -> PersonObs:
    return PersonObs(track_id=track, identity=identity, posture=posture, zone=zone)


def frame(t: float, camera: str, *persons: PersonObs) -> Observation:
    return Observation(t=t, camera=camera, persons=tuple(persons))


def said(t: float, text: str, source: str = "elder", directed: bool = False) -> Observation:
    return Observation(t=t, speech=SpeechObs(text=text, source=source, directed=directed))


# ---------------------------------------------------------------- base helpers

def test_clock_helpers():
    assert parse_clock("02:10:00") == 7800
    assert format_clock(7800 + 86400 + 14) == "02:10:14"
    c = ctx("21:59:00")
    assert not c.is_night(0.0) and c.is_night(60.0)
    assert c.clock(61.0) == "22:00:01"
    assert ctx("06:59:59").is_night(0.0) and not ctx("07:00:00").is_night(0.0)


def test_keyword_matching_uses_word_boundaries_and_normalises_quotes():
    assert find_keyword("Help!", ("help",)) == "help"
    assert find_keyword("that was helpful", ("help",)) is None
    assert find_keyword("I can’t get up", ("can't get up",)) == "can't get up"
    assert find_keyword("我有點暈", ("好暈", "暈")) == "暈"


def test_identity_tracker_confirms_after_dwell():
    tracker = IdentityTracker(confirm_s=1.5)
    tracker, confirmed = tracker.update(frame(0.0, "living", person(2, "family")))
    assert confirmed == () and tracker.next_deadline() == (1.5, "living")
    tracker, confirmed = tracker.update(frame(1.5, "living", person(2, "family")))
    assert confirmed[0].identity == "family" and confirmed[0].first_seen == 0.0
    tracker, confirmed = tracker.update(frame(2.0, "living", person(2, "family")))
    assert confirmed == () and tracker.next_deadline() is None


def test_make_agents_returns_the_six_agents_in_priority_order():
    agents = make_agents(ctx())
    assert {a.name for a in agents} == set(AGENTS)
    assert agents[0].name == "fall" and agents[-1].name == "companion"


# ---------------------------------------------------------------- fall

def test_fall_fires_after_lying_outside_rest_zone_for_2_s():
    agent = FallAgent(ctx())
    agent.on_observation(frame(0.0, "hallway", person(posture="stand", zone="hallway")))
    assert agent.on_observation(frame(10.0, "hallway", person(posture="lie", zone="floor"))) == ()
    assert agent.next_deadline() == (12.0, "hallway")
    assert agent.on_observation(frame(11.9, "hallway", person(posture="lie", zone="floor"))) == ()
    out = agent.on_observation(frame(12.0, "hallway", person(posture="lie", zone="floor")))
    assert out[0].actions == ("A1", "A4", "A3") and out[0].safety and out[0].camera == "hallway"
    assert "2 s" in out[0].reason
    assert agent.on_observation(frame(13.0, "hallway", person(posture="lie", zone="floor"))) == ()


@pytest.mark.parametrize("zone", ["sofa", "bed"])
def test_lying_on_sofa_or_bed_is_not_a_fall(zone):
    agent = FallAgent(ctx())
    agent.on_observation(frame(0.0, "living", person(posture="stand", zone="living")))
    agent.on_observation(frame(1.0, "living", person(posture="lie", zone=zone)))
    assert agent.on_observation(frame(30.0, "living", person(posture="lie", zone=zone))) == ()
    assert agent.next_deadline() is None


def test_lying_without_an_upright_posture_first_is_not_a_fall():
    agent = FallAgent(ctx())
    agent.on_observation(frame(0.0, "living", person(posture="lie", zone="floor")))
    assert agent.on_observation(frame(5.0, "living", person(posture="lie", zone="floor"))) == ()


def test_getting_up_cancels_and_rearms_the_fall_rule():
    agent = FallAgent(ctx())
    agent.on_observation(frame(0.0, "hallway", person(posture="stand", zone="hallway")))
    agent.on_observation(frame(1.0, "hallway", person(posture="lie", zone="floor")))
    agent.on_observation(frame(2.0, "hallway", person(posture="sit", zone="floor")))
    assert agent.on_observation(frame(3.5, "hallway", person(posture="sit", zone="floor"))) == ()
    agent.on_observation(frame(4.0, "hallway", person(posture="lie", zone="floor")))
    assert agent.on_observation(frame(6.0, "hallway", person(posture="lie", zone="floor")))


def test_person_leaving_the_view_resets_the_fall_track():
    agent = FallAgent(ctx())
    agent.on_observation(frame(0.0, "hallway", person(posture="stand", zone="hallway")))
    agent.on_observation(frame(1.0, "hallway", person(posture="lie", zone="floor")))
    agent.on_observation(frame(1.5, "hallway"))
    assert agent.next_deadline() is None
    assert agent.on_time(10.0) == ()


# ---------------------------------------------------------------- voice SOS

@pytest.mark.parametrize("text", ["救命", "我起不來", "Help!", "I can't get up", "I can’t get up"])
def test_immediate_keywords_ask_for_a5(text):
    out = VoiceSosAgent(ctx()).on_observation(said(1.0, text))
    assert out[0].actions == ("A1", "A4", "A5") and out[0].safety


@pytest.mark.parametrize("text", ["我有點暈", "好暈", "I feel dizzy", "胸口痛", "my chest pain is back"])
def test_other_keywords_ask_for_a1_a4(text):
    out = VoiceSosAgent(ctx()).on_observation(said(1.0, text))
    assert out[0].actions == ("A1", "A4")


def test_plain_speech_does_not_trigger_but_a_tv_help_does():
    agent = VoiceSosAgent(ctx())
    # The real system cannot tell a TV from the elder yet: every "help" is real.
    assert agent.on_observation(said(1.0, "Help! Help!", source="tv"))[0].actions == ("A1", "A4", "A5")
    assert agent.on_observation(said(2.0, "that was helpful")) == ()
    assert agent.on_observation(frame(3.0, "living", person())) == ()
    assert agent.on_time(4.0) == () and agent.next_deadline() is None


def test_voice_sos_keywords_are_configurable():
    agent = VoiceSosAgent(ctx(), VoiceSosConfig(immediate=("tasukete",), other=()))
    assert agent.on_observation(said(1.0, "tasukete"))[0].actions == ("A1", "A4", "A5")
    assert agent.on_observation(said(2.0, "help")) == ()


@pytest.mark.parametrize("text", [
    "I'm not dizzy anymore", "I'm OK, I'm not dizzy anymore", "no longer dizzy", "no chest pain",
    "no more chest pain", "my head isn't dizzy", "I'm not feeling dizzy", "I’m not dizzy",
    "不暈", "我不會暈", "沒有胸口痛", "已經不暈了", "我沒事，已經不暈了",
])
def test_a_negated_symptom_word_is_not_an_sos(text):
    # Review (re-review) HIGH: "I'm OK, I'm not dizzy anymore" used to raise a new alarm.
    assert classify_sos(text, VoiceSosConfig()) is None
    assert VoiceSosAgent(ctx()).on_observation(said(1.0, text)) == ()


@pytest.mark.parametrize("text, keyword", [
    ("I'm not dizzy but my chest pain is bad", "chest pain"),
    ("I'm OK, but my chest pain is bad", "chest pain"),
    ("不暈，可是胸口痛", "胸口痛"),
    ("不暈了，又好暈", "好暈"),
    ("oh no, dizzy again", "dizzy"),
])
def test_a_symptom_word_that_is_not_negated_still_counts(text, keyword):
    assert classify_sos(text, VoiceSosConfig()) == (keyword, False)
    out = VoiceSosAgent(ctx()).on_observation(said(1.0, text))
    assert out[0].actions == ("A1", "A4") and keyword in out[0].reason


@pytest.mark.parametrize("text", ["I don't need help", "no help", "不用救命", "我沒有起不來"])
def test_immediate_words_ignore_negation(text):
    # Safety first: negation is never applied to the immediate words (a known limitation).
    out = VoiceSosAgent(ctx()).on_observation(said(1.0, text))
    assert out[0].actions == ("A1", "A4", "A5")


# ---------------------------------------------------------------- night exit

def test_night_exit_fires_between_midnight_and_six():
    agent = NightExitAgent(ctx("02:10:00"))
    out = agent.on_observation(frame(1.0, "front_door", person(posture="stand", zone="front_door")))
    assert out[0].actions == ("A1", "A4") and out[0].safety and "02:10:01" in out[0].reason
    assert agent.on_observation(frame(2.0, "front_door", person(zone="front_door"))) == ()
    agent.on_observation(frame(3.0, "front_door"))
    assert agent.on_observation(frame(4.0, "front_door", person(zone="front_door")))


def test_no_night_exit_in_the_daytime_or_for_family():
    assert NightExitAgent(ctx("14:00:00")).on_observation(frame(1.0, "front_door", person(zone="front_door"))) == ()
    night = NightExitAgent(ctx("03:00:00"))
    assert night.on_observation(frame(1.0, "front_door", person(identity="family", zone="front_door"))) == ()
    assert night.on_time(2.0) == () and night.next_deadline() is None


def test_unknown_person_counts_as_the_elder_when_living_alone():
    agent = NightExitAgent(ctx("03:00:00"))
    assert agent.on_observation(frame(1.0, "front_door", person(identity="unknown", zone="front_door")))


# ---------------------------------------------------------------- inactivity

def test_bathroom_30_min_scaled_by_time_scale():
    agent = InactivityAgent(ctx("10:00:00", time_scale=0.01))
    agent.on_observation(frame(5.0, "hallway", person(zone="bathroom_door")))
    agent.on_observation(frame(10.0, "hallway"))  # walked into the bathroom, out of view
    assert agent.next_deadline() == (pytest.approx(28.0), None)
    assert agent.on_time(27.9) == ()
    out = agent.on_time(28.0)
    assert out[0].actions == ("A1",) and out[0].safety and "bathroom" in out[0].reason
    assert agent.on_time(40.0) == ()


def test_leaving_the_bathroom_resets_the_timer():
    agent = InactivityAgent(ctx("10:00:00", time_scale=0.01))
    agent.on_observation(frame(5.0, "hallway", person(zone="bathroom")))
    agent.on_observation(frame(15.0, "hallway", person(zone="hallway")))
    assert agent.on_time(40.0) == ()


def test_no_movement_2_h_in_the_daytime():
    agent = InactivityAgent(ctx("10:00:00"))
    agent.on_observation(frame(0.0, "living", person(posture="sit", zone="sofa")))
    assert agent.on_time(7199.0) == ()
    out = agent.on_time(7200.0)
    assert out[0].actions == ("A1",) and "2 h" in out[0].reason


def test_movement_or_speech_resets_the_still_timer():
    agent = InactivityAgent(ctx("10:00:00"))
    agent.on_observation(frame(0.0, "living", person(posture="sit", zone="sofa")))
    agent.on_observation(frame(3600.0, "living", person(posture="stand", zone="living")))
    assert agent.on_time(7200.0) == ()
    agent.on_observation(said(7000.0, "where are my glasses"))
    assert agent.on_time(10700.0) == ()


def test_no_still_alert_at_night():
    agent = InactivityAgent(ctx("23:00:00"))
    agent.on_observation(frame(0.0, "bedroom", person(posture="lie", zone="bed")))
    assert agent.on_time(7200.0) == ()


def test_not_up_60_min_after_usual_wake_time():
    agent = InactivityAgent(ctx("06:00:00", wake="06:30:00"))
    agent.on_observation(frame(0.0, "bedroom", person(posture="lie", zone="bed")))
    assert agent.next_deadline() == (pytest.approx(5400.0), None)
    out = agent.on_time(5400.0)
    assert out and "wake" in out[0].reason


def test_up_before_the_check_means_no_wake_alert():
    agent = InactivityAgent(ctx("06:00:00", wake="06:30:00"))
    agent.on_observation(frame(0.0, "bedroom", person(posture="lie", zone="bed")))
    agent.on_observation(frame(2400.0, "bedroom", person(posture="stand", zone="bedside")))
    agent.on_observation(frame(2500.0, "bedroom"))
    assert agent.on_time(5400.0) == ()


# ---------------------------------------------------------------- visitor

def test_caregiver_arrival_and_leaving_are_logged():
    agent = VisitorAgent(ctx("10:00:00"))
    assert agent.on_observation(frame(0.0, "front_door", person(5, "caregiver", zone="front_door"))) == ()
    out = agent.on_observation(frame(1.5, "front_door", person(5, "caregiver", zone="front_door")))
    assert out[0].actions == ("A0",) and "caregiver arrived" in out[0].reason and not out[0].safety
    agent.on_observation(frame(3600.0, "front_door"))
    deadline = agent.next_deadline()
    assert deadline is not None and deadline[1] is None
    left = agent.on_time(deadline[0])
    assert "caregiver left" in left[0].reason and "11:00:00" in left[0].reason


def test_stranger_at_night_asks_for_a4():
    agent = VisitorAgent(ctx("01:30:00"))
    agent.on_observation(frame(0.0, "front_door", person(9, "stranger", zone="front_door")))
    out = agent.on_observation(frame(2.0, "front_door", person(9, "stranger", zone="front_door")))
    assert out[0].actions == ("A4",) and out[0].safety


def test_stranger_in_the_day_is_logged_and_family_is_left_to_companion():
    agent = VisitorAgent(ctx("14:00:00"))
    agent.on_observation(frame(0.0, "living", person(9, "stranger"), person(2, "family")))
    out = agent.on_observation(frame(2.0, "living", person(9, "stranger"), person(2, "family")))
    assert [r.actions for r in out] == [("A0",)]


# ---------------------------------------------------------------- companion

def test_family_greeting_once_per_day_measured_from_entry():
    agent = CompanionAgent(ctx("14:05:00"))
    agent.on_observation(frame(5.0, "living", person(1, "elder", "sit", "sofa"), person(2, "family")))
    out = agent.on_observation(frame(6.5, "living", person(1, "elder", "sit", "sofa"), person(2, "family")))
    assert out[0].actions == ("A2",) and out[0].t == 5.0 and "family" in out[0].reason
    assert not out[0].safety
    agent.on_observation(frame(100.0, "living", person(1, "elder", "sit", "sofa")))
    agent.on_observation(frame(200.0, "living", person(1, "elder", "sit", "sofa"), person(3, "family")))
    assert agent.on_observation(frame(202.0, "living", person(1, "elder", "sit", "sofa"), person(3, "family"))) == ()


def test_directed_speech_asks_for_a2_and_other_speech_is_a0():
    agent = CompanionAgent(ctx())
    assert agent.on_observation(said(1.0, "AhJia, what day is it?", directed=True))[0].actions == ("A2",)
    tv = agent.on_observation(said(2.0, "and now the news", source="tv"))
    assert tv[0].actions == ("A0",) and "tv" in tv[0].reason  # source label kept for the log only
    family = agent.on_observation(said(3.0, "Mom, tea?", source="family"))
    assert family[0].actions == ("A0",) and "not directed" in family[0].reason


def test_companion_follows_up_after_a_soft_sos_word_but_not_after_an_immediate_one():
    agent = CompanionAgent(ctx())
    follow = agent.on_observation(said(1.0, "我有點暈"))
    assert follow[0].actions == ("A2",) and "follow-up" in follow[0].reason
    immediate = agent.on_observation(said(2.0, "I can't get up"))
    assert [r.actions for r in immediate] == [("A0",)] and "safety path" in immediate[0].reason


def test_reminders_and_daily_summary_follow_the_clock():
    agent = CompanionAgent(ctx("08:59:00"), CompanionConfig(reminders=("09:00:00",), summary_clock="09:01:00"))
    assert agent.next_deadline() == (pytest.approx(60.0), None)
    assert agent.on_time(59.0) == ()
    out = agent.on_time(60.0)
    assert out[0].actions == ("A2",) and "reminder" in out[0].reason
    summary = agent.on_time(120.0)
    assert "summary" in summary[0].reason
    assert agent.next_deadline()[0] > 120.0


def test_companion_choice_rests_on_directed_not_on_the_source_label():
    agent = CompanionAgent(ctx())
    assert agent.on_observation(said(1.0, "AhJia, hello", source="tv", directed=True))[0].actions == ("A2",)
    assert agent.on_observation(said(2.0, "AhJia, hello", source="elder", directed=False))[0].actions == ("A0",)
    tv_help = agent.on_observation(said(3.0, "Help!", source="tv"))
    assert tv_help[0].actions == ("A0",) and "safety path" in tv_help[0].reason
    assert agent.on_observation(said(4.0, "I feel dizzy", source="tv"))[0].actions == ("A2",)
