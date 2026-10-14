"""Demo pipeline with a fake detector, a fake engine and a tiny synthetic video."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from ahjia import demo
from ahjia.events import Decision, HudState, Observation, SpeechObs
from ahjia.perception.config import CameraConfig, TimedSpeech
from ahjia.perception.pose import Detection, PoseDetector
from ahjia.perception.zones import build_zone_map
from perception_samples import lying_flat, standing

pytestmark = pytest.mark.unit

W, H = 640, 360


def detection(sample) -> Detection:
    kps, bbox = sample
    return Detection(bbox=bbox, keypoints=kps, score=0.9)


class FakeDetector:
    """Returns the scripted detections for call i (empty after the script ends)."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def detect(self, frame):
        i = self.calls
        self.calls += 1
        return list(self._script[i]) if i < len(self._script) else []


class FakeEngine:
    strategy = "S2"

    def __init__(self):
        self.events: list[tuple[str, float]] = []
        self.observations: list[Observation] = []
        self.last_t = -1.0

    def _move(self, kind, t):
        assert t >= self.last_t, f"time went backwards: {t} < {self.last_t}"
        self.last_t = t
        self.events.append((kind, t))

    def observe(self, obs):
        self._move("observe", obs.t)
        self.observations.append(obs)
        if obs.speech is None:
            return ()
        return (Decision(t=obs.t, clock="14:00:00", strategy="S2", agent="voice_sos",
                         actions=("A1", "A4"), reason="keyword", latency_s=1.9,
                         alert="pre_alert", genai="off"),)

    def advance(self, t):
        self._move("advance", t)
        return ()

    def hud(self):
        return HudState(agent="fall", alert="none")

    def camera_fps(self, camera):
        return 10.0


CAMERA = CameraConfig(
    name="living", start_clock="14:05:00", elder_alone=True, identity="elder",
    zones=build_zone_map({"sofa": [(0.0, 0.5), (0.4, 0.5), (0.4, 1.0), (0.0, 1.0)]}),
)


def frames(n, step=0.1):
    return [(i * step, np.full((H, W, 3), 50, dtype=np.uint8)) for i in range(n)]


def test_observations_carry_identity_zone_and_posture():
    standing_on_sofa = standing(cx=0.2 * 1280, ground=0.9 * 720)
    det = FakeDetector([[detection(standing_on_sofa)]] * 3)
    eng = FakeEngine()
    res = demo.run_pipeline(frames(3), det, eng, CAMERA)
    person = eng.observations[0].persons[0]
    assert (person.identity, person.zone, person.posture, person.track_id) == ("elder", "sofa", "stand", 1)
    assert eng.observations[0].camera == "living"
    assert res.frames == 3 and res.frames_with_person == 3 and res.persons_total == 3
    assert res.detection_rate == 1.0


def test_track_id_is_stable_and_empty_frames_give_empty_observations():
    script = [[detection(standing())], [], [detection(standing())]]
    eng = FakeEngine()
    res = demo.run_pipeline(frames(3), FakeDetector(script), eng, CAMERA)
    assert [len(o.persons) for o in eng.observations] == [1, 0, 1]
    assert eng.observations[0].persons[0].track_id == eng.observations[2].persons[0].track_id
    assert res.frames_with_person == 2


def test_posture_is_smoothed_over_frames():
    script = [[detection(standing())]] * 4 + [[detection(lying_flat())]] + [[detection(standing())]] * 3
    eng = FakeEngine()
    demo.run_pipeline(frames(len(script)), FakeDetector(script), eng, CAMERA)
    assert [o.persons[0].posture for o in eng.observations] == ["stand"] * len(script)


def test_speech_is_delivered_in_time_order_before_the_next_frame():
    speech = (
        TimedSpeech(0.05, SpeechObs("hi", "elder", True)),
        TimedSpeech(0.25, SpeechObs("help", "elder", False)),
    )
    eng = FakeEngine()
    res = demo.run_pipeline(frames(5), FakeDetector([]), eng, CAMERA, speech=speech)
    kinds = [(round(o.t, 2), o.speech.text if o.speech else "frame") for o in eng.observations]
    assert kinds == [(0.0, "frame"), (0.05, "hi"), (0.1, "frame"), (0.2, "frame"), (0.25, "help"),
                     (0.3, "frame"), (0.4, "frame")]
    assert [d.t for d in res.decisions] == [0.05, 0.25]


def test_speech_after_last_frame_but_before_end_is_still_delivered():
    speech = (TimedSpeech(0.45, SpeechObs("late", "elder", False)),
              TimedSpeech(99.0, SpeechObs("never", "elder", False)))
    eng = FakeEngine()
    res = demo.run_pipeline(frames(3), FakeDetector([]), eng, CAMERA, speech=speech, end_t=0.5)
    texts = [o.speech.text for o in eng.observations if o.speech]
    assert texts == ["late"]
    assert eng.events[-1] == ("advance", 0.5)
    assert len(res.decisions) == 1


def test_on_frame_receives_annotated_frames_with_the_engine_hud():
    out = []
    res = demo.run_pipeline(frames(2), FakeDetector([[detection(standing())]] * 2), FakeEngine(),
                            CAMERA, on_frame=out.append)
    assert len(out) == 2
    assert out[0].shape == (H, W, 3)
    assert np.any(out[0] != 50)
    assert res.analysis_s >= 0 and res.analysis_fps >= 0


class HudEngine(FakeEngine):
    """HUD text changes with every observation, to show which frames reuse the HUD."""

    def hud(self):
        return HudState(agent=f"seen{len(self.observations)}")


def test_held_frames_are_drawn_but_not_analysed_or_observed():
    items = [(i * 0.05, np.full((H, W, 3), 50, dtype=np.uint8), i % 2 == 0) for i in range(6)]
    det = FakeDetector([[detection(standing())]] * 3)
    eng, out = HudEngine(), []
    res = demo.run_pipeline(items, det, eng, CAMERA, on_frame=out.append)
    assert det.calls == 3
    assert len(eng.observations) == 3
    assert (res.frames, res.frames_written) == (3, 6)
    assert len(out) == 6
    assert [t for kind, t in eng.events if kind == "observe"] == [0.0, 0.1, 0.2]


def test_held_frame_reuses_latest_people_and_hud():
    items = [(0.0, np.full((H, W, 3), 50, dtype=np.uint8), True),
             (0.05, np.full((H, W, 3), 50, dtype=np.uint8), False)]
    out = []
    demo.run_pipeline(items, FakeDetector([[detection(standing())]]), HudEngine(), CAMERA,
                      on_frame=out.append)
    status_rows = slice(0, 50)  # status text differs between frames; compare below it
    assert np.array_equal(out[0][50:, :], out[1][50:, :])
    assert not np.array_equal(out[0][status_rows], out[1][status_rows])


def test_two_tuples_still_mean_analyse():
    det = FakeDetector([])
    demo.run_pipeline(frames(3), det, FakeEngine(), CAMERA)
    assert det.calls == 3


def test_analysis_fps_is_zero_for_empty_input():
    res = demo.run_pipeline([], FakeDetector([]), FakeEngine(), CAMERA)
    assert (res.frames, res.analysis_fps, res.detection_rate) == (0, 0.0, 0.0)


def test_clock_at_wraps_midnight():
    assert demo.clock_at("14:05:00", 12.4) == "14:05:12"
    assert demo.clock_at("23:59:50", 20) == "00:00:10"


def test_observation_json_is_valid_and_rounded():
    eng = FakeEngine()
    demo.run_pipeline(frames(1), FakeDetector([[detection(standing())]]), eng, CAMERA)
    line = demo.observation_to_json(eng.observations[0])
    record = json.loads(line)
    assert record["camera"] == "living"
    assert len(record["persons"][0]["keypoints"]) == 17
    assert all(len(str(v).split(".")[-1]) <= 3 for v in record["persons"][0]["bbox"])
    assert json.loads(demo.observation_to_json(Observation(t=1.0, speech=SpeechObs("x"))))["speech"]


def write_clip(path, n=20, fps=20.0):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n):
        w.write(np.full((H, W, 3), 30 + i, dtype=np.uint8))
    w.release()


def make_args(tmp_path, **over):
    cam = tmp_path / "cam.yaml"
    cam.write_text('camera: living\nstart_clock: "14:05:00"\nzones:\n  sofa: [[0,0.5],[0.4,0.5],[0.4,1],[0,1]]\n')
    sp = tmp_path / "s.yaml"
    sp.write_text("- {t: 0.3, text: help, source: elder}\n")
    clip = tmp_path / "in.mp4"
    write_clip(clip)
    base = dict(video=str(clip), camera=str(cam), speech=str(sp), out=str(tmp_path / "out"),
                fps=10.0, model="unused.pt")
    base.update(over)
    return argparse.Namespace(**base)


def test_main_writes_video_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, "_make_engine", lambda camera: FakeEngine())
    monkeypatch.setattr(demo, "PoseDetector", lambda model_path: FakeDetector([[detection(standing())]] * 99))
    assert demo.main(make_args(tmp_path)) == 0
    out = tmp_path / "out"
    assert (out / "annotated.mp4").stat().st_size > 0
    decisions = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    assert decisions and decisions[0]["agent"] == "voice_sos"
    obs = (out / "observations.jsonl").read_text().splitlines()
    assert len(obs) == 10 + 1  # 10 analysed frames + 1 speech segment
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20  # every source frame, not only analysed ones
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
    cap.release()


def test_main_reports_missing_engine(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "ahjia.engine", None)  # import raises ImportError
    assert demo.main(make_args(tmp_path)) == 2
    assert "engine" in caplog.text


@pytest.mark.parametrize("key", ["video", "camera", "speech"])
def test_main_reports_missing_inputs(tmp_path, monkeypatch, key):
    monkeypatch.setattr(demo, "_make_engine", lambda camera: FakeEngine())
    args = make_args(tmp_path, **{key: str(tmp_path / "missing.file")})
    assert demo.main(args) == 2


# ---------------------------------------------------------------- failure paths

def _use_fakes(monkeypatch, detector=None):
    monkeypatch.setattr(demo, "_make_engine", lambda camera: FakeEngine())
    det = detector if detector is not None else FakeDetector([[detection(standing())]] * 99)
    monkeypatch.setattr(demo, "PoseDetector", lambda model_path: det)


def _out_files(tmp_path):
    out = tmp_path / "out"
    return sorted(p.name for p in out.iterdir()) if out.exists() else None


@pytest.mark.parametrize("fps", [0.0, -5.0, float("nan"), float("inf")])
def test_main_rejects_bad_fps_before_touching_anything(tmp_path, monkeypatch, caplog, fps):
    _use_fakes(monkeypatch)
    assert demo.main(make_args(tmp_path, fps=fps)) == 2
    assert "--fps must be a positive number" in caplog.text
    assert _out_files(tmp_path) is None  # not even the output folder


def test_main_reports_missing_model_file(tmp_path, monkeypatch, caplog):
    _use_fakes(monkeypatch)
    missing = str(tmp_path / "nowhere" / "model.pt")
    assert demo.main(make_args(tmp_path, model=missing)) == 2
    assert "pose model not found" in caplog.text and missing in caplog.text
    assert _out_files(tmp_path) is None


def test_main_accepts_bare_model_name_and_existing_path(tmp_path, monkeypatch):
    _use_fakes(monkeypatch)
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    assert demo.main(make_args(tmp_path, model=str(weights))) == 0


class ExplodingDetector:
    """Fails on call number `at` (0 = the first call, where real weights would load)."""

    def __init__(self, at, exc=None):
        self.calls, self._at = 0, at
        self._exc = exc or RuntimeError("weights are corrupt")

    def detect(self, frame):
        self.calls += 1
        if self.calls - 1 == self._at:
            raise self._exc
        return []


def test_model_load_failure_leaves_no_files(tmp_path, monkeypatch, caplog):
    """Weights load lazily on the first frame; the video writer must not exist yet."""
    _use_fakes(monkeypatch, ExplodingDetector(0))
    assert demo.main(make_args(tmp_path)) == 2
    assert "demo failed: weights are corrupt" in caplog.text
    assert _out_files(tmp_path) is None


def test_unexpected_error_type_is_named_in_the_message(tmp_path, monkeypatch, caplog):
    _use_fakes(monkeypatch, ExplodingDetector(0, KeyError("boom")))
    assert demo.main(make_args(tmp_path)) == 2
    assert "KeyError" in caplog.text


def test_failure_after_the_writer_started_removes_partial_output(tmp_path, monkeypatch, caplog):
    _use_fakes(monkeypatch, ExplodingDetector(4))
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine")
    assert demo.main(make_args(tmp_path)) == 2
    assert _out_files(tmp_path) == ["keep.txt"]  # annotated.mp4 gone, foreign file kept, folder kept


def test_failure_removes_the_output_folder_it_created(tmp_path, monkeypatch):
    _use_fakes(monkeypatch, ExplodingDetector(4))
    assert demo.main(make_args(tmp_path)) == 2
    assert _out_files(tmp_path) is None


def test_keyboard_interrupt_cleans_up_and_propagates(tmp_path, monkeypatch):
    _use_fakes(monkeypatch, ExplodingDetector(4, KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        demo.main(make_args(tmp_path))
    assert _out_files(tmp_path) is None


def test_failure_while_writing_json_removes_everything(tmp_path, monkeypatch, caplog):
    _use_fakes(monkeypatch)

    def boom(path, lines):
        raise OSError("disk full")
    monkeypatch.setattr(demo, "write_jsonl", boom)
    assert demo.main(make_args(tmp_path)) == 2
    assert "disk full" in caplog.text
    assert _out_files(tmp_path) is None


def _fake_ffmpeg(tmp_path, body):
    script = tmp_path / "ffmpeg"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    return str(script)


@pytest.mark.skipif(sys.platform == "win32", reason="fake ffmpeg is a shell script")
@pytest.mark.parametrize("body", ["exit 3", "cat > /dev/null; exit 3"],
                         ids=["quits-early", "fails-at-close"])
def test_ffmpeg_failure_is_a_clear_error_and_leaves_no_files(tmp_path, monkeypatch, caplog, body):
    from ahjia.perception import video

    _use_fakes(monkeypatch)
    fake = _fake_ffmpeg(tmp_path, body)
    monkeypatch.setattr(video, "find_ffmpeg", lambda: fake)
    assert demo.main(make_args(tmp_path)) == 2
    assert "ffmpeg failed (exit code 3)" in caplog.text
    assert _out_files(tmp_path) is None


def test_video_without_frames_warns_and_writes_json_only(tmp_path, monkeypatch, caplog):
    _use_fakes(monkeypatch)
    monkeypatch.setattr(demo, "iter_all_frames", lambda path, fps: iter(()))
    assert demo.main(make_args(tmp_path)) == 0
    assert "no frames" in caplog.text
    assert _out_files(tmp_path) == ["decisions.jsonl", "observations.jsonl"]


@pytest.mark.integration
def test_real_yolo_on_a_short_clip(tmp_path):
    """Needs weights and a clip: AHJIA_POSE_MODEL and AHJIA_TEST_VIDEO (full paths)."""
    model, video = os.environ.get("AHJIA_POSE_MODEL"), os.environ.get("AHJIA_TEST_VIDEO")
    if not (model and video and Path(model).is_file() and Path(video).is_file()):
        pytest.skip("set AHJIA_POSE_MODEL and AHJIA_TEST_VIDEO to run")
    from ahjia.perception.video import iter_frames

    clip = [item for _, item in zip(range(5), iter_frames(video, 5.0))]
    items = [(i * 0.2, f) for i, f in enumerate(clip)]
    res = demo.run_pipeline(items, PoseDetector(model_path=model), FakeEngine(), CAMERA)
    assert res.frames == 5
    assert res.analysis_s > 0
