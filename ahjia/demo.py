"""`python -m ahjia demo`: video -> live pose -> Care Router engine -> annotated video.

Real here: pose detection on the CPU. Simulated: identity (taken from the camera
config), speech (timed transcript file) and Hailo-10H timing (inside the engine).

run_pipeline() takes the frames, detector and engine as parameters so tests can
drive it with fakes; main() wires the real pieces and writes the output files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Union

import numpy as np

from ahjia.events import Decision, Engine, Observation, PersonObs, RunConfig, decision_to_json
from ahjia.overlay import draw_overlay
from ahjia.perception.config import CameraConfig, TimedSpeech, load_camera_config, load_speech
from ahjia.perception.pose import Detection, PoseDetector, PoseDetectorProtocol
from ahjia.perception.posture import PostureConfig, PostureSmoother, classify_posture
from ahjia.perception.tracker import IouTracker
from ahjia.perception.video import FrameWriter, iter_all_frames, open_writer, video_info
from ahjia.perception.zones import foot_point

log = logging.getLogger(__name__)

STRATEGY = "S2"


@dataclass(frozen=True)
class PipelineResult:
    """What one pass over a video produced."""

    frames: int
    frames_with_person: int
    persons_total: int
    decisions: tuple[Decision, ...]
    observations: tuple[Observation, ...]
    analysis_s: float  # wall time spent in pose + posture only
    frames_written: int = 0  # annotated frames sent to on_frame (all source frames)

    @property
    def analysis_fps(self) -> float:
        return self.frames / self.analysis_s if self.analysis_s > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        return self.frames_with_person / self.frames if self.frames else 0.0


class Perceiver:
    """Detections -> PersonObs: track ids, smoothed posture, zone, simulated identity."""

    def __init__(self, camera: CameraConfig, posture_cfg: PostureConfig = PostureConfig()) -> None:
        self._camera = camera
        self._cfg = posture_cfg
        self._tracker = IouTracker()
        self._smoother = PostureSmoother(posture_cfg.window_s)

    def persons(self, t: float, aspect: float, detections: Sequence[Detection]) -> tuple[PersonObs, ...]:
        ids = self._tracker.update(t, [d.bbox for d in detections])
        self._smoother.prune(set(ids))
        out = []
        for tid, det in zip(ids, detections):
            raw = classify_posture(det.keypoints, det.bbox, aspect, self._cfg)
            fx, fy = foot_point(det.keypoints, det.bbox, self._cfg.min_conf)
            out.append(PersonObs(
                track_id=tid,
                identity=self._camera.identity,  # type: ignore[arg-type]  # simulated face ID
                id_conf=1.0,
                posture=self._smoother.update(tid, t, raw),
                zone=self._camera.zones.zone_at(fx, fy),  # type: ignore[arg-type]
                bbox=det.bbox,
                keypoints=det.keypoints,
            ))
        return tuple(out)


def clock_at(start_clock: str, t: float) -> str:
    h, m, s = (int(x) for x in start_clock.split(":"))
    total = int(h * 3600 + m * 60 + s + t) % 86400
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


FrameItem = Union[tuple[float, np.ndarray], tuple[float, np.ndarray, bool]]


def _as_triples(frames: Iterable[FrameItem]) -> Iterator[tuple[float, np.ndarray, bool]]:
    """Accept (t, frame) as "analyse this frame" and (t, frame, analyse) as given."""
    for item in frames:
        yield (item[0], item[1], item[2] if len(item) == 3 else True)  # type: ignore[misc]


def run_pipeline(
    frames: Iterable[FrameItem],
    detector: PoseDetectorProtocol,
    engine: Engine,
    camera: CameraConfig,
    speech: Sequence[TimedSpeech] = (),
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    end_t: Optional[float] = None,
) -> PipelineResult:
    """Feed every analysed frame (and due speech, in time order) to the engine.

    Frames flagged analyse=False are not analysed and the engine does not see
    them; they are drawn with the latest people and HUD so the output video
    keeps the source frame rate. on_frame receives every annotated frame.
    """
    perceiver = Perceiver(camera)
    pending = list(speech)
    decisions: list[Decision] = []
    observations: list[Observation] = []
    n_frames = n_with = n_persons = n_written = 0
    analysis_s, last_t = 0.0, 0.0
    held_persons: tuple[PersonObs, ...] = ()
    held_hud = engine.hud()
    for t, frame, analyse in _as_triples(frames):
        last_t = max(last_t, t)
        if analyse:
            _deliver_speech(pending, t, engine, decisions, observations)
            started = time.perf_counter()
            held_persons = perceiver.persons(t, frame.shape[1] / frame.shape[0], detector.detect(frame))
            analysis_s += time.perf_counter() - started
            decisions.extend(engine.advance(t))
            obs = Observation(t=t, camera=camera.name, persons=held_persons)
            observations.append(obs)
            decisions.extend(engine.observe(obs))
            held_hud = engine.hud()
            n_frames, n_with = n_frames + 1, n_with + bool(held_persons)
            n_persons += len(held_persons)
        if on_frame is not None:
            status = f"T+{t:5.1f}s  clock {clock_at(camera.start_clock, t)}"
            on_frame(draw_overlay(frame, held_persons, held_hud, camera.zones, status=status))
            n_written += 1
    end = last_t if end_t is None else max(end_t, last_t)
    _deliver_speech(pending, end, engine, decisions, observations)
    decisions.extend(engine.advance(end))
    for item in pending:
        log.warning("speech at t=%.1f s is after the end of the video and was ignored", item.t)
    return PipelineResult(n_frames, n_with, n_persons, tuple(decisions), tuple(observations),
                          analysis_s, n_written)


def _deliver_speech(pending: list[TimedSpeech], t: float, engine: Engine,
                    decisions: list[Decision], observations: list[Observation]) -> None:
    while pending and pending[0].t <= t:
        item = pending.pop(0)
        obs = Observation(t=item.t, speech=item.speech)
        observations.append(obs)
        decisions.extend(engine.observe(obs))


def observation_to_json(obs: Observation) -> str:
    """One debug line per observation; keypoints are rounded to keep the file small."""
    record = asdict(obs)
    for person in record["persons"]:
        if person["keypoints"] is not None:
            person["keypoints"] = [[round(v, 3) for v in kp] for kp in person["keypoints"]]
        if person["bbox"] is not None:
            person["bbox"] = [round(v, 3) for v in person["bbox"]]
    return json.dumps(record, ensure_ascii=False)


def write_jsonl(path: Path, lines: Iterable[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _make_engine(camera: CameraConfig):
    try:
        from ahjia.engine import make_engine  # lazy: written by the core module
    except ImportError as exc:
        raise RuntimeError(f"ahjia.engine is not available yet ({exc})") from exc
    return make_engine(STRATEGY, RunConfig(
        start_clock=camera.start_clock, cameras=(camera.name,), elder_alone=camera.elder_alone))


class _Outputs:
    """The files this run starts to write, so a failed run can remove them again."""

    def __init__(self, out_dir: Path) -> None:
        self.dir = out_dir
        self._made_dir = False
        self._files: list[Path] = []

    def prepare(self) -> None:
        self._made_dir = not self.dir.exists()
        self.dir.mkdir(parents=True, exist_ok=True)

    def track(self, name: str) -> Path:
        path = self.dir / name
        self._files.append(path)  # tracked before writing: ffmpeg may leave a partial file
        return path

    def discard(self) -> None:
        for path in self._files:
            path.unlink(missing_ok=True)
        if self._made_dir:
            try:
                self.dir.rmdir()  # only if empty
            except OSError:
                pass


class _LazyWriter:
    """Opens the output video on the first frame, so a failure before any frame
    (for example a pose model that cannot be loaded) leaves no file behind."""

    def __init__(self, outputs: _Outputs, fps: float, size: tuple[int, int]) -> None:
        self._outputs, self._fps, self._size = outputs, fps, size
        self._writer: Optional[FrameWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._writer = open_writer(self._outputs.track("annotated.mp4"), self._fps, self._size)
        self._writer.write(frame)

    def __enter__(self) -> "_LazyWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._writer is not None:
            self._writer.__exit__(*exc_info)


def _check_args(args: argparse.Namespace) -> None:
    if not (math.isfinite(args.fps) and args.fps > 0):
        raise ValueError(f"--fps must be a positive number, got {args.fps}")
    model = Path(args.model)
    # A bare name such as yolov8n-pose.pt is fetched by Ultralytics; an explicit path must exist.
    if model.parent != Path(".") and not model.is_file():
        raise FileNotFoundError(f"pose model not found: {args.model}")


def _run_demo(args: argparse.Namespace, outputs: _Outputs) -> None:
    _check_args(args)
    camera = load_camera_config(args.camera)
    speech = load_speech(args.speech)
    info = video_info(args.video)
    engine = _make_engine(camera)
    detector = PoseDetector(model_path=args.model)
    outputs.prepare()
    out_fps = info.fps if info.fps > 0 else args.fps  # write every source frame
    with _LazyWriter(outputs, out_fps, (info.width, info.height)) as writer:
        result = run_pipeline(iter_all_frames(args.video, args.fps), detector, engine, camera, speech,
                              on_frame=writer.write, end_t=info.duration_s)
    if result.frames_written == 0:
        log.warning("the video has no frames; no annotated.mp4 was written")
    write_jsonl(outputs.track("decisions.jsonl"), (decision_to_json(d) for d in result.decisions))
    write_jsonl(outputs.track("observations.jsonl"), (observation_to_json(o) for o in result.observations))
    log.info("%d frames, person seen in %.0f %%, pose+posture %.1f fps on this machine, %d decisions",
             result.frames, 100 * result.detection_rate, result.analysis_fps, len(result.decisions))
    log.info("wrote %s", outputs.dir)


def main(args: argparse.Namespace) -> int:
    """Entry point for `python -m ahjia demo`. Returns the process exit code.

    Any failure (bad option, missing file, model or ffmpeg error) is logged as one
    clear message, the files this run started to write are removed, and the exit
    code is 2. Ctrl-C also removes them and then propagates.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    outputs = _Outputs(Path(args.out))
    try:
        _run_demo(args, outputs)
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report every failure the same way
        known = isinstance(exc, (OSError, ValueError, RuntimeError))
        log.error("demo failed: %s", exc if known else f"{type(exc).__name__}: {exc}")
        log.debug("traceback", exc_info=True)
        outputs.discard()
        return 2
    except BaseException:
        outputs.discard()
        raise
    return 0
