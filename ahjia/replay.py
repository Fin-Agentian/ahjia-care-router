"""`python -m ahjia replay`: scenario files -> S0/S1/S2 engines -> decision logs + KPI report.

A scenario (scenarios/*.yaml, format in DESIGN.md) gives the TRUE state of the
world: who each camera sees, what is said, app confirmations. The replay
samples that truth at the frame rate each strategy really processes on each
camera (engine.processed_fps: the router's plan, reduced when the NPU is
overloaded), so a lower frame rate means later detection and lower power.

Frames are event-driven: a camera produces a frame at the first point of its
frame grid after the true state changed, and at the first grid point after an
engine deadline that needs a frame (for example "lying >= 2 s"). Frames whose
content equals the previous frame and at which no rule can fire are skipped;
with noiseless simulated perception this gives the same decisions as feeding
every frame. NPU time and power still count every frame (residency.py
integrates fps x per-frame cost over time).

Scenario extras: t may be seconds or a wall clock "HH:MM:SS"; an event may
carry repeat: {every_s, until}; optional fields time_scale, usual_wake_clock,
synthetic. Label kinds: high_risk, sos_immediate, no_answer, inactivity,
greeting, benign (optional window_s, default 10 s).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ahjia.agents.base import DAY_S, parse_clock
from ahjia.engine import CareEngine, Response, make_engine
from ahjia.events import Decision, Observation, PersonObs, RunConfig, SpeechObs, StrategyName, decision_to_json
from ahjia.residency import MeterTotals
from ahjia.strategies import parse_strategies

log = logging.getLogger(__name__)

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
LABEL_KINDS = ("high_risk", "sos_immediate", "no_answer", "inactivity", "greeting", "benign")
DEFAULT_WINDOW_S = 10.0
GRID_EPS = 1e-6


@dataclass(frozen=True)
class Label:
    t: float
    kind: str
    what: str
    window_s: float = DEFAULT_WINDOW_S


@dataclass(frozen=True)
class WorldEvent:
    t: float
    camera: str | None = None
    persons: tuple[PersonObs, ...] | None = None
    speech: SpeechObs | None = None
    app_confirm: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    source: str
    start_clock: str
    cameras: tuple[str, ...]
    duration_s: float
    labels: tuple[Label, ...]
    events: tuple[WorldEvent, ...]
    elder_alone: bool = True
    time_scale: float = 1.0
    usual_wake_clock: str = "06:30:00"
    synthetic: bool = False
    group: str = "demo"

    def run_config(self, spec_path: str | None = None) -> RunConfig:
        extra = {"spec_path": spec_path} if spec_path else {}
        return RunConfig(start_clock=self.start_clock, cameras=self.cameras, elder_alone=self.elder_alone,
                         time_scale=self.time_scale, usual_wake_clock=self.usual_wake_clock, **extra)

    @property
    def directed_speech(self) -> tuple[WorldEvent, ...]:
        return tuple(e for e in self.events if e.speech is not None and e.speech.directed)


@dataclass(frozen=True)
class RunResult:
    scenario: Scenario
    strategy: StrategyName
    decisions: tuple[Decision, ...]
    responses: tuple[Response, ...]
    totals: MeterTotals
    llm_jobs: int
    llm_loads: int
    max_busy_cameras: int
    frames: int


# ---------------------------------------------------------------- loading

def _time(value: Any, start_s: int) -> float:
    if isinstance(value, str) and ":" in value:
        return float((parse_clock(value) - start_s) % DAY_S)
    return float(value)


def _person(raw: Mapping[str, Any]) -> PersonObs:
    return PersonObs(track_id=int(raw["track_id"]), identity=raw.get("identity", "unknown"),
                     id_conf=float(raw.get("id_conf", 1.0)), posture=raw.get("posture", "unknown"),
                     zone=raw.get("zone", "other"))


def _event(raw: Mapping[str, Any], t: float) -> WorldEvent:
    if "camera" in raw:
        return WorldEvent(t=t, camera=str(raw["camera"]), persons=tuple(_person(p) for p in raw.get("persons") or ()))
    if "speech" in raw:
        sp = raw["speech"]
        return WorldEvent(t=t, speech=SpeechObs(text=str(sp["text"]), source=sp.get("source", "elder"),
                                                directed=bool(sp.get("directed", False)), lang=sp.get("lang", "en")))
    if raw.get("app_confirm"):
        return WorldEvent(t=t, app_confirm=True)
    raise ValueError(f"event at t={t} has no camera, speech or app_confirm")


def _expand(raw: Mapping[str, Any], start_s: int) -> list[WorldEvent]:
    t0 = _time(raw["t"], start_s)
    repeat = raw.get("repeat")
    if not repeat:
        return [_event(raw, t0)]
    every, until = float(repeat["every_s"]), _time(repeat["until"], start_s)
    if every <= 0:
        raise ValueError("repeat.every_s must be > 0")
    count = int(math.floor((until - t0) / every + 1e-9)) + 1
    return [_event(raw, t0 + i * every) for i in range(count)]


def _validate(scn: Scenario) -> None:
    for e in scn.events:
        if not 0 <= e.t <= scn.duration_s:
            raise ValueError(f"{scn.name}: event at t={e.t} is outside the duration {scn.duration_s}")
        if e.camera is not None and e.camera not in scn.cameras:
            raise ValueError(f"{scn.name}: unknown camera {e.camera!r}")
    for label in scn.labels:
        if label.kind not in LABEL_KINDS:
            raise ValueError(f"{scn.name}: label kind must be one of {LABEL_KINDS}, got {label.kind!r}")


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    start = str(data.get("start_clock", "00:00:00"))
    start_s = parse_clock(start)
    events = [e for raw in data.get("events") or () for e in _expand(raw, start_s)]
    labels = tuple(Label(t=_time(raw["t"], start_s), kind=str(raw["kind"]), what=str(raw.get("what", "")),
                         window_s=float(raw.get("window_s", DEFAULT_WINDOW_S))) for raw in data.get("labels") or ())
    group = "scripted" if path.parent.name == "scripted" else ("day" if data.get("synthetic") else "demo")
    scn = Scenario(
        name=str(data.get("name", path.stem)), title=str(data.get("title", path.stem)),
        source=str(data.get("source", "")), start_clock=start, cameras=tuple(data.get("cameras") or ()),
        duration_s=float(data.get("duration_s", 60)), labels=labels,
        events=tuple(sorted(events, key=lambda e: e.t)), elder_alone=bool(data.get("elder_alone", True)),
        time_scale=float(data.get("time_scale", 1.0)), usual_wake_clock=str(data.get("usual_wake_clock", "06:30:00")),
        synthetic=bool(data.get("synthetic", False)), group=group)
    _validate(scn)
    return scn


def find_scenarios(name: str | None, all_: bool, root: Path = SCENARIO_DIR) -> list[Path]:
    if all_:
        return sorted(root.glob("*.yaml")) + sorted((root / "scripted").glob("*.yaml"))
    for candidate in (root / f"{name}.yaml", root / "scripted" / f"{name}.yaml"):
        if candidate.exists():
            return [candidate]
    raise FileNotFoundError(f"no scenario named {name!r} in {root}")


def max_busy_cameras(scn: Scenario) -> int:
    """Most cameras that see someone at the same time (after simultaneous moves)."""
    truth = {cam: False for cam in scn.cameras}
    best, i, events = 0, 0, scn.events
    while i < len(events):
        t = events[i].t
        while i < len(events) and events[i].t == t:
            if events[i].camera is not None:
                truth[events[i].camera] = bool(events[i].persons)
            i += 1
        best = max(best, sum(truth.values()))
    return best


# ---------------------------------------------------------------- sampling loop

def _grid_next(anchor: float, fps: float, at_least: float, after: float) -> float:
    """First frame time anchor + k / fps that is >= at_least and > after."""
    if fps <= 0:
        return math.inf
    k = max(0, math.ceil((at_least - anchor) * fps - GRID_EPS))
    t = anchor + k / fps
    while t <= after + GRID_EPS:
        k += 1
        t = anchor + k / fps
    return t


class _Sampler:
    """True world state per camera plus each camera's frame grid."""

    def __init__(self, engine: CareEngine, cameras: Sequence[str]) -> None:
        self.engine = engine
        self.truth: dict[str, tuple[PersonObs, ...]] = {c: () for c in cameras}
        self.seen: dict[str, tuple[PersonObs, ...] | None] = {c: None for c in cameras}
        self.pending: dict[str, float | None] = {c: 0.0 for c in cameras}
        self.fps = {c: engine.processed_fps(c) for c in cameras}
        self.anchor = {c: 0.0 for c in cameras}
        self.last = {c: -math.inf for c in cameras}
        self.frames = 0

    def next_frame(self) -> tuple[float, str | None]:
        best: tuple[float, str | None] = (math.inf, None)
        for cam, since in self.pending.items():
            if since is not None:
                t = _grid_next(self.anchor[cam], self.fps[cam], since, self.last[cam])
                best = min(best, (t, cam), key=lambda x: x[0])
        return best

    def deadline_frame(self, t: float, cam: str) -> float:
        if cam not in self.fps:
            return math.inf
        return _grid_next(self.anchor[cam], self.fps[cam], t, self.last[cam])

    def world(self, event: WorldEvent) -> None:
        cam = event.camera
        self.truth[cam] = event.persons or ()
        if self.truth[cam] == self.seen[cam]:
            self.pending[cam] = None
        elif self.pending[cam] is None:
            self.pending[cam] = event.t

    def frame(self, t: float, cam: str) -> tuple[Decision, ...]:
        self.frames += 1
        self.seen[cam], self.last[cam], self.pending[cam] = self.truth[cam], t, None
        return self.engine.observe(Observation(t=t, camera=cam, persons=self.truth[cam]))

    def regrid(self, now: float) -> None:
        for cam in self.fps:
            fps = self.engine.processed_fps(cam)
            if abs(fps - self.fps[cam]) > 1e-9:
                self.fps[cam], self.anchor[cam] = fps, now


def run_scenario(scn: Scenario, strategy: StrategyName, spec_path: str | None = None) -> RunResult:
    engine = make_engine(strategy, scn.run_config(spec_path))
    sampler = _Sampler(engine, scn.cameras)
    events, i = scn.events, 0
    while True:
        t_event = events[i].t if i < len(events) else math.inf
        t_frame, cam = sampler.next_frame()
        deadline = engine.next_deadline()
        t_dead, dead_cam = math.inf, None
        if deadline is not None:
            dead_cam = deadline[1]
            t_dead = sampler.deadline_frame(deadline[0], dead_cam) if dead_cam else deadline[0]
        t_next = min(t_event, t_frame, t_dead)
        if t_next > scn.duration_s:
            break
        if t_event <= t_next:
            event = events[i]
            i += 1
            if event.camera is not None:
                sampler.world(event)
            else:
                engine.observe(Observation(t=event.t, speech=event.speech, app_confirm=event.app_confirm))
        elif t_frame <= t_next:
            sampler.frame(t_frame, cam)
        elif dead_cam is not None:
            sampler.frame(t_dead, dead_cam)
        else:
            engine.advance(t_dead)
        sampler.regrid(t_next)
    engine.advance(scn.duration_s)
    return RunResult(scenario=scn, strategy=strategy, decisions=engine.decisions, responses=engine.responses,
                     totals=engine.totals(scn.duration_s), llm_jobs=engine.sim.llm_jobs,
                     llm_loads=engine.sim.llm_loads, max_busy_cameras=max_busy_cameras(scn), frames=sampler.frames)


# ---------------------------------------------------------------- CLI

def write_jsonl(decisions: Sequence[Decision], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(decision_to_json(d) + "\n" for d in decisions), encoding="utf-8")


RESPONSE_ACTIONS = frozenset({"A1", "A2", "A4", "A5"})


def write_responses(responses: Sequence[Response], path: Path) -> None:
    """One line per response action (voice, LLM reply, family text, call) with its simulated timing."""
    lines = []
    for r in responses:
        if r.action not in RESPONSE_ACTIONS:
            continue
        record = {"t_event": round(r.t_event, 3), "agent": r.agent, "action": r.action,
                  "t_done": round(r.t_response, 3), "latency_s": round(r.t_response - r.t_event, 3),
                  "kind": r.kind, "genai_wait_s": round(r.genai_wait_s, 3), "simulated": True}
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    from ahjia.kpi import compute_kpis
    from ahjia.report import write_report

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        strategies = parse_strategies(args.strategies)
        paths = find_scenarios(args.scenario, bool(args.all))
    except (ValueError, FileNotFoundError) as err:
        log.error("%s", err)
        return 2
    out = Path(args.out)
    rows = []
    for path in paths:
        scn = load_scenario(path)
        for strategy in strategies:
            result = run_scenario(scn, strategy)
            write_jsonl(result.decisions, out / "replay" / scn.name / f"{strategy}.jsonl")
            write_responses(result.responses, out / "replay" / scn.name / f"{strategy}.responses.jsonl")
            rows.append(compute_kpis(result))
            log.info("replayed %s under %s: %d decisions", scn.name, strategy, len(result.decisions))
    write_report(rows, out)
    sys.stdout.write(f"wrote {out / 'report.md'}, kpi.csv, kpi.json, kpi.png "
                     f"({len(rows)} runs; timing and power simulated)\n")
    return 0
