"""Hailo-10H (ASUS UGen300) residency, time-sharing and power simulator.

Stage I has no UGen300, so this module stands in for it. Every number comes
from specs/hailo10h.yaml, where each value is tagged official, estimate or
assumption with a source. The model is deliberately simple and transparent:

Residency
    Vision models (person detector, pose, SCRFD, ArcFace) and Whisper-Base are
    loaded at boot and stay resident (P:114, P:386-392). At most one generative
    LLM (Qwen2.5-1.5B) is resident; loading it takes models.llm.load_s
    (5.17 s, official). Loading or unloading never touches the vision models.

NPU time
    The NPU runs one network at a time (HailoRT scheduler time-slicing), so it
    offers npu_capacity = 1 NPU-second per second. Demands, in NPU-s per s:
      vision  = sum over cameras of fps x per-frame cost, where the per-frame
                cost is 1 / FPS(batch 1) of each model the pipeline runs
                (conservative: batching across cameras could do better);
      whisper = 1 while a speech segment is being decoded (tokens / tps);
      llm     = 1 while the LLM decodes a reply (TTFT + tokens / tps).
    With safety-first QoS (S2) vision and Whisper are served first and the LLM
    gets what is left, so only the LLM slows down. Without QoS (S0, S1) all
    demands share the NPU in proportion, so an overloaded NPU drops frames and
    slows the LLM too, and an LLM load blocks the NPU until it ends.

Contention seen by the safety path (no QoS only; S2 waits contention.qos_wait_s)
    LLM loading  -> a safety frame or speech segment waits for the load to end;
    LLM decoding -> each vision frame waits one LLM step (1 / tps) and each
                    Whisper token waits one step (round-robin at step level).

Power (KPI 1, P:443)
    P = sum over states of (time share x state power) with states idle, vision,
    whisper, llm (decode) and load; time shares come from the allocation above.
    Anchors: UGen300 typical 2.5 W (used for CNN work), Qwen2-1.5B inference
    average 2.1 W (both official); idle and load power are assumptions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from ahjia.events import GenAIState

KINDS: tuple[str, ...] = ("official", "estimate", "assumption")
DEFAULT_SPEC = "specs/hailo10h.yaml"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MAX_UTIL_FOR_WAIT = 0.95
MAX_CAMERAS = 64
EPS = 1e-9

Profile = Literal["all_models", "fixed", "cascade"]


# ---------------------------------------------------------------- spec file

@dataclass(frozen=True)
class Param:
    value: float
    kind: str
    source: str


@dataclass(frozen=True)
class HailoSpec:
    params: Mapping[str, Param]
    names: Mapping[str, str] = field(default_factory=dict)
    path: str = ""

    def param(self, key: str) -> Param:
        if key not in self.params:
            raise KeyError(f"spec has no value {key!r}")
        return self.params[key]

    def value(self, key: str) -> float:
        return self.param(key).value

    def name(self, key: str) -> str:
        return self.names.get(key, key)

    def cost_s(self, model: str) -> float:
        """NPU seconds per inference = 1 / FPS(batch 1)."""
        return 1.0 / self.value(f"models.{model}.fps")


def resolve_spec_path(path: str | Path | None) -> Path:
    candidate = Path(path or DEFAULT_SPEC)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PACKAGE_ROOT / candidate


def _flatten(node: Mapping[str, Any], prefix: str, params: dict[str, Param], names: dict[str, str]) -> None:
    for key, child in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(child, Mapping) and "value" in child:
            kind, source = child.get("kind"), str(child.get("source", "")).strip()
            if kind not in KINDS:
                raise ValueError(f"{dotted}: kind must be one of {KINDS}, got {kind!r}")
            if not source:
                raise ValueError(f"{dotted}: source is required")
            params[dotted] = Param(float(child["value"]), str(kind), source)
        elif isinstance(child, Mapping):
            _flatten(child, f"{dotted}.", params, names)
        elif key == "name":
            names[prefix.rstrip(".")] = str(child)


def load_spec(path: str | Path | None = None) -> HailoSpec:
    resolved = resolve_spec_path(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    params: dict[str, Param] = {}
    names: dict[str, str] = {}
    _flatten(data, "", params, names)
    return HailoSpec(params=params, names=names, path=str(resolved))


# ---------------------------------------------------------------- per-frame cost

def frame_cost_s(spec: HailoSpec, profile: Profile, n_persons: int,
                 full_res_pose: bool = False, face_needed: bool = False) -> float:
    """NPU seconds for one frame of one camera.

    all_models (S0): detector + full pose + SCRFD + ArcFace on every frame.
    fixed (S1):      single-stage pose on the full frame every frame; face models
                     whenever someone is in view.
    cascade (S2):    detector every frame; pose only on person crops (full
                     resolution under A3); face models only when needed.
    """
    det, pose, scrfd, arc = (spec.cost_s(m) for m in ("person_detect", "pose", "face_detect", "face_embed"))
    if profile == "all_models":
        return det + pose + scrfd + arc * max(1, n_persons)
    if profile == "fixed":
        return pose + ((scrfd + arc * n_persons) if n_persons else 0.0)
    crop = spec.value("models.pose.crop_cost_factor")
    cost = det
    if n_persons:
        cost += pose if full_res_pose else crop * pose * n_persons
        if face_needed:
            cost += scrfd + arc * n_persons
    return cost


def md1_wait_s(service_s: float, utilization: float) -> float:
    """Mean queueing delay of an M/D/1 queue: rho / (2 (1 - rho)) x service."""
    rho = min(max(utilization, 0.0), MAX_UTIL_FOR_WAIT)
    return service_s * rho / (2.0 * (1.0 - rho))


def whisper_tokens(spec: HailoSpec, text: str) -> float:
    cjk = len(re.findall(r"[㐀-鿿]", text))
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    return (cjk * spec.value("pipeline.whisper_tokens_per_cjk_char")
            + words * spec.value("pipeline.whisper_tokens_per_word")
            + spec.value("pipeline.whisper_overhead_tokens"))


def cameras_supported(budget: float, a3_load: float, busy_load: float, extra_load: float,
                      busy_cameras: int) -> int:
    """KPI 2 (NPU bound): cameras that fit in the NPU budget at the busiest moment.

    Camera 1 is under A3 (an alert), the next busy_cameras - 1 have people in
    high-risk zones, and every extra camera adds extra_load (KPI 2 passes the
    scenario's average NPU demand per camera, i.e. the same occupancy mix).
    """
    used, count = 0.0, 0
    while count < MAX_CAMERAS:
        load = a3_load if count == 0 else busy_load if count < busy_cameras else extra_load
        if used + load > budget + EPS:
            break
        used += load
        count += 1
    return count


# ---------------------------------------------------------------- NPU sharing and power

@dataclass(frozen=True)
class Demand:
    vision: float = 0.0
    whisper: float = 0.0
    llm: float = 0.0
    loading: bool = False
    llm_resident: bool = False


@dataclass(frozen=True)
class Shares:
    vision: float = 0.0
    whisper: float = 0.0
    llm: float = 0.0
    load: float = 0.0
    idle: float = 1.0
    vision_service: float = 1.0
    llm_service: float = 1.0


def _ratio(served: float, wanted: float) -> float:
    return 1.0 if wanted <= EPS else min(1.0, served / wanted)


def allocate(demand: Demand, qos: bool, spec: HailoSpec) -> Shares:
    cap = spec.value("device.npu_capacity")
    if demand.loading and not qos and spec.value("contention.no_qos_load_blocks_vision"):
        unblocked = allocate(replace(demand, loading=False), qos, spec)
        return Shares(load=cap, idle=0.0, vision_service=unblocked.vision_service, llm_service=0.0)
    if qos:
        safety = demand.vision + demand.whisper
        f = _ratio(min(safety, cap), safety)
        vision, whisper = demand.vision * f, demand.whisper * f
        llm = min(demand.llm, cap - vision - whisper)
    else:
        total = demand.vision + demand.whisper + demand.llm
        f = _ratio(cap, total)
        vision, whisper, llm = demand.vision * f, demand.whisper * f, demand.llm * f
    rest = max(0.0, cap - vision - whisper - llm)
    load = rest if demand.loading else 0.0
    return Shares(vision=vision, whisper=whisper, llm=llm, load=load, idle=rest - load,
                  vision_service=_ratio(vision, demand.vision), llm_service=_ratio(llm, demand.llm))


def power_w(shares: Shares, spec: HailoSpec, llm_resident: bool = False) -> float:
    p = spec.value
    watts = (shares.idle * p("power_w.idle") + shares.vision * p("power_w.vision")
             + shares.whisper * p("power_w.whisper") + shares.llm * p("power_w.llm_decode")
             + shares.load * p("power_w.llm_load"))
    return watts + (p("power_w.llm_resident_extra") if llm_resident else 0.0)


@dataclass(frozen=True)
class MeterTotals:
    duration_s: float
    energy_j: float
    state_s: Mapping[str, float]
    peak_vision_demand: float
    overload_s: float
    vision_demand_s: float = 0.0  # integral of the vision demand (NPU-s asked for, served or not)

    @property
    def avg_power_w(self) -> float:
        return self.energy_j / self.duration_s if self.duration_s > 0 else 0.0


class NpuMeter:
    """Integrates power and NPU state time over piecewise-constant demand."""

    STATES = ("idle", "vision", "whisper", "llm", "load")

    def __init__(self, spec: HailoSpec, qos: bool, t0: float = 0.0) -> None:
        self._spec, self._qos = spec, qos
        self._t, self._demand = t0, Demand()
        self._t0 = t0
        self._energy = 0.0
        self._state = dict.fromkeys(self.STATES, 0.0)
        self._peak, self._overload, self._vision_demand = 0.0, 0.0, 0.0

    def _integrate(self, t: float) -> None:
        dt = t - self._t
        if dt <= 0:
            return
        shares = allocate(self._demand, self._qos, self._spec)
        self._energy += power_w(shares, self._spec, self._demand.llm_resident) * dt
        for name in self.STATES:
            self._state[name] += getattr(shares, name) * dt
        self._vision_demand += self._demand.vision * dt
        total = self._demand.vision + self._demand.whisper + self._demand.llm
        if total > self._spec.value("device.npu_capacity") + EPS:
            self._overload += dt
        self._t = t

    def set_demand(self, t: float, demand: Demand) -> None:
        self._integrate(t)
        self._demand = demand
        self._peak = max(self._peak, demand.vision)

    @property
    def demand(self) -> Demand:
        return self._demand

    def totals(self, t_end: float) -> MeterTotals:
        self._integrate(t_end)
        return MeterTotals(duration_s=self._t - self._t0, energy_j=self._energy, state_s=dict(self._state),
                           peak_vision_demand=self._peak, overload_s=self._overload,
                           vision_demand_s=self._vision_demand)


# ---------------------------------------------------------------- the simulated device

@dataclass(frozen=True)
class LlmJob:
    ready: float
    start: float
    first_audio: float
    end: float
    service: float


class Hailo10HSim:
    """One UGen300: resident models, LLM load/decode timeline, NPU shares, power."""

    def __init__(self, spec: HailoSpec, qos: bool) -> None:
        self.spec, self.qos = spec, qos
        self._meter = NpuMeter(spec, qos)
        self._vision = 0.0
        self._resident = False
        self._load_end: float | None = None
        self._jobs: tuple[LlmJob, ...] = ()
        self._whisper_until = 0.0
        self._now = 0.0
        self.llm_jobs = 0
        self.llm_loads = 0

    # ---- state queries
    @property
    def llm_resident(self) -> bool:
        return self._resident

    def loading(self, t: float) -> bool:
        return self._load_end is not None and t < self._load_end - EPS

    def decoding(self, t: float) -> bool:
        return any(j.start - EPS <= t < j.end - EPS for j in self._jobs)

    def genai_state(self, t: float) -> GenAIState:
        if self.loading(t):
            return "loading"
        return "npu_window" if self._resident else "off"

    def _demand(self, t: float) -> Demand:
        return Demand(vision=self._vision, whisper=1.0 if t < self._whisper_until - EPS else 0.0,
                      llm=1.0 if self.decoding(t) else 0.0, loading=self.loading(t),
                      llm_resident=self._resident)

    def shares(self, t: float | None = None) -> Shares:
        return allocate(self._demand(self._now if t is None else t), self.qos, self.spec)

    def vision_service(self) -> float:
        return self.shares().vision_service

    def utilization(self) -> float:
        s = self.shares()
        return s.vision + s.whisper + s.llm

    # ---- time
    def _refresh(self) -> None:
        self._meter.set_demand(self._now, self._demand(self._now))

    def advance(self, t: float) -> None:
        """Move to time t, crossing load / decode / Whisper boundaries in order."""
        while True:
            event = self.next_event(self._now)
            if event is None or event > t + EPS:
                break
            self._now = max(self._now, event)
            if self._load_end is not None and self._now >= self._load_end - EPS:
                self._resident, self._load_end = True, None
            self._jobs = tuple(j for j in self._jobs if j.end > self._now + EPS)
            self._refresh()
        self._now = max(self._now, t)
        self._refresh()

    def next_event(self, t: float) -> float | None:
        times = [self._load_end, self._whisper_until] + [x for j in self._jobs for x in (j.start, j.end)]
        future = [x for x in times if x is not None and x > t + EPS]
        return min(future) if future else None

    def set_vision_demand(self, t: float, vision: float) -> None:
        self.advance(t)
        self._vision = vision
        self._refresh()

    # ---- LLM lifecycle
    def start_load(self, t: float, warm: bool = False) -> float:
        """Begin loading the LLM; returns when it is resident. warm=True means it
        was already resident before the run started (steady state)."""
        self.advance(t)
        if self._resident:
            return t
        if self._load_end is not None:
            return self._load_end
        if warm:
            self._resident = True
        else:
            self._load_end = t + self.spec.value("models.llm.load_s")
            self.llm_loads += 1
        self._refresh()
        return t if warm else self._load_end

    def unload(self, t: float) -> None:
        self.advance(t)
        self._resident, self._load_end, self._jobs = False, None, ()
        self._refresh()

    def start_llm_job(self, ready: float) -> LlmJob:
        if not self._resident and self._load_end is None:
            raise RuntimeError("LLM is not loaded")
        busy = max([ready] + [j.end for j in self._jobs] + ([self._load_end] if self._load_end is not None else []))
        service = allocate(replace(self._demand(busy), llm=1.0, loading=False), self.qos, self.spec).llm_service
        service = max(service, 0.05)
        v = self.spec.value
        first = (v("models.llm.ttft_s") + v("pipeline.llm_first_sentence_tokens") / v("models.llm.tps")) / service
        work = (v("models.llm.ttft_s") + v("pipeline.llm_reply_tokens") / v("models.llm.tps")) / service
        job = LlmJob(ready=ready, start=busy, first_audio=busy + first + v("pipeline.tts_first_audio_s"),
                     end=busy + work, service=service)
        self._jobs = self._jobs + (job,)
        self.llm_jobs += 1
        self._refresh()
        return job

    # ---- speech and contention
    def genai_wait(self, t: float, whisper_tokens: float = 0.0) -> float:
        """Extra wait the safety path sees because of the LLM at time t.

        With safety-first QoS (S2) this is NOT computed: it returns the spec's assumption
        contention.qos_wait_s, so KPI 4 marks S2's count "assumption: qos_wait_s"."""
        if self.qos:
            return self.spec.value("contention.qos_wait_s")
        if self.loading(t) and self.spec.value("contention.no_qos_load_blocks_vision"):
            return self._load_end - t
        if self.decoding(t):
            step = self.spec.value("contention.no_qos_wait_steps") / self.spec.value("models.llm.tps")
            return step * (whisper_tokens if whisper_tokens > 0 else 1.0)
        return 0.0

    def start_whisper(self, t: float, text: str) -> float:
        """Decode one speech segment; returns the time from segment end to transcript."""
        self.advance(t)
        tokens = whisper_tokens(self.spec, text)
        busy = tokens / self.spec.value("models.whisper.tps")
        wait = self.genai_wait(t, whisper_tokens=tokens)
        self._whisper_until = max(self._whisper_until, t + wait + busy)
        self._refresh()
        return self.spec.value("pipeline.vad_tail_s") + wait + busy

    def totals(self, t_end: float) -> MeterTotals:
        self.advance(t_end)
        return self._meter.totals(t_end)
