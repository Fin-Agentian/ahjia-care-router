"""The five KPIs of P:433-443, computed from one replay (no number is typed by hand).

1. Average power = sum of (time share x state power) over the run (residency.py).
2. Cameras per UGen300 while KPI 3 still holds, an estimate (NPU + host decode
   assumption) = min(NPU bound, host.max_streams). NPU bound: cameras that fit
   in the NPU budget (capacity x max_utilization) at the busiest moment: one
   camera under A3, the other busy cameras in high-risk zones, and every added
   camera with the scenario's average NPU demand per camera (same occupancy and
   risk mix as the existing cameras). USB bandwidth is not modelled.
3. Event -> response: from the true event time (label; for a fall, 2 s after
   lying starts, because the rule needs 2 s) to the voice (A1), the family text
   (A4) and the call (A5); talk = directed speech -> reply; greeting = entry ->
   reply.
4. GenAI interference: safety decisions that waited for the LLM (target 0);
   LLM wakes (A2 jobs) per hour; reply rate to directed speech (A1 or A2).
   delayed_safety_basis says where the count comes from: "computed" (no QoS:
   the simulator models the LLM contention) or "assumption: qos_wait_s" (with
   safety-first QoS the wait IS the spec's assumption, so the count follows
   from it; Stage II measures it).
5. High-risk misses: high_risk / inactivity labels with no pre-alert (A4 or A5)
   inside the label window. Guard metric: false alarms = pre-alerts that match
   no risk label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

from ahjia.engine import Response
from ahjia.replay import Label, RunResult
from ahjia.residency import HailoSpec, cameras_supported, load_spec
from ahjia.router import camera_load
from ahjia.strategies import get_policy

FALL_CONFIRM_S = 2.0  # P:144: lying >= 2 s outside bed/sofa
RISK_KINDS = frozenset({"high_risk", "inactivity", "sos_immediate", "no_answer"})
ALERT_KINDS = frozenset({"trigger", "notify"})
COMPUTED_BASIS = "computed"
QOS_ASSUMPTION_BASIS = "assumption: qos_wait_s"
EPS = 1e-6


def percentile(samples: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(samples, dtype=float), q)) if len(samples) else None


@dataclass(frozen=True)
class KpiRow:
    scenario: str
    group: str
    strategy: str
    duration_s: float
    avg_power_w: float | None
    cameras_supported: int | None
    cameras_npu: int | None = None
    cameras_host: int | None = None
    voice_samples: tuple[float, ...] = ()
    text_samples: tuple[float, ...] = ()
    talk_samples: tuple[float, ...] = ()
    a5_immediate_s: float | None = None
    a5_no_answer_s: float | None = None
    greeting_s: float | None = None
    delayed_safety: int = 0
    delayed_safety_basis: str = COMPUTED_BASIS
    llm_wakes: int = 0
    directed: int = 0
    replied: int = 0
    high_risk_labels: int = 0
    misses: int = 0
    false_alarms: int = 0
    frames: int = 0

    @property
    def voice_p50(self) -> float | None:
        return percentile(self.voice_samples, 50)

    @property
    def voice_p95(self) -> float | None:
        return percentile(self.voice_samples, 95)

    @property
    def text_p50(self) -> float | None:
        return percentile(self.text_samples, 50)

    @property
    def text_p95(self) -> float | None:
        return percentile(self.text_samples, 95)

    @property
    def talk_p50(self) -> float | None:
        return percentile(self.talk_samples, 50)

    @property
    def llm_wakes_per_h(self) -> float:
        return self.llm_wakes / (self.duration_s / 3600.0) if self.duration_s > 0 else 0.0

    @property
    def reply_rate(self) -> float | None:
        return self.replied / self.directed if self.directed else None


def _ref(label: Label) -> float:
    return label.t + (FALL_CONFIRM_S if "fall" in label.what.lower() else 0.0)


def _first(responses: Sequence[Response], actions: frozenset[str], start: float, end: float,
           safety_only: bool = True) -> Response | None:
    found = [r for r in responses if r.action in actions and start - EPS <= r.t_response
             and r.t_event <= end + EPS and (r.safety or not safety_only)]
    return min(found, key=lambda r: r.t_response) if found else None


def _latencies(result: RunResult) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"voice": [], "text": [], "a5": [], "no_answer": [], "greeting": []}
    for label in result.scenario.labels:
        ref = _ref(label)
        if label.kind == "high_risk":
            for key, action in (("voice", "A1"), ("text", "A4")):
                hit = _first(result.responses, frozenset({action}), ref, ref + label.window_s)
                if hit is not None:
                    out[key].append(hit.t_response - ref)
        elif label.kind in ("sos_immediate", "no_answer"):
            key = "a5" if label.kind == "sos_immediate" else "no_answer"
            hit = _first(result.responses, frozenset({"A5"}), label.t, label.t + label.window_s)
            if hit is not None:
                out[key].append(hit.t_response - label.t)
        elif label.kind == "greeting":
            hit = _first(result.responses, frozenset({"A1", "A2"}), label.t, label.t + label.window_s,
                         safety_only=False)
            if hit is not None and hit.agent == "companion":
                out["greeting"].append(hit.t_response - label.t)
    return out


def _misses(result: RunResult) -> tuple[int, int]:
    labels = [lb for lb in result.scenario.labels if lb.kind in ("high_risk", "inactivity")]
    alerts = [r for r in result.responses if r.safety and r.action in ("A4", "A5")]
    misses = sum(1 for lb in labels
                 if not any(lb.t - EPS <= r.t_response and r.t_event <= _ref(lb) + lb.window_s + EPS for r in alerts))
    return len(labels), misses


def _false_alarms(result: RunResult) -> int:
    risk = [(lb.t - 1.0, _ref(lb) + lb.window_s) for lb in result.scenario.labels if lb.kind in RISK_KINDS]
    raised = {(r.t_decision, r.agent) for r in result.responses
              if r.safety and r.action == "A4" and r.kind in ALERT_KINDS}
    return sum(1 for t, _ in raised if not any(a - EPS <= t <= b + EPS for a, b in risk))


def _talk(result: RunResult) -> tuple[int, int, list[float]]:
    directed = result.scenario.directed_speech
    samples: list[float] = []
    for event in directed:
        replies = [r for r in result.responses if r.agent == "companion" and r.action in ("A1", "A2")
                   and abs(r.t_event - event.t) < EPS]
        if replies:
            samples.append(min(r.t_response for r in replies) - event.t)
    return len(directed), len(samples), samples


def _cameras(result: RunResult, spec: HailoSpec) -> tuple[int, int]:
    """(NPU bound, host decode bound) for KPI 2."""
    policy = get_policy(result.strategy)
    full = spec.value("device.camera_full_fps")
    budget = spec.value("device.npu_capacity") * spec.value("device.max_utilization")
    camera_seconds = result.totals.duration_s * max(1, len(result.scenario.cameras))
    average = result.totals.vision_demand_s / camera_seconds if camera_seconds > 0 else 0.0
    npu = cameras_supported(budget, a3_load=camera_load(policy, spec, "a3", full),
                            busy_load=camera_load(policy, spec, "high_risk", full),
                            extra_load=average, busy_cameras=max(1, result.max_busy_cameras))
    return npu, int(spec.value("host.max_streams"))


def compute_kpis(result: RunResult, spec: HailoSpec | None = None) -> KpiRow:
    spec = spec or load_spec()
    lat = _latencies(result)
    labels, misses = _misses(result)
    directed, replied, talk = _talk(result)
    delayed = {(r.t_decision, r.agent) for r in result.responses if r.safety and r.genai_wait_s > EPS}
    npu, host = _cameras(result, spec)
    basis = QOS_ASSUMPTION_BASIS if get_policy(result.strategy).qos else COMPUTED_BASIS
    return KpiRow(
        scenario=result.scenario.name, group=result.scenario.group, strategy=result.strategy,
        duration_s=result.scenario.duration_s, avg_power_w=result.totals.avg_power_w,
        cameras_supported=min(npu, host), cameras_npu=npu, cameras_host=host, voice_samples=tuple(lat["voice"]),
        text_samples=tuple(lat["text"]), talk_samples=tuple(talk),
        a5_immediate_s=lat["a5"][0] if lat["a5"] else None,
        a5_no_answer_s=lat["no_answer"][0] if lat["no_answer"] else None,
        greeting_s=lat["greeting"][0] if lat["greeting"] else None,
        delayed_safety=len(delayed), delayed_safety_basis=basis, llm_wakes=result.llm_jobs,
        directed=directed, replied=replied,
        high_risk_labels=labels, misses=misses, false_alarms=_false_alarms(result), frames=result.frames)


def aggregate(rows: Sequence[KpiRow], scenario: str) -> KpiRow:
    """Pool several runs of one strategy (e.g. the scripted set) into one row."""
    strategies = {r.strategy for r in rows}
    if len(strategies) != 1:
        raise ValueError(f"aggregate needs one strategy, got {sorted(strategies)}")
    pooled = {f.name: sum((getattr(r, f.name) for r in rows), ())
              for f in fields(KpiRow) if f.name.endswith("_samples")}
    counts = {name: sum(getattr(r, name) for r in rows)
              for name in ("delayed_safety", "llm_wakes", "directed", "replied", "high_risk_labels",
                           "misses", "false_alarms", "frames")}
    basis = next((r.delayed_safety_basis for r in rows if r.delayed_safety_basis != COMPUTED_BASIS), COMPUTED_BASIS)
    base = KpiRow(scenario=scenario, group="aggregate", strategy=rows[0].strategy,
                  duration_s=sum(r.duration_s for r in rows), avg_power_w=None, cameras_supported=None,
                  delayed_safety_basis=basis)
    return replace(base, **pooled, **counts)
