"""Write outputs/report.md, kpi.csv, kpi.json and kpi.png from KPI rows.

Every number here comes from ahjia/kpi.py (the replay) or specs/hailo10h.yaml.
Timing and power are simulated, not measured; the report says so at the top.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ahjia.kpi import COMPUTED_BASIS, KpiRow, aggregate  # noqa: E402
from ahjia.residency import HailoSpec, load_spec  # noqa: E402

STRATEGY_ORDER = ("S0", "S1", "S2")
# Validated categorical palette (dataviz reference slots 1-3, light surface).
SERIES_COLORS = {"S0": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a"}
INK, MUTED, GRID = "#1a1a19", "#5f5e58", "#e4e3dc"
TARGETS = (
    ("KPI 1 average power (24 h)", "<= 70 % of S0"),
    ("KPI 2 cameras per UGen300", ">= 4"),
    ("KPI 3 voice / family text", "<= 2 s / <= 5 s; A5 at once for 救命 / 起不來, else after 30 s"),
    ("KPI 3 talk / greeting", "talk p50 <= 3 s; greeting <= 4 s"),
    ("KPI 4 GenAI interference", "0 delayed safety events; LLM wakes <= 30 % of S0; reply >= 95 %"),
    ("KPI 5 high-risk misses", "0 on the scripted set; 24 h <= S0; false alarms <= 1 per day"),
)


def _f(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f} %"


ASSUMED_MARK = "†"


def _delayed(row: KpiRow) -> str:
    """Delayed safety events; marked when the count follows from an assumption, not the simulation."""
    return str(row.delayed_safety) + ("" if row.delayed_safety_basis == COMPUTED_BASIS else f" {ASSUMED_MARK}")


def _assumed_note(rows: Sequence[KpiRow], spec: HailoSpec) -> list[str]:
    assumed = sorted({r.strategy for r in rows if r.delayed_safety_basis != COMPUTED_BASIS})
    if not assumed:
        return []
    wait = spec.value("contention.qos_wait_s")
    return [f"{ASSUMED_MARK} {', '.join(assumed)}: follows from the safety-first QoS assumption "
            f"(qos_wait_s = {wait:g}); Stage II measures it.", ""]


def _ordered(rows: Sequence[KpiRow]) -> list[KpiRow]:
    return sorted(rows, key=lambda r: (r.scenario, STRATEGY_ORDER.index(r.strategy)))


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in body]


def _ratio(rows: Sequence[KpiRow], get: Callable[[KpiRow], float | None], row: KpiRow) -> float | None:
    base = next((r for r in rows if r.scenario == row.scenario and r.strategy == "S0"), None)
    value, ref = get(row), (get(base) if base else None)
    return value / ref if value is not None and ref else None


def _day_section(day: Sequence[KpiRow]) -> list[str]:
    body = [[r.strategy, _f(r.avg_power_w), _pct(_ratio(day, lambda x: x.avg_power_w, r)),
             str(r.cameras_supported), _f(r.voice_p50), _f(r.voice_p95), _f(r.text_p50), _f(r.text_p95),
             _f(r.talk_p50), _delayed(r), _f(r.llm_wakes_per_h, 1),
             _pct(_ratio(day, lambda x: x.llm_wakes_per_h, r)), _pct(r.reply_rate), str(r.misses),
             str(r.false_alarms)] for r in _ordered(day)]
    header = ["Strategy", "Power W", "vs S0", "Cameras (est.)", "Voice p50 s", "Voice p95 s", "Text p50 s",
              "Text p95 s", "Talk p50 s", "Delayed safety", "LLM wakes/h", "Wakes vs S0", "Reply",
              "Misses", "False alarms/day"]
    return ["## 24 h synthetic day (KPI 1-5)", ""] + _table(header, body) + [""]


def _demo_section(demo: Sequence[KpiRow]) -> list[str]:
    body = [[r.scenario, r.strategy, _f(r.avg_power_w), _f(r.voice_p50), _f(r.text_p50), _f(r.a5_immediate_s),
             _f(r.a5_no_answer_s), _f(r.talk_p50), _f(r.greeting_s), _delayed(r), str(r.llm_wakes),
             _pct(r.reply_rate), f"{r.misses}/{r.high_risk_labels}", str(r.false_alarms)] for r in _ordered(demo)]
    header = ["Scenario", "Strategy", "Power W", "Voice s", "Text s", "A5 at once s", "A5 30 s rule s",
              "Talk s", "Greeting s", "Delayed safety", "LLM wakes", "Reply", "Misses", "False alarms"]
    return ["## Demo scenarios (P:484-522)", ""] + _table(header, body) + [""]


def _scripted_section(scripted: Sequence[KpiRow]) -> list[str]:
    pooled = [aggregate([r for r in scripted if r.strategy == s], "scripted set")
              for s in STRATEGY_ORDER if any(r.strategy == s for r in scripted)]
    cases = sorted({r.scenario for r in scripted})
    body = [[r.strategy, f"{r.misses}/{r.high_risk_labels}", str(r.false_alarms), _f(r.voice_p50), _f(r.voice_p95),
             _f(r.text_p95), _delayed(r), _pct(r.reply_rate)] for r in pooled]
    header = ["Strategy", "Misses", "False alarms", "Voice p50 s", "Voice p95 s", "Text p95 s",
              "Delayed safety", "Reply"]
    return ([f"## Scripted set ({len(cases)} cases: {', '.join(cases)})", ""] + _table(header, body)
            + ["", "`talk_then_fall` was written to probe KPI 4 (a fall while the LLM answers).", ""])


def _spec_section(spec: HailoSpec) -> list[str]:
    counts = {k: sum(1 for p in spec.params.values() if p.kind == k) for k in ("official", "estimate", "assumption")}
    body = [[key, f"{p.value:g}", p.kind] for key, p in sorted(spec.params.items()) if p.kind != "official"]
    return ([f"## Simulator inputs ({counts['official']} official, {counts['estimate']} estimate, "
             f"{counts['assumption']} assumption)", "",
             "Non-official values below; sources for every value are in `specs/hailo10h.yaml`.", ""]
            + _table(["Parameter", "Value", "Kind"], body) + [""])


def render_markdown(rows: Sequence[KpiRow], spec: HailoSpec) -> str:
    groups = {g: [r for r in rows if r.group == g] for g in ("day", "demo", "scripted")}
    lines = [
        "# AhJia Care Router: Stage I KPI report",
        "",
        "> **Simulated, not measured.** Stage I runs on a laptop without a UGen300. Every timing and power",
        "> number below comes from the Hailo-10H simulator (`ahjia/residency.py`) driven by",
        "> `specs/hailo10h.yaml`, where each value is tagged official, estimate or assumption.",
        "> Stage II replaces these numbers with USB-C power-meter and log measurements.",
        "",
        "Strategies: S0 = fixed (cost ceiling), S1 = rules (baseline), S2 = Care Router (P:423-431).",
        "Times are seconds from the true event (a fall counts from 2 s after lying starts).",
        "KPI 2 cameras per UGen300 is an estimate (NPU + host decode assumption): min(NPU bound, host_max_streams).",
        "NPU bound = one camera under A3, the other busy cameras in high-risk zones, and every added camera with",
        "the scenario's average NPU demand per camera; host_max_streams is an assumption (N100 Quick Sync, not",
        "measured). USB bandwidth is not modelled.",
        "A TV voice saying 'help' is treated as real; telling a TV from the elder is Stage II work.",
        "",
    ]
    if groups["day"]:
        lines += _day_section(groups["day"]) + _assumed_note(groups["day"], spec)
    if groups["demo"]:
        lines += _demo_section(groups["demo"]) + _assumed_note(groups["demo"], spec)
    if groups["scripted"]:
        lines += _scripted_section(groups["scripted"]) + _assumed_note(groups["scripted"], spec)
    lines += ["## Provisional targets (P:437-443)", ""] + _table(["KPI", "Target"], TARGETS) + [""]
    lines += _spec_section(spec)
    return "\n".join(lines)


CSV_FIELDS = ("scenario", "group", "strategy", "duration_s", "avg_power_w", "cameras_supported", "cameras_npu",
              "cameras_host", "voice_p50",
              "voice_p95", "text_p50", "text_p95", "a5_immediate_s", "a5_no_answer_s", "talk_p50", "greeting_s",
              "delayed_safety", "delayed_safety_basis", "llm_wakes", "llm_wakes_per_h", "directed", "replied",
              "reply_rate",
              "high_risk_labels", "misses", "false_alarms", "frames")


def write_csv(rows: Sequence[KpiRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for r in _ordered(rows):
            values = [getattr(r, name) for name in CSV_FIELDS]
            writer.writerow(["" if v is None else (f"{v:.4f}" if isinstance(v, float) else v) for v in values])


def _value(value: object) -> object:
    return round(value, 4) if isinstance(value, float) else value


def write_json(rows: Sequence[KpiRow], spec: HailoSpec, path: Path) -> None:
    """kpi.json: the kpi.csv data as a list of records, flagged as simulated, with the spec's kind counts."""
    kinds = {k: sum(1 for p in spec.params.values() if p.kind == k) for k in ("official", "estimate", "assumption")}
    records = [{name: _value(getattr(r, name)) for name in CSV_FIELDS} for r in _ordered(rows)]
    payload = {"simulated": True, "spec_path": spec.path, "spec_kinds": kinds, "rows": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _panel(ax: plt.Axes, rows: Sequence[KpiRow], get: Callable[[KpiRow], float | None], title: str,
           fmt: str) -> None:
    rows = [r for r in rows if get(r) is not None]
    names = [r.strategy for r in rows]
    values = [float(get(r)) for r in rows]
    bars = ax.bar(names, values, width=0.6, color=[SERIES_COLORS[n] for n in names],
                  edgecolor="#fcfcfb", linewidth=2)
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_ylim(0, max(values + [1e-9]) * 1.2)


def write_png(rows: Sequence[KpiRow], path: Path) -> None:
    day = [r for r in rows if r.group == "day"] or [r for r in rows if r.avg_power_w is not None]
    ordered = sorted(day, key=lambda r: STRATEGY_ORDER.index(r.strategy))
    if day and len({r.scenario for r in day}) > 1:
        ordered = [r for r in ordered if r.scenario == ordered[0].scenario]
    label = ordered[0].scenario if ordered else "-"
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor="#fcfcfb")
    _panel(axes[0], ordered, lambda r: r.avg_power_w, "KPI 1  average power (W)", "{:.2f}")
    _panel(axes[1], ordered, lambda r: r.cameras_supported, "KPI 2  cameras per UGen300 (est.)", "{:.0f}")
    _panel(axes[2], ordered, lambda r: r.llm_wakes_per_h, "KPI 4  LLM wakes per hour", "{:.1f}")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    fig.suptitle(f"AhJia Care Router, scenario '{label}': SIMULATED from specs/hailo10h.yaml, not measured",
                 color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(rows: Sequence[KpiRow], out_dir: str | Path, spec: HailoSpec | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = spec or load_spec()
    report = out / "report.md"
    report.write_text(render_markdown(rows, spec), encoding="utf-8")
    write_csv(rows, out / "kpi.csv")
    write_json(rows, spec, out / "kpi.json")
    write_png(rows, out / "kpi.png")
    return report
