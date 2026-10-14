"""Data shaping for the Workbench: scenario files and replay outputs -> plain JSON-ready dicts.

Nothing here invents a number. The world state comes from the scenario files
(the true state of the world), decisions and responses from the replay's JSONL
files, KPIs from kpi.json (or kpi.csv when kpi.json is missing) and the
official / estimate / assumption counts from the spec file.
"""

from __future__ import annotations

import bisect
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ahjia.replay import Scenario

SPEECH_SHOW_S = 4.0        # a speech bubble stays on the plan this long
APP_CONFIRM_SHOW_S = 4.0   # the "confirmed in the app" badge stays this long
SPEC_KINDS = ("official", "estimate", "assumption")

MAIN_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("family_arrives", "Family arrives"),
    ("dizzy", "“I feel dizzy”"),
    ("night_fall", "Night fall"),
    ("day24h", "24-hour day"),
)

BRANCHES: tuple[tuple[str, str, str, str], ...] = (
    ("ok_upright", "…she says I'm OK and stands up", "ok_and_upright",
     "Rule 3: she clears it herself (I'm OK + stands {standing_clear_s:g} s)"),
    ("ok_down", "…she says I'm OK but stays down", "ok_but_lying",
     "Rule 3: words alone do not clear; her answer stops the call"),
    ("app", "…family confirms in the app", "app_confirm", "Rule 3: family clears it in the app"),
    ("tv", "…the TV shouts help", "tv_help", "Rule 1: any 'help' calls at once, even the TV (a false alarm)"),
    ("no_answer", "…nobody answers (30 s rule)", "dizzy_en", "Rule 2: no answer in 30 s, so AhJia calls"),
)

STRATEGIES: tuple[dict[str, str], ...] = (
    {"name": "S0", "label": "S0 Fixed", "tip": "Cost ceiling: every camera at full rate, the LLM always loaded."},
    {"name": "S1", "label": "S1 Rules", "tip": "Real baseline: fixed frame rate, rule agents, fixed GenAI hours."},
    {"name": "S2", "label": "S2 Care Router",
     "tip": "Our method: frame rate from risk, safety first, GenAI only in a safe window."},
)


# ---------------------------------------------------------------- world state

def _person(p: Any) -> dict[str, Any]:
    return {"id": int(p.track_id), "who": str(p.identity), "pose": str(p.posture), "zone": str(p.zone)}


def world_json(scn: Scenario) -> dict[str, Any]:
    """True world state as change lists: cams[camera] = [[t, persons], ...], plus speech and app confirms."""
    cams: dict[str, list[list[Any]]] = {cam: [] for cam in scn.cameras}
    speech: list[dict[str, Any]] = []
    confirms: list[float] = []
    for e in scn.events:
        if e.camera is not None:
            cams.setdefault(e.camera, []).append([float(e.t), [_person(p) for p in e.persons or ()]])
        elif e.speech is not None:
            speech.append({"t": float(e.t), "text": e.speech.text, "source": e.speech.source,
                           "directed": bool(e.speech.directed), "lang": e.speech.lang})
        elif e.app_confirm:
            confirms.append(float(e.t))
    return {"cams": cams, "speech": speech, "app_confirm": confirms}


def _last_at(changes: Sequence[Sequence[Any]], t: float) -> Any:
    times = [c[0] for c in changes]
    i = bisect.bisect_right(times, t + 1e-9)
    return changes[i - 1][1] if i else []


def world_state_at(world: Mapping[str, Any], t: float, speech_window: float = SPEECH_SHOW_S,
                   confirm_window: float = APP_CONFIRM_SHOW_S) -> dict[str, Any]:
    """Who each camera sees at time t, the speech still on screen and whether an app confirm just came in.

    The Workbench page (app.js, worldAt) implements the same rule.
    """
    cams = {cam: list(_last_at(changes, t)) for cam, changes in world["cams"].items()}
    speech = [s for s in world["speech"] if s["t"] - 1e-9 <= t < s["t"] + speech_window]
    confirm = any(c - 1e-9 <= t < c + confirm_window for c in world["app_confirm"])
    return {"cams": cams, "speech": speech, "app_confirm": confirm}


def labels_json(scn: Scenario) -> list[dict[str, Any]]:
    return [{"t": float(lb.t), "kind": lb.kind, "what": lb.what, "window_s": float(lb.window_s)} for lb in scn.labels]


# ---------------------------------------------------------------- branches

def build_branches(available: Iterable[str], standing_clear_s: float | None = None) -> list[dict[str, str]]:
    """'What if...' buttons; a button whose scripted scenario file is missing is left out.

    The rule texts quote the guardrail's own standing time (guardrails.STANDING_CLEAR_S unless given).
    """
    if standing_clear_s is None:
        from ahjia import guardrails

        standing_clear_s = guardrails.STANDING_CLEAR_S
    have = set(available)
    return [{"id": bid, "label": label, "scenario": scn,
             "rule": rule.format(standing_clear_s=standing_clear_s)} for bid, label, scn, rule in BRANCHES
            if scn in have]


# ---------------------------------------------------------------- replay outputs

def read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    """Records of a JSONL file, or None when the file does not exist."""
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _number(text: str) -> Any:
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return text
    return int(value) if value.is_integer() and "." not in text else value


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [{k: (v if k in ("scenario", "group", "strategy") else _number(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def spec_kind_counts(path: Path) -> dict[str, int]:
    """Count the spec leaves (dicts with value + kind) per kind."""
    counts = dict.fromkeys(SPEC_KINDS, 0)
    if not path.exists():
        return counts

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if "value" in node and "kind" in node:
                kind = str(node["kind"])
                counts[kind] = counts.get(kind, 0) + 1
                return
            for child in node.values():
                walk(child)

    walk(yaml.safe_load(path.read_text(encoding="utf-8")))
    return counts


def default_spec_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "specs" / "hailo10h.yaml"


ASSUMED_DELAY_BASIS = "assumption: qos_wait_s"


def delayed_safety_basis(row: Mapping[str, Any]) -> str:
    """How a row's 'safety events delayed by the LLM' count came about.

    kpi.json rows carry the replay's own `delayed_safety_basis` ("computed" or "assumption: qos_wait_s").
    Older rows and kpi.csv do not; S2's zero follows from the safety-first QoS assumption (qos_wait_s = 0),
    so S2 counts as an assumption and S0 / S1 as computed.
    """
    basis = row.get("delayed_safety_basis")
    if basis:
        return str(basis)
    return ASSUMED_DELAY_BASIS if row.get("strategy") == "S2" else "computed"


def read_kpis(replay_out: Path, spec_path: Path | None = None) -> dict[str, Any]:
    """KPI rows from kpi.json, else kpi.csv; spec kind counts from kpi.json, else the spec file."""
    js, cs = replay_out / "kpi.json", replay_out / "kpi.csv"
    if js.exists():
        payload = json.loads(js.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, Mapping) else list(payload)
        kinds = payload.get("spec_kinds") if isinstance(payload, Mapping) else None
        source: str | None = "kpi.json"
    elif cs.exists():
        rows, kinds, source = _csv_rows(cs), None, "kpi.csv"
    else:
        rows, kinds, source = [], None, None
    if not kinds:
        kinds = spec_kind_counts(spec_path or default_spec_path())
    rows = [{**row, "delayed_safety_basis": delayed_safety_basis(row)} for row in rows]
    return {"source": source, "simulated": True, "rows": rows, "spec_kinds": kinds}


def scripted_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pool the scripted set per strategy (sums of the replay's counts).

    `delayed_safety_assumed` is True when any pooled row's delayed count rests on an assumption.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("group") != "scripted":
            continue
        acc = out.setdefault(str(row["strategy"]), {"cases": 0, "misses": 0, "high_risk_labels": 0,
                                                    "delayed_safety": 0, "false_alarms": 0,
                                                    "delayed_safety_assumed": False})
        acc["cases"] += 1
        acc["delayed_safety_assumed"] = (acc["delayed_safety_assumed"]
                                         or delayed_safety_basis(row).startswith("assumption"))
        for key in ("misses", "high_risk_labels", "delayed_safety", "false_alarms"):
            acc[key] += int(row.get(key) or 0)
    return out


def response_json(r: Any) -> dict[str, Any]:
    """An engine Response in the responses.jsonl shape (used only when that file is missing)."""
    t_event, t_done = float(r.t_event), float(r.t_response)
    return {"t_event": t_event, "agent": r.agent, "action": r.action, "t_done": round(t_done, 3),
            "latency_s": round(t_done - t_event, 3), "kind": getattr(r, "kind", ""),
            "genai_wait_s": round(float(getattr(r, "genai_wait_s", 0.0)), 3), "simulated": True}
