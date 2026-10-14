"""`python -m ahjia workbench --out DIR`: replay every scenario and write one self-contained workbench.html.

Steps:
1. `python -m ahjia replay --all` into a temporary folder (decision logs, responses, kpi.json / kpi.csv).
2. The same replays again in-process with a thin recorder around the engine, to read what the
   decision log does not hold between decisions: the frame rate per camera, the 7 HUD lines and the
   GenAI state after every step (Engine protocol: camera_fps, hud).
3. Scenario files -> true world state (people, zone, posture, speech).
4. Everything is embedded as JSON into template.html with style.css and app.js inlined.

No network: the page has no external script, style or font.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import ahjia.replay as replay_mod
from ahjia.events import (
    ACTIONS,
    HIGH_RISK_ZONES,
    NIGHT_END,
    NIGHT_HIGH_RISK_ZONES,
    NIGHT_START,
    REST_ZONES,
    decision_to_json,
)
from ahjia.workbench import data as wd

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
STRATEGY_NAMES = tuple(s["name"] for s in wd.STRATEGIES)
GITHUB_TEXT = "github.com/Fin-Agentian/ahjia-care-router"


# ---------------------------------------------------------------- engine recorder

class EngineRecorder:
    """Wraps an engine; after every observe / advance it notes frame rates, HUD lines and GenAI state."""

    def __init__(self, engine: Any, cameras: Sequence[str]) -> None:
        self._engine = engine
        self._cameras = tuple(cameras)
        self.timeline: list[dict[str, Any]] = []
        self._snap(0.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def observe(self, obs: Any) -> Any:
        out = self._engine.observe(obs)
        self._snap(float(obs.t))
        return out

    def advance(self, t: float) -> Any:
        out = self._engine.advance(t)
        self._snap(float(t))
        return out

    def _genai(self, t: float) -> str | None:
        try:
            return str(self._engine.sim.genai_state(t))
        except (AttributeError, TypeError):
            return None

    def _snap(self, t: float) -> None:
        fps = {cam: float(self._engine.camera_fps(cam)) for cam in self._cameras}
        state = {"fps": fps, "hud": list(self._engine.hud().lines()), "genai": self._genai(t)}
        if self.timeline and all(self.timeline[-1][k] == v for k, v in state.items()):
            return
        if self.timeline and abs(self.timeline[-1]["t"] - t) < 1e-9:
            self.timeline[-1].update(state)
            return
        self.timeline.append({"t": round(t, 4), **state})


@contextmanager
def recording() -> Iterator[list[EngineRecorder]]:
    """Temporarily make replay build recorded engines (restored afterwards)."""
    original = replay_mod.make_engine
    made: list[EngineRecorder] = []

    def factory(strategy: Any, config: Any) -> EngineRecorder:
        rec = EngineRecorder(original(strategy, config), tuple(config.cameras))
        made.append(rec)
        return rec

    replay_mod.make_engine = factory
    try:
        yield made
    finally:
        replay_mod.make_engine = original


def probe(scn: replay_mod.Scenario, strategy: str) -> dict[str, Any]:
    """Run one scenario in-process; return the step timeline plus decisions and responses."""
    with recording() as made:
        result = replay_mod.run_scenario(scn, strategy)
    return {
        "timeline": made[-1].timeline if made else None,
        "decisions": [json.loads(decision_to_json(d)) for d in result.decisions],
        "responses": [wd.response_json(r) for r in result.responses if r.action in ("A1", "A2", "A4", "A5")],
    }


# ---------------------------------------------------------------- replay outputs

def run_replay(out: Path) -> None:
    """`python -m ahjia replay --all --out OUT` (quiet)."""
    from ahjia.cli import build_parser

    args = build_parser().parse_args(["replay", "--all", "--out", str(out)])
    level = logging.getLogger("ahjia").level
    logging.getLogger("ahjia").setLevel(logging.WARNING)
    try:
        with redirect_stdout(StringIO()):
            code = replay_mod.main(args)
    finally:
        logging.getLogger("ahjia").setLevel(level)
    if code != 0:
        raise RuntimeError(f"replay exited with code {code}")


def _rules() -> dict[str, Any]:
    from ahjia import guardrails
    from ahjia.router import FrameRatePlan

    rates = FrameRatePlan()
    return {
        "escalate_no_answer_s": float(guardrails.ESCALATE_NO_ANSWER_S),
        "checkin_no_answer_s": float(guardrails.CHECKIN_NO_ANSWER_S),
        "standing_clear_s": float(guardrails.STANDING_CLEAR_S),
        "fps": {"empty": rates.empty_fps, "empty_watch": rates.empty_watch_fps, "occupied": rates.occupied_fps,
                "high_risk": rates.high_risk_fps},
    }


def _full_fps(spec_path: Path) -> float:
    import yaml

    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        return float(spec["device"]["camera_full_fps"]["value"])
    except (OSError, KeyError, TypeError, ValueError):
        return 30.0


def _scenario_entry(path: Path, replay_out: Path, main_names: dict[str, str],
                    branch_labels: dict[str, str]) -> dict[str, Any]:
    scn = replay_mod.load_scenario(path)
    runs: dict[str, Any] = {}
    for strategy in STRATEGY_NAMES:
        folder = replay_out / "replay" / scn.name
        probed = probe(scn, strategy)
        decisions = wd.read_jsonl(folder / f"{strategy}.jsonl")
        responses = wd.read_jsonl(folder / f"{strategy}.responses.jsonl")
        runs[strategy] = {
            "decisions": decisions if decisions is not None else probed["decisions"],
            "responses": responses if responses is not None else probed["responses"],
            "responses_source": "responses.jsonl" if responses is not None else "engine",
            "timeline": probed["timeline"],
        }
    return {
        "name": scn.name, "title": scn.title, "source": scn.source, "group": scn.group,
        "main": scn.name in main_names,
        "friendly": main_names.get(scn.name) or branch_labels.get(scn.name) or scn.title,
        "start_clock": scn.start_clock, "cameras": list(scn.cameras), "duration_s": scn.duration_s,
        "synthetic": scn.synthetic, "labels": wd.labels_json(scn), "world": wd.world_json(scn), "runs": runs,
    }


def build_data(replay_out: Path, scenario_root: Path = replay_mod.SCENARIO_DIR) -> dict[str, Any]:
    paths = replay_mod.find_scenarios(None, True, root=scenario_root)
    main_names = dict(wd.MAIN_SCENARIOS)
    branches = wd.build_branches((p.stem for p in paths if p.parent.name == "scripted"),
                                 standing_clear_s=_rules()["standing_clear_s"])
    branch_labels = {b["scenario"]: "What if " + b["label"] for b in branches}
    entries = [_scenario_entry(p, replay_out, main_names, branch_labels) for p in paths]
    order = {name: i for i, (name, _) in enumerate(wd.MAIN_SCENARIOS)}
    entries.sort(key=lambda e: (0, order[e["name"]]) if e["main"] else (1, e["name"]))
    kpis = wd.read_kpis(replay_out)
    kpis["scripted"] = wd.scripted_summary(kpis["rows"])
    spec_path = wd.default_spec_path()
    return {
        "meta": {
            "title": "AhJia Care Router · Workbench",
            "simulated": True,
            "github": GITHUB_TEXT,
            "actions": ACTIONS,
            "full_fps": _full_fps(spec_path),
            "night": [NIGHT_START, NIGHT_END],
            "high_risk_zones": sorted(HIGH_RISK_ZONES),
            "night_high_risk_zones": sorted(NIGHT_HIGH_RISK_ZONES),
            "rest_zones": sorted(REST_ZONES),
            "rules": _rules(),
        },
        "strategies": list(wd.STRATEGIES),
        "scenarios": entries,
        "branches": branches,
        "kpis": kpis,
    }


# ---------------------------------------------------------------- HTML

def _json_for_script(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(payload: Any) -> str:
    template = (HERE / "template.html").read_text(encoding="utf-8")
    style = (HERE / "style.css").read_text(encoding="utf-8")
    app = (HERE / "app.js").read_text(encoding="utf-8")
    parts = template.split("@@")
    pieces = {"STYLE": style, "APP": app, "DATA": _json_for_script(payload)}
    return "".join(pieces.get(p, p) if i % 2 else p for i, p in enumerate(parts))


def main(args: argparse.Namespace) -> int:
    out = Path(args.out)
    try:
        with tempfile.TemporaryDirectory(prefix="ahjia_wb_") as tmp:
            run_replay(Path(tmp))
            payload = build_data(Path(tmp))
    except Exception as err:  # noqa: BLE001 - report any replay failure as a CLI error
        log.error("workbench: replay failed: %s", err)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    target = out / "workbench.html"
    target.write_text(render_html(payload), encoding="utf-8")
    sys.stdout.write(f"wrote {target} ({target.stat().st_size / 1e6:.2f} MB; timing and power simulated)\n")
    return 0
