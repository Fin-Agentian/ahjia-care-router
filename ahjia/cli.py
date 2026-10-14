"""Command line: `python -m ahjia replay ...` and `python -m ahjia demo ...` (see DESIGN.md)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ahjia", description="AhJia Care Router prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="replay scenarios under S0/S1/S2 and write KPIs")
    pick = replay.add_mutually_exclusive_group(required=True)
    pick.add_argument("--scenario", help="scenario name (file name in scenarios/ without .yaml)")
    pick.add_argument("--all", action="store_true", help="replay every scenario")
    replay.add_argument("--strategies", default="S0,S1,S2", help="comma separated, e.g. S0,S2")
    replay.add_argument("--out", default="outputs", help="output folder")

    demo = sub.add_parser("demo", help="run pose + Care Router on a video and write an annotated mp4")
    demo.add_argument("--video", required=True, help="input video file")
    demo.add_argument("--camera", required=True, help="camera config YAML (zones, clock, identity)")
    demo.add_argument("--speech", default=None, help="timed transcript YAML (simulated Whisper output)")
    demo.add_argument("--out", default="outputs/demo", help="output folder")
    demo.add_argument("--fps", type=float, default=10.0, help="frames per second to analyse")
    demo.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose weights")

    bench = sub.add_parser("workbench", help="replay every scenario and write the self-contained Workbench HTML")
    bench.add_argument("--out", default="outputs/workbench", help="output folder")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        from ahjia.replay import main as replay_main

        return replay_main(args)
    if args.command == "workbench":
        from ahjia.workbench.export import main as workbench_main

        return workbench_main(args)
    from ahjia.demo import main as demo_main

    return demo_main(args)
