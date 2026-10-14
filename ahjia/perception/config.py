"""Loaders for the camera config and the simulated speech file (formats in DESIGN.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, get_args

import yaml

from ahjia.events import Identity, SpeechObs, SpeechSource
from ahjia.perception.zones import ZoneMap, build_zone_map

_CLOCK = re.compile(r"^\d{2}:\d{2}:\d{2}$")


@dataclass(frozen=True)
class CameraConfig:
    """One camera for the demo. Face ID is simulated: identity comes from here."""

    name: str
    start_clock: str
    elder_alone: bool
    identity: str
    zones: ZoneMap


@dataclass(frozen=True)
class TimedSpeech:
    """One simulated Whisper segment: it is heard at t seconds into the video."""

    t: float
    speech: SpeechObs


def _read_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_camera_config(path: Union[str, Path]) -> CameraConfig:
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or "camera" not in raw:
        raise ValueError(f"{path}: camera config needs a 'camera' name")
    clock = str(raw.get("start_clock", "12:00:00"))
    if not _CLOCK.match(clock):
        raise ValueError(f"{path}: start_clock must look like HH:MM:SS, got {clock!r}")
    identity = str(raw.get("identity", "elder"))
    if identity not in get_args(Identity):
        raise ValueError(f"{path}: unknown identity {identity!r}")
    return CameraConfig(
        name=str(raw["camera"]),
        start_clock=clock,
        elder_alone=bool(raw.get("elder_alone", True)),
        identity=identity,
        zones=build_zone_map(raw.get("zones"), str(raw.get("default_zone", "floor"))),
    )


def load_speech(path: Union[str, Path, None]) -> tuple[TimedSpeech, ...]:
    """Timed transcript sorted by time; None or an empty file gives no speech."""
    if path is None:
        return ()
    raw = _read_yaml(path) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: speech file must be a list of segments")
    items = [_parse_segment(path, i, entry) for i, entry in enumerate(raw)]
    return tuple(sorted(items, key=lambda it: it.t))


def _parse_segment(path: Any, index: int, entry: Any) -> TimedSpeech:
    if not isinstance(entry, dict) or "t" not in entry or "text" not in entry:
        raise ValueError(f"{path}: segment {index} needs 't' and 'text'")
    source = str(entry.get("source", "elder"))
    if source not in get_args(SpeechSource):
        raise ValueError(f"{path}: segment {index} has unknown source {source!r}")
    speech = SpeechObs(
        text=str(entry["text"]),
        source=source,  # type: ignore[arg-type]
        directed=bool(entry.get("directed", False)),
        lang=str(entry.get("lang", "en")),
    )
    return TimedSpeech(t=float(entry["t"]), speech=speech)
