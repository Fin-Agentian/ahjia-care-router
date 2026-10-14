"""Zones: normalized polygons from the camera config, and the foot point of a person.

The zone of a person is the zone that contains the foot point: the middle of
the ankles when they are confident, else the bottom center of the box.
Zone polygons are checked in config order; the first hit wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, get_args

from ahjia.events import Zone
from ahjia.perception.pose import L_ANKLE, R_ANKLE, Keypoints

Polygon = tuple[tuple[float, float], ...]
VALID_ZONES: frozenset[str] = frozenset(get_args(Zone))


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting; points exactly on an edge count as inside."""
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9


@dataclass(frozen=True)
class ZoneMap:
    """Named polygons plus the zone used for the ground outside all of them."""

    polygons: Mapping[str, Polygon]
    default_zone: str = "floor"

    def zone_at(self, x: float, y: float) -> str:
        for name, poly in self.polygons.items():
            if point_in_polygon(x, y, poly):
                return name
        return self.default_zone


def build_zone_map(zones: Optional[Mapping[str, Sequence[Sequence[float]]]],
                   default_zone: str = "floor") -> ZoneMap:
    """Validate a camera-config zone section and freeze it."""
    polygons: dict[str, Polygon] = {}
    for name, pts in (zones or {}).items():
        _check_zone_name(name)
        if len(pts) < 3:
            raise ValueError(f"zone {name!r} needs at least 3 points")
        poly = tuple((float(p[0]), float(p[1])) for p in pts)
        if any(not (0.0 <= v <= 1.0) for p in poly for v in p):
            raise ValueError(f"zone {name!r} has coordinates outside 0..1")
        polygons[name] = poly
    _check_zone_name(default_zone)
    return ZoneMap(polygons=polygons, default_zone=default_zone)


def _check_zone_name(name: str) -> None:
    if name not in VALID_ZONES:
        raise ValueError(f"unknown zone {name!r}; valid zones: {sorted(VALID_ZONES)}")


def foot_point(
    keypoints: Optional[Keypoints],
    bbox: tuple[float, float, float, float],
    min_conf: float = 0.3,
) -> tuple[float, float]:
    """Mid-ankles if at least one ankle is confident, else the bottom center of the box."""
    if keypoints is not None:
        ankles = [keypoints[i] for i in (L_ANKLE, R_ANKLE) if keypoints[i][2] >= min_conf]
        if ankles:
            return (
                sum(a[0] for a in ankles) / len(ankles),
                sum(a[1] for a in ankles) / len(ankles),
            )
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])
