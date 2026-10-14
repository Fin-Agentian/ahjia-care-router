"""Posture from 17 COCO keypoints and a box: "stand", "sit", "lie" or "unknown".

The cameras sit high in a ceiling corner and look down, so the image is
perspective-distorted: a standing person is foreshortened and a person lying
along the view direction looks vertical. No single cue is trusted. Two scores
are built from several cues and compared:

lie score (evidence for "on the ground")
  * box aspect ratio (width / height in pixels): wide box
  * keypoint spread: horizontal range over vertical range
  * torso angle from vertical (shoulder middle -> hip middle)
  * foreshortening: torso length / shoulder width. Lying along the view axis
    shows a short torso with full shoulder width, even when it looks vertical.
upright score (evidence against)
  * tall narrow box, tall narrow keypoint spread
  * torso close to vertical, but only when the torso is not foreshortened,
    because a vertical torso is exactly what lying toward the camera looks like

"lie" needs the lie score to beat the upright score by lie_margin. Otherwise the
person is upright and the label is split into "sit" and "stand" with
  * knee drop: (knee_y - hip_y) / torso length (small when sitting), if clear
  * else height ratio: box height / torso length (small when legs are hidden)

Keypoints below min_conf are ignored; with fewer than min_points confident
points only the box aspect ratio decides. Thresholds live in PostureConfig and
must be re-tuned on real footage of the final camera position.
A PostureSmoother (majority vote over about 0.6 s) removes frame flicker.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ahjia.events import Posture
from ahjia.perception.pose import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, Keypoints,
)

_EPS = 1e-6
Point = tuple[float, float]


@dataclass(frozen=True)
class PostureConfig:
    """Thresholds of the posture heuristic (all configurable)."""

    min_conf: float = 0.3
    min_points: int = 5
    lie_ar: float = 1.25  # box width/height: clearly wide
    flat_ar: float = 1.0
    upright_ar: float = 0.75  # box width/height: clearly tall
    bbox_only_stand_ar: float = 0.5  # without keypoints, only this narrow means stand
    lie_spread: float = 1.3  # keypoint x range / y range
    flat_spread: float = 1.0
    upright_spread: float = 0.6
    lie_angle_deg: float = 60.0  # torso angle from vertical
    tilt_angle_deg: float = 45.0
    upright_angle_deg: float = 30.0
    strong_foreshortening: float = 0.55  # torso length / shoulder width
    weak_foreshortening: float = 0.75
    lie_margin: float = 2.0
    sit_knee_drop: float = 0.45
    stand_knee_drop: float = 0.6
    sit_height_ratio: float = 2.7  # box height / torso length
    window_s: float = 0.6


@dataclass(frozen=True)
class _Geometry:
    """Pixel-space measurements; None means "could not be measured"."""

    box_ar: float
    box_h: float
    spread: Optional[float]
    torso_len: Optional[float]
    torso_angle: Optional[float]
    foreshortening: Optional[float]
    knee_drop: Optional[float]


def _point(kps: Keypoints, idx: int, min_conf: float, aspect: float) -> Optional[Point]:
    x, y, c = kps[idx]
    return (x * aspect, y) if c >= min_conf else None


def _mid(kps: Keypoints, idxs: tuple[int, int], min_conf: float, aspect: float) -> Optional[Point]:
    pts = [p for p in (_point(kps, i, min_conf, aspect) for i in idxs) if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _spread_ratio(kps: Keypoints, cfg: PostureConfig, aspect: float) -> Optional[float]:
    pts = [p for p in (_point(kps, i, cfg.min_conf, aspect) for i in range(len(kps))) if p]
    if len(pts) < cfg.min_points:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) / max(max(ys) - min(ys), _EPS)


def _torso(kps: Keypoints, cfg: PostureConfig, aspect: float) -> tuple[Optional[Point], Optional[Point]]:
    sh = _mid(kps, (L_SHOULDER, R_SHOULDER), cfg.min_conf, aspect)
    hip = _mid(kps, (L_HIP, R_HIP), cfg.min_conf, aspect)
    return sh, hip


def _measure(kps: Keypoints, bbox: tuple[float, float, float, float], cfg: PostureConfig,
             aspect: float) -> _Geometry:
    box_w = max((bbox[2] - bbox[0]) * aspect, _EPS)
    box_h = max(bbox[3] - bbox[1], _EPS)
    sh, hip = _torso(kps, cfg, aspect)
    torso_len = angle = fore = knee_drop = None
    if sh is not None and hip is not None:
        dx, dy = hip[0] - sh[0], hip[1] - sh[1]
        torso_len = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(abs(dx), abs(dy) + _EPS))
        ls = _point(kps, L_SHOULDER, cfg.min_conf, aspect)
        rs = _point(kps, R_SHOULDER, cfg.min_conf, aspect)
        if ls is not None and rs is not None:
            fore = torso_len / max(math.hypot(ls[0] - rs[0], ls[1] - rs[1]), _EPS)
        knee = _mid(kps, (L_KNEE, R_KNEE), cfg.min_conf, aspect)
        if knee is not None and torso_len > _EPS:
            knee_drop = (knee[1] - hip[1]) / torso_len
    return _Geometry(box_w / box_h, box_h, _spread_ratio(kps, cfg, aspect), torso_len, angle, fore,
                     knee_drop)


def _lie_and_upright_scores(g: _Geometry, cfg: PostureConfig) -> tuple[float, float]:
    lie = up = 0.0
    if g.box_ar >= cfg.lie_ar:
        lie += 2
    elif g.box_ar > cfg.flat_ar:
        lie += 1
    elif g.box_ar <= cfg.upright_ar:
        up += 2
    if g.spread is not None:
        if g.spread >= cfg.lie_spread:
            lie += 2
        elif g.spread > cfg.flat_spread:
            lie += 1
        elif g.spread <= cfg.upright_spread:
            up += 1
    if g.foreshortening is not None:
        if g.foreshortening < cfg.strong_foreshortening:
            lie += 2
        elif g.foreshortening < cfg.weak_foreshortening:
            lie += 1
    lie2, up2 = _angle_scores(g, cfg)
    return lie + lie2, up + up2


def _angle_scores(g: _Geometry, cfg: PostureConfig) -> tuple[float, float]:
    if g.torso_angle is None:
        return 0.0, 0.0
    if g.torso_angle >= cfg.lie_angle_deg:
        return 2.0, 0.0
    if g.torso_angle >= cfg.tilt_angle_deg:
        return 1.0, 0.0
    # A vertical torso only counts as upright when it is not foreshortened.
    foreshortened = g.foreshortening is not None and g.foreshortening < cfg.strong_foreshortening
    if g.torso_angle <= cfg.upright_angle_deg and not foreshortened:
        return 0.0, 2.0
    return 0.0, 0.0


def _sit_or_stand(g: _Geometry, cfg: PostureConfig) -> Posture:
    """Knee drop decides when it is clear; otherwise the box height / torso ratio does."""
    if g.knee_drop is not None:
        if g.knee_drop < cfg.sit_knee_drop:
            return "sit"
        if g.knee_drop >= cfg.stand_knee_drop:
            return "stand"
    if g.torso_len is not None and g.torso_len > _EPS:
        return "sit" if g.box_h / g.torso_len < cfg.sit_height_ratio else "stand"
    return "unknown"


def _box_only(box_ar: float, cfg: PostureConfig) -> Posture:
    if box_ar >= cfg.lie_ar:
        return "lie"
    if box_ar <= cfg.bbox_only_stand_ar:
        return "stand"
    return "unknown"


def classify_posture(
    keypoints: Optional[Keypoints],
    bbox: tuple[float, float, float, float],
    aspect: float = 1.0,
    config: PostureConfig = PostureConfig(),
) -> Posture:
    """Classify one person. keypoints and bbox are normalized; aspect is frame width / height."""
    if keypoints is None:
        box_ar = max((bbox[2] - bbox[0]) * aspect, _EPS) / max(bbox[3] - bbox[1], _EPS)
        return _box_only(box_ar, config)
    geo = _measure(keypoints, bbox, config, aspect)
    confident = sum(1 for _, _, c in keypoints if c >= config.min_conf)
    if confident < config.min_points:
        return _box_only(geo.box_ar, config)
    lie, up = _lie_and_upright_scores(geo, config)
    if lie - up >= config.lie_margin:
        return "lie"
    return _sit_or_stand(geo, config)


class PostureSmoother:
    """Majority vote of the labels seen in the last window_s seconds, per track.

    "unknown" only wins when nothing else was seen. On a tie the label that was
    reported last time is kept, so the output switches once and does not flicker.
    """

    def __init__(self, window_s: float = 0.6) -> None:
        self._window_s = window_s
        self._history: dict[int, deque[tuple[float, Posture]]] = {}
        self._last: dict[int, Posture] = {}

    def update(self, track_id: int, t: float, label: Posture) -> Posture:
        hist = self._history.setdefault(track_id, deque())
        hist.append((t, label))
        while hist and hist[0][0] < t - self._window_s - 1e-9:
            hist.popleft()
        smoothed = self._vote(track_id, [lab for _, lab in hist])
        self._last[track_id] = smoothed
        return smoothed

    def prune(self, active_ids: set[int]) -> None:
        """Forget tracks that are gone."""
        for tid in [k for k in self._history if k not in active_ids]:
            del self._history[tid]
            self._last.pop(tid, None)

    def _vote(self, track_id: int, labels: list[Posture]) -> Posture:
        known = [lab for lab in labels if lab != "unknown"]
        if not known:
            return "unknown"
        counts = {lab: known.count(lab) for lab in set(known)}
        best = max(counts.values())
        leaders = [lab for lab, n in counts.items() if n == best]
        if len(leaders) == 1:
            return leaders[0]
        last = self._last.get(track_id)
        return last if last in leaders else known[-1]
