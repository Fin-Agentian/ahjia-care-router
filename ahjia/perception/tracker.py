"""Small IoU / centroid tracker that keeps track ids stable for one or two people.

Each frame, boxes are matched to existing tracks greedily. A pair matches on
IoU, or on centroid distance when the box shape changes a lot (standing ->
lying changes the box from tall to wide, which drops IoU to almost zero).
Tracks that are not seen for max_age_s are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _diag(b: Box) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


@dataclass(frozen=True)
class _Track:
    track_id: int
    box: Box
    last_seen: float


class IouTracker:
    """Assigns stable ids. update() returns one id per input box, in input order."""

    def __init__(self, iou_threshold: float = 0.3, max_dist: float = 0.6, max_age_s: float = 1.5) -> None:
        self._iou_threshold = iou_threshold
        self._max_dist = max_dist  # centroid distance as a fraction of the larger box diagonal
        self._max_age_s = max_age_s
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, t: float, boxes: Sequence[Box]) -> list[int]:
        self._tracks = [tr for tr in self._tracks if t - tr.last_seen <= self._max_age_s]
        assignment = self._match(boxes)
        ids: list[int] = []
        seen: dict[int, Box] = {}
        for i, box in enumerate(boxes):
            tid = assignment.get(i)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            ids.append(tid)
            seen[tid] = box
        kept = [tr for tr in self._tracks if tr.track_id not in seen]
        self._tracks = kept + [_Track(tid, box, t) for tid, box in seen.items()]
        return ids

    def _match(self, boxes: Sequence[Box]) -> dict[int, int]:
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for bi, box in enumerate(boxes):
                score = self._pair_score(track.box, box)
                if score is not None:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        assignment: dict[int, int] = {}
        for _, ti, bi in pairs:
            if ti in used_t or bi in assignment:
                continue
            used_t.add(ti)
            assignment[bi] = self._tracks[ti].track_id
        return assignment

    def _pair_score(self, track_box: Box, box: Box) -> float | None:
        overlap = iou(track_box, box)
        if overlap >= self._iou_threshold:
            return 1.0 + overlap  # real overlap always beats a centroid-only match
        ax, ay = _center(track_box)
        bx, by = _center(box)
        limit = self._max_dist * max(_diag(track_box), _diag(box))
        dist = math.hypot(ax - bx, ay - by)
        return (1.0 - dist / limit) if limit > 0 and dist <= limit else None
