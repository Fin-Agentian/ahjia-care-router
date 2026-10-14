"""Pose detection: Ultralytics YOLO pose on the CPU behind a small Protocol.

ultralytics (and torch) are imported lazily inside PoseDetector, so the core
engine and the unit tests never need them. Tests inject a fake detector or a
fake model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

# COCO keypoint indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
NUM_KEYPOINTS = 17

# Bones drawn by the overlay (COCO skeleton).
SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

Keypoints = Sequence[tuple[float, float, float]]  # 17 x (x, y, confidence), x and y in 0..1


@dataclass(frozen=True)
class Detection:
    """One person found by the pose model. Coordinates are normalized to 0..1."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: tuple[tuple[float, float, float], ...]
    score: float


class PoseDetectorProtocol(Protocol):
    """Anything that turns a BGR frame into detections (real model or test fake)."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=float)


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def detections_from_arrays(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: Optional[np.ndarray],
    width: int,
    height: int,
) -> list[Detection]:
    """Normalize pixel arrays (N,4), (N,), (N,17,2), (N,17) into Detection records."""
    out: list[Detection] = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        bbox = (_clip01(x1 / width), _clip01(y1 / height), _clip01(x2 / width), _clip01(y2 / height))
        kps = tuple(
            (
                _clip01(kpts_xy[i][k][0] / width),
                _clip01(kpts_xy[i][k][1] / height),
                float(kpts_conf[i][k]) if kpts_conf is not None else 1.0,
            )
            for k in range(NUM_KEYPOINTS)
        )
        out.append(Detection(bbox=bbox, keypoints=kps, score=float(scores[i])))
    return out


class PoseDetector:
    """YOLO pose on the CPU. Weights are loaded on first use.

    model may be a ready object with the Ultralytics call signature (used by
    tests); otherwise model_path is loaded with ultralytics.YOLO.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        conf: float = 0.25,
        imgsz: int = 640,
        model: Any = None,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._conf = conf
        self._imgsz = imgsz
        self._model = model

    def _load(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO  # lazy: heavy import
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "ultralytics is not installed; run: pip install -r requirements-vision.txt"
                ) from exc
            log.info("loading pose model %s on %s", self._model_path, self._device)
            self._model = YOLO(self._model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run the model on one BGR frame and return the people it found."""
        height, width = frame.shape[:2]
        results = self._load()(
            frame, device=self._device, conf=self._conf, imgsz=self._imgsz, verbose=False
        )
        if not results:
            return []
        res = results[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        kconf = getattr(res.keypoints, "conf", None)
        return detections_from_arrays(
            _to_numpy(res.boxes.xyxy),
            _to_numpy(res.boxes.conf),
            _to_numpy(res.keypoints.xy),
            _to_numpy(kconf) if kconf is not None else None,
            width,
            height,
        )
