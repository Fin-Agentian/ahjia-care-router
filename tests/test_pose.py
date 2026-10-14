"""PoseDetector with a fake model (no weights, no ultralytics)."""

import numpy as np
import pytest

from ahjia.perception.pose import Detection, PoseDetector, detections_from_arrays

pytestmark = pytest.mark.unit


class FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeResult:
    def __init__(self, n, with_conf=True):
        xy = np.tile(np.array([[100.0, 50.0]]), (n, 17, 1))
        self.boxes = type("B", (), {
            "xyxy": FakeTensor([[10, 20, 110, 220]] * n),
            "conf": FakeTensor([0.9] * n),
            "__len__": lambda s: n,
        })()
        kp = {"xy": FakeTensor(xy)}
        if with_conf:
            kp["conf"] = FakeTensor(np.full((n, 17), 0.8))
        self.keypoints = type("K", (), kp)()


class FakeYolo:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return self.results


FRAME = np.zeros((200, 400, 3), dtype=np.uint8)


def test_detect_normalizes_to_0_1():
    model = FakeYolo([FakeResult(2)])
    dets = PoseDetector(model=model, device="cpu").detect(FRAME)
    assert len(dets) == 2
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.bbox == pytest.approx((10 / 400, 20 / 200, 110 / 400, 1.0))  # y clipped to 1
    assert len(d.keypoints) == 17
    assert d.keypoints[0] == pytest.approx((0.25, 0.25, 0.8))
    assert d.score == pytest.approx(0.9)
    assert model.calls[0]["device"] == "cpu"


def test_detect_without_keypoint_confidence_defaults_to_one():
    dets = PoseDetector(model=FakeYolo([FakeResult(1, with_conf=False)])).detect(FRAME)
    assert dets[0].keypoints[3][2] == 1.0


def test_detect_empty_cases():
    assert PoseDetector(model=FakeYolo([])).detect(FRAME) == []
    empty = FakeResult(0)
    assert PoseDetector(model=FakeYolo([empty])).detect(FRAME) == []
    none_kp = FakeResult(1)
    none_kp.keypoints = None
    assert PoseDetector(model=FakeYolo([none_kp])).detect(FRAME) == []


def test_detections_from_arrays_clips_out_of_frame_points():
    dets = detections_from_arrays(
        np.array([[-5.0, -5.0, 500.0, 300.0]]),
        np.array([0.5]),
        np.full((1, 17, 2), 1e6),
        None,
        width=400,
        height=200,
    )
    assert dets[0].bbox == (0.0, 0.0, 1.0, 1.0)
    assert dets[0].keypoints[0][:2] == (1.0, 1.0)
