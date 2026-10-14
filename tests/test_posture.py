"""Posture heuristic and temporal smoother."""

import pytest

from ahjia.perception.posture import PostureConfig, PostureSmoother, classify_posture
from perception_samples import (
    ASPECT,
    low_confidence,
    lying_flat,
    lying_toward_camera,
    sitting,
    standing,
    standing_high_angle,
)

pytestmark = pytest.mark.unit


def classify(sample, **kwargs):
    kps, bbox = sample
    return classify_posture(kps, bbox, aspect=ASPECT, **kwargs)


def test_standing_person():
    assert classify(standing()) == "stand"


def test_standing_seen_from_a_high_corner_is_still_stand():
    assert classify(standing_high_angle(squash=0.6)) == "stand"
    assert classify(standing_high_angle(squash=0.8)) == "stand"


def test_sitting_person():
    assert classify(sitting()) == "sit"


def test_sitting_with_legs_hidden_behind_furniture():
    assert classify(sitting(legs_visible=False)) == "sit"


def test_lying_flat():
    assert classify(lying_flat()) == "lie"


def test_lying_along_the_view_axis_is_lie_not_stand():
    assert classify(lying_toward_camera()) == "lie"


def test_low_confidence_keypoints_fall_back_to_the_box():
    # A wide box is still lying, a narrow tall one is standing.
    assert classify(low_confidence(lying_flat())) == "lie"
    assert classify(low_confidence(standing())) == "stand"


def test_low_confidence_with_ambiguous_box_is_unknown():
    kps, _ = low_confidence(standing())
    square_box = (0.4, 0.3, 0.6, 0.65)  # about 0.6 wide/high in pixels
    assert classify_posture(kps, square_box, aspect=ASPECT) == "unknown"


def test_no_keypoints_uses_the_box_only():
    assert classify_posture(None, (0.2, 0.6, 0.8, 0.9), aspect=ASPECT) == "lie"
    assert classify_posture(None, (0.45, 0.1, 0.55, 0.9), aspect=ASPECT) == "stand"


def test_thresholds_are_configurable():
    lenient = PostureConfig(lie_margin=99.0)  # impossible to reach lie
    assert classify(lying_flat(), config=lenient) != "lie"


def test_smoother_removes_single_frame_flicker():
    sm = PostureSmoother(window_s=0.6)
    labels = ["stand", "stand", "lie", "stand", "stand"]
    out = [sm.update(1, i * 0.1, lab) for i, lab in enumerate(labels)]
    assert out == ["stand"] * 5


def test_smoother_follows_a_real_change_within_the_window():
    sm = PostureSmoother(window_s=0.6)
    out = []
    for i in range(12):
        t = i * 0.1
        out.append(sm.update(1, t, "stand" if t < 0.5 else "lie"))
    assert out[0] == "stand"
    assert out[-1] == "lie"
    # One-way switch: no flip back to stand once lie has taken over.
    first_lie = out.index("lie")
    assert set(out[first_lie:]) == {"lie"}


def test_smoother_prefers_known_labels_over_unknown():
    sm = PostureSmoother(window_s=0.6)
    sm.update(1, 0.0, "sit")
    assert sm.update(1, 0.1, "unknown") == "sit"


def test_smoother_all_unknown_is_unknown():
    sm = PostureSmoother(window_s=0.6)
    assert sm.update(1, 0.0, "unknown") == "unknown"


def test_smoother_keeps_tracks_apart_and_can_forget():
    sm = PostureSmoother(window_s=0.6)
    sm.update(1, 0.0, "stand")
    assert sm.update(2, 0.0, "lie") == "lie"
    sm.prune(active_ids={2})
    assert sm.update(1, 5.0, "sit") == "sit"
