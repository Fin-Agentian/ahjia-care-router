"""IoU / centroid tracker."""

import pytest

from ahjia.perception.tracker import IouTracker, iou

pytestmark = pytest.mark.unit

A = (0.40, 0.20, 0.50, 0.80)
B = (0.80, 0.20, 0.90, 0.80)


def test_iou_basics():
    assert iou(A, A) == pytest.approx(1.0)
    assert iou(A, B) == 0.0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_same_person_keeps_the_id_while_moving():
    tr = IouTracker()
    assert tr.update(0.0, [A]) == [1]
    assert tr.update(0.1, [(0.41, 0.20, 0.51, 0.80)]) == [1]
    assert tr.update(0.2, [(0.43, 0.21, 0.53, 0.81)]) == [1]


def test_two_people_get_distinct_stable_ids_in_any_order():
    tr = IouTracker()
    ids = tr.update(0.0, [A, B])
    assert sorted(ids) == [1, 2]
    again = tr.update(0.1, [B, A])
    assert again == [ids[1], ids[0]]


def test_falling_person_keeps_the_id_through_a_shape_change():
    tr = IouTracker()
    standing_box = (0.45, 0.10, 0.55, 0.90)
    lying_box = (0.30, 0.65, 0.70, 0.90)
    assert tr.update(0.0, [standing_box]) == [1]
    assert tr.update(0.1, [lying_box]) == [1]


def test_track_survives_a_short_gap_but_not_a_long_one():
    tr = IouTracker(max_age_s=1.0)
    assert tr.update(0.0, [A]) == [1]
    assert tr.update(0.8, [A]) == [1]
    assert tr.update(5.0, [A]) == [2]


def test_new_person_appears_and_empty_frame_is_fine():
    tr = IouTracker()
    assert tr.update(0.0, []) == []
    assert tr.update(0.1, [A]) == [1]
    assert tr.update(0.2, [A, B]) == [1, 2]


def test_far_apart_boxes_do_not_match():
    tr = IouTracker()
    tr.update(0.0, [(0.05, 0.05, 0.10, 0.15)])
    assert tr.update(0.1, [(0.90, 0.80, 0.95, 0.90)]) == [2]
