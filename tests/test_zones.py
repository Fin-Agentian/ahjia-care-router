"""Zone polygons, foot point and zone lookup."""

import pytest

from ahjia.perception.zones import ZoneMap, build_zone_map, foot_point, point_in_polygon
from perception_samples import lying_flat, standing

pytestmark = pytest.mark.unit

SQUARE = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
L_SHAPE = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]


def test_point_inside_and_outside_a_square():
    assert point_in_polygon(0.4, 0.4, SQUARE)
    assert not point_in_polygon(0.7, 0.4, SQUARE)
    assert not point_in_polygon(0.4, 0.1, SQUARE)


def test_point_on_edge_and_corner_counts_as_inside():
    assert point_in_polygon(0.2, 0.4, SQUARE)
    assert point_in_polygon(0.6, 0.6, SQUARE)


def test_concave_polygon():
    assert point_in_polygon(0.25, 0.25, L_SHAPE)
    assert point_in_polygon(0.75, 0.75, L_SHAPE)
    assert not point_in_polygon(0.75, 0.25, L_SHAPE)  # the notch


def test_zone_map_first_match_and_default():
    zm = build_zone_map({"sofa": SQUARE, "bed": [(0.7, 0.2), (0.9, 0.2), (0.9, 0.6), (0.7, 0.6)]})
    assert zm.zone_at(0.4, 0.4) == "sofa"
    assert zm.zone_at(0.8, 0.4) == "bed"
    assert zm.zone_at(0.65, 0.9) == "floor"


def test_zone_map_custom_default_and_empty():
    assert build_zone_map(None, default_zone="living").zone_at(0.5, 0.5) == "living"
    assert ZoneMap(polygons={}).zone_at(0.1, 0.1) == "floor"


@pytest.mark.parametrize(
    "zones, default, message",
    [
        ({"narnia": SQUARE}, "floor", "unknown zone"),
        ({"sofa": [(0, 0), (1, 1)]}, "floor", "at least 3"),
        ({"sofa": [(0, 0), (2, 0), (1, 1)]}, "floor", "outside 0..1"),
        ({"sofa": SQUARE}, "moon", "unknown zone"),
    ],
)
def test_zone_map_rejects_bad_config(zones, default, message):
    with pytest.raises(ValueError, match=message):
        build_zone_map(zones, default_zone=default)


def test_foot_point_uses_mid_ankles_when_confident():
    kps, bbox = standing(cx=640, ground=650)
    x, y = foot_point(kps, bbox)
    assert x == pytest.approx(0.5, abs=0.01)
    assert y == pytest.approx(650 / 720, abs=0.01)


def test_foot_point_with_one_confident_ankle_uses_that_one():
    kps, bbox = standing()
    kps = tuple((x, y, 0.0) if i == 16 else (x, y, c) for i, (x, y, c) in enumerate(kps))
    x, y = foot_point(kps, bbox)
    assert x == pytest.approx(kps[15][0])


def test_foot_point_falls_back_to_bottom_center_of_box():
    kps, bbox = lying_flat()
    low = tuple((x, y, 0.0) for x, y, _ in kps)
    assert foot_point(low, bbox) == ((bbox[0] + bbox[2]) / 2, bbox[3])
    assert foot_point(None, (0.2, 0.1, 0.4, 0.9)) == (pytest.approx(0.3), 0.9)
