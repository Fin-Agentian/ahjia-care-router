"""Overlay drawing on synthetic frames."""

import numpy as np
import pytest

from ahjia.events import HudState, PersonObs
from ahjia.overlay import CAPTION, alert_color, draw_overlay, to_ascii
from ahjia.perception.zones import build_zone_map
from perception_samples import lying_flat, standing

pytestmark = pytest.mark.unit

SIZES = [(1280, 720), (768, 448)]
HUD = HudState(
    agent="fall",
    decision="A1+A4+A3",
    reason="lying outside bed/sofa >= 2 s",
    latency="1.9 s (simulated)",
    cameras="4 on | living 15 fps",
    genai="off (night)",
    alert="pre-alert",
)


def blank(w, h, value=60):
    return np.full((h, w, 3), value, dtype=np.uint8)


def person(sample, posture="stand", zone="floor", track_id=1):
    kps, bbox = sample
    return PersonObs(track_id=track_id, identity="elder", posture=posture, zone=zone,
                     bbox=bbox, keypoints=kps)


def count_color(img, color):
    return int(np.all(img == np.array(color, dtype=np.uint8), axis=-1).sum())


@pytest.mark.parametrize("w,h", SIZES)
def test_hud_is_drawn_top_right_inside_the_frame(w, h):
    src = blank(w, h)
    out = draw_overlay(src, (), HUD)
    assert out.shape == src.shape
    diff = np.any(out != src, axis=-1)
    ys, xs = np.nonzero(diff[: h - int(0.12 * h)])  # ignore the caption strip
    assert xs.min() > w * 0.45  # right half only
    assert ys.min() < h * 0.1  # starts near the top
    assert ys.max() < h * 0.55 and xs.max() < w


def test_input_frame_is_not_modified():
    src = blank(640, 360)
    before = src.copy()
    draw_overlay(src, (person(standing()),), HUD)
    assert np.array_equal(src, before)


def test_caption_is_drawn_at_the_bottom_and_is_ascii():
    assert to_ascii(CAPTION).isascii()
    assert "Hailo-10H timing simulated" in CAPTION
    src = blank(1280, 720)
    out = draw_overlay(src, (), HUD)
    assert np.any(out[-40:] != src[-40:])


@pytest.mark.parametrize(
    "alert, expected",
    [("none", (80, 200, 80)), ("pre-alert sent", (0, 170, 255)),
     ("escalated", (60, 60, 230)), ("cleared by elder", (230, 150, 40))],
)
def test_alert_color_maps_state_to_accent(alert, expected):
    assert alert_color(alert) == expected


@pytest.mark.parametrize("alert", ["none", "pre_alert", "escalated", "cleared_by_app"])
def test_hud_border_uses_the_alert_color(alert):
    out = draw_overlay(blank(1280, 720), (), HudState(alert=alert))
    assert count_color(out[:300, 640:], alert_color(alert)) > 100


def test_person_skeleton_label_and_zone_are_drawn():
    src = blank(1280, 720)
    p = person(lying_flat(), posture="lie", zone="floor")
    out = draw_overlay(src, (p,), HudState())
    x1, y1, x2, y2 = p.bbox
    region = (slice(int(y1 * 720) - 40, int(y2 * 720) + 5), slice(int(x1 * 1280) - 5, int(x2 * 1280) + 5))
    assert np.any(out[region] != src[region])


def test_person_without_keypoints_or_bbox_is_skipped_safely():
    src = blank(640, 360)
    draw_overlay(src, (PersonObs(track_id=3), person(standing())), HudState())


def test_zone_outlines_are_subtle_but_visible():
    zones = build_zone_map({"sofa": [(0.05, 0.55), (0.35, 0.55), (0.35, 0.95), (0.05, 0.95)]})
    src = blank(1280, 720)
    out = draw_overlay(src, (), HudState(), zones=zones)
    left = np.any(out[:640, :640] != src[:640, :640], axis=-1)
    assert left.any()
    assert left.mean() < 0.02  # thin lines only


def test_status_text_top_left():
    src = blank(1280, 720)
    out = draw_overlay(src, (), HudState(), status="T+12.3s  clock 14:05:12")
    assert np.any(out[:60, :400] != src[:60, :400])


def test_long_hud_lines_are_truncated_to_fit():
    hud = HudState(reason="x" * 300)
    out = draw_overlay(blank(768, 448), (), hud)
    assert out.shape == (448, 768, 3)


def test_to_ascii():
    assert to_ascii("4 on · living") == "4 on | living"
    assert to_ascii("a → b ≥ c") == "a -> b >= c"
    assert to_ascii("救命") == "??"
    assert to_ascii("plain") == "plain"
