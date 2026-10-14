"""Draw the demo overlay: skeletons, posture labels, zones and the 7-line HUD.

Shapes are drawn with OpenCV; text goes through ahjia.overlay_text (TrueType
via Pillow, Hershey as a fallback). The HUD is a semi-transparent dark panel at
the top right in a monospace font, whose border follows the alert state (green
none, amber pre-alert, red escalated, blue cleared). The status line and the
bottom caption sit on dark strips, so they stay readable on any footage.
Sizes scale with the frame height; 1280x720 and 768x448 both stay readable.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from ahjia.events import HudState, PersonObs
from ahjia.overlay_text import Color, TextItem, TextRenderer, default_renderer, to_ascii
from ahjia.perception.pose import SKELETON
from ahjia.perception.zones import ZoneMap

__all__ = ["CAPTION", "alert_color", "draw_overlay", "to_ascii"]

CAPTION = "Laptop prototype · pose runs live · Hailo-10H timing simulated"
HUD_TITLE = "AhJia Care Router"
HUD_MAX_CHARS = 44

GREEN: Color = (80, 200, 80)
AMBER: Color = (0, 170, 255)
RED: Color = (60, 60, 230)
BLUE: Color = (230, 150, 40)
WHITE: Color = (245, 245, 245)
PANEL: Color = (24, 24, 24)
POSTURE_COLORS: dict[str, Color] = {
    "stand": (90, 210, 90), "sit": (230, 200, 60), "lie": (70, 90, 255), "unknown": (170, 170, 170),
}


def alert_color(alert: str) -> Color:
    """Accent color for the HUD border from the free-text alert line."""
    a = alert.lower()
    if "escalat" in a:
        return RED
    if "clear" in a:
        return BLUE
    if "pre" in a:
        return AMBER
    return GREEN


def _scale(frame: np.ndarray) -> float:
    return max(0.85, frame.shape[0] / 720.0)


def _fit(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 2] + ".."


def _dim_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], alpha: float,
              color: Color = PANEL) -> None:
    """Blend a dark rectangle into the image (clipped to the frame)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(w, p2[0]), min(h, p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0)


def _draw_zones(img: np.ndarray, zones: ZoneMap, scale: float, items: list[TextItem]) -> None:
    h, w = img.shape[:2]
    layer = img.copy()
    size = int(12 * scale)
    for name, poly in zones.polygons.items():
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
        cv2.polylines(layer, [pts], True, (210, 210, 210), 1, cv2.LINE_AA)
        items.append(TextItem(name, int(pts[0][0]) + 5, int(pts[0][1]) + size + 3, size,
                              (205, 205, 205), outline=1))
    cv2.addWeighted(layer, 0.55, img, 0.45, 0, img)


def _draw_skeleton(img: np.ndarray, kps: Sequence[tuple[float, float, float]], color: Color,
                   scale: float, min_conf: float = 0.3) -> None:
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) if c >= min_conf else None for x, y, c in kps]
    thick = max(1, int(round(2 * scale)))
    for a, b in SKELETON:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], color, thick, cv2.LINE_AA)
    for pt in pts:
        if pt is not None:
            cv2.circle(img, pt, thick + 1, WHITE, -1, cv2.LINE_AA)


def _draw_person(img: np.ndarray, p: PersonObs, scale: float, tr: TextRenderer,
                 items: list[TextItem]) -> None:
    h, w = img.shape[:2]
    color = POSTURE_COLORS.get(p.posture, POSTURE_COLORS["unknown"])
    if p.keypoints is not None:
        _draw_skeleton(img, p.keypoints, color, scale)
    if p.bbox is None:
        return
    x1, y1, x2, y2 = int(p.bbox[0] * w), int(p.bbox[1] * h), int(p.bbox[2] * w), int(p.bbox[3] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(round(scale))), cv2.LINE_AA)
    size = int(15 * scale)
    label = f"#{p.track_id} {p.identity} · {p.posture} · {p.zone}"
    label = label if tr.unicode_ok else to_ascii(label)
    tw = tr.measure(label, "sans", size)
    base = y1 - int(size * 0.45) if y1 - size * 1.6 > 0 else min(h - 4, y2 + int(size * 1.4))
    lx = max(4, min(x1, w - tw - 6))
    _dim_rect(img, (lx - 4, base - int(size * 1.0)), (lx + tw + 4, base + int(size * 0.35)), 0.7,
              (0, 0, 0))
    items.append(TextItem(label, lx, base, size, color))


def _hud_size(lines: Sequence[str], tr: TextRenderer, w: int, scale: float) -> tuple[int, int, int]:
    """Largest font size (px) whose widest line fits the width budget; returns (size, text_w, pad)."""
    budget = (0.38 if w >= 1100 else 0.47) * w
    pad = int(12 * scale)
    size = int(20 * scale)
    while True:
        text_w = max(tr.measure(s, "mono", size) for s in lines)
        if text_w + 2 * pad <= budget or size <= 9:
            return size, text_w, pad
        size -= 1


def _draw_hud(img: np.ndarray, hud: HudState, scale: float, tr: TextRenderer,
              items: list[TextItem]) -> None:
    h, w = img.shape[:2]
    prep = (lambda s: s) if tr.unicode_ok else to_ascii
    lines = [_fit(prep(line), HUD_MAX_CHARS) for line in hud.lines()]
    size, text_w, pad = _hud_size([HUD_TITLE] + lines, tr, w, scale)
    pitch = int(size * 1.5)
    box_w, box_h = text_w + 2 * pad, pitch * (len(lines) + 1) + pad
    x0, y0 = w - box_w - int(14 * scale), int(14 * scale)
    accent = alert_color(hud.alert)
    _dim_rect(img, (x0, y0), (x0 + box_w, y0 + box_h), 0.74)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), accent, max(2, int(2 * scale)))
    for i, line in enumerate([HUD_TITLE] + lines):
        base = y0 + pad // 2 + i * pitch + pitch // 2 + int(0.36 * size)
        items.append(TextItem(line, x0 + pad, base, size, accent if i == 0 else WHITE, kind="mono"))


def _draw_status(img: np.ndarray, status: str, scale: float, tr: TextRenderer,
                 items: list[TextItem]) -> None:
    size = int(15 * scale)
    x, base = int(14 * scale), int(14 * scale) + size
    tw = tr.measure(status, "sans", size)
    _dim_rect(img, (x - 7, base - size - 5), (x + tw + 7, base + int(size * 0.4)), 0.6)
    items.append(TextItem(status, x, base, size, WHITE))


def _draw_caption(img: np.ndarray, caption: str, scale: float, tr: TextRenderer,
                  items: list[TextItem]) -> None:
    h, w = img.shape[:2]
    size = int(14 * scale)
    while tr.measure(caption, "sans", size) > w - 24 and size > 8:
        size -= 1
    strip = int(size * 2.2)
    _dim_rect(img, (0, h - strip), (w, h), 0.6)
    tw = tr.measure(caption, "sans", size)
    base = h - strip + (strip + int(0.72 * size)) // 2
    items.append(TextItem(caption, (w - tw) // 2, base, size, (230, 230, 230)))


def draw_overlay(
    frame: np.ndarray,
    persons: Sequence[PersonObs],
    hud: HudState,
    zones: Optional[ZoneMap] = None,
    status: Optional[str] = None,
    caption: str = CAPTION,
    text: Optional[TextRenderer] = None,
) -> np.ndarray:
    """Return a new frame with zones, people, the HUD, a status line and the caption."""
    img = frame.copy()
    tr = text or default_renderer()
    prep = (lambda s: s) if tr.unicode_ok else to_ascii
    scale = _scale(img)
    items: list[TextItem] = []
    if zones is not None:
        _draw_zones(img, zones, scale, items)
    for p in persons:
        _draw_person(img, p, scale, tr, items)
    tr.render(img, items)  # zone names and person labels first, so the HUD panel sits on top
    items = []
    _draw_hud(img, hud, scale, tr, items)
    if status:
        _draw_status(img, prep(status), scale, tr, items)
    _draw_caption(img, prep(caption), scale, tr, items)
    tr.render(img, items)
    return img
