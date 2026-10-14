"""Synthetic people for perception tests (no model weights needed).

Poses are built in pixel space on a 1280x720 frame and then normalized,
so aspect-ratio effects are the same as with a real 16:9 video.
"""

from __future__ import annotations

FRAME_W, FRAME_H = 1280, 720
ASPECT = FRAME_W / FRAME_H

# COCO order: nose, eyes(2), ears(2), shoulders(2), elbows(2), wrists(2), hips(2), knees(2), ankles(2)


def _spread(cx: float, y: float, half: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return (cx - half, y), (cx + half, y)


def _finish(points: dict[str, tuple[float, float]], conf: float):
    """Fill the 17 COCO points from the key landmarks and normalize."""
    nose = points["nose"]
    ls, rs = points["ls"], points["rs"]
    lh, rh = points["lh"], points["rh"]
    lk, rk = points["lk"], points["rk"]
    la, ra = points["la"], points["ra"]
    px = [
        nose, nose, nose, nose, nose,  # nose, eyes, ears collapse onto the nose
        ls, rs,
        ls, rs,  # elbows (not used by the heuristic)
        ls, rs,  # wrists
        lh, rh, lk, rk, la, ra,
    ]
    kps = tuple((x / FRAME_W, y / FRAME_H, conf) for x, y in px)
    xs = [p[0] for p in px]
    ys = [p[1] for p in px]
    pad = 12
    bbox = (
        max(0.0, (min(xs) - pad) / FRAME_W),
        max(0.0, (min(ys) - pad) / FRAME_H),
        min(1.0, (max(xs) + pad) / FRAME_W),
        min(1.0, (max(ys) + pad) / FRAME_H),
    )
    return kps, bbox


def standing(cx: float = 640, ground: float = 650, h: float = 300, conf: float = 0.9):
    """Upright person, height h px, camera at eye level."""
    ls, rs = _spread(cx, ground - 0.82 * h, 0.13 * h)
    lh, rh = _spread(cx, ground - 0.52 * h, 0.09 * h)
    lk, rk = _spread(cx, ground - 0.28 * h, 0.09 * h)
    la, ra = _spread(cx, ground, 0.09 * h)
    pts = {"nose": (cx, ground - 0.93 * h), "ls": ls, "rs": rs, "lh": lh, "rh": rh,
           "lk": lk, "rk": rk, "la": la, "ra": ra}
    return _finish(pts, conf)


def standing_high_angle(cx: float = 640, ground: float = 650, h: float = 300, squash: float = 0.6):
    """Upright person seen from a high ceiling corner: vertical extent squashed."""
    s = squash
    ls, rs = _spread(cx, ground - 0.82 * h * s, 0.13 * h)
    lh, rh = _spread(cx, ground - 0.52 * h * s, 0.09 * h)
    lk, rk = _spread(cx, ground - 0.28 * h * s, 0.09 * h)
    la, ra = _spread(cx, ground, 0.09 * h)
    pts = {"nose": (cx, ground - 0.93 * h * s), "ls": ls, "rs": rs, "lh": lh, "rh": rh,
           "lk": lk, "rk": rk, "la": la, "ra": ra}
    return _finish(pts, 0.9)


def sitting(cx: float = 400, hip_y: float = 520, h: float = 300, legs_visible: bool = True):
    """Seated on a chair or sofa, facing right; thighs horizontal."""
    ls, rs = _spread(cx, hip_y - 0.30 * h, 0.13 * h)
    lh, rh = _spread(cx, hip_y, 0.09 * h)
    lk, rk = _spread(cx + 0.25 * h, hip_y + 0.02 * h, 0.09 * h)
    la, ra = _spread(cx + 0.25 * h, hip_y + 0.30 * h, 0.09 * h)
    pts = {"nose": (cx + 0.03 * h, hip_y - 0.45 * h), "ls": ls, "rs": rs, "lh": lh, "rh": rh,
           "lk": lk, "rk": rk, "la": la, "ra": ra}
    kps, bbox = _finish(pts, 0.9)
    if legs_visible:
        return kps, bbox
    # Legs hidden behind furniture: low confidence, box covers the upper body only.
    low = tuple((x, y, 0.05) if i >= 13 else (x, y, c) for i, (x, y, c) in enumerate(kps))
    ymax = (hip_y + 0.1 * h + 12) / FRAME_H
    return low, (bbox[0], bbox[1], bbox[2], min(bbox[3], ymax))


def lying_flat(cx: float = 640, y: float = 600, h: float = 300):
    """Lying on the back, body axis horizontal in the image, head to the left."""
    pts = {
        "nose": (cx - 0.48 * h, y),
        "ls": (cx - 0.30 * h, y - 0.13 * h), "rs": (cx - 0.30 * h, y + 0.13 * h),
        "lh": (cx, y - 0.09 * h), "rh": (cx, y + 0.09 * h),
        "lk": (cx + 0.25 * h, y - 0.09 * h), "rk": (cx + 0.25 * h, y + 0.09 * h),
        "la": (cx + 0.50 * h, y - 0.09 * h), "ra": (cx + 0.50 * h, y + 0.09 * h),
    }
    return _finish(pts, 0.85)


def lying_toward_camera(cx: float = 640, y: float = 450, h: float = 300, squash: float = 0.3):
    """Lying with the body axis along the view direction: heavily foreshortened.

    The torso looks vertical in the image, but it is short compared with the
    shoulder width, which is the cue that separates it from standing.
    """
    s = squash
    ls, rs = _spread(cx, y + 0.18 * h * s, 0.13 * h)
    lh, rh = _spread(cx, y + 0.48 * h * s, 0.09 * h)
    lk, rk = _spread(cx, y + 0.72 * h * s, 0.09 * h)
    la, ra = _spread(cx, y + 1.0 * h * s, 0.09 * h)
    pts = {"nose": (cx, y), "ls": ls, "rs": rs, "lh": lh, "rh": rh,
           "lk": lk, "rk": rk, "la": la, "ra": ra}
    return _finish(pts, 0.85)


def low_confidence(base, conf: float = 0.05):
    """Same pose and box, but every keypoint is below the confidence floor."""
    kps, bbox = base
    return tuple((x, y, conf) for x, y, _ in kps), bbox
