"""Text for the overlay: TrueType through Pillow, with an OpenCV Hershey fallback.

Fonts are looked up on the machine (Linux, WSL, Windows, macOS); no font files
ship with the repo. Set AHJIA_FONT_DIR to a folder to try its fonts first.
With a TrueType font, text keeps its Unicode symbols (middle dot, >=, arrow).
Only the Hershey fallback is ASCII-only, so it substitutes symbols and turns
anything else into '?'.

Text is queued as TextItem records and drawn in one pass per layer, so a frame
is converted between OpenCV and Pillow only a couple of times.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence

import cv2
import numpy as np

Color = tuple[int, int, int]  # BGR
Kind = Literal["mono", "sans"]

FONT_DIRS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/TTF",
    "/usr/share/fonts/dejavu",
    "/usr/share/fonts/liberation",
    "C:/Windows/Fonts",
    "/mnt/c/Windows/Fonts",
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
    "/Library/Fonts",
)
MONO_FONTS: tuple[str, ...] = (
    "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "consola.ttf", "CascadiaMono.ttf", "Menlo.ttc",
)
SANS_FONTS: tuple[str, ...] = (
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "segoeui.ttf", "arial.ttf", "Helvetica.ttc",
)
# Only used for strings that contain CJK characters, which the fonts above lack.
CJK_FONTS: tuple[str, ...] = (
    "NotoSansCJK-Regular.ttc", "msjh.ttc", "msyh.ttc", "PingFang.ttc", "Hiragino Sans GB.ttc",
)
_CJK_START = 0x2E80

_ASCII_MAP = str.maketrans({
    "\u00b7": "|", "\u2022": "|", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00d7": "x", "\u2026": "...",
})
_ASCII_WORDS = {"\u2192": "->", "\u2265": ">=", "\u2264": "<="}


def to_ascii(text: str) -> str:
    """Make text drawable by Hershey: common symbols are mapped, anything else becomes '?'."""
    for src, dst in _ASCII_WORDS.items():
        text = text.replace(src, dst)
    return "".join(ch if ch.isascii() else "?" for ch in text.translate(_ASCII_MAP))


@dataclass(frozen=True)
class FontPaths:
    """Font files found on this machine; None means "not found"."""

    mono: Optional[str] = None
    sans: Optional[str] = None
    cjk: Optional[str] = None

    @property
    def any_found(self) -> bool:
        return self.mono is not None or self.sans is not None


def _first_existing(names: Sequence[str], dirs: Sequence[str]) -> Optional[str]:
    for directory in dirs:
        for name in names:
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def discover_fonts(extra_dir: Optional[str] = None, search_dirs: Sequence[str] = FONT_DIRS) -> FontPaths:
    """Find a monospace, a sans and a CJK font. extra_dir is searched before the standard folders."""
    dirs = ([extra_dir] if extra_dir else []) + list(search_dirs)
    return FontPaths(
        mono=_first_existing(MONO_FONTS, dirs),
        sans=_first_existing(SANS_FONTS, dirs),
        cjk=_first_existing(CJK_FONTS, dirs),
    )


@dataclass(frozen=True)
class TextItem:
    """One piece of text. (x, y) is the left end of the baseline; size is the font size in pixels."""

    text: str
    x: int
    y: int
    size: int
    color: Color
    kind: Kind = "sans"
    outline: int = 0  # dark outline width in pixels, same font and size


class TextRenderer(Protocol):
    unicode_ok: bool

    def measure(self, text: str, kind: Kind, size: int) -> int: ...

    def render(self, img: np.ndarray, items: Sequence[TextItem]) -> None: ...


@lru_cache(maxsize=64)
def _load_font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


def _has_cjk(text: str) -> bool:
    return any(ord(ch) >= _CJK_START for ch in text)


class TtfText:
    """Pillow TrueType renderer."""

    unicode_ok = True

    def __init__(self, paths: FontPaths) -> None:
        if not paths.any_found:
            raise ValueError("no TrueType font available")
        self._mono = paths.mono or paths.sans
        self._sans = paths.sans or paths.mono
        self._cjk = paths.cjk

    def _font(self, text: str, kind: Kind, size: int):
        path = self._cjk if self._cjk and _has_cjk(text) else (self._mono if kind == "mono" else self._sans)
        return _load_font(str(path), max(6, size))

    def measure(self, text: str, kind: Kind, size: int) -> int:
        return int(round(self._font(text, kind, size).getlength(text)))

    def render(self, img: np.ndarray, items: Sequence[TextItem]) -> None:
        if not items:
            return
        from PIL import Image, ImageDraw

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for it in items:
            draw.text(
                (it.x, it.y), it.text, font=self._font(it.text, it.kind, it.size),
                fill=(it.color[2], it.color[1], it.color[0]), anchor="ls",
                stroke_width=it.outline, stroke_fill=(0, 0, 0),
            )
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class HersheyText:
    """OpenCV fallback when no TrueType font exists. ASCII only; mono is drawn on a fixed grid."""

    unicode_ok = False
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _scale(size: int) -> float:
        return max(0.3, size / 30.0)

    def _cell(self, size: int) -> int:
        (mw, _), _ = cv2.getTextSize("M", self._FONT, self._scale(size), 1)
        return max(5, int(mw * 0.82))

    def measure(self, text: str, kind: Kind, size: int) -> int:
        if kind == "mono":
            return len(text) * self._cell(size)
        return cv2.getTextSize(text, self._FONT, self._scale(size), 1)[0][0]

    def render(self, img: np.ndarray, items: Sequence[TextItem]) -> None:
        for it in items:
            if it.outline:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    self._put(img, it, it.x + dx, it.y + dy, (0, 0, 0))
            self._put(img, it, it.x, it.y, it.color)

    def _put(self, img: np.ndarray, it: TextItem, x: int, y: int, color: Color) -> None:
        fs = self._scale(it.size)
        if it.kind == "mono":
            cell = self._cell(it.size)
            for i, ch in enumerate(it.text):
                if ch != " ":
                    cv2.putText(img, ch, (x + i * cell, y), self._FONT, fs, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, it.text, (x, y), self._FONT, fs, color, 1, cv2.LINE_AA)


def make_text_renderer(paths: FontPaths) -> TextRenderer:
    """TrueType when a font was found, else the Hershey fallback."""
    return TtfText(paths) if paths.any_found else HersheyText()


@lru_cache(maxsize=1)
def default_renderer() -> TextRenderer:
    return make_text_renderer(discover_fonts(os.environ.get("AHJIA_FONT_DIR")))
