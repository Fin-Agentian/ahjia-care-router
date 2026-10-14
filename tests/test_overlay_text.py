"""Font discovery and the TrueType / Hershey text renderers."""

import numpy as np
import pytest

from ahjia import overlay_text as ot
from ahjia.events import HudState
from ahjia.overlay import CAPTION, draw_overlay

pytestmark = pytest.mark.unit

FONTS = ot.discover_fonts()
needs_ttf = pytest.mark.skipif(not FONTS.any_found, reason="no TrueType font on this machine")


def test_discovery_finds_nothing_in_empty_dirs(tmp_path):
    paths = ot.discover_fonts(search_dirs=[str(tmp_path / "missing"), str(tmp_path)])
    assert paths == ot.FontPaths(None, None, None)
    assert not paths.any_found


def test_discovery_extra_dir_wins_over_standard_dirs(tmp_path):
    (tmp_path / "LiberationMono-Regular.ttf").write_bytes(b"x")
    (tmp_path / "arial.ttf").write_bytes(b"x")
    (tmp_path / "msjh.ttc").write_bytes(b"x")
    paths = ot.discover_fonts(extra_dir=str(tmp_path))
    assert paths.mono == str(tmp_path / "LiberationMono-Regular.ttf")
    assert paths.sans == str(tmp_path / "arial.ttf")
    assert paths.cjk == str(tmp_path / "msjh.ttc")


def test_discovery_prefers_earlier_names_in_one_dir(tmp_path):
    for name in ("consola.ttf", "DejaVuSansMono.ttf"):
        (tmp_path / name).write_bytes(b"x")
    assert ot.discover_fonts(search_dirs=[str(tmp_path)]).mono.endswith("DejaVuSansMono.ttf")


def test_default_renderer_reads_the_env_override(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ot, "discover_fonts", lambda extra: seen.setdefault("dir", extra) and ot.FontPaths())
    monkeypatch.setenv("AHJIA_FONT_DIR", str(tmp_path))
    ot.default_renderer.cache_clear()
    try:
        assert isinstance(ot.default_renderer(), ot.HersheyText)
        assert seen["dir"] == str(tmp_path)
    finally:
        ot.default_renderer.cache_clear()


def test_no_fonts_gives_the_hershey_fallback_and_ttf_refuses():
    assert isinstance(ot.make_text_renderer(ot.FontPaths()), ot.HersheyText)
    with pytest.raises(ValueError):
        ot.TtfText(ot.FontPaths())


def test_hershey_renderer_draws_ascii_text_with_a_clean_outline():
    r = ot.HersheyText()
    img = np.zeros((60, 300, 3), dtype=np.uint8)
    r.render(img, [ot.TextItem("Alert : none", 10, 40, 20, (255, 255, 255), kind="mono", outline=1),
                   ot.TextItem("caption", 10, 20, 14, (200, 200, 200))])
    assert img.any()
    assert r.measure("abcd", "mono", 20) == 4 * r.measure("a", "mono", 20)
    assert r.measure("abc", "sans", 20) > 0
    assert not r.unicode_ok


def test_overlay_with_hershey_fallback_is_ascii_safe():
    img = np.full((448, 768, 3), 60, dtype=np.uint8)
    hud = HudState(cameras="4 on · living", reason="救命 ≥ 2 s")
    out = draw_overlay(img, (), hud, text=ot.HersheyText())
    assert out.shape == img.shape and np.any(out != img)


@needs_ttf
def test_ttf_measures_monospace_evenly_and_keeps_unicode():
    r = ot.make_text_renderer(FONTS)
    assert isinstance(r, ot.TtfText) and r.unicode_ok
    assert r.measure("iiii", "mono", 20) == r.measure("MMMM", "mono", 20)
    assert r.measure("a · b ≥ c → d", "sans", 16) > r.measure("a b c d", "sans", 16)


@needs_ttf
def test_ttf_text_is_drawn_without_a_second_shifted_copy():
    r = ot.make_text_renderer(FONTS)
    plain = np.zeros((50, 400, 3), dtype=np.uint8)
    r.render(plain, [ot.TextItem("Laptop prototype", 10, 30, 20, (255, 255, 255))])
    outlined = np.zeros_like(plain)
    r.render(outlined, [ot.TextItem("Laptop prototype", 10, 30, 20, (255, 255, 255), outline=2)])
    cols = np.nonzero(plain.any(axis=(0, 2)))[0]
    cols_o = np.nonzero(outlined.any(axis=(0, 2)))[0]
    assert cols_o.min() >= cols.min() - 2 and cols_o.max() <= cols.max() + 2  # only a 2 px halo
    assert r.measure("Laptop prototype", "sans", 20) <= cols.max() - 10 + 3


@needs_ttf
def test_ttf_overlay_keeps_the_middle_dot_in_the_caption():
    r = ot.make_text_renderer(FONTS)
    assert "·" in CAPTION
    img = np.full((448, 768, 3), 60, dtype=np.uint8)
    out = draw_overlay(img, (), HudState(cameras="4 on · a"), text=r)
    assert np.any(out != img)


@needs_ttf
def test_hud_width_budget_at_1280x720():
    r = ot.make_text_renderer(FONTS)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud = HudState(reason="x" * 60, decision="A1+A4+A3", cameras="4 on · hallway 15 fps")
    out = draw_overlay(img, (), hud, text=r)
    cols = np.nonzero(out[:300].any(axis=(0, 2)))[0]
    assert (cols.max() - cols.min()) <= 0.39 * 1280
