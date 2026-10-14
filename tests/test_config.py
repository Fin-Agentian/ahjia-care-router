"""Camera config and speech file loaders."""

from pathlib import Path

import pytest

from ahjia.perception.config import load_camera_config, load_speech

pytestmark = pytest.mark.unit

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_example_camera_config_loads():
    cam = load_camera_config(CONFIGS / "camera_example.yaml")
    assert cam.name == "living"
    assert cam.start_clock == "14:05:00"
    assert cam.identity == "elder"
    assert cam.zones.zone_at(0.2, 0.7) == "sofa"
    assert cam.zones.zone_at(0.8, 0.7) == "bed"
    assert cam.zones.zone_at(0.5, 0.2) == "floor"


def test_example_speech_loads_in_time_order():
    items = load_speech(CONFIGS / "example.speech.yaml")
    assert [i.t for i in items] == [21.5, 41.0]
    assert items[0].speech.text == "I can't get up"
    assert items[1].speech.directed is True


def test_speech_none_and_empty(tmp_path):
    assert load_speech(None) == ()
    empty = tmp_path / "e.yaml"
    empty.write_text("")
    assert load_speech(empty) == ()


def test_speech_is_sorted(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("- {t: 5, text: b}\n- {t: 1, text: a, source: family, lang: zh}\n")
    items = load_speech(f)
    assert [i.speech.text for i in items] == ["a", "b"]
    assert items[0].speech.source == "family" and items[0].speech.lang == "zh"


@pytest.mark.parametrize("body", ["- {t: 1}", "- {text: hi}", "- {t: 1, text: hi, source: alien}", "t: 1"])
def test_bad_speech_is_rejected(tmp_path, body):
    f = tmp_path / "s.yaml"
    f.write_text(body)
    with pytest.raises(ValueError):
        load_speech(f)


@pytest.mark.parametrize(
    "body",
    ["zones: {}", "camera: a\nstart_clock: noon", "camera: a\nidentity: robot", "- 1"],
)
def test_bad_camera_config_is_rejected(tmp_path, body):
    f = tmp_path / "c.yaml"
    f.write_text(body)
    with pytest.raises(ValueError):
        load_camera_config(f)


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_camera_config(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError):
        load_speech(tmp_path / "nope.yaml")
