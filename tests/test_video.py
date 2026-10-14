"""Video input (frame skipping) and output (ffmpeg or OpenCV writer)."""

import sys

import cv2
import numpy as np
import pytest

from ahjia.perception import video

pytestmark = pytest.mark.unit


def write_clip(path, n=30, fps=30.0, size=(64, 48)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), i * 8 % 255, dtype=np.uint8)
        w.write(frame)
    w.release()
    return path


def test_video_info(tmp_path):
    clip = write_clip(tmp_path / "a.mp4", n=30, fps=30.0, size=(64, 48))
    info = video.video_info(clip)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(30.0)
    assert info.frame_count == 30


def test_iter_frames_skips_to_the_target_rate(tmp_path):
    clip = write_clip(tmp_path / "a.mp4", n=30, fps=30.0)
    items = list(video.iter_frames(clip, target_fps=10.0))
    assert len(items) == 10
    ts = [t for t, _ in items]
    assert ts[0] == 0.0
    assert np.diff(ts) == pytest.approx(0.1, abs=1e-6)
    assert items[0][1].shape == (48, 64, 3)


def test_iter_all_frames_flags_the_analysed_ones(tmp_path):
    clip = write_clip(tmp_path / "a.mp4", n=30, fps=30.0)
    items = list(video.iter_all_frames(clip, analysis_fps=10.0))
    assert len(items) == 30
    assert [round(t, 3) for t, _, _ in items[:4]] == [0.0, 0.033, 0.067, 0.1]
    assert [a for _, _, a in items[:7]] == [True, False, False, True, False, False, True]
    assert sum(a for _, _, a in items) == 10


def test_iter_all_frames_rejects_bad_fps(tmp_path):
    clip = write_clip(tmp_path / "a.mp4")
    with pytest.raises(ValueError):
        list(video.iter_all_frames(clip, 0))


def test_iter_frames_never_upsamples(tmp_path):
    clip = write_clip(tmp_path / "a.mp4", n=10, fps=10.0)
    assert len(list(video.iter_frames(clip, target_fps=30.0))) == 10


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        video.video_info(tmp_path / "nope.mp4")
    with pytest.raises(FileNotFoundError):
        list(video.iter_frames(tmp_path / "nope.mp4", 10.0))


def test_iter_frames_rejects_bad_fps(tmp_path):
    clip = write_clip(tmp_path / "a.mp4")
    with pytest.raises(ValueError):
        list(video.iter_frames(clip, target_fps=0))


@pytest.mark.parametrize("use_ffmpeg", [True, False])
def test_writer_produces_a_readable_file(tmp_path, monkeypatch, use_ffmpeg):
    if use_ffmpeg and video.find_ffmpeg() is None:
        pytest.skip("ffmpeg not installed")
    if not use_ffmpeg:
        monkeypatch.setattr(video, "find_ffmpeg", lambda: None)
    out = tmp_path / "out.mp4"
    with video.open_writer(out, fps=10.0, size=(64, 48)) as wr:
        for i in range(5):
            wr.write(np.full((48, 64, 3), 40 * i, dtype=np.uint8))
    info = video.video_info(out)
    assert (info.width, info.height) == (64, 48)
    assert info.frame_count == 5


def test_writer_rejects_wrong_frame_size(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "find_ffmpeg", lambda: None)
    with video.open_writer(tmp_path / "o.mp4", fps=10.0, size=(64, 48)) as wr:
        with pytest.raises(ValueError):
            wr.write(np.zeros((10, 10, 3), dtype=np.uint8))


def _fake_ffmpeg(tmp_path, body):
    script = tmp_path / "ffmpeg"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    return str(script)


@pytest.mark.skipif(sys.platform == "win32", reason="fake ffmpeg is a shell script")
def test_ffmpeg_exit_code_at_close_raises_video_write_error(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "find_ffmpeg", lambda: _fake_ffmpeg(tmp_path, "cat > /dev/null; exit 7"))
    with pytest.raises(video.VideoWriteError, match=r"exit code 7"):
        with video.open_writer(tmp_path / "o.mp4", fps=10.0, size=(64, 48)) as wr:
            wr.write(np.zeros((48, 64, 3), dtype=np.uint8))


@pytest.mark.skipif(sys.platform == "win32", reason="fake ffmpeg is a shell script")
def test_ffmpeg_quitting_early_raises_video_write_error_on_write(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "find_ffmpeg", lambda: _fake_ffmpeg(tmp_path, "exit 9"))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # big enough to overflow the pipe once ffmpeg is gone
    with pytest.raises(video.VideoWriteError, match=r"exit code 9"):
        with video.open_writer(tmp_path / "o.mp4", fps=10.0, size=(640, 480)) as wr:
            for _ in range(200):
                wr.write(frame)


@pytest.mark.skipif(sys.platform == "win32", reason="fake ffmpeg is a shell script")
def test_error_inside_the_block_is_not_masked_by_ffmpeg(tmp_path, monkeypatch):
    """The caller's own exception wins; ffmpeg is stopped, not waited on or judged."""
    monkeypatch.setattr(video, "find_ffmpeg", lambda: _fake_ffmpeg(tmp_path, "cat > /dev/null; exit 7"))
    with pytest.raises(KeyError):
        with video.open_writer(tmp_path / "o.mp4", fps=10.0, size=(64, 48)):
            raise KeyError("caller bug")


def test_opencv_writer_that_cannot_open_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "find_ffmpeg", lambda: None)

    class Closed:
        def __init__(self, *args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass
    monkeypatch.setattr(video.cv2, "VideoWriter", Closed)
    with pytest.raises(video.VideoWriteError, match="cannot open"):
        video.open_writer(tmp_path / "o.mp4", fps=10.0, size=(64, 48))
