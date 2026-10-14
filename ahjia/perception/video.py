"""Video input at a target analysis rate, and an H.264 output writer.

Input uses OpenCV. iter_all_frames() yields every source frame with a flag for
the ones to analyse (smooth output); iter_frames() yields only the analysed ones.
Output goes through ffmpeg (libx264, yuv420p, plays everywhere) when ffmpeg is
installed, else through OpenCV's mp4v writer.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterator, Optional, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)
PathLike = Union[str, Path]


class VideoWriteError(RuntimeError):
    """The output video could not be written (ffmpeg or OpenCV failed)."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _open(path: PathLike) -> cv2.VideoCapture:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {p}")
    return cap


def video_info(path: PathLike) -> VideoInfo:
    """Source fps, size and frame count."""
    cap = _open(path)
    try:
        return VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def iter_frames(path: PathLike, target_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_seconds, BGR frame) at about target_fps; never faster than the source."""
    for t, frame, analyse in iter_all_frames(path, target_fps):
        if analyse:
            yield t, frame


def iter_all_frames(path: PathLike, analysis_fps: float) -> Iterator[tuple[float, np.ndarray, bool]]:
    """Yield (t_seconds, BGR frame, analyse) for every source frame.

    analyse is True for the frames picked for analysis at about analysis_fps
    (never faster than the source); the others are only for smooth output.
    """
    if analysis_fps <= 0:
        raise ValueError("analysis_fps must be positive")
    cap = _open(path)
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or analysis_fps
        step = max(1.0, src_fps / analysis_fps)
        index, next_index = 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            analyse = index + 1e-6 >= next_index
            if analyse:
                next_index += step
            yield index / src_fps, frame, analyse
            index += 1
    finally:
        cap.release()


def find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


class FrameWriter:
    """Writes BGR frames of one fixed size to a video file. Use as a context manager."""

    def __init__(self, path: PathLike, fps: float, size: tuple[int, int]) -> None:
        self._size = size
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._proc: Optional[subprocess.Popen] = None
        self._cv: Optional[cv2.VideoWriter] = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            self._proc = self._start_ffmpeg(ffmpeg, fps, size)
        else:
            log.warning("ffmpeg not found; writing mp4v (less compatible than H.264)")
            self._cv = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )
            if not self._cv.isOpened():
                self._cv.release()
                self._cv = None
                raise VideoWriteError(f"cannot open {self._path} for writing (OpenCV mp4v)")

    def _start_ffmpeg(self, ffmpeg: str, fps: float, size: tuple[int, int]) -> subprocess.Popen:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:g}",
            "-i", "-",
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p needs even sizes
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
            "-movflags", "+faststart", str(self._path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self._size:
            raise ValueError(f"frame size {frame.shape[1]}x{frame.shape[0]} != {self._size}")
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            except OSError as exc:  # BrokenPipeError: ffmpeg quit early
                raise self._ffmpeg_failure() from exc
        elif self._cv is not None:
            self._cv.write(frame)

    def _ffmpeg_failure(self) -> VideoWriteError:
        """Reap the dead ffmpeg process and describe the failure."""
        code = self._proc.wait() if self._proc is not None else -1
        self._proc = None
        return VideoWriteError(
            f"ffmpeg failed (exit code {code}) while writing {self._path}; "
            "its own message, if any, is printed above"
        )

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except OSError:  # ffmpeg already gone; the exit code below tells why
                    pass
            if self._proc.wait() != 0:
                raise self._ffmpeg_failure()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None

    def abort(self) -> None:
        """Stop without finishing the file (used when the caller already failed)."""
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]], exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        if exc_type is not None:
            self.abort()  # keep the original error; do not wait for or judge ffmpeg
        else:
            self.close()


def open_writer(path: PathLike, fps: float, size: tuple[int, int]) -> FrameWriter:
    """size is (width, height)."""
    return FrameWriter(path, fps, size)
