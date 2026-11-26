"""Write grayscale frames to an H.264 MP4.

Each frame is replicated into three channels. ``yuv420p`` needs even dimensions;
the composer is responsible for that. ``macro_block_size=1`` stops imageio from
resizing the frame to a multiple of 16.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class VideoWriter:
    """Context manager around an imageio FFmpeg writer."""

    def __init__(self, path: str | Path, fps: float, frame_size: tuple[int, int]) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        height, width = frame_size
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"frame size must be even, got {height}x{width}")
        self.path = Path(path)
        self.fps = fps
        self.frame_size = frame_size
        self._writer = None

    def __enter__(self) -> VideoWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "18", "-preset", "ultrafast"],
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter.write called outside the context manager")
        if frame.shape != self.frame_size:
            raise ValueError(
                f"frame shape {frame.shape} does not match {self.frame_size}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
        rgb = np.stack((frame, frame, frame), axis=-1)
        self._writer.append_data(rgb)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
