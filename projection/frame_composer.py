"""Center a projection on a fixed even-sized canvas, or draw an error message."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def even_ceil(size: int) -> int:
    """Round a positive size up to an even number (required by yuv420p)."""
    if size <= 0:
        raise ValueError(f"frame size must be positive, got {size}")
    return size if size % 2 == 0 else size + 1


class FrameComposer:
    """Build frames of a fixed ``(height, width)``."""

    def __init__(self, height: int, width: int) -> None:
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"frame size must be even, got {height}x{width}")
        if height <= 0 or width <= 0:
            raise ValueError(f"frame size must be positive, got {height}x{width}")
        self.height = height
        self.width = width

    def compose(self, image: np.ndarray) -> np.ndarray:
        """Center ``image`` on a black canvas. ``image`` must fit."""
        if image.ndim != 2:
            raise ValueError(f"image must be 2D, got shape {image.shape}")
        src_h, src_w = image.shape
        if src_h > self.height or src_w > self.width:
            raise ValueError(
                f"image {src_h}x{src_w} does not fit in {self.height}x{self.width}"
            )
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        y0 = (self.height - src_h) // 2
        x0 = (self.width - src_w) // 2
        canvas[y0 : y0 + src_h, x0 : x0 + src_w] = image
        return canvas

    def error_frame(self, message: str) -> np.ndarray:
        """Black frame with ``message`` in white, wrapped to the canvas width."""
        img = Image.new("L", (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        font = _font(_font_size(self.height))
        margin = 8 if self.width > 24 and self.height > 24 else 2
        wrapped = _wrap(message, draw, font, max(1, self.width - 2 * margin))
        draw.multiline_text((margin, margin), wrapped, fill=255, font=font, spacing=4)
        return np.asarray(img, dtype=np.uint8)


def _font_size(height: int) -> int:
    if height < 40:
        return 10
    if height < 120:
        return 14
    return 18


def _font(size: int) -> ImageFont.ImageFont:
    if os.path.isfile(_FONT_PATH):
        return ImageFont.truetype(_FONT_PATH, size=size)
    return ImageFont.load_default()


def _wrap(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)
