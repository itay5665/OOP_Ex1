"""Project one quad onto a volume.

The projector does not read files. Callers pass a ``Volume`` and a ``Quad``.
All pixel coordinates are built with NumPy broadcasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projection.plane import Quad
from projection.samplers import Sampler
from projection.validation import EPS
from projection.volume import Volume


@dataclass(frozen=True)
class Projection:
    """``values`` holds raw samples. ``in_volume`` marks shape pixels that hit ``M``.

    Pixels outside the shape, and samples outside the volume, are 0.
    ``in_volume`` is false for both, so a later normalizer can leave them black.
    """

    values: np.ndarray
    in_volume: np.ndarray


class Projector:
    """Sample a quad with an injected ``Sampler``."""

    def __init__(self, sampler: Sampler) -> None:
        self.sampler = sampler

    def project(self, volume: Volume, quad: Quad) -> Projection:
        points, inside_shape = _sample_points(quad)
        values, inside_volume = self.sampler.sample(volume, points)
        in_volume = inside_shape & inside_volume
        values = np.where(in_volume, values, np.float32(0.0)).astype(np.float32, copy=False)
        return Projection(values=values, in_volume=in_volume)


def plane_size(quad: Quad) -> tuple[int, int]:
    """Return ``(height, width)`` of the image for this quad."""
    length, base = _lengths(quad)
    return int(np.floor(length)) + 1, int(np.floor(base)) + 1


def _lengths(quad: Quad) -> tuple[float, float]:
    start1 = quad.start1
    end1 = quad.end1
    start2 = quad.start2
    end2 = quad.end2
    length = float(np.linalg.norm(end1 - start1))
    base = float(np.linalg.norm(end2 - end1))
    if length < EPS:
        raise ValueError(f"leg length {length:.4f} is below EPS")
    return length, base


def _sample_points(quad: Quad) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, inside_shape)`` with points shaped ``(H, W, 3)``."""
    length, base = _lengths(quad)
    height = int(np.floor(length)) + 1
    width = int(np.floor(base)) + 1
    start1 = quad.start1.astype(np.float32, copy=False)
    start2 = quad.start2.astype(np.float32, copy=False)
    v1 = (quad.end1 - quad.start1).astype(np.float32) / np.float32(length)
    v2 = (quad.end2 - quad.start2).astype(np.float32) / np.float32(length)
    y = np.arange(height, dtype=np.float32)
    p1 = start1 + y[:, None] * v1
    p2 = start2 + y[:, None] * v2
    delta = p2 - p1
    d12 = np.sqrt(np.sum(delta * delta, axis=1))
    dp = np.zeros_like(delta)
    safe = d12 >= np.float32(EPS)
    dp[safe] = delta[safe] / d12[safe, None]
    n_samples = np.floor(d12).astype(np.int32) + np.int32(1)
    offset = np.rint((np.float32(base) - d12) / np.float32(2.0)).astype(np.int32)
    columns = np.arange(width, dtype=np.float32)
    x1 = columns[None, :] - offset[:, None].astype(np.float32)
    inside_shape = (x1 >= 0) & (x1 < n_samples[:, None])
    x1_used = np.where(inside_shape, x1, np.float32(0.0))
    points = p1[:, None, :] + x1_used[..., None] * dp[:, None, :]
    return points, inside_shape
