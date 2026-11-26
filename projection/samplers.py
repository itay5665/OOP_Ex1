"""Sampling strategies.

Both samplers return float32 values and a boolean mask of points that fall
inside the volume. Points outside the volume contribute 0.

The volume is Fortran-contiguous, so voxel ``(ix, iy, iz)`` lives at
``ix + sx * (iy + sy * iz)`` in the raveled buffer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from projection.volume import Volume


class Sampler(ABC):
    """Read voxel values at continuous coordinates."""

    @abstractmethod
    def sample(
        self, volume: Volume, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``points`` of shape ``(..., 3)``.

        Returns ``(values, inside)``. ``values`` is float32. ``inside`` is true
        where the point is inside the volume.
        """


class NearestNeighborSampler(Sampler):
    """Round each coordinate with ``floor(p + 0.5)`` and read that voxel."""

    def sample(
        self, volume: Volume, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = volume.shape
        rounded = np.floor(points + np.float32(0.5)).astype(np.int32)
        ix = rounded[..., 0]
        iy = rounded[..., 1]
        iz = rounded[..., 2]
        inside = (ix >= 0) & (ix < sx) & (iy >= 0) & (iy < sy) & (iz >= 0) & (iz < sz)
        gathered = _gather(
            volume.data,
            np.clip(ix, 0, sx - 1),
            np.clip(iy, 0, sy - 1),
            np.clip(iz, 0, sz - 1),
        )
        values = np.where(inside, gathered, np.float32(0.0))
        return values.astype(np.float32, copy=False), inside


class TrilinearSampler(Sampler):
    """Trilinear interpolation on the closed index range ``[0, size - 1]``."""

    def sample(
        self, volume: Volume, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        data = volume.data
        sx, sy, sz = volume.shape
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        inside = (
            (x >= 0)
            & (x <= (sx - 1))
            & (y >= 0)
            & (y <= (sy - 1))
            & (z >= 0)
            & (z <= (sz - 1))
        )
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        z0 = np.floor(z).astype(np.int32)
        x1 = x0 + np.int32(1)
        y1 = y0 + np.int32(1)
        z1 = z0 + np.int32(1)
        xd = (x - x0).astype(np.float32)
        yd = (y - y0).astype(np.float32)
        zd = (z - z0).astype(np.float32)

        def at(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
            return _gather(
                data,
                np.clip(ix, 0, sx - 1),
                np.clip(iy, 0, sy - 1),
                np.clip(iz, 0, sz - 1),
            )

        c000 = at(x0, y0, z0)
        c100 = at(x1, y0, z0)
        c010 = at(x0, y1, z0)
        c110 = at(x1, y1, z0)
        c001 = at(x0, y0, z1)
        c101 = at(x1, y0, z1)
        c011 = at(x0, y1, z1)
        c111 = at(x1, y1, z1)
        wx0 = np.float32(1.0) - xd
        wy0 = np.float32(1.0) - yd
        wz0 = np.float32(1.0) - zd
        c00 = c000 * wx0 + c100 * xd
        c01 = c001 * wx0 + c101 * xd
        c10 = c010 * wx0 + c110 * xd
        c11 = c011 * wx0 + c111 * xd
        c0 = c00 * wy0 + c10 * yd
        c1 = c01 * wy0 + c11 * yd
        values = c0 * wz0 + c1 * zd
        values = np.where(inside, values, np.float32(0.0))
        return values.astype(np.float32, copy=False), inside


def _gather(
    data: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray
) -> np.ndarray:
    """Gather voxels from a Fortran-ordered volume. Indices must already be in range."""
    sx, sy, _sz = data.shape
    flat = np.ravel(data, order="F")
    index = ix.astype(np.int64, copy=False)
    index = index + np.int64(sx) * (
        iy.astype(np.int64, copy=False) + np.int64(sy) * iz.astype(np.int64, copy=False)
    )
    return flat[index].astype(np.float32, copy=False)


def make_sampler(name: str) -> Sampler:
    """Build the sampler selected by the command line."""
    if name == "nearest":
        return NearestNeighborSampler()
    if name == "trilinear":
        return TrilinearSampler()
    raise ValueError(f"unknown sampler {name!r}")
