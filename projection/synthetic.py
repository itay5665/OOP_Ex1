"""Small volumes and the oblique quad used by the benchmark.

Tests and the scripts share these builders so a synthetic NRRD and a unit test
cannot drift apart.
"""

from __future__ import annotations

import numpy as np

from projection.plane import Quad

_DTYPES = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
}


def dtype_by_name(name: str) -> np.dtype:
    try:
        return np.dtype(_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"dtype must be one of {tuple(_DTYPES)}, got {name!r}") from exc


def gradient_volume(shape: tuple[int, int, int], axis: int = 0, dtype=np.uint16) -> np.ndarray:
    """``M`` equals the index along ``axis``. The length must fit in ``dtype``."""
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    info = np.iinfo(dtype)
    if shape[axis] - 1 > info.max:
        raise ValueError(
            f"axis length {shape[axis]} does not fit in {np.dtype(dtype).name}"
        )
    coords = np.arange(shape[axis], dtype=dtype)
    reshape = [1, 1, 1]
    reshape[axis] = shape[axis]
    data = np.zeros(shape, dtype=dtype)
    data[...] = coords.reshape(reshape)
    return data


def nested_cubes(shape: tuple[int, int, int], dtype=np.uint8) -> np.ndarray:
    """Nested cubes. Value grows as the Chebyshev distance to the center shrinks."""
    xs = np.arange(shape[0], dtype=np.float64)[:, None, None]
    ys = np.arange(shape[1], dtype=np.float64)[None, :, None]
    zs = np.arange(shape[2], dtype=np.float64)[None, None, :]
    cx = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cz = (shape[2] - 1) / 2.0
    dist = np.maximum(np.maximum(np.abs(xs - cx), np.abs(ys - cy)), np.abs(zs - cz))
    levels = 4
    band = np.floor((1.0 - dist / (dist.max() + 1e-6)) * levels)
    step = max(int(np.iinfo(dtype).max) // levels, 1)
    return (band * step).astype(dtype)


def sphere_volume(shape: tuple[int, int, int], dtype=np.uint8) -> np.ndarray:
    """A ball at mid-gray and a brighter core, on a black background."""
    xs = np.arange(shape[0], dtype=np.float64)[:, None, None]
    ys = np.arange(shape[1], dtype=np.float64)[None, :, None]
    zs = np.arange(shape[2], dtype=np.float64)[None, None, :]
    cx = (shape[0] - 1) / 2.0
    cy = (shape[1] - 1) / 2.0
    cz = (shape[2] - 1) / 2.0
    radius = min(shape) / 2.5
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2)
    data = np.zeros(shape, dtype=dtype)
    hi = int(np.iinfo(dtype).max)
    data[dist <= radius] = hi // 2
    data[dist <= radius / 2.0] = hi
    return data


def benchmark_quad(leg: float = 499.0) -> Quad:
    """Oblique isosceles triangle whose image is about ``leg`` by ``leg``.

    The apex sits near the origin side of a 500³ volume. The second leg is 60°
    from the x axis and tilted in z, so the plane is not axis-aligned.
    """
    start = np.array([0.0, 80.0, 100.0])
    end1 = start + np.array([leg, 0.0, 0.0])
    angle = np.pi / 3.0
    tilt = 0.6
    direction = np.array(
        [
            np.cos(angle),
            np.sin(angle) * np.cos(tilt),
            np.sin(angle) * np.sin(tilt),
        ]
    )
    direction = direction / np.linalg.norm(direction) * leg
    end2 = start + direction
    return Quad(start, end1, start, end2)


def orbit_rectangles(size: int, frames: int, half: float) -> list[np.ndarray]:
    """Rectangles through the center of a cube, rotating about the y axis.

    Each item is an array of shape ``(4, 3)`` in Q&A order ``(a1, a2, b, c)``.
    """
    if frames < 1:
        raise ValueError("frames must be positive")
    if size < 2:
        raise ValueError("size must be at least 2")
    center = np.array([(size - 1) / 2.0, (size - 1) / 2.0, (size - 1) / 2.0])
    rows: list[np.ndarray] = []
    for index in range(frames):
        angle = 2.0 * np.pi * index / frames
        u, v = _plane_basis(angle)
        a1 = center - u * half - v * half
        a2 = center - u * half + v * half
        b = center + u * half - v * half
        c = center + u * half + v * half
        rows.append(np.stack([a1, a2, b, c], axis=0))
    return rows


def _plane_basis(angle: float) -> tuple[np.ndarray, np.ndarray]:
    normal = np.array([np.sin(angle), 0.0, np.cos(angle)])
    helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, normal)
    norm_u = float(np.linalg.norm(u))
    if norm_u < 1e-8:
        u = np.cross(np.array([1.0, 0.0, 0.0]), normal)
        norm_u = float(np.linalg.norm(u))
    u = u / norm_u
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v
