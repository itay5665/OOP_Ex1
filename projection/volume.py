"""In-memory volume. Spacing is stored so it can be reported, and is not used."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ALLOWED = (np.uint8, np.uint16, np.uint32)
_MAX_AXIS = 1024


@dataclass
class Volume:
    """A Fortran-ordered ``[x, y, z]`` array plus unused spatial metadata.

    The array is converted to Fortran order on construction so the sampler can
    take a view of the linear buffer (first index fastest).
    """

    data: np.ndarray
    spacing: tuple[float, float, float] | None = None
    origin: tuple[float, float, float] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise ValueError("volume data must be a numpy array")
        self.data = np.asfortranarray(self.data)
        if self.data.ndim != 3:
            raise ValueError(
                f"volume must be 3D, got {self.data.ndim}D shape {self.data.shape}"
            )
        if self.data.dtype not in _ALLOWED:
            raise ValueError(
                f"volume dtype must be uint8, uint16, or uint32, got {self.data.dtype}"
            )
        if any(axis < 1 or axis >= _MAX_AXIS for axis in self.data.shape):
            raise ValueError(
                f"each axis length must satisfy 1 <= n < {_MAX_AXIS}, got {self.data.shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
