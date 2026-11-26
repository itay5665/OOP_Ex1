"""Map volume samples to 8-bit grayscale with one range for the whole run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projection.volume import Volume


@dataclass
class Normalizer:
    """Linear map from ``[low, high]`` onto ``[0, 255]``.

    Pixels that are not in the volume stay 0, even when ``low`` is above 0.
    A zero range (``high <= low``) maps every in-volume sample to 0.
    """

    low: float
    high: float

    @classmethod
    def from_volume(cls, volume: Volume, mode: str = "minmax") -> Normalizer:
        if mode == "minmax":
            low = float(volume.data.min())
            high = float(volume.data.max())
        elif mode == "percentile":
            low, high = np.percentile(volume.data, [1, 99])
            low = float(low)
            high = float(high)
        else:
            raise ValueError(f"unknown normalize mode {mode!r}")
        return cls(low=low, high=high)

    def apply(self, values: np.ndarray, in_volume: np.ndarray) -> np.ndarray:
        """Return a ``uint8`` image. ``values`` and ``in_volume`` share a shape."""
        out = np.zeros(values.shape, dtype=np.uint8)
        if self.high <= self.low:
            return out
        scaled = (values.astype(np.float32) - np.float32(self.low)) / np.float32(
            self.high - self.low
        )
        scaled = np.clip(scaled, 0.0, 1.0)
        as_u8 = np.rint(scaled * np.float32(255.0)).astype(np.uint8)
        out[in_volume] = as_u8[in_volume]
        return out
