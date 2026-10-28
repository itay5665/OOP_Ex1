"""Global 8-bit normalization."""

import numpy as np

from projection.normalizer import Normalizer
from projection.volume import Volume


def test_minmax_endpoints_and_constant_volume():
    volume = Volume(np.array([[[0]], [[255]]], dtype=np.uint8).reshape(2, 1, 1))
    normalizer = Normalizer.from_volume(volume, "minmax")
    values = np.array([[0.0, 255.0, 127.5]], dtype=np.float32)
    mask = np.array([[True, True, True]])
    out = normalizer.apply(values, mask)
    assert out[0, 0] == 0
    assert out[0, 1] == 255
    assert out[0, 2] == 128

    constant = Normalizer.from_volume(Volume(np.full((2, 2, 2), 7, dtype=np.uint8)))
    painted = constant.apply(np.full((2, 2), 7.0), np.ones((2, 2), dtype=bool))
    assert np.all(painted == 0)


def test_percentile_clips_and_outside_stays_black():
    data = np.zeros((100,), dtype=np.uint8)
    data[:98] = 10
    data[98] = 0
    data[99] = 250
    volume = Volume(data.reshape(100, 1, 1))
    normalizer = Normalizer.from_volume(volume, "percentile")
    # NumPy interpolates the 1st percentile, so it sits just under the mass at 10.
    assert 0 <= normalizer.low <= 10
    assert normalizer.high > normalizer.low

    values = np.array([[50.0, 50.0]], dtype=np.float32)
    mask = np.array([[False, True]])
    mapped = Normalizer(0.0, 100.0).apply(values, mask)
    assert mapped[0, 0] == 0
    assert mapped[0, 1] == 128
