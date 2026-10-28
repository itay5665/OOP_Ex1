"""Nearest neighbour and trilinear sampling."""

import numpy as np

from projection.samplers import NearestNeighborSampler, TrilinearSampler
from projection.volume import Volume


def test_integer_points_agree_and_midpoint_interpolates():
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[0, 0, 0] = 0
    data[1, 0, 0] = 100
    volume = Volume(data)
    integer = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    nearest = NearestNeighborSampler().sample(volume, integer)
    linear = TrilinearSampler().sample(volume, integer)
    np.testing.assert_allclose(nearest[0], linear[0])
    np.testing.assert_array_equal(nearest[0], [[0, 100]])

    mid = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)
    nn_mid, _inside = NearestNeighborSampler().sample(volume, mid)
    tri_mid, inside = TrilinearSampler().sample(volume, mid)
    assert nn_mid[0, 0] == 100
    assert tri_mid[0, 0] == 50
    assert inside[0, 0]


def test_nearest_outside_is_zero():
    volume = Volume(np.full((3, 3, 3), 7, dtype=np.uint8))
    points = np.array([[[-0.6, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    values, inside = NearestNeighborSampler().sample(volume, points)
    assert values[0, 0] == 0
    assert not inside[0, 0]
    assert values[0, 1] == 7
    assert inside[0, 1]
