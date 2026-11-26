"""Projector results on volumes with a known answer."""

import numpy as np

from projection.plane import Quad, TrapezoidPlane, TrianglePlane
from projection.projector import Projector, plane_size
from projection.samplers import NearestNeighborSampler
from projection.synthetic import benchmark_quad, gradient_volume
from projection.volume import Volume


def _project(data, quad):
    volume = Volume(np.asarray(data))
    return Projector(NearestNeighborSampler()).project(volume, quad)


def _axis_rectangle(sx, sy, sz, k):
    return TrapezoidPlane(
        (0, 0, k),
        (0, sy - 1, k),
        (sx - 1, 0, k),
        (sx - 1, sy - 1, k),
    ).to_quad()


def test_axis_aligned_slice_matches_volume_and_gradients():
    shape = (10, 8, 5)
    k = 2
    quad = _axis_rectangle(*shape, k)
    for axis in (0, 1, 2):
        data = gradient_volume(shape, axis=axis, dtype=np.uint16)
        projection = _project(data, quad)
        if axis == 0:
            expected = np.arange(shape[0], dtype=np.float32)[:, None]
            expected = np.broadcast_to(expected, (shape[0], shape[1])).copy()
        elif axis == 1:
            expected = np.arange(shape[1], dtype=np.float32)[None, :]
            expected = np.broadcast_to(expected, (shape[0], shape[1])).copy()
        else:
            expected = np.full((shape[0], shape[1]), k, dtype=np.float32)
        np.testing.assert_allclose(projection.values, expected)
        assert projection.in_volume.all()
    raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape, order="F")
    projection = _project(raw, quad)
    np.testing.assert_array_equal(projection.values, raw[:, :, k].astype(np.float32))


def test_partial_and_full_outside_are_zero_in_the_right_place():
    data = np.full((10, 10, 10), 9, dtype=np.uint8)
    quad = TrapezoidPlane((-3, 0, 1), (-3, 4, 1), (6, 0, 1), (6, 4, 1)).to_quad()
    projection = _project(data, quad)
    assert projection.values.shape == (10, 5)
    assert np.all(projection.values[:3] == 0)
    assert np.all(projection.values[3:] == 9)
    assert not projection.in_volume[:3].any()
    assert projection.in_volume[3:].all()

    outside = Quad((0, 0, -5), (4, 0, -5), (0, 0, -5), (0, 3, -5))
    missed = _project(data, outside)
    assert not missed.in_volume.any()
    assert np.all(missed.values == 0)


def test_apex_row_is_a_single_sample_without_nan():
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[5, 5, 5] = 42
    quad = TrianglePlane((5, 5, 5), (15, 5, 5), (5, 15, 5)).to_quad()
    projection = _project(data, quad)
    assert not np.isnan(projection.values).any()
    row = projection.values[0]
    assert row.sum() == 42
    assert np.count_nonzero(row) == 1


def test_uint_family_agrees():
    shape = (6, 5, 4)
    quad = _axis_rectangle(*shape, 1)
    results = []
    for dtype in (np.uint8, np.uint16, np.uint32):
        data = gradient_volume(shape, axis=0, dtype=dtype)
        results.append(_project(data, quad).values)
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[1], results[2])


def test_benchmark_quad_is_about_500_pixels():
    height, width = plane_size(benchmark_quad())
    assert 450 <= height <= 550
    assert 450 <= width <= 550
