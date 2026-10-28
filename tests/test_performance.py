"""Frame rate of nearest-neighbour projection on a 500³ volume."""

import time

import numpy as np
import pytest

from projection.projector import Projector
from projection.samplers import NearestNeighborSampler
from projection.synthetic import benchmark_quad
from projection.volume import Volume


@pytest.mark.slow
def test_nearest_neighbor_reaches_30_fps():
    rng = np.random.default_rng(0)
    volume = Volume(rng.integers(0, 256, size=(500, 500, 500), dtype=np.uint8))
    quad = benchmark_quad()
    projector = Projector(NearestNeighborSampler())
    projector.project(volume, quad)
    repeats = 3
    start = time.perf_counter()
    for _ in range(repeats):
        projector.project(volume, quad)
    fps = repeats / (time.perf_counter() - start)
    assert fps >= 30, f"nearest-neighbour projection ran at {fps:.1f} FPS"
