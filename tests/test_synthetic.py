"""Synthetic volumes used by the scripts and the experiment."""

import numpy as np

from projection.synthetic import gradient_volume, sphere_volume


def test_gradient_and_sphere():
    data = gradient_volume((5, 4, 3), axis=1, dtype=np.uint8)
    assert data[0, 3, 0] == 3
    assert data[4, 0, 2] == 0
    sphere = sphere_volume((9, 9, 9), dtype=np.uint8)
    assert sphere[4, 4, 4] == 255
    assert sphere[0, 0, 0] == 0
