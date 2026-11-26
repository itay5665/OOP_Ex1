"""NRRD round-trip and rejected volumes."""

import nrrd
import numpy as np
import pytest

from projection.errors import NrrdReadError
from projection.nrrd_reader import NrrdReader
from projection.synthetic import gradient_volume


def test_round_trip_preserves_indexes_and_spacing(tmp_path):
    data = gradient_volume((4, 5, 6), axis=0, dtype=np.uint16)
    path = tmp_path / "grad.nrrd"
    nrrd.write(
        str(path),
        data,
        {"spacings": [1.5, 2.0, 2.5], "space origin": [10.0, 20.0, 30.0]},
    )
    volume = NrrdReader().read(path)
    assert volume.shape == (4, 5, 6)
    assert volume.dtype == np.uint16
    assert volume.data[3, 1, 2] == 3
    assert volume.spacing == pytest.approx((1.5, 2.0, 2.5))
    assert volume.origin == pytest.approx((10.0, 20.0, 30.0))
    assert volume.data.flags.f_contiguous


def test_bad_files_and_illegal_shapes(tmp_path):
    missing = tmp_path / "nope.nrrd"
    with pytest.raises(NrrdReadError, match="not found"):
        NrrdReader().read(missing)

    garbage = tmp_path / "bad.nrrd"
    garbage.write_text("this is not nrrd", encoding="utf-8")
    with pytest.raises(NrrdReadError):
        NrrdReader().read(garbage)

    floats = tmp_path / "float.nrrd"
    nrrd.write(str(floats), np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(NrrdReadError, match="dtype"):
        NrrdReader().read(floats)

    flat = tmp_path / "flat.nrrd"
    nrrd.write(str(flat), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(NrrdReadError, match="3D"):
        NrrdReader().read(flat)

    huge = tmp_path / "huge.nrrd"
    nrrd.write(str(huge), np.zeros((1024, 1, 1), dtype=np.uint8))
    with pytest.raises(NrrdReadError, match="1024"):
        NrrdReader().read(huge)
