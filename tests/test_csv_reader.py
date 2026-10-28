"""CSV parsing, including the lecturer's four sample rows."""

from pathlib import Path

import numpy as np
import pytest

from projection.csv_plane_reader import CsvPlaneReader
from projection.errors import CsvFormatError
from projection.plane import TrapezoidPlane, TrianglePlane

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "Ex0.csv"
EDGES = ROOT / "samples" / "edge_cases.csv"


def test_sample_rows_match_the_assignment_table():
    rows = CsvPlaneReader("auto").read(SAMPLE)
    assert len(rows) == 4

    assert isinstance(rows[0].plane, TrianglePlane)
    assert rows[0].error is None
    assert rows[0].warning is None
    dab = np.linalg.norm(rows[0].plane.b - rows[0].plane.a)
    dac = np.linalg.norm(rows[0].plane.c - rows[0].plane.a)
    assert dab == pytest.approx(447.2135955, abs=1e-4)
    assert dac == pytest.approx(dab)

    assert isinstance(rows[1].plane, TrapezoidPlane)
    assert rows[1].warning is None
    np.testing.assert_allclose(rows[1].plane.a1, (200, 0, 0))
    np.testing.assert_allclose(rows[1].plane.a2, (200, 0, 500))
    np.testing.assert_allclose(rows[1].plane.b, (200, 400, 0))
    np.testing.assert_allclose(rows[1].plane.c, (200, 400, 500))

    assert rows[2].plane is None
    assert "|ab|-|ac|" in rows[2].error
    assert "EPS" in rows[2].error

    assert isinstance(rows[3].plane, TrapezoidPlane)
    assert rows[3].warning is not None
    assert "pairs order" in rows[3].warning
    np.testing.assert_allclose(rows[3].plane.a1, (200, 50, 0))
    np.testing.assert_allclose(rows[3].plane.b, (200, 0, 500))
    np.testing.assert_allclose(rows[3].plane.a2, (200, 350, 0))
    np.testing.assert_allclose(rows[3].plane.c, (200, 400, 500))


def test_point_order_flags_on_sample_rows():
    qa = CsvPlaneReader("qa").read(SAMPLE)
    pairs = CsvPlaneReader("pairs").read(SAMPLE)
    assert qa[3].plane is None
    assert pairs[3].plane is not None
    assert pairs[3].warning is None
    assert qa[1].plane is not None and pairs[1].plane is not None
    qa_quad = qa[1].plane.to_quad()
    pairs_quad = pairs[1].plane.to_quad()
    assert not np.allclose(qa_quad.end1, pairs_quad.end1)


def test_header_blank_line_and_missing_header():
    text = (
        "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
        "3,0,0,0,0,0,10,0,10,0,,,\n"
        "\n"
    )
    rows = CsvPlaneReader().parse(text)
    assert len(rows) == 1
    assert rows[0].plane is not None

    no_header = "3,0,0,0,0,0,10,0,10,0,,,\n"
    rows = CsvPlaneReader().parse(no_header)
    assert len(rows) == 1
    assert isinstance(rows[0].plane, TrianglePlane)


def test_malformed_rows_do_not_raise_and_do_not_stop_the_file():
    text = (
        "h,x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
        "3,0,0,0,1,0,0\n"
        "3,nope,0,0,1,0,0,0,1,0,,,\n"
        "5,0,0,0,1,0,0,0,1,0,,,\n"
        "3,0,0,0,0,0,10,0,10,0,,,\n"
    )
    rows = CsvPlaneReader().parse(text)
    assert [row.error is None for row in rows] == [False, False, False, True]
    assert "13 columns" in rows[0].error
    assert "non-numeric" in rows[1].error
    assert "point count" in rows[2].error
    assert rows[3].plane is not None


def test_extra_trailing_values_are_ignored_with_a_warning():
    text = "3,0,0,0,0,0,10,0,10,0,9,9,9\n"
    rows = CsvPlaneReader().parse(text)
    assert rows[0].plane is not None
    assert rows[0].warning is not None
    assert "ignored" in rows[0].warning


def test_edge_case_file():
    rows = CsvPlaneReader("auto").read(EDGES)
    assert len(rows) == 8
    assert rows[0].error and "collinear" in rows[0].error
    assert rows[1].error and "leg" in rows[1].error
    assert rows[2].error and "coplanar" in rows[2].error
    assert rows[3].error and "point count" in rows[3].error
    assert rows[4].error and "non-numeric" in rows[4].error
    assert rows[5].error and "13 columns" in rows[5].error
    assert rows[6].plane is not None
    assert rows[7].plane is not None
    assert rows[7].warning is not None


def test_missing_file_and_header_only(tmp_path):
    with pytest.raises(CsvFormatError):
        CsvPlaneReader().read(tmp_path / "missing.csv")
    header_only = tmp_path / "header.csv"
    header_only.write_text(
        "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n\n",
        encoding="utf-8",
    )
    with pytest.raises(CsvFormatError, match="no data"):
        CsvPlaneReader().read(header_only)
