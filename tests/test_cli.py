"""Command-line errors stay readable and return a non-zero status."""

from pathlib import Path

import nrrd
import numpy as np

from Ex0 import main

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "Ex0.csv"


def test_demo_command_prints_the_summary(tmp_path, capsys):
    nrrd_path = tmp_path / "vol.nrrd"
    nrrd.write(str(nrrd_path), np.zeros((4, 4, 4), dtype=np.uint8))
    video = tmp_path / "demo.mp4"
    code = main([str(nrrd_path), str(SAMPLE), str(video)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "4 rows, 3 rendered, 1 invalid (row 3)"
    assert "Traceback" not in captured.err
    assert video.is_file()


def test_missing_unreadable_and_empty_inputs(tmp_path, capsys):
    video = tmp_path / "out.mp4"
    code = main([str(tmp_path / "missing.nrrd"), str(SAMPLE), str(video)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "not found" in captured.err.lower()

    garbage = tmp_path / "bad.nrrd"
    garbage.write_text("not-nrrd", encoding="utf-8")
    code = main([str(garbage), str(SAMPLE), str(video)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err

    empty = tmp_path / "empty.csv"
    empty.write_text(
        "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n",
        encoding="utf-8",
    )
    nrrd_path = tmp_path / "vol.nrrd"
    nrrd.write(str(nrrd_path), np.zeros((2, 2, 2), dtype=np.uint8))
    code = main([str(nrrd_path), str(empty), str(video)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert "no data" in captured.err

    code = main([])
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in captured.err
