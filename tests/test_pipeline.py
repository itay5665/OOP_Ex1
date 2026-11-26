"""End-to-end demo: one frame per CSV row, and three frames with --skip-invalid."""

import logging
from pathlib import Path

import imageio.v2 as imageio
import nrrd
import numpy as np

from projection.pipeline import Pipeline, PipelineOptions

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "Ex0.csv"


def _write_volume(path: Path) -> None:
    nrrd.write(str(path), np.zeros((8, 8, 8), dtype=np.uint8))


def _frames(path: Path):
    reader = imageio.get_reader(path)
    frames = [reader.get_data(index) for index in range(reader.count_frames())]
    reader.close()
    return frames


def test_sample_csv_writes_four_frames_and_skip_invalid_writes_three(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    nrrd_path = tmp_path / "vol.nrrd"
    _write_volume(nrrd_path)
    video = tmp_path / "out.mp4"
    summary = Pipeline().run(
        str(nrrd_path),
        str(SAMPLE),
        str(video),
        PipelineOptions(),
    )
    assert summary.format() == "4 rows, 3 rendered, 1 invalid (row 3)"
    assert summary.frame_size[0] % 2 == 0
    assert summary.frame_size[1] % 2 == 0
    frames = _frames(video)
    assert len(frames) == 4
    assert frames[0].shape[:2] == summary.frame_size
    # The tiny volume misses every sample plane, so valid frames stay black
    # and the invalid row is the frame that carries the error text.
    assert frames[2].max() > frames[0].max()
    assert "pairs order" in caplog.text
    assert "outside the volume" in caplog.text

    skipped = tmp_path / "skipped.mp4"
    summary = Pipeline().run(
        str(nrrd_path),
        str(SAMPLE),
        str(skipped),
        PipelineOptions(skip_invalid=True),
    )
    assert summary.rendered == 3
    assert len(summary.invalid_rows) == 1
    assert len(_frames(skipped)) == 3


def test_all_invalid_rows_use_the_fallback_canvas(tmp_path):
    nrrd_path = tmp_path / "vol.nrrd"
    _write_volume(nrrd_path)
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("3,1,2\n", encoding="utf-8")
    video = tmp_path / "err.mp4"
    summary = Pipeline().run(str(nrrd_path), str(csv_path), str(video))
    assert summary.frame_size == (480, 640)
    assert summary.format() == "1 rows, 0 rendered, 1 invalid (row 1)"
    frames = _frames(video)
    assert len(frames) == 1
    assert frames[0].shape[:2] == (480, 640)
