"""Run the full projection: read, validate, project, normalize, encode."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from projection.csv_plane_reader import CsvPlaneReader, PlaneRow
from projection.errors import PipelineError
from projection.frame_composer import FrameComposer, even_ceil
from projection.normalizer import Normalizer
from projection.nrrd_reader import NrrdReader
from projection.projector import Projector, plane_size
from projection.samplers import Sampler, make_sampler
from projection.video_writer import VideoWriter
from projection.volume import Volume

log = logging.getLogger(__name__)

# Used only when every row is invalid, so no plane can define the canvas.
_FALLBACK_HEIGHT = 480
_FALLBACK_WIDTH = 640


@dataclass
class PipelineOptions:
    """Flags from the command line, with the documented defaults."""

    fps: float = 10.0
    sampler: str = "nearest"
    point_order: str = "auto"
    skip_invalid: bool = False
    normalize: str = "minmax"


@dataclass
class RunSummary:
    """Counts printed at the end of a run."""

    rows: int
    rendered: int
    invalid_rows: list[int] = field(default_factory=list)
    frame_size: tuple[int, int] = (0, 0)

    def format(self) -> str:
        invalid = len(self.invalid_rows)
        if invalid == 0:
            detail = "0 invalid"
        elif invalid == 1:
            detail = f"1 invalid (row {self.invalid_rows[0]})"
        else:
            numbers = ", ".join(str(number) for number in self.invalid_rows)
            detail = f"{invalid} invalid (rows {numbers})"
        return f"{self.rows} rows, {self.rendered} rendered, {detail}"


class Pipeline:
    """Orchestrate one video. The sampler can be injected for tests."""

    def __init__(self, sampler: Sampler | None = None) -> None:
        self._sampler = sampler

    def run(
        self,
        nrrd_path: str,
        csv_path: str,
        mp4_path: str,
        options: PipelineOptions | None = None,
    ) -> RunSummary:
        options = options or PipelineOptions()
        if options.fps <= 0:
            raise PipelineError(f"fps must be positive, got {options.fps}")
        volume = NrrdReader().read(nrrd_path)
        rows = CsvPlaneReader(options.point_order).read(csv_path)
        frames_to_write = [
            row for row in rows if row.plane is not None or not options.skip_invalid
        ]
        if not frames_to_write:
            raise PipelineError(
                "every row is invalid and --skip-invalid is set, so there are no frames"
            )
        height, width = _canvas_size(rows)
        composer = FrameComposer(height, width)
        normalizer = Normalizer.from_volume(volume, options.normalize)
        projector = Projector(self._sampler or make_sampler(options.sampler))
        invalid_rows = [row.row_number for row in rows if row.plane is None]
        rendered = 0
        with VideoWriter(mp4_path, options.fps, (height, width)) as writer:
            for row in rows:
                if row.warning:
                    log.warning("Row %s: %s", row.row_number, row.warning)
                if row.plane is None:
                    if options.skip_invalid:
                        log.warning(
                            "Row %s: skipped invalid plane (%s)", row.row_number, row.error
                        )
                        continue
                    log.warning("Row %s: invalid plane (%s)", row.row_number, row.error)
                    writer.write(composer.error_frame(f"Row {row.row_number}: {row.error}"))
                    continue
                frame = _project_frame(projector, normalizer, composer, volume, row)
                writer.write(frame)
                rendered += 1
        summary = RunSummary(
            rows=len(rows),
            rendered=rendered,
            invalid_rows=invalid_rows,
            frame_size=(height, width),
        )
        log.info("%s", summary.format())
        return summary


def _canvas_size(rows: list[PlaneRow]) -> tuple[int, int]:
    sizes = [plane_size(row.plane.to_quad()) for row in rows if row.plane is not None]
    if not sizes:
        return _FALLBACK_HEIGHT, _FALLBACK_WIDTH
    height = even_ceil(max(h for h, _w in sizes))
    width = even_ceil(max(w for _h, w in sizes))
    return height, width


def _project_frame(projector, normalizer, composer, volume: Volume, row: PlaneRow):
    projection = projector.project(volume, row.plane.to_quad())
    if not projection.in_volume.any():
        log.warning(
            "Row %s: plane lies entirely outside the volume; writing a black frame",
            row.row_number,
        )
    image = normalizer.apply(projection.values, projection.in_volume)
    return composer.compose(image)
