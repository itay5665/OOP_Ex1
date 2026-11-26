"""Read an NRRD file into a ``Volume``.

``index_order='F'`` makes the array ``[x, y, z]``, matching ``M[x, y, z]``.
Spacing and origin are parsed for the log and then left unused.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nrrd
import numpy as np

from projection.errors import NrrdReadError
from projection.volume import Volume

log = logging.getLogger(__name__)


class NrrdReader:
    """Load one scalar NRRD volume."""

    def read(self, path: str | Path) -> Volume:
        path = Path(path)
        if not path.is_file():
            raise NrrdReadError(f"NRRD file not found: {path}")
        try:
            data, header = nrrd.read(str(path), index_order="F")
        except NrrdReadError:
            raise
        except Exception as exc:
            raise NrrdReadError(f"could not read NRRD file {path}: {exc}") from exc
        spacing = _spacing(header)
        origin = _origin(header)
        log.info(
            "NRRD %s shape=%s dtype=%s spacing=%s origin=%s "
            "(spacing and origin are not applied; coordinates are voxel indices)",
            path,
            getattr(data, "shape", None),
            getattr(data, "dtype", None),
            spacing,
            origin,
        )
        try:
            return Volume(data=data, spacing=spacing, origin=origin, source=str(path))
        except ValueError as exc:
            raise NrrdReadError(str(exc)) from exc


def _spacing(header: dict) -> tuple[float, float, float] | None:
    raw = header.get("spacings")
    if raw is not None:
        values = [float(v) for v in np.ravel(np.asarray(raw, dtype=float))]
        if len(values) >= 3:
            return (values[0], values[1], values[2])
    directions = header.get("space directions")
    if directions is not None:
        dirs = np.asarray(directions, dtype=float)
        if dirs.ndim == 2 and dirs.shape[0] >= 3:
            return tuple(float(np.linalg.norm(dirs[i])) for i in range(3))  # type: ignore[return-value]
    return None


def _origin(header: dict) -> tuple[float, float, float] | None:
    raw = header.get("space origin")
    if raw is None:
        return None
    values = [float(v) for v in np.ravel(np.asarray(raw, dtype=float))]
    if len(values) < 3:
        return None
    return (values[0], values[1], values[2])
