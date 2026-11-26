"""Parse a CSV of 3-point and 4-point planes.

Row errors stay inside the returned ``PlaneRow``. They do not abort the file.
A missing file, or a file with no data rows, raises ``CsvFormatError``.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from projection.errors import CsvFormatError
from projection.plane import PlaneSpec, TrapezoidPlane, TrianglePlane, as_point
from projection.validation import validate_trapezoid, validate_triangle

_ORDERS = ("auto", "qa", "pairs")


@dataclass
class PlaneRow:
    """One data row.

    ``plane`` is set when validation passed. ``error`` is set when it did not.
    ``warning`` records a non-fatal note (fallback point order, or ignored extras).
    """

    row_number: int
    plane: PlaneSpec | None
    error: str | None = None
    warning: str | None = None


class CsvPlaneReader:
    """Read planes from the lecturer's 13-column CSV."""

    def __init__(self, point_order: str = "auto") -> None:
        if point_order not in _ORDERS:
            raise ValueError(f"point_order must be one of {_ORDERS}, got {point_order!r}")
        self.point_order = point_order

    def read(self, path: str | Path) -> list[PlaneRow]:
        path = Path(path)
        if not path.is_file():
            raise CsvFormatError(f"CSV file not found: {path}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise CsvFormatError(f"could not read CSV file {path}: {exc}") from exc
        return self.parse(text)

    def parse(self, text: str) -> list[PlaneRow]:
        raw_rows = list(csv.reader(io.StringIO(text)))
        data_rows = [row for row in raw_rows if not _is_blank(row)]
        if not data_rows:
            raise CsvFormatError("CSV file has no data rows")
        if not _looks_like_count(data_rows[0][0] if data_rows[0] else ""):
            data_rows = data_rows[1:]
        if not data_rows:
            raise CsvFormatError("CSV file has no data rows")
        return [self._parse_row(index, row) for index, row in enumerate(data_rows, start=1)]

    def _parse_row(self, number: int, raw: list[str]) -> PlaneRow:
        cells = [cell.strip() for cell in raw]
        if len(cells) != 13:
            return PlaneRow(
                number,
                None,
                error=f"expected 13 columns, found {len(cells)}",
            )
        try:
            count = _point_count(cells[0])
        except ValueError as exc:
            return PlaneRow(number, None, error=str(exc))
        if count == 3:
            return self._parse_triangle(number, cells)
        return self._parse_trapezoid(number, cells)

    def _parse_triangle(self, number: int, cells: list[str]) -> PlaneRow:
        warning = None
        if any(cells[10:13]):
            warning = "3-point row has extra trailing values; they were ignored"
        try:
            a = _parse_point(cells[1:4])
            b = _parse_point(cells[4:7])
            c = _parse_point(cells[7:10])
        except ValueError as exc:
            return PlaneRow(number, None, error=str(exc), warning=warning)
        result = validate_triangle(a, b, c)
        if not result.ok:
            return PlaneRow(number, None, error=result.reason, warning=warning)
        return PlaneRow(number, TrianglePlane(a, b, c), warning=warning)

    def _parse_trapezoid(self, number: int, cells: list[str]) -> PlaneRow:
        try:
            points = [
                _parse_point(cells[1:4]),
                _parse_point(cells[4:7]),
                _parse_point(cells[7:10]),
                _parse_point(cells[10:13]),
            ]
        except ValueError as exc:
            return PlaneRow(number, None, error=str(exc))
        orders = _orders_to_try(self.point_order)
        failures: list[str] = []
        for order in orders:
            plane = _trapezoid_in_order(points, order)
            result = validate_trapezoid(plane.a1, plane.a2, plane.b, plane.c)
            if result.ok:
                warning = None
                if self.point_order == "auto" and order == "pairs":
                    warning = (
                        "Q&A order (a1, a2, b, c) failed validation; "
                        "used pairs order (a1, b, a2, c)"
                    )
                return PlaneRow(number, plane, warning=warning)
            failures.append(f"{order}: {result.reason}")
        return PlaneRow(number, None, error="; ".join(failures))


def _orders_to_try(point_order: str) -> tuple[str, ...]:
    if point_order == "qa":
        return ("qa",)
    if point_order == "pairs":
        return ("pairs",)
    return ("qa", "pairs")


def _trapezoid_in_order(points: list[np.ndarray], order: str) -> TrapezoidPlane:
    p1, p2, p3, p4 = points
    if order == "qa":
        return TrapezoidPlane(p1, p2, p3, p4)
    return TrapezoidPlane(p1, p3, p2, p4)


def _parse_point(cells: list[str]) -> np.ndarray:
    if any(cell == "" for cell in cells):
        raise ValueError("missing coordinate")
    try:
        values = [float(cell) for cell in cells]
    except ValueError as exc:
        raise ValueError("non-numeric coordinate") from exc
    try:
        return as_point(values)
    except ValueError as exc:
        raise ValueError("coordinate is not finite") from exc


def _point_count(cell: str) -> int:
    try:
        value = float(cell)
    except ValueError as exc:
        raise ValueError(f"point count is not a number: {cell!r}") from exc
    if value not in (3.0, 4.0):
        raise ValueError(f"point count must be 3 or 4, got {cell}")
    return int(value)


def _looks_like_count(cell: str) -> bool:
    try:
        float(cell.strip())
    except ValueError:
        return False
    return True


def _is_blank(row: list[str]) -> bool:
    return all(cell.strip() == "" for cell in row)
