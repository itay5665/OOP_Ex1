"""Write a CSV of rectangles that rotate through a cube.

Run from the repository root:

    python3 scripts/make_orbit_csv.py samples/orbit_64.csv --size 64 --frames 36
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projection.csv_plane_reader import CsvPlaneReader  # noqa: E402
from projection.synthetic import orbit_rectangles  # noqa: E402

_HEADER = "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4"


def render_csv(size: int, frames: int, half: float) -> str:
    lines = [_HEADER]
    for corners in orbit_rectangles(size, frames, half):
        cells = ["4"]
        for point in corners:
            cells.extend(f"{float(value):.6f}" for value in point)
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an orbit CSV of rectangles.")
    parser.add_argument("path", help="Output CSV path")
    parser.add_argument("--size", type=int, default=64, help="Volume edge the planes are built for")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--half", type=float, default=28.0, help="Half the rectangle side, in voxels")
    args = parser.parse_args(argv)
    text = render_csv(args.size, args.frames, args.half)
    rows = CsvPlaneReader("qa").parse(text)
    invalid = [row for row in rows if row.plane is None]
    if invalid:
        print(f"error: {len(invalid)} orbit rows failed validation", file=sys.stderr)
        for row in invalid[:5]:
            print(f"  row {row.row_number}: {row.error}", file=sys.stderr)
        return 1
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({len(rows)} planes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
