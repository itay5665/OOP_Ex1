"""Write a synthetic NRRD volume.

Run from the repository root:

    python3 scripts/make_synthetic_nrrd.py data/sphere_u8.nrrd --kind sphere --size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nrrd  # noqa: E402

from projection.synthetic import (  # noqa: E402
    dtype_by_name,
    gradient_volume,
    nested_cubes,
    sphere_volume,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic NRRD volume.")
    parser.add_argument("path", help="Output .nrrd path")
    parser.add_argument("--kind", choices=("gradient", "cubes", "sphere"), required=True)
    parser.add_argument("--dtype", default="uint8")
    parser.add_argument("--size", type=int, default=32, help="Cube edge length")
    parser.add_argument("--axis", type=int, default=0, help="Axis used by --kind gradient")
    args = parser.parse_args(argv)
    if args.size < 1:
        print("error: --size must be positive", file=sys.stderr)
        return 1
    dtype = dtype_by_name(args.dtype)
    shape = (args.size, args.size, args.size)
    if args.kind == "gradient":
        data = gradient_volume(shape, axis=args.axis, dtype=dtype)
    elif args.kind == "cubes":
        data = nested_cubes(shape, dtype=dtype)
    else:
        data = sphere_volume(shape, dtype=dtype)
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"spacings": [1.0, 1.0, 1.0], "space origin": [0.0, 0.0, 0.0]}
    nrrd.write(str(path), data, header)
    print(f"wrote {path} shape={data.shape} dtype={data.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
