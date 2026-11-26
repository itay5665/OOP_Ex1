"""Command line: ``python Ex0.py file.nrrd file.csv file.mp4``.

This module only parses arguments and prints the summary. The work lives in
``projection.pipeline``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

from projection.errors import UserFacingError
from projection.pipeline import Pipeline, PipelineOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project planes through an NRRD volume and write an MP4."
    )
    parser.add_argument("nrrd", help="Input NRRD volume")
    parser.add_argument("csv", help="CSV file of 3-point and 4-point planes")
    parser.add_argument("mp4", help="Output MP4 path")
    parser.add_argument(
        "--fps",
        type=_positive_float,
        default=10.0,
        help="Video playback rate (default 10). This is not the projection FPS.",
    )
    parser.add_argument(
        "--sampler",
        choices=("nearest", "trilinear"),
        default="nearest",
        help="How to read a non-integer voxel coordinate (default nearest)",
    )
    parser.add_argument(
        "--point-order",
        choices=("auto", "qa", "pairs"),
        default="auto",
        help="How to read a 4-point row (default auto)",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Drop invalid rows instead of writing an error frame",
    )
    parser.add_argument(
        "--normalize",
        choices=("minmax", "percentile"),
        default="minmax",
        help="Global 8-bit range (default minmax)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the projection. Returns a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 1 if code is None else int(code)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    options = PipelineOptions(
        fps=args.fps,
        sampler=args.sampler,
        point_order=args.point_order,
        skip_invalid=args.skip_invalid,
        normalize=args.normalize,
    )
    try:
        summary = Pipeline().run(args.nrrd, args.csv, args.mp4, options)
    except UserFacingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(summary.format())
    return 0


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"fps must be a number, got {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"fps must be positive, got {text}")
    return value


if __name__ == "__main__":
    sys.exit(main())
