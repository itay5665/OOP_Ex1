"""Measure projection FPS separately from file I/O and video encoding.

    python3 scripts/benchmark.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nrrd  # noqa: E402
import numpy as np  # noqa: E402

from projection.pipeline import Pipeline, PipelineOptions  # noqa: E402
from projection.projector import Projector, plane_size  # noqa: E402
from projection.samplers import make_sampler  # noqa: E402
from projection.synthetic import benchmark_quad, sphere_volume  # noqa: E402
from projection.volume import Volume  # noqa: E402


def measure(volume: Volume, quad, sampler_name: str, repeats: int) -> float:
    projector = Projector(make_sampler(sampler_name))
    projector.project(volume, quad)
    start = time.perf_counter()
    for _ in range(repeats):
        projector.project(volume, quad)
    elapsed = time.perf_counter() - start
    return repeats / elapsed


def main() -> int:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(500, 500, 500), dtype=np.uint8)
    volume = Volume(data)
    quad = benchmark_quad()
    height, width = plane_size(quad)
    print(f"volume 500^3 uint8, image {height}x{width}")
    repeats = 5
    for name in ("nearest", "trilinear"):
        fps = measure(volume, quad, name, repeats)
        print(f"{name}: {fps:.1f} FPS over {repeats} projections")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nrrd_path = tmp_path / "sphere.nrrd"
        csv_path = tmp_path / "one.csv"
        mp4_path = tmp_path / "out.mp4"
        sphere = sphere_volume((32, 32, 32), dtype=np.uint8)
        started = time.perf_counter()
        nrrd.write(str(nrrd_path), sphere)
        load_started = time.perf_counter()
        csv_path.write_text(
            "Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4\n"
            "3,16,16,4,4,16,16,28,16,16,,,\n",
            encoding="utf-8",
        )
        summary = Pipeline().run(
            str(nrrd_path),
            str(csv_path),
            str(mp4_path),
            PipelineOptions(fps=10, sampler="nearest"),
        )
        finished = time.perf_counter()
        print(
            f"end-to-end sample: write+run {finished - started:.3f}s "
            f"(nrrd write {load_started - started:.3f}s), {summary.format()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
