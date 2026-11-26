"""Throughput versus volume size, and correctness checks on the sample CSV.

Run from the repository root:

    python3 scripts/experiment.py

Timings are one measurement after a single warmup projection. Re-running
the script produces new timings. The correctness values are exact properties
of the inputs below.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio  # noqa: E402
import imageio.v2 as iio  # noqa: E402
import nrrd  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from projection.csv_plane_reader import CsvPlaneReader  # noqa: E402
from projection.plane import Quad, TrapezoidPlane  # noqa: E402
from projection.projector import Projector, plane_size  # noqa: E402
from projection.samplers import make_sampler  # noqa: E402
from projection.synthetic import benchmark_quad, gradient_volume  # noqa: E402
from projection.volume import Volume  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIZES = (64, 128, 256, 500)
MIN_SECONDS = 0.5
MIN_REPEATS = 3
MAX_REPEATS = 100_000
SAMPLE_EDGE = 501
SAMPLE_VALUE = 200.0


def main() -> int:
    print_environment()
    print_throughput()
    print_gradient_checks()
    print_midpoint()
    print_sample_csv()
    return run_ex0_command()


def print_environment() -> None:
    print("== environment ==")
    print(f"time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()} ({sys.executable})")
    print(f"machine: {platform.machine()}")
    print(f"cpu_count: {os.cpu_count()}")
    for key, value in _cpuinfo().items():
        print(f"{key}: {value}")
    mem = _mem_total()
    if mem:
        print(f"mem_total_kib: {mem}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"numpy: {np.__version__}")
    print(f"pynrrd: {nrrd.__version__}")
    print(f"imageio: {imageio.__version__}")
    print(f"pillow: {Image.__version__}")
    print("numpy_config:")
    np.show_config()


def print_throughput() -> None:
    print("== throughput ==")
    print(
        "volume: uint8 cube, np.random.default_rng(0).integers(0, 256). "
        "plane: benchmark_quad scaled by (size-1)/499, so size 500 is benchmark_quad(). "
        f"one warmup, then repeats until {MIN_SECONDS}s (at least {MIN_REPEATS}, at most {MAX_REPEATS}). "
        "fps = repeats/seconds. mpix/s = height*width*fps/1e6."
    )
    ref = benchmark_quad()
    got = oblique_quad(500)
    corner_delta = _max_corner_delta(ref, got)
    print(f"size_500_vs_benchmark_quad_max_abs_corner: {corner_delta:.8g}")
    print("size sampler height width repeats seconds fps mpix_per_s inside_frac")
    for size in SIZES:
        quad = oblique_quad(size)
        height, width = plane_size(quad)
        rng = np.random.default_rng(0)
        volume = Volume(rng.integers(0, 256, size=(size, size, size), dtype=np.uint8))
        inside_frac = _inside_fraction(volume, quad)
        for name in ("nearest", "trilinear"):
            repeats, elapsed = measure(volume, quad, name)
            fps = repeats / elapsed
            mpix = height * width * fps / 1_000_000.0
            print(
                f"{size} {name} {height} {width} {repeats} "
                f"{elapsed:.9f} {fps:.4f} {mpix:.4f} {inside_frac:.6f}"
            )
        del volume
        gc.collect()


def print_gradient_checks() -> None:
    print("== gradient slice ==")
    print(
        "axis-aligned rectangle z=2 on a 10x8x5 gradient volume (uint16). "
        "expected matches tests/test_projector.py. "
        "max_abs is the maximum absolute difference."
    )
    shape = (10, 8, 5)
    k = 2
    quad = TrapezoidPlane(
        (0, 0, k),
        (0, shape[1] - 1, k),
        (shape[0] - 1, 0, k),
        (shape[0] - 1, shape[1] - 1, k),
    ).to_quad()
    print(f"image: {plane_size(quad)[0]}x{plane_size(quad)[1]}")
    for axis in (0, 1, 2):
        volume = Volume(gradient_volume(shape, axis=axis, dtype=np.uint16))
        nearest = Projector(make_sampler("nearest")).project(volume, quad)
        trilinear = Projector(make_sampler("trilinear")).project(volume, quad)
        expected = _expected_slice(shape, axis, k)
        nearest_err = float(np.max(np.abs(nearest.values - expected)))
        trilinear_err = float(np.max(np.abs(trilinear.values - expected)))
        pair_err = float(np.max(np.abs(nearest.values - trilinear.values)))
        print(
            f"axis {axis} nearest_max_abs {nearest_err:.8g} "
            f"trilinear_max_abs {trilinear_err:.8g} "
            f"nearest_vs_trilinear_max_abs {pair_err:.8g} "
            f"in_volume_all {bool(nearest.in_volume.all())}"
        )
    dtype_err = _dtype_agreement(shape, quad)
    print(f"uint8_uint16_uint32_max_abs {dtype_err:.8g}")


def print_midpoint() -> None:
    print("== midpoint sample ==")
    print("volume 2x2x2 uint8, M[0,0,0]=0, M[1,0,0]=100, point (0.5, 0, 0).")
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    data[1, 0, 0] = 100
    volume = Volume(data)
    points = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float32)
    for name in ("nearest", "trilinear"):
        values, inside = make_sampler(name).sample(volume, points)
        print(f"{name} value {float(values[0, 0]):.8g} inside {bool(inside[0, 0])}")


def print_sample_csv() -> None:
    print("== samples/Ex0.csv ==")
    print(
        f"reader point_order=auto. gradient cube edge {SAMPLE_EDGE} uint16. "
        "row 1 is checked on M=y (axis 1); rows 2 and 4 on M=x (axis 0). "
        f"known in-plane value {SAMPLE_VALUE:.0f}. row 3 is not projected."
    )
    rows = CsvPlaneReader("auto").read(ROOT / "samples" / "Ex0.csv")
    print(f"data_rows {len(rows)}")
    volumes = {
        0: Volume(gradient_volume((SAMPLE_EDGE,) * 3, axis=0, dtype=np.uint16)),
        1: Volume(gradient_volume((SAMPLE_EDGE,) * 3, axis=1, dtype=np.uint16)),
    }
    axis_for_row = {1: 1, 2: 0, 4: 0}
    for row in rows:
        if row.plane is None:
            print(
                f"row {row.row_number} valid no "
                f"warning {row.warning!r} error {row.error!r}"
            )
            continue
        height, width = plane_size(row.plane.to_quad())
        axis = axis_for_row[row.row_number]
        print(
            f"row {row.row_number} valid yes warning {row.warning!r} "
            f"image {height}x{width} gradient_axis {axis}"
        )
        for name in ("nearest", "trilinear"):
            projection = Projector(make_sampler(name)).project(
                volumes[axis], row.plane.to_quad()
            )
            mask = projection.in_volume
            inside = int(mask.sum())
            total = int(mask.size)
            if inside == 0:
                err = float("nan")
            else:
                err = float(np.max(np.abs(projection.values[mask] - np.float32(SAMPLE_VALUE))))
            nan_count = int(np.isnan(projection.values).sum())
            print(
                f"row {row.row_number} {name} max_abs {err:.8g} "
                f"inside {inside}/{total} nan {nan_count}"
            )


def run_ex0_command() -> int:
    print("== Ex0.py ==")
    print(
        "writes a uint16 gradient NRRD (axis 0, edge 501, spacing 1, origin 0) "
        "and runs Ex0.py on samples/Ex0.csv. decoded frame maxima are after H.264."
    )
    mri = ROOT / "data" / "mri.nrrd"
    print(f"data/mri.nrrd present {mri.is_file()}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        nrrd_path = tmp_path / "gradient_u16.nrrd"
        mp4_path = tmp_path / "ex0.mp4"
        data = gradient_volume((SAMPLE_EDGE,) * 3, axis=0, dtype=np.uint16)
        header = {"spacings": [1.0, 1.0, 1.0], "space origin": [0.0, 0.0, 0.0]}
        started = time.perf_counter()
        nrrd.write(str(nrrd_path), data, header)
        write_seconds = time.perf_counter() - started
        print(f"nrrd_bytes {nrrd_path.stat().st_size}")
        print(f"nrrd_write_seconds {write_seconds:.6f}")
        del data
        command = [
            sys.executable,
            str(ROOT / "Ex0.py"),
            str(nrrd_path),
            str(ROOT / "samples" / "Ex0.csv"),
            str(mp4_path),
        ]
        print("command: " + " ".join(command))
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - started
        print(f"exit {completed.returncode}")
        print(f"wall_seconds {wall:.6f}")
        print("stdout:")
        sys.stdout.write(completed.stdout)
        if completed.stdout and not completed.stdout.endswith("\n"):
            print()
        print("stderr:")
        sys.stdout.write(completed.stderr)
        if completed.stderr and not completed.stderr.endswith("\n"):
            print()
        if completed.returncode != 0:
            return completed.returncode
        reader = iio.get_reader(mp4_path)
        count = reader.count_frames()
        print(f"decoded_frames {count}")
        for index in range(count):
            frame = reader.get_data(index)
            print(
                f"frame {index} shape {frame.shape[0]}x{frame.shape[1]}x{frame.shape[2]} "
                f"dtype {frame.dtype} min {int(frame.min())} max {int(frame.max())}"
            )
        reader.close()
    return 0


def measure(volume: Volume, quad, sampler_name: str) -> tuple[int, float]:
    projector = Projector(make_sampler(sampler_name))
    projector.project(volume, quad)
    gc.collect()
    repeats = 0
    start = time.perf_counter()
    while True:
        projector.project(volume, quad)
        repeats += 1
        elapsed = time.perf_counter() - start
        if repeats >= MAX_REPEATS:
            break
        if repeats >= MIN_REPEATS and elapsed >= MIN_SECONDS:
            break
    if elapsed <= 0:
        raise RuntimeError("projection timing interval was not positive")
    return repeats, elapsed


def oblique_quad(size: int) -> Quad:
    """Scale ``benchmark_quad`` so the triangle fits in a cube of edge ``size``."""
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}")
    leg = float(size - 1)
    raw = benchmark_quad(leg)
    shift = np.array([0.0, 80.0, 100.0]) * (leg / 499.0 - 1.0)
    return Quad(raw.start1 + shift, raw.end1 + shift, raw.start2 + shift, raw.end2 + shift)


def _inside_fraction(volume: Volume, quad) -> float:
    projection = Projector(make_sampler("nearest")).project(volume, quad)
    return float(np.mean(projection.in_volume))


def _max_corner_delta(left, right) -> float:
    deltas = [
        np.max(np.abs(left.start1 - right.start1)),
        np.max(np.abs(left.end1 - right.end1)),
        np.max(np.abs(left.start2 - right.start2)),
        np.max(np.abs(left.end2 - right.end2)),
    ]
    return float(max(deltas))


def _expected_slice(shape: tuple[int, int, int], axis: int, k: int) -> np.ndarray:
    if axis == 0:
        column = np.arange(shape[0], dtype=np.float32)[:, None]
        return np.broadcast_to(column, (shape[0], shape[1])).copy()
    if axis == 1:
        row = np.arange(shape[1], dtype=np.float32)[None, :]
        return np.broadcast_to(row, (shape[0], shape[1])).copy()
    return np.full((shape[0], shape[1]), k, dtype=np.float32)


def _dtype_agreement(shape: tuple[int, int, int], quad) -> float:
    values = []
    for dtype in (np.uint8, np.uint16, np.uint32):
        volume = Volume(gradient_volume(shape, axis=0, dtype=dtype))
        values.append(Projector(make_sampler("nearest")).project(volume, quad).values)
    return float(
        max(
            np.max(np.abs(values[0] - values[1])),
            np.max(np.abs(values[1] - values[2])),
        )
    )


def _cpuinfo() -> dict[str, str]:
    wanted = {
        "vendor_id": "cpu_vendor",
        "cpu family": "cpu_family",
        "model": "cpu_model_id",
        "model name": "cpu_model",
        "stepping": "cpu_stepping",
        "cpu cores": "cpu_cores",
        "siblings": "cpu_siblings",
    }
    found: dict[str, str] = {}
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        label = wanted.get(key.strip())
        if label and label not in found:
            found[label] = value.strip()
    return found


def _mem_total() -> str | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            return line.split(":", 1)[1].strip().split()[0]
    return None


if __name__ == "__main__":
    sys.exit(main())
