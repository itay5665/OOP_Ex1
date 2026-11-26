# Experiment

How projection throughput changes with cube edge length, for nearest-neighbour
and trilinear sampling, and whether `samples/Ex0.csv` matches the notes in
[DATASET.md](DATASET.md) on a gradient volume.

## Setup

One run of `scripts/experiment.py`, started at `2026-11-26T14:34:35+0000`.

| Item | Value |
|---|---|
| Machine | Linux 6.12.113+, x86_64, KVM (`lscpu`: full virtualization, BogoMIPS 4800.00, L3 cache 320 MiB) |
| CPU | GenuineIntel family 6 model 207 stepping 2, `Intel(R) Xeon(R) Processor`, 4 cores, 4 siblings. `cpu_count` 4. `OMP_NUM_THREADS` unset |
| Memory | `MemTotal` 16387328 KiB |
| Python | 3.12.3 (`/usr/bin/python3`) |
| NumPy | 1.26.4. `numpy.show_config()`: system BLAS named `blas` (version unknown), GCC 13.2.0, host SIMD including AVX2 and AVX512F |
| Other libraries | pynrrd 1.1.4, imageio 2.38.2, Pillow 13.0.0, imageio-ffmpeg 0.8.1 (FFmpeg `n9.0.2-17-g2a571b6068-20260930`) |

`data/mri.nrrd` was absent (`present False`). MRI shape, spacing, origin, and timing wait for a run that has that file under `data/`. The program reads spacing and origin and leaves them unused.

### Throughput parameters

- Cube edges 64, 128, 256, and 500. Dtype `uint8`. Values `np.random.default_rng(0).integers(0, 256, size=(edge, edge, edge))`.
- Plane: the oblique isosceles triangle from `benchmark_quad`, with the fixed start `(0, 80, 100)` shifted by `(edge - 1) / 499 - 1`. Edge 500 is `benchmark_quad()` itself. This run's maximum absolute corner difference was 0.
- Samplers: `nearest`, then `trilinear`, through `Projector`.
- Clock: `time.perf_counter`. One untimed warmup projection, then `gc.collect()`, then further projections until the interval reaches 0.5 s, with at least 3 calls and at most 100000. File loading and video encoding are outside this loop.
- `FPS = repeats / seconds`. `Mpix/s = height × width × FPS / 1e6`. Both use the unrounded interval. The table rounds seconds to 9 decimal places and FPS and Mpix/s to 4.

The in-volume fraction is the mean of `in_volume` on one nearest-neighbour projection of that cube, taken before the timed loop. The same fraction is listed on both sampler rows.

### Correctness parameters

- Axis-aligned rectangle on a `10×8×5` `uint16` gradient, plane `z = 2`, corners `(0, 0, 2)`, `(0, 7, 2)`, `(9, 0, 2)`, `(9, 7, 2)`. The expected image is the one in `tests/test_projector.py`: axis 0 varies along rows, axis 1 along columns, axis 2 is the constant 2. The same rectangle is projected for `uint8`, `uint16`, and `uint32` on axis 0.
- Midpoint: `2×2×2` `uint8` with `M[1, 0, 0] = 100` and every other voxel 0. Sample point `(0.5, 0, 0)`.
- `samples/Ex0.csv` with `--point-order auto`. Gradient cube edge 501, `uint16`. Row 1 is compared with `M` equal to the y index, expected value 200. Rows 2 and 4 are compared with `M` equal to the x index, expected value 200. Row 3 is not projected. The maximum absolute error uses only pixels where `in_volume` is true.
- Command check: the script writes that x-gradient as NRRD (spacing `(1, 1, 1)`, origin `(0, 0, 0)`) and runs `Ex0.py` with default flags (`--sampler nearest`, `--point-order auto`, `--normalize minmax`, `--fps 10`).

## How to reproduce

From the repository root, with the libraries in `requirements.txt` installed:

```bash
python3 scripts/experiment.py
```

The process prints the environment, the throughput rows, the correctness lines, and the `Ex0.py` log. Another run keeps the same inputs and will print different timings. The correctness values are fixed by those inputs.

The command the script shells out to is `Ex0.py` on a temporary NRRD and `samples/Ex0.csv`. The same projection, with a file you keep, is:

```bash
python3 scripts/make_synthetic_nrrd.py data/gradient_u16.nrrd --kind gradient --axis 0 --dtype uint16 --size 501
python3 Ex0.py data/gradient_u16.nrrd samples/Ex0.csv output/experiment.mp4
```

`data/` and `output/` are gitignored. The numbers below are from the temporary-file run inside `scripts/experiment.py`. The two commands above are that same projection with paths you choose.

## Results

### Throughput

| Edge | Sampler | Image | Repeats | Seconds | FPS | Mpix/s | In-volume fraction |
|---|---|---|---|---|---|---|---|
| 64 | nearest | 64×63 | 2994 | 0.500133454 | 5986.4022 | 24.1372 | 0.504960 |
| 64 | trilinear | 64×63 | 1174 | 0.500064578 | 2347.6968 | 9.4659 | 0.504960 |
| 128 | nearest | 128×127 | 1127 | 0.500204260 | 2253.0796 | 36.6261 | 0.507074 |
| 128 | trilinear | 128×127 | 345 | 0.501284044 | 688.2326 | 11.1879 | 0.507074 |
| 256 | nearest | 256×255 | 288 | 0.500500656 | 575.4238 | 37.5637 | 0.503646 |
| 256 | trilinear | 256×255 | 109 | 0.501712217 | 217.2560 | 14.1825 | 0.503646 |
| 500 | nearest | 500×499 | 72 | 0.505977456 | 142.2988 | 35.5036 | 0.501780 |
| 500 | trilinear | 500×499 | 20 | 0.513958467 | 38.9137 | 9.7090 | 0.501780 |

`64×63 = 4032` pixels and `500×499 = 249500` pixels (`249500 / 4032 = 61.88`). Nearest-neighbour FPS goes from 5986.4022 to 142.2988 (`5986.4022 / 142.2988 = 42.07`). Dividing the printed FPS values, nearest / trilinear is 2.55 at edge 64, 3.27 at 128, 2.65 at 256, and 3.66 at 500 (each quotient rounded to 2 decimal places).

The in-volume fraction stays near one half. The stored image is the rectangle around the triangle, and pixels outside the triangle are marked out of volume.

### Gradient, dtypes, and the midpoint

| Check | Measured result |
|---|---|
| Slice image size | 10×8 |
| Axis 0, nearest vs expected | max abs 0, every pixel in volume |
| Axis 0, trilinear vs expected | max abs 0 |
| Axis 0, nearest vs trilinear | max abs 0 |
| Axis 1, nearest vs expected | max abs 0, every pixel in volume |
| Axis 1, trilinear vs expected | max abs 0 |
| Axis 1, nearest vs trilinear | max abs 0 |
| Axis 2, nearest vs expected | max abs 0, every pixel in volume |
| Axis 2, trilinear vs expected | max abs 0 |
| Axis 2, nearest vs trilinear | max abs 0 |
| `uint8` / `uint16` / `uint32` nearest, axis 0 | max pairwise abs 0 |
| Midpoint, nearest | value 100, inside True |
| Midpoint, trilinear | value 50, inside True |

### `samples/Ex0.csv`

Four data rows. `point_order` was `auto`.

| Row | Valid | Image | Nearest max abs vs 200 | Trilinear max abs vs 200 | In-volume pixels | NaN count |
|---|---|---|---|---|---|---|
| 1 | yes, no warning | 448×401 | 0 | 0 | 89780/179648 | 0 |
| 2 | yes, no warning | 401×501 | 0 | 0 | 200901/200901 | 0 |
| 3 | no | not projected | not projected | not projected | not projected | not projected |
| 4 | yes, with warning | 503×401 | 0 | 0 | 176273/201703 | 0 |

Row 3 error text: `invalid plane — |ab|-|ac| = 20.5284 >= EPS`.

Row 4 warning text: `Q&A order (a1, a2, b, c) failed validation; used pairs order (a1, b, a2, c)`.

Row 2 fills its rectangle (200901 of 200901 pixels in volume). Rows 1 and 4 are triangles or tapered trapezoids, so part of the rectangular image is outside the shape.

### `Ex0.py` on the 501³ gradient

| Item | Measured result |
|---|---|
| NRRD size on disk | 1220465 bytes |
| NRRD write | 2.042744 s |
| Process exit code | 0 |
| Process wall time | 0.846204 s |
| stdout | `4 rows, 3 rendered, 1 invalid (row 3)` |
| Decoded frames | 4 |
| Decoded frame shape | 504×502×3, `uint8` |
| Frame 0 min, max | 0, 203 |
| Frame 1 min, max | 0, 104 |
| Frame 2 min, max | 0, 255 |
| Frame 3 min, max | 0, 105 |

The in-memory array is `501³ × 2 = 251503002` bytes. The file on disk is smaller than that raw size.

stderr logged the NRRD as `shape=(501, 501, 501) dtype=uint16 spacing=(1.0, 1.0, 1.0) origin=(0.0, 0.0, 0.0)`, repeated the row 3 error and the row 4 pairs-order warning, and logged the same summary as stdout. Decoded minima and maxima are H.264 output. The raw sample checks are the max-abs columns above. The canvas is 504×502. The valid planes measured 448×401, 401×501, and 503×401, so the maximum height is 503 and the maximum width is 501. `even_ceil` lifts those to 504 and 502.

## Conclusion

On this machine the 500³ nearest-neighbour loop reached 142.2988 FPS, which is above the 30 FPS minimum and above the 100 FPS target in the README. Trilinear on the same image reached 38.9137 FPS, above 30 FPS and below 100 FPS. Nearest neighbour was faster at every edge. Mpix/s did not fall with edge length the way FPS did: nearest Mpix/s was 24.1372 at edge 64 and 35.5036 at edge 500, with the highest nearest value at edge 256 (37.5637). Trilinear Mpix/s peaked at edge 256 (14.1825) and was 9.7090 at edge 500.

The known-answer checks came out exact. Gradient slices matched for both samplers, the three unsigned dtypes agreed, and the midpoint sample was 100 for nearest neighbour and 50 for trilinear. On `samples/Ex0.csv`, row 3 was the only invalid row, row 4 carried the pairs-order warning, and every in-volume sample on the three valid planes was 200 for both samplers, with no NaNs. `Ex0.py` exited 0, wrote 4 frames, and printed `4 rows, 3 rendered, 1 invalid (row 3)`. That wall time, 0.846204 s, includes reading the NRRD and encoding the video. The projection FPS above is the timed `Projector.project` loop only.
