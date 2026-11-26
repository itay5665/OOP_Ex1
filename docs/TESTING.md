# Testing strategy

Tests are `pytest` cases in `tests/`. Run them from the repository root:

```bash
python3 -m pytest
python3 -m pytest -m slow
```

The slow mark is the 500³ frame-rate check. The default run skips it.

The projector is tested with in-memory volumes. The demo is tested by writing a real MP4 and reading it back. Exact intensities are checked on the arrays before H.264 encoding, because the codec may change a pixel by a few levels.

## 1. Unit tests

### CSV reader (`tests/test_csv_reader.py`)

- Header is skipped. A file with no header still parses.
- Trailing blank line is ignored.
- Four sample rows match the table in [DATASET.md](DATASET.md): row 1 triangle, row 2 Q&A trapezoid with no warning, row 3 invalid, row 4 pairs-order trapezoid with a warning.
- `--point-order qa` rejects row 4. `--point-order pairs` accepts row 4 with no fallback warning. Both orders accept row 2 and build different quads.
- Wrong column count, a non-numeric cell, and a point count other than 3 or 4 become invalid rows and do not raise.
- Extra numbers on a 3-point row are ignored and recorded as a warning.
- Missing file and a header-only file raise a user-facing error.

### Validation (`tests/test_validation.py`)

- Equal legs pass. A difference of `EPS` fails. A difference of `EPS / 2` passes.
- A short leg and a short base fail.
- Collinear points fail. A clearly non-collinear isosceles triangle passes.
- Trapezoid: equal legs, base order, the `<= |bc| + EPS` boundary, coplanar pass, non-coplanar fail, zero area fail.

### Projector (`tests/test_projector.py`)

- Gradient volumes `M = x`, `M = y`, and `M = z` on an axis-aligned rectangle. Pixel values equal the coordinate that the orientation in [ALGORITHM.md](ALGORITHM.md) predicts, and the `z = k` rectangle equals `M[:, :, k]`.
- A rectangle that crosses `x = 0` is 0 on the outside columns and keeps the inside values.
- A plane with every corner outside the volume is all zeros, and `in_volume` is all false.
- The apex row of a triangle is one sample, the array contains no NaN, and the apex voxel value is present.
- The same geometry on `uint8`, `uint16`, and `uint32` volumes agrees.
- The benchmark quad is about 500 by 500 pixels (this test does not time it).

### Samplers (`tests/test_samplers.py`)

- On integer points, nearest neighbour and trilinear return the same values.
- Halfway between 0 and 100, trilinear returns 50 and nearest neighbour returns one of the endpoints.
- A rounded index outside the volume is 0 for nearest neighbour.

### Normalizer (`tests/test_normalizer.py`)

- Minimum 0 and maximum 255 maps those ends to 0 and 255.
- A constant volume does not divide by zero and maps to 0.
- Percentile mode clips to the 1st and 99th percentiles.
- An out-of-volume pixel stays 0 even when the volume minimum is greater than 0.

### Frame composer (`tests/test_frame_composer.py`)

- The canvas size is the size that was requested, and a smaller image is centered.
- The error frame has that same size and is not blank.

### NRRD (`tests/test_nrrd_reader.py`)

- A written volume round-trips, including `M[x, y, z]` and the spacing/origin metadata.
- A missing file, a float volume, a 2D array, and an axis of length 1024 fail with a user-facing error.

### Synthetic builders (`tests/test_synthetic.py`)

- The gradient equals the chosen axis index.
- The sphere is brightest at the center.

## 2. Demo and integration (`tests/test_pipeline.py`, `tests/test_cli.py`)

- `samples/Ex0.csv` plus a small synthetic NRRD writes an MP4 with exactly 4 frames. The canvas height and width are even. With `--skip-invalid` the file has 3 frames.
- The invalid row is an error frame (brighter than a frame whose plane misses the tiny volume).
- The run summary mentions 4 rows, 3 rendered, and row 3.
- Row 4 logs the pairs-order warning. Planes that miss the volume log an outside-volume warning.
- CLI: missing NRRD, unreadable NRRD, and an empty CSV return a non-zero code and a message with no traceback.

## 3. Performance (`tests/test_performance.py`, `scripts/benchmark.py`)

`@pytest.mark.slow` builds a random `500³` `uint8` volume and an oblique quad of about `500×500`, warms up, and asserts that nearest-neighbour projection reaches at least 30 FPS.

`python3 scripts/benchmark.py` prints FPS for nearest neighbour and for trilinear, then the time to load a small NRRD and to encode a few frames, so those costs are not mixed into the projection number.

## 4. What the demo must show

The assignment demo is:

```bash
python3 Ex0.py file.nrrd file.csv file.mp4
```

The tests above cover that command on a synthetic volume and on `samples/Ex0.csv`: frame count equals the number of data rows, order is preserved, and an invalid row is still a frame. The experiment document repeats the command on a structured synthetic volume and, when `data/mri.nrrd` is present, on the lecturer's file.
