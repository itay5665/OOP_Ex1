# Software Requirements Specification

Plane projection of a 3D volume (Ex1).

## 1. Purpose

The program computes a 2D image of a plane through a 3D raster volume and
writes those images as an MP4. The course grades design, documentation, and
robust behaviour ahead of the code itself. This document records the
requirements and the decisions that close gaps in the assignment sheet.

## 2. Scope

The system:

1. Loads one NRRD volume.
2. Loads a CSV file of planes (3 or 4 points per row).
3. Validates each row.
4. Projects every valid plane.
5. Writes one video frame per CSV row, in order.

It is a batch command, `python Ex0.py file.nrrd file.csv file.mp4`. There is
no GUI.

## 3. Definitions

| Term | Meaning |
|---|---|
| Volume `M` | 3D array, axes `x`, `y`, `z`. Entry `M[x, y, z]` is an unsigned integer of 8, 16, or 32 bits. |
| Voxel index | Integer coordinate into `M`. This project uses voxel indices, not millimetres. |
| EPS | `0.001`. Used for length comparisons. |
| Triangle | Points `(a, b, c)` with `abs(\|a-b\| - \|a-c\|) < EPS`. |
| Trapezoid | Points `(a1, a2, b, c)`. Legs are `a1→b` and `a2→c`. |
| Quad | The common internal shape. A triangle is a trapezoid whose leg starts coincide: `Quad(a, b, a, c)`. A trapezoid is `Quad(a1, b, a2, c)`. |
| `L` | Shared leg length `\|end1 - start1\|`. |
| `Dbc` | Far-base length `\|end2 - end1\|`. |
| Frame | One image in the MP4. Row `k` of the CSV produces frame `k` unless `--skip-invalid` is set. |
| Projection FPS | How many planes are projected per second. This is not the MP4 playback rate. |

## 4. Functional requirements

1. The program accepts `python Ex0.py <file.nrrd> <file.csv> <file.mp4>`.
2. The NRRD is read with `pynrrd` and `index_order='F'`, so the array is indexed `[x, y, z]`.
3. Spacing and origin are read from the header and written to the log. They are not applied to coordinates.
4. The CSV header is skipped. Each remaining non-empty row is one plane. Rows are processed in order. The reader does not assume a row count.
5. Coordinates are parsed as floating-point numbers.
6. A valid triangle or trapezoid is projected with the algorithm in [ALGORITHM.md](ALGORITHM.md).
7. A sample that falls outside `M` contributes the value 0.
8. The default sampler is nearest neighbour. `--sampler trilinear` selects trilinear interpolation.
9. Samples are normalized to 8-bit with one range for the whole volume (minimum/maximum, or the 1st and 99th percentiles).
10. Pixels outside the plane shape, and pixels outside the volume, stay black after normalization.
11. Every frame of one video has the same even width and height. Each projection is centered with black padding.
12. An invalid row produces a black frame of that same size with the error text drawn on it, and a warning is logged. `--skip-invalid` drops the frame instead.
13. A plane that is valid but lies entirely outside the volume is a black frame plus a warning, not an error.
14. When the run finishes, the program prints a summary of the form `4 rows, 3 rendered, 1 invalid (row 3)`.
15. A missing file, an unreadable NRRD, a volume of the wrong rank, dtype, or size, or a CSV with no data rows exits with a non-zero status and a short message. It does not print a stack trace.
16. A malformed row (wrong column count, point count other than 3 or 4, missing or non-numeric field) is an invalid row. It does not abort the run.

## 5. Non-functional requirements

1. Python 3.10 or newer. Type hints and docstrings on public classes and functions.
2. Warnings go through the `logging` module. The end-of-run summary is printed to standard output.
3. The projector does not read files or write video. Tests can call it with an in-memory volume.
4. Sampling is a strategy object passed into the projector. Planes are value objects. There is no global mutable state.
5. The projection of one plane does not use a Python loop over pixels.
6. On a `500³` `uint8` volume, nearest-neighbour projection of an oblique plane of about `500×500` pixels must reach at least 30 FPS. 100 FPS is the target.
7. Dependencies are `numpy`, `pynrrd`, `imageio`, `imageio-ffmpeg`, `Pillow`, and `pytest`.

## 6. Inputs and outputs

### NRRD

- Rank 3, dtype `uint8`, `uint16`, or `uint32`.
- Each axis length `n` satisfies `1 <= n < 1024`.
- Encoding may be raw or gzip. `pynrrd` handles endianness.

### CSV

Thirteen columns:

```text
Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
```

- The first row of the lecturer's file is a header. A leading row whose first cell is not a number is treated as a header and skipped. A file that already starts with data is accepted as well.
- A 3-point row leaves the last three cells empty.
- A blank row (including a trailing blank line from a spreadsheet export) is ignored.
- Data-row numbers in messages are 1-based among the non-blank rows after the header. They match the sample analysis: row 3 is the invalid triangle.

### MP4

- H.264 (`libx264`), pixel format `yuv420p`, even width and height.
- Playback rate from `--fps` (default 10). This does not change the projection benchmark.
- Each stored frame is a grayscale image copied into three equal channels so the codec receives RGB.

### Command line

| Argument | Default | Role |
|---|---|---|
| `file.nrrd` | required | Input volume |
| `file.csv` | required | Planes, one per row |
| `file.mp4` | required | Output video |
| `--fps` | `10` | Playback rate, must be positive |
| `--sampler` | `nearest` | `nearest` or `trilinear` |
| `--point-order` | `auto` | `auto`, `qa`, or `pairs` |
| `--skip-invalid` | off | Drop invalid rows |
| `--normalize` | `minmax` | `minmax` or `percentile` |

## 7. Design decisions

These match the course prompt. The algorithm document explains the math.

| # | Issue | Decision |
|---|---|---|
| 1 | Coordinate space | Voxel indices. Spacing and origin are reported and not applied. |
| 2 | Non-integer sample position | Strategy pattern. `NearestNeighborSampler` is the default. `TrilinearSampler` is optional. |
| 3 | Sample outside `M` | Value 0, using a vectorized mask. Those pixels stay black after normalization. |
| 4 | Pseudocode skips the row of point `a` | Sample `start + y · v` for `y = 0 .. H-1`, so the first row is included. |
| 5 | Non-integer sizes | `H = floor(L) + 1`, `W = floor(Dbc) + 1`. Row `y` has `n(y) = floor(d12(y)) + 1` samples. Horizontal offset is `round((Dbc - d12(y)) / 2)`. |
| 6 | Index order | The image array is `image[y, x]` (NumPy row-major). That corresponds to the sheet's `E[x][y]`. |
| 7 | `d12 = 0` | One sample at `p1`. No division. |
| 8 | CSV layout | Section 6. |
| 8b | Order of the four points | `--point-order auto` (default) tries the Q&A order `(a1, a2, b, c)` first. If that fails, it tries the pairs order `(a1, b, a2, c)`. A successful fallback is logged as a warning. `--point-order qa` and `--point-order pairs` force one reading. |
| 9 | Invalid row | A black frame with the error text, plus a warning, and a line in the summary. `--skip-invalid` drops the frame. The default keeps one frame per row. |
| 10 | Frames have different sizes | First pass: parse and validate every row and record `H` and `W` of each valid plane. The canvas is the maximum height and width, each rounded up to an even number. Images are centered. Black padding fills the rest. |
| 11 | 8/16/32-bit values into an 8-bit video | One global range, computed once from the volume. `minmax` uses the minimum and maximum. `percentile` uses the 1st and 99th percentiles. Per-frame ranges would flicker. |
| 12 | NRRD | `pynrrd`, `index_order='F'`, array shaped `[x, y, z]`. |

### Validation rules

Triangle `(a, b, c)`:

- `\|a-b\| >= EPS` and `\|a-c\| >= EPS`.
- `abs(\|a-b\| - \|a-c\|) < EPS`.
- `\|b-c\| >= EPS`.
- Not collinear. Let `s = \|(b-a) × (c-a)\| / (\|b-a\| \|a-c\|)`. Reject when `s < 0.001`. `s` is the absolute sine of the angle at `a`.

Trapezoid `(a1, a2, b, c)`:

- Both legs have length `>= EPS`.
- `abs(\|a1-b\| - \|a2-c\|) < EPS`.
- `\|b-c\| >= EPS`.
- `\|a1-a2\| <= \|b-c\| + EPS`.
- Coplanar. Let `n = (a2-a1) × (c-a1)`. If `n` is zero, `a1`, `a2`, and `c` are collinear, so the four points are coplanar and this check passes. Otherwise reject when `abs(n · (b-a1)) / (\|n\| \|b-a1\|) >= 0.001`. That quantity is the absolute sine of the angle between leg `a1→b` and the plane of `a1`, `a2`, `c`.
- Area at least `EPS`. The area is the sum of the areas of triangles `a1bc` and `a1ca2`. A flat line fails this check.

A failed check produces one reason string. The first failing check wins.

## 8. Ambiguities

Resolved above, and recorded here so a reader can see what the sheet did not say:

- The sheet's loops use a non-integer bound and skip the apex row (decisions 4 and 5).
- `M[p]` is undefined for a non-integer `p` (decision 2).
- The video needs a fixed frame size; planes do not (decision 10).
- The sheet does not say how to turn 16-bit and 32-bit samples into an 8-bit frame (decision 11).
- The sheet does not say what to do with a broken CSV row (decision 9).

### Open question for the lecturer (decision 8b)

The Q&A names the four points `(a1, a2, b, c)` and requires:

1. `abs(\|a1-b\| - \|a2-c\|) < EPS`
2. `\|a1-a2\| <= \|b-c\|`
3. `a1→b` and `a2→c` lie on one plane

The sample file's fourth row, read in that order, has legs 300 and 400, so it fails. Read as `(a1, b, a2, c)` — that is, columns `(p1, p2, p3, p4) = (a1, b, a2, c)` — both legs are 502.49 and the bases are 300 and 400, so it passes.

The second row passes both readings. The two readings swap which edge runs across the image, so they produce different (transposed) pictures of the same rectangle.

**Question:** which column order is authoritative for a 4-point row?

Until there is an answer, the Q&A order has priority. `auto` falls back to the pairs order and logs a warning, so both the written Q&A and the published sample file work.

## 9. Assumptions

- A leading non-numeric row is a header. Blank rows are not data.
- Row numbers in messages count data rows, starting at 1.
- On a 3-point row, numbers in the last three columns are ignored. The reader logs a warning.
- When both 4-point orders fail, the error text contains both reasons.
- Nearest neighbour rounds with `floor(p + 0.5)` (half away from −∞). The sample is inside the volume when the rounded index lies in range.
- Trilinear interpolation treats a point as inside when each continuous coordinate lies in `[0, size - 1]`. On an integer point it matches nearest neighbour.
- If the volume's minimum equals its maximum, every in-volume sample maps to 0. That avoids dividing by zero.
- If every row is invalid, the canvas is 480×640 (height × width), both even, because no plane can supply a size.
- If `--skip-invalid` removes every row, the program exits with an error instead of writing a video with zero frames.
- The parent directory of the MP4 is created if needed.
- The encoder uses `-preset ultrafast` and `-crf 18`. The demo cares about a correct frame sequence more than about the smallest file.
- Out-of-range volume sizes and unsupported dtypes are rejected up front with a clear message.

## 10. Limitations

- Spacing and origin are not applied. A future version could multiply voxel indices by the spacing vectors.
- The image size follows the plane, not the volume. Very long coordinates allocate a large image.
- Trilinear filtering is not required to hit 100 FPS. The target is for nearest neighbour.
- The MP4 is grayscale stored as RGB. There is no colour map.
- Compression can change pixel values slightly. Tests that need exact intensities use the arrays before encoding.
- The lecturer's MRI file is not part of the repository.
