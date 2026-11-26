# Dataset

Coordinates in every CSV file are voxel indices. See [SRS.md](SRS.md) decision 1.

## 1. Lecturer MRI

The assignment asks for an MRI NRRD from the 3D Slicer project. This repository
does not download it. Place the file at:

```text
data/mri.nrrd
```

`data/` is gitignored. After the file is there, record its dtype, axis
lengths, spacing, and origin in [EXPERIMENT.md](EXPERIMENT.md). Spacing is
reported by the program and is not applied.

I am not writing a download URL. The assignment PDF points at a file without
printing the link in the text that was available here, and a guessed URL would
be worse than none. The 3D Slicer Sample Data module is the place to confirm
the file.

## 2. Further Slicer volumes

These are optional comparisons. Obtain them from the Sample Data module inside
3D Slicer (Download Sample Data). I am not listing URLs, because the module is
the source I trust and I do not want to publish a link I have not just
checked.

Confirm dtype and size in Slicer after download. The names below are the
module's names; the sizes are the usual ones and should be checked rather than
trusted blindly.

| Sample Data name | What it is useful for |
|---|---|
| MRHead | Small head MRI. A second scalar volume beside the lecturer file, so the same CSV can be compared on two scans. |
| CTACardio | CT angiography of the heart. Larger than MRHead, and a different modality, so contrast and the percentile normalizer are easier to judge. |
| CBCTDentalSurgery | Cone-beam CT. Another dtype/size combination, with a small field of view. |

Copy them into `data/` under those names, for example `data/MRHead.nrrd`.

## 3. Synthetic volumes

`projection/synthetic.py` builds arrays in memory. `scripts/make_synthetic_nrrd.py` writes them as NRRD. Tests use the in-memory builders so they do not depend on a downloaded file.

```bash
python3 scripts/make_synthetic_nrrd.py data/gradient_u16.nrrd --kind gradient --axis 0 --dtype uint16 --size 32
python3 scripts/make_synthetic_nrrd.py data/cubes_u8.nrrd --kind cubes --dtype uint8 --size 64
python3 scripts/make_synthetic_nrrd.py data/sphere_u8.nrrd --kind sphere --dtype uint8 --size 64
```

| Kind | Contents |
|---|---|
| `gradient` | `M[x, y, z]` equals the index along `--axis` (0, 1, or 2). The axis length must fit in the dtype. |
| `cubes` | Nested cubes. The value depends on the Chebyshev distance from the center, so a slice shows rectangular shells. |
| `sphere` | A ball and a brighter core. Useful for an orbit video, where a moving plane should show a circle that changes size. |

Dtypes: `uint8`, `uint16`, `uint32`. Each axis length must satisfy `1 <= n < 1024`.

The benchmark does not write a file. It fills a `500³` `uint8` array from a fixed random seed.

## 4. CSV files

### `samples/Ex0.csv`

The lecturer's sample, including the header and a trailing blank line.

| Row | Expected result under `--point-order auto` |
|---|---|
| 1 | Valid triangle. `\|ab\| = \|ac\| ≈ 447.21`, plane `y = 200`. |
| 2 | Valid trapezoid in both orders (a rectangle in the plane `x = 200`). `auto` keeps the Q&A order and does not warn. The two orders transpose the image. |
| 3 | Invalid triangle. The legs differ by about 20.53, which is greater than `EPS`. Some coordinates are negative. |
| 4 | Invalid in the Q&A order (legs 300 and 400). Valid in the pairs order (both legs ≈ 502.49, bases 300 and 400). `auto` uses the pairs order and warns. |

### `samples/edge_cases.csv`

| Row | Case |
|---|---|
| 1 | Collinear isosceles triangle. |
| 2 | Legs shorter than `EPS`. |
| 3 | Non-coplanar trapezoid with equal legs. |
| 4 | Point count 5. |
| 5 | Non-numeric coordinate. |
| 6 | Too few columns. |
| 7 | Valid triangle with every sample at negative `z` (outside a normal volume). |
| 8 | Valid 3-point triangle with extra trailing numbers (ignored, with a warning). |

### `samples/orbit_64.csv`

Thirty-six rectangles through the center of a `64³` volume, rotating around the y axis. Regenerate with:

```bash
python3 scripts/make_orbit_csv.py samples/orbit_64.csv --size 64 --frames 36 --half 28
```

The script refuses to write a row that fails validation. The rectangle is stored in the Q&A column order, so `auto` accepts it without the pairs-order warning. `--half` is half the side length of the rectangle in voxels. `28` keeps the square inside a volume of size 64 for every angle.
