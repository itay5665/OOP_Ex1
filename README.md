# Plane projection of a 3D volume

Course project for Introduction to Software Engineering (OOP / OOD).

Given a 3D raster volume and a plane, the program builds the 2D image of that
plane and writes one video frame per row of a CSV file. The original problem
statement leaves several choices open (sampling, image size, invalid rows,
video layout). Those choices are fixed in [docs/SRS.md](docs/SRS.md) and the
sampling math is written out in [docs/ALGORITHM.md](docs/ALGORITHM.md).

## Problem

`M` is a 3D array of unsigned 8-, 16-, or 32-bit integers. Each axis length
`n` satisfies `1 <= n < 1024`. A plane is an isosceles triangle (3 points) or
an isosceles trapezoid (4 points). The program samples `M` over that shape and
returns a 2D image. Coordinates are voxel indices: `M[x, y, z]`.

## Scope

Included:

- Reading an NRRD volume and a CSV list of planes.
- Validating each plane and projecting it.
- Writing an MP4 with one frame per CSV row, in order.
- A nearest-neighbour sampler and an optional trilinear sampler.
- Tests, a benchmark, and a short experiment.

Not included:

- Physical millimetre coordinates (NRRD spacing and origin are read and
  reported, then ignored).
- Interactive viewing, and any download of the lecturer's MRI file.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 Ex0.py data/volume.nrrd samples/Ex0.csv output/demo.mp4
```

The three positional arguments are the volume, the plane list, and the output
video. Useful flags:

```text
--fps 10                     video playback rate (default 10)
--sampler nearest|trilinear nearest is the default
--point-order auto|qa|pairs auto is the default (see SRS decision 8b)
--skip-invalid               drop invalid rows instead of writing an error frame
--normalize minmax|percentile
                             minmax is the default; percentile uses the 1st and 99th
```

Other commands, from the repository root:

```bash
python3 scripts/make_synthetic_nrrd.py data/sphere_u8.nrrd --kind sphere --size 64
python3 scripts/make_orbit_csv.py samples/orbit_64.csv --size 64 --frames 36
python3 scripts/benchmark.py
python3 -m pytest
python3 -m pytest -m slow
```

Put NRRD files in `data/`. That directory is gitignored because the files are large.

## Layout

```text
Ex0.py                  command-line entry point
projection/             library (no video I/O inside the projector)
scripts/                synthetic volumes, orbit CSV, benchmark
tests/                  pytest
docs/                   SRS, algorithm, dataset, testing, experiment
samples/                Ex0.csv, edge cases, orbit
data/                   local NRRD files (not in git)
```

## Performance

On a `500³` volume the projection step must reach at least 30 frames per
second. 100 FPS is the expected target. "FPS" here means projection
computations per second, not the video playback rate. `scripts/benchmark.py`
reports the two numbers separately from file loading and video encoding.

## Documentation

| Document | Contents |
|---|---|
| [docs/SRS.md](docs/SRS.md) | Requirements and every design decision |
| [docs/ALGORITHM.md](docs/ALGORITHM.md) | Corrected projection algorithm |
| [docs/DATASET.md](docs/DATASET.md) | NRRD and CSV data |
| [docs/TESTING.md](docs/TESTING.md) | What is tested, including the demo |
| [docs/EXPERIMENT.md](docs/EXPERIMENT.md) | Measured results |

## Requirements

Python 3.10 or newer. Libraries are listed in `requirements.txt`.
