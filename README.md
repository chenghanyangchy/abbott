# abbott
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI Status](https://github.com/pelkmanslab/abbott/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/pelkmanslab/abbott/actions/workflows/build_and_test.yml)
[![codecov](https://codecov.io/github/pelkmanslab/abbott/graph/badge.svg?token=BF9NP4YLO6)](https://codecov.io/github/pelkmanslab/abbott)

3D Multiplexed Image Analysis Workflows

## Available Tasks

| Task | Description | Passing |
| --- | --- | --- |
| Convert Cellvoyager Multiplexing to existing OME-Zarr | Converts CV7000/CV8000 images and extends to existing OME-Zarr file.| WIP |
| Compute Registration (elastix) | Compute rigid/affine/b-spline registration for aligning images in multiplexed image analysis.|x|
| Apply Registration (elastix) | Apply rigid/affine/b-spline registration to images.|x|

## Installation

To install this task package on a Fractal server, get the whl in the Github release and use the local task collection.
To install this package locally:
```
git clone https://github.com/pelkmanslab/abbott
cd abbott
pip install -e .
```

For development:
```
git clone https://github.com/pelkmanslab/abbott
cd abbott
pip install -e ".[dev]"
```
