# RFA Model

Python model for electron trajectory tracking, secondary/backscattered electron emission, cascade emission, and electrode-current accounting in a spherical retarding-field analyzer.

## Current status

Implemented:

- STL geometry alignment
- voxel field loading
- electric-field interpolation
- trajectory integration
- analytic grid/collector spherical surfaces
- stochastic grid transparency
- JMONSEL/Bronstein yield and sampler loading
- sampler diagnostics
- primary beam tracking
- first-generation sample-emission batch runs
- cascade emission
- cascade current accounting
- serial and parallel batch runners

## Coordinate convention

The RFA axis is +X. The sample outward normal points toward +X. Primary electrons start at small positive x and travel toward -X to hit the sample. Emitted electrons launch generally toward +X.

## Data files

Large files are not tracked by Git. Place local data in `data/`, for example:

- STL geometry files
- field `.npz` files
- JMONSEL sampler CSV files
- Bronstein yield CSV files

See `data/README.md`.

## Basic usage

```python
from pathlib import Path
import numpy as np

from rfa_model.io import load_field_npz
from rfa_model.fields import (
    build_field_interpolators,
    build_potential_interpolator,
    attach_default_owner_name_map,
)
from rfa_model.samplers import load_default_surface_models
from rfa_model.cascade import run_cascade_batch_parallel, print_cascade_batch_summary

field = load_field_npz("data/field_500eV.npz")
field = attach_default_owner_name_map(field)

Ex_interp, Ey_interp, Ez_interp = build_field_interpolators(field)
Phi_interp = build_potential_interpolator(field)