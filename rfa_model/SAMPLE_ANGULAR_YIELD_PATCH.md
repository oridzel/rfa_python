# Cu sample incidence-angle-dependent yield patch

This patch replaces the **sample only** capped-secant angular yield law with the
JMONSEL Cu table in `yieldsCu100TDDFT10ED.csv`.

## What changes

For every impact on the Cu sample, including later cascade electrons that return
and hit the sample again, the code computes the actual incidence angle from the
incoming velocity and the sample surface normal and uses separate JMONSEL gains

`G_SE(E,theta)  = SEY_JMONSEL(E,theta)  / SEY_JMONSEL(E,0)`

`G_BSE(E,theta) = BSEY_JMONSEL(E,theta) / BSEY_JMONSEL(E,0)`

The RFA model's existing normal-incidence Cu curves remain the absolute yield
normalization:

`SEY_used  = SEY_current_normal(E)  * G_SE(E,theta)`

`BSEY_used = BSEY_current_normal(E) * G_BSE(E,theta)`

So this patch does **not** replace the established normal-incidence sample yield
with the absolute numbers from the new TDDFT table.

## Interpolation

The supplied table is a complete 23-energy x 15-angle grid. Interpolation is
linear in incidence angle and linear in log(incident energy). Queries are
clamped to the tabulated range. Angles above 89 degrees use the 89-degree row;
there is no extrapolation toward 90 degrees.

The old `cos_theta >= 0.05` floor was removed from the sample emission path so
that 88-89 degree impacts query their true incidence angle. The existing
`angular_yield_gain()` remains capped and safe for all other solid surfaces.

## Installation

Replace these files in your `rfa_model` package:

- `samplers.py`
- `cascade.py`

`plotting.py` is also included and is the latest energy-colored/solid-STL
trajectory plotting version from the previous patch.

Copy this data file into the same `model_dir` that contains the other sampler
CSVs:

- `yieldsCu100TDDFT10ED.csv`

No notebook change is required if you already call `load_default_surface_models`
with its normal arguments. The new option defaults to on:

```python
yield_models, energy_models, theta_models = load_default_surface_models(
    model_dir,
    bronstein_dir=bronstein_dir,
    use_measured_carbon_coating=True,
    use_measured_carbon_for_grids=True,
)
```

To reproduce the historical sample secant model explicitly:

```python
yield_models, energy_models, theta_models = load_default_surface_models(
    model_dir,
    bronstein_dir=bronstein_dir,
    use_measured_carbon_coating=True,
    use_measured_carbon_for_grids=True,
    use_sample_angle_dependent_yields=False,
)
```

If the new model is enabled but the CSV is missing from `model_dir`, the loader
raises a clear `FileNotFoundError` rather than silently falling back to the old
law.

## Quick audit

After loading the models:

```python
from rfa_model.samplers import describe_sample_angular_yields

describe_sample_angular_yields(yield_models)
```

This prints the separate SEY and BSEY gains at 200, 500, 1000, 2000, 5000 and
10000 eV for 0, 30, 45, 60, 75, 80 and 85 degrees.

The cascade log now also distinguishes sample-table angular handling from the
capped secant law and records separate `terminal_sey_gain_used` and
`terminal_bsey_gain_used` for sample re-impacts.

## Validation performed

- `samplers.py`, `cascade.py`, and `plotting.py` pass `py_compile`.
- At 0 degrees the new and old sample event samplers are bit-for-bit identical
  for the same RNG seed because both gains are exactly 1.
- At 1 keV and 80 degrees the code reproduces the table ratios exactly:
  - SEY gain = 1.4261682243
  - BSEY gain = 1.9795918367
- An 89-degree incidence is preserved as 89 degrees rather than being truncated
  to the old `acos(0.05) = 87.13 deg` floor.
- Monte Carlo sampling at 1 keV / 80 degrees agrees with the expected scaled
  SEY and BSEY within statistical error.
