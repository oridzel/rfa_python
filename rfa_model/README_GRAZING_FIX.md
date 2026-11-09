# RFA grazing-incidence joint-sampler update

This update addresses two effects exposed by the 89° sample-bias BSEY run:

1. The electrostatic field changes the **actual primary impact angle**, so the
   first-generation sample joint sampler must be selected from the impact
   direction rather than from the mechanical sample angle.
2. The voxelized sample boundary can protrude into analytic vacuum at grazing
   angles. For emitted electrons, sample-owned fixed voxels in analytic vacuum
   must not be treated as physical collisions.

## 1. Files

Replace the current RFA files with:

```text
rfa_model/samplers.py   <- samplers.py
rfa_model/cascade.py    <- cascade.py
```

The supplied `cascade.py` includes the earlier first-generation sample-normal
fix: gun-to-sample joint emission uses the authoritative analytic sample normal.

For the emitted-electron tracker, apply the patch to your **current**
`trajectories.py` so unrelated recent changes are preserved:

```bash
python apply_emitted_grazing_voxel_fix.py rfa_model/trajectories.py
```

The patcher creates:

```text
rfa_model/trajectories.py.pre_grazing_emit_fix.bak
```

and runs `py_compile` after patching.

## 2. Multi-angle joint sampler catalog

`load_default_surface_models()` now accepts a mapping from incidence angle to
sampler directory:

```python
from pathlib import Path

root = Path("sampler_library/Cu_B50_abrupt_grazing_joint")

joint_sampler_dirs = {
    float(a): root / f"alpha_{a}deg"
    for a in range(80, 90)
}

yield_models, energy_models, theta_models = load_default_surface_models(
    model_dir=model_dir,
    bronstein_dir=bronstein_dir,
    use_measured_carbon_coating=True,
    use_measured_carbon_for_grids=True,
    sample_quantum_reflection_mode="disabled",
    sample_gun_incidence_dirs=joint_sampler_dirs,
    sample_gun_incidence_selection="stochastic_bracket",
    sample_gun_incidence_tolerance_deg=0.5,
    require_sample_gun_joint_sampler=True,
)
```

The old single-angle API remains supported:

```python
sample_gun_incidence_dir=...
sample_gun_incidence_angle_deg=75.0
```

Do not pass the single-angle and multi-angle APIs at the same time.

## 3. How angle selection works

For every first-generation gun -> sample hit, the code computes

```text
actual_angle = acos(-v_in_hat . n_sample)
```

using the **actual impact velocity after electrostatic deflection** and the
analytic physical sample normal.

With `sample_gun_incidence_selection="stochastic_bracket"`, a hit between two
available angles chooses one complete lower/upper joint population with linear
probability. Example:

```text
actual angle = 87.97°
87° table weight ~ 0.03
88° table weight ~ 0.97
```

This is preferable to interpolating individual E/theta/phi vectors: every
sampled electron remains an event from a real SEEMC joint population, while the
ensemble yield and barrier-reflection fraction vary smoothly with angle.

At an exact tabulated angle, only that table is used.

The reconstructed joint direction uses the **actual primary beam-back axis**
`-v_in_hat`, not the nominal fixed +X axis. This keeps the joint coordinate
system consistent with the deflected primary.

The cascade output now records:

```text
sample_gun_incidence_actual_angle_deg
sample_gun_incidence_sampler_angle_deg
sample_gun_incidence_sampler_delta_deg
sample_gun_incidence_selection
```

along with the existing joint-sampler provenance.

## 4. Generate 1° grazing samplers with SEEMC

A single run can generate 80, 81, ..., 89° into one output root:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 80 81 82 83 84 85 86 87 88 89 \
  --primaries 100000 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --barrier-model abrupt \
  --workers 18 \
  --output sampler_library/Cu_B50_abrupt_grazing_joint
```

The current generator writes the six legacy CSVs plus the joint SE/BSE NPZ
samplers in each `alpha_*deg` directory.

For final high-grazing production libraries, 1M primaries can be used:

```bash
python3 generate_plane_sampler_grid.py MaterialDatabase.pkl \
  --material Cu \
  --angles-deg 86 87 88 89 \
  --primaries 1000000 \
  --elastic-low-energy-model browning \
  --elastic-cutoff-ev 50 \
  --barrier-model abrupt \
  --workers 18 \
  --output sampler_library/Cu_B50_abrupt_grazing_joint_1M
```

The 100k-vs-1M comparison at 89° showed that 100k was already well converged in
bulk BSEY/E/theta/phi; 1M mainly improves rare grazing tails and the precision
of the large specular-reflection population.

### Optional 79° guard table

If a mechanical 80° sample-bias run is deflected below 80°, add a 79° library
as an endpoint guard:

```bash
--angles-deg 79 80 81 82 83 84 85 86 87 88 89
```

The catalog deliberately refuses to extrapolate far outside its supplied angle
range.

## 5. Emitted-electron voxel-artifact behavior

The emitted-electron patch does **not** make a long straight-line geometric
jump through the voxel staircase.

That would be inappropriate for grazing BSEs: at 1 keV and ~89° a reflected
electron can have only a fraction of an eV of kinetic energy normal to the
sample, so the +50 V sample-bias field can physically turn it around.

Instead:

- the finite analytic sample plane remains the physical collision surface;
- a sample-owned fixed voxel lying on the analytic-vacuum side is ignored only
  as a collision label;
- the ordinary electrostatic integrator continues step-by-step;
- a genuine crossing onto the solid side of the finite analytic sample face is
  returned as `reason="hit_sample"`;
- STL, grids, holder, frame, collector, and other collision tests remain active.

After patching, the old grazing-specific emitted trajectory termination

```text
sample_voxel_artifact_failed
```

should disappear or be reduced to zero rather than being counted as a physical
sample return.

## 6. Recommended validation after rerunning 89°

Check:

```python
# Primary tracking
result["df_primary"]["reason"].value_counts()

# Emitted/cascade trajectories
result["df_cascade"]["reason"].value_counts()

# Actual impact-angle distribution
result["df_primary"].loc[
    result["df_primary"]["reason"].eq("hit_sample"),
    "incidence_angle_deg",
].describe()

# Which angle tables were actually selected
result["df_cascade"].loc[
    result["df_cascade"]["generation"].eq(1),
    [
        "sample_gun_incidence_actual_angle_deg",
        "sample_gun_incidence_sampler_angle_deg",
        "sample_gun_incidence_sampler_delta_deg",
    ],
].value_counts()
```

For a mechanical 89° / +50 V sample-bias run similar to the previous case, the
selected table population should be dominated by ~88° rather than 89°.
