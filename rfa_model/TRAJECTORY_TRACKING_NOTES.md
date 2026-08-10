# RFA trajectory tracking update — 2026-08-10

## What changed

1. **Continuous sample plane is now the physical collision surface at grazing incidence.**
   The voxelized rotated sample remains an electrostatic boundary condition, but a fixed sample voxel that protrudes into physical vacuum no longer automatically terminates an electron. A local ballistic bypass traverses only the sample-voxel staircase while checking the finite continuous sample plane and STL geometry.

2. **Point-by-point trajectories are saved automatically when present.**
   `save_cascade_batch_tables(...)` now writes `<prefix>_trajectories.npz` whenever tracked primary or cascade trajectories exist. The NPZ is compressed and pickle-free; variable-length paths are stored as concatenated point/velocity arrays plus offsets.

3. **Primary CSV contains beam-steering diagnostics.**
   New columns include launch position/velocity, hit velocity, launch-to-hit distance, direction-change angle, incidence cosine/angle, and whether a sample-voxel artifact traversal occurred.

4. **Primary trajectories can be plotted directly.**
   `plotting.py` adds `plot_primary_trajectory_projections(...)` and `plot_primary_trajectories_3d(...)`.

5. **Saved NPZ trajectories can be reloaded later.**
   Use `load_tracked_trajectories_npz(...)` from `plotting.py`.

6. **Optional farther-upstream primary launch.**
   `run_cascade_batch_parallel(...)` now accepts `primary_launch_distance_m=None`. When supplied, this is an explicit along-beam launch distance from the geometric sample-plane intersection. The old `primary_launch_clearance_h=2.0` behavior remains the default.

## Suggested 80-degree beam-steering test

```python
cascade = rfa.run_cascade_batch_parallel(
    ...,
    sample_theta_deg=80.0,
    track_points=True,
    track_stride=1,
    track_primary_only=True,
    tracked_primary_indices=set(range(1000)),
    primary_launch_distance_m=0.010,  # example: 10 mm upstream along beam
)

paths = rfa.save_cascade_batch_tables(
    cascade,
    out_dir,
    prefix="cascade_cu_1000eV_BSEY_alpha_80_traj",
)
print(paths.get("trajectories"))
```

Reload later:

```python
from rfa_model.plotting import (
    load_tracked_trajectories_npz,
    plot_primary_trajectory_projections,
)

tracks = load_tracked_trajectories_npz(
    "cascade_cu_1000eV_BSEY_alpha_80_traj_trajectories.npz"
)

fig, axes = plot_primary_trajectory_projections(
    tracks["primary_results"],
    n=100,
)
```

For a controlled 0 V versus +50 V steering comparison, use the same sample angle, seed, beam size, and launch distance in both runs.


## Sample geometry v3

The analytic sample face is no longer centred from the post-rotation axis-aligned
sample bounds. `run_cascade_batch_parallel()` now infers the exposed front face
from sample-owned triangles in the primary collision STL and uses that face for:

- analytic launch-plane centre and normal,
- beam centring,
- finite analytic sample return geometry.

A sample-owned STL collision is now classified as `reason="hit_sample"` while
retaining `kind="stl"` and adding `sample_stl_hit=True`. Thus the primary STL is
the authoritative physical sample surface, while the analytic face is a matching
continuous geometry used for launch and return logic.

At startup, verbose output reports the analytic face `center`, `normal`, geometry
`source`, and plane offset. For a sample whose exposed face pivots through the
origin, `plane_offset` should be approximately zero at every angle.
