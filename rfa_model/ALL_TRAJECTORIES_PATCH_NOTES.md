# All-trajectories / sub-barrier visualization patch

This patch adds a presentation mode for sample-bias simulations without changing the historical current/yield accounting.

## New simulation option

```python
track_sub_barrier_sample_emissions=True
```

Default is `False`, so ordinary runs behave exactly as before.

When enabled, sample-emitted SE/BSE electrons whose surface energy is below the positive sample-bias escape threshold are launched and integrated instead of being discarded. They are tagged:

```text
sub_barrier=True
escape_eligible=False
visualization_only=True
```

These trajectories use an independent per-primary RNG, do not consume the ordinary physics cascade electron budget, do not create child emissions after returning, and are excluded from current/yield/grid-event accounting. This keeps the normal simulated results unchanged while making the biased return paths visible.

## Track every trajectory for a representative run

```python
result = rfa.run_cascade_batch_parallel(
    ...,
    N_primary=100,
    track_points=True,
    track_stride=2,
    track_primary_only=False,
    tracked_primary_indices=None,       # all primaries
    track_sub_barrier_sample_emissions=True,
)

paths = rfa.save_cascade_batch_tables(
    result,
    out_dir=out_dir,
    prefix=prefix,
    save_trajectories=True,
)
```

For a presentation image, 50-150 primaries is usually more readable than tracking thousands.

## Plot ALL saved trajectories over solid aligned STL parts

```python
from rfa_model.plotting import plot_saved_trajectories_with_stls_plotly

fig = plot_saved_trajectories_with_stls_plotly(
    paths["trajectories"],
    meshes=meshes,
    frame_meshes=frame_meshes,
    n_primary=None,     # ALL tracked primaries
    n_cascade=None,     # ALL tracked cascade electrons
    geometry_opacity=0.22,
    trajectory_width=4.0,
    cascade_width=2.0,
    color_cascade_by_kind=True,
    title="1 keV sample-bias trajectories",
)
fig.show()
```

Color convention in the solid-STL Plotly view:

- primary electrons: crimson
- ordinary SEs: deepskyblue
- BSEs: darkorange
- sub-barrier return electrons: magenta
- quantum reflections: mediumseagreen

`n_cascade=0` still means do not draw cascade trajectories. `None` means draw all.

## Files changed

- `samplers.py`: optional generation/tagging of sub-barrier sample emissions with a separate visualization RNG.
- `cascade.py`: threads the option, tracks visualization-only particles, preserves the physics RNG/caps/accounting, and saves the new metadata in CSV/NPZ.
- `plotting.py`: loads the new metadata, remains backward-compatible with old trajectory NPZ files, supports `None = all`, and colors trajectory classes separately on solid STL geometry.

`primary.py` and `trajectories.py` are included unchanged from the current aligned-sample-geometry package.
