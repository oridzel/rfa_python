Added `cascade_filter` to `plot_stl_trajectories_plotly()` and
`plot_saved_trajectories_with_stls_plotly()`.

Allowed values:
- `"all"` (default)
- `"returned_to_sample"`
- `"sub_barrier_return"`
- `"not_returned_to_sample"`

Example:
```python
fig = rplt.plot_saved_trajectories_with_stls_plotly(
    paths["trajectories"],
    meshes=meshes,
    frame_meshes=frame_meshes,
    visual_meshes=visual_shell_meshes,
    visual_parts=visual_shell_parts,
    n_primary=None,
    n_cascade=None,
    trajectory_color_by="energy",
    energy_scale="log",
    energy_range_eV=(0.1, 1100.0),
    cascade_filter="returned_to_sample",
    show_trajectory_legend=False,
)
```
