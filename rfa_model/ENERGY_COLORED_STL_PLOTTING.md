# Solid STL + energy-colored trajectory plotting

The Plotly trajectory renderer now supports:

- `visual_meshes`: additional presentation-only STL meshes such as collector and shield shells.
- `visual_parts`: `SimpleNamespace`/dict descriptors with per-part `name`, `color`, and `opacity`.
- `part_opacities`: optional direct per-part opacity overrides.
- `trajectory_color_by="energy"`: color at each saved trajectory point by instantaneous electron kinetic energy calculated from the saved velocity.
- `energy_scale="log"` (default): recommended for sub-eV SEs through ~keV primaries.
- `energy_scale="linear"`: optional linear color mapping.
- `energy_range_eV=(emin, emax)`: optional fixed range for consistent color scales across several figures.

Example using the collector/shield objects already created in the notebook:

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
    energy_colorscale="Turbo",
    energy_range_eV=(0.1, 1100.0),
    geometry_opacity=0.22,
    trajectory_width=4.0,
    cascade_width=2.0,
    title="1 keV sample-bias trajectories",
)
fig.show()
```

For normal-incidence and grazing-angle figures that should be directly comparable, use the same explicit `energy_range_eV`, e.g. `(0.1, 1100.0)` for 1 keV landing-energy runs.

The color is kinetic energy, not total mechanical energy. It is derived point-by-point from the trajectory's saved velocity as `0.5*m_e*|v|^2/e`.
