# Solid STL + trajectory plotting

Use the real aligned `Trimesh` objects returned by the geometry loaders, then overlay the saved point-by-point trajectories with Plotly `Mesh3d` surfaces.

For an all-trajectories presentation run:

```python
fig = plot_saved_trajectories_with_stls_plotly(
    trajectory_npz,
    meshes=meshes,
    frame_meshes=frame_meshes,
    n_primary=None,
    n_cascade=None,
    geometry_opacity=0.22,
    trajectory_width=4.0,
    cascade_width=2.0,
    color_cascade_by_kind=True,
)
fig.show()
```

`None` means all tracked trajectories. Use `n_cascade=0` to suppress cascade trajectories.

For sample-bias demonstration runs, enable `track_sub_barrier_sample_emissions=True` during the simulation. The returning sub-barrier trajectories are rendered magenta and are excluded from the normal current/yield accounting.
