# Solid STL trajectory plotting

The new Plotly helpers render the actual aligned `trimesh.Trimesh` STL surfaces as filled, shaded `go.Mesh3d` objects, matching the solid-part style used in `simelec.ipynb`.

## Typical 80 degree run

```python
from pathlib import Path
from rfa_model.plotting import plot_saved_trajectories_with_stls_plotly

stl_dir = Path("rfa stl")

# IMPORTANT: use the same sample rotation as the simulation run.
meshes, sample_parts = rfa.load_and_align_sample_assembly(
    stl_dir,
    alpha_deg=80.0,
)
frame_meshes, frame_parts, frame_info = rfa.load_and_align_grid_frames(
    stl_dir,
    verbose=False,
)

fig = plot_saved_trajectories_with_stls_plotly(
    "cascade_cu_1000eV_BSEY_Tg=0p93_alpha_80_5000_traj_trajectories.npz",
    meshes=meshes,
    frame_meshes=frame_meshes,
    n_primary=80,
    n_cascade=0,
    geometry_opacity=0.28,
    trajectory_width=4.0,
    show_hits=True,
    title="1 keV primaries, +50 V sample bias, alpha=80 deg",
)
fig.show()
```

Any additional real STL (for example an aligned drift-tube STL) can be added to `meshes` or `frame_meshes` before plotting. Analytic grid-wire/collector shells are not STL bodies and are therefore not drawn by this real-STL renderer unless explicitly added as separate analytic geometry.
