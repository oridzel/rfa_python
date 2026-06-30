import numpy as np
import plotly.graph_objects as go


def mesh_to_plotly(
    mesh,
    name,
    color="lightgray",
    opacity=0.35,
    max_faces=15000,
):
    """
    Convert trimesh mesh to a Plotly Mesh3d trace.
    Decimates by random face subsampling if mesh is large.
    """
    V = mesh.vertices
    F = mesh.faces

    if len(F) > max_faces:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(F), size=max_faces, replace=False)
        Fp = F[idx]
    else:
        Fp = F

    return go.Mesh3d(
        x=V[:, 0],
        y=V[:, 1],
        z=V[:, 2],
        i=Fp[:, 0],
        j=Fp[:, 1],
        k=Fp[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=True,
        showscale=False,
    )


def add_line(fig, p0, p1, name, color="black", width=5):
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)

    fig.add_trace(go.Scatter3d(
        x=[p0[0], p1[0]],
        y=[p0[1], p1[1]],
        z=[p0[2], p1[2]],
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
    ))


def add_point(fig, p, name, color="red", size=5):
    p = np.asarray(p)

    fig.add_trace(go.Scatter3d(
        x=[p[0]],
        y=[p[1]],
        z=[p[2]],
        mode="markers",
        name=name,
        marker=dict(color=color, size=size),
    ))


def plot_rfa_geometry_alignment(
    meshes,
    frame_meshes=None,
    sample_y_bounds=None,
    sample_z_bounds=None,
    show_grid_frames=True,
    show_rays=True,
    x_ray_end=0.025,
):
    fig = go.Figure()

    colors = {
        "sample": "gold",
        "holder": "gray",
        "receiver": "lightblue",
        "rod": "black",
    }

    opacities = {
        "sample": 0.75,
        "holder": 0.35,
        "receiver": 0.25,
        "rod": 0.4,
    }

    for name in ["sample", "holder", "receiver", "rod"]:
        if name in meshes:
            fig.add_trace(mesh_to_plotly(
                meshes[name],
                name=name,
                color=colors.get(name, "lightgray"),
                opacity=opacities.get(name, 0.35),
                max_faces=12000,
            ))

    if show_grid_frames and frame_meshes is not None:
        frame_colors = {
            "g1_low_frame": "orange",
            "g1_upper_frame": "orange",
            "g2_low_frame": "green",
            "g2_upper_frame": "green",
            "g3_low_frame": "purple",
            "g3_upper_frame": "purple",
        }

        for name, mesh in frame_meshes.items():
            fig.add_trace(mesh_to_plotly(
                mesh,
                name=name,
                color=frame_colors.get(name, "purple"),
                opacity=0.18,
                max_faces=6000,
            ))

    # RFA axis: +X through y=0,z=0.
    add_line(
        fig,
        p0=[-0.005, 0.0, 0.0],
        p1=[0.085, 0.0, 0.0],
        name="RFA axis y=0,z=0",
        color="red",
        width=6,
    )

    if sample_y_bounds is None or sample_z_bounds is None:
        bounds = meshes["sample"].bounds
        sample_y_bounds = bounds[:, 1]
        sample_z_bounds = bounds[:, 2]

    y_center = 0.5 * (sample_y_bounds[0] + sample_y_bounds[1])
    z_center = 0.5 * (sample_z_bounds[0] + sample_z_bounds[1])

    sample_center = np.array([0.0, y_center, z_center])

    add_point(
        fig,
        sample_center,
        name="sample center",
        color="red",
        size=6,
    )

    if show_rays:
        add_line(
            fig,
            p0=[0.0, y_center, z_center],
            p1=[x_ray_end, y_center, z_center],
            name="straight +X from sample center",
            color="red",
            width=5,
        )

        # Also show a straight line on the RFA axis.
        add_line(
            fig,
            p0=[0.0, 0.0, 0.0],
            p1=[x_ray_end, 0.0, 0.0],
            name="straight +X on RFA axis",
            color="blue",
            width=5,
        )

    fig.update_layout(
        title="RFA geometry after alignment",
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        width=1000,
        height=800,
        legend=dict(itemsizing="constant"),
    )

    return fig