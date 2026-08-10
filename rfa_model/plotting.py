# rfa_model/plotting.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _to_mm(a):
    return 1e3 * np.asarray(a, dtype=float)


def _get_traj(res):
    traj = res.get("traj", None)
    if traj is None:
        return None

    traj = np.asarray(traj, dtype=float)

    if traj.ndim != 2 or traj.shape[1] != 3 or len(traj) < 2:
        return None

    return traj


def _tracked_result_indices(cascade_results_all):
    """Return list indices for cascade results that actually store a trajectory."""
    return [
        i for i, res in enumerate(cascade_results_all)
        if _get_traj(res) is not None
    ]


def _set_axes_equal_3d(ax):
    """
    Make 3D axes approximately equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def select_cascade_indices(
    df_cascade: pd.DataFrame,
    n: int = 20,
    seed: int = 1,
    terminal_electrode: str | None = None,
    source_electrode: str | None = None,
    generation: int | None = None,
    emission_kind: str | None = None,
    reason: str | None = None,
    slowest: bool = False,
):
    """
    Select row indices from df_cascade for trajectory plotting.

    Returns dataframe indices, which should correspond to the same order as
    cascade_results_all.
    """
    df = df_cascade.copy()

    mask = pd.Series(True, index=df.index)

    if terminal_electrode is not None:
        mask &= df["terminal_electrode"] == terminal_electrode

    if source_electrode is not None:
        mask &= df["source_electrode"] == source_electrode

    if generation is not None:
        mask &= df["generation"] == generation

    if emission_kind is not None:
        mask &= df["emission_kind"] == emission_kind

    if reason is not None:
        mask &= df["reason"] == reason

    df_sel = df.loc[mask]

    if df_sel.empty:
        return []

    if slowest:
        return list(df_sel.sort_values("steps", ascending=False).head(n).index)

    rng = np.random.default_rng(seed)

    if len(df_sel) <= n:
        return list(df_sel.index)

    return list(rng.choice(df_sel.index.to_numpy(), size=n, replace=False))


def plot_trajectories_3d(
    cascade_results_all: list[dict],
    df_cascade: pd.DataFrame | None = None,
    indices=None,
    n: int = 20,
    seed: int = 1,
    terminal_electrode: str | None = None,
    source_electrode: str | None = None,
    generation: int | None = None,
    emission_kind: str | None = None,
    reason: str | None = None,
    slowest: bool = False,
    show_hits: bool = True,
    title: str | None = None,
    ax=None,
):
    """
    Plot selected cascade trajectories in 3D.

    Coordinates are shown in mm.
    """
    if indices is None:
        tracked_indices = _tracked_result_indices(cascade_results_all)
        if df_cascade is None:
            indices = tracked_indices[:n]
        else:
            # df_cascade is generated in the same order as
            # cascade_results_all. Restrict selection to rows for which point
            # tracking was actually enabled, otherwise a random request for N
            # trajectories may silently select mostly traj=None rows.
            available = df_cascade.loc[
                df_cascade.index.intersection(tracked_indices)
            ]
            indices = select_cascade_indices(
                available,
                n=n,
                seed=seed,
                terminal_electrode=terminal_electrode,
                source_electrode=source_electrode,
                generation=generation,
                emission_kind=emission_kind,
                reason=reason,
                slowest=slowest,
            )

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    plotted = 0

    for idx in indices:
        if idx < 0 or idx >= len(cascade_results_all):
            continue

        res = cascade_results_all[idx]
        traj = _get_traj(res)

        if traj is None:
            continue

        tr = _to_mm(traj)

        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], linewidth=0.8, alpha=0.8)

        if show_hits:
            ax.scatter(
                tr[0, 0], tr[0, 1], tr[0, 2],
                marker="o",
                s=15,
                alpha=0.8,
            )
            ax.scatter(
                tr[-1, 0], tr[-1, 1], tr[-1, 2],
                marker="x",
                s=25,
                alpha=0.9,
            )

        plotted += 1

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    if title is None:
        title = f"Cascade trajectories, N = {plotted}"

    ax.set_title(title)

    if plotted > 0:
        _set_axes_equal_3d(ax)

    return fig, ax


def plot_trajectory_projections(
    cascade_results_all: list[dict],
    df_cascade: pd.DataFrame | None = None,
    indices=None,
    n: int = 20,
    seed: int = 1,
    terminal_electrode: str | None = None,
    source_electrode: str | None = None,
    generation: int | None = None,
    emission_kind: str | None = None,
    reason: str | None = None,
    slowest: bool = False,
    show_hits: bool = True,
    title: str | None = None,
):
    """
    Plot x-y, x-z, and y-z projections of selected trajectories.
    """
    if indices is None:
        tracked_indices = _tracked_result_indices(cascade_results_all)
        if df_cascade is None:
            indices = tracked_indices[:n]
        else:
            # df_cascade is generated in the same order as
            # cascade_results_all. Restrict selection to rows for which point
            # tracking was actually enabled, otherwise a random request for N
            # trajectories may silently select mostly traj=None rows.
            available = df_cascade.loc[
                df_cascade.index.intersection(tracked_indices)
            ]
            indices = select_cascade_indices(
                available,
                n=n,
                seed=seed,
                terminal_electrode=terminal_electrode,
                source_electrode=source_electrode,
                generation=generation,
                emission_kind=emission_kind,
                reason=reason,
                slowest=slowest,
            )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        (0, 1, "x (mm)", "y (mm)", "x-y"),
        (0, 2, "x (mm)", "z (mm)", "x-z"),
        (1, 2, "y (mm)", "z (mm)", "y-z"),
    ]

    plotted = 0

    for idx in indices:
        if idx < 0 or idx >= len(cascade_results_all):
            continue

        res = cascade_results_all[idx]
        traj = _get_traj(res)

        if traj is None:
            continue

        tr = _to_mm(traj)

        for ax, (i, j, xlabel, ylabel, panel_title) in zip(axes, panels):
            ax.plot(tr[:, i], tr[:, j], linewidth=0.8, alpha=0.8)

            if show_hits:
                ax.scatter(tr[0, i], tr[0, j], marker="o", s=12, alpha=0.8)
                ax.scatter(tr[-1, i], tr[-1, j], marker="x", s=22, alpha=0.9)

        plotted += 1

    for ax, (_, _, xlabel, ylabel, panel_title) in zip(axes, panels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.set_aspect("equal", adjustable="box")

    if title is None:
        title = f"Cascade trajectory projections, N = {plotted}"

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_hit_points(
    df_cascade: pd.DataFrame,
    plane: str = "yz",
    color_by: str = "terminal_electrode",
    terminal_electrode: str | None = None,
    source_electrode: str | None = None,
    generation: int | None = None,
    emission_kind: str | None = None,
    reason: str | None = None,
    ax=None,
    title: str | None = None,
    s: float = 8,
    alpha: float = 0.7,
):
    """
    Plot terminal hit points from df_cascade.

    plane can be:
        "xy", "xz", or "yz".

    Coordinates are shown in mm.
    """
    if plane not in ["xy", "xz", "yz"]:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    df = df_cascade.copy()

    mask = pd.Series(True, index=df.index)

    if terminal_electrode is not None:
        mask &= df["terminal_electrode"] == terminal_electrode

    if source_electrode is not None:
        mask &= df["source_electrode"] == source_electrode

    if generation is not None:
        mask &= df["generation"] == generation

    if emission_kind is not None:
        mask &= df["emission_kind"] == emission_kind

    if reason is not None:
        mask &= df["reason"] == reason

    df = df.loc[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    plane_map = {
        "xy": ("x_hit", "y_hit", "x (mm)", "y (mm)"),
        "xz": ("x_hit", "z_hit", "x (mm)", "z (mm)"),
        "yz": ("y_hit", "z_hit", "y (mm)", "z (mm)"),
    }

    xcol, ycol, xlabel, ylabel = plane_map[plane]

    if color_by is None or color_by not in df.columns:
        ax.scatter(
            1e3 * df[xcol],
            1e3 * df[ycol],
            s=s,
            alpha=alpha,
        )
    else:
        for label, group in df.groupby(color_by):
            ax.scatter(
                1e3 * group[xcol],
                1e3 * group[ycol],
                s=s,
                alpha=alpha,
                label=str(label),
            )
        ax.legend(markerscale=2, fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")

    if title is None:
        title = f"Hit points on {plane} plane"

    ax.set_title(title)

    return fig, ax


def plot_hit_points_3d(
    df_cascade: pd.DataFrame,
    color_by: str = "terminal_electrode",
    terminal_electrode: str | None = None,
    source_electrode: str | None = None,
    generation: int | None = None,
    emission_kind: str | None = None,
    reason: str | None = None,
    ax=None,
    title: str | None = None,
    s: float = 8,
    alpha: float = 0.7,
):
    """
    Plot terminal hit points in 3D.
    """
    df = df_cascade.copy()

    mask = pd.Series(True, index=df.index)

    if terminal_electrode is not None:
        mask &= df["terminal_electrode"] == terminal_electrode

    if source_electrode is not None:
        mask &= df["source_electrode"] == source_electrode

    if generation is not None:
        mask &= df["generation"] == generation

    if emission_kind is not None:
        mask &= df["emission_kind"] == emission_kind

    if reason is not None:
        mask &= df["reason"] == reason

    df = df.loc[mask]

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if color_by is None or color_by not in df.columns:
        ax.scatter(
            1e3 * df["x_hit"],
            1e3 * df["y_hit"],
            1e3 * df["z_hit"],
            s=s,
            alpha=alpha,
        )
    else:
        for label, group in df.groupby(color_by):
            ax.scatter(
                1e3 * group["x_hit"],
                1e3 * group["y_hit"],
                1e3 * group["z_hit"],
                s=s,
                alpha=alpha,
                label=str(label),
            )
        ax.legend(markerscale=2, fontsize=8)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    if title is None:
        title = "3D hit points"

    ax.set_title(title)

    if len(df) > 0:
        _set_axes_equal_3d(ax)

    return fig, ax


def plot_terminal_counts(
    df_cascade: pd.DataFrame,
    normalize: bool = True,
    N_primary: int | None = None,
    ax=None,
    title: str | None = None,
):
    """
    Plot terminal electrode counts from df_cascade.
    """
    counts = df_cascade["terminal_electrode"].value_counts()

    order = [
        "sample",
        "holder",
        "receiver",
        "rod",
        "grid1",
        "grid2",
        "grid3",
        "collector",
        "drifttube",
        "escaped",
        "unknown",
    ]

    counts = counts.reindex(order, fill_value=0)

    if normalize:
        if N_primary is None:
            raise ValueError("N_primary is required when normalize=True")
        values = counts / N_primary
        ylabel = "Terminal arrivals per primary"
    else:
        values = counts
        ylabel = "Terminal arrivals"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    values.plot(kind="bar", ax=ax)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Terminal electrode")

    if title is None:
        title = "Terminal electrode distribution"

    ax.set_title(title)
    fig.tight_layout()

    return fig, ax


def plot_current_balance(
    current_counts: pd.DataFrame,
    column: str = "net_per_primary",
    ax=None,
    title: str | None = None,
):
    """
    Plot cascade current/electron balance by electrode.
    """
    if column not in current_counts.columns:
        raise ValueError(f"{column!r} not found in current_counts")

    order = [
        "sample",
        "holder",
        "receiver",
        "rod",
        "grid1",
        "grid2",
        "grid3",
        "collector",
        "drifttube",
        "escaped",
        "unknown",
    ]

    values = current_counts[column].reindex(order)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    values.plot(kind="bar", ax=ax)

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Electrode")
    ax.set_ylabel(column)

    if title is None:
        title = "Cascade electron balance"

    ax.set_title(title)
    fig.tight_layout()

    return fig, ax



def plot_meshes_3d(
    meshes: dict,
    frame_meshes: dict | None = None,
    max_faces_per_mesh: int = 3000,
    alpha: float = 0.25,
    title: str = "STL meshes",
    ax=None,
):
    """
    Plot STL meshes in 3D.

    Coordinates are shown in mm.

    Parameters
    ----------
    meshes:
        Dict name -> trimesh object.
    frame_meshes:
        Optional dict name -> trimesh object for grid frames.
    max_faces_per_mesh:
        Randomly downsample faces for faster plotting.
    alpha:
        Mesh transparency.
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    all_meshes = {}

    if meshes is not None:
        all_meshes.update(meshes)

    if frame_meshes is not None:
        all_meshes.update(frame_meshes)

    rng = np.random.default_rng(1)

    for name, mesh in all_meshes.items():
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        if len(faces) > max_faces_per_mesh:
            idx = rng.choice(len(faces), size=max_faces_per_mesh, replace=False)
            faces_plot = faces[idx]
        else:
            faces_plot = faces

        tri = 1e3 * verts[faces_plot]

        poly = Poly3DCollection(
            tri,
            alpha=alpha,
            linewidths=0.05,
        )

        poly.set_label(name)
        ax.add_collection3d(poly)

    # Autoscale from all mesh bounds.
    bounds_all = []

    for mesh in all_meshes.values():
        bounds_all.append(mesh.bounds)

    if bounds_all:
        bounds_all = np.asarray(bounds_all)
        xyz_min = bounds_all[:, 0, :].min(axis=0)
        xyz_max = bounds_all[:, 1, :].max(axis=0)

        ax.set_xlim(1e3 * xyz_min[0], 1e3 * xyz_max[0])
        ax.set_ylim(1e3 * xyz_min[1], 1e3 * xyz_max[1])
        ax.set_zlim(1e3 * xyz_min[2], 1e3 * xyz_max[2])

        _set_axes_equal_3d(ax)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)

    return fig, ax


def plot_fixed_voxels_3d(
    field: dict,
    owner_names: list[str] | None = None,
    max_points: int = 100_000,
    s: float = 1.0,
    alpha: float = 0.35,
    title: str = "Fixed-potential voxels",
    ax=None,
):
    """
    Plot fixed voxels from field['fixed'] and field['owner'].

    Coordinates are shown in mm.
    """
    if ax is None:
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    fixed = np.asarray(field["fixed"], dtype=bool)
    owner = np.asarray(field["owner"])

    owner_name_map = field.get("owner_name_map", {})

    if owner_names is None:
        owner_ids = sorted(np.unique(owner[fixed]))
        owner_ids = [int(o) for o in owner_ids if int(o) != 0]
    else:
        name_to_id = field.get("owner_id_map", None)

        if name_to_id is None:
            raise ValueError("field must contain owner_id_map when owner_names is used")

        owner_ids = [int(name_to_id[name]) for name in owner_names]

    rng = np.random.default_rng(1)

    for owner_id in owner_ids:
        mask = fixed & (owner == owner_id)

        idx = np.argwhere(mask)

        if len(idx) == 0:
            continue

        if len(idx) > max_points:
            keep = rng.choice(len(idx), size=max_points, replace=False)
            idx = idx[keep]

        x = field["x"][idx[:, 0]]
        y = field["y"][idx[:, 1]]
        z = field["z"][idx[:, 2]]

        owner_name = owner_name_map.get(owner_id, f"owner_{owner_id}")

        ax.scatter(
            1e3 * x,
            1e3 * y,
            1e3 * z,
            s=s,
            alpha=alpha,
            label=owner_name,
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    ax.legend(markerscale=5, fontsize=8)

    _set_axes_equal_3d(ax)

    return fig, ax


def plot_owner_slice(
    field: dict,
    plane: str = "xz",
    coord: float = 0.0,
    title: str | None = None,
    ax=None,
):
    """
    Plot owner IDs on a 2D slice.

    plane:
        'xy' means constant z = coord.
        'xz' means constant y = coord.
        'yz' means constant x = coord.

    coord is in meters.
    """
    if plane not in ["xy", "xz", "yz"]:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    owner = field["owner"]

    x = field["x"]
    y = field["y"]
    z = field["z"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if plane == "xy":
        k = int(np.argmin(np.abs(z - coord)))
        img = owner[:, :, k].T
        extent = [1e3 * x[0], 1e3 * x[-1], 1e3 * y[0], 1e3 * y[-1]]
        xlabel = "x (mm)"
        ylabel = "y (mm)"
        actual = z[k]

    elif plane == "xz":
        j = int(np.argmin(np.abs(y - coord)))
        img = owner[:, j, :].T
        extent = [1e3 * x[0], 1e3 * x[-1], 1e3 * z[0], 1e3 * z[-1]]
        xlabel = "x (mm)"
        ylabel = "z (mm)"
        actual = y[j]

    else:
        i = int(np.argmin(np.abs(x - coord)))
        img = owner[i, :, :].T
        extent = [1e3 * y[0], 1e3 * y[-1], 1e3 * z[0], 1e3 * z[-1]]
        xlabel = "y (mm)"
        ylabel = "z (mm)"
        actual = x[i]

    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("owner ID")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Owner slice {plane}, coord = {1e3 * actual:.3f} mm"

    ax.set_title(title)

    return fig, ax


def plot_potential_slice(
    field: dict,
    plane: str = "xz",
    coord: float = 0.0,
    title: str | None = None,
    ax=None,
):
    """
    Plot potential V on a 2D slice.
    """
    if plane not in ["xy", "xz", "yz"]:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    V = field["V"]

    x = field["x"]
    y = field["y"]
    z = field["z"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if plane == "xy":
        k = int(np.argmin(np.abs(z - coord)))
        img = V[:, :, k].T
        extent = [1e3 * x[0], 1e3 * x[-1], 1e3 * y[0], 1e3 * y[-1]]
        xlabel = "x (mm)"
        ylabel = "y (mm)"
        actual = z[k]

    elif plane == "xz":
        j = int(np.argmin(np.abs(y - coord)))
        img = V[:, j, :].T
        extent = [1e3 * x[0], 1e3 * x[-1], 1e3 * z[0], 1e3 * z[-1]]
        xlabel = "x (mm)"
        ylabel = "z (mm)"
        actual = y[j]

    else:
        i = int(np.argmin(np.abs(x - coord)))
        img = V[i, :, :].T
        extent = [1e3 * y[0], 1e3 * y[-1], 1e3 * z[0], 1e3 * z[-1]]
        xlabel = "y (mm)"
        ylabel = "z (mm)"
        actual = x[i]

    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Potential (V)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Potential slice {plane}, coord = {1e3 * actual:.3f} mm"

    ax.set_title(title)

    return fig, ax


def collect_tracked_trajectories(primary_results, cascade_results_all):
    tracked = {
        "primary": [],
        "cascade": [],
    }

    for res in primary_results:
        traj = res.get("traj", None)
        if traj is not None:
            tracked["primary"].append({
                "primary_index": res.get("primary_index", None),
                "traj": traj,
                "reason": res.get("reason", None),
            })

    for res in cascade_results_all:
        traj = res.get("traj", None)
        if traj is not None:
            tracked["cascade"].append({
                "primary_index": res.get("primary_index", None),
                "electron_id": res.get("electron_id", None),
                "parent_id": res.get("parent_id", None),
                "generation": res.get("generation", None),
                "terminal_electrode": res.get("terminal_electrode", None),
                "emission_kind": res.get("emission_kind", None),
                "traj": traj,
            })

    return tracked
