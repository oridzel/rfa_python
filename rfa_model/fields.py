"""
fields.py

Voxel field construction, Laplace solver, owner map handling,
and field/potential interpolation utilities for the RFA model.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .constants import e_charge, m_e


# ============================================================
# Owner IDs and names
# ============================================================

OWNER_ID = {
    "free": 0,

    "sample": 1,
    "holder": 2,
    "receiver": 3,
    "rod": 4,

    "g1frame": 5,
    "g2frame": 6,
    "g3frame": 7,

    "drifttube": 8,

    "g1_shell": 9,
    "g2_shell": 10,
    "g3_shell": 11,
    "collector_shell": 12,
}


OWNER_NAME = {v: k for k, v in OWNER_ID.items()}


def attach_default_owner_name_map(field: dict) -> dict:
    """
    Attach default owner-name map to a field dictionary.
    """
    field["owner_name_map"] = OWNER_NAME.copy()
    field["owner_id_map"] = OWNER_ID.copy()
    return field


def owner_id_from_name(name: str | None) -> int:
    """
    Convert owner name to integer owner ID.
    """
    if name is None:
        return OWNER_ID["free"]

    name = str(name)

    aliases = {
        "grid1": "g1_shell",
        "grid2": "g2_shell",
        "grid3": "g3_shell",
        "collector": "collector_shell",

        "g1_low_frame": "g1frame",
        "g1_upper_frame": "g1frame",
        "g2_low_frame": "g2frame",
        "g2_upper_frame": "g2frame",
        "g3_low_frame": "g3frame",
        "g3_upper_frame": "g3frame",
    }

    name = aliases.get(name, name)

    return OWNER_ID.get(name, OWNER_ID["free"])


def owner_name_from_id(owner_id: int) -> str:
    """
    Convert integer owner ID to owner name.
    """
    return OWNER_NAME.get(int(owner_id), f"owner_{int(owner_id)}")


# ============================================================
# Grid construction
# ============================================================

def make_cartesian_grid(
    xyz_min,
    xyz_max,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build regular Cartesian grid vectors.

    Parameters
    ----------
    xyz_min, xyz_max:
        Length-3 coordinate bounds in meters.
    h:
        Grid spacing in meters.

    Returns
    -------
    x, y, z:
        1D coordinate arrays.
    """
    xyz_min = np.asarray(xyz_min, dtype=float)
    xyz_max = np.asarray(xyz_max, dtype=float)

    x = np.arange(xyz_min[0], xyz_max[0] + 0.5 * h, h)
    y = np.arange(xyz_min[1], xyz_max[1] + 0.5 * h, h)
    z = np.arange(xyz_min[2], xyz_max[2] + 0.5 * h, h)

    return x, y, z


def make_empty_field_grid(
    xyz_min=(-0.083, -0.083, -0.083),
    xyz_max=(0.083, 0.083, 0.083),
    h: float = 0.5e-3,
) -> dict:
    """
    Create an empty field dictionary with regular grid vectors.
    """
    x, y, z = make_cartesian_grid(xyz_min, xyz_max, h)

    shape = (len(x), len(y), len(z))

    field = {
        "x": x,
        "y": y,
        "z": z,
        "h": float(h),

        "V": np.zeros(shape, dtype=float),
        "fixed": np.zeros(shape, dtype=bool),
        "owner": np.zeros(shape, dtype=np.int16),

        "update_region": np.ones(shape, dtype=bool),
    }

    attach_default_owner_name_map(field)

    return field


def meshgrid_coordinates(field: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return full 3D coordinate arrays X, Y, Z for a field dictionary.
    """
    return np.meshgrid(
        field["x"],
        field["y"],
        field["z"],
        indexing="ij",
    )


# ============================================================
# Geometry-to-voxel assignment
# ============================================================

def mark_spherical_shell(
    field: dict,
    radius: float,
    voltage: float,
    owner_name: str,
    thickness: float | None = None,
    center=(0.0, 0.0, 0.0),
    mask_openings: bool = True,
) -> dict:
    """
    Mark an analytic spherical shell as fixed potential.

    This is used for g1/g2/g3 analytic grid shells and collector shell.
    """
    from .collisions import sphere_crossing_is_opening

    if thickness is None:
        thickness = 1.25 * float(field["h"])

    X, Y, Z = meshgrid_coordinates(field)

    center = np.asarray(center, dtype=float)

    R = np.sqrt(
        (X - center[0])**2
        + (Y - center[1])**2
        + (Z - center[2])**2
    )

    mask = np.abs(R - radius) <= 0.5 * thickness

    if mask_openings:
        # Vectorized opening check is awkward because it uses several custom
        # opening rules. This loop is only over shell voxels, so it is okay.
        idx = np.argwhere(mask)

        keep = np.ones(len(idx), dtype=bool)

        for k, (i, j, l) in enumerate(idx):
            p = np.array([field["x"][i], field["y"][j], field["z"][l]])

            if sphere_crossing_is_opening(p, owner_name, field):
                keep[k] = False

        mask2 = np.zeros_like(mask, dtype=bool)

        if len(idx) > 0:
            idx_keep = idx[keep]
            mask2[idx_keep[:, 0], idx_keep[:, 1], idx_keep[:, 2]] = True

        mask = mask2

    owner_id = owner_id_from_name(owner_name)

    field["V"][mask] = voltage
    field["fixed"][mask] = True
    field["owner"][mask] = owner_id

    return field


def mark_box_region(
    field: dict,
    bounds,
    voltage: float,
    owner_name: str,
) -> dict:
    """
    Mark a rectangular box region as fixed potential.

    Useful for quick tests and simple electrodes.
    """
    bounds = np.asarray(bounds, dtype=float)

    X, Y, Z = meshgrid_coordinates(field)

    mask = (
        (X >= bounds[0, 0]) & (X <= bounds[1, 0])
        & (Y >= bounds[0, 1]) & (Y <= bounds[1, 1])
        & (Z >= bounds[0, 2]) & (Z <= bounds[1, 2])
    )

    owner_id = owner_id_from_name(owner_name)

    field["V"][mask] = voltage
    field["fixed"][mask] = True
    field["owner"][mask] = owner_id

    return field


def mark_mesh_voxels_by_contains(
    field: dict,
    mesh,
    voltage: float,
    owner_name: str,
    chunk_size: int = 250_000,
) -> dict:
    """
    Mark voxels whose centers are inside an STL mesh.

    This is slower than analytic surfaces, but useful for sample holder,
    receiver, rod, drift tube, and frame parts.

    Requires trimesh contains support, usually with rtree installed.
    """
    X, Y, Z = meshgrid_coordinates(field)

    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
    ])

    inside = np.zeros(len(points), dtype=bool)

    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        inside[start:stop] = mesh.contains(points[start:stop])

    mask = inside.reshape(field["V"].shape)

    owner_id = owner_id_from_name(owner_name)

    field["V"][mask] = voltage
    field["fixed"][mask] = True
    field["owner"][mask] = owner_id

    return field


def mark_mesh_bounds_shell(
    field: dict,
    mesh,
    voltage: float,
    owner_name: str,
    padding: float | None = None,
) -> dict:
    """
    Approximate marking of a mesh using its bounding box.

    This is not a substitute for accurate STL voxelization, but can be useful
    for fast field-debugging tests.
    """
    if padding is None:
        padding = float(field["h"])

    bounds = mesh.bounds.copy()
    bounds[0, :] -= padding
    bounds[1, :] += padding

    return mark_box_region(
        field=field,
        bounds=bounds,
        voltage=voltage,
        owner_name=owner_name,
    )


def mark_named_meshes(
    field: dict,
    meshes: dict,
    voltages: dict,
    name_to_voltage_key: dict | None = None,
    method: str = "contains",
) -> dict:
    """
    Mark multiple named STL meshes as fixed-potential voxels.

    Parameters
    ----------
    meshes:
        Dict name -> trimesh.
    voltages:
        Voltage dictionary, e.g. {"Vs":0, "Vdt":0, ...}.
    name_to_voltage_key:
        Optional mapping from mesh name to voltage key.
    method:
        "contains" or "bounds".
    """
    if name_to_voltage_key is None:
        name_to_voltage_key = {
            "sample": "Vs",
            "holder": "Vs",
            "receiver": "Vs",
            "rod": "Vs",

            "drifttube": "Vdt",

            "g1frame": "Vg1",
            "g2frame": "Vg2",
            "g3frame": "Vg3",

            "g1_low_frame": "Vg1",
            "g1_upper_frame": "Vg1",
            "g2_low_frame": "Vg2",
            "g2_upper_frame": "Vg2",
            "g3_low_frame": "Vg3",
            "g3_upper_frame": "Vg3",
        }

    for name, mesh in meshes.items():
        voltage_key = name_to_voltage_key.get(name, None)

        if voltage_key is None:
            continue

        voltage = float(voltages.get(voltage_key, 0.0))

        if method == "contains":
            mark_mesh_voxels_by_contains(
                field=field,
                mesh=mesh,
                voltage=voltage,
                owner_name=name,
            )
        elif method == "bounds":
            mark_mesh_bounds_shell(
                field=field,
                mesh=mesh,
                voltage=voltage,
                owner_name=name,
            )
        else:
            raise ValueError("method must be 'contains' or 'bounds'")

    return field


def mark_analytic_rfa_surfaces(
    field: dict,
    voltages: dict,
    R_g1: float,
    R_g2: float,
    R_g3: float,
    R_col: float,
    shell_thickness: float | None = None,
) -> dict:
    """
    Mark analytic g1/g2/g3 grid shells and collector shell.
    """
    field["R_g1"] = float(R_g1)
    field["R_g2"] = float(R_g2)
    field["R_g3"] = float(R_g3)
    field["R_col"] = float(R_col)

    mark_spherical_shell(
        field,
        radius=R_g1,
        voltage=float(voltages.get("Vg1", 0.0)),
        owner_name="g1_shell",
        thickness=shell_thickness,
    )

    mark_spherical_shell(
        field,
        radius=R_g2,
        voltage=float(voltages.get("Vg2", 0.0)),
        owner_name="g2_shell",
        thickness=shell_thickness,
    )

    mark_spherical_shell(
        field,
        radius=R_g3,
        voltage=float(voltages.get("Vg3", 0.0)),
        owner_name="g3_shell",
        thickness=shell_thickness,
    )

    mark_spherical_shell(
        field,
        radius=R_col,
        voltage=float(voltages.get("Vc", 0.0)),
        owner_name="collector_shell",
        thickness=shell_thickness,
    )

    return field


# ============================================================
# Update-region handling
# ============================================================

def set_update_region_spherical(
    field: dict,
    r_max: float | None = None,
    r_min: float = 0.0,
    center=(0.0, 0.0, 0.0),
) -> dict:
    """
    Restrict Laplace updates to a spherical region.
    """
    X, Y, Z = meshgrid_coordinates(field)
    center = np.asarray(center, dtype=float)

    R = np.sqrt(
        (X - center[0])**2
        + (Y - center[1])**2
        + (Z - center[2])**2
    )

    mask = R >= r_min

    if r_max is not None:
        mask &= R <= r_max

    field["update_region"] = mask

    return field


def set_outer_boundary_fixed(
    field: dict,
    voltage: float = 0.0,
    owner_name: str = "drifttube",
) -> dict:
    """
    Fix the six outer faces of the grid domain.
    """
    owner_id = owner_id_from_name(owner_name)

    V = field["V"]
    fixed = field["fixed"]
    owner = field["owner"]

    slices = [
        (0, slice(None), slice(None)),
        (-1, slice(None), slice(None)),
        (slice(None), 0, slice(None)),
        (slice(None), -1, slice(None)),
        (slice(None), slice(None), 0),
        (slice(None), slice(None), -1),
    ]

    for slc in slices:
        V[slc] = voltage
        fixed[slc] = True
        owner[slc] = owner_id

    return field


# ============================================================
# Laplace solver
# ============================================================

def initialize_potential_linear_x(
    field: dict,
    V_left: float | None = None,
    V_right: float | None = None,
) -> dict:
    """
    Initialize potential with a linear x-gradient between the two x-boundaries.

    Fixed voxels are restored afterwards.
    """
    x = field["x"]
    V = field["V"]

    fixed = field["fixed"]
    V_fixed = V.copy()

    if V_left is None:
        V_left = float(np.nanmean(V[0, :, :]))

    if V_right is None:
        V_right = float(np.nanmean(V[-1, :, :]))

    alpha = (x - x[0]) / (x[-1] - x[0])
    line = (1.0 - alpha) * V_left + alpha * V_right

    V[:, :, :] = line[:, None, None]

    V[fixed] = V_fixed[fixed]

    return field


def solve_laplace_jacobi(
    field: dict,
    max_iter: int = 20_000,
    tol: float = 1e-5,
    omega: float = 1.0,
    check_every: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Solve Laplace equation on a regular grid using weighted Jacobi iteration.

    This solves ∇²V = 0 on non-fixed voxels inside update_region.

    Parameters
    ----------
    field:
        Field dictionary with V, fixed, update_region.
    max_iter:
        Maximum iterations.
    tol:
        Stop when maximum update is below this value, in volts.
    omega:
        Weighted Jacobi relaxation parameter. omega=1 is ordinary Jacobi.
        For Jacobi, values <=1 are safest.
    check_every:
        Print/check convergence every this many iterations.
    verbose:
        Print progress.

    Returns
    -------
    field:
        Updated field dictionary. Adds solver metadata.
    """
    V = np.asarray(field["V"], dtype=float)
    fixed = np.asarray(field["fixed"], dtype=bool)
    update_region = np.asarray(field.get("update_region", np.ones_like(fixed)), dtype=bool)

    update = update_region & (~fixed)

    Vnew = V.copy()

    t0 = time.perf_counter()
    last_delta = np.inf

    if verbose:
        print("Solving Laplace equation with weighted Jacobi")
        print(f"grid shape: {V.shape}")
        print(f"update voxels: {int(update.sum())}")
        print(f"fixed voxels:  {int(fixed.sum())}")

    for it in range(1, max_iter + 1):
        avg = (
            V[:-2, 1:-1, 1:-1]
            + V[2:, 1:-1, 1:-1]
            + V[1:-1, :-2, 1:-1]
            + V[1:-1, 2:, 1:-1]
            + V[1:-1, 1:-1, :-2]
            + V[1:-1, 1:-1, 2:]
        ) / 6.0

        inner_update = update[1:-1, 1:-1, 1:-1]

        if omega == 1.0:
            Vnew[1:-1, 1:-1, 1:-1][inner_update] = avg[inner_update]
        else:
            old = V[1:-1, 1:-1, 1:-1]
            Vnew_inner = Vnew[1:-1, 1:-1, 1:-1]
            Vnew_inner[inner_update] = (
                (1.0 - omega) * old[inner_update]
                + omega * avg[inner_update]
            )

        if it % check_every == 0 or it == 1:
            diff = np.abs(Vnew - V)
            last_delta = float(diff[update].max()) if update.any() else 0.0

            if verbose:
                elapsed = time.perf_counter() - t0
                print(
                    f"iter {it:7d}: max update = {last_delta:.6e} V "
                    f"elapsed = {elapsed:.1f} s"
                )

            if last_delta < tol:
                V[:, :, :] = Vnew
                break

        V, Vnew = Vnew, V

    field["V"] = V
    field["solver"] = {
        "method": "weighted_jacobi",
        "iterations": it,
        "tol": tol,
        "last_delta": last_delta,
        "omega": omega,
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(last_delta < tol),
    }

    if verbose:
        print("Laplace solver finished")
        print(field["solver"])

    return field


def solve_laplace_red_black_sor(
    field: dict,
    max_iter: int = 20_000,
    tol: float = 1e-5,
    omega: float = 1.85,
    check_every: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Solve Laplace equation using red-black SOR.

    This is usually much faster than Jacobi.

    Notes
    -----
    For very large grids, omega around 1.7-1.95 is typical. If the solver
    becomes unstable, reduce omega.
    """
    V = field["V"]
    fixed = np.asarray(field["fixed"], dtype=bool)
    update_region = np.asarray(field.get("update_region", np.ones_like(fixed)), dtype=bool)

    update = update_region & (~fixed)

    nx, ny, nz = V.shape

    I, J, K = np.indices((nx, ny, nz))
    red = ((I + J + K) % 2 == 0) & update
    black = (~red) & update

    # Do not update outermost boundary.
    boundary = np.zeros_like(update, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    red &= ~boundary
    black &= ~boundary

    red_inner = red[1:-1, 1:-1, 1:-1]
    black_inner = black[1:-1, 1:-1, 1:-1]

    t0 = time.perf_counter()
    last_delta = np.inf

    if verbose:
        print("Solving Laplace equation with red-black SOR")
        print(f"grid shape: {V.shape}")
        print(f"update voxels: {int(update.sum())}")
        print(f"fixed voxels:  {int(fixed.sum())}")
        print(f"omega:         {omega}")

    for it in range(1, max_iter + 1):
        max_delta_iter = 0.0

        for mask_inner in [red_inner, black_inner]:
            center = V[1:-1, 1:-1, 1:-1]

            avg = (
                V[:-2, 1:-1, 1:-1]
                + V[2:, 1:-1, 1:-1]
                + V[1:-1, :-2, 1:-1]
                + V[1:-1, 2:, 1:-1]
                + V[1:-1, 1:-1, :-2]
                + V[1:-1, 1:-1, 2:]
            ) / 6.0

            delta = omega * (avg[mask_inner] - center[mask_inner])

            center[mask_inner] += delta

            if delta.size > 0:
                max_delta_iter = max(
                    max_delta_iter,
                    float(np.max(np.abs(delta))),
                )

        if it % check_every == 0 or it == 1:
            last_delta = max_delta_iter

            if verbose:
                elapsed = time.perf_counter() - t0
                print(
                    f"iter {it:7d}: max update = {last_delta:.6e} V "
                    f"elapsed = {elapsed:.1f} s"
                )

            if last_delta < tol:
                break

    field["solver"] = {
        "method": "red_black_sor",
        "iterations": it,
        "tol": tol,
        "last_delta": last_delta,
        "omega": omega,
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(last_delta < tol),
    }

    if verbose:
        print("Laplace solver finished")
        print(field["solver"])

    return field


# ============================================================
# Electric field calculation
# ============================================================

def calculate_electric_field(field: dict) -> dict:
    """
    Calculate electric field from potential.

    E = -grad(V)

    Assumes uniform grid spacing h in all directions.
    """
    h = float(field["h"])
    V = field["V"]

    dVdx, dVdy, dVdz = np.gradient(V, h, h, h, edge_order=2)

    field["Ex"] = -dVdx
    field["Ey"] = -dVdy
    field["Ez"] = -dVdz

    return field


# Backwards-compatible alias.
compute_electric_field = calculate_electric_field


# ============================================================
# Interpolators and field evaluation
# ============================================================

def build_field_interpolators(field: dict):
    """
    Build RegularGridInterpolator objects for Ex, Ey, Ez.
    """
    points = (field["x"], field["y"], field["z"])

    Ex_interp = RegularGridInterpolator(
        points,
        field["Ex"],
        bounds_error=False,
        fill_value=np.nan,
    )

    Ey_interp = RegularGridInterpolator(
        points,
        field["Ey"],
        bounds_error=False,
        fill_value=np.nan,
    )

    Ez_interp = RegularGridInterpolator(
        points,
        field["Ez"],
        bounds_error=False,
        fill_value=np.nan,
    )

    return Ex_interp, Ey_interp, Ez_interp


def build_potential_interpolator(field: dict):
    """
    Build RegularGridInterpolator for potential.
    """
    return RegularGridInterpolator(
        (field["x"], field["y"], field["z"]),
        field["V"],
        bounds_error=False,
        fill_value=np.nan,
    )


def evaluate_field(p, Ex_interp, Ey_interp, Ez_interp):
    """
    Evaluate electric field vector at one point.
    """
    p = np.asarray(p, dtype=float).reshape(1, 3)

    E = np.array([
        float(Ex_interp(p)[0]),
        float(Ey_interp(p)[0]),
        float(Ez_interp(p)[0]),
    ])

    return E


def evaluate_potential(p, Phi_interp):
    """
    Evaluate potential at one point.
    """
    p = np.asarray(p, dtype=float).reshape(1, 3)
    return float(Phi_interp(p)[0])


def classify_grid_point(p, field: dict) -> dict:
    """
    Classify a point relative to the field grid and fixed voxels.

    Returns
    -------
    dict with keys:
        status:
            "free", "hit_fixed", or "left_grid"
        owner_id:
            integer owner ID if fixed
        owner_name:
            owner name if fixed
        index:
            nearest grid index
    """
    p = np.asarray(p, dtype=float)

    x = field["x"]
    y = field["y"]
    z = field["z"]
    h = float(field["h"])

    i = int(np.round((p[0] - x[0]) / h))
    j = int(np.round((p[1] - y[0]) / h))
    k = int(np.round((p[2] - z[0]) / h))

    if (
        i < 0 or i >= len(x)
        or j < 0 or j >= len(y)
        or k < 0 or k >= len(z)
    ):
        return {
            "status": "left_grid",
            "index": (i, j, k),
            "owner_id": None,
            "owner_name": None,
        }

    update_region = field.get("update_region", None)

    if update_region is not None and not bool(update_region[i, j, k]):
        return {
            "status": "left_update_region",
            "index": (i, j, k),
            "owner_id": None,
            "owner_name": None,
        }

    if bool(field["fixed"][i, j, k]):
        owner_id = int(field["owner"][i, j, k])

        owner_name_map = field.get("owner_name_map", OWNER_NAME)
        owner_name = owner_name_map.get(owner_id, f"owner_{owner_id}")

        return {
            "status": "hit_fixed",
            "index": (i, j, k),
            "owner_id": owner_id,
            "owner_name": owner_name,
        }

    return {
        "status": "free",
        "index": (i, j, k),
        "owner_id": 0,
        "owner_name": "free",
    }


# ============================================================
# Complete field build helper
# ============================================================

def build_rfa_field(
    xyz_min=(-0.083, -0.083, -0.083),
    xyz_max=(0.083, 0.083, 0.083),
    h: float = 0.5e-3,
    voltages: dict | None = None,
    meshes: dict | None = None,
    frame_meshes: dict | None = None,
    R_g1: float = 0.0451904,
    R_g2: float = 0.0579265,
    R_g3: float = 0.0710762,
    R_col: float = 0.08255,
    mesh_method: str = "contains",
    outer_boundary_voltage: float | None = None,
    solver: str = "sor",
    max_iter: int = 20_000,
    tol: float = 1e-5,
    omega: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Build and solve a complete RFA electrostatic field.

    This creates a voxel field, marks fixed electrodes, solves Laplace's
    equation, and calculates Ex/Ey/Ez.

    Parameters
    ----------
    meshes:
        Sample/holder/receiver/rod/drifttube meshes, dict name -> mesh.
    frame_meshes:
        Grid-frame meshes, dict name -> mesh.
    voltages:
        Dict containing Vs, Vg1, Vg2, Vg3, Vc, Vdt.

    Returns
    -------
    field:
        Complete field dictionary.
    """
    if voltages is None:
        voltages = {
            "Vs": 0.0,
            "Vg1": 0.0,
            "Vg2": 0.0,
            "Vg3": 0.0,
            "Vc": 50.0,
            "Vdt": 0.0,
        }

    field = make_empty_field_grid(
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        h=h,
    )

    field["voltages"] = dict(voltages)

    if outer_boundary_voltage is None:
        outer_boundary_voltage = float(voltages.get("Vdt", 0.0))

    set_outer_boundary_fixed(
        field,
        voltage=outer_boundary_voltage,
        owner_name="drifttube",
    )

    if meshes is not None:
        mark_named_meshes(
            field,
            meshes=meshes,
            voltages=voltages,
            method=mesh_method,
        )

    if frame_meshes is not None:
        mark_named_meshes(
            field,
            meshes=frame_meshes,
            voltages=voltages,
            method=mesh_method,
        )

    mark_analytic_rfa_surfaces(
        field,
        voltages=voltages,
        R_g1=R_g1,
        R_g2=R_g2,
        R_g3=R_g3,
        R_col=R_col,
    )

    initialize_potential_linear_x(
        field,
        V_left=outer_boundary_voltage,
        V_right=float(voltages.get("Vc", outer_boundary_voltage)),
    )

    if solver == "sor":
        if omega is None:
            omega = 1.85

        solve_laplace_red_black_sor(
            field,
            max_iter=max_iter,
            tol=tol,
            omega=omega,
            verbose=verbose,
        )
    elif solver == "jacobi":
        if omega is None:
            omega = 1.0

        solve_laplace_jacobi(
            field,
            max_iter=max_iter,
            tol=tol,
            omega=omega,
            verbose=verbose,
        )
    else:
        raise ValueError("solver must be 'sor' or 'jacobi'")

    calculate_electric_field(field)
    attach_default_owner_name_map(field)

    return field


def E_at_point(p, Ex_interp, Ey_interp, Ez_interp):
    """
    Backwards-compatible alias for evaluate_field().
    """
    return evaluate_field(p, Ex_interp, Ey_interp, Ez_interp)


def potential_at_point(p, Phi_interp):
    """
    Backwards-compatible alias for evaluate_potential().
    """
    return evaluate_potential(p, Phi_interp)


def evaluate_field(p, Ex_interp, Ey_interp, Ez_interp):
    """
    Evaluate electric field vector at one point.
    """
    p = np.asarray(p, dtype=float).reshape(1, 3)

    E = np.array([
        float(Ex_interp(p)[0]),
        float(Ey_interp(p)[0]),
        float(Ez_interp(p)[0]),
    ])

    return E


def evaluate_potential(p, Phi_interp):
    """
    Evaluate potential at one point.
    """
    p = np.asarray(p, dtype=float).reshape(1, 3)
    return float(Phi_interp(p)[0])


def E_at_point(p, Ex_interp, Ey_interp, Ez_interp):
    """
    Backwards-compatible alias for evaluate_field().
    """
    return evaluate_field(p, Ex_interp, Ey_interp, Ez_interp)


def potential_at_point(p, Phi_interp):
    """
    Backwards-compatible alias for evaluate_potential().
    """
    return evaluate_potential(p, Phi_interp)