"""
fields.py

Voxel field construction, Laplace solver, owner map handling,
and field/potential interpolation utilities for the RFA model.
mesh_method:
    "voxelized", "contains", or "bounds".
"""

from __future__ import annotations

import time
import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator


from .constants import e_charge, m_e, COLLECTOR_OPENING_ALPHA_DEG

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
    "outer_boundary": 13,
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


def attach_rfa_metadata(
    field: dict,
    voltages: dict,
    R_g1: float,
    R_g2: float,
    R_g3: float,
    R_col: float,
) -> dict:
    """
    Attach RFA geometry and voltage metadata required by trajectory,
    collision, and cascade code.
    """
    field["R_g1"] = float(R_g1)
    field["R_g2"] = float(R_g2)
    field["R_g3"] = float(R_g3)
    field["R_col"] = float(R_col)
    field["voltages"] = dict(voltages)

    # Physical opening through the supporting frames around the +X drift-tube
    # axis.  This is 5.6 mm; it is not the tube's inner bore.  Keep the older
    # metadata key as a compatibility alias for existing notebooks.
    field.setdefault("grid_frame_opening_radius", 0.0056)
    field.setdefault(
        "drifttube_aperture_radius",
        field["grid_frame_opening_radius"],
    )

    attach_default_owner_name_map(field)

    return field


def validate_field_metadata(field: dict) -> None:
    """
    Raise an error if required RFA field metadata are missing.
    """
    required = [
        "x", "y", "z", "h",
        "V", "fixed", "owner", "update_region",
        "R_g1", "R_g2", "R_g3", "R_col",
        "voltages",
        "owner_name_map", "owner_id_map",
    ]

    missing = [key for key in required if key not in field]

    if missing:
        raise KeyError(f"Field is missing required keys: {missing}")

    # Ex/Ey/Ez are present only after calculate_electric_field().
    if all(key in field for key in ["Ex", "Ey", "Ez"]):
        return


def _log(verbose: bool, message: str):
    """
    Print a progress message when verbose=True.
    """
    if verbose:
        print(message, flush=True)


def _count_updatable_voxels(
    update_region: np.ndarray,
    fixed: np.ndarray,
    chunk_size: int = 8_000_000,
) -> int:
    """Count active non-fixed cells without a full-grid temporary mask."""
    update_flat = np.ravel(update_region)
    fixed_flat = np.ravel(fixed)
    total = 0
    for start in range(0, update_flat.size, chunk_size):
        stop = min(start + chunk_size, update_flat.size)
        active = np.logical_not(fixed_flat[start:stop])
        np.logical_and(update_flat[start:stop], active, out=active)
        total += int(np.count_nonzero(active))
    return total


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


def mark_mesh_voxels_by_voxelized(
    field: dict,
    mesh,
    voltage: float,
    owner_name: str,
    pitch: float | None = None,
    verbose: bool = False,
) -> dict:
    """
    Mark fixed-potential voxels using ``trimesh.Trimesh.voxelized``.

    Rod and drift-tube meshes are deliberately redirected to the bounding-box
    limited ``mesh.contains`` path.  Trimesh's voxelizer named ``"contains"``
    is a ray-based *surface* voxelizer and produced direction-dependent gaps
    in these long curved parts; it is not the same operation as calling
    ``mesh.contains(points)`` to classify field-grid cell centers.
    """
    if pitch is None:
        pitch = float(field["h"])

    if owner_name in {"rod", "drifttube"}:
        _log(
            verbose,
            f"  Redirecting {owner_name!r} to solid mesh.contains "
            "voxelization ...",
        )
        return mark_mesh_voxels_by_contains(
            field=field,
            mesh=mesh,
            voltage=voltage,
            owner_name=owner_name,
            verbose=verbose,
        )

    _log(
        verbose,
        f"  Voxelizing {owner_name!r} with pitch = {pitch:.3e} m "
        "(method=subdivide) ...",
    )

    t0 = time.perf_counter()

    vox = mesh.voxelized(pitch=pitch)
    points = np.asarray(vox.points, dtype=float)

    _log(
        verbose,
        f"    voxelized points: {len(points):,} "
        f"elapsed = {time.perf_counter() - t0:.2f} s"
    )

    if points.size == 0:
        _log(verbose, f"    WARNING: no voxel points for {owner_name!r}")
        return field

    x = field["x"]
    y = field["y"]
    z = field["z"]
    h = float(field["h"])

    i = np.round((points[:, 0] - x[0]) / h).astype(int)
    j = np.round((points[:, 1] - y[0]) / h).astype(int)
    k = np.round((points[:, 2] - z[0]) / h).astype(int)

    valid = (
        (i >= 0) & (i < len(x))
        & (j >= 0) & (j < len(y))
        & (k >= 0) & (k < len(z))
    )

    n_valid_points = int(valid.sum())
    n_outside = int(len(valid) - n_valid_points)

    i = i[valid]
    j = j[valid]
    k = k[valid]

    # Several voxelized points may round to the same field cell. Deduplicate
    # the integer indices so metadata reports actual cells, not raw rows.
    flat = np.ravel_multi_index((i, j, k), field["V"].shape)
    flat = np.unique(flat)
    i, j, k = np.unravel_index(flat, field["V"].shape)
    n_unique = int(flat.size)

    field.setdefault("mesh_voxel_counts", {})[owner_name] = n_unique
    field.setdefault("mesh_voxel_point_counts", {})[owner_name] = n_valid_points
    field.setdefault("mesh_voxel_methods", {})[owner_name] = "subdivide"

    owner_id = owner_id_from_name(owner_name)

    field["V"][i, j, k] = voltage
    field["fixed"][i, j, k] = True
    field["owner"][i, j, k] = owner_id

    _log(
        verbose,
        f"    assigned unique fixed voxels: {n_unique:,}; "
        f"in-bounds voxel points: {n_valid_points:,}; "
        f"outside grid: {n_outside:,}; "
        f"V = {voltage:g} V; owner_id = {owner_id}"
    )

    return field


def mark_mesh_voxels_by_contains(
    field: dict,
    mesh,
    voltage: float,
    owner_name: str,
    chunk_size: int = 250_000,
    padding: float | None = None,
    require_watertight: bool = True,
    validate_connectivity: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Mark field-grid cell centers inside a watertight STL mesh.

    Only points within the mesh bounding box (plus a small padding) are
    generated, and they are classified in chunks.  This avoids the three full
    coordinate arrays previously created for the entire field domain.  The
    local boolean mask is completed before the field is mutated, so a failed
    contains query cannot leave a partly assigned conductor.

    This is the production path for the rod and drift-tube STLs.  It calls
    ``mesh.contains(points)`` directly; it does *not* call the ray-based
    ``mesh.voxelized(method="contains")`` surface voxelizer.

    Parameters
    ----------
    chunk_size:
        Maximum number of candidate cell centers in one contains query.
    padding:
        Bounding-box expansion used to select candidate grid centers. The
        default is half a field voxel. It does not enlarge the geometry;
        ``mesh.contains`` still decides which centers are inside.
    require_watertight:
        Refuse classification when Trimesh reports a non-watertight mesh.
        Point containment is not reliable for an open surface.
    validate_connectivity:
        Record 26-neighbour connected-component diagnostics for rod and drift
        tube masks. Fragmentation is reported as a warning rather than raised,
        because legitimate ownership overlaps are applied later.

    Notes
    -----
    Requires Trimesh containment support, normally provided by ``rtree``.
    """
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be at least 1")
    chunk_size = int(chunk_size)

    h = float(field["h"])
    if padding is None:
        padding = 0.5 * h
    padding = float(padding)
    if padding < 0.0:
        raise ValueError("padding must be non-negative")

    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            f"Mesh {owner_name!r} has invalid bounds: {bounds!r}"
        )

    is_watertight = bool(getattr(mesh, "is_watertight", False))
    field.setdefault("mesh_watertight", {})[owner_name] = is_watertight

    if require_watertight and not is_watertight:
        raise ValueError(
            f"Mesh {owner_name!r} is not watertight; mesh.contains cannot "
            "reliably define its metal volume. Repair or reload the STL "
            "with processing enabled before rebuilding the field."
        )

    x = np.asarray(field["x"])
    y = np.asarray(field["y"])
    z = np.asarray(field["z"])

    axes = (x, y, z)
    local_indices = []
    for axis_number, axis in enumerate(axes):
        lo = bounds[0, axis_number] - padding
        hi = bounds[1, axis_number] + padding
        local_indices.append(np.flatnonzero((axis >= lo) & (axis <= hi)))

    ix, iy, iz = local_indices

    if len(ix) == 0 or len(iy) == 0 or len(iz) == 0:
        _log(
            verbose,
            f"    WARNING: {owner_name!r} bounding box does not intersect "
            "the field grid",
        )
        field.setdefault("mesh_voxel_methods", {})[owner_name] = "contains"
        field.setdefault("mesh_voxel_counts", {})[owner_name] = 0
        field.setdefault("mesh_voxel_point_counts", {})[owner_name] = 0
        field.setdefault("mesh_contains_candidate_counts", {})[owner_name] = 0
        return field

    local_shape = (len(ix), len(iy), len(iz))
    n_candidates = int(np.prod(local_shape, dtype=np.int64))
    local_mask = np.zeros(local_shape, dtype=bool)

    _log(
        verbose,
        f"  Voxelizing {owner_name!r} by solid mesh.contains: "
        f"local shape={local_shape}, candidates={n_candidates:,}, "
        f"chunk_size={chunk_size:,} ...",
    )

    t0 = time.perf_counter()
    nyz = len(iy) * len(iz)
    nz = len(iz)

    for start in range(0, n_candidates, chunk_size):
        stop = min(start + chunk_size, n_candidates)
        flat_local = np.arange(start, stop, dtype=np.int64)

        local_i = flat_local // nyz
        remainder = flat_local - local_i * nyz
        local_j = remainder // nz
        local_k = remainder - local_j * nz

        points = np.column_stack((
            x[ix[local_i]],
            y[iy[local_j]],
            z[iz[local_k]],
        ))

        try:
            inside = np.asarray(mesh.contains(points), dtype=bool).reshape(-1)
        except ModuleNotFoundError as exc:
            if exc.name == "rtree":
                raise RuntimeError(
                    f"Solid mesh.contains voxelization for {owner_name!r} "
                    "requires the 'rtree' package. Install it in the notebook "
                    "environment with `pip install rtree`, restart the "
                    "kernel, and rebuild the field from scratch."
                ) from exc
            raise

        if inside.size != stop - start:
            raise RuntimeError(
                f"mesh.contains returned {inside.size} classifications for "
                f"{stop - start} points of {owner_name!r}"
            )

        if np.any(inside):
            local_mask[
                local_i[inside],
                local_j[inside],
                local_k[inside],
            ] = True

    n_inside = int(np.count_nonzero(local_mask))

    field.setdefault("mesh_voxel_methods", {})[owner_name] = "contains"
    field.setdefault("mesh_voxel_counts", {})[owner_name] = n_inside
    field.setdefault("mesh_voxel_point_counts", {})[owner_name] = n_inside
    field.setdefault("mesh_contains_candidate_counts", {})[
        owner_name
    ] = n_candidates

    if n_inside == 0:
        raise RuntimeError(
            f"Solid mesh.contains voxelization produced no field voxels for "
            f"{owner_name!r}. The conductor may be thinner than the field "
            "spacing or its mesh may not enclose the expected volume."
        )

    local_i, local_j, local_k = np.nonzero(local_mask)
    global_i = ix[local_i]
    global_j = iy[local_j]
    global_k = iz[local_k]

    owner_id = owner_id_from_name(owner_name)

    field["V"][global_i, global_j, global_k] = voltage
    field["fixed"][global_i, global_j, global_k] = True
    field["owner"][global_i, global_j, global_k] = owner_id

    _log(
        verbose,
        f"    assigned solid fixed voxels: {n_inside:,}; "
        f"elapsed={time.perf_counter() - t0:.2f} s; "
        f"V={voltage:g} V; owner_id={owner_id}; "
        f"watertight={is_watertight}",
    )

    if validate_connectivity and owner_name in {"rod", "drifttube"}:
        from scipy.ndimage import generate_binary_structure, label

        structure = generate_binary_structure(3, 3)
        labels, n_components = label(local_mask, structure=structure)
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(sizes.max()) if sizes.size else 0
        largest_fraction = largest / n_inside if n_inside else 0.0

        field.setdefault("mesh_connectivity", {})[owner_name] = {
            "components_26_neighbor": int(n_components),
            "largest_component_voxels": largest,
            "largest_component_fraction": float(largest_fraction),
        }

        _log(
            verbose,
            f"    connectivity: components={n_components:,}, "
            f"largest_fraction={largest_fraction:.6f}",
        )

        if n_components != 1:
            _log(
                verbose,
                f"    WARNING: {owner_name!r} contains mask has "
                f"{n_components} connected components before ownership "
                "overlaps are applied",
            )

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
    method: str = "voxelized",
    verbose: bool = False,
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
        ``"voxelized"`` uses Trimesh subdivision for ordinary STL parts and
        automatically uses solid, bounding-box-limited ``mesh.contains`` for
        rod and drift tube. ``"contains"`` forces solid containment for every
        named mesh. ``"bounds"`` is only a debugging approximation.
    """
    if name_to_voltage_key is None:
        name_to_voltage_key = {
            "sample": "Vs",
            "holder": "Vs",
            "receiver": "Vs",
            "rod": "Vrod",

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

        if method == "voxelized":
            mark_mesh_voxels_by_voxelized(
                field=field,
                mesh=mesh,
                voltage=voltage,
                owner_name=name,
                pitch=float(field["h"]),
                verbose=verbose,
            )
                
        elif method == "contains":
            _log(verbose, f"  Voxelizing {name!r} by mesh.contains ...")
        
            mark_mesh_voxels_by_contains(
                field=field,
                mesh=mesh,
                voltage=voltage,
                owner_name=name,
                verbose=verbose,
            )
        
        elif method == "bounds":
            mark_mesh_bounds_shell(
                field=field,
                mesh=mesh,
                voltage=voltage,
                owner_name=name,
            )
        
        else:
            raise ValueError("method must be 'voxelized', 'contains', or 'bounds'")

    return field


def mark_analytic_rfa_surfaces(
    field: dict,
    voltages: dict,
    R_g1: float,
    R_g2: float,
    R_g3: float,
    R_col: float,
    shell_thickness: float | None = None,
    include_analytic_drifttube: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Mark analytic g1/g2/g3 grid shells, collector shell,
    and optionally the legacy analytic drift-tube boundary.

    This version stores the masks in the field dictionary.

    Set ``include_analytic_drifttube=False`` when the aligned physical
    ``meshes["drifttube"]`` STL has already been voxelized.  This prevents the
    legacy cylindrical approximation from duplicating or overwriting the STL.
    """
    attach_rfa_metadata(
        field=field,
        voltages=voltages,
        R_g1=R_g1,
        R_g2=R_g2,
        R_g3=R_g3,
        R_col=R_col,
    )

    _log(verbose, "Creating analytic RFA boundary masks ...")

    make_analytic_rfa_boundary_masks(
        field=field,
        R_g1=R_g1,
        R_g2=R_g2,
        R_g3=R_g3,
        R_col=R_col,
        include_drifttube=include_analytic_drifttube,
    )

    _log(verbose, "Assigning analytic fixed-potential boundaries ...")

    n_before = int(np.count_nonzero(field["fixed"]))

    if include_analytic_drifttube:
        _set_fixed_mask(
            field,
            field["drift_bc"],
            voltage=float(voltages.get("Vdt", 0.0)),
            owner_name="drifttube",
        )

    _set_fixed_mask(
        field,
        field["g1_bdry"],
        voltage=float(voltages.get("Vg1", 0.0)),
        owner_name="g1_shell",
    )

    _set_fixed_mask(
        field,
        field["g2_bdry"],
        voltage=float(voltages.get("Vg2", 0.0)),
        owner_name="g2_shell",
    )

    _set_fixed_mask(
        field,
        field["g3_bdry"],
        voltage=float(voltages.get("Vg3", 0.0)),
        owner_name="g3_shell",
    )

    _set_fixed_mask(
        field,
        field["col_bdry"],
        voltage=float(voltages.get("Vc", 0.0)),
        owner_name="collector_shell",
    )

    n_after = int(np.count_nonzero(field["fixed"]))

    _log(verbose, f"  analytic fixed voxels added: {n_after - n_before:,}")
    if verbose:
        _log(
            True,
            "  update voxels: "
            f"{_count_updatable_voxels(field['update_region'], field['fixed']):,}",
        )

    return field


# ============================================================
# Update-region handling
# ============================================================

def _set_fixed_mask(
    field: dict,
    mask: np.ndarray,
    voltage: float,
    owner_name: str,
) -> dict:
    """
    Assign a fixed-potential mask to the field.
    """
    owner_id = owner_id_from_name(owner_name)

    field["fixed"][mask] = True
    field["V"][mask] = float(voltage)
    field["owner"][mask] = owner_id

    return field


def make_analytic_rfa_boundary_masks(
    field: dict,
    R_g1: float,
    R_g2: float,
    R_g3: float,
    R_col: float,
    r_hole: float = 0.0056,
    r_rod: float = 0.011,
    r_dt_i: float = 4.3e-3,
    t_dt: float = 0.25e-3,
    include_drifttube: bool = True,
) -> dict:
    """
    Create and store analytic RFA boundary masks.

    Stores:
        field["g1_bdry"]
        field["g2_bdry"]
        field["g3_bdry"]
        field["col_bdry"]
        field["drift_bc"]
        field["inside"]
        field["update_region"]

    This follows the old notebook convention.
    """
    x = field["x"]
    y = field["y"]
    z = field["z"]
    h = float(field["h"])

    Nx, Ny, Nz = field["V"].shape
    full_shape = (Nx, Ny, Nz)

    X = x[:, None, None]
    Y = y[None, :, None]
    Z = z[None, None, :]

    rho_yz = np.sqrt(Y**2 + Z**2)
    rho_xy = np.sqrt(X**2 + Y**2)
    R = np.sqrt(X**2 + Y**2 + Z**2)

    eps_hit = 0.5 * h
    band_grid = 2.0 * eps_hit      # = h
    band_col = 2.0 * h
    pad = 2.0 * h

    # This is used only by the legacy analytic-cylinder fallback.  The normal
    # build path uses the physical STL whose nose is aligned at x=0.047 m.
    x_dt_near = 0.047
    x_dt_far = R_col + pad

    # --------------------------------------------------------
    # Analytic spherical shells
    # --------------------------------------------------------
    g1_bdry = np.abs(R - R_g1) <= band_grid
    g2_bdry = np.abs(R - R_g2) <= band_grid
    g3_bdry = np.abs(R - R_g3) <= band_grid
    col_bdry = np.abs(R - R_col) <= band_col

    # --------------------------------------------------------
    # Drift-tube aperture through shells
    # --------------------------------------------------------
    x_ap_near = R_g1 - 2.0 * h

    aperture_range = (X >= x_ap_near) & (X <= x_dt_far)
    drift_axis_open = aperture_range & (rho_yz <= r_hole)

    g1_bdry = g1_bdry & ~drift_axis_open
    g2_bdry = g2_bdry & ~drift_axis_open
    g3_bdry = g3_bdry & ~drift_axis_open
    col_bdry = col_bdry & ~drift_axis_open

    # --------------------------------------------------------
    # Rod holes through shells
    # --------------------------------------------------------
    rod_axis = (rho_xy <= r_rod) & (Z <= 0.0)

    g1_bdry = g1_bdry & ~rod_axis
    g2_bdry = g2_bdry & ~rod_axis
    g3_bdry = g3_bdry & ~rod_axis
    col_bdry = col_bdry & ~rod_axis

    # --------------------------------------------------------
    # Side spherical cap openings
    # --------------------------------------------------------
    def spherical_cap_hole(R, X, Y, Z, Rshell, band_shell, u_axis, alpha_deg):
        u = np.asarray(u_axis, dtype=float)
        u = u / np.linalg.norm(u)

        shell = np.abs(R - Rshell) <= band_shell

        dot = X * u[0] + Y * u[1] + Z * u[2]
        cosang = dot / np.maximum(R, 1e-30)

        return shell & (cosang >= np.cos(np.deg2rad(alpha_deg)))

    u_open = np.array([
        np.cos(np.deg2rad(225.0)),
        np.sin(np.deg2rad(225.0)),
        0.0,
    ])

    open_g1 = spherical_cap_hole(R, X, Y, Z, R_g1, band_grid, u_open, 20.0)
    open_g2 = spherical_cap_hole(R, X, Y, Z, R_g2, band_grid, u_open, 18.0)
    open_g3 = spherical_cap_hole(R, X, Y, Z, R_g3, band_grid, u_open, 14.0)
    open_col = spherical_cap_hole(
        R,
        X,
        Y,
        Z,
        R_col,
        band_col,
        u_open,
        COLLECTOR_OPENING_ALPHA_DEG,
    )

    g1_bdry = g1_bdry & ~open_g1
    g2_bdry = g2_bdry & ~open_g2
    g3_bdry = g3_bdry & ~open_g3
    col_bdry = col_bdry & ~open_col

    # --------------------------------------------------------
    # Drift tube cylindrical boundary
    # --------------------------------------------------------
    in_range = (X >= x_dt_near - 5.0 * h) & (X <= x_dt_far)
    dt_band_bc = max(t_dt, 2.0 * h)

    if include_drifttube:
        drift_bc = (
            in_range
            & (rho_yz >= r_dt_i)
            & (rho_yz <= r_dt_i + dt_band_bc)
        )
    else:
        # Keep a full-size compatibility mask, but do not add any approximate
        # drift-tube conductor when the real STL is present.
        drift_bc = np.zeros(full_shape, dtype=bool)

    # Broadcast/copy to real full-size boolean arrays.
    g1_bdry = np.broadcast_to(g1_bdry, full_shape).copy()
    g2_bdry = np.broadcast_to(g2_bdry, full_shape).copy()
    g3_bdry = np.broadcast_to(g3_bdry, full_shape).copy()
    col_bdry = np.broadcast_to(col_bdry, full_shape).copy()
    drift_bc = np.broadcast_to(drift_bc, full_shape).copy()

    inside = np.broadcast_to(
        R <= (R_col + band_col),
        full_shape,
    ).copy()

    update_region = (
        inside
        & ~g1_bdry
        & ~g2_bdry
        & ~g3_bdry
        & ~col_bdry
    )

    field["g1_bdry"] = g1_bdry
    field["g2_bdry"] = g2_bdry
    field["g3_bdry"] = g3_bdry
    field["col_bdry"] = col_bdry
    field["drift_bc"] = drift_bc
    field["inside"] = inside
    field["update_region"] = update_region

    return field
    

def set_rfa_update_region(field: dict) -> dict:
    """
    Match the old notebook update region:

        inside collector domain,
        excluding analytic grid and collector boundary shells.

    Requires field to contain:
        g1_bdry, g2_bdry, g3_bdry, col_bdry, R_col.
    """
    h = float(field["h"])
    R_col = float(field["R_col"])

    X = field["x"][:, None, None]
    Y = field["y"][None, :, None]
    Z = field["z"][None, None, :]

    R = np.sqrt(X**2 + Y**2 + Z**2)

    band = 2.0 * h

    inside = np.broadcast_to(
        R <= (R_col + band),
        field["V"].shape,
    ).copy()

    update_region = (
        inside
        & ~field["g1_bdry"]
        & ~field["g2_bdry"]
        & ~field["g3_bdry"]
        & ~field["col_bdry"]
    )

    field["inside"] = inside
    field["update_region"] = update_region

    return field
    

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
    owner_name: str = "outer_boundary",
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

@njit
def _solve_laplace_sor_numba(
    V,
    Vfix,
    fixed,
    update_region,
    omega,
    tol,
    maxit,
):
    Nx, Ny, Nz = V.shape

    for it in range(1, maxit + 1):
        dmax = 0.0

        for k in range(1, Nz - 1):
            for j in range(1, Ny - 1):
                for i in range(1, Nx - 1):
                    if (not update_region[i, j, k]) or fixed[i, j, k]:
                        continue

                    rhs = (
                        V[i - 1, j, k] + V[i + 1, j, k]
                        + V[i, j - 1, k] + V[i, j + 1, k]
                        + V[i, j, k - 1] + V[i, j, k + 1]
                    ) / 6.0

                    vij = V[i, j, k]
                    vnew = vij + omega * (rhs - vij)

                    V[i, j, k] = vnew

                    d = abs(vnew - vij)
                    if d > dmax:
                        dmax = d

        # Re-enforce fixed values.
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    if fixed[i, j, k]:
                        V[i, j, k] = Vfix[i, j, k]

        if dmax < tol:
            return V, it, dmax

    return V, maxit, dmax


def solve_laplace_sor_numba(
    field: dict,
    max_iter: int = 20_000,
    tol: float = 2e-5,
    omega: float = 1.90,
    verbose: bool = True,
) -> dict:
    """
    Solve Laplace equation using the same Numba in-place SOR method
    used in the old test notebook.
    """
    Vfix = field.get("Vfix", field["V"].copy())
    fixed = field["fixed"]
    update_region = field["update_region"]

    V = field["V"].copy()

    if verbose:
        print("Solving Laplace with Numba SOR ...", flush=True)
        print(f"  grid shape: {V.shape}", flush=True)
        print(f"  fixed voxels: {int(np.count_nonzero(fixed)):,}", flush=True)
        print(f"  update voxels: {int(np.count_nonzero(update_region & ~fixed)):,}", flush=True)
        print(f"  omega = {omega}", flush=True)
        print(f"  tol   = {tol}", flush=True)

    t0 = time.perf_counter()

    V, it, dmax = _solve_laplace_sor_numba(
        V,
        Vfix,
        fixed,
        update_region,
        float(omega),
        float(tol),
        int(max_iter),
    )

    field["V"] = V
    field["solver"] = {
        "method": "numba_sor",
        "iterations": int(it),
        "tol": float(tol),
        "last_delta": float(dmax),
        "omega": float(omega),
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(dmax < tol),
    }

    if verbose:
        print(f"Finished: it = {it}, max dV = {dmax:.3e} V", flush=True)
        print(f"runtime = {field['solver']['runtime_s']:.2f} s", flush=True)

    return field


def solve_laplace_sor_taichi(
    field: dict,
    max_iter: int = 20_000,
    tol: float = 1e-5,
    omega: float = 1.85,
    check_every: int = 25,
    arch: str = "cpu",
    precision: str | None = None,
    cpu_max_num_threads: int | None = None,
    restore_best_on_max_iter: bool = True,
    record_history: bool = False,
    verbose: bool = True,
) -> dict:
    """Solve Laplace's equation with optional Taichi red-black SOR.

    ``arch="cpu"`` defaults to f64 and operates directly on the existing
    NumPy arrays.  ``arch="metal"`` defaults to f32 and keeps device arrays
    resident for the full iterative solve.  Geometry and boundary masks are
    not rebuilt or modified by this function.
    """
    try:
        from .taichi_sor import solve_red_black_sor_taichi
    except ImportError as exc:
        raise ImportError(
            "solver='taichi_sor' requires the optional Taichi package and "
            "the rfa_model/taichi_sor.py backend module. Install Taichi with "
            "`python -m pip install taichi`, then restart the kernel."
        ) from exc

    update_region = field.get("update_region")
    if update_region is None:
        update_region = np.ones_like(field["fixed"], dtype=bool)

    V, solver_metadata = solve_red_black_sor_taichi(
        V=field["V"],
        fixed=field["fixed"],
        update_region=update_region,
        max_iter=max_iter,
        tol=tol,
        omega=omega,
        check_every=check_every,
        arch=arch,
        precision=precision,
        cpu_max_num_threads=cpu_max_num_threads,
        restore_best_on_max_iter=restore_best_on_max_iter,
        record_history=record_history,
        verbose=verbose,
    )

    field["V"] = V
    field["solver"] = solver_metadata
    return field
    

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
    mesh_method: str = "voxelized",
    outer_boundary_voltage: float | None = None,
    solver: str = "numba_sor",
    max_iter: int = 20_000,
    tol: float = 1e-5,
    omega: float | None = None,
    taichi_arch: str = "cpu",
    taichi_precision: str | None = None,
    taichi_check_every: int = 25,
    taichi_cpu_threads: int | None = None,
    taichi_restore_best_on_max_iter: bool = True,
    taichi_record_history: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Build and solve a complete RFA electrostatic field.

    If ``meshes`` contains ``"drifttube"``, that aligned STL is voxelized at
    ``pitch=h`` and used as the only drift-tube conductor.  The legacy
    analytic cylinder is disabled automatically.  A fresh call to this
    function is therefore required after changing the drift-tube alignment.

    Select ``solver="taichi_sor"`` for the optional Taichi red-black SOR
    backend.  Its validation default is CPU/f64.  On macOS, request
    ``taichi_arch="metal"`` for the Apple GPU; Metal uses f32.
    If the tolerance is not reached, ``taichi_restore_best_on_max_iter=True``
    returns the checked iterate with the smallest maximum SOR update.  This
    requires one additional full potential array for the checkpoint.
    Set ``taichi_record_history=True`` to retain the small checkpoint-delta
    history used by the omega benchmark.  ``solver="initialize_only"`` builds
    the geometry and common linear initial potential without relaxing it.
    """

    if voltages is None:
        voltages = {
            "Vs": 0.0,
            "Vrod": 0.0,
            "Vg1": 0.0,
            "Vg2": 0.0,
            "Vg3": 0.0,
            "Vc": 50.0,
            "Vdt": 0.0,
        }

    t_total = time.perf_counter()

    _log(verbose, "Building RFA field")
    _log(verbose, "==================")
    _log(verbose, f"h = {h:.3e} m")
    _log(verbose, f"domain min = {xyz_min}")
    _log(verbose, f"domain max = {xyz_max}")
    _log(verbose, f"voltages = {voltages}")
    _log(verbose, "\n[1/7] Creating grid ...")

    field = make_empty_field_grid(
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        h=h,
    )

    _log(verbose, f"  grid shape = {field['V'].shape}")
    _log(verbose, f"  total voxels = {field['V'].size:,}")

    attach_rfa_metadata(
        field=field,
        voltages=voltages,
        R_g1=R_g1,
        R_g2=R_g2,
        R_g3=R_g3,
        R_col=R_col,
    )

    has_drifttube_stl = bool(
        meshes is not None
        and "drifttube" in meshes
        and meshes["drifttube"] is not None
    )
    field["drifttube_geometry"] = "stl" if has_drifttube_stl else "analytic"

    if has_drifttube_stl:
        field["drifttube_bounds_m"] = np.asarray(
            meshes["drifttube"].bounds,
            dtype=float,
        ).copy()
        field["drifttube_nose_x_m"] = float(
            field["drifttube_bounds_m"][0, 0]
        )
        _log(
            verbose,
            "Using aligned drift-tube STL: "
            f"nose x = {1e3 * field['drifttube_nose_x_m']:.3f} mm",
        )
    else:
        field["drifttube_nose_x_m"] = 0.047
        _log(
            verbose,
            "No drift-tube STL supplied; using the legacy analytic "
            "cylinder at x = 47.000 mm.",
        )

    if outer_boundary_voltage is None:
        outer_boundary_voltage = float(voltages.get("Vdt", 0.0))

    _log(verbose, "\n[2/7] Marking outer boundary ...")

    set_outer_boundary_fixed(
        field,
        voltage=outer_boundary_voltage,
        owner_name="outer_boundary",
    )

    _log(verbose, f"  fixed voxels after boundary = {int(field['fixed'].sum()):,}")

    if meshes is not None:
        _log(verbose, "\n[3/7] Voxelizing sample assembly meshes ...")

        mark_named_meshes(
            field,
            meshes=meshes,
            voltages=voltages,
            method=mesh_method,
            verbose=verbose,
        )

        _log(verbose, f"  fixed voxels after sample assembly = {int(field['fixed'].sum()):,}")
    else:
        _log(verbose, "\n[3/7] No sample assembly meshes provided.")

    if frame_meshes is not None:
        _log(verbose, "\n[4/7] Voxelizing grid-frame meshes ...")

        mark_named_meshes(
            field,
            meshes=frame_meshes,
            voltages=voltages,
            method=mesh_method,
            verbose=verbose,
        )

        _log(verbose, f"  fixed voxels after grid frames = {int(field['fixed'].sum()):,}")
    else:
        _log(verbose, "\n[4/7] No grid-frame meshes provided.")

    _log(verbose, "\n[5/7] Marking analytic grid/collector shells ...")

    mark_analytic_rfa_surfaces(
        field,
        voltages=voltages,
        R_g1=R_g1,
        R_g2=R_g2,
        R_g3=R_g3,
        R_col=R_col,
        include_analytic_drifttube=not has_drifttube_stl,
        verbose=verbose,
    )
    
    _log(verbose, f"  fixed voxels after analytic shells = {int(field['fixed'].sum()):,}")
    if verbose:
        _log(
            True,
            "  update voxels = "
            f"{_count_updatable_voxels(field['update_region'], field['fixed']):,}",
        )

    field["Vfix"] = field["V"].copy()

    _log(verbose, "\n[6/7] Initializing potential ...")

    initialize_potential_linear_x(
        field,
        V_left=outer_boundary_voltage,
        V_right=float(voltages.get("Vc", outer_boundary_voltage)),
    )

    _log(verbose, "\n[7/7] Solving Laplace equation ...")

    if solver in {"taichi", "taichi_sor"}:
        solve_laplace_sor_taichi(
            field,
            max_iter=max_iter,
            tol=tol,
            omega=omega if omega is not None else 1.85,
            check_every=taichi_check_every,
            arch=taichi_arch,
            precision=taichi_precision,
            cpu_max_num_threads=taichi_cpu_threads,
            restore_best_on_max_iter=taichi_restore_best_on_max_iter,
            record_history=taichi_record_history,
            verbose=verbose,
        )

    elif solver == "numba_sor":
        solve_laplace_sor_numba(
            field,
            max_iter=max_iter,
            tol=tol,
            omega=omega if omega is not None else 1.90,
            verbose=verbose,
        )

    elif solver == "sor":
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

    elif solver in {"initialize_only", "none"}:
        field["solver"] = {
            "method": "initial_potential_only",
            "iterations": 0,
            "tol": float(tol),
            "last_delta": None,
            "omega": None if omega is None else float(omega),
            "runtime_s": 0.0,
            "converged": False,
            "termination_reason": "not_solved",
        }

    else:
        raise ValueError(
            "solver must be 'taichi_sor', 'numba_sor', 'sor', 'jacobi', "
            "or 'initialize_only'"
        )

    _log(verbose, "\nCalculating electric field Ex, Ey, Ez ...")
    calculate_electric_field(field)

    attach_default_owner_name_map(field)

    if "validate_field_metadata" in globals():
        validate_field_metadata(field)

    _log(verbose, "\nField build complete")
    _log(verbose, f"total fixed voxels = {int(field['fixed'].sum()):,}")
    _log(verbose, f"solver converged = {field.get('solver', {}).get('converged', None)}")
    _log(verbose, f"total elapsed = {time.perf_counter() - t_total:.2f} s")

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
