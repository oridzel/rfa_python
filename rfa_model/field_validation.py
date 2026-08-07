"""Memory-conscious numerical and visual validation for RFA fields.

The routines in this module never create full-grid coordinate arrays and do
not require two production-size fields to be resident at the same time.  A
small compressed signature stores one diagnostic plane, the axial field line,
and the corresponding geometry masks for later solver-to-solver comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_PLANES = {"xy", "xz", "yz"}


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _require_field(field: dict, *, electric: bool = False) -> None:
    required = ["x", "y", "z", "h", "V", "fixed", "update_region"]
    if electric:
        required.extend(["Ex", "Ey", "Ez"])
    missing = [name for name in required if name not in field]
    if missing:
        raise KeyError(f"Field is missing required keys: {missing}")

    shape = np.asarray(field["V"]).shape
    if len(shape) != 3:
        raise ValueError("field['V'] must be three-dimensional")
    for name in ["fixed", "update_region"]:
        if np.asarray(field[name]).shape != shape:
            raise ValueError(f"field[{name!r}] does not match V.shape")
    if electric:
        for name in ["Ex", "Ey", "Ez"]:
            if np.asarray(field[name]).shape != shape:
                raise ValueError(f"field[{name!r}] does not match V.shape")


def _nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - float(target))))


def _plane_spec(field: dict, plane: str, coordinate: float) -> dict[str, Any]:
    plane = str(plane).lower()
    if plane not in _PLANES:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    if plane == "xy":
        normal_name, normal_axis = "z", 2
        horizontal_name, vertical_name = "x", "y"
        horizontal_axis, vertical_axis = 0, 1
        component_names = ("Ex", "Ey")
    elif plane == "xz":
        normal_name, normal_axis = "y", 1
        horizontal_name, vertical_name = "x", "z"
        horizontal_axis, vertical_axis = 0, 2
        component_names = ("Ex", "Ez")
    else:
        normal_name, normal_axis = "x", 0
        horizontal_name, vertical_name = "y", "z"
        horizontal_axis, vertical_axis = 1, 2
        component_names = ("Ey", "Ez")

    normal_values = np.asarray(field[normal_name], dtype=float)
    index = _nearest_index(normal_values, coordinate)
    return {
        "plane": plane,
        "coordinate_requested_m": float(coordinate),
        "coordinate_actual_m": float(normal_values[index]),
        "index": index,
        "normal_name": normal_name,
        "normal_axis": normal_axis,
        "horizontal_name": horizontal_name,
        "vertical_name": vertical_name,
        "horizontal_axis": horizontal_axis,
        "vertical_axis": vertical_axis,
        "component_names": component_names,
    }


def _take_plane(array: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    array = np.asarray(array)
    axis = int(spec["normal_axis"])
    plane = np.take(array, int(spec["index"]), axis=axis)

    # np.take leaves the remaining axes in their original order.  Return
    # [horizontal, vertical] so plotting consistently uses plane.T.
    remaining = [axis_index for axis_index in range(3) if axis_index != axis]
    desired = [int(spec["horizontal_axis"]), int(spec["vertical_axis"])]
    if remaining != desired:
        plane = plane.T
    return plane


def _chunk_slices(length: int, chunk_size: int):
    for start in range(0, length, int(chunk_size)):
        yield slice(start, min(start + int(chunk_size), length))


def _finite_stats(array: np.ndarray, chunk_size: int = 8_000_000) -> dict:
    flat = np.ravel(np.asarray(array))
    nan_count = 0
    inf_count = 0
    finite_count = 0
    minimum = np.inf
    maximum = -np.inf

    for slc in _chunk_slices(flat.size, chunk_size):
        part = flat[slc]
        nan_count += int(np.count_nonzero(np.isnan(part)))
        inf_count += int(np.count_nonzero(np.isinf(part)))
        finite = part[np.isfinite(part)]
        finite_count += int(finite.size)
        if finite.size:
            minimum = min(minimum, float(np.min(finite)))
            maximum = max(maximum, float(np.max(finite)))

    return {
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "minimum": float(minimum) if finite_count else np.nan,
        "maximum": float(maximum) if finite_count else np.nan,
    }


def laplace_residual_stats(
    field: dict,
    *,
    chunk_x: int = 8,
) -> dict[str, float | int]:
    """Return exact residual statistics on active interior vacuum voxels.

    The intuitive residual is ``neighbor_mean - V`` in volts.  The physical
    discrete Laplacian is also reported in V/m^2.  Computation is chunked along
    x so the 0.25 mm production grid does not require a second full 3-D array.
    """
    _require_field(field)
    V = np.asarray(field["V"])
    fixed = np.asarray(field["fixed"], dtype=bool)
    update = np.asarray(field["update_region"], dtype=bool)
    h = float(field["h"])

    n_active = 0
    sum_square = 0.0
    max_abs = 0.0

    for i0 in range(1, V.shape[0] - 1, int(chunk_x)):
        i1 = min(i0 + int(chunk_x), V.shape[0] - 1)
        center = V[i0:i1, 1:-1, 1:-1]
        neighbor_mean = (
            V[i0 - 1:i1 - 1, 1:-1, 1:-1]
            + V[i0 + 1:i1 + 1, 1:-1, 1:-1]
            + V[i0:i1, :-2, 1:-1]
            + V[i0:i1, 2:, 1:-1]
            + V[i0:i1, 1:-1, :-2]
            + V[i0:i1, 1:-1, 2:]
        ) / 6.0
        active = (
            update[i0:i1, 1:-1, 1:-1]
            & ~fixed[i0:i1, 1:-1, 1:-1]
        )
        if not np.any(active):
            continue
        values = (neighbor_mean - center)[active]
        n_active += int(values.size)
        sum_square += float(np.dot(values, values))
        max_abs = max(max_abs, float(np.max(np.abs(values))))

    rms = np.sqrt(sum_square / n_active) if n_active else 0.0
    return {
        "active_interior_voxels": n_active,
        "max_abs_neighbor_mean_residual_V": max_abs,
        "rms_neighbor_mean_residual_V": float(rms),
        "max_abs_discrete_laplacian_V_per_m2": float(6.0 * max_abs / h**2),
        "rms_discrete_laplacian_V_per_m2": float(6.0 * rms / h**2),
    }


def fixed_boundary_error_stats(
    field: dict,
    *,
    chunk_x: int = 16,
) -> dict[str, float | int | None]:
    """Compare fixed voxels with field['Vfix'] without a full-grid temporary."""
    _require_field(field)
    if "Vfix" not in field:
        return {
            "fixed_voxels": int(np.count_nonzero(field["fixed"])),
            "max_abs_fixed_error_V": None,
            "rms_fixed_error_V": None,
            "note": "field['Vfix'] is unavailable",
        }

    V = np.asarray(field["V"])
    Vfix = np.asarray(field["Vfix"])
    fixed = np.asarray(field["fixed"], dtype=bool)
    if Vfix.shape != V.shape:
        raise ValueError("field['Vfix'] does not match V.shape")

    count = 0
    sum_square = 0.0
    max_abs = 0.0
    for slc in _chunk_slices(V.shape[0], chunk_x):
        mask = fixed[slc]
        if not np.any(mask):
            continue
        values = (V[slc] - Vfix[slc])[mask]
        count += int(values.size)
        sum_square += float(np.dot(values, values))
        max_abs = max(max_abs, float(np.max(np.abs(values))))

    rms = np.sqrt(sum_square / count) if count else 0.0
    return {
        "fixed_voxels": count,
        "max_abs_fixed_error_V": max_abs,
        "rms_fixed_error_V": float(rms),
    }


def _interior_owner_count(
    field: dict,
    owner_name: str,
    *,
    chunk_x: int = 8,
) -> int | None:
    """Count physical owner cells while excluding the numerical box faces."""
    if "owner" not in field:
        return None
    owner_id_map = field.get("owner_id_map", {})
    if owner_name not in owner_id_map:
        return None

    owner = np.asarray(field["owner"])
    if owner.ndim != 3 or any(size < 3 for size in owner.shape):
        return None

    owner_id = int(owner_id_map[owner_name])
    count = 0
    for i0 in range(1, owner.shape[0] - 1, int(chunk_x)):
        i1 = min(i0 + int(chunk_x), owner.shape[0] - 1)
        count += int(
            np.count_nonzero(owner[i0:i1, 1:-1, 1:-1] == owner_id)
        )
    return count


def validate_field_numerics(
    field: dict,
    *,
    residual_chunk_x: int = 8,
) -> dict[str, Any]:
    """Run the production-field checks that do not need a reference solver."""
    _require_field(field, electric=True)
    solver = dict(field.get("solver", {}))
    potential = _finite_stats(field["V"])
    electric = {name: _finite_stats(field[name]) for name in ("Ex", "Ey", "Ez")}
    fixed = fixed_boundary_error_stats(field)
    residual = laplace_residual_stats(field, chunk_x=residual_chunk_x)

    stored_mesh_voxels = field.get("mesh_voxel_counts", {}).get("drifttube")
    raw_mesh_points = field.get("mesh_voxel_point_counts", {}).get("drifttube")
    physical_dt_voxels = _interior_owner_count(
        field,
        "drifttube",
        chunk_x=residual_chunk_x,
    )
    geometry = {
        "shape": tuple(int(value) for value in np.asarray(field["V"]).shape),
        "h_m": float(field["h"]),
        "drifttube_geometry": field.get("drifttube_geometry"),
        "drifttube_nose_x_m": field.get("drifttube_nose_x_m"),
        "drifttube_voxels": (
            physical_dt_voxels
            if physical_dt_voxels is not None
            else stored_mesh_voxels
        ),
        "drifttube_unique_voxels_at_assignment": stored_mesh_voxels,
        "drifttube_raw_voxel_points": raw_mesh_points,
        "legacy_drift_bc_voxels": (
            int(np.count_nonzero(field["drift_bc"]))
            if "drift_bc" in field else None
        ),
    }

    tolerance = solver.get("tol")
    residual_limit = None if tolerance is None else 5.0 * float(tolerance)
    strict_convergence = bool(solver.get("converged", False))
    residual_acceptable = (
        None
        if residual_limit is None
        else residual["max_abs_neighbor_mean_residual_V"] <= residual_limit
    )
    required_checks = {
        "potential_is_finite": potential["nan_count"] == 0 and potential["inf_count"] == 0,
        "electric_field_is_finite": all(
            stats["nan_count"] == 0 and stats["inf_count"] == 0
            for stats in electric.values()
        ),
        "fixed_voxels_unchanged": fixed["max_abs_fixed_error_V"] in (0.0, None),
        "residual_within_5x_solver_tolerance": residual_acceptable,
    }
    solution_acceptable = all(
        value for value in required_checks.values() if value is not None
    )
    checks = {
        "solver_tolerance_reached": strict_convergence,
        **required_checks,
        "solution_acceptable": solution_acceptable,
        "all_required_checks_pass": solution_acceptable,
    }

    return {
        "solver": solver,
        "geometry": geometry,
        "potential": potential,
        "electric_field": electric,
        "fixed_boundary": fixed,
        "laplace_residual": residual,
        "checks": checks,
    }


def print_validation_report(report: dict[str, Any]) -> None:
    """Print a compact, readable report from validate_field_numerics()."""
    solver = report["solver"]
    fixed = report["fixed_boundary"]
    residual = report["laplace_residual"]
    geometry = report["geometry"]

    print("RFA field validation")
    print("--------------------")
    print(
        f"solver: {solver.get('method', 'unknown')} | "
        f"converged={solver.get('converged')} | "
        f"iterations={solver.get('iterations')} | "
        f"last_delta={solver.get('last_delta')}"
    )
    if solver.get("best_iteration") is not None:
        print(
            f"best checked iterate: {solver.get('best_iteration')} | "
            f"best_delta={solver.get('best_delta')} | "
            f"selected={solver.get('selected_iteration')} | "
            f"restored_best={solver.get('restored_best')}"
        )
    print(f"shape: {geometry['shape']} | h={1e3 * geometry['h_m']:.3f} mm")
    print(
        "fixed boundary max error: "
        f"{fixed['max_abs_fixed_error_V']} V"
    )
    print(
        "Laplace residual: max="
        f"{residual['max_abs_neighbor_mean_residual_V']:.6e} V, rms="
        f"{residual['rms_neighbor_mean_residual_V']:.6e} V"
    )
    print(
        f"drift tube: {geometry['drifttube_geometry']}, "
        f"nose={geometry['drifttube_nose_x_m']}, "
        f"physical voxels={geometry['drifttube_voxels']}, "
        "unique at assignment="
        f"{geometry['drifttube_unique_voxels_at_assignment']}, "
        f"raw points={geometry['drifttube_raw_voxel_points']}"
    )
    for name, passed in report["checks"].items():
        if passed is None:
            status = "N/A"
        elif passed:
            status = "PASS"
        elif (
            name == "solver_tolerance_reached"
            and report["checks"].get("solution_acceptable", False)
        ):
            status = "WARN"
        else:
            status = "FAIL"
        print(f"{status:4s}  {name}")


def save_field_signature(
    field: dict,
    path: str | Path,
    *,
    plane: str = "xy",
    coordinate: float = 0.0,
) -> Path:
    """Save a compact solver-comparison reference instead of the full field."""
    _require_field(field, electric=True)
    spec = _plane_spec(field, plane, coordinate)
    ix_y0 = _nearest_index(field["y"], 0.0)
    ix_z0 = _nearest_index(field["z"], 0.0)

    metadata = {
        "format_version": 1,
        "plane": spec["plane"],
        "coordinate_requested_m": spec["coordinate_requested_m"],
        "coordinate_actual_m": spec["coordinate_actual_m"],
        "shape": tuple(int(value) for value in np.asarray(field["V"]).shape),
        "h_m": float(field["h"]),
        "solver": field.get("solver", {}),
        "drifttube_geometry": field.get("drifttube_geometry"),
        "drifttube_nose_x_m": field.get("drifttube_nose_x_m"),
        "mesh_voxel_counts": field.get("mesh_voxel_counts", {}),
    }

    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, default=_json_default)),
        "x": np.asarray(field["x"]),
        "y": np.asarray(field["y"]),
        "z": np.asarray(field["z"]),
        "V_plane": _take_plane(field["V"], spec),
        "Ex_plane": _take_plane(field["Ex"], spec),
        "Ey_plane": _take_plane(field["Ey"], spec),
        "Ez_plane": _take_plane(field["Ez"], spec),
        "fixed_plane": _take_plane(field["fixed"], spec),
        "update_plane": _take_plane(field["update_region"], spec),
        "V_axis": np.asarray(field["V"][:, ix_y0, ix_z0]),
        "Ex_axis": np.asarray(field["Ex"][:, ix_y0, ix_z0]),
        "Ey_axis": np.asarray(field["Ey"][:, ix_y0, ix_z0]),
        "Ez_axis": np.asarray(field["Ez"][:, ix_y0, ix_z0]),
    }
    if "owner" in field:
        payload["owner_plane"] = _take_plane(field["owner"], spec)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def _percentile_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            f"{prefix}_max": np.nan,
            f"{prefix}_rms": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
        }
    absolute = np.abs(values)
    return {
        f"{prefix}_max": float(np.max(absolute)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_p95": float(np.percentile(absolute, 95.0)),
        f"{prefix}_p99": float(np.percentile(absolute, 99.0)),
    }


def compare_field_to_signature(
    field: dict,
    reference_path: str | Path,
) -> dict[str, Any]:
    """Compare one diagnostic plane and axis line with a saved reference."""
    _require_field(field, electric=True)
    with np.load(reference_path, allow_pickle=False) as reference:
        metadata = json.loads(str(reference["metadata_json"]))
        spec = _plane_spec(
            field,
            metadata["plane"],
            float(metadata["coordinate_actual_m"]),
        )

        coordinate_match = all(
            np.array_equal(np.asarray(field[name]), reference[name])
            for name in ("x", "y", "z")
        )
        if not coordinate_match:
            raise ValueError(
                "Current field coordinates do not exactly match the reference. "
                "Use identical bounds and h for solver comparison."
            )

        fixed_current = _take_plane(field["fixed"], spec)
        update_current = _take_plane(field["update_region"], spec)
        fixed_reference = reference["fixed_plane"].astype(bool)
        update_reference = reference["update_plane"].astype(bool)

        fixed_mismatch = int(np.count_nonzero(fixed_current != fixed_reference))
        update_mismatch = int(np.count_nonzero(update_current != update_reference))
        owner_mismatch = None
        if "owner_plane" in reference.files and "owner" in field:
            owner_mismatch = int(
                np.count_nonzero(
                    _take_plane(field["owner"], spec) != reference["owner_plane"]
                )
            )

        valid = (
            update_current
            & update_reference
            & ~fixed_current
            & ~fixed_reference
        )
        V_current = _take_plane(field["V"], spec)
        dV = V_current - reference["V_plane"]

        components_current = np.stack(
            [_take_plane(field[name], spec) for name in ("Ex", "Ey", "Ez")],
            axis=0,
        )
        components_reference = np.stack(
            [reference[f"{name}_plane"] for name in ("Ex", "Ey", "Ez")],
            axis=0,
        )
        dE = np.linalg.norm(components_current - components_reference, axis=0)
        E_reference = np.linalg.norm(components_reference, axis=0)

        metrics = {}
        metrics.update(_percentile_metrics(dV[valid], "abs_dV_V"))
        metrics.update(_percentile_metrics(dE[valid], "abs_dE_V_per_m"))
        reference_e_rms = float(np.sqrt(np.mean(E_reference[valid] ** 2)))
        dE_rms = metrics["abs_dE_V_per_m_rms"]
        metrics["E_vector_normalized_rms"] = (
            float(dE_rms / reference_e_rms) if reference_e_rms > 0.0 else np.nan
        )

        dV_axis = np.asarray(field["V"][:, _nearest_index(field["y"], 0.0), _nearest_index(field["z"], 0.0)]) - reference["V_axis"]
        dEx_axis = np.asarray(field["Ex"][:, _nearest_index(field["y"], 0.0), _nearest_index(field["z"], 0.0)]) - reference["Ex_axis"]
        metrics.update(_percentile_metrics(dV_axis, "axis_abs_dV_V"))
        metrics.update(_percentile_metrics(dEx_axis, "axis_abs_dEx_V_per_m"))

    return {
        "reference_metadata": metadata,
        "current_solver": dict(field.get("solver", {})),
        "geometry": {
            "fixed_plane_mismatch_voxels": fixed_mismatch,
            "update_plane_mismatch_voxels": update_mismatch,
            "owner_plane_mismatch_voxels": owner_mismatch,
        },
        "metrics": metrics,
        "checks": {
            "plane_geometry_matches": (
                fixed_mismatch == 0
                and update_mismatch == 0
                and owner_mismatch in (0, None)
            )
        },
    }


def _draw_fixed_contour(ax, horizontal_mm, vertical_mm, fixed_plane) -> None:
    if np.any(fixed_plane) and np.any(~fixed_plane):
        ax.contour(
            horizontal_mm,
            vertical_mm,
            fixed_plane.T.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=0.45,
            alpha=0.85,
        )


def _mark_axial_electrodes(ax, field: dict) -> None:
    markers = [
        (field.get("R_g1"), "G1"),
        (field.get("R_g2"), "G2"),
        (field.get("R_g3"), "G3"),
        (field.get("R_col"), "collector"),
        (field.get("drifttube_nose_x_m"), "DT nose"),
    ]
    for position, label in markers:
        if position is None:
            continue
        ax.axvline(1e3 * float(position), color="0.4", linewidth=0.6, alpha=0.5)
        ax.annotate(
            label,
            xy=(1e3 * float(position), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color="0.3",
        )


def plot_field_overview(
    field: dict,
    *,
    plane: str = "xy",
    coordinate: float = 0.0,
    arrows_per_axis: int = 32,
):
    """Plot potential, field magnitude/direction, and the on-axis profiles."""
    import matplotlib.pyplot as plt

    _require_field(field, electric=True)
    spec = _plane_spec(field, plane, coordinate)
    horizontal = 1e3 * np.asarray(field[spec["horizontal_name"]])
    vertical = 1e3 * np.asarray(field[spec["vertical_name"]])
    potential = _take_plane(field["V"], spec)
    fixed = _take_plane(field["fixed"], spec).astype(bool)
    E_components = {
        name: _take_plane(field[name], spec) for name in ("Ex", "Ey", "Ez")
    }
    E_magnitude = np.sqrt(sum(component**2 for component in E_components.values()))
    E_horizontal = E_components[spec["component_names"][0]]
    E_vertical = E_components[spec["component_names"][1]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    image_v = axes[0, 0].pcolormesh(
        horizontal, vertical, potential.T, shading="auto", cmap="viridis"
    )
    levels = np.linspace(float(np.nanmin(potential)), float(np.nanmax(potential)), 15)
    if np.unique(levels).size > 1:
        axes[0, 0].contour(
            horizontal,
            vertical,
            potential.T,
            levels=levels,
            colors="white",
            linewidths=0.35,
            alpha=0.55,
        )
    _draw_fixed_contour(axes[0, 0], horizontal, vertical, fixed)
    fig.colorbar(image_v, ax=axes[0, 0], label="Potential (V)")
    axes[0, 0].set_title(
        f"Potential in {spec['plane']} plane at "
        f"{spec['normal_name']}={1e3 * spec['coordinate_actual_m']:.3f} mm"
    )

    positive = E_magnitude[E_magnitude > 0.0]
    floor = max(float(np.min(positive)) if positive.size else 1.0, 1e-12)
    log_e = np.log10(np.maximum(E_magnitude, floor))
    image_e = axes[0, 1].pcolormesh(
        horizontal, vertical, log_e.T, shading="auto", cmap="magma"
    )
    _draw_fixed_contour(axes[0, 1], horizontal, vertical, fixed)

    step_h = max(1, len(horizontal) // int(arrows_per_axis))
    step_v = max(1, len(vertical) // int(arrows_per_axis))
    qh = E_horizontal[::step_h, ::step_v]
    qv = E_vertical[::step_h, ::step_v]
    qnorm = np.hypot(qh, qv)
    qh = np.divide(qh, qnorm, out=np.zeros_like(qh), where=qnorm > 0.0)
    qv = np.divide(qv, qnorm, out=np.zeros_like(qv), where=qnorm > 0.0)
    axes[0, 1].quiver(
        horizontal[::step_h],
        vertical[::step_v],
        qh.T,
        qv.T,
        color="white",
        alpha=0.65,
        scale=42,
        width=0.002,
        pivot="mid",
    )
    fig.colorbar(image_e, ax=axes[0, 1], label=r"log$_{10}|E|$ (V/m)")
    axes[0, 1].set_title("Electric-field magnitude and direction")

    iy0 = _nearest_index(field["y"], 0.0)
    iz0 = _nearest_index(field["z"], 0.0)
    x_mm = 1e3 * np.asarray(field["x"])
    axes[1, 0].plot(x_mm, np.asarray(field["V"])[:, iy0, iz0], linewidth=1.4)
    axes[1, 0].set_ylabel("Potential (V)")
    axes[1, 0].set_title("Potential on the analyzer axis (y=z=0)")
    _mark_axial_electrodes(axes[1, 0], field)

    axes[1, 1].plot(x_mm, np.asarray(field["Ex"])[:, iy0, iz0], linewidth=1.4)
    axes[1, 1].axhline(0.0, color="0.5", linewidth=0.6)
    axes[1, 1].set_ylabel(r"$E_x$ (V/m)")
    axes[1, 1].set_title("Axial electric field (y=z=0)")
    _mark_axial_electrodes(axes[1, 1], field)

    for ax in axes.flat:
        ax.set_xlabel(f"{spec['horizontal_name']} (mm)" if ax in axes[0] else "x (mm)")
        if ax in axes[0]:
            ax.set_ylabel(f"{spec['vertical_name']} (mm)")
            ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    return fig, axes


def plot_field_comparison(
    field: dict,
    reference_path: str | Path,
):
    """Plot current-minus-reference differences and axial overlays."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    _require_field(field, electric=True)
    with np.load(reference_path, allow_pickle=False) as reference:
        metadata = json.loads(str(reference["metadata_json"]))
        spec = _plane_spec(
            field,
            metadata["plane"],
            float(metadata["coordinate_actual_m"]),
        )
        for name in ("x", "y", "z"):
            if not np.array_equal(np.asarray(field[name]), reference[name]):
                raise ValueError("Current and reference field coordinates differ")

        horizontal = 1e3 * np.asarray(field[spec["horizontal_name"]])
        vertical = 1e3 * np.asarray(field[spec["vertical_name"]])
        dV_mV = 1e3 * (_take_plane(field["V"], spec) - reference["V_plane"])
        fixed = _take_plane(field["fixed"], spec).astype(bool)
        dE = np.sqrt(
            sum(
                (
                    _take_plane(field[name], spec) - reference[f"{name}_plane"]
                ) ** 2
                for name in ("Ex", "Ey", "Ez")
            )
        )
        x_mm = 1e3 * np.asarray(field["x"])
        iy0 = _nearest_index(field["y"], 0.0)
        iz0 = _nearest_index(field["z"], 0.0)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        vmax = float(np.nanmax(np.abs(dV_mV)))
        if vmax == 0.0:
            vmax = np.finfo(float).eps
        image_v = axes[0, 0].pcolormesh(
            horizontal,
            vertical,
            dV_mV.T,
            shading="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        )
        _draw_fixed_contour(axes[0, 0], horizontal, vertical, fixed)
        fig.colorbar(image_v, ax=axes[0, 0], label="Current - reference (mV)")
        axes[0, 0].set_title("Potential difference")

        positive = dE[dE > 0.0]
        floor = max(float(np.min(positive)) if positive.size else 1e-30, 1e-30)
        image_e = axes[0, 1].pcolormesh(
            horizontal,
            vertical,
            np.log10(np.maximum(dE, floor)).T,
            shading="auto",
            cmap="magma",
        )
        _draw_fixed_contour(axes[0, 1], horizontal, vertical, fixed)
        fig.colorbar(image_e, ax=axes[0, 1], label=r"log$_{10}|\Delta E|$ (V/m)")
        axes[0, 1].set_title("Electric-field vector difference")
        if not positive.size:
            axes[0, 1].text(
                0.5,
                0.5,
                r"$|\Delta E|=0$ V/m",
                transform=axes[0, 1].transAxes,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        axes[1, 0].plot(x_mm, reference["V_axis"], label="reference", linewidth=1.3)
        axes[1, 0].plot(
            x_mm,
            np.asarray(field["V"])[:, iy0, iz0],
            label="current",
            linewidth=1.0,
            linestyle="--",
        )
        axes[1, 0].set_ylabel("Potential (V)")
        axes[1, 0].set_title("On-axis potential")
        axes[1, 0].legend()
        _mark_axial_electrodes(axes[1, 0], field)

        axes[1, 1].plot(x_mm, reference["Ex_axis"], label="reference", linewidth=1.3)
        axes[1, 1].plot(
            x_mm,
            np.asarray(field["Ex"])[:, iy0, iz0],
            label="current",
            linewidth=1.0,
            linestyle="--",
        )
        axes[1, 1].set_ylabel(r"$E_x$ (V/m)")
        axes[1, 1].set_title("On-axis electric field")
        axes[1, 1].legend()
        _mark_axial_electrodes(axes[1, 1], field)

        for ax in axes.flat:
            ax.set_xlabel(f"{spec['horizontal_name']} (mm)" if ax in axes[0] else "x (mm)")
            if ax in axes[0]:
                ax.set_ylabel(f"{spec['vertical_name']} (mm)")
                ax.set_aspect("equal", adjustable="box")
            ax.grid(False)

    return fig, axes
