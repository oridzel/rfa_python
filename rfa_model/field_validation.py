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

    geometry = {
        "shape": tuple(int(value) for value in np.asarray(field["V"]).shape),
        "h_m": float(field["h"]),
        "drifttube_geometry": field.get("drifttube_geometry"),
        "drifttube_nose_x_m": field.get("drifttube_nose_x_m"),
        "drifttube_voxels": field.get("mesh_voxel_counts", {}).get("drifttube"),
        "legacy_drift_bc_voxels": (
            int(np.count_nonzero(field["drift_bc"]))
            if "drift_bc" in field else None
        ),
    }

    tolerance = solver.get("tol")
    residual_limit = None if tolerance is None else 5.0 * float(tolerance)
    checks = {
        "solver_converged": bool(solver.get("converged", False)),
        "potential_is_finite": potential["nan_count"] == 0 and potential["inf_count"] == 0,
        "electric_field_is_finite": all(
            stats["nan_count"] == 0 and stats["inf_count"] == 0
            for stats in electric.values()
        ),
        "fixed_voxels_unchanged": fixed["max_abs_fixed_error_V"] in (0.0, None),
        "residual_within_5x_solver_tolerance": (
            None
            if residual_limit is None
            else residual["max_abs_neighbor_mean_residual_V"] <= residual_limit
        ),
    }
    checks["all_available_checks_pass"] = all(
        value for value in checks.values() if value is not None
    )

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
        f"voxels={geometry['drifttube_voxels']}"
    )
    for name, passed in report["checks"].items():
        status = "N/A" if passed is None else ("PASS" if passed else "FAIL")
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


# ---------------------------------------------------------------------------
# Presentation figures
# ---------------------------------------------------------------------------

_PRESENTATION_GEOMETRY_GROUPS = (
    (
        "Collector",
        ("collector_shell",),
        "#7A7A7A",
    ),
    (
        "Grids and frames",
        (
            "g1_shell", "g2_shell", "g3_shell",
            "g1frame", "g2frame", "g3frame",
        ),
        "#4C78A8",
    ),
    (
        "Drift tube",
        ("drifttube",),
        "#009E73",
    ),
    (
        "Sample support",
        ("holder", "receiver", "rod"),
        "#8C6D4F",
    ),
    (
        "Sample",
        ("sample",),
        "#E69F00",
    ),
)


def _presentation_domain_plane(field: dict, spec: dict[str, Any]) -> np.ndarray:
    """Return the solved/displayable domain for one plane."""
    if "inside" in field:
        domain = _take_plane(field["inside"], spec).astype(bool)
        if domain.shape == _take_plane(field["V"], spec).shape:
            return domain

    # Older fields may not retain ``inside``.  Reconstruct the spherical
    # collector domain when possible instead of showing the frozen box.
    if field.get("R_col") is not None:
        horizontal = np.asarray(field[spec["horizontal_name"]], dtype=float)
        vertical = np.asarray(field[spec["vertical_name"]], dtype=float)
        normal = float(spec["coordinate_actual_m"])
        H = horizontal[:, None]
        W = vertical[None, :]
        radius = np.sqrt(H**2 + W**2 + normal**2)
        return radius <= float(field["R_col"]) + 2.0 * float(field["h"])

    return np.ones_like(_take_plane(field["V"], spec), dtype=bool)


def _presentation_owner_plane(
    field: dict,
    spec: dict[str, Any],
) -> np.ndarray | None:
    if "owner" not in field:
        return None
    owner = _take_plane(field["owner"], spec)
    if owner.shape != _take_plane(field["V"], spec).shape:
        return None
    return owner


def _presentation_potential_limits(
    potential: np.ndarray,
    domain: np.ndarray,
    limits: tuple[float, float] | None,
) -> tuple[float, float]:
    if limits is not None:
        vmin, vmax = map(float, limits)
    else:
        values = np.asarray(potential)[domain & np.isfinite(potential)]
        if not values.size:
            raise ValueError("No finite potential values exist inside the display domain")
        vmin = float(np.min(values))
        vmax = float(np.max(values))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin > vmax:
        raise ValueError("potential_limits must be finite and increasing")
    if vmin == vmax:
        padding = max(1.0, abs(vmin) * 0.01)
        vmin -= padding
        vmax += padding
    return vmin, vmax


def _presentation_field_limits(
    field_v_per_mm: np.ndarray,
    vacuum: np.ndarray,
    limits: tuple[float, float] | None,
) -> tuple[float, float]:
    if limits is not None:
        lower, upper = map(float, limits)
    else:
        values = np.asarray(field_v_per_mm)[
            vacuum & np.isfinite(field_v_per_mm) & (field_v_per_mm > 0.0)
        ]
        if not values.size:
            return 1e-15, 1e-14
        upper = float(np.percentile(values, 99.8))
        lower = float(np.percentile(values, 2.0))
        lower = max(lower, upper * 1e-7, 1e-15)

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or lower >= upper
    ):
        raise ValueError(
            "field_limits_v_per_mm must contain two positive increasing values"
        )
    return lower, upper


def _presentation_zoom_limits_mm(
    field: dict,
    spec: dict[str, Any],
    horizontal_mm: np.ndarray,
    vertical_mm: np.ndarray,
    zoom_limits_mm: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if zoom_limits_mm is not None:
        if len(zoom_limits_mm) != 4:
            raise ValueError("zoom_limits_mm must be (left, right, bottom, top)")
        left, right, bottom, top = map(float, zoom_limits_mm)
        if not (left < right and bottom < top):
            raise ValueError("zoom_limits_mm must be increasing")
        return left, right, bottom, top

    if spec["horizontal_name"] == "x" and field.get("R_col") is not None:
        radius_mm = 1e3 * float(field["R_col"])
        return (
            -0.45 * radius_mm,
            1.05 * radius_mm,
            -0.55 * radius_mm,
            0.55 * radius_mm,
        )

    return (
        float(horizontal_mm[0]),
        float(horizontal_mm[-1]),
        float(vertical_mm[0]),
        float(vertical_mm[-1]),
    )


def _draw_presentation_geometry(
    ax,
    field: dict,
    spec: dict[str, Any],
    horizontal_mm: np.ndarray,
    vertical_mm: np.ndarray,
    owner_plane: np.ndarray | None,
    fixed_plane: np.ndarray,
):
    """Overlay conductor silhouettes and return legend handles."""
    from matplotlib.patches import Patch

    handles = []
    owner_ids = field.get("owner_id_map", {})

    if owner_plane is not None and owner_ids:
        for label, names, color in _PRESENTATION_GEOMETRY_GROUPS:
            ids = [int(owner_ids[name]) for name in names if name in owner_ids]
            if not ids:
                continue
            mask = np.isin(owner_plane, ids)
            if not np.any(mask):
                continue

            display = np.ma.masked_where(~mask.T, np.ones(mask.T.shape))
            ax.contourf(
                horizontal_mm,
                vertical_mm,
                display,
                levels=[0.5, 1.5],
                colors=[color],
                alpha=0.94,
                antialiased=False,
                zorder=6,
            )
            if np.any(~mask):
                ax.contour(
                    horizontal_mm,
                    vertical_mm,
                    mask.T.astype(float),
                    levels=[0.5],
                    colors="#202020",
                    linewidths=0.55,
                    zorder=7,
                )
            handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="#202020",
                    linewidth=0.55,
                    label=label,
                )
            )
        return handles

    # Compatibility fallback for fields saved without ownership metadata.
    if np.any(fixed_plane):
        mask = fixed_plane.astype(bool)
        display = np.ma.masked_where(~mask.T, np.ones(mask.T.shape))
        ax.contourf(
            horizontal_mm,
            vertical_mm,
            display,
            levels=[0.5, 1.5],
            colors=["#777777"],
            alpha=0.92,
            antialiased=False,
            zorder=6,
        )
        if np.any(~mask):
            ax.contour(
                horizontal_mm,
                vertical_mm,
                mask.T.astype(float),
                levels=[0.5],
                colors="#202020",
                linewidths=0.55,
                zorder=7,
            )
        handles.append(
            Patch(
                facecolor="#777777",
                edgecolor="#202020",
                linewidth=0.55,
                label="Fixed conductors",
            )
        )
    return handles


def _draw_presentation_potential(
    ax,
    *,
    horizontal_mm: np.ndarray,
    vertical_mm: np.ndarray,
    potential: np.ndarray,
    domain: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    equipotential_count: int,
    label_equipotentials: bool,
):
    from matplotlib.ticker import MaxNLocator

    display = np.ma.masked_where(~domain.T, potential.T)
    filled_levels = np.linspace(vmin, vmax, 96)
    image = ax.contourf(
        horizontal_mm,
        vertical_mm,
        display,
        levels=filled_levels,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extend="both",
        antialiased=True,
        zorder=1,
    )

    contour_set = None
    if equipotential_count >= 2:
        contour_levels = MaxNLocator(nbins=int(equipotential_count)).tick_values(
            vmin, vmax
        )
        contour_levels = contour_levels[
            (contour_levels > vmin) & (contour_levels < vmax)
        ]
        contour_set = ax.contour(
            horizontal_mm,
            vertical_mm,
            display,
            levels=contour_levels,
            colors="white",
            linewidths=0.55,
            alpha=0.70,
            zorder=3,
        )
        if label_equipotentials and len(contour_set.levels):
            ax.clabel(
                contour_set,
                contour_set.levels[::2],
                inline=True,
                fontsize=7,
                fmt=lambda value: f"{value:g} V",
            )
    return image


def _draw_presentation_streamlines(
    ax,
    *,
    horizontal_mm: np.ndarray,
    vertical_mm: np.ndarray,
    e_horizontal: np.ndarray,
    e_vertical: np.ndarray,
    e_magnitude_v_per_mm: np.ndarray,
    vacuum: np.ndarray,
    min_field_v_per_mm: float,
    density: float,
    max_points_per_axis: int,
) -> bool:
    if min_field_v_per_mm < 0.0:
        raise ValueError("min_stream_field_v_per_mm must be non-negative")

    step_h = max(1, int(np.ceil(len(horizontal_mm) / max_points_per_axis)))
    step_v = max(1, int(np.ceil(len(vertical_mm) / max_points_per_axis)))

    x_plot = horizontal_mm[::step_h]
    y_plot = vertical_mm[::step_v]
    u = np.asarray(e_horizontal)[::step_h, ::step_v]
    v = np.asarray(e_vertical)[::step_h, ::step_v]
    magnitude = np.asarray(e_magnitude_v_per_mm)[::step_h, ::step_v]
    valid = (
        np.asarray(vacuum)[::step_h, ::step_v]
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(magnitude)
        & (magnitude >= float(min_field_v_per_mm))
    )
    if np.count_nonzero(valid) < 4 or len(x_plot) < 2 or len(y_plot) < 2:
        return False

    u_plot = np.ma.masked_where(~valid.T, u.T)
    v_plot = np.ma.masked_where(~valid.T, v.T)
    magnitude_plot = magnitude.T
    reference = float(np.percentile(magnitude[valid], 95.0))
    if reference <= 0.0:
        return False
    linewidth = 0.45 + 1.25 * np.sqrt(
        np.clip(magnitude_plot / reference, 0.0, 1.0)
    )

    ax.streamplot(
        x_plot,
        y_plot,
        u_plot,
        v_plot,
        density=float(density),
        color="white",
        linewidth=linewidth,
        arrowsize=0.75,
        arrowstyle="-|>",
        minlength=0.18,
        integration_direction="both",
        broken_streamlines=True,
        zorder=4,
    )
    return True


def _style_presentation_axis(
    ax,
    *,
    spec: dict[str, Any],
    exterior_color: str,
) -> None:
    ax.set_facecolor(exterior_color)
    ax.set_xlabel(f"{spec['horizontal_name']} (mm)")
    ax.set_ylabel(f"{spec['vertical_name']} (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=3)
    ax.grid(False)


def plot_presentation_cutaway(
    field: dict,
    *,
    plane: str = "xy",
    coordinate: float = 0.0,
    title: str | None = None,
    potential_limits: tuple[float, float] | None = None,
    zoom_limits_mm: tuple[float, float, float, float] | None = None,
    equipotential_count: int = 10,
    min_stream_field_v_per_mm: float = 1e-3,
    stream_density: float = 1.05,
    cmap: str = "viridis",
    exterior_color: str = "#ECEFF1",
    show: bool = True,
):
    """Create a presentation-ready RFA electrostatic cutaway.

    The broad, unsolved exterior of the computational box is masked.  The main
    panel shows potential, labeled equipotentials, conductor silhouettes, and
    field lines.  A built-in inset enlarges the sample/grid/drift-tube region.

    ``min_stream_field_v_per_mm`` suppresses numerical-noise streamlines.  The
    default is 1 V/m = 1e-3 V/mm.
    """
    import matplotlib.pyplot as plt

    _require_field(field, electric=True)
    spec = _plane_spec(field, plane, coordinate)
    horizontal_mm = 1e3 * np.asarray(field[spec["horizontal_name"]], dtype=float)
    vertical_mm = 1e3 * np.asarray(field[spec["vertical_name"]], dtype=float)
    potential = _take_plane(field["V"], spec)
    fixed = _take_plane(field["fixed"], spec).astype(bool)
    domain = _presentation_domain_plane(field, spec)
    owner = _presentation_owner_plane(field, spec)
    vacuum = domain & (~fixed if owner is None else owner == 0)

    e_horizontal = _take_plane(field[spec["component_names"][0]], spec)
    e_vertical = _take_plane(field[spec["component_names"][1]], spec)
    e_magnitude_v_per_mm = 1e-3 * np.sqrt(
        sum(_take_plane(field[name], spec) ** 2 for name in ("Ex", "Ey", "Ez"))
    )
    vmin, vmax = _presentation_potential_limits(potential, domain, potential_limits)
    zoom = _presentation_zoom_limits_mm(
        field, spec, horizontal_mm, vertical_mm, zoom_limits_mm
    )

    fig, ax = plt.subplots(figsize=(10.8, 8.4))
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.14, top=0.90)
    image = _draw_presentation_potential(
        ax,
        horizontal_mm=horizontal_mm,
        vertical_mm=vertical_mm,
        potential=potential,
        domain=domain,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        equipotential_count=equipotential_count,
        label_equipotentials=True,
    )
    _draw_presentation_streamlines(
        ax,
        horizontal_mm=horizontal_mm,
        vertical_mm=vertical_mm,
        e_horizontal=e_horizontal,
        e_vertical=e_vertical,
        e_magnitude_v_per_mm=e_magnitude_v_per_mm,
        vacuum=vacuum,
        min_field_v_per_mm=min_stream_field_v_per_mm,
        density=stream_density,
        max_points_per_axis=240,
    )
    handles = _draw_presentation_geometry(
        ax, field, spec, horizontal_mm, vertical_mm, owner, fixed
    )
    _style_presentation_axis(ax, spec=spec, exterior_color=exterior_color)
    ax.set_xlim(float(horizontal_mm[0]), float(horizontal_mm[-1]))
    ax.set_ylim(float(vertical_mm[0]), float(vertical_mm[-1]))

    inset = ax.inset_axes([0.56, 0.53, 0.41, 0.41])
    _draw_presentation_potential(
        inset,
        horizontal_mm=horizontal_mm,
        vertical_mm=vertical_mm,
        potential=potential,
        domain=domain,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        equipotential_count=max(5, equipotential_count),
        label_equipotentials=False,
    )
    _draw_presentation_streamlines(
        inset,
        horizontal_mm=horizontal_mm,
        vertical_mm=vertical_mm,
        e_horizontal=e_horizontal,
        e_vertical=e_vertical,
        e_magnitude_v_per_mm=e_magnitude_v_per_mm,
        vacuum=vacuum,
        min_field_v_per_mm=min_stream_field_v_per_mm,
        density=max(1.20, stream_density),
        max_points_per_axis=220,
    )
    _draw_presentation_geometry(
        inset, field, spec, horizontal_mm, vertical_mm, owner, fixed
    )
    _style_presentation_axis(inset, spec=spec, exterior_color=exterior_color)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_xlim(zoom[0], zoom[1])
    inset.set_ylim(zoom[2], zoom[3])
    inset.set_title("Active region", fontsize=9, pad=3)
    inset.tick_params(labelsize=7)
    ax.indicate_inset_zoom(inset, edgecolor="#303030", alpha=0.55, linewidth=0.7)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Potential (V)")

    if title is None:
        title = (
            f"RFA electrostatic cutaway — {spec['plane'].upper()} slice at "
            f"{spec['normal_name']}={1e3 * spec['coordinate_actual_m']:.2f} mm"
        )
    fig.suptitle(title, fontsize=15, y=0.965)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.47, 0.025),
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=9,
        )

    if show:
        plt.show()
    return fig, ax, inset


def plot_presentation_potential_and_field(
    field: dict,
    *,
    plane: str = "xy",
    coordinate: float = 0.0,
    title: str | None = None,
    potential_limits: tuple[float, float] | None = None,
    field_limits_v_per_mm: tuple[float, float] | None = None,
    zoom_limits_mm: tuple[float, float, float, float] | None = None,
    equipotential_count: int = 10,
    potential_cmap: str = "viridis",
    field_cmap: str = "inferno",
    exterior_color: str = "#ECEFF1",
    show: bool = True,
):
    """Plot full-analyzer potential beside zoomed log10 field strength.

    Field strength is calculated from the full 3-D vector magnitude and shown
    in V/mm.  The unsolved computational-box exterior and conductor interiors
    are masked in the field-strength panel.
    """
    import matplotlib.pyplot as plt

    _require_field(field, electric=True)
    spec = _plane_spec(field, plane, coordinate)
    horizontal_mm = 1e3 * np.asarray(field[spec["horizontal_name"]], dtype=float)
    vertical_mm = 1e3 * np.asarray(field[spec["vertical_name"]], dtype=float)
    potential = _take_plane(field["V"], spec)
    fixed = _take_plane(field["fixed"], spec).astype(bool)
    domain = _presentation_domain_plane(field, spec)
    owner = _presentation_owner_plane(field, spec)
    vacuum = domain & (~fixed if owner is None else owner == 0)
    e_magnitude_v_per_mm = 1e-3 * np.sqrt(
        sum(_take_plane(field[name], spec) ** 2 for name in ("Ex", "Ey", "Ez"))
    )

    vmin, vmax = _presentation_potential_limits(potential, domain, potential_limits)
    field_min, field_max = _presentation_field_limits(
        e_magnitude_v_per_mm, vacuum, field_limits_v_per_mm
    )
    zoom = _presentation_zoom_limits_mm(
        field, spec, horizontal_mm, vertical_mm, zoom_limits_mm
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4))
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.17, top=0.86, wspace=0.28)

    image_v = _draw_presentation_potential(
        axes[0],
        horizontal_mm=horizontal_mm,
        vertical_mm=vertical_mm,
        potential=potential,
        domain=domain,
        vmin=vmin,
        vmax=vmax,
        cmap=potential_cmap,
        equipotential_count=equipotential_count,
        label_equipotentials=True,
    )
    handles = _draw_presentation_geometry(
        axes[0], field, spec, horizontal_mm, vertical_mm, owner, fixed
    )
    _style_presentation_axis(axes[0], spec=spec, exterior_color=exterior_color)
    axes[0].set_xlim(float(horizontal_mm[0]), float(horizontal_mm[-1]))
    axes[0].set_ylim(float(vertical_mm[0]), float(vertical_mm[-1]))
    axes[0].set_title("Electrostatic potential")
    cbar_v = fig.colorbar(image_v, ax=axes[0], fraction=0.047, pad=0.03)
    cbar_v.set_label("Potential (V)")

    log_field = np.log10(np.clip(e_magnitude_v_per_mm, field_min, field_max))
    log_display = np.ma.masked_where(~vacuum.T, log_field.T)
    image_e = axes[1].pcolormesh(
        horizontal_mm,
        vertical_mm,
        log_display,
        shading="auto",
        vmin=np.log10(field_min),
        vmax=np.log10(field_max),
        cmap=field_cmap,
        zorder=1,
    )
    _draw_presentation_geometry(
        axes[1], field, spec, horizontal_mm, vertical_mm, owner, fixed
    )
    _style_presentation_axis(axes[1], spec=spec, exterior_color=exterior_color)
    axes[1].set_xlim(zoom[0], zoom[1])
    axes[1].set_ylim(zoom[2], zoom[3])
    axes[1].set_title("Field concentration in the active region")
    cbar_e = fig.colorbar(image_e, ax=axes[1], fraction=0.047, pad=0.03)
    cbar_e.set_label(r"$\log_{10}|E|$  (V/mm)")

    if title is None:
        title = (
            f"RFA potential and field strength — {spec['plane'].upper()} slice at "
            f"{spec['normal_name']}={1e3 * spec['coordinate_actual_m']:.2f} mm"
        )
    fig.suptitle(title, fontsize=15, y=0.955)
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=9,
        )

    if show:
        plt.show()
    return fig, axes
