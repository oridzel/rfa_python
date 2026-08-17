"""
cascade.py

Cascade-emission runner for the RFA model.

This module extends the first-generation model:

    primary electron -> sample impact -> sample SE/BSE emission

to a cascade model:

    emitted electron hits grid/collector/holder/etc.
    -> that impact can emit additional SE/BSE electrons
    -> those electrons are tracked recursively

The cascade is controlled by:
    max_generation
    max_total_electrons
    min_incident_energy_eV

Current accounting is intentionally kept separate. This module returns
trajectory results with enough metadata for later cascade current accounting.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import time
import pandas as pd

from .constants import kinetic_energy_eV_from_velocity
from .collisions import make_sample_plane_geometry
from .trajectories import (
    unit,
    classify_effective_vacuum_point,
    integrate_one_electron,
    place_emitted_particle_in_vacuum,
)
from .primary import (
    fly_primary_to_sample,
    make_primary_beam_near_sample,
)
from .samplers import (
    generate_surface_emissions,
    canonical_surface_name,
    surface_family,
    angular_yield_gain,
    sample_angle_dependent_yield_gains,
    ANALYTIC_MESH_SURFACES,
    MAX_ANGULAR_YIELD_GAIN,
)
from .accounting import (
    terminal_owner_from_result,
    electrode_from_owner,
    summarize_cascade_accounting,
    add_per_primary_to_current_counts,
    grid_events_to_dataframe_many
)





# ============================================================
# Sample front-face analytic geometry
# ============================================================

def _patch_sample_geometry_extents(
    geometry: dict,
    center,
    normal,
    u_axis,
    v_axis,
    u_bounds,
    v_bounds,
    *,
    theta_deg: float,
    source: str,
) -> dict:
    """Patch/augment a sample-geometry dictionary without assuming key names.

    ``make_sample_plane_geometry`` remains the compatibility constructor for
    the collision helpers.  This function replaces the quantities that must be
    tied to the *physical front face*, and also writes common aliases for local
    finite-face bounds.  Unknown extra keys are harmless.
    """
    g = dict(geometry)
    center = np.asarray(center, dtype=float)
    normal = unit(np.asarray(normal, dtype=float))
    u_axis = unit(np.asarray(u_axis, dtype=float))
    v_axis = unit(np.asarray(v_axis, dtype=float))
    u_bounds = tuple(float(x) for x in u_bounds)
    v_bounds = tuple(float(x) for x in v_bounds)

    g.update({
        "center": center,
        "normal": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "tangent_u": u_axis,
        "tangent_v": v_axis,
        "u_hat": u_axis,
        "v_hat": v_axis,
        "u_bounds": u_bounds,
        "v_bounds": v_bounds,
        "sample_u_bounds": u_bounds,
        "sample_v_bounds": v_bounds,
        "half_width_u": 0.5 * (u_bounds[1] - u_bounds[0]),
        "half_width_v": 0.5 * (v_bounds[1] - v_bounds[0]),
        "half_width": 0.5 * (u_bounds[1] - u_bounds[0]),
        "half_height": 0.5 * (v_bounds[1] - v_bounds[0]),
        "theta_deg": float(theta_deg),
        "geometry_source": str(source),
        "plane_offset_m": float(np.dot(normal, center)),
    })
    return g


def infer_sample_front_face_geometry_from_stl(
    collision_mesh,
    face_owner,
    theta_deg: float,
    sample_y_bounds,
    sample_z_bounds,
    *,
    x_sample: float = 0.0,
    normal_alignment_min: float = 0.95,
    plane_tolerance_m: float = 2.0e-6,
    fallback_center=None,
) -> dict:
    """Build a pivot-anchored analytic sample face using STL dimensions.

    The sample solid has finite thickness, so the midpoint of its *rotated
    axis-aligned bounding box* is not the front-face centre at grazing angle.
    Instead, identify sample-owned triangles whose normals are parallel to the
    expected rotated front-face normal and select the outermost plane along
    that normal.  The local U/V *spans* are measured from those triangles, but
    their absolute STL coordinates are deliberately not copied into the
    analytic collision geometry.  The analytic face is anchored at the known
    rotation pivot/front-face position.

    ``fallback_center`` is retained as the argument name for compatibility,
    but when supplied it is the authoritative face anchor even when STL
    inference succeeds.  Otherwise the anchor is
    ``(x_sample, 0, midpoint(sample_z_bounds))``.

    If STL inference is impossible, use the supplied sample bounds on the same
    pivot-anchored plane.
    """
    a = np.deg2rad(float(theta_deg))
    n_expected = unit(np.array([np.cos(a), np.sin(a), 0.0], dtype=float))
    u_axis = unit(np.array([-np.sin(a), np.cos(a), 0.0], dtype=float))
    v_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    # Start from the existing constructor solely so all collision-helper keys
    # expected by the user's collisions.py remain present.
    base = make_sample_plane_geometry(
        sample_y_bounds=sample_y_bounds,
        sample_z_bounds=sample_z_bounds,
        theta_deg=float(theta_deg),
        x_sample=float(x_sample),
    )

    if fallback_center is None:
        z0 = 0.5 * (float(sample_z_bounds[0]) + float(sample_z_bounds[1]))
        face_anchor = np.array([float(x_sample), 0.0, z0], dtype=float)
    else:
        face_anchor = np.asarray(fallback_center, dtype=float)

    if face_anchor.shape != (3,) or not np.all(np.isfinite(face_anchor)):
        raise ValueError("sample face anchor must be a finite three-vector")

    try:
        owners = np.asarray(face_owner, dtype=object)
        sample_face_ids = np.array([
            i for i, owner in enumerate(owners)
            if canonical_surface_name(owner) == "sample"
        ], dtype=int)

        if sample_face_ids.size == 0:
            raise ValueError("no sample-owned faces in primary collision mesh")

        faces = np.asarray(collision_mesh.faces, dtype=int)
        vertices = np.asarray(collision_mesh.vertices, dtype=float)
        normals = np.asarray(collision_mesh.face_normals, dtype=float)

        tri = vertices[faces[sample_face_ids]]
        centroids = tri.mean(axis=1)
        face_normals = normals[sample_face_ids]
        face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]

        alignment = np.abs(face_normals @ n_expected)
        planar = alignment >= float(normal_alignment_min)
        if not np.any(planar):
            raise ValueError("no sample faces parallel to expected front-face normal")

        ids_planar = sample_face_ids[planar]
        cent_planar = centroids[planar]
        s_planar = cent_planar @ n_expected

        # The sample body extends behind the exposed face, so the vacuum/front
        # face is the maximum coordinate along the outward normal.
        s_front = float(np.max(s_planar))
        front_sel = np.abs(s_planar - s_front) <= float(plane_tolerance_m)
        front_face_ids = ids_planar[front_sel]
        if front_face_ids.size == 0:
            raise ValueError("front-face plane cluster is empty")

        front_vertices = vertices[faces[front_face_ids]].reshape(-1, 3)
        d = float(np.median(front_vertices @ n_expected))
        u_abs = front_vertices @ u_axis
        v_abs = front_vertices @ v_axis

        u_min_abs, u_max_abs = float(np.min(u_abs)), float(np.max(u_abs))
        v_min_abs, v_max_abs = float(np.min(v_abs)), float(np.max(v_abs))
        u_center_raw = 0.5 * (u_min_abs + u_max_abs)
        v_center_raw = 0.5 * (v_min_abs + v_max_abs)
        stl_center_raw = (
            d * n_expected
            + u_center_raw * u_axis
            + v_center_raw * v_axis
        )

        # The aligned collision STL may carry an assembly translation.  That
        # translation must not redefine the analytical sample coordinate
        # system or move the primary beam.  Preserve only the measured face
        # dimensions and place them symmetrically about the known pivot.
        u_span = u_max_abs - u_min_abs
        v_span = v_max_abs - v_min_abs
        if not (np.isfinite(u_span) and u_span > 0.0):
            raise ValueError("sample STL front face has invalid U span")
        if not (np.isfinite(v_span) and v_span > 0.0):
            raise ValueError("sample STL front face has invalid V span")

        center = face_anchor.copy()
        u_bounds = (-0.5 * u_span, 0.5 * u_span)
        v_bounds = (-0.5 * v_span, 0.5 * v_span)

        g = _patch_sample_geometry_extents(
            base,
            center=center,
            normal=n_expected,
            u_axis=u_axis,
            v_axis=v_axis,
            u_bounds=u_bounds,
            v_bounds=v_bounds,
            theta_deg=float(theta_deg),
            source="sample_stl_extents_pivot_anchored",
        )
        g.update({
            "front_face_triangle_count": int(front_face_ids.size),
            "front_face_ids": front_face_ids,
            "normal_alignment_min": float(normal_alignment_min),
            "front_plane_tolerance_m": float(plane_tolerance_m),
            "stl_front_face_center_raw": stl_center_raw,
            "stl_front_plane_offset_raw_m": d,
            "stl_to_analytic_center_shift_m": center - stl_center_raw,
        })
        return g

    except Exception as exc:
        center = face_anchor.copy()

        # Preserve generous finite bounds on fallback.  The key correction is
        # that the plane itself passes through the pivot/front-face location,
        # not through the rotated solid's AABB centre.
        span_y = abs(float(sample_y_bounds[1]) - float(sample_y_bounds[0]))
        span_z = abs(float(sample_z_bounds[1]) - float(sample_z_bounds[0]))
        g = _patch_sample_geometry_extents(
            base,
            center=center,
            normal=n_expected,
            u_axis=u_axis,
            v_axis=v_axis,
            u_bounds=(-0.5 * span_y, 0.5 * span_y),
            v_bounds=(-0.5 * span_z, 0.5 * span_z),
            theta_deg=float(theta_deg),
            source="rotation_pivot_fallback",
        )
        g["stl_geometry_inference_error"] = f"{type(exc).__name__}: {exc}"
        return g


# ============================================================
# Surface classification
# ============================================================

def is_emitting_surface(owner_name: str) -> bool:
    """
    Return True if this terminal surface is allowed to produce cascade emission.

    This is a modeling choice. For now we allow emission from all physical
    conducting/solid surfaces that have yield models or can be mapped to one.
    """
    owner = canonical_surface_name(owner_name)

    emitting = {
        "sample",
        "holder",
        "receiver",
        "rod",
        "drifttube",

        "g1_shell",
        "g2_shell",
        "g3_shell",
        "g1mesh",
        "g2mesh",
        "g3mesh",

        "g1frame",
        "g2frame",
        "g3frame",

        "collector",
        "collector_shell",
    }

    return owner in emitting


def cascade_surface_name(owner_name: str) -> str:
    """
    Convert terminal owner name to a name accepted by samplers.py.
    """
    owner = canonical_surface_name(owner_name)

    mapping = {
        "collector_shell": "collector",

        "g1_shell": "g1mesh",
        "g2_shell": "g2mesh",
        "g3_shell": "g3mesh",

        "g1mesh": "g1mesh",
        "g2mesh": "g2mesh",
        "g3mesh": "g3mesh",

        "g1frame": "g1frame",
        "g2frame": "g2frame",
        "g3frame": "g3frame",
    }

    return mapping.get(owner, owner)


# ============================================================
# Surface-normal estimation
# ============================================================

def estimate_surface_normal(
    owner_name: str,
    r_hit,
    v_in,
    hit_info: dict | None = None,
    sample_normal=None,
) -> np.ndarray:
    """
    Estimate local outward normal for secondary emission.

    For analytic spherical shells:
        grid shells: radial normal
        collector shell: inward radial normal, because the vacuum side is inside

    For sample:
        use the normal carried by hit_info, or an explicitly supplied rotated
        sample normal.  +X is no longer assumed for rotated geometry.

    For STL/fixed solids without a reliable face normal:
        use -v_in direction, i.e. emit back into the incident half-space.
        This is a safe fallback until we pass exact STL face normals through
        every hit result.
    """
    owner = canonical_surface_name(owner_name)

    r_hit = np.asarray(r_hit, dtype=float)
    v_in = np.asarray(v_in, dtype=float)

    if hit_info is not None:
        n = hit_info.get("normal", None)
        if n is not None:
            n = unit(n)

            # Orient normal against incoming velocity so that -v dot n > 0.
            if np.dot(v_in, n) > 0:
                n = -n

            return n

    if owner == "sample":
        if sample_normal is not None:
            return unit(np.asarray(sample_normal, dtype=float))

        # Last-resort fallback only.  Correct sample impacts produced by the
        # rotated-plane tracker carry an exact hit normal in hit_info above.
        return -unit(v_in)

    if owner in ["g1_shell", "g2_shell", "g3_shell", "g1mesh", "g2mesh", "g3mesh"]:
        return unit(r_hit)

    if owner in ["collector", "collector_shell"]:
        # Vacuum side of collector is inside the spherical collector.
        return -unit(r_hit)

    # Fallback: emit back into the side the incident electron came from.
    return -unit(v_in)


# ============================================================
# Safe launch helper
# ============================================================

def make_emissions_safe_to_launch(
    emissions: list[dict],
    r_hit,
    field: dict,
    launch_step_fraction_of_h: float = 0.10,
    max_advance_tries: int = 60,
    Phi_interp=None,
    surface_name: str | None = None,
    n_vacuum=None,
    fallback_vacuum_point=None,
) -> tuple[list[dict], list[dict]]:
    """
    Place newly emitted electrons in a free-vacuum voxel.

    Solid-surface emissions are displaced along the vacuum-side surface
    normal until a free voxel is reached. Analytic grid-wire emissions use a
    tiny displacement along their own sampled direction so forward and
    backward emission are both preserved. Analytic collector emissions use a
    tiny displacement along the inward vacuum normal. Grid/collector fixed
    voxels are field boundary conditions, not collision geometry.

    Returns
    -------
    safe, failed : tuple[list[dict], list[dict]]
        Successfully placed emissions and explicit failed-launch records.
        Failed launches are never passed to the trajectory integrator and are
        therefore not misclassified as physical escapes.
    """
    from .fields import evaluate_potential
    from .constants import speed_from_energy_eV

    safe: list[dict] = []
    failed: list[dict] = []

    r_hit = np.asarray(r_hit, dtype=float)
    surface = canonical_surface_name(surface_name)
    analytic_grid_surfaces = {
        "g1_shell", "g2_shell", "g3_shell",
        "g1mesh", "g2mesh", "g3mesh",
    }
    analytic_collector_surfaces = {"collector", "collector_shell"}

    if n_vacuum is None:
        raise ValueError(
            f"n_vacuum is required for emitted-particle placement "
            f"(surface={surface_name!r})"
        )

    # Keep the normal search local even if an older notebook passes the former
    # transmission-style value (for example 0.75 h).
    normal_step_fraction = min(float(launch_step_fraction_of_h), 0.10)
    if normal_step_fraction <= 0.0:
        raise ValueError("launch_step_fraction_of_h must be positive")

    for emission in emissions:
        e = dict(emission)

        v0 = np.asarray(e["v0"], dtype=float)
        direction = unit(v0)

        if surface in analytic_grid_surfaces:
            # A wire is represented by a zero-thickness analytic sphere
            # during tracking. Preserve the sampled forward/backward side and
            # do not demand that the launch point clear the field-solver's
            # fixed grid-shell voxels.
            launch_eps = max(1.0e-9, 1.0e-3 * float(field["h"]))
            p_safe = r_hit + launch_eps * direction
            cls = dict(classify_effective_vacuum_point(p_safe, field))
            cls.update({
                "placement_method": "analytic_grid_along_emission_direction",
                "placement_attempts": 1,
                "placement_offset_m": launch_eps,
            })
            success = cls["status"] == "free"

        elif surface in analytic_collector_surfaces:
            # The collector's vacuum side is radially inward. Its fixed shell
            # also exists only to impose the electrostatic boundary.
            launch_eps = max(1.0e-9, 1.0e-3 * float(field["h"]))
            p_safe = r_hit + launch_eps * unit(n_vacuum)
            cls = dict(classify_effective_vacuum_point(p_safe, field))
            cls.update({
                "placement_method": "analytic_collector_vacuum_normal",
                "placement_attempts": 1,
                "placement_offset_m": launch_eps,
            })
            success = cls["status"] == "free"

        else:
            # For sample, holder, receiver, rod, drift tube, and frames, the
            # fixed/STL geometry is physical. Move along the known vacuum-side
            # normal; using a tangential sampled velocity can remain trapped
            # in the solid for the full search distance.
            p_safe, cls, success = place_emitted_particle_in_vacuum(
                r_hit=r_hit,
                n_vacuum=n_vacuum,
                field=field,
                max_tries=max_advance_tries,
                step_fraction_of_h=normal_step_fraction,
                fallback_vacuum_point=fallback_vacuum_point,
            )
            cls = dict(cls)
            cls.setdefault("placement_method", "solid_vacuum_normal_search")

        e["raw_hit_location"] = r_hit.copy()
        e["launch_offset_m"] = float(np.linalg.norm(p_safe - r_hit))
        e["launch_grid_classification"] = cls
        e["launch_placement_method"] = cls.get("placement_method", None)

        if not success:
            e["launch_failed"] = True
            e["launch_failure_reason"] = cls.get("status", "no_free_voxel")
            failed.append(e)
            continue

        e["p0"] = p_safe
        e["launch_failed"] = False

        if Phi_interp is not None:
            Phi_emit = e.get("Phi_emit", np.nan)
            E_surface = e.get("E_emit_eV", np.nan)

            if np.isfinite(Phi_emit) and np.isfinite(E_surface):
                Phi_launch = float(evaluate_potential(p_safe, Phi_interp))

                if np.isfinite(Phi_launch):
                    # Electron energy conservation in eV:
                    # K_launch - Phi_launch = K_surface - Phi_emit
                    # K_launch = K_surface + Phi_launch - Phi_emit
                    dE = Phi_launch - Phi_emit
                    E_launch = float(E_surface) + dE

                    e["Phi_launch"] = Phi_launch
                    e["phi_launch_correction_eV"] = dE
                    e["E_launch_eV_unclipped"] = E_launch

                    if E_launch <= 0.0:
                        e["launch_failed"] = True
                        e["launch_failure_reason"] = (
                            "insufficient_energy_to_launch_point"
                        )
                        failed.append(e)
                        continue

                    e["E_launch_eV"] = E_launch
                    e["v0"] = direction * speed_from_energy_eV(E_launch)

        safe.append(e)

    return safe, failed

def generate_cascade_emissions_from_hit(
    surface_name: str,
    r_hit,
    v_in,
    Einc_eV: float,
    field: dict,
    yield_models: dict,
    energy_models: dict,
    theta_models: dict,
    voltages: dict,
    rng,
    origin: str,
    hit_info: dict | None = None,
    launch_step_fraction_of_h: float = 0.10,
    Phi_interp=None,
    sample_geometry: dict | None = None,
    track_sub_barrier_sample_emissions: bool = False,
    visualization_rng=None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Generate cascade emissions and place them safely in free vacuum.

    Returns successful emissions, explicit failed-launch records, and
    per-impact emission diagnostics (including sampled wire incidence data).
    """
    surface_name = cascade_surface_name(surface_name)

    n_out = estimate_surface_normal(
        surface_name,
        r_hit=r_hit,
        v_in=v_in,
        hit_info=hit_info,
        sample_normal=(
            None if sample_geometry is None else sample_geometry.get("normal", None)
        ),
    )

    # Use very small normal offset in generate_surface_emissions; the safe
    # launch point is found by make_emissions_safe_to_launch() using the
    # surface normal (for absorbing surfaces) or velocity (for grid shells).
    emissions, emission_event_info = generate_surface_emissions(
        surface_name=surface_name,
        r_hit=r_hit,
        v_in=v_in,
        n_out=n_out,
        Einc=Einc_eV,
        yield_models=yield_models,
        energy_models=energy_models,
        theta_models=theta_models,
        voltages=voltages,
        rng=rng,
        origin=origin,
        sample_launch_eps=1.0e-6,
        U0=15.0,
        Phi_interp=Phi_interp,
        track_sub_barrier_sample_emissions=track_sub_barrier_sample_emissions,
        visualization_rng=visualization_rng,
    )

    safe_emissions, failed_emissions = make_emissions_safe_to_launch(
        emissions,
        r_hit=r_hit,
        field=field,
        launch_step_fraction_of_h=launch_step_fraction_of_h,
        Phi_interp=Phi_interp,
        surface_name=surface_name,
        n_vacuum=n_out,   # n_out points from the surface into vacuum
        fallback_vacuum_point=(
            None if hit_info is None else hit_info.get("p_before", None)
        ),
    )

    return safe_emissions, failed_emissions, emission_event_info


# ============================================================
# Cascade runner for one primary
# ============================================================

def run_one_primary_with_cascade(
    p_primary,
    v_primary,
    field,
    Ex_interp,
    Ey_interp,
    Ez_interp,
    Phi_interp,

    intersector_primary,
    face_owner_primary,
    collision_mesh_primary,
    stl_boxes_primary,

    intersector_emit,
    face_owner_emit,
    collision_mesh_emit,
    stl_boxes_emit,

    grid_transparency,

    yield_models,
    energy_models,
    theta_models,
    voltages,
    rng,
    
    sample_y_bounds,
    sample_z_bounds,
    
    max_generation: int = 5,
    max_total_electrons: int = 500,
    min_incident_energy_eV: float = 0.1,

    emitted_max_steps: int = 20000,
    emitted_dt_max: float = 5.0e-11,
    emitted_max_step_fraction_of_h: float = 0.75,

    launch_step_fraction_of_h: float = 0.10,
    surface_skip_eps: float = 1.0e-6,
    integrator: str = "verlet",
    sample_geometry: dict | None = None,
    track_points: bool = False,
    track_stride: int = 1,
    track_primary_only: bool = False,
    track_this_primary: bool = False,
    track_sub_barrier_sample_emissions: bool = False,
    visualization_rng=None,
):
    """
    Run one primary electron with full cascade emission.

    Returns
    -------
    primary_result:
        Primary trajectory result.

    cascade_results:
        List of trajectory results for all emitted/cascade electrons.

    cascade_log:
        Lightweight list of generated-emission records.
    """

    track_primary_points = bool(track_points and track_this_primary)

    primary_result = fly_primary_to_sample(
        p0=p_primary,
        v0=v_primary,
        field=field,
        Ex_interp=Ex_interp,
        Ey_interp=Ey_interp,
        Ez_interp=Ez_interp,
        Phi_interp=Phi_interp,
        intersector=intersector_primary,
        face_owner=face_owner_primary,
        collision_mesh=collision_mesh_primary,
        stl_boxes=stl_boxes_primary,
        sample_y_bounds=sample_y_bounds,
        sample_z_bounds=sample_z_bounds,
        sample_geometry=sample_geometry,
        adaptive_dt=True,
        dt_min=1.0e-13,
        dt_max=2.0e-11,
        max_step_fraction_of_h=0.10,
        track_points=track_primary_points,
        track_stride=track_stride,
    )

    cascade_results = []
    cascade_log = []

    if primary_result["reason"] != "hit_sample":
        return primary_result, cascade_results, cascade_log

    hit = primary_result["hit_info"]

    p_hit = hit["location"]
    v_in = hit["v_in"]
    E_inc_eV = hit["KE_hit_eV"]

    queue = deque()
    next_electron_id = 0
    # The existing max_total_electrons cap and accounting apply only to the
    # historical physics cascade. Visualization-only sub-barrier electrons do
    # not consume that budget.
    physics_results_count = 0
    physics_generated_count = 0

    # First-generation sample emission.
    (
        first_emissions,
        first_launch_failures,
        first_emission_event_info,
    ) = generate_cascade_emissions_from_hit(
        surface_name="sample",
        r_hit=p_hit,
        v_in=v_in,
        Einc_eV=E_inc_eV,
        field=field,
        yield_models=yield_models,
        energy_models=energy_models,
        theta_models=theta_models,
        voltages=voltages,
        rng=rng,
        origin="gun",
        hit_info=hit,
        launch_step_fraction_of_h=launch_step_fraction_of_h,
        Phi_interp=Phi_interp,
        sample_geometry=sample_geometry,
        track_sub_barrier_sample_emissions=track_sub_barrier_sample_emissions,
        visualization_rng=visualization_rng,
    )

    # Record the primary sample-emission decision itself, including the Cu
    # angular-yield and quantum-reflection diagnostics.  This makes grazing-
    # incidence validation possible even when no emitted electron survives the
    # safe-launch placement step.
    cascade_log.append({
        "event": "first_generation_emissions_sampled",
        "electron_id": None,
        "parent_id": -1,
        "generation": 1,
        "source_owner": "sample",
        "source_electrode": "sample",
        "source_Einc_eV": E_inc_eV,
        "N_child_emissions": len(first_emissions),
        "N_launch_failures": len(first_launch_failures),
        "sample_incidence_theta_deg": first_emission_event_info.get(
            "sample_incidence_theta_deg", np.nan
        ),
        "sample_sey_gain_used": first_emission_event_info.get(
            "sample_sey_gain_used", np.nan
        ),
        "sample_bsey_gain_used": first_emission_event_info.get(
            "sample_bsey_gain_used", np.nan
        ),
        "sample_sey_mean_used": first_emission_event_info.get(
            "sample_sey_mean_used", np.nan
        ),
        "sample_bsey_mean_used": first_emission_event_info.get(
            "sample_bsey_mean_used", np.nan
        ),
        "sample_angular_yield_model": first_emission_event_info.get(
            "sample_angular_yield_model", None
        ),
        "sample_bse_multiplicity_sampled": first_emission_event_info.get(
            "sample_bse_multiplicity_sampled", np.nan
        ),
        "sample_quantum_reflection_mode": first_emission_event_info.get(
            "sample_quantum_reflection_mode", None
        ),
        "sample_quantum_reflection_probability": first_emission_event_info.get(
            "sample_quantum_reflection_probability", np.nan
        ),
        "sample_quantum_reflection_applied": first_emission_event_info.get(
            "sample_quantum_reflection_applied", False
        ),
        "sample_gun_incidence_sampler_used": first_emission_event_info.get(
            "sample_gun_incidence_sampler_used", False
        ),
        "sample_gun_incidence_sampler_angle_deg": first_emission_event_info.get(
            "sample_gun_incidence_sampler_angle_deg", np.nan
        ),
        "sample_gun_incidence_sampler_source": first_emission_event_info.get(
            "sample_gun_incidence_sampler_source", None
        ),
        "sample_emission_polar_axis": first_emission_event_info.get(
            "sample_emission_polar_axis", None
        ),
        "sample_gun_joint_sampler_used": first_emission_event_info.get(
            "sample_gun_joint_sampler_used", False
        ),
        "sample_gun_joint_sampler_source": first_emission_event_info.get(
            "sample_gun_joint_sampler_source", None
        ),
        "sample_gun_azimuth_model": first_emission_event_info.get(
            "sample_gun_azimuth_model", None
        ),
    })

    for failed in first_launch_failures:
        cascade_log.append({
            "event": "launch_failed",
            "electron_id": None,
            "parent_id": -1,
            "generation": 1,
            "source_owner": "sample",
            "source_electrode": "sample",
            "source_Einc_eV": E_inc_eV,
            "emission_kind": failed.get("kind", None),
            "E_emit_eV": failed.get("E_emit_eV", np.nan),
            "launch_offset_m": failed.get("launch_offset_m", np.nan),
            "sub_barrier": bool(failed.get("sub_barrier", False)),
            "escape_eligible": bool(failed.get("escape_eligible", True)),
            "visualization_only": bool(failed.get("visualization_only", False)),
            "launch_failure_reason": failed.get("launch_failure_reason", None),
            "launch_grid_status": failed.get(
                "launch_grid_classification", {}
            ).get("status", None),
            "launch_raw_grid_status": failed.get(
                "launch_grid_classification", {}
            ).get("raw_status", None),
            "launch_raw_owner_id": failed.get(
                "launch_grid_classification", {}
            ).get("raw_owner_id", None),
            "launch_raw_owner_name": failed.get(
                "launch_grid_classification", {}
            ).get("raw_owner_name", None),
            "launch_ignored_fixed_owner": failed.get(
                "launch_grid_classification", {}
            ).get("ignored_fixed_owner", None),
            "launch_placement_method": failed.get(
                "launch_placement_method", None
            ),
        })

    for e in first_emissions:
        record = {
            "electron_id": next_electron_id,
            "parent_id": -1,
            "generation": 1,
            "source_owner": "sample",
            "source_electrode": "sample",
            "source_Einc_eV": E_inc_eV,
            "emission": e,
        }

        queue.append(record)
        if not bool(e.get("visualization_only", False)):
            physics_generated_count += 1
        next_electron_id += 1

    # Main cascade loop.
    while queue:
        item = queue.popleft()

        e = item["emission"]
        visualization_only = bool(e.get("visualization_only", False))

        # Preserve the old physics-electron cap exactly while still allowing
        # visualization-only return trajectories to be drawn.
        if (not visualization_only) and physics_results_count >= max_total_electrons:
            continue

        track_emitted_points = bool(
            track_points
            and track_this_primary
            and (not track_primary_only)
        )

        res = integrate_one_electron(
            p0=e["p0"],
            v0=e["v0"],
            field=field,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,
            Phi_interp=Phi_interp,
            intersector=intersector_emit,
            face_owner=face_owner_emit,
            collision_mesh=collision_mesh_emit,
            integrator=integrator,
            dt=1.0e-12,
            max_steps=emitted_max_steps,
            surface_eps=surface_skip_eps,
            grid_transparency=grid_transparency,
            rng=(visualization_rng if visualization_only and visualization_rng is not None else rng),
            adaptive_dt=True,
            dt_min=1.0e-13,
            dt_max=emitted_dt_max,
            max_step_fraction_of_h=emitted_max_step_fraction_of_h,
            stl_boxes=stl_boxes_emit,
            sample_plane_return=True,
            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,
            min_sample_return_distance=5.0e-7,
            sample_geometry=sample_geometry,
            track_points=track_emitted_points,
            track_stride=track_stride,
        )

        # Attach cascade metadata.
        res["electron_id"] = item["electron_id"]
        res["parent_id"] = item["parent_id"]
        res["generation"] = item["generation"]
        res["source_owner"] = item["source_owner"]
        res["source_electrode"] = item["source_electrode"]
        res["source_Einc_eV"] = item["source_Einc_eV"]

        res["E_emit_eV"] = e.get("E_emit_eV", np.nan)
        res["emission_kind"] = e.get("kind", None)
        res["launch_offset_m"] = e.get("launch_offset_m", np.nan)
        res["Phi_emit"] = e.get("Phi_emit", np.nan)
        res["Phi_launch"] = e.get("Phi_launch", np.nan)
        res["phi_launch_correction_eV"] = e.get(
            "phi_launch_correction_eV", np.nan
        )
        res["E_launch_eV"] = e.get("E_launch_eV", e.get("E_emit_eV", np.nan))
        res["primary_E_inc_eV"] = E_inc_eV
        res["primary_cos_theta"] = e.get("cos_theta", np.nan)
        res["emission_theta_deg"] = e.get("emission_theta_deg", np.nan)
        res["emission_phi_deg"] = e.get("emission_phi_deg", np.nan)
        res["emission_polar_axis"] = e.get("emission_polar_axis", None)
        res["emission_outward_cosine"] = e.get(
            "emission_outward_cosine", np.nan
        )
        res["sample_gun_incidence_sampler_used"] = bool(
            e.get("sample_gun_incidence_sampler_used", False)
        )
        res["sample_gun_joint_sampler_used"] = bool(
            e.get("sample_gun_joint_sampler_used", False)
        )
        res["sample_gun_joint_sampler_source"] = e.get(
            "sample_gun_joint_sampler_source", None
        )
        res["joint_sampler_incident_energy_eV"] = e.get(
            "joint_sampler_incident_energy_eV", np.nan
        )
        res["joint_sampler_event_index"] = e.get(
            "joint_sampler_event_index", -1
        )
        res["joint_sampler_Eout_raw_eV"] = e.get(
            "joint_sampler_Eout_raw_eV", np.nan
        )
        res["joint_sampler_energy_clipped"] = bool(
            e.get("joint_sampler_energy_clipped", False)
        )
        res["emission_mu_beam_back"] = e.get(
            "emission_mu_beam_back", np.nan
        )
        res["emission_mu_toward_normal"] = e.get(
            "emission_mu_toward_normal", np.nan
        )
        res["emission_mu_side"] = e.get("emission_mu_side", np.nan)
        res["sub_barrier"] = bool(e.get("sub_barrier", False))
        res["escape_eligible"] = bool(e.get("escape_eligible", True))
        res["visualization_only"] = visualization_only

        cascade_results.append(res)
        if not visualization_only:
            physics_results_count += 1

        cascade_log.append({
            "electron_id": item["electron_id"],
            "parent_id": item["parent_id"],
            "generation": item["generation"],
            "source_owner": item["source_owner"],
            "source_electrode": item["source_electrode"],
            "source_Einc_eV": item["source_Einc_eV"],
            "emission_kind": e.get("kind", None),
            "E_emit_eV": e.get("E_emit_eV", np.nan),
            "launch_offset_m": e.get("launch_offset_m", np.nan),
            "emission_theta_deg": e.get("emission_theta_deg", np.nan),
            "emission_phi_deg": e.get("emission_phi_deg", np.nan),
            "emission_polar_axis": e.get("emission_polar_axis", None),
            "emission_outward_cosine": e.get(
                "emission_outward_cosine", np.nan
            ),
            "sample_gun_incidence_sampler_used": bool(
                e.get("sample_gun_incidence_sampler_used", False)
            ),
            "sample_gun_joint_sampler_used": bool(
                e.get("sample_gun_joint_sampler_used", False)
            ),
            "sample_gun_joint_sampler_source": e.get(
                "sample_gun_joint_sampler_source", None
            ),
            "joint_sampler_incident_energy_eV": e.get(
                "joint_sampler_incident_energy_eV", np.nan
            ),
            "joint_sampler_event_index": e.get(
                "joint_sampler_event_index", -1
            ),
            "joint_sampler_Eout_raw_eV": e.get(
                "joint_sampler_Eout_raw_eV", np.nan
            ),
            "joint_sampler_energy_clipped": bool(
                e.get("joint_sampler_energy_clipped", False)
            ),
            "emission_mu_beam_back": e.get(
                "emission_mu_beam_back", np.nan
            ),
            "emission_mu_toward_normal": e.get(
                "emission_mu_toward_normal", np.nan
            ),
            "emission_mu_side": e.get("emission_mu_side", np.nan),
            "sub_barrier": bool(e.get("sub_barrier", False)),
            "escape_eligible": bool(e.get("escape_eligible", True)),
            "visualization_only": visualization_only,
        })

        # Visualization-only sub-barrier electrons exist solely to show the
        # biased return path. They must not create children or alter the
        # historical cascade/current calculation when they re-hit the sample.
        if visualization_only:
            continue

        # Stop cascade if generation limit reached.
        if item["generation"] >= max_generation:
            continue

        hit_info = res.get("hit_info", None)

        if hit_info is None:
            continue

        owner_name_map = field.get("owner_name_map", None)

        terminal_owner = terminal_owner_from_result(
            res,
            owner_name_map=owner_name_map,
            field=field,
        )

        terminal_electrode = electrode_from_owner(terminal_owner)

        if not is_emitting_surface(terminal_owner):
            continue

        r_term = hit_info.get("location", None)
        v_term = hit_info.get("v_in", None)
        E_term = hit_info.get("KE_hit_eV", None)

        if r_term is None or v_term is None or E_term is None:
            continue

        E_term = float(E_term)

        if not np.isfinite(E_term) or E_term < min_incident_energy_eV:
            continue

        n_term = estimate_surface_normal(
            terminal_owner,
            r_hit=r_term,
            v_in=v_term,
            hit_info=hit_info,
            sample_normal=(
                None if sample_geometry is None else sample_geometry.get("normal", None)
            ),
        )

        vhat_term = unit(v_term)

        terminal_cos_theta_raw = -float(np.dot(vhat_term, n_term))
        terminal_surface_canonical = cascade_surface_name(terminal_owner)

        terminal_sey_gain_used = np.nan
        terminal_bsey_gain_used = np.nan

        if terminal_surface_canonical in ANALYTIC_MESH_SURFACES:
            # The shell normal is only the mesh-plane normal. The actual yield
            # calculation samples a local cylindrical-wire normal inside
            # generate_surface_emissions(), so there is no single pre-sampling
            # incidence cosine or gain to report here.
            terminal_angle_model = "sampled_effective_wire_normal"
            terminal_cos_theta_used = np.nan
            terminal_angular_gain_used = np.nan
            terminal_theta_used_deg = np.nan
        elif terminal_surface_canonical == "sample":
            terminal_cos_theta_used = float(
                np.clip(terminal_cos_theta_raw, 0.0, 1.0)
            )
            (
                terminal_sey_gain_used,
                terminal_bsey_gain_used,
                terminal_theta_used_deg,
                sample_angle_source,
            ) = sample_angle_dependent_yield_gains(
                yield_models=yield_models,
                Einc=E_term,
                cos_theta=terminal_cos_theta_used,
            )
            terminal_angle_model = (
                "jmonsel_sample_angular_table"
                if sample_angle_source != "capped_secant"
                else "capped_secant_from_surface_normal"
            )
            # There is intentionally no single gain for the sample because SEY
            # and BSEY use distinct angular dependencies.
            terminal_angular_gain_used = np.nan
        else:
            terminal_angle_model = "capped_secant_from_surface_normal"
            terminal_cos_theta_used = max(
                float(np.clip(terminal_cos_theta_raw, 0.0, 1.0)),
                1.0 / float(MAX_ANGULAR_YIELD_GAIN),
            )
            terminal_angular_gain_used = angular_yield_gain(
                terminal_cos_theta_raw
            )
            terminal_sey_gain_used = terminal_angular_gain_used
            terminal_bsey_gain_used = terminal_angular_gain_used
            terminal_theta_used_deg = float(
                np.degrees(
                    np.arccos(
                        np.clip(terminal_cos_theta_used, 0.0, 1.0)
                    )
                )
            )

        terminal_theta_raw_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(terminal_cos_theta_raw, -1.0, 1.0)
                )
            )
        )

        cascade_log.append({
            "electron_id": item["electron_id"],
            "parent_id": item["parent_id"],
            "generation": item["generation"],
            "event": "terminal_hit_for_possible_cascade",
            "terminal_owner": terminal_owner,
            "terminal_electrode": terminal_electrode,
            "terminal_Einc_eV": E_term,
            "is_emitting_surface": is_emitting_surface(terminal_owner),

            "terminal_cos_theta_raw": terminal_cos_theta_raw,
            "terminal_cos_theta_used": terminal_cos_theta_used,
            "terminal_theta_raw_deg": terminal_theta_raw_deg,
            "terminal_theta_used_deg": terminal_theta_used_deg,
            "terminal_angular_gain_used": terminal_angular_gain_used,
            "terminal_sey_gain_used": terminal_sey_gain_used,
            "terminal_bsey_gain_used": terminal_bsey_gain_used,
            "terminal_angle_model": terminal_angle_model,
        })

        (
            child_emissions,
            child_launch_failures,
            emission_event_info,
        ) = generate_cascade_emissions_from_hit(
            surface_name=terminal_owner,
            r_hit=r_term,
            v_in=v_term,
            Einc_eV=E_term,
            field=field,
            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
            rng=rng,
            origin=res.get("emission_kind", "cascade"),
            hit_info=hit_info,
            launch_step_fraction_of_h=launch_step_fraction_of_h,
            Phi_interp=Phi_interp,
            sample_geometry=sample_geometry,
            track_sub_barrier_sample_emissions=track_sub_barrier_sample_emissions,
            visualization_rng=visualization_rng,
        )

        cascade_log.append({
            "electron_id": item["electron_id"],
            "parent_id": item["parent_id"],
            "generation": item["generation"],
            "event": "child_emissions_sampled",
            "terminal_owner": terminal_owner,
            "terminal_electrode": terminal_electrode,
            "terminal_Einc_eV": E_term,
            "N_child_emissions": len(child_emissions),

            "terminal_cos_theta_raw": terminal_cos_theta_raw,
            "terminal_cos_theta_used": terminal_cos_theta_used,
            "terminal_theta_raw_deg": terminal_theta_raw_deg,
            "terminal_theta_used_deg": terminal_theta_used_deg,
            "terminal_angular_gain_used": terminal_angular_gain_used,
            "terminal_sey_gain_used": terminal_sey_gain_used,
            "terminal_bsey_gain_used": terminal_bsey_gain_used,
            "terminal_angle_model": terminal_angle_model,

            # Actual event-by-event wire geometry sampled inside samplers.py.
            # These remain NaN for non-grid surfaces. They are logged here,
            # rather than only on emitted electrons, so zero-emission grid hits
            # are represented too.
            "sampled_wire_cos_theta": emission_event_info.get(
                "sampled_wire_cos_theta", np.nan
            ),
            "sampled_wire_angular_gain": emission_event_info.get(
                "sampled_wire_angular_gain", np.nan
            ),
            "wire_sey_gain_used": emission_event_info.get(
                "wire_sey_gain_used", np.nan
            ),
            "wire_bsey_gain_used": emission_event_info.get(
                "wire_bsey_gain_used", np.nan
            ),
            "sample_incidence_theta_deg": emission_event_info.get(
                "sample_incidence_theta_deg", np.nan
            ),
            "sample_sey_gain_used": emission_event_info.get(
                "sample_sey_gain_used", np.nan
            ),
            "sample_bsey_gain_used": emission_event_info.get(
                "sample_bsey_gain_used", np.nan
            ),
            "sample_sey_mean_used": emission_event_info.get(
                "sample_sey_mean_used", np.nan
            ),
            "sample_bsey_mean_used": emission_event_info.get(
                "sample_bsey_mean_used", np.nan
            ),
            "sample_angular_yield_model": emission_event_info.get(
                "sample_angular_yield_model", None
            ),
            "sample_bse_multiplicity_sampled": emission_event_info.get(
                "sample_bse_multiplicity_sampled", np.nan
            ),
            "sample_quantum_reflection_mode": emission_event_info.get(
                "sample_quantum_reflection_mode", None
            ),
            "sample_quantum_reflection_probability": emission_event_info.get(
                "sample_quantum_reflection_probability", np.nan
            ),
            "sample_quantum_reflection_applied": emission_event_info.get(
                "sample_quantum_reflection_applied", False
            ),
            "sample_gun_incidence_sampler_used": emission_event_info.get(
                "sample_gun_incidence_sampler_used", False
            ),
            "sample_gun_incidence_sampler_angle_deg": emission_event_info.get(
                "sample_gun_incidence_sampler_angle_deg", np.nan
            ),
            "sample_gun_incidence_sampler_source": emission_event_info.get(
                "sample_gun_incidence_sampler_source", None
            ),
            "sample_emission_polar_axis": emission_event_info.get(
                "sample_emission_polar_axis", None
            ),
            "sample_gun_joint_sampler_used": emission_event_info.get(
                "sample_gun_joint_sampler_used", False
            ),
            "sample_gun_joint_sampler_source": emission_event_info.get(
                "sample_gun_joint_sampler_source", None
            ),
            "sample_gun_azimuth_model": emission_event_info.get(
                "sample_gun_azimuth_model", None
            ),
        })

        for failed in child_launch_failures:
            cascade_log.append({
                "event": "launch_failed",
                "electron_id": None,
                "parent_id": item["electron_id"],
                "generation": item["generation"] + 1,
                "source_owner": terminal_owner,
                "source_electrode": terminal_electrode,
                "source_Einc_eV": E_term,
                "emission_kind": failed.get("kind", None),
                "E_emit_eV": failed.get("E_emit_eV", np.nan),
                "launch_offset_m": failed.get("launch_offset_m", np.nan),
                "sub_barrier": bool(failed.get("sub_barrier", False)),
                "escape_eligible": bool(failed.get("escape_eligible", True)),
                "visualization_only": bool(failed.get("visualization_only", False)),
                "launch_failure_reason": failed.get(
                    "launch_failure_reason", None
                ),
                "launch_grid_status": failed.get(
                    "launch_grid_classification", {}
                ).get("status", None),
                "launch_raw_grid_status": failed.get(
                    "launch_grid_classification", {}
                ).get("raw_status", None),
                "launch_raw_owner_id": failed.get(
                    "launch_grid_classification", {}
                ).get("raw_owner_id", None),
                "launch_raw_owner_name": failed.get(
                    "launch_grid_classification", {}
                ).get("raw_owner_name", None),
                "launch_ignored_fixed_owner": failed.get(
                    "launch_grid_classification", {}
                ).get("ignored_fixed_owner", None),
                "launch_placement_method": failed.get(
                    "launch_placement_method", None
                ),
            })

        for child in child_emissions:
            child_visual_only = bool(child.get("visualization_only", False))
            if (not child_visual_only) and physics_generated_count >= max_total_electrons:
                continue

            queue.append({
                "electron_id": next_electron_id,
                "parent_id": item["electron_id"],
                "generation": item["generation"] + 1,
                "source_owner": terminal_owner,
                "source_electrode": terminal_electrode,
                "source_Einc_eV": E_term,
                "emission": child,
            })

            if not child_visual_only:
                physics_generated_count += 1
            next_electron_id += 1

    return primary_result, cascade_results, cascade_log


def cascade_results_to_dataframe(
    cascade_results: list[dict],
    owner_name_map: dict | None = None,
    field: dict | None = None,
):
    """
    Convert cascade trajectory results into a compact dataframe.
    """
    import pandas as pd

    rows = []

    for res in cascade_results:
        hit_info = res.get("hit_info", None)

        terminal_owner = terminal_owner_from_result(
            res,
            owner_name_map=owner_name_map,
            field=field,
        )
        terminal_electrode = electrode_from_owner(terminal_owner)

        if hit_info is None:
            owner_id = None
            KE_hit_eV = np.nan
            location = np.array([np.nan, np.nan, np.nan])
        else:
            owner_id = hit_info.get("owner_id", None)
            KE_hit_eV = hit_info.get("KE_hit_eV", np.nan)
            location = hit_info.get("location", None)

            if location is None:
                traj = res.get("traj", None)
                if traj is not None and len(traj) > 0:
                    location = np.asarray(traj)[-1]
                else:
                    location = np.array([np.nan, np.nan, np.nan])

        location = np.asarray(location, dtype=float)

        rows.append({
            "electron_id": res.get("electron_id", None),
            "parent_id": res.get("parent_id", None),
            "generation": res.get("generation", None),

            "source_owner": res.get("source_owner", None),
            "source_electrode": res.get("source_electrode", None),
            "source_Einc_eV": res.get("source_Einc_eV", np.nan),

            "emission_kind": res.get("emission_kind", None),
            "E_emit_eV": res.get("E_emit_eV", np.nan),
            "launch_offset_m": res.get("launch_offset_m", np.nan),
            "Phi_emit": res.get("Phi_emit", np.nan),
            "Phi_launch": res.get("Phi_launch", np.nan),
            "phi_launch_correction_eV": res.get(
                "phi_launch_correction_eV", np.nan
            ),
            "E_launch_eV": res.get("E_launch_eV", np.nan),
            "emission_theta_deg": res.get("emission_theta_deg", np.nan),
            "emission_phi_deg": res.get("emission_phi_deg", np.nan),
            "emission_polar_axis": res.get("emission_polar_axis", None),
            "emission_outward_cosine": res.get(
                "emission_outward_cosine", np.nan
            ),
            "sample_gun_incidence_sampler_used": bool(
                res.get("sample_gun_incidence_sampler_used", False)
            ),
            "sample_gun_joint_sampler_used": bool(
                res.get("sample_gun_joint_sampler_used", False)
            ),
            "sample_gun_joint_sampler_source": res.get(
                "sample_gun_joint_sampler_source", None
            ),
            "joint_sampler_incident_energy_eV": res.get(
                "joint_sampler_incident_energy_eV", np.nan
            ),
            "joint_sampler_event_index": res.get(
                "joint_sampler_event_index", -1
            ),
            "joint_sampler_Eout_raw_eV": res.get(
                "joint_sampler_Eout_raw_eV", np.nan
            ),
            "joint_sampler_energy_clipped": bool(
                res.get("joint_sampler_energy_clipped", False)
            ),
            "emission_mu_beam_back": res.get(
                "emission_mu_beam_back", np.nan
            ),
            "emission_mu_toward_normal": res.get(
                "emission_mu_toward_normal", np.nan
            ),
            "emission_mu_side": res.get("emission_mu_side", np.nan),
            "sub_barrier": bool(res.get("sub_barrier", False)),
            "escape_eligible": bool(res.get("escape_eligible", True)),
            "visualization_only": bool(res.get("visualization_only", False)),

            "reason": res.get("reason", None),
            "terminal_owner": terminal_owner,
            "terminal_electrode": terminal_electrode,
            "owner_id": owner_id,
            "KE_hit_eV": KE_hit_eV,
            "steps": res.get("steps", np.nan),

            "x_hit": location[0],
            "y_hit": location[1],
            "z_hit": location[2],
            "primary_index": res.get("primary_index", None),
        })

    return pd.DataFrame(rows)


def cascade_log_to_dataframe(cascade_log: list[dict]):
    """
    Convert cascade generation log into dataframe.
    """
    import pandas as pd

    return pd.DataFrame(cascade_log)


# ============================================================
# Cascade batch runners
# ============================================================

def _run_cascade_chunk(
    chunk_index: int,
    p0s_chunk,
    v0s_chunk,
    primary_index_offset: int,
    seed: int,

    field,
    Phi_interp,
    Ex_interp,
    Ey_interp,
    Ez_interp,

    intersector_primary,
    face_owner_primary,
    collision_mesh_primary,
    stl_boxes_primary,

    intersector_emit,
    face_owner_emit,
    collision_mesh_emit,
    stl_boxes_emit,

    grid_transparency,

    yield_models,
    energy_models,
    theta_models,
    voltages,
    
    sample_y_bounds,
    sample_z_bounds,
    
    max_generation,
    max_total_electrons_per_primary,
    min_incident_energy_eV,
    
    emitted_max_steps,
    emitted_dt_max,
    emitted_max_step_fraction_of_h,
    
    launch_step_fraction_of_h,
    
    integrator: str = "verlet",
    sample_geometry: dict | None = None,

    track_points: bool = False,
    track_stride: int = 1,
    track_primary_only: bool = False,
    tracked_primary_indices=None,
    track_sub_barrier_sample_emissions: bool = False,
):
    """
    Worker function for one cascade chunk.
    """
    rng = np.random.default_rng(seed)

    primary_results = []
    cascade_results_all = []
    cascade_logs_all = []

    for i in range(len(p0s_chunk)):
        primary_index = primary_index_offset + i

        # Dedicated visualization RNG. Enabling sub-barrier trajectories must
        # not perturb the main physics RNG stream.
        visualization_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(primary_index), 0x5EED5E])
        )

        track_this_primary = (
            track_points
            and (
                tracked_primary_indices is None
                or primary_index in tracked_primary_indices
            )
        )

        primary_res_i, cas_i, log_i = run_one_primary_with_cascade(
            p_primary=p0s_chunk[i],
            v_primary=v0s_chunk[i],
            field=field,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,
            Phi_interp=Phi_interp,

            intersector_primary=intersector_primary,
            face_owner_primary=face_owner_primary,
            collision_mesh_primary=collision_mesh_primary,
            stl_boxes_primary=stl_boxes_primary,

            intersector_emit=intersector_emit,
            face_owner_emit=face_owner_emit,
            collision_mesh_emit=collision_mesh_emit,
            stl_boxes_emit=stl_boxes_emit,

            grid_transparency=grid_transparency,

            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
            rng=rng,

            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,

            max_generation=max_generation,
            max_total_electrons=max_total_electrons_per_primary,
            min_incident_energy_eV=min_incident_energy_eV,

            emitted_max_steps=emitted_max_steps,
            emitted_dt_max=emitted_dt_max,
            emitted_max_step_fraction_of_h=emitted_max_step_fraction_of_h,

            launch_step_fraction_of_h=launch_step_fraction_of_h,
            integrator=integrator,
            sample_geometry=sample_geometry,
            track_points=track_points,
            track_stride=track_stride,
            track_primary_only=track_primary_only,
            track_this_primary=track_this_primary,
            track_sub_barrier_sample_emissions=track_sub_barrier_sample_emissions,
            visualization_rng=visualization_rng,
        )

        primary_res_i["primary_index"] = primary_index

        for r in cas_i:
            r["primary_index"] = primary_index

        for row in log_i:
            row["primary_index"] = primary_index

        primary_results.append(primary_res_i)
        cascade_results_all.extend(cas_i)
        cascade_logs_all.extend(log_i)

    return {
        "chunk_index": chunk_index,
        "primary_results": primary_results,
        "cascade_results_all": cascade_results_all,
        "cascade_logs_all": cascade_logs_all,
    }


def primary_results_to_dataframe(
    primary_results: list[dict],
    p0s=None,
    v0s=None,
    sample_geometry: dict | None = None,
) -> pd.DataFrame:
    """Convert primary results into a compact beam-steering dataframe.

    When launch arrays are supplied, the CSV also records launch position and
    velocity, impact velocity, direction-change angle, path displacement, and
    actual incidence angle on the continuous sample plane.  These diagnostics
    remain available even when point-by-point trajectory storage is disabled.
    """
    rows = []

    p0s_arr = None if p0s is None else np.asarray(p0s, dtype=float)
    v0s_arr = None if v0s is None else np.asarray(v0s, dtype=float)
    sample_normal = None
    if sample_geometry is not None:
        n = np.asarray(sample_geometry.get("normal", [np.nan] * 3), dtype=float)
        if n.shape == (3,) and np.all(np.isfinite(n)) and np.linalg.norm(n) > 0:
            sample_normal = n / np.linalg.norm(n)

    for i, res in enumerate(primary_results):
        hit_info = res.get("hit_info", None)
        primary_index = int(res.get("primary_index", i))

        p0 = np.full(3, np.nan)
        v0 = np.full(3, np.nan)
        if p0s_arr is not None and 0 <= primary_index < len(p0s_arr):
            p0 = np.asarray(p0s_arr[primary_index], dtype=float)
        if v0s_arr is not None and 0 <= primary_index < len(v0s_arr):
            v0 = np.asarray(v0s_arr[primary_index], dtype=float)

        if hit_info is None:
            KE_hit_eV = np.nan
            location = np.full(3, np.nan)
            v_hit = np.full(3, np.nan)
            reason = res.get("reason", None)
            kind = None
        else:
            KE_hit_eV = hit_info.get("KE_hit_eV", np.nan)
            location = hit_info.get("location", None)
            v_hit = hit_info.get("v_in", None)
            reason = res.get("reason", None)
            kind = hit_info.get("kind", None)

            if location is None:
                traj = res.get("traj", None)
                if traj is not None and len(traj) > 0:
                    location = np.asarray(traj)[-1]
                else:
                    location = np.full(3, np.nan)
            if v_hit is None:
                vel = res.get("vel", None)
                if vel is not None and len(vel) > 0:
                    v_hit = np.asarray(vel)[-1]
                else:
                    v_hit = np.full(3, np.nan)

        location = np.asarray(location, dtype=float)
        v_hit = np.asarray(v_hit, dtype=float)

        launch_to_hit_distance_m = np.nan
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(location)):
            launch_to_hit_distance_m = float(np.linalg.norm(location - p0))

        direction_change_deg = np.nan
        if (
            np.all(np.isfinite(v0))
            and np.all(np.isfinite(v_hit))
            and np.linalg.norm(v0) > 0
            and np.linalg.norm(v_hit) > 0
        ):
            c = float(np.dot(v0, v_hit) / (np.linalg.norm(v0) * np.linalg.norm(v_hit)))
            direction_change_deg = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

        incidence_angle_deg = np.nan
        incidence_cos = np.nan
        if (
            sample_normal is not None
            and np.all(np.isfinite(v_hit))
            and np.linalg.norm(v_hit) > 0
            and reason == "hit_sample"
        ):
            vhat = v_hit / np.linalg.norm(v_hit)
            incidence_cos = float(np.clip(-np.dot(vhat, sample_normal), -1.0, 1.0))
            incidence_angle_deg = float(np.degrees(np.arccos(incidence_cos)))

        rows.append({
            "primary_index": primary_index,
            "reason": reason,
            "kind": kind,
            "KE_hit_eV": KE_hit_eV,
            "steps": res.get("steps", np.nan),
            "x0": p0[0],
            "y0": p0[1],
            "z0": p0[2],
            "vx0": v0[0],
            "vy0": v0[1],
            "vz0": v0[2],
            "x_hit": location[0],
            "y_hit": location[1],
            "z_hit": location[2],
            "vx_hit": v_hit[0],
            "vy_hit": v_hit[1],
            "vz_hit": v_hit[2],
            "launch_to_hit_distance_m": launch_to_hit_distance_m,
            "direction_change_deg": direction_change_deg,
            "incidence_cos": incidence_cos,
            "incidence_angle_deg": incidence_angle_deg,
            "sample_voxel_artifact_traversal": (
                False if hit_info is None
                else bool(hit_info.get("sample_voxel_artifact_traversal", False))
            ),
            "sample_voxel_artifact_distance_m": (
                np.nan if hit_info is None
                else hit_info.get("sample_voxel_artifact_distance_m", np.nan)
            ),
        })

    return pd.DataFrame(rows)


def run_cascade_batch_parallel(
    N_primary: int,
    E0_eV: float,
    field: dict,
    Phi_interp,
    Ex_interp,
    Ey_interp,
    Ez_interp,

    intersector_primary,
    face_owner_primary,
    collision_mesh_primary,
    stl_boxes_primary,

    intersector_emit,
    face_owner_emit,
    collision_mesh_emit,
    stl_boxes_emit,

    grid_transparency: dict,

    yield_models: dict,
    energy_models: dict,
    theta_models: dict,
    voltages: dict,

    sample_y_bounds,
    sample_z_bounds,
    
    x_start: float | None = None,
    beam_sigma: float = 150e-6,
    energy_spread_eV: float = 0.0,
    angular_sigma_deg: float = 0.0,
    seed: int = 1,

    max_generation: int = 4,
    max_total_electrons_per_primary: int = 200,
    min_incident_energy_eV: float = 0.5,

    emitted_max_step_fraction_of_h: float = 0.75,
    emitted_dt_max: float = 5.0e-11,
    emitted_max_steps: int = 20000,

    launch_step_fraction_of_h: float = 0.10,
    integrator: str = "verlet",

    n_jobs: int = 4,
    chunk_size: int = 5,
    verbose: int = 10,
    sample_theta_deg: float | None = None,
    primary_launch_clearance_h: float = 2.0,
    primary_launch_distance_m: float | None = None,
    primary_launch_retreat_step_h: float = 0.25,
    primary_launch_max_tries: int = 80,
    track_points: bool = False,
    track_stride: int = 1,
    track_primary_only: bool = False,
    tracked_primary_indices: set[int] | None = None,
    track_sub_barrier_sample_emissions: bool = False,
):
    """
    Parallel cascade batch runner.

    Returns
    -------
    result:
        dict containing primary/cascade dataframes, logs, accounting, and runtime.
    """
    from joblib import Parallel, delayed
    from pathlib import Path
    import os

    if N_primary <= 0:
        raise ValueError("N_primary must be positive")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if track_stride <= 0:
        raise ValueError("track_stride must be positive")

    # Point storage is opt-in; per-primary selection is resolved in each worker.

    yield_model_sources = {}
    for family, family_models in yield_models.items():
        yield_model_sources[family] = {}
        for yield_kind, model in family_models.items():
            if isinstance(model, dict):
                yield_model_sources[family][yield_kind] = model.get(
                    "source", "legacy table"
                )
            else:
                yield_model_sources[family][yield_kind] = "unknown"

    emission_sampler_sources = {"energy": {}, "theta": {}}
    for family, family_models in energy_models.items():
        emission_sampler_sources["energy"][family] = {}
        for kind, model in family_models.items():
            if isinstance(model, dict):
                emission_sampler_sources["energy"][family][kind] = model.get(
                    "source", "legacy table"
                )
            else:
                emission_sampler_sources["energy"][family][kind] = str(model)

    for family, family_models in theta_models.items():
        emission_sampler_sources["theta"][family] = {}
        for kind, model in family_models.items():
            if isinstance(model, dict):
                emission_sampler_sources["theta"][family][kind] = model.get(
                    "source", "legacy table"
                )
            else:
                emission_sampler_sources["theta"][family][kind] = str(model)

    # Announce which curve every family actually runs on.
    #
    # A loader flag silently defaulting to False is invisible in the results:
    # a run made with the JMONSEL FromWire grid curves and one made with the
    # measured C-on-SS curves differ only in numbers that look plausible
    # either way. Printing provenance at the start means the run reports what
    # it used, instead of the operator having to recall which cell ran last.
    if verbose:
        print("[cascade] yield curves in use:")
        for family in sorted(yield_model_sources):
            for yield_kind in ("SEY", "BSEY"):
                if yield_kind not in yield_model_sources[family]:
                    continue
                model = yield_models.get(family, {}).get(yield_kind, {})
                geom = str(model.get("geometry", "-")) if isinstance(model, dict) else "?"
                print(
                    f"    {family:11}{yield_kind:6}{geom:8}"
                    f"{yield_model_sources[family][yield_kind]}"
                )

        print("[cascade] emitted-energy / angle samplers in use:")
        for family in sorted(set(energy_models) | set(theta_models)):
            for kind in ("SE", "BSE"):
                e_src = emission_sampler_sources["energy"].get(
                    family, {}
                ).get(kind, "<missing>")
                t_src = emission_sampler_sources["theta"].get(
                    family, {}
                ).get(kind, "<missing>")
                print(
                    f"    {family:11}{kind:4} energy={e_src}; theta={t_src}"
                )

    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    # Prefer an explicitly supplied angle.  If absent, allow the field setup
    # to carry it; otherwise normal incidence remains the backward-compatible
    # default.
    if sample_theta_deg is None:
        sample_theta_deg = float(field.get("sample_theta_deg", 0.0))

    # An incidence-specific sample model is an explicit one-angle experiment.
    # Validate it once at batch start so a 75-degree table can never be used
    # silently for a different configured sample rotation.  Small primary-ray
    # deflections by the field are handled event by event; this check concerns
    # the nominal mechanical sample angle.
    sample_gun_configs = {
        "yield": yield_models.get("sample", {}).get("gun_incidence", None),
        "energy": energy_models.get("sample", {}).get("gun_incidence", None),
        "theta": theta_models.get("sample", {}).get("gun_incidence", None),
    }
    configured = [cfg is not None for cfg in sample_gun_configs.values()]
    if any(configured):
        if not all(configured):
            raise ValueError(
                "sample gun-incidence model must be present in yield, energy, "
                "and theta model dictionaries"
            )
        sampler_angles = np.asarray([
            float(cfg["incidence_angle_deg"])
            for cfg in sample_gun_configs.values()
        ])
        sampler_tolerances = np.asarray([
            float(cfg["angle_tolerance_deg"])
            for cfg in sample_gun_configs.values()
        ])
        if not np.allclose(
            sampler_angles, sampler_angles[0], rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("sample gun-incidence model angles are inconsistent")
        if not np.allclose(
            sampler_tolerances, sampler_tolerances[0], rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "sample gun-incidence model angle tolerances are inconsistent"
            )

        sampler_angle = float(sampler_angles[0])
        sampler_tolerance = float(sampler_tolerances[0])
        angle_delta = abs(float(sample_theta_deg) - sampler_angle)
        if angle_delta > sampler_tolerance:
            raise ValueError(
                f"configured sample angle is {float(sample_theta_deg):g} deg, "
                f"but the gun-incidence sample model is for {sampler_angle:g} "
                f"deg (allowed difference {sampler_tolerance:g} deg)"
            )
        if verbose:
            theta_cfg = sample_gun_configs["theta"]
            yield_cfg = sample_gun_configs["yield"]
            energy_cfg = sample_gun_configs["energy"]
            print(
                "[cascade] sample gun-incidence model: "
                f"angle={sampler_angle:g} deg, "
                f"tolerance={sampler_tolerance:g} deg, "
                "polar axis=fixed +X beam-back/drift-tube axis, "
                f"azimuth={theta_cfg.get('azimuth_model', '?')}"
            )
            print(
                "    yields: "
                f"SE={yield_cfg['SEY'].get('source', '?')}; "
                f"BSE={yield_cfg['BSEY'].get('source', '?')}"
            )
            print(
                "    energy: "
                f"SE={energy_cfg['SE'].get('source', '?')}; "
                f"BSE={energy_cfg['BSE'].get('source', '?')}"
            )
            print(
                "    theta:  "
                f"SE={theta_cfg['SE'].get('source', '?')}; "
                f"BSE={theta_cfg['BSE'].get('source', '?')}"
            )
            joint_cfg = theta_cfg.get("joint", None)
            if joint_cfg is not None:
                print(
                    "    joint:  "
                    f"SE={joint_cfg['SE'].get('source', '?')} "
                    f"({joint_cfg['SE'].get('n_events', '?')} events); "
                    f"BSE={joint_cfg['BSE'].get('source', '?')} "
                    f"({joint_cfg['BSE'].get('n_events', '?')} events)"
                )
            else:
                print(
                    "    WARNING: no joint emitted-event sampler loaded; "
                    "oblique azimuth is still the legacy conditional-uniform model"
                )

    sample_x = float(field.get("sample_x", 0.0))
    explicit_face_center = field.get("sample_face_center", None)
    if explicit_face_center is None:
        sample_face_anchor = np.array([
            sample_x,
            0.0,
            0.5 * (float(sample_z_bounds[0]) + float(sample_z_bounds[1])),
        ], dtype=float)
    else:
        sample_face_anchor = np.asarray(explicit_face_center, dtype=float)

    sample_geometry = infer_sample_front_face_geometry_from_stl(
        collision_mesh=collision_mesh_primary,
        face_owner=face_owner_primary,
        theta_deg=float(sample_theta_deg),
        sample_y_bounds=sample_y_bounds,
        sample_z_bounds=sample_z_bounds,
        x_sample=sample_x,
        fallback_center=sample_face_anchor,
    )

    if all(configured):
        beam_back_axis = unit(np.asarray(
            sample_gun_configs["theta"]["beam_back_axis"], dtype=float
        ))
        sample_normal = unit(np.asarray(sample_geometry["normal"], dtype=float))
        geometry_angle = float(np.degrees(np.arccos(np.clip(
            float(np.dot(beam_back_axis, sample_normal)), -1.0, 1.0
        ))))
        if abs(geometry_angle - sampler_angle) > sampler_tolerance:
            raise ValueError(
                f"sample normal is {geometry_angle:g} deg from the configured "
                f"beam-back polar axis, but the incidence-specific sampler is "
                f"for {sampler_angle:g} deg"
            )

    # This is a coordinate-system invariant: STL placement can supply face
    # dimensions, never move the analytical face away from its configured
    # pivot.  Fail loudly if a future geometry change violates it.
    if not np.allclose(
        np.asarray(sample_geometry["center"], dtype=float),
        sample_face_anchor,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "analytic sample face moved away from configured sample_face_center"
        )

    # Beam transverse coordinates must be centred on the *front face*, not on
    # the midpoint of the rotated solid's axis-aligned Y bounds.
    y0 = float(np.asarray(sample_geometry["center"], dtype=float)[1])
    z0 = float(np.asarray(sample_geometry["center"], dtype=float)[2])

    if verbose:
        print(
            "[cascade] sample geometry: "
            f"theta={float(sample_theta_deg):.3f} deg, "
            f"normal={sample_geometry['normal']}, "
            f"center={sample_geometry['center']}, "
            f"source={sample_geometry.get('geometry_source', '?')}, "
            f"plane_offset={1e6 * sample_geometry.get('plane_offset_m', np.nan):.3f} um, "
            f"primary launch="
            + (
                f"{1e3 * float(primary_launch_distance_m):.3f} mm along ray"
                if primary_launch_distance_m is not None
                else f"{float(primary_launch_clearance_h):.2f} h normal-clearance"
            )
        )
        raw_stl_center = sample_geometry.get("stl_front_face_center_raw", None)
        if raw_stl_center is not None:
            print(
                "[cascade] raw sample-STL face center "
                f"{np.asarray(raw_stl_center, dtype=float)} was used for "
                "dimensions only; its assembly translation was ignored"
            )

    p0s, v0s, K0s, Phi0s = make_primary_beam_near_sample(
        N=N_primary,
        E0_eV=E0_eV,
        field=field,
        Phi_interp=Phi_interp,
        x_start=x_start,  # legacy-only; ignored when sample_geometry is supplied
        y0=y0,
        z0=z0,
        beam_sigma=beam_sigma,
        energy_spread_eV=energy_spread_eV,
        angular_sigma_deg=angular_sigma_deg,
        sample_voltage=voltages.get("Vs", 0.0),
        rng=rng,
        sample_geometry=sample_geometry,
        primary_launch_clearance_h=primary_launch_clearance_h,
        primary_launch_distance_m=primary_launch_distance_m,
        primary_launch_retreat_step_h=primary_launch_retreat_step_h,
        primary_launch_max_tries=primary_launch_max_tries,
    )

    chunks = []

    for start in range(0, N_primary, chunk_size):
        stop = min(start + chunk_size, N_primary)
        chunks.append((start, stop))

    chunk_seeds = seed + 5000 + np.arange(len(chunks))

    tmp_base = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    
    tmp_dir = Path(tmp_base) / "joblib_rfa_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        max_nbytes="10M",
        mmap_mode="r",
        temp_folder=str(tmp_dir),
        verbose=verbose,
    )(
        delayed(_run_cascade_chunk)(
            chunk_index=ic,
            p0s_chunk=p0s[start:stop],
            v0s_chunk=v0s[start:stop],
            primary_index_offset=start,
            seed=int(chunk_seeds[ic]),
    
            field=field,
            Phi_interp=Phi_interp,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,
    
            intersector_primary=intersector_primary,
            face_owner_primary=face_owner_primary,
            collision_mesh_primary=collision_mesh_primary,
            stl_boxes_primary=stl_boxes_primary,
    
            intersector_emit=intersector_emit,
            face_owner_emit=face_owner_emit,
            collision_mesh_emit=collision_mesh_emit,
            stl_boxes_emit=stl_boxes_emit,
    
            grid_transparency=grid_transparency,
    
            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
    
            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,
    
            max_generation=max_generation,
            max_total_electrons_per_primary=max_total_electrons_per_primary,
            min_incident_energy_eV=min_incident_energy_eV,
    
            emitted_max_steps=emitted_max_steps,
            emitted_dt_max=emitted_dt_max,
            emitted_max_step_fraction_of_h=emitted_max_step_fraction_of_h,
    
            launch_step_fraction_of_h=launch_step_fraction_of_h,
            integrator=integrator,
            sample_geometry=sample_geometry,

            track_points=track_points,
            track_stride=track_stride,
            track_primary_only=track_primary_only,
            tracked_primary_indices=tracked_primary_indices,
            track_sub_barrier_sample_emissions=track_sub_barrier_sample_emissions,
        )
        for ic, (start, stop) in enumerate(chunks)
    )

    chunk_results = sorted(chunk_results, key=lambda d: d["chunk_index"])

    primary_results = []
    cascade_results_all = []
    cascade_logs_all = []

    for cr in chunk_results:
        primary_results.extend(cr["primary_results"])
        cascade_results_all.extend(cr["cascade_results_all"])
        cascade_logs_all.extend(cr["cascade_logs_all"])

    owner_name_map = field.get("owner_name_map", None)

    df_primary = primary_results_to_dataframe(
        primary_results,
        p0s=p0s,
        v0s=v0s,
        sample_geometry=sample_geometry,
    )

    # Guard against a translated sample STL intercepting primaries before the
    # pivot-anchored analytical plane.  A few edge/bevel hits may legitimately
    # be off-plane, so use the median absolute residual rather than the maximum.
    # For a correctly aligned front-face population the median is effectively
    # zero (up to the integrator/collision tolerance).
    sample_hit_plane_residual_median_m = np.nan
    sample_hit_plane_residual_p95_m = np.nan
    if not df_primary.empty and "reason" in df_primary.columns:
        sample_hit_mask = df_primary["reason"].eq("hit_sample").to_numpy()
        if np.any(sample_hit_mask):
            hit_xyz = df_primary.loc[
                sample_hit_mask, ["x_hit", "y_hit", "z_hit"]
            ].to_numpy(dtype=float)
            finite_hit = np.all(np.isfinite(hit_xyz), axis=1)
            hit_xyz = hit_xyz[finite_hit]
            if len(hit_xyz):
                face_center = np.asarray(sample_geometry["center"], dtype=float)
                face_normal = unit(np.asarray(sample_geometry["normal"], dtype=float))
                plane_residual = np.abs((hit_xyz - face_center) @ face_normal)
                sample_hit_plane_residual_median_m = float(
                    np.median(plane_residual)
                )
                sample_hit_plane_residual_p95_m = float(
                    np.percentile(plane_residual, 95.0)
                )
                residual_tolerance_m = max(
                    5.0e-6,
                    0.25 * float(field["h"]),
                )
                if sample_hit_plane_residual_median_m > residual_tolerance_m:
                    raise RuntimeError(
                        "primary hit_sample population is displaced from the "
                        "configured analytic sample face: median normal "
                        f"residual={1e3 * sample_hit_plane_residual_median_m:.6f} "
                        f"mm, tolerance={1e3 * residual_tolerance_m:.6f} mm. "
                        "The translated sample STL is probably intercepting "
                        "the beam and must be excluded from primary collisions "
                        "or realigned to the sample pivot."
                    )

    if verbose and np.isfinite(sample_hit_plane_residual_median_m):
        print(
            "[cascade] primary sample-plane residual: "
            f"median={1e6 * sample_hit_plane_residual_median_m:.3f} um, "
            f"p95={1e6 * sample_hit_plane_residual_p95_m:.3f} um"
        )

    df_cascade = cascade_results_to_dataframe(
        cascade_results_all,
        owner_name_map=owner_name_map,
        field=field,
    )

    df_log = cascade_log_to_dataframe(cascade_logs_all)

    if not df_log.empty and "event" in df_log.columns:
        launch_failure_mask = df_log["event"].eq("launch_failed")
        if "visualization_only" in df_log.columns:
            launch_failure_mask &= ~df_log["visualization_only"].fillna(False).astype(bool)
        N_launch_failed = int(launch_failure_mask.sum())
        launch_failures_by_surface = (
            df_log.loc[launch_failure_mask, "source_electrode"]
            .value_counts(dropna=False)
            .to_dict()
        )
    else:
        N_launch_failed = 0
        launch_failures_by_surface = {}

    # Visualization-only sub-barrier electrons are deliberately absent from
    # the historical physics/current accounting. This keeps enabling the
    # presentation mode from changing TEY/BSEY/current results.
    cascade_results_physics = [
        r for r in cascade_results_all
        if not bool(r.get("visualization_only", False))
    ]

    acct = summarize_cascade_accounting(
        cascade_results=cascade_results_physics,
        N_primary=N_primary,
        owner_name_map=owner_name_map,
        field=field,
    )

    acct = add_per_primary_to_current_counts(
        acct,
        N_primary=N_primary,
    )

    N_visualization_only = int(sum(
        bool(r.get("visualization_only", False)) for r in cascade_results_all
    ))
    acct["summary"]["N_visualization_only"] = N_visualization_only

    # Reuse grid-event helper from first-generation accounting.
    # It expects list[list[dict]], so group cascade results by primary.
    cascade_by_primary = [[] for _ in range(N_primary)]

    for res in cascade_results_physics:
        pi = res.get("primary_index", None)
        if pi is not None and 0 <= int(pi) < N_primary:
            cascade_by_primary[int(pi)].append(res)

    df_grid_events = grid_events_to_dataframe_many(
        cascade_by_primary,
        owner_name_map=owner_name_map,
    )

    runtime_s = time.perf_counter() - t0

    result = {
        "primary_results": primary_results,
        "cascade_results_all": cascade_results_all,
        "cascade_logs_all": cascade_logs_all,

        "df_primary": df_primary,
        "df_cascade": df_cascade,
        "df_cascade_log": df_log,
        "df_grid_events": df_grid_events,

        "accounting": acct,
        "current_counts": acct["current_counts"],
        "summary": acct["summary"],

        "runtime_s": runtime_s,
        "runtime_per_primary_s": runtime_s / N_primary,
        "N_launch_failed": N_launch_failed,
        "launch_failures_by_surface": launch_failures_by_surface,
        "N_visualization_only": N_visualization_only,
        "sample_hit_plane_residual_median_m": (
            sample_hit_plane_residual_median_m
        ),
        "sample_hit_plane_residual_p95_m": sample_hit_plane_residual_p95_m,

        "p0s": p0s,
        "v0s": v0s,
        "K0s": K0s,
        "Phi0s": Phi0s,

        "N_primary": N_primary,
        "E0_eV": E0_eV,
        "integrator": integrator,

        "grid_transparency": grid_transparency,
        "yield_model_sources": yield_model_sources,
        "emission_sampler_sources": emission_sampler_sources,

        "max_generation": max_generation,
        "max_total_electrons_per_primary": max_total_electrons_per_primary,
        "min_incident_energy_eV": min_incident_energy_eV,

        "emitted_max_step_fraction_of_h": emitted_max_step_fraction_of_h,
        "emitted_dt_max": emitted_dt_max,
        "emitted_max_steps": emitted_max_steps,
        "launch_step_fraction_of_h": launch_step_fraction_of_h,
        "sample_theta_deg": float(sample_theta_deg),
        "sample_geometry": sample_geometry,
        "primary_launch_clearance_h": float(primary_launch_clearance_h),
        "primary_launch_distance_m": (
            None if primary_launch_distance_m is None
            else float(primary_launch_distance_m)
        ),
        "primary_launch_retreat_step_h": float(primary_launch_retreat_step_h),
        "primary_launch_max_tries": int(primary_launch_max_tries),

        "n_jobs": n_jobs,
        "chunk_size": chunk_size,

        "track_points": track_points,
        "track_stride": track_stride,
        "track_primary_only": track_primary_only,
        "track_sub_barrier_sample_emissions": bool(track_sub_barrier_sample_emissions),
        "tracked_primary_indices": (
            None if tracked_primary_indices is None
            else sorted(tracked_primary_indices)
        ),
    }

    return result


def print_cascade_batch_summary(result: dict):
    """
    Print compact cascade batch summary.
    """
    s = result["summary"]

    print("Cascade batch summary")
    print("---------------------")
    print(f"N primary:               {s['N_primary']}")
    print(f"N cascade electrons:     {s['N_cascade_electrons']}")
    print(f"N SE:                    {s['N_SE']}")
    print(f"N BSE:                   {s['N_BSE']}")
    print(f"Max generation:          {s['max_generation']}")
    print(f"Per primary electrons:   {s['per_primary_cascade_electrons']:.5f}")

    print(f"\nRuntime:                 {result['runtime_s']:.2f} s")
    print(f"Runtime per primary:     {result['runtime_per_primary_s']:.4f} s")
    print(f"Launch failures:         {result.get('N_launch_failed', 0)}")
    if result.get("N_visualization_only", 0):
        print(f"Visualization-only e-:  {result['N_visualization_only']}")
    if result.get("launch_failures_by_surface"):
        print(f"Failures by surface:     {result['launch_failures_by_surface']}")

    print("\nElectron-count balance:")
    print(result["current_counts"])

    df_cascade = result["df_cascade"]
    if not df_cascade.empty and "visualization_only" in df_cascade.columns:
        df_cascade_physics = df_cascade.loc[~df_cascade["visualization_only"].astype(bool)]
    else:
        df_cascade_physics = df_cascade

    if not df_cascade_physics.empty:
        print("\nTerminal electrodes (physics accounting):")
        print(df_cascade_physics["terminal_electrode"].value_counts())

        print("\nSource electrodes:")
        print(df_cascade_physics["source_electrode"].value_counts())

        print("\nGenerations:")
        print(df_cascade_physics["generation"].value_counts().sort_index())

    df_grid_events = result.get("df_grid_events", pd.DataFrame())

    if not df_grid_events.empty:
        print("\nGrid transmissions:")
        print(df_grid_events["electrode"].value_counts().reindex(
            ["grid1", "grid2", "grid3"],
            fill_value=0,
        ))

    print("\nCharge balance check:")
    print(result["current_counts"]["net_count"].sum())


def save_tracked_trajectories_npz(result: dict, path):
    """Save variable-length tracked trajectories in a compact pickle-free NPZ.

    Only records whose ``traj`` is not None are written.  Points and velocities
    are concatenated into flat arrays with offsets, so loading does not require
    ``allow_pickle=True``.  Metadata is stored alongside the trajectories.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    primary_results = result.get("primary_results", [])
    cascade_results = result.get("cascade_results_all", [])
    df_cascade = result.get("df_cascade", pd.DataFrame())

    def text(value):
        return "" if value is None else str(value)

    def pack(records, kind):
        points = []
        velocities = []
        offsets = [0]
        kept_indices = []
        valid_records = []

        for idx, res in enumerate(records):
            traj = res.get("traj", None)
            if traj is None:
                continue
            traj = np.asarray(traj, dtype=float)
            if traj.ndim != 2 or traj.shape[1] != 3 or len(traj) == 0:
                continue

            vel = res.get("vel", None)
            if vel is None:
                vel = np.full_like(traj, np.nan)
            else:
                vel = np.asarray(vel, dtype=float)
                if vel.shape != traj.shape:
                    vel_fixed = np.full_like(traj, np.nan)
                    ncopy = min(len(vel_fixed), len(vel)) if vel.ndim == 2 else 0
                    if ncopy and vel.shape[1] == 3:
                        vel_fixed[:ncopy] = vel[:ncopy]
                    vel = vel_fixed

            points.append(traj)
            velocities.append(vel)
            offsets.append(offsets[-1] + len(traj))
            kept_indices.append(idx)
            valid_records.append(res)

        points_arr = (
            np.concatenate(points, axis=0) if points
            else np.empty((0, 3), dtype=float)
        )
        vel_arr = (
            np.concatenate(velocities, axis=0) if velocities
            else np.empty((0, 3), dtype=float)
        )

        payload = {
            f"{kind}_points": points_arr,
            f"{kind}_velocities": vel_arr,
            f"{kind}_offsets": np.asarray(offsets, dtype=np.int64),
            f"{kind}_result_index": np.asarray(kept_indices, dtype=np.int64),
        }
        return payload, valid_records, kept_indices

    payload = {
        "format_version": np.asarray([2], dtype=np.int64),
        "track_stride": np.asarray([int(result.get("track_stride", 1))], dtype=np.int64),
        "sample_theta_deg": np.asarray([float(result.get("sample_theta_deg", np.nan))]),
        "E0_eV": np.asarray([float(result.get("E0_eV", np.nan))]),
    }

    p_payload, p_records, p_indices = pack(primary_results, "primary")
    c_payload, c_records, c_indices = pack(cascade_results, "cascade")
    payload.update(p_payload)
    payload.update(c_payload)

    payload.update({
        "primary_primary_index": np.asarray([
            int(r.get("primary_index", -1)) for r in p_records
        ], dtype=np.int64),
        "primary_reason": np.asarray([text(r.get("reason")) for r in p_records]),
        "primary_kind": np.asarray([
            text((r.get("hit_info") or {}).get("kind")) for r in p_records
        ]),
    })

    c_meta_rows = []
    for result_idx, res in zip(c_indices, c_records):
        if isinstance(df_cascade, pd.DataFrame) and result_idx < len(df_cascade):
            row = df_cascade.iloc[result_idx]
        else:
            row = None
        c_meta_rows.append((res, row))

    def meta_value(res, row, key, default=None):
        if key in res and res.get(key) is not None:
            return res.get(key)
        if row is not None and key in row.index and pd.notna(row[key]):
            return row[key]
        return default

    payload.update({
        "cascade_primary_index": np.asarray([
            int(meta_value(r, row, "primary_index", -1)) for r, row in c_meta_rows
        ], dtype=np.int64),
        "cascade_electron_id": np.asarray([
            int(meta_value(r, row, "electron_id", -1)) for r, row in c_meta_rows
        ], dtype=np.int64),
        "cascade_parent_id": np.asarray([
            int(meta_value(r, row, "parent_id", -1)) for r, row in c_meta_rows
        ], dtype=np.int64),
        "cascade_generation": np.asarray([
            int(meta_value(r, row, "generation", -1)) for r, row in c_meta_rows
        ], dtype=np.int64),
        "cascade_reason": np.asarray([
            text(meta_value(r, row, "reason")) for r, row in c_meta_rows
        ]),
        "cascade_source_owner": np.asarray([
            text(meta_value(r, row, "source_owner")) for r, row in c_meta_rows
        ]),
        "cascade_source_electrode": np.asarray([
            text(meta_value(r, row, "source_electrode")) for r, row in c_meta_rows
        ]),
        "cascade_terminal_owner": np.asarray([
            text(meta_value(r, row, "terminal_owner")) for r, row in c_meta_rows
        ]),
        "cascade_terminal_electrode": np.asarray([
            text(meta_value(r, row, "terminal_electrode")) for r, row in c_meta_rows
        ]),
        "cascade_emission_kind": np.asarray([
            text(meta_value(r, row, "emission_kind")) for r, row in c_meta_rows
        ]),
        "cascade_E_emit_eV": np.asarray([
            float(meta_value(r, row, "E_emit_eV", np.nan)) for r, row in c_meta_rows
        ], dtype=float),
        "cascade_E_launch_eV": np.asarray([
            float(meta_value(r, row, "E_launch_eV", np.nan)) for r, row in c_meta_rows
        ], dtype=float),
        "cascade_sub_barrier": np.asarray([
            bool(meta_value(r, row, "sub_barrier", False)) for r, row in c_meta_rows
        ], dtype=bool),
        "cascade_escape_eligible": np.asarray([
            bool(meta_value(r, row, "escape_eligible", True)) for r, row in c_meta_rows
        ], dtype=bool),
        "cascade_visualization_only": np.asarray([
            bool(meta_value(r, row, "visualization_only", False)) for r, row in c_meta_rows
        ], dtype=bool),
    })

    np.savez_compressed(path, **payload)
    return path


def save_cascade_batch_tables(
    result: dict,
    out_dir,
    prefix: str = "cascade",
    save_trajectories: bool = True,
):
    """Save cascade tables and, when present, tracked trajectories.

    CSVs remain compact tabular summaries.  Point-by-point trajectories are
    saved separately as ``<prefix>_trajectories.npz`` so they can be reloaded
    losslessly without embedding arrays inside CSV cells.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    paths["df_primary"] = out_dir / f"{prefix}_primary.csv"
    paths["df_cascade"] = out_dir / f"{prefix}_cascade.csv"
    paths["df_cascade_log"] = out_dir / f"{prefix}_cascade_log.csv"
    paths["df_grid_events"] = out_dir / f"{prefix}_grid_events.csv"
    paths["current_counts"] = out_dir / f"{prefix}_current_counts.csv"
    paths["summary"] = out_dir / f"{prefix}_summary.csv"

    result["df_primary"].to_csv(paths["df_primary"], index=False)
    result["df_cascade"].to_csv(paths["df_cascade"], index=False)
    result["df_cascade_log"].to_csv(paths["df_cascade_log"], index=False)
    result["df_grid_events"].to_csv(paths["df_grid_events"], index=False)
    result["current_counts"].to_csv(paths["current_counts"])

    pd.DataFrame([result["summary"]]).to_csv(paths["summary"], index=False)

    if save_trajectories:
        has_primary_tracks = any(
            r.get("traj", None) is not None
            for r in result.get("primary_results", [])
        )
        has_cascade_tracks = any(
            r.get("traj", None) is not None
            for r in result.get("cascade_results_all", [])
        )
        if has_primary_tracks or has_cascade_tracks:
            paths["trajectories"] = out_dir / f"{prefix}_trajectories.npz"
            save_tracked_trajectories_npz(result, paths["trajectories"])

    return paths
