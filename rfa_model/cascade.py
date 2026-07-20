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
from .trajectories import (
    unit,
    integrate_one_electron,
    advance_until_free,
    place_emitted_particle_in_vacuum,
    place_grid_emission_in_vacuum,
)
from .primary import (
    fly_primary_to_sample,
    make_primary_beam_near_sample,
    sample_center_from_bounds
)
from .samplers import (
    generate_surface_emissions,
    canonical_surface_name,
    surface_family,
)
from .accounting import (
    terminal_owner_from_result,
    electrode_from_owner,
    summarize_cascade_accounting,
    add_per_primary_to_current_counts,
    grid_events_to_dataframe_many
)




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
) -> np.ndarray:
    """
    Estimate local outward normal for secondary emission.

    For analytic spherical shells:
        grid shells: radial normal
        collector shell: inward radial normal, because the vacuum side is inside

    For sample:
        +X is the vacuum-side normal.

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
        return np.array([1.0, 0.0, 0.0])

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
    max_advance_tries: int = 20,
    Phi_interp=None,
    surface_name: str | None = None,
    n_vacuum=None,
) -> tuple[list[dict], list[dict]]:
    """
    Place newly emitted electrons in a free-vacuum voxel.

    Every particle passed to this function is a NEW surface emission, so its
    launch position is displaced along the vacuum-side surface normal.  The
    sampled emission velocity is preserved.  Velocity-directed stepping is
    reserved for already-travelling electrons transmitted through grid shells
    inside integrate_one_electron().

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

        owner = canonical_surface_name(surface_name) if surface_name is not None else None
        grid_surfaces = {
            "g1_shell", "g2_shell", "g3_shell",
            "g1mesh", "g2mesh", "g3mesh",
        }

        if owner in grid_surfaces:
            # Grid emission is special: the angular sampler has already
            # selected the physical emitted direction. Use that direction
            # to determine which side of the voxelized spherical shell the
            # electron belongs on, but do not rotate or reflect v0.
            p_safe, cls, success = place_grid_emission_in_vacuum(
                r_hit=r_hit,
                emission_direction=direction,
                field=field,
                max_forward_tries=max_advance_tries,
                max_radial_tries=max_advance_tries,
                step_fraction_of_h=normal_step_fraction,
            )
        else:
            # For ordinary solid surfaces, move only the launch position
            # along the known vacuum-side normal. Using v0 here can trap
            # nearly tangential emissions inside a thick voxelized surface
            # and was the source of collector/holder/etc. launch failures.
            p_safe, cls, success = place_emitted_particle_in_vacuum(
                r_hit=r_hit,
                n_vacuum=n_vacuum,
                field=field,
                max_tries=max_advance_tries,
                step_fraction_of_h=normal_step_fraction,
            )

        e["raw_hit_location"] = r_hit.copy()
        e["launch_offset_m"] = float(np.linalg.norm(p_safe - r_hit))
        e["launch_grid_classification"] = cls

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
    grid_SEY_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    Phi_interp=None,
) -> tuple[list[dict], list[dict]]:
    """
    Generate cascade emissions and place them safely in free vacuum.

    Returns successful emissions and explicit failed-launch records.
    """
    surface_name = cascade_surface_name(surface_name)

    n_out = estimate_surface_normal(
        surface_name,
        r_hit=r_hit,
        v_in=v_in,
        hit_info=hit_info,
    )

    # Use very small normal offset in generate_surface_emissions; the safe
    # launch point is found by make_emissions_safe_to_launch() using the
    # surface normal (for absorbing surfaces) or velocity (for grid shells).
    emissions = generate_surface_emissions(
        surface_name=surface_name,
        r_hit=r_hit,
        v_in=v_in,
        n_out=n_out,
        Einc=Einc_eV,
        yield_models=yield_models,
        energy_models=energy_models,
        theta_models=theta_models,
        voltages=voltages,
        grid_SEY_mult=grid_SEY_mult,
        collector_BSE_mult=collector_BSE_mult,
        rng=rng,
        origin=origin,
        sample_launch_eps=1.0e-6,
        U0=15.0,
        Phi_interp=Phi_interp,
    )

    # Save emitted-direction diagnostics before launch placement.
    v_in_hat = unit(v_in)
    r_hat = unit(r_hit)

    for e in emissions:
        v_emit_hat = unit(e["v0"])

        # Relative to incoming electron:
        # +1 = continues forward, -1 = reverses direction.
        forward_cosine = float(np.dot(v_emit_hat, v_in_hat))

        # Relative to spherical grid radius:
        # +1 = toward larger radius, -1 = toward smaller radius.
        radial_cosine = float(np.dot(v_emit_hat, r_hat))

        e["emission_forward_cosine"] = forward_cosine
        e["emission_radial_cosine"] = radial_cosine

        e["emission_scattering_angle_deg"] = float(
            np.degrees(
                np.arccos(np.clip(forward_cosine, -1.0, 1.0))
            )
        )

        if forward_cosine > 0.0:
            e["emission_forward_backward"] = "forward"
        elif forward_cosine < 0.0:
            e["emission_forward_backward"] = "backward"
        else:
            e["emission_forward_backward"] = "tangential"

        if radial_cosine > 0.0:
            e["emission_radial_direction"] = "outward"
        elif radial_cosine < 0.0:
            e["emission_radial_direction"] = "inward"
        else:
            e["emission_radial_direction"] = "tangential"

    safe_emissions, failed_emissions = make_emissions_safe_to_launch(
        emissions,
        r_hit=r_hit,
        field=field,
        launch_step_fraction_of_h=launch_step_fraction_of_h,
        Phi_interp=Phi_interp,
        surface_name=surface_name,
        n_vacuum=n_out,   # n_out points from the surface into vacuum
    )

    return safe_emissions, failed_emissions


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
    
    grid_SEY_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    grid_wire_geometry=None,

    max_generation: int = 5,
    max_total_electrons: int = 500,
    min_incident_energy_eV: float = 0.1,

    emitted_max_steps: int = 20000,
    emitted_dt_max: float = 5.0e-11,
    emitted_max_step_fraction_of_h: float = 0.75,

    launch_step_fraction_of_h: float = 0.10,
    surface_skip_eps: float = 1.0e-6,
    integrator: str = "verlet",
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
        adaptive_dt=True,
        dt_min=1.0e-13,
        dt_max=2.0e-11,
        max_step_fraction_of_h=0.10,
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

    # First-generation sample emission.
    first_emissions, first_launch_failures = generate_cascade_emissions_from_hit(
        surface_name="sample",
        r_hit=p_hit,
        v_in=v_in,
        Einc_eV=E_inc_eV,
        field=field,
        yield_models=yield_models,
        energy_models=energy_models,
        theta_models=theta_models,
        voltages=voltages,
        grid_SEY_mult=grid_SEY_mult,
        collector_BSE_mult=collector_BSE_mult,
        rng=rng,
        origin="gun",
        hit_info=hit,
        launch_step_fraction_of_h=launch_step_fraction_of_h,
        Phi_interp=Phi_interp,
    )

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
            "launch_failure_reason": failed.get("launch_failure_reason", None),
            "launch_grid_status": failed.get(
                "launch_grid_classification", {}
            ).get("status", None),
            "emission_forward_cosine": failed.get(
                "emission_forward_cosine", np.nan
            ),
            "emission_radial_cosine": failed.get(
                "emission_radial_cosine", np.nan
            ),
            "emission_scattering_angle_deg": failed.get(
                "emission_scattering_angle_deg", np.nan
            ),
            "emission_forward_backward": failed.get(
                "emission_forward_backward", None
            ),
            "emission_radial_direction": failed.get(
                "emission_radial_direction", None
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
        next_electron_id += 1

    # Main cascade loop.
    while queue:
        item = queue.popleft()

        if len(cascade_results) >= max_total_electrons:
            break

        e = item["emission"]

        res = integrate_one_electron(
            p0=e["p0"],
            v0=e["v0"],
            field=field,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,
            intersector=intersector_emit,
            face_owner=face_owner_emit,
            collision_mesh=collision_mesh_emit,
            integrator=integrator,
            dt=1.0e-12,
            max_steps=emitted_max_steps,
            surface_eps=surface_skip_eps,
            grid_transparency=grid_transparency,
            grid_wire_geometry=grid_wire_geometry,
            rng=rng,
            adaptive_dt=True,
            dt_min=1.0e-13,
            dt_max=emitted_dt_max,
            max_step_fraction_of_h=emitted_max_step_fraction_of_h,
            stl_boxes=stl_boxes_emit,
            sample_plane_return=True,
            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,
            min_sample_return_distance=5.0e-7,
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
        res["emission_forward_cosine"] = e.get(
            "emission_forward_cosine", np.nan
        )
        res["emission_radial_cosine"] = e.get(
            "emission_radial_cosine", np.nan
        )
        res["emission_scattering_angle_deg"] = e.get(
            "emission_scattering_angle_deg", np.nan
        )
        res["emission_forward_backward"] = e.get(
            "emission_forward_backward", None
        )
        res["emission_radial_direction"] = e.get(
            "emission_radial_direction", None
        )
        res["launch_offset_m"] = e.get("launch_offset_m", np.nan)
        res["Phi_emit"] = e.get("Phi_emit", np.nan)
        res["Phi_launch"] = e.get("Phi_launch", np.nan)
        res["phi_launch_correction_eV"] = e.get(
            "phi_launch_correction_eV", np.nan
        )
        res["E_launch_eV"] = e.get("E_launch_eV", e.get("E_emit_eV", np.nan))
        res["primary_E_inc_eV"] = E_inc_eV
        res["primary_cos_theta"] = e.get("cos_theta", np.nan)

        cascade_results.append(res)

        cascade_log.append({
            "event": "tracked_emission",
            "electron_id": item["electron_id"],
            "parent_id": item["parent_id"],
            "generation": item["generation"],
            "source_owner": item["source_owner"],
            "source_electrode": item["source_electrode"],
            "source_Einc_eV": item["source_Einc_eV"],
            "emission_kind": e.get("kind", None),
            "E_emit_eV": e.get("E_emit_eV", np.nan),
            "launch_offset_m": e.get("launch_offset_m", np.nan),
            "emission_forward_cosine": e.get(
                "emission_forward_cosine", np.nan
            ),
            "emission_radial_cosine": e.get(
                "emission_radial_cosine", np.nan
            ),
            "emission_scattering_angle_deg": e.get(
                "emission_scattering_angle_deg", np.nan
            ),
            "emission_forward_backward": e.get(
                "emission_forward_backward", None
            ),
            "emission_radial_direction": e.get(
                "emission_radial_direction", None
            ),
        })

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
        )

        vhat_term = unit(v_term)

        terminal_cos_theta_raw = -float(np.dot(vhat_term, n_term))
        terminal_cos_theta_used = max(terminal_cos_theta_raw, 0.05)

        terminal_theta_raw_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(terminal_cos_theta_raw, -1.0, 1.0)
                )
            )
        )

        terminal_theta_used_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(terminal_cos_theta_used, 0.0, 1.0)
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
        })

        child_emissions, child_launch_failures = generate_cascade_emissions_from_hit(
            surface_name=terminal_owner,
            r_hit=r_term,
            v_in=v_term,
            Einc_eV=E_term,
            field=field,
            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
            grid_SEY_mult=grid_SEY_mult,
            collector_BSE_mult=collector_BSE_mult,
            rng=rng,
            origin=res.get("emission_kind", "cascade"),
            hit_info=hit_info,
            launch_step_fraction_of_h=launch_step_fraction_of_h,
            Phi_interp=Phi_interp,
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
            "N_child_launch_failures": len(child_launch_failures),
            "N_child_sampled_total": (
                len(child_emissions) + len(child_launch_failures)
            ),

            "terminal_cos_theta_raw": terminal_cos_theta_raw,
            "terminal_cos_theta_used": terminal_cos_theta_used,
            "terminal_theta_raw_deg": terminal_theta_raw_deg,
            "terminal_theta_used_deg": terminal_theta_used_deg,
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
                "launch_failure_reason": failed.get(
                    "launch_failure_reason", None
                ),
                "launch_grid_status": failed.get(
                    "launch_grid_classification", {}
                ).get("status", None),
                "emission_forward_cosine": failed.get(
                    "emission_forward_cosine", np.nan
                ),
                "emission_radial_cosine": failed.get(
                    "emission_radial_cosine", np.nan
                ),
                "emission_scattering_angle_deg": failed.get(
                    "emission_scattering_angle_deg", np.nan
                ),
                "emission_forward_backward": failed.get(
                    "emission_forward_backward", None
                ),
                "emission_radial_direction": failed.get(
                    "emission_radial_direction", None
                ),
            })

        for child in child_emissions:
            if next_electron_id >= max_total_electrons:
                break

            queue.append({
                "electron_id": next_electron_id,
                "parent_id": item["electron_id"],
                "generation": item["generation"] + 1,
                "source_owner": terminal_owner,
                "source_electrode": terminal_electrode,
                "source_Einc_eV": E_term,
                "emission": child,
            })

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
            "emission_forward_cosine": res.get(
                "emission_forward_cosine", np.nan
            ),
            "emission_radial_cosine": res.get(
                "emission_radial_cosine", np.nan
            ),
            "emission_scattering_angle_deg": res.get(
                "emission_scattering_angle_deg", np.nan
            ),
            "emission_forward_backward": res.get(
                "emission_forward_backward", None
            ),
            "emission_radial_direction": res.get(
                "emission_radial_direction", None
            ),
            "Phi_emit": res.get("Phi_emit", np.nan),
            "Phi_launch": res.get("Phi_launch", np.nan),
            "phi_launch_correction_eV": res.get(
                "phi_launch_correction_eV", np.nan
            ),
            "E_launch_eV": res.get("E_launch_eV", np.nan),

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
    
    grid_SEY_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    grid_wire_geometry=None,
    integrator: str = "verlet",
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
            grid_SEY_mult=grid_SEY_mult,
            collector_BSE_mult=collector_BSE_mult,
            grid_wire_geometry=grid_wire_geometry,
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


def primary_results_to_dataframe(primary_results: list[dict]) -> pd.DataFrame:
    """
    Convert primary trajectory results into a compact dataframe.
    """
    rows = []

    for i, res in enumerate(primary_results):
        hit_info = res.get("hit_info", None)

        primary_index = res.get("primary_index", i)

        if hit_info is None:
            KE_hit_eV = np.nan
            location = np.array([np.nan, np.nan, np.nan])
            reason = res.get("reason", None)
            kind = None
        else:
            KE_hit_eV = hit_info.get("KE_hit_eV", np.nan)
            location = hit_info.get("location", None)
            reason = res.get("reason", None)
            kind = hit_info.get("kind", None)

            if location is None:
                traj = res.get("traj", None)
                if traj is not None and len(traj) > 0:
                    location = np.asarray(traj)[-1]
                else:
                    location = np.array([np.nan, np.nan, np.nan])

        location = np.asarray(location, dtype=float)

        rows.append({
            "primary_index": primary_index,
            "reason": reason,
            "kind": kind,
            "KE_hit_eV": KE_hit_eV,
            "steps": res.get("steps", np.nan),
            "x_hit": location[0],
            "y_hit": location[1],
            "z_hit": location[2],
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
    
    grid_SEY_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    grid_wire_geometry: dict | None = None,

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

    if grid_SEY_mult is None:
        grid_SEY_mult = 1.0

    if collector_BSE_mult is None:
        collector_BSE_mult = 1.0

    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    if x_start is None:
        x_start = 0.75 * float(field["h"])

    y0, z0 = sample_center_from_bounds(
        sample_y_bounds,
        sample_z_bounds,
    )

    p0s, v0s, K0s, Phi0s = make_primary_beam_near_sample(
        N=N_primary,
        E0_eV=E0_eV,
        field=field,
        Phi_interp=Phi_interp,
        x_start=x_start,
        y0=y0,
        z0=z0,
        beam_sigma=beam_sigma,
        energy_spread_eV=energy_spread_eV,
        angular_sigma_deg=angular_sigma_deg,
        sample_voltage=voltages.get("Vs", 0.0),
        rng=rng,
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
    
            grid_SEY_mult=grid_SEY_mult,
            collector_BSE_mult=collector_BSE_mult,
            grid_wire_geometry=grid_wire_geometry,
    
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

    df_primary = primary_results_to_dataframe(primary_results)

    df_cascade = cascade_results_to_dataframe(
        cascade_results_all,
        owner_name_map=owner_name_map,
        field=field,
    )

    df_log = cascade_log_to_dataframe(cascade_logs_all)

    if not df_log.empty and "event" in df_log.columns:
        launch_failure_mask = df_log["event"].eq("launch_failed")
        N_launch_failed = int(launch_failure_mask.sum())
        launch_failures_by_surface = (
            df_log.loc[launch_failure_mask, "source_electrode"]
            .value_counts(dropna=False)
            .to_dict()
        )
    else:
        N_launch_failed = 0
        launch_failures_by_surface = {}

    acct = summarize_cascade_accounting(
        cascade_results=cascade_results_all,
        N_primary=N_primary,
        owner_name_map=owner_name_map,
        field=field,
    )

    acct = add_per_primary_to_current_counts(
        acct,
        N_primary=N_primary,
    )

    # Reuse grid-event helper from first-generation accounting.
    # It expects list[list[dict]], so group cascade results by primary.
    cascade_by_primary = [[] for _ in range(N_primary)]

    for res in cascade_results_all:
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

        "p0s": p0s,
        "v0s": v0s,
        "K0s": K0s,
        "Phi0s": Phi0s,

        "N_primary": N_primary,
        "E0_eV": E0_eV,
        "integrator": integrator,

        "grid_transparency": grid_transparency,
        "grid_wire_geometry": grid_wire_geometry,
        "grid_SEY_mult": grid_SEY_mult,
        "collector_BSE_mult": collector_BSE_mult,

        "max_generation": max_generation,
        "max_total_electrons_per_primary": max_total_electrons_per_primary,
        "min_incident_energy_eV": min_incident_energy_eV,

        "emitted_max_step_fraction_of_h": emitted_max_step_fraction_of_h,
        "emitted_dt_max": emitted_dt_max,
        "emitted_max_steps": emitted_max_steps,
        "launch_step_fraction_of_h": launch_step_fraction_of_h,

        "n_jobs": n_jobs,
        "chunk_size": chunk_size,
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
    if result.get("launch_failures_by_surface"):
        print(f"Failures by surface:     {result['launch_failures_by_surface']}")

    print("\nElectron-count balance:")
    print(result["current_counts"])

    df_cascade = result["df_cascade"]

    if not df_cascade.empty:
        print("\nTerminal electrodes:")
        print(df_cascade["terminal_electrode"].value_counts())

        print("\nSource electrodes:")
        print(df_cascade["source_electrode"].value_counts())

        print("\nGenerations:")
        print(df_cascade["generation"].value_counts().sort_index())

    df_grid_events = result.get("df_grid_events", pd.DataFrame())

    if not df_grid_events.empty:
        print("\nGrid transmissions:")
        print(df_grid_events["electrode"].value_counts().reindex(
            ["grid1", "grid2", "grid3"],
            fill_value=0,
        ))

    print("\nCharge balance check:")
    print(result["current_counts"]["net_count"].sum())


def save_cascade_batch_tables(result: dict, out_dir, prefix: str = "cascade"):
    """
    Save cascade batch result tables to CSV.
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

    return paths