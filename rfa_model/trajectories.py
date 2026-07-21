"""
trajectories.py

Electron trajectory integration through the RFA field, including
STL collisions, analytic spherical grid/collector surfaces, grid
transparency, openings, adaptive timestep, and analytic sample-plane
return.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    e_charge,
    m_e,
    kinetic_energy_eV_from_velocity,
    speed_from_energy_eV,
)
from .fields import E_at_point, evaluate_potential
from .collisions import (
    first_segment_hit,
    first_analytic_grid_hit,
    segment_near_any_stl_box,
    segment_hits_sample_plane,
    classify_sphere_event,
    nearest_hit,
)


def _trajectory_failure(
    reason,
    p,
    v,
    traj,
    vel,
    step,
    hit_info=None,
    extra=None,
):
    if hit_info is None:
        hit_info = {}

    if extra is None:
        extra = {}

    out = {
        "reason": reason,
        "p_final": np.asarray(p, dtype=float),
        "v_final": np.asarray(v, dtype=float),
        "traj": np.asarray(traj, dtype=float),
        "vel": np.asarray(vel, dtype=float),
        "steps": step,
        "hit_info": hit_info,
    }

    out.update(extra)

    return out


def unit(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n == 0:
        raise ValueError("Cannot normalize zero vector.")

    return v / n


# ============================================================
# Grid classification helpers
# ============================================================

def point_to_grid_index(p, x, y, z):
    """
    Convert a Cartesian point to nearest grid index.
    """
    p = np.asarray(p, dtype=float)

    if p.shape[0] != 3 or not np.all(np.isfinite(p)):
        return -1, -1, -1, False

    hx = x[1] - x[0]
    hy = y[1] - y[0]
    hz = z[1] - z[0]

    i = int(np.round((p[0] - x[0]) / hx))
    j = int(np.round((p[1] - y[0]) / hy))
    k = int(np.round((p[2] - z[0]) / hz))

    inside = (
        0 <= i < len(x)
        and 0 <= j < len(y)
        and 0 <= k < len(z)
    )

    return i, j, k, inside


def classify_grid_point(p, field):
    """
    Classify nearest field-grid point as free/fixed/outside/update-region.
    """
    x = field["x"]
    y = field["y"]
    z = field["z"]

    i, j, k, inside = point_to_grid_index(p, x, y, z)

    if not inside:
        return {
            "status": "left_grid",
            "i": i,
            "j": j,
            "k": k,
            "owner_id": None,
        }

    fixed = field["fixed"][i, j, k]
    update = field["update_region"][i, j, k]
    owner_id = int(field["owner"][i, j, k])

    if fixed:
        return {
            "status": "hit_fixed",
            "i": i,
            "j": j,
            "k": k,
            "owner_id": owner_id,
        }

    if not update:
        return {
            "status": "left_update_region",
            "i": i,
            "j": j,
            "k": k,
            "owner_id": owner_id,
        }

    return {
        "status": "free",
        "i": i,
        "j": j,
        "k": k,
        "owner_id": owner_id,
    }




def is_drifttube_escape_candidate(p, v, field, aperture_radius=None):
    """Return True when a trajectory is leaving through the +X DT aperture.

    The test is intentionally conservative: the electron must be close to the
    positive-X edge of the field/update domain, moving toward +X, and inside
    the circular drift-tube aperture in the YZ plane.
    """
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)

    if p.shape != (3,) or v.shape != (3,):
        return False
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        return False
    if v[0] <= 0.0:
        return False

    h = float(field["h"])
    x_max = float(np.asarray(field["x"])[-1])

    if aperture_radius is None:
        aperture_radius = float(
            field.get("drifttube_aperture_radius", 6.0e-3)
        )

    radial_yz = float(np.hypot(p[1], p[2]))
    near_positive_x_boundary = p[0] >= x_max - 1.5 * h

    return near_positive_x_boundary and radial_yz <= aperture_radius


def drifttube_escape_result(p, v, traj, vel, step, grid_events=None, extra=None):
    """Build a clean terminal result for a physical DT-aperture escape."""
    hit_info = {
        "owner_name": "escaped",
        "owner": "escaped",
        "escape_opening": "drifttube",
        "kind": "drifttube_aperture_escape",
        "location": np.asarray(p, dtype=float).copy(),
        "KE_hit_eV": kinetic_energy_eV_from_velocity(v),
        "v_in": np.asarray(v, dtype=float).copy(),
    }
    payload = {"grid_events": [] if grid_events is None else grid_events}
    if extra:
        payload.update(extra)
    return _trajectory_failure(
        reason="escaped_drifttube",
        p=p,
        v=v,
        traj=traj,
        vel=vel,
        step=step,
        hit_info=hit_info,
        extra=payload,
    )

# ============================================================
# Adaptive timestep
# ============================================================

def distance_to_nearest_sphere_surface(p, field):
    """
    Distance from point to nearest analytic spherical surface.
    """
    r = np.linalg.norm(p)

    distances = [
        abs(r - field["R_g1"]),
        abs(r - field["R_g2"]),
        abs(r - field["R_g3"]),
        abs(r - field["R_col"]),
    ]

    return min(distances)


def choose_adaptive_dt_with_surfaces(
    p,
    v,
    field,
    dt_min: float = 1e-13,
    dt_max: float = 2e-11,
    max_step_fraction_of_h: float = 0.25,
    surface_safety: float = 0.25,
):
    """
    Choose timestep so that segment length remains small relative to:
        - grid spacing h
        - distance to nearest analytic spherical surface
    """
    speed = np.linalg.norm(v)

    if speed <= 0:
        return dt_min

    h = field["h"]

    max_step_grid = max_step_fraction_of_h * h

    d_surf = distance_to_nearest_sphere_surface(p, field)
    max_step_surface = surface_safety * d_surf
    max_step_surface = max(max_step_surface, 0.05 * h)

    max_step = min(max_step_grid, max_step_surface)

    dt = max_step / speed

    dt = min(dt, dt_max)
    dt = max(dt, dt_min)

    return dt


# ============================================================
# RK4
# ============================================================

def acceleration_at_point(
    p,
    Ex_interp,
    Ey_interp,
    Ez_interp,
):
    """
    Electron acceleration at position p in the electrostatic field.

    Equation:
        a = q E / m

    For an electron, q = -e.
    """
    E = E_at_point(
        p,
        Ex_interp,
        Ey_interp,
        Ez_interp,
    )

    return (-e_charge / m_e) * E


def velocity_verlet_step(
    p,
    v,
    dt,
    Ex_interp,
    Ey_interp,
    Ez_interp,
):
    """
    One velocity-Verlet step for an electron in a static electric field.

    System:
        dr/dt = v
        dv/dt = a(r)

    Update:
        r_new = r + v dt + 0.5 a_old dt^2
        v_new = v + 0.5 (a_old + a_new) dt
    """
    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)
    dt = float(dt)

    a_old = acceleration_at_point(
        p,
        Ex_interp,
        Ey_interp,
        Ez_interp,
    )

    p_new = p + v * dt + 0.5 * a_old * dt * dt

    a_new = acceleration_at_point(
        p_new,
        Ex_interp,
        Ey_interp,
        Ez_interp,
    )

    v_new = v + 0.5 * (a_old + a_new) * dt

    return p_new, v_new
    

def rk4_step(p, v, dt, Ex_interp, Ey_interp, Ez_interp):
    """
    One RK4 step for:
        dp/dt = v
        dv/dt = a(p)
    """

    def f(p_i, v_i):
        a_i = acceleration_at_point(
            p_i,
            Ex_interp,
            Ey_interp,
            Ez_interp,
        )
        return v_i, a_i

    k1_p, k1_v = f(p, v)

    k2_p, k2_v = f(
        p + 0.5 * dt * k1_p,
        v + 0.5 * dt * k1_v,
    )

    k3_p, k3_v = f(
        p + 0.5 * dt * k2_p,
        v + 0.5 * dt * k2_v,
    )

    k4_p, k4_v = f(
        p + dt * k3_p,
        v + dt * k3_v,
    )

    p_new = p + dt / 6.0 * (
        k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
    )

    v_new = v + dt / 6.0 * (
        k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
    )

    return p_new, v_new


# ============================================================
# Main emitted-electron integrator
# ============================================================

def integrate_one_electron(
    p0,
    v0,
    field,
    Ex_interp,
    Ey_interp,
    Ez_interp,
    Phi_interp,
    intersector,
    face_owner,
    collision_mesh,
    integrator: str = "verlet",
    dt: float = 1e-12,
    max_steps: int = 20000,
    surface_eps: float = 1e-7,
    grid_transparency=None,
    rng=None,
    adaptive_dt: bool = False,
    dt_min: float = 1e-13,
    dt_max: float = 2e-11,
    max_step_fraction_of_h: float = 0.25,
    stl_boxes=None,
    sample_plane_return: bool = False,
    sample_y_bounds=None,
    sample_z_bounds=None,
    min_sample_return_distance: float = 5e-7,
):
    """
    Integrate one electron trajectory through the RFA.

    Parameters
    ----------
    p0, v0:
        Initial position and velocity.
    field:
        RFA field dictionary.
    Ex_interp, Ey_interp, Ez_interp:
        Electric-field interpolators.
    intersector, face_owner, collision_mesh:
        STL ray-intersection data.
    grid_transparency:
        Dict such as {"g1_shell": 0.90, ...}.
    sample_plane_return:
        If True, hits of the analytic sample plane x=0 are terminal.

    Returns
    -------
    dict:
        Trajectory result with reason, hit_info, traj, vel, events.
    """
    if rng is None:
        rng = np.random.default_rng()

    if grid_transparency is None:
        grid_transparency = {
            "g1_shell": 0.93,
            "g2_shell": 0.93,
            "g3_shell": 0.93,
        }

    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    
    traj = [p.copy()]
    vel = [v.copy()]
    grid_events = []
    events = []
    
    ignore_sphere_owners = set()
    
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        return _trajectory_failure(
            reason="nan_state",
            p=p,
            v=v,
            traj=traj,
            vel=vel,
            step=0,
        )

    for step in range(max_steps):
        cls = classify_grid_point(p, field)

        if cls["status"] != "free":
            hit_info = dict(cls)

            if cls["status"] == "hit_fixed":
                owner_id = cls.get("owner_id", None)
                owner = fixed_owner_name(owner_id, field=field)

                hit_info["kind"] = "fixed_voxel"
                hit_info["owner"] = owner
                hit_info["location"] = p.copy()
                hit_info["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v)
                hit_info["v_in"] = v.copy()

                # ----------------------------------------------------
                # Special case: fixed spherical grid shell.
                #
                # These voxels represent the fixed-potential grid shell
                # in the field solution. They should not automatically
                # absorb electrons. Apply stochastic grid transparency.
                # ----------------------------------------------------
                if is_grid_shell_owner(owner):
                    if grid_transparency is None:
                        T = 1.0
                    else:
                        T = transparency_for_owner(grid_transparency, owner, default=1.0)

                    if rng is None:
                        rng_local = np.random.default_rng()
                    else:
                        rng_local = rng

                    if rng_local.random() > T:
                        # Absorbed by grid wire.
                        hit_info["kind"] = "grid_wire_fixed_voxel"

                        return {
                            "reason": "hit_grid_wire",
                            "hit_info": hit_info,
                            "traj": np.asarray(traj),
                            "vel": np.asarray(vel),
                            "steps": step,
                            "grid_events": grid_events,
                        }

                    # Transmitted through the grid shell.
                    grid_events.append({
                        "owner": owner,
                        "location": p.copy(),
                        "steps": step,
                        "type": "transmit_fixed_voxel",
                    })

                    # The current point is the first fixed-shell voxel reached
                    # from free vacuum.  Preserve that incoming-side information
                    # so the placement search exits on the physically transmitted
                    # side of the shell rather than jumping back to the side from
                    # which the electron arrived.
                    p_before_shell = (
                        np.asarray(traj[-2], dtype=float)
                        if len(traj) >= 2
                        else None
                    )

                    p_before_move = p.copy()
                    v_before_move = v.copy()

                    p, cls_after = place_transmitted_particle_beyond_grid(
                        p_shell=p_before_move,
                        v=v_before_move,
                        owner=owner,
                        field=field,
                        p_before=p_before_shell,
                        clearance_fraction_of_h=0.75,
                        verify_step_fraction_of_h=0.10,
                        max_verify_tries=20,
                    )

                    if cls_after["status"] == "free":
                        # Preserve electron total energy while moving it from the
                        # first fixed-shell voxel to the verified free point.
                        # In eV for an electron: K - Phi = constant.
                        if Phi_interp is not None:
                            phi_ref_point = (
                                p_before_shell
                                if p_before_shell is not None
                                else p_before_move
                            )
                            Phi_before = float(evaluate_potential(
                                phi_ref_point, Phi_interp
                            ))
                            Phi_after = float(evaluate_potential(p, Phi_interp))
                            K_before = float(
                                kinetic_energy_eV_from_velocity(v_before_move)
                            )
                            K_after = K_before + Phi_after - Phi_before

                            cls_after["Phi_before_V"] = Phi_before
                            cls_after["Phi_after_V"] = Phi_after
                            cls_after["K_before_eV"] = K_before
                            cls_after["K_after_eV"] = K_after
                            cls_after["energy_correction_eV"] = (
                                Phi_after - Phi_before
                            )

                            if not np.isfinite(K_after) or K_after <= 0.0:
                                cls_after["status"] = (
                                    "insufficient_energy_after_grid_placement"
                                )
                            else:
                                v = unit(v_before_move) * speed_from_energy_eV(
                                    K_after
                                )

                    traj.append(p.copy())
                    vel.append(v.copy())

                    # Save placement diagnostics in the transmission event.
                    grid_events[-1].update({
                        "placement_status": cls_after.get("status"),
                        "placement_method": cls_after.get("placement_method"),
                        "placement_attempts": cls_after.get("placement_attempts"),
                        "placement_offset_m": cls_after.get("placement_offset_m"),
                        "placement_target_radius_m": cls_after.get("placement_target_radius_m"),
                        "Phi_before_V": cls_after.get("Phi_before_V"),
                        "Phi_after_V": cls_after.get("Phi_after_V"),
                        "K_before_eV": cls_after.get("K_before_eV"),
                        "K_after_eV": cls_after.get("K_after_eV"),
                        "energy_correction_eV": cls_after.get("energy_correction_eV"),
                    })

                    if cls_after["status"] != "free":
                        return _trajectory_failure(
                            reason="grid_transmission_placement_failed",
                            p=p,
                            v=v,
                            traj=traj,
                            vel=vel,
                            step=step,
                            hit_info={
                                **cls_after,
                                "owner": owner,
                                "kind": "grid_transmission_placement_failed",
                            },
                            extra={"grid_events": grid_events},
                        )

                    # Avoid immediately detecting the same analytic spherical
                    # grid after leaving its fixed-potential voxel shell.
                    ignore_sphere_owners = {owner}

                    # Continue integration after stepping through shell.
                    continue

                # Ordinary fixed voxel: absorbing electrode.
                return {
                    "reason": "hit_fixed",
                    "hit_info": hit_info,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "steps": step,
                    "grid_events": grid_events,
                }

            if cls["status"] in {"left_grid", "left_update_region"}:
                if is_drifttube_escape_candidate(p, v, field):
                    return drifttube_escape_result(
                        p=p,
                        v=v,
                        traj=traj,
                        vel=vel,
                        step=step,
                        grid_events=grid_events,
                        extra={"original_grid_status": cls["status"]},
                    )

            return {
                "reason": cls["status"],
                "hit_info": hit_info,
                "traj": np.asarray(traj),
                "vel": np.asarray(vel),
                "steps": step,
                "grid_events": grid_events,
            }

        if adaptive_dt:
            dt_step = choose_adaptive_dt_with_surfaces(
                p,
                v,
                field,
                dt_min=dt_min,
                dt_max=dt_max,
                max_step_fraction_of_h=max_step_fraction_of_h,
            )
        else:
            dt_step = dt

        if integrator == "verlet":
            p_new, v_new = velocity_verlet_step(
                p, v, dt_step,
                Ex_interp, Ey_interp, Ez_interp,
            )
        
        elif integrator == "rk4":
            p_new, v_new = rk4_step(
                p, v, dt_step,
                Ex_interp, Ey_interp, Ez_interp,
            )
        
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        if not np.all(np.isfinite(p_new)) or not np.all(np.isfinite(v_new)):
            # Field interpolation commonly becomes nonfinite immediately after
            # a valid electron crosses the +X drift-tube aperture.  Classify
            # that physical boundary exit before calling it a numerical error.
            if is_drifttube_escape_candidate(p, v, field):
                return drifttube_escape_result(
                    p=p,
                    v=v,
                    traj=traj,
                    vel=vel,
                    step=step + 1,
                    grid_events=grid_events,
                    extra={
                        "p_before": p.copy(),
                        "v_before": v.copy(),
                        "dt_step": dt_step,
                        "nonfinite_after_boundary_step": True,
                    },
                )

            return _trajectory_failure(
                reason="nan_state_after_step",
                p=p_new,
                v=v_new,
                traj=traj,
                vel=vel,
                step=step + 1,
                hit_info={
                    "owner_name": "escaped",
                    "owner": "escaped",
                    "numerical_failure": True,
                },
                extra={
                    "p_before": p.copy(),
                    "v_before": v.copy(),
                    "dt_step": dt_step,
                },
            )

        # Analytic sample-plane return.
        hit_sample = None
        if sample_plane_return:
            if p[0] > 0.0 and p_new[0] <= 0.0:
                hit_sample = segment_hits_sample_plane(
                    p,
                    p_new,
                    x_sample=0.0,
                    sample_y_bounds=sample_y_bounds,
                    sample_z_bounds=sample_z_bounds,
                )

                if hit_sample is not None:
                    if np.linalg.norm(hit_sample["location"] - p0) < min_sample_return_distance:
                        hit_sample = None

        # STL hit only if broad-phase says segment is near an STL box.
        hit_stl = None
        if stl_boxes is None or segment_near_any_stl_box(p, p_new, stl_boxes):
            hit_stl = first_segment_hit(
                p,
                p_new,
                intersector,
                face_owner,
                collision_mesh,
            )

        # Analytic grid/collector hit.
        hit_grid = first_analytic_grid_hit(
            p,
            p_new,
            field,
            ignore_owners=ignore_sphere_owners,
        )

        hit = nearest_hit(hit_sample, hit_stl, hit_grid)

        if hit is not None:
            if hit["kind"] == "sample_plane":
                traj.append(hit["location"].copy())
                vel.append(v_new.copy())

                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "events": events,
                    "steps": step + 1,
                }

            if hit["kind"] == "stl":
                traj.append(hit["location"].copy())
                vel.append(v_new.copy())

                return {
                    "reason": "hit_stl",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "events": events,
                    "steps": step + 1,
                }

            if hit["kind"] == "sphere":
                event_type = classify_sphere_event(hit)
                owner = hit["owner"]

                if event_type == "hit_collector":
                    traj.append(hit["location"].copy())
                    vel.append(v_new.copy())

                    return {
                        "reason": "hit_collector",
                        "hit_info": hit,
                        "traj": np.asarray(traj),
                        "vel": np.asarray(vel),
                        "events": events,
                        "steps": step + 1,
                    }

                if event_type == "transmit_grid":
                    T = transparency_for_owner(grid_transparency, owner, default=1.0)

                    u = rng.random()

                    if u > T:
                        traj.append(hit["location"].copy())
                        vel.append(v_new.copy())

                        return {
                            "reason": "hit_grid_wire",
                            "hit_info": hit,
                            "traj": np.asarray(traj),
                            "vel": np.asarray(vel),
                            "events": events,
                            "steps": step + 1,
                        }

                    events.append({
                        "type": "transmit_grid",
                        "owner": owner,
                        "location": hit["location"],
                        "steps": step,
                        "u": u,
                        "T": T,
                    })

                    # Step just past the spherical surface to avoid
                    # detecting the same surface again.
                    p = hit["location"] + surface_eps * unit(v_new)
                    v = v_new.copy()

                    traj.append(p.copy())
                    vel.append(v.copy())

                    ignore_sphere_owners = {owner}
                    continue

        ignore_sphere_owners = set()

        p = p_new
        v = v_new

        traj.append(p.copy())
        vel.append(v.copy())

    return {
        "reason": "max_steps",
        "hit_info": None,
        "traj": np.asarray(traj),
        "vel": np.asarray(vel),
        "events": events,
        "steps": max_steps,
    }


def fixed_owner_name(owner_id, field=None):
    """
    Decode fixed-potential voxel owner ID.

    Prefer field["owner_name_map"] if available.
    """
    if owner_id is None:
        return None

    owner_id = int(owner_id)

    if field is not None and "owner_name_map" in field:
        return field["owner_name_map"].get(owner_id, f"owner_{owner_id}")

    return f"owner_{owner_id}"


def is_grid_shell_owner(owner_name: str) -> bool:
    """
    True for fixed-potential spherical grid-shell voxels.

    These should be treated with stochastic grid transparency,
    not as ordinary absorbing fixed solids.
    """
    return owner_name in ["g1_shell", "g2_shell", "g3_shell"]


def grid_shell_transparency_key(owner_name: str) -> str:
    """
    Convert fixed-owner grid shell name to transparency dictionary key.
    """
    mapping = {
        "g1_shell": "g1_shell",
        "g2_shell": "g2_shell",
        "g3_shell": "g3_shell",
    }

    return mapping[owner_name]


def grid_radius_for_owner(owner: str, field: dict) -> float:
    """Return the analytic spherical radius for a grid-shell owner."""
    mapping = {
        "g1_shell": "R_g1",
        "g2_shell": "R_g2",
        "g3_shell": "R_g3",
    }
    try:
        return float(field[mapping[owner]])
    except KeyError as exc:
        raise ValueError(f"Unknown analytic grid-shell owner: {owner}") from exc


def place_transmitted_particle_beyond_grid(
    p_shell,
    v,
    owner: str,
    field: dict,
    p_before=None,
    clearance_fraction_of_h: float = 0.75,
    verify_step_fraction_of_h: float = 0.10,
    max_verify_tries: int = 20,
):
    """Place a transmitted electron directly beyond an analytic grid shell.

    The outgoing radial side is inferred from the last free point before the
    fixed voxel layer.  The particle is placed at the known analytic grid
    radius plus a modest clearance on that same side.  A short radial search
    then verifies that the selected point belongs to a free field voxel.

    This avoids numerically walking through a staircase-like fixed voxel shell.
    Only the position is selected here; the caller applies potential-based
    kinetic-energy correction after the verified point is found.
    """
    p_shell = np.asarray(p_shell, dtype=float)
    vhat = unit(v)
    rhat = unit(p_shell)
    h = float(field["h"])
    R_grid = grid_radius_for_owner(owner, field)

    if clearance_fraction_of_h <= 0.0:
        raise ValueError("clearance_fraction_of_h must be positive")
    if verify_step_fraction_of_h <= 0.0:
        raise ValueError("verify_step_fraction_of_h must be positive")
    if max_verify_tries < 0:
        raise ValueError("max_verify_tries must be nonnegative")

    radial_component = float(np.dot(vhat, rhat))
    if p_before is not None:
        p_prev = np.asarray(p_before, dtype=float)
        if np.all(np.isfinite(p_prev)):
            dr_enter = float(np.linalg.norm(p_shell) - np.linalg.norm(p_prev))
        else:
            dr_enter = 0.0
    else:
        dr_enter = 0.0

    if abs(dr_enter) > 1.0e-12:
        radial_sign = 1.0 if dr_enter > 0.0 else -1.0
    else:
        radial_sign = 1.0 if radial_component >= 0.0 else -1.0

    clearance = float(clearance_fraction_of_h) * h
    verify_ds = float(verify_step_fraction_of_h) * h
    transmitted_radial = radial_sign * rhat

    target_radius = R_grid + radial_sign * clearance
    if target_radius <= 0.0:
        raise RuntimeError(
            f"Invalid target radius {target_radius} for {owner}"
        )

    candidate = target_radius * rhat
    cls = classify_grid_point(candidate, field)

    for attempt in range(max_verify_tries + 1):
        if cls["status"] == "free":
            enriched = dict(cls)
            enriched.update({
                "placement_method": "analytic_radius_then_radial_verify",
                "placement_attempts": attempt,
                "placement_offset_m": float(
                    np.linalg.norm(candidate - p_shell)
                ),
                "placement_radial_sign": radial_sign,
                "placement_grid_radius_m": R_grid,
                "placement_target_radius_m": float(np.linalg.norm(candidate)),
            })
            return candidate, enriched

        if cls["status"] in {"left_grid", "left_update_region"}:
            break

        candidate = candidate + verify_ds * transmitted_radial
        cls = classify_grid_point(candidate, field)

    failed = dict(cls)
    failed.update({
        "placement_method": "analytic_radius_then_radial_verify_failed",
        "placement_attempts": max_verify_tries,
        "placement_offset_m": float(np.linalg.norm(candidate - p_shell)),
        "placement_radial_sign": radial_sign,
        "placement_grid_radius_m": R_grid,
        "placement_target_radius_m": float(np.linalg.norm(candidate)),
    })
    return candidate, failed


def advance_until_free(
    p,
    v,
    field,
    p_before=None,
    max_tries: int = 160,
    step_fraction_of_h: float = 0.05,
):
    """Place a grid-transmitted electron beyond a fixed voxel shell.

    The search preserves the physical crossing direction.  It first follows
    the actual velocity and, if the trajectory is nearly tangent to the
    voxelized spherical shell, progressively biases the search toward the
    radial direction on the transmitted side.  It never searches toward the
    incident side of the shell.

    Parameters
    ----------
    p : array (3,)
        First point classified inside the fixed grid-shell voxel layer.
    v : array (3,)
        Electron velocity at the crossing.
    field : dict
        Field-grid data.
    p_before : array (3,), optional
        Last point in free vacuum before entering the fixed shell.  When
        supplied, its radius relative to ``p`` determines which radial side is
        the transmitted side more reliably than velocity alone.
    max_tries : int
        Number of samples per search direction.
    step_fraction_of_h : float
        Search increment as a fraction of the field-grid spacing.

    Returns
    -------
    p_safe, cls
        ``cls["status"]`` is ``"free"`` on success.  Diagnostic fields are
        added: ``placement_method``, ``placement_attempts``, and
        ``placement_offset_m``.
    """
    p0 = np.asarray(p, dtype=float).copy()
    vhat = unit(v)
    rhat = unit(p0)

    h = float(field["h"])
    ds = float(step_fraction_of_h) * h
    if ds <= 0.0:
        raise ValueError("step_fraction_of_h must be positive")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive")

    # Determine the radial side on which the transmitted electron must exit.
    # If the prior free point is available, moving from its radius to the
    # fixed-shell radius gives the crossing sense even for tangential motion.
    radial_component = float(np.dot(vhat, rhat))
    if p_before is not None:
        p_prev = np.asarray(p_before, dtype=float)
        if np.all(np.isfinite(p_prev)):
            dr_enter = float(np.linalg.norm(p0) - np.linalg.norm(p_prev))
            if abs(dr_enter) > 1.0e-12:
                radial_sign = 1.0 if dr_enter > 0.0 else -1.0
            else:
                radial_sign = 1.0 if radial_component >= 0.0 else -1.0
        else:
            radial_sign = 1.0 if radial_component >= 0.0 else -1.0
    else:
        radial_sign = 1.0 if radial_component >= 0.0 else -1.0

    transmitted_radial = radial_sign * rhat

    # Every candidate direction has a component toward the transmitted side.
    # The sequence starts with the true velocity and adds increasing radial
    # bias only as needed for a staircase-like voxel shell.
    direction_specs = [
        ("velocity", vhat),
        ("velocity_plus_0p10_radial", unit(vhat + 0.10 * transmitted_radial)),
        ("velocity_plus_0p25_radial", unit(vhat + 0.25 * transmitted_radial)),
        ("velocity_plus_0p50_radial", unit(vhat + 0.50 * transmitted_radial)),
        ("transmitted_radial", transmitted_radial),
    ]

    # Remove duplicate directions while preserving order.
    unique_specs = []
    for name, direction in direction_specs:
        direction = unit(direction)
        if not any(
            float(np.dot(direction, old_direction)) > 1.0 - 1.0e-10
            for _, old_direction in unique_specs
        ):
            unique_specs.append((name, direction))

    best = None
    last_p = p0.copy()
    last_cls = classify_grid_point(last_p, field)

    for method, direction in unique_specs:
        for attempt in range(1, max_tries + 1):
            candidate = p0 + attempt * ds * direction
            cls = classify_grid_point(candidate, field)
            last_p = candidate
            last_cls = cls

            if cls["status"] == "free":
                offset = float(np.linalg.norm(candidate - p0))
                enriched = dict(cls)
                enriched.update({
                    "placement_method": method,
                    "placement_attempts": attempt,
                    "placement_offset_m": offset,
                    "placement_radial_sign": radial_sign,
                })

                # Prefer the shortest valid placement among all directions.
                if best is None or offset < best[0]:
                    best = (offset, candidate.copy(), enriched)
                break

            # Once this ray leaves the modeled domain it cannot re-enter in a
            # physically useful way for this local shell crossing.
            if cls["status"] in {"left_grid", "left_update_region"}:
                break

    if best is not None:
        return best[1], best[2]

    failed = dict(last_cls)
    failed.update({
        "placement_method": "failed_all_transmitted_side_directions",
        "placement_attempts": max_tries,
        "placement_offset_m": float(np.linalg.norm(last_p - p0)),
        "placement_radial_sign": radial_sign,
    })
    return last_p, failed


def place_emitted_particle_in_vacuum(
    r_hit,
    n_vacuum,
    field,
    max_tries: int = 60,
    step_fraction_of_h: float = 0.10,
):
    """
    Find a free-vacuum launch point for a NEWLY EMITTED particle.

    Unlike advance_until_free(), this steps along the vacuum-side surface
    NORMAL rather than the sampled emission velocity. This is the
    geometrically correct approach for emission from curved or thick
    fixed-voxel surfaces (especially the collector shell), where a
    tangential emission direction can stay inside the shell for many steps.

    The sampled emission velocity is unchanged; only the starting position
    is displaced along the normal.

    Parameters
    ----------
    r_hit : array (3,)
        Exact surface hit location.
    n_vacuum : array (3,)
        Unit outward normal pointing toward vacuum from the surface.
        For collector_shell this is -r_hat (inward radial direction).
        For g*_shell this is +r_hat.
        For sample this is +X.
    field : dict
        Field dictionary containing at least "h" (voxel size).
    max_tries : int
        Maximum number of normal-directed steps.  The default searches up to
        6h, which is sufficient for the voxelized grid-shell layers observed
        in validation runs.
    step_fraction_of_h : float
        Step size as a fraction of h.

    Returns
    -------
    p_safe : ndarray (3,)
        Launch point (free voxel if success, else last attempted point).
    cls : dict
        Grid classification at p_safe.
    success : bool
        True only if a free voxel was found within max_tries steps.
        If False the caller MUST NOT use p_safe as a physical launch point.
    """
    p = np.asarray(r_hit, dtype=float).copy()
    n_vac = unit(np.asarray(n_vacuum, dtype=float))

    h = float(field["h"])
    ds = step_fraction_of_h * h

    cls = classify_grid_point(p, field)

    for _ in range(max_tries):
        p = p + ds * n_vac
        cls = classify_grid_point(p, field)

        if cls["status"] == "free":
            return p, cls, True

        # Stepped outside the field entirely — give up immediately.
        if cls["status"] in {"left_grid", "left_update_region"}:
            return p, cls, False

    # Exhausted all tries without finding free vacuum.
    return p, cls, False


def transparency_for_owner(grid_transparency, owner, default=1.0):
    if grid_transparency is None:
        return float(default)

    if isinstance(grid_transparency, dict):
        return float(grid_transparency.get(owner, default))

    return float(grid_transparency)
