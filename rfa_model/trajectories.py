"""
trajectories.py

Electron trajectory integration through the RFA field, including
STL collisions, analytic spherical grid/collector surfaces, grid
transparency, openings, adaptive timestep, and analytic sample-plane
return.
"""

from __future__ import annotations

import numpy as np

from .constants import e_charge, m_e, kinetic_energy_eV_from_velocity
from .fields import E_at_point
from .collisions import (
    first_segment_hit,
    first_analytic_grid_hit,
    segment_near_any_stl_box,
    segment_hits_sample_plane,
    classify_sphere_event,
    nearest_hit,
)


def _trajectory_failure(reason, p, v, traj, vel, step, extra=None):
    hit_info = {
        "kind": reason,
        "location": np.asarray(p, dtype=float),
        "v_in": np.asarray(v, dtype=float),
        "KE_hit_eV": np.nan,
    }

    if extra is not None:
        hit_info.update(extra)

    return {
        "reason": reason,
        "hit_info": hit_info,
        "traj": np.asarray(traj),
        "vel": np.asarray(vel),
        "steps": step,
    }


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
            "g1_shell": 1.0,
            "g2_shell": 1.0,
            "g3_shell": 1.0,
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
                        "step": step,
                        "type": "transmit_fixed_voxel",
                    })

                    p, cls_after = advance_until_free(
                        p,
                        v,
                        field,
                        max_tries=20,
                        step_fraction_of_h=0.25,
                    )

                    traj.append(p.copy())
                    vel.append(v.copy())

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
                        "step": step,
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


def advance_until_free(
    p,
    v,
    field,
    max_tries: int = 20,
    step_fraction_of_h: float = 0.25,
):
    """
    Advance a transmitted electron along its velocity until it is outside
    the current fixed voxel layer.

    This is needed for fixed grid-shell voxels, because surface_eps alone
    can be much smaller than the field-grid voxel thickness.
    """
    p = np.asarray(p, dtype=float).copy()
    direction = unit(v)

    h = float(field["h"])
    ds = step_fraction_of_h * h

    for _ in range(max_tries):
        p = p + ds * direction
        cls = classify_grid_point(p, field)

        if cls["status"] == "free":
            return p, cls

    return p, cls


def transparency_for_owner(grid_transparency, owner, default=1.0):
    if grid_transparency is None:
        return float(default)

    if isinstance(grid_transparency, dict):
        return float(grid_transparency.get(owner, default))

    return float(grid_transparency)
