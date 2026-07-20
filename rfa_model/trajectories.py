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


# ============================================================
# Grid-wire geometry (angle-dependent mesh transparency)
# ============================================================
#
# Physical mesh spec (Unique Wire Weaving Co. tungsten cloth, per the RFA
# instrument paper): 25.4 um diameter wires, 500 um opening, 513 um pitch,
# giving ~93% optical transparency for a single grid at normal incidence.
# All three grids (g1, g2, g3) are fabricated from the same mesh stock.
DEFAULT_WIRE_DIAMETER_M = 25.4e-6
DEFAULT_WIRE_PITCH_M = 513e-6

DEFAULT_GRID_WIRE_GEOMETRY = {
    "g1_shell": {
        "wire_diameter_m": DEFAULT_WIRE_DIAMETER_M,
        "pitch_m": DEFAULT_WIRE_PITCH_M,
    },
    "g2_shell": {
        "wire_diameter_m": DEFAULT_WIRE_DIAMETER_M,
        "pitch_m": DEFAULT_WIRE_PITCH_M,
    },
    "g3_shell": {
        "wire_diameter_m": DEFAULT_WIRE_DIAMETER_M,
        "pitch_m": DEFAULT_WIRE_PITCH_M,
    },
}


def wire_geometry_for_owner(grid_wire_geometry, owner, default=None):
    """
    Look up per-shell wire geometry ({"wire_diameter_m", "pitch_m"}).

    Falls back to `default` (or DEFAULT_GRID_WIRE_GEOMETRY's per-shell
    entry) if grid_wire_geometry is None or does not have this owner.
    """
    if default is None:
        default = DEFAULT_GRID_WIRE_GEOMETRY.get(
            owner,
            {
                "wire_diameter_m": DEFAULT_WIRE_DIAMETER_M,
                "pitch_m": DEFAULT_WIRE_PITCH_M,
            },
        )

    if grid_wire_geometry is None:
        return default

    return grid_wire_geometry.get(owner, default)


def mesh_angular_falloff(wire_diameter_m: float, pitch_m: float, cos_theta_local: float) -> float:
    """
    Normalized (dimensionless) falloff factor for woven-mesh transparency
    as a function of local incidence angle, derived from wire geometry.

    Model: the mesh is treated locally as two orthogonal sets of parallel
    cylindrical wires (valid since the mesh pitch is tiny compared to the
    grid's radius of curvature). Viewed at polar angle theta from the
    local mesh-plane normal, a wire's projected shadow width grows as
    d / cos(theta), reducing the 1D linear transmittance of one wire set
    from t0 = (pitch - diameter) / pitch to:

        t(theta) = 1 - (1 - t0) / cos(theta)

    Two orthogonal wire sets combine multiplicatively, so areal
    transparency scales as t(theta)^2. This function returns that falloff
    normalized to 1.0 at normal incidence (t(theta)/t0)^2, so it can be
    multiplied directly against a separately calibrated, measured
    normal-incidence transparency (see angle_corrected_grid_transparency).

    Returns 0.0 once the local wire shadow fully occludes the opening
    (the classic "mesh looks opaque at grazing incidence" limit).
    """
    if pitch_m <= 0.0:
        raise ValueError("pitch_m must be positive")

    if wire_diameter_m < 0.0 or wire_diameter_m >= pitch_m:
        raise ValueError("wire_diameter_m must be in [0, pitch_m)")

    cos_theta_local = float(np.clip(cos_theta_local, 0.0, 1.0))

    t0 = (pitch_m - wire_diameter_m) / pitch_m

    if cos_theta_local <= 0.0:
        return 0.0

    t = 1.0 - (1.0 - t0) / cos_theta_local

    if t <= 0.0:
        return 0.0

    return float((t / t0) ** 2)


def angle_corrected_grid_transparency(
    T0_normal: float,
    wire_diameter_m: float,
    pitch_m: float,
    cos_theta_local: float,
) -> float:
    """
    Angle-dependent grid transparency, anchored to a measured/calibrated
    normal-incidence transparency T0_normal (e.g. grid_transparency["g1_shell"]),
    with the falloff shape/rate derived purely from wire diameter and pitch.

    T(theta) = T0_normal * mesh_angular_falloff(...)

    so T(theta=0) == T0_normal exactly, and T decreases toward grazing
    incidence at the rate set by the real wire geometry.
    """
    falloff = mesh_angular_falloff(wire_diameter_m, pitch_m, cos_theta_local)

    return float(np.clip(T0_normal * falloff, 0.0, 1.0))


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
    grid_wire_geometry=None,
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
        Dict such as {"g1_shell": 0.90, ...}. This is treated as each
        shell's calibrated NORMAL-INCIDENCE transparency; the actual
        transmission probability used at each crossing is angle-corrected
        using grid_wire_geometry (see below).
    grid_wire_geometry:
        Optional dict such as {"g1_shell": {"wire_diameter_m": 25.4e-6,
        "pitch_m": 513e-6}, ...}. Used to derive how transparency falls
        off away from normal incidence, from the real woven-mesh wire
        diameter and pitch (see mesh_angular_falloff /
        angle_corrected_grid_transparency). Defaults to
        DEFAULT_GRID_WIRE_GEOMETRY (the as-fabricated tungsten mesh spec)
        for any shell not present in the dict.
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
    # grid_events and events historically tracked the same kind of
    # "electron passed through a grid shell" bookkeeping, but from two
    # different collision-detection code paths (fixed-voxel classification
    # vs. analytic sphere hit). They are aliased to the same underlying
    # list here so that every return path can expose both keys
    # consistently, instead of "grid_events" only appearing on returns
    # triggered by the fixed-voxel path and "events" only appearing on
    # returns triggered by the analytic-sphere path.
    grid_events = []
    events = grid_events
    
    ignore_sphere_owners = set()
    
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        return _trajectory_failure(
            reason="nan_state",
            p=p,
            v=v,
            traj=traj,
            vel=vel,
            step=0,
            extra={"grid_events": grid_events, "events": events},
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
                    # Provide an auto-orientable local normal so that
                    # estimate_surface_normal() in cascade.py can correctly
                    # orient cascade-emission direction/placement against
                    # this electron's actual approach direction. Without
                    # this, a terminal hit reached via the fixed-voxel
                    # classification path (as opposed to the analytic
                    # sphere-hit path) has no "normal" key, so
                    # estimate_surface_normal() falls back to a fixed
                    # outward-radial normal regardless of whether the
                    # electron approached from inside or outside the
                    # shell - misorienting emission for inward-travelling
                    # electrons and, downstream, misplacing their launch
                    # point search direction as well.
                    r_norm = np.linalg.norm(p)
                    if r_norm > 0:
                        n_raw = p / r_norm
                        if np.dot(v, n_raw) > 0:
                            n_raw = -n_raw
                        hit_info["normal"] = n_raw

                        speed = np.linalg.norm(v)
                        cos_theta_local = (
                            float(-np.dot(v, n_raw) / speed) if speed > 0 else 1.0
                        )
                    else:
                        cos_theta_local = 1.0

                    if grid_transparency is None:
                        T = 1.0
                    else:
                        T0_normal = transparency_for_owner(grid_transparency, owner, default=1.0)
                        wire_geom = wire_geometry_for_owner(grid_wire_geometry, owner)
                        T = angle_corrected_grid_transparency(
                            T0_normal,
                            wire_geom["wire_diameter_m"],
                            wire_geom["pitch_m"],
                            cos_theta_local,
                        )

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
                            "events": events,
                            "grid_events": grid_events,
                        }

                    # Transmitted through the grid shell.
                    grid_events.append({
                        "owner": owner,
                        "location": p.copy(),
                        "steps": step,
                        "type": "transmit_fixed_voxel",
                        "T": T,
                        "T0_normal": T0_normal,
                        "cos_theta_local": cos_theta_local,
                    })

                    p, cls_after = advance_grid_transmission_until_free(
                        p,
                        v,
                        field,
                        owner=owner,
                        max_tries=80,
                        step_fraction_of_h=0.10,
                    )

                    traj.append(p.copy())
                    vel.append(v.copy())

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
                            extra={"grid_events": grid_events, "events": events},
                        )

                    # Continue integration after stepping through shell.
                    continue

                # Ordinary fixed voxel: absorbing electrode.
                return {
                    "reason": "hit_fixed",
                    "hit_info": hit_info,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "steps": step,
                    "events": events,
                    "grid_events": grid_events,
                }

            return {
                "reason": cls["status"],
                "hit_info": hit_info,
                "traj": np.asarray(traj),
                "vel": np.asarray(vel),
                "steps": step,
                "events": events,
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

        # A Verlet step can produce a finite p_new but a nonfinite v_new when
        # p_new has crossed outside the interpolation/update domain and the
        # field evaluation at the new point returns NaN.  Classify that as a
        # clean boundary exit rather than as an internal numerical failure.
        if np.all(np.isfinite(p_new)) and not np.all(np.isfinite(v_new)):
            cls_new = classify_grid_point(p_new, field)
            if cls_new["status"] in {"left_grid", "left_update_region"}:
                return _trajectory_failure(
                    reason=cls_new["status"],
                    p=p_new,
                    v=v,
                    traj=traj + [p_new.copy()],
                    vel=vel + [v.copy()],
                    step=step + 1,
                    hit_info={
                        **cls_new,
                        "owner_name": "escaped",
                        "owner": "escaped",
                        "location": p_new.copy(),
                        "v_in": v.copy(),
                        "KE_hit_eV": kinetic_energy_eV_from_velocity(v),
                        "boundary_exit": True,
                    },
                    extra={
                        "p_before": p.copy(),
                        "v_before": v.copy(),
                        "dt_step": dt_step,
                        "grid_events": grid_events,
                        "events": events,
                    },
                )

        if not np.all(np.isfinite(p_new)) or not np.all(np.isfinite(v_new)):
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
                    "grid_events": grid_events,
                    "events": events,
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
                    "grid_events": grid_events,
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
                    "grid_events": grid_events,
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
                        "grid_events": grid_events,
                        "steps": step + 1,
                    }

                if event_type == "transmit_grid":
                    T0_normal = transparency_for_owner(grid_transparency, owner, default=1.0)

                    # Local incidence angle at the actual crossing point,
                    # used to geometrically derate transparency away from
                    # normal incidence. Prefer the hit's own normal if the
                    # collision module already supplies one; otherwise fall
                    # back to the radial direction (valid for the analytic
                    # spherical grid shells).
                    n_hit = hit.get("normal", None)
                    if n_hit is None:
                        r_hit_norm = np.linalg.norm(hit["location"])
                        n_hit = (
                            hit["location"] / r_hit_norm if r_hit_norm > 0 else None
                        )

                    speed = np.linalg.norm(v_new)
                    if n_hit is not None and speed > 0:
                        cos_theta_local = float(
                            abs(np.dot(v_new, unit(n_hit))) / speed
                        )
                    else:
                        cos_theta_local = 1.0

                    wire_geom = wire_geometry_for_owner(grid_wire_geometry, owner)
                    T = angle_corrected_grid_transparency(
                        T0_normal,
                        wire_geom["wire_diameter_m"],
                        wire_geom["pitch_m"],
                        cos_theta_local,
                    )

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
                            "grid_events": grid_events,
                            "steps": step + 1,
                        }

                    events.append({
                        "type": "transmit_grid",
                        "owner": owner,
                        "location": hit["location"],
                        "steps": step,
                        "u": u,
                        "T": T,
                        "T0_normal": T0_normal,
                        "cos_theta_local": cos_theta_local,
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
        "grid_events": grid_events,
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
    Advance a TRANSMITTED electron along its velocity until it leaves
    the current fixed voxel layer.

    This is the original function, intended for grid-shell transmissions
    where advancing along the velocity direction is geometrically correct.

    Returns (p, cls) — same signature as before for backward compatibility.
    The caller should check cls["status"] == "free" if it needs to know
    whether the search succeeded.
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

        # Early exit if already outside valid region — no point continuing.
        if cls["status"] in {"left_grid", "left_update_region"}:
            return p, cls

    return p, cls



def place_grid_emission_in_vacuum(
    r_hit,
    emission_direction,
    field,
    max_forward_tries: int = 20,
    max_radial_tries: int = 20,
    step_fraction_of_h: float = 0.10,
    tangential_tol: float = 0.05,
):
    """
    Place a NEW electron emitted from a spherical grid shell in free vacuum.

    The angular sampler has already selected the physical emission velocity.
    This helper preserves that velocity and uses it only to select the
    physically consistent side of the voxelized shell.

    Search order:
      1. Try moving directly along the sampled emission direction.
      2. If voxelization traps the point, search radially toward the side
         indicated by v_emit dot r_hat.
      3. For a nearly tangential direction, search both radial sides and
         choose the closest free point.

    Only the launch position is changed; the caller keeps v0 unchanged.
    """
    r_hit = np.asarray(r_hit, dtype=float)
    d_emit = unit(np.asarray(emission_direction, dtype=float))
    r_hat = unit(r_hit)

    h = float(field["h"])
    ds = float(step_fraction_of_h) * h

    if ds <= 0.0:
        raise ValueError("step_fraction_of_h must be positive")

    # First honor the sampled direction directly.
    p = r_hit.copy()
    last_cls = classify_grid_point(p, field)

    for _ in range(max_forward_tries):
        p = p + ds * d_emit
        last_cls = classify_grid_point(p, field)

        if last_cls["status"] == "free":
            return p, last_cls, True

        if last_cls["status"] in {"left_grid", "left_update_region"}:
            break

    radial_component = float(np.dot(d_emit, r_hat))

    if radial_component > tangential_tol:
        signs = (1.0,)
    elif radial_component < -tangential_tol:
        signs = (-1.0,)
    else:
        signs = (1.0, -1.0)

    candidates = []

    for sign in signs:
        p_try = r_hit.copy()
        cls_try = classify_grid_point(p_try, field)

        for _ in range(max_radial_tries):
            p_try = p_try + sign * ds * r_hat
            cls_try = classify_grid_point(p_try, field)

            if cls_try["status"] == "free":
                candidates.append(
                    (float(np.linalg.norm(p_try - r_hit)), p_try.copy(), cls_try)
                )
                break

            if cls_try["status"] in {"left_grid", "left_update_region"}:
                break

        last_cls = cls_try

    if candidates:
        _, p_best, cls_best = min(candidates, key=lambda item: item[0])
        return p_best, cls_best, True

    return p, last_cls, False


def advance_grid_transmission_until_free(
    p,
    v,
    field,
    owner: str | None = None,
    max_tries: int = 80,
    step_fraction_of_h: float = 0.10,
    tangential_tol: float = 1.0e-6,
):
    """
    Move a transmitted electron across the artificial voxelized grid shell
    without changing its velocity.
    """
    p = np.asarray(p, dtype=float).copy()
    v = np.asarray(v, dtype=float)

    d = unit(v)
    r = float(np.linalg.norm(p))

    if r <= 0.0:
        return p, classify_grid_point(p, field)

    r_hat = p / r
    h = float(field["h"])
    ds = float(step_fraction_of_h) * h

    radius_map = {
        "g1_shell": "R_g1",
        "g2_shell": "R_g2",
        "g3_shell": "R_g3",
    }

    radius_key = radius_map.get(owner)
    R_shell = float(field[radius_key]) if radius_key in field else r

    radial_component = float(np.dot(d, r_hat))

    if radial_component > tangential_tol:
        sign = 1.0
    elif radial_component < -tangential_tol:
        sign = -1.0
    else:
        # For a nearly tangential trajectory, determine the side from
        # the current radius relative to the analytic shell.
        sign = 1.0 if r >= R_shell else -1.0

    def is_correct_side(point):
        radius = float(np.linalg.norm(point))
        return sign * (radius - R_shell) > 0.0

    last_p = p.copy()
    last_cls = classify_grid_point(last_p, field)

    # 1. Radial search from the analytic shell.
    for itry in range(1, max_tries + 1):
        clearance = itry * ds
        p_try = (R_shell + sign * clearance) * r_hat
        cls_try = classify_grid_point(p_try, field)

        last_p = p_try
        last_cls = cls_try

        if cls_try["status"] == "free" and is_correct_side(p_try):
            return p_try, cls_try

        if cls_try["status"] in {"left_grid", "left_update_region"}:
            break

    # 2. Search along the actual trajectory, but only accept a point
    # on the correct side of the analytic shell.
    p_try = p.copy()

    for _ in range(max_tries):
        p_try = p_try + ds * d
        cls_try = classify_grid_point(p_try, field)

        last_p = p_try
        last_cls = cls_try

        if cls_try["status"] == "free" and is_correct_side(p_try):
            return p_try, cls_try

        if cls_try["status"] in {"left_grid", "left_update_region"}:
            break

    # 3. Small lateral searches around the radial direction to escape
    # stair-stepped voxel chains.
    axis_candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    tangent = None

    for axis in axis_candidates:
        candidate = np.cross(r_hat, axis)
        norm_candidate = np.linalg.norm(candidate)

        if norm_candidate > 1.0e-12:
            tangent = candidate / norm_candidate
            break

    if tangent is not None:
        bitangent = unit(np.cross(r_hat, tangent))

        lateral_directions = [
            tangent,
            -tangent,
            bitangent,
            -bitangent,
            unit(tangent + bitangent),
            unit(tangent - bitangent),
            unit(-tangent + bitangent),
            unit(-tangent - bitangent),
        ]

        for lateral in lateral_directions:
            for itry in range(1, max_tries + 1):
                radial_clearance = itry * ds
                lateral_offset = min(itry, 4) * 0.25 * ds

                p_try = (
                    (R_shell + sign * radial_clearance) * r_hat
                    + lateral_offset * lateral
                )

                cls_try = classify_grid_point(p_try, field)

                last_p = p_try
                last_cls = cls_try

                if cls_try["status"] == "free" and is_correct_side(p_try):
                    return p_try, cls_try

                if cls_try["status"] in {
                    "left_grid",
                    "left_update_region",
                }:
                    break

    return last_p, last_cls

def place_emitted_particle_in_vacuum(
    r_hit,
    n_vacuum,
    field,
    max_tries: int = 12,
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
        Maximum number of normal-directed steps.  12 steps × 0.10h ≈ 1.2h
        which is safely larger than the voxel layer thickness without
        overshooting into the next electrode.
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