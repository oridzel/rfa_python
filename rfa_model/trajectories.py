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
)
from .fields import E_at_point
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

    The test is intentionally conservative: the electron must be moving toward
    +X, be close to the drift tube's exit plane, and lie inside the bore in the
    YZ plane.

    Two geometry sources, resolved through the same helper that
    collisions.is_in_drift_tube_aperture() uses so the two can never disagree:

    Preferred - field["drifttube_bore"] from compute_drifttube_bore_geometry(),
    giving the real bore radius, the bore's offset from the RFA axis, and the
    tube's own x extent. The escape plane is then x_exit = min(tube +X end,
    domain +X edge) rather than always the domain edge. This matters because the
    drift tube is now real collision geometry: an electron inside the bore that
    strikes the wall is DT current and is caught by the STL test, so the only
    electrons that should be classified as escapes are those still inside the
    bore when they reach the exit. Testing at the domain edge with a nominal
    on-axis circle could label an electron an escape at a radius the real tube
    wall would have intercepted.

    Legacy - field["drifttube_aperture_radius"] (default 5.6 mm) as a circle
    centred exactly on the axis, evaluated at the domain +X edge. Retained so
    setups that have not called compute_drifttube_bore_geometry() are unchanged.

    An explicit aperture_radius argument always wins and is interpreted on-axis
    at the domain edge.
    """
    from .collisions import drifttube_bore_from_field

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
        bore = drifttube_bore_from_field(field)

        if bore is not None:
            y0, z0 = bore["center_yz"]
            radial_yz = float(np.hypot(p[1] - float(y0), p[2] - float(z0)))
            x_exit = min(float(bore["x_exit"]), x_max)

            return (p[0] >= x_exit - 1.5 * h) and (
                radial_yz <= float(bore["radius"])
            )

        # Must match collisions.is_in_drift_tube_aperture()'s default so the
        # domain-boundary escape check and the grid-opening check agree on
        # the same physical drift-tube bore radius.
        aperture_radius = float(
            field.get("drifttube_aperture_radius", 5.6e-3)
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
        # Use the same effective collision classification everywhere,
        # including launch placement. Grid/collector shell voxels impose the
        # field boundary but are not physical collision geometry.
        cls = classify_effective_vacuum_point(p, field)

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
                if len(traj) >= 2:
                    hit_info["p_before"] = np.asarray(
                        traj[-2], dtype=float
                    ).copy()

                # Grid-shell owners can never reach this point: they were
                # already neutralized to "free" above, since grid crossing
                # is now handled exclusively by the analytic segment-sphere
                # test later in this loop. Any remaining fixed-voxel hit
                # here is therefore an ordinary absorbing electrode.
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
                hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                hit["v_in"] = np.asarray(v_new, dtype=float).copy()

                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "grid_events": grid_events,
                    "events": grid_events,
                    "steps": step + 1,
                }

            if hit["kind"] == "stl":
                traj.append(hit["location"].copy())
                vel.append(v_new.copy())
                hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                hit["v_in"] = np.asarray(v_new, dtype=float).copy()
                hit["p_before"] = np.asarray(p, dtype=float).copy()

                return {
                    "reason": "hit_stl",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "grid_events": grid_events,
                    "events": grid_events,
                    "steps": step + 1,
                }

            if hit["kind"] == "sphere":
                event_type = classify_sphere_event(hit)
                owner = hit["owner"]

                if event_type == "hit_collector":
                    traj.append(hit["location"].copy())
                    vel.append(v_new.copy())
                    hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                    hit["v_in"] = np.asarray(v_new, dtype=float).copy()

                    return {
                        "reason": "hit_collector",
                        "hit_info": hit,
                        "traj": np.asarray(traj),
                        "vel": np.asarray(vel),
                        "grid_events": grid_events,
                        "events": grid_events,
                        "steps": step + 1,
                    }

                if event_type == "transmit_grid":
                    T = transparency_for_owner(grid_transparency, owner, default=1.0)

                    u = rng.random()

                    if u > T:
                        traj.append(hit["location"].copy())
                        vel.append(v_new.copy())
                        hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                        hit["v_in"] = np.asarray(v_new, dtype=float).copy()

                        return {
                            "reason": "hit_grid_wire",
                            "hit_info": hit,
                            "traj": np.asarray(traj),
                            "vel": np.asarray(vel),
                            "grid_events": grid_events,
                            "events": grid_events,
                            "steps": step + 1,
                        }

                    grid_events.append({
                        "type": "transmit_grid",
                        "owner": owner,
                        "location": hit["location"],
                        "step": step,
                        "u": u,
                        "T": T,
                    })

                    # Move only a tiny distance onto the transmitted side
                    # to avoid detecting the same analytic sphere again.
                    # No voxel-clearing jump is needed because grid-shell
                    # voxels do not act as collision geometry.
                    analytic_eps = max(
                        float(surface_eps),
                        1.0e-6 * float(field["h"]),
                    )
                    p = hit["location"] + analytic_eps * unit(v_new)
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
        "grid_events": grid_events,
        "events": grid_events,
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


IGNORED_FIXED_BOUNDARY_OWNERS = frozenset({
    "g1_shell",
    "g2_shell",
    "g3_shell",
    "collector_shell",
})
IGNORED_FIXED_BOUNDARY_OWNER_IDS = frozenset({9, 10, 11, 12})


def classify_effective_vacuum_point(p, field):
    """
    Classify a point using the same collision rule as trajectory tracking.

    The voxelized grid and collector shells are electrostatic boundary
    conditions only.  They must not block a trajectory or a newly emitted
    particle.  Physical STL owners, including g1frame/g2frame/g3frame, remain
    solid.

    The returned dictionary preserves the raw voxel classification and owner
    so launch-failure diagnostics can distinguish an ignored analytic shell
    from a real solid.
    """
    raw = dict(classify_grid_point(p, field))
    owner_id = raw.get("owner_id", None)
    owner_name = fixed_owner_name(owner_id, field=field)

    effective = dict(raw)
    effective["raw_status"] = raw.get("status", None)
    effective["raw_owner_id"] = owner_id
    effective["raw_owner_name"] = owner_name

    if (
        raw.get("status", None) == "hit_fixed"
        and (
            owner_name in IGNORED_FIXED_BOUNDARY_OWNERS
            or owner_id in IGNORED_FIXED_BOUNDARY_OWNER_IDS
        )
    ):
        effective["status"] = "free"
        effective["ignored_fixed_owner"] = owner_name

    return effective


def is_grid_shell_owner(owner_name: str) -> bool:
    """
    True for fixed-potential spherical grid-shell voxels.

    These are retained only as electrostatic boundary conditions in the
    field solve. They are ignored as collision geometry: physical grid
    crossings are handled exclusively by the analytic segment-sphere test
    (first_analytic_grid_hit / classify_sphere_event) further down in
    integrate_one_electron().
    """
    return owner_name in ["g1_shell", "g2_shell", "g3_shell"]


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
    fallback_vacuum_point=None,
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
    fallback_vacuum_point : array (3,), optional
        Last known vacuum point before the surface hit.  If the surface-normal
        search fails, retry toward this point.  This is a geometry-grounded
        fallback for grazing STL hits and is safer than guessing with -v_in.

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

    cls = classify_effective_vacuum_point(p, field)

    for attempt in range(1, max_tries + 1):
        p = np.asarray(r_hit, dtype=float) + attempt * ds * n_vac
        cls = classify_effective_vacuum_point(p, field)

        if cls["status"] == "free":
            cls = dict(cls)
            cls.update({
                "placement_method": "solid_vacuum_normal_search",
                "placement_attempts": attempt,
                "placement_offset_m": float(np.linalg.norm(
                    p - np.asarray(r_hit, dtype=float)
                )),
            })
            return p, cls, True

        # Stepped outside the field entirely — give up immediately.
        if cls["status"] in {"left_grid", "left_update_region"}:
            break

    # A real STL face normal can be poorly oriented or nearly tangential for
    # overlapping/non-manifold geometry.  The preceding trajectory point is
    # known to be on the incident vacuum side, so it provides a safe fallback.
    if fallback_vacuum_point is not None:
        p_prev = np.asarray(fallback_vacuum_point, dtype=float)
        fallback_vec = p_prev - np.asarray(r_hit, dtype=float)
        fallback_norm = float(np.linalg.norm(fallback_vec))

        if np.all(np.isfinite(p_prev)) and fallback_norm > 0.0:
            fallback_dir = fallback_vec / fallback_norm

            for attempt in range(1, max_tries + 1):
                p = (
                    np.asarray(r_hit, dtype=float)
                    + attempt * ds * fallback_dir
                )
                cls = classify_effective_vacuum_point(p, field)

                if cls["status"] == "free":
                    cls = dict(cls)
                    cls.update({
                        "placement_method":
                            "solid_previous_vacuum_point_search",
                        "placement_attempts": attempt,
                        "placement_offset_m": float(np.linalg.norm(
                            p - np.asarray(r_hit, dtype=float)
                        )),
                    })
                    return p, cls, True

                if cls["status"] in {"left_grid", "left_update_region"}:
                    break

    # Exhausted all tries without finding free vacuum.
    cls = dict(cls)
    cls.update({
        "placement_method": "failed_solid_launch_searches",
        "placement_attempts": max_tries,
        "placement_offset_m": float(np.linalg.norm(
            p - np.asarray(r_hit, dtype=float)
        )),
    })
    return p, cls, False


def transparency_for_owner(grid_transparency, owner, default=1.0):
    if grid_transparency is None:
        return float(default)

    if isinstance(grid_transparency, dict):
        return float(grid_transparency.get(owner, default))

    return float(grid_transparency)
