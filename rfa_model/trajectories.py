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
    sample_plane_coordinates,
    sample_point_is_in_bounds,
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
        "traj": None if traj is None else np.asarray(traj, dtype=float),
        "vel": None if vel is None else np.asarray(vel, dtype=float),
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




def advance_through_sample_voxel_artifact(
    p,
    v,
    field,
    sample_geometry,
    sample_owner_id: int = 1,
    max_tries: int = 240,
    step_fraction_of_h: float = 0.05,
):
    """Traverse the voxelized sample staircase without treating it as solid.

    The rotated sample is voxelized for the electrostatic boundary condition,
    but the *continuous finite sample plane* is the physical collision surface.
    At grazing incidence a fixed sample voxel may protrude into physical vacuum.
    Starting from such a voxel, advance locally along the current electron
    direction until the continuous sample plane is crossed or the sample-owned
    voxel staircase is cleared.

    Motion inside the staircase artifact is ballistic because the field inside
    the fixed sample voxels is itself a discretization artifact.  Callers should
    still test the returned bypass segment for STL collisions.
    """
    if sample_geometry is None:
        raise ValueError("sample_geometry is required")
    if max_tries <= 0:
        raise ValueError("max_tries must be positive")
    if step_fraction_of_h <= 0.0:
        raise ValueError("step_fraction_of_h must be positive")

    p0 = np.asarray(p, dtype=float).copy()
    direction = unit(np.asarray(v, dtype=float))
    h = float(field["h"])
    ds = float(step_fraction_of_h) * h

    p_prev = p0.copy()
    last_cls = dict(classify_grid_point(p0, field))

    for attempt in range(1, int(max_tries) + 1):
        candidate = p0 + attempt * ds * direction

        sample_hit = segment_hits_sample_plane(
            p_prev,
            candidate,
            x_sample=0.0,
            sample_y_bounds=None,
            sample_z_bounds=None,
            sample_geometry=sample_geometry,
        )

        if sample_hit is not None:
            hit = dict(sample_hit)
            hit["sample_voxel_artifact_traversal"] = True
            hit["sample_voxel_artifact_attempts"] = attempt
            hit["sample_voxel_artifact_distance_m"] = float(
                np.linalg.norm(hit["location"] - p0)
            )
            return {
                "status": "hit_sample",
                "point": np.asarray(hit["location"], dtype=float).copy(),
                "hit_info": hit,
                "classification": dict(last_cls),
                "attempts": attempt,
                "distance_m": float(np.linalg.norm(hit["location"] - p0)),
            }

        cls = dict(classify_grid_point(candidate, field))
        last_cls = cls
        owner_id = cls.get("owner_id", None)
        is_sample_voxel = (
            cls.get("status") == "hit_fixed"
            and owner_id is not None
            and int(owner_id) == int(sample_owner_id)
        )

        if not is_sample_voxel:
            return {
                "status": "cleared",
                "point": candidate.copy(),
                "hit_info": None,
                "classification": cls,
                "attempts": attempt,
                "distance_m": float(np.linalg.norm(candidate - p0)),
            }

        p_prev = candidate

    return {
        "status": "failed",
        "point": p_prev.copy(),
        "hit_info": None,
        "classification": last_cls,
        "attempts": int(max_tries),
        "distance_m": float(np.linalg.norm(p_prev - p0)),
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



def ballistic_continue_to_drifttube_exit(
    p,
    v,
    field,
    intersector,
    face_owner,
    collision_mesh,
    *,
    exit_epsilon: float = 1.0e-9,
):
    """Continue a +X drift-tube escape candidate to the real STL exit.

    The electrostatic field is unavailable beyond the solved domain, but the
    remaining section of the grounded drift tube is treated as field-free.
    The electron is therefore propagated ballistically from the field boundary
    to the physical +X end of the drift-tube bore.  The continuation segment is
    tested against the same STL collision mesh used by the normal integrator.

    Returns a dict with ``status`` equal to:

    ``"hit_stl"``
        The continuation intersects physical CAD before exiting.

    ``"escaped"``
        The continuation reaches the physical drift-tube exit without a hit.

    ``"unavailable"``
        Real drift-tube bore metadata is not present; callers should preserve
        legacy boundary behavior.
    """
    from .collisions import drifttube_bore_from_field

    p = np.asarray(p, dtype=float)
    v = np.asarray(v, dtype=float)

    bore = drifttube_bore_from_field(field)
    if bore is None:
        return {"status": "unavailable"}

    if p.shape != (3,) or v.shape != (3,):
        return {"status": "unavailable"}
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        return {"status": "unavailable"}

    direction = unit(v)
    if direction[0] <= 0.0:
        return {"status": "unavailable"}

    x_exit = float(bore["x_exit"])

    # If the field domain already reaches or exceeds the physical tube exit,
    # there is nothing left to continue ballistically.
    if p[0] >= x_exit - float(exit_epsilon):
        return {
            "status": "escaped",
            "p_exit": p.copy(),
            "physical_x_exit": x_exit,
            "continuation_distance_m": 0.0,
            "ballistic_continuation": False,
        }

    path_length = (x_exit - p[0]) / direction[0]
    if path_length <= 0.0 or not np.isfinite(path_length):
        return {"status": "unavailable"}

    # Extend a tiny amount beyond the nominal exit plane so that an annular
    # end face / lip at x_exit is included in the segment collision test.
    p_exit = p + path_length * direction
    p_test_end = p + (path_length + float(exit_epsilon)) * direction

    hit = first_segment_hit(
        p,
        p_test_end,
        intersector,
        face_owner,
        collision_mesh,
    )

    if hit is not None:
        return {
            "status": "hit_stl",
            "hit_info": dict(hit),
            "p_exit": p_exit.copy(),
            "physical_x_exit": x_exit,
            "continuation_distance_m": float(
                np.linalg.norm(np.asarray(hit["location"], dtype=float) - p)
            ),
            "ballistic_continuation": True,
        }

    return {
        "status": "escaped",
        "p_exit": p_exit.copy(),
        "physical_x_exit": x_exit,
        "continuation_distance_m": float(path_length),
        "ballistic_continuation": True,
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
    sample_geometry: dict | None = None,
    track_points: bool = False,
    track_stride: int = 1,
):
    """Integrate one electron trajectory through the RFA.

    Point-by-point storage is disabled by default.  When enabled, the launch
    point, every ``track_stride`` accepted integration steps, analytic grid
    crossings, and exact terminal hit locations are retained.  Collision
    physics never depends on the stored trajectory history.
    """
    if rng is None:
        rng = np.random.default_rng()

    if grid_transparency is None:
        grid_transparency = {
            "g1_shell": 0.93,
            "g2_shell": 0.93,
            "g3_shell": 0.93,
        }

    if track_stride <= 0:
        raise ValueError("track_stride must be positive")

    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()

    if track_points:
        traj = [p.copy()]
        vel = [v.copy()]
    else:
        traj = None
        vel = None

    # Keep the previous accepted vacuum point for p_before/fallback logic.
    # This must not depend on optional trajectory storage.
    p_previous = None

    def append_track(position, velocity, *, force=False, accepted_step=None):
        if not track_points:
            return
        if not force:
            if accepted_step is None or (accepted_step % track_stride) != 0:
                return
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if len(traj) == 0 or not np.array_equal(traj[-1], position):
            traj.append(position.copy())
            vel.append(velocity.copy())

    def packed_track():
        return (
            np.asarray(traj, dtype=float) if track_points else None,
            np.asarray(vel, dtype=float) if track_points else None,
        )

    def finalize_drifttube_boundary_escape(position, velocity, terminal_step, extra=None):
        """Resolve a field-boundary DT candidate against the remaining STL."""
        continuation = ballistic_continue_to_drifttube_exit(
            p=position,
            v=velocity,
            field=field,
            intersector=intersector,
            face_owner=face_owner,
            collision_mesh=collision_mesh,
        )

        if continuation.get("status") == "hit_stl":
            hit = dict(continuation["hit_info"])
            hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(velocity)
            hit["v_in"] = np.asarray(velocity, dtype=float).copy()
            hit["p_before"] = np.asarray(position, dtype=float).copy()
            hit["ballistic_drifttube_continuation"] = True
            hit["continuation_distance_m"] = continuation.get(
                "continuation_distance_m", np.nan
            )
            hit["physical_x_exit"] = continuation.get(
                "physical_x_exit", np.nan
            )

            append_track(hit["location"], velocity, force=True)
            traj_out, vel_out = packed_track()
            return {
                "reason": "hit_stl",
                "hit_info": hit,
                "traj": traj_out,
                "vel": vel_out,
                "grid_events": grid_events,
                "events": grid_events,
                "steps": terminal_step,
                "ballistic_drifttube_continuation": True,
            }

        # Legacy behavior if real bore metadata is unavailable.
        p_escape = np.asarray(
            continuation.get("p_exit", position), dtype=float
        )
        append_track(p_escape, velocity, force=True)

        payload = {
            "ballistic_drifttube_continuation": bool(
                continuation.get("ballistic_continuation", False)
            ),
            "continuation_distance_m": continuation.get(
                "continuation_distance_m", 0.0
            ),
            "physical_x_exit": continuation.get(
                "physical_x_exit", np.nan
            ),
            "field_boundary_p": np.asarray(position, dtype=float).copy(),
        }
        if extra:
            payload.update(extra)

        return drifttube_escape_result(
            p=p_escape,
            v=velocity,
            traj=traj,
            vel=vel,
            step=terminal_step,
            grid_events=grid_events,
            extra=payload,
        )

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
        cls = classify_effective_vacuum_point(p, field)

        # GRAZING_ANALYTIC_SAMPLE_VOXEL_BYPASS_V1
        # ---------------------------------------------------------------
        # The voxelized sample exists to impose the electrostatic boundary;
        # the finite analytic sample plane is the physical collision surface.
        # At grazing incidence the voxel staircase can protrude a long way
        # into physical vacuum along an emitted-electron ray. Do NOT terminate
        # such a trajectory merely because its nearest field node is a sample
        # fixed voxel.
        #
        # Unlike the high-energy primary artifact helper, we deliberately do
        # not jump geometrically through many millimeters here. A grazing BSE
        # can have tiny normal kinetic energy and can be physically turned by
        # the sample-bias field. Therefore the ordinary field integrator keeps
        # advancing while only the collision classification of an analytically
        # vacuum sample voxel is neutralized.
        if (
            sample_plane_return
            and sample_geometry is not None
            and cls.get("status") == "hit_fixed"
        ):
            _sample_owner_id = cls.get("owner_id", None)
            _sample_owner = fixed_owner_name(_sample_owner_id, field=field)

            if _sample_owner == "sample":
                _C_sample = np.asarray(sample_geometry["center"], dtype=float)
                _n_sample = unit(
                    np.asarray(sample_geometry["normal"], dtype=float)
                )
                _signed_sample = float(np.dot(p - _C_sample, _n_sample))
                _projected_sample = p - _signed_sample * _n_sample
                _projected_in_bounds = sample_point_is_in_bounds(
                    _projected_sample, sample_geometry
                )
                _travel_from_launch = float(
                    np.linalg.norm(
                        _projected_sample - np.asarray(p0, dtype=float)
                    )
                )

                # If numerical stepping has already placed the electron on the
                # solid side of the true finite face, recover the physical
                # analytic-plane return instead of calling it a voxel hit.
                if (
                    _signed_sample <= 0.0
                    and _projected_in_bounds
                    and _travel_from_launch >= float(min_sample_return_distance)
                ):
                    _, _sample_u, _sample_v = sample_plane_coordinates(
                        _projected_sample, sample_geometry
                    )
                    _sample_hit = {
                        "kind": "sample_plane_recovered_from_voxel",
                        "location": _projected_sample.copy(),
                        "distance": float(
                            np.linalg.norm(_projected_sample - p)
                        ),
                        "owner": "sample",
                        "owner_name": "sample",
                        "owner_id": _sample_owner_id,
                        "normal": _n_sample.copy(),
                        "sample_u": _sample_u,
                        "sample_v": _sample_v,
                        "sample_theta_deg": sample_geometry.get(
                            "theta_deg", np.nan
                        ),
                        "grid_classification": dict(cls),
                        "sample_voxel_artifact_recovered": True,
                        "sample_voxel_artifact_signed_distance_m": (
                            _signed_sample
                        ),
                        "KE_hit_eV": kinetic_energy_eV_from_velocity(v),
                        "v_in": np.asarray(v, dtype=float).copy(),
                    }
                    if traj is not None and len(traj) >= 2:
                        _sample_hit["p_before"] = np.asarray(
                            traj[-2], dtype=float
                        ).copy()

                    return {
                        "reason": "hit_sample",
                        "hit_info": _sample_hit,
                        "traj": (
                            np.asarray(traj, dtype=float)
                            if traj is not None
                            else None
                        ),
                        "vel": (
                            np.asarray(vel, dtype=float)
                            if vel is not None
                            else None
                        ),
                        "steps": step,
                        "grid_events": grid_events,
                        "events": grid_events,
                    }

                # Still in physical vacuum (or outside the finite sample face):
                # ignore only the sample-voxel collision label. The field
                # integration, STL tests, analytic sample-plane tests, grid
                # crossings, and all other physics continue normally below.
                cls = dict(cls)
                cls.update({
                    "status": "free",
                    "ignored_fixed_owner": "sample",
                    "sample_voxel_artifact_ignored": True,
                    "sample_voxel_artifact_signed_distance_m": (
                        _signed_sample
                    ),
                    "sample_voxel_artifact_projected_in_bounds": bool(
                        _projected_in_bounds
                    ),
                })

        if cls["status"] != "free":
            hit_info = dict(cls)

            if cls["status"] == "hit_fixed":
                owner_id = cls.get("owner_id", None)
                owner = fixed_owner_name(owner_id, field=field)

                if (
                    sample_plane_return
                    and owner == "sample"
                    and sample_geometry is not None
                ):
                    # The fixed sample voxels are electrostatic boundary
                    # conditions, not the physical collision surface.  At
                    # grazing incidence bypass the local staircase artifact
                    # while retaining the continuous finite sample plane as
                    # the true surface.
                    bypass = advance_through_sample_voxel_artifact(
                        p=p,
                        v=v,
                        field=field,
                        sample_geometry=sample_geometry,
                        sample_owner_id=owner_id,
                    )
                    bypass_end = np.asarray(bypass["point"], dtype=float)
                    sample_hit = bypass.get("hit_info", None)
                    stl_hit = None

                    if (
                        stl_boxes is None
                        or segment_near_any_stl_box(p, bypass_end, stl_boxes)
                    ):
                        stl_hit = first_segment_hit(
                            p,
                            bypass_end,
                            intersector,
                            face_owner,
                            collision_mesh,
                        )

                    hit = stl_hit if stl_hit is not None else sample_hit

                    if hit is not None:
                        hit = dict(hit)
                        hit["sample_voxel_artifact_traversal"] = True
                        hit["sample_voxel_artifact_attempts"] = bypass["attempts"]
                        hit["sample_voxel_artifact_distance_m"] = bypass["distance_m"]
                        hit["grid_classification"] = dict(cls)
                        hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v)
                        hit["v_in"] = v.copy()
                        hit["p_before"] = p.copy()

                        append_track(hit["location"], v, force=True)
                        traj_out, vel_out = packed_track()
                        return {
                            "reason": (
                                "hit_sample"
                                if hit.get("kind") == "sample_plane"
                                else "hit_stl"
                            ),
                            "hit_info": hit,
                            "traj": traj_out,
                            "vel": vel_out,
                            "steps": step,
                            "grid_events": grid_events,
                            "events": grid_events,
                        }

                    if bypass["status"] == "cleared":
                        p_previous = p.copy()
                        p = bypass_end
                        append_track(p, v, force=True)
                        continue

                    hit_info.update({
                        "kind": "sample_voxel_artifact",
                        "owner": "sample",
                        "owner_name": "sample",
                        "location": p.copy(),
                        "sample_voxel_artifact_attempts": bypass["attempts"],
                        "sample_voxel_artifact_distance_m": bypass["distance_m"],
                    })
                    append_track(p, v, force=True)
                    traj_out, vel_out = packed_track()
                    return {
                        "reason": "sample_voxel_artifact_failed",
                        "hit_info": hit_info,
                        "traj": traj_out,
                        "vel": vel_out,
                        "steps": step,
                        "grid_events": grid_events,
                        "events": grid_events,
                    }

                hit_info["kind"] = "fixed_voxel"
                hit_info["owner"] = owner
                hit_info["location"] = p.copy()
                hit_info["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v)
                hit_info["v_in"] = v.copy()
                if p_previous is not None:
                    hit_info["p_before"] = p_previous.copy()

                append_track(p, v, force=True)
                traj_out, vel_out = packed_track()
                return {
                    "reason": "hit_fixed",
                    "hit_info": hit_info,
                    "traj": traj_out,
                    "vel": vel_out,
                    "steps": step,
                    "grid_events": grid_events,
                }

            if cls["status"] in {"left_grid", "left_update_region"}:
                if is_drifttube_escape_candidate(p, v, field):
                    return finalize_drifttube_boundary_escape(
                        position=p,
                        velocity=v,
                        terminal_step=step,
                        extra={"original_grid_status": cls["status"]},
                    )

            append_track(p, v, force=True)
            traj_out, vel_out = packed_track()
            return {
                "reason": cls["status"],
                "hit_info": hit_info,
                "traj": traj_out,
                "vel": vel_out,
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
            if is_drifttube_escape_candidate(p, v, field):
                return finalize_drifttube_boundary_escape(
                    position=p,
                    velocity=v,
                    terminal_step=step + 1,
                    extra={
                        "p_before": p.copy(),
                        "v_before": v.copy(),
                        "dt_step": dt_step,
                        "nonfinite_after_boundary_step": True,
                    },
                )

            append_track(p, v, force=True)
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

        hit_sample = None
        if sample_plane_return:
            hit_sample = segment_hits_sample_plane(
                p,
                p_new,
                x_sample=0.0,
                sample_y_bounds=sample_y_bounds,
                sample_z_bounds=sample_z_bounds,
                sample_geometry=sample_geometry,
            )

            if hit_sample is not None:
                if np.linalg.norm(hit_sample["location"] - p0) < min_sample_return_distance:
                    hit_sample = None

        hit_stl = None
        if stl_boxes is None or segment_near_any_stl_box(p, p_new, stl_boxes):
            hit_stl = first_segment_hit(
                p,
                p_new,
                intersector,
                face_owner,
                collision_mesh,
            )

        hit_grid = first_analytic_grid_hit(
            p,
            p_new,
            field,
            ignore_owners=ignore_sphere_owners,
        )

        hit = nearest_hit(hit_sample, hit_stl, hit_grid)

        if hit is not None:
            if hit["kind"] == "sample_plane":
                t_hit = float(np.clip(hit.get("t", 1.0), 0.0, 1.0))
                v_hit = v + t_hit * (v_new - v)
                append_track(hit["location"], v_hit, force=True)
                hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_hit)
                hit["v_in"] = np.asarray(v_hit, dtype=float).copy()
                hit["p_before"] = np.asarray(p, dtype=float).copy()

                traj_out, vel_out = packed_track()
                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": traj_out,
                    "vel": vel_out,
                    "grid_events": grid_events,
                    "events": grid_events,
                    "steps": step + 1,
                }

            if hit["kind"] == "stl":
                append_track(hit["location"], v_new, force=True)
                hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                hit["v_in"] = np.asarray(v_new, dtype=float).copy()
                hit["p_before"] = np.asarray(p, dtype=float).copy()

                traj_out, vel_out = packed_track()
                return {
                    "reason": "hit_stl",
                    "hit_info": hit,
                    "traj": traj_out,
                    "vel": vel_out,
                    "grid_events": grid_events,
                    "events": grid_events,
                    "steps": step + 1,
                }

            if hit["kind"] == "sphere":
                event_type = classify_sphere_event(hit)
                owner = hit["owner"]

                if event_type == "hit_collector":
                    append_track(hit["location"], v_new, force=True)
                    hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                    hit["v_in"] = np.asarray(v_new, dtype=float).copy()

                    traj_out, vel_out = packed_track()
                    return {
                        "reason": "hit_collector",
                        "hit_info": hit,
                        "traj": traj_out,
                        "vel": vel_out,
                        "grid_events": grid_events,
                        "events": grid_events,
                        "steps": step + 1,
                    }

                if event_type == "transmit_grid":
                    T = transparency_for_owner(grid_transparency, owner, default=1.0)
                    u = rng.random()

                    if u > T:
                        append_track(hit["location"], v_new, force=True)
                        hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_new)
                        hit["v_in"] = np.asarray(v_new, dtype=float).copy()

                        traj_out, vel_out = packed_track()
                        return {
                            "reason": "hit_grid_wire",
                            "hit_info": hit,
                            "traj": traj_out,
                            "vel": vel_out,
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

                    # Preserve the exact crossing in presentation trajectories,
                    # regardless of track_stride, then move a tiny distance to
                    # the transmitted side to avoid redetecting the same shell.
                    append_track(hit["location"], v_new, force=True)
                    analytic_eps = max(
                        float(surface_eps),
                        1.0e-6 * float(field["h"]),
                    )
                    p_previous = p.copy()
                    p = hit["location"] + analytic_eps * unit(v_new)
                    v = v_new.copy()
                    append_track(p, v, force=True)

                    ignore_sphere_owners = {owner}
                    continue

        ignore_sphere_owners = set()

        p_previous = p.copy()
        p = p_new
        v = v_new
        append_track(p, v, accepted_step=step + 1)

    append_track(p, v, force=True)
    traj_out, vel_out = packed_track()
    return {
        "reason": "max_steps",
        "hit_info": None,
        "traj": traj_out,
        "vel": vel_out,
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
