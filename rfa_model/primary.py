"""
primary.py

Primary-beam launch, primary-to-sample tracking, impact-energy/angle
calculation, and first-generation sample-emission tracking.

This module connects:

    primary electron -> sample impact -> surface sampler -> emitted electron tracking

It does not implement full cascade emission yet.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    e_charge,
    m_e,
    speed_from_energy_eV,
    kinetic_energy_eV_from_velocity,
)

from .fields import (
    E_at_point,
    potential_at_point,
)

from .collisions import (
    first_segment_hit,
    segment_near_any_stl_box,
    segment_hits_sample_plane,
    sample_plane_coordinates,
    nearest_hit,
)

from .trajectories import (
    unit,
    classify_grid_point,
    classify_effective_vacuum_point,
    advance_through_sample_voxel_artifact,
    choose_adaptive_dt_with_surfaces,
    integrate_one_electron,
)

from .samplers import generate_surface_emissions


# ============================================================
# Primary beam generation
# ============================================================

def make_primary_beam_near_sample(
    N: int,
    E0_eV: float,
    field: dict,
    Phi_interp,
    x_start: float | None = 1.0e-3,
    y0: float = 0.0,
    z0: float = 0.0,
    beam_sigma: float = 150e-6,
    energy_spread_eV: float = 0.0,
    angular_sigma_deg: float = 0.0,
    sample_voltage: float = 0.0,
    rng=None,
    sample_geometry: dict | None = None,
    primary_launch_clearance_h: float = 2.0,
    primary_launch_distance_m: float | None = None,
    primary_launch_retreat_step_h: float = 0.25,
    primary_launch_max_tries: int = 80,
    min_incidence_cos: float = 1.0e-3,
):
    """Generate primary electrons near the sample and flying toward it.

    With ``sample_geometry`` supplied, every electron is placed from its own
    beam-line intersection with the *continuous rotated sample plane*:

        1. Sample a finite beam coordinate (y, z) and direction.
        2. Intersect that individual beam line with the rotated sample plane.
        3. Move upstream along the same ray.  By default the normal clearance
           is ``primary_launch_clearance_h * field['h']``.  If
           ``primary_launch_distance_m`` is supplied, that explicit along-ray
           distance is used instead (useful for beam-steering studies).
        4. If voxelization still classifies the point as solid, retreat a
           little farther upstream along that same ray until free vacuum is
           found.

    Thus the field voxel size controls only the safety clearance, not the
    physical impact position.  At grazing incidence different transverse beam
    coordinates naturally produce different x positions at the sample.

    If ``sample_geometry`` is None, the legacy fixed-``x_start`` behavior is
    retained for backward compatibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    if N <= 0:
        raise ValueError("N must be positive")
    if primary_launch_clearance_h <= 0.0:
        raise ValueError("primary_launch_clearance_h must be positive")
    if primary_launch_distance_m is not None and primary_launch_distance_m <= 0.0:
        raise ValueError("primary_launch_distance_m must be positive when supplied")
    if primary_launch_retreat_step_h <= 0.0:
        raise ValueError("primary_launch_retreat_step_h must be positive")
    if primary_launch_max_tries <= 0:
        raise ValueError("primary_launch_max_tries must be positive")

    p0s = np.zeros((N, 3))
    v0s = np.zeros((N, 3))
    K0s = np.zeros(N)
    Phi0s = np.zeros(N)

    ys = y0 + beam_sigma * rng.standard_normal(N)
    zs = z0 + beam_sigma * rng.standard_normal(N)

    if energy_spread_eV > 0:
        E_land = np.maximum(
            E0_eV + energy_spread_eV * rng.standard_normal(N),
            1.0e-6,
        )
    else:
        E_land = E0_eV * np.ones(N)

    dirs = np.tile(np.array([-1.0, 0.0, 0.0]), (N, 1))

    if angular_sigma_deg > 0:
        sig = np.deg2rad(angular_sigma_deg)
        dirs[:, 1] += sig * rng.standard_normal(N)
        dirs[:, 2] += sig * rng.standard_normal(N)
        dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]

    h = float(field["h"])

    if sample_geometry is not None:
        C = np.asarray(sample_geometry["center"], dtype=float)
        n = unit(np.asarray(sample_geometry["normal"], dtype=float))
        normal_clearance = float(primary_launch_clearance_h) * h
        normal_retreat_step = float(primary_launch_retreat_step_h) * h
        explicit_ray_distance = (
            None if primary_launch_distance_m is None
            else float(primary_launch_distance_m)
        )

    for i in range(N):
        d = unit(dirs[i])

        if sample_geometry is None:
            if x_start is None:
                x_start = 0.75 * h
            p = np.array([float(x_start), ys[i], zs[i]], dtype=float)
        else:
            # The transverse beam distribution is defined in a plane
            # perpendicular to the nominal -X beam and passing through the
            # sample centre.  The intersection uses the full infinite beam
            # line, so it works whether the tilted surface at this (y,z) lies
            # at positive or negative x relative to the centre plane.
            p_ref = np.array([C[0], ys[i], zs[i]], dtype=float)
            denom = float(np.dot(n, d))
            mu = -denom  # positive for a ray entering the vacuum-side face

            if (not np.isfinite(mu)) or mu <= float(min_incidence_cos):
                raise RuntimeError(
                    "Primary beam is parallel to or directed away from the "
                    f"sample for electron {i}: -d·n={mu:.6g}. "
                    "Check sample_theta_deg/angular_sigma_deg."
                )

            s_hit = float(np.dot(n, C - p_ref) / denom)
            p_hit_geom = p_ref + s_hit * d

            # Move upstream along the *same electron ray*.  By default the
            # requested clearance is measured normal to the sample plane, so
            # dividing by mu makes that normal clearance independent of angle.
            # For beam-steering studies an explicit along-ray launch distance
            # can instead be supplied, allowing the primary to start much
            # farther upstream while retaining backward-compatible defaults.
            if explicit_ray_distance is None:
                launch_ray_distance = normal_clearance / mu
                ds_ray = normal_retreat_step / mu
            else:
                launch_ray_distance = explicit_ray_distance
                ds_ray = normal_retreat_step

            p = p_hit_geom - launch_ray_distance * d

            cls = classify_effective_vacuum_point(p, field)
            if cls["status"] != "free":
                p_base = p.copy()
                found = False

                for attempt in range(1, primary_launch_max_tries + 1):
                    candidate = p_base - attempt * ds_ray * d
                    cls = classify_effective_vacuum_point(candidate, field)

                    if cls["status"] == "free":
                        p = candidate
                        found = True
                        break

                    if cls["status"] in {"left_grid", "left_update_region"}:
                        break

                if not found:
                    _, sample_u, sample_v = sample_plane_coordinates(
                        p_hit_geom, sample_geometry
                    )
                    raise RuntimeError(
                        "Could not place primary in free vacuum upstream of "
                        f"rotated sample (electron {i}, theta="
                        f"{sample_geometry.get('theta_deg', np.nan):.3f} deg, "
                        f"sample_u={sample_u:.6g} m, sample_v={sample_v:.6g} m, "
                        f"last_status={cls.get('status')!r})."
                    )

        Phi_start = potential_at_point(p, Phi_interp)

        if not np.isfinite(Phi_start):
            raise RuntimeError(f"Phi_start is NaN at p={p}")

        K_start = E_land[i] + (Phi_start - sample_voltage)
        K_start = max(K_start, 1.0e-6)

        vmag = speed_from_energy_eV(K_start)

        p0s[i] = p
        v0s[i] = vmag * d
        K0s[i] = K_start
        Phi0s[i] = Phi_start

    return p0s, v0s, K0s, Phi0s


def sample_center_from_bounds(sample_y_bounds, sample_z_bounds):
    """
    Return y0,z0 sample center from sample bounds.
    """
    y0 = 0.5 * (sample_y_bounds[0] + sample_y_bounds[1])
    z0 = 0.5 * (sample_z_bounds[0] + sample_z_bounds[1])

    return y0, z0


# ============================================================
# Impact angle helpers
# ============================================================

def sample_outward_normal(theta_deg: float = 0.0) -> np.ndarray:
    """
    Sample outward normal.

    For the current aligned geometry, sample outward normal toward the RFA
    is +X. The optional theta_deg rotates it around Z.
    """
    a = np.deg2rad(theta_deg)

    R_z = np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a),  np.cos(a), 0.0],
        [0.0,        0.0,       1.0],
    ])

    n = R_z @ np.array([1.0, 0.0, 0.0])

    return n / np.linalg.norm(n)


def impact_cos_theta(
    v_in,
    n_out,
    cos_min: float = 0.05,
) -> float:
    """
    Cosine of incident angle relative to outward normal.

    Incoming primary velocity points into the surface, so:

        cos(theta) = -vhat dot n_out
    """
    vhat = unit(v_in)
    nhat = unit(n_out)

    cos_theta = -np.dot(vhat, nhat)

    return max(cos_min, float(cos_theta))


# ============================================================
# Primary trajectory to sample
# ============================================================

def fly_primary_to_sample(
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
    stl_boxes=None,
    sample_y_bounds=None,
    sample_z_bounds=None,
    x_sample: float = 0.0,
    dt: float = 1.0e-12,
    max_steps: int = 20000,
    adaptive_dt: bool = True,
    dt_min: float = 1.0e-13,
    dt_max: float = 2.0e-11,
    max_step_fraction_of_h: float = 0.10,
    sample_owner_id: int = 1,
    sample_geometry: dict | None = None,
    track_points: bool = False,
    track_stride: int = 1,
):
    """Track one primary until it hits the finite oriented sample or an STL.

    ``sample_geometry`` is the preferred path.  The segment/plane test is then
    fully orientation independent; there is no global-x crossing condition.
    The old x=0 behavior remains available when no geometry is supplied.

    Point-by-point trajectory storage is optional.  With ``track_points=False``
    (the default), only the state needed by the integrator/collision logic is
    retained.  With tracking enabled, the launch point, every ``track_stride``
    accepted integration steps, and exact terminal hit locations are stored.
    """
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

    # Physics/collision logic occasionally needs the immediately preceding
    # vacuum point.  Keep it independently of optional trajectory storage.
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

    q_over_m = -e_charge / m_e

    for step in range(max_steps):
        cls = classify_grid_point(p, field)

        if cls["status"] != "free":
            if (
                cls["status"] == "hit_fixed"
                and cls.get("owner_id", None) == sample_owner_id
                and sample_geometry is not None
            ):
                # The fixed sample voxels exist to impose the electrostatic
                # boundary.  At grazing incidence their staircase can protrude
                # into physical vacuum, so it must not become the collision
                # surface.  Traverse only that local artifact while continuing
                # to test the continuous finite sample plane.
                bypass = advance_through_sample_voxel_artifact(
                    p=p,
                    v=v,
                    field=field,
                    sample_geometry=sample_geometry,
                    sample_owner_id=sample_owner_id,
                )

                bypass_end = np.asarray(bypass["point"], dtype=float)
                hit_sample_bypass = bypass.get("hit_info", None)
                hit_stl_bypass = None

                if (
                    stl_boxes is None
                    or segment_near_any_stl_box(p, bypass_end, stl_boxes)
                ):
                    hit_stl_bypass = first_segment_hit(
                        p,
                        bypass_end,
                        intersector,
                        face_owner,
                        collision_mesh,
                    )

                hit = hit_stl_bypass if hit_stl_bypass is not None else hit_sample_bypass

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
                            "hit_sample" if hit.get("kind") == "sample_plane"
                            else "hit_stl"
                        ),
                        "hit_info": hit,
                        "traj": traj_out,
                        "vel": vel_out,
                        "steps": step,
                    }

                if bypass["status"] == "cleared":
                    p_previous = p.copy()
                    p = bypass_end
                    append_track(p, v, force=True)
                    continue

                # Reaching this branch means the local staircase could not be
                # cleared within the conservative search window.  Keep it
                # explicit rather than silently turning a voxel artifact into a
                # physical sample hit.
                fail_info = dict(cls)
                fail_info.update({
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
                    "hit_info": fail_info,
                    "traj": traj_out,
                    "vel": vel_out,
                    "steps": step,
                }

            if cls["status"] == "hit_fixed" and cls.get("owner_id", None) == sample_owner_id:
                # Legacy behavior when no continuous sample geometry is supplied.
                hit = {
                    "kind": "sample_voxel",
                    "location": p.copy(),
                    "distance": 0.0,
                    "owner": "sample",
                    "owner_name": "sample",
                    "owner_id": sample_owner_id,
                    "normal": np.array([1.0, 0.0, 0.0]),
                    "grid_classification": dict(cls),
                    "KE_hit_eV": kinetic_energy_eV_from_velocity(v),
                    "v_in": v.copy(),
                }
                if p_previous is not None:
                    hit["p_before"] = p_previous.copy()
                append_track(hit["location"], v, force=True)
                traj_out, vel_out = packed_track()
                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": traj_out,
                    "vel": vel_out,
                    "steps": step,
                }

            append_track(p, v, force=True)
            traj_out, vel_out = packed_track()
            return {
                "reason": cls["status"],
                "hit_info": cls,
                "traj": traj_out,
                "vel": vel_out,
                "steps": step,
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

        E = E_at_point(p, Ex_interp, Ey_interp, Ez_interp)
        a = q_over_m * E

        p_new = p + v * dt_step + 0.5 * a * dt_step**2

        E_new = E_at_point(p_new, Ex_interp, Ey_interp, Ez_interp)
        a_new = q_over_m * E_new
        v_new = v + 0.5 * (a + a_new) * dt_step

        hit_sample = segment_hits_sample_plane(
            p,
            p_new,
            x_sample=x_sample,
            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,
            sample_geometry=sample_geometry,
        )

        hit_stl = None

        if stl_boxes is None or segment_near_any_stl_box(p, p_new, stl_boxes):
            hit_stl = first_segment_hit(
                p,
                p_new,
                intersector,
                face_owner,
                collision_mesh,
            )

        hit = nearest_hit(hit_sample, hit_stl)

        if hit is not None:
            hit = dict(hit)
            t_hit = float(hit.get("t", 1.0))
            t_hit = float(np.clip(t_hit, 0.0, 1.0))
            v_hit = v + t_hit * (v_new - v)

            append_track(hit["location"], v_hit, force=True)

            if hit["kind"] == "sample_plane":
                hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_hit)
                hit["v_in"] = v_hit.copy()
                hit["p_before"] = p.copy()

                traj_out, vel_out = packed_track()
                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": traj_out,
                    "vel": vel_out,
                    "steps": step + 1,
                }

            if hit["kind"] == "stl":
                hit["KE_hit_eV"] = kinetic_energy_eV_from_velocity(v_hit)
                hit["v_in"] = v_hit.copy()
                hit["p_before"] = p.copy()

                traj_out, vel_out = packed_track()
                return {
                    "reason": "hit_stl",
                    "hit_info": hit,
                    "traj": traj_out,
                    "vel": vel_out,
                    "steps": step + 1,
                }

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
        "steps": max_steps,
    }


# ============================================================
# One-primary first-generation sample-emission runner
# ============================================================

def run_one_primary_with_model_emission(
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
    sample_normal=None,
    sample_launch_eps=None,
    surface_skip_eps: float = 1.0e-6,
    emitted_max_steps: int = 20000,
    emitted_dt_max: float = 2.0e-11,
    emitted_max_step_fraction_of_h: float = 0.40,
    sample_geometry: dict | None = None,
):
    """
    Run one primary electron and track first-generation sample emissions.

    This does not yet implement full cascade emission from subsequent
    grid/collector/holder/sample hits.

    Parameters
    ----------
    sample_launch_eps:
        Offset for launching emitted electrons away from the sample.
        If None, uses 0.75 * field["h"] to avoid starting inside fixed
        sample voxels.
    surface_skip_eps:
        Small offset for stepping past transmitted grid shells.

    Returns
    -------
    primary:
        Primary trajectory result.
    emitted_results:
        List of emitted-electron trajectory results.
    """
    if sample_normal is None:
        if sample_geometry is not None:
            sample_normal = unit(np.asarray(sample_geometry["normal"], dtype=float))
        else:
            sample_normal = np.array([1.0, 0.0, 0.0])

    if sample_launch_eps is None:
        sample_launch_eps = 0.75 * float(field["h"])

    primary = fly_primary_to_sample(
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
    )

    emitted_results = []

    if primary["reason"] != "hit_sample":
        return primary, emitted_results

    hit = primary["hit_info"]

    p_hit = hit["location"]
    E_inc_eV = hit["KE_hit_eV"]
    v_in = hit["v_in"]

    emitted_electrons, _emission_event_info = generate_surface_emissions(
        surface_name="sample",
        r_hit=p_hit,
        v_in=v_in,
        n_out=unit(hit.get("normal", sample_normal)),
        Einc=E_inc_eV,
        yield_models=yield_models,
        energy_models=energy_models,
        theta_models=theta_models,
        voltages=voltages,
        rng=rng,
        origin="gun",
        sample_launch_eps=sample_launch_eps,
        U0=15.0,
    )

    for e in emitted_electrons:
        res_emit = integrate_one_electron(
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
            dt=1.0e-12,
            max_steps=emitted_max_steps,
            surface_eps=surface_skip_eps,
            grid_transparency=grid_transparency,
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
            sample_geometry=sample_geometry,
        )

        res_emit["E_emit_eV"] = e["E_emit_eV"]
        res_emit["emission_kind"] = e["kind"]
        res_emit["primary_E_inc_eV"] = E_inc_eV
        res_emit["primary_cos_theta"] = e["cos_theta"]

        emitted_results.append(res_emit)

    return primary, emitted_results
