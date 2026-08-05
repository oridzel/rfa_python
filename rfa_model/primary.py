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
    nearest_hit,
)

from .trajectories import (
    unit,
    classify_grid_point,
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
    x_start: float = 1.0e-3,
    y0: float = 0.0,
    z0: float = 0.0,
    beam_sigma: float = 150e-6,
    energy_spread_eV: float = 0.0,
    angular_sigma_deg: float = 0.0,
    sample_voltage: float = 0.0,
    rng=None,
):
    """
    Generate primary electrons starting near the sample and flying toward -X.

    E0_eV is the desired landing energy at the sample potential.

    For an electron, kinetic energy evolves as:

        K2 = K1 + Phi2 - Phi1

    Therefore, if the desired landing energy at Phi_sample is E0_eV,
    the starting kinetic energy is:

        K_start = E0_eV + Phi_start - Phi_sample

    Parameters
    ----------
    N:
        Number of primary electrons.
    E0_eV:
        Desired landing energy at sample, eV.
    field:
        Field dictionary.
    Phi_interp:
        Potential interpolator.
    x_start:
        Primary start x coordinate, normally small positive x.
    y0, z0:
        Beam center.
    beam_sigma:
        Gaussian beam sigma in y and z.
    energy_spread_eV:
        Optional Gaussian landing-energy spread.
    angular_sigma_deg:
        Optional small angular spread around -X.
    sample_voltage:
        Sample potential, usually Vs.
    rng:
        NumPy random generator.

    Returns
    -------
    p0s, v0s, K0s, Phi0s
    """
    if rng is None:
        rng = np.random.default_rng()

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

    for i in range(N):
        p = np.array([x_start, ys[i], zs[i]], dtype=float)

        Phi_start = potential_at_point(p, Phi_interp)

        if not np.isfinite(Phi_start):
            raise RuntimeError(f"Phi_start is NaN at p={p}")

        K_start = E_land[i] + (Phi_start - sample_voltage)
        K_start = max(K_start, 1.0e-6)

        vmag = speed_from_energy_eV(K_start)

        p0s[i] = p
        v0s[i] = vmag * dirs[i]
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
):
    """
    Track one primary electron until it hits the analytic sample plane
    or another terminal object.

    The sample plane is checked analytically before STL collision so that
    primary hits on the intended sample face are robust.

    Returns
    -------
    result dict with:
        reason
        hit_info
        traj
        vel
        steps
    """
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()

    traj = [p.copy()]
    vel = [v.copy()]

    q_over_m = -e_charge / m_e

    for step in range(max_steps):
        cls = classify_grid_point(p, field)

        if cls["status"] != "free":
            # If the primary has entered the sample fixed-voxel layer,
            # count this as a sample impact. The finite field grid marks
            # sample voxels slightly in front of the analytic x=0 plane.
            if cls["status"] == "hit_fixed" and cls.get("owner_id", None) == sample_owner_id:
                hit = {
                    "kind": "sample_voxel",
                    "location": p.copy(),
                    "distance": 0.0,
                    "owner": "sample",
                    "normal": np.array([1.0, 0.0, 0.0]),
                    "grid_classification": cls,
                    "KE_hit_eV": kinetic_energy_eV_from_velocity(v),
                    "v_in": v.copy(),
                }

                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "steps": step,
                }

            return {
                "reason": cls["status"],
                "hit_info": cls,
                "traj": np.asarray(traj),
                "vel": np.asarray(vel),
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

        # Primary sample hit: primary travels from +X toward x=0/-X.
        hit_sample = None

        if p[0] > x_sample and p_new[0] <= x_sample:
            hit_sample = segment_hits_sample_plane(
                p,
                p_new,
                x_sample=x_sample,
                sample_y_bounds=sample_y_bounds,
                sample_z_bounds=sample_z_bounds,
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
            traj.append(hit["location"].copy())
            vel.append(v_new.copy())

            if hit["kind"] == "sample_plane":
                KE_hit_eV = kinetic_energy_eV_from_velocity(v_new)

                hit = dict(hit)
                hit["KE_hit_eV"] = KE_hit_eV
                hit["v_in"] = v_new.copy()

                return {
                    "reason": "hit_sample",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "steps": step + 1,
                }

            if hit["kind"] == "stl":
                KE_hit_eV = kinetic_energy_eV_from_velocity(v_new)

                hit = dict(hit)
                hit["KE_hit_eV"] = KE_hit_eV
                hit["v_in"] = v_new.copy()

                return {
                    "reason": "hit_stl",
                    "hit_info": hit,
                    "traj": np.asarray(traj),
                    "vel": np.asarray(vel),
                    "steps": step + 1,
                }

        p = p_new
        v = v_new

        traj.append(p.copy())
        vel.append(v.copy())

    return {
        "reason": "max_steps",
        "hit_info": None,
        "traj": np.asarray(traj),
        "vel": np.asarray(vel),
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

    emitted_electrons = generate_surface_emissions(
        surface_name="sample",
        r_hit=p_hit,
        v_in=v_in,
        n_out=sample_normal,
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
        )

        res_emit["E_emit_eV"] = e["E_emit_eV"]
        res_emit["emission_kind"] = e["kind"]
        res_emit["primary_E_inc_eV"] = E_inc_eV
        res_emit["primary_cos_theta"] = e["cos_theta"]

        emitted_results.append(res_emit)

    return primary, emitted_results
