"""Utilities for exporting correlated oblique-incidence emission samplers.

The RFA sampler loader accepts one pickle-free NPZ per emitted population:

    BSEJointFromPlaneSampler_uncoatedCuFPA.npz
    SEJointFromPlaneSampler_uncoatedCuFPA.npz

Each row is one *escaped* electron.  The important rule is that Eout and the
three direction components stay on the same row; do not independently sort or
CDF-transform them.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def _unit_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("v_out must have shape (N, 3)")
    n = np.linalg.norm(v, axis=1)
    if np.any(~np.isfinite(n)) or np.any(n <= 0.0):
        raise ValueError("v_out contains non-finite or zero vectors")
    return v / n[:, None]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        raise ValueError("axis/normal must be a three-vector")
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("axis/normal must be finite and nonzero")
    return v / n


def rfa_oblique_basis(
    sample_normal,
    beam_back_axis=(1.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the RFA local basis (beam_back, toward_normal, side)."""
    axis = _unit(beam_back_axis)
    n_out = _unit(sample_normal)

    c = float(np.dot(axis, n_out))
    if c < -1.0e-12:
        raise ValueError("sample normal opposes beam_back_axis")
    c = float(np.clip(c, 0.0, 1.0))

    transverse = n_out - c * axis
    s = float(np.linalg.norm(transverse))
    if s <= 1.0e-14:
        raise ValueError(
            "oblique basis is undefined at normal incidence; use the existing "
            "rotationally symmetric sampler there"
        )

    toward_normal = transverse / s
    side = np.cross(axis, toward_normal)
    side = side / np.linalg.norm(side)
    return axis, toward_normal, side


def write_joint_sampler_npz(
    path,
    *,
    Einc_eV,
    Eout_eV,
    v_out,
    sample_normal,
    beam_back_axis=(1.0, 0.0, 0.0),
) -> Path:
    """Write one correlated SE or BSE event sampler.

    Parameters
    ----------
    Einc_eV:
        Scalar incident energy for all rows, or an array of length N.
    Eout_eV:
        Escaped-electron kinetic energy at the physical sample surface.
    v_out:
        Escaped-electron velocity/direction vectors in the same Cartesian frame
        as ``sample_normal`` and ``beam_back_axis``.  Magnitude is ignored.
    sample_normal:
        Vacuum-side sample normal for this incidence angle.
    beam_back_axis:
        Fixed direction back toward the gun.  In the current RFA convention it
        is +X = (1, 0, 0).
    """
    path = Path(path)
    Eout = np.asarray(Eout_eV, dtype=float).reshape(-1)
    dirs = _unit_rows(v_out)
    n = len(Eout)
    if len(dirs) != n or n == 0:
        raise ValueError("Eout_eV and v_out must contain the same nonzero N")

    Einc = np.asarray(Einc_eV, dtype=float)
    if Einc.ndim == 0:
        Einc = np.full(n, float(Einc), dtype=float)
    else:
        Einc = Einc.reshape(-1)
    if len(Einc) != n:
        raise ValueError("Einc_eV must be scalar or length N")

    if np.any(~np.isfinite(Einc)) or np.any(Einc <= 0.0):
        raise ValueError("Einc_eV must be finite and positive")
    if np.any(~np.isfinite(Eout)) or np.any(Eout <= 0.0):
        raise ValueError("Eout_eV must be finite and positive")

    axis, toward_normal, side = rfa_oblique_basis(
        sample_normal=sample_normal,
        beam_back_axis=beam_back_axis,
    )
    n_out = _unit(sample_normal)

    outward = dirs @ n_out
    if np.any(outward <= 1.0e-12):
        raise ValueError(
            f"input contains {np.count_nonzero(outward <= 1.0e-12)} "
            "inward/tangent directions; joint files must contain escaped "
            "vacuum-side electrons only"
        )

    u0 = dirs @ axis
    u1 = dirs @ toward_normal
    u2 = dirs @ side

    theta_deg = np.rad2deg(np.arccos(np.clip(u0, -1.0, 1.0)))
    phi_deg = np.rad2deg(np.arctan2(u2, u1))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        Einc_eV=Einc,
        Eout_eV=Eout,
        dir_beam_back=u0,
        dir_toward_normal=u1,
        dir_side=u2,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )
    return path


def sample_normal_from_rfa_tilt(tilt_deg: float) -> np.ndarray:
    """Current RFA sample normal for a rotation about +Z."""
    a = np.deg2rad(float(tilt_deg))
    return np.array([np.cos(a), np.sin(a), 0.0], dtype=float)
