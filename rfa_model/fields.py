"""
fields.py

Electric-field and potential interpolation utilities for the RFA model.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .constants import kinetic_energy_array_eV


def attach_default_owner_name_map(field: dict) -> dict:
    """
    Attach default owner-id to physical-name mapping for the current RFA field.

    This mapping matches the current voxelized field convention:

        1  sample
        2  holder
        3  receiver
        4  rod
        5  g1frame
        6  g2frame
        7  g3frame
        8  drifttube
        9  g1_shell
        10 g2_shell
        11 g3_shell
        12 collector_shell

    Returns the same field dictionary, modified in place.
    """
    field["owner_name_map"] = {
        0: "free",

        1: "sample",
        2: "holder",
        3: "receiver",
        4: "rod",

        5: "g1frame",
        6: "g2frame",
        7: "g3frame",

        8: "drifttube",

        9: "g1_shell",
        10: "g2_shell",
        11: "g3_shell",
        12: "collector_shell",
    }

    return field


def build_field_interpolators(field: dict):
    """
    Build interpolators for Ex, Ey, Ez.

    Parameters
    ----------
    field:
        Field dictionary containing x, y, z, Ex, Ey, Ez.

    Returns
    -------
    Ex_interp, Ey_interp, Ez_interp
    """
    x = field["x"]
    y = field["y"]
    z = field["z"]

    Ex_interp = RegularGridInterpolator(
        (x, y, z),
        field["Ex"],
        bounds_error=False,
        fill_value=0.0,
    )

    Ey_interp = RegularGridInterpolator(
        (x, y, z),
        field["Ey"],
        bounds_error=False,
        fill_value=0.0,
    )

    Ez_interp = RegularGridInterpolator(
        (x, y, z),
        field["Ez"],
        bounds_error=False,
        fill_value=0.0,
    )

    return Ex_interp, Ey_interp, Ez_interp


def build_potential_interpolator(field: dict):
    """
    Build interpolator for electrostatic potential V.

    Parameters
    ----------
    field:
        Field dictionary containing x, y, z, V.

    Returns
    -------
    Phi_interp
    """
    return RegularGridInterpolator(
        (field["x"], field["y"], field["z"]),
        field["V"],
        bounds_error=False,
        fill_value=np.nan,
    )


def E_at_point(
    p: np.ndarray,
    Ex_interp,
    Ey_interp,
    Ez_interp,
) -> np.ndarray:
    """
    Evaluate electric field at one point.

    Returns
    -------
    np.ndarray, shape (3,)
    """
    pp = np.asarray(p, dtype=float).reshape(1, 3)

    return np.array([
        Ex_interp(pp)[0],
        Ey_interp(pp)[0],
        Ez_interp(pp)[0],
    ], dtype=float)


def potential_at_point(p: np.ndarray, Phi_interp) -> float:
    """
    Evaluate electrostatic potential at one point.
    """
    pp = np.asarray(p, dtype=float).reshape(1, 3)
    return float(Phi_interp(pp)[0])


def trajectory_energy_diagnostic(res: dict, Phi_interp) -> dict:
    """
    Check approximate energy conservation along one trajectory.

    For an electron in electrostatic potential Phi, the conserved
    quantity in eV is approximately:

        H = K_eV - Phi_V

    because the electron charge is -e.

    Parameters
    ----------
    res:
        Trajectory result dictionary from integrate_one_electron.
        Must contain "traj" and "vel".
    Phi_interp:
        Potential interpolator.

    Returns
    -------
    dict with K_eV, Phi, H, H_drift.
    """
    traj = np.asarray(res["traj"], dtype=float)
    vel = np.asarray(res["vel"], dtype=float)

    K_eV = kinetic_energy_array_eV(vel)

    Phi = np.array([
        potential_at_point(p, Phi_interp)
        for p in traj
    ])

    H = K_eV - Phi

    return {
        "K_eV": K_eV,
        "Phi": Phi,
        "H": H,
        "H_drift": H - H[0],
    }


def summarize_energy_drift(res: dict, Phi_interp) -> dict:
    """
    Compact energy-conservation summary for one trajectory.
    """
    diag = trajectory_energy_diagnostic(res, Phi_interp)

    H_drift = diag["H_drift"]

    return {
        "K_start_eV": float(diag["K_eV"][0]),
        "K_end_eV": float(diag["K_eV"][-1]),
        "Phi_start_V": float(diag["Phi"][0]),
        "Phi_end_V": float(diag["Phi"][-1]),
        "H_start_eV": float(diag["H"][0]),
        "H_end_eV": float(diag["H"][-1]),
        "H_drift_min_eV": float(np.nanmin(H_drift)),
        "H_drift_max_eV": float(np.nanmax(H_drift)),
        "H_drift_final_eV": float(H_drift[-1]),
    }