"""
constants.py

Physical constants used by the RFA model.
"""

from __future__ import annotations

import numpy as np


e_charge = 1.602176634e-19
m_e = 9.1093837015e-31
COLLECTOR_OPENING_ALPHA_DEG = 11.8


def speed_from_energy_eV(E_eV: float | np.ndarray) -> float | np.ndarray:
    """
    Electron speed from kinetic energy in eV.
    Nonrelativistic.
    """
    return np.sqrt(2.0 * np.asarray(E_eV) * e_charge / m_e)


def kinetic_energy_eV_from_velocity(v: np.ndarray) -> float:
    """
    Electron kinetic energy in eV from velocity vector.
    """
    v = np.asarray(v, dtype=float)
    return float(0.5 * m_e * np.dot(v, v) / e_charge)


def kinetic_energy_array_eV(vel: np.ndarray) -> np.ndarray:
    """
    Electron kinetic energy in eV for an array of velocities.
    """
    vel = np.asarray(vel, dtype=float)
    return 0.5 * m_e * np.sum(vel**2, axis=1) / e_charge