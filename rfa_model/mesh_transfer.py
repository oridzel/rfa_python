"""
Cell-scale mesh field: transfer function for grid crossings.

The digital twin solves the field on a grid much coarser than the mesh
(0.2 mm vs 0.508 mm pitch and 25 um wires), so each grid is a smooth
equipotential shell.  In reality the field leaks through the openings: the
potential in an opening differs from the wire potential by ~ dE p ln(p/pi a)/(4 pi)
(0.3-1 V here), and within ~p/2pi of the mesh the field has transverse
components comparable to the applied field.  Fast electrons are not affected,
but slow electrons crossing a mesh with a field on one side, or turned back
just beyond it, pick up transverse energy of the order of that potential
variation and are intercepted more often than the open-area fraction suggests.

This module applies that effect at every crossing of a grid shell, from
tables computed with a one-cell model of the 50x50 woven tungsten mesh
(see make_mesh_tables.py).  Everything scales with

    kappa = K / (e * dE * p)

K: kinetic energy at the crossing (eV); dE: field difference across the mesh
(V/m); p: pitch.  Two mesh types by the sign of the mesh charge:
  "L": field side at lower potential than the wires (e.g. G1 in RG mode);
       an electron entering the field side with kappa <= 0.8 turns back within
       the mesh field and is treated as one round trip.
  "H": field side at higher potential (e.g. G2 in RG mode seen from G1, G3
       toward the collector).

Enabled with grid_transparency["mesh_transfer"] = True (default tables) or a
dict with keys "tables" (path), "kappa_max" (default 40), "delta_m" (radial
offset used to read the field on each side, default 0.6 mm).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .fields import E_at_point

_QM = 1.75882001e11
_DEFAULT_TABLES = Path(__file__).with_name("data") / "mesh_transfer_tables.json"
SHELL_RADIUS_KEY = {"g1_shell": "R_g1", "g2_shell": "R_g2", "g3_shell": "R_g3"}


@lru_cache(maxsize=4)
def _load(path: str):
    with open(path) as f:
        t = json.load(f)
    q = np.asarray(t["quantile_levels"], float)

    def pack(rows, drop_empty=False):
        rows = [r for r in rows if not (drop_empty and r["survival"] <= 0.0)]
        k = np.array([r["kappa"] for r in rows], float)
        s = np.array([r["survival"] for r in rows], float)
        et = np.array([r["Et_quantiles"] for r in rows], float)
        return dict(logk=np.log(k), S=s, Et=et)

    L_s = pack(t["L_single"], drop_empty=True)
    H_s = pack(t["H_single"])
    S_inf = 0.5 * (L_s["S"][-1] + H_s["S"][-1])
    return dict(q=q, kappa_rt=float(t["kappa_round_trip_max"]), pitch=float(t["pitch_m"]),
                L_rt=pack(t["L_round_trip"]), L_single=L_s, H_single=H_s, S_inf=S_inf)


def spec_from(grid_transparency):
    """Return the parsed mesh-transfer settings, or None when disabled."""
    if not isinstance(grid_transparency, dict):
        return None
    s = grid_transparency.get("mesh_transfer")
    if not s:
        return None
    s = {} if s is True else dict(s)
    tables = _load(str(s.get("tables", _DEFAULT_TABLES)))
    return dict(tables=tables, kappa_max=float(s.get("kappa_max", 40.0)),
                delta_m=float(s.get("delta_m", 6.0e-4)))


def _interp_row(tab, kappa):
    """Survival and the E_t quantile vector at kappa (log-kappa interpolation, clamped)."""
    lk = np.clip(np.log(max(kappa, 1e-12)), tab["logk"][0], tab["logk"][-1])
    S = float(np.interp(lk, tab["logk"], tab["S"]))
    j = int(np.clip(np.searchsorted(tab["logk"], lk), 1, len(tab["logk"]) - 1))
    w = (lk - tab["logk"][j - 1]) / (tab["logk"][j] - tab["logk"][j - 1])
    et = (1 - w) * tab["Et"][j - 1] + w * tab["Et"][j]
    return S, et


def crossing_info(spec, owner, location, v, field, Ex_interp, Ey_interp, Ez_interp):
    """Classify one shell crossing.  Returns None when the mesh field is negligible."""
    if owner not in SHELL_RADIUS_KEY:
        return None
    r = np.asarray(location, float)
    rn = float(np.linalg.norm(r))
    if rn <= 0.0:
        return None
    rh = r / rn
    d = spec["delta_m"]
    try:
        Ein = float(np.dot(np.asarray(E_at_point(rh * (rn - d), Ex_interp, Ey_interp, Ez_interp), float), rh))
        Eout = float(np.dot(np.asarray(E_at_point(rh * (rn + d), Ex_interp, Ey_interp, Ez_interp), float), rh))
    except Exception:
        return None
    dE = Eout - Ein                    # = surface charge / eps0
    if not np.isfinite(dE) or abs(dE) < 1.0:
        return None
    vv = np.asarray(v, float)
    K = 0.5 * float(np.dot(vv, vv)) / _QM          # eV
    tabs = spec["tables"]
    kappa = K / (abs(dE) * tabs["pitch"])
    if kappa > spec["kappa_max"]:
        return None
    mesh_type = "L" if dE > 0 else "H"
    field_outer = abs(Eout) >= abs(Ein)
    outward = float(np.dot(vv, rh)) > 0.0
    return dict(kappa=kappa, type=mesh_type, gp=abs(dE) * tabs["pitch"],
                enters_field_side=(outward == field_outer), r_hat=rh)


def round_trip_survival_factor(spec, info):
    """Round-trip survival from the table relative to two plain crossings far from
    threshold; multiplies T_model**2 at the first crossing."""
    t = spec["tables"]
    S, _ = _interp_row(t["L_rt"], info["kappa"])
    return S / t["S_inf"] ** 2


def opacity_ratio(spec, info):
    t = spec["tables"]
    tab = t["L_single"] if info["type"] == "L" else t["H_single"]
    S, _ = _interp_row(tab, max(info["kappa"], np.exp(tab["logk"][0])))
    return (1.0 - S) / (1.0 - t["S_inf"])


def apply_kick(spec, info, v, rng, round_trip=False):
    """Add a transverse velocity drawn from the table and restore the speed."""
    t = spec["tables"]
    if round_trip:
        tab = t["L_rt"]
    else:
        tab = t["L_single"] if info["type"] == "L" else t["H_single"]
    _, et = _interp_row(tab, max(info["kappa"], np.exp(tab["logk"][0])))
    Et = float(np.interp(rng.random(), t["q"], et)) * info["gp"]     # eV
    if Et <= 0.0:
        return v
    dv = np.sqrt(2.0 * _QM * Et)
    n = info["r_hat"]
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    ph = 2 * np.pi * rng.random()
    vv = np.asarray(v, float)
    speed = float(np.linalg.norm(vv))
    w = vv + dv * (np.cos(ph) * t1 + np.sin(ph) * t2)
    wn = float(np.linalg.norm(w))
    return w * (speed / wn) if wn > 0 else vv
