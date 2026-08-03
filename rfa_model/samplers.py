"""
samplers.py

Surface-yield, emitted-energy, emitted-angle, and secondary-emission
sampling utilities for the RFA model.
"""

from __future__ import annotations

from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .constants import speed_from_energy_eV
from .trajectories import unit


GRID_SURFACES = {
    # canonical analytic grid mesh names
    "g1mesh",
    "g2mesh",
    "g3mesh",

    # shell aliases
    "g1_shell",
    "g2_shell",
    "g3_shell",

    # canonical frame names
    "g1frame",
    "g2frame",
    "g3frame",

    # possible aliases from STL names
    "g1_frame",
    "g2_frame",
    "g3_frame",

    "g1_low_frame",
    "g1_upper_frame",
    "g2_low_frame",
    "g2_upper_frame",
    "g3_low_frame",
    "g3_upper_frame",

    # possible electrode labels
    "grid1",
    "grid2",
    "grid3",
    "grid",
}

COLLECTOR_SURFACES = {
    "collector",
    "collector_shell",
}

# Every surface carrying the airbrushed colloidal-graphite coating.
#
# Per the instrument paper: "The upper and lower collector shells, sample
# support rod, and the outer surface of the drift tube were coated with
# graphite". All three therefore share ONE physical coating and, at a given
# impact energy and angle, one SEY and one BSEY.
#
# This is why the yield multipliers below default to acting on the whole set
# rather than on the collector alone. Before this change, bse_multiplier_for_
# surface() applied collector_BSE_mult only to COLLECTOR_SURFACES while
# surface_family() routed rod and drifttube to the SAME "collector" yield
# curves -- so a fitted multiplier silently gave the three identically-coated
# electrodes two different effective yields.
CARBON_COATED_SURFACES = COLLECTOR_SURFACES | {
    "rod",
    "drifttube",
}

# Analytic zero-thickness spheres standing in for the woven wire mesh.
#
# These are the ONLY surfaces excluded from the incidence-angle yield law.
# Two independent reasons:
#   1. The normal used for a mesh hit is the sphere's radial direction, which
#      is not the local normal of a wire. The true incidence angle on the wire
#      depends on where around the wire circumference the electron lands, and
#      that information does not exist in the analytic-sphere representation.
#   2. The grid yield curves are JMONSEL "FromWire" tables, which already
#      average over wire geometry. Applying a secant law on top would
#      double-count the geometric enhancement.
#
# Grid FRAMES are NOT in this set: they are real STL solids with genuine face
# normals from collisions.py, so they take the angular law like any other solid.
ANALYTIC_MESH_SURFACES = {
    "g1mesh",
    "g2mesh",
    "g3mesh",
}

# Cap on the 1/cos(theta) incidence-angle yield enhancement.
#
# The secant law diverges at grazing incidence, while real surfaces roll over
# because of roughness and finite escape depth. The previous cos_theta floor of
# 0.05 permitted a 20x enhancement, which would put the Poisson mean for a
# measured C-on-SS SEY of ~0.72 at ~14 secondaries from a single grazing hit.
# That matters now that the drift tube is real STL geometry: grazing incidence
# on a tube bore is the common case, not the exception.
MAX_ANGULAR_YIELD_GAIN = 4.0

# Measured normal-incidence yields for the carbon coating on stainless steel.
#
# TEY is the arithmetic mean of all three 2026-07-28 measurements.
# BSEY is the arithmetic mean of measurements 1 and 2; measurement 3 was
# excluded from the BSEY average because of its isolated 400 eV outlier.
# SEY in the CSV is calculated consistently as TEY_mean_1_to_3 -
# BSEY_mean_1_to_2.
MEASURED_CARBON_BSEY_FILENAME = "BSEYFromPlane_measured_C_on_SS.csv"
MEASURED_CARBON_SEY_FILENAME = "SEYFromPlane_measured_C_on_SS.csv"

MEASURED_CARBON_SOURCE = (
    "C-on-SS measured 2026-07-28: TEY mean runs 1-3; "
    "BSEY mean runs 1-2; SEY=TEY-BSEY"
)


def canonical_surface_name_for_sey(surface_name):
    if surface_name is None:
        return "unknown"

    s = str(surface_name)

    aliases = {
        "grid1": "g1mesh",
        "grid2": "g2mesh",
        "grid3": "g3mesh",

        "g1_shell": "g1mesh",
        "g2_shell": "g2mesh",
        "g3_shell": "g3mesh",

        "g1_frame": "g1frame",
        "g2_frame": "g2frame",
        "g3_frame": "g3frame",

        "g1_low_frame": "g1frame",
        "g1_upper_frame": "g1frame",
        "g2_low_frame": "g2frame",
        "g2_upper_frame": "g2frame",
        "g3_low_frame": "g3frame",
        "g3_upper_frame": "g3frame",

        "collector_shell": "collector",
    }

    return aliases.get(s, s)


def orient_normal_against_incoming(n_out, v_in):
    """
    Orient ordinary surface normal so it faces the incoming electron.
    """
    n_out = unit(n_out)
    vhat = unit(v_in)

    if np.dot(vhat, n_out) > 0:
        n_out = -n_out

    return n_out


def _carbon_family_multiplier(
    canonical_name: str,
    shared: float | None,
    collector: float | None,
    rod: float | None,
    drifttube: float | None,
    fallback: float | None,
) -> float:
    """
    Resolve the multiplier for one carbon-coated surface.

    Precedence, most specific first:
        1. the per-surface override for this electrode
        2. the shared carbon-coating multiplier
        3. the global fallback multiplier
        4. 1.0

    Prefer the shared knob. The collector, rod, and drift tube carry the SAME
    graphite coating, so their true yields cannot differ except through impact
    energy and angle, which the sampler already handles. Per-surface overrides
    exist for diagnostics -- if the fit wants very different values for the
    three, that is evidence of a GEOMETRY error (e.g. the known rod-current
    deficit) being absorbed into a yield parameter, not of a real coating
    difference.
    """
    per_surface = {
        "collector": collector,
        "rod": rod,
        "drifttube": drifttube,
    }

    value = per_surface.get(canonical_name, None)

    if value is not None:
        return float(value)

    if shared is not None:
        return float(shared)

    if fallback is not None:
        return float(fallback)

    return 1.0


def bse_multiplier_for_surface(
    surface_name,
    BSE_mult: float = 1.0,
    grid_BSE_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    carbon_BSE_mult: float | None = None,
    rod_BSE_mult: float | None = None,
    drifttube_BSE_mult: float | None = None,
) -> float:
    """
    Surface-specific BSEY multiplier.

    Parameters
    ----------
    grid_BSE_mult:
        Applies to grid meshes and grid frames.
    carbon_BSE_mult:
        Applies to every graphite-coated electrode: collector, rod, and drift
        tube. This is the physically correct knob, since they share a coating.
    collector_BSE_mult, rod_BSE_mult, drifttube_BSE_mult:
        Per-electrode overrides that beat carbon_BSE_mult. collector_BSE_mult
        is kept for backward compatibility with earlier fits; note that using
        it ALONE scales the collector while leaving the identically-coated rod
        and drift tube at 1.0.
    BSE_mult:
        Global fallback for grid and carbon surfaces when nothing more
        specific is supplied.
    """
    s = canonical_surface_name_for_sey(surface_name)

    if s in GRID_SURFACES:
        return float(
            grid_BSE_mult if grid_BSE_mult is not None else BSE_mult
        )

    if s in CARBON_COATED_SURFACES:
        return _carbon_family_multiplier(
            canonical_name=s,
            shared=carbon_BSE_mult,
            collector=collector_BSE_mult,
            rod=rod_BSE_mult,
            drifttube=drifttube_BSE_mult,
            fallback=BSE_mult,
        )

    # sample, holder, receiver: not tuned here.
    return 1.0


def sey_multiplier_for_surface(
    surface_name,
    grid_SEY_mult: float | None = None,
    SEY_mult: float = 1.0,
    carbon_SEY_mult: float | None = None,
    collector_SEY_mult: float | None = None,
    rod_SEY_mult: float | None = None,
    drifttube_SEY_mult: float | None = None,
) -> float:
    """
    Surface-specific SEY multiplier.

    Mirrors bse_multiplier_for_surface. carbon_SEY_mult is the preferred knob
    for the graphite-coated electrodes (collector, rod, drift tube); the
    per-electrode overrides beat it.

    Note that collector SEY is NOT self-cancelling in this geometry: with the
    collector biased positive relative to the grounded grids, secondaries born
    on the collector are pulled back to it, but at +1 kV the grid-mesh field
    leak-through extracts a fraction of them inward. Scaling collector SEY is
    therefore a meaningful degree of freedom, which the earlier hard-coded 1.0
    denied.
    """
    s = canonical_surface_name_for_sey(surface_name)

    if s in GRID_SURFACES:
        return float(
            grid_SEY_mult if grid_SEY_mult is not None else SEY_mult
        )

    if s in CARBON_COATED_SURFACES:
        return _carbon_family_multiplier(
            canonical_name=s,
            shared=carbon_SEY_mult,
            collector=collector_SEY_mult,
            rod=rod_SEY_mult,
            drifttube=drifttube_SEY_mult,
            fallback=SEY_mult,
        )

    # sample, holder, receiver: not tuned here.
    return 1.0


def angular_yield_gain(
    cos_theta: float,
    max_gain: float = MAX_ANGULAR_YIELD_GAIN,
) -> float:
    """
    Capped secant-law incidence-angle yield enhancement.

    The measured C-on-SS tables and the JMONSEL "FromPlane" tables are all
    NORMAL-INCIDENCE yields, so using them at oblique incidence requires an
    explicit angular law. The standard escape-depth argument gives

        Y(theta) = Y(0) / cos(theta)

    because the primary deposits its energy over a path length that is longer
    by 1/cos(theta) within the shallow escape layer.

    The pure secant law is unbounded, so it is capped at max_gain. See
    MAX_ANGULAR_YIELD_GAIN for why the cap matters here specifically.

    Parameters
    ----------
    cos_theta:
        Cosine of the incidence angle measured against the vacuum-side normal.
        Non-finite or non-positive values return max_gain.
    max_gain:
        Upper bound on the enhancement factor.

    Returns
    -------
    Multiplicative gain in [1.0, max_gain].
    """
    max_gain = float(max_gain)

    if max_gain < 1.0:
        raise ValueError("max_gain must be >= 1.0")

    c = float(cos_theta)

    if not np.isfinite(c) or c <= 0.0:
        return max_gain

    return float(min(1.0 / c, max_gain))


# ============================================================
# CSV loaders
# ============================================================

def load_yield_curve_csv(path: str | Path) -> dict:
    """
    Load yield curve CSV.

    Expected format:
        line 1: description
        line 2: "beamE (eV)","SEY" or "BSEY"
        then data

    Returns
    -------
    dict with keys:
        E
        Y
    """
    path = Path(path)

    df = pd.read_csv(path, skiprows=1)

    E = np.asarray(df.iloc[:, 0], dtype=float)
    Y = np.asarray(df.iloc[:, 1], dtype=float)

    good = np.isfinite(E) & np.isfinite(Y)
    E = E[good]
    Y = Y[good]

    order = np.argsort(E)

    return {
        "E": E[order],
        "Y": Y[order],
    }


def _clean_sampler_table(r, values):
    """
    Sort sampler table by r and remove duplicate r values.

    PchipInterpolator requires strictly increasing x values.
    """
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)

    good = np.isfinite(r) & np.isfinite(values)
    r = r[good]
    values = values[good]

    order = np.argsort(r)
    r = r[order]
    values = values[order]

    r_unique, idx = np.unique(r, return_index=True)
    values_unique = values[idx]

    return r_unique, values_unique


def load_energy_sampler_csv(path: str | Path) -> dict:
    """
    Load long-format emitted-energy sampler CSV.

    Expected format:
        line 1: description
        line 2: description
        line 3: "beamE (eV)","r","eE (eV)"
        then data

    Returns
    -------
    dict with keys:
        E
        tables

    Each table has:
        r
        Eout
    """
    path = Path(path)

    df = pd.read_csv(path, skiprows=2)

    e_col = df.columns[0]
    r_col = df.columns[1]
    val_col = df.columns[2]

    Egrid = np.sort(df[e_col].unique().astype(float))
    tables = []

    for E in Egrid:
        g = df[df[e_col] == E].copy()

        r, Eout = _clean_sampler_table(
            g[r_col].to_numpy(),
            g[val_col].to_numpy(),
        )

        tables.append({
            "r": r,
            "Eout": Eout,
        })

    return {
        "E": Egrid,
        "tables": tables,
    }


def load_theta_sampler_csv(path: str | Path) -> dict:
    """
    Load long-format emitted-angle sampler CSV.

    Expected format:
        line 1: description
        line 2: description
        line 3: "beamE (eV)","r","theta (deg)"
        then data

    Returns
    -------
    dict with keys:
        E
        tables

    Each table has:
        r
        theta
    """
    path = Path(path)

    df = pd.read_csv(path, skiprows=2)

    e_col = df.columns[0]
    r_col = df.columns[1]
    val_col = df.columns[2]

    Egrid = np.sort(df[e_col].unique().astype(float))
    tables = []

    for E in Egrid:
        g = df[df[e_col] == E].copy()

        r, theta = _clean_sampler_table(
            g[r_col].to_numpy(),
            g[val_col].to_numpy(),
        )

        tables.append({
            "r": r,
            "theta": theta,
        })

    return {
        "E": Egrid,
        "tables": tables,
    }


# ============================================================
# Default model loader
# ============================================================

def load_default_surface_models(
    model_dir: str | Path,
    bronstein_dir: str | Path | None = None,
    use_measured_carbon_coating: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Load the same surface models used in the MATLAB setup.

    Parameters
    ----------
    model_dir:
        Directory containing JMONSEL CSV files.
    bronstein_dir:
        Directory containing Bronstein Mo/Ti yield files.
        If None, uses model_dir.
    use_measured_carbon_coating:
        If True (default), use the measured C-on-SS yield curves for the
        collector, rod, and drift tube. Their emitted-energy and angle
        distributions remain the JMONSEL thick-glassy-carbon distributions.
        Below 150 eV, the JMONSEL shape is scaled to join the measured curve
        continuously, with a zero-yield anchor at 0 eV.

    Returns
    -------
    yield_models, energy_models, theta_models
    """
    model_dir = Path(model_dir)

    if bronstein_dir is None:
        bronstein_dir = model_dir
    else:
        bronstein_dir = Path(bronstein_dir)

    theta_models = {
        "sample": {
            "BSE": load_theta_sampler_csv(model_dir / "BSEThetaFromPlaneSampler_uncoatedCuFPA.csv"),
            "SE": load_theta_sampler_csv(model_dir / "SEThetaFromPlaneSampler_uncoatedCuFPA.csv"),
        },
        "grid": {
            "BSE": load_theta_sampler_csv(model_dir / "BSEThetaFromWireSampler_glassyCarbon_t70nmWFPA.csv"),
            "SE": load_theta_sampler_csv(model_dir / "SEThetaFromWireSampler_glassyCarbon_t70nmWFPA.csv"),
        },
        "collector": {
            "BSE": load_theta_sampler_csv(model_dir / "BSEThetaFromPlaneSampler_glassyCarbon_t150000nmCuFPA.csv"),
            "SE": load_theta_sampler_csv(model_dir / "SEThetaFromPlaneSampler_glassyCarbon_t150000nmCuFPA.csv"),
        },
        "holder": {
            "BSE": "cosine",
            "SE": "cosine",
        },
        "receiver": {
            "BSE": "cosine",
            "SE": "cosine",
        },
    }

    energy_models = {
        "sample": {
            "BSE": load_energy_sampler_csv(model_dir / "BSEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv"),
            "SE": load_energy_sampler_csv(model_dir / "SEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv"),
        },
        "grid": {
            "BSE": load_energy_sampler_csv(model_dir / "BSEeEFromWireSampler_glassyCarbon_t70nmWFPA.csv"),
            "SE": load_energy_sampler_csv(model_dir / "SEeEFromWireSampler_glassyCarbon_t70nmWFPA.csv"),
        },
        "collector": {
            "BSE": load_energy_sampler_csv(model_dir / "BSEeEFromPlaneSampler_glassyCarbon_t150000nmCuFPA.csv"),
            "SE": load_energy_sampler_csv(model_dir / "SEeEFromPlaneSampler_glassyCarbon_t150000nmCuFPA.csv"),
        },
    }

    # MATLAB used sample energy distributions for holder/receiver.
    energy_models["holder"] = {
        "BSE": energy_models["sample"]["BSE"],
        "SE": energy_models["sample"]["SE"],
    }

    energy_models["receiver"] = {
        "BSE": energy_models["sample"]["BSE"],
        "SE": energy_models["sample"]["SE"],
    }

    collector_bsey_jmonsel = load_yield_curve_csv(
        model_dir / "BSEYFromPlane_glassyCarbon_t150000nmCuFPA.csv"
    )
    collector_sey_jmonsel = load_yield_curve_csv(
        model_dir / "SEYFromPlane_glassyCarbon_t150000nmCuFPA.csv"
    )

    if use_measured_carbon_coating:
        measured_carbon_bsey = load_yield_curve_csv(
            model_dir / MEASURED_CARBON_BSEY_FILENAME
        )
        measured_carbon_sey = load_yield_curve_csv(
            model_dir / MEASURED_CARBON_SEY_FILENAME
        )

        collector_bsey = _make_measured_carbon_hybrid_curve(
            jmonsel_model=collector_bsey_jmonsel,
            measured_model=measured_carbon_bsey,
            yield_kind="BSEY",
        )
        collector_sey = _make_measured_carbon_hybrid_curve(
            jmonsel_model=collector_sey_jmonsel,
            measured_model=measured_carbon_sey,
            yield_kind="SEY",
        )
    else:
        collector_bsey = collector_bsey_jmonsel
        collector_sey = collector_sey_jmonsel

    yield_models = {
        "sample": {
            "BSEY": load_yield_curve_csv(model_dir / "BSEYFromPlane_SEVaccum_t0nmCuFPA.csv"),
            "SEY": load_yield_curve_csv(model_dir / "SEYFromPlane_SEVaccum_t0nmCuFPA.csv"),
        },
        "holder": {
            "BSEY": load_yield_curve_csv(bronstein_dir / "BSEY_Mo_Bronstein.csv"),
            "SEY": load_yield_curve_csv(bronstein_dir / "SEY_Mo_Bronstein.csv"),
        },
        "receiver": {
            "BSEY": load_yield_curve_csv(bronstein_dir / "BSEY_Ti_Bronstein.csv"),
            "SEY": load_yield_curve_csv(bronstein_dir / "SEY_Ti_Bronstein.csv"),
        },
        "grid": {
            "BSEY": load_yield_curve_csv(model_dir / "BSEYFromWire_glassyCarbon_t70nmWFPA.csv"),
            "SEY": load_yield_curve_csv(model_dir / "SEYFromWire_glassyCarbon_t70nmWFPA.csv"),
        },
        "collector": {
            "BSEY": collector_bsey,
            "SEY": collector_sey,
        },
    }

    return yield_models, energy_models, theta_models


def _make_measured_carbon_hybrid_curve(
    jmonsel_model: dict,
    measured_model: dict,
    yield_kind: str,
) -> dict:
    """
    Build the collector-family yield curve used by the cascade model.

    The measured table is used from 150 eV through 10 keV. Below 150 eV,
    the original JMONSEL curve is scaled to meet the measured 150 eV value.
    A (0 eV, 0) anchor avoids holding the first JMONSEL value constant all
    the way to zero incident energy.
    """
    measured_E = np.asarray(measured_model["E"], dtype=float)
    measured_y = np.asarray(measured_model["Y"], dtype=float)

    if (
        measured_E.ndim != 1
        or measured_y.ndim != 1
        or measured_E.size != measured_y.size
        or measured_E.size < 2
    ):
        raise ValueError(
            f"Invalid measured carbon {yield_kind} table"
        )
    if not np.all(np.diff(measured_E) > 0.0):
        raise ValueError(
            f"Measured carbon {yield_kind} energies must be strictly increasing"
        )
    if not np.isclose(measured_E[0], 150.0):
        raise ValueError(
            f"Measured carbon {yield_kind} table must begin at 150 eV"
        )
    if measured_E[-1] < 10000.0:
        raise ValueError(
            f"Measured carbon {yield_kind} table must extend to 10000 eV"
        )

    jE = np.asarray(jmonsel_model["E"], dtype=float)
    jY = np.asarray(jmonsel_model["Y"], dtype=float)

    join_energy_eV = float(measured_E[0])
    y_jmonsel_at_150 = float(np.interp(join_energy_eV, jE, jY))
    if not np.isfinite(y_jmonsel_at_150) or y_jmonsel_at_150 <= 0.0:
        raise ValueError(
            f"Cannot scale JMONSEL {yield_kind} below 150 eV"
        )

    scale = float(measured_y[0]) / y_jmonsel_at_150
    low_mask = (jE > 0.0) & (jE < join_energy_eV)
    low_E = jE[low_mask]
    low_Y = scale * jY[low_mask]

    E = np.concatenate((
        np.array([0.0]),
        low_E,
        measured_E,
    ))
    Y = np.concatenate((
        np.array([0.0]),
        low_Y,
        measured_y,
    ))

    return {
        "E": E,
        "Y": Y,
        "source": MEASURED_CARBON_SOURCE,
        "yield_kind": str(yield_kind),
        "measured_energy_min_eV": float(measured_E[0]),
        "measured_energy_max_eV": float(measured_E[-1]),
        "low_energy_model": "scaled JMONSEL shape with zero-energy anchor",
        "jmonsel_scale_below_150_eV": scale,
        # Multipliers are ALLOWED on this curve (previously blocked).
        # Kept as metadata so downstream code/logs can still tell that
        # this is measured data rather than a model curve.
        "allow_collector_BSE_mult": True,
        "is_measured": True,
    }


# ============================================================
# Surface name mapping
# ============================================================

def canonical_surface_name(name: str) -> str:
    """
    Convert trajectory owner/outcome names to canonical surface names.
    """
    name = str(name)

    if name in [
        "sample",
        "holder",
        "receiver",
        "grid",
        "collector",
        "rod",
        "drifttube",
        "escaped",
    ]:
        return name

    mapping = {
        "g1_shell": "g1mesh",
        "g2_shell": "g2mesh",
        "g3_shell": "g3mesh",

        "g1_low_frame": "g1frame",
        "g1_upper_frame": "g1frame",
        "g2_low_frame": "g2frame",
        "g2_upper_frame": "g2frame",
        "g3_low_frame": "g3frame",
        "g3_upper_frame": "g3frame",

        "collector_shell": "collector",
        "escaped_grid": "escaped",
        "left_grid": "escaped",
    }

    return mapping.get(name, name)


def surface_family(surface_name: str) -> str:
    """
    Map a specific surface name to one of:
        sample, holder, receiver, grid, collector
    """
    s = canonical_surface_name(surface_name)

    if s == "sample":
        return "sample"

    if s == "holder":
        return "holder"

    if s == "receiver":
        return "receiver"

    if s == "grid":
        return "grid"

    if s in [
        "g1mesh", "g2mesh", "g3mesh",
        "g1frame", "g2frame", "g3frame",
    ]:
        return "grid"

    if s == "collector":
        return "collector"

    if s in ["rod", "drifttube"]:
        return "collector"

    raise ValueError(f"Unknown surface name: {surface_name}")


def is_fullsphere_surface(surface_name: str) -> bool:
    """
    True for grid mesh/frame emission where MATLAB allowed full-sphere emission.
    """
    s = canonical_surface_name(surface_name)

    return s in [
        "g1mesh", "g2mesh", "g3mesh",
        "g1frame", "g2frame", "g3frame",
        "grid",
    ]


def surface_voltage(surface_name: str, voltages: dict) -> float:
    """
    Return electrode voltage for a surface name.
    """
    s = canonical_surface_name(surface_name)

    if s in ["sample", "holder", "receiver"]:
        return voltages.get("Vs", 0.0)

    if s == "rod":
        return voltages.get("Vr", 0.0)

    if s in ["g1frame", "g1mesh"]:
        return voltages.get("Vg1", 0.0)

    if s in ["g2frame", "g2mesh", "grid"]:
        return voltages.get("Vg2", 0.0)

    if s in ["g3frame", "g3mesh"]:
        return voltages.get("Vg3", 0.0)

    if s == "drifttube":
        return voltages.get("Vdt", 0.0)

    if s == "collector":
        return voltages.get("Vc", 0.0)

    return np.nan


# ============================================================
# Yield sampling
# ============================================================

def interp_yield_model(model: dict, Einc: float) -> float:
    """
    Interpolate yield curve at incident energy.
    """
    E = np.asarray(model["E"], dtype=float)
    Y = np.asarray(model["Y"], dtype=float)

    return float(np.interp(Einc, E, Y, left=Y[0], right=Y[-1]))


_MEASURED_SCALING_WARNED: set = set()


def _warn_scaling_measured_curve(model, mult, kind, surface_name):
    """
    Emit one warning per (curve, kind, surface) when a MEASURED yield table is
    scaled by a multiplier other than 1.

    The point is provenance, not prohibition. A fitted value that stays close
    to 1 says the witness-coupon measurement transfers well to the in-situ
    electrodes; a value far from 1 is a real result worth reporting, and
    should not be quietly folded into a "measured" label.
    """
    if model is None or np.isclose(float(mult), 1.0):
        return

    source = str(model.get("source", "")) if isinstance(model, dict) else ""

    if "measured" not in source.lower():
        return

    key = (source, str(kind), canonical_surface_name_for_sey(surface_name))

    if key in _MEASURED_SCALING_WARNED:
        return

    _MEASURED_SCALING_WARNED.add(key)

    warnings.warn(
        f"{kind} multiplier {float(mult):.4g} applied on top of the measured "
        f"curve '{source}' for surface '{surface_name}'. Results no longer "
        f"reflect the measured yields at face value; report the fitted "
        f"multiplier alongside them.",
        RuntimeWarning,
        stacklevel=2,
    )


def reset_measured_scaling_warnings():
    """Clear the one-time warning cache (useful between parameter-scan runs)."""
    _MEASURED_SCALING_WARNED.clear()


def sample_surface_event(
    yield_models: dict,
    surface_name: str,
    Einc: float,
    cos_theta: float,
    rng,
    grid_SEY_mult: float | None = None,
    BSE_mult: float = 1.0,
    grid_BSE_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    SEY_mult: float = 1.0,
    carbon_SEY_mult: float | None = None,
    carbon_BSE_mult: float | None = None,
    collector_SEY_mult: float | None = None,
    rod_SEY_mult: float | None = None,
    rod_BSE_mult: float | None = None,
    drifttube_SEY_mult: float | None = None,
    drifttube_BSE_mult: float | None = None,
) -> tuple[bool, int]:
    """
    Sample whether a BSE occurs and how many SE electrons are emitted.

    Multipliers
    -----------
    Grid surfaces take grid_SEY_mult / grid_BSE_mult.
    Graphite-coated surfaces (collector, rod, drift tube) take
    carbon_SEY_mult / carbon_BSE_mult, with optional per-electrode overrides.
    See bse_multiplier_for_surface for the precedence rules.

    Returns
    -------
    did_bse:
        True/False.
    Nse:
        Poisson-sampled number of secondary electrons.
    """
    fam = surface_family(surface_name)
    surf = canonical_surface_name(surface_name)

    sey_mdl = yield_models[fam]["SEY"]
    bsey_mdl = yield_models[fam]["BSEY"]

    sey_base = interp_yield_model(sey_mdl, Einc)
    bsey_base = interp_yield_model(bsey_mdl, Einc)

    # Incidence-angle yield enhancement now applies to every surface that has a
    # physically meaningful local normal:
    #
    #   sample / holder / receiver  real STL or analytic plane
    #   rod / drifttube            real STL solids (drift tube is now a real mesh)
    #   g1frame / g2frame / g3frame real STL solids with genuine face normals
    #   collector                  exact analytic sphere, so its radial normal
    #                              IS the true local normal
    #
    # Only the analytic wire meshes are excluded; see ANALYTIC_MESH_SURFACES.
    if surf in ANALYTIC_MESH_SURFACES:
        angular_gain = 1.0
    else:
        angular_gain = angular_yield_gain(cos_theta)

    sey_val = sey_base * angular_gain
    bsey_val = bsey_base * angular_gain

    # Yield multipliers.
    #
    # Both SEY and BSEY are now scalable on both families. The measured
    # carbon-coating tables no longer BLOCK a multiplier: scaling a measured
    # curve is a legitimate fit degree of freedom (coating thickness, surface
    # roughness, and the airbrushed-vs-witness-sample difference are all real),
    # so it is reported rather than forbidden. See _warn_scaling_measured_curve.
    sey_mult = sey_multiplier_for_surface(
        surface_name=surface_name,
        grid_SEY_mult=grid_SEY_mult,
        SEY_mult=SEY_mult,
        carbon_SEY_mult=carbon_SEY_mult,
        collector_SEY_mult=collector_SEY_mult,
        rod_SEY_mult=rod_SEY_mult,
        drifttube_SEY_mult=drifttube_SEY_mult,
    )

    bse_mult = bse_multiplier_for_surface(
        surface_name=surface_name,
        BSE_mult=BSE_mult,
        grid_BSE_mult=grid_BSE_mult,
        collector_BSE_mult=collector_BSE_mult,
        carbon_BSE_mult=carbon_BSE_mult,
        rod_BSE_mult=rod_BSE_mult,
        drifttube_BSE_mult=drifttube_BSE_mult,
    )

    _warn_scaling_measured_curve(sey_mdl, sey_mult, "SEY", surface_name)
    _warn_scaling_measured_curve(bsey_mdl, bse_mult, "BSEY", surface_name)

    sey_val *= sey_mult
    bsey_val *= bse_mult

    sey_val = max(0.0, float(sey_val))
    bsey_val = max(0.0, min(float(bsey_val), 0.99))

    did_bse = False if Einc <= 50.0 else (rng.random() < bsey_val)
    Nse = rng.poisson(sey_val)

    return did_bse, Nse


# ============================================================
# Energy and theta sampling
# ============================================================

def sample_direction_about_axis(axis, theta_rad, rng):
    """
    Sample a direction making polar angle theta_rad with respect to axis.
    theta = 0   -> along axis
    theta = pi  -> opposite axis
    """
    axis = unit(axis)

    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = unit(np.cross(axis, tmp))
    e2 = unit(np.cross(axis, e1))

    phi = 2.0 * np.pi * rng.random()

    vhat = (
        np.cos(theta_rad) * axis
        + np.sin(theta_rad) * (
            np.cos(phi) * e1
            + np.sin(phi) * e2
        )
    )

    return unit(vhat)
    

def _choose_sampler_table(model: dict, Einc: float, rng):
    """
    Choose nearest/interpolated incident-energy table.
    Matches MATLAB-style stochastic bracketing between adjacent tables.
    """
    Egrid = np.asarray(model["E"], dtype=float)
    tables = model["tables"]

    Einc = float(Einc)

    if Einc <= Egrid[0]:
        return tables[0]

    if Einc >= Egrid[-1]:
        return tables[-1]

    ilo = np.where(Egrid < Einc)[0][-1]
    ihi = np.where(Egrid > Einc)[0][0]

    Elo = Egrid[ilo]
    Ehi = Egrid[ihi]

    w = (Einc - Elo) / (Ehi - Elo)

    if rng.random() < w:
        return tables[ihi]

    return tables[ilo]


def sample_energy_from_table(model: dict, Einc: float, rng) -> float:
    """
    Sample emitted energy from inverse-CDF table using interpolation
    between incident-energy tables.

    This avoids artificial pile-up when the upper incident-energy table
    produces values above the requested Einc and those values are clipped.
    """
    Egrid = np.asarray(model["E"], dtype=float)
    tables = model["tables"]

    Einc = float(Einc)
    u = rng.random()

    def eval_table(tab, u):
        r = np.asarray(tab["r"], dtype=float)
        Eout = np.asarray(tab["Eout"], dtype=float)

        u_clip = np.clip(u, r[0], r[-1])

        f = PchipInterpolator(r, Eout, extrapolate=False)
        return float(f(u_clip))

    if Einc <= Egrid[0]:
        return eval_table(tables[0], u)

    if Einc >= Egrid[-1]:
        return eval_table(tables[-1], u)

    ilo = np.where(Egrid <= Einc)[0][-1]
    ihi = np.where(Egrid >= Einc)[0][0]

    if ilo == ihi:
        return eval_table(tables[ilo], u)

    Elo = Egrid[ilo]
    Ehi = Egrid[ihi]
    w = (Einc - Elo) / (Ehi - Elo)

    E_lo = eval_table(tables[ilo], u)
    E_hi = eval_table(tables[ihi], u)

    return (1.0 - w) * E_lo + w * E_hi


def sample_theta_from_table(model: dict, Einc: float, rng) -> float:
    """
    Sample emitted polar angle in degrees from inverse-CDF table.
    """
    tab = _choose_sampler_table(model, Einc, rng)

    r = np.asarray(tab["r"], dtype=float)
    theta = np.asarray(tab["theta"], dtype=float)

    u = rng.random()
    u = np.clip(u, r[0], r[-1])

    f = PchipInterpolator(r, theta, extrapolate=False)

    return float(f(u))


def sample_surface_energy(
    energy_models: dict,
    surface_name: str,
    kind: str,
    Einc: float,
    rng,
) -> float:
    fam = surface_family(surface_name)
    kind = kind.upper()

    mdl = energy_models[fam][kind]

    Einc = max(float(Einc), 0.01)

    Eraw = sample_energy_from_table(mdl, Einc, rng)

    if kind == "SE":
        Elo = 0.01
        Ehi = min(50.0, Einc)

    elif kind == "BSE":
        if Einc <= 50.0:
            return max(0.01, min(Einc, Eraw))

        Elo = 50.0
        Ehi = Einc

    else:
        raise ValueError(f"Unknown emission kind: {kind}")

    return min(max(Eraw, Elo), Ehi)


def sample_surface_theta(
    theta_models: dict,
    surface_name: str,
    kind: str,
    Einc: float,
    rng,
) -> float:
    """
    Sample emitted polar angle in degrees.

    For cosine models:
        theta = asin(sqrt(u))
    """
    fam = surface_family(surface_name)
    kind = kind.upper()

    mdl = theta_models[fam][kind]

    if isinstance(mdl, str) and mdl.lower() == "cosine":
        theta_rad = np.arcsin(np.sqrt(rng.random()))
        return float(np.rad2deg(theta_rad))

    return sample_theta_from_table(mdl, Einc, rng)


# ============================================================
# Direction generation
# ============================================================

def emit_local(
    n,
    theta_rad: float,
    phi_rad: float,
    speed: float,
) -> np.ndarray:
    """
    Build a velocity vector from local surface normal and polar/azimuthal angles.
    """
    n = unit(n)

    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])

    t1 = np.cross(n, a)
    t1 = unit(t1)

    t2 = np.cross(t1, n)
    t2 = unit(t2)

    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    direction = (
        c * n
        + s * np.cos(phi_rad) * t1
        + s * np.sin(phi_rad) * t2
    )

    return speed * unit(direction)


def launch_electron_about_axis(
    theta_rad: float,
    phi_rad: float,
    Eout_eV: float,
    axis,
) -> np.ndarray:
    """
    Launch electron with polar angle theta_rad relative to a chosen axis.

    For grid-wire JMONSEL angular samplers:
        axis = -v_in_hat

    Then:
        theta = 0 deg   -> backward, opposite incoming direction
        theta = 90 deg  -> sideways
        theta = 180 deg -> forward, along incoming direction
    """
    speed = speed_from_energy_eV(Eout_eV)
    axis = unit(axis)

    return emit_local(
        n=axis,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        speed=speed,
    )


def launch_surface_electron(
    theta_rad: float,
    phi_rad: float,
    Eout_eV: float,
    n_out,
    use_full_sphere: bool = False,
) -> np.ndarray:
    """
    Launch emitted electron from a surface.

    If use_full_sphere=False, force emission into outward hemisphere.
    """
    speed = speed_from_energy_eV(Eout_eV)

    n_out = unit(n_out)
    v = emit_local(n_out, theta_rad, phi_rad, speed)

    if not use_full_sphere:
        if np.dot(v, n_out) <= 0:
            v = -v

    return v


def sample_effective_wire_normal_for_grid_hit(n_grid, v_in, rng):
    vhat = unit(v_in)
    n_grid = unit(n_grid)

    # Make n_grid oppose incoming direction.
    if np.dot(vhat, n_grid) > 0:
        n_grid = -n_grid

    # Choose a random tangent direction in the grid plane.
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, n_grid)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    t1 = unit(np.cross(n_grid, tmp))
    t2 = unit(np.cross(n_grid, t1))

    # Random wire-edge direction in tangent plane.
    t_edge = unit(rng.normal() * t1 + rng.normal() * t2)

    # Impact parameter across wire shadow.
    b = rng.uniform(-1.0, 1.0)

    # Local cylinder normal on front half of wire.
    # b=0 gives normal against incoming/grid-normal direction.
    # |b| near 1 gives side/edge normal.
    n_wire = np.sqrt(max(0.0, 1.0 - b*b)) * n_grid + b * t_edge
    n_wire = unit(n_wire)

    # Ensure it faces the incoming electron.
    if np.dot(vhat, n_wire) > 0:
        n_wire = -n_wire

    return n_wire


# ============================================================
# Surface-emission generation
# ============================================================

def generate_surface_emissions(
    surface_name: str,
    r_hit,
    v_in,
    n_out,
    Einc: float,
    yield_models: dict,
    energy_models: dict,
    theta_models: dict,
    voltages: dict,
    rng,
    origin: str,
    grid_SEY_mult: float | None = None,
    BSE_mult: float = 1.0,
    grid_BSE_mult: float | None = None,
    collector_BSE_mult: float | None = None,
    SEY_mult: float = 1.0,
    carbon_SEY_mult: float | None = None,
    carbon_BSE_mult: float | None = None,
    collector_SEY_mult: float | None = None,
    rod_SEY_mult: float | None = None,
    rod_BSE_mult: float | None = None,
    drifttube_SEY_mult: float | None = None,
    drifttube_BSE_mult: float | None = None,
    sample_launch_eps: float = 1.0e-6,
    U0: float = 15.0,
    Phi_interp=None,
) -> list[dict]:
    """
    Generate emitted electrons from one surface impact.

    Returns
    -------
    emitted:
        List of dicts with p0, v0, E_emit_eV, kind, cos_theta.
    """
    emitted = []

    surface_name = canonical_surface_name(surface_name)

    r_hit = np.asarray(r_hit, dtype=float)
    v_in = np.asarray(v_in, dtype=float)

    fam = surface_family(surface_name)
    surf = canonical_surface_name(surface_name)

    # Orient the normal against the incoming velocity for every surface that
    # has a real local normal. Grid FRAMES are included now: their STL face
    # normal is already oriented by cascade.estimate_surface_normal(), so this
    # is a no-op there, but it makes the sign of cos_theta safe by construction
    # rather than by assumption.
    if surf not in ANALYTIC_MESH_SURFACES:
        n_out = orient_normal_against_incoming(n_out, v_in)

    vhat = unit(v_in)

    if surf in ANALYTIC_MESH_SURFACES:
        # Zero-thickness analytic sphere standing in for woven wire: the radial
        # normal is not the local wire normal, so no meaningful incidence angle
        # exists here. Leave the yield at its normal-incidence value.
        cos_theta = 1.0
    else:
        cos_theta = max(0.05, -float(np.dot(vhat, n_out)))

    Einc = float(Einc)
    Phi_emit = surface_voltage(surface_name, voltages)

    # Quantum reflection term copied from MATLAB-style model.
    # For ordinary runs this matters only for primary gun electrons on sample.
    E_perp = Einc * cos_theta**2

    k1 = np.sqrt(max(E_perp, 0.0))
    k2 = np.sqrt(max(E_perp + U0, 0.0))

    if (k1 + k2) > 0:
        R_quantum = ((k1 - k2) / (k1 + k2))**2
    else:
        R_quantum = 0.0

    p0_surface = r_hit + sample_launch_eps * n_out

    if Phi_interp is not None:
        from .fields import evaluate_potential
        Phi0 = evaluate_potential(p0_surface, Phi_interp)
        phi_correction = Phi0 - Phi_emit
    else:
        phi_correction = 0.0

    if surface_name == "sample" and origin == "gun":
        if rng.random() < R_quantum:
            v_reflect = v_in - 2.0 * np.dot(v_in, n_out) * n_out

            emitted.append({
                "p0": p0_surface,
                "v0": v_reflect,
                "E_emit_eV": Einc,
                "Phi_emit": Phi_emit,
                "kind": "quantum_reflection",
                "cos_theta": cos_theta,
            })

            return emitted

    did_bse, Nse = sample_surface_event(
        yield_models=yield_models,
        surface_name=surface_name,
        Einc=Einc,
        cos_theta=cos_theta,
        rng=rng,
        grid_SEY_mult=grid_SEY_mult,
        BSE_mult=BSE_mult,
        grid_BSE_mult=grid_BSE_mult,
        collector_BSE_mult=collector_BSE_mult,
        SEY_mult=SEY_mult,
        carbon_SEY_mult=carbon_SEY_mult,
        carbon_BSE_mult=carbon_BSE_mult,
        collector_SEY_mult=collector_SEY_mult,
        rod_SEY_mult=rod_SEY_mult,
        rod_BSE_mult=rod_BSE_mult,
        drifttube_SEY_mult=drifttube_SEY_mult,
        drifttube_BSE_mult=drifttube_BSE_mult,
    )

    if did_bse:
        E_bse = sample_surface_energy(
            energy_models=energy_models,
            surface_name=surface_name,
            kind="BSE",
            Einc=Einc,
            rng=rng,
        )

        # Cutoff check uses raw surface energy (escape condition).
        if not (surface_name == "sample" and E_bse < Phi_emit):
            if E_bse > 0:
                theta_bs = np.deg2rad(
                    sample_surface_theta(
                        theta_models=theta_models,
                        surface_name=surface_name,
                        kind="BSE",
                        Einc=Einc,
                        rng=rng,
                    )
                )

                phi_bs = 2.0 * np.pi * rng.random()

                # Apply launch-point potential correction to kinetic energy.
                # Clamp to a small positive value so speed stays real.
                E_bse_launch = max(E_bse + phi_correction, 1.0e-3)

                if fam == "grid":
                    axis_backscatter = -unit(v_in)

                    v_bse = launch_electron_about_axis(
                        theta_rad=theta_bs,
                        phi_rad=phi_bs,
                        Eout_eV=E_bse_launch,
                        axis=axis_backscatter,
                    )

                    p0_bse = r_hit + sample_launch_eps * unit(v_bse)

                else:
                    use_full = is_fullsphere_surface(surface_name)

                    v_bse = launch_surface_electron(
                        theta_rad=theta_bs,
                        phi_rad=phi_bs,
                        Eout_eV=E_bse_launch,
                        n_out=n_out,
                        use_full_sphere=use_full,
                    )

                    p0_bse = p0_surface

                emitted.append({
                    "p0": p0_bse,
                    "v0": v_bse,
                    "E_emit_eV": E_bse,        # record physical surface energy
                    "E_launch_eV": E_bse_launch,
                    "Phi_emit": Phi_emit,
                    "kind": "BSE",
                    "cos_theta": cos_theta,
                })

    for _ in range(Nse):
        E_se = sample_surface_energy(
            energy_models=energy_models,
            surface_name=surface_name,
            kind="SE",
            Einc=Einc,
            rng=rng,
        )

        # Cutoff check uses raw surface energy (escape condition).
        if surface_name == "sample" and E_se < Phi_emit:
            continue

        if E_se <= 0:
            continue

        theta_se = np.deg2rad(
            sample_surface_theta(
                theta_models=theta_models,
                surface_name=surface_name,
                kind="SE",
                Einc=Einc,
                rng=rng,
            )
        )

        phi_se = 2.0 * np.pi * rng.random()

        # Apply launch-point potential correction to kinetic energy.
        # Clamp to a small positive value so speed stays real.
        E_se_launch = max(E_se + phi_correction, 1.0e-3)

        if fam == "grid":
            axis_backscatter = -unit(v_in)

            v_se = launch_electron_about_axis(
                theta_rad=theta_se,
                phi_rad=phi_se,
                Eout_eV=E_se_launch,
                axis=axis_backscatter,
            )

            p0_se = r_hit + sample_launch_eps * unit(v_se)

        else:
            use_full = is_fullsphere_surface(surface_name)

            v_se = launch_surface_electron(
                theta_rad=theta_se,
                phi_rad=phi_se,
                Eout_eV=E_se_launch,
                n_out=n_out,
                use_full_sphere=use_full,
            )

            p0_se = p0_surface

        emitted.append({
            "p0": p0_se,
            "v0": v_se,
            "E_emit_eV": E_se,          # record physical surface energy
            "E_launch_eV": E_se_launch,
            "kind": "SE",
            "cos_theta": cos_theta,
        })

    return emitted
