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
# They all read the same measured C-on-SS curve, differing only through impact
# energy and incidence angle. (A collector-only multiplier used to exist while
# surface_family() routed rod and drifttube to those same curves, which
# silently gave three identically-coated electrodes two different effective
# yields. All yield multipliers have since been removed.)
CARBON_COATED_SURFACES = COLLECTOR_SURFACES | {
    "rod",
    "drifttube",
}

# Analytic zero-thickness spheres standing in for the woven wire mesh.
#
# The sphere's radial normal describes the local mesh plane, not a wire
# circumference. For a plane-derived measured yield, every mesh impact samples
# an effective cylindrical-wire normal using the incoming trajectory and a
# random impact parameter; the capped angular law is then applied to that event.
# For a JMONSEL FromWire curve, the sampled cosine is ignored for yield magnitude
# because the FromWire table already averages over wire geometry.
#
# Grid FRAMES are NOT in this set: they are real STL solids with genuine face
# normals from collisions.py and use plane-surface emission logic.
ANALYTIC_MESH_SURFACES = {
    "g1mesh",
    "g2mesh",
    "g3mesh",
}

# Grid support frames: DMLS 316 stainless steel, carbon sputter-coated.
#
# This is the SAME material system as the measured C-on-SS witness coupon, and
# the frames are flat hoops rather than wires, so the measured flat-plane curve
# applies to them directly with no geometry correction. They are also real STL
# solids, so they already take the ordinary 1/cos incidence law.
GRID_FRAME_SURFACES = {
    "g1frame",
    "g2frame",
    "g3frame",
}

# Analytical reference for an uncapped secant law averaged over the
# illuminated half of a cylindrical wire. The cascade no longer applies this
# as one fixed multiplier. For plane-derived grid yields it samples an
# effective local wire normal for every impact and applies the capped angular
# law to that event. Keeping the reference value is useful for diagnostics and
# backward-compatible imports.
WIRE_GEOMETRY_GAIN = float(np.pi / 2.0)

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
        # Provenance, so the run banner and describe_surface_yields() can name
        # the file a curve came from instead of printing "<untagged>".
        "source": path.name,
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
        "source": path.name,
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
        "source": path.name,
    }


# ============================================================
# Default model loader
# ============================================================

def load_default_surface_models(
    model_dir: str | Path,
    bronstein_dir: str | Path | None = None,
    use_measured_carbon_coating: bool = True,
    use_measured_carbon_for_grids: bool = False,
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
    use_measured_carbon_for_grids:
        If True, also use the measured C-on-SS curves for the grid meshes and
        grid frames, replacing the JMONSEL glassy-carbon-on-tungsten "FromWire"
        curves. The grid wires and frames are carbon sputter-coated, and the
        frames are carbon on 316 stainless -- the same system as the witness
        coupon. Because the measured curve is a flat-plane measurement, each
        mesh hit samples an effective local cylindrical-wire normal and applies
        the capped angular law event by event. Grid frames use plane-carbon
        energy/angle samplers and vacuum-hemisphere emission from their STL
        normals; mesh energy/angle distributions remain JMONSEL FromWire.
        Requires use_measured_carbon_coating=True.

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

    # Grid frames are real, flat STL solids. They use the same plane-carbon
    # energy and angular distributions as the collector family, not the woven-
    # wire FromWire samplers used by the analytic grid meshes.
    theta_models["gridframe"] = {
        "BSE": theta_models["collector"]["BSE"],
        "SE": theta_models["collector"]["SE"],
    }
    energy_models["gridframe"] = {
        "BSE": energy_models["collector"]["BSE"],
        "SE": energy_models["collector"]["SE"],
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

    # Cosmetic: "geometry" is only consulted for ANALYTIC_MESH_SURFACES, which
    # never includes the collector/rod/drift tube, but tagging it keeps the
    # audit output from showing a bare None that looks like a bug.
    collector_bsey = dict(collector_bsey, geometry="plane")
    collector_sey = dict(collector_sey, geometry="plane")

    # Grid curves. Two options:
    #   use_measured_carbon_for_grids=False (default, unchanged behaviour)
    #       JMONSEL glassy-carbon-on-tungsten "FromWire" curves. Wire geometry
    #       is already averaged into these, so geometry="wire".
    #   use_measured_carbon_for_grids=True
    #       The measured C-on-SS curves, which are FLAT-PLANE measurements, so
    #       geometry="plane". Each mesh impact samples a local wire normal and
    #       receives the capped event-specific angular gain; frames use their
    #       real STL normals with no wire conversion.
    grid_bsey_jmonsel = load_yield_curve_csv(
        model_dir / "BSEYFromWire_glassyCarbon_t70nmWFPA.csv"
    )
    grid_sey_jmonsel = load_yield_curve_csv(
        model_dir / "SEYFromWire_glassyCarbon_t70nmWFPA.csv"
    )
    grid_bsey_jmonsel = dict(grid_bsey_jmonsel, geometry="wire")
    grid_sey_jmonsel = dict(grid_sey_jmonsel, geometry="wire")

    if use_measured_carbon_for_grids:
        if not use_measured_carbon_coating:
            raise ValueError(
                "use_measured_carbon_for_grids=True requires "
                "use_measured_carbon_coating=True"
            )
        grid_bsey = dict(collector_bsey, geometry="plane")
        grid_sey = dict(collector_sey, geometry="plane")
        gridframe_bsey = dict(collector_bsey, geometry="plane")
        gridframe_sey = dict(collector_sey, geometry="plane")
    else:
        grid_bsey = grid_bsey_jmonsel
        grid_sey = grid_sey_jmonsel
        gridframe_bsey = grid_bsey_jmonsel
        gridframe_sey = grid_sey_jmonsel

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
            "BSEY": grid_bsey,
            "SEY": grid_sey,
        },
        "gridframe": {
            "BSEY": gridframe_bsey,
            "SEY": gridframe_sey,
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
    True only for analytic woven-wire mesh emission.

    Grid frames are finite STL solids and must emit into their vacuum-side
    hemisphere like the collector, rod, and drift tube.
    """
    s = canonical_surface_name(surface_name)

    return s in [
        "g1mesh", "g2mesh", "g3mesh",
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


def _surface_model_key(models: dict, surface_name: str) -> str:
    """Return the model-dictionary key for a specific physical surface."""
    surf = canonical_surface_name(surface_name)
    fam = surface_family(surface_name)

    if surf in GRID_FRAME_SURFACES and "gridframe" in models:
        return "gridframe"

    return fam


def sample_surface_event(
    yield_models: dict,
    surface_name: str,
    Einc: float,
    cos_theta: float,
    rng,
) -> tuple[bool, int]:
    """
    Sample whether a BSE occurs and how many SE electrons are emitted.

    Multipliers
    -----------
    Yields come straight from the curves in yield_models; there are no
    tunable multipliers.

    Returns
    -------
    did_bse:
        True/False.
    Nse:
        Poisson-sampled number of secondary electrons.
    """
    surf = canonical_surface_name(surface_name)
    model_key = _surface_model_key(yield_models, surface_name)

    sey_mdl = yield_models[model_key]["SEY"]
    bsey_mdl = yield_models[model_key]["BSEY"]

    sey_base = interp_yield_model(sey_mdl, Einc)
    bsey_base = interp_yield_model(bsey_mdl, Einc)

    # Geometry gain. Three cases:
    #
    #   analytic mesh + "wire" curve   gain 1.0
    #       JMONSEL FromWire already averages over wire geometry.
    #   analytic mesh + "plane" curve  event-specific capped 1/cos(theta)
    #       generate_surface_emissions() samples an effective local cylindrical
    #       wire normal for this impact and passes its incidence cosine here.
    #   every real solid surface        event-specific capped 1/cos(theta)
    #       The STL/local spherical normal is physically meaningful.
    #
    # SEY and BSEY are checked independently so a mixed pair cannot silently
    # pick up the wrong treatment.
    def _geometry_gain(model):
        if surf in ANALYTIC_MESH_SURFACES:
            geom = "wire"
            if isinstance(model, dict):
                geom = str(model.get("geometry", "wire"))
            return angular_yield_gain(cos_theta) if geom == "plane" else 1.0
        return angular_yield_gain(cos_theta)

    sey_val = sey_base * _geometry_gain(sey_mdl)
    bsey_val = bsey_base * _geometry_gain(bsey_mdl)

    # No yield multipliers.
    #
    # Every surface runs directly off its curve:
    #   sample                      JMONSEL Cu
    #   grid mesh / grid frames     measured C-on-SS (event-sampled wire
    #                               incidence on mesh only), or JMONSEL FromWire
    #   collector / rod / drifttube measured C-on-SS
    #   holder / receiver           Bronstein Mo / Ti
    #
    # Scale factors were removed deliberately. They let a geometry error (for
    # example the known rod line-of-sight deficit) be absorbed into a yield
    # parameter, giving a good chi-squared for the wrong reason; and because
    # they were threaded through five call layers, a surface's effective yield
    # depended on which call site happened to forward which keyword. What the
    # curves say is now what the model uses.
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
    model_key = _surface_model_key(energy_models, surface_name)
    kind = kind.upper()

    mdl = energy_models[model_key][kind]

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
    model_key = _surface_model_key(theta_models, surface_name)
    kind = kind.upper()

    mdl = theta_models[model_key][kind]

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
    sample_launch_eps: float = 1.0e-6,
    U0: float = 15.0,
    Phi_interp=None,
) -> tuple[list[dict], dict]:
    """
    Generate emitted electrons from one surface impact.

    Returns
    -------
    emitted:
        List of dicts with p0, v0, E_emit_eV, kind, cos_theta.
    event_info:
        Per-impact diagnostics. For analytic grid-wire hits this records the
        sampled local-wire incidence cosine and angular gain even when the hit
        produces zero emitted electrons. Non-grid fields are NaN.
    """
    emitted = []
    event_info = {
        "sampled_wire_cos_theta": np.nan,
        "sampled_wire_angular_gain": np.nan,
        "wire_sey_gain_used": np.nan,
        "wire_bsey_gain_used": np.nan,
    }

    surface_name = canonical_surface_name(surface_name)

    r_hit = np.asarray(r_hit, dtype=float)
    v_in = np.asarray(v_in, dtype=float)

    surf = canonical_surface_name(surface_name)

    vhat = unit(v_in)

    if surf in ANALYTIC_MESH_SURFACES:
        # The analytic shell only provides the local mesh-plane normal. Sample
        # a cylindrical-wire normal at this impact so a plane-derived measured
        # curve responds to the actual incoming direction and random impact
        # parameter. JMONSEL FromWire yield curves ignore this cosine because
        # they already average over wire geometry, but the sampled normal is
        # still recorded for diagnostics.
        n_out = sample_effective_wire_normal_for_grid_hit(
            n_grid=n_out,
            v_in=v_in,
            rng=rng,
        )
    else:
        # Real surfaces, including grid frames, use their STL/local surface
        # normal oriented toward the incident vacuum half-space.
        n_out = orient_normal_against_incoming(n_out, v_in)

    cos_theta = max(0.05, -float(np.dot(vhat, n_out)))

    if surf in ANALYTIC_MESH_SURFACES:
        # Record the actual local-wire geometry sampled for this impact.
        # sampled_wire_angular_gain is the geometric capped-secant factor.
        # The *_gain_used fields make clear whether the active yield curve
        # actually uses that factor (plane-derived measured curve) or unity
        # (JMONSEL FromWire, which already contains wire geometry).
        wire_gain = angular_yield_gain(cos_theta)
        model_key = _surface_model_key(yield_models, surface_name)
        sey_model = yield_models[model_key]["SEY"]
        bsey_model = yield_models[model_key]["BSEY"]

        sey_geom = str(sey_model.get("geometry", "wire"))
        bsey_geom = str(bsey_model.get("geometry", "wire"))

        event_info.update({
            "sampled_wire_cos_theta": float(cos_theta),
            "sampled_wire_angular_gain": float(wire_gain),
            "wire_sey_gain_used": float(wire_gain if sey_geom == "plane" else 1.0),
            "wire_bsey_gain_used": float(wire_gain if bsey_geom == "plane" else 1.0),
        })

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

            return emitted, event_info

    did_bse, Nse = sample_surface_event(
        yield_models=yield_models,
        surface_name=surface_name,
        Einc=Einc,
        cos_theta=cos_theta,
        rng=rng,
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

                if surf in ANALYTIC_MESH_SURFACES:
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

        if surf in ANALYTIC_MESH_SURFACES:
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

    return emitted, event_info


def describe_surface_yields(
    yield_models: dict,
    energies=(20.0, 100.0, 500.0, 5000.0),
    cos_theta: float = 1.0,
    n_samples: int = 20000,
    seed: int = 0,
    verbose: bool = True,
):
    """
    Audit curve provenance and effective sampled yields for representative
    surfaces.

    For an analytic grid mesh using a plane-derived measured curve, this audit
    follows the real cascade path: it samples a local cylindrical-wire normal
    for every trial, computes that event's incidence cosine, and applies the
    capped angular law. A JMONSEL FromWire curve remains at gain 1 because its
    wire geometry is already included.
    """
    probes = [
        ("g1mesh", "grid mesh (analytic sphere, woven wire)"),
        ("g1frame", "grid frame (STL, flat hoop)"),
        ("collector", "collector shell"),
        ("rod", "sample rod"),
        ("drifttube", "drift tube"),
        ("sample", "sample"),
        ("holder", "holder"),
    ]

    c_macro = float(np.clip(cos_theta, 0.0, 1.0))
    n_grid_audit = np.array([1.0, 0.0, 0.0])
    v_in_audit = np.array([
        -c_macro,
        np.sqrt(max(0.0, 1.0 - c_macro * c_macro)),
        0.0,
    ])

    if verbose:
        print("=" * 82)
        print("CURVE PROVENANCE")
        print("=" * 82)
        print(f"{'family':12}{'kind':6}{'geometry':10}source")

        for fam in (
            "sample", "grid", "gridframe", "collector", "holder", "receiver"
        ):
            if fam not in yield_models:
                print(f"{fam:12}{'--':6}{'--':10}<MISSING from yield_models>")
                continue
            for kind in ("SEY", "BSEY"):
                m = yield_models[fam].get(kind, {})
                print(
                    f"{fam:12}{kind:6}"
                    f"{str(m.get('geometry', 'wire' if fam == 'grid' else 'plane')):10}"
                    f"{m.get('source', '<untagged>')}"
                )

        print()
        print("=" * 82)
        print(
            "EFFECTIVE YIELDS through sample_surface_event  "
            f"(macroscopic cos_theta={c_macro:g}, N={n_samples:,})"
        )
        print("=" * 82)
        header = f"{'surface':11}{'mean gain':>11}"
        for E in energies:
            header += f"{'SEY@' + format(E, '.0f'):>12}{'BSEY':>8}"
        print(header)

    rows = []

    for surf, _label in probes:
        model_key = _surface_model_key(yield_models, surf)
        if model_key not in yield_models:
            continue

        canon = canonical_surface_name(surf)
        sey_model = yield_models[model_key]["SEY"]
        sey_geom = str(sey_model.get("geometry", "wire"))

        gain_rng = np.random.default_rng(seed + 991)
        if canon in ANALYTIC_MESH_SURFACES and sey_geom == "plane":
            gains = []
            for _ in range(n_samples):
                n_wire = sample_effective_wire_normal_for_grid_hit(
                    n_grid=n_grid_audit,
                    v_in=v_in_audit,
                    rng=gain_rng,
                )
                c_event = max(
                    0.05,
                    -float(np.dot(unit(v_in_audit), n_wire)),
                )
                gains.append(angular_yield_gain(c_event))
            mean_gain = float(np.mean(gains))
        elif canon in ANALYTIC_MESH_SURFACES:
            mean_gain = 1.0
        else:
            mean_gain = angular_yield_gain(c_macro)

        row = {"surface": surf, "geometry_gain": mean_gain}
        line = f"{surf:11}{mean_gain:11.3f}"

        for E in energies:
            rng = np.random.default_rng(seed)
            n_bse = 0
            n_se = 0

            for _ in range(n_samples):
                c_event = c_macro
                if canon in ANALYTIC_MESH_SURFACES and sey_geom == "plane":
                    n_wire = sample_effective_wire_normal_for_grid_hit(
                        n_grid=n_grid_audit,
                        v_in=v_in_audit,
                        rng=rng,
                    )
                    c_event = max(
                        0.05,
                        -float(np.dot(unit(v_in_audit), n_wire)),
                    )

                did_bse, nse = sample_surface_event(
                    yield_models=yield_models,
                    surface_name=surf,
                    Einc=float(E),
                    cos_theta=c_event,
                    rng=rng,
                )
                n_bse += bool(did_bse)
                n_se += int(nse)

            sey = n_se / n_samples
            bsey = n_bse / n_samples
            row[f"SEY@{E:.0f}"] = sey
            row[f"BSEY@{E:.0f}"] = bsey
            line += f"{sey:12.3f}{bsey:8.3f}"

        rows.append(row)

        if verbose:
            print(line)

    if verbose:
        print()
        print(
            "Expected when use_measured_carbon_for_grids=True:\n"
            "  grid/gridframe/collector show the measured C-on-SS source;\n"
            "  g1mesh uses an event-sampled wire incidence gain rather than a\n"
            "  fixed pi/2 multiplier; and g1frame uses plane-carbon samplers\n"
            "  with vacuum-hemisphere emission from its STL normal."
        )

    return rows
