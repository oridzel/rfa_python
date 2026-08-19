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

# Incidence-angle-dependent Cu yields calculated with JMONSEL.
#
# IMPORTANT: these data are used only as ANGULAR GAINS, normalized to the
# 0-degree row at the same incident energy. The existing normal-incidence Cu
# yield curves therefore remain the absolute normalization of the RFA model.
# This lets us replace the crude secant law without changing the established
# normal-incidence yield normalization.  In v8 the default quantum-reflection
# treatment is also corrected separately; that can make a tiny stochastic
# difference even at 0 degrees.
SAMPLE_ANGULAR_YIELD_FILENAME = "yieldsCu100TDDFT10ED.csv"
SAMPLE_ANGULAR_YIELD_SOURCE = (
    "Cu TDDFT b=1, l=0, April 2026; Browning elastic below 100 eV"
)

# Optional incidence-specific sample tables.  The filenames deliberately match
# the normal-incidence tables because JMONSEL writes one self-contained folder
# per incidence angle.  Keep the tilted files in a separate directory and pass
# that directory explicitly to load_default_surface_models().
SAMPLE_GUN_INCIDENCE_FILENAMES = {
    "BSEY": "BSEYFromPlane_SEVaccum_t0nmCuFPA.csv",
    "SEY": "SEYFromPlane_SEVaccum_t0nmCuFPA.csv",
    "BSE_energy": "BSEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv",
    "SE_energy": "SEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv",
    "BSE_theta": "BSEThetaFromPlaneSampler_uncoatedCuFPA.csv",
    "SE_theta": "SEThetaFromPlaneSampler_uncoatedCuFPA.csv",
}

# Optional event-level emitted-electron samplers for oblique gun incidence.
#
# These files preserve the correlation that is lost when emitted energy and
# polar angle are reduced to separate inverse-CDF tables. Each NPZ contains
# one row per emitted electron and must provide either
#
#   Einc_eV, Eout_eV, theta_deg, phi_deg
#
# or
#
#   Einc_eV, Eout_eV,
#   dir_beam_back, dir_toward_normal, dir_side
#
# The local basis is the same one used by launch_sample_electron_beam_relative:
#   beam_back      = fixed +X axis back toward the gun/drift tube;
#   toward_normal  = in the incidence plane, perpendicular to beam_back and
#                    pointing toward the sample's outward normal;
#   side           = beam_back x toward_normal.
#
# Having separate SE and BSE files avoids string/object arrays and keeps the
# NPZ pickle-free.
SAMPLE_GUN_JOINT_FILENAMES = {
    "BSE": "BSEJointFromPlaneSampler_uncoatedCuFPA.npz",
    "SE": "SEJointFromPlaneSampler_uncoatedCuFPA.npz",
}

# The RFA coordinate convention has +X pointing from the sample toward the
# drift tube/electron gun.  The tilted JMONSEL theta tables are explicitly
# referenced to this fixed axis (the untilted-sample normal), not to a primary
# velocity after it has been slightly deflected by the electrostatic field.
SAMPLE_BEAM_BACK_AXIS = np.array([1.0, 0.0, 0.0], dtype=float)

# The historical model also contained a separate surface-barrier quantum-
# reflection branch for gun electrons hitting Cu.  When a full JMONSEL
# incidence-angle yield table is active, applying that branch on top of the
# tabulated BSEY/SEY can double count reflected electrons and, more seriously,
# the historical early-return branch suppresses all ordinary SE/BSE emission
# for that impact.  ``auto`` therefore disables the separate branch whenever
# the JMONSEL angular table is present, while retaining exact legacy behavior
# when that table is not used.
SAMPLE_QUANTUM_REFLECTION_MODES = {"auto", "legacy_replace", "disabled"}


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


def load_joint_emission_sampler_npz(path: str | Path) -> dict:
    """Load event-level correlated emitted-electron samples.

    Required arrays
    ---------------
    Einc_eV, Eout_eV

    Direction may be represented in either of two equivalent schemas:

    1. ``theta_deg`` and ``phi_deg`` in the beam/sample basis used by
       :func:`launch_sample_electron_beam_relative`; or
    2. the corresponding local direction cosines
       ``dir_beam_back``, ``dir_toward_normal``, and ``dir_side``.

    The loader converts both forms to normalized local direction cosines and
    groups rows by incident energy.  Keeping one table row per emitted electron
    preserves the full E-theta-phi correlation during resampling.
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = {"Einc_eV", "Eout_eV"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(
                f"Joint sampler {path.name!r} is missing arrays: {missing}"
            )

        Einc = np.asarray(data["Einc_eV"], dtype=float).reshape(-1)
        Eout = np.asarray(data["Eout_eV"], dtype=float).reshape(-1)

        angle_schema = {"theta_deg", "phi_deg"}.issubset(keys)
        component_schema = {
            "dir_beam_back",
            "dir_toward_normal",
            "dir_side",
        }.issubset(keys)

        if not angle_schema and not component_schema:
            raise ValueError(
                f"Joint sampler {path.name!r} must contain either "
                "theta_deg/phi_deg or dir_beam_back/dir_toward_normal/dir_side"
            )

        if component_schema:
            u0 = np.asarray(data["dir_beam_back"], dtype=float).reshape(-1)
            u1 = np.asarray(data["dir_toward_normal"], dtype=float).reshape(-1)
            u2 = np.asarray(data["dir_side"], dtype=float).reshape(-1)
            schema = "direction_cosines"
        else:
            theta_deg = np.asarray(data["theta_deg"], dtype=float).reshape(-1)
            phi_deg = np.asarray(data["phi_deg"], dtype=float).reshape(-1)
            theta_rad = np.deg2rad(theta_deg)
            phi_rad = np.deg2rad(phi_deg)
            u0 = np.cos(theta_rad)
            u1 = np.sin(theta_rad) * np.cos(phi_rad)
            u2 = np.sin(theta_rad) * np.sin(phi_rad)
            schema = "theta_phi"

    arrays = [Einc, Eout, u0, u1, u2]
    n = len(Einc)
    if n == 0:
        raise ValueError(f"Joint sampler {path.name!r} contains no events")
    if any(len(a) != n for a in arrays):
        raise ValueError(
            f"Joint sampler {path.name!r} arrays do not have equal length"
        )

    good = np.ones(n, dtype=bool)
    for a in arrays:
        good &= np.isfinite(a)
    good &= Einc > 0.0
    good &= Eout > 0.0

    if not np.any(good):
        raise ValueError(f"Joint sampler {path.name!r} has no valid events")

    Einc = Einc[good]
    Eout = Eout[good]
    dirs = np.column_stack((u0[good], u1[good], u2[good]))
    norms = np.linalg.norm(dirs, axis=1)
    nonzero = norms > 1.0e-12
    if not np.all(nonzero):
        Einc = Einc[nonzero]
        Eout = Eout[nonzero]
        dirs = dirs[nonzero]
        norms = norms[nonzero]
    if len(Einc) == 0:
        raise ValueError(
            f"Joint sampler {path.name!r} has no nonzero direction vectors"
        )
    dirs = dirs / norms[:, None]

    Egrid = np.unique(Einc)
    Egrid.sort()
    tables = []
    for E in Egrid:
        sel = np.isclose(Einc, E, rtol=0.0, atol=1.0e-12)
        tables.append({
            "Einc_eV": float(E),
            "Eout_eV": Eout[sel],
            "direction_local": dirs[sel],
        })

    return {
        "E": Egrid,
        "tables": tables,
        "source": path.name,
        "schema": schema,
        "n_events": int(len(Einc)),
        "direction_basis": (
            "beam_back, toward_sample_normal_in_incidence_plane, side"
        ),
    }


def load_sample_gun_incidence_model(
    sampler_dir: str | Path,
    incidence_angle_deg: float,
    angle_tolerance_deg: float = 0.5,
) -> dict:
    """Load one incidence-specific Cu sample model.

    The six incidence-specific tables are a matched set: absolute SEY/BSEY,
    emitted-energy inverse CDFs, and emitted-polar-angle inverse CDFs.  If the
    two optional SEEMC ``*JointFromPlaneSampler*.npz`` files are present, they are
    loaded as the preferred first-generation emission model because they retain
    the event-by-event E-theta-phi correlation.  Without those files the legacy
    theta-only model remains available as a backward-compatible fallback.

    The returned model is intentionally marked ``gun_only``.  A later cascade
    electron that re-impacts the sample generally has a different incident
    direction, so this one-angle model must not be reused for that event.
    """
    sampler_dir = Path(sampler_dir)
    angle = float(incidence_angle_deg)
    tolerance = float(angle_tolerance_deg)

    if not np.isfinite(angle) or not (0.0 < angle < 90.0):
        raise ValueError(
            "sample gun-incidence angle must satisfy 0 < angle < 90 deg"
        )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("sample gun-incidence angle tolerance must be >= 0 deg")

    paths = {
        key: sampler_dir / filename
        for key, filename in SAMPLE_GUN_INCIDENCE_FILENAMES.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incidence-specific Cu sample model is incomplete; missing: "
            + ", ".join(missing)
        )

    joint_paths = {
        kind: sampler_dir / filename
        for kind, filename in SAMPLE_GUN_JOINT_FILENAMES.items()
    }
    joint_present = {kind: path.is_file() for kind, path in joint_paths.items()}
    if any(joint_present.values()) and not all(joint_present.values()):
        missing_joint = [
            str(joint_paths[kind])
            for kind, present in joint_present.items()
            if not present
        ]
        raise FileNotFoundError(
            "Oblique-incidence joint emission sampling requires both SE and "
            "BSE NPZ files; missing: " + ", ".join(missing_joint)
        )

    joint_model = None
    if all(joint_present.values()):
        joint_model = {
            kind: load_joint_emission_sampler_npz(path)
            for kind, path in joint_paths.items()
        }

    model = {
        "incidence_angle_deg": angle,
        "angle_tolerance_deg": tolerance,
        "gun_only": True,
        "polar_axis": "fixed_beam_back_axis",
        "beam_back_axis": SAMPLE_BEAM_BACK_AXIS.copy(),
        "azimuth_model": (
            "joint_emitted_event"
            if joint_model is not None
            else "uniform_over_outward_part_of_polar_cone"
        ),
        "yield": {
            "BSEY": load_yield_curve_csv(paths["BSEY"]),
            "SEY": load_yield_curve_csv(paths["SEY"]),
        },
        "energy": {
            "BSE": load_energy_sampler_csv(paths["BSE_energy"]),
            "SE": load_energy_sampler_csv(paths["SE_energy"]),
        },
        "theta": {
            "BSE": load_theta_sampler_csv(paths["BSE_theta"]),
            "SE": load_theta_sampler_csv(paths["SE_theta"]),
        },
        "joint": joint_model,
        "source_dir": sampler_dir.name,
    }

    # All six files must describe the same incident-energy grid.  A mismatch
    # would otherwise mix yields and distributions from different simulations.
    grids = {
        "BSEY": np.asarray(model["yield"]["BSEY"]["E"], dtype=float),
        "SEY": np.asarray(model["yield"]["SEY"]["E"], dtype=float),
        "BSE energy": np.asarray(model["energy"]["BSE"]["E"], dtype=float),
        "SE energy": np.asarray(model["energy"]["SE"]["E"], dtype=float),
        "BSE theta": np.asarray(model["theta"]["BSE"]["E"], dtype=float),
        "SE theta": np.asarray(model["theta"]["SE"]["E"], dtype=float),
    }
    reference_name, reference_grid = next(iter(grids.items()))
    for name, grid in grids.items():
        if grid.shape != reference_grid.shape or not np.allclose(
            grid, reference_grid, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"Incidence-specific sample energy grid in {name} does not "
                f"match {reference_name}"
            )

    if joint_model is not None:
        for kind in ("BSE", "SE"):
            joint_grid = np.asarray(joint_model[kind]["E"], dtype=float)
            if (
                joint_grid.shape != reference_grid.shape
                or not np.allclose(
                    joint_grid, reference_grid, rtol=0.0, atol=1.0e-12
                )
            ):
                raise ValueError(
                    f"{kind} joint sampler incident-energy grid does not match "
                    f"the six incidence-specific CSV tables"
                )

    # For incidence alpha and a polar axis back toward the gun, the outward
    # vacuum hemisphere terminates at theta = 90 deg + alpha.  The supplied
    # 75-degree tables therefore end at 165 degrees.  Validate this convention
    # explicitly so a normal-relative table cannot be misinterpreted silently.
    expected_theta_max = 90.0 + angle
    for kind in ("BSE", "SE"):
        tables = model["theta"][kind]["tables"]
        theta_min = min(float(np.min(tab["theta"])) for tab in tables)
        theta_max = max(float(np.max(tab["theta"])) for tab in tables)
        if theta_min < -1.0e-9 or theta_max > expected_theta_max + 1.0e-6:
            raise ValueError(
                f"{kind} tilted-sample polar angles [{theta_min:g}, "
                f"{theta_max:g}] deg exceed the physical range "
                f"[0, {expected_theta_max:g}] deg"
            )
        if not np.isclose(theta_max, expected_theta_max, rtol=0.0, atol=0.25):
            raise ValueError(
                f"{kind} tilted-sample table ends at {theta_max:g} deg; "
                f"expected {expected_theta_max:g} deg for {angle:g}-degree "
                "incidence and a beam-relative polar axis"
            )

    if joint_model is not None:
        c_alpha = float(np.cos(np.deg2rad(angle)))
        s_alpha = float(np.sin(np.deg2rad(angle)))
        for kind in ("BSE", "SE"):
            for tab in joint_model[kind]["tables"]:
                d = np.asarray(tab["direction_local"], dtype=float)
                outward = d[:, 0] * c_alpha + d[:, 1] * s_alpha
                if np.any(outward <= 1.0e-12):
                    n_bad = int(np.count_nonzero(outward <= 1.0e-12))
                    raise ValueError(
                        f"{kind} joint sampler contains {n_bad} inward/tangent "
                        f"events at {angle:g} deg. Expected local basis "
                        "(beam_back, toward_normal, side); check phi convention."
                    )

    return model


# ============================================================
# Sample incidence-angle-dependent yield table
# ============================================================

def load_sample_angular_yield_csv(path: str | Path) -> dict:
    """
    Load the JMONSEL Cu yield-vs-energy-and-incidence-angle table.

    Expected format
    ---------------
    line 1: description
    line 2: E_beam (eV), Sample tilt (deg), BSEY, SEY
    remaining lines: one complete rectangular E x angle grid

    The absolute values are retained for interpolation, but the RFA model uses
    them only through Y(E, theta) / Y(E, 0). Thus the existing normal-incidence
    sample yield curves remain the absolute normalization.
    """
    path = Path(path)
    df = pd.read_csv(path, skiprows=1)

    required = ["E_beam (eV)", "Sample tilt (deg)", "BSEY", "SEY"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sample angular-yield table {path.name!r} is missing columns: {missing}"
        )

    work = df[required].copy()
    for c in required:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()

    Egrid = np.sort(work["E_beam (eV)"].unique().astype(float))
    Agrid = np.sort(work["Sample tilt (deg)"].unique().astype(float))

    if Egrid.size < 2 or Agrid.size < 2:
        raise ValueError("Sample angular-yield table needs at least 2 energies and 2 angles")
    if not np.any(np.isclose(Agrid, 0.0, atol=1.0e-12)):
        raise ValueError("Sample angular-yield table must contain a 0-degree row")
    if np.any(Egrid <= 0.0):
        raise ValueError("Sample angular-yield energies must be positive")
    if np.any(Agrid < 0.0) or np.any(Agrid >= 90.0):
        raise ValueError("Sample angular-yield angles must satisfy 0 <= theta < 90 deg")

    matrices = {}
    for kind in ("BSEY", "SEY"):
        piv = work.pivot(
            index="E_beam (eV)",
            columns="Sample tilt (deg)",
            values=kind,
        ).reindex(index=Egrid, columns=Agrid)

        if piv.isna().any().any():
            raise ValueError(
                f"Sample angular-yield table is not a complete rectangular grid for {kind}"
            )

        Y = piv.to_numpy(dtype=float)
        if np.any(~np.isfinite(Y)) or np.any(Y < 0.0):
            raise ValueError(f"Sample angular-yield {kind} values must be finite and non-negative")
        matrices[kind] = Y

    return {
        "E": Egrid,
        "theta_deg": Agrid,
        "BSEY": matrices["BSEY"],
        "SEY": matrices["SEY"],
        "source": path.name,
        "description": SAMPLE_ANGULAR_YIELD_SOURCE,
        "interpolation": "linear_in_angle_linear_in_log_energy",
        "normalization": "gain = JMONSEL(E,theta) / JMONSEL(E,0deg)",
    }


def interp_sample_angular_yield(
    angular_model: dict,
    Einc: float,
    theta_deg: float,
    kind: str,
) -> float:
    """
    Interpolate the absolute JMONSEL angular-yield table.

    Interpolation is deliberately shape-safe and inexpensive because this is
    called for every sample impact in a cascade:
      * linear in incidence angle;
      * linear in log(incident energy).

    Queries are clamped to the tabulated E and angle ranges. In particular,
    angles above the last row (89 deg in the supplied table) use the 89-degree
    value rather than extrapolating into the 90-degree singular limit.
    """
    kind = str(kind).upper()
    if kind not in ("SEY", "BSEY"):
        raise ValueError(f"Unknown sample angular-yield kind: {kind}")

    Egrid = np.asarray(angular_model["E"], dtype=float)
    Agrid = np.asarray(angular_model["theta_deg"], dtype=float)
    Y = np.asarray(angular_model[kind], dtype=float)

    E = float(np.clip(float(Einc), Egrid[0], Egrid[-1]))
    theta = float(np.clip(float(theta_deg), Agrid[0], Agrid[-1]))
    logE = float(np.log(E))
    logEgrid = np.log(Egrid)

    # Bracket angle, then interpolate each bracketing angular curve in energy.
    ihi = int(np.searchsorted(Agrid, theta, side="left"))
    if ihi <= 0:
        ilo = ihi = 0
        wa = 0.0
    elif ihi >= Agrid.size:
        ilo = ihi = Agrid.size - 1
        wa = 0.0
    elif np.isclose(theta, Agrid[ihi], atol=1.0e-12):
        ilo = ihi
        wa = 0.0
    else:
        ilo = ihi - 1
        wa = (theta - Agrid[ilo]) / (Agrid[ihi] - Agrid[ilo])

    ylo = float(np.interp(logE, logEgrid, Y[:, ilo]))
    if ihi == ilo:
        return max(0.0, ylo)

    yhi = float(np.interp(logE, logEgrid, Y[:, ihi]))
    return max(0.0, (1.0 - wa) * ylo + wa * yhi)


def sample_angle_dependent_yield_gains(
    yield_models: dict,
    Einc: float,
    cos_theta: float,
    reference_theta_deg: float = 0.0,
) -> tuple[float, float, float, str]:
    """
    Return (SEY_gain, BSEY_gain, theta_deg, model_label) for a Cu sample hit.

    ``reference_theta_deg`` is the incidence angle represented by the absolute
    SEY/BSEY curves being multiplied.  It is zero for the standard sample
    curves and 75 degrees for the optional 75-degree absolute-yield curves.
    This prevents the 75-degree angular enhancement from being counted twice.

    If no angular table is attached to yield_models["sample"], this function
    uses the ratio of capped-secant gains at the actual and reference angles.
    """
    c = float(np.clip(float(cos_theta), 0.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(c)))
    reference_theta_deg = float(reference_theta_deg)
    if not np.isfinite(reference_theta_deg) or not (
        0.0 <= reference_theta_deg < 90.0
    ):
        raise ValueError("sample yield reference angle must satisfy 0 <= angle < 90 deg")

    sample_models = yield_models.get("sample", {})
    angular_model = sample_models.get("angular_yields", None)

    if angular_model is None:
        g_actual = angular_yield_gain(cos_theta)
        g_reference = angular_yield_gain(
            float(np.cos(np.deg2rad(reference_theta_deg)))
        )
        g = float(g_actual / g_reference)
        return g, g, theta_deg, "capped_secant_ratio"

    y0_se = interp_sample_angular_yield(
        angular_model, Einc, reference_theta_deg, "SEY"
    )
    yth_se = interp_sample_angular_yield(angular_model, Einc, theta_deg, "SEY")
    y0_bse = interp_sample_angular_yield(
        angular_model, Einc, reference_theta_deg, "BSEY"
    )
    yth_bse = interp_sample_angular_yield(angular_model, Einc, theta_deg, "BSEY")

    # The supplied BSEY table has exactly zero normal-incidence BSEY at 50 eV.
    # BSE generation is disabled at Einc <= 50 eV elsewhere in the model, so a
    # neutral gain is the only well-defined choice at that exact zero.
    sey_gain = 1.0 if y0_se <= 1.0e-15 else yth_se / y0_se
    bsey_gain = 1.0 if y0_bse <= 1.0e-15 else yth_bse / y0_bse

    sey_gain = max(0.0, float(sey_gain))
    bsey_gain = max(0.0, float(bsey_gain))

    return (
        sey_gain,
        bsey_gain,
        theta_deg,
        (
            f"{angular_model.get('source', SAMPLE_ANGULAR_YIELD_FILENAME)}"
            f" normalized_at_{reference_theta_deg:g}deg"
        ),
    )


def resolve_sample_quantum_reflection_mode(yield_models: dict) -> str:
    """Return the active Cu-sample quantum-reflection treatment.

    Modes
    -----
    auto
        Disable the separate MATLAB-style quantum-reflection branch when the
        JMONSEL angular-yield table is loaded; otherwise reproduce the legacy
        replacement branch.
    legacy_replace
        Historical behavior: if quantum reflection is selected for a gun
        electron, emit one specular reflected primary and return immediately,
        without sampling ordinary SE/BSE emission from that impact.
    disabled
        Do not apply a separate quantum-reflection branch.
    """
    sample_models = yield_models.get("sample", {})
    requested = str(sample_models.get("quantum_reflection_mode", "auto")).strip().lower()

    if requested not in SAMPLE_QUANTUM_REFLECTION_MODES:
        raise ValueError(
            "sample quantum_reflection_mode must be one of "
            f"{sorted(SAMPLE_QUANTUM_REFLECTION_MODES)}, got {requested!r}"
        )

    if requested == "auto":
        return (
            "disabled"
            if sample_models.get("angular_yields", None) is not None
            else "legacy_replace"
        )

    return requested


def _sample_integer_count_from_mean(mean: float, rng) -> int:
    """Sample a non-negative integer with the requested mean.

    This is floor-plus-Bernoulli rather than Poisson.  For 0 <= mean < 1 it
    is exactly the historical Bernoulli sampler (including one RNG draw), so
    all established sub-unity BSE behavior is preserved.  For mean > 1 it
    naturally permits multiple BSEs while retaining the requested mean.
    """
    mean = max(0.0, float(mean))
    n_floor = int(np.floor(mean))
    frac = float(mean - n_floor)

    if n_floor == 0:
        return int(rng.random() < frac)
    if frac <= 0.0:
        return n_floor
    return n_floor + int(rng.random() < frac)


def describe_sample_angular_yields(
    yield_models: dict,
    energies=(200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0),
    angles_deg=(0.0, 30.0, 45.0, 60.0, 75.0, 80.0, 85.0),
    verbose: bool = True,
):
    """Return/print the SEY and BSEY angular gains actually used on the sample."""
    rows = []
    angular_model = yield_models.get("sample", {}).get("angular_yields", None)
    source = (
        angular_model.get("source", "<unknown>")
        if angular_model is not None
        else "capped_secant (no angular table loaded)"
    )

    if verbose:
        print("=" * 78)
        print("SAMPLE ANGULAR YIELD GAINS")
        print("=" * 78)
        print(f"model: {source}")
        print("normal-incidence absolute Cu yield curves are unchanged")
        print(
            "sample quantum reflection: "
            f"{resolve_sample_quantum_reflection_mode(yield_models)}"
        )
        print()
        print(f"{'E (eV)':>9} {'angle':>7} {'SEY gain':>11} {'BSEY gain':>11}")

    for E in energies:
        for theta in angles_deg:
            c = float(np.cos(np.deg2rad(float(theta))))
            sey_gain, bsey_gain, theta_used, model_label = (
                sample_angle_dependent_yield_gains(
                    yield_models=yield_models,
                    Einc=float(E),
                    cos_theta=c,
                )
            )
            row = {
                "E_eV": float(E),
                "theta_deg": float(theta_used),
                "SEY_gain": float(sey_gain),
                "BSEY_gain": float(bsey_gain),
                "model": model_label,
            }
            rows.append(row)
            if verbose:
                print(
                    f"{float(E):9.0f} {float(theta_used):7.1f} "
                    f"{float(sey_gain):11.4f} {float(bsey_gain):11.4f}"
                )

    return rows


# ============================================================
# Default model loader
# ============================================================

def load_default_surface_models(
    model_dir: str | Path,
    bronstein_dir: str | Path | None = None,
    use_measured_carbon_coating: bool = True,
    use_measured_carbon_for_grids: bool = False,
    use_sample_angle_dependent_yields: bool = True,
    sample_angular_yield_filename: str = SAMPLE_ANGULAR_YIELD_FILENAME,
    sample_quantum_reflection_mode: str = "auto",
    sample_gun_incidence_dir: str | Path | None = None,
    sample_gun_incidence_angle_deg: float = 75.0,
    sample_gun_incidence_tolerance_deg: float = 0.5,
    require_sample_gun_joint_sampler: bool = False,
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
    use_sample_angle_dependent_yields:
        If True (default), replace the sample's capped-secant angular yield law
        with the JMONSEL Cu yield-vs-energy-and-angle table. Only the angular
        gain Y(E,theta)/Y(E,0) is used, separately for SEY and BSEY, so the
        existing normal-incidence Cu yield curves remain unchanged.
    sample_angular_yield_filename:
        CSV filename, relative to model_dir, containing E_beam, sample tilt,
        BSEY, and SEY. The default is yieldsCu100TDDFT10ED.csv.
    sample_quantum_reflection_mode:
        ``"auto"`` (default) disables the separate MATLAB-style quantum-
        reflection replacement branch when the JMONSEL angular-yield table is
        active, avoiding double counting/suppression of the tabulated yield.
        ``"legacy_replace"`` reproduces the historical early-return branch;
        ``"disabled"`` always disables it.
    sample_gun_incidence_dir:
        Optional separate directory containing the six matched Cu yield,
        emitted-energy, and emitted-angle CSVs for one oblique gun-incidence
        angle.  These files are used only for first-generation gun impacts on
        the sample.  Keep this directory separate from ``model_dir`` because
        the incidence-specific files use the same filenames as the standard
        normal-incidence files.
    sample_gun_incidence_angle_deg:
        Incidence angle represented by ``sample_gun_incidence_dir``.  The
        current test data use 75 degrees.
    sample_gun_incidence_tolerance_deg:
        Allowed difference between the batch sample angle and the sampler
        angle.  The batch runner validates this before launching primaries.
    require_sample_gun_joint_sampler:
        If True, fail unless both event-level SE/BSE joint NPZ files are
        present in ``sample_gun_incidence_dir``.  This is recommended for
        oblique-incidence production runs because the legacy theta-only path
        must invent an azimuth and cannot preserve theta-phi correlation.

    Returns
    -------
    yield_models, energy_models, theta_models
    """
    model_dir = Path(model_dir)

    if sample_gun_incidence_dir is not None:
        sample_gun_incidence_dir = Path(sample_gun_incidence_dir)
        try:
            same_directory = (
                sample_gun_incidence_dir.resolve() == model_dir.resolve()
            )
        except OSError:
            same_directory = False
        if same_directory:
            raise ValueError(
                "sample_gun_incidence_dir must be separate from model_dir "
                "because the tilted and normal tables have identical filenames"
            )

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

    requested_qr_mode = str(sample_quantum_reflection_mode).strip().lower()
    if requested_qr_mode not in SAMPLE_QUANTUM_REFLECTION_MODES:
        raise ValueError(
            "sample_quantum_reflection_mode must be one of "
            f"{sorted(SAMPLE_QUANTUM_REFLECTION_MODES)}, got {sample_quantum_reflection_mode!r}"
        )

    sample_yield_models = {
        "BSEY": load_yield_curve_csv(model_dir / "BSEYFromPlane_SEVaccum_t0nmCuFPA.csv"),
        "SEY": load_yield_curve_csv(model_dir / "SEYFromPlane_SEVaccum_t0nmCuFPA.csv"),
        "quantum_reflection_mode": requested_qr_mode,
    }

    if use_sample_angle_dependent_yields:
        angular_path = model_dir / sample_angular_yield_filename
        if not angular_path.exists():
            raise FileNotFoundError(
                f"Angle-dependent Cu sample yields are enabled, but {angular_path} "
                "does not exist. Put yieldsCu100TDDFT10ED.csv in model_dir, or "
                "call load_default_surface_models(..., "
                "use_sample_angle_dependent_yields=False) to reproduce the old "
                "capped-secant sample model."
            )
        sample_yield_models["angular_yields"] = load_sample_angular_yield_csv(
            angular_path
        )

    yield_models = {
        "sample": sample_yield_models,
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

    if sample_gun_incidence_dir is not None:
        incidence_model = load_sample_gun_incidence_model(
            sampler_dir=sample_gun_incidence_dir,
            incidence_angle_deg=sample_gun_incidence_angle_deg,
            angle_tolerance_deg=sample_gun_incidence_tolerance_deg,
        )

        if incidence_model.get("joint", None) is None:
            msg = (
                "sample_gun_incidence_dir has no joint emitted-event NPZ "
                "samplers; falling back to the legacy theta-only model with "
                "uniform conditional azimuth. This is not physically complete "
                "at oblique incidence."
            )
            if require_sample_gun_joint_sampler:
                raise FileNotFoundError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        common = {
            "incidence_angle_deg": incidence_model["incidence_angle_deg"],
            "angle_tolerance_deg": incidence_model["angle_tolerance_deg"],
            "gun_only": True,
            "source_dir": incidence_model["source_dir"],
        }
        yield_models["sample"]["gun_incidence"] = {
            **common,
            "BSEY": incidence_model["yield"]["BSEY"],
            "SEY": incidence_model["yield"]["SEY"],
        }
        energy_models["sample"]["gun_incidence"] = {
            **common,
            "BSE": incidence_model["energy"]["BSE"],
            "SE": incidence_model["energy"]["SE"],
        }
        theta_models["sample"]["gun_incidence"] = {
            **common,
            "polar_axis": incidence_model["polar_axis"],
            "beam_back_axis": incidence_model["beam_back_axis"],
            "azimuth_model": incidence_model["azimuth_model"],
            "joint": incidence_model.get("joint", None),
            "BSE": incidence_model["theta"]["BSE"],
            "SE": incidence_model["theta"]["SE"],
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
    sample_yield_override: dict | None = None,
    sample_yield_reference_angle_deg: float = 0.0,
) -> tuple[int, int]:
    """
    Sample the number of BSE and SE electrons emitted by one surface impact.

    Multipliers
    -----------
    Yields come straight from the curves in yield_models; there are no
    tunable multipliers.

    Returns
    -------
    Nbse:
        Integer number of BSE electrons.  This remains exactly Bernoulli for
        sub-unity yields.  When the Cu JMONSEL angular table requests BSEY > 1,
        floor-plus-Bernoulli sampling permits multiple BSEs with the correct
        mean instead of silently clipping BSEY to 0.99.
    Nse:
        Poisson-sampled number of secondary electrons.
    """
    surf = canonical_surface_name(surface_name)
    model_key = _surface_model_key(yield_models, surface_name)

    if surf == "sample" and sample_yield_override is not None:
        sey_mdl = sample_yield_override["SEY"]
        bsey_mdl = sample_yield_override["BSEY"]
    else:
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

    sample_has_angular_table = bool(
        surf == "sample"
        and yield_models.get("sample", {}).get("angular_yields", None) is not None
    )

    if surf == "sample":
        sey_gain, bsey_gain, _theta_deg, _model_label = (
            sample_angle_dependent_yield_gains(
                yield_models=yield_models,
                Einc=Einc,
                cos_theta=cos_theta,
                reference_theta_deg=sample_yield_reference_angle_deg,
            )
        )
        sey_val = sey_base * sey_gain
        bsey_val = bsey_base * bsey_gain
    else:
        sey_val = sey_base * _geometry_gain(sey_mdl)
        bsey_val = bsey_base * _geometry_gain(bsey_mdl)

    # No yield multipliers.
    #
    # Every surface runs directly off its curve.  The Cu sample is special only
    # in one respect: when its explicit JMONSEL angular table is active, BSEY is
    # allowed to exceed unity and is represented by an integer multiplicity.
    # For every legacy path (including sample capped-secant mode and coated
    # hardware) retain the historical 0.99 BSE cap so existing runs remain
    # reproducible.
    sey_val = max(0.0, float(sey_val))
    if sample_has_angular_table:
        bsey_val = max(0.0, float(bsey_val))
    else:
        bsey_val = max(0.0, min(float(bsey_val), 0.99))

    if Einc <= 50.0:
        Nbse = 0
    else:
        Nbse = _sample_integer_count_from_mean(bsey_val, rng)

    Nse = int(rng.poisson(sey_val))

    return int(Nbse), Nse


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


def sample_joint_emission_event(
    model: dict,
    Einc: float,
    rng,
) -> dict:
    """Sample one correlated emitted-electron event.

    The incident-energy table is selected with the same stochastic bracketing
    convention used by the historical theta sampler.  A complete row is then
    drawn from that table, so emitted energy and all direction components stay
    coupled.  No independent phi draw is made.
    """
    Egrid = np.asarray(model["E"], dtype=float)
    tables = model["tables"]
    Equery = float(Einc)

    # Honor exact tabulated energies. The historical theta helper brackets
    # strict < and > neighbors even at an exact interior grid point; that is
    # not acceptable for event-level resampling because a 200 eV impact must
    # draw from the actual 200 eV SEEMC population.
    exact = np.where(np.isclose(Egrid, Equery, rtol=0.0, atol=1.0e-12))[0]
    if exact.size:
        tab = tables[int(exact[0])]
    else:
        tab = _choose_sampler_table(model, Equery, rng)

    Eout = np.asarray(tab["Eout_eV"], dtype=float)
    directions = np.asarray(tab["direction_local"], dtype=float)

    if Eout.ndim != 1 or directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("invalid joint emission sampler table shape")
    if len(Eout) == 0 or len(Eout) != len(directions):
        raise ValueError("joint emission sampler table is empty or inconsistent")

    idx = int(rng.integers(0, len(Eout)))
    d = unit(directions[idx])

    theta = float(np.arccos(np.clip(d[0], -1.0, 1.0)))
    phi = float(np.arctan2(d[2], d[1]))

    return {
        "Eout_eV": float(Eout[idx]),
        "direction_local": d,
        "theta_rad": theta,
        "phi_rad": phi,
        "sampler_Einc_eV": float(tab["Einc_eV"]),
        "sampler_event_index": idx,
        "source": model.get("source", "<joint sampler>"),
    }


def sample_surface_energy(
    energy_models: dict,
    surface_name: str,
    kind: str,
    Einc: float,
    rng,
    model_override: dict | None = None,
) -> float:
    model_key = _surface_model_key(energy_models, surface_name)
    kind = kind.upper()

    mdl = (
        model_override[kind]
        if model_override is not None
        else energy_models[model_key][kind]
    )

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
    model_override: dict | None = None,
) -> float:
    """
    Sample emitted polar angle in degrees.

    For cosine models:
        theta = asin(sqrt(u))
    """
    model_key = _surface_model_key(theta_models, surface_name)
    kind = kind.upper()

    mdl = (
        model_override[kind]
        if model_override is not None
        else theta_models[model_key][kind]
    )

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


def launch_sample_electron_beam_relative(
    theta_rad: float,
    Eout_eV: float,
    beam_back_axis,
    n_out,
    rng,
) -> tuple[np.ndarray, float, float]:
    """Launch a tilted-sample electron without changing its sampled polar angle.

    ``theta_rad`` is measured from the fixed axis back toward the gun/drift
    tube.  For the 75-degree tables it may span 0--165 degrees.  At fixed polar angle, sample
    azimuth uniformly only over the portion of that cone satisfying

        dot(direction, n_out) > 0,

    where ``n_out`` is the vacuum-side sample normal.  This is conditional
    azimuth sampling: theta is never rejected, reflected, or clamped, so the
    inverse-CDF polar distribution supplied by JMONSEL is preserved exactly.

    Returns
    -------
    velocity, phi_rad, outward_cosine
        ``phi_rad=0`` lies in the beam/sample-normal plane toward ``n_out``.
    """
    axis = unit(beam_back_axis)
    n_out = unit(n_out)
    theta = float(theta_rad)

    if not np.isfinite(theta) or theta < 0.0 or theta > np.pi:
        raise ValueError("beam-relative emission theta must lie in [0, pi]")

    # Orienting n_out against v_in should already guarantee c >= 0.  Retain a
    # tiny numerical tolerance but fail on a genuinely inconsistent geometry.
    c_alpha = float(np.dot(axis, n_out))
    if c_alpha < -1.0e-12:
        raise ValueError(
            "beam-back axis and sample outward normal point to opposite hemispheres"
        )
    c_alpha = float(np.clip(c_alpha, 0.0, 1.0))
    s_alpha = float(np.sqrt(max(0.0, 1.0 - c_alpha * c_alpha)))

    # e1 is the beam-transverse direction toward the outward normal.  e2
    # completes a right-handed basis around the beam-back polar axis.
    if s_alpha > 1.0e-14:
        e1 = unit(n_out - c_alpha * axis)
    else:
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, axis))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = unit(np.cross(axis, tmp))
    e2 = unit(np.cross(axis, e1))

    c_theta = float(np.cos(theta))
    s_theta = float(np.sin(theta))
    A = c_theta * c_alpha
    B = s_theta * s_alpha

    # A + B*cos(phi) > 0.  Draw from the open interval so an RNG value exactly
    # equal to zero cannot select a tangent boundary direction.
    u = float(np.clip(
        rng.random(),
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    ))

    if B <= 1.0e-15:
        if A <= 0.0:
            raise ValueError(
                "sampled polar angle has no outward azimuth for this sample incidence"
            )
        phi = -np.pi + 2.0 * np.pi * u
    else:
        q = -A / B
        if q <= -1.0:
            phi = -np.pi + 2.0 * np.pi * u
        elif q >= 1.0:
            raise ValueError(
                "sampled polar angle exceeds the outward vacuum-cone limit"
            )
        else:
            phi_max = float(np.arccos(np.clip(q, -1.0, 1.0)))
            phi = -phi_max + 2.0 * phi_max * u

    direction = unit(
        c_theta * axis
        + s_theta * (np.cos(phi) * e1 + np.sin(phi) * e2)
    )
    outward_cosine = float(np.dot(direction, n_out))
    if not outward_cosine > 0.0:
        raise RuntimeError(
            "outward-azimuth sampler produced a non-outward sample direction"
        )

    speed = speed_from_energy_eV(Eout_eV)
    return speed * direction, float(phi), outward_cosine


def launch_sample_electron_joint(
    direction_local,
    Eout_eV: float,
    beam_back_axis,
    n_out,
) -> tuple[np.ndarray, float]:
    """Launch one event from the correlated oblique-incidence sampler.

    ``direction_local`` contains components in the orthonormal basis

        (beam_back, toward_normal, side)

    used by the joint NPZ format.  Unlike the legacy tilted-sample launcher,
    this function performs no azimuth randomization, rejection, reflection, or
    clamping.  An event-level sampler represents electrons that already escaped
    the material, so a reconstructed inward direction indicates a coordinate-
    convention error and is rejected loudly.
    """
    axis = unit(beam_back_axis)
    n_out = unit(n_out)
    dloc = unit(np.asarray(direction_local, dtype=float))

    c_alpha = float(np.dot(axis, n_out))
    if c_alpha < -1.0e-12:
        raise ValueError(
            "beam-back axis and sample outward normal point to opposite hemispheres"
        )
    c_alpha = float(np.clip(c_alpha, 0.0, 1.0))
    s_alpha = float(np.sqrt(max(0.0, 1.0 - c_alpha * c_alpha)))

    if s_alpha > 1.0e-14:
        e1 = unit(n_out - c_alpha * axis)
    else:
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, axis))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = unit(np.cross(axis, tmp))
    e2 = unit(np.cross(axis, e1))

    direction = unit(dloc[0] * axis + dloc[1] * e1 + dloc[2] * e2)
    outward_cosine = float(np.dot(direction, n_out))
    if not outward_cosine > 1.0e-12:
        raise ValueError(
            "joint sample emission reconstructed inward/tangent to the Cu "
            "surface; check the NPZ phi/direction basis convention"
        )

    speed = speed_from_energy_eV(Eout_eV)
    return speed * direction, outward_cosine


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
    track_sub_barrier_sample_emissions: bool = False,
    visualization_rng=None,
) -> tuple[list[dict], dict]:
    """
    Generate emitted electrons from one surface impact.

    Returns
    -------
    emitted:
        List of dicts with p0, v0, E_emit_eV, kind, cos_theta. Optional
        sub-barrier sample emissions are tagged for visualization only.
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
        "sample_incidence_theta_deg": np.nan,
        "sample_sey_gain_used": np.nan,
        "sample_bsey_gain_used": np.nan,
        "sample_sey_mean_used": np.nan,
        "sample_bsey_mean_used": np.nan,
        "sample_angular_yield_model": None,
        "sample_bse_multiplicity_sampled": np.nan,
        "sample_quantum_reflection_mode": None,
        "sample_quantum_reflection_probability": np.nan,
        "sample_quantum_reflection_applied": False,
        "sample_gun_incidence_sampler_used": False,
        "sample_gun_incidence_sampler_angle_deg": np.nan,
        "sample_gun_incidence_sampler_source": None,
        "sample_emission_polar_axis": None,
        "sample_gun_joint_sampler_used": False,
        "sample_gun_joint_sampler_source": None,
        "sample_gun_azimuth_model": None,
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

    # Preserve the true incidence cosine all the way to grazing incidence.
    # The historical 0.05 floor was only a numerical guard for 1/cos(theta).
    # angular_yield_gain() is already capped safely, while the Cu sample now
    # needs the actual angle to query the JMONSEL angular table.
    cos_theta = float(np.clip(-float(np.dot(vhat, n_out)), 0.0, 1.0))

    sample_yield_override = None
    sample_energy_override = None
    sample_theta_override = None
    sample_joint_override = None
    sample_yield_reference_angle_deg = 0.0
    use_sample_gun_incidence = False

    if surf == "sample" and str(origin).lower() == "gun":
        incidence_configs = {
            "yield": yield_models.get("sample", {}).get("gun_incidence", None),
            "energy": energy_models.get("sample", {}).get("gun_incidence", None),
            "theta": theta_models.get("sample", {}).get("gun_incidence", None),
        }
        configured = [cfg is not None for cfg in incidence_configs.values()]
        if any(configured) and not all(configured):
            raise ValueError(
                "sample gun-incidence model must be present in yield, energy, "
                "and theta model dictionaries"
            )
        if all(configured):
            angles = np.asarray([
                float(cfg["incidence_angle_deg"])
                for cfg in incidence_configs.values()
            ])
            tolerances = np.asarray([
                float(cfg["angle_tolerance_deg"])
                for cfg in incidence_configs.values()
            ])
            if not np.allclose(angles, angles[0], rtol=0.0, atol=1.0e-12):
                raise ValueError("sample gun-incidence model angles are inconsistent")
            if not np.allclose(
                tolerances, tolerances[0], rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    "sample gun-incidence model angle tolerances are inconsistent"
                )

            use_sample_gun_incidence = True
            sample_yield_override = incidence_configs["yield"]
            sample_energy_override = incidence_configs["energy"]
            sample_theta_override = incidence_configs["theta"]
            sample_joint_override = sample_theta_override.get("joint", None)
            sample_yield_reference_angle_deg = float(angles[0])
            event_info.update({
                "sample_gun_incidence_sampler_used": True,
                "sample_gun_incidence_sampler_angle_deg": float(angles[0]),
                "sample_gun_incidence_sampler_source": str(
                    sample_theta_override.get("source_dir", "<unknown>")
                ),
                "sample_emission_polar_axis": "fixed_beam_back_axis",
                "sample_gun_joint_sampler_used": bool(
                    sample_joint_override is not None
                ),
                "sample_gun_joint_sampler_source": (
                    None
                    if sample_joint_override is None
                    else "; ".join(
                        str(sample_joint_override[k].get("source", "?"))
                        for k in ("SE", "BSE")
                    )
                ),
                "sample_gun_azimuth_model": sample_theta_override.get(
                    "azimuth_model", None
                ),
            })

    if surf == "sample":
        sample_sey_gain, sample_bsey_gain, sample_theta_deg, sample_model = (
            sample_angle_dependent_yield_gains(
                yield_models=yield_models,
                Einc=Einc,
                cos_theta=cos_theta,
                reference_theta_deg=sample_yield_reference_angle_deg,
            )
        )
        sample_absolute_yields = (
            sample_yield_override
            if sample_yield_override is not None
            else yield_models["sample"]
        )
        sample_sey_mean = max(
            0.0,
            interp_yield_model(sample_absolute_yields["SEY"], Einc)
            * float(sample_sey_gain),
        )
        sample_bsey_mean = max(
            0.0,
            interp_yield_model(sample_absolute_yields["BSEY"], Einc)
            * float(sample_bsey_gain),
        )
        if yield_models.get("sample", {}).get("angular_yields", None) is None:
            sample_bsey_mean = min(sample_bsey_mean, 0.99)

        event_info.update({
            "sample_incidence_theta_deg": float(sample_theta_deg),
            "sample_sey_gain_used": float(sample_sey_gain),
            "sample_bsey_gain_used": float(sample_bsey_gain),
            "sample_sey_mean_used": float(sample_sey_mean),
            "sample_bsey_mean_used": float(sample_bsey_mean),
            "sample_angular_yield_model": sample_model,
        })

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

    qr_mode = None
    if surface_name == "sample":
        qr_mode = resolve_sample_quantum_reflection_mode(yield_models)
        event_info.update({
            "sample_quantum_reflection_mode": qr_mode,
            "sample_quantum_reflection_probability": float(R_quantum),
        })

        if (
            str(origin).lower() == "gun"
            and sample_joint_override is not None
            and qr_mode != "disabled"
        ):
            raise ValueError(
                "The joint SEEMC gun-incidence sampler already contains "
                "vacuum->Cu barrier-reflected primaries. Set "
                "sample_quantum_reflection_mode='disabled' (recommended), or "
                "use auto only when it resolves to disabled. The legacy RFA "
                "reflection branch must not be combined with the joint sampler."
            )

    if surface_name == "sample" and origin == "gun" and qr_mode == "legacy_replace":
        if rng.random() < R_quantum:
            v_reflect = v_in - 2.0 * np.dot(v_in, n_out) * n_out

            emitted.append({
                "p0": p0_surface,
                "v0": v_reflect,
                "E_emit_eV": Einc,
                "Phi_emit": Phi_emit,
                "kind": "quantum_reflection",
                "cos_theta": cos_theta,
                "sub_barrier": False,
                "escape_eligible": True,
                "visualization_only": False,
            })
            event_info["sample_quantum_reflection_applied"] = True

            return emitted, event_info

    Nbse, Nse = sample_surface_event(
        yield_models=yield_models,
        surface_name=surface_name,
        Einc=Einc,
        cos_theta=cos_theta,
        rng=rng,
        sample_yield_override=sample_yield_override,
        sample_yield_reference_angle_deg=sample_yield_reference_angle_deg,
    )

    if surface_name == "sample":
        event_info["sample_bse_multiplicity_sampled"] = int(Nbse)

    for _ in range(Nbse):
        joint_event = None
        joint_energy_raw = np.nan
        joint_energy_clipped = False

        if use_sample_gun_incidence and sample_joint_override is not None:
            joint_event = sample_joint_emission_event(
                sample_joint_override["BSE"], Einc=Einc, rng=rng
            )
            joint_energy_raw = float(joint_event["Eout_eV"])
            E_bse = float(np.clip(joint_energy_raw, 50.0, Einc))
            joint_energy_clipped = not np.isclose(
                E_bse, joint_energy_raw, rtol=0.0, atol=1.0e-12
            )
        else:
            E_bse = sample_surface_energy(
                energy_models=energy_models,
                surface_name=surface_name,
                kind="BSE",
                Einc=Einc,
                rng=rng,
                model_override=sample_energy_override,
            )

        # Optional visualization-only launch of sample electrons below the
        # positive sample-bias escape threshold. The historical physics model
        # omitted these because they cannot escape; when enabled we launch them
        # so the field can visibly turn them back to the sample.
        sub_barrier_bse = bool(surface_name == "sample" and E_bse < Phi_emit)
        if sub_barrier_bse and not track_sub_barrier_sample_emissions:
            continue
        if E_bse <= 0.0:
            continue

        # In the legacy path, visualization-only angles/azimuths use a separate
        # RNG so enabling trajectory display does not perturb the physics RNG.
        # The joint path has already sampled the complete event above, so no
        # additional random draw is needed for its direction.
        rng_emit = (
            visualization_rng
            if (
                joint_event is None
                and sub_barrier_bse
                and visualization_rng is not None
            )
            else rng
        )

        if joint_event is not None:
            theta_bs = float(joint_event["theta_rad"])
            phi_bs = float(joint_event["phi_rad"])
        else:
            theta_bs = np.deg2rad(
                sample_surface_theta(
                    theta_models=theta_models,
                    surface_name=surface_name,
                    kind="BSE",
                    Einc=Einc,
                    rng=rng_emit,
                    model_override=sample_theta_override,
                )
            )

        # Apply launch-point potential correction to kinetic energy.
        E_bse_launch = max(E_bse + phi_correction, 1.0e-3)

        if joint_event is not None:
            v_bse, emission_outward_cosine = launch_sample_electron_joint(
                direction_local=joint_event["direction_local"],
                Eout_eV=E_bse_launch,
                beam_back_axis=sample_theta_override["beam_back_axis"],
                n_out=n_out,
            )
            emission_polar_axis = "fixed_beam_back_axis_joint"
            p0_bse = p0_surface

        elif use_sample_gun_incidence:
            v_bse, phi_bs, emission_outward_cosine = (
                launch_sample_electron_beam_relative(
                    theta_rad=theta_bs,
                    Eout_eV=E_bse_launch,
                    beam_back_axis=sample_theta_override["beam_back_axis"],
                    n_out=n_out,
                    rng=rng_emit,
                )
            )
            emission_polar_axis = "fixed_beam_back_axis"
            p0_bse = p0_surface

        elif surf in ANALYTIC_MESH_SURFACES:
            phi_bs = 2.0 * np.pi * rng_emit.random()
            axis_backscatter = -unit(v_in)

            v_bse = launch_electron_about_axis(
                theta_rad=theta_bs,
                phi_rad=phi_bs,
                Eout_eV=E_bse_launch,
                axis=axis_backscatter,
            )

            p0_bse = r_hit + sample_launch_eps * unit(v_bse)
            emission_polar_axis = "opposite_incoming_direction"
            emission_outward_cosine = float(np.dot(unit(v_bse), n_out))

        else:
            phi_bs = 2.0 * np.pi * rng_emit.random()
            use_full = is_fullsphere_surface(surface_name)

            v_bse = launch_surface_electron(
                theta_rad=theta_bs,
                phi_rad=phi_bs,
                Eout_eV=E_bse_launch,
                n_out=n_out,
                use_full_sphere=use_full,
            )

            p0_bse = p0_surface
            emission_polar_axis = "surface_normal"
            emission_outward_cosine = float(np.dot(unit(v_bse), n_out))

        dloc = (
            np.asarray(joint_event["direction_local"], dtype=float)
            if joint_event is not None
            else np.full(3, np.nan)
        )
        emitted.append({
            "p0": p0_bse,
            "v0": v_bse,
            "E_emit_eV": E_bse,
            "E_launch_eV": E_bse_launch,
            "Phi_emit": Phi_emit,
            "kind": "BSE",
            "cos_theta": cos_theta,
            "emission_theta_deg": float(np.rad2deg(theta_bs)),
            "emission_phi_deg": float(np.rad2deg(phi_bs)),
            "emission_polar_axis": emission_polar_axis,
            "emission_outward_cosine": emission_outward_cosine,
            "emission_mu_beam_back": float(dloc[0]),
            "emission_mu_toward_normal": float(dloc[1]),
            "emission_mu_side": float(dloc[2]),
            "sample_gun_incidence_sampler_used": bool(use_sample_gun_incidence),
            "sample_gun_joint_sampler_used": bool(joint_event is not None),
            "sample_gun_joint_sampler_source": (
                None if joint_event is None else joint_event.get("source", None)
            ),
            "joint_sampler_incident_energy_eV": (
                np.nan
                if joint_event is None
                else float(joint_event["sampler_Einc_eV"])
            ),
            "joint_sampler_event_index": (
                -1
                if joint_event is None
                else int(joint_event["sampler_event_index"])
            ),
            "joint_sampler_Eout_raw_eV": joint_energy_raw,
            "joint_sampler_energy_clipped": bool(joint_energy_clipped),
            "sub_barrier": sub_barrier_bse,
            "escape_eligible": not sub_barrier_bse,
            "visualization_only": sub_barrier_bse,
        })

    for _ in range(Nse):
        joint_event = None
        joint_energy_raw = np.nan
        joint_energy_clipped = False

        if use_sample_gun_incidence and sample_joint_override is not None:
            joint_event = sample_joint_emission_event(
                sample_joint_override["SE"], Einc=Einc, rng=rng
            )
            joint_energy_raw = float(joint_event["Eout_eV"])
            Ehi = min(50.0, Einc)
            E_se = float(np.clip(joint_energy_raw, min(0.01, Ehi), Ehi))
            joint_energy_clipped = not np.isclose(
                E_se, joint_energy_raw, rtol=0.0, atol=1.0e-12
            )
        else:
            E_se = sample_surface_energy(
                energy_models=energy_models,
                surface_name=surface_name,
                kind="SE",
                Einc=Einc,
                rng=rng,
                model_override=sample_energy_override,
            )

        sub_barrier_se = bool(surface_name == "sample" and E_se < Phi_emit)
        if sub_barrier_se and not track_sub_barrier_sample_emissions:
            continue
        if E_se <= 0.0:
            continue

        rng_emit = (
            visualization_rng
            if (
                joint_event is None
                and sub_barrier_se
                and visualization_rng is not None
            )
            else rng
        )

        if joint_event is not None:
            theta_se = float(joint_event["theta_rad"])
            phi_se = float(joint_event["phi_rad"])
        else:
            theta_se = np.deg2rad(
                sample_surface_theta(
                    theta_models=theta_models,
                    surface_name=surface_name,
                    kind="SE",
                    Einc=Einc,
                    rng=rng_emit,
                    model_override=sample_theta_override,
                )
            )

        E_se_launch = max(E_se + phi_correction, 1.0e-3)

        if joint_event is not None:
            v_se, emission_outward_cosine = launch_sample_electron_joint(
                direction_local=joint_event["direction_local"],
                Eout_eV=E_se_launch,
                beam_back_axis=sample_theta_override["beam_back_axis"],
                n_out=n_out,
            )
            emission_polar_axis = "fixed_beam_back_axis_joint"
            p0_se = p0_surface

        elif use_sample_gun_incidence:
            v_se, phi_se, emission_outward_cosine = (
                launch_sample_electron_beam_relative(
                    theta_rad=theta_se,
                    Eout_eV=E_se_launch,
                    beam_back_axis=sample_theta_override["beam_back_axis"],
                    n_out=n_out,
                    rng=rng_emit,
                )
            )
            emission_polar_axis = "fixed_beam_back_axis"
            p0_se = p0_surface

        elif surf in ANALYTIC_MESH_SURFACES:
            phi_se = 2.0 * np.pi * rng_emit.random()
            axis_backscatter = -unit(v_in)

            v_se = launch_electron_about_axis(
                theta_rad=theta_se,
                phi_rad=phi_se,
                Eout_eV=E_se_launch,
                axis=axis_backscatter,
            )

            p0_se = r_hit + sample_launch_eps * unit(v_se)
            emission_polar_axis = "opposite_incoming_direction"
            emission_outward_cosine = float(np.dot(unit(v_se), n_out))

        else:
            phi_se = 2.0 * np.pi * rng_emit.random()
            use_full = is_fullsphere_surface(surface_name)

            v_se = launch_surface_electron(
                theta_rad=theta_se,
                phi_rad=phi_se,
                Eout_eV=E_se_launch,
                n_out=n_out,
                use_full_sphere=use_full,
            )

            p0_se = p0_surface
            emission_polar_axis = "surface_normal"
            emission_outward_cosine = float(np.dot(unit(v_se), n_out))

        dloc = (
            np.asarray(joint_event["direction_local"], dtype=float)
            if joint_event is not None
            else np.full(3, np.nan)
        )
        emitted.append({
            "p0": p0_se,
            "v0": v_se,
            "E_emit_eV": E_se,
            "E_launch_eV": E_se_launch,
            "Phi_emit": (Phi_emit if sub_barrier_se else np.nan),
            "kind": "SE",
            "cos_theta": cos_theta,
            "emission_theta_deg": float(np.rad2deg(theta_se)),
            "emission_phi_deg": float(np.rad2deg(phi_se)),
            "emission_polar_axis": emission_polar_axis,
            "emission_outward_cosine": emission_outward_cosine,
            "emission_mu_beam_back": float(dloc[0]),
            "emission_mu_toward_normal": float(dloc[1]),
            "emission_mu_side": float(dloc[2]),
            "sample_gun_incidence_sampler_used": bool(use_sample_gun_incidence),
            "sample_gun_joint_sampler_used": bool(joint_event is not None),
            "sample_gun_joint_sampler_source": (
                None if joint_event is None else joint_event.get("source", None)
            ),
            "joint_sampler_incident_energy_eV": (
                np.nan
                if joint_event is None
                else float(joint_event["sampler_Einc_eV"])
            ),
            "joint_sampler_event_index": (
                -1
                if joint_event is None
                else int(joint_event["sampler_event_index"])
            ),
            "joint_sampler_Eout_raw_eV": joint_energy_raw,
            "joint_sampler_energy_clipped": bool(joint_energy_clipped),
            "sub_barrier": sub_barrier_se,
            "escape_eligible": not sub_barrier_se,
            "visualization_only": sub_barrier_se,
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

                nbse, nse = sample_surface_event(
                    yield_models=yield_models,
                    surface_name=surf,
                    Einc=float(E),
                    cos_theta=c_event,
                    rng=rng,
                )
                n_bse += int(nbse)
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