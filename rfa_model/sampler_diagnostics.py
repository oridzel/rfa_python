"""
sampler_diagnostics.py

Diagnostic utilities for validating RFA surface-emission samplers.

This module checks that the Monte Carlo sampling routines in samplers.py
reproduce the tabulated yield, energy, and angular distributions.

The diagnostics are intended for notebooks, especially:

    04_sampler_debug.ipynb

Main diagnostic categories
--------------------------

1. Yield validation
   Compare input SEY/BSEY curves with Monte Carlo sampled averages.

2. Energy inverse-CDF validation
   Compare the tabulated inverse-CDF r -> Eout with the empirical
   inverse-CDF from sampled energies.

3. Theta inverse-CDF validation
   Compare the tabulated inverse-CDF r -> theta with the empirical
   inverse-CDF from sampled angles.

4. Histogram diagnostics
   Plot sampled SE/BSE energy or theta histograms.

Notation
--------

Einc:
    Incident electron kinetic energy at the surface, in eV.

Eout:
    Emitted electron kinetic energy, in eV.

r:
    Uniform random variable / CDF probability in [0, 1].
    The tabulated sampler files store inverse-CDF tables:
        r -> Eout
        r -> theta

theta:
    Polar emission angle measured from the local outward surface normal,
    in degrees in the CSV tables and diagnostics.

kind:
    Emission type, either "SE" or "BSE".

surface_name:
    Specific surface name such as:
        "sample"
        "holder"
        "receiver"
        "g1_shell"
        "g2_shell"
        "g3_shell"
        "collector_shell"

fam:
    Surface family used for model lookup:
        "sample"
        "holder"
        "receiver"
        "grid"
        "collector"

SEY:
    Mean number of true secondary electrons emitted per incident electron.

BSEY:
    Probability of one BSE event per incident electron in this model.
    BSE sampling is Bernoulli, not Poisson.

Important convention
--------------------

For BSE energy sampling, Eout is clipped to:

    50 eV <= Eout <= Einc

For SE energy sampling, Eout is clipped to:

    0.01 eV <= Eout <= min(50 eV, Einc)

So if an upper incident-energy table is sampled directly and then clipped,
an artificial pile-up at Einc can appear. The production sampler avoids this
by interpolating inverse-CDF values between neighboring incident-energy tables.

This module can show whether a suspicious histogram feature is caused by:
    - the input inverse-CDF table,
    - interpolation method,
    - clipping,
    - or Monte Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .samplers import (
    surface_family,
    interp_yield_model,
    sample_surface_event,
    sample_surface_energy,
    sample_surface_theta,
)


# ============================================================
# General helpers
# ============================================================

def empirical_inverse_cdf(samples):
    """
    Compute empirical inverse-CDF from sampled values.

    Parameters
    ----------
    samples:
        1D array-like sampled values.

    Returns
    -------
    r:
        Empirical CDF probabilities.
    q:
        Sorted sample values.
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    samples = np.sort(samples)

    r = (np.arange(len(samples)) + 0.5) / len(samples)

    return r, samples


def get_bracketing_table_indices(model: dict, Einc: float) -> list[int]:
    """
    Return incident-energy table indices bracketing Einc.
    """
    Egrid = np.asarray(model["E"], dtype=float)

    if Einc <= Egrid[0]:
        return [0]

    if Einc >= Egrid[-1]:
        return [len(Egrid) - 1]

    ilo = np.where(Egrid <= Einc)[0][-1]
    ihi = np.where(Egrid >= Einc)[0][0]

    return sorted(set([int(ilo), int(ihi)]))


def get_nearest_sampler_table(model: dict, Einc: float):
    """
    Return nearest incident-energy table.
    """
    Egrid = np.asarray(model["E"], dtype=float)
    idx = int(np.argmin(np.abs(Egrid - Einc)))

    return idx, Egrid[idx], model["tables"][idx]


# ============================================================
# Yield diagnostics
# ============================================================

def validate_yield_sampling_at_energy(
    yield_models,
    surface_name,
    E0,
    N=100_000,
    cos_theta=1.0,
    seed=1,
):
    """
    Compare sampled SEY/BSEY with expected input values at one energy.
    """
    rng = np.random.default_rng(seed)

    bse_count = 0
    se_count = 0

    for _ in range(int(N)):
        did_bse, Nse = sample_surface_event(
            yield_models=yield_models,
            surface_name=surface_name,
            Einc=E0,
            grid_SEY_mult=1.0,
            BSE_mult=1.0,
            grid_BSE_mult=1.0,
            collector_BSE_mult=1.0,
            cos_theta=cos_theta,
            rng=rng,
        )

        bse_count += int(did_bse)
        se_count += Nse

    bse_mc = bse_count / N
    se_mc = se_count / N

    fam = surface_family(surface_name)

    bse_raw = interp_yield_model(yield_models[fam]["BSEY"], E0)
    se_raw = interp_yield_model(yield_models[fam]["SEY"], E0)

    bse_expected = bse_raw / max(cos_theta, 0.05)
    bse_expected = max(0.0, min(0.99, bse_expected))

    if E0 <= 50:
        bse_expected = 0.0

    se_expected = se_raw / max(cos_theta, 0.05)

    if fam in ["grid", "collector"]:
        se_expected *= SEY_mult

    print(f"Surface: {surface_name}")
    print(f"Family: {fam}")
    print(f"Energy: {E0:.1f} eV")
    print(f"Input BSEY raw      = {bse_raw:.5f}")
    print(f"Expected BSEY used  = {bse_expected:.5f}")
    print(f"Sampled BSEY        = {bse_mc:.5f}")
    print(f"Input SEY raw       = {se_raw:.5f}")
    print(f"Expected SEY used   = {se_expected:.5f}")
    print(f"Sampled SEY         = {se_mc:.5f}")

    return {
        "surface": surface_name,
        "family": fam,
        "E0": E0,
        "N": N,
        "bse_input_raw": bse_raw,
        "bse_expected": bse_expected,
        "bse_sampled": bse_mc,
        "se_input_raw": se_raw,
        "se_expected": se_expected,
        "se_sampled": se_mc,
    }


def sweep_yield_sampling(
    yield_models,
    surface_name,
    Etest,
    N=50_000,
    cos_theta=1.0,
    seed=1,
):
    """
    Sweep incident energy and compare expected vs sampled SEY/BSEY.
    """
    rng = np.random.default_rng(seed)
    fam = surface_family(surface_name)

    rows = []

    for E0 in Etest:
        bse_count = 0
        se_count = 0

        for _ in range(int(N)):
            did_bse, Nse = sample_surface_event(
                yield_models=yield_models,
                surface_name=surface_name,
                Einc=E0,
                grid_SEY_mult=1.0,
                BSE_mult=1.0,
                grid_BSE_mult=1.0,
                collector_BSE_mult=1.0,
                cos_theta=cos_theta,
                rng=rng,
            )

            bse_count += int(did_bse)
            se_count += Nse

        bse_raw = interp_yield_model(yield_models[fam]["BSEY"], E0)
        se_raw = interp_yield_model(yield_models[fam]["SEY"], E0)

        bse_expected = bse_raw / max(cos_theta, 0.05)
        bse_expected = max(0.0, min(0.99, bse_expected))

        if E0 <= 50:
            bse_expected = 0.0

        se_expected = se_raw / max(cos_theta, 0.05)

        rows.append({
            "E_eV": E0,
            "BSEY_input_raw": bse_raw,
            "SEY_input_raw": se_raw,
            "BSEY_expected": bse_expected,
            "SEY_expected": se_expected,
            "BSEY_sampled": bse_count / N,
            "SEY_sampled": se_count / N,
        })

    return pd.DataFrame(rows)


def plot_yield_sampling_sweep(df, title=None):
    """
    Plot expected and sampled SEY/BSEY from sweep_yield_sampling().
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df["E_eV"], df["BSEY_expected"], "-", label="BSEY expected", linewidth=1.5)
    ax.plot(df["E_eV"], df["BSEY_sampled"], "o", label="BSEY sampled")

    ax.plot(df["E_eV"], df["SEY_expected"], "-", label="SEY expected", linewidth=1.5)
    ax.plot(df["E_eV"], df["SEY_sampled"], "o", label="SEY sampled")

    ax.set_xlabel("Incident energy (eV)")
    ax.set_ylabel("Yield")
    ax.grid(True)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    plt.show()

    return fig, ax


# ============================================================
# Distribution sampling helpers
# ============================================================

def sample_energy_distribution(
    energy_models,
    surface_name,
    kind,
    Einc,
    N=100_000,
    seed=1,
):
    """
    Draw N emitted-energy samples.
    """
    rng = np.random.default_rng(seed)

    samples = np.array([
        sample_surface_energy(
            energy_models=energy_models,
            surface_name=surface_name,
            kind=kind,
            Einc=Einc,
            rng=rng,
        )
        for _ in range(int(N))
    ])

    return samples


def sample_theta_distribution(
    theta_models,
    surface_name,
    kind,
    Einc,
    N=100_000,
    seed=1,
):
    """
    Draw N theta samples.
    """
    rng = np.random.default_rng(seed)

    samples = np.array([
        sample_surface_theta(
            theta_models=theta_models,
            surface_name=surface_name,
            kind=kind,
            Einc=Einc,
            rng=rng,
        )
        for _ in range(int(N))
    ])

    return samples


# ============================================================
# Energy diagnostics
# ============================================================

def plot_raw_energy_sampler_table(
    energy_models,
    surface_name,
    kind,
    Einc,
):
    """
    Plot raw bracketing inverse-CDF energy tables.
    """
    fam = surface_family(surface_name)
    kind = kind.upper()

    model = energy_models[fam][kind]
    Egrid = np.asarray(model["E"], dtype=float)
    idxs = get_bracketing_table_indices(model, Einc)

    fig, ax = plt.subplots(figsize=(7, 5))

    for idx in idxs:
        tab = model["tables"][idx]
        ax.plot(
            tab["r"],
            tab["Eout"],
            "o-",
            markersize=3,
            label=f"table E={Egrid[idx]:.0f} eV",
        )

    ax.set_xlabel("CDF probability r")
    ax.set_ylabel("Emitted energy (eV)")
    ax.set_title(f"Raw inverse-CDF table: {surface_name} {kind}, Einc={Einc:.0f} eV")
    ax.grid(True)
    ax.legend()
    plt.show()

    return fig, ax


def plot_energy_inverse_cdf_validation(
    energy_models,
    surface_name,
    kind,
    Einc,
    N=100_000,
    seed=1,
    bins=80,
):
    """
    Compare bracketing input inverse-CDF tables with empirical sampled quantile.
    """
    fam = surface_family(surface_name)
    kind = kind.upper()

    model = energy_models[fam][kind]
    Egrid = np.asarray(model["E"], dtype=float)

    samples = sample_energy_distribution(
        energy_models=energy_models,
        surface_name=surface_name,
        kind=kind,
        Einc=Einc,
        N=N,
        seed=seed,
    )

    r_emp, q_emp = empirical_inverse_cdf(samples)
    idxs = get_bracketing_table_indices(model, Einc)

    fig, ax = plt.subplots(figsize=(7, 5))

    for idx in idxs:
        tab = model["tables"][idx]
        ax.plot(
            tab["r"],
            tab["Eout"],
            "-",
            linewidth=1.2,
            label=f"Input table E={Egrid[idx]:.0f} eV",
        )

    ax.plot(
        r_emp,
        q_emp,
        "--",
        linewidth=1.8,
        label=f"Sampled empirical quantile, Einc={Einc:.0f} eV",
    )

    if kind == "BSE":
        ax.axhline(Einc, linestyle=":", linewidth=1.2, label="Incident energy cutoff")

    ax.set_xlabel("CDF probability r")
    ax.set_ylabel("Emitted energy (eV)")
    ax.set_title(f"{surface_name} {kind} energy inverse-CDF validation")
    ax.grid(True)
    ax.legend()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(samples, bins=bins, density=True, alpha=0.7)
    ax2.set_xlabel("Emitted energy (eV)")
    ax2.set_ylabel("Probability density")
    ax2.set_title(f"{surface_name} {kind} sampled energy histogram, Einc={Einc:.0f} eV")
    ax2.grid(True)
    plt.show()

    print("Sample summary:")
    print(pd.Series(samples).describe())

    return samples


# ============================================================
# Theta diagnostics
# ============================================================

def plot_theta_inverse_cdf_validation(
    theta_models,
    surface_name,
    kind,
    Einc,
    N=100_000,
    seed=1,
    bins=80,
):
    """
    Compare input theta inverse-CDF with empirical sampled quantile.
    """
    fam = surface_family(surface_name)
    kind = kind.upper()

    model = theta_models[fam][kind]

    samples = sample_theta_distribution(
        theta_models=theta_models,
        surface_name=surface_name,
        kind=kind,
        Einc=Einc,
        N=N,
        seed=seed,
    )

    r_emp, q_emp = empirical_inverse_cdf(samples)

    fig, ax = plt.subplots(figsize=(7, 5))

    if isinstance(model, str) and model.lower() == "cosine":
        r_input = np.linspace(0, 1, 500)
        theta_input = np.rad2deg(np.arcsin(np.sqrt(r_input)))

        ax.plot(
            r_input,
            theta_input,
            "-",
            linewidth=1.5,
            label="Input cosine inverse-CDF",
        )
    else:
        Egrid = np.asarray(model["E"], dtype=float)
        idxs = get_bracketing_table_indices(model, Einc)

        for idx in idxs:
            tab = model["tables"][idx]
            ax.plot(
                tab["r"],
                tab["theta"],
                "-",
                linewidth=1.2,
                label=f"Input table E={Egrid[idx]:.0f} eV",
            )

    ax.plot(
        r_emp,
        q_emp,
        "--",
        linewidth=1.8,
        label=f"Sampled empirical quantile, Einc={Einc:.0f} eV",
    )

    ax.set_xlabel("CDF probability r")
    ax.set_ylabel("Theta from surface normal (deg)")
    ax.set_title(f"{surface_name} {kind} theta inverse-CDF validation")
    ax.grid(True)
    ax.legend()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(samples, bins=bins, density=True, alpha=0.7)
    ax2.set_xlabel("Theta from surface normal (deg)")
    ax2.set_ylabel("Probability density")
    ax2.set_title(f"{surface_name} {kind} sampled theta histogram, Einc={Einc:.0f} eV")
    ax2.grid(True)
    plt.show()

    print("Sample summary:")
    print(pd.Series(samples).describe())

    return samples