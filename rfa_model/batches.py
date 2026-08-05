"""
batches.py

Batch runners for the RFA model.

This module runs many primary electrons through:

    primary electron -> sample impact -> first-generation sample emission
    -> emitted-electron tracking -> accounting tables

At this stage, this module supports first-generation sample emission only.
Full cascade emission will be added later.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .primary import (
    make_primary_beam_near_sample,
    run_one_primary_with_model_emission,
    sample_center_from_bounds,
)

from .accounting import (
    summarize_many_first_generation,
    grid_events_to_dataframe_many,
)


# ============================================================
# Serial batch runner
# ============================================================

def run_first_generation_batch_serial(
    N_primary: int,
    E0_eV: float,
    field: dict,
    Phi_interp,
    Ex_interp,
    Ey_interp,
    Ez_interp,

    intersector_primary,
    face_owner_primary,
    collision_mesh_primary,
    stl_boxes_primary,

    intersector_emit,
    face_owner_emit,
    collision_mesh_emit,
    stl_boxes_emit,

    grid_transparency: dict,

    yield_models: dict,
    energy_models: dict,
    theta_models: dict,
    voltages: dict,
    sample_y_bounds,
    sample_z_bounds,

    x_start: float | None = None,
    beam_sigma: float = 150e-6,
    energy_spread_eV: float = 0.0,
    angular_sigma_deg: float = 0.0,
    seed: int = 1,

    progress_every: int | None = None,

    emitted_max_step_fraction_of_h: float = 0.40,
    emitted_dt_max: float = 2.0e-11,
    emitted_max_steps: int = 20000,
):
    """
    Run a serial first-generation sample-emission batch.

    Parameters
    ----------
    N_primary:
        Number of primary electrons.

    E0_eV:
        Desired landing energy at the sample.

    x_start:
        Primary start x position. If None, uses 0.75 * field["h"].

    progress_every:
        If not None, print progress every given number of primaries.

    Returns
    -------
    result:
        dict containing:
            primary_results
            emitted_results_all
            df_primary
            df_emit
            df_grid_events
            summary
            electrode_counts
            owner_counts
            kind_counts
            runtime_s
    """
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    if N_primary <= 0:
        raise ValueError("N_primary must be positive")

    if x_start is None:
        x_start = 0.75 * float(field["h"])

    y0, z0 = sample_center_from_bounds(
        sample_y_bounds,
        sample_z_bounds,
    )

    p0s, v0s, K0s, Phi0s = make_primary_beam_near_sample(
        N=N_primary,
        E0_eV=E0_eV,
        field=field,
        Phi_interp=Phi_interp,
        x_start=x_start,
        y0=y0,
        z0=z0,
        beam_sigma=beam_sigma,
        energy_spread_eV=energy_spread_eV,
        angular_sigma_deg=angular_sigma_deg,
        sample_voltage=voltages.get("Vs", 0.0),
        rng=rng,
    )

    primary_results = []
    emitted_results_all = []

    for i in range(N_primary):
        primary_res_i, emitted_i = run_one_primary_with_model_emission(
            p_primary=p0s[i],
            v_primary=v0s[i],
            field=field,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,
            Phi_interp=Phi_interp,

            intersector_primary=intersector_primary,
            face_owner_primary=face_owner_primary,
            collision_mesh_primary=collision_mesh_primary,
            stl_boxes_primary=stl_boxes_primary,

            intersector_emit=intersector_emit,
            face_owner_emit=face_owner_emit,
            collision_mesh_emit=collision_mesh_emit,
            stl_boxes_emit=stl_boxes_emit,

            grid_transparency=grid_transparency,

            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
            rng=rng,

            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,
            emitted_max_step_fraction_of_h=emitted_max_step_fraction_of_h,
            emitted_dt_max=emitted_dt_max,
            emitted_max_steps=emitted_max_steps,
        )

        primary_results.append(primary_res_i)
        emitted_results_all.append(emitted_i)

        if progress_every is not None:
            if (i + 1) % progress_every == 0 or (i + 1) == N_primary:
                print(f"{i + 1}/{N_primary} primaries complete")

    owner_name_map = field.get("owner_name_map", None)

    acct = summarize_many_first_generation(
        primary_results=primary_results,
        emitted_results_all=emitted_results_all,
        owner_name_map=owner_name_map,
    )

    df_grid_events = grid_events_to_dataframe_many(
        emitted_results_all,
        owner_name_map=owner_name_map,
    )

    runtime_s = time.perf_counter() - t0

    result = {
        "primary_results": primary_results,
        "emitted_results_all": emitted_results_all,

        "df_primary": acct["df_primary"],
        "df_emit": acct["df_emit"],
        "df_grid_events": df_grid_events,

        "summary": acct["summary"],
        "electrode_counts": acct["electrode_counts"],
        "owner_counts": acct["owner_counts"],
        "kind_counts": acct["kind_counts"],

        "runtime_s": runtime_s,
        "runtime_per_primary_s": runtime_s / N_primary if N_primary > 0 else np.nan,

        "p0s": p0s,
        "v0s": v0s,
        "K0s": K0s,
        "Phi0s": Phi0s,

        "emitted_max_step_fraction_of_h": emitted_max_step_fraction_of_h,
        "emitted_dt_max": emitted_dt_max,
        "emitted_max_steps": emitted_max_steps,
    }

    result = add_step_diagnostics(result)

    return result


# ============================================================
# Printing / summary helpers
# ============================================================

def print_batch_summary(result: dict):
    """
    Print compact batch summary.
    """
    s = result["summary"]

    print("First-generation batch summary")
    print("------------------------------")
    print(f"N primary:                 {s['N_primary']}")
    print(f"N primary hit sample:      {s['N_primary_hit_sample']}")
    print(f"Hit-sample fraction:       {s['primary_hit_sample_fraction']:.5f}")

    print(f"\nN emitted total:           {s['N_emitted_total']}")
    print(f"N SE:                      {s['N_SE']}")
    print(f"N BSE:                     {s['N_BSE']}")
    print(f"N quantum refl.:           {s['N_quantum_reflection']}")

    print(f"\nPer primary emitted total: {s['per_primary_emitted_total']:.5f}")
    print(f"Per primary SE:            {s['per_primary_SE']:.5f}")
    print(f"Per primary BSE:           {s['per_primary_BSE']:.5f}")

    print(f"\nRuntime:                   {result['runtime_s']:.2f} s")
    print(f"Runtime per primary:       {result['runtime_per_primary_s']:.4f} s")

    print("\nTerminal electrodes:")
    print(result["electrode_counts"])

    print("\nTerminal owners:")
    print(result["owner_counts"])

    print("\nEmission kinds:")
    print(result["kind_counts"])

    df_grid_events = result.get("df_grid_events", pd.DataFrame())

    if not df_grid_events.empty:
        print("\nGrid transmissions:")
        print(df_grid_events["electrode"].value_counts().reindex(
            ["grid1", "grid2", "grid3"],
            fill_value=0,
        ))


# ============================================================
# Saving helpers
# ============================================================

def save_batch_tables(result: dict, out_dir, prefix: str = "batch"):
    """
    Save df_primary, df_emit, df_grid_events, and summary to CSV files.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    df_primary_path = out_dir / f"{prefix}_primary.csv"
    df_emit_path = out_dir / f"{prefix}_emit.csv"
    df_grid_path = out_dir / f"{prefix}_grid_events.csv"
    summary_path = out_dir / f"{prefix}_summary.csv"

    result["df_primary"].to_csv(df_primary_path, index=False)
    result["df_emit"].to_csv(df_emit_path, index=False)
    result["df_grid_events"].to_csv(df_grid_path, index=False)

    pd.DataFrame([result["summary"]]).to_csv(summary_path, index=False)

    paths["df_primary"] = df_primary_path
    paths["df_emit"] = df_emit_path
    paths["df_grid_events"] = df_grid_path
    paths["summary"] = summary_path

    return paths


def add_step_diagnostics(result: dict) -> dict:
    """
    Add simple step/runtime diagnostics to a batch result.
    """
    df_emit = result["df_emit"]

    if df_emit.empty:
        result["step_diagnostics"] = {}
        return result

    diag = {
        "N_emit": int(len(df_emit)),
        "steps_mean": float(df_emit["steps"].mean()),
        "steps_median": float(df_emit["steps"].median()),
        "steps_max": int(df_emit["steps"].max()),
        "steps_p90": float(df_emit["steps"].quantile(0.90)),
        "steps_p95": float(df_emit["steps"].quantile(0.95)),
        "steps_p99": float(df_emit["steps"].quantile(0.99)),
        "slowest_electrons": df_emit.sort_values("steps", ascending=False).head(10),
    }

    result["step_diagnostics"] = diag

    return result


def print_step_diagnostics(result: dict):
    """
    Print step diagnostics from add_step_diagnostics().
    """
    if "step_diagnostics" not in result:
        result = add_step_diagnostics(result)

    d = result["step_diagnostics"]

    if not d:
        print("No emitted electrons.")
        return

    print("Step diagnostics")
    print("----------------")
    print(f"N emitted:     {d['N_emit']}")
    print(f"steps mean:    {d['steps_mean']:.1f}")
    print(f"steps median:  {d['steps_median']:.1f}")
    print(f"steps p90:     {d['steps_p90']:.1f}")
    print(f"steps p95:     {d['steps_p95']:.1f}")
    print(f"steps p99:     {d['steps_p99']:.1f}")
    print(f"steps max:     {d['steps_max']}")

    print("\nSlowest emitted electrons:")
    display_cols = [
        "primary_index",
        "electron_index",
        "emission_kind",
        "E_emit_eV",
        "terminal_owner",
        "terminal_electrode",
        "KE_hit_eV",
        "steps",
    ]

    print(d["slowest_electrons"][display_cols])


# ============================================================
# Parallel batch runner
# ============================================================

def _run_first_generation_chunk(
    chunk_index: int,
    p0s_chunk,
    v0s_chunk,
    seed: int,

    field,
    Phi_interp,
    Ex_interp,
    Ey_interp,
    Ez_interp,

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
    sample_y_bounds,
    sample_z_bounds,

    emitted_max_step_fraction_of_h,
    emitted_dt_max,
    emitted_max_steps,
):
    """
    Worker function for one chunk.

    Note: this assumes the intersectors can be serialized by joblib.
    If they cannot, we will switch to building intersectors inside each worker.
    """
    rng = np.random.default_rng(seed)

    primary_results = []
    emitted_results_all = []

    for i in range(len(p0s_chunk)):
        primary_res_i, emitted_i = run_one_primary_with_model_emission(
            p_primary=p0s_chunk[i],
            v_primary=v0s_chunk[i],
            field=field,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,
            Phi_interp=Phi_interp,

            intersector_primary=intersector_primary,
            face_owner_primary=face_owner_primary,
            collision_mesh_primary=collision_mesh_primary,
            stl_boxes_primary=stl_boxes_primary,

            intersector_emit=intersector_emit,
            face_owner_emit=face_owner_emit,
            collision_mesh_emit=collision_mesh_emit,
            stl_boxes_emit=stl_boxes_emit,

            grid_transparency=grid_transparency,

            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
            rng=rng,

            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,
            
            emitted_max_step_fraction_of_h=emitted_max_step_fraction_of_h,
            emitted_dt_max=emitted_dt_max,
            emitted_max_steps=emitted_max_steps,
        )

        primary_results.append(primary_res_i)
        emitted_results_all.append(emitted_i)

    return {
        "chunk_index": chunk_index,
        "primary_results": primary_results,
        "emitted_results_all": emitted_results_all,
    }


def run_first_generation_batch_parallel(
    N_primary: int,
    E0_eV: float,
    field: dict,
    Phi_interp,
    Ex_interp,
    Ey_interp,
    Ez_interp,

    intersector_primary,
    face_owner_primary,
    collision_mesh_primary,
    stl_boxes_primary,

    intersector_emit,
    face_owner_emit,
    collision_mesh_emit,
    stl_boxes_emit,

    grid_transparency: dict,

    yield_models: dict,
    energy_models: dict,
    theta_models: dict,
    voltages: dict,
    sample_y_bounds,
    sample_z_bounds,

    x_start: float | None = None,
    beam_sigma: float = 150e-6,
    energy_spread_eV: float = 0.0,
    angular_sigma_deg: float = 0.0,
    seed: int = 1,

    n_jobs: int = 4,
    chunk_size: int = 10,
    verbose: int = 10,

    emitted_max_step_fraction_of_h: float = 0.40,
    emitted_dt_max: float = 2.0e-11,
    emitted_max_steps: int = 20000,
):
    """
    Parallel first-generation batch runner using joblib.

    If this fails because trimesh/ray intersectors cannot be pickled,
    we will make a second version that builds intersectors inside workers.
    """
    from joblib import Parallel, delayed

    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if N_primary <= 0:
        raise ValueError("N_primary must be positive")

    if x_start is None:
        x_start = 0.75 * float(field["h"])

    y0, z0 = sample_center_from_bounds(
        sample_y_bounds,
        sample_z_bounds,
    )

    p0s, v0s, K0s, Phi0s = make_primary_beam_near_sample(
        N=N_primary,
        E0_eV=E0_eV,
        field=field,
        Phi_interp=Phi_interp,
        x_start=x_start,
        y0=y0,
        z0=z0,
        beam_sigma=beam_sigma,
        energy_spread_eV=energy_spread_eV,
        angular_sigma_deg=angular_sigma_deg,
        sample_voltage=voltages.get("Vs", 0.0),
        rng=rng,
    )

    chunks = []

    for start in range(0, N_primary, chunk_size):
        stop = min(start + chunk_size, N_primary)
        chunks.append((start, stop))

    chunk_seeds = seed + 1000 + np.arange(len(chunks))

    chunk_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=verbose,
    )(
        delayed(_run_first_generation_chunk)(
            chunk_index=ic,
            p0s_chunk=p0s[start:stop],
            v0s_chunk=v0s[start:stop],
            seed=int(chunk_seeds[ic]),

            field=field,
            Phi_interp=Phi_interp,
            Ex_interp=Ex_interp,
            Ey_interp=Ey_interp,
            Ez_interp=Ez_interp,

            intersector_primary=intersector_primary,
            face_owner_primary=face_owner_primary,
            collision_mesh_primary=collision_mesh_primary,
            stl_boxes_primary=stl_boxes_primary,

            intersector_emit=intersector_emit,
            face_owner_emit=face_owner_emit,
            collision_mesh_emit=collision_mesh_emit,
            stl_boxes_emit=stl_boxes_emit,

            grid_transparency=grid_transparency,

            yield_models=yield_models,
            energy_models=energy_models,
            theta_models=theta_models,
            voltages=voltages,
            sample_y_bounds=sample_y_bounds,
            sample_z_bounds=sample_z_bounds,

            emitted_max_step_fraction_of_h=emitted_max_step_fraction_of_h,
            emitted_dt_max=emitted_dt_max,
            emitted_max_steps=emitted_max_steps,
        )
        for ic, (start, stop) in enumerate(chunks)
    )

    # Preserve chunk order.
    chunk_results = sorted(chunk_results, key=lambda d: d["chunk_index"])

    primary_results = []
    emitted_results_all = []

    for cr in chunk_results:
        primary_results.extend(cr["primary_results"])
        emitted_results_all.extend(cr["emitted_results_all"])

    owner_name_map = field.get("owner_name_map", None)

    acct = summarize_many_first_generation(
        primary_results=primary_results,
        emitted_results_all=emitted_results_all,
        owner_name_map=owner_name_map,
    )

    df_grid_events = grid_events_to_dataframe_many(
        emitted_results_all,
        owner_name_map=owner_name_map,
    )

    runtime_s = time.perf_counter() - t0

    result = {
        "primary_results": primary_results,
        "emitted_results_all": emitted_results_all,

        "df_primary": acct["df_primary"],
        "df_emit": acct["df_emit"],
        "df_grid_events": df_grid_events,

        "summary": acct["summary"],
        "electrode_counts": acct["electrode_counts"],
        "owner_counts": acct["owner_counts"],
        "kind_counts": acct["kind_counts"],

        "runtime_s": runtime_s,
        "runtime_per_primary_s": runtime_s / N_primary if N_primary > 0 else np.nan,

        "p0s": p0s,
        "v0s": v0s,
        "K0s": K0s,
        "Phi0s": Phi0s,

        "n_jobs": n_jobs,
        "chunk_size": chunk_size,

        "emitted_max_step_fraction_of_h": emitted_max_step_fraction_of_h,
        "emitted_dt_max": emitted_dt_max,
        "emitted_max_steps": emitted_max_steps,
    }

    result = add_step_diagnostics(result)

    return result