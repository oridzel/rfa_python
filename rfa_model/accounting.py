"""
accounting.py

Current and yield accounting for RFA trajectory results.

This module converts terminal trajectory outcomes into electrode-level
counts/currents.

At this stage it supports first-generation sample-emission results:

    primary electron hits sample
    sample emits SE/BSE electrons
    emitted electrons terminate on grid/collector/holder/sample/escape

Later this can be extended to full cascade accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Electrode / owner-name normalization
# ============================================================

def canonical_owner_name(name) -> str:
    """
    Normalize terminal owner/surface names.

    Input names may come from:
        STL owner
        analytic shell owner
        fixed voxel owner
        trajectory reason
    """
    if name is None:
        return "unknown"

    name = str(name)

    mapping = {
        # grid shells
        "g1_shell": "g1_shell",
        "g2_shell": "g2_shell",
        "g3_shell": "g3_shell",

        # grid frames
        "g1frame": "g1frame",
        "g2frame": "g2frame",
        "g3frame": "g3frame",
        "g1_low_frame": "g1frame",
        "g1_upper_frame": "g1frame",
        "g2_low_frame": "g2frame",
        "g2_upper_frame": "g2frame",
        "g3_low_frame": "g3frame",
        "g3_upper_frame": "g3frame",

        # collector
        "collector": "collector",
        "collector_shell": "collector_shell",

        # sample assembly
        "sample": "sample",
        "sample_plane": "sample",
        "sample_voxel": "sample",
        "holder": "holder",
        "receiver": "receiver",
        "rod": "rod",

        # drift tube / escape
        "drifttube": "drifttube",
        "left_grid": "escaped",
        "escaped_grid": "escaped",
        "escaped": "escaped",

        # raw reasons
        "hit_sample": "sample",
        "hit_collector": "collector_shell",
        "hit_grid_wire": "grid_wire",
        "transmit_grid": "grid_transmit",
        "hit_stl": "stl",
        "hit_fixed": "fixed_voxel",
    }

    return mapping.get(name, name)


def electrode_from_owner(owner_name) -> str:
    """
    Collapse physical owner/surface names into measured electrode channels.

    Electrode channels:
        sample
        holder
        receiver
        rod
        grid1
        grid2
        grid3
        collector
        drifttube
        escaped
        unknown
    """
    name = canonical_owner_name(owner_name)

    if name in ["sample"]:
        return "sample"

    if name == "holder":
        return "holder"

    if name == "receiver":
        return "receiver"

    if name == "rod":
        return "rod"

    if name in ["g1_shell", "g1frame"]:
        return "grid1"

    if name in ["g2_shell", "g2frame"]:
        return "grid2"

    if name in ["g3_shell", "g3frame"]:
        return "grid3"

    if name in ["collector", "collector_shell"]:
        return "collector"

    if name == "drifttube":
        return "drifttube"

    if name == "escaped":
        return "escaped"

    return "unknown"


def terminal_owner_from_result(res, owner_name_map=None):
    reason = res.get("reason", None)

    if reason in ["left_grid", "left_update_region", "escaped"]:
        return "escaped"

    hit_info = res.get("hit_info", None)

    if hit_info is None:
        return "unknown"

    owner_name = hit_info.get("owner_name", None)

    if owner_name is not None:
        return canonical_owner_name(owner_name)

    owner_id = hit_info.get("owner_id", None)

    if owner_id is not None and owner_name_map is not None:
        return canonical_owner_name(owner_name_map.get(int(owner_id), "unknown"))

    return "unknown"


def terminal_electrode_from_result(
    res: dict,
    owner_name_map: dict | None = None,
) -> str:
    """
    Extract terminal electrode channel from one trajectory result.
    """
    owner = terminal_owner_from_result(
        res,
        owner_name_map=owner_name_map,
    )

    return electrode_from_owner(owner)


# ============================================================
# Result table conversion
# ============================================================

def emitted_results_to_dataframe(
    emitted_results: list[dict],
    owner_name_map: dict | None = None,
) -> pd.DataFrame:
    """
    Convert emitted-electron trajectory results to a compact dataframe.

    Each row is one emitted electron.
    """
    rows = []

    for i, res in enumerate(emitted_results):
        hit_info = res.get("hit_info", None)

        owner = terminal_owner_from_result(
            res,
            owner_name_map=owner_name_map,
        )

        electrode = electrode_from_owner(owner)

        if hit_info is None:
            owner_id = None
            KE_hit_eV = np.nan
            location = np.array([np.nan, np.nan, np.nan])
        else:
            owner_id = hit_info.get("owner_id", None)
            KE_hit_eV = hit_info.get("KE_hit_eV", np.nan)
            location = hit_info.get("location", None)

            if location is None:
                traj = res.get("traj", None)
                if traj is not None and len(traj) > 0:
                    location = np.asarray(traj)[-1]
                else:
                    location = np.array([np.nan, np.nan, np.nan])

        location = np.asarray(location, dtype=float)

        rows.append({
            "electron_index": i,
            "emission_kind": res.get("emission_kind", None),
            "E_emit_eV": res.get("E_emit_eV", np.nan),
            "primary_E_inc_eV": res.get("primary_E_inc_eV", np.nan),
            "primary_cos_theta": res.get("primary_cos_theta", np.nan),

            "reason": res.get("reason", None),
            "terminal_owner": owner,
            "terminal_electrode": electrode,
            "owner_id": owner_id,
            "KE_hit_eV": KE_hit_eV,
            "steps": res.get("steps", np.nan),

            "x_hit": location[0],
            "y_hit": location[1],
            "z_hit": location[2],
        })

    return pd.DataFrame(rows)


def primary_result_to_row(primary_result: dict, primary_index: int = 0) -> dict:
    """
    Convert one primary result to a compact row.
    """
    hit_info = primary_result.get("hit_info", None)

    if hit_info is None:
        owner = terminal_owner_from_result(primary_result)
        electrode = electrode_from_owner(owner)
        KE_hit_eV = np.nan
        location = np.array([np.nan, np.nan, np.nan])
    else:
        owner = terminal_owner_from_result(primary_result)
        electrode = electrode_from_owner(owner)
        KE_hit_eV = hit_info.get("KE_hit_eV", np.nan)
        location = hit_info.get("location", None)

        if location is None:
            traj = primary_result.get("traj", None)
            if traj is not None and len(traj) > 0:
                location = np.asarray(traj)[-1]
            else:
                location = np.array([np.nan, np.nan, np.nan])

    location = np.asarray(location, dtype=float)

    return {
        "primary_index": primary_index,
        "reason": primary_result.get("reason", None),
        "terminal_owner": owner,
        "terminal_electrode": electrode,
        "KE_hit_eV": KE_hit_eV,
        "steps": primary_result.get("steps", np.nan),
        "x_hit": location[0],
        "y_hit": location[1],
        "z_hit": location[2],
    }


# ============================================================
# First-generation sample-emission accounting
# ============================================================

def count_by_electrode(df_emit: pd.DataFrame) -> pd.Series:
    """
    Count emitted electrons terminating on each electrode.
    """
    channels = [
        "sample",
        "holder",
        "receiver",
        "rod",
        "grid1",
        "grid2",
        "grid3",
        "collector",
        "drifttube",
        "escaped",
        "unknown",
    ]

    if df_emit.empty:
        return pd.Series(0, index=channels, dtype=int)

    counts = df_emit["terminal_electrode"].value_counts()

    return counts.reindex(channels, fill_value=0).astype(int)


def count_by_owner(df_emit: pd.DataFrame) -> pd.Series:
    """
    Count emitted electrons terminating on detailed owner names.
    """
    if df_emit.empty:
        return pd.Series(dtype=int)

    return df_emit["terminal_owner"].value_counts()


def count_by_kind(df_emit: pd.DataFrame) -> pd.Series:
    """
    Count emitted electrons by emission kind: SE, BSE, quantum_reflection, etc.
    """
    if df_emit.empty:
        return pd.Series(dtype=int)

    return df_emit["emission_kind"].value_counts()


def summarize_first_generation(
    primary_result: dict,
    emitted_results: list[dict],
    N_primary: int = 1,
    owner_name_map: dict | None = None,
) -> dict:
    """
    Summarize first-generation sample emission for one or more primaries.

    For one primary, N_primary=1.

    This function reports counts per primary. It does not yet include
    full cascade secondary emission.
    """
    df_emit = emitted_results_to_dataframe(
        emitted_results,
        owner_name_map=owner_name_map,
    )

    electrode_counts = count_by_electrode(df_emit)
    owner_counts = count_by_owner(df_emit)
    kind_counts = count_by_kind(df_emit)

    primary_hit_sample = primary_result.get("reason") == "hit_sample"

    summary = {
        "N_primary": int(N_primary),
        "primary_hit_sample": int(primary_hit_sample),

        "N_emitted_total": int(len(df_emit)),
        "N_SE": int(kind_counts.get("SE", 0)),
        "N_BSE": int(kind_counts.get("BSE", 0)),
        "N_quantum_reflection": int(kind_counts.get("quantum_reflection", 0)),

        "per_primary_emitted_total": len(df_emit) / N_primary,
        "per_primary_SE": kind_counts.get("SE", 0) / N_primary,
        "per_primary_BSE": kind_counts.get("BSE", 0) / N_primary,
    }

    for electrode, count in electrode_counts.items():
        summary[f"N_to_{electrode}"] = int(count)
        summary[f"per_primary_to_{electrode}"] = count / N_primary

    return {
        "summary": summary,
        "df_emit": df_emit,
        "electrode_counts": electrode_counts,
        "owner_counts": owner_counts,
        "kind_counts": kind_counts,
    }


def print_first_generation_summary(accounting_result: dict):
    """
    Pretty-print summarize_first_generation() output.
    """
    s = accounting_result["summary"]

    print("First-generation sample-emission accounting")
    print("------------------------------------------")
    print(f"N primary:       {s['N_primary']}")
    print(f"Primary sample hits: {s['primary_hit_sample']}")
    print(f"N emitted total: {s['N_emitted_total']}")
    print(f"N SE:            {s['N_SE']}")
    print(f"N BSE:           {s['N_BSE']}")
    print(f"N quantum refl.: {s['N_quantum_reflection']}")

    print("\nTerminal electrodes:")
    print(accounting_result["electrode_counts"])

    print("\nTerminal owners:")
    print(accounting_result["owner_counts"])

    print("\nEmission kinds:")
    print(accounting_result["kind_counts"])


def summarize_many_first_generation(
    primary_results: list[dict],
    emitted_results_all: list[list[dict]],
    owner_name_map: dict | None = None,
) -> dict:
    """
    Summarize first-generation sample emission for many primaries.

    Parameters
    ----------
    primary_results:
        List of primary trajectory results.

    emitted_results_all:
        List where each element is the emitted_results list for one primary.

    owner_name_map:
        Optional owner-id decoder, usually field["owner_name_map"].

    Returns
    -------
    dict with summary, df_primary, df_emit, counts.
    """
    N_primary = len(primary_results)

    primary_rows = []
    emit_dfs = []

    for i, primary_result in enumerate(primary_results):
        primary_rows.append(
            primary_result_to_row(
                primary_result,
                primary_index=i,
            )
        )

        emitted_results = emitted_results_all[i]

        df_i = emitted_results_to_dataframe(
            emitted_results,
            owner_name_map=owner_name_map,
        )

        if not df_i.empty:
            df_i.insert(0, "primary_index", i)

        emit_dfs.append(df_i)

    df_primary = pd.DataFrame(primary_rows)

    if len(emit_dfs) > 0:
        df_emit = pd.concat(emit_dfs, ignore_index=True)
    else:
        df_emit = pd.DataFrame()

    electrode_counts = count_by_electrode(df_emit)
    owner_counts = count_by_owner(df_emit)
    kind_counts = count_by_kind(df_emit)

    N_hit_sample = int((df_primary["reason"] == "hit_sample").sum())

    summary = {
        "N_primary": int(N_primary),
        "N_primary_hit_sample": N_hit_sample,
        "primary_hit_sample_fraction": N_hit_sample / N_primary if N_primary > 0 else np.nan,

        "N_emitted_total": int(len(df_emit)),
        "N_SE": int(kind_counts.get("SE", 0)),
        "N_BSE": int(kind_counts.get("BSE", 0)),
        "N_quantum_reflection": int(kind_counts.get("quantum_reflection", 0)),

        "per_primary_emitted_total": len(df_emit) / N_primary if N_primary > 0 else np.nan,
        "per_primary_SE": kind_counts.get("SE", 0) / N_primary if N_primary > 0 else np.nan,
        "per_primary_BSE": kind_counts.get("BSE", 0) / N_primary if N_primary > 0 else np.nan,
    }

    for electrode, count in electrode_counts.items():
        summary[f"N_to_{electrode}"] = int(count)
        summary[f"per_primary_to_{electrode}"] = count / N_primary if N_primary > 0 else np.nan

    return {
        "summary": summary,
        "df_primary": df_primary,
        "df_emit": df_emit,
        "electrode_counts": electrode_counts,
        "owner_counts": owner_counts,
        "kind_counts": kind_counts,
    }


def print_many_first_generation_summary(accounting_result: dict):
    """
    Pretty-print summarize_many_first_generation() output.
    """
    s = accounting_result["summary"]

    print("Many-primary first-generation accounting")
    print("---------------------------------------")
    print(f"N primary:             {s['N_primary']}")
    print(f"N primary hit sample:  {s['N_primary_hit_sample']}")
    print(f"Hit-sample fraction:   {s['primary_hit_sample_fraction']:.5f}")

    print(f"\nN emitted total:       {s['N_emitted_total']}")
    print(f"N SE:                  {s['N_SE']}")
    print(f"N BSE:                 {s['N_BSE']}")
    print(f"N quantum refl.:       {s['N_quantum_reflection']}")

    print(f"\nPer primary emitted:   {s['per_primary_emitted_total']:.5f}")
    print(f"Per primary SE:        {s['per_primary_SE']:.5f}")
    print(f"Per primary BSE:       {s['per_primary_BSE']:.5f}")

    print("\nTerminal electrodes:")
    print(accounting_result["electrode_counts"])

    print("\nTerminal owners:")
    print(accounting_result["owner_counts"])

    print("\nEmission kinds:")
    print(accounting_result["kind_counts"])


# ============================================================
# Grid-event accounting
# ============================================================

def grid_events_to_dataframe(
    emitted_results: list[dict],
    owner_name_map: dict | None = None,
) -> pd.DataFrame:
    """
    Convert transmitted-grid events from emitted-electron results
    into a dataframe.

    Each row is one grid crossing/transmission event.
    """
    rows = []

    for electron_index, res in enumerate(emitted_results):
        grid_events = res.get("grid_events", [])

        for event_index, ev in enumerate(grid_events):
            owner = ev.get("owner", None)

            # Some older results may store owner_id instead.
            owner_id = ev.get("owner_id", None)

            if owner is None and owner_id is not None and owner_name_map is not None:
                owner = owner_name_map.get(int(owner_id), f"owner_{owner_id}")

            owner = canonical_owner_name(owner)
            electrode = electrode_from_owner(owner)

            location = ev.get("location", None)
            if location is None:
                location = np.array([np.nan, np.nan, np.nan])
            location = np.asarray(location, dtype=float)

            rows.append({
                "electron_index": electron_index,
                "event_index": event_index,
                "event_type": ev.get("type", None),
                "owner": owner,
                "electrode": electrode,
                "step": ev.get("step", np.nan),
                "x": location[0],
                "y": location[1],
                "z": location[2],
            })

    return pd.DataFrame(rows)


def count_grid_transmissions(
    emitted_results: list[dict],
    owner_name_map: dict | None = None,
) -> pd.Series:
    """
    Count transmitted grid crossings by electrode.
    """
    df_events = grid_events_to_dataframe(
        emitted_results,
        owner_name_map=owner_name_map,
    )

    channels = ["grid1", "grid2", "grid3"]

    if df_events.empty:
        return pd.Series(0, index=channels, dtype=int)

    counts = df_events["electrode"].value_counts()

    return counts.reindex(channels, fill_value=0).astype(int)


def grid_events_to_dataframe_many(
    emitted_results_all: list[list[dict]],
    owner_name_map: dict | None = None,
) -> pd.DataFrame:
    """
    Convert grid events from many-primary emitted results into a dataframe.
    """
    dfs = []

    for primary_index, emitted_results in enumerate(emitted_results_all):
        df_i = grid_events_to_dataframe(
            emitted_results,
            owner_name_map=owner_name_map,
        )

        if not df_i.empty:
            df_i.insert(0, "primary_index", primary_index)

        dfs.append(df_i)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# ============================================================
# Cascade accounting
# ============================================================

def cascade_results_to_dataframe_for_accounting(
    cascade_results: list[dict],
    owner_name_map: dict | None = None,
) -> pd.DataFrame:
    """
    Convert cascade results into a compact dataframe for current accounting.

    This is similar to cascade.cascade_results_to_dataframe(), but lives here
    so accounting.py can be used without importing cascade.py.
    """
    rows = []

    for res in cascade_results:
        hit_info = res.get("hit_info", None)

        terminal_owner = terminal_owner_from_result(
            res,
            owner_name_map=owner_name_map,
        )
        terminal_electrode = electrode_from_owner(terminal_owner)

        if hit_info is None:
            owner_id = None
            KE_hit_eV = np.nan
            location = np.array([np.nan, np.nan, np.nan])
        else:
            owner_id = hit_info.get("owner_id", None)
            KE_hit_eV = hit_info.get("KE_hit_eV", np.nan)
            location = hit_info.get("location", None)

            if location is None:
                traj = res.get("traj", None)
                if traj is not None and len(traj) > 0:
                    location = np.asarray(traj)[-1]
                else:
                    location = np.array([np.nan, np.nan, np.nan])

        location = np.asarray(location, dtype=float)

        source_owner = res.get("source_owner", None)
        source_electrode = res.get("source_electrode", None)

        if source_electrode is None:
            source_electrode = electrode_from_owner(source_owner)

        rows.append({
            "primary_index": res.get("primary_index", None),
            "electron_id": res.get("electron_id", None),
            "parent_id": res.get("parent_id", None),
            "generation": res.get("generation", None),

            "source_owner": source_owner,
            "source_electrode": source_electrode,
            "source_Einc_eV": res.get("source_Einc_eV", np.nan),

            "emission_kind": res.get("emission_kind", None),
            "E_emit_eV": res.get("E_emit_eV", np.nan),

            "reason": res.get("reason", None),
            "terminal_owner": terminal_owner,
            "terminal_electrode": terminal_electrode,
            "owner_id": owner_id,
            "KE_hit_eV": KE_hit_eV,
            "steps": res.get("steps", np.nan),

            "x_hit": location[0],
            "y_hit": location[1],
            "z_hit": location[2],
        })

    return pd.DataFrame(rows)


def cascade_current_counts(
    df_cascade: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute source, terminal, and net electron-count contributions.

    Convention here is electron-count bookkeeping:

        source_count:
            +1 when an electron is emitted from an electrode.

        terminal_count:
            -1 when an electron lands on an electrode.

        net_count:
            source_count + terminal_count.

    This is not yet conventional current sign in amperes; it is the
    electron balance per simulated primary.
    """
    channels = [
        "sample",
        "holder",
        "receiver",
        "rod",
        "grid1",
        "grid2",
        "grid3",
        "collector",
        "drifttube",
        "escaped",
        "unknown",
    ]

    if df_cascade.empty:
        return pd.DataFrame({
            "source_count": pd.Series(0, index=channels, dtype=float),
            "terminal_count": pd.Series(0, index=channels, dtype=float),
            "net_count": pd.Series(0, index=channels, dtype=float),
        })

    source = df_cascade["source_electrode"].value_counts()
    terminal = df_cascade["terminal_electrode"].value_counts()

    source = source.reindex(channels, fill_value=0).astype(float)
    terminal = terminal.reindex(channels, fill_value=0).astype(float)

    # Electron leaves source electrode: +1.
    # Electron arrives at terminal electrode: -1.
    out = pd.DataFrame(index=channels)
    out["source_count"] = source
    out["terminal_count"] = -terminal
    out["net_count"] = out["source_count"] + out["terminal_count"]

    return out


def summarize_cascade_accounting(
    cascade_results: list[dict],
    N_primary: int,
    owner_name_map: dict | None = None,
) -> dict:
    """
    Summarize cascade electron balance per electrode.
    """
    df_cascade = cascade_results_to_dataframe_for_accounting(
        cascade_results,
        owner_name_map=owner_name_map,
    )

    counts = cascade_current_counts(df_cascade)

    summary = {
        "N_primary": int(N_primary),
        "N_cascade_electrons": int(len(df_cascade)),
        "N_SE": int((df_cascade["emission_kind"] == "SE").sum()) if not df_cascade.empty else 0,
        "N_BSE": int((df_cascade["emission_kind"] == "BSE").sum()) if not df_cascade.empty else 0,
        "max_generation": int(df_cascade["generation"].max()) if not df_cascade.empty else 0,
        "per_primary_cascade_electrons": len(df_cascade) / N_primary if N_primary > 0 else np.nan,
    }

    for electrode in counts.index:
        summary[f"{electrode}_source_count"] = counts.loc[electrode, "source_count"]
        summary[f"{electrode}_terminal_count"] = counts.loc[electrode, "terminal_count"]
        summary[f"{electrode}_net_count"] = counts.loc[electrode, "net_count"]

        summary[f"{electrode}_net_per_primary"] = (
            counts.loc[electrode, "net_count"] / N_primary
            if N_primary > 0 else np.nan
        )

    return {
        "summary": summary,
        "df_cascade": df_cascade,
        "current_counts": counts,
    }


def print_cascade_accounting_summary(result: dict):
    """
    Pretty-print cascade accounting summary.
    """
    s = result["summary"]

    print("Cascade accounting summary")
    print("--------------------------")
    print(f"N primary:              {s['N_primary']}")
    print(f"N cascade electrons:    {s['N_cascade_electrons']}")
    print(f"N SE:                   {s['N_SE']}")
    print(f"N BSE:                  {s['N_BSE']}")
    print(f"Max generation:         {s['max_generation']}")
    print(f"Per primary electrons:  {s['per_primary_cascade_electrons']:.5f}")

    print("\nElectron-count balance by electrode:")
    print(result["current_counts"])


def add_per_primary_to_current_counts(
    accounting_result: dict,
    N_primary: int | None = None,
) -> dict:
    """
    Add per-primary source/terminal/net columns to cascade current counts.
    """
    if N_primary is None:
        N_primary = accounting_result["summary"]["N_primary"]

    counts = accounting_result["current_counts"].copy()

    counts["source_per_primary"] = counts["source_count"] / N_primary
    counts["terminal_per_primary"] = counts["terminal_count"] / N_primary
    counts["net_per_primary"] = counts["net_count"] / N_primary

    accounting_result["current_counts"] = counts

    return accounting_result
