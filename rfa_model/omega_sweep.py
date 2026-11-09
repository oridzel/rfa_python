"""Production-resolution relaxation-factor benchmark for Taichi RFA SOR.

Every candidate starts from the same host-side potential snapshot.  Only one
candidate field is solved at a time, so the sweep does not retain several
665-cubed fields.  The input field is restored before the function returns.
"""

from __future__ import annotations

import csv
import gc
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .field_validation import laplace_residual_stats
from .taichi_sor import solve_red_black_sor_taichi


DEFAULT_OMEGAS = (1.85, 1.90, 1.94, 1.97, 1.98, 1.985)


def optimal_omega(shape: int | Iterable[int]) -> float:
    """Return the empty-cube estimate ``2 / (1 + sin(pi / N))``.

    This is a comparison guide, not an automatic choice for an RFA domain
    containing internal fixed-potential conductors.
    """
    if np.isscalar(shape):
        n = int(shape)
    else:
        sizes = tuple(int(value) for value in shape)
        if not sizes:
            raise ValueError("shape must not be empty")
        n = max(sizes)
    if n < 3:
        raise ValueError("each benchmark dimension must be at least 3")
    return float(2.0 / (1.0 + math.sin(math.pi / n)))


def _validate_field(field: dict) -> None:
    required = ("V", "fixed", "update_region", "h")
    missing = [name for name in required if name not in field]
    if missing:
        raise KeyError(f"field is missing required keys: {missing}")
    shape = np.asarray(field["V"]).shape
    if len(shape) != 3:
        raise ValueError("field['V'] must be three-dimensional")
    if np.asarray(field["fixed"]).shape != shape:
        raise ValueError("field['fixed'] does not match V.shape")
    if np.asarray(field["update_region"]).shape != shape:
        raise ValueError("field['update_region'] does not match V.shape")


def _history_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "first_checked_delta_V": None,
            "last_checked_delta_V": None,
            "delta_reduction_ratio": None,
            "delta_increase_fraction": None,
            "history_stable": False,
        }

    deltas = np.asarray(
        [float(item["max_delta"]) for item in history],
        dtype=float,
    )
    finite = bool(np.all(np.isfinite(deltas)))
    if deltas.size > 1:
        increases = np.count_nonzero(np.diff(deltas) > 0.0)
        increase_fraction = float(increases / (deltas.size - 1))
    else:
        increase_fraction = 0.0
    first = float(deltas[0])
    last = float(deltas[-1])
    ratio = float(last / first) if first > 0.0 else 0.0

    return {
        "first_checked_delta_V": first,
        "last_checked_delta_V": last,
        "delta_reduction_ratio": ratio,
        "delta_increase_fraction": increase_fraction,
        "history_stable": bool(
            finite
            and last <= first
            and increase_fraction <= 0.5
        ),
    }


def run_omega_sweep(
    field: dict,
    *,
    omegas: Iterable[float] = DEFAULT_OMEGAS,
    iterations: int = 750,
    check_every: int = 25,
    arch: str = "metal",
    precision: str | None = None,
    cpu_max_num_threads: int | None = None,
    residual_chunk_x: int = 8,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark candidate relaxation factors from one identical potential.

    The sweep deliberately disables best-checkpoint restoration and uses an
    effectively unreachable tolerance so each candidate receives the same
    iteration budget.  The caller's potential and solver metadata are restored
    even if a candidate raises ``FloatingPointError``.
    """
    _validate_field(field)
    omega_values = tuple(float(value) for value in omegas)
    if not omega_values:
        raise ValueError("omegas must contain at least one value")
    if len(set(omega_values)) != len(omega_values):
        raise ValueError("omegas must not contain duplicate values")
    if any(not (0.0 < value < 2.0) for value in omega_values):
        raise ValueError("every omega must satisfy 0 < omega < 2")
    if int(iterations) < 1:
        raise ValueError("iterations must be at least 1")

    potential = np.asarray(field["V"])
    initial_potential = np.array(potential, copy=True, order="C")
    original_solver = field.get("solver")
    theory = optimal_omega(potential.shape)
    results: list[dict[str, Any]] = []

    if verbose:
        print("Taichi SOR omega sweep")
        print(f"  shape:             {potential.shape}")
        print(f"  candidates:        {omega_values}")
        print(f"  iterations each:   {int(iterations)}")
        print(f"  check every:       {int(check_every)}")
        print(f"  empty-cube guide:  {theory:.6f}")

    try:
        for omega in omega_values:
            np.copyto(potential, initial_potential)
            row: dict[str, Any] = {
                "omega": float(omega),
                "status": "ok",
                "empty_cube_omega": theory,
                "shape": tuple(int(value) for value in potential.shape),
                "iterations_requested": int(iterations),
            }
            try:
                _, metadata = solve_red_black_sor_taichi(
                    potential,
                    field["fixed"],
                    field["update_region"],
                    max_iter=int(iterations),
                    tol=1e-30,
                    omega=float(omega),
                    check_every=int(check_every),
                    arch=arch,
                    precision=precision,
                    cpu_max_num_threads=cpu_max_num_threads,
                    restore_best_on_max_iter=False,
                    record_history=True,
                    verbose=False,
                )
                residual = laplace_residual_stats(
                    field,
                    chunk_x=int(residual_chunk_x),
                )
                history = list(metadata.get("history", []))
                row.update(
                    {
                        "architecture": metadata["architecture"],
                        "precision": metadata["precision"],
                        "iterations_completed": metadata["iterations"],
                        "runtime_s": metadata["runtime_s"],
                        "solve_s": metadata["solve_s"],
                        "checked_iterations": metadata["checked_iterations"],
                        "fast_iterations": metadata["fast_iterations"],
                        "final_delta_V": metadata["final_iteration_delta"],
                        "best_delta_V": metadata["best_delta"],
                        "best_iteration": metadata["best_iteration"],
                        "residual_max_V": residual[
                            "max_abs_neighbor_mean_residual_V"
                        ],
                        "residual_rms_V": residual[
                            "rms_neighbor_mean_residual_V"
                        ],
                        "history": history,
                    }
                )
                row.update(_history_stats(history))
            except FloatingPointError as exc:
                row.update(
                    {
                        "status": "non_finite",
                        "error": str(exc),
                        "history_stable": False,
                    }
                )
            results.append(row)

            if verbose:
                if row["status"] == "ok":
                    print(
                        f"  omega={omega:.5f}: "
                        f"rms={row['residual_rms_V']:.6e} V, "
                        f"max={row['residual_max_V']:.6e} V, "
                        f"last dV={row['final_delta_V']:.6e} V, "
                        f"time={row['runtime_s']:.1f} s, "
                        f"stable={row['history_stable']}",
                        flush=True,
                    )
                else:
                    print(
                        f"  omega={omega:.5f}: REJECTED ({row['status']})",
                        flush=True,
                    )
            gc.collect()
    finally:
        np.copyto(potential, initial_potential)
        if original_solver is None:
            field.pop("solver", None)
        else:
            field["solver"] = original_solver

    successful = [row for row in results if row["status"] == "ok"]
    successful.sort(key=lambda row: float(row["residual_rms_V"]))
    for rank, row in enumerate(successful, start=1):
        row["residual_rank"] = int(rank)
    return results


def recommended_omega(results: Iterable[dict[str, Any]]) -> float:
    """Return the lowest-RMS stable candidate, or lowest-RMS successful one."""
    rows = [row for row in results if row.get("status") == "ok"]
    if not rows:
        raise ValueError("omega sweep contains no successful candidates")
    stable = [row for row in rows if bool(row.get("history_stable", False))]
    candidates = stable or rows
    best = min(candidates, key=lambda row: float(row["residual_rms_V"]))
    return float(best["omega"])


def save_omega_sweep_csv(
    results: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Save scalar benchmark results; checkpoint histories remain in memory."""
    rows = list(results)
    if not rows:
        raise ValueError("results must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"history", "shape"}
            and not isinstance(value, (dict, list, tuple))
        }
    )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
    return output


def plot_omega_sweep(results: Iterable[dict[str, Any]]):
    """Plot residual versus omega and checkpoint convergence histories."""
    import matplotlib.pyplot as plt

    rows = [row for row in results if row.get("status") == "ok"]
    if not rows:
        raise ValueError("omega sweep contains no successful candidates")
    rows.sort(key=lambda row: float(row["omega"]))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    omegas = np.asarray([row["omega"] for row in rows], dtype=float)
    rms = np.asarray([row["residual_rms_V"] for row in rows], dtype=float)
    maximum = np.asarray([row["residual_max_V"] for row in rows], dtype=float)
    axes[0].semilogy(omegas, rms, "o-", label="RMS residual")
    axes[0].semilogy(omegas, maximum, "s--", label="Maximum residual")
    axes[0].axvline(
        float(rows[0]["empty_cube_omega"]),
        color="0.5",
        linestyle=":",
        label="Empty-cube guide",
    )
    axes[0].set_xlabel(r"Relaxation factor $\omega$")
    axes[0].set_ylabel("Mean-neighbor residual (V)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    for row in rows:
        history = row.get("history", [])
        axes[1].semilogy(
            [item["iteration"] for item in history],
            [item["max_delta"] for item in history],
            label=f"{float(row['omega']):g}",
        )
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Maximum checked update (V)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(title=r"$\omega$", ncol=2)
    return figure, axes

