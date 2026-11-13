"""Experimental CUDA f64 extension for the RFA Taichi red-black SOR solver.

This module does NOT modify the installed rfa_model package.  It dispatches
only arch='cuda' calls through rfa_model.taichi_sor to a resident-device solver,
reusing rfa_model.taichi_sor's existing numerical kernels. CPU and Metal use
the installed code unchanged.

REQUIREMENTS: RFA package exposing _red_black_step_f64,
_red_black_step_fast_f64, _reset_max_f64; Taichi CUDA environment.

IMPORTANT: Not yet validated on the user's A100. Run smoke_test_cuda()
BEFORE the 831^3 production job. Compare fields with a known-good field.
"""
# IMPORTANT: no postponed annotations here. Taichi 1.7.3
# inspects @ti.kernel annotations as concrete Taichi types at import time.

import time
from typing import Any

import numpy as np
import taichi as ti

_INSTALLED_ORIGINAL = None


@ti.kernel
def _copy_3d_f64(
    src: ti.types.ndarray(dtype=ti.f64, ndim=3),
    dst: ti.types.ndarray(dtype=ti.f64, ndim=3),
):
    ti.loop_config(block_dim=256)
    for i, j, k in ti.ndrange(src.shape[0], src.shape[1], src.shape[2]):
        dst[i, j, k] = src[i, j, k]


def _cuda_initialized() -> bool:
    try:
        return ti.lang.impl.get_runtime().prog is not None
    except Exception:
        return False


def _ensure_cuda_f64():
    if _cuda_initialized():
        cfg = ti.lang.impl.current_cfg()
        arch = str(cfg.arch).lower()
        if "cuda" not in arch:
            raise RuntimeError(
                f"Taichi was already initialized with {arch!r}, not CUDA. "
                "Restart the Jupyter kernel and initialize with ti.cuda."
            )
        if bool(getattr(cfg, 'fast_math', False)):
            raise RuntimeError(
                "Existing Taichi runtime has fast_math=True. Restart the kernel "
                "and initialize with ti.init(arch=ti.cuda, "
                "default_fp=ti.f64, fast_math=False)."
            )
        return

    # Asking for CUDA explicitly avoids silent CPU fallback.
    ti.init(arch=ti.cuda, default_fp=ti.f64, fast_math=False)
    arch = str(ti.lang.impl.current_cfg().arch).lower()
    if "cuda" not in arch:
        raise RuntimeError(f"Requested CUDA but Taichi selected {arch!r}.")


def _validate(V, fixed, update_region, max_iter, tol, omega, check_every):
    if V.ndim != 3 or min(V.shape) < 3 or V.dtype not in (np.float32, np.float64):
        raise ValueError("V must be a 3D float32/float64 array with each extent >=3")
    if fixed.shape != V.shape or update_region.shape != V.shape:
        raise ValueError("V, fixed, and update_region must have matching shapes")
    if int(max_iter) < 1 or int(check_every) < 1:
        raise ValueError("max_iter and check_every must be positive")
    if float(tol) <= 0 or not (0 < float(omega) < 2):
        raise ValueError("Expected tol > 0 and 0 < omega < 2")
    if not np.isfinite(V).all():
        raise ValueError("Initial potential contains nonfinite values")


def solve_red_black_sor_cuda(
    V, fixed, update_region, *, max_iter=20_000, tol=1e-5,
    omega=1.85, check_every=25, arch="cuda", precision="f64",
    cpu_max_num_threads=None, restore_best_on_max_iter=True,
    record_history=False, verbose=True, **kwargs,
):
    """Perform f64 red-black SOR on CUDA; keep all 3D arrays device-resident.

    This uses the package's original step/step-fast kernels so the mathematical
    update rule remains the same as the project's CPU implementation.
    """
    if str(arch).lower() != "cuda":
        raise ValueError("This temporary solver handles only arch='cuda'")
    if str(precision).lower() != "f64":
        raise ValueError("CUDA patch currently supports f64 only (as requested)")
    if kwargs:
        raise TypeError(f"Unsupported CUDA solver arguments: {sorted(kwargs)}")

    import rfa_model.taichi_sor as ts
    required = ("_red_black_step_f64", "_red_black_step_fast_f64", "_reset_max_f64")
    missing = [key for key in required if not hasattr(ts, key)]
    if missing:
        raise RuntimeError(
            "Installed taichi_sor.py differs from the known RFA source; "
            f"missing kernels: {missing}. Please supply the installed taichi_sor.py."
        )

    V_input = np.asarray(V)
    f = np.asarray(fixed)
    u = np.asarray(update_region)
    _validate(V_input, f, u, max_iter, tol, omega, check_every)
    _ensure_cuda_f64()

    t0 = time.perf_counter()
    shape = tuple(map(int, V_input.shape))
    # Explicit on-device allocations. No 3D NumPy copies during the iteration loop.
    dv = ti.ndarray(dtype=ti.f64, shape=shape)
    df = ti.ndarray(dtype=ti.u8, shape=shape)
    du = ti.ndarray(dtype=ti.u8, shape=shape)
    dmax = ti.ndarray(dtype=ti.f64, shape=(1,))
    best = ti.ndarray(dtype=ti.f64, shape=shape) if restore_best_on_max_iter else None

    dv.from_numpy(np.ascontiguousarray(V_input, dtype=np.float64))
    df.from_numpy(np.ascontiguousarray(f, dtype=np.uint8))
    du.from_numpy(np.ascontiguousarray(u, dtype=np.uint8))

    n = V_input.size
    allocated_gib = n * (8 + 1 + 1 + (8 if best is not None else 0)) / 2**30
    if verbose:
        print("RFA Taichi CUDA red-black SOR (experimental patch)", flush=True)
        print(f"  grid: {shape}; precision: f64; omega: {omega}", flush=True)
        print(f"  minimum resident array memory: {allocated_gib:.2f} GiB", flush=True)
        print(f"  maximum iterations: {max_iter}; tolerance: {tol}", flush=True)

    best_delta = float('inf')
    last_delta = float('inf')
    best_it = None
    history = []
    converged = False

    for it in range(1, int(max_iter) + 1):
        track = (it == 1 or it % int(check_every) == 0 or it == int(max_iter))
        if track:
            ts._reset_max_f64(dmax)
            ts._red_black_step_f64(dv, df, du, 0, float(omega), dmax)
            ts._red_black_step_f64(dv, df, du, 1, float(omega), dmax)
            last_delta = float(dmax.to_numpy()[0])  # one scalar transfer/check
            if not np.isfinite(last_delta):
                raise FloatingPointError("Nonfinite maximum update on CUDA")
            if last_delta < best_delta:
                best_delta = last_delta
                best_it = it
                if best is not None:
                    _copy_3d_f64(dv, best)
            if record_history:
                history.append({"iteration": it, "max_delta": last_delta,
                                "elapsed_s": time.perf_counter() - t0})
            if verbose:
                print(f"iter {it:6d}: max update = {last_delta:.6e} V; "
                      f"elapsed = {time.perf_counter()-t0:.1f} s", flush=True)
            if last_delta < float(tol):
                converged = True
                break
        else:
            ts._red_black_step_fast_f64(dv, df, du, 0, float(omega))
            ts._red_black_step_fast_f64(dv, df, du, 1, float(omega))

    restored_best = bool(not converged and best is not None and best_it != it)
    if restored_best:
        _copy_3d_f64(best, dv)
    ti.sync()

    # Copy to RAM only once, at the end; caller receives its familiar NumPy V.
    out = dv.to_numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("CUDA solver produced NaN or Inf")
    if V_input.flags.writeable:
        V_input[...] = out.astype(V_input.dtype, copy=False)
        out = V_input
    metadata: dict[str, Any] = {
        "method": "taichi_red_black_sor",
        "architecture": "cuda",
        "precision": "f64",
        "iterations": int(it),
        "tol": float(tol),
        "last_delta": float(last_delta),
        "best_delta": float(best_delta),
        "best_iteration": best_it,
        "omega": float(omega),
        "check_every": int(check_every),
        "converged": bool(converged),
        "restored_best_on_max_iter": restored_best,
        "runtime_s": time.perf_counter() - t0,
        "taichi_version": str(ti.__version__),
        "patched_cuda": True,
    }
    if record_history:
        metadata["history"] = history
    if verbose:
        print("RFA CUDA solver complete:", metadata, flush=True)
    return out, metadata


def enable_cuda_sor():
    """Enable CUDA dispatch at the module actually defining the RFA solver.

    fields.solve_laplace_sor_taichi imports the routine from
    rfa_model.taichi_sor internally in the installed package. Patching an
    attribute on fields fails because no such module-level attribute exists.
    All non-CUDA requests are delegated to the untouched original routine.
    """
    import rfa_model.fields as fields
    import rfa_model.taichi_sor as ts

    required = ("_red_black_step_f64", "_red_black_step_fast_f64", "_reset_max_f64")
    missing = [key for key in required if not hasattr(ts, key)]
    if missing:
        raise RuntimeError(f"Unsupported installed rfa_model.taichi_sor; missing {missing}")

    global _INSTALLED_ORIGINAL
    current = ts.solve_red_black_sor_taichi
    if getattr(current, "_rfa_cuda_patch", False):
        print("CUDA solver patch already enabled")
        return

    _INSTALLED_ORIGINAL = current

    def _dispatch(*args, **kwargs):
        if str(kwargs.get("arch", "cpu")).lower() == "cuda":
            return solve_red_black_sor_cuda(*args, **kwargs)
        return _INSTALLED_ORIGINAL(*args, **kwargs)

    _dispatch._rfa_cuda_patch = True
    ts.solve_red_black_sor_taichi = _dispatch

    # Support package revisions that imported the function at module scope.
    # Do not create the attribute when absent, as in the user's installation.
    if getattr(fields, "solve_red_black_sor_taichi", None) is current:
        fields.solve_red_black_sor_taichi = _dispatch

    assert getattr(ts.solve_red_black_sor_taichi, "_rfa_cuda_patch", False)
    print("Enabled experimental CUDA f64 dispatch in rfa_model.taichi_sor")


def smoke_test_cuda():
    """Check CUDA boundary handling, f64 arithmetic, and convergence on 17^3."""
    _ensure_cuda_f64()
    n = 17
    x = np.linspace(0.0, 1.0, n)
    exact = 50.0 * np.broadcast_to(x[:, None, None], (n, n, n)).copy()
    V = np.zeros((n, n, n), dtype=np.float64)
    fixed = np.zeros((n, n, n), dtype=bool)
    fixed[[0, -1], :, :] = True
    fixed[:, [0, -1], :] = True
    fixed[:, :, [0, -1]] = True
    V[fixed] = exact[fixed]
    update = np.ones_like(fixed)
    result, meta = solve_red_black_sor_cuda(
        V, fixed, update, max_iter=800, tol=1e-8, omega=1.75,
        check_every=10, restore_best_on_max_iter=True, verbose=False,
    )
    max_error = float(np.max(np.abs(result - exact)))
    boundary_error = float(np.max(np.abs(result[fixed] - exact[fixed])))
    print(f"CUDA test: max error = {max_error:.3e} V; "
          f"fixed boundary error = {boundary_error:.3e} V")
    print(f"  iterations={meta['iterations']}; converged={meta['converged']}; "
          f"backend={meta['architecture']}")
    if not meta['converged'] or max_error > 1e-4 or boundary_error > 1e-12:
        raise AssertionError("CUDA solver smoke test failed. Do not run production field.")
    return meta
