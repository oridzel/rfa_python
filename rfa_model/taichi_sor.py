"""Optional Taichi red-black SOR backend for the RFA field solver.

The CPU path updates the existing NumPy arrays directly and therefore avoids
duplicating the multi-gigabyte potential array.  The Metal path keeps Taichi
ndarrays resident on the device for all iterations and transfers the data only
before and after the solve.

Taichi is intentionally imported only when this module is requested by
``fields.solve_laplace_sor_taichi``.  The rest of ``rfa_model`` therefore keeps
working when Taichi is not installed.
"""

import platform
import time
from typing import Any

import numpy as np

try:
    import taichi as ti
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "The Taichi SOR backend requires the optional 'taichi' package. "
        "Install rfa_model in a Python version supported by Taichi and run "
        "`python -m pip install taichi`."
    ) from exc


_SUPPORTED_ARCHES = {"cpu", "metal"}
_SUPPORTED_PRECISIONS = {"f32", "f64"}


def _runtime_is_initialized() -> bool:
    """Return whether the process already owns a live Taichi runtime."""
    try:
        return ti.lang.impl.get_runtime().prog is not None
    except Exception:
        return False


def _arch_name(arch: Any) -> str:
    """Normalize Taichi's enum/string representation to ``cpu`` or ``metal``."""
    text = str(arch).lower()
    if "metal" in text:
        return "metal"
    if "cpu" in text or "x64" in text or "arm64" in text:
        return "cpu"
    return text.split(".")[-1]


def _current_arch_name() -> str:
    try:
        return _arch_name(ti.lang.impl.current_cfg().arch)
    except Exception:
        return "unknown"


def _initialize_taichi(
    arch: str,
    precision: str,
    cpu_max_num_threads: int | None,
) -> str:
    """Initialize Taichi once and reject incompatible existing runtimes."""
    arch = arch.lower()
    precision = precision.lower()

    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported Taichi architecture {arch!r}; choose 'cpu' or 'metal'."
        )
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported Taichi precision {precision!r}; choose 'f32' or 'f64'."
        )
    if arch == "metal" and precision != "f32":
        raise ValueError(
            "Taichi's Metal backend does not support f64. Use "
            "taichi_precision='f32' with Metal, or taichi_arch='cpu' for f64."
        )
    if arch == "metal" and platform.system() != "Darwin":
        raise ValueError("The Taichi Metal backend is available only on macOS.")

    if _runtime_is_initialized():
        cfg = ti.lang.impl.current_cfg()
        current_arch = _current_arch_name()
        if current_arch != arch:
            raise RuntimeError(
                "Taichi is already initialized with architecture "
                f"{current_arch!r}, but the RFA solver requested {arch!r}. "
                "Restart the Python/Jupyter kernel before changing "
                "taichi_arch."
            )
        if bool(getattr(cfg, "fast_math", False)):
            raise RuntimeError(
                "Taichi is already initialized with fast_math=True. Restart "
                "the Python/Jupyter kernel and let the RFA solver initialize "
                "Taichi with fast_math=False."
            )
        requested_default_fp = ti.f64 if precision == "f64" else ti.f32
        current_default_fp = getattr(cfg, "default_fp", None)
        if (
            current_default_fp is not None
            and current_default_fp != requested_default_fp
        ):
            raise RuntimeError(
                "Taichi is already initialized with default_fp="
                f"{current_default_fp}, but the RFA solver requested "
                f"{precision}. Restart the Python/Jupyter kernel before "
                "changing taichi_precision. The current kernels use explicit "
                "types, but rejecting the mismatch prevents future untyped "
                "literals from silently using the wrong precision."
            )
        if arch == "cpu" and cpu_max_num_threads is not None:
            actual_threads = int(getattr(cfg, "cpu_max_num_threads", 0))
            if actual_threads != int(cpu_max_num_threads):
                raise RuntimeError(
                    "Taichi is already initialized with "
                    f"cpu_max_num_threads={actual_threads}, but the RFA "
                    f"solver requested {int(cpu_max_num_threads)}. Restart "
                    "the Python/Jupyter kernel before changing the thread "
                    "count."
                )
        return current_arch

    init_kwargs: dict[str, Any] = {
        "arch": ti.cpu if arch == "cpu" else ti.metal,
        "default_fp": ti.f64 if precision == "f64" else ti.f32,
        "fast_math": False,
    }
    if arch == "cpu" and cpu_max_num_threads is not None:
        threads = int(cpu_max_num_threads)
        if threads < 1:
            raise ValueError("taichi_cpu_threads must be at least 1.")
        init_kwargs["cpu_max_num_threads"] = threads

    ti.init(**init_kwargs)
    return _current_arch_name()


@ti.kernel
def _reset_max_f64(value: ti.types.ndarray(dtype=ti.f64, ndim=1)):
    value[0] = 0.0


@ti.kernel
def _reset_max_f32(value: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    value[0] = 0.0


@ti.kernel
def _copy_3d_f32(
    source: ti.types.ndarray(dtype=ti.f32, ndim=3),
    destination: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    """Copy one device-resident f32 potential array to another."""
    ti.loop_config(block_dim=256)
    for i, j, k in ti.ndrange(
        source.shape[0],
        source.shape[1],
        source.shape[2],
    ):
        destination[i, j, k] = source[i, j, k]


@ti.kernel
def _red_black_step_f64(
    V: ti.types.ndarray(dtype=ti.f64, ndim=3),
    fixed: ti.types.ndarray(dtype=ti.u8, ndim=3),
    update_region: ti.types.ndarray(dtype=ti.u8, ndim=3),
    parity: ti.i32,
    omega: ti.f64,
    max_delta: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    # Cells of one checkerboard color have only opposite-color neighbours, so
    # every iteration of this loop is independent and safe to parallelize.
    ti.loop_config(block_dim=256)
    for i, j, k in ti.ndrange(
        (1, V.shape[0] - 1),
        (1, V.shape[1] - 1),
        (1, V.shape[2] - 1),
    ):
        if (
            ((i + j + k) & 1) == parity
            and update_region[i, j, k] != 0
            and fixed[i, j, k] == 0
        ):
            old = V[i, j, k]
            average = (
                V[i - 1, j, k]
                + V[i + 1, j, k]
                + V[i, j - 1, k]
                + V[i, j + 1, k]
                + V[i, j, k - 1]
                + V[i, j, k + 1]
            ) / 6.0
            delta = omega * (average - old)
            V[i, j, k] = old + delta
            ti.atomic_max(max_delta[0], ti.abs(delta))


@ti.kernel
def _red_black_step_f32(
    V: ti.types.ndarray(dtype=ti.f32, ndim=3),
    fixed: ti.types.ndarray(dtype=ti.u8, ndim=3),
    update_region: ti.types.ndarray(dtype=ti.u8, ndim=3),
    parity: ti.i32,
    omega: ti.f32,
    max_delta: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    ti.loop_config(block_dim=256)
    for i, j, k in ti.ndrange(
        (1, V.shape[0] - 1),
        (1, V.shape[1] - 1),
        (1, V.shape[2] - 1),
    ):
        if (
            ((i + j + k) & 1) == parity
            and update_region[i, j, k] != 0
            and fixed[i, j, k] == 0
        ):
            old = V[i, j, k]
            average = (
                V[i - 1, j, k]
                + V[i + 1, j, k]
                + V[i, j - 1, k]
                + V[i, j + 1, k]
                + V[i, j, k - 1]
                + V[i, j, k + 1]
            ) / 6.0
            delta = omega * (average - old)
            V[i, j, k] = old + delta
            ti.atomic_max(max_delta[0], ti.abs(delta))


@ti.kernel
def _red_black_step_fast_f64(
    V: ti.types.ndarray(dtype=ti.f64, ndim=3),
    fixed: ti.types.ndarray(dtype=ti.u8, ndim=3),
    update_region: ti.types.ndarray(dtype=ti.u8, ndim=3),
    parity: ti.i32,
    omega: ti.f64,
):
    """Perform one checkerboard update without a global reduction."""
    ti.loop_config(block_dim=256)
    for i, j, k in ti.ndrange(
        (1, V.shape[0] - 1),
        (1, V.shape[1] - 1),
        (1, V.shape[2] - 1),
    ):
        if (
            ((i + j + k) & 1) == parity
            and update_region[i, j, k] != 0
            and fixed[i, j, k] == 0
        ):
            old = V[i, j, k]
            average = (
                V[i - 1, j, k]
                + V[i + 1, j, k]
                + V[i, j - 1, k]
                + V[i, j + 1, k]
                + V[i, j, k - 1]
                + V[i, j, k + 1]
            ) / 6.0
            V[i, j, k] = old + omega * (average - old)


@ti.kernel
def _red_black_step_fast_f32(
    V: ti.types.ndarray(dtype=ti.f32, ndim=3),
    fixed: ti.types.ndarray(dtype=ti.u8, ndim=3),
    update_region: ti.types.ndarray(dtype=ti.u8, ndim=3),
    parity: ti.i32,
    omega: ti.f32,
):
    """Perform one checkerboard update without a global reduction."""
    ti.loop_config(block_dim=256)
    for i, j, k in ti.ndrange(
        (1, V.shape[0] - 1),
        (1, V.shape[1] - 1),
        (1, V.shape[2] - 1),
    ):
        if (
            ((i + j + k) & 1) == parity
            and update_region[i, j, k] != 0
            and fixed[i, j, k] == 0
        ):
            old = V[i, j, k]
            average = (
                V[i - 1, j, k]
                + V[i + 1, j, k]
                + V[i, j - 1, k]
                + V[i, j + 1, k]
                + V[i, j, k - 1]
                + V[i, j, k + 1]
            ) / 6.0
            V[i, j, k] = old + omega * (average - old)


def _u8_mask_view(mask: np.ndarray, name: str) -> np.ndarray:
    """Return a C-contiguous u8 representation, avoiding a copy when possible."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional array.")
    if array.dtype == np.bool_ and array.flags.c_contiguous:
        return array.view(np.uint8)
    result = np.ascontiguousarray(array, dtype=np.uint8)
    return result


def _has_non_finite(
    array: np.ndarray,
    *,
    chunk_size: int = 8_000_000,
) -> bool:
    """Scan a large array for NaN/Inf without a full-size temporary."""
    flat = np.ravel(np.asarray(array))
    for start in range(0, flat.size, int(chunk_size)):
        stop = min(start + int(chunk_size), flat.size)
        if not bool(np.all(np.isfinite(flat[start:stop]))):
            return True
    return False


def _raise_if_non_finite(array: np.ndarray, stage: str) -> None:
    if _has_non_finite(array):
        raise FloatingPointError(
            f"Non-finite potential detected {stage}. The field was not "
            "accepted because NaN/Inf values are invisible to the "
            "atomic maximum-update reduction."
        )


def _count_active(update_region: np.ndarray, fixed: np.ndarray) -> int:
    """Count updateable cells with a bounded-size temporary buffer."""
    update_flat = np.ravel(update_region)
    fixed_flat = np.ravel(fixed)
    chunk_size = 8_000_000
    total = 0
    for start in range(0, update_flat.size, chunk_size):
        stop = min(start + chunk_size, update_flat.size)
        active_chunk = np.logical_not(fixed_flat[start:stop])
        np.logical_and(update_flat[start:stop], active_chunk, out=active_chunk)
        total += int(np.count_nonzero(active_chunk))
    return total


def _validate_inputs(
    V: np.ndarray,
    fixed: np.ndarray,
    update_region: np.ndarray,
    max_iter: int,
    tol: float,
    omega: float,
    check_every: int,
) -> None:
    if V.ndim != 3 or any(n < 3 for n in V.shape):
        raise ValueError("field['V'] must be a 3-D array with each dimension >= 3.")
    if V.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("field['V'] must have dtype float32 or float64.")
    if fixed.shape != V.shape or update_region.shape != V.shape:
        raise ValueError("V, fixed, and update_region must have identical shapes.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1.")
    if float(tol) <= 0.0:
        raise ValueError("tol must be positive.")
    if not (0.0 < float(omega) < 2.0):
        raise ValueError("Red-black SOR requires 0 < omega < 2.")
    if int(check_every) < 1:
        raise ValueError("taichi_check_every must be at least 1.")


def _solve_cpu_external_arrays(
    V: np.ndarray,
    fixed_u8: np.ndarray,
    update_u8: np.ndarray,
    precision: str,
    max_iter: int,
    tol: float,
    omega: float,
    check_every: int,
    restore_best_on_max_iter: bool,
    record_history: bool,
    verbose: bool,
    t0: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    dtype = np.float64 if precision == "f64" else np.float32
    if V.dtype != dtype or not V.flags.c_contiguous:
        V_work = np.ascontiguousarray(V, dtype=dtype)
    else:
        V_work = V

    max_delta = np.zeros(1, dtype=dtype)
    reset = _reset_max_f64 if precision == "f64" else _reset_max_f32
    step = _red_black_step_f64 if precision == "f64" else _red_black_step_f32
    fast_step = (
        _red_black_step_fast_f64
        if precision == "f64"
        else _red_black_step_fast_f32
    )

    last_delta = np.inf
    best_delta = np.inf
    best_iteration = 0
    best_V = np.empty_like(V_work) if restore_best_on_max_iter else None
    checked_iterations = 0
    history: list[dict[str, float | int]] = []
    for it in range(1, int(max_iter) + 1):
        checked = (
            it == 1
            or it % int(check_every) == 0
            or it == int(max_iter)
        )
        if checked:
            reset(max_delta)
            step(V_work, fixed_u8, update_u8, 0, omega, max_delta)
            step(V_work, fixed_u8, update_u8, 1, omega, max_delta)
            last_delta = float(max_delta[0])
            checked_iterations += 1
            if record_history:
                history.append({"iteration": int(it), "max_delta": last_delta})
            if last_delta < best_delta:
                best_delta = last_delta
                best_iteration = it
                if best_V is not None:
                    np.copyto(best_V, V_work)
            if verbose:
                print(
                    f"iter {it:7d}: max update = {last_delta:.6e} V "
                    f"elapsed = {time.perf_counter() - t0:.1f} s",
                    flush=True,
                )
            if last_delta < tol:
                break
        else:
            fast_step(V_work, fixed_u8, update_u8, 0, omega)
            fast_step(V_work, fixed_u8, update_u8, 1, omega)

    final_delta = last_delta
    selected_delta = final_delta
    selected_iteration = it
    restored_best = False
    if (
        final_delta >= tol
        and best_V is not None
        and best_iteration != it
    ):
        np.copyto(V_work, best_V)
        selected_delta = best_delta
        selected_iteration = best_iteration
        restored_best = True

    if V_work is not V:
        V[...] = V_work.astype(V.dtype, copy=False)
    return V, {
        "iterations": int(it),
        "final_delta": float(final_delta),
        "best_delta": float(best_delta),
        "best_iteration": int(best_iteration),
        "selected_delta": float(selected_delta),
        "selected_iteration": int(selected_iteration),
        "restored_best": bool(restored_best),
        "checked_iterations": int(checked_iterations),
        "fast_iterations": int(it - checked_iterations),
        "history": history,
    }


def _solve_metal_device_arrays(
    V: np.ndarray,
    fixed_u8: np.ndarray,
    update_u8: np.ndarray,
    max_iter: int,
    tol: float,
    omega: float,
    check_every: int,
    restore_best_on_max_iter: bool,
    record_history: bool,
    verbose: bool,
    t0: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = tuple(int(n) for n in V.shape)
    V_device = ti.ndarray(dtype=ti.f32, shape=shape)
    fixed_device = ti.ndarray(dtype=ti.u8, shape=shape)
    update_device = ti.ndarray(dtype=ti.u8, shape=shape)
    max_delta = ti.ndarray(dtype=ti.f32, shape=(1,))
    best_device = (
        ti.ndarray(dtype=ti.f32, shape=shape)
        if restore_best_on_max_iter
        else None
    )

    V_device.from_numpy(np.ascontiguousarray(V, dtype=np.float32))
    fixed_device.from_numpy(fixed_u8)
    update_device.from_numpy(update_u8)

    last_delta = np.inf
    best_delta = np.inf
    best_iteration = 0
    checked_iterations = 0
    history: list[dict[str, float | int]] = []
    for it in range(1, int(max_iter) + 1):
        checked = (
            it == 1
            or it % int(check_every) == 0
            or it == int(max_iter)
        )
        if checked:
            _reset_max_f32(max_delta)
            _red_black_step_f32(
                V_device, fixed_device, update_device, 0, omega, max_delta
            )
            _red_black_step_f32(
                V_device, fixed_device, update_device, 1, omega, max_delta
            )
            ti.sync()
            last_delta = float(max_delta.to_numpy()[0])
            checked_iterations += 1
            if record_history:
                history.append({"iteration": int(it), "max_delta": last_delta})
            if last_delta < best_delta:
                best_delta = last_delta
                best_iteration = it
                if best_device is not None:
                    _copy_3d_f32(V_device, best_device)
            if verbose:
                print(
                    f"iter {it:7d}: max update = {last_delta:.6e} V "
                    f"elapsed = {time.perf_counter() - t0:.1f} s",
                    flush=True,
                )
            if last_delta < tol:
                break
        else:
            _red_black_step_fast_f32(
                V_device, fixed_device, update_device, 0, omega
            )
            _red_black_step_fast_f32(
                V_device, fixed_device, update_device, 1, omega
            )

    final_delta = last_delta
    selected_delta = final_delta
    selected_iteration = it
    restored_best = False
    selected_device = V_device
    if (
        final_delta >= tol
        and best_device is not None
        and best_iteration != it
    ):
        selected_device = best_device
        selected_delta = best_delta
        selected_iteration = best_iteration
        restored_best = True

    ti.sync()
    V[...] = selected_device.to_numpy().astype(V.dtype, copy=False)
    return V, {
        "iterations": int(it),
        "final_delta": float(final_delta),
        "best_delta": float(best_delta),
        "best_iteration": int(best_iteration),
        "selected_delta": float(selected_delta),
        "selected_iteration": int(selected_iteration),
        "restored_best": bool(restored_best),
        "checked_iterations": int(checked_iterations),
        "fast_iterations": int(it - checked_iterations),
        "history": history,
    }


def solve_red_black_sor_taichi(
    V: np.ndarray,
    fixed: np.ndarray,
    update_region: np.ndarray,
    *,
    max_iter: int = 20_000,
    tol: float = 1e-5,
    omega: float = 1.85,
    check_every: int = 25,
    arch: str = "cpu",
    precision: str | None = None,
    cpu_max_num_threads: int | None = None,
    restore_best_on_max_iter: bool = True,
    record_history: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a masked 3-D Laplace problem with Taichi red-black SOR.

    Parameters are deliberately NumPy-based so the backend stays independent
    of the RFA field dictionary and can be regression-tested on small domains.
    CPU defaults to f64; Metal is constrained to f32.  When
    ``restore_best_on_max_iter`` is true and the requested tolerance is not
    reached, the returned potential is the checked iterate with the smallest
    maximum SOR update.  Keeping that checkpoint costs one additional full
    potential array (f64 on CPU or f32 on Metal).
    """
    arch = str(arch).lower()
    if precision is None:
        precision = "f32" if arch == "metal" else "f64"
    precision = str(precision).lower()

    V_array = np.asarray(V)
    if not V_array.flags.writeable:
        V_array = V_array.copy()
    fixed_array = np.asarray(fixed)
    update_array = np.asarray(update_region)
    _validate_inputs(
        V_array,
        fixed_array,
        update_array,
        max_iter,
        tol,
        omega,
        check_every,
    )

    finite_scan_start = time.perf_counter()
    _raise_if_non_finite(V_array, "before the Taichi solve")
    pre_solve_finite_scan_s = time.perf_counter() - finite_scan_start

    fixed_u8 = _u8_mask_view(fixed_array, "fixed")
    update_u8 = _u8_mask_view(update_array, "update_region")

    t0 = time.perf_counter()
    actual_arch = _initialize_taichi(arch, precision, cpu_max_num_threads)
    setup_s = time.perf_counter() - t0

    if verbose:
        active_count = _count_active(update_array, fixed_array)
        print("Solving Laplace equation with Taichi red-black SOR", flush=True)
        print(f"  grid shape:    {V_array.shape}", flush=True)
        print(f"  update voxels: {active_count:,}", flush=True)
        print(f"  fixed voxels:  {int(np.count_nonzero(fixed_array)):,}", flush=True)
        print(f"  architecture:  {actual_arch}", flush=True)
        print(f"  precision:     {precision}", flush=True)
        print(f"  omega:         {float(omega):g}", flush=True)
        print(f"  tolerance:     {float(tol):.3e} V", flush=True)

    solve_start = time.perf_counter()
    if actual_arch == "metal":
        V_array, outcome = _solve_metal_device_arrays(
            V_array,
            fixed_u8,
            update_u8,
            max_iter,
            tol,
            omega,
            check_every,
            restore_best_on_max_iter,
            record_history,
            verbose,
            t0,
        )
    else:
        V_array, outcome = _solve_cpu_external_arrays(
            V_array,
            fixed_u8,
            update_u8,
            precision,
            max_iter,
            tol,
            omega,
            check_every,
            restore_best_on_max_iter,
            record_history,
            verbose,
            t0,
        )
    solve_s = time.perf_counter() - solve_start

    finite_scan_start = time.perf_counter()
    _raise_if_non_finite(V_array, "after the Taichi solve")
    post_solve_finite_scan_s = time.perf_counter() - finite_scan_start

    metadata: dict[str, Any] = {
        "method": "taichi_red_black_sor",
        "architecture": actual_arch,
        "precision": precision,
        "iterations": int(outcome["iterations"]),
        "tol": float(tol),
        # Compatibility: last_delta describes the potential actually returned.
        "last_delta": float(outcome["selected_delta"]),
        "final_iteration_delta": float(outcome["final_delta"]),
        "best_delta": float(outcome["best_delta"]),
        "best_iteration": int(outcome["best_iteration"]),
        "selected_iteration": int(outcome["selected_iteration"]),
        "restored_best": bool(outcome["restored_best"]),
        "restore_best_on_max_iter": bool(restore_best_on_max_iter),
        "checked_iterations": int(outcome["checked_iterations"]),
        "fast_iterations": int(outcome["fast_iterations"]),
        "omega": float(omega),
        "check_every": int(check_every),
        "setup_s": float(setup_s),
        "solve_s": float(solve_s),
        "pre_solve_finite_scan_s": float(pre_solve_finite_scan_s),
        "post_solve_finite_scan_s": float(post_solve_finite_scan_s),
        "runtime_s": float(time.perf_counter() - t0),
        "converged": bool(outcome["best_delta"] < tol),
        "termination_reason": (
            "tolerance"
            if outcome["best_delta"] < tol
            else "max_iter"
        ),
        "taichi_version": str(getattr(ti, "__version__", "unknown")),
    }
    if record_history:
        metadata["history"] = list(outcome["history"])
    if cpu_max_num_threads is not None and actual_arch == "cpu":
        metadata["cpu_max_num_threads"] = int(cpu_max_num_threads)

    if verbose:
        print("Taichi Laplace solver finished", flush=True)
        print(metadata, flush=True)

    return V_array, metadata
