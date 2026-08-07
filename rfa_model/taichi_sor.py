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
    """Identical to _red_black_step_f64 but without the residual reduction.

    The atomic_max fires once per updated voxel and costs ~60% of the sweep
    on CPU (worse on Metal, where every thread contends for one global
    address).  The residual is only consumed on check iterations, so every
    other iteration runs this variant instead.
    """
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
    if array.dtype == np.bool_ and array.flags.c_contiguous:
        return array.view(np.uint8)
    result = np.ascontiguousarray(array, dtype=np.uint8)
    if result.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional array.")
    return result


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


def _has_non_finite(array: np.ndarray, chunk_size: int = 8_000_000) -> bool:
    """Scan for NaN/Inf without allocating a full-grid temporary."""
    flat = np.ravel(array)
    for start in range(0, flat.size, chunk_size):
        chunk = flat[start:start + chunk_size]
        if not np.isfinite(chunk).all():
            return True
    return False


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
    verbose: bool,
    t0: float,
) -> tuple[np.ndarray, int, float]:
    dtype = np.float64 if precision == "f64" else np.float32
    if V.dtype != dtype or not V.flags.c_contiguous:
        V_work = np.ascontiguousarray(V, dtype=dtype)
    else:
        V_work = V

    max_delta = np.zeros(1, dtype=dtype)
    reset = _reset_max_f64 if precision == "f64" else _reset_max_f32
    step = _red_black_step_f64 if precision == "f64" else _red_black_step_f32
    step_fast = (
        _red_black_step_fast_f64 if precision == "f64" else _red_black_step_fast_f32
    )

    last_delta = np.inf
    for it in range(1, int(max_iter) + 1):
        tracking = it == 1 or it % int(check_every) == 0 or it == int(max_iter)

        if tracking:
            reset(max_delta)
            step(V_work, fixed_u8, update_u8, 0, omega, max_delta)
            step(V_work, fixed_u8, update_u8, 1, omega, max_delta)
        else:
            step_fast(V_work, fixed_u8, update_u8, 0, omega)
            step_fast(V_work, fixed_u8, update_u8, 1, omega)

        if tracking:
            last_delta = float(max_delta[0])
            if verbose:
                print(
                    f"iter {it:7d}: max update = {last_delta:.6e} V "
                    f"elapsed = {time.perf_counter() - t0:.1f} s",
                    flush=True,
                )
            if last_delta < tol:
                break

    if V_work is not V:
        V[...] = V_work.astype(V.dtype, copy=False)
    return V, it, last_delta


def _solve_metal_device_arrays(
    V: np.ndarray,
    fixed_u8: np.ndarray,
    update_u8: np.ndarray,
    max_iter: int,
    tol: float,
    omega: float,
    check_every: int,
    verbose: bool,
    t0: float,
) -> tuple[np.ndarray, int, float]:
    shape = tuple(int(n) for n in V.shape)
    V_device = ti.ndarray(dtype=ti.f32, shape=shape)
    fixed_device = ti.ndarray(dtype=ti.u8, shape=shape)
    update_device = ti.ndarray(dtype=ti.u8, shape=shape)
    max_delta = ti.ndarray(dtype=ti.f32, shape=(1,))

    V_device.from_numpy(np.ascontiguousarray(V, dtype=np.float32))
    fixed_device.from_numpy(fixed_u8)
    update_device.from_numpy(update_u8)

    last_delta = np.inf
    for it in range(1, int(max_iter) + 1):
        tracking = it == 1 or it % int(check_every) == 0 or it == int(max_iter)

        if tracking:
            _reset_max_f32(max_delta)
            _red_black_step_f32(
                V_device, fixed_device, update_device, 0, omega, max_delta
            )
            _red_black_step_f32(
                V_device, fixed_device, update_device, 1, omega, max_delta
            )
        else:
            _red_black_step_fast_f32(
                V_device, fixed_device, update_device, 0, omega
            )
            _red_black_step_fast_f32(
                V_device, fixed_device, update_device, 1, omega
            )

        if tracking:
            last_delta = float(max_delta.to_numpy()[0])
            if verbose:
                print(
                    f"iter {it:7d}: max update = {last_delta:.6e} V "
                    f"elapsed = {time.perf_counter() - t0:.1f} s",
                    flush=True,
                )
            if last_delta < tol:
                break

    ti.sync()
    V[...] = V_device.to_numpy().astype(V.dtype, copy=False)
    return V, it, last_delta


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
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a masked 3-D Laplace problem with Taichi red-black SOR.

    Parameters are deliberately NumPy-based so the backend stays independent
    of the RFA field dictionary and can be regression-tested on small domains.
    CPU defaults to f64; Metal is constrained to f32.
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
        V_array, iterations, last_delta = _solve_metal_device_arrays(
            V_array,
            fixed_u8,
            update_u8,
            max_iter,
            tol,
            omega,
            check_every,
            verbose,
            t0,
        )
    else:
        V_array, iterations, last_delta = _solve_cpu_external_arrays(
            V_array,
            fixed_u8,
            update_u8,
            precision,
            max_iter,
            tol,
            omega,
            check_every,
            verbose,
            t0,
        )
    solve_s = time.perf_counter() - solve_start

    # A NaN inside the relaxed region is invisible to the residual: abs(NaN)
    # never wins an atomic_max, so the solver would otherwise report a clean
    # convergence while handing back a corrupted potential.
    if _has_non_finite(V_array):
        raise FloatingPointError(
            "Taichi SOR produced a non-finite potential after "
            f"{int(iterations)} iterations (reported max update "
            f"{float(last_delta):.3e} V). The residual reduction cannot see "
            "NaN/Inf, so this would otherwise be reported as converged. "
            "Check the initial potential and the fixed-voltage values."
        )

    metadata: dict[str, Any] = {
        "method": "taichi_red_black_sor",
        "architecture": actual_arch,
        "precision": precision,
        "iterations": int(iterations),
        "tol": float(tol),
        "last_delta": float(last_delta),
        "omega": float(omega),
        "check_every": int(check_every),
        "setup_s": float(setup_s),
        "solve_s": float(solve_s),
        "runtime_s": float(time.perf_counter() - t0),
        "converged": bool(last_delta < tol),
        "taichi_version": str(getattr(ti, "__version__", "unknown")),
    }
    if cpu_max_num_threads is not None and actual_arch == "cpu":
        metadata["cpu_max_num_threads"] = int(cpu_max_num_threads)

    if verbose:
        print("Taichi Laplace solver finished", flush=True)
        print(metadata, flush=True)

    return V_array, metadata
