"""Small-domain validation for the optional RFA Taichi SOR backend."""

import argparse

import numpy as np

from rfa_model.taichi_sor import solve_red_black_sor_taichi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--threads", type=int, default=18)
    args = parser.parse_args()

    precision = "f32" if args.arch == "metal" else "f64"
    tolerance = 2e-6 if precision == "f32" else 1e-10
    error_limit = 5e-6 if precision == "f32" else 2e-10

    n = 24
    x = np.linspace(0.0, 1.0, n)
    exact = np.broadcast_to(x[:, None, None], (n, n, n)).copy()

    V = np.zeros_like(exact)
    fixed = np.zeros(V.shape, dtype=bool)
    fixed[[0, -1], :, :] = True
    fixed[:, [0, -1], :] = True
    fixed[:, :, [0, -1]] = True
    V[fixed] = exact[fixed]
    fixed_before = V[fixed].copy()

    V, metadata = solve_red_black_sor_taichi(
        V,
        fixed,
        np.ones_like(fixed),
        max_iter=4_000,
        tol=tolerance,
        omega=1.85,
        check_every=10,
        arch=args.arch,
        precision=precision,
        cpu_max_num_threads=args.threads if args.arch == "cpu" else None,
        verbose=True,
    )

    max_error = float(np.max(np.abs(V - exact)))
    fixed_change = float(np.max(np.abs(V[fixed] - fixed_before)))

    print(f"maximum analytical error: {max_error:.6e} V")
    print(f"maximum fixed-voxel change: {fixed_change:.6e} V")

    if not metadata["converged"]:
        raise SystemExit("FAIL: solver did not converge")
    if max_error > error_limit:
        raise SystemExit(
            f"FAIL: analytical error {max_error:.3e} exceeds {error_limit:.3e}"
        )
    if fixed_change != 0.0:
        raise SystemExit("FAIL: fixed-potential voxels changed")

    # The reduction-free kernels must produce exactly the same potential as
    # measuring the maximum update on every iteration.
    fast_shape = (17, 16, 15)
    fast_fixed = np.zeros(fast_shape, dtype=bool)
    fast_fixed[[0, -1], :, :] = True
    fast_fixed[:, [0, -1], :] = True
    fast_fixed[:, :, [0, -1]] = True
    fast_initial = np.zeros(fast_shape, dtype=V.dtype)
    fast_initial[-1, :, :] = 1.0

    every_V, every_metadata = solve_red_black_sor_taichi(
        fast_initial.copy(),
        fast_fixed,
        np.ones_like(fast_fixed),
        max_iter=31,
        tol=1e-30,
        omega=1.80,
        check_every=1,
        arch=args.arch,
        precision=precision,
        cpu_max_num_threads=args.threads if args.arch == "cpu" else None,
        restore_best_on_max_iter=False,
        verbose=False,
    )
    sparse_V, sparse_metadata = solve_red_black_sor_taichi(
        fast_initial.copy(),
        fast_fixed,
        np.ones_like(fast_fixed),
        max_iter=31,
        tol=1e-30,
        omega=1.80,
        check_every=7,
        arch=args.arch,
        precision=precision,
        cpu_max_num_threads=args.threads if args.arch == "cpu" else None,
        restore_best_on_max_iter=False,
        verbose=False,
    )
    if not np.array_equal(every_V, sparse_V):
        difference = float(np.max(np.abs(every_V - sparse_V)))
        raise SystemExit(
            "FAIL: reduction-free iterations changed the potential; "
            f"maximum difference={difference:.3e} V"
        )
    if every_metadata["fast_iterations"] != 0:
        raise SystemExit("FAIL: check_every=1 unexpectedly used fast iterations")
    if sparse_metadata["fast_iterations"] != 25:
        raise SystemExit("FAIL: check_every=7 fast-iteration count is incorrect")

    # A non-finite input must never be mistaken for clean convergence.
    bad_V = fast_initial.copy()
    bad_V[fast_shape[0] // 2, fast_shape[1] // 2, fast_shape[2] // 2] = np.nan
    try:
        solve_red_black_sor_taichi(
            bad_V,
            fast_fixed,
            np.ones_like(fast_fixed),
            max_iter=10,
            tol=tolerance,
            omega=1.80,
            check_every=5,
            arch=args.arch,
            precision=precision,
            cpu_max_num_threads=args.threads if args.arch == "cpu" else None,
            verbose=False,
        )
    except FloatingPointError:
        pass
    else:
        raise SystemExit("FAIL: NaN potential was not rejected")

    # Exercise max-iteration checkpoint restoration with a deliberately slow,
    # mildly oscillatory small problem.
    checkpoint_shape = (11, 10, 9)
    checkpoint_fixed = np.zeros(checkpoint_shape, dtype=bool)
    checkpoint_fixed[[0, -1], :, :] = True
    checkpoint_fixed[:, [0, -1], :] = True
    checkpoint_fixed[:, :, [0, -1]] = True
    rng = np.random.default_rng(0)
    checkpoint_V = rng.normal(size=checkpoint_shape).astype(V.dtype)
    checkpoint_V[checkpoint_fixed] = 0.0

    _, checkpoint_metadata = solve_red_black_sor_taichi(
        checkpoint_V,
        checkpoint_fixed,
        np.ones_like(checkpoint_fixed),
        max_iter=8,
        tol=1e-30,
        omega=1.99,
        check_every=1,
        arch=args.arch,
        precision=precision,
        cpu_max_num_threads=args.threads if args.arch == "cpu" else None,
        restore_best_on_max_iter=True,
        verbose=False,
    )
    if not checkpoint_metadata["restored_best"]:
        raise SystemExit("FAIL: earlier best checkpoint was not restored")
    if checkpoint_metadata["last_delta"] != checkpoint_metadata["best_delta"]:
        raise SystemExit("FAIL: returned field metadata does not match best checkpoint")
    if not (
        checkpoint_metadata["best_delta"]
        < checkpoint_metadata["final_iteration_delta"]
    ):
        raise SystemExit("FAIL: checkpoint regression did not exercise rollback")

    print(
        "PASS: Taichi convergence, boundaries, finite guard, fast kernels, "
        "and best-checkpoint restoration validated."
    )


if __name__ == "__main__":
    main()
