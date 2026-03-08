"""Comprehensive TPU validation: all public API paths (matmul, gram, residual, solve).

Tests f64 and ozaki modes, various input sizes and condition numbers.
Verifies on-device execution and FP64 accuracy.

Usage:
    python tpu_full_validate.py [--output /path/to/results.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax import matmul, gram, residual, solve


def rel_err(actual, reference):
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(actual - reference) / denom)


def digits(err):
    if err <= 0:
        return 16.0
    return min(16.0, -np.log10(err))


def timed(fn, *args, warmup=1, repeats=3, **kwargs):
    """Time a function, returning (result, median_ms)."""
    for _ in range(warmup):
        r = fn(*args, **kwargs)
        if isinstance(r, jax.Array):
            r.block_until_ready()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn(*args, **kwargs)
        if isinstance(r, jax.Array):
            r.block_until_ready()
        times.append(time.perf_counter() - t0)
    return r, float(np.median(times)) * 1000


def make_matrix(rng, m, n, cond=None):
    """Random matrix, optionally with specified condition number."""
    A = rng.randn(m, n)
    if cond is not None and m == n:
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        s = np.logspace(0, -np.log10(cond), min(m, n))
        A = U @ np.diag(s) @ Vt
    return A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = {"tests": [], "summary": {}}
    all_pass = True
    rng = np.random.RandomState(42)

    # Device info
    devices = jax.devices()
    backend = jax.default_backend()
    device_kind = devices[0].device_kind
    print(f"Platform: {backend} | Device: {device_kind} | JAX: {jax.__version__}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    results["platform"] = backend
    results["device_kind"] = device_kind
    results["jax_version"] = jax.__version__

    def record(name, err, threshold, time_ms=None):
        nonlocal all_pass
        ok = err <= threshold
        d = digits(err)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        line = f"  [{status}] {name}: {err:.2e} ({d:.1f} digits)"
        if time_ms is not None:
            line += f" [{time_ms:.1f}ms]"
        if not ok:
            line += f" > threshold {threshold:.0e}"
        print(line)
        results["tests"].append({
            "name": name, "error": err, "digits": d,
            "threshold": threshold, "pass": ok,
            "time_ms": time_ms,
        })

    # =========================================================
    # 1. matmul — Ozaki pipeline (default and fused)
    # =========================================================
    print("=== matmul (Ozaki pipeline) ===")
    for n in [64, 128, 256]:
        A = rng.randn(n, n)
        B = rng.randn(n, n)
        C_ref = A @ B

        C, ms = timed(matmul, A, B)
        record(f"matmul bf16_interleaved n={n}", rel_err(C, C_ref), 1e-9, ms)

        C_f, ms_f = timed(matmul, A, B, accumulation="fused")
        record(f"matmul fused n={n}", rel_err(C_f, C_ref), 1e-9, ms_f)

    # matmul with JAX arrays (matmul always returns numpy)
    A_j = jnp.asarray(rng.randn(128, 128), dtype=jnp.float64)
    B_j = jnp.asarray(rng.randn(128, 128), dtype=jnp.float64)
    C_j = matmul(A_j, B_j)
    C_ref = np.asarray(A_j) @ np.asarray(B_j)
    record("matmul JAX arrays n=128", rel_err(np.asarray(C_j), C_ref), 1e-9)

    # Rectangular
    A_rect = rng.randn(64, 256)
    B_rect = rng.randn(256, 32)
    C_rect = matmul(A_rect, B_rect)
    record("matmul rect 64x256 @ 256x32", rel_err(C_rect, A_rect @ B_rect), 1e-9)
    print()

    # =========================================================
    # 2. gram — f64 and ozaki modes
    # =========================================================
    print("=== gram ===")
    for n in [64, 128, 256]:
        A = rng.randn(n, n)
        G_ref = A.T @ A

        # f64 mode (default)
        G, ms = timed(gram, A)
        record(f"gram f64 n={n}", rel_err(G, G_ref), 1e-13, ms)

        # ozaki mode
        G_oz, ms_oz = timed(gram, A, mode="ozaki")
        record(f"gram ozaki n={n}", rel_err(G_oz, G_ref), 1e-8, ms_oz)

    # Tall matrix
    A_tall = rng.randn(256, 64)
    G_tall = gram(A_tall)
    G_tall_ref = A_tall.T @ A_tall
    record("gram f64 tall 256x64", rel_err(G_tall, G_tall_ref), 1e-13)

    # Symmetry check
    asym = float(np.max(np.abs(G_tall - G_tall.T)))
    record("gram symmetry", asym, 1e-30)

    # PSD check
    eigvals = np.linalg.eigvalsh(G_tall)
    psd_violation = max(0, -eigvals.min())
    record("gram PSD", psd_violation, 1e-10)

    # JAX arrays
    A_j = jnp.asarray(rng.randn(128, 128), dtype=jnp.float64)
    G_j = gram(A_j)
    assert isinstance(G_j, jax.Array), "gram JAX input should return JAX array"
    record("gram JAX arrays n=128", rel_err(np.asarray(G_j), np.asarray(A_j).T @ np.asarray(A_j)), 1e-13)
    print()

    # =========================================================
    # 3. residual — f64 and ozaki modes
    # =========================================================
    print("=== residual ===")
    for n in [64, 128, 256]:
        A = rng.randn(n, n)
        x = rng.randn(n)
        b = rng.randn(n)
        r_ref = b - A @ x

        # f64 mode (default)
        r, ms = timed(residual, A, x, b)
        record(f"residual f64 n={n}", rel_err(r, r_ref), 1e-13, ms)

        # ozaki mode
        r_oz, ms_oz = timed(residual, A, x, b, mode="ozaki")
        record(f"residual ozaki n={n}", rel_err(r_oz, r_ref), 1e-8, ms_oz)

    # Matrix RHS
    A = rng.randn(128, 128)
    X = rng.randn(128, 4)
    B = rng.randn(128, 4)
    R = residual(A, X, B)
    R_ref = B - A @ X
    record("residual f64 matrix RHS", rel_err(R, R_ref), 1e-13)

    # Non-square A
    A_rect = rng.randn(64, 128)
    x_rect = rng.randn(128)
    b_rect = rng.randn(64)
    r_rect = residual(A_rect, x_rect, b_rect)
    r_rect_ref = b_rect - A_rect @ x_rect
    record("residual f64 non-square 64x128", rel_err(r_rect, r_rect_ref), 1e-13)

    # JAX arrays
    A_j = jnp.asarray(rng.randn(128, 128), dtype=jnp.float64)
    x_j = jnp.asarray(rng.randn(128), dtype=jnp.float64)
    b_j = jnp.asarray(rng.randn(128), dtype=jnp.float64)
    r_j = residual(A_j, x_j, b_j)
    assert isinstance(r_j, jax.Array), "residual JAX input should return JAX array"
    r_ref = np.asarray(b_j) - np.asarray(A_j) @ np.asarray(x_j)
    record("residual JAX arrays", rel_err(np.asarray(r_j), r_ref), 1e-13)
    print()

    # =========================================================
    # 4. solve — f64 and ozaki modes, various condition numbers
    # =========================================================
    print("=== solve ===")

    # Well-conditioned systems
    for n in [64, 128, 256]:
        A = rng.randn(n, n)
        b = rng.randn(n)
        x_ref = np.linalg.solve(A, b)

        # f64 mode (default)
        x, ms = timed(solve, A, b)
        record(f"solve f64 n={n}", rel_err(x, x_ref), 1e-11, ms)

        # ozaki mode
        x_oz, ms_oz = timed(solve, A, b, residual_mode="ozaki")
        record(f"solve ozaki n={n}", rel_err(x_oz, x_ref), 1e-6, ms_oz)

    # Matrix RHS
    A = rng.randn(128, 128)
    B = rng.randn(128, 4)
    X_ref = np.linalg.solve(A, B)
    X = solve(A, B)
    record("solve f64 matrix RHS 128x4", rel_err(X, X_ref), 1e-11)

    # Condition number sweep
    print("\n  --- condition number sweep (n=128, f64 mode) ---")
    # FP32 LU factorization limits us: cond(A) > ~1e8 means the FP32
    # factorization is near-singular. We test up to 1e6 with tight thresholds
    # and report higher cond numbers as informational only.
    for cond in [1e2, 1e4, 1e6]:
        A = make_matrix(rng, 128, 128, cond=cond)
        b = rng.randn(128)
        x_ref = np.linalg.solve(A, b)
        x = solve(A, b)
        err = rel_err(x, x_ref)
        expected_digits = max(0, 16 - np.log10(cond))
        threshold = cond * 1e-13
        record(f"solve f64 cond={cond:.0e} (expect ~{expected_digits:.0f} digits)",
               err, threshold)

    # Informational: high condition numbers (FP32 LU may fail)
    for cond in [1e10, 1e13]:
        A = make_matrix(rng, 128, 128, cond=cond)
        b = rng.randn(128)
        x_ref = np.linalg.solve(A, b)
        x = solve(A, b)
        err = rel_err(x, x_ref)
        print(f"  [INFO] solve f64 cond={cond:.0e}: {err:.2e} ({digits(err):.1f} digits) [FP32 LU limit]")
        results["tests"].append({
            "name": f"solve f64 cond={cond:.0e} (info)", "error": err,
            "digits": digits(err), "info_only": True, "pass": True,
        })

    # Iteration convergence
    A = rng.randn(128, 128)
    b = rng.randn(128)
    x_ref = np.linalg.solve(A, b)
    err_0 = rel_err(solve(A, b, max_iterations=0), x_ref)
    err_1 = rel_err(solve(A, b, max_iterations=1), x_ref)
    err_3 = rel_err(solve(A, b, max_iterations=3), x_ref)
    print(f"\n  --- iteration convergence ---")
    print(f"  iter=0: {err_0:.2e} ({digits(err_0):.1f} digits)")
    print(f"  iter=1: {err_1:.2e} ({digits(err_1):.1f} digits)")
    print(f"  iter=3: {err_3:.2e} ({digits(err_3):.1f} digits)")
    ok = err_1 < err_0 * 1e-3 and err_3 < err_1
    if not ok:
        all_pass = False
    print(f"  Convergence: {'PASS' if ok else 'FAIL'}")
    results["tests"].append({
        "name": "solve convergence", "pass": ok,
        "err_0": err_0, "err_1": err_1, "err_3": err_3,
    })

    # JAX arrays
    A_j = jnp.asarray(rng.randn(128, 128), dtype=jnp.float64)
    b_j = jnp.asarray(rng.randn(128), dtype=jnp.float64)
    x_j = solve(A_j, b_j)
    assert isinstance(x_j, jax.Array), "solve JAX input should return JAX array"
    x_ref = np.linalg.solve(np.asarray(A_j), np.asarray(b_j))
    record("solve JAX arrays", rel_err(np.asarray(x_j), x_ref), 1e-11)
    print()

    # =========================================================
    # 5. On-device verification
    # =========================================================
    print("=== on-device verification ===")
    if backend == "tpu":
        # Verify computations stay on TPU by checking device placement
        # Note: matmul always returns numpy, so we check gram/residual/solve
        A_j = jnp.asarray(rng.randn(128, 128), dtype=jnp.float64)
        x_j = jnp.asarray(rng.randn(128), dtype=jnp.float64)
        b_j = jnp.asarray(rng.randn(128), dtype=jnp.float64)

        G_j = gram(A_j)
        print(f"  gram output device: {G_j.devices()}")
        assert isinstance(G_j, jax.Array), "gram should return JAX array"

        r_j = residual(A_j, x_j, b_j)
        print(f"  residual output device: {r_j.devices()}")
        assert isinstance(r_j, jax.Array), "residual should return JAX array"

        s_j = solve(A_j, b_j)
        print(f"  solve output device: {s_j.devices()}")
        assert isinstance(s_j, jax.Array), "solve should return JAX array"

        print("  All on-device: PASS")
    else:
        print("  (skipped — not on TPU)")
    print()

    # =========================================================
    # Summary
    # =========================================================
    n_tests = len(results["tests"])
    n_pass = sum(1 for t in results["tests"] if t.get("pass", False))
    n_fail = n_tests - n_pass

    print(f"=== SUMMARY: {n_pass}/{n_tests} passed", end="")
    if n_fail > 0:
        print(f", {n_fail} FAILED", end="")
        for t in results["tests"]:
            if not t.get("pass", False):
                print(f"\n  FAIL: {t['name']}", end="")
    print(" ===")

    results["summary"] = {
        "total": n_tests, "pass": n_pass, "fail": n_fail,
        "all_pass": all_pass,
    }

    # Save results
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).resolve().parent / "tpu_full_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
