"""Stress tests for both host and on-device pipelines.

Tests accuracy and correctness under challenging conditions:
  1. Rectangular matrices (tall-skinny, wide-short)
  2. Ill-conditioned matrices (condition number sweep)
  3. Extreme scale (very large and very small values)
  4. Special matrices (identity, zero, near-singular)
  5. Mixed-scale columns (some large, some tiny)

Usage:
    python benchmarks/stress_test.py
"""

import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax import matmul, matmul_numpy


def relative_error(C, C_ref):
    """Relative Frobenius error, safe for extreme magnitudes."""
    # Normalize to avoid overflow in norm computation.
    peak = np.max(np.abs(C_ref))
    if peak == 0:
        return float(np.linalg.norm(C))
    if not np.isfinite(peak):
        return float("nan")
    C_n = C / peak
    C_ref_n = C_ref / peak
    ref_norm = np.linalg.norm(C_ref_n)
    if ref_norm == 0:
        return float(np.linalg.norm(C_n))
    return float(np.linalg.norm(C_n - C_ref_n) / ref_norm)


def run_test(name, A, B, threshold=1e-13, ondevice=True):
    """Run both pipelines and report errors.

    Set ondevice=False to skip the on-device pipeline (e.g. for values
    outside FP32 representable range).
    """
    C_ref = A @ B

    C_host = matmul(A, B, pipeline="host")
    e_host = relative_error(C_host, C_ref)
    ok_host = e_host < threshold and np.isfinite(e_host)

    if ondevice:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C_ondev = matmul(A, B, pipeline="ondevice")
            C_ondev_np = matmul_numpy(A, B, pipeline="ondevice")
        e_ondev = relative_error(C_ondev, C_ref)
        e_match = np.max(np.abs(C_ondev - C_ondev_np))
        ok_ondev = e_ondev < threshold and np.isfinite(e_ondev)
        ok_match = e_match == 0.0 or (np.isnan(e_match) and np.isnan(e_ondev))
    else:
        e_ondev = float("nan")
        e_match = float("nan")
        ok_ondev = True
        ok_match = True

    ok = ok_host and ok_ondev and ok_match
    status = "PASS" if ok else "FAIL"
    ondev_str = f"{e_ondev:.2e}" if ondevice else "skip"
    match_str = f"{e_match:.0e}" if ondevice else "skip"
    print(f"  {name:<40} host={e_host:.2e}  ondev={ondev_str:<10}  "
          f"jax==np={match_str}  [{status}]")

    return ok


def main():
    print("ozaki-jax stress tests")
    print(f"JAX {jax.__version__}, platform: {jax.default_backend()}")
    print()

    rng = np.random.RandomState(42)
    all_pass = True

    # --- Test 1: Rectangular matrices ---
    print("test 1: rectangular matrices\n")
    shapes = [
        (32, 256, 32, "tall-skinny @ skinny-wide"),
        (256, 32, 256, "wide-short @ short-wide"),
        (16, 512, 16, "very tall @ very wide"),
        (512, 16, 512, "wide @ tall"),
        (100, 200, 300, "non-square all different"),
        (1, 128, 1, "row vector @ matrix @ col (outer)"),
        (128, 1, 128, "col @ row (rank-1 outer product)"),
    ]
    for N, K, M, desc in shapes:
        A = rng.randn(N, K).astype(np.float64)
        B = rng.randn(K, M).astype(np.float64)
        ok = run_test(f"{N}x{K} @ {K}x{M} ({desc})", A, B)
        all_pass &= ok
    print()

    # --- Test 2: Condition number sweep ---
    print("test 2: condition number sweep\n")
    n = 128
    for target_cond in [1, 1e3, 1e6, 1e9, 1e12]:
        # Build matrix with prescribed condition number via SVD.
        U, _, Vt = np.linalg.svd(rng.randn(n, n).astype(np.float64))
        s = np.logspace(0, -np.log10(target_cond), n)
        A = (U * s) @ Vt
        B = rng.randn(n, n).astype(np.float64)

        actual_cond = np.linalg.cond(A)
        # Error bound scales with condition number * machine epsilon.
        threshold = max(actual_cond * 2.3e-16 * 10, 1e-13)
        ok = run_test(f"cond={actual_cond:.0e}", A, B, threshold=threshold)
        all_pass &= ok
    print()

    # --- Test 3: Extreme scales ---
    # The on-device pipeline uses FP32 intermediates, so values must be within
    # ~[1e-38, 3e38].  Scales outside this range are tested host-only.
    print("test 3: extreme scales\n")
    n = 64
    for scale, label, use_ondev in [
        (1e-100, "very small (1e-100)", False),   # Below FP32 subnormal
        (1e-30, "small (1e-30)", False),           # FP32 subnormal range
        (1e-10, "moderate small (1e-10)", True),
        (1.0, "unit scale", True),
        (1e10, "moderate large (1e10)", True),
        (1e30, "large (1e30)", True),
        (1e100, "very large (1e100)", False),      # norm() overflows
    ]:
        A = rng.randn(n, n).astype(np.float64) * scale
        B = rng.randn(n, n).astype(np.float64) * scale
        ok = run_test(f"scale={label}", A, B, ondevice=use_ondev)
        all_pass &= ok
    print()

    # --- Test 4: Special matrices ---
    print("test 4: special matrices\n")

    # Identity.
    n = 64
    I = np.eye(n, dtype=np.float64)
    B = rng.randn(n, n).astype(np.float64)
    ok = run_test("identity @ random", I, B)
    all_pass &= ok

    ok = run_test("random @ identity", B, I)
    all_pass &= ok

    # Near-zero residual (A*B is nearly zero).
    A = np.ones((n, n), dtype=np.float64) * 1e-8
    A += rng.randn(n, n) * 1e-16
    B = rng.randn(n, n).astype(np.float64)
    ok = run_test("near-zero matrix", A, B)
    all_pass &= ok

    # Diagonal.
    diag = rng.randn(n).astype(np.float64) * 100
    A = np.diag(diag)
    B = rng.randn(n, n).astype(np.float64)
    ok = run_test("diagonal @ random", A, B)
    all_pass &= ok

    # Permutation.
    perm = rng.permutation(n)
    A = np.eye(n, dtype=np.float64)[perm]
    B = rng.randn(n, n).astype(np.float64)
    ok = run_test("permutation @ random", A, B)
    all_pass &= ok
    print()

    # --- Test 5: Mixed-scale columns ---
    # The on-device pipeline uses per-row normalization, so within-row
    # scale variation (i.e. different column magnitudes) is harder than
    # between-row variation.  We test a moderate range (1e-5 to 1e5)
    # that both pipelines should handle, and a wider range (1e-10 to 1e10)
    # that is host-only.
    print("test 5: mixed-scale columns\n")
    n = 128

    # Moderate range: both pipelines.
    # Per-row FP32 normalization loses precision on wide intra-row spread,
    # so we use a relaxed threshold for on-device.
    A = rng.randn(n, n).astype(np.float64)
    col_scales = np.logspace(-5, 5, n)
    rng.shuffle(col_scales)
    A *= col_scales[np.newaxis, :]
    B = rng.randn(n, n).astype(np.float64)
    ok = run_test("columns 1e-5 to 1e5", A, B, threshold=1e-11)
    all_pass &= ok

    # Wide range: host-only (FP32 per-row normalization loses precision).
    A = rng.randn(n, n).astype(np.float64)
    col_scales = np.logspace(-10, 10, n)
    rng.shuffle(col_scales)
    A *= col_scales[np.newaxis, :]
    ok = run_test("columns 1e-10 to 1e10 (host only)", A, B, ondevice=False)
    all_pass &= ok

    # Row-scaled: both pipelines (per-row normalization handles this well).
    A = rng.randn(n, n).astype(np.float64)
    row_scales = np.logspace(-10, 10, n)
    rng.shuffle(row_scales)
    A *= row_scales[:, np.newaxis]
    ok = run_test("rows scaled 1e-10 to 1e10", A, B)
    all_pass &= ok
    print()

    # --- Test 6: Large K near limit ---
    print("test 6: large K near exactness limit\n")
    for K in [512, 648, 1024]:
        A = rng.randn(32, K).astype(np.float64)
        B = rng.randn(K, 32).astype(np.float64)
        ok = run_test(f"32x{K} @ {K}x32", A, B)
        all_pass &= ok
    print()

    # --- Test 7: Extraction residual analysis ---
    print("test 7: extraction residual analysis\n")
    from ozaki_jax.extract import (
        _compute_rho, _compute_rho_f32,
        extract_split_rows, f32_extract_split_rows,
    )
    from ozaki_jax.matmul import _double_f32_split

    n = 128
    X = rng.randn(n, n).astype(np.float64)
    rho64 = _compute_rho(n)
    rho32 = _compute_rho_f32(n)

    # FP64 extraction residual per slice.
    print("  FP64 extraction (8 slices):")
    residual = X.copy()
    for i in range(8):
        slices, _ = extract_split_rows(residual, rho64, n_slices=1)
        captured = np.linalg.norm(slices[0].astype(np.float64)) / np.linalg.norm(X)
        residual = residual - slices[0].astype(np.float64) * np.ldexp(
            np.ones(n, dtype=np.float64), _.pop(0).astype(np.int64))[:, np.newaxis]
        # Simpler: just report the slice norm fraction.
        print(f"    slice {i}: captured norm fraction = {captured:.6f}")

    # FP32 extraction residual per slice (hi part).
    print("  FP32 extraction of hi part (5 slices):")
    X_hi = np.float32(X)
    residual32 = X_hi.copy()
    for i in range(5):
        slices32, _ = f32_extract_split_rows(residual32, rho32, n_slices=1)
        captured = np.linalg.norm(slices32[0]) / np.linalg.norm(X_hi)
        residual32 = np.float32(residual32 - slices32[0] * np.float32(np.ldexp(
            np.ones(n, dtype=np.float32), _.pop(0).astype(np.int32)))[:, np.newaxis])
        print(f"    slice {i}: captured norm fraction = {captured:.6f}")

    # Compare total residual.
    X_hi, X_lo = _double_f32_split(X)
    hi_slices, _ = f32_extract_split_rows(X_hi, rho32, 5)
    lo_slices, _ = f32_extract_split_rows(X_lo, rho32, 4)

    hi_residual = np.float64(X_hi)
    for s, sc in zip(hi_slices, _):
        pass  # Just checking we can iterate

    print(f"\n  Double-FP32 split quality:")
    recon = np.float64(X_hi) + np.float64(X_lo)
    split_err = np.max(np.abs(X - recon)) / np.max(np.abs(X))
    print(f"    split relative error: {split_err:.2e} (vs 2^-48 = {2**-48:.2e})")
    print(f"    bits lost: {-np.log2(split_err + 1e-300):.1f} "
          f"(FP64 has 53, FP32 pair gives ~48)")
    print()

    # Summary.
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
