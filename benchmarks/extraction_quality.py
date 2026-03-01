"""Analyze extraction quality: FP32 vs FP64, bits per slice, residual decay.

Measures:
  1. How many bits each slice captures (FP64 vs FP32 extraction)
  2. Residual energy decay curve
  3. BF16-exactness verification across matrix sizes
  4. Reconstruction error: sum-of-slices vs original

Usage:
    python benchmarks/extraction_quality.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax.extract import (
    _compute_rho, _compute_rho_f32,
    extract_split_rows, extract_split_cols,
    f32_extract_split_rows, f32_extract_split_cols,
)
from ozaki_jax.matmul import _double_f32_split


def bits_captured(slice_vals, original):
    """Estimate bits of the original captured by this slice."""
    s_norm = np.linalg.norm(slice_vals.astype(np.float64))
    o_norm = np.linalg.norm(original.astype(np.float64))
    if o_norm == 0 or s_norm == 0:
        return 0.0
    ratio = s_norm / o_norm
    if ratio >= 1.0:
        return 53.0  # Entire signal
    return -np.log2(1.0 - ratio + 1e-300)


def main():
    print("ozaki-jax extraction quality analysis")
    print(f"JAX {jax.__version__}, platform: {jax.default_backend()}")
    print()

    rng = np.random.RandomState(42)

    # === Test 1: Residual decay curves ===
    print("test 1: residual decay per extraction round\n")
    n = 256
    X = rng.randn(n, n).astype(np.float64)
    X_f32_hi, X_f32_lo = _double_f32_split(X)

    rho64 = _compute_rho(n)
    rho32 = _compute_rho_f32(n)

    print(f"  rho_f64={rho64}, bits/slice={53 - rho64}")
    print(f"  rho_f32={rho32}, bits/slice={24 - rho32}")
    print()

    # FP64 extraction.
    print("  FP64 extraction residual (fraction of original norm):")
    residual_f64 = X.copy()
    x_norm = np.linalg.norm(X)
    for i in range(8):
        slices, scales = extract_split_rows(residual_f64, rho64, n_slices=1)
        v_unscaled = slices[0].astype(np.float64)
        row_scale = np.ldexp(np.ones(n, dtype=np.float64),
                             scales[0].astype(np.int64))
        v_original = v_unscaled * row_scale[:, np.newaxis]
        residual_f64 = residual_f64 - v_original
        r_frac = np.linalg.norm(residual_f64) / x_norm
        print(f"    after slice {i}: residual = {r_frac:.2e}")
    print()

    # FP32 extraction of hi part.
    print("  FP32 extraction of hi part (residual fraction of hi norm):")
    residual_f32 = X_f32_hi.copy()
    hi_norm = np.linalg.norm(X_f32_hi.astype(np.float64))
    for i in range(5):
        slices, scales = f32_extract_split_rows(residual_f32, rho32, n_slices=1)
        v_unscaled = slices[0]
        row_scale = np.float32(np.ldexp(
            np.ones(n, dtype=np.float32), scales[0].astype(np.int32)))
        v_original = np.float32(v_unscaled * row_scale[:, np.newaxis])
        residual_f32 = np.float32(residual_f32 - v_original)
        r_frac = np.linalg.norm(residual_f32.astype(np.float64)) / hi_norm
        print(f"    after slice {i}: residual = {r_frac:.2e}")
    print()

    # FP32 extraction of lo part.
    print("  FP32 extraction of lo part (residual fraction of lo norm):")
    residual_f32 = X_f32_lo.copy()
    lo_norm = np.linalg.norm(X_f32_lo.astype(np.float64))
    if lo_norm > 0:
        for i in range(4):
            slices, scales = f32_extract_split_rows(residual_f32, rho32, n_slices=1)
            v_unscaled = slices[0]
            row_scale = np.float32(np.ldexp(
                np.ones(n, dtype=np.float32), scales[0].astype(np.int32)))
            v_original = np.float32(v_unscaled * row_scale[:, np.newaxis])
            residual_f32 = np.float32(residual_f32 - v_original)
            r_frac = np.linalg.norm(residual_f32.astype(np.float64)) / lo_norm
            print(f"    after slice {i}: residual = {r_frac:.2e}")
    else:
        print("    lo part is zero (no residual)")
    print()

    # === Test 2: BF16-exactness across sizes ===
    print("test 2: BF16-exactness across sizes\n")
    all_exact = True
    for n in [32, 64, 128, 256, 512]:
        X = rng.randn(n, n).astype(np.float32)
        rho = _compute_rho_f32(n)
        slices, _ = f32_extract_split_rows(X, rho, n_slices=5)

        max_err = 0.0
        for s in slices:
            bf16_rt = np.array(jnp.float32(jnp.bfloat16(jnp.float32(s))))
            max_err = max(max_err, np.max(np.abs(s - bf16_rt)))

        ok = max_err == 0.0
        all_exact &= ok
        print(f"  n={n:>4}: max BF16 roundtrip error = {max_err:.0e} "
              f"[{'PASS' if ok else 'FAIL'}]")
    print()

    # === Test 3: Slice count sensitivity ===
    print("test 3: accuracy vs number of hi/lo slices\n")
    from ozaki_jax import matmul as ozaki_matmul
    from ozaki_jax.matmul import (
        _ondevice_gemms_jit, _accumulate_block_products,
    )

    n = 128
    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)
    C_ref = A @ B
    ref_norm = np.linalg.norm(C_ref)

    # Reference: host pipeline.
    C_host = ozaki_matmul(A, B, pipeline="host")
    e_host = np.linalg.norm(C_host - C_ref) / ref_norm

    print(f"  Host pipeline (8 slices, 36 GEMMs): {e_host:.2e}")
    print()

    rho = _compute_rho_f32(n)
    A_hi, A_lo = _double_f32_split(A)
    B_hi, B_lo = _double_f32_split(B)

    configs = [(3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]
    print(f"  {'n_hi':>4} {'n_lo':>4} {'GEMMs':>6} {'Error':>12} {'vs Host':>8}")
    print(f"  {'-'*4} {'-'*4} {'-'*6} {'-'*12} {'-'*8}")

    for n_hi, n_lo in configs:
        A_hi_sl, A_hi_sc = f32_extract_split_rows(A_hi, rho, n_hi)
        A_lo_sl, A_lo_sc = f32_extract_split_rows(A_lo, rho, n_lo)
        B_hi_sl, B_hi_sc = f32_extract_split_cols(B_hi, rho, n_hi)
        B_lo_sl, B_lo_sc = f32_extract_split_cols(B_lo, rho, n_lo)

        n_gemms = n_hi * n_hi + n_hi * n_lo + n_lo * n_hi

        # Compute via numpy (no JAX needed for this analysis).
        products = []
        for i in range(n_hi):
            for j in range(n_hi):
                products.append(np.float32(A_hi_sl[i] @ B_hi_sl[j]))
        for i in range(n_hi):
            for j in range(n_lo):
                products.append(np.float32(A_hi_sl[i] @ B_lo_sl[j]))
        for i in range(n_lo):
            for j in range(n_hi):
                products.append(np.float32(A_lo_sl[i] @ B_hi_sl[j]))

        C = _accumulate_block_products(
            np.stack(products), A_hi_sc, A_lo_sc,
            B_hi_sc, B_lo_sc, n, n, n_hi, n_lo, n_hi)

        err = np.linalg.norm(C - C_ref) / ref_norm
        ratio = err / e_host if e_host > 0 else float("inf")
        print(f"  {n_hi:>4} {n_lo:>4} {n_gemms:>6} {err:>12.2e} {ratio:>7.1f}x")
    print()

    # === Test 4: Reconstruction completeness ===
    print("test 4: extraction reconstruction error\n")
    n = 128
    for label, dtype, extract_fn, compute_rho, n_slices in [
        ("FP64", np.float64, extract_split_rows, _compute_rho, 8),
        ("FP32", np.float32, f32_extract_split_rows, _compute_rho_f32, 5),
    ]:
        X = rng.randn(n, n).astype(dtype)
        rho = compute_rho(n)
        slices, scales = extract_fn(X, rho, n_slices)

        # Reconstruct from slices.
        recon = np.zeros((n, n), dtype=np.float64)
        for s, sc in zip(slices, scales):
            s_f64 = s.astype(np.float64)
            if dtype == np.float64:
                row_scale = np.ldexp(np.ones(n, dtype=np.float64),
                                     sc.astype(np.int64))
            else:
                row_scale = np.ldexp(np.ones(n, dtype=np.float64),
                                     sc.astype(np.int64))
            recon += s_f64 * row_scale[:, np.newaxis]

        err = np.max(np.abs(recon - X.astype(np.float64))) / np.max(np.abs(X))
        print(f"  {label} extraction ({n_slices} slices): "
              f"reconstruction error = {err:.2e}")
    print()

    # Summary.
    print(f"BF16-exactness: {'ALL PASS' if all_exact else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
