"""Accumulation optimization v2: focused on the most promising approach.

v1 findings:
  - factored_mixed is 2x faster but kills accuracy (FP32 inner sum)
  - factored FP64 is 1.2x (all FP64, too slow)
  - Key insight: inner sum has only 4-5 terms spanning ~2^28.
    Kahan FP32 gives ~2^-48 precision — exactly what double-FP32 needs.

This script tests:
  1. baseline: current _accumulate_block_products
  2. factored_mixed: FP32 inner sum (fast but broken, for reference)
  3. kahan_inner_f32: Kahan compensated FP32 inner sum + FP64 outer
  4. ldexp_factored: use ldexp instead of multiply for power-of-2 scales
  5. kahan_inner_ldexp: combine Kahan inner + ldexp scaling

Usage:
    python benchmarks/accumulation_experiments_v2.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax.extract import (
    _compute_rho_f32,
    f32_extract_split_rows, f32_extract_split_cols,
)
from ozaki_jax.matmul import (
    _double_f32_split, _accumulate_block_products,
    _ONDEVICE_N_HI, _ONDEVICE_N_LO,
)


def rel_err(C, C_ref):
    return np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)


def prepare_data(n, rng):
    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)
    C_ref = A @ B

    A_hi, A_lo = _double_f32_split(A)
    B_hi, B_lo = _double_f32_split(B)

    rho = _compute_rho_f32(n)
    n_hi, n_lo = _ONDEVICE_N_HI, _ONDEVICE_N_LO

    A_hi_sl, A_hi_sc = f32_extract_split_rows(A_hi, rho, n_hi)
    A_lo_sl, A_lo_sc = f32_extract_split_rows(A_lo, rho, n_lo)
    B_hi_sl, B_hi_sc = f32_extract_split_cols(B_hi, rho, n_hi)
    B_lo_sl, B_lo_sc = f32_extract_split_cols(B_lo, rho, n_lo)

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

    products_np = np.stack(products)

    return {
        "A": A, "B": B, "C_ref": C_ref,
        "products_np": products_np,
        "A_hi_sc": A_hi_sc, "A_lo_sc": A_lo_sc,
        "B_hi_sc": B_hi_sc, "B_lo_sc": B_lo_sc,
        "n_hi": n_hi, "n_lo": n_lo,
        "N": n, "M": n,
    }


# ====================================================================
# Method 1: Baseline
# ====================================================================

def accum_baseline(d):
    return _accumulate_block_products(
        d["products_np"], d["A_hi_sc"], d["A_lo_sc"],
        d["B_hi_sc"], d["B_lo_sc"], d["N"], d["M"],
        d["n_hi"], d["n_lo"], d["n_hi"])


# ====================================================================
# Method 2: Factored mixed (fast but broken — reference)
# ====================================================================

def accum_factored_mixed(d):
    """FP32 inner sum, FP64 outer. Fast but ~1e-8 accuracy."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi_64 = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
                 for i in range(n_hi)]
    row_lo_64 = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
                 for i in range(n_lo)]
    col_hi_32 = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
                 for j in range(n_hi)]
    col_lo_32 = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
                 for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    inner = np.zeros((N, M), dtype=np.float32)
    idx = 0

    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_hi):
            inner += products_np[idx] * col_hi_32[j][None, :]
            idx += 1
        C += np.float64(inner) * row_hi_64[i][:, None]

    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_lo):
            inner += products_np[idx] * col_lo_32[j][None, :]
            idx += 1
        C += np.float64(inner) * row_hi_64[i][:, None]

    for i in range(n_lo):
        inner[:] = 0.0
        for j in range(n_hi):
            inner += products_np[idx] * col_hi_32[j][None, :]
            idx += 1
        C += np.float64(inner) * row_lo_64[i][:, None]

    return C


# ====================================================================
# Method 3: Kahan FP32 inner sum + FP64 outer
# ====================================================================

def accum_kahan_inner(d):
    """Kahan-compensated FP32 inner sum (4-5 terms), FP64 for outer sum.

    Inner sum: sum_{j} P_{ij} * col_j  (4-5 terms, span ~2^28)
    Kahan gives ~2^-48 precision on this sum.
    Then promote to FP64, apply row_scale, accumulate into C.
    """
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi_64 = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
                 for i in range(n_hi)]
    row_lo_64 = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
                 for i in range(n_lo)]
    col_hi_32 = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
                 for j in range(n_hi)]
    col_lo_32 = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
                 for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    S = np.zeros((N, M), dtype=np.float32)
    comp = np.zeros((N, M), dtype=np.float32)
    buf = np.empty((N, M), dtype=np.float32)
    idx = 0

    def kahan_group(col_scales, n_b, row_scale_64):
        """Kahan-sum n_b col-scaled products, apply row_scale in FP64."""
        nonlocal idx
        S[:] = 0.0
        comp[:] = 0.0
        for j in range(n_b):
            np.multiply(products_np[idx], col_scales[j][None, :], out=buf)
            # Kahan step: y = term - comp; t = S + y; comp = (t-S) - y; S = t
            np.subtract(buf, comp, out=buf)  # y = term - comp
            t = np.float32(S + buf)          # t = S + y
            np.subtract(t, S, out=comp)      # comp = (t - S)
            np.subtract(comp, buf, out=comp) # comp -= y  (now comp = (t-S)-y)
            np.copyto(S, t)
            idx += 1
        # Promote to FP64, apply row scale, accumulate.
        C.__iadd__(np.float64(S) * row_scale_64[:, None])

    for i in range(n_hi):
        kahan_group(col_hi_32, n_hi, row_hi_64[i])
    for i in range(n_hi):
        kahan_group(col_lo_32, n_lo, row_hi_64[i])
    for i in range(n_lo):
        kahan_group(col_hi_32, n_hi, row_lo_64[i])

    return C


# ====================================================================
# Method 4: Kahan inner with ldexp scaling (avoid FP multiply for scales)
# ====================================================================

def accum_kahan_inner_ldexp(d):
    """Same as method 3 but use ldexp for power-of-2 col scaling."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi_64 = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
                 for i in range(n_hi)]
    row_lo_64 = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
                 for i in range(n_lo)]
    # Integer exponents for ldexp.
    col_hi_exp = [d["B_hi_sc"][j].astype(np.int32) for j in range(n_hi)]
    col_lo_exp = [d["B_lo_sc"][j].astype(np.int32) for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    S = np.zeros((N, M), dtype=np.float32)
    comp = np.zeros((N, M), dtype=np.float32)
    buf = np.empty((N, M), dtype=np.float32)
    idx = 0

    def kahan_group_ldexp(col_exps, n_b, row_scale_64):
        nonlocal idx
        S[:] = 0.0
        comp[:] = 0.0
        for j in range(n_b):
            # ldexp: shift exponent by col_exp (exact for power-of-2).
            np.ldexp(products_np[idx], col_exps[j][None, :], out=buf)
            np.subtract(buf, comp, out=buf)
            t = np.float32(S + buf)
            np.subtract(t, S, out=comp)
            np.subtract(comp, buf, out=comp)
            np.copyto(S, t)
            idx += 1
        C.__iadd__(np.float64(S) * row_scale_64[:, None])

    for i in range(n_hi):
        kahan_group_ldexp(col_hi_exp, n_hi, row_hi_64[i])
    for i in range(n_hi):
        kahan_group_ldexp(col_lo_exp, n_lo, row_hi_64[i])
    for i in range(n_lo):
        kahan_group_ldexp(col_hi_exp, n_hi, row_lo_64[i])

    return C


# ====================================================================
# Method 5: Factored FP64 (best of v1 FP64 methods)
# ====================================================================

def accum_factored_f64(d):
    """Inner sum in FP64, factored row-scale. Reference for accuracy."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float64), d["B_hi_sc"][j].astype(np.int64))
              for j in range(n_hi)]
    col_lo = [np.ldexp(np.ones(M, np.float64), d["B_lo_sc"][j].astype(np.int64))
              for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    inner = np.zeros((N, M), dtype=np.float64)
    buf = np.empty((N, M), dtype=np.float64)
    idx = 0

    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_hi):
            np.copyto(buf, products_np[idx], casting="unsafe")
            buf *= col_hi[j][None, :]
            inner += buf
            idx += 1
        inner *= row_hi[i][:, None]
        C += inner

    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_lo):
            np.copyto(buf, products_np[idx], casting="unsafe")
            buf *= col_lo[j][None, :]
            inner += buf
            idx += 1
        inner *= row_hi[i][:, None]
        C += inner

    for i in range(n_lo):
        inner[:] = 0.0
        for j in range(n_hi):
            np.copyto(buf, products_np[idx], casting="unsafe")
            buf *= col_hi[j][None, :]
            inner += buf
            idx += 1
        inner *= row_lo[i][:, None]
        C += inner

    return C


# ====================================================================
# Method 6: Double-compensation approach
#   Inner sum: (S_hi, S_lo) tracked via error-free add
#   Both promoted to FP64 for outer sum
# ====================================================================

def accum_double_inner(d):
    """Track inner sum as (S_hi + S_lo) FP32 pair for ~48-bit accuracy.

    Uses 2Sum error-free transformation:
      S_new = fl(S + x)
      error = fl((S - S_new) + x)
    Sum of errors tracked in S_lo. Final: FP64(S_hi) + FP64(S_lo).
    """
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi_64 = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
                 for i in range(n_hi)]
    row_lo_64 = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
                 for i in range(n_lo)]
    col_hi_32 = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
                 for j in range(n_hi)]
    col_lo_32 = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
                 for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    S_hi = np.zeros((N, M), dtype=np.float32)
    S_lo = np.zeros((N, M), dtype=np.float32)
    buf = np.empty((N, M), dtype=np.float32)
    t = np.empty((N, M), dtype=np.float32)
    idx = 0

    def twosum_group(col_scales, n_b, row_scale_64, S_hi, S_lo, buf, t):
        """2Sum inner accumulation in FP32 pair."""
        nonlocal idx
        S_hi[:] = 0.0
        S_lo[:] = 0.0
        for j in range(n_b):
            np.multiply(products_np[idx], col_scales[j][None, :], out=buf)
            # 2Sum: t = fl(S_hi + buf); e = fl((S_hi - t) + buf)
            np.add(S_hi, buf, out=t)
            # e = (S_hi - t) + buf
            e = np.float32(np.float32(S_hi - t) + buf)
            S_lo += e
            np.copyto(S_hi, t)
            idx += 1
        # Combine: FP64(S_hi) + FP64(S_lo), then row-scale.
        inner_64 = np.float64(S_hi) + np.float64(S_lo)
        C.__iadd__(inner_64 * row_scale_64[:, None])

    for i in range(n_hi):
        twosum_group(col_hi_32, n_hi, row_hi_64[i], S_hi, S_lo, buf, t)
    for i in range(n_hi):
        twosum_group(col_lo_32, n_lo, row_hi_64[i], S_hi, S_lo, buf, t)
    for i in range(n_lo):
        twosum_group(col_hi_32, n_hi, row_lo_64[i], S_hi, S_lo, buf, t)

    return C


# ====================================================================
# Method 7: Pre-scale products then FP64 pairwise sum
#   Apply both scales in FP32 (exact power-of-2), then promote to FP64
#   and accumulate. Avoids FP64 multiply entirely.
# ====================================================================

def accum_prescale_f32(d):
    """Apply row+col scales in FP32 (exact), promote, accumulate in FP64.

    Since scales are powers of 2, multiplying in FP32 is exact (just shifts
    the exponent). The product value doesn't change, only its magnitude.
    Then we promote to FP64 for the addition.

    Trade-off: 65 FP64 additions (no FP64 multiplies), 130 FP32 multiplies.
    """
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi_32 = [np.ldexp(np.ones(N, np.float32), d["A_hi_sc"][i].astype(np.int32))
                 for i in range(n_hi)]
    row_lo_32 = [np.ldexp(np.ones(N, np.float32), d["A_lo_sc"][i].astype(np.int32))
                 for i in range(n_lo)]
    col_hi_32 = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
                 for j in range(n_hi)]
    col_lo_32 = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
                 for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    buf = np.empty((N, M), dtype=np.float32)
    idx = 0

    idx = 0
    for i in range(n_hi):
        for j in range(n_hi):
            np.multiply(products_np[idx], row_hi_32[i][:, None], out=buf)
            buf *= col_hi_32[j][None, :]
            C.__iadd__(buf.astype(np.float64))
            idx += 1
    for i in range(n_hi):
        for j in range(n_lo):
            np.multiply(products_np[idx], row_hi_32[i][:, None], out=buf)
            buf *= col_lo_32[j][None, :]
            C.__iadd__(buf.astype(np.float64))
            idx += 1
    for i in range(n_lo):
        for j in range(n_hi):
            np.multiply(products_np[idx], row_lo_32[i][:, None], out=buf)
            buf *= col_hi_32[j][None, :]
            C.__iadd__(buf.astype(np.float64))
            idx += 1

    return C


# ====================================================================
# Method 8: Pre-scale FP32 + factored row-scale
#   Col-scale in FP32, sum per group in FP64, row-scale in FP64
# ====================================================================

def accum_prescale_factored(d):
    """Col-scale in FP32, promote to FP64, inner-sum, row-scale in FP64.

    Combines: FP32 col-scale (exact), FP64 inner summation, FP64 row-scale.
    FP64 ops: 65 promotions + 65 inner-adds + 14 row-muls + 14 outer-adds = 158
    FP32 ops: 65 col-muls
    """
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi_64 = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
                 for i in range(n_hi)]
    row_lo_64 = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
                 for i in range(n_lo)]
    col_hi_32 = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
                 for j in range(n_hi)]
    col_lo_32 = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
                 for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    inner = np.zeros((N, M), dtype=np.float64)
    buf32 = np.empty((N, M), dtype=np.float32)
    idx = 0

    def do_group(col_scales, n_b, row_scale_64, inner_buf, tmp_buf):
        nonlocal idx
        inner_buf[:] = 0.0
        for j in range(n_b):
            np.multiply(products_np[idx], col_scales[j][None, :], out=tmp_buf)
            inner_buf += tmp_buf  # numpy auto-promotes FP32+FP64 → FP64
            idx += 1
        inner_buf *= row_scale_64[:, None]
        C.__iadd__(inner_buf)

    for i in range(n_hi):
        do_group(col_hi_32, n_hi, row_hi_64[i], inner, buf32)
    for i in range(n_hi):
        do_group(col_lo_32, n_lo, row_hi_64[i], inner, buf32)
    for i in range(n_lo):
        do_group(col_hi_32, n_hi, row_lo_64[i], inner, buf32)

    return C


# ====================================================================
# Timing + Accuracy harness
# ====================================================================

def time_fn(fn, d, n_calls=20, warmup=3):
    for _ in range(warmup):
        fn(d)
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        fn(d)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    rng = np.random.RandomState(42)

    methods = [
        ("baseline",           accum_baseline),
        ("factored_mixed",     accum_factored_mixed),
        ("kahan_inner",        accum_kahan_inner),
        ("kahan_inner_ldexp",  accum_kahan_inner_ldexp),
        ("factored_f64",       accum_factored_f64),
        ("double_inner",       accum_double_inner),
        ("prescale_f32",       accum_prescale_f32),
        ("prescale_factored",  accum_prescale_factored),
    ]

    # ================================================================
    # Part 1: Accuracy (n=256, 5 trials)
    # ================================================================
    print("=" * 70)
    print("PART 1: Accuracy (n=256, 5 trials)")
    print("=" * 70)
    print()

    errors = {name: [] for name, _ in methods}

    for trial in range(5):
        d = prepare_data(256, np.random.RandomState(42 + trial))
        C_ref = d["C_ref"]
        for name, fn in methods:
            C = fn(d)
            errors[name].append(rel_err(C, C_ref))

    baseline_err = np.mean(errors["baseline"])
    print(f"  {'Method':<24} {'Mean Error':>12} {'vs Baseline':>12} {'Status':>10}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*10}")
    for name, _ in methods:
        e = np.mean(errors[name])
        ratio = e / baseline_err if baseline_err > 0 else 0
        if e > 1e-10:
            status = "BROKEN"
        elif ratio > 2.0:
            status = "degraded"
        elif ratio < 1.1:
            status = "EXACT"
        else:
            status = "ok"
        print(f"  {name:<24} {e:>12.2e} {ratio:>11.2f}x  {status:>10}")
    print()

    # ================================================================
    # Part 2: Stress test accuracy across scales and conditions
    # ================================================================
    print("=" * 70)
    print("PART 2: Accuracy stress test (challenging matrices)")
    print("=" * 70)
    print()

    stress_cases = []

    # Large scale.
    A = rng.randn(128, 128).astype(np.float64) * 1e10
    B = rng.randn(128, 128).astype(np.float64) * 1e10
    stress_cases.append(("scale=1e10", A, B))

    # Mixed-scale columns.
    A = rng.randn(128, 128).astype(np.float64)
    col_scales = np.logspace(-5, 5, 128)
    rng.shuffle(col_scales)
    A *= col_scales[np.newaxis, :]
    B = rng.randn(128, 128).astype(np.float64)
    stress_cases.append(("mixed_cols", A, B))

    # Ill-conditioned.
    U, _, Vt = np.linalg.svd(rng.randn(128, 128).astype(np.float64))
    s = np.logspace(0, -6, 128)
    A = (U * s) @ Vt
    B = rng.randn(128, 128).astype(np.float64)
    stress_cases.append(("cond=1e6", A, B))

    # Rectangular.
    A = rng.randn(64, 256).astype(np.float64)
    B = rng.randn(256, 64).astype(np.float64)
    stress_cases.append(("64x256@256x64", A, B))

    # Filter to methods with acceptable base accuracy.
    good_methods = [(n, f) for n, f in methods
                    if np.mean(errors[n]) < 1e-10]

    for case_name, A, B in stress_cases:
        C_ref = A @ B

        A_hi, A_lo = _double_f32_split(A)
        B_hi, B_lo = _double_f32_split(B)

        rho = _compute_rho_f32(A.shape[1])
        n_hi, n_lo = _ONDEVICE_N_HI, _ONDEVICE_N_LO

        A_hi_sl, A_hi_sc = f32_extract_split_rows(A_hi, rho, n_hi)
        A_lo_sl, A_lo_sc = f32_extract_split_rows(A_lo, rho, n_lo)
        B_hi_sl, B_hi_sc = f32_extract_split_cols(B_hi, rho, n_hi)
        B_lo_sl, B_lo_sc = f32_extract_split_cols(B_lo, rho, n_lo)

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

        dd = {
            "A": A, "B": B, "C_ref": C_ref,
            "products_np": np.stack(products),
            "A_hi_sc": A_hi_sc, "A_lo_sc": A_lo_sc,
            "B_hi_sc": B_hi_sc, "B_lo_sc": B_lo_sc,
            "n_hi": n_hi, "n_lo": n_lo,
            "N": A.shape[0], "M": B.shape[1],
        }

        print(f"  {case_name}:")
        base_e = rel_err(accum_baseline(dd), C_ref)
        for name, fn in good_methods:
            e = rel_err(fn(dd), C_ref)
            ratio = e / base_e if base_e > 0 else 0
            ok = "PASS" if ratio < 2.0 else "FAIL"
            print(f"    {name:<24} {e:.2e}  ({ratio:.2f}x)  [{ok}]")
        print()

    # ================================================================
    # Part 3: Timing
    # ================================================================
    for n in [256, 512]:
        print("=" * 70)
        print(f"PART 3: Timing (n={n}, median of 20 calls)")
        print("=" * 70)
        print()

        d = prepare_data(n, rng)
        baseline_time = time_fn(accum_baseline, d)

        print(f"  {'Method':<24} {'Time (ms)':>10} {'Speedup':>8} {'Accuracy':>10}")
        print(f"  {'-'*24} {'-'*10} {'-'*8} {'-'*10}")
        for name, fn in methods:
            t = time_fn(fn, d)
            speedup = baseline_time / t if t > 0 else 0
            e = np.mean(errors[name])
            status = "OK" if e < 1e-10 else "BROKEN"
            print(f"  {name:<24} {t:>10.2f} {speedup:>7.2f}x  {status:>10}")
        print()

    # ================================================================
    # Part 4: Detailed breakdown of winning method
    # ================================================================
    print("=" * 70)
    print("PART 4: Winner analysis")
    print("=" * 70)
    print()

    # Find the fastest accurate method.
    n = 512
    d = prepare_data(n, rng)
    best_name, best_time = None, float("inf")
    for name, fn in methods:
        if np.mean(errors[name]) > 1e-10:
            continue
        t = time_fn(fn, d)
        if t < best_time:
            best_name, best_time = name, t

    baseline_time = time_fn(accum_baseline, d)
    print(f"  Winner: {best_name}")
    print(f"  Baseline: {baseline_time:.2f}ms")
    print(f"  Winner:   {best_time:.2f}ms")
    print(f"  Speedup:  {baseline_time/best_time:.2f}x")
    print()

    # Scale sweep.
    print("  Scale sweep (winner vs baseline):")
    print(f"  {'n':>6} {'Baseline (ms)':>14} {'Winner (ms)':>12} {'Speedup':>8}")
    print(f"  {'-'*6} {'-'*14} {'-'*12} {'-'*8}")
    winner_fn = dict(methods)[best_name]
    for n in [64, 128, 256, 512, 1024]:
        d = prepare_data(n, rng)
        t_base = time_fn(accum_baseline, d, n_calls=10)
        t_win = time_fn(winner_fn, d, n_calls=10)
        print(f"  {n:>6} {t_base:>14.2f} {t_win:>12.2f} {t_base/t_win:>7.2f}x")
    print()


if __name__ == "__main__":
    main()
