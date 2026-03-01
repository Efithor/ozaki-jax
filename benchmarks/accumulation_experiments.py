"""Investigate accumulation optimizations for the on-device pipeline.

The accumulation step rescales 65 GEMM products from normalized FP32 back to
FP64, applying per-row and per-column scale factors, then sums. On TPU v6e
this takes 7.1ms (63% of pipeline time), making it the bottleneck.

Experiments:
  1. Baseline: current _accumulate_block_products (Python loop, per-product ldexp)
  2. Precompute scales: compute all scale vectors once, reuse in loop
  3. In-place buffer reuse: stop allocating new FP64 arrays each iteration
  4. Factored accumulation: inner-sum per A-slice group, then row-scale
  5. Vectorized broadcast: apply all scales via numpy broadcasting, sum axis=0
  6. Combined: precompute + factored + in-place (best of 2-4)
  7. Kahan FP32: compensated summation without FP64 (accuracy test)

Usage:
    python benchmarks/accumulation_experiments.py
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
    """Generate test matrices and extract slices + products."""
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

    # Compute GEMM products (FP32).
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

    products_np = np.stack(products)  # (65, N, M)

    return {
        "A": A, "B": B, "C_ref": C_ref,
        "products_np": products_np,
        "A_hi_sc": A_hi_sc, "A_lo_sc": A_lo_sc,
        "B_hi_sc": B_hi_sc, "B_lo_sc": B_lo_sc,
        "n_hi": n_hi, "n_lo": n_lo,
        "N": n, "M": n,
    }


# ====================================================================
# Method 1: Baseline (current _accumulate_block_products)
# ====================================================================

def accum_baseline(d):
    return _accumulate_block_products(
        d["products_np"], d["A_hi_sc"], d["A_lo_sc"],
        d["B_hi_sc"], d["B_lo_sc"], d["N"], d["M"],
        d["n_hi"], d["n_lo"], d["n_hi"])


# ====================================================================
# Method 2: Precompute scale vectors
# ====================================================================

def accum_precompute(d):
    """Precompute all scale vectors once, reuse in loop."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    # Precompute all row/col scale vectors (18 ldexp calls instead of 130).
    row_hi = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float64), d["B_hi_sc"][j].astype(np.int64))
              for j in range(n_hi)]
    col_lo = [np.ldexp(np.ones(M, np.float64), d["B_lo_sc"][j].astype(np.int64))
              for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    idx = 0

    for i in range(n_hi):
        for j in range(n_hi):
            P = products_np[idx].astype(np.float64)
            P *= row_hi[i][:, None]
            P *= col_hi[j][None, :]
            C += P
            idx += 1
    for i in range(n_hi):
        for j in range(n_lo):
            P = products_np[idx].astype(np.float64)
            P *= row_hi[i][:, None]
            P *= col_lo[j][None, :]
            C += P
            idx += 1
    for i in range(n_lo):
        for j in range(n_hi):
            P = products_np[idx].astype(np.float64)
            P *= row_lo[i][:, None]
            P *= col_hi[j][None, :]
            C += P
            idx += 1
    return C


# ====================================================================
# Method 3: In-place buffer reuse
# ====================================================================

def accum_inplace(d):
    """Reuse a single FP64 buffer instead of allocating per product."""
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
    buf = np.empty((N, M), dtype=np.float64)
    idx = 0

    def _accum(a_sc, b_sc):
        nonlocal idx
        # Copy FP32 → FP64 into pre-allocated buffer.
        np.copyto(buf, products_np[idx], casting="unsafe")
        np.multiply(buf, a_sc[:, None], out=buf)
        np.multiply(buf, b_sc[None, :], out=buf)
        np.add(C, buf, out=C)
        idx += 1

    for i in range(n_hi):
        for j in range(n_hi):
            _accum(row_hi[i], col_hi[j])
    for i in range(n_hi):
        for j in range(n_lo):
            _accum(row_hi[i], col_lo[j])
    for i in range(n_lo):
        for j in range(n_hi):
            _accum(row_lo[i], col_hi[j])
    return C


# ====================================================================
# Method 4: Factored accumulation
# ====================================================================

def accum_factored(d):
    """Apply col-scales and sum per A-group first, then apply row-scale once.

    C = sum_{i,j} P_{ij} * row_i[:, None] * col_j[None, :]
      = sum_i row_i[:, None] * (sum_j P_{ij} * col_j[None, :])

    Reduces row-scale operations from 65 to 14.
    """
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

    # hi × hi: group by A_hi slice i.
    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_hi):
            np.copyto(buf, products_np[idx])
            buf *= col_hi[j][None, :]
            inner += buf
            idx += 1
        inner *= row_hi[i][:, None]
        C += inner

    # hi × lo: group by A_hi slice i.
    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_lo):
            np.copyto(buf, products_np[idx])
            buf *= col_lo[j][None, :]
            inner += buf
            idx += 1
        inner *= row_hi[i][:, None]
        C += inner

    # lo × hi: group by A_lo slice i.
    for i in range(n_lo):
        inner[:] = 0.0
        for j in range(n_hi):
            np.copyto(buf, products_np[idx])
            buf *= col_hi[j][None, :]
            inner += buf
            idx += 1
        inner *= row_lo[i][:, None]
        C += inner

    return C


# ====================================================================
# Method 5: Vectorized broadcast
# ====================================================================

def accum_vectorized(d):
    """Apply all scales via numpy broadcasting, sum over product axis."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    # Build (65, N) row-scale array and (65, M) col-scale array.
    n_total = n_hi * n_hi + n_hi * n_lo + n_lo * n_hi
    row_sc = np.empty((n_total, N), dtype=np.float64)
    col_sc = np.empty((n_total, M), dtype=np.float64)

    row_hi = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float64), d["B_hi_sc"][j].astype(np.int64))
              for j in range(n_hi)]
    col_lo = [np.ldexp(np.ones(M, np.float64), d["B_lo_sc"][j].astype(np.int64))
              for j in range(n_lo)]

    idx = 0
    for i in range(n_hi):
        for j in range(n_hi):
            row_sc[idx] = row_hi[i]
            col_sc[idx] = col_hi[j]
            idx += 1
    for i in range(n_hi):
        for j in range(n_lo):
            row_sc[idx] = row_hi[i]
            col_sc[idx] = col_lo[j]
            idx += 1
    for i in range(n_lo):
        for j in range(n_hi):
            row_sc[idx] = row_lo[i]
            col_sc[idx] = col_hi[j]
            idx += 1

    # Vectorized: (65, N, M) * (65, N, 1) * (65, 1, M) → sum → (N, M)
    scaled = products_np.astype(np.float64)
    scaled *= row_sc[:, :, None]
    scaled *= col_sc[:, None, :]
    return scaled.sum(axis=0)


# ====================================================================
# Method 6: Combined (precompute + factored + in-place)
# ====================================================================

def accum_combined(d):
    """Best of methods 2-4: precompute, factor, reuse buffers."""
    # Same as accum_factored — it already includes all three optimizations.
    return accum_factored(d)


# ====================================================================
# Method 7: Kahan FP32 (no FP64 accumulation)
# ====================================================================

def accum_kahan_f32(d):
    """Kahan compensated summation in FP32. Tests if FP64 can be avoided."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    # Precompute scales in FP64, but apply in FP32 (power-of-2, so exact).
    # Actually, ldexp in FP32 can overflow/underflow. Row/col scales are
    # exponents like c_x which are ~ log2(max(row)). For unit-scale matrices,
    # c_x ~ 0..7. The combined scale for a product element is 2^(c_x_i + c_y_j),
    # which for hi slices could be 2^14 or so — well within FP32 range.
    row_hi = [np.ldexp(np.ones(N, np.float32), d["A_hi_sc"][i].astype(np.int32))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float32), d["A_lo_sc"][i].astype(np.int32))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
              for j in range(n_hi)]
    col_lo = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
              for j in range(n_lo)]

    # Kahan summation state.
    S = np.zeros((N, M), dtype=np.float32)    # running sum
    comp = np.zeros((N, M), dtype=np.float32)  # compensation
    buf = np.empty((N, M), dtype=np.float32)
    idx = 0

    def _kahan_add(val):
        nonlocal idx
        np.copyto(buf, products_np[idx])
        idx += 1

    # Process all products.
    idx = 0
    for i in range(n_hi):
        for j in range(n_hi):
            buf = np.float32(products_np[idx] * row_hi[i][:, None] * col_hi[j][None, :])
            y = buf - comp
            t = S + y
            comp = (t - S) - y
            S = t
            idx += 1
    for i in range(n_hi):
        for j in range(n_lo):
            buf = np.float32(products_np[idx] * row_hi[i][:, None] * col_lo[j][None, :])
            y = buf - comp
            t = S + y
            comp = (t - S) - y
            S = t
            idx += 1
    for i in range(n_lo):
        for j in range(n_hi):
            buf = np.float32(products_np[idx] * row_lo[i][:, None] * col_hi[j][None, :])
            y = buf - comp
            t = S + y
            comp = (t - S) - y
            S = t
            idx += 1

    return np.float64(S)


# ====================================================================
# Method 8: Sorted Kahan FP32 (sort products by magnitude)
# ====================================================================

def accum_sorted_kahan_f32(d):
    """Sort products by ascending magnitude before Kahan FP32 summation."""
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi = [np.ldexp(np.ones(N, np.float32), d["A_hi_sc"][i].astype(np.int32))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float32), d["A_lo_sc"][i].astype(np.int32))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
              for j in range(n_hi)]
    col_lo = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
              for j in range(n_lo)]

    # Pre-scale all products and estimate their magnitudes.
    scaled_products = []
    idx = 0
    for i in range(n_hi):
        for j in range(n_hi):
            sp = np.float32(products_np[idx] * row_hi[i][:, None] * col_hi[j][None, :])
            scaled_products.append(sp)
            idx += 1
    for i in range(n_hi):
        for j in range(n_lo):
            sp = np.float32(products_np[idx] * row_hi[i][:, None] * col_lo[j][None, :])
            scaled_products.append(sp)
            idx += 1
    for i in range(n_lo):
        for j in range(n_hi):
            sp = np.float32(products_np[idx] * row_lo[i][:, None] * col_hi[j][None, :])
            scaled_products.append(sp)
            idx += 1

    # Sort by Frobenius norm (ascending — add small values first).
    magnitudes = [np.linalg.norm(sp) for sp in scaled_products]
    order = np.argsort(magnitudes)

    # Kahan summation in sorted order.
    S = np.zeros((N, M), dtype=np.float32)
    comp = np.zeros((N, M), dtype=np.float32)
    for k in order:
        y = scaled_products[k] - comp
        t = S + y
        comp = (t - S) - y
        S = t

    return np.float64(S)


# ====================================================================
# Method 9: Hybrid Kahan — FP32 summation within blocks, FP64 across blocks
# ====================================================================

def accum_hybrid_kahan(d):
    """FP32 Kahan within each block (hh/hl/lh), FP64 across blocks.

    Products within a block have similar magnitude (same scale structure),
    so FP32 Kahan is safe. The three block sums are then combined in FP64.
    """
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    row_hi = [np.ldexp(np.ones(N, np.float32), d["A_hi_sc"][i].astype(np.int32))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float32), d["A_lo_sc"][i].astype(np.int32))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
              for j in range(n_hi)]
    col_lo = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
              for j in range(n_lo)]

    def kahan_block(idx_start, a_scales, b_scales, na, nb):
        """Kahan-sum a block in FP32."""
        S = np.zeros((N, M), dtype=np.float32)
        comp = np.zeros((N, M), dtype=np.float32)
        idx = idx_start
        for i in range(na):
            for j in range(nb):
                p = np.float32(products_np[idx] * a_scales[i][:, None]
                               * b_scales[j][None, :])
                y = p - comp
                t = S + y
                comp = (t - S) - y
                S = t
                idx += 1
        return S, idx

    idx = 0
    S_hh, idx = kahan_block(idx, row_hi, col_hi, n_hi, n_hi)
    S_hl, idx = kahan_block(idx, row_hi, col_lo, n_hi, n_lo)
    S_lh, idx = kahan_block(idx, row_lo, col_hi, n_lo, n_hi)

    # Combine blocks in FP64.
    return np.float64(S_hh) + np.float64(S_hl) + np.float64(S_lh)


# ====================================================================
# Method 10: Factored with FP32 inner sum (col-scale in FP32, row-scale in FP64)
# ====================================================================

def accum_factored_mixed(d):
    """Col-scale and inner-sum in FP32 (same magnitude), row-scale in FP64.

    Within each A-group, the products P_{i,0..k} share the same row scale.
    The col-scaled products have similar magnitude (same A_slice i),
    so FP32 summation of 4-5 terms is safe. Then promote to FP64 for
    the row-scale and final accumulation.
    """
    N, M = d["N"], d["M"]
    n_hi, n_lo = d["n_hi"], d["n_lo"]
    products_np = d["products_np"]

    # FP64 row scales (need precision for final accumulation).
    row_hi_64 = [np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][i].astype(np.int64))
                 for i in range(n_hi)]
    row_lo_64 = [np.ldexp(np.ones(N, np.float64), d["A_lo_sc"][i].astype(np.int64))
                 for i in range(n_lo)]
    # FP32 col scales (power-of-2, exact in FP32).
    col_hi_32 = [np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][j].astype(np.int32))
                 for j in range(n_hi)]
    col_lo_32 = [np.ldexp(np.ones(M, np.float32), d["B_lo_sc"][j].astype(np.int32))
                 for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    inner = np.zeros((N, M), dtype=np.float32)
    idx = 0

    # hi × hi
    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_hi):
            inner += products_np[idx] * col_hi_32[j][None, :]
            idx += 1
        C += np.float64(inner) * row_hi_64[i][:, None]

    # hi × lo
    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_lo):
            inner += products_np[idx] * col_lo_32[j][None, :]
            idx += 1
        C += np.float64(inner) * row_hi_64[i][:, None]

    # lo × hi
    for i in range(n_lo):
        inner[:] = 0.0
        for j in range(n_hi):
            inner += products_np[idx] * col_hi_32[j][None, :]
            idx += 1
        C += np.float64(inner) * row_lo_64[i][:, None]

    return C


# ====================================================================
# Timing + Accuracy harness
# ====================================================================

def time_fn(fn, d, n_calls=20, warmup=3):
    """Median time in ms."""
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
        ("baseline",            accum_baseline),
        ("precompute_scales",   accum_precompute),
        ("inplace_buffer",      accum_inplace),
        ("factored",            accum_factored),
        ("vectorized",          accum_vectorized),
        ("kahan_f32",           accum_kahan_f32),
        ("sorted_kahan_f32",    accum_sorted_kahan_f32),
        ("hybrid_kahan",        accum_hybrid_kahan),
        ("factored_mixed",      accum_factored_mixed),
    ]

    # ================================================================
    # Part 1: Accuracy check (n=256)
    # ================================================================
    print("=" * 70)
    print("PART 1: Accuracy (n=256, 5 trials)")
    print("=" * 70)
    print()

    n = 256
    errors = {name: [] for name, _ in methods}

    for trial in range(5):
        d = prepare_data(n, np.random.RandomState(42 + trial))
        C_ref = d["C_ref"]

        for name, fn in methods:
            C = fn(d)
            errors[name].append(rel_err(C, C_ref))

    # Also get the full on-device pipeline error for reference.
    from ozaki_jax import matmul_numpy
    ref_errs = []
    for trial in range(5):
        d = prepare_data(n, np.random.RandomState(42 + trial))
        C_full = matmul_numpy(d["A"], d["B"], pipeline="ondevice")
        ref_errs.append(rel_err(C_full, d["C_ref"]))

    print(f"  {'Method':<24} {'Mean Error':>12} {'vs Baseline':>12}")
    print(f"  {'-'*24} {'-'*12} {'-'*12}")
    baseline_err = np.mean(errors["baseline"])
    for name, _ in methods:
        e = np.mean(errors[name])
        ratio = e / baseline_err if baseline_err > 0 else 0
        marker = ""
        if e > 1e-10:
            marker = " *** DEGRADED"
        elif ratio > 2.0:
            marker = " * worse"
        print(f"  {name:<24} {e:>12.2e} {ratio:>11.2f}x{marker}")
    print(f"  {'full_ondevice':<24} {np.mean(ref_errs):>12.2e} "
          f"{np.mean(ref_errs)/baseline_err:>11.2f}x  (reference)")
    print()

    # ================================================================
    # Part 2: Timing (n=256 and n=512)
    # ================================================================
    for n in [256, 512]:
        print("=" * 70)
        print(f"PART 2: Timing (n={n}, median of 20 calls)")
        print("=" * 70)
        print()

        d = prepare_data(n, rng)

        print(f"  {'Method':<24} {'Time (ms)':>10} {'Speedup':>8}")
        print(f"  {'-'*24} {'-'*10} {'-'*8}")
        baseline_time = time_fn(accum_baseline, d)
        for name, fn in methods:
            t = time_fn(fn, d)
            speedup = baseline_time / t if t > 0 else 0
            print(f"  {name:<24} {t:>10.2f} {speedup:>7.2f}x")
        print()

    # ================================================================
    # Part 3: Analysis — where does the time go?
    # ================================================================
    print("=" * 70)
    print("PART 3: Microbenchmarks (n=256)")
    print("=" * 70)
    print()

    n = 256
    d = prepare_data(n, rng)
    N, M = d["N"], d["M"]
    P = d["products_np"][0]

    def time_micro(label, fn, n_calls=1000):
        for _ in range(100):
            fn()
        times = []
        for _ in range(n_calls):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        t = np.median(times) * 1000
        print(f"  {label:<40} {t*1000:>8.1f} µs")

    time_micro("astype FP32→FP64",
               lambda: P.astype(np.float64))
    time_micro("np.copyto (FP32→FP64 preallocated)",
               lambda: np.copyto(np.empty((N, M), np.float64), P))
    time_micro("ldexp (N vector)",
               lambda: np.ldexp(np.ones(N, np.float64),
                                d["A_hi_sc"][0].astype(np.int64)))
    buf64 = np.empty((N, M), dtype=np.float64)
    row_s = np.ldexp(np.ones(N, np.float64), d["A_hi_sc"][0].astype(np.int64))
    col_s = np.ldexp(np.ones(M, np.float64), d["B_hi_sc"][0].astype(np.int64))
    time_micro("broadcast multiply row (N×M)",
               lambda: np.multiply(buf64, row_s[:, None], out=buf64))
    time_micro("broadcast multiply col (N×M)",
               lambda: np.multiply(buf64, col_s[None, :], out=buf64))
    C_test = np.zeros((N, M), dtype=np.float64)
    time_micro("accumulate += (N×M FP64)",
               lambda: C_test.__iadd__(buf64))
    # FP32 equivalents.
    P32 = d["products_np"][0]
    buf32 = np.empty((N, M), dtype=np.float32)
    row_s32 = np.ldexp(np.ones(N, np.float32), d["A_hi_sc"][0].astype(np.int32))
    col_s32 = np.ldexp(np.ones(M, np.float32), d["B_hi_sc"][0].astype(np.int32))
    time_micro("broadcast multiply row (N×M FP32)",
               lambda: np.multiply(P32, row_s32[:, None], out=buf32))
    time_micro("broadcast multiply col (N×M FP32)",
               lambda: np.multiply(buf32, col_s32[None, :], out=buf32))
    C32 = np.zeros((N, M), dtype=np.float32)
    time_micro("accumulate += (N×M FP32)",
               lambda: C32.__iadd__(buf32))
    time_micro("FP32→FP64 promotion (N×M)",
               lambda: buf32.astype(np.float64))
    print()

    # ================================================================
    # Part 4: Theoretical analysis
    # ================================================================
    print("=" * 70)
    print("PART 4: Operation count analysis")
    print("=" * 70)
    print()

    print("  Current baseline (65 products):")
    print("    65 × astype FP32→FP64    (alloc + copy)")
    print("    130 × ldexp              (scale vector creation)")
    print("    130 × broadcast multiply (N×M each, FP64)")
    print("    65 × accumulate +=       (N×M each, FP64)")
    print(f"    Total FP64 N×M ops: {130 + 65} = 195")
    print()

    print("  Factored (method 4):")
    print("    65 × copyto FP32→FP64    (into pre-allocated buf)")
    print("    18 × ldexp               (precomputed)")
    print("    65 × broadcast multiply  (col-scale in inner, FP64)")
    print("    65 × accumulate inner += (N×M each, FP64)")
    print("    14 × broadcast multiply  (row-scale on group sum)")
    print("    14 × accumulate C +=     (N×M each, FP64)")
    print(f"    Total FP64 N×M ops: {65 + 65 + 14 + 14} = 158")
    print()

    print("  Factored mixed (method 10):")
    print("    65 × col-scale multiply  (N×M each, FP32)")
    print("    65 × accumulate inner += (N×M each, FP32)")
    print("    14 × FP32→FP64 promotion (N×M each)")
    print("    14 × row-scale multiply  (N×M each, FP64)")
    print("    14 × accumulate C +=     (N×M each, FP64)")
    print(f"    Total FP32 N×M ops: {65 + 65} = 130")
    print(f"    Total FP64 N×M ops: {14 + 14 + 14} = 42")
    print()

    print("  Kahan FP32 (method 7):")
    print("    65 × scale (2 muls)       (N×M each, FP32)")
    print("    65 × Kahan step (4 ops)   (N×M each, FP32)")
    print("    1 × FP32→FP64 promotion")
    print(f"    Total FP32 N×M ops: {65*2 + 65*4} = {65*6}")
    print(f"    Total FP64 N×M ops: 1")
    print()


if __name__ == "__main__":
    main()
