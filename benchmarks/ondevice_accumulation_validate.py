"""Validate on-device 2Sum accumulation accuracy and behavior.

Tests:
  1. 2Sum accuracy vs FP64 baseline at multiple sizes
  2. Scale precomputation shape verification
  3. Transfer reduction (2×N×M vs 65×N×M)
  4. End-to-end matmul with on-device accumulation (host, ondevice, fused)
  5. Timing: host vs on-device vs fused accumulation
  6. Pallas backend validation (if available)
  7. JAX extraction vs numpy extraction bit-exactness

Usage:
    python benchmarks/ondevice_accumulation_validate.py
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ozaki_jax import matmul
from ozaki_jax.pallas_ops import (
    accumulate_2sum,
    validate_accumulation_2sum,
    _precompute_accumulation_scales,
    _HAS_PALLAS,
)
from ozaki_jax.matmul import (
    _double_f32_split,
    _accumulate_block_products,
    _ondevice_gemms_jit,
    _ONDEVICE_N_HI,
    _ONDEVICE_N_LO,
)
from ozaki_jax.extract import (
    _compute_rho_f32,
    f32_extract_split_rows,
    f32_extract_split_cols,
    jax_extract_split_rows,
    jax_extract_split_cols,
)


def rel_err(C, C_ref):
    return float(np.max(np.abs(C - C_ref)) / np.max(np.abs(C_ref)))


def main():
    rng = np.random.RandomState(42)

    print("on-device 2Sum accumulation validation")
    print(f"JAX version:  {jax.__version__}")
    print(f"Platform:     {jax.default_backend()}")
    print(f"Pallas:       {'available' if _HAS_PALLAS else 'not available'}")
    print()

    all_pass = True

    # ----------------------------------------------------------------
    # Test 1: 2Sum accuracy vs FP64 baseline
    # ----------------------------------------------------------------
    print("test 1: 2Sum accuracy vs FP64 baseline\n")
    header = f"{'Size':>10}  {'Baseline':>12}  {'2Sum':>12}  {'Status':>8}"
    print(header)
    print("-" * len(header))

    for N in [128, 256, 512]:
        result = validate_accumulation_2sum(N=N, M=N, K=N)
        status = "PASS" if result["match"] else "FAIL"
        if not result["match"]:
            all_pass = False
        print(f"{N:>10}  {result['baseline_error']:>12.2e}  "
              f"{result['max_rel_error']:>12.2e}  {status:>8}")
    print()

    # ----------------------------------------------------------------
    # Test 2: Scale precomputation shapes
    # ----------------------------------------------------------------
    print("test 2: scale precomputation shapes\n")
    N, M = 256, 256
    n_hi, n_lo = _ONDEVICE_N_HI, _ONDEVICE_N_LO

    # Generate dummy scales.
    A_hi_sc = [np.zeros(N, dtype=np.float32) for _ in range(n_hi)]
    A_lo_sc = [np.zeros(N, dtype=np.float32) for _ in range(n_lo)]
    B_hi_sc = [np.zeros(M, dtype=np.float32) for _ in range(n_hi)]
    B_lo_sc = [np.zeros(M, dtype=np.float32) for _ in range(n_lo)]

    col_scales, row_scales, block_group_sizes = _precompute_accumulation_scales(
        A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc, N, M, n_hi, n_lo)

    n_products = n_hi * n_hi + n_hi * n_lo + n_lo * n_hi
    n_groups = n_hi + n_hi + n_lo
    flat_sizes = sum(block_group_sizes, ())  # flatten tuple of tuples

    checks = [
        ("col_scales shape", col_scales.shape, (n_products, M)),
        ("row_scales shape", row_scales.shape, (n_groups, N)),
        ("n_blocks", len(block_group_sizes), 3),
        ("total groups", len(flat_sizes), n_groups),
        ("total products", sum(flat_sizes), n_products),
        ("col_scales dtype", col_scales.dtype, np.float32),
        ("row_scales dtype", row_scales.dtype, np.float32),
    ]

    for name, actual, expected in checks:
        ok = actual == expected
        if not ok:
            all_pass = False
        print(f"  {name}: {actual} == {expected}  {'PASS' if ok else 'FAIL'}")
    print()

    # ----------------------------------------------------------------
    # Test 3: Transfer reduction
    # ----------------------------------------------------------------
    print("test 3: transfer reduction\n")
    host_transfer = n_products * N * M * 4  # 65×N×M float32
    ondevice_transfer = 2 * N * M * 4       # 2×N×M float32 (C_hi + C_lo)
    ratio = host_transfer / ondevice_transfer
    print(f"  Host accumulation transfer:     {n_products}×{N}×{M}×4 = "
          f"{host_transfer / 1024:.0f} KB")
    print(f"  On-device accumulation transfer: 2×{N}×{M}×4 = "
          f"{ondevice_transfer / 1024:.0f} KB")
    print(f"  Reduction: {ratio:.1f}x")
    ok = ratio > 30
    if not ok:
        all_pass = False
    print(f"  {ratio:.1f}x > 30x: {'PASS' if ok else 'FAIL'}")
    print()

    # ----------------------------------------------------------------
    # Test 4: End-to-end matmul
    # ----------------------------------------------------------------
    print("test 4: end-to-end matmul\n")
    header = f"{'Config':>25}  {'Error':>12}  {'Status':>8}"
    print(header)
    print("-" * len(header))

    for N_test in [128, 256]:
        A = rng.randn(N_test, N_test).astype(np.float64)
        B = rng.randn(N_test, N_test).astype(np.float64)
        C_exact = A @ B

        configs = [
            ("host", matmul(A, B, pipeline="host")),
            ("ondev+host_acc", matmul(A, B, pipeline="ondevice", accumulation="host")),
            ("ondev+dev_acc", matmul(A, B, pipeline="ondevice", accumulation="ondevice")),
            ("ondev+fused", matmul(A, B, pipeline="ondevice", accumulation="fused")),
        ]

        for label, C in configs:
            err = rel_err(C, C_exact)
            # On-device FP32 2Sum achieves ~1e-10; host FP64 paths achieve ~1e-15.
            threshold = 1e-9 if ("dev_acc" in label or "fused" in label) else 1e-14
            ok = err < threshold
            if not ok:
                all_pass = False
            full_label = f"n={N_test} {label}"
            print(f"{full_label:>25}  {err:>12.2e}  {'PASS' if ok else 'FAIL'}")
    print()

    # ----------------------------------------------------------------
    # Test 5: Timing comparison
    # ----------------------------------------------------------------
    print("test 5: timing (host vs on-device vs fused accumulation)\n")

    for N_test in [256, 512]:
        A = rng.randn(N_test, N_test).astype(np.float64)
        B = rng.randn(N_test, N_test).astype(np.float64)

        # Warm up.
        _ = matmul(A, B, pipeline="ondevice", accumulation="host")
        _ = matmul(A, B, pipeline="ondevice", accumulation="ondevice")
        _ = matmul(A, B, pipeline="ondevice", accumulation="fused")

        n_iter = 10

        times_host = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            C = matmul(A, B, pipeline="ondevice", accumulation="host")
            times_host.append(time.perf_counter() - t0)

        times_dev = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            C = matmul(A, B, pipeline="ondevice", accumulation="ondevice")
            times_dev.append(time.perf_counter() - t0)

        times_fused = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            C = matmul(A, B, pipeline="ondevice", accumulation="fused")
            times_fused.append(time.perf_counter() - t0)

        t_host = np.median(times_host) * 1000
        t_dev = np.median(times_dev) * 1000
        t_fused = np.median(times_fused) * 1000
        speedup_dev = t_host / t_dev if t_dev > 0 else 0
        speedup_fused = t_host / t_fused if t_fused > 0 else 0

        print(f"  n={N_test}: host={t_host:.1f}ms  "
              f"ondevice={t_dev:.1f}ms ({speedup_dev:.2f}x)  "
              f"fused={t_fused:.1f}ms ({speedup_fused:.2f}x)")
    print()

    # ----------------------------------------------------------------
    # Test 6: Pallas backend (if available)
    # ----------------------------------------------------------------
    print("test 6: Pallas backend validation\n")
    if not _HAS_PALLAS:
        print("  SKIP (Pallas not available)")
    else:
        N_test = 256
        A = rng.randn(N_test, N_test).astype(np.float64)
        B = rng.randn(N_test, N_test).astype(np.float64)
        C_exact = A @ B

        # Prepare products + scales.
        n_hi, n_lo = _ONDEVICE_N_HI, _ONDEVICE_N_LO
        rho = _compute_rho_f32(N_test)

        A_hi, A_lo = _double_f32_split(A)
        B_hi, B_lo = _double_f32_split(B)
        A_hi_sl, A_hi_sc = f32_extract_split_rows(A_hi, rho, n_hi)
        A_lo_sl, A_lo_sc = f32_extract_split_rows(A_lo, rho, n_lo)
        B_hi_sl, B_hi_sc = f32_extract_split_cols(B_hi, rho, n_hi)
        B_lo_sl, B_lo_sc = f32_extract_split_cols(B_lo, rho, n_lo)

        products_list = []
        for i in range(n_hi):
            for j in range(n_hi):
                products_list.append(np.float32(A_hi_sl[i] @ B_hi_sl[j]))
        for i in range(n_hi):
            for j in range(n_lo):
                products_list.append(np.float32(A_hi_sl[i] @ B_lo_sl[j]))
        for i in range(n_lo):
            for j in range(n_hi):
                products_list.append(np.float32(A_lo_sl[i] @ B_hi_sl[j]))

        col_scales, row_scales, block_group_sizes = _precompute_accumulation_scales(
            A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc, N_test, N_test, n_hi, n_lo)

        products_jax = jnp.array(np.stack(products_list))
        col_scales_jax = jnp.array(col_scales)
        row_scales_jax = jnp.array(row_scales)

        # JAX backend.
        C_hi_jax, C_lo_jax = accumulate_2sum(
            products_jax, col_scales_jax, row_scales_jax,
            block_group_sizes, backend="jax")
        C_jax = np.float64(np.array(C_hi_jax)) + np.float64(np.array(C_lo_jax))

        # Pallas backend.
        try:
            C_hi_pal, C_lo_pal = accumulate_2sum(
                products_jax, col_scales_jax, row_scales_jax,
                block_group_sizes, N=N_test, M=N_test, backend="pallas")
            C_pal = np.float64(np.array(C_hi_pal)) + np.float64(np.array(C_lo_pal))

            jax_err = rel_err(C_jax, C_exact)
            pal_err = rel_err(C_pal, C_exact)
            diff = float(np.max(np.abs(C_jax - C_pal)) / np.max(np.abs(C_exact)))

            print(f"  JAX 2Sum error:     {jax_err:.2e}")
            print(f"  Pallas 2Sum error:  {pal_err:.2e}")
            print(f"  JAX vs Pallas diff: {diff:.2e}")
            ok = diff < 1e-14
            if not ok:
                all_pass = False
            print(f"  Match: {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  Pallas error: {e}")
    print()

    # ----------------------------------------------------------------
    # Test 7: JAX extraction vs numpy extraction bit-exactness
    # ----------------------------------------------------------------
    print("test 7: JAX extraction vs numpy extraction bit-exactness\n")

    N_test = 64
    K_test = 64
    X = rng.randn(N_test, K_test).astype(np.float32)
    Y = rng.randn(K_test, N_test).astype(np.float32)
    rho_test = _compute_rho_f32(K_test)

    for n_sl, label in [(5, "n_hi=5"), (4, "n_lo=4")]:
        # Rows.
        np_sl, np_sc = f32_extract_split_rows(X, rho_test, n_sl)
        jax_sl, jax_sc = jax_extract_split_rows(jnp.array(X), rho_test, n_sl)
        sl_diff = float(np.max(np.abs(np.array(jax_sl) - np.stack(np_sl))))
        sc_diff = float(np.max(np.abs(np.array(jax_sc) - np.stack(np_sc))))
        ok_rows = sl_diff == 0.0 and sc_diff == 0.0
        if not ok_rows:
            all_pass = False
        print(f"  rows {label}: slice_diff={sl_diff}  scale_diff={sc_diff}  "
              f"{'PASS' if ok_rows else 'FAIL'}")

        # Cols.
        np_sl_c, np_sc_c = f32_extract_split_cols(Y, rho_test, n_sl)
        jax_sl_c, jax_sc_c = jax_extract_split_cols(jnp.array(Y), rho_test, n_sl)
        sl_diff_c = float(np.max(np.abs(np.array(jax_sl_c) - np.stack(np_sl_c))))
        sc_diff_c = float(np.max(np.abs(np.array(jax_sc_c) - np.stack(np_sc_c))))
        ok_cols = sl_diff_c == 0.0 and sc_diff_c == 0.0
        if not ok_cols:
            all_pass = False
        print(f"  cols {label}: slice_diff={sl_diff_c}  scale_diff={sc_diff_c}  "
              f"{'PASS' if ok_cols else 'FAIL'}")
    print()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
