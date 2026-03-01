"""Diagnose the ~3x accuracy gap between host and on-device pipelines.

Isolates error sources by building hybrid pipelines that swap one component
at a time:

  Experiment 1: Split isolation
    - Run HOST extraction on double-FP32 reconstructed input
    - Isolates: does losing 5 bits in the split cause the gap?

  Experiment 2: Extraction isolation
    - Run FP64 extraction on hi/lo separately, use all-pairs accumulation
    - Isolates: does FP32 sigma trick add error beyond the split?

  Experiment 3: GEMM structure isolation
    - Run on-device extraction, but accumulate with triangular pairing
      within the hi×hi block
    - Isolates: does all-pairs vs triangular pairing matter?

  Experiment 4: Slice count sweep
    - Vary n_hi and n_lo independently
    - Shows the error floor as a function of slice budget

  Experiment 5: Per-element error analysis
    - Look at error distribution, not just Frobenius norm
    - Shows whether error is uniform or concentrated

Usage:
    python benchmarks/accuracy_diagnosis.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax import matmul, matmul_numpy
from ozaki_jax.extract import (
    _compute_rho, _compute_rho_f32,
    extract_split_rows, extract_split_cols,
    f32_extract_split_rows, f32_extract_split_cols,
)
from ozaki_jax.matmul import (
    _double_f32_split, _accumulate_products, _accumulate_block_products,
)


def rel_err(C, C_ref):
    return np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)


def main():
    rng = np.random.RandomState(42)
    n = 256
    n_trials = 5

    print("accuracy diagnosis: isolating the ~3x gap\n")

    # ================================================================
    # Experiment 1: Is the double-FP32 split the cause?
    # ================================================================
    print("experiment 1: split isolation")
    print("  Run HOST pipeline on double-FP32 reconstructed input.\n")

    errs_host = []
    errs_split_host = []
    errs_ondev = []

    for _ in range(n_trials):
        scale = 10.0 ** rng.uniform(-2, 2)
        A = rng.randn(n, n).astype(np.float64) * scale
        B = rng.randn(n, n).astype(np.float64) * scale
        C_ref = A @ B

        # Standard host pipeline.
        C_host = matmul_numpy(A, B, pipeline="host")
        errs_host.append(rel_err(C_host, C_ref))

        # Double-FP32 split, then reconstruct to FP64, then host pipeline.
        A_hi, A_lo = _double_f32_split(A)
        B_hi, B_lo = _double_f32_split(B)
        A_recon = np.float64(A_hi) + np.float64(A_lo)
        B_recon = np.float64(B_hi) + np.float64(B_lo)
        C_split_host = matmul_numpy(A_recon, B_recon, pipeline="host")
        errs_split_host.append(rel_err(C_split_host, C_ref))

        # Standard on-device pipeline.
        C_ondev = matmul_numpy(A, B, pipeline="ondevice")
        errs_ondev.append(rel_err(C_ondev, C_ref))

    print(f"  host(original):     {np.mean(errs_host):.2e}")
    print(f"  host(split+recon):  {np.mean(errs_split_host):.2e}")
    print(f"  ondevice:           {np.mean(errs_ondev):.2e}")
    ratio_split = np.mean(errs_split_host) / np.mean(errs_host)
    ratio_ondev = np.mean(errs_ondev) / np.mean(errs_host)
    print(f"  split+recon ratio:  {ratio_split:.1f}x worse than host")
    print(f"  ondevice ratio:     {ratio_ondev:.1f}x worse than host")
    if ratio_split > 1.5:
        print("  → Split IS a significant contributor.")
    else:
        print("  → Split is NOT the main cause.")
    print()

    # ================================================================
    # Experiment 2: Is FP32 extraction adding error?
    # ================================================================
    print("experiment 2: extraction isolation")
    print("  Use FP64 extraction on hi/lo parts (bypass FP32 sigma trick).\n")

    errs_f64_on_split = []

    rho64 = _compute_rho(n)

    for _ in range(n_trials):
        scale = 10.0 ** rng.uniform(-2, 2)
        A = rng.randn(n, n).astype(np.float64) * scale
        B = rng.randn(n, n).astype(np.float64) * scale
        C_ref = A @ B

        # Split to FP32, promote back to FP64, then use FP64 extraction.
        A_hi, A_lo = _double_f32_split(A)
        B_hi, B_lo = _double_f32_split(B)

        # FP64-extract the hi and lo parts separately.
        A_hi_sl, A_hi_sc = extract_split_rows(np.float64(A_hi), rho64, 5)
        A_lo_sl, A_lo_sc = extract_split_rows(np.float64(A_lo), rho64, 4)
        B_hi_sl, B_hi_sc = extract_split_cols(np.float64(B_hi), rho64, 5)
        B_lo_sl, B_lo_sc = extract_split_cols(np.float64(B_lo), rho64, 4)

        # All-pairs GEMMs in numpy, same structure as on-device.
        products = []
        n_hi, n_lo = 5, 4
        for i in range(n_hi):
            for j in range(n_hi):
                products.append(np.float32(
                    A_hi_sl[i].astype(np.float32) @ B_hi_sl[j].astype(np.float32)))
        for i in range(n_hi):
            for j in range(n_lo):
                products.append(np.float32(
                    A_hi_sl[i].astype(np.float32) @ B_lo_sl[j].astype(np.float32)))
        for i in range(n_lo):
            for j in range(n_hi):
                products.append(np.float32(
                    A_lo_sl[i].astype(np.float32) @ B_hi_sl[j].astype(np.float32)))

        C = _accumulate_block_products(
            np.stack(products), A_hi_sc, A_lo_sc,
            B_hi_sc, B_lo_sc, n, n, n_hi, n_lo, n_hi)
        errs_f64_on_split.append(rel_err(C, C_ref))

    print(f"  FP64-extract on split:  {np.mean(errs_f64_on_split):.2e}")
    print(f"  FP32-extract on split:  {np.mean(errs_ondev):.2e}")
    ratio_ext = np.mean(errs_ondev) / np.mean(errs_f64_on_split)
    print(f"  FP32 vs FP64 extract:   {ratio_ext:.2f}x")
    if ratio_ext > 1.5:
        print("  → FP32 extraction IS adding error beyond the split.")
    else:
        print("  → FP32 extraction is NOT adding significant extra error.")
    print()

    # ================================================================
    # Experiment 3: All-pairs vs triangular within hi×hi
    # ================================================================
    print("experiment 3: GEMM structure")
    print("  Compare all-pairs vs triangular within hi×hi block.\n")

    errs_allpairs = []
    errs_triangular = []

    rho32 = _compute_rho_f32(n)

    for _ in range(n_trials):
        scale = 10.0 ** rng.uniform(-2, 2)
        A = rng.randn(n, n).astype(np.float64) * scale
        B = rng.randn(n, n).astype(np.float64) * scale
        C_ref = A @ B

        A_hi, A_lo = _double_f32_split(A)
        B_hi, B_lo = _double_f32_split(B)

        A_hi_sl, A_hi_sc = f32_extract_split_rows(A_hi, rho32, 5)
        A_lo_sl, A_lo_sc = f32_extract_split_rows(A_lo, rho32, 4)
        B_hi_sl, B_hi_sc = f32_extract_split_cols(B_hi, rho32, 5)
        B_lo_sl, B_lo_sc = f32_extract_split_cols(B_lo, rho32, 4)

        def gemm(a, b):
            return np.float32(a.astype(np.float32) @ b.astype(np.float32))

        # All-pairs (standard on-device).
        C_ap = np.zeros((n, n), dtype=np.float64)
        for i in range(5):
            for j in range(5):
                P = gemm(A_hi_sl[i], B_hi_sl[j]).astype(np.float64)
                rs = np.ldexp(np.ones(n, np.float64), A_hi_sc[i].astype(np.int64))
                cs = np.ldexp(np.ones(n, np.float64), B_hi_sc[j].astype(np.int64))
                C_ap += P * rs[:, None] * cs[None, :]
        for i in range(5):
            for j in range(4):
                P = gemm(A_hi_sl[i], B_lo_sl[j]).astype(np.float64)
                rs = np.ldexp(np.ones(n, np.float64), A_hi_sc[i].astype(np.int64))
                cs = np.ldexp(np.ones(n, np.float64), B_lo_sc[j].astype(np.int64))
                C_ap += P * rs[:, None] * cs[None, :]
        for i in range(4):
            for j in range(5):
                P = gemm(A_lo_sl[i], B_hi_sl[j]).astype(np.float64)
                rs = np.ldexp(np.ones(n, np.float64), A_lo_sc[i].astype(np.int64))
                cs = np.ldexp(np.ones(n, np.float64), B_hi_sc[j].astype(np.int64))
                C_ap += P * rs[:, None] * cs[None, :]
        errs_allpairs.append(rel_err(C_ap, C_ref))

        # Triangular within hi×hi (i+j <= 4), all-pairs for cross blocks.
        C_tr = np.zeros((n, n), dtype=np.float64)
        tri_gemms = 0
        for i in range(5):
            for j in range(5):
                if i + j > 4:
                    continue
                tri_gemms += 1
                P = gemm(A_hi_sl[i], B_hi_sl[j]).astype(np.float64)
                rs = np.ldexp(np.ones(n, np.float64), A_hi_sc[i].astype(np.int64))
                cs = np.ldexp(np.ones(n, np.float64), B_hi_sc[j].astype(np.int64))
                C_tr += P * rs[:, None] * cs[None, :]
        for i in range(5):
            for j in range(4):
                P = gemm(A_hi_sl[i], B_lo_sl[j]).astype(np.float64)
                rs = np.ldexp(np.ones(n, np.float64), A_hi_sc[i].astype(np.int64))
                cs = np.ldexp(np.ones(n, np.float64), B_lo_sc[j].astype(np.int64))
                C_tr += P * rs[:, None] * cs[None, :]
        for i in range(4):
            for j in range(5):
                P = gemm(A_lo_sl[i], B_hi_sl[j]).astype(np.float64)
                rs = np.ldexp(np.ones(n, np.float64), A_lo_sc[i].astype(np.int64))
                cs = np.ldexp(np.ones(n, np.float64), B_hi_sc[j].astype(np.int64))
                C_tr += P * rs[:, None] * cs[None, :]
        errs_triangular.append(rel_err(C_tr, C_ref))

    print(f"  All-pairs (25+20+20=65): {np.mean(errs_allpairs):.2e}")
    print(f"  Triangular hi×hi ({tri_gemms}+20+20={tri_gemms+40}): "
          f"{np.mean(errs_triangular):.2e}")
    ratio_struct = np.mean(errs_triangular) / np.mean(errs_allpairs)
    print(f"  Triangular / all-pairs: {ratio_struct:.2f}x")
    print()

    # ================================================================
    # Experiment 4: Error floor vs slice count
    # ================================================================
    print("experiment 4: error floor vs slice count")
    print("  Sweep n_hi from 4 to 8, n_lo from 0 to 6.\n")

    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)
    C_ref = A @ B
    ref_norm = np.linalg.norm(C_ref)

    A_hi, A_lo = _double_f32_split(A)
    B_hi, B_lo = _double_f32_split(B)

    # Also compute "oracle" error: matmul on reconstructed input.
    A_recon = np.float64(A_hi) + np.float64(A_lo)
    B_recon = np.float64(B_hi) + np.float64(B_lo)
    C_oracle = A_recon @ B_recon
    oracle_err = rel_err(C_oracle, C_ref)
    print(f"  Oracle (recon @ recon, no extraction): {oracle_err:.2e}")
    print(f"  Host pipeline:                         {rel_err(matmul_numpy(A, B, pipeline='host'), C_ref):.2e}")
    print()

    print(f"  {'n_hi':>4} {'n_lo':>4} {'GEMMs':>6} {'Error':>12} {'vs Oracle':>10}")
    print(f"  {'-'*4} {'-'*4} {'-'*6} {'-'*12} {'-'*10}")

    for n_hi in [4, 5, 6, 7, 8]:
        for n_lo in [0, 2, 4, 6]:
            a_hi_sl, a_hi_sc = f32_extract_split_rows(A_hi, rho32, n_hi)
            b_hi_sl, b_hi_sc = f32_extract_split_cols(B_hi, rho32, n_hi)

            n_gemms = n_hi * n_hi
            C = np.zeros((n, n), dtype=np.float64)

            def gemm(a, b):
                return np.float32(a.astype(np.float32) @ b.astype(np.float32))

            # hi × hi
            for i in range(n_hi):
                for j in range(n_hi):
                    P = gemm(a_hi_sl[i], b_hi_sl[j]).astype(np.float64)
                    rs = np.ldexp(np.ones(n, np.float64), a_hi_sc[i].astype(np.int64))
                    cs = np.ldexp(np.ones(n, np.float64), b_hi_sc[j].astype(np.int64))
                    C += P * rs[:, None] * cs[None, :]

            if n_lo > 0:
                a_lo_sl, a_lo_sc = f32_extract_split_rows(A_lo, rho32, n_lo)
                b_lo_sl, b_lo_sc = f32_extract_split_cols(B_lo, rho32, n_lo)
                n_gemms += n_hi * n_lo + n_lo * n_hi

                for i in range(n_hi):
                    for j in range(n_lo):
                        P = gemm(a_hi_sl[i], b_lo_sl[j]).astype(np.float64)
                        rs = np.ldexp(np.ones(n, np.float64), a_hi_sc[i].astype(np.int64))
                        cs = np.ldexp(np.ones(n, np.float64), b_lo_sc[j].astype(np.int64))
                        C += P * rs[:, None] * cs[None, :]
                for i in range(n_lo):
                    for j in range(n_hi):
                        P = gemm(a_lo_sl[i], b_hi_sl[j]).astype(np.float64)
                        rs = np.ldexp(np.ones(n, np.float64), a_lo_sc[i].astype(np.int64))
                        cs = np.ldexp(np.ones(n, np.float64), b_hi_sc[j].astype(np.int64))
                        C += P * rs[:, None] * cs[None, :]

            err = rel_err(C, C_ref)
            vs_oracle = err / oracle_err if oracle_err > 0 else float("inf")
            marker = " ← current" if n_hi == 5 and n_lo == 4 else ""
            print(f"  {n_hi:>4} {n_lo:>4} {n_gemms:>6} {err:>12.2e} {vs_oracle:>9.1f}x{marker}")

    print()

    # ================================================================
    # Experiment 5: Per-element error distribution
    # ================================================================
    print("experiment 5: per-element error distribution\n")

    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)
    C_ref = A @ B
    peak = np.max(np.abs(C_ref))

    C_host = matmul_numpy(A, B, pipeline="host")
    C_ondev = matmul_numpy(A, B, pipeline="ondevice")

    err_host = np.abs(C_host - C_ref) / peak
    err_ondev = np.abs(C_ondev - C_ref) / peak

    for label, err in [("host", err_host), ("ondevice", err_ondev)]:
        print(f"  {label}:")
        print(f"    max:    {np.max(err):.2e}")
        print(f"    mean:   {np.mean(err):.2e}")
        print(f"    median: {np.median(err):.2e}")
        print(f"    p99:    {np.percentile(err, 99):.2e}")
        print(f"    p99.9:  {np.percentile(err, 99.9):.2e}")
    ratio_max = np.max(err_ondev) / np.max(err_host)
    ratio_mean = np.mean(err_ondev) / np.mean(err_host)
    print(f"  ondev/host ratio: max={ratio_max:.1f}x, mean={ratio_mean:.1f}x")
    print()

    # ================================================================
    # Experiment 6: Is the error from the split or from matmul structure?
    # ================================================================
    print("experiment 6: decompose error sources\n")
    print("  A@B = (A_hi+A_lo)@(B_hi+B_lo)")
    print("      = A_hi@B_hi + A_hi@B_lo + A_lo@B_hi + A_lo@B_lo\n")

    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)
    C_ref = A @ B

    A_hi, A_lo = _double_f32_split(A)
    B_hi, B_lo = _double_f32_split(B)

    # Each cross-product computed in FP64 (no extraction, no slicing).
    hh = np.float64(A_hi) @ np.float64(B_hi)
    hl = np.float64(A_hi) @ np.float64(B_lo)
    lh = np.float64(A_lo) @ np.float64(B_hi)
    ll = np.float64(A_lo) @ np.float64(B_lo)

    C_no_ll = hh + hl + lh
    C_with_ll = hh + hl + lh + ll

    err_no_ll = rel_err(C_no_ll, C_ref)
    err_with_ll = rel_err(C_with_ll, C_ref)

    print(f"  hh+hl+lh (skip ll): {err_no_ll:.2e}")
    print(f"  hh+hl+lh+ll (all):  {err_with_ll:.2e}")
    print(f"  ||ll|| / ||C_ref||: {np.linalg.norm(ll) / np.linalg.norm(C_ref):.2e}")
    print()

    # Now: how much error does the extraction+GEMM add on top of the split?
    C_ondev = matmul_numpy(A, B, pipeline="ondevice")
    err_ondev = rel_err(C_ondev, C_ref)
    err_extraction = rel_err(C_ondev, C_with_ll)

    print(f"  on-device vs C_ref:   {err_ondev:.2e}  (total error)")
    print(f"  on-device vs C_split: {err_extraction:.2e}  (extraction+GEMM error only)")
    print(f"  split error (C_split vs C_ref): {err_with_ll:.2e}")
    print()
    print(f"  Interpretation:")
    print(f"    split contributes:      {err_with_ll:.2e}")
    print(f"    extraction+GEMM adds:   {err_extraction:.2e}")
    print(f"    total on-device error:  {err_ondev:.2e}")


if __name__ == "__main__":
    main()
