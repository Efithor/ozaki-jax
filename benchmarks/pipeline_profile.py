"""Profile where time is spent in the host vs on-device pipelines.

Breaks each pipeline into three stages:
  1. Extraction (splitting matrices into BF16-exact slices)
  2. GEMMs (matrix multiplications)
  3. Accumulation (rescaling and summing in FP64)

This is the core measurement for the on-device pipeline's thesis:
the host pipeline is extraction-bottlenecked, and FP32 extraction
eliminates that bottleneck.

Usage:
    python benchmarks/pipeline_profile.py
"""

import sys
import time
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
from ozaki_jax.matmul import (
    _double_f32_split, _ozaki_gemms_jit, _ondevice_gemms_jit,
    _accumulate_products, _accumulate_block_products,
    _ONDEVICE_N_HI, _ONDEVICE_N_LO,
)


def profile_host_pipeline(A_f64, B_f64, n_slices=8):
    """Profile the host pipeline, returning per-stage times."""
    N, K = A_f64.shape
    M = B_f64.shape[1]
    rho = _compute_rho(K)

    # Stage 1: extraction.
    t0 = time.perf_counter()
    A_slices, A_scales = extract_split_rows(A_f64, rho, n_slices)
    B_slices, B_scales = extract_split_cols(B_f64, rho, n_slices)
    t_extract = time.perf_counter() - t0

    # Stage 2: GEMMs.
    A_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_slices])
    B_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_slices])

    t0 = time.perf_counter()
    products = _ozaki_gemms_jit(A_stack, B_stack, n_slices)
    products.block_until_ready()
    t_gemms = time.perf_counter() - t0

    # Stage 3: accumulation.
    products_np = np.array(products, dtype=np.float64)
    t0 = time.perf_counter()
    C = _accumulate_products(products_np, A_scales, B_scales, N, M, n_slices)
    t_accum = time.perf_counter() - t0

    return {"extract": t_extract, "gemms": t_gemms, "accum": t_accum,
            "total": t_extract + t_gemms + t_accum, "result": C}


def profile_ondevice_pipeline(A_f64, B_f64):
    """Profile the on-device pipeline, returning per-stage times."""
    N = A_f64.shape[0]
    K = A_f64.shape[1]
    M = B_f64.shape[1]
    rho = _compute_rho_f32(K)
    n_hi = _ONDEVICE_N_HI
    n_lo = _ONDEVICE_N_LO

    # Stage 1a: double-FP32 split.
    t0 = time.perf_counter()
    A_hi, A_lo = _double_f32_split(A_f64)
    B_hi, B_lo = _double_f32_split(B_f64)
    t_split = time.perf_counter() - t0

    # Stage 1b: FP32 extraction.
    t0 = time.perf_counter()
    A_hi_slices, A_hi_scales = f32_extract_split_rows(A_hi, rho, n_hi)
    A_lo_slices, A_lo_scales = f32_extract_split_rows(A_lo, rho, n_lo)
    B_hi_slices, B_hi_scales = f32_extract_split_cols(B_hi, rho, n_hi)
    B_lo_slices, B_lo_scales = f32_extract_split_cols(B_lo, rho, n_lo)
    t_extract = time.perf_counter() - t0

    # Stage 2: GEMMs.
    A_hi_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_hi_slices])
    A_lo_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_lo_slices])
    B_hi_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_hi_slices])
    B_lo_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_lo_slices])

    t0 = time.perf_counter()
    products = _ondevice_gemms_jit(A_hi_stack, A_lo_stack,
                                   B_hi_stack, B_lo_stack,
                                   n_hi, n_lo, n_hi)
    products.block_until_ready()
    t_gemms = time.perf_counter() - t0

    # Stage 3: accumulation.
    products_np = np.array(products, dtype=np.float64)
    t0 = time.perf_counter()
    C = _accumulate_block_products(products_np, A_hi_scales, A_lo_scales,
                                   B_hi_scales, B_lo_scales, N, M,
                                   n_hi, n_lo, n_hi)
    t_accum = time.perf_counter() - t0

    total_extract = t_split + t_extract
    return {"split": t_split, "extract": t_extract,
            "total_extract": total_extract,
            "gemms": t_gemms, "accum": t_accum,
            "total": total_extract + t_gemms + t_accum, "result": C}


def main():
    print("ozaki-jax pipeline profiler")
    print(f"JAX {jax.__version__}, platform: {jax.default_backend()}")
    print()

    rng = np.random.RandomState(42)
    sizes = [128, 256, 512]
    n_warmup = 2
    n_measure = 5

    # Warm up JIT for all sizes.
    print("warming up JIT...", end=" ", flush=True)
    for n in sizes:
        A = rng.randn(n, n).astype(np.float64)
        B = rng.randn(n, n).astype(np.float64)
        for _ in range(n_warmup):
            profile_host_pipeline(A, B)
            profile_ondevice_pipeline(A, B)
    print("done.\n")

    # Profile each size.
    for n in sizes:
        A = rng.randn(n, n).astype(np.float64)
        B = rng.randn(n, n).astype(np.float64)

        host_runs = [profile_host_pipeline(A, B) for _ in range(n_measure)]
        ondev_runs = [profile_ondevice_pipeline(A, B) for _ in range(n_measure)]

        def median(runs, key):
            return np.median([r[key] for r in runs]) * 1000

        h_ext = median(host_runs, "extract")
        h_gem = median(host_runs, "gemms")
        h_acc = median(host_runs, "accum")
        h_tot = median(host_runs, "total")

        o_spl = median(ondev_runs, "split")
        o_ext = median(ondev_runs, "extract")
        o_tex = median(ondev_runs, "total_extract")
        o_gem = median(ondev_runs, "gemms")
        o_acc = median(ondev_runs, "accum")
        o_tot = median(ondev_runs, "total")

        print(f"n={n}")
        print(f"  {'Stage':<20} {'Host (ms)':>10} {'OnDevice (ms)':>14} {'Speedup':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*8}")
        print(f"  {'Extraction':<20} {h_ext:>10.2f} {o_tex:>14.2f} {h_ext/o_tex:>7.1f}x")
        print(f"    {'(FP32 split)':<18} {'-':>10} {o_spl:>14.2f}")
        print(f"    {'(sigma trick)':<18} {'-':>10} {o_ext:>14.2f}")
        print(f"  {'GEMMs':<20} {h_gem:>10.2f} {o_gem:>14.2f} {h_gem/o_gem if o_gem > 0 else 0:>7.1f}x")
        print(f"  {'Accumulation':<20} {h_acc:>10.2f} {o_acc:>14.2f} {h_acc/o_acc if o_acc > 0 else 0:>7.1f}x")
        print(f"  {'Total':<20} {h_tot:>10.2f} {o_tot:>14.2f} {h_tot/o_tot:>7.1f}x")

        # What fraction of time is extraction?
        print(f"  Host extraction fraction:     {h_ext/h_tot*100:.0f}%")
        print(f"  OnDevice extraction fraction: {o_tex/o_tot*100:.0f}%")
        print()


if __name__ == "__main__":
    main()
