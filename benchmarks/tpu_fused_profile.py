"""Profile fused pipeline stages on TPU.

Breaks down wall-clock time for each phase of the pipeline:
  1. Host: double_f32_split (CPU)
  2. Transfer: 4 FP32 matrices to device
  3. Device: extraction + GEMMs + scales + 2Sum (fused JIT)
  4. Transfer: 2 FP32 results back to host
  5. Host: FP64 combine

Also profiles the fused JIT internals by compiling separate JIT functions
for each phase (extraction, GEMMs, scale precompute, 2Sum accumulation)
to measure their individual device time.

Usage:
    PYTHONPATH=~/ozaki-jax python3 benchmarks/tpu_fused_profile.py
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from ozaki_jax import matmul
from ozaki_jax.extract import (
    _compute_rho_f32,
    jax_extract_split_rows,
    jax_extract_split_cols,
)
from ozaki_jax.matmul import (
    _double_f32_split, _fused_ondevice_jit,
    _ONDEVICE_N_HI, _ONDEVICE_N_LO,
)
from ozaki_jax.pallas_ops import _accumulate_2sum_logic


# ── Stage-isolated JIT functions for profiling ──────────────────────


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def _profile_extract(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo):
    """Just extraction."""
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)
    B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)
    return A_hi_sl, A_hi_sc, A_lo_sl, A_lo_sc, B_hi_sl, B_hi_sc, B_lo_sl, B_lo_sc


# For GEMMs profiling: take pre-extracted slices
@functools.partial(jax.jit, static_argnums=(4, 5))
def _profile_gemms(A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl, n_hi, n_lo):
    """Just GEMMs."""
    products = []
    for i in range(n_hi):
        for j in range(n_hi):
            products.append(jnp.dot(A_hi_sl[i], B_hi_sl[j]))
    for i in range(n_hi):
        for j in range(n_lo):
            products.append(jnp.dot(A_hi_sl[i], B_lo_sl[j]))
    for i in range(n_lo):
        for j in range(n_hi):
            products.append(jnp.dot(A_lo_sl[i], B_hi_sl[j]))
    return jnp.stack(products)


@functools.partial(jax.jit, static_argnums=(4, 5))
def _profile_scales(A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc, n_hi, n_lo):
    """Just scale precomputation."""
    col_scales = jnp.exp2(jnp.concatenate([
        jnp.tile(B_hi_sc, (n_hi, 1)),
        jnp.tile(B_lo_sc, (n_hi, 1)),
        jnp.tile(B_hi_sc, (n_lo, 1)),
    ], axis=0))

    row_scales = jnp.exp2(jnp.concatenate([
        A_hi_sc,
        A_hi_sc,
        A_lo_sc,
    ], axis=0))
    return col_scales, row_scales


@functools.partial(jax.jit, static_argnums=(3,))
def _profile_accumulate(products, col_scales, row_scales, block_group_sizes):
    """Just 2Sum accumulation."""
    return _accumulate_2sum_logic(products, col_scales, row_scales,
                                  block_group_sizes)


def time_fn(fn, *args, n_warmup=3, n_iter=20):
    """Time a function, blocking until device completion."""
    for _ in range(n_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    return np.median(times) * 1000, np.array(times) * 1000, result


def main():
    print("=" * 70)
    print("ozaki-jax fused pipeline stage profiling")
    print("=" * 70)
    print(f"JAX version:  {jax.__version__}")
    print(f"Platform:     {jax.default_backend()}")
    print(f"Devices:      {jax.devices()}")
    print()

    rng = np.random.RandomState(42)
    n_hi = _ONDEVICE_N_HI
    n_lo = _ONDEVICE_N_LO

    for N in [256, 512, 1024]:
        print(f"--- n={N} ---\n")

        A = rng.randn(N, N).astype(np.float64)
        B = rng.randn(N, N).astype(np.float64)
        rho = _compute_rho_f32(N)

        block_group_sizes = (
            tuple([n_hi] * n_hi),
            tuple([n_lo] * n_hi),
            tuple([n_hi] * n_lo),
        )

        # ── End-to-end pipeline breakdown ──────────────────────────

        # Phase 0: double_f32_split (CPU)
        times_split = []
        for _ in range(20):
            t0 = time.perf_counter()
            A_hi, A_lo = _double_f32_split(A)
            B_hi, B_lo = _double_f32_split(B)
            times_split.append(time.perf_counter() - t0)
        t_split = np.median(times_split) * 1000

        # Phase 1: transfer to device
        times_xfer_in = []
        for _ in range(20):
            t0 = time.perf_counter()
            A_hi_j = jax.device_put(jnp.array(A_hi))
            A_lo_j = jax.device_put(jnp.array(A_lo))
            B_hi_j = jax.device_put(jnp.array(B_hi))
            B_lo_j = jax.device_put(jnp.array(B_lo))
            jax.block_until_ready((A_hi_j, A_lo_j, B_hi_j, B_lo_j))
            times_xfer_in.append(time.perf_counter() - t0)
        t_xfer_in = np.median(times_xfer_in) * 1000

        # Phase 2: fused JIT (device)
        t_fused, _, (C_hi, C_lo) = time_fn(
            _fused_ondevice_jit,
            A_hi_j, A_lo_j, B_hi_j, B_lo_j,
            rho, n_hi, n_lo, block_group_sizes)

        # Phase 3: transfer back
        times_xfer_out = []
        for _ in range(20):
            t0 = time.perf_counter()
            c_hi_np = np.array(C_hi)
            c_lo_np = np.array(C_lo)
            times_xfer_out.append(time.perf_counter() - t0)
        t_xfer_out = np.median(times_xfer_out) * 1000

        # Phase 4: FP64 combine (CPU)
        times_combine = []
        for _ in range(20):
            t0 = time.perf_counter()
            C = np.float64(c_hi_np) + np.float64(c_lo_np)
            times_combine.append(time.perf_counter() - t0)
        t_combine = np.median(times_combine) * 1000

        total = t_split + t_xfer_in + t_fused + t_xfer_out + t_combine

        print("  end-to-end pipeline breakdown:")
        print(f"    double_f32_split (CPU):     {t_split:>8.2f}ms  ({t_split/total*100:>5.1f}%)")
        print(f"    transfer in (4 matrices):   {t_xfer_in:>8.2f}ms  ({t_xfer_in/total*100:>5.1f}%)")
        print(f"    fused JIT (device):         {t_fused:>8.2f}ms  ({t_fused/total*100:>5.1f}%)")
        print(f"    transfer out (2 matrices):  {t_xfer_out:>8.2f}ms  ({t_xfer_out/total*100:>5.1f}%)")
        print(f"    FP64 combine (CPU):         {t_combine:>8.2f}ms  ({t_combine/total*100:>5.1f}%)")
        print(f"    ────────────────────────────────────")
        print(f"    total:                      {total:>8.2f}ms")
        print()

        # ── Device-side stage breakdown ────────────────────────────

        # Extraction only
        t_extract, _, ext_result = time_fn(
            _profile_extract,
            A_hi_j, A_lo_j, B_hi_j, B_lo_j,
            rho, n_hi, n_lo)
        A_hi_sl, A_hi_sc, A_lo_sl, A_lo_sc, B_hi_sl, B_hi_sc, B_lo_sl, B_lo_sc = ext_result

        # GEMMs only
        t_gemms, _, products = time_fn(
            _profile_gemms,
            A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl,
            n_hi, n_lo)

        # Scale precomputation only
        t_scales, _, (col_scales, row_scales) = time_fn(
            _profile_scales,
            A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc,
            n_hi, n_lo)

        # 2Sum accumulation only
        t_accum, _, _ = time_fn(
            _profile_accumulate,
            products, col_scales, row_scales,
            block_group_sizes)

        device_total = t_extract + t_gemms + t_scales + t_accum

        print("  device-side stage breakdown (separate JITs):")
        print(f"    extraction:                 {t_extract:>8.2f}ms  ({t_extract/device_total*100:>5.1f}%)")
        print(f"    65 GEMMs:                   {t_gemms:>8.2f}ms  ({t_gemms/device_total*100:>5.1f}%)")
        print(f"    scale precompute:           {t_scales:>8.2f}ms  ({t_scales/device_total*100:>5.1f}%)")
        print(f"    2Sum accumulation:          {t_accum:>8.2f}ms  ({t_accum/device_total*100:>5.1f}%)")
        print(f"    ────────────────────────────────────")
        print(f"    sum of stages:              {device_total:>8.2f}ms")
        print(f"    actual fused JIT:           {t_fused:>8.2f}ms")
        print(f"    fusion overhead savings:    {device_total - t_fused:>8.2f}ms")
        print()

    print("done.")


if __name__ == "__main__":
    main()
