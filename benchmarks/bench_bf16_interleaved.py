"""Benchmark BF16 interleaved pipeline vs stock (broadcast) pipeline on TPU.

The BF16 interleaved pipeline casts extracted slices to BF16 before matmul,
using preferred_element_type=float32 for BF16->FP32 MXU output. Each GEMM
result is immediately 2Sum-accumulated, keeping only ONE (N,M) product alive.

This combines two potential optimizations:
1. BF16 MXU path (higher throughput than FP32 matmul)
2. Interleaved accumulation (no (24, N, M) intermediate in HBM)
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.extract import _compute_rho_f32, jax_extract_split_rows, jax_extract_split_cols
from ozaki_jax.matmul import (
    _jax_double_f32_split,
    _fused_pipeline_logic,
    _interleaved_pipeline_logic,
)

N_HI, N_LO = 4, 1
BGS = (tuple([N_HI] * N_HI), tuple([N_LO] * N_HI), tuple([N_HI] * N_LO))


def _bf16_interleaved_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo):
    """Interleaved pipeline with BF16 casts for MXU throughput.

    Same structure as _interleaved_pipeline_logic but casts extracted
    slices to BF16 before matmul, using preferred_element_type=float32.
    """
    # Phase 1: Extraction.
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)

    if n_lo > 0:
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

    # Cast slices to BF16 for MXU throughput.
    A_hi_sl_bf = jnp.bfloat16(A_hi_sl)
    B_hi_sl_bf = jnp.bfloat16(B_hi_sl)
    if n_lo > 0:
        A_lo_sl_bf = jnp.bfloat16(A_lo_sl)
        B_lo_sl_bf = jnp.bfloat16(B_lo_sl)

    # Pre-compute scale vectors (FP32 power-of-2).
    A_hi_sc_pow = jnp.exp2(A_hi_sc)
    B_hi_sc_pow = jnp.exp2(B_hi_sc)
    if n_lo > 0:
        A_lo_sc_pow = jnp.exp2(A_lo_sc)
        B_lo_sc_pow = jnp.exp2(B_lo_sc)

    N = A_hi.shape[0]
    M = B_hi.shape[1]

    def twosum_add(s_hi, s_lo, x):
        t = s_hi + x
        e = (s_hi - t) + x
        return t, s_lo + e

    def do_dot(a, b):
        return jnp.matmul(a, b, preferred_element_type=jnp.float32)

    # Block 1: hi x hi
    blk_hh_hi = jnp.zeros((N, M), dtype=jnp.float32)
    blk_hh_lo = jnp.zeros((N, M), dtype=jnp.float32)
    for i in range(n_hi):
        inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
        inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for j in range(n_hi):
            product = do_dot(A_hi_sl_bf[i], B_hi_sl_bf[j])
            scaled = product * B_hi_sc_pow[j][jnp.newaxis, :]
            inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
        inner_hi = inner_hi * A_hi_sc_pow[i][:, jnp.newaxis]
        inner_lo = inner_lo * A_hi_sc_pow[i][:, jnp.newaxis]
        blk_hh_hi, blk_hh_lo = twosum_add(blk_hh_hi, blk_hh_lo, inner_hi)
        blk_hh_hi, blk_hh_lo = twosum_add(blk_hh_hi, blk_hh_lo, inner_lo)

    if n_lo > 0:
        # Block 2: hi x lo
        blk_hl_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_hl_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for i in range(n_hi):
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
            for j in range(n_lo):
                product = do_dot(A_hi_sl_bf[i], B_lo_sl_bf[j])
                scaled = product * B_lo_sc_pow[j][jnp.newaxis, :]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
            inner_hi = inner_hi * A_hi_sc_pow[i][:, jnp.newaxis]
            inner_lo = inner_lo * A_hi_sc_pow[i][:, jnp.newaxis]
            blk_hl_hi, blk_hl_lo = twosum_add(blk_hl_hi, blk_hl_lo, inner_hi)
            blk_hl_hi, blk_hl_lo = twosum_add(blk_hl_hi, blk_hl_lo, inner_lo)

        # Block 3: lo x hi
        blk_lh_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_lh_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for i in range(n_lo):
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
            for j in range(n_hi):
                product = do_dot(A_lo_sl_bf[i], B_hi_sl_bf[j])
                scaled = product * B_hi_sc_pow[j][jnp.newaxis, :]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
            inner_hi = inner_hi * A_lo_sc_pow[i][:, jnp.newaxis]
            inner_lo = inner_lo * A_lo_sc_pow[i][:, jnp.newaxis]
            blk_lh_hi, blk_lh_lo = twosum_add(blk_lh_hi, blk_lh_lo, inner_hi)
            blk_lh_hi, blk_lh_lo = twosum_add(blk_lh_hi, blk_lh_lo, inner_lo)

    # Final combine.
    C_hi = jnp.zeros((N, M), dtype=jnp.float32)
    C_lo = jnp.zeros((N, M), dtype=jnp.float32)
    C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hh_hi)
    if n_lo > 0:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hl_hi)
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lh_hi)
    C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hh_lo)
    if n_lo > 0:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hl_lo)
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lh_lo)

    return C_hi, C_lo


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def stock_fn(A, B, rho, n_hi, n_lo, bgs):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _fused_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo, bgs)
    return jnp.float64(Ch) + jnp.float64(Cl)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def interleaved_f32_fn(A, B, rho, n_hi, n_lo):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _interleaved_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo)
    return jnp.float64(Ch) + jnp.float64(Cl)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def interleaved_bf16_fn(A, B, rho, n_hi, n_lo):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _bf16_interleaved_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo)
    return jnp.float64(Ch) + jnp.float64(Cl)


def time_fn(fn, *args, warmup=3, timed=15):
    for _ in range(warmup):
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), out)
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree.map(lambda x: x.block_until_ready(), out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000, out


def main():
    print(f"Device: {jax.devices()[0].device_kind}")
    print(f"Preset: high (n_hi={N_HI}, n_lo={N_LO}, 24 GEMMs)")
    print()

    header = (f"{'n':>6}  {'stock_ms':>10}  {'f32_ms':>10}  {'bf16_ms':>10}  "
              f"{'bf16/stock':>10}  {'stock_dig':>10}  {'f32_dig':>10}  "
              f"{'bf16_dig':>10}")
    print(header)
    print("-" * len(header))

    for n in [512, 1024, 2048, 4096, 8192]:
        rng = np.random.RandomState(42)
        A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)
        C_ref = np.asarray(A) @ np.asarray(B)

        # Stock broadcast pipeline
        t_stock, C_stock = time_fn(stock_fn, A, B, rho, N_HI, N_LO, BGS)
        err_s = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_s = -np.log10(max(err_s, 1e-16))

        # FP32 interleaved
        t_f32, C_f32 = time_fn(interleaved_f32_fn, A, B, rho, N_HI, N_LO)
        err_f32 = float(np.max(np.abs(np.asarray(C_f32) - C_ref))
                        / np.max(np.abs(C_ref)))
        dig_f32 = -np.log10(max(err_f32, 1e-16))

        # BF16 interleaved
        t_bf16, C_bf16 = time_fn(interleaved_bf16_fn, A, B, rho, N_HI, N_LO)
        err_bf16 = float(np.max(np.abs(np.asarray(C_bf16) - C_ref))
                         / np.max(np.abs(C_ref)))
        dig_bf16 = -np.log10(max(err_bf16, 1e-16))

        speedup = t_stock / t_bf16

        print(f"{n:>6}  {t_stock:>10.2f}  {t_f32:>10.2f}  {t_bf16:>10.2f}  "
              f"{speedup:>9.2f}x  {dig_s:>10.1f}  {dig_f32:>10.1f}  "
              f"{dig_bf16:>10.1f}")


if __name__ == "__main__":
    main()
