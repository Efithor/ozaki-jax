"""Benchmark BF16 broadcast pipeline vs FP32 broadcast (stock) on TPU.

Tests whether BF16 cast of extracted slices + broadcast matmul gives
speedup over stock FP32 broadcast matmul, while preserving accuracy.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.extract import _compute_rho_f32, jax_extract_split_rows, jax_extract_split_cols
from ozaki_jax.matmul import _jax_double_f32_split
from ozaki_jax.pallas_ops import _accumulate_2sum_logic

N_HI, N_LO = 4, 1
BGS = (tuple([N_HI] * N_HI), tuple([N_LO] * N_HI), tuple([N_HI] * N_LO))


def _bf16_broadcast_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo,
                                    block_group_sizes):
    """Broadcast pipeline with BF16 casts on extracted slices.

    Same structure as _fused_pipeline_logic but casts slices to BF16
    before broadcast matmul.
    """
    # Phase 1: Extraction.
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)

    # Cast to BF16.
    A_hi_sl_bf = jnp.bfloat16(A_hi_sl)
    B_hi_sl_bf = jnp.bfloat16(B_hi_sl)

    # Phase 2: Broadcast-batched GEMMs with BF16 inputs.
    hh = jnp.matmul(A_hi_sl_bf[:, None, :, :], B_hi_sl_bf[None, :, :, :],
                     preferred_element_type=jnp.float32)
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]

    if n_lo > 0:
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

        A_lo_sl_bf = jnp.bfloat16(A_lo_sl)
        B_lo_sl_bf = jnp.bfloat16(B_lo_sl)

        hl = jnp.matmul(A_hi_sl_bf[:, None, :, :], B_lo_sl_bf[None, :, :, :],
                         preferred_element_type=jnp.float32)
        parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
        lh = jnp.matmul(A_lo_sl_bf[:, None, :, :], B_hi_sl_bf[None, :, :, :],
                         preferred_element_type=jnp.float32)
        parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))

    products = jnp.concatenate(parts, axis=0)

    # Phase 3: Scale precomputation.
    col_parts = [jnp.tile(B_hi_sc, (n_hi, 1))]
    row_parts = [A_hi_sc]
    if n_lo > 0:
        col_parts.append(jnp.tile(B_lo_sc, (n_hi, 1)))
        col_parts.append(jnp.tile(B_hi_sc, (n_lo, 1)))
        row_parts.append(A_hi_sc)
        row_parts.append(A_lo_sc)

    col_scales = jnp.exp2(jnp.concatenate(col_parts, axis=0))
    row_scales = jnp.exp2(jnp.concatenate(row_parts, axis=0))

    # Phase 4: 2Sum accumulation.
    return _accumulate_2sum_logic(products, col_scales, row_scales,
                                  block_group_sizes)


def _stock_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo,
                           block_group_sizes):
    """Stock FP32 broadcast pipeline (reference)."""
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)

    hh = jnp.matmul(A_hi_sl[:, None, :, :], B_hi_sl[None, :, :, :])
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]

    if n_lo > 0:
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

        hl = jnp.matmul(A_hi_sl[:, None, :, :], B_lo_sl[None, :, :, :])
        parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
        lh = jnp.matmul(A_lo_sl[:, None, :, :], B_hi_sl[None, :, :, :])
        parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))

    products = jnp.concatenate(parts, axis=0)

    col_parts = [jnp.tile(B_hi_sc, (n_hi, 1))]
    row_parts = [A_hi_sc]
    if n_lo > 0:
        col_parts.append(jnp.tile(B_lo_sc, (n_hi, 1)))
        col_parts.append(jnp.tile(B_hi_sc, (n_lo, 1)))
        row_parts.append(A_hi_sc)
        row_parts.append(A_lo_sc)

    col_scales = jnp.exp2(jnp.concatenate(col_parts, axis=0))
    row_scales = jnp.exp2(jnp.concatenate(row_parts, axis=0))

    return _accumulate_2sum_logic(products, col_scales, row_scales,
                                  block_group_sizes)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def stock_fn(A, B, rho, n_hi, n_lo, bgs):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _stock_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo, bgs)
    return jnp.float64(Ch) + jnp.float64(Cl)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def bf16_broadcast_fn(A, B, rho, n_hi, n_lo, bgs):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _bf16_broadcast_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo, bgs)
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

    header = (f"{'n':>6}  {'stock_ms':>10}  {'bf16bc_ms':>10}  "
              f"{'speedup':>8}  {'stock_dig':>10}  {'bf16bc_dig':>11}")
    print(header)
    print("-" * len(header))

    for n in [512, 1024, 2048, 4096, 8192]:
        rng = np.random.RandomState(42)
        A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)
        C_ref = np.asarray(A) @ np.asarray(B)

        t_stock, C_stock = time_fn(stock_fn, A, B, rho, N_HI, N_LO, BGS)
        err_s = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_s = -np.log10(max(err_s, 1e-16))

        t_bf16, C_bf16 = time_fn(bf16_broadcast_fn, A, B, rho, N_HI, N_LO, BGS)
        err_bf16 = float(np.max(np.abs(np.asarray(C_bf16) - C_ref))
                         / np.max(np.abs(C_ref)))
        dig_bf16 = -np.log10(max(err_bf16, 1e-16))

        speedup = t_stock / t_bf16

        print(f"{n:>6}  {t_stock:>10.2f}  {t_bf16:>10.2f}  "
              f"{speedup:>7.2f}x  {dig_s:>10.1f}  {dig_bf16:>11.1f}")


if __name__ == "__main__":
    main()
