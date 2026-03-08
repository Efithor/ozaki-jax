"""Benchmark Pallas v2 kernel: 1 dot per grid step with scalar prefetch.

Previous kernel had 24 dots per K-step → MXU couldn't pipeline.
This version flattens (pair, K-tile) into a single reduction dimension,
giving ONE dot per grid step for proper DMA/MXU overlap.

Uses scalar prefetch to map each step to the right A/B slice indices.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ozaki_jax.extract import (
    _compute_rho_f32, jax_extract_split_rows, jax_extract_split_cols,
)
from ozaki_jax.matmul import _jax_double_f32_split, _fused_pipeline_logic
from ozaki_jax.pallas_gemm_accum import _make_pair_schedule


def _build_pallas_v2(n_hi, n_lo, N, K, M, bm, bk, bn):
    """Build the v2 Pallas kernel and return a callable.

    Returns a function f(a_slices, b_slices, a_scales_pow2, b_scales_pow2)
    that returns (C_hi, C_lo).
    """
    n_total = n_hi + n_lo
    nsteps = K // bk
    pair_a, pair_b, n_pairs, groups, block_boundaries = \
        _make_pair_schedule(n_hi, n_lo)

    total_steps = n_pairs * nsteps

    # Build prefetch index arrays: for each flattened step, which A/B slice
    # and K-tile to load.
    step_a_idx = []  # A slice index for each step
    step_b_idx = []  # B slice index for each step
    step_pair = []   # Which pair accumulator to write to
    step_k = []      # Which K-tile
    for p in range(n_pairs):
        for k in range(nsteps):
            step_a_idx.append(pair_a[p])
            step_b_idx.append(pair_b[p])
            step_pair.append(p)
            step_k.append(k)

    step_a_idx = jnp.array(step_a_idx, dtype=jnp.int32)
    step_b_idx = jnp.array(step_b_idx, dtype=jnp.int32)
    step_pair_arr = jnp.array(step_pair, dtype=jnp.int32)
    step_k_arr = jnp.array(step_k, dtype=jnp.int32)

    def kernel(step_a_ref, step_b_ref, step_pair_ref, step_k_ref,
               a_sl_ref, b_sl_ref, a_sc_ref, b_sc_ref,
               c_hi_ref, c_lo_ref,
               accum_ref):
        """One dot per grid step. Scalar prefetch drives slice selection."""
        s = pl.program_id(2)

        # Initialize accumulators on first step.
        @pl.when(s == 0)
        def _():
            accum_ref[...] = jnp.zeros_like(accum_ref)

        # Accumulate the single dot product for this step.
        p = step_pair_ref[s]
        accum_ref[p, :, :] += jnp.dot(
            a_sl_ref[0, :, :],
            b_sl_ref[0, :, :],
            preferred_element_type=jnp.float32,
        )

        # Finalize on last step: apply scales and 2Sum.
        @pl.when(s == total_steps - 1)
        def _():
            bm_loc = c_hi_ref.shape[0]
            bn_loc = c_hi_ref.shape[1]

            def twosum_add(s_hi, s_lo, x):
                t = jnp.float32(s_hi + x)
                e = jnp.float32(jnp.float32(s_hi - t) + x)
                return t, jnp.float32(s_lo + e)

            block_hi_list = []
            block_lo_list = []
            prev = 0

            for blk_end in block_boundaries:
                blk_hi = jnp.zeros((bm_loc, bn_loc), dtype=jnp.float32)
                blk_lo = jnp.zeros((bm_loc, bn_loc), dtype=jnp.float32)

                for g_idx in range(prev, blk_end):
                    a_idx, g_pairs = groups[g_idx]
                    inner_hi = jnp.zeros((bm_loc, bn_loc), dtype=jnp.float32)
                    inner_lo = jnp.zeros((bm_loc, bn_loc), dtype=jnp.float32)

                    for pp in g_pairs:
                        col_sc = b_sc_ref[pair_b[pp], :]
                        scaled = jnp.float32(
                            accum_ref[pp, :, :] * col_sc[jnp.newaxis, :])
                        inner_hi, inner_lo = twosum_add(
                            inner_hi, inner_lo, scaled)

                    row_sc = a_sc_ref[a_idx, :]
                    inner_hi = jnp.float32(
                        inner_hi * row_sc[:, jnp.newaxis])
                    inner_lo = jnp.float32(
                        inner_lo * row_sc[:, jnp.newaxis])

                    blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_hi)
                    blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_lo)

                block_hi_list.append(blk_hi)
                block_lo_list.append(blk_lo)
                prev = blk_end

            final_hi = jnp.zeros((bm_loc, bn_loc), dtype=jnp.float32)
            final_lo = jnp.zeros((bm_loc, bn_loc), dtype=jnp.float32)
            for bh in block_hi_list:
                final_hi, final_lo = twosum_add(final_hi, final_lo, bh)
            for bl in block_lo_list:
                final_hi, final_lo = twosum_add(final_hi, final_lo, bl)

            c_hi_ref[...] = final_hi
            c_lo_ref[...] = final_lo

    grid = (N // bm, M // bn, total_steps)

    # BlockSpec index maps use scalar prefetch refs to pick the right slices.
    # Scalar prefetch args come first in both index_map and kernel.
    def a_map(i, j, s, sa_ref, sb_ref, sp_ref, sk_ref):
        return (sa_ref[s], i, sk_ref[s])

    def b_map(i, j, s, sa_ref, sb_ref, sp_ref, sk_ref):
        return (sb_ref[s], sk_ref[s], j)

    def a_sc_map(i, j, s, sa_ref, sb_ref, sp_ref, sk_ref):
        return (0, i)

    def b_sc_map(i, j, s, sa_ref, sb_ref, sp_ref, sk_ref):
        return (0, j)

    def out_map(i, j, s, sa_ref, sb_ref, sp_ref, sk_ref):
        return (i, j)

    call_fn = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=grid,
            in_specs=[
                # A slice: load ONE slice's (bm, bk) tile per step
                pl.BlockSpec((1, bm, bk), a_map),
                # B slice: load ONE slice's (bk, bn) tile per step
                pl.BlockSpec((1, bk, bn), b_map),
                # A scales: all scales for current i block
                pl.BlockSpec((n_total, bm), a_sc_map),
                # B scales: all scales for current j block
                pl.BlockSpec((n_total, bn), b_sc_map),
            ],
            out_specs=[
                pl.BlockSpec((bm, bn), out_map),
                pl.BlockSpec((bm, bn), out_map),
            ],
            scratch_shapes=[
                pltpu.VMEM((n_pairs, bm, bn), jnp.float32),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((N, M), jnp.float32),
            jax.ShapeDtypeStruct((N, M), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )

    def run(a_slices, b_slices, a_scales_pow2, b_scales_pow2):
        return call_fn(step_a_idx, step_b_idx, step_pair_arr, step_k_arr,
                       a_slices, b_slices, a_scales_pow2, b_scales_pow2)

    return run


def fused_ozaki_v2(A_f64, B_f64, rho, n_hi, n_lo, *, bm=128, bk=128, bn=128):
    """Full pipeline using v2 Pallas kernel."""
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)

    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)
    A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
    B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

    a_slices = jnp.concatenate([A_hi_sl, A_lo_sl], axis=0)
    b_slices = jnp.concatenate([B_hi_sl, B_lo_sl], axis=0)
    a_scales = jnp.concatenate([A_hi_sc, A_lo_sc], axis=0)
    b_scales = jnp.concatenate([B_hi_sc, B_lo_sc], axis=0)

    a_scales_pow2 = jnp.float32(jnp.exp2(jnp.float32(a_scales)))
    b_scales_pow2 = jnp.float32(jnp.exp2(jnp.float32(b_scales)))

    N, K = A_f64.shape
    M = B_f64.shape[1]

    kernel_fn = _build_pallas_v2(n_hi, n_lo, N, K, M, bm, bk, bn)
    C_hi, C_lo = kernel_fn(a_slices, b_slices, a_scales_pow2, b_scales_pow2)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


N_HI, N_LO = 4, 1
BGS = (tuple([N_HI] * N_HI), tuple([N_LO] * N_HI), tuple([N_HI] * N_LO))


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def stock_fn(A, B, rho, n_hi, n_lo, bgs):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _fused_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo, bgs)
    return jnp.float64(Ch) + jnp.float64(Cl)


def time_fn(fn, *args, warmup=3, timed=10):
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
    print(f"Pallas v2: 1 dot per grid step with scalar prefetch")
    print()

    header = (f"{'n':>6}  {'stock_ms':>10}  {'v2_ms':>10}  "
              f"{'speedup':>8}  {'stock_dig':>10}  {'v2_dig':>10}")
    print(header)
    print("-" * len(header))

    for n in [512, 1024, 2048, 4096]:
        rng = np.random.RandomState(42)
        A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)
        C_ref = np.asarray(A) @ np.asarray(B)

        t_stock, C_stock = time_fn(stock_fn, A, B, rho, N_HI, N_LO, BGS)
        err_s = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_s = -np.log10(max(err_s, 1e-16))

        bm = min(256, n)
        bn = min(256, n)
        bk = min(128, n)

        @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
        def v2_fn(A, B, rho, n_hi, n_lo, bm, bk, bn):
            return fused_ozaki_v2(A, B, rho, n_hi, n_lo, bm=bm, bk=bk, bn=bn)

        t_v2, C_v2 = time_fn(v2_fn, A, B, rho, N_HI, N_LO, bm, bk, bn)
        err_v2 = float(np.max(np.abs(np.asarray(C_v2) - C_ref))
                       / np.max(np.abs(C_ref)))
        dig_v2 = -np.log10(max(err_v2, 1e-16))

        speedup = t_stock / t_v2
        print(f"{n:>6}  {t_stock:>10.2f}  {t_v2:>10.2f}  "
              f"{speedup:>7.2f}x  {dig_s:>10.1f}  {dig_v2:>10.1f}")


if __name__ == "__main__":
    main()
