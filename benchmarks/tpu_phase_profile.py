"""Per-phase profiling on TPU: where does time go at large n?"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax import matmul
from ozaki_jax.matmul import (
    _jax_double_f32_split, _fused_pipeline_logic, _accumulate_2sum_logic,
)
from ozaki_jax.extract import jax_extract_split_rows, jax_extract_split_cols, _compute_rho_f32

N_HI, N_LO = 4, 1
BGS = (
    tuple([N_HI] * N_HI),
    tuple([N_LO] * N_HI),
    tuple([N_HI] * N_LO),
)

# ── Phase JITs ──────────────────────────────────────────────────────

@jax.jit
def phase_split(A_f64, B_f64):
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    return A_hi, A_lo, B_hi, B_lo


@functools.partial(jax.jit, static_argnums=(4,))
def phase_extract(A_hi, A_lo, B_hi, B_lo, rho):
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, N_HI)
    A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, N_LO)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, N_HI)
    B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, N_LO)
    return A_hi_sl, A_hi_sc, A_lo_sl, A_lo_sc, B_hi_sl, B_hi_sc, B_lo_sl, B_lo_sc


@jax.jit
def phase_gemms(A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl):
    hh = jnp.matmul(A_hi_sl[:, None, :, :], B_hi_sl[None, :, :, :])
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]
    hl = jnp.matmul(A_hi_sl[:, None, :, :], B_lo_sl[None, :, :, :])
    parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
    lh = jnp.matmul(A_lo_sl[:, None, :, :], B_hi_sl[None, :, :, :])
    parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))
    return jnp.concatenate(parts, axis=0)


@jax.jit
def phase_scales(A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc):
    col_parts = [jnp.tile(B_hi_sc, (N_HI, 1)),
                 jnp.tile(B_lo_sc, (N_HI, 1)),
                 jnp.tile(B_hi_sc, (N_LO, 1))]
    row_parts = [A_hi_sc, A_hi_sc, A_lo_sc]
    col_scales = jnp.exp2(jnp.concatenate(col_parts, axis=0))
    row_scales = jnp.exp2(jnp.concatenate(row_parts, axis=0))
    return col_scales, row_scales


@functools.partial(jax.jit, static_argnums=(3,))
def phase_accum(products, col_scales, row_scales, bgs):
    C_hi, C_lo = _accumulate_2sum_logic(products, col_scales, row_scales, bgs)
    return C_hi, C_lo


@jax.jit
def phase_combine(C_hi, C_lo):
    return jnp.float64(C_hi) + jnp.float64(C_lo)


# ── Fused JIT (the real pipeline) ──────────────────────────────────

@functools.partial(jax.jit, static_argnums=(2,))
def fused_all(A_f64, B_f64, rho):
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    C_hi, C_lo = _fused_pipeline_logic(
        A_hi, A_lo, B_hi, B_lo, rho, N_HI, N_LO, BGS)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


# ── Timing helper ──────────────────────────────────────────────────

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
    platform = jax.devices()[0].platform
    device_kind = jax.devices()[0].device_kind
    print(f"Device: {device_kind} ({platform})")
    print()

    for n in [512, 1024, 2048, 4096, 8192]:
        rng = np.random.RandomState(42)
        A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)

        # Fused (real pipeline)
        t_fused, _ = time_fn(fused_all, A, B, rho)

        # Per-phase
        t_split, (Ah, Al, Bh, Bl) = time_fn(phase_split, A, B)
        t_extract, (Ahs, Ahsc, Als, Alsc, Bhs, Bhsc, Bls, Blsc) = time_fn(
            phase_extract, Ah, Al, Bh, Bl, rho)
        t_gemms, products = time_fn(phase_gemms, Ahs, Als, Bhs, Bls)
        t_scales, (cs, rs) = time_fn(phase_scales, Ahsc, Alsc, Bhsc, Blsc)
        t_accum, (C_hi, C_lo) = time_fn(phase_accum, products, cs, rs, BGS)
        t_combine, _ = time_fn(phase_combine, C_hi, C_lo)

        phases_sum = t_split + t_extract + t_gemms + t_scales + t_accum + t_combine

        # Theoretical GEMM-only time
        gemm_flops = 24 * 2.0 * n**3
        gemm_theoretical = gemm_flops / 918e12 * 1000  # ms at 100% MXU

        print(f"n={n:5d}  fused={t_fused:.2f} ms  phases_sum={phases_sum:.2f} ms  "
              f"gemm_floor={gemm_theoretical:.2f} ms")
        print(f"  split:    {t_split:7.2f} ms  ({t_split/phases_sum*100:4.1f}%)")
        print(f"  extract:  {t_extract:7.2f} ms  ({t_extract/phases_sum*100:4.1f}%)")
        print(f"  gemms:    {t_gemms:7.2f} ms  ({t_gemms/phases_sum*100:4.1f}%)  "
              f"[MXU util: {gemm_theoretical/t_gemms*100:.0f}%]")
        print(f"  scales:   {t_scales:7.2f} ms  ({t_scales/phases_sum*100:4.1f}%)")
        print(f"  accum:    {t_accum:7.2f} ms  ({t_accum/phases_sum*100:4.1f}%)")
        print(f"  combine:  {t_combine:7.2f} ms  ({t_combine/phases_sum*100:4.1f}%)")
        print(f"  overhead: fused is {phases_sum/t_fused:.2f}x sum-of-phases "
              f"(XLA fusion {'helps' if phases_sum > t_fused else 'hurts'})")
        print()


if __name__ == "__main__":
    main()
