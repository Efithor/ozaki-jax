"""MXU utilization diagnostic: why only 11% on TPU v6e?

Tests three hypotheses:
  1. FP32 vs BF16 precision — are we using the fast MXU datapath?
  2. Broadcast matmul vs sequential dots — is the broadcast pattern suboptimal?
  3. Explicit BF16 cast — does casting Ozaki slices to BF16 before matmul help?

Also tests the full fused pipeline with and without BF16 casts.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

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
V6E_BF16_TFLOPS = 918.0


# ── Variant 1: Current broadcast FP32 (baseline) ──────────────────

@jax.jit
def gemms_broadcast_f32(A_sl, B_sl):
    """Broadcast matmul with FP32 inputs (current library behavior)."""
    out = jnp.matmul(A_sl[:, None, :, :], B_sl[None, :, :, :])
    return out.reshape(-1, out.shape[-2], out.shape[-1])


# ── Variant 2: Broadcast with explicit BF16 cast ──────────────────

@jax.jit
def gemms_broadcast_bf16(A_sl, B_sl):
    """Broadcast matmul with explicit BF16 cast before matmul."""
    A_bf = jnp.bfloat16(A_sl)
    B_bf = jnp.bfloat16(B_sl)
    out = jnp.matmul(A_bf[:, None, :, :], B_bf[None, :, :, :])
    return out.reshape(-1, out.shape[-2], out.shape[-1])


# ── Variant 3: Sequential dots FP32 ───────────────────────────────

@functools.partial(jax.jit, static_argnums=(2,))
def gemms_sequential_f32(A_sl, B_sl, n):
    """Sequential jnp.dot with FP32 (XLA loop-unrolled)."""
    results = []
    for i in range(n):
        for j in range(n):
            results.append(jnp.dot(A_sl[i], B_sl[j]))
    return jnp.stack(results)


# ── Variant 4: Sequential dots BF16 ───────────────────────────────

@functools.partial(jax.jit, static_argnums=(2,))
def gemms_sequential_bf16(A_sl, B_sl, n):
    """Sequential jnp.dot with explicit BF16 cast."""
    results = []
    A_bf = jnp.bfloat16(A_sl)
    B_bf = jnp.bfloat16(B_sl)
    for i in range(n):
        for j in range(n):
            results.append(jnp.dot(A_bf[i], B_bf[j]))
    return jnp.stack(results)


# ── Variant 5: Broadcast FP32 with precision=DEFAULT explicit ─────

@jax.jit
def gemms_broadcast_f32_default(A_sl, B_sl):
    """Broadcast matmul with explicit precision=DEFAULT."""
    out = jnp.matmul(A_sl[:, None, :, :], B_sl[None, :, :, :],
                     precision=jax.lax.Precision.DEFAULT)
    return out.reshape(-1, out.shape[-2], out.shape[-1])


# ── Variant 6: Broadcast FP32 with precision=HIGHEST ──────────────

@jax.jit
def gemms_broadcast_f32_highest(A_sl, B_sl):
    """Broadcast matmul with explicit precision=HIGHEST (full FP32)."""
    out = jnp.matmul(A_sl[:, None, :, :], B_sl[None, :, :, :],
                     precision=jax.lax.Precision.HIGHEST)
    return out.reshape(-1, out.shape[-2], out.shape[-1])


# ── Variant 7: Single large matmul (sanity check for MXU peak) ────

@jax.jit
def single_matmul_f32(A, B):
    """Single large FP32 matmul — establishes MXU baseline."""
    return jnp.matmul(A, B)


@jax.jit
def single_matmul_bf16(A, B):
    """Single large BF16 matmul — establishes MXU ceiling."""
    return jnp.matmul(jnp.bfloat16(A), jnp.bfloat16(B))


# ── Full fused pipeline variants ──────────────────────────────────

@functools.partial(jax.jit, static_argnums=(2,))
def fused_pipeline_stock(A_f64, B_f64, rho):
    """Current fused pipeline (FP32 matmuls)."""
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    C_hi, C_lo = _fused_pipeline_logic(
        A_hi, A_lo, B_hi, B_lo, rho, N_HI, N_LO, BGS)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


@functools.partial(jax.jit, static_argnums=(2,))
def fused_pipeline_bf16_gemms(A_f64, B_f64, rho):
    """Fused pipeline with BF16 casts before matmul."""
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)

    # Extraction (unchanged).
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, N_HI)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, N_HI)
    A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, N_LO)
    B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, N_LO)

    # GEMMs with BF16 cast.
    A_hi_bf = jnp.bfloat16(A_hi_sl)
    B_hi_bf = jnp.bfloat16(B_hi_sl)
    A_lo_bf = jnp.bfloat16(A_lo_sl)
    B_lo_bf = jnp.bfloat16(B_lo_sl)

    hh = jnp.matmul(A_hi_bf[:, None, :, :], B_hi_bf[None, :, :, :])
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]
    hl = jnp.matmul(A_hi_bf[:, None, :, :], B_lo_bf[None, :, :, :])
    parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
    lh = jnp.matmul(A_lo_bf[:, None, :, :], B_hi_bf[None, :, :, :])
    parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))
    products = jnp.concatenate(parts, axis=0)

    # Scales (unchanged).
    col_scales = jnp.exp2(jnp.concatenate([
        jnp.tile(B_hi_sc, (N_HI, 1)),
        jnp.tile(B_lo_sc, (N_HI, 1)),
        jnp.tile(B_hi_sc, (N_LO, 1)),
    ], axis=0))
    row_scales = jnp.exp2(jnp.concatenate([
        A_hi_sc, A_hi_sc, A_lo_sc,
    ], axis=0))

    # 2Sum accumulation.
    C_hi, C_lo = _accumulate_2sum_logic(products, col_scales, row_scales, BGS)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


# ── Timing helper ──────────────────────────────────────────────────

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
    platform = jax.devices()[0].platform
    device_kind = jax.devices()[0].device_kind
    print(f"Device: {device_kind} ({platform})")
    print()

    for n in [2048, 4096, 8192]:
        rng = np.random.RandomState(42)
        A_f64 = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B_f64 = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)

        # Generate extracted slices for GEMM-only tests.
        A_hi = jnp.float32(A_f64)
        B_hi = jnp.float32(B_f64)
        A_hi_sl, _ = jax_extract_split_rows(A_hi, rho, N_HI)
        B_hi_sl, _ = jax_extract_split_cols(B_hi, rho, N_HI)
        # Force materialization.
        A_hi_sl.block_until_ready()
        B_hi_sl.block_until_ready()

        print(f"{'='*80}")
        print(f"n={n}  rho={rho}  n_hi={N_HI}  n_gemms=24")
        print(f"{'='*80}")

        # Theoretical GEMM floor at full BF16 MXU utilization.
        total_flops = 24 * 2.0 * n**3
        gemm_floor_ms = total_flops / (V6E_BF16_TFLOPS * 1e12) * 1000
        print(f"  Theoretical BF16 floor: {gemm_floor_ms:.2f} ms")
        print()

        # ── Section 1: GEMM variants (hi×hi only = 16 GEMMs) ──
        n_hh_gemms = N_HI * N_HI
        hh_flops = n_hh_gemms * 2.0 * n**3
        hh_floor = hh_flops / (V6E_BF16_TFLOPS * 1e12) * 1000

        print(f"  GEMM variants (hi×hi only, {n_hh_gemms} GEMMs, BF16 floor={hh_floor:.2f} ms):")

        variants = [
            ("broadcast f32 (current)", gemms_broadcast_f32, (A_hi_sl, B_hi_sl)),
            ("broadcast bf16 cast",     gemms_broadcast_bf16, (A_hi_sl, B_hi_sl)),
            ("broadcast f32 DEFAULT",   gemms_broadcast_f32_default, (A_hi_sl, B_hi_sl)),
            ("broadcast f32 HIGHEST",   gemms_broadcast_f32_highest, (A_hi_sl, B_hi_sl)),
            ("sequential f32",          gemms_sequential_f32, (A_hi_sl, B_hi_sl, N_HI)),
            ("sequential bf16",         gemms_sequential_bf16, (A_hi_sl, B_hi_sl, N_HI)),
        ]

        for name, fn, args in variants:
            t_ms, _ = time_fn(fn, *args)
            mxu = hh_floor / t_ms * 100
            print(f"    {name:30s}  {t_ms:8.2f} ms  MXU {mxu:5.1f}%")
        print()

        # ── Section 2: Single matmul baseline ──
        single_flops = 2.0 * n**3
        single_floor = single_flops / (V6E_BF16_TFLOPS * 1e12) * 1000

        print(f"  Single matmul baseline (1 GEMM, BF16 floor={single_floor:.2f} ms):")
        t_f32, _ = time_fn(single_matmul_f32, A_hi, B_hi)
        t_bf16, _ = time_fn(single_matmul_bf16, A_hi, B_hi)
        print(f"    {'single f32':30s}  {t_f32:8.2f} ms  MXU {single_floor/t_f32*100:5.1f}%")
        print(f"    {'single bf16':30s}  {t_bf16:8.2f} ms  MXU {single_floor/t_bf16*100:5.1f}%")
        print(f"    f32/bf16 ratio: {t_f32/t_bf16:.2f}x")
        print()

        # ── Section 3: Full pipeline comparison ──
        print(f"  Full fused pipeline (24 GEMMs, BF16 floor={gemm_floor_ms:.2f} ms):")
        t_stock, C_stock = time_fn(fused_pipeline_stock, A_f64, B_f64, rho)
        t_bf16p, C_bf16 = time_fn(fused_pipeline_bf16_gemms, A_f64, B_f64, rho)

        # Check accuracy impact.
        C_ref = np.asarray(A_f64) @ np.asarray(B_f64)
        err_stock = float(np.max(np.abs(np.asarray(C_stock) - C_ref)) / np.max(np.abs(C_ref)))
        err_bf16 = float(np.max(np.abs(np.asarray(C_bf16) - C_ref)) / np.max(np.abs(C_ref)))
        dig_stock = -np.log10(max(err_stock, 1e-16))
        dig_bf16 = -np.log10(max(err_bf16, 1e-16))

        print(f"    {'stock fused (f32 GEMMs)':30s}  {t_stock:8.2f} ms  "
              f"{dig_stock:.1f} digits  MXU {gemm_floor_ms/t_stock*100:5.1f}%")
        print(f"    {'bf16 cast fused':30s}  {t_bf16p:8.2f} ms  "
              f"{dig_bf16:.1f} digits  MXU {gemm_floor_ms/t_bf16p*100:5.1f}%")
        print(f"    speedup: {t_stock/t_bf16p:.2f}x")
        print()


if __name__ == "__main__":
    main()
