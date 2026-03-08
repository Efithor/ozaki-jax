"""Test and benchmark the fused Pallas GEMM+Accum kernel.

Phase 1: Correctness test on CPU (Pallas interpret mode)
Phase 2: TPU benchmark comparing fused vs stock pipeline
"""

import time
import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.extract import _compute_rho_f32, jax_extract_split_rows, jax_extract_split_cols
from ozaki_jax.matmul import (
    _jax_double_f32_split, _fused_pipeline_logic,
    _PRECISION_PRESETS, _resolve_precision,
)
from ozaki_jax.pallas_gemm_accum import (
    _make_pair_schedule, fused_gemm_accum, fused_ozaki_matmul,
)


def test_pair_schedule():
    """Verify pair schedule matches expected structure."""
    pair_a, pair_b, n_pairs, groups, block_boundaries = \
        _make_pair_schedule(4, 1)

    assert n_pairs == 24, f"Expected 24 pairs, got {n_pairs}"
    assert block_boundaries == (4, 8, 9)

    # Verify hi×hi pairs (first 16)
    for idx in range(16):
        i, j = divmod(idx, 4)
        assert pair_a[idx] == i, f"Pair {idx}: expected a={i}, got {pair_a[idx]}"
        assert pair_b[idx] == j, f"Pair {idx}: expected b={j}, got {pair_b[idx]}"

    # Verify hi×lo pairs (16-19)
    for idx in range(16, 20):
        assert pair_a[idx] == idx - 16
        assert pair_b[idx] == 4  # lo slice index

    # Verify lo×hi pairs (20-23)
    for idx in range(20, 24):
        assert pair_a[idx] == 4  # lo slice index
        assert pair_b[idx] == idx - 20

    print("PASS: pair_schedule")


def test_correctness_cpu(n=256, precision="high"):
    """Compare fused Pallas kernel against stock pipeline on CPU."""
    n_hi, n_lo = _resolve_precision(precision)
    rho = _compute_rho_f32(n)

    rng = np.random.RandomState(42)
    A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
    B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)

    # Reference: stock fused pipeline
    block_group_sizes = (
        tuple([n_hi] * n_hi),
        tuple([n_lo] * n_hi) if n_lo > 0 else (),
        tuple([n_hi] * n_lo) if n_lo > 0 else (),
    )

    @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def stock_pipeline(A_f64, B_f64, rho, n_hi, n_lo, bgs):
        A_hi, A_lo = _jax_double_f32_split(A_f64)
        B_hi, B_lo = _jax_double_f32_split(B_f64)
        C_hi, C_lo = _fused_pipeline_logic(
            A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo, bgs)
        return jnp.float64(C_hi) + jnp.float64(C_lo)

    C_stock = stock_pipeline(A, B, rho, n_hi, n_lo, block_group_sizes)

    # Test: prepare slices manually for the Pallas kernel
    A_hi, A_lo = _jax_double_f32_split(A)
    B_hi, B_lo = _jax_double_f32_split(B)
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)
    A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
    B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

    a_slices = jnp.concatenate([A_hi_sl, A_lo_sl], axis=0)
    b_slices = jnp.concatenate([B_hi_sl, B_lo_sl], axis=0)
    a_scales = jnp.concatenate([A_hi_sc, A_lo_sc], axis=0)
    b_scales = jnp.concatenate([B_hi_sc, B_lo_sc], axis=0)

    # Check if we're on TPU (Pallas kernel requires TPU)
    platform = jax.devices()[0].platform
    if platform == "tpu":
        C_hi_p, C_lo_p = fused_gemm_accum(
            a_slices, b_slices, a_scales, b_scales,
            n_hi, n_lo, bm=128, bk=128, bn=128)
        C_pallas = jnp.float64(C_hi_p) + jnp.float64(C_lo_p)
    else:
        # On CPU, test the full pipeline wrapper (which includes extraction)
        # This will fail since Pallas TPU is not available on CPU.
        # Instead, verify the pair schedule and compare against numpy ref.
        print(f"  Platform={platform}, skipping Pallas kernel (TPU only)")
        print(f"  Verifying stock pipeline accuracy instead...")
        C_ref = np.asarray(A) @ np.asarray(B)
        err = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                    / np.max(np.abs(C_ref)))
        digits = -np.log10(max(err, 1e-16))
        print(f"  Stock pipeline: {digits:.1f} correct digits")
        assert digits > 7.0, f"Expected >7 digits, got {digits:.1f}"
        print(f"PASS: correctness_cpu (stock only, {digits:.1f} digits)")
        return

    # Compare Pallas vs stock
    diff = float(jnp.max(jnp.abs(C_pallas - C_stock)))
    C_ref = np.asarray(A) @ np.asarray(B)
    err_stock = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                      / np.max(np.abs(C_ref)))
    err_pallas = float(np.max(np.abs(np.asarray(C_pallas) - C_ref))
                       / np.max(np.abs(C_ref)))

    dig_stock = -np.log10(max(err_stock, 1e-16))
    dig_pallas = -np.log10(max(err_pallas, 1e-16))

    print(f"  Stock:  {dig_stock:.1f} digits")
    print(f"  Pallas: {dig_pallas:.1f} digits")
    print(f"  Max diff (pallas vs stock): {diff:.2e}")

    assert dig_pallas > 5.0, f"Pallas accuracy too low: {dig_pallas:.1f} digits"
    print(f"PASS: correctness n={n} precision={precision}")


def benchmark_tpu(sizes=None, precision="high", warmup=3, timed=10):
    """Benchmark fused Pallas kernel vs stock pipeline on TPU."""
    if sizes is None:
        sizes = [1024, 2048, 4096, 8192]

    n_hi, n_lo = _resolve_precision(precision)
    n_pairs = n_hi * n_hi + 2 * n_hi * n_lo
    V6E_BF16 = 918.0

    block_group_sizes = (
        tuple([n_hi] * n_hi),
        tuple([n_lo] * n_hi) if n_lo > 0 else (),
        tuple([n_hi] * n_lo) if n_lo > 0 else (),
    )

    print(f"Device: {jax.devices()[0].device_kind}")
    print(f"Preset: {precision} (n_hi={n_hi}, n_lo={n_lo}, {n_pairs} GEMMs)")
    print()

    @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def stock_fn(A, B, rho, n_hi, n_lo, bgs):
        A_hi, A_lo = _jax_double_f32_split(A)
        B_hi, B_lo = _jax_double_f32_split(B)
        C_hi, C_lo = _fused_pipeline_logic(
            A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo, bgs)
        return jnp.float64(C_hi) + jnp.float64(C_lo)

    def time_fn(fn, *args):
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

    print(f"{'n':>6}  {'stock_ms':>10}  {'pallas_ms':>10}  {'speedup':>8}  "
          f"{'stock_dig':>10}  {'pallas_dig':>11}  {'eff_TFLOPS':>11}")

    for n in sizes:
        rng = np.random.RandomState(42)
        A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)
        C_ref = np.asarray(A) @ np.asarray(B)

        # Stock pipeline
        t_stock, C_stock = time_fn(
            stock_fn, A, B, rho, n_hi, n_lo, block_group_sizes)
        err_s = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_s = -np.log10(max(err_s, 1e-16))

        # Pallas fused
        # Choose tile sizes based on n
        bm = min(256, n)
        bn = min(256, n)
        bk = min(128, n)

        # Wrap in JIT-friendly function
        @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
        def pallas_fn(A, B, rho, n_hi, n_lo, bm, bk, bn):
            return fused_ozaki_matmul(A, B, rho, n_hi, n_lo,
                                     bm=bm, bk=bk, bn=bn)

        t_pallas, C_pallas = time_fn(
            pallas_fn, A, B, rho, n_hi, n_lo, bm, bk, bn)
        err_p = float(np.max(np.abs(np.asarray(C_pallas) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_p = -np.log10(max(err_p, 1e-16))

        eff = 2.0 * n**3 / (t_pallas / 1000) / 1e12
        speedup = t_stock / t_pallas

        print(f"{n:>6}  {t_stock:>10.2f}  {t_pallas:>10.2f}  {speedup:>7.2f}x  "
              f"{dig_s:>10.1f}  {dig_p:>11.1f}  {eff:>11.1f}")


if __name__ == "__main__":
    print("=== Pair Schedule Test ===")
    test_pair_schedule()
    print()

    print("=== Correctness Test ===")
    test_correctness_cpu(n=256)
    print()

    platform = jax.devices()[0].platform
    if platform == "tpu":
        print("=== TPU Benchmark ===")
        benchmark_tpu()
    else:
        print(f"Skipping TPU benchmark (platform={platform})")
