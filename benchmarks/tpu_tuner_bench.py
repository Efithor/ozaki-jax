"""GEMM tuner benchmark: accuracy × throughput × batching strategy.

Tests:
  - (n_hi, n_lo) configs: (4,1), (4,2), (5,4)
  - GEMM strategies: sequential (current), broadcast batched
  - Matrix sizes: 256, 512, 1024
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.matmul import _jax_double_f32_split, _accumulate_2sum_logic
from ozaki_jax.extract import (
    _compute_rho_f32, jax_extract_split_rows, jax_extract_split_cols,
)


# -- GEMM strategies --

def _gemms_sequential(A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl, n_hi, n_lo):
    """Current approach: Python loops, unrolled at trace time."""
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


def _gemms_broadcast(A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl, n_hi, n_lo):
    """Broadcast matmul: A[:, None] @ B[None, :] for all-pairs."""
    # hi × hi: (n_hi, 1, N, K) @ (1, n_hi, K, M) → (n_hi, n_hi, N, M)
    hh = jnp.matmul(A_hi_sl[:, None, :, :], B_hi_sl[None, :, :, :])
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]

    if n_lo > 0:
        hl = jnp.matmul(A_hi_sl[:, None, :, :], B_lo_sl[None, :, :, :])
        parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
        lh = jnp.matmul(A_lo_sl[:, None, :, :], B_hi_sl[None, :, :, :])
        parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))

    return jnp.concatenate(parts, axis=0)


def _gemms_vmap(A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl, n_hi, n_lo):
    """Nested vmap: express all-pairs as batched operation."""
    def all_pairs(A_sl, B_sl):
        return jax.vmap(lambda a: jax.vmap(lambda b: jnp.dot(a, b))(B_sl))(A_sl)

    hh = all_pairs(A_hi_sl, B_hi_sl)  # (n_hi, n_hi, N, M)
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]

    if n_lo > 0:
        hl = all_pairs(A_hi_sl, B_lo_sl)
        parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
        lh = all_pairs(A_lo_sl, B_hi_sl)
        parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))

    return jnp.concatenate(parts, axis=0)


STRATEGIES = {
    "sequential": _gemms_sequential,
    "broadcast": _gemms_broadcast,
    "vmap": _gemms_vmap,
}


# -- Build JIT for each (config, strategy) --

def _make_fused_jit(gemm_fn, n_hi, n_lo):
    """Create a fully-fused JIT for a given GEMM strategy and slice config."""
    block_group_sizes = (
        tuple([n_hi] * n_hi),
        tuple([n_lo] * n_hi) if n_lo > 0 else (),
        tuple([n_hi] * n_lo) if n_lo > 0 else (),
    )

    @functools.partial(jax.jit, static_argnums=(2,))
    def fused(A_f64, B_f64, rho):
        A_hi, A_lo = _jax_double_f32_split(A_f64)
        B_hi, B_lo = _jax_double_f32_split(B_f64)

        A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

        products = gemm_fn(A_hi_sl, A_lo_sl, B_hi_sl, B_lo_sl, n_hi, n_lo)

        col_scales = jnp.exp2(jnp.concatenate([
            jnp.tile(B_hi_sc, (n_hi, 1)),
        ] + ([
            jnp.tile(B_lo_sc, (n_hi, 1)),
            jnp.tile(B_hi_sc, (n_lo, 1)),
        ] if n_lo > 0 else []), axis=0))

        row_scales = jnp.exp2(jnp.concatenate([
            A_hi_sc,
        ] + ([
            A_hi_sc,
            A_lo_sc,
        ] if n_lo > 0 else []), axis=0))

        C_hi, C_lo = _accumulate_2sum_logic(
            products, col_scales, row_scales, block_group_sizes)
        return jnp.float64(C_hi) + jnp.float64(C_lo)

    return fused


def time_ms(fn, warmup=5, trials=20):
    for _ in range(warmup):
        fn().block_until_ready()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn().block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sorted(times)[len(times) // 2]


def bench_config(n, n_hi, n_lo, strategy_name, rng):
    A = rng.randn(n, n)
    B = rng.randn(n, n)
    C_exact = A @ B
    rho = _compute_rho_f32(n)

    A_j = jnp.float64(jnp.array(A))
    B_j = jnp.float64(jnp.array(B))

    n_gemms = n_hi * n_hi + n_hi * n_lo + n_lo * n_hi

    gemm_fn = STRATEGIES[strategy_name]
    fused_fn = _make_fused_jit(gemm_fn, n_hi, n_lo)

    # Accuracy
    C_out = np.asarray(fused_fn(A_j, B_j, rho))
    err = np.max(np.abs(C_out - C_exact)) / np.max(np.abs(C_exact))
    digits = -np.log10(max(err, 1e-16))

    # Throughput
    t = time_ms(lambda: fused_fn(A_j, B_j, rho), warmup=5, trials=20)
    flops = 2 * n**3
    tflops = flops / (t * 1e-3) / 1e12

    return {
        'n': n, 'n_hi': n_hi, 'n_lo': n_lo, 'strategy': strategy_name,
        'gemms': n_gemms, 'error': err, 'digits': digits,
        'time_ms': t, 'tflops': tflops,
    }


if __name__ == "__main__":
    print("ozaki-jax GEMM tuner benchmark")
    print(f"Platform: {jax.devices()[0].platform}")
    print(f"Device:   {jax.devices()[0].device_kind}")

    rng = np.random.RandomState(42)

    configs = [
        (3, 1),   # 15 GEMMs, ~7 digits
        (4, 1),   # 24 GEMMs, ~9.5 digits  ← sweet spot?
        (4, 2),   # 32 GEMMs, ~9.7 digits
        (5, 4),   # 65 GEMMs, ~10 digits (current)
    ]

    sizes = [256, 512, 1024]
    strategies = ["sequential", "broadcast", "vmap"]

    results = []
    for n in sizes:
        print(f"\n{'='*75}")
        print(f"  n = {n}")
        print(f"{'='*75}")
        print(f"  {'config':>8}  {'strategy':>12}  {'GEMMs':>5}  {'time_ms':>8}"
              f"  {'TFLOPS':>7}  {'digits':>6}  {'error':>10}")
        print(f"  {'--------':>8}  {'------------':>12}  {'-----':>5}  {'--------':>8}"
              f"  {'-------':>7}  {'------':>6}  {'----------':>10}")

        for n_hi, n_lo in configs:
            for strat in strategies:
                r = bench_config(n, n_hi, n_lo, strat, rng)
                results.append(r)
                tag = f"({n_hi},{n_lo})"
                print(f"  {tag:>8}  {strat:>12}  {r['gemms']:>5}"
                      f"  {r['time_ms']:>8.2f}  {r['tflops']:>7.3f}"
                      f"  {r['digits']:>6.1f}  {r['error']:>10.2e}")

    # Summary: best config per size
    print(f"\n\n{'='*75}")
    print("  BEST CONFIGS (highest TFLOPS per accuracy tier)")
    print(f"{'='*75}")
    for n in sizes:
        nr = [r for r in results if r['n'] == n]
        print(f"\n  n={n}:")
        # Best for ~10 digits
        tier10 = [r for r in nr if r['digits'] >= 9.0]
        if tier10:
            best = max(tier10, key=lambda r: r['tflops'])
            print(f"    ≥9 digits:  ({best['n_hi']},{best['n_lo']}) {best['strategy']:>12}"
                  f"  {best['gemms']} GEMMs  {best['time_ms']:.2f}ms"
                  f"  {best['tflops']:.3f} TFLOPS  {best['digits']:.1f} digits")
        # Best for ~7 digits
        tier7 = [r for r in nr if r['digits'] >= 6.5]
        if tier7:
            best = max(tier7, key=lambda r: r['tflops'])
            print(f"    ≥7 digits:  ({best['n_hi']},{best['n_lo']}) {best['strategy']:>12}"
                  f"  {best['gemms']} GEMMs  {best['time_ms']:.2f}ms"
                  f"  {best['tflops']:.3f} TFLOPS  {best['digits']:.1f} digits")
