"""Benchmark interleaved pipeline vs stock (broadcast) pipeline on TPU.

The interleaved pipeline computes each GEMM and immediately 2Sum-accumulates,
keeping only ONE (N,M) product alive at a time instead of materializing
all 24 products. XLA may fuse the matmul→scale→twosum chain.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.extract import _compute_rho_f32
from ozaki_jax.matmul import (
    _jax_double_f32_split,
    _fused_pipeline_logic,
    _interleaved_pipeline_logic,
)

N_HI, N_LO = 4, 1
BGS = (tuple([N_HI] * N_HI), tuple([N_LO] * N_HI), tuple([N_HI] * N_LO))
V6E_BF16 = 918.0


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def stock_fn(A, B, rho, n_hi, n_lo, bgs):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _fused_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo, bgs)
    return jnp.float64(Ch) + jnp.float64(Cl)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def interleaved_fn(A, B, rho, n_hi, n_lo):
    Ah, Al = _jax_double_f32_split(A)
    Bh, Bl = _jax_double_f32_split(B)
    Ch, Cl = _interleaved_pipeline_logic(Ah, Al, Bh, Bl, rho, n_hi, n_lo)
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

    header = (f"{'n':>6}  {'stock_ms':>10}  {'inter_ms':>10}  "
              f"{'speedup':>8}  {'stock_dig':>10}  {'inter_dig':>10}  "
              f"{'eff_TFLOPS':>11}")
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

        t_inter, C_inter = time_fn(
            interleaved_fn, A, B, rho, N_HI, N_LO)
        err_i = float(np.max(np.abs(np.asarray(C_inter) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_i = -np.log10(max(err_i, 1e-16))

        speedup = t_stock / t_inter
        eff = 2.0 * n**3 / (min(t_stock, t_inter) / 1000) / 1e12
        bf16_floor = 24 * 2.0 * n**3 / (V6E_BF16 * 1e12) * 1000

        print(f"{n:>6}  {t_stock:>10.2f}  {t_inter:>10.2f}  "
              f"{speedup:>7.2f}x  {dig_s:>10.1f}  {dig_i:>10.1f}  "
              f"{eff:>11.1f}")


if __name__ == "__main__":
    main()
