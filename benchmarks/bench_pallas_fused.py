"""Benchmark fused Pallas GEMM+Accum kernel vs stock pipeline on TPU."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.extract import _compute_rho_f32
from ozaki_jax.matmul import _jax_double_f32_split, _fused_pipeline_logic
from ozaki_jax.pallas_gemm_accum import fused_ozaki_matmul

N_HI, N_LO = 4, 1
BGS = (tuple([N_HI] * N_HI), tuple([N_LO] * N_HI), tuple([N_HI] * N_LO))
V6E_BF16 = 918.0


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def stock_fn(A, B, rho, n_hi, n_lo, bgs):
    A_hi, A_lo = _jax_double_f32_split(A)
    B_hi, B_lo = _jax_double_f32_split(B)
    C_hi, C_lo = _fused_pipeline_logic(
        A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo, bgs)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


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
    print()

    header = f"{'n':>6}  {'stock_ms':>10}  {'pallas_ms':>10}  {'speedup':>8}  {'stock_dig':>10}  {'pallas_dig':>11}  {'eff_TFLOPS':>11}"
    print(header)
    print("-" * len(header))

    for n in [512, 1024, 2048, 4096, 8192]:
        rng = np.random.RandomState(42)
        A = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        B = jnp.asarray(rng.randn(n, n), dtype=jnp.float64)
        rho = _compute_rho_f32(n)
        C_ref = np.asarray(A) @ np.asarray(B)

        # Stock pipeline
        t_stock, C_stock = time_fn(stock_fn, A, B, rho, N_HI, N_LO, BGS)
        err_s = float(np.max(np.abs(np.asarray(C_stock) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_s = -np.log10(max(err_s, 1e-16))

        # Pallas fused
        bm = min(256, n)
        bn = min(256, n)
        bk = min(128, n)

        @functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
        def pallas_fn(A, B, rho, n_hi, n_lo, bm, bk, bn):
            return fused_ozaki_matmul(A, B, rho, n_hi, n_lo,
                                     bm=bm, bk=bk, bn=bn)

        t_pallas, C_pallas = time_fn(
            pallas_fn, A, B, rho, N_HI, N_LO, bm, bk, bn)
        err_p = float(np.max(np.abs(np.asarray(C_pallas) - C_ref))
                      / np.max(np.abs(C_ref)))
        dig_p = -np.log10(max(err_p, 1e-16))

        speedup = t_stock / t_pallas
        eff = 2.0 * n**3 / (t_pallas / 1000) / 1e12

        print(f"{n:>6}  {t_stock:>10.2f}  {t_pallas:>10.2f}  "
              f"{speedup:>7.2f}x  {dig_s:>10.1f}  {dig_p:>11.1f}  "
              f"{eff:>11.1f}")


if __name__ == "__main__":
    main()
