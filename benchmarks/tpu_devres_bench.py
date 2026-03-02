"""v0.4.0 device-resident API benchmark on TPU v6e.

Compares:
  A) numpy in → numpy out (PCIe transfer each call)
  B) JAX in → JAX out (device-resident, zero transfer)
  C) Chained matmul (3 ops, data stays on device)
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax import matmul


def _block(x):
    if hasattr(x, 'block_until_ready'):
        x.block_until_ready()


def time_ms(fn, warmup=5, trials=20):
    for _ in range(warmup):
        _block(fn())
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _block(fn())
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sorted(times)[len(times) // 2]


def bench_size(n, trials=20):
    rng = np.random.RandomState(42)
    A_np = rng.randn(n, n)
    B_np = rng.randn(n, n)
    C_np = rng.randn(n, n)
    C_exact = A_np @ B_np

    # Pre-stage on device
    A_jax = jnp.float64(jnp.array(A_np))
    B_jax = jnp.float64(jnp.array(B_np))
    C_jax = jnp.float64(jnp.array(C_np))

    print(f"\n{'='*65}")
    print(f"  n = {n}")
    print(f"{'='*65}")

    # A) numpy → numpy (full PCIe round-trip)
    t_np = time_ms(lambda: matmul(A_np, B_np, pipeline='ondevice'),
                   warmup=5, trials=trials)

    # B) JAX → JAX (device-resident, no PCIe)
    t_jax = time_ms(lambda: matmul(A_jax, B_jax, pipeline='ondevice'),
                    warmup=5, trials=trials)

    # Accuracy check
    C_out = matmul(A_jax, B_jax, pipeline='ondevice')
    err = float(jnp.max(jnp.abs(C_out - jnp.array(C_exact))) /
                jnp.max(jnp.abs(jnp.array(C_exact))))

    speedup = t_np / t_jax if t_jax > 0 else float('inf')
    saved = t_np - t_jax

    print(f"  numpy in/out:     {t_np:8.2f} ms")
    print(f"  JAX (on-device):  {t_jax:8.2f} ms")
    print(f"  Speedup:          {speedup:8.2f}x  (saved {saved:.2f} ms)")
    print(f"  Accuracy:         {err:.2e}")

    # C) Chained matmul: A @ B @ C
    def chain_np():
        r = matmul(A_np, B_np, pipeline='ondevice')
        return matmul(r, C_np, pipeline='ondevice')

    def chain_jax():
        r = matmul(A_jax, B_jax, pipeline='ondevice')
        return matmul(r, C_jax, pipeline='ondevice')

    t_chain_np = time_ms(chain_np, warmup=3, trials=max(trials // 2, 5))
    t_chain_jax = time_ms(chain_jax, warmup=3, trials=max(trials // 2, 5))
    chain_speedup = t_chain_np / t_chain_jax if t_chain_jax > 0 else float('inf')

    print(f"\n  Chained (A @ B @ C):")
    print(f"    numpy chain:    {t_chain_np:8.2f} ms")
    print(f"    JAX chain:      {t_chain_jax:8.2f} ms")
    print(f"    Speedup:        {chain_speedup:8.2f}x")

    # D) Effective TFLOPS
    flops = 2 * n**3
    tflops_np = flops / (t_np * 1e-3) / 1e12
    tflops_jax = flops / (t_jax * 1e-3) / 1e12
    print(f"\n  Effective FP64 TFLOPS:")
    print(f"    numpy path:     {tflops_np:8.3f}")
    print(f"    JAX path:       {tflops_jax:8.3f}")

    return {
        'n': n, 't_np': t_np, 't_jax': t_jax,
        'speedup': speedup, 'error': err,
        't_chain_np': t_chain_np, 't_chain_jax': t_chain_jax,
        'chain_speedup': chain_speedup,
        'tflops_np': tflops_np, 'tflops_jax': tflops_jax,
    }


if __name__ == "__main__":
    print("ozaki-jax v0.4.0 — device-resident API benchmark")
    print(f"Platform: {jax.devices()[0].platform}")
    print(f"Device:   {jax.devices()[0].device_kind}")

    sizes = [128, 256, 512, 1024]
    results = []
    for n in sizes:
        results.append(bench_size(n))

    print(f"\n\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'n':>6}  {'numpy ms':>9}  {'JAX ms':>8}  {'speedup':>8}"
          f"  {'chain spd':>10}  {'TFLOPS':>7}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}")
    for r in results:
        print(f"  {r['n']:>6}  {r['t_np']:>9.2f}  {r['t_jax']:>8.2f}"
              f"  {r['speedup']:>7.2f}x  {r['chain_speedup']:>9.2f}x"
              f"  {r['tflops_jax']:>7.3f}")
