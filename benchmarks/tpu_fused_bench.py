"""TPU benchmark: host vs ondevice vs fused accumulation pipelines.

Measures wall-clock time for each accumulation mode across matrix sizes.
Includes JIT warmup separation so we time steady-state performance.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

from ozaki_jax import matmul

def bench_one(A, B, pipeline, accumulation, n_warmup=3, n_iter=20):
    """Benchmark a single config. Returns median ms."""
    # Warmup (includes JIT compilation).
    for _ in range(n_warmup):
        C = matmul(A, B, pipeline=pipeline, accumulation=accumulation)
        # Block until complete on TPU.
        if hasattr(C, 'block_until_ready'):
            C.block_until_ready()
        else:
            np.asarray(C)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        C = matmul(A, B, pipeline=pipeline, accumulation=accumulation)
        if hasattr(C, 'block_until_ready'):
            C.block_until_ready()
        else:
            np.asarray(C)
        times.append(time.perf_counter() - t0)

    return np.median(times) * 1000, np.array(times) * 1000


def main():
    print("=" * 70)
    print("ozaki-jax TPU fused pipeline benchmark")
    print("=" * 70)
    print(f"JAX version:  {jax.__version__}")
    print(f"Platform:     {jax.default_backend()}")
    print(f"Devices:      {jax.devices()}")
    print()

    rng = np.random.RandomState(42)

    sizes = [128, 256, 512, 1024]
    configs = [
        ("host",     "ondevice", "host"),
        ("ondevice", "ondevice", "ondevice"),
        ("fused",    "ondevice", "fused"),
    ]

    # ----------------------------------------------------------------
    # Accuracy check first
    # ----------------------------------------------------------------
    print("--- accuracy check (n=256) ---")
    A = rng.randn(256, 256); B = rng.randn(256, 256); C_exact = A @ B
    for label, pipeline, accum in configs:
        C = matmul(A, B, pipeline=pipeline, accumulation=accum)
        err = np.max(np.abs(np.asarray(C, dtype=np.float64) - C_exact)) / np.max(np.abs(C_exact))
        print(f"  {label:12s} error: {err:.2e}")
    print()

    # ----------------------------------------------------------------
    # Timing
    # ----------------------------------------------------------------
    print("--- timing (median of 20 iterations, 3 warmup) ---\n")

    header = f"{'Size':>6}"
    for label, _, _ in configs:
        header += f"  {label:>12s}"
    header += f"  {'fused_speedup':>14s}"
    print(header)
    print("-" * len(header))

    for N in sizes:
        A = rng.randn(N, N).astype(np.float64)
        B = rng.randn(N, N).astype(np.float64)

        results = {}
        for label, pipeline, accum in configs:
            med, all_times = bench_one(A, B, pipeline, accum)
            results[label] = (med, all_times)

        row = f"{N:>6}"
        for label, _, _ in configs:
            med = results[label][0]
            row += f"  {med:>10.1f}ms"
        # Speedup: fused vs host accumulation
        speedup = results["host"][0] / results["fused"][0] if results["fused"][0] > 0 else 0
        row += f"  {speedup:>12.2f}x"
        print(row)

    print()

    # ----------------------------------------------------------------
    # Detailed breakdown at largest size
    # ----------------------------------------------------------------
    N = sizes[-1]
    print(f"--- detailed stats at n={N} ---\n")
    A = rng.randn(N, N).astype(np.float64)
    B = rng.randn(N, N).astype(np.float64)

    for label, pipeline, accum in configs:
        med, all_times = bench_one(A, B, pipeline, accum, n_warmup=3, n_iter=30)
        p5 = np.percentile(all_times, 5)
        p95 = np.percentile(all_times, 95)
        print(f"  {label:12s}: median={med:.1f}ms  p5={p5:.1f}ms  p95={p95:.1f}ms")
    print()

    # ----------------------------------------------------------------
    # JIT compilation time (first call)
    # ----------------------------------------------------------------
    print(f"--- JIT compilation time (n=256, first call) ---\n")
    # Use different random seed to avoid any caching
    rng2 = np.random.RandomState(99)
    A = rng2.randn(256, 256).astype(np.float64)
    B = rng2.randn(256, 256).astype(np.float64)

    # We need fresh JIT traces — but they're already compiled from above.
    # Just report this informational note.
    print("  (JIT already compiled from prior runs — see warmup overhead in first bench call)")
    print()

    print("done.")


if __name__ == "__main__":
    main()
