"""v0.4.0 profiling: fully-fused pipeline stage breakdown on TPU v6e.

Compares v0.3.0 (CPU split + 4 FP32 transfer + fused JIT) vs
v0.4.0 (2 FP64 transfer + fully-fused JIT).

Also profiles individual stages with separate JIT functions.
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax.matmul import (
    _double_f32_split, _jax_double_f32_split,
    _fused_ondevice_jit, _fully_fused_ondevice_jit,
    _fused_pipeline_logic,
    _ONDEVICE_N_HI, _ONDEVICE_N_LO,
)
from ozaki_jax.extract import _compute_rho_f32


# -- Separate JIT functions for stage profiling --

@jax.jit
def _profile_split(A_f64, B_f64):
    """Just the FP64 → FP32 split on device."""
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    return A_hi, A_lo, B_hi, B_lo


def _block(out):
    """Block until JAX arrays in output are ready."""
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()
    elif isinstance(out, tuple):
        for item in out:
            _block(item)


def time_fn(fn, *args, warmup=3, trials=10):
    """Time a function with warmup, return median ms."""
    for _ in range(warmup):
        _block(fn(*args))

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _block(fn(*args))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sorted(times)[len(times) // 2]


def profile_size(n, trials=10):
    """Profile all stages for a given matrix size."""
    rng = np.random.RandomState(42)
    A = rng.randn(n, n)
    B = rng.randn(n, n)
    C_exact = A @ B

    K = n
    rho = _compute_rho_f32(K)
    n_hi = _ONDEVICE_N_HI
    n_lo = _ONDEVICE_N_LO
    block_group_sizes = (
        tuple([n_hi] * n_hi),
        tuple([n_lo] * n_hi),
        tuple([n_hi] * n_lo),
    )

    print(f"\n{'='*60}")
    print(f"  n = {n}  (rho={rho}, n_hi={n_hi}, n_lo={n_lo})")
    print(f"{'='*60}")

    # -- v0.3.0 approach: CPU split + 4 FP32 transfer + fused JIT --
    print("\n  v0.3.0 (CPU split + 4 FP32 transfers):")

    # Stage: CPU double_f32_split
    def cpu_split():
        return _double_f32_split(A), _double_f32_split(B)
    t_cpu_split = time_fn(lambda: cpu_split(), warmup=3, trials=trials)
    print(f"    CPU split:         {t_cpu_split:8.2f} ms")

    # Stage: Transfer 4 FP32 matrices
    A_hi, A_lo = _double_f32_split(A)
    B_hi, B_lo = _double_f32_split(B)

    def transfer_4_f32():
        a_hi = jax.device_put(A_hi)
        a_lo = jax.device_put(A_lo)
        b_hi = jax.device_put(B_hi)
        b_lo = jax.device_put(B_lo)
        return a_hi, a_lo, b_hi, b_lo
    t_xfer_4f32 = time_fn(lambda: transfer_4_f32(), warmup=3, trials=trials)
    print(f"    Transfer 4×FP32:   {t_xfer_4f32:8.2f} ms")

    # Stage: Fused JIT (extract + GEMMs + accum)
    A_hi_j, A_lo_j, B_hi_j, B_lo_j = transfer_4_f32()
    t_fused_v03 = time_fn(
        lambda: _fused_ondevice_jit(A_hi_j, A_lo_j, B_hi_j, B_lo_j,
                                     rho, n_hi, n_lo, block_group_sizes),
        warmup=3, trials=trials)
    print(f"    Fused JIT (v0.3):  {t_fused_v03:8.2f} ms")

    # Stage: Transfer back + combine
    C_hi, C_lo = _fused_ondevice_jit(A_hi_j, A_lo_j, B_hi_j, B_lo_j,
                                      rho, n_hi, n_lo, block_group_sizes)
    def xfer_back_combine(c_hi, c_lo):
        return np.float64(np.array(c_hi)) + np.float64(np.array(c_lo))
    t_back_v03 = time_fn(lambda: xfer_back_combine(C_hi, C_lo), warmup=3, trials=trials)
    print(f"    Transfer+combine:  {t_back_v03:8.2f} ms")

    total_v03 = t_cpu_split + t_xfer_4f32 + t_fused_v03 + t_back_v03
    print(f"    TOTAL v0.3.0:      {total_v03:8.2f} ms")

    # -- v0.4.0 approach: 2 FP64 transfer + fully-fused JIT --
    print(f"\n  v0.4.0 (2 FP64 transfers, split on device):")

    # Stage: Transfer 2 FP64 matrices
    def transfer_2_f64():
        a = jax.device_put(A)
        b = jax.device_put(B)
        return a, b
    t_xfer_2f64 = time_fn(lambda: transfer_2_f64(), warmup=3, trials=trials)
    print(f"    Transfer 2×FP64:   {t_xfer_2f64:8.2f} ms")

    # Stage: Fully-fused JIT (split + extract + GEMMs + accum)
    A_j, B_j = jnp.array(A), jnp.array(B)
    t_fused_v04 = time_fn(
        lambda: _fully_fused_ondevice_jit(A_j, B_j, rho, n_hi, n_lo,
                                           block_group_sizes),
        warmup=3, trials=trials)
    print(f"    Fully-fused JIT:   {t_fused_v04:8.2f} ms")

    # Stage: Transfer back + combine
    C_hi2, C_lo2 = _fully_fused_ondevice_jit(A_j, B_j, rho, n_hi, n_lo,
                                               block_group_sizes)
    t_back_v04 = time_fn(lambda: xfer_back_combine(C_hi2, C_lo2), warmup=3, trials=trials)
    print(f"    Transfer+combine:  {t_back_v04:8.2f} ms")

    total_v04 = t_xfer_2f64 + t_fused_v04 + t_back_v04
    print(f"    TOTAL v0.4.0:      {total_v04:8.2f} ms")

    # -- Device-only split cost --
    print(f"\n  Device split overhead:")
    t_split_device = time_fn(lambda: _profile_split(A_j, B_j), warmup=3, trials=trials)
    print(f"    On-device split:   {t_split_device:8.2f} ms")
    print(f"    JIT overhead:      {t_fused_v04 - t_fused_v03:+8.2f} ms (fully-fused − fused)")

    # -- Summary --
    speedup = total_v03 / total_v04 if total_v04 > 0 else float('inf')
    saved = total_v03 - total_v04
    print(f"\n  Summary:")
    print(f"    v0.3.0 total:      {total_v03:8.2f} ms")
    print(f"    v0.4.0 total:      {total_v04:8.2f} ms")
    print(f"    Saved:             {saved:8.2f} ms ({speedup:.2f}x speedup)")

    # Accuracy check
    C_out = np.float64(np.array(C_hi2)) + np.float64(np.array(C_lo2))
    err = np.max(np.abs(C_out - C_exact)) / np.max(np.abs(C_exact))
    print(f"    Accuracy:          {err:.2e} rel error")

    return {
        'n': n,
        'v03_total': total_v03,
        'v04_total': total_v04,
        'speedup': speedup,
        'cpu_split': t_cpu_split,
        'xfer_4f32': t_xfer_4f32,
        'xfer_2f64': t_xfer_2f64,
        'fused_v03': t_fused_v03,
        'fused_v04': t_fused_v04,
        'split_device': t_split_device,
        'error': err,
    }


if __name__ == "__main__":
    print("ozaki-jax v0.4.0 profiling — fully-fused pipeline")
    print(f"Platform: {jax.devices()[0].platform}")
    print(f"Device:   {jax.devices()[0].device_kind}")

    sizes = [128, 256, 512, 1024]
    results = []
    for n in sizes:
        results.append(profile_size(n))

    print(f"\n\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"  {'n':>6}  {'v0.3 ms':>8}  {'v0.4 ms':>8}  {'speedup':>8}  {'split(dev)':>10}  {'error':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['n']:>6}  {r['v03_total']:>8.2f}  {r['v04_total']:>8.2f}"
              f"  {r['speedup']:>8.2f}x  {r['split_device']:>9.2f}ms  {r['error']:>10.2e}")
