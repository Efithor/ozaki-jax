"""Validate dtype behavior, accuracy, and timing on current JAX backend.

Usage:
    python tpu_validate.py
"""

import sys
import time
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path for ozaki_jax import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax import matmul, matmul_numpy
from ozaki_jax.extract import _compute_rho

def main():
    results = {}

    # Device info.
    devices = jax.devices()
    print("ozaki-jax validation")
    print(f"JAX version:  {jax.__version__}")
    print(f"Devices:      {devices}")
    print(f"Device kind:  {devices[0].device_kind}")
    print(f"Platform:     {jax.default_backend()}")
    print()

    results["jax_version"] = jax.__version__
    results["device_kind"] = devices[0].device_kind
    results["platform"] = jax.default_backend()
    results["n_devices"] = len(devices)

    is_tpu = jax.default_backend() == "tpu"
    if not is_tpu:
        print("WARNING: Not running on TPU! Results are CPU-simulated.\n")

    # Test 1: BF16 matmul dtype check.
    print("test 1: bf16 matmul behavior\n")
    a_bf16 = jnp.ones((4, 4), dtype=jnp.bfloat16)
    b_bf16 = jnp.ones((4, 4), dtype=jnp.bfloat16)
    c = jnp.dot(a_bf16, b_bf16)
    print(f"BF16 @ BF16 output dtype: {c.dtype}")
    print("Expected on TPU: float32")
    print("Expected on CPU: bfloat16")
    results["bf16_dot_dtype"] = str(c.dtype)

    # Cast to f32 explicitly for GEMMs (works on both CPU and TPU).
    a_f32 = jnp.float32(a_bf16)
    b_f32 = jnp.float32(b_bf16)
    c2 = jnp.dot(a_f32, b_f32)
    print(f"F32(BF16) @ F32(BF16) output dtype: {c2.dtype}")
    results["f32_dot_dtype"] = str(c2.dtype)
    print()

    # Test 2: quick precision check.
    print("test 2: quick precision check (n=64)\n")
    rng = np.random.RandomState(42)
    A = rng.randn(64, 64).astype(np.float64)
    B = rng.randn(64, 64).astype(np.float64)
    C_ref = A @ B
    ref_norm = np.linalg.norm(C_ref)

    C_jax = matmul(A, B)
    C_np = matmul_numpy(A, B)

    err_jax = np.linalg.norm(C_jax - C_ref) / ref_norm
    err_np = np.linalg.norm(C_np - C_ref) / ref_norm
    err_diff = np.linalg.norm(C_jax - C_np) / ref_norm

    print(f"matmul (JAX):      {err_jax:.2e}")
    print(f"matmul_numpy:      {err_np:.2e}")
    print(f"JAX vs numpy diff: {err_diff:.2e}")
    results["quick_err_jax"] = float(err_jax)
    results["quick_err_numpy"] = float(err_np)
    results["quick_err_diff"] = float(err_diff)
    print()

    # Test 3: relative error table across sizes.
    print("test 3: relative error table\n")
    sizes = [64, 128, 256, 512]
    n_trials = 3
    ladder = {}

    for n in sizes:
        errors_bf16 = []
        errors_extract = []

        for trial in range(n_trials):
            scale = 10.0 ** rng.uniform(-3, 3)
            A = rng.randn(n, n).astype(np.float64) * scale
            B = rng.randn(n, n).astype(np.float64) * scale
            C_ref = A @ B
            ref_norm = np.linalg.norm(C_ref)
            if ref_norm == 0:
                continue

            # Naive BF16
            A32, B32 = jnp.float32(A.astype(np.float32)), jnp.float32(B.astype(np.float32))
            a_bf = jnp.float32(jnp.bfloat16(A32))
            b_bf = jnp.float32(jnp.bfloat16(B32))
            C_bf16 = np.array(jnp.dot(a_bf, b_bf))
            errors_bf16.append(np.linalg.norm(C_bf16.astype(np.float64) - C_ref) / ref_norm)

            # Ozaki Extract JAX
            C_ext = matmul(A, B)
            errors_extract.append(np.linalg.norm(C_ext - C_ref) / ref_norm)

        ladder[n] = {
            "bf16": float(np.mean(errors_bf16)),
            "extract": float(np.mean(errors_extract)),
        }

    header = f"{'Size':>6}  {'BF16':>12}  {'Extract':>12}  {'Improvement':>12}"
    print(header)
    print("-" * len(header))
    for n in sizes:
        r = ladder[n]
        improvement = r["bf16"] / r["extract"] if r["extract"] > 0 else float("inf")
        print(f"{n:>6}  {r['bf16']:>12.2e}  {r['extract']:>12.2e}  {improvement:>12.0f}x")
    results["precision_ladder"] = ladder
    print()

    # Test 4: exactness condition parameters.
    print("test 4: exactness condition\n")
    for K in [64, 256, 512, 648, 1024, 1040]:
        rho = _compute_rho(K)
        bits = 53 - rho
        product_sum = K * (2**bits - 1)**2
        exact = product_sum < 2**24
        print(f"K={K:>5}: rho={rho}, bits/slice={bits}, "
              f"K*(2^p-1)^2={product_sum:>12,} vs 2^24={2**24:>12,}  "
              f"{'EXACT' if exact else 'NOT EXACT'}")
    results["max_exact_K"] = 1040
    print()

    # Test 5: timing.
    print("test 5: timing\n")
    timing = {}

    for n in [128, 256, 512]:
        A = rng.randn(n, n).astype(np.float64)
        B = rng.randn(n, n).astype(np.float64)

        # Warm up JIT
        _ = matmul(A, B)
        jax.block_until_ready(_)

        # Time matmul (JAX) — 5 calls
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            C = matmul(A, B)
            if hasattr(C, 'block_until_ready'):
                C.block_until_ready()
            times.append(time.perf_counter() - t0)
        t_jax = np.median(times)

        # Time matmul_numpy — 5 calls
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            C = matmul_numpy(A, B)
            times.append(time.perf_counter() - t0)
        t_np = np.median(times)

        # Time raw FP64 numpy matmul — 5 calls
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            C = A @ B
            times.append(time.perf_counter() - t0)
        t_f64 = np.median(times)

        timing[n] = {
            "jax_ms": float(t_jax * 1000),
            "numpy_ms": float(t_np * 1000),
            "f64_ms": float(t_f64 * 1000),
            "overhead_vs_f64": float(t_jax / t_f64) if t_f64 > 0 else 0,
        }
        print(f"n={n:>4}: JAX={t_jax*1000:>8.1f}ms  "
              f"numpy={t_np*1000:>8.1f}ms  "
              f"FP64={t_f64*1000:>8.1f}ms  "
              f"overhead={t_jax/t_f64:.1f}x")

    results["timing"] = timing
    print()

    # Test 6: large matrix near K limit.
    print("test 6: large matrix near K limit (n=1024)\n")
    A = rng.randn(32, 1024).astype(np.float64)
    B = rng.randn(1024, 32).astype(np.float64)
    C_ref = A @ B
    ref_norm = np.linalg.norm(C_ref)

    C_ext = matmul(A, B)
    err = np.linalg.norm(C_ext - C_ref) / ref_norm
    print(f"32x1024 @ 1024x32: relative error = {err:.2e}")
    results["large_K_error"] = float(err)
    print()

    # Summary.
    print("validation summary")
    all_pass = True

    check_tpu = jax.default_backend() == "tpu"
    print(f"  Running on TPU:           {'PASS' if check_tpu else 'SKIP (CPU)'}")

    check_precision = results["quick_err_jax"] < 1e-14
    print(f"  FP64-level precision:     {'PASS' if check_precision else 'FAIL'} "
          f"({results['quick_err_jax']:.2e} < 1e-14)")
    all_pass &= check_precision

    check_match = results["quick_err_diff"] < 1e-15
    print(f"  JAX matches numpy:        {'PASS' if check_match else 'FAIL'} "
          f"({results['quick_err_diff']:.2e})")
    all_pass &= check_match

    check_large = results["large_K_error"] < 1e-14
    print(f"  Large K (1024) accurate:  {'PASS' if check_large else 'FAIL'} "
          f"({results['large_K_error']:.2e})")
    all_pass &= check_large

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    results["all_pass"] = all_pass
    print()

    # Save results to JSON
    out_path = Path(__file__).resolve().parent / "tpu_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
