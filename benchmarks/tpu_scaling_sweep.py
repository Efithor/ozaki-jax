"""Scaling sweep: effective FP64 TFLOPS vs matrix size on TPU v6e."""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax import matmul
from ozaki_jax.matmul import _resolve_precision

V6E_BF16_TFLOPS = 918.0

SIZES = [256, 512, 1024, 2048, 4096, 8192]
PRESETS = ["medium", "high"]
N_WARMUP = 3
N_TIMED = 20


def bench_one(n, precision):
    """Benchmark a single (size, precision) configuration."""
    n_hi, n_lo = _resolve_precision(precision)
    n_gemms = n_hi * n_hi + n_hi * n_lo + n_lo * n_hi

    # Generate on host, transfer once.
    rng = np.random.RandomState(42)
    A_np = rng.randn(n, n).astype(np.float64)
    B_np = rng.randn(n, n).astype(np.float64)

    A = jnp.asarray(A_np, dtype=jnp.float64)
    B = jnp.asarray(B_np, dtype=jnp.float64)

    # JIT warmup.
    for _ in range(N_WARMUP):
        C = matmul(A, B, pipeline="ondevice", precision=precision)
        C.block_until_ready()

    # Timed runs.
    times = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        C = matmul(A, B, pipeline="ondevice", precision=precision)
        C.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_s = float(np.median(times))
    min_s = float(np.min(times))
    flops_fp64 = 2.0 * n ** 3

    eff_tflops = flops_fp64 / median_s / 1e12
    bf16_throughput = n_gemms * flops_fp64 / median_s / 1e12
    mxu_util = bf16_throughput / V6E_BF16_TFLOPS * 100

    # Accuracy check.
    C_ref = A_np @ B_np
    C_out = np.asarray(C)
    rel_err = float(np.max(np.abs(C_out - C_ref)) / np.max(np.abs(C_ref)))
    digits = -np.log10(max(rel_err, 1e-16))

    return {
        "n": n,
        "precision": precision,
        "n_hi": n_hi,
        "n_lo": n_lo,
        "n_gemms": n_gemms,
        "median_ms": median_s * 1000,
        "min_ms": min_s * 1000,
        "eff_fp64_tflops": round(eff_tflops, 3),
        "bf16_tflops": round(bf16_throughput, 1),
        "mxu_util_pct": round(mxu_util, 1),
        "digits": round(digits, 1),
        "rel_err": rel_err,
    }


def main():
    platform = jax.devices()[0].platform
    device_kind = jax.devices()[0].device_kind
    print(f"Device: {device_kind} ({platform})")
    print(f"Warmup: {N_WARMUP}  Timed: {N_TIMED}")
    print()

    results = []
    for n in SIZES:
        for prec in PRESETS:
            label = f"n={n:5d}  {prec:7s}"
            print(f"  {label} ... ", end="", flush=True)
            try:
                r = bench_one(n, prec)
                results.append(r)
                print(
                    f"{r['median_ms']:8.2f} ms  "
                    f"{r['eff_fp64_tflops']:7.2f} TFLOPS  "
                    f"MXU {r['mxu_util_pct']:5.1f}%  "
                    f"{r['digits']:.1f} digits"
                )
            except Exception as e:
                print(f"FAILED: {e}")

    # Summary table.
    print()
    print("=" * 90)
    print(
        f"{'n':>6} {'preset':>8} {'GEMMs':>5} {'ms':>9} "
        f"{'FP64 TFLOPS':>12} {'MXU %':>6} {'digits':>6}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['n']:>6} {r['precision']:>8} {r['n_gemms']:>5} "
            f"{r['median_ms']:>9.2f} {r['eff_fp64_tflops']:>12.3f} "
            f"{r['mxu_util_pct']:>6.1f} {r['digits']:>6.1f}"
        )
    print("=" * 90)

    # Dump JSON for archiving.
    out = {
        "device": device_kind,
        "platform": platform,
        "v6e_bf16_tflops": V6E_BF16_TFLOPS,
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "results": results,
    }
    with open("scaling_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nJSON written to scaling_sweep.json")


if __name__ == "__main__":
    main()
