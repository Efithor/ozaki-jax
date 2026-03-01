#!/usr/bin/env python3
"""Self-contained TPU benchmark for the on-device pipeline.

Designed to run on a fresh TPU VM or Kaggle/Colab TPU.
Installs ozaki-jax from local source, runs all benchmarks, prints results.

Usage (TPU VM):
    pip install -e /path/to/ozaki-jax && python tpu_ondevice_bench.py

Usage (Colab/Kaggle):
    !pip install git+https://github.com/efithor/ozaki-jax.git
    %run tpu_ondevice_bench.py
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Verify TPU.
devices = jax.devices()
platform = jax.default_backend()
print(f"JAX {jax.__version__}")
print(f"Devices: {devices}")
print(f"Platform: {platform}")
if platform != "tpu":
    print("WARNING: Not on TPU. Results will not reflect TPU performance.")
print()

from ozaki_jax import matmul, matmul_numpy
from ozaki_jax.extract import _compute_rho, _compute_rho_f32
from ozaki_jax.matmul import _double_f32_split, _ONDEVICE_N_HI, _ONDEVICE_N_LO

results = {"platform": platform, "jax_version": jax.__version__,
           "device_kind": devices[0].device_kind}

rng = np.random.RandomState(42)

# === 1. Accuracy comparison ===
print("=" * 60)
print("1. ACCURACY: host vs on-device")
print("=" * 60)

for n in [64, 128, 256, 512]:
    errs = {"host": [], "ondevice": [], "match": []}
    for _ in range(5):
        scale = 10.0 ** rng.uniform(-3, 3)
        A = rng.randn(n, n).astype(np.float64) * scale
        B = rng.randn(n, n).astype(np.float64) * scale
        C_ref = A @ B
        ref_norm = np.linalg.norm(C_ref)
        if ref_norm == 0:
            continue

        C_h = matmul(A, B, pipeline="host")
        C_o = matmul(A, B, pipeline="ondevice")
        C_on = matmul_numpy(A, B, pipeline="ondevice")

        errs["host"].append(np.linalg.norm(C_h - C_ref) / ref_norm)
        errs["ondevice"].append(np.linalg.norm(C_o - C_ref) / ref_norm)
        errs["match"].append(np.max(np.abs(C_o - C_on)))

    eh = np.mean(errs["host"])
    eo = np.mean(errs["ondevice"])
    em = np.max(errs["match"])
    ratio = eo / eh if eh > 0 else float("inf")
    print(f"  n={n:>4}: host={eh:.2e}  ondevice={eo:.2e}  "
          f"ratio={ratio:.1f}x  jax==numpy={em:.0e}")
    results[f"acc_{n}"] = {"host": float(eh), "ondevice": float(eo),
                           "ratio": float(ratio), "match": float(em)}
print()

# === 2. Timing: full pipeline ===
print("=" * 60)
print("2. TIMING: full pipeline (median of 10 calls after warmup)")
print("=" * 60)

for n in [128, 256, 512]:
    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)

    # Warmup (3 calls each).
    for _ in range(3):
        C = matmul(A, B, pipeline="host")
        if hasattr(C, 'block_until_ready'):
            C.block_until_ready()
        C = matmul(A, B, pipeline="ondevice")
        if hasattr(C, 'block_until_ready'):
            C.block_until_ready()

    # Measure host.
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        C = matmul(A, B, pipeline="host")
        if hasattr(C, 'block_until_ready'):
            C.block_until_ready()
        times.append(time.perf_counter() - t0)
    t_host = np.median(times)

    # Measure on-device.
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        C = matmul(A, B, pipeline="ondevice")
        if hasattr(C, 'block_until_ready'):
            C.block_until_ready()
        times.append(time.perf_counter() - t0)
    t_ondev = np.median(times)

    # Measure raw FP64.
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        C = A @ B
        times.append(time.perf_counter() - t0)
    t_f64 = np.median(times)

    speedup = t_host / t_ondev if t_ondev > 0 else float("inf")
    print(f"  n={n:>4}: host={t_host*1000:>8.2f}ms  "
          f"ondevice={t_ondev*1000:>8.2f}ms  "
          f"FP64={t_f64*1000:>8.2f}ms  "
          f"speedup={speedup:.1f}x")
    results[f"time_{n}"] = {
        "host_ms": float(t_host * 1000),
        "ondevice_ms": float(t_ondev * 1000),
        "f64_ms": float(t_f64 * 1000),
        "speedup": float(speedup),
    }
print()

# === 3. Timing: stage breakdown ===
print("=" * 60)
print("3. TIMING: stage breakdown (n=256)")
print("=" * 60)

from ozaki_jax.extract import extract_split_rows, extract_split_cols
from ozaki_jax.extract import f32_extract_split_rows, f32_extract_split_cols
from ozaki_jax.matmul import (
    _ozaki_gemms_jit, _ondevice_gemms_jit,
    _accumulate_products, _accumulate_block_products,
)

n = 256
A = rng.randn(n, n).astype(np.float64)
B = rng.randn(n, n).astype(np.float64)
rho64 = _compute_rho(n)
rho32 = _compute_rho_f32(n)

# Warmup.
_ = matmul(A, B, pipeline="host")
_ = matmul(A, B, pipeline="ondevice")

def time_fn(fn, n_calls=10):
    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

# Host stages.
def host_extract():
    return (extract_split_rows(A, rho64, 8), extract_split_cols(B, rho64, 8))

A_sl, A_sc = extract_split_rows(A, rho64, 8)
B_sl, B_sc = extract_split_cols(B, rho64, 8)
A_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_sl])
B_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_sl])

def host_gemms():
    return _ozaki_gemms_jit(A_stack, B_stack, 8)

prods = np.array(_ozaki_gemms_jit(A_stack, B_stack, 8), dtype=np.float64)

def host_accum():
    return _accumulate_products(prods, A_sc, B_sc, n, n, 8)

h_ext = time_fn(host_extract)
h_gem = time_fn(host_gemms)
h_acc = time_fn(host_accum)

# On-device stages.
A_hi, A_lo = _double_f32_split(A)
B_hi, B_lo = _double_f32_split(B)
n_hi, n_lo = _ONDEVICE_N_HI, _ONDEVICE_N_LO

def ondev_split():
    return _double_f32_split(A), _double_f32_split(B)

def ondev_extract():
    return (f32_extract_split_rows(A_hi, rho32, n_hi),
            f32_extract_split_rows(A_lo, rho32, n_lo),
            f32_extract_split_cols(B_hi, rho32, n_hi),
            f32_extract_split_cols(B_lo, rho32, n_lo))

Ahi_sl, Ahi_sc = f32_extract_split_rows(A_hi, rho32, n_hi)
Alo_sl, Alo_sc = f32_extract_split_rows(A_lo, rho32, n_lo)
Bhi_sl, Bhi_sc = f32_extract_split_cols(B_hi, rho32, n_hi)
Blo_sl, Blo_sc = f32_extract_split_cols(B_lo, rho32, n_lo)

Ahi_stk = jnp.stack([jnp.float32(jnp.array(s)) for s in Ahi_sl])
Alo_stk = jnp.stack([jnp.float32(jnp.array(s)) for s in Alo_sl])
Bhi_stk = jnp.stack([jnp.float32(jnp.array(s)) for s in Bhi_sl])
Blo_stk = jnp.stack([jnp.float32(jnp.array(s)) for s in Blo_sl])

def ondev_gemms():
    return _ondevice_gemms_jit(Ahi_stk, Alo_stk, Bhi_stk, Blo_stk,
                               n_hi, n_lo, n_hi)

oprods = np.array(ondev_gemms(), dtype=np.float64)

def ondev_accum():
    return _accumulate_block_products(oprods, Ahi_sc, Alo_sc,
                                      Bhi_sc, Blo_sc, n, n, n_hi, n_lo, n_hi)

o_spl = time_fn(ondev_split)
o_ext = time_fn(ondev_extract)
o_gem = time_fn(ondev_gemms)
o_acc = time_fn(ondev_accum)

print(f"  {'Stage':<20} {'Host (ms)':>10} {'OnDevice (ms)':>14} {'Speedup':>8}")
print(f"  {'-'*20} {'-'*10} {'-'*14} {'-'*8}")
print(f"  {'Extraction':<20} {h_ext:>10.2f} {o_spl+o_ext:>14.2f} "
      f"{h_ext/(o_spl+o_ext):>7.1f}x")
print(f"    {'(FP32 split)':<18} {'':>10} {o_spl:>14.2f}")
print(f"    {'(sigma trick)':<18} {'':>10} {o_ext:>14.2f}")
print(f"  {'GEMMs (36 vs 65)':<20} {h_gem:>10.2f} {o_gem:>14.2f} "
      f"{h_gem/o_gem if o_gem > 0 else 0:>7.1f}x")
print(f"  {'Accumulation':<20} {h_acc:>10.2f} {o_acc:>14.2f} "
      f"{h_acc/o_acc if o_acc > 0 else 0:>7.1f}x")
h_tot = h_ext + h_gem + h_acc
o_tot = o_spl + o_ext + o_gem + o_acc
print(f"  {'TOTAL':<20} {h_tot:>10.2f} {o_tot:>14.2f} {h_tot/o_tot:>7.1f}x")
print(f"\n  Host bottleneck:     extraction = {h_ext/h_tot*100:.0f}% of total")
print(f"  OnDevice bottleneck: extraction = {(o_spl+o_ext)/o_tot*100:.0f}% of total")

results["breakdown_256"] = {
    "host": {"extract_ms": h_ext, "gemms_ms": h_gem, "accum_ms": h_acc},
    "ondevice": {"split_ms": o_spl, "extract_ms": o_ext,
                 "gemms_ms": o_gem, "accum_ms": o_acc},
}
print()

# === 4. BF16 exactness + Pallas ===
print("=" * 60)
print("4. CORRECTNESS: BF16 exactness + sigma trick rounding")
print("=" * 60)

X = rng.randn(128, 128).astype(np.float32)
rho = _compute_rho_f32(128)
slices, _ = f32_extract_split_rows(X, rho, n_slices=5)
all_exact = True
for i, s in enumerate(slices):
    bf16_rt = np.array(jnp.float32(jnp.bfloat16(jnp.float32(s))))
    err = np.max(np.abs(s - bf16_rt))
    ok = err == 0.0
    all_exact &= ok
    print(f"  Slice {i} BF16 roundtrip: {err:.0e} [{'PASS' if ok else 'FAIL'}]")

try:
    from ozaki_jax.pallas_ops import validate_sigma_trick_rounding
    r = validate_sigma_trick_rounding(K=256)
    if r.get("match") is not None:
        print(f"  JAX sigma trick: max_diff={r['max_diff']:.0e} "
              f"[{'PASS' if r['match'] else 'FAIL'}]")
    if "pallas_match" in r:
        print(f"  Pallas sigma trick: max_diff={r['pallas_max_diff']:.0e} "
              f"[{'PASS' if r['pallas_match'] else 'FAIL'}]")
    elif "pallas_error" in r:
        print(f"  Pallas: skipped ({r['pallas_error']})")
except Exception as e:
    print(f"  Pallas: {e}")

results["bf16_exact"] = bool(all_exact)
print()

# === 5. Summary ===
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Platform:       {platform}")
print(f"  Device:         {devices[0].device_kind}")
for n in [128, 256, 512]:
    k = f"time_{n}"
    if k in results:
        r = results[k]
        print(f"  n={n}: {r['host_ms']:.1f}ms -> {r['ondevice_ms']:.1f}ms "
              f"({r['speedup']:.1f}x speedup)")
print(f"  BF16-exact:     {results['bf16_exact']}")
print()

# Save JSON.
with open("tpu_ondevice_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to tpu_ondevice_results.json")
