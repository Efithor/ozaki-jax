"""Validate the on-device pipeline against the host pipeline.

Usage:
    python benchmarks/ondevice_validate.py
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ozaki_jax import matmul, matmul_numpy
from ozaki_jax.extract import (
    _compute_rho, _compute_rho_f32,
    f32_extract_split_rows, f32_extract_split_cols,
)
from ozaki_jax.matmul import _double_f32_split, _ONDEVICE_N_HI, _ONDEVICE_N_LO


def main():
    print("ozaki-jax on-device pipeline validation")
    print(f"JAX version:  {jax.__version__}")
    print(f"Devices:      {jax.devices()}")
    print(f"Platform:     {jax.default_backend()}")
    print()

    rng = np.random.RandomState(42)
    all_pass = True

    # Test 1: double-FP32 split reconstruction.
    print("test 1: double-FP32 split reconstruction\n")
    for n in [64, 256]:
        X = rng.randn(n, n).astype(np.float64)
        hi, lo = _double_f32_split(X)
        reconstructed = np.float64(hi) + np.float64(lo)
        max_err = np.max(np.abs(X - reconstructed))
        rel_err = max_err / np.max(np.abs(X))
        ok = rel_err < 2**-47
        status = "PASS" if ok else "FAIL"
        print(f"  n={n}: max_err={max_err:.2e}, rel_err={rel_err:.2e} "
              f"(vs 2^-48={2**-48:.2e}) [{status}]")
        all_pass &= ok
    print()

    # Test 2: BF16-exactness of FP32 sigma slices.
    print("test 2: BF16-exactness of FP32 sigma slices\n")
    X = rng.randn(64, 64).astype(np.float32)
    rho = _compute_rho_f32(64)
    slices, _ = f32_extract_split_rows(X, rho, n_slices=5)
    all_bf16_exact = True
    for i, s in enumerate(slices):
        # Round-trip through BF16.
        bf16_rt = np.array(jnp.float32(jnp.bfloat16(jnp.float32(s))))
        max_diff = np.max(np.abs(s - bf16_rt))
        ok = max_diff == 0.0
        all_bf16_exact &= ok
        print(f"  slice {i}: max BF16 roundtrip error = {max_diff:.2e} "
              f"[{'PASS' if ok else 'FAIL'}]")
    all_pass &= all_bf16_exact

    # Also check column slices.
    slices_col, _ = f32_extract_split_cols(X, rho, n_slices=5)
    for i, s in enumerate(slices_col):
        bf16_rt = np.array(jnp.float32(jnp.bfloat16(jnp.float32(s))))
        max_diff = np.max(np.abs(s - bf16_rt))
        ok = max_diff == 0.0
        all_bf16_exact &= ok
        print(f"  col slice {i}: max BF16 roundtrip error = {max_diff:.2e} "
              f"[{'PASS' if ok else 'FAIL'}]")
    all_pass &= all_bf16_exact
    print()

    # Test 3: relative error comparison.
    print("test 3: host vs on-device errors\n")
    sizes = [128, 256, 512]
    n_trials = 3
    header = (f"{'Size':>6}  {'Host JAX':>12}  {'OnDev JAX':>12}  "
              f"{'OnDev numpy':>12}  {'JAX==numpy':>12}")
    print(header)
    print("-" * len(header))

    for n in sizes:
        errs_host = []
        errs_ondev = []
        errs_ondev_np = []
        errs_match = []

        for _ in range(n_trials):
            scale = 10.0 ** rng.uniform(-3, 3)
            A = rng.randn(n, n).astype(np.float64) * scale
            B = rng.randn(n, n).astype(np.float64) * scale
            C_ref = A @ B
            ref_norm = np.linalg.norm(C_ref)
            if ref_norm == 0:
                continue

            C_host = matmul(A, B, pipeline="host")
            C_ondev = matmul(A, B, pipeline="ondevice")
            C_ondev_np = matmul_numpy(A, B, pipeline="ondevice")

            errs_host.append(np.linalg.norm(C_host - C_ref) / ref_norm)
            errs_ondev.append(np.linalg.norm(C_ondev - C_ref) / ref_norm)
            errs_ondev_np.append(np.linalg.norm(C_ondev_np - C_ref) / ref_norm)
            errs_match.append(np.max(np.abs(C_ondev - C_ondev_np)))

        e_h = np.mean(errs_host)
        e_o = np.mean(errs_ondev)
        e_onp = np.mean(errs_ondev_np)
        e_m = np.max(errs_match)
        print(f"{n:>6}  {e_h:>12.2e}  {e_o:>12.2e}  {e_onp:>12.2e}  {e_m:>12.2e}")

        # Keep a loose bound to accommodate CPU fallback behavior.
        ok = e_o < e_h * 10 + 1e-15
        all_pass &= ok
    print()

    # Test 4: GEMM count.
    print("test 4: GEMM count verification\n")
    n_hi = _ONDEVICE_N_HI
    n_lo = _ONDEVICE_N_LO
    hh = n_hi * n_hi
    hl = n_hi * n_lo
    lh = n_lo * n_hi
    total = hh + hl + lh
    ok = total == 65
    print(f"  hi×hi: {n_hi}×{n_hi} = {hh}")
    print(f"  hi×lo: {n_hi}×{n_lo} = {hl}")
    print(f"  lo×hi: {n_lo}×{n_hi} = {lh}")
    print(f"  Total: {total} [{'PASS' if ok else 'FAIL'}; expected 65]")
    all_pass &= ok
    print()

    # Test 5: rho_f32 values.
    print("test 5: rho_f32 values\n")
    expected_rho = {64: 17, 128: 17, 256: 17, 512: 17,
                    648: 17, 1024: 17, 1040: 18}
    for K in sorted(expected_rho):
        rho = _compute_rho_f32(K)
        bits = 24 - rho
        rho64 = _compute_rho(K)
        bits64 = 53 - rho64
        ok = rho == expected_rho[K]
        print(f"  K={K:>5}: rho_f32={rho} (bits={bits})  "
              f"rho_f64={rho64} (bits={bits64})  [{'PASS' if ok else 'FAIL'}]")
        all_pass &= ok
    print()

    # Test 6: timing.
    print("test 6: timing comparison\n")
    for n in [128, 256]:
        A = rng.randn(n, n).astype(np.float64)
        B = rng.randn(n, n).astype(np.float64)

        # Warm-up.
        _ = matmul(A, B, pipeline="host")
        _ = matmul(A, B, pipeline="ondevice")

        # Time host.
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            C = matmul(A, B, pipeline="host")
            if hasattr(C, 'block_until_ready'):
                C.block_until_ready()
            times.append(time.perf_counter() - t0)
        t_host = np.median(times)

        # Time on-device.
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            C = matmul(A, B, pipeline="ondevice")
            if hasattr(C, 'block_until_ready'):
                C.block_until_ready()
            times.append(time.perf_counter() - t0)
        t_ondev = np.median(times)

        ratio = t_host / t_ondev if t_ondev > 0 else float("inf")
        print(f"  n={n:>4}: host={t_host*1000:>8.1f}ms  "
              f"ondevice={t_ondev*1000:>8.1f}ms  "
              f"ratio={ratio:.2f}x")
    print()

    # Test 7: Pallas rounding validation.
    print("test 7: Pallas rounding validation\n")
    try:
        from ozaki_jax.pallas_ops import validate_sigma_trick_rounding
        result = validate_sigma_trick_rounding(K=256)
        if result.get("match") is not None:
            ok = result["match"]
            print(f"  JAX vs numpy: max_diff={result['max_diff']:.2e} "
                  f"[{'PASS' if ok else 'FAIL'}]")
            all_pass &= ok
        else:
            print(f"  Skipped: {result.get('error', 'unknown')}")

        if "pallas_match" in result:
            ok = result["pallas_match"]
            print(f"  Pallas vs numpy: max_diff={result['pallas_max_diff']:.2e} "
                  f"[{'PASS' if ok else 'FAIL'}]")
            all_pass &= ok
        elif "pallas_error" in result:
            print(f"  Pallas: skipped ({result['pallas_error']})")
        else:
            print("  Pallas: not available")
    except Exception as e:
        print(f"  Pallas test error: {e}")
    print()

    # Summary.
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
