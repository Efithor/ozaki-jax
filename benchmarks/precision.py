"""Compare matmul variants against FP64 reference results.

Usage:
    python benchmarks/precision.py
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Library imports
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from ozaki_jax import matmul, matmul_numpy
from ozaki_jax.extract import _compute_rho


# Comparison methods used only in this benchmark script.

def split_f64_to_bf16(X_f64, n_slices=8):
    """Split FP64 into BF16 slices via iterative truncation."""
    slices = []
    residual = np.float64(X_f64)
    for _ in range(n_slices):
        hi_f32 = residual.astype(np.float32)
        hi_bf16 = np.array(jnp.bfloat16(jnp.float32(hi_f32)))
        residual = residual - hi_bf16.astype(np.float64)
        slices.append(hi_bf16)
    return slices


def split_f32_to_bf16(X_f32, n_slices=4):
    """Split FP32 into BF16 slices."""
    slices = []
    residual = jnp.float32(X_f32)
    for _ in range(n_slices):
        hi = jnp.bfloat16(residual)
        residual = residual - jnp.float32(hi)
        slices.append(hi)
    return slices


def twosum_f32(a, b):
    """TwoSum: s + e = a + b exactly (in FP32 arithmetic)."""
    s = np.float32(a + b)
    v = np.float32(s - a)
    e = np.float32(np.float32(a - np.float32(s - v)) + np.float32(b - v))
    return s, e


def bf16_gemm(A_f32, B_f32):
    """Simulate TPU's native BF16 matmul: truncate to BF16, accumulate in FP32."""
    a = jnp.float32(jnp.bfloat16(A_f32))
    b = jnp.float32(jnp.bfloat16(B_f32))
    return jnp.dot(a, b)


@jax.jit
def ozaki_f32(A, B):
    """FP32-accurate matmul using only BF16 GEMMs (16 GEMMs)."""
    A_slices = split_f32_to_bf16(A, n_slices=4)
    B_slices = split_f32_to_bf16(B, n_slices=4)
    C = jnp.zeros((A.shape[0], B.shape[1]), dtype=jnp.float32)
    for a_i in A_slices:
        for b_j in B_slices:
            C = C + jnp.dot(jnp.float32(a_i), jnp.float32(b_j))
    return C


def ozaki_dgemm(A_f64, B_f64, n_slices=8, block_k=1):
    """FP64 matmul via simple FP64->BF16 splitting + TwoSum accumulation."""
    N, K = A_f64.shape
    _, M = B_f64.shape

    A_slices = split_f64_to_bf16(A_f64, n_slices)
    B_slices = split_f64_to_bf16(B_f64, n_slices)

    C_hi = np.zeros((N, M), dtype=np.float32)
    C_lo = np.zeros((N, M), dtype=np.float32)

    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)
        for a_slice in A_slices:
            ab = np.array(jnp.float32(jnp.bfloat16(jnp.float32(
                a_slice[:, k_start:k_end]))))
            for b_slice in B_slices:
                bb = np.array(jnp.float32(jnp.bfloat16(jnp.float32(
                    b_slice[k_start:k_end, :]))))
                p = np.float32(ab @ bb)
                C_hi, e = twosum_f32(C_hi, p)
                C_lo = np.float32(C_lo + e)

    C_hi, e = twosum_f32(C_hi, C_lo)
    return C_hi.astype(np.float64) + e.astype(np.float64)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(sizes=None, n_trials=5, seed=42):
    """Compare all precision levels against FP64 ground truth."""
    if sizes is None:
        sizes = [64, 128, 256]

    rng = np.random.RandomState(seed)
    results = {}

    for n in sizes:
        errors = {"bf16": [], "ozaki_f32": [], "simple_bk1": [],
                  "extract_np": [], "extract_jax": []}

        for trial in range(n_trials):
            scale = 10.0 ** rng.uniform(-3, 3)
            A = rng.randn(n, n).astype(np.float64) * scale
            B = rng.randn(n, n).astype(np.float64) * scale
            C_ref = A @ B
            ref_norm = np.linalg.norm(C_ref)
            if ref_norm == 0:
                continue

            A32, B32 = A.astype(np.float32), B.astype(np.float32)

            # Level 0: Naive BF16
            C_bf16 = np.array(bf16_gemm(jnp.float32(A32), jnp.float32(B32)))
            errors["bf16"].append(
                np.linalg.norm(C_bf16.astype(np.float64) - C_ref) / ref_norm)

            # Level 1: Ozaki BF16->FP32 (16 GEMMs)
            C_f32 = np.array(ozaki_f32(jnp.float32(A32), jnp.float32(B32)))
            errors["ozaki_f32"].append(
                np.linalg.norm(C_f32.astype(np.float64) - C_ref) / ref_norm)

            # Level 2: Simple splitting + TwoSum (64*K GEMMs)
            C_bk1 = ozaki_dgemm(A, B, block_k=1)
            errors["simple_bk1"].append(
                np.linalg.norm(C_bk1 - C_ref) / ref_norm)

            # Level 3: Ozaki Extract numpy (36 GEMMs)
            C_ext = matmul_numpy(A, B)
            errors["extract_np"].append(
                np.linalg.norm(C_ext - C_ref) / ref_norm)

            # Level 3b: Ozaki Extract JAX (36 GEMMs, JIT)
            C_jax = matmul(A, B)
            errors["extract_jax"].append(
                np.linalg.norm(C_jax - C_ref) / ref_norm)

        results[n] = {k: float(np.mean(v)) for k, v in errors.items() if v}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ozaki-jax benchmark")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")

    # Quick Extract parity check (numpy vs JAX).
    print("\nextract parity check (n=64)")
    rng = np.random.RandomState(42)
    A = rng.randn(64, 64).astype(np.float64)
    B = rng.randn(64, 64).astype(np.float64)
    C_ref = A @ B
    ref_norm = np.linalg.norm(C_ref)

    C_np = matmul_numpy(A, B)
    C_jx = matmul(A, B)

    err_np = np.linalg.norm(C_np - C_ref) / ref_norm
    err_jx = np.linalg.norm(C_jx - C_ref) / ref_norm
    err_diff = np.linalg.norm(C_np - C_jx) / ref_norm

    print(f"Extract (numpy): {err_np:.2e}")
    print(f"Extract (JAX):   {err_jx:.2e}")
    print(f"numpy vs JAX:    {err_diff:.2e}")

    rho = _compute_rho(64)
    bits = 53 - rho
    max_sum = 64 * (2**bits - 1)**2
    print(f"\nrho={rho}, bits/slice={bits}, "
          f"K*(2^p-1)^2={max_sum:,} vs 2^24={2**24:,}")

    # Relative error table vs FP64 ground truth.
    print("\nrelative error table")
    results = validate(sizes=[64, 128, 256], n_trials=3)

    cols = ["bf16", "ozaki_f32", "simple_bk1", "extract_np", "extract_jax"]
    labels = ["BF16", "Ozaki->F32", "Simple bk=1", "Extract-np", "Extract-JAX"]
    header = f"{'Size':>6}" + "".join(f"  {l:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for n, r in sorted(results.items()):
        row = f"{n:>6}"
        for c in cols:
            row += f"  {r.get(c, 0):>12.2e}"
        print(row)

    # GEMM count comparison.
    print("\ngemm count comparison")
    for K in [64, 256, 648]:
        rho = _compute_rho(K)
        bits = 53 - rho
        s = 8
        extract_gemms = s * (s + 1) // 2
        simple_gemms = 64 * K
        print(f"K={K:>4}: Extract={extract_gemms} GEMMs, "
              f"Simple={simple_gemms:,} GEMMs, "
              f"ratio={simple_gemms/extract_gemms:.0f}x, "
              f"rho={rho}, bits/slice={bits}, "
              f"exact={'yes' if K*(2**bits-1)**2 < 2**24 else 'NO'}")

    # Timing comparison (numpy vs JAX Extract).
    print("\ntiming (n=256, 5 calls)")
    A = rng.randn(256, 256).astype(np.float64)
    B = rng.randn(256, 256).astype(np.float64)

    # Warm up JIT
    _ = matmul(A, B)

    t0 = time.perf_counter()
    for _ in range(5):
        matmul_numpy(A, B)
    t_np = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        matmul(A, B)
    t_jx = (time.perf_counter() - t0) / 5

    print(f"numpy Extract:  {t_np*1000:.1f} ms/call")
    print(f"JAX Extract:    {t_jx*1000:.1f} ms/call")
    print(f"Ratio:          {t_np/t_jx:.2f}x")

    print("\ncomplete.")
