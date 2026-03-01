"""FP64-accurate matrix multiplication via Ozaki Extract.

Provides `matmul` (JAX backend) and `matmul_numpy` (NumPy-only fallback).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .extract import _compute_rho, extract_split_rows, extract_split_cols


@functools.partial(jax.jit, static_argnums=(2,))
def _ozaki_gemms_jit(A_stack, B_stack, n_slices):
    """Compute triangular slice-pair GEMMs and stack results."""
    results = []
    for i in range(n_slices):
        for j in range(n_slices):
            if i + j > n_slices - 1:
                continue
            results.append(jnp.dot(A_stack[i], B_stack[j]))
    return jnp.stack(results)


def _accumulate_products(products_np, A_scales, B_scales, N, M, n_slices):
    """Rescale normalized products and accumulate in FP64."""
    C = np.zeros((N, M), dtype=np.float64)
    pair_idx = 0
    for i in range(n_slices):
        for j in range(n_slices):
            if i + j > n_slices - 1:
                continue

            P_f64 = products_np[pair_idx]
            row_scale = np.ldexp(np.ones(N, dtype=np.float64),
                                 A_scales[i].astype(np.int64))
            col_scale = np.ldexp(np.ones(M, dtype=np.float64),
                                 B_scales[j].astype(np.int64))
            P_f64 *= row_scale[:, np.newaxis]
            P_f64 *= col_scale[np.newaxis, :]

            C += P_f64
            pair_idx += 1

    return C


def _check_exactness(K, bits_per_slice):
    """Verify the exactness condition for BF16->FP32 inner products."""
    max_product_sum = K * (2**bits_per_slice - 1)**2
    if max_product_sum >= 2**24:
        max_K = int(2**24 / (2**bits_per_slice - 1)**2)
        raise ValueError(
            f"K={K} too large for exact BF16->FP32 inner products "
            f"(need K <= {max_K})")


def matmul(A_f64, B_f64, n_slices=8):
    """Multiply two FP64 matrices using JAX GEMMs over Extract slices."""
    A_f64 = np.asarray(A_f64, dtype=np.float64)
    B_f64 = np.asarray(B_f64, dtype=np.float64)
    N, K = A_f64.shape
    _, M = B_f64.shape

    rho = _compute_rho(K)
    bits_per_slice = 53 - rho
    _check_exactness(K, bits_per_slice)

    # Extract split in FP64 on host.
    A_slices, A_scales = extract_split_rows(A_f64, rho, n_slices)
    B_slices, B_scales = extract_split_cols(B_f64, rho, n_slices)

    # Convert slices for device GEMMs.
    A_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_slices])
    B_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_slices])

    # Compute slice-pair GEMMs on device.
    products = _ozaki_gemms_jit(A_stack, B_stack, n_slices)

    # Rescale and accumulate on host in FP64.
    products_np = np.array(products, dtype=np.float64)
    return _accumulate_products(products_np, A_scales, B_scales, N, M, n_slices)


def matmul_numpy(A_f64, B_f64, n_slices=8):
    """Multiply two FP64 matrices using NumPy-only Extract slice GEMMs."""
    A_f64 = np.asarray(A_f64, dtype=np.float64)
    B_f64 = np.asarray(B_f64, dtype=np.float64)
    N, K = A_f64.shape
    _, M = B_f64.shape

    rho = _compute_rho(K)
    bits_per_slice = 53 - rho
    _check_exactness(K, bits_per_slice)

    # Extract split in FP64.
    A_slices, A_scales = extract_split_rows(A_f64, rho, n_slices)
    B_slices, B_scales = extract_split_cols(B_f64, rho, n_slices)

    # Multiply slice pairs and accumulate in FP64.
    C = np.zeros((N, M), dtype=np.float64)
    for i in range(n_slices):
        for j in range(n_slices):
            if i + j > n_slices - 1:
                continue

            a_f32 = A_slices[i].astype(np.float32)
            b_f32 = B_slices[j].astype(np.float32)
            P = np.float32(a_f32 @ b_f32)

            # Restore per-row/per-column scales in FP64.
            P_f64 = P.astype(np.float64)
            row_scale = np.ldexp(np.ones(N, dtype=np.float64),
                                 A_scales[i].astype(np.int64))
            col_scale = np.ldexp(np.ones(M, dtype=np.float64),
                                 B_scales[j].astype(np.int64))
            P_f64 *= row_scale[:, np.newaxis]
            P_f64 *= col_scale[np.newaxis, :]

            C += P_f64

    return C
