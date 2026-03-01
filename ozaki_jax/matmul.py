"""FP64-accurate matrix multiplication via Ozaki Extract.

Provides `matmul` (JAX backend) and `matmul_numpy` (NumPy-only fallback).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .extract import (
    _compute_rho, extract_split_rows, extract_split_cols,
    _compute_rho_f32, f32_extract_split_rows, f32_extract_split_cols,
)
from .pallas_ops import _precompute_accumulation_scales, accumulate_2sum


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
    """Rescale normalized products and accumulate in FP64.

    Uses factored row-scaling: inner-sum col-scaled products, then apply
    row-scale once per group.  Host extraction uses FP64 exponents that can
    exceed FP32 range, so all scaling stays in FP64.
    """
    row_sc = [np.ldexp(np.ones(N, np.float64), A_scales[i].astype(np.int64))
              for i in range(n_slices)]
    col_sc = [np.ldexp(np.ones(M, np.float64), B_scales[j].astype(np.int64))
              for j in range(n_slices)]

    C = np.zeros((N, M), dtype=np.float64)
    inner = np.zeros((N, M), dtype=np.float64)
    buf = np.empty((N, M), dtype=np.float64)
    pair_idx = 0

    for i in range(n_slices):
        inner[:] = 0.0
        for j in range(n_slices):
            if i + j > n_slices - 1:
                continue
            np.copyto(buf, products_np[pair_idx], casting="unsafe")
            buf *= col_sc[j][np.newaxis, :]
            inner += buf
            pair_idx += 1
        inner *= row_sc[i][:, np.newaxis]
        C += inner

    return C


def _validate_shapes(A_f64, B_f64):
    """Validate matrix ranks and compatibility."""
    if A_f64.ndim != 2 or B_f64.ndim != 2:
        raise ValueError("Inputs must be rank-2 matrices.")
    if A_f64.shape[1] != B_f64.shape[0]:
        raise ValueError(
            f"Incompatible shapes for matmul: {A_f64.shape} and {B_f64.shape}."
        )


def _ozaki_safety_report(A_f64, B_f64, n_slices):
    """Return safety checks for the host Extract pipeline."""
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1.")

    _validate_shapes(A_f64, B_f64)
    K = A_f64.shape[1]
    rho = _compute_rho(K)
    bits_per_slice = 53 - rho

    reasons = []

    if bits_per_slice <= 0:
        reasons.append(
            f"non-positive bits per slice ({bits_per_slice}); K={K} is too large "
            "for this BF16->FP32 exactness model"
        )
    else:
        max_product_sum = K * (2**bits_per_slice - 1)**2
        if max_product_sum >= 2**24:
            max_K = int(2**24 / (2**bits_per_slice - 1)**2)
            reasons.append(
                f"K={K} violates BF16->FP32 exact inner-product bound (need K <= {max_K})"
            )

    if n_slices * max(bits_per_slice, 0) < 53:
        reasons.append(
            f"insufficient slice budget: n_slices*bits_per_slice="
            f"{n_slices * max(bits_per_slice, 0)} < 53"
        )

    if not np.isfinite(A_f64).all() or not np.isfinite(B_f64).all():
        reasons.append("inputs contain NaN or Inf")

    return {
        "safe": len(reasons) == 0,
        "rho": rho,
        "bits_per_slice": bits_per_slice,
        "reasons": reasons,
    }


def _handle_unsafe_preflight(A_f64, B_f64, report, safe_mode):
    """Handle unsafe inputs by raising or falling back to `A @ B`."""
    allowed_modes = {"raise", "fallback"}
    if safe_mode not in allowed_modes:
        raise ValueError(
            f"Invalid safe_mode={safe_mode!r}; expected one of {sorted(allowed_modes)}."
        )

    if report["safe"]:
        return None

    message = (
        "Ozaki preflight marked this input as unsafe: " + "; ".join(report["reasons"])
    )
    if safe_mode == "fallback":
        return A_f64 @ B_f64
    raise ValueError(message)


# On-device pipeline configuration.

# Fixed config for on-device pipeline
_ONDEVICE_N_HI = 5
_ONDEVICE_N_LO = 4


def _double_f32_split(X_f64):
    """Split FP64 values into `hi + lo` FP32 terms."""
    hi = np.float32(X_f64)
    lo = np.float32(X_f64 - np.float64(hi))
    return hi, lo


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def _ondevice_gemms_jit(A_hi_stack, A_lo_stack, B_hi_stack, B_lo_stack,
                        n_hi, n_lo, n_hi_b):
    """Compute hi-hi, hi-lo, and lo-hi GEMM blocks."""
    results = []
    # hi × hi
    for i in range(n_hi):
        for j in range(n_hi_b):
            results.append(jnp.dot(A_hi_stack[i], B_hi_stack[j]))
    # hi × lo
    for i in range(n_hi):
        for j in range(n_lo):
            results.append(jnp.dot(A_hi_stack[i], B_lo_stack[j]))
    # lo × hi
    for i in range(n_lo):
        for j in range(n_hi_b):
            results.append(jnp.dot(A_lo_stack[i], B_hi_stack[j]))
    return jnp.stack(results)


def _accumulate_block_products(products_np, A_hi_scales, A_lo_scales,
                               B_hi_scales, B_lo_scales, N, M,
                               n_hi, n_lo, n_hi_b):
    """Rescale block products and accumulate in FP64.

    Uses prescale-factored strategy: col-scale in FP32 (exact power-of-2
    shift), inner-sum in FP64, factored row-scale in FP64 (14 muls
    instead of 65).
    """
    # Precompute scale vectors.
    row_hi = [np.ldexp(np.ones(N, np.float64), A_hi_scales[i].astype(np.int64))
              for i in range(n_hi)]
    row_lo = [np.ldexp(np.ones(N, np.float64), A_lo_scales[i].astype(np.int64))
              for i in range(n_lo)]
    col_hi = [np.ldexp(np.ones(M, np.float32), B_hi_scales[j].astype(np.int32))
              for j in range(n_hi_b)]
    col_lo = [np.ldexp(np.ones(M, np.float32), B_lo_scales[j].astype(np.int32))
              for j in range(n_lo)]

    C = np.zeros((N, M), dtype=np.float64)
    inner = np.zeros((N, M), dtype=np.float64)
    buf = np.empty((N, M), dtype=np.float32)
    idx = 0

    # hi × hi: col-scale in FP32, inner-sum in FP64, row-scale in FP64.
    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_hi_b):
            np.multiply(products_np[idx], col_hi[j][np.newaxis, :], out=buf)
            inner += buf  # FP32→FP64 auto-promotion in addition
            idx += 1
        inner *= row_hi[i][:, np.newaxis]
        C += inner

    # hi × lo
    for i in range(n_hi):
        inner[:] = 0.0
        for j in range(n_lo):
            np.multiply(products_np[idx], col_lo[j][np.newaxis, :], out=buf)
            inner += buf
            idx += 1
        inner *= row_hi[i][:, np.newaxis]
        C += inner

    # lo × hi
    for i in range(n_lo):
        inner[:] = 0.0
        for j in range(n_hi_b):
            np.multiply(products_np[idx], col_hi[j][np.newaxis, :], out=buf)
            inner += buf
            idx += 1
        inner *= row_lo[i][:, np.newaxis]
        C += inner

    return C


def _ondevice_safety_report(A_f64, B_f64):
    """Return safety checks for the on-device pipeline."""
    _validate_shapes(A_f64, B_f64)
    K = A_f64.shape[1]
    rho = _compute_rho_f32(K)
    bits_per_slice = 24 - rho  # FP32 mantissa = 24

    reasons = []

    if bits_per_slice <= 0:
        reasons.append(
            f"non-positive bits per slice ({bits_per_slice}); K={K} is too "
            "large for FP32 extraction"
        )
    else:
        max_product_sum = K * (2**bits_per_slice - 1)**2
        if max_product_sum >= 2**24:
            max_K = int(2**24 / (2**bits_per_slice - 1)**2)
            reasons.append(
                f"K={K} violates BF16->FP32 exact inner-product bound "
                f"(need K <= {max_K})"
            )

    n_hi = _ONDEVICE_N_HI
    if n_hi * max(bits_per_slice, 0) < 24:
        reasons.append(
            f"insufficient hi slice budget: {n_hi}*{max(bits_per_slice, 0)}"
            f"={n_hi * max(bits_per_slice, 0)} < 24"
        )

    if not np.isfinite(A_f64).all() or not np.isfinite(B_f64).all():
        reasons.append("inputs contain NaN or Inf")

    return {
        "safe": len(reasons) == 0,
        "rho": rho,
        "bits_per_slice": bits_per_slice,
        "n_hi": _ONDEVICE_N_HI,
        "n_lo": _ONDEVICE_N_LO,
        "n_gemms": (_ONDEVICE_N_HI * _ONDEVICE_N_HI +
                    _ONDEVICE_N_HI * _ONDEVICE_N_LO +
                    _ONDEVICE_N_LO * _ONDEVICE_N_HI),
        "reasons": reasons,
    }


def _matmul_ondevice(A_f64, B_f64, safe_mode, accumulation="host"):
    """On-device path: FP32 extraction and fixed GEMM blocks.

    Args:
        accumulation: 'host' (FP64 on CPU) or 'ondevice' (2Sum on device)
    """
    report = _ondevice_safety_report(A_f64, B_f64)
    fallback = _handle_unsafe_preflight(A_f64, B_f64, report, safe_mode)
    if fallback is not None:
        return fallback

    N = A_f64.shape[0]
    M = B_f64.shape[1]
    rho = report["rho"]
    n_hi = report["n_hi"]
    n_lo = report["n_lo"]

    # 1) Double-FP32 split.
    A_hi, A_lo = _double_f32_split(A_f64)
    B_hi, B_lo = _double_f32_split(B_f64)

    # 2) FP32 Extract split.
    A_hi_slices, A_hi_scales = f32_extract_split_rows(A_hi, rho, n_hi)
    A_lo_slices, A_lo_scales = f32_extract_split_rows(A_lo, rho, n_lo)
    B_hi_slices, B_hi_scales = f32_extract_split_cols(B_hi, rho, n_hi)
    B_lo_slices, B_lo_scales = f32_extract_split_cols(B_lo, rho, n_lo)

    # 3) Stack and dispatch GEMMs.
    A_hi_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_hi_slices])
    A_lo_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in A_lo_slices])
    B_hi_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_hi_slices])
    B_lo_stack = jnp.stack([jnp.float32(jnp.array(s)) for s in B_lo_slices])

    products = _ondevice_gemms_jit(A_hi_stack, A_lo_stack,
                                   B_hi_stack, B_lo_stack,
                                   n_hi, n_lo, n_hi)

    # 4) Accumulate.
    if accumulation == "ondevice":
        # Products stay on device; precompute scales and send to device.
        col_scales, row_scales, block_group_sizes = _precompute_accumulation_scales(
            A_hi_scales, A_lo_scales, B_hi_scales, B_lo_scales,
            N, M, n_hi, n_lo)
        col_scales_jax = jnp.array(col_scales)
        row_scales_jax = jnp.array(row_scales)

        C_hi, C_lo = accumulate_2sum(products, col_scales_jax,
                                      row_scales_jax, block_group_sizes)
        # Transfer only 2×N×M back and combine in FP64.
        return np.float64(np.array(C_hi)) + np.float64(np.array(C_lo))

    # Default: host accumulation in FP64.
    products_np = np.array(products, dtype=np.float64)
    return _accumulate_block_products(
        products_np, A_hi_scales, A_lo_scales,
        B_hi_scales, B_lo_scales, N, M, n_hi, n_lo, n_hi)


def _matmul_ondevice_numpy(A_f64, B_f64, safe_mode):
    """On-device path implemented with NumPy GEMMs for testing."""
    report = _ondevice_safety_report(A_f64, B_f64)
    fallback = _handle_unsafe_preflight(A_f64, B_f64, report, safe_mode)
    if fallback is not None:
        return fallback

    N = A_f64.shape[0]
    M = B_f64.shape[1]
    rho = report["rho"]
    n_hi = report["n_hi"]
    n_lo = report["n_lo"]

    # 1) Double-FP32 split.
    A_hi, A_lo = _double_f32_split(A_f64)
    B_hi, B_lo = _double_f32_split(B_f64)

    # 2) FP32 Extract split.
    A_hi_slices, A_hi_scales = f32_extract_split_rows(A_hi, rho, n_hi)
    A_lo_slices, A_lo_scales = f32_extract_split_rows(A_lo, rho, n_lo)
    B_hi_slices, B_hi_scales = f32_extract_split_cols(B_hi, rho, n_hi)
    B_lo_slices, B_lo_scales = f32_extract_split_cols(B_lo, rho, n_lo)

    # 3) Compute GEMMs in NumPy.
    products = []

    def _gemm(a, b):
        return np.float32(a.astype(np.float32) @ b.astype(np.float32))

    # hi × hi
    for i in range(n_hi):
        for j in range(n_hi):
            products.append(_gemm(A_hi_slices[i], B_hi_slices[j]))
    # hi × lo
    for i in range(n_hi):
        for j in range(n_lo):
            products.append(_gemm(A_hi_slices[i], B_lo_slices[j]))
    # lo × hi
    for i in range(n_lo):
        for j in range(n_hi):
            products.append(_gemm(A_lo_slices[i], B_hi_slices[j]))

    products_np = np.stack(products)

    # 4) Accumulate in FP64.
    return _accumulate_block_products(
        products_np, A_hi_scales, A_lo_scales,
        B_hi_scales, B_lo_scales, N, M, n_hi, n_lo, n_hi)


def matmul(A_f64, B_f64, n_slices=8, safe_mode="raise", pipeline="host",
           accumulation="host"):
    """FP64 matmul via Ozaki Extract.

    Args:
        pipeline: 'host' or 'ondevice'
        accumulation: 'host' (FP64 on CPU) or 'ondevice' (2Sum on device).
            Only used when pipeline='ondevice'; ignored for host pipeline.
    """
    A_f64 = np.asarray(A_f64, dtype=np.float64)
    B_f64 = np.asarray(B_f64, dtype=np.float64)

    if pipeline == "ondevice":
        return _matmul_ondevice(A_f64, B_f64, safe_mode, accumulation)
    if pipeline != "host":
        raise ValueError(f"Unknown pipeline={pipeline!r}; expected 'host' or 'ondevice'.")

    report = _ozaki_safety_report(A_f64, B_f64, n_slices)
    fallback = _handle_unsafe_preflight(A_f64, B_f64, report, safe_mode)
    if fallback is not None:
        return fallback
    N = A_f64.shape[0]
    M = B_f64.shape[1]
    rho = report["rho"]

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


def matmul_numpy(A_f64, B_f64, n_slices=8, safe_mode="raise", pipeline="host"):
    """NumPy implementation of `matmul` with matching pipeline options."""
    A_f64 = np.asarray(A_f64, dtype=np.float64)
    B_f64 = np.asarray(B_f64, dtype=np.float64)

    if pipeline == "ondevice":
        return _matmul_ondevice_numpy(A_f64, B_f64, safe_mode)
    if pipeline != "host":
        raise ValueError(f"Unknown pipeline={pipeline!r}; expected 'host' or 'ondevice'.")

    report = _ozaki_safety_report(A_f64, B_f64, n_slices)
    fallback = _handle_unsafe_preflight(A_f64, B_f64, report, safe_mode)
    if fallback is not None:
        return fallback
    N = A_f64.shape[0]
    M = B_f64.shape[1]
    rho = report["rho"]

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
