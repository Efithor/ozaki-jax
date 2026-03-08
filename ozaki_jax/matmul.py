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
    jax_extract_split_rows, jax_extract_split_cols,
)
from .pallas_ops import (
    _precompute_accumulation_scales, accumulate_2sum,
    _accumulate_2sum_logic,
)


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


def _validate_accumulation_mode(accumulation):
    """Validate accumulation mode for on-device pipeline."""
    allowed = {"fused", "bf16_interleaved", "ondevice", "host"}
    if accumulation not in allowed:
        raise ValueError(
            f"Unknown accumulation={accumulation!r}; expected one of {sorted(allowed)}."
        )


def _is_x64_enabled():
    """Return whether JAX x64 mode is enabled."""
    try:
        return bool(jax.config.read("jax_enable_x64"))
    except Exception:
        return bool(getattr(jax.config, "jax_enable_x64", False))


def _require_x64_for_fused():
    """Require JAX x64 for the fully-fused on-device path."""
    if not _is_x64_enabled():
        raise ValueError(
            "accumulation='fused' requires JAX x64 mode. "
            "Enable it before running matmul, e.g. "
            "jax.config.update('jax_enable_x64', True)."
        )


# On-device pipeline precision presets.
# Each maps to (n_hi, n_lo): number of hi/lo extraction slices.
# More slices = more GEMMs = higher accuracy but slower.
_PRECISION_PRESETS = {
    "high":   (4, 1),  # 24 GEMMs
    "medium": (3, 1),  # 15 GEMMs
    "max":    (5, 4),  # 65 GEMMs
}


def _resolve_precision(precision):
    """Resolve precision preset name to (n_hi, n_lo) config.

    Accepts a preset name ('high', 'medium', 'max') or a custom
    (n_hi, n_lo) tuple where both values are >= 1.
    """
    if isinstance(precision, tuple) and len(precision) == 2:
        n_hi, n_lo = precision
        if n_hi < 1 or n_lo < 1:
            raise ValueError(
                f"Custom precision tuple must have n_hi >= 1 and n_lo >= 1, "
                f"got ({n_hi}, {n_lo})."
            )
        return precision
    if precision not in _PRECISION_PRESETS:
        raise ValueError(
            f"Unknown precision={precision!r}; expected one of "
            f"{sorted(_PRECISION_PRESETS)} or a (n_hi, n_lo) tuple."
        )
    return _PRECISION_PRESETS[precision]


def _double_f32_split(X_f64):
    """Split FP64 values into `hi + lo` FP32 terms."""
    hi = np.float32(X_f64)
    lo = np.float32(X_f64 - np.float64(hi))
    return hi, lo


def _jax_double_f32_split(X_f64):
    """Split FP64 values into hi + lo FP32 terms (JAX, no JIT decorator).

    Called inside a larger JIT. On TPU, FP64 is emulated as double-float
    pairs so jnp.float32(X_f64) is essentially extracting the hi part of
    the emulation — very cheap.
    """
    hi = jnp.float32(X_f64)
    lo = jnp.float32(X_f64 - jnp.float64(hi))
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


def _fused_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo,
                          block_group_sizes):
    """Core fused pipeline: extraction + broadcast GEMMs + scales + 2Sum.

    Called by both _fused_ondevice_jit (4 FP32 inputs) and
    _fully_fused_ondevice_jit (2 FP64 inputs, split on device).
    Uses broadcast matmul for better MXU pipelining on TPU.
    """
    # Phase 1: Extraction (on device).
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)

    # Phase 2: Broadcast-batched GEMMs.
    # hi × hi: (n_hi, 1, N, K) @ (1, n_hi, K, M) → (n_hi, n_hi, N, M)
    hh = jnp.matmul(A_hi_sl[:, None, :, :], B_hi_sl[None, :, :, :])
    parts = [hh.reshape(-1, hh.shape[-2], hh.shape[-1])]

    if n_lo > 0:
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

        hl = jnp.matmul(A_hi_sl[:, None, :, :], B_lo_sl[None, :, :, :])
        parts.append(hl.reshape(-1, hl.shape[-2], hl.shape[-1]))
        lh = jnp.matmul(A_lo_sl[:, None, :, :], B_hi_sl[None, :, :, :])
        parts.append(lh.reshape(-1, lh.shape[-2], lh.shape[-1]))

    products = jnp.concatenate(parts, axis=0)

    # Phase 3: Scale precomputation (on device).
    col_parts = [jnp.tile(B_hi_sc, (n_hi, 1))]
    row_parts = [A_hi_sc]

    if n_lo > 0:
        col_parts.append(jnp.tile(B_lo_sc, (n_hi, 1)))
        col_parts.append(jnp.tile(B_hi_sc, (n_lo, 1)))
        row_parts.append(A_hi_sc)
        row_parts.append(A_lo_sc)

    col_exp = jnp.concatenate(col_parts, axis=0)
    row_exp = jnp.concatenate(row_parts, axis=0)
    col_scales = jnp.ldexp(jnp.ones_like(col_exp), jnp.int32(col_exp))
    row_scales = jnp.ldexp(jnp.ones_like(row_exp), jnp.int32(row_exp))

    # Phase 4: 2Sum accumulation.
    return _accumulate_2sum_logic(products, col_scales, row_scales,
                                  block_group_sizes)


def _interleaved_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo):
    """Interleaved pipeline: extract → GEMM → scale → accumulate per-pair.

    Instead of materializing all 24 products then accumulating, computes
    each matmul and immediately 2Sum-accumulates the scaled result.
    This keeps only ONE (N, M) product alive at a time, allowing XLA to
    fuse the matmul→scale→twosum chain without HBM round-trips.
    """
    # Phase 1: Extraction.
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)

    if n_lo > 0:
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

    # Pre-compute scale vectors (FP32 power-of-2).
    # Use ldexp instead of exp2 for exact power-of-2 values.
    A_hi_sc_pow = jnp.ldexp(jnp.ones_like(A_hi_sc), jnp.int32(A_hi_sc))
    B_hi_sc_pow = jnp.ldexp(jnp.ones_like(B_hi_sc), jnp.int32(B_hi_sc))
    if n_lo > 0:
        A_lo_sc_pow = jnp.ldexp(jnp.ones_like(A_lo_sc), jnp.int32(A_lo_sc))
        B_lo_sc_pow = jnp.ldexp(jnp.ones_like(B_lo_sc), jnp.int32(B_lo_sc))

    N = A_hi.shape[0]
    M = B_hi.shape[1]

    def twosum_add(s_hi, s_lo, x):
        t = s_hi + x
        e = (s_hi - t) + x
        return t, s_lo + e

    # Phase 2-4: Interleaved GEMM + scale + 2Sum per group.
    # Block 1: hi × hi
    blk_hh_hi = jnp.zeros((N, M), dtype=jnp.float32)
    blk_hh_lo = jnp.zeros((N, M), dtype=jnp.float32)
    for i in range(n_hi):
        inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
        inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for j in range(n_hi):
            product = jnp.dot(A_hi_sl[i], B_hi_sl[j])
            scaled = product * B_hi_sc_pow[j][jnp.newaxis, :]
            inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
        inner_hi = inner_hi * A_hi_sc_pow[i][:, jnp.newaxis]
        inner_lo = inner_lo * A_hi_sc_pow[i][:, jnp.newaxis]
        blk_hh_hi, blk_hh_lo = twosum_add(blk_hh_hi, blk_hh_lo, inner_hi)
        blk_hh_hi, blk_hh_lo = twosum_add(blk_hh_hi, blk_hh_lo, inner_lo)

    if n_lo > 0:
        # Block 2: hi × lo
        blk_hl_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_hl_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for i in range(n_hi):
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
            for j in range(n_lo):
                product = jnp.dot(A_hi_sl[i], B_lo_sl[j])
                scaled = product * B_lo_sc_pow[j][jnp.newaxis, :]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
            inner_hi = inner_hi * A_hi_sc_pow[i][:, jnp.newaxis]
            inner_lo = inner_lo * A_hi_sc_pow[i][:, jnp.newaxis]
            blk_hl_hi, blk_hl_lo = twosum_add(
                blk_hl_hi, blk_hl_lo, inner_hi)
            blk_hl_hi, blk_hl_lo = twosum_add(
                blk_hl_hi, blk_hl_lo, inner_lo)

        # Block 3: lo × hi
        blk_lh_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_lh_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for i in range(n_lo):
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
            for j in range(n_hi):
                product = jnp.dot(A_lo_sl[i], B_hi_sl[j])
                scaled = product * B_hi_sc_pow[j][jnp.newaxis, :]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
            inner_hi = inner_hi * A_lo_sc_pow[i][:, jnp.newaxis]
            inner_lo = inner_lo * A_lo_sc_pow[i][:, jnp.newaxis]
            blk_lh_hi, blk_lh_lo = twosum_add(
                blk_lh_hi, blk_lh_lo, inner_hi)
            blk_lh_hi, blk_lh_lo = twosum_add(
                blk_lh_hi, blk_lh_lo, inner_lo)

    # Final combine: hi parts first, then lo parts.
    C_hi = jnp.zeros((N, M), dtype=jnp.float32)
    C_lo = jnp.zeros((N, M), dtype=jnp.float32)
    C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hh_hi)
    if n_lo > 0:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hl_hi)
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lh_hi)
    C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hh_lo)
    if n_lo > 0:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hl_lo)
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lh_lo)

    return C_hi, C_lo


def _bf16_interleaved_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo):
    """BF16 interleaved pipeline: extract → BF16 cast → GEMM → scale → 2Sum.

    Casts extracted slices to BF16 before matmul for higher MXU throughput,
    then immediately 2Sum-accumulates each scaled result. Keeps only ONE
    (N, M) product alive at a time, reducing HBM intermediate traffic.

    Accuracy is preserved because the sigma trick already constrains mantissa
    bits within BF16 range. Gives 1.05-1.16x speedup over broadcast on TPU.
    """
    # Phase 1: Extraction.
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)

    if n_lo > 0:
        A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
        B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

    # Cast to BF16 for MXU throughput.
    A_hi_sl_bf = jnp.bfloat16(A_hi_sl)
    B_hi_sl_bf = jnp.bfloat16(B_hi_sl)
    if n_lo > 0:
        A_lo_sl_bf = jnp.bfloat16(A_lo_sl)
        B_lo_sl_bf = jnp.bfloat16(B_lo_sl)

    # Pre-compute scale vectors (FP32 power-of-2).
    # Use ldexp instead of exp2 for exact power-of-2 values.
    A_hi_sc_pow = jnp.ldexp(jnp.ones_like(A_hi_sc), jnp.int32(A_hi_sc))
    B_hi_sc_pow = jnp.ldexp(jnp.ones_like(B_hi_sc), jnp.int32(B_hi_sc))
    if n_lo > 0:
        A_lo_sc_pow = jnp.ldexp(jnp.ones_like(A_lo_sc), jnp.int32(A_lo_sc))
        B_lo_sc_pow = jnp.ldexp(jnp.ones_like(B_lo_sc), jnp.int32(B_lo_sc))

    N = A_hi.shape[0]
    M = B_hi.shape[1]

    def twosum_add(s_hi, s_lo, x):
        t = s_hi + x
        e = (s_hi - t) + x
        return t, s_lo + e

    def do_dot(a, b):
        return jnp.matmul(a, b, preferred_element_type=jnp.float32)

    # Block 1: hi x hi
    blk_hh_hi = jnp.zeros((N, M), dtype=jnp.float32)
    blk_hh_lo = jnp.zeros((N, M), dtype=jnp.float32)
    for i in range(n_hi):
        inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
        inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for j in range(n_hi):
            product = do_dot(A_hi_sl_bf[i], B_hi_sl_bf[j])
            scaled = product * B_hi_sc_pow[j][jnp.newaxis, :]
            inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
        inner_hi = inner_hi * A_hi_sc_pow[i][:, jnp.newaxis]
        inner_lo = inner_lo * A_hi_sc_pow[i][:, jnp.newaxis]
        blk_hh_hi, blk_hh_lo = twosum_add(blk_hh_hi, blk_hh_lo, inner_hi)
        blk_hh_hi, blk_hh_lo = twosum_add(blk_hh_hi, blk_hh_lo, inner_lo)

    if n_lo > 0:
        # Block 2: hi x lo
        blk_hl_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_hl_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for i in range(n_hi):
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
            for j in range(n_lo):
                product = do_dot(A_hi_sl_bf[i], B_lo_sl_bf[j])
                scaled = product * B_lo_sc_pow[j][jnp.newaxis, :]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
            inner_hi = inner_hi * A_hi_sc_pow[i][:, jnp.newaxis]
            inner_lo = inner_lo * A_hi_sc_pow[i][:, jnp.newaxis]
            blk_hl_hi, blk_hl_lo = twosum_add(
                blk_hl_hi, blk_hl_lo, inner_hi)
            blk_hl_hi, blk_hl_lo = twosum_add(
                blk_hl_hi, blk_hl_lo, inner_lo)

        # Block 3: lo x hi
        blk_lh_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_lh_lo = jnp.zeros((N, M), dtype=jnp.float32)
        for i in range(n_lo):
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)
            for j in range(n_hi):
                product = do_dot(A_lo_sl_bf[i], B_hi_sl_bf[j])
                scaled = product * B_hi_sc_pow[j][jnp.newaxis, :]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
            inner_hi = inner_hi * A_lo_sc_pow[i][:, jnp.newaxis]
            inner_lo = inner_lo * A_lo_sc_pow[i][:, jnp.newaxis]
            blk_lh_hi, blk_lh_lo = twosum_add(
                blk_lh_hi, blk_lh_lo, inner_hi)
            blk_lh_hi, blk_lh_lo = twosum_add(
                blk_lh_hi, blk_lh_lo, inner_lo)

    # Final combine.
    C_hi = jnp.zeros((N, M), dtype=jnp.float32)
    C_lo = jnp.zeros((N, M), dtype=jnp.float32)
    C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hh_hi)
    if n_lo > 0:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hl_hi)
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lh_hi)
    C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hh_lo)
    if n_lo > 0:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hl_lo)
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lh_lo)

    return C_hi, C_lo


@functools.partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _fused_ondevice_jit(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo,
                         block_group_sizes):
    """Fused extraction + GEMMs + accumulation. Takes 4 FP32 inputs."""
    return _fused_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo,
                                  block_group_sizes)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _fully_fused_ondevice_jit(A_f64, B_f64, rho, n_hi, n_lo,
                               block_group_sizes):
    """Fully fused: split + extraction + GEMMs + accumulation.

    Takes 2 FP64 matrices — eliminates CPU double_f32_split bottleneck.
    Returns (C_hi, C_lo) FP32 pair.
    Requires jax_enable_x64=True.
    """
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    return _fused_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo,
                                  block_group_sizes)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def _fully_fused_f64_jit(A_f64, B_f64, rho, n_hi, n_lo,
                          block_group_sizes):
    """Fully fused with FP64 combine. Returns single FP64 JAX array.

    Fuses split + extraction + GEMMs + 2Sum + FP64 combine into one JIT.
    Avoids separate dispatch and extra PCIe round-trip for the combine.
    """
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    C_hi, C_lo = _fused_pipeline_logic(A_hi, A_lo, B_hi, B_lo, rho, n_hi,
                                        n_lo, block_group_sizes)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _bf16_interleaved_f64_jit(A_f64, B_f64, rho, n_hi, n_lo):
    """BF16 interleaved with FP64 combine. Returns single FP64 JAX array.

    Uses BF16-cast extracted slices with interleaved 2Sum accumulation.
    Gives 1.05-1.16x speedup over broadcast on TPU v6e.
    """
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)
    C_hi, C_lo = _bf16_interleaved_pipeline_logic(
        A_hi, A_lo, B_hi, B_lo, rho, n_hi, n_lo)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


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


def _ondevice_safety_report(A_f64, B_f64, n_hi, n_lo):
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

    if isinstance(A_f64, jax.Array):
        if not (bool(jnp.all(jnp.isfinite(A_f64))) and
                bool(jnp.all(jnp.isfinite(B_f64)))):
            reasons.append("inputs contain NaN or Inf")
    elif not np.isfinite(A_f64).all() or not np.isfinite(B_f64).all():
        reasons.append("inputs contain NaN or Inf")

    return {
        "safe": len(reasons) == 0,
        "rho": rho,
        "bits_per_slice": bits_per_slice,
        "n_hi": n_hi,
        "n_lo": n_lo,
        "n_gemms": n_hi * n_hi + n_hi * n_lo + n_lo * n_hi,
        "reasons": reasons,
    }


def _matmul_ondevice(A_f64, B_f64, safe_mode, accumulation="fused",
                     n_hi=4, n_lo=1):
    """On-device path: FP32 extraction and configurable GEMM blocks.

    Args:
        accumulation: 'fused' (extraction + GEMMs + accumulation in one JIT call),
            'ondevice' (2Sum on device), or 'host' (FP64 on CPU)
        n_hi: Number of hi extraction slices (from precision preset).
        n_lo: Number of lo extraction slices (from precision preset).
    """
    _validate_accumulation_mode(accumulation)
    report = _ondevice_safety_report(A_f64, B_f64, n_hi, n_lo)
    fallback = _handle_unsafe_preflight(A_f64, B_f64, report, safe_mode)
    if fallback is not None:
        return fallback

    N = A_f64.shape[0]
    M = B_f64.shape[1]
    rho = report["rho"]
    n_hi = report["n_hi"]
    n_lo = report["n_lo"]

    # Fully-fused path: everything on device, returns JAX FP64 array.
    # jnp.asarray is a no-op for JAX arrays already on device.
    if accumulation == "fused":
        _require_x64_for_fused()
        A_j = jnp.asarray(A_f64, dtype=jnp.float64)
        B_j = jnp.asarray(B_f64, dtype=jnp.float64)

        block_group_sizes = (
            tuple([n_hi] * n_hi),
            tuple([n_lo] * n_hi) if n_lo > 0 else (),
            tuple([n_hi] * n_lo) if n_lo > 0 else (),
        )

        return _fully_fused_f64_jit(
            A_j, B_j, rho, n_hi, n_lo, block_group_sizes)

    # BF16 interleaved path: BF16-cast slices + interleaved 2Sum.
    # ~1.05-1.16x faster than broadcast on TPU due to reduced HBM traffic.
    if accumulation == "bf16_interleaved":
        _require_x64_for_fused()
        A_j = jnp.asarray(A_f64, dtype=jnp.float64)
        B_j = jnp.asarray(B_f64, dtype=jnp.float64)

        return _bf16_interleaved_f64_jit(A_j, B_j, rho, n_hi, n_lo)

    # 1) Double-FP32 split (for ondevice/host paths — needs numpy).
    A_np = np.asarray(A_f64, dtype=np.float64)
    B_np = np.asarray(B_f64, dtype=np.float64)
    A_hi, A_lo = _double_f32_split(A_np)
    B_hi, B_lo = _double_f32_split(B_np)

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


def _matmul_ondevice_numpy(A_f64, B_f64, safe_mode, n_hi=4, n_lo=1):
    """On-device path implemented with NumPy GEMMs for testing."""
    report = _ondevice_safety_report(A_f64, B_f64, n_hi, n_lo)
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


def matmul(A, B, n_slices=8, safe_mode="raise", pipeline="host",
           accumulation="fused", precision="high"):
    """FP64 matmul via Ozaki Extract.

    Accepts numpy arrays or JAX arrays.
    For `pipeline='ondevice'`, output type matches input type:
    numpy inputs -> numpy output, JAX inputs -> JAX output.

    Args:
        A, B: Input matrices of the same array family (both numpy or both JAX).
        pipeline: 'host' or 'ondevice'
        accumulation: 'fused' (default; split + extraction + GEMMs + accumulation
            in one JIT call), 'bf16_interleaved' (BF16-cast slices with
            interleaved 2Sum; ~1.05-1.16x faster on TPU), 'ondevice' (2Sum on
            device with CPU split), or 'host' (FP64 on CPU).
            Only used when pipeline='ondevice'; ignored for host pipeline.
        precision: Accuracy/speed tradeoff for on-device pipeline.
            'high' (default): 24 GEMMs, ~9.5 correct digits.
            'medium': 15 GEMMs, ~7 correct digits (FP32-level).
            'max': 65 GEMMs, ~10 correct digits.
            Also accepts a custom (n_hi, n_lo) tuple.
    """
    a_is_jax = isinstance(A, jax.Array)
    b_is_jax = isinstance(B, jax.Array)
    if a_is_jax != b_is_jax:
        raise ValueError(
            "A and B must both be numpy arrays or both be JAX arrays."
        )
    input_is_jax = a_is_jax

    if pipeline == "ondevice":
        n_hi, n_lo = _resolve_precision(precision)
        if not input_is_jax:
            A = np.asarray(A, dtype=np.float64)
            B = np.asarray(B, dtype=np.float64)
        result = _matmul_ondevice(A, B, safe_mode, accumulation, n_hi, n_lo)
        if not input_is_jax and isinstance(result, jax.Array):
            return np.asarray(result)
        if input_is_jax and not isinstance(result, jax.Array):
            return jnp.asarray(result, dtype=jnp.float64)
        return result

    if pipeline != "host":
        raise ValueError(f"Unknown pipeline={pipeline!r}; expected 'host' or 'ondevice'.")

    # Host pipeline needs numpy arrays.
    A_f64 = np.asarray(A, dtype=np.float64)
    B_f64 = np.asarray(B, dtype=np.float64)

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


def matmul_numpy(A_f64, B_f64, n_slices=8, safe_mode="raise", pipeline="host",
                 precision="high"):
    """NumPy implementation of `matmul` with matching pipeline options."""
    A_f64 = np.asarray(A_f64, dtype=np.float64)
    B_f64 = np.asarray(B_f64, dtype=np.float64)

    if pipeline == "ondevice":
        n_hi, n_lo = _resolve_precision(precision)
        return _matmul_ondevice_numpy(A_f64, B_f64, safe_mode, n_hi, n_lo)
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
