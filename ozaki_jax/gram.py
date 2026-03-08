"""FP64-accurate Gram matrix (A^T @ A)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .matmul import (
    _jax_double_f32_split,
    _bf16_interleaved_pipeline_logic,
    _fused_pipeline_logic,
    _resolve_precision,
    _is_x64_enabled,
)
from .extract import _compute_rho_f32


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _gram_ozaki_jit(A_f64, rho, n_hi, n_lo, accumulation):
    """JIT-compiled A^T @ A via Ozaki with symmetrization."""
    At = A_f64.T
    At_hi, At_lo = _jax_double_f32_split(At)
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    if accumulation == "bf16_interleaved":
        C_hi, C_lo = _bf16_interleaved_pipeline_logic(
            At_hi, At_lo, A_hi, A_lo, rho, n_hi, n_lo)
    else:
        bgs = (
            tuple([n_hi] * n_hi),
            tuple([n_lo] * n_hi) if n_lo > 0 else (),
            tuple([n_hi] * n_lo) if n_lo > 0 else (),
        )
        C_hi, C_lo = _fused_pipeline_logic(
            At_hi, At_lo, A_hi, A_lo, rho, n_hi, n_lo, bgs)
    G = jnp.float64(C_hi) + jnp.float64(C_lo)
    return (G + G.T) * 0.5


@jax.jit
def _gram_f64_jit(A_f64):
    """JIT-compiled A^T @ A via native FP64 matmul with symmetrization."""
    G = jnp.matmul(A_f64.T, A_f64)
    return (G + G.T) * 0.5


def gram(A, precision="high", accumulation="bf16_interleaved", mode="f64"):
    """Compute A^T @ A with FP64 accuracy.

    The result is symmetrized: G = (G + G^T) / 2.

    Args:
        A: Matrix (N, K). numpy or JAX array.
        precision: 'high', 'medium', 'max', or (n_hi, n_lo) tuple.
            Only used when mode='ozaki'.
        accumulation: 'bf16_interleaved' or 'fused'.
            Only used when mode='ozaki'.
        mode: How to compute A^T @ A.
            'f64' (default): native FP64 matmul. Gives ~15 digits.
            'ozaki': Ozaki matmul (FP32/BF16 pipeline). ~10 digits.
                Use when FP64 is unavailable or too slow.

    Returns:
        G: Symmetric matrix (K, K), same type as A.
    """
    if not _is_x64_enabled():
        raise ValueError("gram() requires jax_enable_x64=True.")

    allowed_modes = {"f64", "ozaki"}
    if mode not in allowed_modes:
        raise ValueError(
            f"mode={mode!r}; expected one of {sorted(allowed_modes)}."
        )

    if mode == "ozaki":
        allowed_accum = {"bf16_interleaved", "fused"}
        if accumulation not in allowed_accum:
            raise ValueError(
                f"accumulation={accumulation!r}; expected one of "
                f"{sorted(allowed_accum)}."
            )

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got ndim={A.ndim}.")

    a_is_jax = isinstance(A, jax.Array)
    A_j = jnp.asarray(A, dtype=jnp.float64)

    if mode == "f64":
        G = _gram_f64_jit(A_j)
    else:
        n_hi, n_lo = _resolve_precision(precision)
        N = A.shape[0]
        rho = _compute_rho_f32(N)
        G = _gram_ozaki_jit(A_j, rho, n_hi, n_lo, accumulation)

    if not a_is_jax:
        return np.asarray(G)
    return G
