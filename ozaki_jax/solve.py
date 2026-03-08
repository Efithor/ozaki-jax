"""FP64-accurate residual and linear system solver.

- residual(A, x, b): accurate b - A @ x via Ozaki matmul.
- solve(A, b): iterative refinement over FP32 LU, FP64 residual.
"""

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


def _validate_solve_inputs(A, b):
    """Validate shapes for solve(A, b)."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")
    N = A.shape[0]
    if b.ndim == 1:
        if b.shape[0] != N:
            raise ValueError(f"b length {b.shape[0]} != A size {N}.")
    elif b.ndim == 2:
        if b.shape[0] != N:
            raise ValueError(f"b rows {b.shape[0]} != A size {N}.")
    else:
        raise ValueError(f"b must be 1D or 2D, got ndim={b.ndim}.")


def _ozaki_matvec_f64(A_f64, x_f64, rho, n_hi, n_lo, accumulation):
    """Accurate A @ x via Ozaki. Called within JIT context."""
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    x_hi, x_lo = _jax_double_f32_split(x_f64)
    if accumulation == "bf16_interleaved":
        C_hi, C_lo = _bf16_interleaved_pipeline_logic(
            A_hi, A_lo, x_hi, x_lo, rho, n_hi, n_lo)
    else:
        bgs = (
            tuple([n_hi] * n_hi),
            tuple([n_lo] * n_hi) if n_lo > 0 else (),
            tuple([n_hi] * n_lo) if n_lo > 0 else (),
        )
        C_hi, C_lo = _fused_pipeline_logic(
            A_hi, A_lo, x_hi, x_lo, rho, n_hi, n_lo, bgs)
    return jnp.float64(C_hi) + jnp.float64(C_lo)


def _validate_residual_inputs(A, x, b):
    """Validate shapes for residual(A, x, b)."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got ndim={A.ndim}.")
    N, K = A.shape
    if x.ndim == 1:
        if x.shape[0] != K:
            raise ValueError(f"x length {x.shape[0]} != A columns {K}.")
        if b.ndim != 1 or b.shape[0] != N:
            raise ValueError(
                f"b shape {b.shape} incompatible with A @ x shape ({N},).")
    elif x.ndim == 2:
        if x.shape[0] != K:
            raise ValueError(f"x rows {x.shape[0]} != A columns {K}.")
        if b.ndim != 2 or b.shape[0] != N or b.shape[1] != x.shape[1]:
            raise ValueError(
                f"b shape {b.shape} incompatible with A @ x shape "
                f"({N}, {x.shape[1]}).")
    else:
        raise ValueError(f"x must be 1D or 2D, got ndim={x.ndim}.")


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _residual_ozaki_jit(A_f64, x_f64, b_f64, rho, n_hi, n_lo, accumulation):
    """JIT-compiled residual b - A @ x via Ozaki."""
    Ax = _ozaki_matvec_f64(A_f64, x_f64, rho, n_hi, n_lo, accumulation)
    return b_f64 - Ax


@jax.jit
def _residual_f64_jit(A_f64, x_f64, b_f64):
    """JIT-compiled residual b - A @ x via native FP64 matmul."""
    return b_f64 - jnp.matmul(A_f64, x_f64)


def residual(A, x, b, precision="high", accumulation="bf16_interleaved",
             mode="f64"):
    """Compute b - A @ x with FP64 accuracy.

    Args:
        A: Matrix (N, K). numpy or JAX array.
        x: Vector (K,) or matrix (K, M).
        b: Vector (N,) or matrix (N, M), matching A @ x shape.
        precision: 'high', 'medium', 'max', or (n_hi, n_lo) tuple.
            Only used when mode='ozaki'.
        accumulation: 'bf16_interleaved' or 'fused'.
            Only used when mode='ozaki'.
        mode: How to compute A @ x.
            'f64' (default): native FP64 matmul. Gives ~15 digits.
            'ozaki': Ozaki matmul (FP32/BF16 pipeline). ~10 digits.
                Use when FP64 is unavailable or too slow.

    Returns:
        r: Residual b - A @ x, same type/shape as b.
    """
    if not _is_x64_enabled():
        raise ValueError("residual() requires jax_enable_x64=True.")

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

    a_is_jax = isinstance(A, jax.Array)
    _validate_residual_inputs(A, x, b)

    A_j = jnp.asarray(A, dtype=jnp.float64)
    x_j = jnp.asarray(x, dtype=jnp.float64)
    b_j = jnp.asarray(b, dtype=jnp.float64)
    x_is_vector = (x_j.ndim == 1)
    if x_is_vector:
        x_j = x_j[:, None]
        b_j = b_j[:, None]

    if mode == "f64":
        r = _residual_f64_jit(A_j, x_j, b_j)
    else:
        n_hi, n_lo = _resolve_precision(precision)
        K = A.shape[1]
        rho = _compute_rho_f32(K)
        r = _residual_ozaki_jit(A_j, x_j, b_j, rho, n_hi, n_lo, accumulation)

    if x_is_vector:
        r = r[:, 0]
    if not a_is_jax:
        return np.asarray(r)
    return r


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _solve_iterref_jit(A_f64, b_f64, rho, n_hi, n_lo, max_iterations,
                       accumulation):
    """JIT-compiled iterative refinement core (Ozaki residual)."""
    A_f32 = jnp.float32(A_f64)
    x = jnp.float64(jnp.linalg.solve(A_f32, jnp.float32(b_f64)))

    for _ in range(max_iterations):
        Ax = _ozaki_matvec_f64(A_f64, x, rho, n_hi, n_lo, accumulation)
        r = b_f64 - Ax
        d = jnp.float64(jnp.linalg.solve(A_f32, jnp.float32(r)))
        x = x + d

    return x


@functools.partial(jax.jit, static_argnums=(2,))
def _solve_iterref_f64_jit(A_f64, b_f64, max_iterations):
    """JIT-compiled iterative refinement with native FP64 residual.

    Uses jnp.matmul for the residual (1 FP64 matmul vs 24-65 BF16 GEMMs
    for Ozaki). On TPU, FP64 is emulated (~4 MXU calls). This is both
    cheaper and more accurate than Ozaki for the residual computation.
    """
    A_f32 = jnp.float32(A_f64)
    x = jnp.float64(jnp.linalg.solve(A_f32, jnp.float32(b_f64)))

    for _ in range(max_iterations):
        r = b_f64 - jnp.matmul(A_f64, x)
        d = jnp.float64(jnp.linalg.solve(A_f32, jnp.float32(r)))
        x = x + d

    return x


def solve(A, b, precision="high", accumulation="bf16_interleaved",
          max_iterations=3, residual_mode="f64"):
    """Solve Ax = b with FP64 accuracy via iterative refinement.

    Uses FP32 LU factorization with iterative refinement. Each iteration
    computes an accurate residual and applies an FP32 correction. With
    residual_mode='f64' (default), converges to ~14 digits in 2-3
    iterations for well-conditioned systems.

    Args:
        A: Square matrix (N, N). numpy or JAX array.
        b: RHS vector (N,) or matrix (N, K).
        precision: 'high', 'medium', 'max', or (n_hi, n_lo) tuple.
            Only used when residual_mode='ozaki'.
        accumulation: 'bf16_interleaved' or 'fused'.
            Only used when residual_mode='ozaki'.
        max_iterations: Refinement iterations (default 3).
        residual_mode: How to compute the residual b - A @ x.
            'f64' (default): native FP64 matmul. Faster and more accurate.
                Converges to cond(A) * u_f64 (~14 digits).
            'ozaki': Ozaki matmul (FP32/BF16 extraction pipeline).
                Limited to ~10 digit matmul accuracy, solve plateaus
                at ~8-9 digits. Use when FP64 is unavailable or too slow.

    Returns:
        x: Solution, same type/shape as b.
    """
    if not _is_x64_enabled():
        raise ValueError("solve() requires jax_enable_x64=True.")

    allowed_residual = {"f64", "ozaki"}
    if residual_mode not in allowed_residual:
        raise ValueError(
            f"residual_mode={residual_mode!r}; expected one of "
            f"{sorted(allowed_residual)}."
        )

    if residual_mode == "ozaki":
        allowed_accum = {"bf16_interleaved", "fused"}
        if accumulation not in allowed_accum:
            raise ValueError(
                f"accumulation={accumulation!r}; expected one of "
                f"{sorted(allowed_accum)}."
            )

    a_is_jax = isinstance(A, jax.Array)
    _validate_solve_inputs(A, b)

    # Convert to JAX FP64.
    A_j = jnp.asarray(A, dtype=jnp.float64)
    b_j = jnp.asarray(b, dtype=jnp.float64)
    b_is_vector = (b_j.ndim == 1)
    if b_is_vector:
        b_j = b_j[:, None]

    if residual_mode == "f64":
        x = _solve_iterref_f64_jit(A_j, b_j, max_iterations)
    else:
        n_hi, n_lo = _resolve_precision(precision)
        N = A.shape[0]
        rho = _compute_rho_f32(N)
        x = _solve_iterref_jit(A_j, b_j, rho, n_hi, n_lo, max_iterations,
                               accumulation)

    if b_is_vector:
        x = x[:, 0]
    if not a_is_jax:
        return np.asarray(x)
    return x
