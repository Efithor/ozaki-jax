"""FP64-accurate least squares via iterative refinement over FP32 QR.

lstsq(A, b): minimize ||A x - b||_2 for full-rank A with rows >= cols.

Mixed-precision refinement on the augmented (Bjorck) system

    [ I   A ] [r]   [b]
    [ A^T 0 ] [x] = [0]

whose solution gives the LS residual r = b - A x and the LS solution x.
A = QR is factored once in FP32; each iteration computes the augmented
residual at high precision and solves for corrections (dr, dx) via

    dx = R^-1 ( Q^T f - R^-T g ),   dr = f - A dx

where f, g are the two block residuals. Refining r as well as x is what
makes this converge to ~FP64 accuracy even when the LS residual is large
(generic b) -- plain residual refinement of x alone stalls at ~u_f32 there.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from .matmul import _resolve_precision, _is_x64_enabled
from .solve import _ozaki_matvec_f64
from .extract import _compute_rho_f32


def _validate_lstsq_inputs(A, b):
    """Validate shapes for lstsq(A, b). Requires rows >= cols (full rank)."""
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got ndim={A.ndim}.")
    M, N = A.shape
    if M < N:
        raise ValueError(
            f"lstsq requires rows >= cols (got {M}x{N}); underdetermined "
            "systems are not supported.")
    if b.ndim == 1:
        if b.shape[0] != M:
            raise ValueError(f"b length {b.shape[0]} != A rows {M}.")
    elif b.ndim == 2:
        if b.shape[0] != M:
            raise ValueError(f"b rows {b.shape[0]} != A rows {M}.")
    else:
        raise ValueError(f"b must be 1D or 2D, got ndim={b.ndim}.")


def _ls_solve_f32(Q, R, rhs_f64):
    """FP32 LS solve x = R^-1 Q^T rhs for an FP64 RHS, returned in FP64."""
    z = jnp.matmul(Q.T, jnp.float32(rhs_f64))
    return jnp.float64(solve_triangular(R, z, lower=False))


def _aug_correction_f32(Q, R, f_f64, g_f64):
    """Solve the augmented correction system for (dr, dx) in FP32.

    dx = R^-1 ( Q^T f - R^-T g ),  dr = f - A dx,  with A = QR.
    """
    f32 = jnp.float32
    Qtf = jnp.matmul(Q.T, f32(f_f64))                       # (n, k)
    Rtg = solve_triangular(R.T, f32(g_f64), lower=True)      # R^-T g
    dx = solve_triangular(R, Qtf - Rtg, lower=False)         # (n, k)
    dr = f32(f_f64) - jnp.matmul(Q, jnp.matmul(R, dx))       # A dx = Q R dx
    return jnp.float64(dr), jnp.float64(dx)


@functools.partial(jax.jit, static_argnums=(2,))
def _lstsq_iterref_f64_jit(A_f64, b_f64, max_iterations):
    """Augmented-system refinement with native FP64 residuals."""
    A_f32 = jnp.float32(A_f64)
    Q, R = jnp.linalg.qr(A_f32, mode="reduced")

    x = _ls_solve_f32(Q, R, b_f64)
    r = b_f64 - jnp.matmul(A_f64, x)
    for _ in range(max_iterations):
        f = b_f64 - r - jnp.matmul(A_f64, x)        # block-1 residual
        g = -jnp.matmul(A_f64.T, r)                 # block-2 residual (=0 at opt)
        dr, dx = _aug_correction_f32(Q, R, f, g)
        x = x + dx
        r = r + dr
    return x


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _lstsq_iterref_ozaki_jit(A_f64, b_f64, rho, n_hi, n_lo, max_iterations,
                             accumulation):
    """Augmented-system refinement with Ozaki (FP32/BF16) residuals."""
    A_f32 = jnp.float32(A_f64)
    Q, R = jnp.linalg.qr(A_f32, mode="reduced")
    At_f64 = A_f64.T

    x = _ls_solve_f32(Q, R, b_f64)
    Ax = _ozaki_matvec_f64(A_f64, x, rho, n_hi, n_lo, accumulation)
    r = b_f64 - Ax
    for _ in range(max_iterations):
        Ax = _ozaki_matvec_f64(A_f64, x, rho, n_hi, n_lo, accumulation)
        Atr = _ozaki_matvec_f64(At_f64, r, rho, n_hi, n_lo, accumulation)
        f = b_f64 - r - Ax
        g = -Atr
        dr, dx = _aug_correction_f32(Q, R, f, g)
        x = x + dx
        r = r + dr
    return x


def lstsq(A, b, precision="high", accumulation="bf16_interleaved",
          max_iterations=3, residual_mode="f64"):
    """Solve the least-squares problem min ||A x - b||_2 with FP64 accuracy.

    Uses an FP32 QR factorization with iterative refinement on high-precision
    residuals. For well-conditioned, full-rank A this converges to ~13-14
    digits in 2-3 iterations.

    Args:
        A: Matrix (M, N) with M >= N, full column rank. numpy or JAX array.
        b: RHS vector (M,) or matrix (M, K).
        precision: 'high', 'medium', 'max', or (n_hi, n_lo) tuple.
            Only used when residual_mode='ozaki'.
        accumulation: 'bf16_interleaved' or 'fused'.
            Only used when residual_mode='ozaki'.
        max_iterations: Refinement iterations (default 3).
        residual_mode: How to compute the residual b - A @ x.
            'f64' (default): native FP64 matmul. Faster and more accurate.
            'ozaki': Ozaki matmul (FP32/BF16 extraction pipeline).

    Returns:
        x: Least-squares solution (N,) or (N, K), same type as b.
    """
    if not _is_x64_enabled():
        raise ValueError("lstsq() requires jax_enable_x64=True.")

    allowed_residual = {"f64", "ozaki"}
    if residual_mode not in allowed_residual:
        raise ValueError(
            f"residual_mode={residual_mode!r}; expected one of "
            f"{sorted(allowed_residual)}.")

    if residual_mode == "ozaki":
        allowed_accum = {"bf16_interleaved", "fused"}
        if accumulation not in allowed_accum:
            raise ValueError(
                f"accumulation={accumulation!r}; expected one of "
                f"{sorted(allowed_accum)}.")

    a_is_jax = isinstance(A, jax.Array)
    _validate_lstsq_inputs(A, b)

    A_j = jnp.asarray(A, dtype=jnp.float64)
    b_j = jnp.asarray(b, dtype=jnp.float64)
    b_is_vector = (b_j.ndim == 1)
    if b_is_vector:
        b_j = b_j[:, None]

    if residual_mode == "f64":
        x = _lstsq_iterref_f64_jit(A_j, b_j, max_iterations)
    else:
        n_hi, n_lo = _resolve_precision(precision)
        # Refinement contracts over both N (A@x) and M (A^T@r); size rho for
        # the larger dimension so the sigma split is safe for both products.
        rho = _compute_rho_f32(max(A.shape[0], A.shape[1]))
        x = _lstsq_iterref_ozaki_jit(A_j, b_j, rho, n_hi, n_lo,
                                     max_iterations, accumulation)

    if b_is_vector:
        x = x[:, 0]
    if not a_is_jax:
        return np.asarray(x)
    return x
