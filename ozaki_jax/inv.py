"""FP64-accurate matrix inverse via iterative-refinement solve."""

import jax
import jax.numpy as jnp
import numpy as np

from .matmul import _is_x64_enabled
from .solve import solve


def inv(A, precision="high", accumulation="bf16_interleaved",
        max_iterations=3, residual_mode="f64"):
    """Compute A^-1 with FP64 accuracy.

    Solves A X = I column-block by iterative refinement (see solve()).
    Prefer solve(A, b) when you only need A^-1 @ b -- forming the explicit
    inverse is less accurate and more expensive.

    Args:
        A: Square matrix (N, N). numpy or JAX array.
        precision: 'high', 'medium', 'max', or (n_hi, n_lo) tuple.
            Only used when residual_mode='ozaki'.
        accumulation: 'bf16_interleaved' or 'fused'.
            Only used when residual_mode='ozaki'.
        max_iterations: Refinement iterations (default 3).
        residual_mode: How to compute the residual I - A @ X.
            'f64' (default): native FP64 matmul. Faster and more accurate.
            'ozaki': Ozaki matmul (FP32/BF16 extraction pipeline).

    Returns:
        Ainv: Inverse matrix (N, N), same type as A.
    """
    if not _is_x64_enabled():
        raise ValueError("inv() requires jax_enable_x64=True.")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}.")

    N = A.shape[0]
    a_is_jax = isinstance(A, jax.Array)
    # solve() returns output matching A's framework; build I to match.
    eye = jnp.eye(N, dtype=jnp.float64) if a_is_jax else np.eye(N)

    return solve(A, eye, precision=precision, accumulation=accumulation,
                 max_iterations=max_iterations, residual_mode=residual_mode)
