"""FP64-accurate vector and matrix norms.

Most orders are exact FP64 reductions (no Ozaki needed). The matrix
spectral norm (ord=2) is the largest singular value, computed as
sqrt(lambda_max(A^T A)) using the accurate Gram matrix -- this is where
the Ozaki pipeline contributes, via mode='ozaki'.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from .matmul import _is_x64_enabled
from .gram import gram

_INF = float("inf")


def _vector_norm(x, ord):
    if ord is None or ord == 2:
        return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))
    if ord == 1:
        return jnp.sum(jnp.abs(x))
    if ord == _INF:
        return jnp.max(jnp.abs(x))
    if ord == -_INF:
        return jnp.min(jnp.abs(x))
    if ord == 0:
        return jnp.sum(x != 0).astype(jnp.float64)
    if isinstance(ord, str):
        raise ValueError(f"Invalid norm order {ord!r} for a vector.")
    return jnp.sum(jnp.abs(x) ** ord) ** (1.0 / ord)


def _matrix_norm(A, ord, precision, accumulation, mode):
    if ord is None or ord == "fro":
        return jnp.sqrt(jnp.sum(jnp.abs(A) ** 2))
    if ord == 1:
        return jnp.max(jnp.sum(jnp.abs(A), axis=0))
    if ord == -1:
        return jnp.min(jnp.sum(jnp.abs(A), axis=0))
    if ord == _INF:
        return jnp.max(jnp.sum(jnp.abs(A), axis=1))
    if ord == -_INF:
        return jnp.min(jnp.sum(jnp.abs(A), axis=1))
    if ord == 2:
        # Largest singular value = sqrt(lambda_max(A^T A)). Use the smaller
        # Gram matrix and clip tiny negative eigenvalues from rounding.
        G = gram(A, precision=precision, accumulation=accumulation, mode=mode)
        lam = jnp.linalg.eigvalsh(jnp.asarray(G, dtype=jnp.float64))[-1]
        return jnp.sqrt(jnp.maximum(lam, 0.0))
    raise ValueError(
        f"Invalid / unsupported norm order {ord!r} for a matrix. "
        "Supported: 'fro', 1, -1, 2, inf, -inf.")


def norm(x, ord=None, precision="high", accumulation="bf16_interleaved",
         mode="f64"):
    """Compute a vector or matrix norm with FP64 accuracy.

    Vector (1D) orders: None/2 (Euclidean), 1, inf, -inf, 0, or any p.
    Matrix (2D) orders: None/'fro', 1, -1, inf, -inf, 2 (spectral).

    Only the matrix spectral norm (ord=2) uses the Ozaki pipeline (via the
    Gram matrix); precision/accumulation/mode are ignored for other orders.

    Accuracy note: a general p-norm (p not in {1, 2, inf}) is computed as
    (sum |x|^p)^(1/p). The x**p / **(1/p) transcendentals run at ~fp32
    precision on TPU (even under emulated fp64), so a non-standard p gives
    ~1e-8 there; it is exact to fp64 on CPU/GPU. The standard orders use only
    multiply/add/sqrt and stay near machine fp64 on every backend.

    Args:
        x: Vector (N,) or matrix (M, N). numpy or JAX array.
        ord: Norm order (see above). Default: Euclidean for vectors,
            Frobenius for matrices.
        precision: 'high', 'medium', 'max', or (n_hi, n_lo). ord=2 only.
        accumulation: 'bf16_interleaved' or 'fused'. ord=2 only.
        mode: 'f64' (default) or 'ozaki'. ord=2 only.

    Returns:
        The norm as a scalar (np.float64 for numpy input, 0-d JAX array
        for JAX input).
    """
    if not _is_x64_enabled():
        raise ValueError("norm() requires jax_enable_x64=True.")

    x_is_jax = isinstance(x, jax.Array)
    x_j = jnp.asarray(x, dtype=jnp.float64)

    if x_j.ndim == 1:
        result = _vector_norm(x_j, ord)
    elif x_j.ndim == 2:
        result = _matrix_norm(x_j, ord, precision, accumulation, mode)
    else:
        raise ValueError(f"x must be 1D or 2D, got ndim={x_j.ndim}.")

    if not x_is_jax:
        return np.float64(result)
    return result
