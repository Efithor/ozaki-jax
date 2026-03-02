"""Extract-splitting utilities for FP64 inputs and BF16-like slices."""

import numpy as np

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _compute_rho(K, m1=53, m2=8, m3=24):
    """Compute the rho lower bound from storage and accumulation constraints."""
    # Storage constraint (BF16 representation)
    xi = m1 + 1 - m2
    # Accumulation constraint (FP32 inner product exactness)
    gamma = int(np.ceil(m1 - (m3 - np.ceil(np.log2(max(K, 2)))) / 2))
    return max(gamma, xi)


def extract_split_rows(X_f64, rho, n_slices=8):
    """Split an FP64 matrix into row-scaled slices."""
    N, K = X_f64.shape
    slices = []
    scales = []
    residual = np.float64(X_f64.copy())

    for _ in range(n_slices):
        # Row-wise scaling anchor.
        row_max = np.max(np.abs(residual), axis=1)  # (N,)

        # Avoid log2(0) on empty-magnitude rows.
        zero_mask = (row_max == 0)
        row_max_safe = np.where(zero_mask, 1.0, row_max)

        # Exponent per row.
        c_x = np.floor(np.log2(row_max_safe))
        c_x = np.where(zero_mask, 0.0, c_x)

        # Extract constant sigma = 0.75 * 2^(rho + c_x).
        sigma = np.float64(0.75) * np.ldexp(np.ones(N, dtype=np.float64),
                                             (rho + c_x).astype(np.int64))
        sigma_2d = sigma[:, np.newaxis]  # (N, 1) for broadcasting

        # Extract rounded slice.
        v = (residual + sigma_2d) - sigma_2d

        # Keep all-zero rows exact.
        v = np.where(zero_mask[:, np.newaxis], 0.0, v)

        # Remove extracted slice from residual.
        residual = residual - v

        # Normalize by row scales.
        inv_scale = np.ldexp(np.ones(N, dtype=np.float64),
                             (-c_x).astype(np.int64))
        v_norm = v * inv_scale[:, np.newaxis]

        # NumPy has no native BF16 dtype; use float32 storage.
        v_bf16 = v_norm.astype(np.float32)

        slices.append(v_bf16)
        scales.append(c_x)

    return slices, scales


def extract_split_cols(X_f64, rho, n_slices=8):
    """Split an FP64 matrix into column-scaled slices."""
    K, M = X_f64.shape
    slices = []
    scales = []
    residual = np.float64(X_f64.copy())

    for _ in range(n_slices):
        # Column-wise scaling anchor.
        col_max = np.max(np.abs(residual), axis=0)  # (M,)

        zero_mask = (col_max == 0)
        col_max_safe = np.where(zero_mask, 1.0, col_max)

        c_y = np.floor(np.log2(col_max_safe))
        c_y = np.where(zero_mask, 0.0, c_y)

        sigma = np.float64(0.75) * np.ldexp(np.ones(M, dtype=np.float64),
                                             (rho + c_y).astype(np.int64))
        sigma_2d = sigma[np.newaxis, :]  # (1, M)

        v = (residual + sigma_2d) - sigma_2d
        v = np.where(zero_mask[np.newaxis, :], 0.0, v)
        residual = residual - v

        inv_scale = np.ldexp(np.ones(M, dtype=np.float64),
                             (-c_y).astype(np.int64))
        v_norm = v * inv_scale[np.newaxis, :]

        # NumPy has no native BF16 dtype; use float32 storage.
        v_bf16 = v_norm.astype(np.float32)

        slices.append(v_bf16)
        scales.append(c_y)

    return slices, scales


# FP32 extraction helpers for the on-device pipeline.


def _compute_rho_f32(K, m1=24, m2=8, m3=24):
    """Compute rho lower bound for FP32 working precision."""
    xi = m1 + 1 - m2  # 24+1-8 = 17
    gamma = int(np.ceil(m1 - (m3 - np.ceil(np.log2(max(K, 2)))) / 2))
    return max(gamma, xi)


def f32_extract_split_rows(X_f32, rho, n_slices=5):
    """Split an FP32 matrix into row-scaled Extract slices."""
    X_f32 = np.float32(X_f32)
    N, K = X_f32.shape
    slices = []
    scales = []
    residual = X_f32.copy()

    for _ in range(n_slices):
        row_max = np.float32(np.max(np.abs(residual), axis=1))  # (N,)

        zero_mask = (row_max == 0)
        row_max_safe = np.where(zero_mask, np.float32(1.0), row_max)

        c_x = np.floor(np.float32(np.log2(row_max_safe)))
        c_x = np.where(zero_mask, np.float32(0.0), c_x)

        # Sigma = 0.75 * 2^(rho + c_x) in FP32.
        sigma = np.float32(np.float32(0.75) * np.ldexp(
            np.ones(N, dtype=np.float32),
            (rho + c_x).astype(np.int32)))
        sigma_2d = sigma[:, np.newaxis]

        # Sigma trick extracts top bits under FP32 rounding.
        v = np.float32(np.float32(residual + sigma_2d) - sigma_2d)
        v = np.where(zero_mask[:, np.newaxis], np.float32(0.0), v)
        residual = np.float32(residual - v)

        # Normalize by row scales.
        inv_scale = np.float32(np.ldexp(
            np.ones(N, dtype=np.float32),
            (-c_x).astype(np.int32)))
        v_norm = np.float32(v * inv_scale[:, np.newaxis])

        # Store as float32 values (BF16-exact by construction).
        slices.append(v_norm)
        scales.append(c_x)

    return slices, scales


def f32_extract_split_cols(X_f32, rho, n_slices=5):
    """Split an FP32 matrix into column-scaled Extract slices."""
    X_f32 = np.float32(X_f32)
    K, M = X_f32.shape
    slices = []
    scales = []
    residual = X_f32.copy()

    for _ in range(n_slices):
        col_max = np.float32(np.max(np.abs(residual), axis=0))  # (M,)

        zero_mask = (col_max == 0)
        col_max_safe = np.where(zero_mask, np.float32(1.0), col_max)

        c_y = np.floor(np.float32(np.log2(col_max_safe)))
        c_y = np.where(zero_mask, np.float32(0.0), c_y)

        sigma = np.float32(np.float32(0.75) * np.ldexp(
            np.ones(M, dtype=np.float32),
            (rho + c_y).astype(np.int32)))
        sigma_2d = sigma[np.newaxis, :]

        v = np.float32(np.float32(residual + sigma_2d) - sigma_2d)
        v = np.where(zero_mask[np.newaxis, :], np.float32(0.0), v)
        residual = np.float32(residual - v)

        inv_scale = np.float32(np.ldexp(
            np.ones(M, dtype=np.float32),
            (-c_y).astype(np.int32)))
        v_norm = np.float32(v * inv_scale[np.newaxis, :])

        slices.append(v_norm)
        scales.append(c_y)

    return slices, scales


# JAX extraction helpers for the fused on-device pipeline.


def jax_extract_split_rows(X_f32, rho, n_slices=5):
    """Split an FP32 matrix into row-scaled Extract slices (JAX).

    Same algorithm as f32_extract_split_rows but using jnp ops.
    NOT decorated with @jax.jit — called inside a larger JIT.
    Python loop is unrolled at trace time.

    Returns:
        (slices, scales): stacked arrays (n_slices, N, K) and (n_slices, N)
    """
    X_f32 = jnp.float32(X_f32)
    N = X_f32.shape[0]
    slices = []
    scales = []
    residual = X_f32

    for _ in range(n_slices):
        row_max = jnp.float32(jnp.max(jnp.abs(residual), axis=1))  # (N,)

        zero_mask = (row_max == 0)
        row_max_safe = jnp.where(zero_mask, jnp.float32(1.0), row_max)

        c_x = jnp.floor(jnp.float32(jnp.log2(row_max_safe)))
        c_x = jnp.where(zero_mask, jnp.float32(0.0), c_x)

        # Sigma = 0.75 * 2^(rho + c_x) in FP32.
        sigma = jnp.float32(jnp.float32(0.75) * jnp.ldexp(
            jnp.ones(N, dtype=jnp.float32),
            jnp.int32(rho + c_x)))
        sigma_2d = sigma[:, jnp.newaxis]

        # Sigma trick extracts top bits under FP32 rounding.
        v = jnp.float32(jnp.float32(residual + sigma_2d) - sigma_2d)
        v = jnp.where(zero_mask[:, jnp.newaxis], jnp.float32(0.0), v)
        residual = jnp.float32(residual - v)

        # Normalize by row scales.
        inv_scale = jnp.float32(jnp.ldexp(
            jnp.ones(N, dtype=jnp.float32),
            jnp.int32(-c_x)))
        v_norm = jnp.float32(v * inv_scale[:, jnp.newaxis])

        slices.append(v_norm)
        scales.append(c_x)

    return jnp.stack(slices), jnp.stack(scales)


def jax_extract_split_cols(X_f32, rho, n_slices=5):
    """Split an FP32 matrix into column-scaled Extract slices (JAX).

    Same algorithm as f32_extract_split_cols but using jnp ops.
    NOT decorated with @jax.jit — called inside a larger JIT.

    Returns:
        (slices, scales): stacked arrays (n_slices, K, M) and (n_slices, M)
    """
    X_f32 = jnp.float32(X_f32)
    M = X_f32.shape[1]
    slices = []
    scales = []
    residual = X_f32

    for _ in range(n_slices):
        col_max = jnp.float32(jnp.max(jnp.abs(residual), axis=0))  # (M,)

        zero_mask = (col_max == 0)
        col_max_safe = jnp.where(zero_mask, jnp.float32(1.0), col_max)

        c_y = jnp.floor(jnp.float32(jnp.log2(col_max_safe)))
        c_y = jnp.where(zero_mask, jnp.float32(0.0), c_y)

        sigma = jnp.float32(jnp.float32(0.75) * jnp.ldexp(
            jnp.ones(M, dtype=jnp.float32),
            jnp.int32(rho + c_y)))
        sigma_2d = sigma[jnp.newaxis, :]

        v = jnp.float32(jnp.float32(residual + sigma_2d) - sigma_2d)
        v = jnp.where(zero_mask[jnp.newaxis, :], jnp.float32(0.0), v)
        residual = jnp.float32(residual - v)

        inv_scale = jnp.float32(jnp.ldexp(
            jnp.ones(M, dtype=jnp.float32),
            jnp.int32(-c_y)))
        v_norm = jnp.float32(v * inv_scale[jnp.newaxis, :])

        slices.append(v_norm)
        scales.append(c_y)

    return jnp.stack(slices), jnp.stack(scales)
