"""Extract-splitting utilities for FP64 inputs and BF16-like slices."""

import numpy as np


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
