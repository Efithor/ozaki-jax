"""Sigma-trick and 2Sum accumulation helpers for JAX and optional Pallas."""

import functools

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

try:
    from jax.experimental import pallas as pl
    _HAS_PALLAS = True
except ImportError:
    _HAS_PALLAS = False


def _sigma_trick_kernel(x_ref, sigma_ref, out_ref):
    """Pallas kernel: element-wise sigma trick in FP32."""
    x = x_ref[...]
    sigma = sigma_ref[...]
    # Force FP32 intermediate: add then subtract sigma
    tmp = (x + sigma).astype(jnp.float32)
    out_ref[...] = (tmp - sigma).astype(jnp.float32)


def pallas_sigma_trick(X_f32, sigma_2d):
    """Apply sigma trick via Pallas kernel."""
    if not _HAS_PALLAS:
        raise RuntimeError("Pallas not available; install jax[tpu] or use jax_sigma_trick.")

    X_f32 = jnp.asarray(X_f32, dtype=jnp.float32)
    sigma_2d = jnp.broadcast_to(jnp.asarray(sigma_2d, dtype=jnp.float32), X_f32.shape)

    return pl.pallas_call(
        _sigma_trick_kernel,
        out_shape=jax.ShapeDtypeStruct(X_f32.shape, jnp.float32),
    )(X_f32, sigma_2d)


def jax_sigma_trick(X_f32, sigma_2d):
    """Apply sigma trick via standard JAX ops."""
    if not _HAS_JAX:
        raise RuntimeError("JAX not available.")

    X_f32 = jnp.float32(X_f32)
    sigma_2d = jnp.float32(sigma_2d)
    return jnp.float32(jnp.float32(X_f32 + sigma_2d) - sigma_2d)


def sigma_trick(X_f32, sigma_2d, backend="auto"):
    """Dispatch sigma trick backend: `auto`, `jax`, or `pallas`."""
    if backend == "pallas":
        return pallas_sigma_trick(X_f32, sigma_2d)
    if backend == "jax" or backend == "auto":
        return jax_sigma_trick(X_f32, sigma_2d)
    raise ValueError(f"Unknown backend={backend!r}; expected 'auto', 'jax', or 'pallas'.")


def validate_sigma_trick_rounding(K=256):
    """Compare sigma-trick outputs against NumPy FP32 reference."""
    rng = np.random.RandomState(42)
    N = 64
    X = rng.randn(N, K).astype(np.float32)

    # Build sigma per row (same logic as f32_extract_split_rows)
    row_max = np.float32(np.max(np.abs(X), axis=1))
    c_x = np.floor(np.float32(np.log2(np.where(row_max == 0, 1.0, row_max))))
    rho = 17  # FP32 rho
    sigma = np.float32(np.float32(0.75) * np.ldexp(
        np.ones(N, dtype=np.float32), (rho + c_x).astype(np.int32)))
    sigma_2d = sigma[:, np.newaxis]

    # Numpy FP32 reference
    np_result = np.float32(np.float32(X + sigma_2d) - sigma_2d)

    # JAX result
    if not _HAS_JAX:
        return {"match": None, "max_diff": None, "error": "JAX not available"}

    jax_result = np.array(jax_sigma_trick(X, sigma_2d))
    max_diff = float(np.max(np.abs(jax_result - np_result)))

    result = {"match": max_diff == 0.0, "max_diff": max_diff}

    # Also test Pallas if available
    if _HAS_PALLAS:
        try:
            pallas_result = np.array(pallas_sigma_trick(X, sigma_2d))
            pallas_diff = float(np.max(np.abs(pallas_result - np_result)))
            result["pallas_match"] = pallas_diff == 0.0
            result["pallas_max_diff"] = pallas_diff
        except Exception as e:
            result["pallas_error"] = str(e)

    return result


# ====================================================================
# 2Sum accumulation for on-device pipeline
# ====================================================================


def _precompute_accumulation_scales(A_hi_scales, A_lo_scales,
                                    B_hi_scales, B_lo_scales,
                                    N, M, n_hi, n_lo):
    """Precompute flat scale arrays for on-device 2Sum accumulation.

    Returns:
        col_scales: (n_products, M) FP32 — per-product column scale vectors
        row_scales: (n_groups, N) FP32 — per-group row scale vectors
        block_group_sizes: tuple of tuples — ((sizes for hixhi), (hixlo), (loxhi))
    """
    n_hi_b = n_hi  # B's hi slice count matches A's hi count

    # col_scales: one per product in GEMM order
    col_scales_list = []

    # hixhi: n_hi groups x n_hi_b products each
    for i in range(n_hi):
        for j in range(n_hi_b):
            col_scales_list.append(
                np.ldexp(np.ones(M, dtype=np.float32),
                         B_hi_scales[j].astype(np.int32)))

    # hixlo: n_hi groups x n_lo products each
    for i in range(n_hi):
        for j in range(n_lo):
            col_scales_list.append(
                np.ldexp(np.ones(M, dtype=np.float32),
                         B_lo_scales[j].astype(np.int32)))

    # loxhi: n_lo groups x n_hi_b products each
    for i in range(n_lo):
        for j in range(n_hi_b):
            col_scales_list.append(
                np.ldexp(np.ones(M, dtype=np.float32),
                         B_hi_scales[j].astype(np.int32)))

    col_scales_f32 = np.stack(col_scales_list)  # (65, M)

    # row_scales: one per group
    row_scales_list = []

    # Groups 0..n_hi-1 (hixhi): row = A_hi_scales[i]
    for i in range(n_hi):
        row_scales_list.append(
            np.ldexp(np.ones(N, dtype=np.float32),
                     A_hi_scales[i].astype(np.int32)))

    # Groups n_hi..2*n_hi-1 (hixlo): same A_hi row-scales
    for i in range(n_hi):
        row_scales_list.append(
            np.ldexp(np.ones(N, dtype=np.float32),
                     A_hi_scales[i].astype(np.int32)))

    # Groups 2*n_hi..2*n_hi+n_lo-1 (loxhi): row = A_lo_scales[i]
    for i in range(n_lo):
        row_scales_list.append(
            np.ldexp(np.ones(N, dtype=np.float32),
                     A_lo_scales[i].astype(np.int32)))

    row_scales_f32 = np.stack(row_scales_list)  # (14, N)

    # Block structure: 3 blocks with different dynamic ranges.
    block_group_sizes = (
        tuple([n_hi_b] * n_hi),   # hixhi: 5 groups of 5
        tuple([n_lo] * n_hi),     # hixlo: 5 groups of 4
        tuple([n_hi_b] * n_lo),   # loxhi: 4 groups of 5
    )

    return col_scales_f32, row_scales_f32, block_group_sizes


def _accumulate_2sum_logic(products, col_scales, row_scales, block_group_sizes):
    """Core 2Sum accumulation logic (no JIT).

    Called by both the standalone JIT wrapper and the fused pipeline.

    Args:
        products: (n_products, N, M) FP32 — raw GEMM outputs
        col_scales: (n_products, M) FP32 — per-product column scales
        row_scales: (n_groups, N) FP32 — per-group row scales
        block_group_sizes: tuple of tuples — ((sizes for hixhi), (hixlo), (loxhi))

    Returns:
        (C_hi, C_lo): FP32 arrays of shape (N, M)
    """
    N = products.shape[1]
    M = products.shape[2]

    def twosum_add(s_hi, s_lo, x):
        """2Sum: add x into (s_hi, s_lo) accumulator."""
        t = s_hi + x
        e = (s_hi - t) + x
        return t, s_lo + e

    # Accumulate each block separately to preserve precision.
    block_results = []
    prod_idx = 0
    group_idx = 0

    for block_sizes in block_group_sizes:
        blk_hi = jnp.zeros((N, M), dtype=jnp.float32)
        blk_lo = jnp.zeros((N, M), dtype=jnp.float32)

        for g_size in block_sizes:
            inner_hi = jnp.zeros((N, M), dtype=jnp.float32)
            inner_lo = jnp.zeros((N, M), dtype=jnp.float32)

            for _ in range(g_size):
                scaled = products[prod_idx] * col_scales[prod_idx]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
                prod_idx += 1

            # Row-scale is exact power-of-2: multiply is exact in FP32.
            inner_hi = inner_hi * row_scales[group_idx][:, None]
            inner_lo = inner_lo * row_scales[group_idx][:, None]

            blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_hi)
            blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_lo)
            group_idx += 1

        block_results.append((blk_hi, blk_lo))

    # Combine block results: add hi parts first (largest to smallest),
    # then lo parts. The 3 blocks have different magnitudes (hixhi >>
    # hixlo ≈ loxhi), so order matters.
    C_hi = jnp.zeros((N, M), dtype=jnp.float32)
    C_lo = jnp.zeros((N, M), dtype=jnp.float32)
    for blk_hi, _ in block_results:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_hi)
    for _, blk_lo in block_results:
        C_hi, C_lo = twosum_add(C_hi, C_lo, blk_lo)

    return C_hi, C_lo


@functools.partial(jax.jit, static_argnums=(3,))
def _jax_accumulate_2sum(products, col_scales, row_scales, block_group_sizes):
    """Standalone JIT wrapper for 2Sum accumulation (backward compat)."""
    return _accumulate_2sum_logic(products, col_scales, row_scales,
                                  block_group_sizes)


def _pallas_accumulate_2sum_kernel(products_ref, col_scales_ref,
                                   row_scales_ref, c_hi_ref, c_lo_ref):
    """Pallas kernel: tiled 2Sum accumulation over GEMM products.

    Uses block-level accumulation matching the JAX path. The group
    structure (5 groups of 5, 5 groups of 4, 4 groups of 5) is
    hard-coded to match n_hi=5, n_lo=4.
    """
    def twosum_add(s_hi, s_lo, x):
        t = s_hi + x
        e = (s_hi - t) + x
        return t, s_lo + e

    BN = c_hi_ref.shape[0]
    BM = c_hi_ref.shape[1]

    # Block structure: [5]*5 + [4]*5 + [5]*4 = 14 groups, 65 products
    block_group_sizes = ((5, 5, 5, 5, 5), (4, 4, 4, 4, 4), (5, 5, 5, 5))

    block_results_hi = []
    block_results_lo = []
    prod_idx = 0
    group_idx = 0

    for block_sizes in block_group_sizes:
        blk_hi = jnp.zeros((BN, BM), dtype=jnp.float32)
        blk_lo = jnp.zeros((BN, BM), dtype=jnp.float32)

        for g_size in block_sizes:
            inner_hi = jnp.zeros((BN, BM), dtype=jnp.float32)
            inner_lo = jnp.zeros((BN, BM), dtype=jnp.float32)

            for _ in range(g_size):
                scaled = products_ref[prod_idx] * col_scales_ref[prod_idx]
                inner_hi, inner_lo = twosum_add(inner_hi, inner_lo, scaled)
                prod_idx += 1

            row_sc = row_scales_ref[group_idx][:, None]
            inner_hi = inner_hi * row_sc
            inner_lo = inner_lo * row_sc

            blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_hi)
            blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_lo)
            group_idx += 1

        block_results_hi.append(blk_hi)
        block_results_lo.append(blk_lo)

    # Combine: hi parts first, then lo parts.
    C_hi = jnp.zeros((BN, BM), dtype=jnp.float32)
    C_lo = jnp.zeros((BN, BM), dtype=jnp.float32)
    for bh in block_results_hi:
        C_hi, C_lo = twosum_add(C_hi, C_lo, bh)
    for bl in block_results_lo:
        C_hi, C_lo = twosum_add(C_hi, C_lo, bl)

    c_hi_ref[...] = C_hi
    c_lo_ref[...] = C_lo


_ACCUM_BLOCK = 128  # TPU-friendly tile size


def _pad_to_multiple(x, block, axis):
    """Pad array along axis to a multiple of block size."""
    size = x.shape[axis]
    remainder = size % block
    if remainder == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, block - remainder)
    return jnp.pad(x, pad_width)


def pallas_accumulate_2sum(products, col_scales, row_scales,
                           block_group_sizes, N, M):
    """2Sum accumulation via Pallas tiled kernel.

    Handles padding to BLOCK multiples and strips padding from output.
    """
    if not _HAS_PALLAS:
        raise RuntimeError("Pallas not available; use backend='jax'.")

    BLOCK = _ACCUM_BLOCK
    n_products = products.shape[0]
    n_groups = row_scales.shape[0]

    # Pad N and M dimensions to block multiples.
    N_pad = N + (-N % BLOCK)
    M_pad = M + (-M % BLOCK)

    products_p = _pad_to_multiple(
        _pad_to_multiple(products, BLOCK, axis=1), BLOCK, axis=2)
    col_scales_p = _pad_to_multiple(col_scales, BLOCK, axis=1)
    row_scales_p = _pad_to_multiple(row_scales, BLOCK, axis=1)

    grid = (N_pad // BLOCK, M_pad // BLOCK)

    c_hi, c_lo = pl.pallas_call(
        _pallas_accumulate_2sum_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((N_pad, M_pad), jnp.float32),
            jax.ShapeDtypeStruct((N_pad, M_pad), jnp.float32),
        ),
        in_specs=[
            pl.BlockSpec((n_products, BLOCK, BLOCK),
                         lambda i, j: (0, i, j)),
            pl.BlockSpec((n_products, BLOCK),
                         lambda i, j: (0, j)),
            pl.BlockSpec((n_groups, BLOCK),
                         lambda i, j: (0, i)),
        ],
        out_specs=[
            pl.BlockSpec((BLOCK, BLOCK), lambda i, j: (i, j)),
            pl.BlockSpec((BLOCK, BLOCK), lambda i, j: (i, j)),
        ],
        grid=grid,
    )(products_p, col_scales_p, row_scales_p)

    # Strip padding.
    return c_hi[:N, :M], c_lo[:N, :M]


def accumulate_2sum(products, col_scales, row_scales, block_group_sizes,
                    N=None, M=None, backend="auto"):
    """Dispatch 2Sum accumulation: 'auto'/'jax' or 'pallas'.

    Args:
        products: (n_products, N, M) FP32 JAX array
        col_scales: (n_products, M) FP32 JAX array
        row_scales: (n_groups, N) FP32 JAX array
        block_group_sizes: tuple of tuples — ((hixhi sizes), (hixlo), (loxhi))
        N, M: original matrix dimensions (needed for Pallas padding)
        backend: 'auto', 'jax', or 'pallas'

    Returns:
        (C_hi, C_lo): FP32 JAX arrays of shape (N, M)
    """
    if backend == "pallas":
        if N is None or M is None:
            N, M = products.shape[1], products.shape[2]
        return pallas_accumulate_2sum(products, col_scales, row_scales,
                                      block_group_sizes, N, M)
    if backend in ("auto", "jax"):
        return _jax_accumulate_2sum(products, col_scales, row_scales,
                                    block_group_sizes)
    raise ValueError(
        f"Unknown backend={backend!r}; expected 'auto', 'jax', or 'pallas'.")


def validate_accumulation_2sum(N=128, M=128, K=128):
    """Compare 2Sum accumulation against FP64 baseline.

    Returns dict with match status and error metrics.
    """
    from .matmul import (
        _double_f32_split, _accumulate_block_products,
        _PRECISION_PRESETS,
    )
    from .extract import _compute_rho_f32, f32_extract_split_rows, f32_extract_split_cols

    rng = np.random.RandomState(42)
    A = rng.randn(N, K).astype(np.float64)
    B = rng.randn(K, M).astype(np.float64)
    C_exact = A @ B

    n_hi, n_lo = _PRECISION_PRESETS["max"]
    rho = _compute_rho_f32(K)

    A_hi, A_lo = _double_f32_split(A)
    B_hi, B_lo = _double_f32_split(B)

    A_hi_sl, A_hi_sc = f32_extract_split_rows(A_hi, rho, n_hi)
    A_lo_sl, A_lo_sc = f32_extract_split_rows(A_lo, rho, n_lo)
    B_hi_sl, B_hi_sc = f32_extract_split_cols(B_hi, rho, n_hi)
    B_lo_sl, B_lo_sc = f32_extract_split_cols(B_lo, rho, n_lo)

    # Compute products (numpy FP32 GEMMs).
    products_list = []
    for i in range(n_hi):
        for j in range(n_hi):
            products_list.append(np.float32(A_hi_sl[i] @ B_hi_sl[j]))
    for i in range(n_hi):
        for j in range(n_lo):
            products_list.append(np.float32(A_hi_sl[i] @ B_lo_sl[j]))
    for i in range(n_lo):
        for j in range(n_hi):
            products_list.append(np.float32(A_lo_sl[i] @ B_hi_sl[j]))
    products_np = np.stack(products_list)

    # FP64 baseline.
    C_baseline = _accumulate_block_products(
        products_np, A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc, N, M, n_hi, n_lo, n_hi)
    baseline_err = float(np.max(np.abs(C_baseline - C_exact)) / np.max(np.abs(C_exact)))

    # 2Sum accumulation.
    col_scales, row_scales, block_group_sizes = _precompute_accumulation_scales(
        A_hi_sc, A_lo_sc, B_hi_sc, B_lo_sc, N, M, n_hi, n_lo)

    products_jax = jnp.array(products_np)
    col_scales_jax = jnp.array(col_scales)
    row_scales_jax = jnp.array(row_scales)

    C_hi, C_lo = accumulate_2sum(products_jax, col_scales_jax, row_scales_jax,
                                  block_group_sizes)
    C_2sum = np.float64(np.array(C_hi)) + np.float64(np.array(C_lo))
    twosum_err = float(np.max(np.abs(C_2sum - C_exact)) / np.max(np.abs(C_exact)))

    # FP32 2Sum achieves ~1e-10 accuracy (limited by FP32 outer accumulation).
    # This is much better than naive FP32 (~1e-4) but worse than FP64 host
    # accumulation (~1e-15). The 2Sum error scales as O(n * eps_f32^2 * cond).
    match = twosum_err < 1e-9

    return {
        "match": match,
        "max_rel_error": twosum_err,
        "baseline_error": baseline_err,
        "N": N, "M": M, "K": K,
    }
