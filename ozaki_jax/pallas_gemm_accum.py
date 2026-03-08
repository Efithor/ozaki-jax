"""Fused GEMM + 2Sum accumulation Pallas kernel for TPU.

Eliminates the (n_pairs, N, M) intermediate product tensor by keeping
per-pair accumulators in VMEM scratch and applying 2Sum at the end of
the K reduction loop.

Data flow (current vs fused):
  Current: slices → HBM(products) → HBM read → 2Sum → output
  Fused:   slices → VMEM(accumulators) → 2Sum (in VMEM) → output
  Savings: ~13 GB HBM traffic eliminated at n=8192
"""

import functools

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    _HAS_PALLAS_TPU = True
except ImportError:
    _HAS_PALLAS_TPU = False


def _make_pair_schedule(n_hi, n_lo):
    """Precompute pair indices and group structure for the kernel.

    Returns static (hashable) structures suitable for JIT static_argnums.
    """
    n_total = n_hi + n_lo

    # Pair schedule: (a_slice_idx, b_slice_idx) for each GEMM.
    # A slices: 0..n_hi-1 are hi, n_hi..n_total-1 are lo.
    # B slices: same indexing.
    pair_a = []
    pair_b = []

    # hi × hi
    for i in range(n_hi):
        for j in range(n_hi):
            pair_a.append(i)
            pair_b.append(j)
    # hi × lo
    for i in range(n_hi):
        for j in range(n_lo):
            pair_a.append(i)
            pair_b.append(n_hi + j)
    # lo × hi
    for i in range(n_lo):
        for j in range(n_hi):
            pair_a.append(n_hi + i)
            pair_b.append(j)

    pair_a = tuple(pair_a)
    pair_b = tuple(pair_b)
    n_pairs = len(pair_a)

    # Group structure for 2Sum: each group shares an A-slice row scale.
    # groups[g] = (a_scale_idx, (pair_idx_0, pair_idx_1, ...))
    groups = []
    idx = 0
    # hi×hi: n_hi groups of n_hi pairs
    for i in range(n_hi):
        g_pairs = tuple(range(idx, idx + n_hi))
        groups.append((i, g_pairs))
        idx += n_hi
    # hi×lo: n_hi groups of n_lo pairs
    for i in range(n_hi):
        g_pairs = tuple(range(idx, idx + n_lo))
        groups.append((i, g_pairs))
        idx += n_lo
    # lo×hi: n_lo groups of n_hi pairs
    for i in range(n_lo):
        g_pairs = tuple(range(idx, idx + n_hi))
        groups.append((n_hi + i, g_pairs))
        idx += n_hi

    groups = tuple((a, ps) for a, ps in groups)

    # Block boundaries for the 3-block 2Sum structure.
    # Block 0: hi×hi (groups 0..n_hi-1)
    # Block 1: hi×lo (groups n_hi..2*n_hi-1)
    # Block 2: lo×hi (groups 2*n_hi..2*n_hi+n_lo-1)
    block_boundaries = (n_hi, 2 * n_hi, 2 * n_hi + n_lo)

    return pair_a, pair_b, n_pairs, groups, block_boundaries


def _make_kernel(n_pairs, pair_a, pair_b, groups, block_boundaries, nsteps):
    """Build the fused GEMM+2Sum kernel function.

    All loop bounds and indices are compile-time constants, so the Python
    loops get unrolled at JAX trace time.

    Scales are pre-computed as FP32 power-of-2 values (not exponents)
    to avoid exp2 inside the kernel, which would trigger FP64 ops
    unsupported by Pallas TPU lowering.
    """

    def kernel(a_sl_ref, b_sl_ref, a_sc_ref, b_sc_ref,
               c_hi_ref, c_lo_ref, accum_ref):
        """Pallas kernel body.

        Refs:
            a_sl_ref:  (n_total, bm, bk) — A slice tiles
            b_sl_ref:  (n_total, bk, bn) — B slice tiles
            a_sc_ref:  (n_total, bm) — A row scales (pre-computed 2^exponent)
            b_sc_ref:  (n_total, bn) — B col scales (pre-computed 2^exponent)
            c_hi_ref:  (bm, bn) — output hi part
            c_lo_ref:  (bm, bn) — output lo part
            accum_ref: (n_pairs, bm, bn) — VMEM scratch accumulators
        """
        k = pl.program_id(2)

        # Initialize accumulators on first K tile.
        @pl.when(k == 0)
        def _():
            accum_ref[...] = jnp.zeros_like(accum_ref)

        # Accumulate partial dot products for all pairs.
        for p in range(n_pairs):
            sa = pair_a[p]
            sb = pair_b[p]
            accum_ref[p, :, :] += jnp.dot(
                a_sl_ref[sa, :, :],
                b_sl_ref[sb, :, :],
                preferred_element_type=jnp.float32,
            )

        # On the last K tile: apply scales and 2Sum accumulate.
        @pl.when(k == nsteps - 1)
        def _():
            bm = c_hi_ref.shape[0]
            bn = c_hi_ref.shape[1]

            def twosum_add(s_hi, s_lo, x):
                t = jnp.float32(s_hi + x)
                e = jnp.float32(jnp.float32(s_hi - t) + x)
                return t, jnp.float32(s_lo + e)

            # Process each block (hi×hi, hi×lo, lo×hi) separately.
            block_hi_list = []
            block_lo_list = []
            prev = 0

            for blk_end in block_boundaries:
                blk_hi = jnp.zeros((bm, bn), dtype=jnp.float32)
                blk_lo = jnp.zeros((bm, bn), dtype=jnp.float32)

                for g_idx in range(prev, blk_end):
                    a_idx, g_pairs = groups[g_idx]
                    inner_hi = jnp.zeros((bm, bn), dtype=jnp.float32)
                    inner_lo = jnp.zeros((bm, bn), dtype=jnp.float32)

                    for p in g_pairs:
                        # Scales are pre-computed: already 2^exponent
                        col_sc = b_sc_ref[pair_b[p], :]
                        scaled = jnp.float32(
                            accum_ref[p, :, :] * col_sc[jnp.newaxis, :])
                        inner_hi, inner_lo = twosum_add(
                            inner_hi, inner_lo, scaled)

                    row_sc = a_sc_ref[a_idx, :]
                    inner_hi = jnp.float32(
                        inner_hi * row_sc[:, jnp.newaxis])
                    inner_lo = jnp.float32(
                        inner_lo * row_sc[:, jnp.newaxis])

                    blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_hi)
                    blk_hi, blk_lo = twosum_add(blk_hi, blk_lo, inner_lo)

                block_hi_list.append(blk_hi)
                block_lo_list.append(blk_lo)
                prev = blk_end

            # Combine block results: hi parts first, then lo parts.
            final_hi = jnp.zeros((bm, bn), dtype=jnp.float32)
            final_lo = jnp.zeros((bm, bn), dtype=jnp.float32)
            for bh in block_hi_list:
                final_hi, final_lo = twosum_add(final_hi, final_lo, bh)
            for bl in block_lo_list:
                final_hi, final_lo = twosum_add(final_hi, final_lo, bl)

            c_hi_ref[...] = final_hi
            c_lo_ref[...] = final_lo

    return kernel


def fused_gemm_accum(a_slices, b_slices, a_scales, b_scales,
                     n_hi, n_lo, *, bm=128, bk=128, bn=128):
    """Fused GEMM + 2Sum accumulation via Pallas on TPU.

    Takes pre-extracted slices and computes the full Ozaki product
    with 2Sum accumulation, keeping all intermediates in VMEM.

    Args:
        a_slices: (n_total, N, K) FP32 — stacked A extraction slices
        b_slices: (n_total, K, M) FP32 — stacked B extraction slices
        a_scales: (n_total, N) FP32 — A row scale exponents
        b_scales: (n_total, M) FP32 — B col scale exponents
        n_hi: Number of hi extraction slices
        n_lo: Number of lo extraction slices
        bm, bk, bn: Tile sizes (must be multiples of 128)

    Returns:
        (C_hi, C_lo): FP32 arrays of shape (N, M)
    """
    if not _HAS_PALLAS_TPU:
        raise RuntimeError(
            "Pallas TPU not available. Install jax[tpu].")

    n_total = n_hi + n_lo
    N = a_slices.shape[1]
    K = a_slices.shape[2]
    M = b_slices.shape[2]
    nsteps = K // bk

    pair_a, pair_b, n_pairs, groups, block_boundaries = \
        _make_pair_schedule(n_hi, n_lo)

    kernel_fn = _make_kernel(
        n_pairs, pair_a, pair_b, groups, block_boundaries, nsteps)

    # Pre-compute scales: convert exponents to FP32 power-of-2 values.
    # This avoids exp2 inside the Pallas kernel (which triggers FP64 ops
    # unsupported by Mosaic TPU lowering).
    a_scales_pow2 = jnp.ldexp(jnp.ones_like(a_scales, dtype=jnp.float32),
                              jnp.int32(a_scales))
    b_scales_pow2 = jnp.ldexp(jnp.ones_like(b_scales, dtype=jnp.float32),
                              jnp.int32(b_scales))

    grid = (N // bm, M // bn, nsteps)

    c_hi, c_lo = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[
                # A slices: load all n_total slices for current (i, k)
                pl.BlockSpec(
                    (n_total, bm, bk),
                    lambda i, j, k: (0, i, k)),
                # B slices: load all n_total slices for current (k, j)
                pl.BlockSpec(
                    (n_total, bk, bn),
                    lambda i, j, k: (0, k, j)),
                # A scales (pre-computed 2^exp): all for current i
                pl.BlockSpec(
                    (n_total, bm),
                    lambda i, j, k: (0, i)),
                # B scales (pre-computed 2^exp): all for current j
                pl.BlockSpec(
                    (n_total, bn),
                    lambda i, j, k: (0, j)),
            ],
            out_specs=[
                pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
                pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            ],
            scratch_shapes=[
                pltpu.VMEM((n_pairs, bm, bn), jnp.float32),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((N, M), jnp.float32),
            jax.ShapeDtypeStruct((N, M), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(a_slices, b_slices, a_scales_pow2, b_scales_pow2)

    return c_hi, c_lo


def fused_ozaki_matmul(A_f64, B_f64, rho, n_hi, n_lo, *,
                       bm=128, bk=128, bn=128):
    """Full Ozaki pipeline with fused Pallas GEMM+Accum.

    Does split + extraction in standard JAX, then fuses GEMMs + 2Sum
    in a single Pallas kernel. Returns FP64 result.

    Args:
        A_f64, B_f64: FP64 JAX arrays
        rho: Extraction parameter (from _compute_rho_f32)
        n_hi, n_lo: Extraction slice counts
        bm, bk, bn: Pallas tile sizes
    """
    from .matmul import _jax_double_f32_split
    from .extract import jax_extract_split_rows, jax_extract_split_cols

    # Phase 1: Split FP64 → (FP32_hi, FP32_lo)
    A_hi, A_lo = _jax_double_f32_split(A_f64)
    B_hi, B_lo = _jax_double_f32_split(B_f64)

    # Phase 2: Extraction (standard JAX — runs on vector unit)
    A_hi_sl, A_hi_sc = jax_extract_split_rows(A_hi, rho, n_hi)
    B_hi_sl, B_hi_sc = jax_extract_split_cols(B_hi, rho, n_hi)
    A_lo_sl, A_lo_sc = jax_extract_split_rows(A_lo, rho, n_lo)
    B_lo_sl, B_lo_sc = jax_extract_split_cols(B_lo, rho, n_lo)

    # Stack slices: (n_hi + n_lo, N, K) and (n_hi + n_lo, K, M)
    a_slices = jnp.concatenate([A_hi_sl, A_lo_sl], axis=0)
    b_slices = jnp.concatenate([B_hi_sl, B_lo_sl], axis=0)
    a_scales = jnp.concatenate([A_hi_sc, A_lo_sc], axis=0)
    b_scales = jnp.concatenate([B_hi_sc, B_lo_sc], axis=0)

    # Phase 3: Fused GEMM + 2Sum (Pallas kernel)
    C_hi, C_lo = fused_gemm_accum(
        a_slices, b_slices, a_scales, b_scales,
        n_hi, n_lo, bm=bm, bk=bk, bn=bn)

    # Phase 4: Combine to FP64
    return jnp.float64(C_hi) + jnp.float64(C_lo)
