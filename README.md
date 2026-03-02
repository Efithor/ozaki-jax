# ozaki-jax

FP64-accurate matrix multiplication built from BF16 GEMMs using Ozaki Extract in JAX.

For square test matrices, the default implementation uses 36 BF16 GEMMs and reaches about `1e-16` relative error against FP64 reference results.

## Usage

```python
import numpy as np
from ozaki_jax import matmul

A = np.random.randn(256, 256)
B = np.random.randn(256, 256)
C = matmul(A, B)
```

Without JAX runtime dependency:

```python
from ozaki_jax import matmul_numpy
C = matmul_numpy(A, B)
```

On-device extraction path (fixed 65 GEMMs, default `accumulation="fused"`):

```python
C = matmul(A, B, pipeline="ondevice")
```

On-device extraction with explicit on-device 2Sum accumulation:

```python
C = matmul(A, B, pipeline="ondevice", accumulation="ondevice")
```

For `accumulation="fused"`, enable JAX x64 before using `matmul`:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Note: TPU does not provide high-throughput native FP64 in the same way as CPU/GPU
HPC backends. In this project, fused mode uses x64 primarily for on-device
`hi/lo` splitting before FP32/BF16-style stages. It can reduce host overhead, but
speedups are workload and device dependent; profile before treating it as the
fastest default for a given deployment.

Safety preflight with explicit fallback:

```python
# Reject unsafe Ozaki configurations (default behavior).
C = matmul(A, B, safe_mode="raise")

# Fallback to plain FP64 matmul when preflight fails.
C = matmul(A, B, safe_mode="fallback")
```

## Method summary

Given `C = A @ B`:

1. Split `A` and `B` into magnitude-controlled slices with Extract.
2. Run BF16-derived GEMMs for the selected pipeline.
3. Rescale and accumulate products.

Pipeline options:

- `pipeline="host"` (default): FP64 host extraction, triangular pairing, 36 GEMMs at `n_slices=8`.
- `pipeline="ondevice"`: FP32 extraction (`hi/lo` split), fixed block structure, 65 GEMMs.

Accumulation options (`pipeline="ondevice"` only):

- `accumulation="fused"` (default): split, extraction, GEMMs, and 2Sum accumulation in one JIT call (requires JAX x64 enabled).
- `accumulation="ondevice"`: separate on-device FP32 2Sum accumulation path.
- `accumulation="host"`: transfer products and accumulate in FP64 on host.

## Exactness condition

For BF16 slice values bounded by `2^p - 1`, exact FP32 accumulation requires:

```
K * (2^p - 1)^2 < 2^24
```

In the default BF16 setting (`p = 7`), this gives `K <= 1040`.

## Safety preflight

`matmul()` and `matmul_numpy()` run a preflight safety check before the Ozaki path.
An input is considered safe only if all checks pass:

- BF16->FP32 exactness bound: `K * (2^p - 1)^2 < 2^24`
- Mantissa coverage: `n_slices * p >= 53`
- Inputs are finite (no `NaN`/`Inf`)
- Shapes are rank-2 and matmul-compatible

On preflight failure:

- `safe_mode="raise"`: raises `ValueError`
- `safe_mode="fallback"`: returns plain FP64 `A @ B`

For `pipeline="ondevice"`, preflight uses FP32 extraction constraints and the fixed
slice configuration (`n_hi=5`, `n_lo=4`).

## Notes on rho for BF16

For FP64 source values (`m1=53`) and BF16 storage (`m2=8`), storage exactness requires:

```
rho >= m1 + 1 - m2 = 46
```

The accumulation constraint is also at least 46 for typical `K` values in this project, so the implementation uses `rho=46` as the lower bound.

## Limitations

- `K <= 1040` for the default unblocked exactness condition.
- JAX on CPU does not represent TPU BF16 execution behavior.
- Blocking for larger `K` is not implemented.

## Benchmarks

```bash
python benchmarks/precision.py
python benchmarks/tpu_validate.py
python benchmarks/ondevice_validate.py
```

## References

- Mukunoki, D., Ogita, T., & Imamura, T. (2020). "DGEMM using Tensor Cores, and Its Accurate and Reproducible Versions." *ISC High Performance 2020*.
- Mukunoki, D. (2025). "Ozaki Scheme-Based Accurate Matrix Multiplication on FP8 Tensor Cores." [arXiv:2508.00441](https://arxiv.org/abs/2508.00441)
- Ozaki, K., Ogita, T., Oishi, S., & Rump, S.M. (2012). "Error-Free Transformations of Matrix Multiplication by Using Fast Routines of Matrix Multiplication and Its Applications." *Numerical Algorithms*, 59(1).

## License

Apache 2.0
