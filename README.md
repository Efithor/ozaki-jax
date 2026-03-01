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

## Method summary

Given `C = A @ B`:

1. Split `A` and `B` into `n_slices` magnitude-controlled slices with Extract.
2. Compute BF16-derived GEMM pairs where `i + j <= n_slices - 1`.
3. Rescale and accumulate products in FP64.

With `n_slices = 8`, this keeps `8 * 9 / 2 = 36` slice-pair GEMMs.

## Exactness condition

For BF16 slice values bounded by `2^p - 1`, exact FP32 accumulation requires:

```
K * (2^p - 1)^2 < 2^24
```

In the default BF16 setting (`p = 7`), this gives `K <= 1040`.

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
```

## References

- Mukunoki, D., Ogita, T., & Imamura, T. (2020). "DGEMM using Tensor Cores, and Its Accurate and Reproducible Versions." *ISC High Performance 2020*.
- Mukunoki, D. (2025). "Ozaki Scheme-Based Accurate Matrix Multiplication on FP8 Tensor Cores." [arXiv:2508.00441](https://arxiv.org/abs/2508.00441)
- Ozaki, K., Ogita, T., Oishi, S., & Rump, S.M. (2012). "Error-Free Transformations of Matrix Multiplication by Using Fast Routines of Matrix Multiplication and Its Applications." *Numerical Algorithms*, 59(1).

## License

Apache 2.0
