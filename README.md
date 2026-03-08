# ozaki-jax

FP64-accurate linear algebra on BF16/FP32-oriented hardware (e.g. TPU) using Ozaki Extract in JAX.

`ozaki-jax` features accurate matmul with higher-level routines for Gram matrices, residuals, and iterative-refinement solves.

## Install

```bash
# inside an active environment
uv pip install -e .
```

Optional TPU dependency:

```bash
uv pip install -e ".[tpu]"
```

## Public API

```python
from ozaki_jax import gram, matmul, matmul_numpy, residual, solve
```

- `matmul(A, B, ...)`: accurate matrix multiply with host or on-device Ozaki pipelines.
- `matmul_numpy(A, B, ...)`: NumPy-only mirror for host path and testing.
- `gram(A, ...)`: computes symmetric `A.T @ A`.
- `residual(A, x, b, ...)`: computes accurate `b - A @ x`.
- `solve(A, b, ...)`: iterative refinement solve with FP32 factorization plus accurate residuals.

Input/output typing:

- NumPy input -> NumPy output
- JAX input -> JAX output
- Mixed NumPy/JAX inputs are rejected

## Quick Start

### 1) Accurate Matmul (Default Host Pipeline)

```python
import numpy as np
from ozaki_jax import matmul

A = np.random.randn(256, 256)
B = np.random.randn(256, 256)
C = matmul(A, B)
```

### 2) On-device Pipeline

```python
import jax
from ozaki_jax import matmul

jax.config.update("jax_enable_x64", True)

C = matmul(A, B, pipeline="ondevice", accumulation="fused")
# or:
C2 = matmul(A, B, pipeline="ondevice", accumulation="bf16_interleaved")
```

### 3) Gram / Residual / Solve

```python
import numpy as np
import jax
from ozaki_jax import gram, residual, solve

jax.config.update("jax_enable_x64", True)

A = np.random.randn(256, 256)
b = np.random.randn(256)
x = solve(A, b)                # default: residual_mode="f64"
r = residual(A, x, b)          # default: mode="f64"
G = gram(A)                    # default: mode="f64", symmetric result
```

## Matmul Options

`matmul(A, B, n_slices=8, safe_mode="raise", pipeline="host", accumulation="fused", precision="high")`

Pipeline:

- `pipeline="host"` (default): FP64 host extraction + triangular pair scheduling.
- `pipeline="ondevice"`: FP64 input -> FP32 hi/lo split + on-device extract/GEMM/accumulation.

On-device accumulation (`pipeline="ondevice"` only):

- `fused` (default): split + extraction + GEMMs + 2Sum in one JIT call.
- `bf16_interleaved`: BF16-cast extracted slices, interleaved GEMM+2Sum.
- `ondevice`: separate on-device 2Sum accumulation after GEMMs.
- `host`: transfer products and accumulate on host in FP64.

On-device precision presets (`precision=`):

- `high` -> `(n_hi=4, n_lo=1)` -> 24 GEMMs
- `medium` -> `(n_hi=3, n_lo=1)` -> 15 GEMMs
- `max` -> `(n_hi=5, n_lo=4)` -> 65 GEMMs
- custom tuple `(n_hi, n_lo)` is supported

Safety behavior (`safe_mode=`):

- `raise` (default): raises `ValueError` when preflight fails.
- `fallback`: returns plain FP64 `A @ B` when preflight fails.

## Gram / Residual / Solve Details

### `gram(A, precision="high", accumulation="bf16_interleaved", mode="f64")`

- Computes `A.T @ A`, then symmetrizes via `(G + G.T) / 2`.
- `mode="f64"` (default): native FP64 matmul, highest accuracy.
- `mode="ozaki"`: uses Ozaki pipeline and `precision`/`accumulation`.

### `residual(A, x, b, precision="high", accumulation="bf16_interleaved", mode="f64")`

- Computes `b - A @ x` for vector or matrix `x`.
- Supports rectangular `A`.
- `mode="f64"` (default) or `mode="ozaki"`.

### `solve(A, b, precision="high", accumulation="bf16_interleaved", max_iterations=3, residual_mode="f64")`

- Solves `A x = b` using iterative refinement.
- Uses FP32 solve steps with accurate residual recomputation.
- `residual_mode="f64"` (default): best accuracy and usually best convergence.
- `residual_mode="ozaki"`: available when FP64 throughput is constrained.

## x64 Requirement

Enable x64 before using:

- `matmul(..., pipeline="ondevice", accumulation in {"fused", "bf16_interleaved"})`
- `gram()`
- `residual()`
- `solve()`

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Safety and Exactness

Preflight checks (for `matmul`/`matmul_numpy`) include:

- Rank/shape compatibility
- Finite inputs (no `NaN`/`Inf`)
- Mantissa/extract budget constraints
- BF16->FP32 exact inner-product bound

Core exactness condition for BF16-bounded slice values:

```
K * (2^p - 1)^2 < 2^24
```

In the default BF16 setting (`p = 7`), this gives `K <= 1040`.

## Validation and Benchmark Scripts

Run from repo root with `uv run`:

```bash
# Core accuracy/validation
uv run python benchmarks/ci_cpu_validate.py
uv run python benchmarks/ci_gram_validate.py
uv run python benchmarks/ci_solve_validate.py
uv run python benchmarks/tpu_validate.py
uv run python benchmarks/tpu_full_validate.py

# Performance / profiling sweeps
uv run python benchmarks/bench_bf16_interleaved.py
uv run python benchmarks/bench_bf16_broadcast.py
uv run python benchmarks/bench_interleaved.py
uv run python benchmarks/tpu_phase_profile.py
uv run python benchmarks/tpu_scaling_sweep.py
```

## Limitations

- Unblocked exactness model still limits supported `K` for strict guarantees.
- CPU execution does not represent TPU BF16/MXU behavior.
- TPU speedups depend on matrix size, backend, and accumulation mode; profile on target hardware.

## References

- Mukunoki, D., Ogita, T., & Imamura, T. (2020). "DGEMM using Tensor Cores, and Its Accurate and Reproducible Versions." *ISC High Performance 2020*.
- Mukunoki, D. (2025). "Ozaki Scheme-Based Accurate Matrix Multiplication on FP8 Tensor Cores." [arXiv:2508.00441](https://arxiv.org/abs/2508.00441)
- Ozaki, K., Ogita, T., Oishi, S., & Rump, S.M. (2012). "Error-Free Transformations of Matrix Multiplication by Using Fast Routines of Matrix Multiplication and Its Applications." *Numerical Algorithms*, 59(1).

## License

Apache 2.0
