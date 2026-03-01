"""ozaki-jax: FP64-accurate matrix multiplication from BF16 hardware via Ozaki Extract.

>>> import numpy as np
>>> from ozaki_jax import matmul
>>> A = np.random.randn(128, 128)
>>> B = np.random.randn(128, 128)
>>> C = matmul(A, B)  # FP64-accurate, uses 36 BF16 GEMMs
"""

__version__ = "0.1.1"

from .matmul import matmul, matmul_numpy
from .extract import extract_split_rows, extract_split_cols

__all__ = ["matmul", "matmul_numpy", "extract_split_rows", "extract_split_cols"]
