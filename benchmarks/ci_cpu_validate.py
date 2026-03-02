"""CPU-only CI smoke checks for ozaki-jax."""

import jax
import jax.numpy as jnp
import numpy as np

# Fused on-device path requires x64; set before importing ozaki_jax.
jax.config.update("jax_enable_x64", True)

from ozaki_jax import matmul, matmul_numpy


def rel_err(actual, reference):
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        denom = np.linalg.norm(actual)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(actual - reference) / denom)


def assert_leq(value, threshold, label):
    if value > threshold:
        raise AssertionError(f"{label}: {value:.3e} > {threshold:.3e}")


def main():
    rng = np.random.RandomState(0)
    n = 64
    A = rng.randn(n, n).astype(np.float64)
    B = rng.randn(n, n).astype(np.float64)
    C_ref = A @ B

    # Host pipeline (FP64 extraction) — always high accuracy.
    C_host = matmul(A, B, pipeline="host")
    C_np_host = matmul_numpy(A, B, pipeline="host")
    assert_leq(rel_err(C_host, C_ref), 1e-14, "host accuracy")
    assert_leq(rel_err(C_np_host, C_ref), 1e-14, "numpy host accuracy")

    # On-device default (precision="high", n_hi=4, n_lo=1): ~9.5 digits.
    C_ondev_host = matmul(A, B, pipeline="ondevice", accumulation="host")
    C_ondev_dev = matmul(A, B, pipeline="ondevice", accumulation="ondevice")
    C_ondev_fused = matmul(A, B, pipeline="ondevice", accumulation="fused")
    C_np_ondev = matmul_numpy(A, B, pipeline="ondevice")
    assert_leq(rel_err(C_ondev_host, C_ref), 1e-9, "ondevice+host accuracy")
    assert_leq(rel_err(C_np_ondev, C_ref), 1e-9, "numpy ondevice accuracy")
    assert_leq(rel_err(C_ondev_dev, C_ref), 1e-9, "ondevice+ondevice accuracy")
    assert_leq(rel_err(C_ondev_fused, C_ref), 1e-9, "ondevice+fused accuracy")
    assert_leq(rel_err(C_ondev_fused, C_ondev_dev), 1e-9, "fused vs ondevice parity")

    # On-device precision="max" (n_hi=5, n_lo=4): ~10 digits, host accum → ~14 digits.
    C_max_host = matmul(A, B, pipeline="ondevice", accumulation="host", precision="max")
    C_max_fused = matmul(A, B, pipeline="ondevice", precision="max")
    C_np_max = matmul_numpy(A, B, pipeline="ondevice", precision="max")
    assert_leq(rel_err(C_max_host, C_ref), 1e-14, "max+host accuracy")
    assert_leq(rel_err(C_np_max, C_ref), 1e-14, "numpy max accuracy")
    assert_leq(rel_err(np.asarray(C_max_fused), C_ref), 1e-9, "max+fused accuracy")

    # Precision presets: medium (~7 digits).
    C_medium = matmul(A, B, pipeline="ondevice", precision="medium")
    assert_leq(rel_err(np.asarray(C_medium), C_ref), 1e-6, "precision=medium accuracy")

    # Custom (n_hi, n_lo) tuple.
    C_custom = matmul(A, B, pipeline="ondevice", precision=(4, 2))
    assert_leq(rel_err(np.asarray(C_custom), C_ref), 1e-9, "custom precision accuracy")

    # Invalid precision must raise.
    try:
        _ = matmul(A, B, pipeline="ondevice", precision="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid precision did not raise ValueError")

    # JAX-input on-device path should keep JAX outputs for all accumulation modes.
    A_j = jnp.asarray(A, dtype=jnp.float64)
    B_j = jnp.asarray(B, dtype=jnp.float64)
    Cj_fused = matmul(A_j, B_j, pipeline="ondevice", accumulation="fused")
    Cj_ondev = matmul(A_j, B_j, pipeline="ondevice", accumulation="ondevice")
    Cj_host = matmul(A_j, B_j, pipeline="ondevice", accumulation="host")
    if not isinstance(Cj_fused, jax.Array):
        raise AssertionError("JAX-input fused path did not return JAX array")
    if not isinstance(Cj_ondev, jax.Array):
        raise AssertionError("JAX-input ondevice path did not return JAX array")
    if not isinstance(Cj_host, jax.Array):
        raise AssertionError("JAX-input host-acc path did not return JAX array")
    assert_leq(rel_err(np.asarray(Cj_fused), C_ref), 1e-9, "jax fused accuracy")
    assert_leq(rel_err(np.asarray(Cj_ondev), C_ref), 1e-9, "jax ondevice accuracy")
    assert_leq(rel_err(np.asarray(Cj_host), C_ref), 1e-9, "jax host-acc accuracy")

    # Mixed array families must be rejected.
    try:
        _ = matmul(A_j, B, pipeline="ondevice", accumulation="fused")
    except ValueError:
        pass
    else:
        raise AssertionError("mixed input types did not raise ValueError")

    # Ensure invalid accumulation values are rejected explicitly.
    try:
        _ = matmul(A, B, pipeline="ondevice", accumulation="invalid_mode")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid accumulation mode did not raise ValueError")

    # Ensure unsafe host configurations can still succeed with fallback mode.
    A_bad = rng.randn(2, 10_000).astype(np.float64)
    B_bad = rng.randn(10_000, 2).astype(np.float64)
    C_fallback = matmul(A_bad, B_bad, pipeline="host", safe_mode="fallback")
    assert_leq(rel_err(C_fallback, A_bad @ B_bad), 1e-14, "safe_mode fallback")

    print("CPU CI validation: PASS")


if __name__ == "__main__":
    main()
