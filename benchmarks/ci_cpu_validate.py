"""CPU-only CI smoke checks for ozaki-jax."""

import numpy as np

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

    C_host = matmul(A, B, pipeline="host")
    C_ondev_host = matmul(A, B, pipeline="ondevice", accumulation="host")
    C_ondev_dev = matmul(A, B, pipeline="ondevice", accumulation="ondevice")
    C_ondev_fused = matmul(A, B, pipeline="ondevice", accumulation="fused")
    C_np_host = matmul_numpy(A, B, pipeline="host")
    C_np_ondev = matmul_numpy(A, B, pipeline="ondevice")

    assert_leq(rel_err(C_host, C_ref), 1e-14, "host accuracy")
    assert_leq(rel_err(C_ondev_host, C_ref), 1e-14, "ondevice+host accuracy")
    assert_leq(rel_err(C_np_host, C_ref), 1e-14, "numpy host accuracy")
    assert_leq(rel_err(C_np_ondev, C_ref), 1e-14, "numpy ondevice accuracy")
    assert_leq(rel_err(C_ondev_dev, C_ref), 1e-9, "ondevice+ondevice accuracy")
    assert_leq(rel_err(C_ondev_fused, C_ref), 1e-9, "ondevice+fused accuracy")
    assert_leq(rel_err(C_ondev_fused, C_ondev_dev), 1e-9, "fused vs ondevice parity")

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
