"""CPU/TPU validation for ozaki-jax Tier-1 0.7 APIs: inv(), lstsq(), norm().

Runs on whatever backend JAX selects. On a TPU host this also confirms the
new routines stay on-device.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax import inv, lstsq, norm


def rel_err(actual, reference):
    denom = np.linalg.norm(np.asarray(reference))
    if denom == 0.0:
        denom = np.linalg.norm(np.asarray(actual))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(np.asarray(actual) - np.asarray(reference)) / denom)


def assert_leq(value, threshold, label):
    if value > threshold:
        raise AssertionError(f"{label}: {value:.3e} > {threshold:.3e}")
    print(f"  ok  {label}: {value:.3e} <= {threshold:.3e}")


def well_conditioned(rng, m, n, cond=1e2):
    """Tall (m>=n) matrix with controlled condition number via SVD."""
    U, _ = np.linalg.qr(rng.randn(m, m))
    V, _ = np.linalg.qr(rng.randn(n, n))
    s = np.logspace(0, -np.log10(cond), n)
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)
    return U @ S @ V.T


def test_inv(rng):
    print("inv():")
    n = 128
    A = well_conditioned(rng, n, n, cond=1e2)
    Ainv_ref = np.linalg.inv(A)

    Ainv = inv(A)
    assert isinstance(Ainv, np.ndarray), "numpy in -> numpy out"
    assert Ainv.shape == (n, n)
    assert_leq(rel_err(Ainv, Ainv_ref), 1e-11, "inv accuracy (f64)")

    # A @ A^-1 == I
    assert_leq(rel_err(A @ Ainv, np.eye(n)), 1e-11, "inv residual A@Ainv=I")

    # JAX in -> JAX out, stays on device.
    A_j = jnp.asarray(A, dtype=jnp.float64)
    Ainv_j = inv(A_j)
    assert isinstance(Ainv_j, jax.Array), "JAX in -> JAX out"
    assert_leq(rel_err(Ainv_j, Ainv_ref), 1e-11, "inv accuracy (JAX)")

    # Ozaki residual mode (lower accuracy).
    Ainv_oz = inv(A, residual_mode="ozaki", precision="high")
    assert_leq(rel_err(Ainv_oz, Ainv_ref), 1e-6, "inv accuracy (ozaki)")

    # Non-square rejected.
    try:
        inv(rng.randn(3, 4))
    except ValueError:
        print("  ok  non-square raises ValueError")
    else:
        raise AssertionError("non-square inv did not raise")


def test_lstsq(rng):
    print("lstsq():")
    m, n = 256, 128
    A = well_conditioned(rng, m, n, cond=1e2)
    b = rng.randn(m)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]

    x = lstsq(A, b)
    assert isinstance(x, np.ndarray), "numpy in -> numpy out"
    assert x.shape == (n,)
    assert_leq(rel_err(x, x_ref), 1e-10, "lstsq accuracy (f64, vector)")

    # Matrix RHS.
    B = rng.randn(m, 3)
    X_ref = np.linalg.lstsq(A, B, rcond=None)[0]
    X = lstsq(A, B)
    assert X.shape == (n, 3)
    assert_leq(rel_err(X, X_ref), 1e-10, "lstsq accuracy (f64, matrix)")

    # Square case reduces to a solve.
    As = well_conditioned(rng, n, n, cond=1e2)
    bs = rng.randn(n)
    assert_leq(rel_err(lstsq(As, bs), np.linalg.solve(As, bs)), 1e-10,
               "lstsq square == solve")

    # Refinement improves over the raw FP32 QR solution.
    err_0 = rel_err(lstsq(A, b, max_iterations=0), x_ref)
    err_3 = rel_err(lstsq(A, b, max_iterations=3), x_ref)
    if not (err_3 < err_0):
        raise AssertionError(f"refinement didn't help: {err_3:.3e} vs {err_0:.3e}")
    print(f"  ok  refinement improves: {err_0:.3e} -> {err_3:.3e}")

    # JAX in -> JAX out.
    x_j = lstsq(jnp.asarray(A, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64))
    assert isinstance(x_j, jax.Array), "JAX in -> JAX out"
    assert_leq(rel_err(x_j, x_ref), 1e-10, "lstsq accuracy (JAX)")

    # Ozaki residual mode.
    x_oz = lstsq(A, b, residual_mode="ozaki", precision="high")
    assert_leq(rel_err(x_oz, x_ref), 1e-6, "lstsq accuracy (ozaki)")

    # Underdetermined rejected.
    try:
        lstsq(rng.randn(64, 128), rng.randn(64))
    except ValueError:
        print("  ok  underdetermined raises ValueError")
    else:
        raise AssertionError("underdetermined lstsq did not raise")


def test_norm(rng):
    print("norm():")
    # Vector norms. Standard orders (None/2/1/inf/-inf) use only
    # multiply/add/sqrt and stay near machine FP64 on every backend.
    v = rng.randn(512)
    for ord in [None, 2, 1, np.inf, -np.inf]:
        got = norm(v, ord=ord)
        ref = np.linalg.norm(v, ord=ord)
        assert_leq(abs(float(got) - float(ref)) / abs(float(ref)), 1e-13,
                   f"vector norm ord={ord}")

    # General p-norm (p not in {1,2,inf}) uses x**p / **(1/p). On TPU the
    # transcendental pow path runs at ~fp32 precision even under emulated
    # fp64, so allow a looser bound here (exact on CPU, ~1e-8 on TPU v6e).
    got = norm(v, ord=3)
    ref = np.linalg.norm(v, ord=3)
    assert_leq(abs(float(got) - float(ref)) / abs(float(ref)), 1e-7,
               "vector norm ord=3 (general-p; TPU pow ~fp32)")

    # Matrix norms.
    A = rng.randn(192, 128)
    for ord in [None, "fro", 1, np.inf]:
        got = norm(A, ord=ord)
        ref = np.linalg.norm(A, ord=ord)
        assert_leq(abs(float(got) - float(ref)) / abs(float(ref)), 1e-13,
                   f"matrix norm ord={ord}")

    # Spectral norm (ord=2) via accurate Gram.
    s2_ref = np.linalg.norm(A, ord=2)
    assert_leq(abs(float(norm(A, ord=2)) - s2_ref) / s2_ref, 1e-10,
               "matrix spectral norm ord=2 (f64)")
    assert_leq(abs(float(norm(A, ord=2, mode="ozaki")) - s2_ref) / s2_ref, 1e-6,
               "matrix spectral norm ord=2 (ozaki)")

    # numpy in -> numpy scalar; JAX in -> JAX scalar.
    assert isinstance(norm(v), np.float64), "numpy in -> np.float64"
    assert isinstance(norm(jnp.asarray(v, dtype=jnp.float64)), jax.Array), \
        "JAX in -> JAX scalar"

    # Bad inputs.
    try:
        norm(rng.randn(2, 2, 2))
    except ValueError:
        print("  ok  3D input raises ValueError")
    else:
        raise AssertionError("3D norm did not raise")
    try:
        norm(A, ord="nuc")
    except ValueError:
        print("  ok  unsupported matrix ord raises ValueError")
    else:
        raise AssertionError("unsupported ord did not raise")


def main():
    devices = jax.devices()
    print(f"backend: {jax.default_backend()}  device: {devices[0].device_kind}\n")
    rng = np.random.RandomState(0)
    test_inv(rng)
    test_lstsq(rng)
    test_norm(rng)
    print("\nTier-1 (inv/lstsq/norm) validation: PASS")


if __name__ == "__main__":
    main()
