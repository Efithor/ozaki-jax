"""CPU-only CI smoke checks for ozaki-jax gram()."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax import gram


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
    rng = np.random.RandomState(42)

    # 1. Default mode (f64): near FP64 accuracy.
    A = rng.randn(128, 128)
    G = gram(A)
    G_ref = A.T @ A
    assert isinstance(G, np.ndarray), "numpy input should return numpy output"
    assert G.shape == (128, 128)
    assert_leq(rel_err(G, G_ref), 1e-14, "square gram accuracy (f64)")

    # 2. Tall matrix (N > K).
    A_tall = rng.randn(256, 64)
    G_tall = gram(A_tall)
    G_tall_ref = A_tall.T @ A_tall
    assert G_tall.shape == (64, 64)
    assert_leq(rel_err(G_tall, G_tall_ref), 1e-14, "tall gram accuracy (f64)")

    # 3. Wide matrix (N < K).
    A_wide = rng.randn(32, 128)
    G_wide = gram(A_wide)
    G_wide_ref = A_wide.T @ A_wide
    assert G_wide.shape == (128, 128)
    assert_leq(rel_err(G_wide, G_wide_ref), 1e-14, "wide gram accuracy (f64)")

    # 4. Result is symmetric.
    assert_leq(float(np.max(np.abs(G - G.T))), 1e-30, "symmetry")
    assert_leq(float(np.max(np.abs(G_tall - G_tall.T))), 1e-30, "tall symmetry")

    # 5. Result is positive semi-definite (eigenvalues >= 0).
    eigvals = np.linalg.eigvalsh(G_tall)
    assert np.all(eigvals >= -1e-10), (
        f"gram matrix not PSD: min eigenvalue = {eigvals.min():.3e}"
    )

    # 6. Ozaki mode (backward compat).
    G_ozaki = gram(A, mode="ozaki", precision="high")
    assert_leq(rel_err(G_ozaki, G_ref), 1e-9, "ozaki high accuracy")

    G_ozaki_med = gram(A, mode="ozaki", precision="medium")
    assert_leq(rel_err(G_ozaki_med, G_ref), 1e-5, "ozaki medium accuracy")

    # 7. Accumulation modes (ozaki path).
    G_fused = gram(A, mode="ozaki", accumulation="fused")
    assert_leq(rel_err(G_fused, G_ref), 1e-9, "ozaki fused accuracy")

    # 8. JAX array inputs return JAX arrays.
    A_j = jnp.asarray(A, dtype=jnp.float64)
    G_j = gram(A_j)
    assert isinstance(G_j, jax.Array), "JAX input should return JAX array"
    assert_leq(rel_err(np.asarray(G_j), G_ref), 1e-14, "JAX input accuracy")

    # 9. Error cases.
    try:
        gram(rng.randn(10))
    except ValueError:
        pass
    else:
        raise AssertionError("1D input did not raise ValueError")

    try:
        gram(A, mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode did not raise ValueError")

    print("gram() CPU CI validation: PASS")


if __name__ == "__main__":
    main()
