"""CPU-only CI smoke checks for ozaki-jax residual() and solve()."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ozaki_jax import residual, solve


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
    n = 128

    A = rng.randn(n, n)
    b_vec = rng.randn(n)
    x_ref_vec = np.linalg.solve(A, b_vec)

    # 1. Default mode (residual_mode='f64'): near FP64 accuracy.
    # With FP64 residual, solve converges to cond(A) * u_f64 ~ 1e-13.
    x = solve(A, b_vec)
    assert isinstance(x, np.ndarray), "numpy input should return numpy output"
    assert x.shape == (n,), f"expected shape ({n},), got {x.shape}"
    assert_leq(rel_err(x, x_ref_vec), 1e-12, "vector RHS accuracy (f64)")

    # 2. Matrix RHS correctness.
    B = rng.randn(n, 4)
    X_ref = np.linalg.solve(A, B)
    X = solve(A, B)
    assert X.shape == (n, 4), f"expected shape ({n}, 4), got {X.shape}"
    assert_leq(rel_err(X, X_ref), 1e-12, "matrix RHS accuracy (f64)")

    # 3. Ozaki residual mode (backward compat, lower accuracy).
    x_ozaki = solve(A, b_vec, residual_mode="ozaki", precision="high")
    assert_leq(rel_err(x_ozaki, x_ref_vec), 1e-7, "ozaki residual accuracy")

    x_ozaki_med = solve(A, b_vec, residual_mode="ozaki", precision="medium")
    assert_leq(rel_err(x_ozaki_med, x_ref_vec), 1e-5, "ozaki medium accuracy")

    # 4. Accumulation modes (ozaki path).
    x_bf16 = solve(A, b_vec, residual_mode="ozaki",
                   accumulation="bf16_interleaved")
    x_fused = solve(A, b_vec, residual_mode="ozaki", accumulation="fused")
    assert_leq(rel_err(x_bf16, x_ref_vec), 1e-7, "bf16_interleaved accuracy")
    assert_leq(rel_err(x_fused, x_ref_vec), 1e-7, "fused accuracy")

    # 5. Iteration convergence: FP64 residual should converge fast.
    err_0 = rel_err(solve(A, b_vec, max_iterations=0), x_ref_vec)
    err_1 = rel_err(solve(A, b_vec, max_iterations=1), x_ref_vec)
    err_3 = rel_err(solve(A, b_vec, max_iterations=3), x_ref_vec)
    assert err_1 < err_0 * 1e-4, (
        f"1 iteration didn't improve enough: {err_1:.3e} vs {err_0:.3e}"
    )
    assert err_3 < err_1, (
        f"3 iterations didn't improve over 1: {err_3:.3e} vs {err_1:.3e}"
    )

    # 6. JAX array inputs return JAX arrays.
    A_j = jnp.asarray(A, dtype=jnp.float64)
    b_j = jnp.asarray(b_vec, dtype=jnp.float64)
    x_j = solve(A_j, b_j)
    assert isinstance(x_j, jax.Array), "JAX input should return JAX array"
    assert_leq(rel_err(np.asarray(x_j), x_ref_vec), 1e-12, "JAX input accuracy")

    # 7. Error cases.
    # Non-square A.
    try:
        solve(rng.randn(3, 4), rng.randn(3))
    except ValueError:
        pass
    else:
        raise AssertionError("non-square A did not raise ValueError")

    # Incompatible b dimension.
    try:
        solve(rng.randn(3, 3), rng.randn(4))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible b did not raise ValueError")

    # Invalid residual_mode.
    try:
        solve(A, b_vec, residual_mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid residual_mode did not raise ValueError")

    # --- residual() tests ---
    # Use random x (not a near-solution) so the residual is large and
    # relative error is meaningful.

    # 8. Vector residual correctness.
    x_rand = rng.randn(n)
    r = residual(A, x_rand, b_vec)
    r_ref = b_vec - A @ x_rand
    assert isinstance(r, np.ndarray), "numpy input should return numpy output"
    assert r.shape == (n,), f"expected shape ({n},), got {r.shape}"
    assert_leq(rel_err(r, r_ref), 1e-14, "residual vector accuracy (f64)")

    # 9. Matrix residual correctness.
    X_rand = rng.randn(n, 4)
    R = residual(A, X_rand, B)
    R_ref = B - A @ X_rand
    assert R.shape == (n, 4), f"expected shape ({n}, 4), got {R.shape}"
    assert_leq(rel_err(R, R_ref), 1e-14, "residual matrix accuracy (f64)")

    # 10. Non-square A works for residual (unlike solve).
    A_rect = rng.randn(64, 128)
    x_rect = rng.randn(128)
    b_rect = rng.randn(64)
    r_rect = residual(A_rect, x_rect, b_rect)
    r_rect_ref = b_rect - A_rect @ x_rect
    assert r_rect.shape == (64,)
    assert_leq(rel_err(r_rect, r_rect_ref), 1e-14, "residual non-square (f64)")

    # 11. JAX array inputs return JAX arrays.
    r_j = residual(A_j, jnp.asarray(x_rand, dtype=jnp.float64), b_j)
    assert isinstance(r_j, jax.Array), "JAX input should return JAX array"
    assert_leq(rel_err(np.asarray(r_j), r_ref), 1e-14, "residual JAX accuracy")

    # 12. Ozaki mode for residual (backward compat).
    r_ozaki = residual(A, x_rand, b_vec, mode="ozaki")
    assert_leq(rel_err(r_ozaki, r_ref), 1e-9, "residual ozaki accuracy")

    r_ozaki_fused = residual(A, x_rand, b_vec, mode="ozaki",
                             accumulation="fused")
    assert_leq(rel_err(r_ozaki_fused, r_ref), 1e-9, "residual ozaki fused")

    # 13. Residual error cases.
    try:
        residual(rng.randn(3, 4), rng.randn(5), rng.randn(3))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible x did not raise ValueError")

    try:
        residual(rng.randn(3, 4), rng.randn(4), rng.randn(4))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible b did not raise ValueError")

    try:
        residual(A, x_rand, b_vec, mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode did not raise ValueError")

    print("residual() + solve() CPU CI validation: PASS")


if __name__ == "__main__":
    main()
