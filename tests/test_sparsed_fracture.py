"""Regression tests for the scalar 1D fracture operator."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from pinn.operators import SparseD


def _grid(n):
    return np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]


def _quadratic_error(n, delta):
    x = _grid(n)
    operator = SparseD(x, delta, kernel="cubic")
    values = np.asarray(operator @ jnp.asarray(x[:, 0] ** 2))
    interior = (x[:, 0] >= delta) & (x[:, 0] <= 1.0 - delta)
    return float(np.mean(np.abs(values[interior] + 2.0)))


class SparseDFractureTest(unittest.TestCase):
    def test_float64_operator_matches_scipy_matvec(self):
        jax.config.update("jax_enable_x64", True)
        x = np.linspace(0.0, 1.0, 81, dtype=np.float64)[:, None]
        operator = SparseD(x, 0.1, kernel="cubic")
        values = jnp.sin(2.0 * jnp.pi * jnp.asarray(x[:, 0]))
        self.assertEqual(operator.diagonal.dtype, jnp.float64)
        np.testing.assert_allclose(
            np.asarray(operator.to_scipy() @ np.asarray(values)),
            np.asarray(operator @ values),
            rtol=1e-12,
            atol=1e-11,
        )

    def test_constants_are_annihilated(self):
        x = _grid(81)
        operator = SparseD(x, 0.1, kernel="cubic")
        result = np.asarray(operator @ jnp.ones(len(x)))
        np.testing.assert_allclose(result, 0.0, rtol=0.0, atol=2e-6)

    def test_subgraph_scalar_matvec_matches_full_on_core_rows(self):
        x = _grid(81)
        operator = SparseD(x, 0.15, kernel="cubic")
        values = jnp.sin(2.0 * jnp.pi * jnp.asarray(x[:, 0]))
        full = np.asarray(operator @ values)

        all_idx, sub_edges, core_mask, _ = operator.sub_sample(seed=17, K=20)
        sub_row, sub_col, sub_values, sub_diag = sub_edges
        values_sub = values[all_idx]
        sub = np.asarray(
            jnp.asarray(sub_diag) * values_sub
            + jax_segment_sum(sub_values * values_sub[sub_col], sub_row,
                              len(all_idx))
        )
        np.testing.assert_allclose(
            sub[np.asarray(core_mask)],
            full[np.asarray(all_idx)[np.asarray(core_mask)]],
            # Full and subgraph paths use different FP32 accumulation orders.
            rtol=2e-6,
            atol=2e-4,
        )

    def test_quadratic_has_negative_second_derivative(self):
        x = _grid(161)
        operator = SparseD(x, 0.1, kernel="cubic")
        result = np.asarray(operator @ jnp.asarray(x[:, 0] ** 2))
        interior = (x[:, 0] >= 0.1) & (x[:, 0] <= 0.9)
        np.testing.assert_allclose(result[interior], -2.0, rtol=0.0,
                                   atol=2e-3)

    def test_fixed_horizon_refines_with_grid_spacing(self):
        errors = [_quadratic_error(n, 0.1) for n in (41, 81, 161)]
        print(f"SparseD fixed-delta quadratic errors: {errors}")
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_coupled_horizon_grid_refinement_is_finite(self):
        cases = ((41, 0.2), (81, 0.1), (161, 0.05))
        errors = [_quadratic_error(n, delta) for n, delta in cases]
        print(f"SparseD coupled refinement (n, delta, error): "
              f"{[(n, delta, error) for (n, delta), error in zip(cases, errors)]}")
        self.assertTrue(np.all(np.isfinite(errors)))
        self.assertLessEqual(errors[-1], 2.0 * errors[0] + 1e-6)

    def test_quadratic_form_is_nonnegative(self):
        x = _grid(101)
        operator = SparseD(x, 0.12, kernel="cubic")
        rng = np.random.default_rng(23)
        values = jnp.asarray(rng.normal(size=len(x)), dtype=jnp.float32)
        h = 1.0 / (len(x) - 1)
        quadratic_form = h * jnp.dot(values, operator @ values)
        self.assertGreaterEqual(float(quadratic_form), -2e-5)

    def test_scipy_matrix_matches_jax_matvec(self):
        x = _grid(81)
        operator = SparseD(x, 0.1, kernel="cubic")
        values = jnp.sin(2.0 * jnp.pi * jnp.asarray(x[:, 0]))
        np.testing.assert_allclose(
            np.asarray(operator.to_scipy() @ np.asarray(values)),
            np.asarray(operator @ values),
            rtol=2e-6,
            # SciPy and JAX use different accumulation orders for FP32 weights.
            atol=3e-4,
        )

    def test_boundary_excluded_error_is_no_larger_than_full_domain_error(self):
        x = _grid(161)
        operator = SparseD(x, 0.1, kernel="cubic")
        result = np.asarray(operator @ jnp.asarray(x[:, 0] ** 2))
        error = np.abs(result + 2.0)
        interior = (x[:, 0] >= 0.1) & (x[:, 0] <= 0.9)
        print(f"SparseD quadratic full/interior MAE: {error.mean():.6e}/"
              f"{error[interior].mean():.6e}")
        self.assertLessEqual(error[interior].mean(), error.mean() + 1e-7)


def jax_segment_sum(values, indices, num_segments):
    """Keep the scalar subgraph calculation explicit in the regression test."""
    import jax

    return jax.ops.segment_sum(values, indices, num_segments=num_segments)


if __name__ == "__main__":
    unittest.main()
