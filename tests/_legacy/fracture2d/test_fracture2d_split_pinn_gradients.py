"""Analytic operator pullbacks must match AD and independently checked derivatives."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.split_horizon_model import build_model
from examples.fracture2d._legacy.split_pinn_gradients import (
    make_analytic_energy, make_analytic_matvec, make_padded_energy, make_padded_matvec,
)
from tests._legacy.fracture2d.test_fracture2d_split_horizon import square_cloud


class SplitAnalyticGradientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        cls.model = build_model(square_cloud(), delta_u=0.4, delta_d=0.6,
            grip_extent=0.6, h_grip=0.2, visibility_tolerance=1e-12,
            E=210000., nu=0.3, k_res=1e-8, rank_rtol=1e-12, Gc=2.7, l0=0.024)
        cls.backend = FP64A2Backend(cls.model.static, cls.model.constraints, jax.devices()[0])

    def test_energy_pullback_matches_ad_force_and_directional_difference(self):
        backend = self.backend
        energy = make_analytic_energy(backend)
        factor = backend._edge_factor(jnp.linspace(0., 0.8, self.model.n_material))
        rng = np.random.default_rng(731)
        q = jnp.asarray(rng.normal(size=self.model.n_dofs) * 1e-4)
        old_value, old_gradient = jax.jit(jax.value_and_grad(backend._energy_from_factor, argnums=(0, 1)))(factor, q)
        value, gradient = jax.jit(jax.value_and_grad(energy, argnums=(0, 1)))(factor, q)
        np.testing.assert_allclose(value, old_value, rtol=1e-13, atol=1e-15)
        for actual, expected in zip(gradient, old_gradient):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
        padded, info = make_padded_energy(backend)
        padded_value, padded_gradient = jax.jit(jax.value_and_grad(padded, argnums=(0, 1)))(factor, q)
        np.testing.assert_allclose(padded_value, old_value, rtol=1e-12, atol=1e-15)
        for actual, expected in zip(padded_gradient, old_gradient):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
        self.assertEqual(info["support_occurrences"], len(backend.occurrence_particle))
        np.testing.assert_allclose(gradient[1], backend._residual_from_factor(factor, q), rtol=1e-12, atol=1e-12)
        direction = jnp.asarray(rng.normal(size=len(q)))
        step = 1e-7
        difference = (backend._energy_from_factor(factor, q + step * direction)
                      - backend._energy_from_factor(factor, q - step * direction)) / (2 * step)
        self.assertAlmostEqual(float(difference / jnp.dot(gradient[1], direction)), 1., places=8)

    def test_scalar_pullback_matches_nonsymmetric_csr_transpose(self):
        operator = self.model.material_operator
        matvec = make_analytic_matvec(operator)
        padded, info = make_padded_matvec(operator)
        matrix = operator.to_scipy().toarray()
        self.assertGreater(np.max(abs(matrix - matrix.T)), 1.)
        rng = np.random.default_rng(19)
        for columns in (1, 3):
            shape = (self.model.n_material,) if columns == 1 else (self.model.n_material, columns)
            x, z = (jnp.asarray(rng.normal(size=shape)) for _ in range(2))
            value, pullback = jax.vjp(matvec, x)
            actual = jax.jit(pullback)(z)[0]
            np.testing.assert_allclose(value, operator.deterministic_matvec(x), rtol=0, atol=0)
            np.testing.assert_allclose(actual, matrix.T @ np.asarray(z), rtol=1e-12, atol=1e-12)
            old = jax.vjp(operator.deterministic_matvec, x)[1](z)[0]
            np.testing.assert_allclose(actual, old, rtol=1e-12, atol=1e-12)
            padded_value, padded_pullback = jax.vjp(padded, x)
            np.testing.assert_allclose(padded_value, matrix @ np.asarray(x), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(padded_pullback(z)[0], matrix.T @ np.asarray(z), rtol=1e-12, atol=1e-12)
            self.assertAlmostEqual(float(jnp.sum(z * value) / jnp.sum(actual * x)), 1., places=12)
        self.assertEqual(info["edge_count"], len(self.model.geometry.material_damage_edges))


if __name__ == "__main__":
    unittest.main()
