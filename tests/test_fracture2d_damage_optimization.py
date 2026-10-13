"""Independently verify the fixed-H functional and residual-based stopping."""
import unittest
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.damage_optimization import damage_energy, optimize_lbfgs, validate_energy_operator

jax.config.update('jax_enable_x64', True)


class DamageOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.V = np.array([1., 2., .5])
        conductance = np.array([[0., 1., .5], [1., 0., .2], [.5, .2, 0.]])
        K = np.diag(conductance.sum(axis=1))-conductance
        self.D = K/self.V[:, None]
        self.H = jnp.array([.1, .8, .3])
        self.A = np.diag(2+2*np.asarray(self.H))+3*self.D
        self.exact = np.linalg.solve(self.A, 2*np.asarray(self.H))

    def test_energy_gradient_is_mass_times_original_residual(self):
        d, v = jnp.array([.2, .4, .3]), np.array([.1, -.2, .4])
        matvec = lambda x: jnp.asarray(self.D)@x
        energy = lambda x: damage_energy(x, self.H, jnp.asarray(self.V), matvec, 2., 3.)
        expected = self.V*(self.A@np.asarray(d)-2*np.asarray(self.H))
        np.testing.assert_allclose(jax.grad(energy)(d), expected, rtol=1e-13, atol=1e-14)
        delta = 1e-5
        finite_difference = float((energy(d+delta*v)-energy(d-delta*v))/(2*delta))
        self.assertAlmostEqual(finite_difference, expected@v, places=10)
        np.testing.assert_allclose(jax.grad(energy)(jnp.asarray(self.exact)), 0., atol=1e-14)

    def test_nonuniform_volume_symmetry_is_required(self):
        matrix = csr_matrix(self.D)
        operator = SimpleNamespace(to_scipy=lambda: matrix)
        validate_energy_operator(operator, self.V, roundoff_factor=1024.)
        matrix[0, 1] += .1
        with self.assertRaisesRegex(ValueError, 'volume symmetry'):
            validate_energy_operator(operator, self.V, roundoff_factor=1024.)

    def test_lbfgs_stops_on_field_residual_and_rejects_small_budget(self):
        net = eqx.nn.Linear(3, 1, key=jax.random.PRNGKey(0), dtype=jnp.float64)
        matvec = lambda x: jnp.asarray(self.D)@x
        def objective(n, H, load):
            return damage_energy(n.weight[0], H, jnp.asarray(self.V), matvec, 2., 3.)
        def metrics(n, H, load):
            d = n.weight[0]
            r = jnp.asarray(self.A)@d-2*H
            return dict(AT2_relative=jnp.max(abs(r)), damage_min=jnp.min(d), damage_max=jnp.max(d))
        controls = dict(maxiter=100, check_every=1, memory=5, max_linesearch=30,
            gtol=1e-14, ftol=0., residual_limit=1e-9, range_tolerance=1e-10)
        result, info = optimize_lbfgs(net, objective, metrics, self.H, jnp.array(1.),
            controls, progress=lambda row: None)
        self.assertEqual(info['status'], 'residual_reached')
        np.testing.assert_allclose(result.weight[0], self.exact, atol=1e-9)
        controls['maxiter'] = 1
        _, info = optimize_lbfgs(net, objective, metrics, self.H, jnp.array(1.),
            controls, progress=lambda row: None)
        self.assertEqual(info['status'], 'iteration_budget_exhausted')
        self.assertFalse(info['rows'][-1]['ready'])


if __name__ == '__main__':
    unittest.main()
