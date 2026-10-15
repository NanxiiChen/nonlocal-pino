"""Independent small-mesh weak-form and solve checks for the local control."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from examples.fracture2d._legacy.at2_cg import damage_solver
from examples.fracture2d._legacy.local_damage_comparison import LocalDamageOperator
from tests.test_fracture2d_local_mechanics import model


class LocalDamageComparisonTest(unittest.TestCase):
    def test_independent_gradient_weak_form_and_fixed_history_solve(self):
        jax.config.update('jax_enable_x64', True)
        mechanics = model()
        operator = LocalDamageOperator(mechanics)
        # Unit-square two-triangle P1 weak form, derived from basis gradients.
        expected = np.array([[1., -.5, -.5, 0.], [-.5, 1., 0., -.5],
                             [-.5, 0., 1., -.5], [0., -.5, -.5, 1.]])
        np.testing.assert_allclose(operator.stiffness.toarray(), expected, atol=1e-15)
        d = np.array([.1, .2, .4, .7])
        np.testing.assert_allclose(operator.gpu_matvec()(jnp.asarray(d)),
                                   expected @ d / mechanics.volumes, atol=1e-14)
        H = np.array([.1, 1., 3., 5.])
        gc0, gc1 = 112.5, .0648
        mass = mechanics.volumes
        solution = spsolve(diags(mass * (gc0 + 2 * H)) + gc1 * operator.stiffness, 2 * mass * H)
        actual = damage_solver(operator, mass, gc0, gc1, tol=1e-12, atol=1e-14, maxiter=100)(
            jnp.asarray(H), jnp.zeros(4, jnp.float64))
        np.testing.assert_allclose(actual, solution, rtol=1e-12, atol=1e-14)
        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))


if __name__ == '__main__':
    unittest.main()
