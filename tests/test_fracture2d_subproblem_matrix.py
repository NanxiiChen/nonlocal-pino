"""GPU/CPU-compatible AT2 CG must use the volume-SPD equation and explicit checks."""
import unittest
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.subproblem_matrix import damage_solver


class OperatorFixture:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape
    def to_scipy(self):
        return csr_matrix(self.matrix)
    def deterministic_matvec(self, d):
        return jnp.asarray(self.matrix) @ d


class DamageCGTest(unittest.TestCase):
    def setUp(self):
        self.volumes = np.asarray([1., 2., 3.])
        L = np.asarray([[3., -1., -2.], [-1., 2., -1.], [-2., -1., 3.]])
        self.D = L/self.volumes[:, None]
        self.operator = OperatorFixture(self.D)
        self.H = np.asarray([.2, 1., 2.])

    def test_nonsymmetric_strong_operator_matches_independent_dense_solution(self):
        solve = damage_solver(self.operator, self.volumes, 2., .3,
                              tol=1e-12, atol=1e-14, maxiter=100)
        d = np.asarray(solve(jnp.asarray(self.H), jnp.zeros(3, jnp.float64)))
        expected = np.linalg.solve(np.diag(2.+2*self.H)+.3*self.D, 2*self.H)
        np.testing.assert_allclose(d, expected, rtol=1e-10, atol=1e-12)

    def test_truncated_cg_is_not_mistaken_for_convergence(self):
        solve = damage_solver(self.operator, self.volumes, 2., .3,
                              tol=1e-12, atol=1e-14, maxiter=1)
        with self.assertRaisesRegex(RuntimeError, "explicit residual"):
            solve(jnp.asarray(self.H), jnp.zeros(3, jnp.float64))

    def test_negative_history_or_wrong_mass_symmetry_fails_loudly(self):
        solve = damage_solver(self.operator, self.volumes, 2., .3,
                              tol=1e-12, atol=1e-14, maxiter=100)
        with self.assertRaisesRegex(ValueError, "invalid history"):
            solve(jnp.asarray([-1., 1., 2.]), jnp.zeros(3, jnp.float64))
        with self.assertRaisesRegex(ValueError, "invalid history"):
            solve(jnp.asarray(self.H), jnp.asarray([np.nan, 0., 0.]))
        with self.assertRaisesRegex(ValueError, "volume-symmetric"):
            damage_solver(self.operator, np.ones(3), 2., .3,
                          tol=1e-12, atol=1e-14, maxiter=100)


if __name__ == '__main__':
    unittest.main()
