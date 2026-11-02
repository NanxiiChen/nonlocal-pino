"""Independent checks for the frozen-velocity CH diagnostic, not production training."""
import unittest
import numpy as np
from examples.channel_nsch2d.configs.train_operator_width128_gmls_first_online import get_config
from examples.channel_nsch2d.data import geometry_data
from docs.tasks.channel_frozen_ch import FrozenCH


class FrozenCHTests(unittest.TestCase):
    def setUp(self):
        self.c = get_config()
        self.c.sample.Nx, self.c.sample.Ny = 24, 10
        data, _ = geometry_data(np.array([.12, .08]), self.c)
        self.scheme = FrozenCH(self.c, data, (24, 10))

    def test_equilibrium_and_polynomial_divergence_free_transport(self):
        s = self.scheme
        state = np.r_[np.ones(s.n), np.zeros(s.n)]
        velocity = np.column_stack((s.points[:, 1]*(1 - s.points[:, 1]), np.zeros(s.n)))
        residual = s.residual(state, state[:s.n], velocity, np.zeros((s.n, 2)), 1., .025, jacobian=False)
        np.testing.assert_allclose(residual, 0., atol=1e-9)

    def test_jacobian_with_nonzero_boundary_data_against_finite_difference(self):
        s = self.scheme
        rng = np.random.default_rng(6911)
        z = s.initial() + .01*rng.normal(size=2*s.n)
        history = s.initial()[:s.n]
        velocity = .02*rng.normal(size=(s.n, 2))
        natural = .03*rng.normal(size=(s.n, 2))
        direction = rng.normal(size=2*s.n)
        direction /= np.linalg.norm(direction)
        _, jac = s.residual(z, history, velocity, natural, 1.5, .025, jacobian=True)
        h = 2e-6
        actual = (s.residual(z + h*direction, history, velocity, natural, 1.5, .025, jacobian=False)
                  - s.residual(z - h*direction, history, velocity, natural, 1.5, .025, jacobian=False))/(2*h)
        expected = jac@direction
        self.assertLess(np.linalg.norm(actual - expected)/np.linalg.norm(expected), 2e-7)


if __name__ == '__main__':
    unittest.main()
