"""Independent polynomial flow, outlet-pressure and direct baseline algebra checks."""
import unittest
import numpy as np
from examples.channel_nsch2d.configs.train_base import get_config
from docs.tasks.channel_nonlocal_reference import Collocation, algebra_check


class NonlocalReferenceTests(unittest.TestCase):
    def test_original_laplacian_and_nonlinear_jacobian(self):
        result = algebra_check(get_config())
        self.assertLess(result['jacobian_relative_fd_gap'], 2e-7)
        self.assertLess(result['equilibrium_max_gap'], 1e-8)

    def test_analytic_poiseuille_stress_and_pressure_anchor(self):
        c = get_config(); c.geometry.upper = 0.; c.geometry.lower = 0.
        scheme = Collocation(c, (24, 10), 1, 'interior')
        x, y = scheme.points.T; q = c.physical.mean_velocity
        u = 6*q*y*(1-y); nu = c.physical.viscosity
        pressure = 12*nu*q*(c.geometry.length-scheme.pressure_points[:, 0])
        state = np.concatenate((np.ones(scheme.n), np.zeros(scheme.n),
            u, np.zeros(scheme.n), pressure))
        actual = scheme.residual(state, state, 1., 1., .025, False)
        # Developed Poiseuille has NONZERO tangential outlet traction. Verify
        # it rather than silently changing the physical zero-traction BC.
        expected = np.zeros(scheme.size)
        expected[3*scheme.n:4*scheme.n][scheme.outlet] = (
            nu*6*q*(1-2*y[scheme.outlet])/c.loss.momentum_scale)
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
        shifted = state.copy(); shifted[4*scheme.n:] += 1
        change = scheme.residual(shifted, state, 1., 1., .025, False)-actual
        expected = np.zeros(scheme.size)
        expected[2*scheme.n:3*scheme.n][scheme.outlet] = -1/c.loss.momentum_scale
        np.testing.assert_allclose(change, expected, atol=2e-10, rtol=0)

    def test_direct_and_krylov_solve_same_nonlinear_time_step(self):
        scheme = Collocation(get_config(), (24, 10), 1, 'interior')
        old = scheme.initial()
        direct, _ = scheme.newton(old.copy(), old, .025, 1., .025, 1e-9, 16,
            dict(kind='direct_lu', refresh_steps=20, group=None, preconditioner=None))
        krylov, info = scheme.newton(old.copy(), old, .025, 1., .025, 1e-9, 16,
            dict(kind='gmres_lu', refresh_steps=20, group=None, preconditioner=None))
        np.testing.assert_allclose(krylov, direct, atol=2e-8, rtol=2e-8)
        self.assertLess(info['residual_max'], 1e-9)
        self.assertLess(scheme.diagnostic(krylov, .025)['divergence_fine_interior_rms'], 1e-9)


if __name__ == '__main__':
    unittest.main()
