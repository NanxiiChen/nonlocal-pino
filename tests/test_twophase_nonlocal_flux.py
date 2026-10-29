"""Independent symbols, conservation, adjoints and mixed residual derivatives."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.twophase2d.configs.train_vortex_conserved import get_config as old_config
from examples.twophase2d.configs.train_vortex_nonlocal_flux import get_config
from examples.twophase2d.fields import predict_fields
from examples.twophase2d.losses import make_loss_fn, make_residual_fn
from examples.twophase2d.operators import PeriodicGradient, PeriodicLaplacian
from examples.twophase2d.problem import identity, initial_field, validate_config, velocity_field
from examples.twophase2d.solver import first_derivative_symbols, grid, symbols, solve_reference


class SmoothFields(eqx.Module):
    a: jax.Array

    def __call__(self, x, t):
        value = self.a*jnp.exp(-t)
        phi = value*(jnp.sin(2*jnp.pi*x[0])+.2*jnp.cos(2*jnp.pi*x[1]))+.1*t*t
        q = .3*value*jnp.cos(2*jnp.pi*x[0])+.2*t
        return jnp.stack((phi, q))


class NonlocalFluxTests(unittest.TestCase):
    def test_symbols_adjoint_mass_and_known_checkerboard_nullspace(self):
        c = get_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        validate_config(c)
        n = c.sample.Nx
        rng = np.random.default_rng(4)
        values = rng.normal(size=n*n)
        vectors = rng.normal(size=(n*n, 2))
        with jax.enable_x64(True):
            op = PeriodicGradient(n, 1., c.nl.delta, dtype=np.float64)
            gx, gy = first_derivative_symbols(n, c, 'nonlocal')
            hat = np.fft.fftn(values.reshape(n, n))
            expected_gradient = np.stack((np.fft.ifftn(gx*hat).real.ravel(),
                                          np.fft.ifftn(gy*hat).real.ravel()), axis=-1)
            vector_hat = np.fft.fftn(vectors.reshape(n, n, 2), axes=(0, 1))
            expected_divergence = np.fft.ifftn(gx*vector_hat[..., 0]+gy*vector_hat[..., 1]).real.ravel()
            np.testing.assert_allclose(op @ jnp.asarray(values), expected_gradient, atol=2e-14)
            np.testing.assert_allclose(op.divergence(jnp.asarray(vectors)), expected_divergence, atol=2e-14)
            self.assertLess(abs(float(jnp.sum(op.divergence(jnp.asarray(vectors))))), 1e-13)
            np.testing.assert_allclose(op @ jnp.ones(n*n), 0., atol=1e-15)
            left = jnp.dot(jnp.asarray(values), op.divergence(jnp.asarray(vectors)))
            right = -jnp.sum((op @ jnp.asarray(values))*jnp.asarray(vectors))
            np.testing.assert_allclose(left, right, atol=1e-13)
            transpose_action = jax.grad(lambda f:jnp.sum((op @ f)*jnp.asarray(vectors)))(jnp.asarray(values))
            np.testing.assert_allclose(transpose_action, -expected_divergence, atol=2e-14)
            checkerboard = (-1.)**np.indices((n, n)).sum(axis=0).ravel()
            np.testing.assert_allclose(op @ jnp.asarray(checkerboard), 0., atol=2e-14)
            lap = PeriodicLaplacian(n, 1., c.nl.delta, dtype=np.float64)
            self.assertGreater(np.linalg.norm(lap @ jnp.asarray(checkerboard)), 1.)

    def test_mixed_residual_and_nonzero_parameter_gradient(self):
        c = get_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        n = c.sample.Nx
        with jax.enable_x64(True):
            points = jnp.asarray(grid(n, 1.)); t = jnp.asarray(.37)
            net = SmoothFields(jnp.asarray(.4))
            D = PeriodicLaplacian(n, 1., c.nl.delta, dtype=np.float64)
            fields = np.asarray(predict_fields(net, points, t, c)).reshape(n, n, 2)
            initial_mean = float(jnp.mean(initial_field(points, c, xp=jnp)))
            velocity = np.asarray(velocity_field(points, t, c, xp=jnp)).reshape(n, n, 2)
            gx, gy = first_derivative_symbols(n, c, 'nonlocal')
            lam, _ = symbols(n, c, 'nonlocal')
            flux_hat = np.fft.fftn(fields[..., 0, None]*velocity, axes=(0, 1))
            advection = np.fft.ifftn(gx*flux_hat[..., 0]+gy*flux_hat[..., 1]).real
            negative_laplacians = np.fft.ifftn(lam[..., None]*np.fft.fftn(fields, axes=(0, 1)), axes=(0, 1)).real
            phase = -(fields[..., 0]-initial_mean)+advection+c.physical.mobility*negative_laplacians[..., 1]
            chemical = fields[..., 1]-fields[..., 0]**3+fields[..., 0]-c.physical.epsilon**2*negative_laplacians[..., 0]
            expected = np.stack((phase, chemical), axis=-1).reshape(n*n, 2)
            actual, _ = make_residual_fn(points, D, c)(net, t)
            np.testing.assert_allclose(actual, expected, atol=5e-15)
            loss = make_loss_fn(points, D, c)
            f = lambda a:loss(SmoothFields(a), jnp.asarray([.2, .7]), jnp.asarray(initial_mean))[0]
            finite = (f(net.a+1e-6)-f(net.a-1e-6))/2e-6
            self.assertGreater(abs(float(jax.grad(f)(net.a))), 1e-5)
            np.testing.assert_allclose(jax.grad(f)(net.a), finite, rtol=3e-7)

    def test_local_target_and_old_identity_remain_unchanged(self):
        old = old_config(); new = get_config()
        unsupported = get_config(); unsupported.nl.method = 'local_ad'
        with self.assertRaisesRegex(ValueError, 'requires nonlocal diffusion'):
            validate_config(unsupported)
        old_identity = identity(old)
        self.assertNotIn('advection', old_identity)
        new_identity = identity(new)
        self.assertEqual(new_identity.pop('advection'), 'nonlocal_flux')
        self.assertEqual(new_identity, old_identity)
        for a, b in zip(first_derivative_symbols(8, old, 'local_ad'),
                        first_derivative_symbols(8, new, 'local_ad'), strict=True):
            np.testing.assert_array_equal(a, b)
        # Short independent reference evolution detects unintended changes to
        # the common local target, rather than just inspecting config wiring.
        for c in (old, new):
            c.physical.terminal_time = c.physical.velocity.period = .002
            c.reference.snapshots = 3
        a = solve_reference(old, 8, 'local_ad', .0001)
        b = solve_reference(new, 8, 'local_ad', .0001)
        for name in ('phi', 'q', 'energy'):
            np.testing.assert_array_equal(a[name], b[name])


if __name__ == '__main__':
    unittest.main()
