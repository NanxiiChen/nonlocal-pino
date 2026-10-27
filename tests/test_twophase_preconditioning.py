"""Independent block inversion, coordinate derivatives and physical zero checks."""
import unittest
import tempfile
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.twophase2d.configs.train_vortex_cubic_schur import get_config
from examples.twophase2d.network import PhaseNetwork, inverse_cubic
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.preconditioning import make_schur_transform, make_causal_transform
from examples.twophase2d.problem import validate_config
from examples.twophase2d.solver import grid
from examples.twophase2d.losses import make_loss_fn, make_residual_fn
from examples.twophase2d.refinement import refine
from examples.twophase2d.configs.train_vortex_refined import get_config as refined_config
from examples.twophase2d.configs.train_vortex_conserved import get_config as conserved_config
from examples.twophase2d.fields import predict_fields
from examples.twophase2d.problem import initial_field
from examples.twophase2d.problem import velocity_field
from examples.twophase2d.advection import make_flux_divergence
from tests.test_twophase_residual import SmoothFields, ConstantFields


class CHPreconditioningTests(unittest.TestCase):
    def test_local_hessian_with_mass_projection_and_conservative_flux(self):
        c = conserved_config(); c.nl.method = 'local_ad'
        c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        validate_config(c)

        class Manufactured(eqx.Module):
            a: jax.Array

            def __call__(self, x, t):
                k = 2*jnp.pi
                amplitude = self.a*jnp.exp(-t)
                phi = amplitude*(jnp.sin(k*x[0])+.2*jnp.cos(k*x[1]))+.1*t*t
                q = .3*amplitude*jnp.cos(k*x[0])+.2*t
                return jnp.stack((phi, q))

        with jax.enable_x64(True):
            points = jnp.asarray(grid(8, 1.)); t = jnp.asarray(.37)
            net = Manufactured(jnp.asarray(.4))
            initial_mean = jnp.mean(initial_field(points, c, xp=jnp))
            x, y = np.asarray(points).T; k = 2*np.pi
            amplitude = .4*np.exp(-float(t))
            fluct = amplitude*(np.sin(k*x)+.2*np.cos(k*y))
            phi = float(initial_mean)+fluct
            q = .3*amplitude*np.cos(k*x)+.2*float(t)
            gradient = amplitude*k*np.stack((np.cos(k*x), -.2*np.sin(k*y)), axis=-1)
            velocity = np.asarray(velocity_field(points, t, c, xp=jnp))
            expected = np.stack((
                -fluct+np.sum(velocity*gradient, axis=-1)
                +c.physical.mobility*k*k*(q-.2*float(t)),
                q-phi**3+phi-c.physical.epsilon**2*k*k*fluct), axis=-1)
            residual, physical_phi = make_residual_fn(points, None, c)(net, t)
            np.testing.assert_allclose(physical_phi, phi, atol=5e-16)
            np.testing.assert_allclose(residual, expected, atol=5e-15)
            loss = make_loss_fn(points, None, c)
            f = lambda a: loss(Manufactured(a), jnp.asarray([.2, .7]), initial_mean)[0]
            finite = (f(net.a+1e-6)-f(net.a-1e-6))/2e-6
            np.testing.assert_allclose(jax.grad(f)(net.a), finite, rtol=2e-8)

    def test_initial_time_collocation_detects_unresolved_initial_transient(self):
        c = get_config(); del c.preconditioner
        c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        del c.physical.velocity
        c.physical.velocity = (0., 0.)
        class InitialTransient(eqx.Module):
            a: jax.Array

            def __call__(self, x, t):
                return jnp.stack((self.a*jnp.exp(-50*t)*jnp.sin(2*jnp.pi*x[0]), jnp.zeros_like(t)))

        with jax.enable_x64(True):
            points = jnp.asarray(grid(8, 1.)); D = PeriodicLaplacian(8, 1., .3, dtype=np.float64)
            net = InitialTransient(jnp.asarray(.3)); times = jnp.asarray([.4, .7])
            interior = make_loss_fn(points, D, c)(net, times, jnp.asarray(0.))[0]
            c.sample.include_initial_time = True; validate_config(c)
            loss = make_loss_fn(points, D, c)
            endpoint = loss(net, times, jnp.asarray(0.))[0]
            self.assertGreater(float(endpoint), 1e12*float(interior))
            f = lambda a: loss(InitialTransient(a), times, jnp.asarray(0.))[0]
            finite = (f(net.a+1e-6)-f(net.a-1e-6))/2e-6
            np.testing.assert_allclose(jax.grad(f)(net.a), finite, rtol=1e-8)

    def test_conservative_flux_matches_manufactured_advection_and_preserves_mean(self):
        c = conserved_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        with jax.enable_x64(True):
            points = jnp.asarray(grid(8, 1.))
            function = lambda x: jnp.sin(2*jnp.pi*x[0])*jnp.cos(2*jnp.pi*x[1])+.3
            phi = jax.vmap(function)(points)
            gradient = jax.vmap(jax.grad(function))(points)
            flux = make_flux_divergence(c, dtype=np.float64)
            for t in (.2, .7, 1.4):
                velocity = velocity_field(points, t, c, xp=jnp)
                actual = flux(phi, velocity)
                expected = jnp.sum(velocity*gradient, axis=-1)
                np.testing.assert_allclose(actual, expected, atol=6e-15)
                self.assertLess(abs(float(jnp.mean(actual))), 3e-16)

    def test_mass_projected_fields_time_derivative_and_full_loss_gradient(self):
        c = conserved_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        validate_config(c)
        with jax.enable_x64(True):
            points = jnp.asarray(grid(8, 1.)); net = PhaseNetwork(c, jax.random.PRNGKey(42))
            initial = initial_field(points, c, xp=jnp)
            for t in (0., .37, 1.7):
                fields, dt = jax.jvp(lambda tt:predict_fields(net, points, tt, c),
                                     (jnp.asarray(t),), (jnp.asarray(1.),))
                np.testing.assert_allclose(jnp.mean(fields[:, 0]), jnp.mean(initial), atol=3e-16)
                self.assertLess(abs(float(jnp.mean(dt[:, 0]))), 2e-16)
                if t == 0.:np.testing.assert_allclose(fields[:, 0], initial, atol=3e-16)
            D = PeriodicLaplacian(8, 1., .3, dtype=np.float64)
            loss = make_loss_fn(points, D, c)
            def f(bias):
                candidate = eqx.tree_at(lambda m:m.net.output.bias, net, bias)
                return loss(candidate, jnp.asarray([.2, .7]), jnp.mean(initial))[0]
            bias = net.net.output.bias
            step = jnp.asarray([1e-6, 0.])
            finite = (f(bias+step)-f(bias-step))/2e-6
            np.testing.assert_allclose(jax.grad(f)(bias)[0], finite, rtol=3e-6)
            _, parts = loss(net, jnp.asarray([.2, .7]), jnp.mean(initial))
            self.assertEqual(parts.shape, (3,))
            self.assertLess(float(parts[2]), 1e-29)

    def test_causal_transform_matches_dense_space_time_inverse_and_gradient(self):
        c = get_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        c.preconditioner.kind = 'ch_causal'; c.sample.time_rule = 'open_stratified'
        validate_config(c)
        with jax.enable_x64(True):
            D = PeriodicLaplacian(8, 1., .3, dtype=np.float64)
            L = np.asarray(D @ jnp.eye(64)); B = 2*np.eye(64)+c.physical.epsilon**2*L
            times = jnp.asarray([.1, .4, .9]); dt = np.diff(np.r_[0., times])
            A = np.zeros((384, 384))
            for i, interval in enumerate(dt):
                block = np.block([[np.eye(64)/interval, c.physical.mobility*L], [-B, np.eye(64)]])
                A[i*128:(i+1)*128, i*128:(i+1)*128] = block
                if i: A[i*128:i*128+64, (i-1)*128:(i-1)*128+64] = -np.eye(64)/interval
            r = np.random.default_rng(4).normal(size=(3, 64, 2))
            rhs = r.transpose(0, 2, 1).ravel()
            expected = np.linalg.solve(A, rhs).reshape(3, 2, 64).transpose(0, 2, 1)
            expected[..., 0] /= c.preconditioner.time_scale
            transform = make_causal_transform(D, c, dtype=np.float64)
            np.testing.assert_allclose(transform(jnp.asarray(r), times), expected, atol=3e-14)
            # Sorting input time/residual pairs preserves the causal metric.
            np.testing.assert_allclose(transform(jnp.asarray(r[::-1].copy()), times[::-1]), expected, atol=3e-14)
            points = jnp.asarray(grid(8, 1.)); loss = make_loss_fn(points, D, c)
            f = lambda a: loss(SmoothFields(a), times, jnp.asarray(0.))[0]
            a = jnp.asarray(.7); finite = (f(a+1e-6)-f(a-1e-6))/2e-6
            np.testing.assert_allclose(jax.grad(f)(a), finite, rtol=3e-7)

    def test_refinement_solves_manufactured_time_pde_without_field_targets(self):
        c = refined_config(); c.train.refinement.time_points = 4
        c.train.refinement.iterations = 10
        with jax.enable_x64(True), tempfile.TemporaryDirectory() as directory:
            x = jnp.asarray([.21, .4])
            # phi_t + 2 phi = exp(-t) sin(2*pi*x); amplitude must be one.
            def loss(model, times, initial_mean):
                def residual(t):
                    value, dt = jax.jvp(lambda tt: model(x, tt)[0], (t,), (jnp.ones_like(t),))
                    return dt+2*value-jnp.exp(-t)*jnp.sin(2*jnp.pi*x[0])
                return jnp.mean(jax.vmap(residual)(times)**2), jnp.zeros(3)
            net, result = refine(SmoothFields(jnp.asarray(.3)), loss,
                                 jnp.asarray(0.), c, Path(directory), lambda model, step: None)
            np.testing.assert_allclose(net.a, 1., atol=1e-12)
            self.assertTrue(result['success'])
            self.assertLess(result['final_loss'], 1e-24)

    def test_periodic_modes_stay_fixed_under_parameter_updates(self):
        c = get_config()
        with jax.enable_x64(True):
            net = PhaseNetwork(c, jax.random.PRNGKey(42))
            expected = np.asarray(net.modes).copy()
            loss = lambda model: jnp.sum(model(jnp.asarray([.23, .41]), .7)**2)
            for _ in range(3):
                grad = eqx.filter_grad(loss)(net)
                np.testing.assert_array_equal(grad.modes, np.zeros_like(expected))
                net = eqx.apply_updates(net, jax.tree.map(lambda a: -.01*a, grad))
            np.testing.assert_array_equal(net.modes, expected)
            for axis in (0, 1):
                left = jnp.asarray([.37, .41]).at[axis].set(0.)
                right = left.at[axis].set(1.)
                for t in (.2, 1., 2.):
                    np.testing.assert_allclose(net(left, t), net(right, t), atol=1e-14)
                    np.testing.assert_allclose(jax.jacfwd(net)(left, t), jax.jacfwd(net)(right, t), atol=1e-13)

    def test_coordinate_inverse_derivative_and_exact_ic(self):
        c = get_config(); validate_config(c)
        with jax.enable_x64(True):
            phi = jnp.linspace(-10., 10., 101)
            u = phi+phi**3
            np.testing.assert_allclose(inverse_cubic(u), phi, atol=1e-14)
            derivative = jax.vmap(jax.grad(inverse_cubic))(u)
            np.testing.assert_allclose(derivative, 1/(1+3*phi**2), atol=1e-15)
            net = PhaseNetwork(c, jax.random.PRNGKey(42))
            points = jnp.asarray(grid(8, 1.))
            predicted = jax.vmap(net, in_axes=(0, None))(points, 0.)[:, 0]
            np.testing.assert_array_equal(predicted, jax.vmap(net.initial)(points))

    def test_fft_block_inverse_matches_independent_dense_solve(self):
        c = get_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        with jax.enable_x64(True):
            D = PeriodicLaplacian(8, 1., .3, dtype=np.float64)
            L = np.asarray(D @ jnp.eye(64))
            tau = c.preconditioner.time_scale
            B = c.preconditioner.curvature*np.eye(64)+c.physical.epsilon**2*L
            A = np.block([[np.eye(64)/tau, c.physical.mobility*L], [-B, np.eye(64)]])
            r = np.random.default_rng(3).normal(size=(64, 2))
            z = np.linalg.solve(A, r.T.ravel()).reshape(2, 64).T
            z[:, 0] /= tau
            actual = make_schur_transform(D, c, dtype=np.float64)(jnp.asarray(r))
            np.testing.assert_allclose(actual, z, atol=2e-14)
            # A constant phase-transport residual is retained, not projected out.
            constant = jnp.tile(jnp.asarray([1., 0.]), (64, 1))
            transformed = make_schur_transform(D, c, dtype=np.float64)(constant)
            np.testing.assert_allclose(transformed[:, 0], 1., atol=1e-15)

    def test_full_parameter_gradient_and_physical_zeros(self):
        c = get_config(); c.sample.Nx = c.sample.Ny = 8; c.nl.delta = .3
        with jax.enable_x64(True):
            points = jnp.asarray(grid(8, 1.))
            for D in (None, PeriodicLaplacian(8, 1., .3, dtype=np.float64)):
                loss = make_loss_fn(points, D, c)
                f = lambda a: loss(SmoothFields(a), jnp.asarray([.2, .7]), jnp.asarray(0.))[0]
                a = jnp.asarray(.7)
                finite = (f(a+1e-6)-f(a-1e-6))/2e-6
                np.testing.assert_allclose(jax.grad(f)(a), finite, rtol=2e-7)
                for phase in (-1.05, 0., 1.05):
                    value, _ = loss(ConstantFields(jnp.asarray(phase)), jnp.asarray([.3]), jnp.asarray(phase))
                    self.assertLess(float(value), 1e-27)


if __name__ == '__main__':
    unittest.main()
