"""Analytic local residual/normal derivatives and independent reference checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d.residual_experiment import grid, nondim, solve_reference
from examples.corrosion2d.local_reference import negative_laplacian
from examples.electropolish2d.configs.train_operator_local import get_config
from examples.electropolish2d.operator_data import validate_config
from examples.electropolish2d.operator_local import (
    encoded_fields, local_residual, make_loss_vector, normal_derivatives, spatial_laplacians)
from examples.electropolish2d.operator_reference import make_system, solve


class AnalyticRaw(eqx.Module):
    w: jax.Array

    def encode(self, image):
        return self.w*jnp.mean(image)

    def prepare_encoded(self, encoded, x, local):
        return jnp.column_stack([x, jnp.full(len(x), encoded)])

    def decode(self, prepared, times):
        x, y, z = prepared.T
        t = times[:, None]
        a = self.w*x[None, :]+.3*y[None, :]+.2*t+.1*z[None, :]
        b = .4*x[None, :]**2+.2*y[None, :]**2+.3*t+.1*z[None, :]
        return jnp.stack([a, b], -1)


class LocalOperatorTests(unittest.TestCase):
    def test_fd_mirror_cosine_eigenmode_and_constant(self):
        nx, ny = 11, 6
        xx, yy = np.meshgrid(np.linspace(0., 1., nx), np.linspace(0., .5, ny))
        matrix = negative_laplacian(nx, ny)
        mode = (np.cos(np.pi*xx)*np.cos(2*np.pi*yy)).ravel()
        hx, hy = 1/(nx-1), .5/(ny-1)
        eigenvalue = 4*np.sin(np.pi*hx/2)**2/hx**2+4*np.sin(np.pi*hy)**2/hy**2
        np.testing.assert_allclose(matrix@mode, eigenvalue*mode, rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(matrix@np.ones(nx*ny), 0., atol=1e-12)

    def test_total_concentration_has_no_unit_interval_maximum_principle(self):
        with jax.enable_x64(True):
            c = get_config(); c.surface.seed = 1688060238
            c.sample.Nx, c.sample.Ny = 11, 6
            _, _, x = grid(c)
            rhs, _, state, _ = make_system(c, negative_laplacian(11, 6), x)
            free = (x[:, 1] > 0.) & (x[:, 1] < .5)
            ids = np.flatnonzero(free); nf = len(ids)
            phi = .2+.4*np.exp(-((x[:, 0]-.5)**2+(x[:, 1]-.2)**2)/.02)
            state[:nf] = phi[free]; state[nf:] = 1.
            center = np.flatnonzero(np.isclose(x[ids, 0], .5) & np.isclose(x[ids, 1], .2))[0]
            self.assertGreater(rhs(0., state)[nf+center], 0.)

    def test_analytic_residual_neumann_mask_and_parameter_gradient(self):
        with jax.enable_x64(True):
            c = get_config(); validate_config(c)
            net = AnalyticRaw(jnp.array(.7))
            image = jnp.linspace(0, 1, 32).reshape(1, 4, 8)
            x = jnp.array([[.2, .1], [.7, .4], [.3, .2], [.4, .3],
                           [.5, .5], [.5, 0.], [0., .2], [1., .2]])
            times = jnp.array([.2, .8])
            encoded = net.encode(image)
            raw = np.asarray(net.decode(net.prepare_encoded(encoded, x, jnp.zeros_like(x)), times))
            v, u = raw[..., 0], raw[..., 1]
            phi = .5*(1+np.tanh(v)); sech2 = 1-np.tanh(v)**2
            cle = c.physical.c_le; dc = c.physical.c_se-cle
            q = cle*(u-1)
            lapphi = -sech2*np.tanh(v)*(.7**2+.3**2)
            lapq = np.full_like(phi, 1.2*cle)
            actual = spatial_laplacians(net, encoded, x, times, c)
            np.testing.assert_allclose(actual, np.stack([lapphi, lapq], -1), rtol=1e-11, atol=1e-12)
            pt = .1*sech2
            ct = .3*cle+dc*6*phi*(1-phi)*pt
            nch, n1, n2, n3, _, _ = nondim(c)
            ac = pt-n3*lapphi+n2*(4*phi**3-6*phi**2+2*phi)-n1*q*dc*6*phi*(1-phi)
            ch = ct-nch*lapq
            ap, cp = local_residual(net, encoded, x, times, nondim(c), c)
            np.testing.assert_allclose(ap, ac, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(cp, ch, rtol=1e-11, atol=1e-11)
            normals = jnp.array([[-1., 0.], [1., 0.]])
            normal = normal_derivatives(net, encoded, x[-2:], normals, times, c)
            expected = np.stack([.5*sech2[:, -2:]*.7*np.array([-1., 1.]),
                                 np.broadcast_to([0., .8], (2, 2))], -1)
            np.testing.assert_allclose(normal, expected, rtol=1e-11, atol=1e-12)
            known = jnp.full((1, len(x), 2), .2)
            mask = jnp.array([True, False])
            fn = make_loss_vector(c, nondim(c), 2, 1, 2)
            values = fn(net, times, (image[None], known, x), None, mask)
            np.testing.assert_allclose(values[:2], [np.mean(abs(ac[:, 0])), np.mean(abs(ch[:, 0]))], rtol=1e-11)
            np.testing.assert_allclose(values[-1], np.mean(np.sum(expected**2, -1)), rtol=1e-11)
            initial = encoded_fields(net, encoded, x[2:4], jnp.zeros(1), c)[0]
            np.testing.assert_allclose(values[4], jnp.mean(jnp.sum((initial-.2)**2, -1)), rtol=1e-11)
            objective = eqx.filter_jit(lambda n: fn(n, times, (image[None], known, x), None, mask).sum())
            h = 1e-5
            plus = eqx.tree_at(lambda n: n.w, net, net.w+h)
            minus = eqx.tree_at(lambda n: n.w, net, net.w-h)
            finite = (objective(plus)-objective(minus))/(2*h)
            ad = eqx.filter_grad(objective)(net).w
            np.testing.assert_allclose(ad, finite, rtol=1e-6, atol=1e-7)

    def test_local_requires_explicit_boundary_and_zero_query_features(self):
        c = get_config(); validate_config(c)
        for field, value in (('arch.local_input', 'initial'), ('loss.use_causal', True),
                             ('loss.neumann_scale', 0.), ('loss.gradnorm_bias', (1.,)*5)):
            broken = c.copy_and_resolve_references()
            section, name = field.split('.')
            broken[section][name] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(broken)

    def test_independent_reference_jacobian_and_existing_solver_agreement(self):
        with jax.enable_x64(True):
            c = get_config(); c.surface.seed = 1688060238
            c.sample.Nx, c.sample.Ny = 101, 51
            _, _, x = grid(c)
            matrix = negative_laplacian(c.sample.Nx, c.sample.Ny)
            rhs, jac, state, _ = make_system(c, matrix, x)
            tangent = np.random.default_rng(0).normal(size=len(state))
            h = 1e-5
            finite = (rhs(0., state+h*tangent)-rhs(0., state-h*tangent))/(2*h)
            np.testing.assert_allclose(jac(0., state)@tangent, finite, rtol=2e-6, atol=1e-7)
            times, old, _ = solve_reference(c, matrix, x, 1e-8, 1e-11, .005)
            new, _ = solve(c, matrix, x, times, 1e-8, 1e-11, .005)
            np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
