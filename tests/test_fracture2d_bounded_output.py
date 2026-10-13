"""Independent head derivatives, matching initial fields and AT2 algebra."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.bounded_damage_output import (
    MODES, head, inverse_head, make_image, reconstruct, validate_graph,
)
from examples.fracture2d.compact_damage_features import Readout
from examples.fracture2d.compact_damage_loading import JointNetwork
from examples.fracture2d.configs.output_damage_trial import get_config
from examples.fracture2d.output_damage_trial import make_problem, normalization


class BoundedOutputTest(unittest.TestCase):
    def test_ranges_inverses_and_exponential_stationary_point(self):
        jax.config.update('jax_enable_x64', True)
        values = jnp.asarray([1e-10, .001, .2, .99])
        for mode in MODES:
            z = inverse_head(values, mode, floor=1e-8)
            np.testing.assert_allclose(head(z, mode, floor=1e-8), values, rtol=1e-10, atol=1e-16)
            for value in (.3, -1.7):
                analytic = jax.grad(lambda x: head(x, mode, floor=1e-8))(value)
                eps = 1e-6
                fd = (head(value+eps, mode, floor=1e-8)-head(value-eps, mode, floor=1e-8))/(2*eps)
                np.testing.assert_allclose(analytic, fd, rtol=1e-7, atol=1e-10)
        self.assertEqual(float(head(0., 'exp_square', floor=1e-8)), 1.)
        self.assertEqual(float(jax.grad(lambda z: head(z, 'exp_square', floor=1e-8))(0.)), 0.)
        self.assertGreater(float(jnp.exp(-(-1.))), 1.)
        extreme = jnp.asarray([-30., -1., 0., 1., 30.])
        for mode in MODES[1:]:
            bounded = head(extreme, mode, floor=1e-8)
            self.assertGreaterEqual(float(bounded.min()), 0.)
            self.assertLessEqual(float(bounded.max()), 1.)

    def test_reconstruction_equivalence_and_parameter_gradients(self):
        jax.config.update('jax_enable_x64', True)
        D = np.asarray([[3., -1., -2.], [-1., 2., -1.], [-2., -1., 3.]])
        H = jnp.asarray([.1, .2, .4])
        g0, g1 = 2., .3
        diagonal, _ = validate_graph(csr_matrix(D), np.asarray(H), gc0=g0, gc1=g1, roundoff_factor=1024.)
        diagonal = jnp.asarray(diagonal)
        P = np.eye(3)-D/np.asarray(diagonal)[:, None]
        sigma = jnp.asarray([.1, .25, .8])
        d = reconstruct(sigma, H, diagonal, gc0=g0, gc1=g1)
        r = (g0+2*H)*d+g1*(D@d)-2*H
        np.testing.assert_allclose(r, g1*diagonal*(sigma-P@d), atol=1e-15)
        self.assertGreaterEqual(float(d.min()), 0.)
        self.assertLess(float(d.max()), 1.)
        B = jnp.asarray([[1., .2], [.3, 1.], [.5, .4]])
        beta = jnp.zeros(2, jnp.float64)
        common = None
        for mode in MODES:
            image, _ = make_image(lambda h, load: jax.nn.sigmoid(-2.+h)*load,
                lambda b, h, load: B@b, diagonal, mode=mode, gc0=g0, gc1=g1,
                floor=1e-8, initialization_epsilon=1e-10)
            compiled = jax.jit(image)
            initial = compiled(beta, H, jnp.asarray(.5))
            if common is None:
                common = initial
            np.testing.assert_allclose(initial, common, rtol=1e-13, atol=1e-14)
            self.assertGreater(float(jnp.max(abs(compiled(beta, H*2, .7)-initial))), 0.)
            def loss(b):
                damage = compiled(b, H, .5)
                rho = (g0+2*H)*damage+g1*(D@damage)-2*H
                return jnp.mean(rho**2)
            direction = jnp.asarray([.1, -.2])
            eps = 1e-5
            analytic = jnp.dot(jax.grad(loss)(beta), direction)
            fd = (loss(beta+eps*direction)-loss(beta-eps*direction))/(2*eps)
            np.testing.assert_allclose(analytic, fd, rtol=1e-7, atol=1e-11)
        bad = D.copy()
        bad[0, 1] = 1.
        with self.assertRaises(ValueError):
            validate_graph(csr_matrix(bad), np.asarray(H), gc0=g0, gc1=g1, roundoff_factor=1024.)

    def test_joint_network_tangent_normalization_and_peak_gradients(self):
        jax.config.update('jax_enable_x64', True)
        c = get_config()
        D = csr_matrix(np.asarray([[3., -1., -2.], [-1., 2., -1.], [-2., -1., 3.]]))
        B = csr_matrix(np.asarray([[1., .2], [.3, 1.], [.5, .4]]))
        H = np.asarray([1., 2., 4.])
        s = dict(D=D, matrix=B, H_np=H, H=jnp.asarray(H), load=jnp.asarray(.0033),
                 reference_load=.0048, diagonal=D.diagonal(), operator=None,
                 features=lambda h, load: h[:, None],
                 frozen_image=lambda h, load: .01+.001*h)
        linear = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0), dtype=jnp.float64)
        linear = eqx.tree_at(lambda n: (n.weight, n.bias), linear,
            (jnp.zeros_like(linear.weight), jnp.zeros_like(linear.bias)))
        net = JointNetwork(linear, Readout(jnp.zeros(2, jnp.float64)))
        with patch('examples.fracture2d.output_damage_trial.make_padded_matvec',
                   return_value=(lambda d: jnp.asarray(D.toarray())@d, {})):
            for mode in MODES:
                columns, gain, _ = normalization(c, s, mode)
                image, _, objectives, _ = make_problem(c, s, mode, columns, gain)
                # Compare initial residual Jacobian columns with the independent
                # host normalization, including the load ratio and decoder.
                gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
                A = jnp.asarray(np.diag(gc0+2*H)+gc1*D.toarray())
                jac = jax.jacfwd(lambda beta: A@image(
                    JointNetwork(linear, Readout(beta)), s['H'], s['load']))(net.readout.coefficients)
                np.testing.assert_allclose(jnp.mean((jac/(2*np.sqrt(np.mean(H**2))))**2, axis=0),
                                           np.ones(2), rtol=1e-12)
                delta = jax.tree.map(lambda x: jnp.full_like(x, .01), eqx.filter(net, eqx.is_inexact_array))
                for objective in objectives.values():
                    grad = eqx.filter_grad(objective)(net, s['H'], s['load'])
                    self.assertGreater(float(jnp.linalg.norm(grad.trunk.weight)), 0.)
                    self.assertGreater(float(jnp.linalg.norm(grad.readout.coefficients)), 0.)
                    eps = 1e-5
                    plus = eqx.apply_updates(net, jax.tree.map(lambda x: eps*x, delta))
                    minus = eqx.apply_updates(net, jax.tree.map(lambda x: -eps*x, delta))
                    fd = (objective(plus, s['H'], s['load'])-objective(minus, s['H'], s['load']))/(2*eps)
                    analytic = sum(jnp.sum(a*b) for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(delta)))
                    np.testing.assert_allclose(fd, analytic, rtol=1e-6, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
