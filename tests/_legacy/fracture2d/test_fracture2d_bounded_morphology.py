"""Bounded dynamic-history gradients and physical morphology diagnostics."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.bounded_damage_morphology_trial import make_problem, morphology
from examples.fracture2d._legacy.configs.bounded_damage_morphology_trial import get_config
from examples.fracture2d._legacy.plain_damage_head import fresh_network, initialization


class BoundedMorphologyTest(unittest.TestCase):
    def test_dynamic_context_is_detached_but_network_gradient_is_correct(self):
        jax.config.update('jax_enable_x64', True)
        c = get_config()
        c.arch.hidden_dim = 4
        c.arch.num_layers = 1
        B = csr_matrix(np.eye(3, dtype=np.float64))
        columns = jnp.arange(3, dtype=jnp.int32)[:, None]
        values = jnp.ones((3, 1), jnp.float64)
        D = jnp.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
        ctx = dict(H=jnp.array([1., 2., 3.]), features=jnp.array([[1., 2.], [2., 3.], [3., 4.]]))
        for head in c.head.coupled_heads:
            gain = initialization(head, c.head.initial_damage)[2]
            net = fresh_network(c, 2, B, head=head)
            image, objective, metrics = make_problem(columns, values, lambda d: D@d,
                head=head, gain=gain, gc0=112.5, gc1=.0648, power=16, floor=1e-15,
                volumes=jnp.ones(3), thresholds=(.1, .5, .9))
            load = jnp.asarray(.0052)
            np.testing.assert_allclose(image(net, ctx, load), .03, rtol=1e-13)
            context_gradient = jax.grad(lambda context: objective(net, context, load))(ctx)
            for array in context_gradient.values():
                np.testing.assert_array_equal(array, jnp.zeros_like(array))
            gradient = eqx.filter_grad(objective)(net, ctx, load)
            eps = 1e-6
            def shifted(delta):
                return eqx.tree_at(lambda n: n.readout.coefficients, net,
                    net.readout.coefficients+jnp.array([delta, 0., 0.]))
            fd = (objective(shifted(eps), ctx, load)-objective(shifted(-eps), ctx, load))/(2*eps)
            np.testing.assert_allclose(gradient.readout.coefficients[0], fd, rtol=1e-6, atol=1e-9)
            extreme = shifted(100.)
            output = np.asarray(image(extreme, ctx, load))
            self.assertTrue(np.all((output >= 0) & (output <= 1)))
            self.assertTrue(np.isfinite(list(metrics(extreme, ctx, load).values())).all())

    def test_background_measure_is_volume_weighted_and_validated(self):
        xy = np.array([[.2, .1], [.7, .5], [.9, .8]])
        d = np.array([.2, .99, .6])
        v = np.array([1., 10., 3.])
        result = morphology(d, xy, v, band_width=.072, threshold=.5)
        self.assertAlmostEqual(result['outside_band_mean'], .5)
        self.assertAlmostEqual(result['outside_band_fraction_ge_threshold'], .75)
        self.assertTrue(result['mask_is_evaluation_only'])
        with self.assertRaises(ValueError):
            morphology(d, xy, np.zeros(3), band_width=.072, threshold=.5)
        with self.assertRaises(ValueError):
            morphology(d, xy, v, band_width=2., threshold=.5)


if __name__ == '__main__':
    unittest.main()
