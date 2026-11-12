"""Row-sampled dense residuals retain the original global physics and gradients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tests.test_confined_ch_operator import small_fixture
from examples.confined_ch2d.operator_training import build_network, make_loss
from examples.confined_ch2d.operator_spatial import residual_rows


class ResidualRowTests(unittest.TestCase):
    def test_full_coverage_and_weighted_one_row_expectation(self):
        with jax.enable_x64(True):
            c, data = small_fixture()
            # Off-diagonal coupling makes selecting rows before/after the
            # dense metric distinguishable (identity would miss that bug).
            data['metric'] = jnp.array([[1., .3, .2, 0.], [.1, .9, .2, 0.], [.3, .2, .8, 0.], [0., 0., 0., 0.]])
            net = build_network(c, jax.random.key(25))
            fn = eqx.filter_jit(eqx.filter_value_and_grad(make_loss(c)))
            times = jnp.array([0., .05])
            batch = lambda d: jax.tree.map(lambda a: jnp.asarray(a)[None], d)
            value, gradient = fn(net, batch(data), times)
            weights = np.asarray(data['volumes'])/np.sum(data['volumes'])
            sample = dict(data, residual_core_indices=jnp.arange(4), residual_core_weights=weights)
            actual, derivative = fn(net, batch(sample), times)
            np.testing.assert_allclose(actual, value, rtol=1e-11)
            for a, b in zip(jax.tree.leaves(derivative), jax.tree.leaves(gradient), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
            components = []
            derivatives = []
            for i in range(3):
                sample = dict(data, residual_core_indices=jnp.array([i]), residual_core_weights=jnp.ones(1))
                v, g = fn(net, batch(sample), times)
                components.append(v)
                derivatives.append(g)
            np.testing.assert_allclose(weights[:3]@np.asarray(components), value, rtol=1e-11)
            expected = jax.tree.map(lambda *leaves: sum(w*x for w, x in zip(weights[:3], leaves, strict=True)), *derivatives)
            for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(gradient), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    def test_padding_excluded_and_complete_global_arrays_preserved(self):
        _, data = small_fixture()
        sampled = residual_rows(data, 32, np.random.default_rng(12))
        self.assertEqual(sampled['residual_core_indices'].shape, (32,))
        self.assertTrue(np.all(sampled['residual_core_indices'] < 3))
        for name in data:
            self.assertIs(data[name], sampled[name])
        np.testing.assert_allclose(sampled['residual_core_weights'].sum(), 1.)


if __name__ == '__main__':
    unittest.main()
