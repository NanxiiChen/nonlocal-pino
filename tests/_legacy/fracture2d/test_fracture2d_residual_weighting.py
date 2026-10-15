"""Detached weighting and exact support must preserve physical residuals."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.compact_damage_features import residual_power_loss
from examples.fracture2d._legacy.damage_residual_weighting import (
    make_weights, validate_weights, weighted_power_loss, build_core_halo,
    core_matvec, importance_moment,
)

jax.config.update('jax_enable_x64', True)


class ResidualWeightTest(unittest.TestCase):
    def test_uniform_loss_and_gradient_match_original(self):
        h = jnp.array([.2, .4, .1])
        r = jnp.array([-.03, .07, .11])
        original = lambda x: residual_power_loss(x, h, power=16, floor=1e-12)
        weighted = lambda x: weighted_power_loss(x, h, jnp.ones_like(h), power=16, floor=1e-12)
        for a, b in zip(jax.value_and_grad(original)(r), jax.value_and_grad(weighted)(r)):
            np.testing.assert_array_equal(a, b)

    def test_weights_positive_normalized_and_detached(self):
        points = jnp.array([[.5, .5], [.8, .5], [.8, .8]])
        d = jnp.array([.7, .2, -.0001])
        controls = dict(boost=9., damage_power=2., start=[.5, .5], end=[1., .5], width=.024)
        for mode in ('uniform', 'static', 'damage'):
            fn = lambda x: make_weights(points, x, mode=mode, **controls)
            w = fn(d)
            validate_weights(w, jnp.ones_like(d))
            self.assertAlmostEqual(float(w.mean()), 1.)
            self.assertLessEqual(float(w.max()/w.min()), 10.)
            np.testing.assert_array_equal(jax.jacrev(fn)(d), np.zeros((3, 3)))
        self.assertGreater(float(make_weights(points, d, mode='static', **controls)[1]),
                           float(make_weights(points, d, mode='static', **controls)[2]))
        self.assertGreater(float(make_weights(points, d, mode='damage', **controls)[0]),
                           float(make_weights(points, d, mode='damage', **controls)[1]))
        np.testing.assert_array_equal(d, [.7, .2, -.0001])

    def test_frozen_weight_gradient_is_true_line_search_gradient(self):
        h, w = jnp.array([.4, .7]), jnp.array([.5, 1.5])
        fn = lambda p: weighted_power_loss(jnp.array([p[0]+p[1], 2*p[0]-p[1]]),
                                           h, w, power=16, floor=1e-12)
        p = jnp.array([.3, -.2])
        analytic = jax.grad(fn)(p)
        finite = [(float(fn(p.at[i].add(1e-6)))-float(fn(p.at[i].add(-1e-6))))/2e-6 for i in range(2)]
        np.testing.assert_allclose(analytic, finite, rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(jax.grad(lambda weights: weighted_power_loss(p, h, weights, power=16, floor=1e-12))(w), [0., 0.])
        for bad in (jnp.array([0., 2.]), jnp.array([1., jnp.nan]), jnp.ones(3)):
            with self.assertRaises(ValueError): validate_weights(bad, h)

    def test_core_halo_primal_adjoint_and_padding_equal_original_rows(self):
        D = csr_matrix(np.array([[2., -1., -1., 0.], [-1., 3., -1., -1.],
                                [-1., -1., 2., 0.], [0., -1., 0., 1.]]))
        ids = np.array([0, 3, 0], np.int32)
        packed = build_core_halo(D, ids, node_capacity=7, row_width=5)
        def sampled(x):
            halo = x[packed['global_ids']]*packed['node_mask']
            return core_matvec(halo, packed['columns'], packed['values'])
        x = jnp.array([.2, .7, .4, -.1])
        np.testing.assert_allclose(sampled(x), (D@np.asarray(x))[ids], atol=1e-15)
        expected = np.asarray(D[ids].toarray()).T@np.array([.3, -.7, .2])
        np.testing.assert_allclose(jax.grad(lambda z: jnp.dot(sampled(z), jnp.array([.3, -.7, .2])))(x), expected, atol=1e-15)
        for capacity, width in ((2, 5), (7, 1)):
            with self.assertRaisesRegex(ValueError, 'truncate'):
                build_core_halo(D, ids, node_capacity=capacity, row_width=width)

    def test_importance_factor_recovers_full_moment_and_is_detached(self):
        p, f = jnp.array([.1, .2, .7]), jnp.array([2., 5., 11.])
        estimates = jnp.array([importance_moment(f[i:i+1], p, jnp.array([i]), population_size=3) for i in range(3)])
        np.testing.assert_allclose(jnp.dot(p, estimates), f.mean(), rtol=1e-15)
        np.testing.assert_array_equal(jax.grad(lambda q: importance_moment(f, q, jnp.arange(3), population_size=3))(p), jnp.zeros(3))


if __name__ == '__main__':
    unittest.main()
