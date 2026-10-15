"""Known-history feature and neural residual directional-gradient checks."""
import unittest
from types import SimpleNamespace
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.conditioned_damage import make_features, make_problem, _build_network
from examples.fracture2d._legacy.configs.conditioned_damage_trial import get_config


class ConditionedDamageTest(unittest.TestCase):
    def test_feature_average_and_residual_gradient(self):
        jax.config.update('jax_enable_x64', True)
        D = csr_matrix(np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]]))
        op = SimpleNamespace(to_scipy=lambda: D)
        points = np.array([[.45, .5], [.5, .45], [.55, .5]])
        H = np.array([1., 2., 4.])
        f, info = make_features(op, points, gc0=112.5, length=.024, tip=[.5, .5],
                               powers=[0, 1, 2, 4], roundoff_factor=1024.)
        actual = np.asarray(f(jnp.asarray(H)))
        P = np.eye(3)-D.toarray()/2
        for j, k in enumerate([0, 1, 2, 4]):
            ratio = 2*np.linalg.matrix_power(P, k)@H/112.5
            np.testing.assert_allclose(actual[:, 6+2*j], np.log1p(ratio), rtol=1e-14)
            np.testing.assert_allclose(actual[:, 7+2*j], ratio/(1+ratio), rtol=1e-14)
        c = get_config()
        c.arch.hidden_dim, c.arch.num_layers = 8, 2
        net = _build_network(c, info['input_dimension'])
        net = eqx.tree_at(lambda n: n.output.bias, net, jnp.array([.2]))
        image, metrics, objectives = make_problem(f, lambda d: jnp.asarray(D.toarray())@d, c)
        for objective in objectives.values():
            value, gradient = eqx.filter_value_and_grad(objective)(net, jnp.asarray(H), jnp.asarray(.0048))
            direction = jax.tree.map(lambda x: x*.01, eqx.filter(net, eqx.is_inexact_array))
            step = 1e-4
            plus = eqx.apply_updates(net, jax.tree.map(lambda x: step*x, direction))
            minus = eqx.apply_updates(net, jax.tree.map(lambda x: -step*x, direction))
            fd = (objective(plus, jnp.asarray(H), .0048)-objective(minus, jnp.asarray(H), .0048))/(2*step)
            analytic = sum(jnp.sum(x*y) for x, y in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction)))
            np.testing.assert_allclose(fd, analytic, rtol=1e-6, atol=1e-10)
            self.assertTrue(np.isfinite(float(value)))
        predicted = np.asarray(image(net, jnp.asarray(H), .0048))
        self.assertTrue(np.all(predicted > 0))
        self.assertTrue(np.isfinite(float(metrics(net, jnp.asarray(H), .0048)['AT2_relative'])))


if __name__ == '__main__':
    unittest.main()
