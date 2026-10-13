"""Neural loading uses the changing own H/load and trainable coefficients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.compact_damage_features import Readout
from examples.fracture2d.compact_damage_loading import make_problem, JointNetwork, history_feature_input
from examples.fracture2d.configs.compact_damage_loading import get_config


class CompactLoadingTest(unittest.TestCase):
    def test_known_history_load_scaling(self):
        jax.config.update('jax_enable_x64', True)
        H = jnp.asarray([1., 2., 3.])
        ref = .0048
        compiled = jax.jit(lambda h, load: history_feature_input(h, load, ref, 'load_squared'))
        np.testing.assert_array_equal(compiled(H, ref), H)
        np.testing.assert_allclose(compiled(H/4, ref/2), H, rtol=1e-14)
        np.testing.assert_array_equal(jax.jacfwd(lambda h: compiled(h, ref))(H), np.zeros((3, 3)))
        np.testing.assert_array_equal(history_feature_input(H, ref/2, ref, 'physical'), H)

    def test_changing_history_load_and_gradients(self):
        jax.config.update('jax_enable_x64', True)
        c = get_config()
        B = jnp.asarray([[1., .2], [.3, 1.], [.5, .5]])
        D = jnp.asarray([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
        reference = .0048
        def trunk(n, H, load):
            return (load/reference)**2*(.01+.0001*H)
        image, metrics, objectives = make_problem(trunk, lambda n: B@n.coefficients, lambda d: D@d, c, reference)
        net = Readout(jnp.asarray([.002, -.001]))
        H = jnp.asarray([1., 2., 3.])
        compiled = jax.jit(image)
        first = compiled(net, H, reference)
        changed = compiled(net, H*2, reference)
        self.assertGreater(float(jnp.max(abs(first-changed))), 0.)
        raw = trunk(net, H, reference)+B@net.coefficients
        np.testing.assert_allclose(first, .5*(raw+jnp.sqrt(raw**2+c.physical.damage_floor**2)), atol=1e-15)
        smaller = compiled(net, H, reference/2)
        np.testing.assert_allclose(smaller, first/4, atol=1e-13)
        direction = jnp.asarray([.1, -.05])
        eps = 1e-5
        for objective in objectives.values():
            def loss(beta):
                return objective(Readout(beta), H, reference)
            analytic = jnp.dot(jax.grad(loss)(net.coefficients), direction)
            fd = (loss(net.coefficients+eps*direction)-loss(net.coefficients-eps*direction))/(2*eps)
            np.testing.assert_allclose(fd, analytic, rtol=1e-7, atol=1e-10)
        self.assertTrue(np.isfinite(float(metrics(net, H, reference)['AT2_relative'])))
        linear = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0), dtype=jnp.float64)
        linear = eqx.tree_at(lambda n: (n.weight, n.bias), linear,
                             (jnp.asarray([[.0001]]), jnp.asarray([.01])))
        joint = JointNetwork(linear, net)
        def learned_trunk(n, history, prescribed):
            return (prescribed/reference)**2*jax.vmap(n.trunk)(history[:, None])[:, 0]
        _, _, joint_objectives = make_problem(learned_trunk, lambda n: B@n.readout.coefficients, lambda d: D@d, c, reference)
        delta = jax.tree.map(lambda x: x*.1, eqx.filter(joint, eqx.is_inexact_array))
        for objective in joint_objectives.values():
            grad = eqx.filter_grad(objective)(joint, H, reference)
            self.assertGreater(float(jnp.linalg.norm(grad.trunk.weight)), 0.)
            self.assertGreater(float(jnp.linalg.norm(grad.readout.coefficients)), 0.)
            plus = eqx.apply_updates(joint, jax.tree.map(lambda x: eps*x, delta))
            minus = eqx.apply_updates(joint, jax.tree.map(lambda x: -eps*x, delta))
            fd = (objective(plus, H, reference)-objective(minus, H, reference))/(2*eps)
            analytic = sum(jnp.sum(x*y) for x, y in zip(jax.tree.leaves(grad), jax.tree.leaves(delta)))
            np.testing.assert_allclose(fd, analytic, rtol=1e-7, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
