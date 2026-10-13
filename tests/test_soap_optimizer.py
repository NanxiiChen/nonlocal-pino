"""Upstream SOAP moment, public factory, JIT and checkpoint regression checks."""
import tempfile
import unittest
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from pinn.optimizers import make_optimizer, soap


class SoapOptimizerTest(unittest.TestCase):
    def test_elementwise_moments_are_updated_without_precondition_refresh(self):
        opt = soap(1e-3, b1=.9, b2=.95, precondition_frequency=5,
                   max_precond_dim=8, precondition_1d=False)
        params = jnp.zeros(2)
        state = opt.init(params)
        update = jax.jit(opt.update)
        _, state = update(jnp.array([1., 2.]), state, params)
        _, state = update(jnp.array([2., 4.]), state, params)
        np.testing.assert_allclose(state[0].exp_avg_sq, [.2, .8], rtol=1e-6)
        _, state = update(jnp.array([3., 1.]), state, params)
        np.testing.assert_allclose(state[0].exp_avg_sq, [.64, .81], rtol=1e-6)
        self.assertEqual(int(state[0].count), 3)

    def test_public_factory_checkpoint_continuation_matches_uninterrupted(self):
        config = ConfigDict(dict(train=dict(optimizer='soap', grad_clip=1000.,
            lr_init=1e-3, lr_decay_every=16, lr_decay_rate=.1, lr_min=1e-4,
            soap_b1=.95, soap_b2=.95, soap_precond_freq=2,
            soap_max_precond_dim=8, soap_precond_1d=False)))
        opt = make_optimizer(config)
        params = {'matrix': jnp.array([[.5, -.3], [.2, .8]]), 'bias': jnp.zeros(2)}
        state = opt.init(params)
        update = jax.jit(opt.update)
        gradients = jax.tree.map(lambda p: p+1, params)
        for _ in range(4):
            updates, state = update(gradients, state, params)
            params = eqx.apply_updates(params, updates)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'optimizer.eqx'
            eqx.tree_serialise_leaves(path, state)
            restored = eqx.tree_deserialise_leaves(path, opt.init(params))
        expected, continued = update(gradients, state, params)
        actual, resumed = update(gradients, restored, params)
        self.assertTrue(eqx.tree_equal(expected, actual))
        self.assertTrue(eqx.tree_equal(continued, resumed))
        self.assertEqual(int(resumed[1][0].count), 5)


if __name__ == '__main__':
    unittest.main()
