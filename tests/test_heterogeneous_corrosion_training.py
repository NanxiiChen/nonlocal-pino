"""Optimizer and loss-count integration for the matched derivative controls."""
import unittest
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.heterogeneous_corrosion2d.configs.train_base import get_config
from examples.heterogeneous_corrosion2d.losses import make_step
from nonlocal_pinn.training.optimizers import make_optimizer
from examples.heterogeneous_corrosion2d.replay import _check_config


class HeterogeneousOptimizerTests(unittest.TestCase):
    def test_adam_and_soap_handle_five_or_seven_losses(self):
        for optimizer_name, count in (('adam', 5), ('soap', 5), ('soap', 7)):
            with self.subTest(optimizer=optimizer_name, losses=count):
                c = get_config()
                c.train.optimizer = optimizer_name
                c.loss.gradnorm_bias = (1.,) * count
                model = eqx.nn.Linear(2, 2, key=jax.random.key(17))
                original = np.asarray(model.weight).copy()

                def losses(net, times, payload, sub, mask):
                    predicted = net(jnp.array([.3, -.2]))
                    return jnp.stack([jnp.sum((predicted-.1*(i+1))**2) for i in range(count)])

                optimizer = make_optimizer(c)
                state = optimizer.init(eqx.filter(model, eqx.is_array))
                step = make_step(losses, optimizer, c, tuple(f'loss{i}' for i in range(count)))
                weights = jnp.ones(count)
                for refresh in (True, False, True):
                    model, state, weights, values, finite = step(
                        model, state, weights, jnp.array([.5]), None, None, None, jnp.asarray(refresh))
                    self.assertTrue(bool(finite))
                    self.assertEqual(values.shape, (count,))
                    self.assertTrue(np.isfinite(np.asarray(weights)).all())
                self.assertGreater(float(np.linalg.norm(np.asarray(model.weight)-original)), 0)

    def test_bias_must_match_declared_loss_count(self):
        c = get_config()
        with self.assertRaisesRegex(ValueError, 'every declared loss'):
            make_step(None, make_optimizer(c), c, ('only_one',))

    def test_paired_lr_exception_does_not_unlock_physics_or_schedule(self):
        c = get_config()
        source = json.loads(c.to_json())
        c.train.optimizer = 'soap'
        c.train.lr_init = 3e-4
        target = json.loads(c.to_json())
        with self.assertRaisesRegex(ValueError, 'train.lr_init'):
            _check_config(source, target)
        _check_config(source, target, allow_learning_rate_change=True)
        target['train']['lr_decay_rate'] = .8
        with self.assertRaisesRegex(ValueError, 'lr_decay_rate'):
            _check_config(source, target, allow_learning_rate_change=True)
        target = json.loads(c.to_json())
        target['physical']['n_ch'] = 40.
        with self.assertRaisesRegex(ValueError, 'n_ch'):
            _check_config(source, target, allow_learning_rate_change=True)


if __name__ == '__main__':
    unittest.main()
