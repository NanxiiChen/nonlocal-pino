"""Precision controls preserve experiment inputs and differentiable dtypes."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d.configs.train_base import get_config
from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.training_precision import (
    computation_dtype, cast_floating, audit_floating,
)
from nonlocal_pinn.training.runtime import with_run_seed


class CorrosionPrecisionTests(unittest.TestCase):
    def test_precision_validation_and_default(self):
        c = get_config()
        self.assertEqual(computation_dtype(c), np.dtype('float64'))
        c.train.compute_dtype = 'bfloat16'
        with self.assertRaisesRegex(ValueError, 'compute_dtype'):
            computation_dtype(c)

    def test_common_initialization_times_and_fp64_gradient_optimizer(self):
        c = with_run_seed(get_config(), 42)
        with jax.enable_x64(False):
            kn, kt = jax.random.split(jax.random.PRNGKey(42))
            net32 = r.build_model(c, kn)
            x32 = jnp.asarray([.5, .1], dtype=jnp.float32)
            t32 = r.sample_training_times(c, kt, 4999)
        with jax.enable_x64(True):
            dtype = np.dtype('float64')
            net64, x64, t64 = cast_floating((net32, x32, t32), dtype)
            for a, b in zip(jax.tree.leaves(eqx.filter(net32, eqx.is_array)),
                            jax.tree.leaves(eqx.filter(net64, eqx.is_array))):
                np.testing.assert_array_equal(np.asarray(a, dtype=np.float64), b)
            np.testing.assert_array_equal(np.asarray(t32, dtype=np.float64), t64)
            value, grad = eqx.filter_value_and_grad(
                lambda n: jnp.sum(n(x64, t64[0])**2))(net64)
            self.assertEqual(value.dtype, dtype)
            audit_floating(grad, dtype)
            optimizer = r.make_tail_optimizer(c)
            state = optimizer.init(eqx.filter(net64, eqx.is_array))
            updates, state = optimizer.update(grad, state, net64)
            audit_floating((eqx.apply_updates(net64, updates), state), dtype)
            with self.assertRaises(TypeError):
                audit_floating(net32, dtype)


if __name__ == '__main__':
    unittest.main()
