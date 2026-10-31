"""Preserve the expected initial/positive-time PDE measure as Nt changes."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_problem import validate
from examples.confined_ch2d.operator_training import build_network,make_loss
from test_confined_ch_operator import small_fixture


class OperatorTimeWeightTests(unittest.TestCase):
    def test_original_four_positive_mean_value_and_full_gradient_are_preserved(self):
        with jax.enable_x64(True):
            c,data=small_fixture();c.sample.Nt=4
            net=build_network(c,jax.random.key(42));batch=jax.tree.map(lambda a:a[None],data)
            times=jnp.array([0.,.1,.2,.5,.7])
            old_value,old_grad=eqx.filter_value_and_grad(make_loss(c))(net,batch,times)
            changed=ConfigDict(c.to_dict());changed.loss.initial_residual_fraction=.2
            validate(changed)
            value,grad=eqx.filter_value_and_grad(make_loss(changed))(net,batch,times)
            np.testing.assert_allclose(value,old_value,atol=1e-14,rtol=1e-13)
            for a,b in zip(jax.tree.leaves(grad),jax.tree.leaves(old_grad),strict=True):
                if eqx.is_array(a):np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-12)

    def test_larger_fixed_shape_batch_backprop_and_invalid_fraction(self):
        with jax.enable_x64(True):
            c,data=small_fixture();c.sample.Nt=16;c.operator.condition_batch=16
            c.loss.initial_residual_fraction=.2;validate(c)
            net=build_network(c,jax.random.key(42));batch=jax.tree.map(lambda a:jnp.stack([a]*16),data)
            times=jnp.concatenate((jnp.zeros(1),(jnp.arange(16)+.5)*c.physical.terminal_time/16))
            value,grad=eqx.filter_jit(eqx.filter_value_and_grad(make_loss(c)))(net,batch,times)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(grad) if eqx.is_array(a)))
            for bad in (-.1,1.,float('nan'),True):
                broken=ConfigDict(c.to_dict())
                with broken.loss.ignore_type():broken.loss.initial_residual_fraction=bad
                with self.assertRaises(ValueError):validate(broken)


if __name__=='__main__':unittest.main()
