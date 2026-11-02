"""Independent weighting algebra and complete coupled differentiated controls."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from tests.test_channel_nsch import small
from examples.channel_nsch2d.balance import balanced_value_and_grad,group_losses,group_names,make_balanced_fn
from examples.channel_nsch2d.configs.train_stream_gradnorm import get_config
from examples.channel_nsch2d.network import build_network


class BalanceTests(unittest.TestCase):
    def test_analytic_weights_frozen_gradient_and_refresh(self):
        with jax.enable_x64(True):
            settings=ConfigDict(dict(epsilon=1e-6,cap=1000.,new_weight_fraction=.9))
            fn=lambda p:(jnp.array([p[0]**2,3*p[1]**2,(p[0]-p[1])**2]),p)
            apply=eqx.filter_jit(lambda p,w,flag:balanced_value_and_grad(fn,p,w,flag,settings))
            p=jnp.array([2.,1.]);old=jnp.array([1.,2.,3.])
            total,_,weights,norms,gradient=apply(p,old,jnp.array(True))
            np.testing.assert_allclose(norms,[4.,6.,4.],atol=1e-12)
            np.testing.assert_allclose(weights,[1.15,.9,1.35],atol=1e-12)
            np.testing.assert_allclose(gradient,[7.3,2.7],atol=1e-12)
            self.assertAlmostEqual(float(total),8.65)
            _,_,same,zero,g=apply(p,weights,jnp.array(False))
            np.testing.assert_array_equal(same,weights);np.testing.assert_array_equal(zero,0.)
            np.testing.assert_allclose(g,gradient,atol=1e-12)
            jac=jax.jacfwd(lambda x:balanced_value_and_grad(fn,x,old,jnp.array(True),settings)[2])(p)
            np.testing.assert_array_equal(jac,0.)

    def test_grouping_and_complete_physics_gradients(self):
        with jax.enable_x64(True):
            for stream in (False,True):
                c,data,model=small();c.balance=get_config().balance
                if not stream:
                    del c.velocity
                    model=build_network(c,jax.random.PRNGKey(42))
                batch=jax.tree.map(lambda a:a[None],data)
                apply=eqx.filter_jit(make_balanced_fn(c));n=len(group_names(c))
                previous=jnp.ones(n);times=jnp.array([.3,1.])
                total,parts,weights,norms,g=apply(model,batch,times,previous,jnp.array(True))
                self.assertTrue(np.isfinite(float(total)))
                self.assertTrue(np.isfinite(np.asarray(norms)).all())
                self.assertEqual(len(parts),9 if stream else 10)
                self.assertEqual(n,4 if stream else 5)
                np.testing.assert_allclose(group_losses(parts,c).sum(),parts.sum(),atol=1e-12)
                np.testing.assert_allclose(total,jnp.dot(weights,group_losses(parts,c)),atol=1e-12)
                self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g) if eqx.is_array(v)))
                _,_,held,_,_=apply(model,batch,times,weights,jnp.array(False))
                np.testing.assert_array_equal(held,weights)


if __name__=='__main__':unittest.main()
