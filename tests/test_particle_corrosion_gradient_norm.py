"""Stable norm arithmetic and complete ordinary/causal optimizer updates."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from examples.particle_corrosion2d import losses,strategies
from tests.test_particle_corrosion_losses import small_config


class GradientNormTests(unittest.TestCase):
    def test_norms_zero_and_large_finite_leaves(self):
        tree={'a':jnp.array([3.,-4.]),'b':jnp.zeros((2,2)),'c':None}
        self.assertEqual(float(strategies.parameter_gradient_norm(tree,'l1')),7.)
        self.assertEqual(float(strategies.parameter_gradient_norm(tree,'l2')),5.)
        self.assertEqual(float(strategies.parameter_gradient_norm(jnp.zeros(3),'l2')),0.)
        large=jnp.array([1e30,-1e30],dtype=jnp.float32)
        actual=strategies.parameter_gradient_norm(large,'l2')
        self.assertTrue(np.isfinite(float(actual)))
        np.testing.assert_allclose(actual,np.sqrt(2)*1e30,rtol=2e-6)

    def check_update(self,kind,causal):
        c=small_config()
        if kind is not None:c.loss.gradnorm_norm=kind
        model=eqx.nn.Linear(2,2,key=jax.random.key(72));x=jnp.array([.3,-.4]);times=jnp.array([.1,.6])
        def vector(net,times,data):
            y=net(data)
            return jnp.stack(((y[0]-.2)**2,(y[1]+.6)**2,(y.sum()+.4)**2,(y[0]-y[1]-.7)**2,jnp.sum((y-.5)**2)))
        weights=jnp.array([.7,.9,1.2,1.4,.8]);norms=[]
        for i in range(5):
            g=eqx.filter_grad(lambda net:vector(net,times,x)[i])(model)
            arrays=[np.asarray(a,dtype=np.float64) for a in jax.tree.leaves(g) if eqx.is_array(a)]
            norms.append(np.sqrt(sum(np.sum(a*a) for a in arrays)) if kind=='l2'
                         else sum(np.sum(np.abs(a)) for a in arrays))
        norms=np.clip(norms,1e-6,1e6);proposal=np.clip(np.mean(norms)/norms,1e-6,1000.)
        expected_weights=jnp.asarray(.9*proposal+.1*np.asarray(weights),dtype=weights.dtype)
        optimizer=optax.adam(1e-3);state=optimizer.init(eqx.filter(model,eqx.is_array))
        gradient=eqx.filter_grad(lambda net:jnp.sum(vector(net,times,x)*expected_weights))(model)
        updates,expected_state=optimizer.update(gradient,state,model);expected=eqx.apply_updates(model,updates)
        if causal:
            c.causal=ConfigDict(dict(step_size=1.1,max_eps=100.,max_last_weight=.9,min_mean_weight=.5))
            def components(net,times,data,epsilon):
                return vector(net,times,data),jnp.array([[1.,.6],[2.,3.]])
            result=strategies.make_causal_step(components,optimizer,c,losses.LOSS_NAMES)(
                model,state,weights,jnp.array(1.),times,x,jnp.array(True))
            actual,actual_state,actual_weights,eps,_,_,finite=result
            self.assertEqual(float(eps),1.)
        else:
            actual,actual_state,actual_weights,_,finite=losses.make_step(vector,optimizer,c,losses.LOSS_NAMES)(
                model,state,weights,times,x,jnp.array(True))
        self.assertTrue(bool(finite));np.testing.assert_allclose(actual_weights,expected_weights,rtol=2e-6)
        for a,b in zip(jax.tree.leaves((actual,actual_state)),jax.tree.leaves((expected,expected_state)),strict=True):
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7)

    def test_complete_ordinary_updates_preserve_l1_and_select_l2(self):
        self.check_update(None,False);self.check_update('l2',False)

    def test_complete_causal_l2_update(self):
        self.check_update('l2',True)


if __name__=='__main__':unittest.main()
