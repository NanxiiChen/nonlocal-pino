"""Complete optimizer/mask and independent stopped-causal formula checks."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict
from tests.test_particle_corrosion_losses import small_config,fixture
from examples.particle_corrosion2d import losses,strategies


class StrategyTests(unittest.TestCase):
    def test_data_warmup_masks_only_physics_and_holds_causal_epsilon(self):
        c=small_config();c.loss.data_warmup_steps=3
        np.testing.assert_array_equal(strategies.active_losses(c,0,5),[0,0,1,1,1])
        np.testing.assert_array_equal(strategies.active_losses(c,3,5),[1,1,1,1,1])
        c.causal=ConfigDict(dict(step_size=1.1,max_eps=100.,max_last_weight=.9,min_mean_weight=.5))
        net=eqx.nn.Linear(2,2,key=jax.random.key(53));x=jnp.array([.1,.4]);ts=jnp.array([.1,.3,.8])
        optimizer=optax.adam(1e-3);state=optimizer.init(eqx.filter(net,eqx.is_array));weights=jnp.ones(5)
        def vector(model):
            y=model(x)
            return jnp.stack((y[0]**2,y[1]**2,(y.sum()-.3)**2,(y[0]-y[1]+.1)**2,jnp.sum((y+.7)**2)))
        def components(model,times,payload,epsilon):
            return vector(model),jnp.stack((payload,jnp.ones_like(payload)))
        step=strategies.make_causal_step(components,optimizer,c,losses.LOSS_NAMES)
        active=jnp.asarray([False,False,True,True,True])
        actual,state_actual,_,eps,_,_,finite=step(net,state,weights,jnp.array(1.),ts,jnp.array([1.,.1,.01]),jnp.array(False),active)
        gradient=eqx.filter_grad(lambda model:vector(model)[2:].sum())(net)
        updates,state_expected=optimizer.update(gradient,state,net);expected=eqx.apply_updates(net,updates)
        self.assertTrue(bool(finite));self.assertEqual(float(eps),1.)
        for a,b in zip(jax.tree.leaves((actual,state_actual)),jax.tree.leaves((expected,state_expected)),strict=True):
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7)

    def test_stagger_full_adam_update_keeps_data_terms(self):
        c=small_config();c.loss.stagger_steps=50;c.loss.stagger_switch_every=25
        model=eqx.nn.Linear(2,2,key=jax.random.key(51));x=jnp.array([.2,-.7])
        optimizer=optax.adam(1e-3);state=optimizer.init(eqx.filter(model,eqx.is_array))
        weights=jnp.array([.7,1.2,.9,1.5,.6]);ts=jnp.array([.1,.7])
        def vector(net,times,data):
            y=net(data)
            return jnp.stack(((y[0]-.3)**2,(y[1]+.4)**2,(y.sum()-.7)**2,(y[0]-y[1])**2,jnp.sum(y*y)))
        step=losses.make_step(vector,optimizer,c,losses.LOSS_NAMES)
        for iteration,expected_active in [(0,[1,0,1,1,1]),(25,[0,1,1,1,1]),(50,[1,1,1,1,1])]:
            active=strategies.active_losses(c,iteration,5)
            np.testing.assert_array_equal(active,np.array(expected_active,dtype=bool))
            gradient=eqx.filter_grad(lambda net:jnp.sum(vector(net,ts,x)*weights*jnp.asarray(expected_active)))(model)
            updates,expected_state=optimizer.update(gradient,state,model)
            expected=eqx.apply_updates(model,updates)
            actual,actual_state,_,_,finite=step(model,state,weights,ts,x,jnp.array(False),jnp.asarray(active))
            self.assertTrue(bool(finite))
            for a,b in zip(jax.tree.leaves((actual,actual_state)),jax.tree.leaves((expected,expected_state)),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7)

    def test_causal_physical_loss_matches_numpy_prefix_formula(self):
        with jax.enable_x64(True):
            c,net,data,_=fixture();times=jnp.array([.71,.03,.2]);epsilon=jnp.array(.8)
            ordinary=np.stack([np.asarray(losses.condition_losses(net,t[None],data,c)) for t in times])
            c.loss.use_causal=True
            values,diagnostics=losses.condition_losses(net,times,data,c,epsilon,True)
            order=np.argsort(np.asarray(times));segments=ordinary[order,:2].T
            total=segments.sum(axis=0);weights=np.r_[1.,np.exp(-.8*np.cumsum(total[:-1]))]
            np.testing.assert_allclose(diagnostics[0],weights,rtol=1e-13)
            np.testing.assert_allclose(diagnostics[1],total,rtol=1e-13)
            np.testing.assert_allclose(values[:2],np.mean(segments*weights,axis=1),rtol=1e-13)
            np.testing.assert_allclose(values[2:],ordinary[:,2:].mean(axis=0),rtol=1e-13)
            batch=jax.tree.map(lambda a:jnp.stack((a,a)),data)
            actual,aux=losses.make_loss_components(c)(net,times,batch,epsilon)
            np.testing.assert_allclose(actual,values,rtol=1e-13)
            np.testing.assert_allclose(aux,diagnostics,rtol=1e-13)

    def test_causal_controller_and_complete_adam_update(self):
        c=small_config();c.causal=ConfigDict(dict(step_size=1.1,max_eps=100.,max_last_weight=.9,min_mean_weight=.5))
        net=eqx.nn.Linear(2,2,key=jax.random.key(7));ts=jnp.array([.1,.3,.8]);x=jnp.array([.2,.5])
        optimizer=optax.adam(1e-3);state=optimizer.init(eqx.filter(net,eqx.is_array));weights=jnp.ones(5)
        def vector(model):
            y=model(x)
            return jnp.stack((y[0]**2,y[1]**2,(y.sum()-.3)**2,(y[0]-y[1]+.1)**2,jnp.sum((y+.7)**2)))
        def components(model,times,payload,epsilon):
            return vector(model),jnp.stack((payload,jnp.ones_like(payload)))
        step=strategies.make_causal_step(components,optimizer,c,losses.LOSS_NAMES)
        for temporal,factor in [([1.,.1,.01],1/1.1),([1.,.98,.97],1.1),([1.,.7,.6],1.)]:
            actual,state_actual,w,e,values,aux,finite=step(net,state,weights,jnp.array(1.),ts,jnp.asarray(temporal),jnp.array(False))
            gradient=eqx.filter_grad(lambda model:vector(model).sum())(net)
            updates,state_expected=optimizer.update(gradient,state,net)
            expected=eqx.apply_updates(net,updates)
            self.assertTrue(bool(finite));np.testing.assert_allclose(e,factor,rtol=1e-6)
            for a,b in zip(jax.tree.leaves((actual,state_actual)),jax.tree.leaves((expected,state_expected)),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-7)


if __name__=='__main__':unittest.main()
