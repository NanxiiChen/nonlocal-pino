"""Check positive region metrics, lagged derivatives and stagger head isolation."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.phase_metric import region_weights
from examples.channel_nsch2d.partition import stage_at,mask_output_updates
from examples.channel_nsch2d.configs.train_stream_regions import get_config as regions
from examples.channel_nsch2d.configs.train_stream_stagger import get_config as stagger
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.losses import residual_components,make_loss_fn
from tests.test_channel_nsch import small
from nonlocal_pinn.training.optimizers import make_optimizer


class Update(eqx.Module):
    output: object
    trunk: object


class PhaseControls(unittest.TestCase):
    def test_region_metric_independent_budget_and_stopped_weights(self):
        with jax.enable_x64(True):
            c=regions();validate_config(c);s=c.phase_metric
            p=np.array([[-1.,0.,1.,.5]]);v=np.array([1.,2.,3.,4.])
            a=np.stack(((1-p)/2,(1+p)/2,1-p**2),axis=-1)
            f=(a*v[None,:,None]).sum(axis=1)/v.sum()
            expected=.5+.5*np.mean(a/np.maximum(f[:,None,:],.001),axis=-1)
            actual=region_weights(jnp.asarray(p),jnp.asarray(v),s)
            np.testing.assert_allclose(actual,expected,atol=1e-14)
            np.testing.assert_allclose(np.sum(np.asarray(actual)*v)/v.sum(),1.,atol=1e-14)
            self.assertTrue(np.all(np.asarray(actual)>=.5))
            jac=jax.jacfwd(lambda phi:region_weights(phi,jnp.asarray(v),s))(jnp.asarray(p))
            np.testing.assert_array_equal(jac,0.)
            pure=np.asarray(region_weights(jnp.ones_like(actual)*2,jnp.asarray(v),s))
            np.testing.assert_allclose(pure,2/3,atol=1e-14)

    def test_schedule_and_inactive_soap_update_rows(self):
        c=stagger();validate_config(c)
        stages=[stage_at(i,c.stagger)[0] for i in range(c.train.iterations)]
        self.assertEqual({k:stages.count(k) for k in ('phase','flow','joint')},
                         dict(phase=2000,flow=2000,joint=1000))
        self.assertEqual(stages[24:26],['phase','flow'])
        u=Update(eqx.nn.Linear(3,4,key=jax.random.PRNGKey(4)),jnp.array([3.,4.]))
        out=mask_output_updates(u,jnp.array([1.,0.]))
        np.testing.assert_array_equal(out.output.weight[2:],0.)
        np.testing.assert_array_equal(out.output.bias[2:],0.)
        np.testing.assert_array_equal(out.output.weight[:2],u.output.weight[:2])
        np.testing.assert_array_equal(out.trunk,u.trunk)
        c.stagger.alternating_steps=4001
        with self.assertRaisesRegex(ValueError,'cycles'):validate_config(c)

    def test_lagging_preserves_values_and_blocks_only_cross_physics(self):
        with jax.enable_x64(True):
            c=stagger();x=jnp.array([.3,.7,.2,.4,.6]);g=jnp.array([.9,.8,.1,.5,.2])
            z=jnp.zeros_like(x)
            fn=lambda a,b,mask:residual_components(a,z,b,g,z,c,mask)
            expected=fn(x,g,None)
            for active in (jnp.array([1.,0.]),jnp.array([0.,1.]),jnp.ones(2)):
                np.testing.assert_allclose(fn(x,g,active),expected,atol=0,rtol=0)
            phase_jac=jax.jacfwd(lambda a:fn(a,g,jnp.array([1.,0.])))(x)
            self.assertEqual(float(phase_jac[0,2]),0.)
            self.assertEqual(float(phase_jac[0,3]),0.)
            flow_jac=jax.jacfwd(lambda a:fn(a,g,jnp.array([0.,1.])))(x)
            self.assertEqual(float(flow_jac[2,0]),0.)
            flow_mu_jac=jax.jacfwd(lambda b:fn(x,b,jnp.array([0.,1.])))(g)
            self.assertEqual(float(flow_mu_jac[2,1]),0.)

    def test_mask_blocks_real_soap_momentum_after_switch(self):
        c=stagger();model=Update(eqx.nn.Linear(3,4,key=jax.random.PRNGKey(14)),jnp.array([3.,4.]))
        optimizer=make_optimizer(c);state=optimizer.init(eqx.filter(model,eqx.is_array))
        fn=lambda net: jnp.sum(net.output.weight[:2]**2)+jnp.sum(net.trunk**2)
        # Upstream SOAP's first update initializes its preconditioner and emits
        # zero updates. Build actual phase momentum before changing the block.
        for _ in range(3):
            gradient=eqx.filter_grad(fn)(model)
            updates,state=optimizer.update(gradient,state,model)
            model=eqx.apply_updates(model,mask_output_updates(updates,jnp.array([1.,0.])))
        fn=lambda net: jnp.sum(net.output.weight[2:]**2)+jnp.sum(net.trunk**2)
        updates,_=optimizer.update(eqx.filter_grad(fn)(model),state,model)
        self.assertGreater(float(jnp.linalg.norm(updates.output.weight[:2])),0.)
        masked=mask_output_updates(updates,jnp.array([0.,1.]))
        np.testing.assert_array_equal(masked.output.weight[:2],0.)
        np.testing.assert_array_equal(masked.output.bias[:2],0.)

    def test_full_pde_optional_metrics_and_joint_identity(self):
        with jax.enable_x64(True):
            c,data,net=small();batch=jax.tree.map(lambda a:a[None],data);t=jnp.array([.3,1.])
            original=eqx.filter_jit(make_loss_fn(c));value,parts=original(net,batch,t)
            c.stagger=stagger().stagger
            fn=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))
            for mask in (jnp.array([1.,0.]),jnp.array([0.,1.]),jnp.ones(2)):
                (total,p),gradient=fn(net,batch,t,mask)
                np.testing.assert_allclose(p,parts,atol=1e-10)
                if bool(jnp.all(mask)):np.testing.assert_allclose(total,value,atol=1e-10)
                self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(gradient) if eqx.is_array(a)))
            del c.stagger;c.phase_metric=regions().phase_metric
            (total,_),gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(net,batch,t)
            self.assertTrue(np.isfinite(float(total)))
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(gradient) if eqx.is_array(a)))


if __name__=='__main__':unittest.main()
