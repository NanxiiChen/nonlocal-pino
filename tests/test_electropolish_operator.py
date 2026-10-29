"""Independent split, derivative and conditioned-physics regression checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from examples.electropolish2d.configs.train_operator_legacy import get_config
from examples.electropolish2d.operator_training import make_loss_vector, predict_fields
from examples.electropolish2d.operator_data import (
    condition_split, sample_condition_indices, training_condition_indices,
    validate_config,
)
from examples.corrosion2d.residual_experiment import nondim


def model(kind):
    return ConditionalFieldOperator(image_shape=(1,4,8), patch_shape=(2,2),
        extent=(1.,.5), width=8, encoder_depth=1, heads=2, mlp_ratio=2,
        decoder_depth=2, spatial_features=3, spatial_scale=1., time_features=2,
        time_scale=1., local_dim=2, out_dim=2, conditioning=kind,
        norm_eps=1e-5, key=jax.random.key(8))


class OperatorTests(unittest.TestCase):
    def test_initial_query_features_match_legacy_saved_config(self):
        c=get_config();legacy=c.copy_and_resolve_references();del legacy.arch['local_input']
        net=model('cross_attention')
        image=jnp.linspace(0,1,32).reshape(1,4,8)
        x=jnp.array([[.2,.1],[.7,.4]]);initial=jnp.array([[.9,.8],[.1,.05]])
        ts=jnp.array([.1,.7])
        np.testing.assert_array_equal(predict_fields(net,image,x,ts,initial,c),
            predict_fields(net,image,x,ts,initial,legacy))

    def test_zero_query_features_preserve_true_initial_supervision(self):
        with jax.enable_x64(True):
            c=get_config();c.arch.local_input='zero';validate_config(c)
            net=model('cross_attention')
            x=jnp.array([[.2,.1],[.7,.4],[.3,.2],[.4,.3],[.5,.5],[.5,0.]])
            image=jnp.linspace(0,1,32).reshape(1,1,4,8)
            known=jnp.array([[[.9,.8],[.1,.05],[.7,.6],[.2,.1],[0.,0.],[1.,1.]]])
            ts=jnp.array([.2,.8])
            sub=(jnp.array([0,1]),jnp.array([1,0]),jnp.array([-2.,-2.]),jnp.array([2.,2.]))
            mask=jnp.array([True,True])
            fn=make_loss_vector(c,nondim(c),2,1)
            a=fn(net,ts,(image,known,x),sub,mask)
            b=fn(net,ts,(image,jnp.zeros_like(known),x),sub,mask)
            np.testing.assert_allclose(a[:4],b[:4],rtol=0,atol=0)
            predicted=predict_fields(net,image[0],x,jnp.zeros(1),known[0],c)[0]
            expected=jnp.mean(jnp.sum((predicted[2:4]-known[0,2:4])**2,-1))
            np.testing.assert_allclose(a[4],expected,rtol=1e-13,atol=1e-13)
            self.assertGreater(abs(float(a[4]-b[4])),1e-4)
            np.testing.assert_array_equal(predict_fields(net,image[0],x,ts,known[0],c),
                predict_fields(net,image[0],x,ts,jnp.zeros_like(known[0]),c))
            c.arch.local_input='invalid'
            with self.assertRaisesRegex(ValueError,'local_input'):validate_config(c)

    def test_split_pairs_time_jvp_and_condition_response(self):
        with jax.enable_x64(True):
            for kind in ('cross_attention', 'film'):
                net=model(kind)
                image=jnp.linspace(0,1,32).reshape(1,4,8)
                x=jnp.array([[.2,.1],[.7,.4],[.4,.2]])
                local=jnp.array([[.9,.8],[.1,.05],[.7,.6]])
                ts=jnp.array([.1,.7])
                split=eqx.filter_jit(net)(image,x,ts,local)
                xx=jnp.tile(x,(len(ts),1));tt=jnp.repeat(ts,len(x));ll=jnp.tile(local,(len(ts),1))
                paired=eqx.filter_jit(net.paired)(image,xx,tt,ll).reshape(split.shape)
                np.testing.assert_allclose(split,paired,rtol=1e-12,atol=1e-12)
                f=lambda t:net(image,x,t,local)
                _,dt=jax.jvp(f,(ts,),(jnp.ones_like(ts),))
                h=1e-5
                finite=(f(ts+h)-f(ts-h))/(2*h)
                np.testing.assert_allclose(dt,finite,rtol=2e-6,atol=1e-9)
                # There must be no derivative with respect to another query time.
                jac=jax.jacfwd(f)(ts)
                np.testing.assert_allclose(jac[0,:,:,1],0,atol=1e-14)
                np.testing.assert_allclose(jac[1,:,:,0],0,atol=1e-14)
                change=np.linalg.norm(np.asarray(split-net(1-image,x,ts,local)))
                self.assertGreater(change,1e-5)
                # Frozen coordinate frequencies must not drift under parameter AD.
                grad=eqx.filter_grad(lambda n:jnp.sum(n(image,x,ts,local)**2))(net)
                np.testing.assert_array_equal(grad.spatial_frequencies,0)

    def test_physical_residual_and_parameter_gradient(self):
        with jax.enable_x64(True):
            c=get_config();validate_config(c)
            net=model('cross_attention')
            x=jnp.array([[.2,.1],[.7,.4],[.3,.2],[.4,.3],[.5,.5],[.5,0.]])
            image=jnp.linspace(0,1,32).reshape(1,1,4,8)
            known=jnp.array([[[.9,.8],[.1,.05],[.7,.6],[.2,.1],[0.,0.],[1.,1.]]])
            times=jnp.array([.2,.8])
            row=jnp.array([0,1]);col=jnp.array([1,0]);vals=jnp.array([-2.,-2.]);diag=jnp.array([2.,2.])
            sub=(row,col,vals,diag);mask=jnp.array([True,True])
            loss=make_loss_vector(c,nondim(c),2,1)
            actual=loss(net,times,(image,known,x),sub,mask)
            h=1e-5
            def fields(t):
                raw=net(image[0],x,t,known[0]);phi=.5*(1+jnp.tanh(raw[...,0]))
                conc=c.physical.c_le*raw[...,1]+(c.physical.c_se-c.physical.c_le)*(3*phi**2-2*phi**3)
                return jnp.stack([phi,conc],-1)
            f=np.asarray(fields(times));dt=np.asarray((fields(times+h)-fields(times-h))/(2*h))
            phi=f[:,:2,0].T;conc=f[:,:2,1].T
            A=np.array([[2.,-2.],[-2.,2.]])
            nch,n1,n2,n3,dc,cle=nondim(c)
            q=conc-(3*phi**2-2*phi**3)*dc-cle
            ac=dt[:,:2,0].T+n3*(A@phi)+n2*(4*phi**3-6*phi**2+2*phi)-n1*q*dc*(6*phi-6*phi**2)
            ch=dt[:,:2,1].T+nch*(A@q)
            ic=np.asarray(fields(jnp.zeros(1)))[0,2:4]
            expected=np.array([np.mean(abs(ac)),np.mean(abs(ch)),
                np.mean(np.sum(f[:,4:5]**2,-1)),np.mean(np.sum((f[:,5:6]-1)**2,-1)),
                np.mean(np.sum((ic-np.asarray(known[0,2:4]))**2,-1))])
            np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=3e-8)
            theta,unravel=ravel_pytree(eqx.filter(net,eqx.is_inexact_array))
            _,static=eqx.partition(net,eqx.is_inexact_array)
            objective=eqx.filter_jit(lambda v:jnp.sum(loss(eqx.combine(unravel(v),static),times,(image,known,x),sub,mask)))
            direction=np.random.default_rng(2).normal(size=theta.shape)
            tangent=unravel(jnp.asarray(direction))
            tangent=eqx.tree_at(lambda n:n.spatial_frequencies,tangent,jnp.zeros_like(net.spatial_frequencies))
            direction=np.array(ravel_pytree(tangent)[0]);direction/=np.linalg.norm(direction)
            finite=(objective(theta+h*direction)-objective(theta-h*direction))/(2*h)
            ad=jnp.dot(jax.grad(objective)(theta),direction)
            np.testing.assert_allclose(ad,finite,rtol=2e-5,atol=2e-7)

    def test_training_subset_exclusion_and_legacy_random_schedule(self):
        from examples.electropolish2d.configs.train_operator_single import get_config as single_config
        c=get_config();eligible=training_condition_indices(c)
        np.testing.assert_array_equal(eligible,np.arange(16))
        for i in range(32):
            key=jax.random.fold_in(jax.random.key(42),i)
            expected=jax.random.choice(key,16,(2,),replace=False)
            np.testing.assert_array_equal(sample_condition_indices(key,eligible,2),expected)
        c=single_config();validate_config(c)
        eligible=training_condition_indices(c)
        np.testing.assert_array_equal(eligible,[0])
        for i in range(8):
            actual=sample_condition_indices(jax.random.key(i),eligible,1)
            np.testing.assert_array_equal(actual,[0])
        self.assertEqual(condition_split(0,16,eligible),'train')
        self.assertEqual(condition_split(1,16,eligible),'unused_train')
        self.assertEqual(condition_split(16,16,eligible),'test')
        for subset in ((),(0,0),(16,),(-1,),(True,),(0,False),(0.,)):
            broken=single_config();broken.operator.training_indices=subset
            with self.subTest(subset=subset), self.assertRaisesRegex(ValueError,'training_indices'):
                validate_config(broken)
        c.operator.condition_batch=2
        with self.assertRaisesRegex(ValueError,'batch'):
            validate_config(c)

    def test_invalid_condition_settings(self):
        c=get_config();c.operator.condition_batch=c.operator.train_conditions+1
        with self.assertRaisesRegex(ValueError,'batch'):
            validate_config(c)
        n=model('cross_attention')
        with self.assertRaisesRegex(ValueError,'Condition shape'):
            n.encode(jnp.zeros((1,3,8)))
        with self.assertRaisesRegex(ValueError,'aligned'):
            n.prepare(jnp.zeros((1,4,8)),jnp.zeros((2,2)),jnp.zeros((2,1)))


if __name__=='__main__':
    unittest.main()
