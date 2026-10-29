"""Independent causal/core normalization, stopped AD, and step parity checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d.residual_experiment import make_step, nondim
from examples.electropolish2d.configs.train_operator_causal import get_config
from examples.electropolish2d.operator_data import validate_config
from examples.electropolish2d.operator_training import (
    _make_causal_step, make_loss_components, make_loss_vector,
)
from nonlocal_pinn.training.optimizers import make_optimizer
from tests.test_electropolish_operator import model


def fixture():
    net = model('cross_attention')
    c = get_config()
    x = jnp.array([[.2,.1],[.7,.4],[.3,.2],[.4,.3],[.5,.2],[.5,.5],[.5,0.]])
    images = jnp.linspace(0,1,64).reshape(2,1,4,8)
    local = jnp.array([[[.9,.8],[.1,.05],[.7,.6],[.2,.1],[.5,.4],[0.,0.],[1.,1.]],
                       [[.7,.6],[.2,.1],[.5,.4],[.4,.3],[.8,.7],[0.,0.],[1.,1.]]])
    row = jnp.array([0,0,2,2]); col = jnp.array([1,2,0,1])
    sub = row, col, jnp.array([-2.,-1.,-1.,-3.]), jnp.array([3.,0.,4.])
    mask = jnp.array([True,False,True]); times = jnp.array([.8,.01,.3])
    return net, c, times, (images,local,x), sub, mask


class OperatorCausalTests(unittest.TestCase):
    def test_numpy_formula_mask_sort_condition_independence_and_stopped_gradient(self):
        with jax.enable_x64(True):
            net,c,ts,payload,sub,mask = fixture()
            fn = make_loss_components(c,nondim(c),2,1)
            eps = jnp.array([.5,.7])
            parts,aux = fn(net,ts,(*payload,eps),sub,mask)
            nch,n1,n2,n3,dc,cle = nondim(c)
            A = np.array([[3.,-2.,-1.],[0.,0.,0.],[-1.,-3.,4.]])
            expected_parts=[]; expected_aux=[]
            for image,known in zip(payload[0],payload[1]):
                def physical(t):
                    raw=net(image,payload[2],t,known)
                    p=.5*(1+jnp.tanh(raw[...,0]))
                    conc=cle*raw[...,1]+dc*(3*p**2-2*p**3)
                    return jnp.stack([p,conc],-1)
                f=np.asarray(physical(ts)); h=1e-5
                dt=np.asarray((physical(ts+h)-physical(ts-h))/(2*h))
                p=f[:,:3,0].T; conc=f[:,:3,1].T
                q=conc-dc*(3*p**2-2*p**3)-cle
                ac=dt[:,:3,0].T+n3*(A@p)+n2*(4*p**3-6*p**2+2*p)-n1*q*dc*(6*p-6*p**2)
                ch=dt[:,:3,1].T+nch*(A@q)
                segments=np.array([abs(ac)[[0,2]].mean(0),abs(ch)[[0,2]].mean(0)])[:,np.argsort(ts)]
                weights=np.exp(-.5*np.c_[np.zeros(2),np.cumsum(segments[:,:-1],axis=1)])
                expected_aux.append(np.r_[weights,segments])
                ic=np.asarray(physical(jnp.zeros(1)))[0,3:5]
                expected_parts.append([*(weights*segments).mean(1),
                    np.mean(np.sum(f[:,5:6]**2,-1)),np.mean(np.sum((f[:,6:7]-1)**2,-1)),
                    np.mean(np.sum((ic-np.asarray(known[3:5]))**2,-1))])
            np.testing.assert_allclose(parts,np.mean(expected_parts,0),rtol=3e-7,atol=3e-8)
            np.testing.assert_allclose(aux,expected_aux,rtol=3e-7,atol=3e-8)
            np.testing.assert_array_equal(aux[:,:2,0],1.)
            self.assertFalse(np.array_equal(aux[0,0],aux[1,0]))
            # Zero eps recovers the original physical objective (no normalization by sum w).
            plain=c.copy_and_resolve_references();plain.loss.use_causal=False
            baseline=make_loss_vector(plain,nondim(c),2,1)(net,ts,payload,sub,mask)
            zero=fn(net,ts,(*payload,jnp.zeros(2)),sub,mask)[0]
            np.testing.assert_allclose(zero,baseline,rtol=1e-13,atol=1e-13)
            # Independently freeze weights at theta0; this detects accidental weight AD.
            frozen=jax.lax.stop_gradient(aux[:,:2])
            def varying(theta):
                m=eqx.tree_at(lambda n:n.output.bias,net,net.output.bias+jnp.array([theta,0.]))
                return fn(m,ts,(*payload,eps),sub,mask)
            actual=jax.grad(lambda a:varying(a)[0][:2].sum())(0.)
            expected=jax.grad(lambda a:jnp.mean(jnp.sum(jnp.mean(frozen*varying(a)[1][:,2:],axis=-1),axis=-1)))(0.)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            jac=jax.jacfwd(lambda a:varying(a)[1][:,:2])(0.)
            np.testing.assert_array_equal(jac,0.)

    def test_aux_step_matches_existing_gradient_balance_and_optimizer(self):
        with jax.enable_x64(True):
            net,c,ts,payload,sub,mask=fixture()
            eps=jnp.array([.5,.7]); weights=jnp.ones(5)
            opt=make_optimizer(c);state=opt.init(eqx.filter(net,eqx.is_array))
            standard=make_step(make_loss_vector(c,nondim(c),2,1),opt,c)
            causal=_make_causal_step(make_loss_components(c,nondim(c),2,1),opt,c)
            a=standard(net,state,weights,ts,(*payload,eps),sub,mask,jnp.array(True),jnp.ones(5,dtype=bool))
            b=causal(net,state,weights,eps,ts,payload,sub,mask,jnp.array(True))
            for left,right in zip(jax.tree.leaves((a[0],a[1],a[2],a[3])),jax.tree.leaves((b[0],b[1],b[2],b[4]))):
                if eqx.is_array(left):
                    np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-12)
                else:
                    self.assertEqual(left,right)
            self.assertTrue(bool(a[4]) and bool(b[6]))
            mean_weights=np.asarray(b[5])[:,:2].mean(0)
            expected=np.asarray(eps).copy()
            for i,w in enumerate(mean_weights):
                if w[-1]>c.causal.max_last_weight and expected[i]<c.causal.max_eps:
                    expected[i]*=c.causal.step_size
                if w.mean()<c.causal.min_mean_weight:expected[i]/=c.causal.step_size
            expected=np.clip(expected,0,c.causal.max_eps)
            np.testing.assert_allclose(b[3],expected,rtol=1e-13)

    def test_invalid_causal_settings(self):
        validate_config(get_config())
        for key,value in [('n_chunks',4),('n_chunks',True),('n_chunks',0.),('eps_init',0.),
                          ('eps_init',float('nan')),('max_eps',.5),('step_size',1.),
                          ('max_last_weight',1.),('min_mean_weight',0.)]:
            c=get_config()
            with c.ignore_type():
                c.causal[key]=value
            with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'causal'):
                validate_config(c)


if __name__=='__main__':unittest.main()
