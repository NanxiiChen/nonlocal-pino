"""Independent local CH jets, constant spatial mass shift, and update checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.confined_ch2d.configs.train_operator_compare_local_ad import get_config
from examples.confined_ch2d.operator_local_ad import (
    initial_jets, spatial_jets, chunked_jets, one_time, condition_parts, make_loss, make_step)
from examples.confined_ch2d.operator_problem import fixed_initial, condition


class Polynomial(eqx.Module):
    a: jax.Array

    def encode(self, features, weights):
        return features

    def prepare_encoded(self, encoded, points):
        return points

    def decode(self, points, times):
        x,y=points.T
        return self.a*(x**4+x*x*y*y+2*y**4)[None,:]+times[:,None]*(x**3*y)[None,:]


def config():
    c=get_config(); c.train.seed=42; c.train.compute_dtype='float64'
    c.operator.local_ad.query_chunk=2
    c.geometry.boundary_outer=4; c.geometry.boundary_per_hole=2
    return c


def data(c):
    x=jnp.array([[.23,.32],[.4,.27],[.54,.66],[.73,.42]])
    v=jnp.array([.12,.2,.17,.11])
    boundary,weights=condition(np.array([[.4,.5,.07],[.65,.5,.06]]),c)
    # Nontrivial constant-preserving weighted-self-adjoint positive metric.
    metric=.6*jnp.eye(4)+.4*jnp.broadcast_to(v/v.sum(),(4,4))
    return dict(points=x,volumes=v,initial=fixed_initial(x,c,xp=jnp),
        condition=jnp.asarray(boundary),condition_weights=jnp.asarray(weights),metric=metric)


class ConfinedOperatorADTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=jax.config.jax_enable_x64; jax.config.update('jax_enable_x64',True)

    @classmethod
    def tearDownClass(cls):
        jax.config.update('jax_enable_x64',cls.old)

    def test_known_initial_fourier_jets(self):
        c=config(); x=data(c)['points']; actual=initial_jets(x,c)
        f=lambda q:fixed_initial(q[None],c,xp=jnp)[0]
        lap=lambda q:jnp.trace(jax.hessian(f)(q))
        np.testing.assert_allclose(actual[0],jax.vmap(f)(x),rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(actual[1],jax.vmap(jax.grad(f))(x),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(actual[2],jax.vmap(lap)(x),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(actual[3],jax.vmap(lambda q:jnp.trace(jax.hessian(lap)(q)))(x),rtol=2e-12,atol=1e-9)
        np.testing.assert_allclose(actual[4],jax.vmap(jax.grad(lap))(x),rtol=2e-12,atol=1e-11)

    def test_fourth_order_polynomial_and_chunks(self):
        c=config(); p=data(c)['points']; a=.7; t=.4; net=Polynomial(jnp.asarray(a))
        x,y=np.asarray(p).T; tau=t/c.physical.terminal_time
        fun=lambda q:net.decode(q,jnp.array([tau]))[0]
        expected_g=np.stack((a*(4*x**3+2*x*y*y)+3*tau*x*x*y,
                             a*(2*x*x*y+8*y**3)+tau*x**3),axis=-1)
        expected_l=a*(14*x*x+26*y*y)+6*tau*x*y
        expected_b=np.full(len(p),80*a)
        for values in (spatial_jets(fun,p,True),chunked_jets(net,None,p,jnp.asarray(t),c,True)):
            for actual,expected in zip(values,(expected_g,expected_l,expected_b),strict=True):
                np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)

    def test_eliminated_ch_against_direct_nested_physical_ad(self):
        c=config(); d=data(c); net=Polynomial(jnp.asarray(.45)); t=jnp.asarray(.65)
        prepared=d['points']; actual=one_time(net,d,None,prepared,t,c)
        v=d['volumes']; points=d['points']; T=c.physical.terminal_time

        def phi(point,time):
            s=time/T
            mean=jnp.dot(v,net.decode(points,jnp.reshape(s,(1,)))[0])/v.sum()
            return fixed_initial(point[None],c,xp=jnp)[0]+s*(net.decode(point[None],jnp.reshape(s,(1,)))[0,0]-mean)

        def mu(point,time):
            value=phi(point,time)
            return c.physical.beta*(value**3-value)-c.physical.epsilon**2*jnp.trace(jax.hessian(lambda q:phi(q,time))(point))

        dt=jax.vmap(lambda q:jax.grad(lambda tt:phi(q,tt))(t))(points)
        lm=jax.vmap(lambda q:jnp.trace(jax.hessian(lambda xx:mu(xx,t))(q)))(points)
        res=T*(dt-c.physical.mobility*lm); filtered=d['metric']@res
        b=d['condition'][:,:2]; n=d['condition'][:,2:4]; w=d['condition_weights']
        pn=jnp.sum(jax.vmap(jax.grad(lambda q:phi(q,t)))(b)*n,axis=1)
        mn=jnp.sum(jax.vmap(jax.grad(lambda q:mu(q,t)))(b)*n,axis=1)
        expected=jnp.array([v@(filtered**2)/v.sum(),w@(pn**2)/w.sum(),w@(mn**2)/w.sum()])
        np.testing.assert_allclose(actual,expected,rtol=3e-11,atol=3e-10)
        self.assertAlmostEqual(float(v@dt),0.,places=13)
        vals=jax.vmap(lambda q:phi(q,t))(points)
        self.assertAlmostEqual(float(v@(vals-d['initial'])),0.,places=13)
        # Taking a tangent of all integration points would differentiate the
        # mass mean in space. The above independent query formulation does not.

    def test_condition_accumulation_matches_full_objective_and_one_update(self):
        c=config(); c.loss.initial_residual_fraction=.2
        first=data(c); second=dict(first,metric=jnp.eye(4))
        batch=jax.tree.map(lambda a,b:jnp.stack((a,b)),first,second)
        times=jnp.array([0.,.45,1.7]); net=Polynomial(jnp.asarray(.4))
        loss=make_loss(c)
        expected,gradient=eqx.filter_value_and_grad(loss)(net,batch,times)
        optimizer=optax.sgd(1e-6); state=optimizer.init(eqx.filter(net,eqx.is_array))
        updated,_,value,finite,parts=make_step(c,optimizer)(net,state,batch,times)
        self.assertTrue(bool(finite))
        np.testing.assert_allclose(value,expected,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(updated.a,net.a-1e-6*gradient.a,rtol=2e-12,atol=2e-12)
        allparts=jax.vmap(lambda t:one_time(net,first,None,first['points'],t,c))(times)
        p=condition_parts(net,first,times,c)
        np.testing.assert_allclose(p[0],.2*allparts[0,0]+.8*jnp.mean(allparts[1:,0]),rtol=2e-12)
        np.testing.assert_allclose(p[1:],jnp.mean(allparts[1:,1:],axis=0),rtol=2e-12)
        self.assertEqual(parts.shape,(3,))
        c.loss.initial_residual_fraction=0.
        positive=condition_parts(net,first,times,c)
        np.testing.assert_allclose(positive,jnp.mean(allparts[1:],axis=0),rtol=2e-12)
        zero_loss=make_loss(c)
        full_value,full_grad=eqx.filter_value_and_grad(zero_loss)(net,batch,times)
        update,_,val,ok,_=make_step(c,optimizer)(net,state,batch,times)
        self.assertTrue(bool(ok))
        np.testing.assert_allclose(val,full_value,rtol=2e-12)
        np.testing.assert_allclose(update.a,net.a-1e-6*full_grad.a,rtol=2e-12)

    def test_real_two_stream_operator_fourth_derivative_and_parameter_backward(self):
        from examples.confined_ch2d.operator_training import build_network,encode
        from examples.confined_ch2d.operator_local_ad import raw
        c=config();c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        d=data(c);net=build_network(c,jax.random.key(11));encoded=encode(net,d)
        points=d['points'][:2];t=jnp.asarray(.5)
        fun=lambda x:raw(net,encoded,x,t,c)
        g,l,b=spatial_jets(fun,points,True)
        for i,x in enumerate(points):
            scalar=lambda q:fun(q[None])[0]
            np.testing.assert_allclose(g[i],jax.grad(scalar)(x),rtol=2e-10,atol=2e-10)
            np.testing.assert_allclose(l[i],jnp.trace(jax.hessian(scalar)(x)),rtol=2e-9,atol=2e-9)
        h=1e-4;lap=lambda x:spatial_jets(fun,x,False)[1]
        finite=sum((lap(points+h*e)-2*l+lap(points-h*e))/h**2 for e in jnp.eye(2))
        np.testing.assert_allclose(b,finite,rtol=2e-3,atol=2e-3)
        # Full parameter backward includes encoder/context, wall third jets and
        # the parameter-dependent hard mass shift, not only output parameters.
        times=jnp.array([0.,.3]);batch=jax.tree.map(lambda x:x[None],d)
        value,gradient=eqx.filter_value_and_grad(make_loss(c))(net,batch,times)
        leaves=[np.asarray(a) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(np.isfinite(a).all() for a in leaves))
        self.assertGreater(sum(float(np.sum(a*a)) for a in leaves),0)


if __name__=='__main__':
    unittest.main()
