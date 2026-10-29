"""Implicit tanh mass map, independent chain rule, transformed CH and gradients."""
import unittest
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.chebyshev import chebvander,chebder
import jax,jax.numpy as jnp,equinox as eqx
from jax.flatten_util import ravel_pytree
from examples.confined_ch2d.tanh_mapping import mass_shift,tanh_jacobian,phase_state,stable_tanh
from examples.confined_ch2d.configs.train_coarsening_tanh_jacobian import get_config
from examples.confined_ch2d.configs.train_coarsening_global_strong import get_config as original
from examples.confined_ch2d.problem import initial_field,validate_config
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.losses import make_loss_fn
from tests.test_confined_ch import fixture


class TanhTests(unittest.TestCase):
    def test_implicit_map_jacobian_and_all_input_derivatives(self):
        with jax.enable_x64(True):
            u=np.array([-2.1,-.5,.3,1.7]);v=np.array([.1,.2,.4,.3]);mean=.12;a=1.1
            expected=brentq(lambda b:v@(a*np.tanh(u+b))/v.sum()-mean,-10,10,xtol=1e-14)
            shift=mass_shift(jnp.asarray(u),jnp.asarray(v),jnp.asarray(mean),a)
            self.assertAlmostEqual(float(shift),expected,places=13)
            phi=a*np.tanh(u+expected);J=a*(1-np.tanh(u+expected)**2)
            B=np.diag(J)-np.outer(J,v*J)/(v@J)
            fun=lambda z:a*jnp.tanh(z+mass_shift(z,jnp.asarray(v),jnp.asarray(mean),a))
            np.testing.assert_allclose(jax.jacfwd(fun)(jnp.asarray(u)),B,atol=1e-13)
            du=np.array([.2,-.3,.7,.1]);dv=np.array([.01,.02,-.02,.03]);dm=.04
            _,actual=jax.jvp(lambda z,w,m:mass_shift(z,w,m,a),tuple(jnp.asarray(z) for z in (u,v,mean)),tuple(jnp.asarray(z) for z in (du,dv,dm)))
            h=1e-5
            direct=lambda q:brentq(lambda b:(v+q*dv)@(a*np.tanh(u+q*du+b))/(v+q*dv).sum()-(mean+q*dm),-10,10,xtol=1e-14)
            np.testing.assert_allclose(actual,(direct(h)-direct(-h))/(2*h),rtol=1e-8,atol=1e-10)
            np.testing.assert_allclose(tanh_jacobian(jnp.asarray([0.,20.,-20.]),a),[a,4*a*np.exp(-40),4*a*np.exp(-40)],rtol=1e-14)
            for z in (-20.,20.):
                np.testing.assert_allclose(jax.grad(stable_tanh)(jnp.asarray(z)),4*np.exp(-40),rtol=1e-14)
                np.testing.assert_allclose(jax.grad(jax.grad(stable_tanh))(jnp.asarray(z)),-2*np.sign(z)*4*np.exp(-40),rtol=1e-14)

    def test_physical_CH_chain_rule_loss_and_parameter_gradient(self):
        _,x,v=fixture();c=get_config();c.nl.delta=.2;c.arch.rank=4;c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;validate_config(c)
        with jax.enable_x64(True):
            points=jnp.asarray(x);vol=jnp.asarray(v);p=c.physical;a=c.phase.output_map.amplitude
            phi0=initial_field(x,c,xp=np);mean=jnp.dot(vol,jnp.asarray(phi0))/vol.sum();A=native_matrix(x,v,c).toarray();D=RegionalLaplacian(x,v,c)
            key=jax.random.PRNGKey(71);net=_build_network(c,key)
            legacy=original();legacy.arch.hidden_dim=8;legacy.arch.num_layers=2;legacy.arch.spatial_modes=2;legacy.arch.rank=4
            old=_build_network(legacy,key)
            for actual,expected in zip(jax.tree.leaves(eqx.filter(net,eqx.is_array)),jax.tree.leaves(eqx.filter(old,eqx.is_array)),strict=True):np.testing.assert_array_equal(actual,expected)
            np.testing.assert_allclose(predict_fields(net,points,0.,mean,vol,D)[:,0],phi0,atol=1e-14)
            coeff=np.asarray(jax.vmap(lambda point:net.net(net.inputs(point)[0]))(points));rank=c.arch.rank;L=np.log1p(p.terminal_time/c.phase.initial_time_scale)
            root=np.sqrt(v);S=root[:,None]*A/root[None,:];q=root/np.linalg.norm(root);inverse=np.linalg.inv(S+np.outer(q,q))
            times=jnp.asarray([.002,.1,1.5]);expected=[]
            for t in np.r_[0.,times]:
                s=np.log1p(t/c.phase.initial_time_scale)/L;u=2*s-1;basis=chebvander(u,rank-1).ravel();derivative=(chebvander(u,rank-2)@chebder(np.eye(rank),axis=0)).ravel()
                latent=np.arctanh(phi0/a)+s*(coeff@basis);rate=coeff@(basis+2*s*derivative)/((t+c.phase.initial_time_scale)*L)
                offset=brentq(lambda b:v@(a*np.tanh(latent+b))/v.sum()-float(mean),-10,10,xtol=1e-14)
                phi=a*np.tanh(latent+offset);J=a*(1-np.tanh(latent+offset)**2);dt=J*(rate-(v*J)@rate/(v@J));mu=p.beta*(phi**3-phi)+p.epsilon**2*(A@phi)
                actual,actual_dt=jax.jvp(lambda z:predict_fields(net,points,z,mean,vol,D),(jnp.asarray(t),),(jnp.asarray(1.),))
                np.testing.assert_allclose(actual[:,0],phi,atol=2e-14);np.testing.assert_allclose(actual[:,1],mu,atol=2e-13)
                np.testing.assert_allclose(actual_dt[:,0],dt,rtol=2e-11,atol=2e-11)
                np.testing.assert_allclose(v@np.asarray(actual_dt[:,0]),0.,atol=1e-12)
                residual=dt+p.mobility*(A@mu);transformed=residual/J
                latent_residual=rate-(v*J)@rate/(v@J)+p.mobility*(A@mu)/J
                np.testing.assert_allclose(transformed,latent_residual,rtol=2e-12,atol=2e-12)
                transformed_inverse=(inverse@(root*transformed))/root/p.mobility
                expected.append(v@(transformed_inverse**2)/v.sum())
                got,gotJ,_=phase_state(net,points,jnp.asarray(t),mean,vol)
                np.testing.assert_allclose(gotJ,J,rtol=2e-13,atol=2e-14)
                self.assertLess(float(np.max(abs(got))),a)
            loss=make_loss_fn(points,vol,D,None,None,None,c)
            (value,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,times,mean)
            np.testing.assert_allclose(value,np.mean(expected),rtol=2e-10,atol=1e-9);self.assertEqual(parts.shape,(1,))
            parameters,static=eqx.partition(net,eqx.is_inexact_array);theta,unravel=ravel_pytree(parameters);g,_=ravel_pytree(gradient)
            direction=np.random.default_rng(3).normal(size=theta.shape);direction/=np.linalg.norm(direction)
            objective=eqx.filter_jit(lambda vector:loss(eqx.combine(unravel(vector),static),times,mean)[0]);h=1e-5
            fd=(objective(theta+h*direction)-objective(theta-h*direction))/(2*h)
            np.testing.assert_allclose(jnp.dot(g,direction),fd,rtol=3e-5,atol=2e-6)


if __name__=='__main__':unittest.main()
