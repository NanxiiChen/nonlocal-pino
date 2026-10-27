"""Bounded CH residual norms, exact physical zeros and full parameter gradients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.twophase2d.configs.train_vortex_residual import get_config
from examples.twophase2d.losses import inverse_jacobian_weight, make_loss_fn, make_residual_fn
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.problem import validate_config, identity
from examples.twophase2d.solver import grid
from examples.twophase2d.network import PhaseNetwork, inverse_leaky_tanh
from examples.twophase2d.configs.train_vortex_latent import get_config as latent_config


class SmoothFields(eqx.Module):
    a: jax.Array

    def __call__(self, x, t):
        phi = self.a*jnp.exp(-t)*jnp.sin(2*jnp.pi*x[0])
        return jnp.stack((phi, .3*phi))


class ConstantFields(eqx.Module):
    phase: jax.Array

    def __call__(self, x, t):
        return jnp.stack((self.phase, self.phase**3-self.phase))


class MappedFields(eqx.Module):
    a: jax.Array

    def __call__(self, x, t):
        u=self.a*jnp.exp(-t)*jnp.sin(2*jnp.pi*x[0])
        phi=jnp.tanh(u)+.1*u
        return jnp.stack((phi,.3*phi))

    def phase_jacobian(self, x, t):
        u=self.a*jnp.exp(-t)*jnp.sin(2*jnp.pi*x[0])
        return 1-jnp.tanh(u)**2+.1


class TwoPhaseResidualTests(unittest.TestCase):
    def test_unbounded_map_inverse_implicit_derivative_and_exact_network_ic(self):
        c=latent_config();validate_config(c)
        with jax.enable_x64(True):
            values=jnp.linspace(-1.,1.,401)
            for leak in (.1,1.):
                u=inverse_leaky_tanh(values,leak,8)
                np.testing.assert_allclose(jnp.tanh(u)+leak*u,values,atol=3e-16)
                du=jax.vmap(jax.grad(lambda phi:inverse_leaky_tanh(phi,leak,8)))(values)
                np.testing.assert_allclose(du,1/(1-jnp.tanh(u)**2+leak),atol=3e-14)
            net=PhaseNetwork(c,jax.random.PRNGKey(42));points=jnp.asarray(grid(16,1.))
            predicted=jax.vmap(net,in_axes=(0,None))(points,0.)[:,0]
            np.testing.assert_array_equal(predicted,jax.vmap(net.initial)(points))
            x=jnp.asarray([.7,.4]);t=.37
            physical_dt=jax.grad(lambda tt:net(x,tt)[0])(t)
            latent_dt=jax.grad(lambda tt:(tt/net.duration)*net.raw(x,tt)[0])(t)
            np.testing.assert_allclose(physical_dt,net.phase_jacobian(x,t)*latent_dt,atol=1e-14)
            self.assertGreater(float(jnp.tanh(10.)+.1*10),1.)

    def test_exact_inverse_map_residual_and_full_weight_gradient(self):
        c=latent_config();c.sample.Nx=c.sample.Ny=16;c.nl.delta=.15
        with jax.enable_x64(True):
            points=jnp.asarray(grid(16,1.));D=PeriodicLaplacian(16,1.,.15,dtype=np.float64)
            times=jnp.asarray([.1,.7]);net=MappedFields(jnp.asarray(2.))
            raw,phi=jax.vmap(make_residual_fn(points,D,c),in_axes=(None,0))(net,times)
            jac=jax.vmap(lambda t:jax.vmap(net.phase_jacobian,in_axes=(0,None))(points,t))(times)
            for mode in ('map_transport','map_both'):
                c.residual.mode=mode;loss=make_loss_fn(points,D,c)
                factors=np.ones_like(raw);factors[...,0]=1.1/np.asarray(jac)
                if mode=='map_both':factors[...,1]=1.1/np.asarray(jac)
                expected=np.r_[np.mean((np.asarray(raw)*factors)**2,axis=(0,1)),np.mean(np.asarray(phi).mean(axis=1)**2)]
                actual,parts=loss(net,times,jnp.asarray(0.))
                np.testing.assert_allclose(parts,expected,rtol=2e-14,atol=1e-12)
                f=lambda a:loss(MappedFields(a),times,jnp.asarray(0.))[0]
                finite=(f(net.a+1e-6)-f(net.a-1e-6))/2e-6
                np.testing.assert_allclose(jax.grad(f)(net.a),finite,rtol=2e-6,atol=1e-7)

    def test_weight_range_overshoots_and_inverse_chain_limit(self):
        with jax.enable_x64(True):
            phi=jnp.asarray([-10.,-1.1,-1.,-.8,0.,.8,1.,1.1,10.])
            w=np.asarray(inverse_jacobian_weight(phi,.1))
            self.assertTrue(np.all(w>=1))
            self.assertLessEqual(w.max(),np.sqrt(1.01)/.1)
            self.assertEqual(w[0],1.)
            self.assertEqual(w[-1],1.)
            self.assertEqual(w[4],1.)
            np.testing.assert_allclose(w,w[::-1],atol=1e-14)
            # Independent tanh chain-rule check, away from singular endpoints.
            u=jnp.linspace(-1.,1.,19)
            J=jax.vmap(jax.grad(jnp.tanh))(u)
            np.testing.assert_allclose(inverse_jacobian_weight(jnp.tanh(u),1e-7)*J,1.,atol=3e-14)
            self.assertTrue(np.isfinite(jax.vmap(jax.grad(lambda p:inverse_jacobian_weight(p,.1)))(phi)).all())

    def test_weighted_mixed_norm_matches_numpy_and_full_parameter_gradient(self):
        c=get_config();c.sample.Nx=c.sample.Ny=16;c.nl.delta=.15
        with jax.enable_x64(True):
            points=jnp.asarray(grid(16,1.));D=PeriodicLaplacian(16,1.,.15,dtype=np.float64)
            net=SmoothFields(jnp.asarray(1.3));times=jnp.asarray([.1,.7])
            raw,phi=jax.vmap(make_residual_fn(points,D,c),in_axes=(None,0))(net,times)
            phi=np.asarray(phi);raw=np.asarray(raw)
            w=np.sqrt(1.01)/np.sqrt(.01+np.minimum((1-phi**2)**2,1))
            for mode in ('original','transport','both'):
                c.residual.mode=mode;loss=make_loss_fn(points,D,c)
                actual,parts=loss(net,times,jnp.asarray(0.))
                factors=np.ones_like(raw)
                if mode!='original':factors[...,0]=w
                if mode=='both':factors[...,1]=w
                expected=np.r_[np.mean((raw*factors)**2,axis=(0,1)),np.mean(phi.mean(axis=1)**2)]
                np.testing.assert_allclose(parts,expected,rtol=2e-14,atol=1e-12)
                np.testing.assert_allclose(actual,expected.sum(),rtol=2e-14)
                f=lambda a:loss(SmoothFields(a),times,jnp.asarray(0.))[0]
                derivative=jax.grad(f)(net.a)
                step=1e-6
                finite=(f(net.a+step)-f(net.a-step))/(2*step)
                np.testing.assert_allclose(derivative,finite,rtol=2e-6,atol=1e-7)
            # Mode is an optimization choice, absent from physical reference identity.
            old=identity(c);c.residual.mode='transport';self.assertEqual(old,identity(c))

    def test_physical_constant_zero_and_legacy_original_schema(self):
        c=get_config();c.sample.Nx=c.sample.Ny=16;c.nl.delta=.15
        with jax.enable_x64(True):
            points=jnp.asarray(grid(16,1.));D=PeriodicLaplacian(16,1.,.15,dtype=np.float64)
            for mode in ('original','transport','both'):
                c.residual.mode=mode;validate_config(c)
                for phase in (-1.05,0.,1.05):
                    loss,_=make_loss_fn(points,D,c)(ConstantFields(jnp.asarray(phase)),jnp.asarray([.3]),jnp.asarray(phase))
                    self.assertLess(float(loss),1e-27)
            c.residual.mode='original'
            net=SmoothFields(jnp.asarray(.4))
            extended=make_loss_fn(points,D,c)(net,jnp.asarray([.2]),jnp.asarray(0.))
            legacy=ConfigDict(c.to_dict());del legacy.residual
            original=make_loss_fn(points,D,legacy)(net,jnp.asarray([.2]),jnp.asarray(0.))
            np.testing.assert_array_equal(extended[0],original[0])
            c.residual.eta=0
            with self.assertRaisesRegex(ValueError,'eta'):validate_config(c)


if __name__=='__main__':unittest.main()
