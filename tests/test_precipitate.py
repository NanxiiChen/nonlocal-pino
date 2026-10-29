"""Independent energy, equilibrium and coupled-residual checks."""
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.precipitate2d.configs.train_base import get_config
from examples.precipitate2d.problem import grid, initial_field, validate_config
from examples.precipitate2d.mechanics import ElasticEquilibrium
from examples.precipitate2d.network import PhaseNetwork, predict_fields
from examples.precipitate2d.losses import make_loss_fn
from examples.precipitate2d.solver import coefficients, elastic_fields, solve_reference
from examples.twophase2d.operators import PeriodicLaplacian


class PrecipitateTests(unittest.TestCase):
    def setUp(self):
        self.context = jax.enable_x64(True)
        self.context.__enter__()
        self.c = get_config()
        self.c.sample.Nx = self.c.sample.Ny = 15
        self.c.nl.delta = .15
        self.c.reference.snapshots = 3

    def tearDown(self):self.context.__exit__(None,None,None)

    def test_energy_derivative_independent_stress_equilibrium_and_translation_gauge(self):
        c = self.c
        x = grid(15,1.)
        p = -.7+.2*np.cos(2*np.pi*x[:,0])+.15*np.sin(2*np.pi*(x[:,0]+x[:,1]))
        mech = ElasticEquilibrium(c)
        chemical,u,stress = map(np.asarray,mech(jnp.asarray(p)))
        ref = elastic_fields(p.reshape(15,15),c,coefficients(c,15,'local_spectral'))
        for a,b in zip((chemical,u,stress),ref[:3],strict=True):
            np.testing.assert_allclose(a,b.reshape(a.shape),rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(mech.energy(jnp.asarray(p)),ref[3],atol=1e-14)
        gradient = jax.grad(mech.energy)(jnp.asarray(p))*len(p)
        np.testing.assert_allclose(gradient,chemical,atol=1e-13)
        sx,sy,ss = np.moveaxis(stress.reshape(15,15,3),-1,0)
        kx,ky = np.asarray(mech.kx),np.asarray(mech.ky)
        force = np.stack((np.fft.ifft2(1j*(kx*np.fft.fft2(sx)+ky*np.fft.fft2(ss))).real,
                          np.fft.ifft2(1j*(kx*np.fft.fft2(ss)+ky*np.fft.fft2(sy))).real),axis=-1)
        self.assertLess(np.max(np.abs(force)),1e-12)
        np.testing.assert_allclose(u.mean(axis=0),0,atol=1e-16)
        self.assertGreater(float(mech.energy(jnp.asarray(p))),0)

    def test_nonlocal_symbol_and_discrete_total_energy_gradient(self):
        c = self.c
        D = PeriodicLaplacian(15,1.,.15,dtype=np.float64)
        L,*_ = coefficients(c,15,'nonlocal')
        p = jnp.asarray(initial_field(grid(15,1.),c,xp=np))
        actual = D@p
        expected = np.fft.ifft2(L*np.fft.fft2(np.asarray(p).reshape(15,15))).real.ravel()
        np.testing.assert_allclose(actual,expected,atol=2e-12)
        mech = ElasticEquilibrium(c)
        def energy(phi):
            return jnp.mean(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*(D@phi))+mech.energy(phi)
        q = c.physical.beta*(p**3-p)+c.physical.epsilon**2*(D@p)+mech(p)[0]
        np.testing.assert_allclose(jax.grad(energy)(p)*len(p),q,atol=1e-12)
        self.assertLess(float(-c.physical.mobility*jnp.mean(q*(D@q))),0)

    def test_initial_mass_periodicity_and_mixed_gradient(self):
        c = self.c
        c.arch.hidden_dim = 8
        c.arch.num_layers = 2
        x = jnp.asarray(grid(15,1.))
        p0 = initial_field(x,c,xp=jnp)
        net = PhaseNetwork(c,jax.random.PRNGKey(42))
        mean = jnp.mean(p0)
        np.testing.assert_allclose(predict_fields(net,x,0.,mean)[:,0],p0,atol=1e-14)
        np.testing.assert_allclose(jax.vmap(net,in_axes=(0,None))(x+.0,.4),jax.vmap(net,in_axes=(0,None))(x+jnp.array([1.,0.]),.4),atol=1e-13)
        self.assertLess(abs(float(predict_fields(net,x,.5,mean)[:,0].mean()-mean)),1e-14)
        D = PeriodicLaplacian(15,1.,.15,dtype=np.float64)
        loss = make_loss_fn(x,D,ElasticEquilibrium(c),c)
        (value,parts),g = eqx.filter_value_and_grad(loss,has_aux=True)(net,jnp.asarray([.2,.8]),mean)
        self.assertTrue(np.isfinite(float(value)))
        self.assertGreater(float(jnp.linalg.norm(jnp.stack(parts))),0)
        self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(eqx.filter(g,eqx.is_array))))

    def test_reference_constant_state_mass_energy_and_early_validation(self):
        c = self.c
        validate_config(c)
        with patch('examples.precipitate2d.solver.initial_field',side_effect=lambda points,config,xp:np.full(len(points),-.4)):
            for method in ('nonlocal','local_spectral'):
                constant = solve_reference(c,15,method,.001)
                np.testing.assert_allclose(constant['phi'],-.4,atol=1e-14)
                np.testing.assert_allclose(constant['u'],0,atol=1e-15)
                np.testing.assert_allclose(constant['stress'],np.broadcast_to(constant['stress'][0:1],constant['stress'].shape),atol=1e-12)
        d = solve_reference(c,15,'nonlocal',.0005)
        np.testing.assert_allclose(d['phi'].mean(axis=(1,2)),d['phi'][0].mean(),atol=1e-13)
        self.assertLess(np.max(np.diff(d['energy'])),1e-9)
        c.sample.Nx=c.sample.Ny=16
        with self.assertRaisesRegex(ValueError,'odd'):validate_config(c)

    def test_mixed_local_ad_residual_matches_manufactured_laplacians(self):
        class Fields(eqx.Module):
            amplitude: jax.Array
            def __call__(self,x,t):
                p = self.amplitude*jnp.exp(-t)*jnp.sin(2*jnp.pi*x[0])
                q = .3*jnp.cos(4*jnp.pi*x[1])
                return jnp.stack((p,q))
        c = self.c
        c.nl.method = 'local_ad'
        validate_config(c)
        x = jnp.asarray(grid(15,1.))
        net = Fields(jnp.asarray(.2))
        mechanics = ElasticEquilibrium(c)
        loss = make_loss_fn(x,None,mechanics,c)
        times = jnp.asarray([.2,.7])
        def independent(model):
            def at(t):
                p,q = predict_fields(model,x,t,jnp.asarray(-.3)).T
                # The mean projection contributes -0.3 to phase, and zero
                # spatial/time derivatives for this zero-mean sine field.
                original_p = p+.3
                rp = -original_p+c.physical.mobility*(4*jnp.pi)**2*q
                rq = q-c.physical.beta*(p**3-p)-c.physical.epsilon**2*(2*jnp.pi)**2*original_p-mechanics(p)[0]
                return jnp.stack((jnp.mean(rp**2),jnp.mean(rq**2)))
            return jnp.mean(jax.vmap(at)(times),axis=0)
        (actual,parts),grad = eqx.filter_value_and_grad(loss,has_aux=True)(net,times,jnp.asarray(-.3))
        expected = independent(net)
        np.testing.assert_allclose(parts,expected,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(actual,jnp.sum(expected),rtol=1e-12)
        exact_grad = eqx.filter_grad(lambda model:jnp.sum(independent(model)))(net)
        np.testing.assert_allclose(grad.amplitude,exact_grad.amplitude,rtol=1e-12,atol=1e-13)
        spectral = c.copy_and_resolve_references()
        spectral.nl.method = 'local_spectral'
        (spectral_value,spectral_parts),spectral_grad = eqx.filter_value_and_grad(
            make_loss_fn(x,None,mechanics,spectral),has_aux=True)(net,times,jnp.asarray(-.3))
        np.testing.assert_allclose(spectral_parts,parts,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(spectral_value,actual,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(spectral_grad.amplitude,grad.amplitude,rtol=1e-12,atol=1e-13)


if __name__=='__main__':unittest.main()
