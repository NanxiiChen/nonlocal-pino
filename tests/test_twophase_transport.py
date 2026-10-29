"""Periodic moments/AD mixed residuals and independent reference consistency."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.twophase2d.configs.train_base import get_config
from examples.twophase2d.configs.train_vortex import get_config as vortex_config
from examples.twophase2d.losses import make_loss_fn
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.operators import PeriodicLaplacian
from examples.twophase2d.problem import (initial_field, validate_config, identity,
                                       load_reference, velocity_field)
from examples.twophase2d.solver import grid, symbols, solve_reference


class SmoothFields(eqx.Module):
    a: jax.Array
    epsilon: float = eqx.field(static=True)

    def __call__(self, x, t):
        phi = self.a*jnp.exp(-t)*jnp.sin(2*jnp.pi*x[0])
        q = phi**3-phi+self.epsilon**2*(2*jnp.pi)**2*phi
        return jnp.stack((phi, q))


class TwoPhaseTransportTests(unittest.TestCase):
    def test_vortex_divergence_periodicity_reversal_and_mixed_advection(self):
        c = vortex_config()
        validate_config(c)
        with jax.enable_x64(True):
            points = jnp.asarray(grid(16, 1.))
            t = .3
            flow = lambda x, time: velocity_field(x, time, c, xp=jnp)
            jacobian = jax.vmap(jax.jacfwd(flow, argnums=0), in_axes=(0, None))(points, t)
            np.testing.assert_allclose(jnp.trace(jacobian, axis1=-2, axis2=-1), 0, atol=2e-15)
            np.testing.assert_allclose(flow(points, t), -flow(points, 2-t), atol=1e-15)
            np.testing.assert_allclose(flow(points, 1.), 0, atol=1e-15)
            np.testing.assert_allclose(flow(points+jnp.asarray([1., 0.]), t), flow(points, t), atol=1e-15)
            model = SmoothFields(jnp.asarray(.4), c.physical.epsilon)
            amplitude = .4*np.exp(-t)
            angle = 2*np.pi*np.asarray(points[:, 0]); k = 2*np.pi
            phi = amplitude*np.sin(angle)
            a1 = .75*amplitude**3-amplitude+c.physical.epsilon**2*k*k*amplitude
            a3 = -.25*amplitude**3
            x, y = np.asarray(points).T
            u = -2*np.sin(np.pi*x)**2*np.sin(np.pi*y)*np.cos(np.pi*y)*np.cos(np.pi*t/2)
            expected = -phi+u*k*amplitude*np.cos(angle)+c.physical.mobility*k*k*(a1*np.sin(angle)+9*a3*np.sin(3*angle))
            _, parts = make_loss_fn(points, None, c)(model, jnp.asarray([t]), jnp.asarray(0.))
            np.testing.assert_allclose(parts[0], np.mean(expected**2), atol=1e-12, rtol=0)
            self.assertLess(float(parts[1]), 1e-25)

    def test_vortex_reference_flux_preserves_constant_and_mass(self):
        c = vortex_config(); c.reference.snapshots = 3
        with self.assertRaisesRegex(ValueError, 'actual grid size'):
            symbols(16, c, 'nonlocal')
        c.nl.delta = .15
        with patch('examples.twophase2d.solver.initial_field', side_effect=lambda x, config, xp:np.full(len(x), .3)):
            for method in ('local_ad', 'nonlocal'):
                result = solve_reference(c, 16, method, .01)
                np.testing.assert_allclose(result['phi'], .3, atol=1e-14)
        result = solve_reference(c, 32, 'local_ad', .002)
        means = result['phi'].mean(axis=(1, 2))
        np.testing.assert_allclose(means, means[0], atol=1e-13)

    def test_periodic_operator_scipy_symbol_nullspace_energy_and_transpose(self):
        c = get_config()
        with jax.enable_x64(True):
            n = c.sample.Nx
            x = grid(n, c.geometry.length)
            D = PeriodicLaplacian(n, c.geometry.length, c.nl.delta, dtype=np.float64)
            lam, _ = symbols(n, c, 'nonlocal')
            np.testing.assert_array_equal(D@jnp.ones(n*n), 0)
            field = np.sin(2*np.pi*x[:, 0])+0.2*np.cos(4*np.pi*x[:, 1])
            actual = np.asarray(D@jnp.asarray(field))
            expected = np.fft.ifftn(lam*np.fft.fftn(field.reshape(n, n))).real.ravel()
            np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
            self.assertLess(abs(actual.sum()), 1e-10)
            self.assertGreater(field@actual, 0)
            _, pullback = jax.vjp(lambda a:D@a, jnp.asarray(field))
            np.testing.assert_allclose(pullback(jnp.asarray(field))[0], actual, atol=1e-11)

    def test_periodic_network_exact_initial_values_and_differentiable_parameters(self):
        c = get_config()
        net = PhaseNetwork(c, jax.random.PRNGKey(42))
        points = jnp.asarray([[.1,.2],[.9,.75]], dtype=jnp.float32)
        initial = jax.vmap(net, in_axes=(0,None))(points, jnp.float32(0))[:,0]
        np.testing.assert_array_equal(initial, initial_field(points,c,xp=jnp))
        for displacement in (jnp.array([1.,0.]), jnp.array([0.,1.])):
            np.testing.assert_allclose(jax.vmap(net,in_axes=(0,None))(points+displacement,.4),
                                       jax.vmap(net,in_axes=(0,None))(points,.4), atol=3e-6)
        loss, grad = eqx.filter_value_and_grad(lambda model:jnp.sum(model(points[0],.4)**2))(net)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(eqx.filter(grad,eqx.is_array))))

    def test_local_mixed_residual_matches_analytic_derivatives(self):
        c = get_config()
        with jax.enable_x64(True):
            points = jnp.asarray(grid(16,1.))
            model = SmoothFields(jnp.asarray(.4),c.physical.epsilon)
            loss = make_loss_fn(points,None,c)
            t=.3
            _, parts = loss(model,jnp.asarray([t]),jnp.asarray(0.))
            amplitude=.4*np.exp(-t);angle=2*np.pi*np.asarray(points[:,0]);k=2*np.pi
            phi=amplitude*np.sin(angle)
            a1=.75*amplitude**3-amplitude+c.physical.epsilon**2*k*k*amplitude
            a3=-.25*amplitude**3
            expected=-phi+c.physical.velocity[0]*k*amplitude*np.cos(angle)+c.physical.mobility*k*k*(a1*np.sin(angle)+9*a3*np.sin(3*angle))
            np.testing.assert_allclose(parts[0],np.mean(expected**2),atol=1e-12,rtol=0)
            self.assertLess(float(parts[1]),1e-25)
            self.assertLess(float(parts[2]),1e-25)
            self.assertEqual(parts.shape,(3,))
            shifted_value,shifted_parts = loss(model,jnp.asarray([t]),jnp.asarray(.25))
            np.testing.assert_allclose(shifted_parts[:2],parts[:2],atol=1e-13)
            np.testing.assert_allclose(shifted_parts[2],.25**2,atol=1e-13)
            np.testing.assert_allclose(shifted_value,jnp.dot(shifted_parts,jnp.asarray([
                c.loss.phase_weight,c.loss.chemical_weight,c.loss.mass_weight])),atol=1e-13)
            D=PeriodicLaplacian(16,1.,c.nl.delta,dtype=np.float64)
            _, nlparts=make_loss_fn(points,D,c)(model,jnp.asarray([t]),jnp.asarray(0.))
            symbol,_=symbols(16,c,'nonlocal')
            expected_nl=-phi+c.physical.velocity[0]*k*amplitude*np.cos(angle)+c.physical.mobility*(symbol[0,1]*a1*np.sin(angle)+symbol[0,3]*a3*np.sin(3*angle))
            chemical=c.physical.epsilon**2*(k*k-symbol[0,1])*phi
            np.testing.assert_allclose(nlparts[0],np.mean(expected_nl**2),atol=1e-12,rtol=0)
            np.testing.assert_allclose(nlparts[1],np.mean(chemical**2),atol=1e-12,rtol=0)

    def test_reference_rejects_shuffled_nodes_before_training(self):
        c=get_config();c.sample.Nx=c.sample.Ny=8
        with TemporaryDirectory() as directory:
            c.reference.path=str(Path(directory)/'reference.npz')
            shape=(c.reference.snapshots,8,8)
            data=dict(identity=json.dumps(identity(c)),points=grid(8,1.),
                time=np.linspace(0,1,c.reference.snapshots),
                **{key:np.ones(shape) for key in ['phi_local','phi_nonlocal','q_local','q_nonlocal']})
            np.savez(c.reference.path,**data)
            load_reference(c)
            data['points']=data['points'][::-1]
            np.savez(c.reference.path,**data)
            with self.assertRaisesRegex(ValueError,'ordering'):load_reference(c)

    def test_independent_reference_preserves_constant_and_configuration_fails_early(self):
        c=get_config();validate_config(c)
        c.reference.snapshots=3
        with patch('examples.twophase2d.solver.initial_field',side_effect=lambda x,config,xp:np.full(len(x),.3)):
            for method in ('local_ad','nonlocal'):
                result=solve_reference(c,16,method,.01)
                np.testing.assert_allclose(result['phi'],.3,atol=1e-14)
                np.testing.assert_allclose(result['q'],.3**3-.3,atol=1e-14)
        c.nl.delta=0.6
        with self.assertRaisesRegex(ValueError,'delta'):validate_config(c)


if __name__=='__main__':unittest.main()
