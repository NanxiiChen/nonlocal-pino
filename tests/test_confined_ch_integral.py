"""Global integral moments, manufactured evolution and physical network gradients."""
import unittest
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from jax.flatten_util import ravel_pytree
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.network import PhaseNetwork,predict_fields
from examples.confined_ch2d.problem import initial_field,validate_config
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.configs.train_coarsening_integral import get_config
from tests.test_confined_ch import fixture


class IntegralTests(unittest.TestCase):
    def test_moments_and_manufactured_evolution_convergence(self):
        errors=[]
        for n in (16,32):
            t=integration_nodes(count=n,duration=1.5,time_scale=.005)
            W=integration_matrix(t,time_scale=.005)
            np.testing.assert_allclose(W@np.ones(len(t)),t,rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(W@t,t*t/2,rtol=1e-10,atol=1e-12)
            rate=100.;truth=np.exp(-rate*t)
            sol=np.linalg.solve(np.eye(n)+rate*W[1:,1:],1-rate*W[1:,0])
            errors.append(np.max(abs(sol-truth[1:])))
        self.assertLess(errors[1],1e-8);self.assertLess(errors[1],errors[0]*1e-3)

    def test_physical_loss_dense_resolvent_and_parameter_gradient(self):
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2
        _,x,v=fixture();c.nl.delta=.2
        full=integration_nodes(count=8,duration=1.5,time_scale=c.loss.phase_metric.time_scale)
        c.sample.temporal.nodes=tuple(full[1:]);c.sample.Nt=8
        c.train.refinement.time_nodes=tuple(full[1:]);c.train.refinement.time_points=8;validate_config(c)
        A=native_matrix(x,v,c).toarray();K=c.physical.mobility*c.physical.epsilon**2*(A@A)
        W=integration_matrix(full,time_scale=c.loss.phase_metric.time_scale)
        with jax.enable_x64(True):
            p=jnp.asarray(x);vol=jnp.asarray(v);ts=jnp.asarray(full[1:]);D=RegionalLaplacian(x,v,c)
            phi0=np.asarray(initial_field(p,c,xp=jnp));mean=jnp.dot(vol,jnp.asarray(phi0))/vol.sum()
            net=PhaseNetwork(c,jax.random.PRNGKey(17))
            values=np.asarray(jax.vmap(lambda t:predict_fields(net,p,t,mean,vol,D))(jnp.asarray(full)))
            mu=c.physical.beta*(values[...,0]**3-values[...,0])+c.physical.epsilon**2*(values[...,0]@A.T)
            residual=values[1:,:,0]-phi0+c.physical.mobility*((W[1:]@mu)@A.T)
            expected=0.
            for i,t in enumerate(full[1:]):
                e=np.linalg.solve(np.eye(len(x))+t*K,residual[i])
                expected+=W[-1,i+1]*np.sum(v*e**2)/v.sum()/c.physical.terminal_time
            loss=make_loss_fn(p,vol,D,None,None,None,c)
            actual,parts=eqx.filter_jit(loss)(net,ts,mean)
            np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-13)
            self.assertEqual(parts.shape,(1,))
            c.loss.phase_metric.normalization='time_average'
            averaged_loss=make_loss_fn(p,vol,D,None,None,None,c)
            expected_average=0.
            for i,t in enumerate(full[1:]):
                e=np.linalg.solve(np.eye(len(x))+t*K,residual[i])/t
                expected_average+=W[-1,i+1]*np.sum(v*e**2)/v.sum()/c.physical.terminal_time
            averaged_value,_=eqx.filter_jit(averaged_loss)(net,ts,mean)
            np.testing.assert_allclose(averaged_value,expected_average,rtol=2e-11,atol=2e-13)
            params,static=eqx.partition(net,eqx.is_inexact_array);vector,unravel=ravel_pytree(params)
            objective=lambda z:loss(eqx.combine(unravel(z),static),ts,mean)[0]
            grad=np.asarray(jax.jit(jax.grad(objective))(vector));step=1e-6
            direction=np.random.default_rng(13).normal(size=len(vector));direction/=np.linalg.norm(direction)
            fd=(float(objective(vector+step*direction))-float(objective(vector-step*direction)))/(2*step)
            np.testing.assert_allclose(grad@direction,fd,rtol=2e-6,atol=2e-7)
            self.assertTrue(np.isfinite(grad).all());self.assertGreater(np.linalg.norm(grad),0.)
        c.train.refinement.time_nodes=tuple(full[1:]*.99)
        with self.assertRaisesRegex(ValueError,'identical'):validate_config(c)


if __name__=='__main__':unittest.main()
