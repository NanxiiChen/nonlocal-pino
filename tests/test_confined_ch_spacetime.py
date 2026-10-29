"""Independent dense block solve and gradients for the global residual metric."""
import unittest
import numpy as np
import jax,jax.numpy as jnp
from examples.confined_ch2d.preconditioning import SpaceTimeMetric
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from tests.test_confined_ch import fixture


class SpaceTimeTests(unittest.TestCase):
    def test_block_inverse_and_parameter_gradient(self):
        c,x,v=fixture();A=native_matrix(x,v,c).toarray()
        K=c.physical.mobility*c.physical.epsilon**2*(A@A)
        times=np.array([.003,.017,.12,.6,1.]);h=np.diff(np.r_[0.,times]);n=len(v)
        rng=np.random.default_rng(8);r=rng.normal(size=(len(times),n))
        B=np.zeros((len(times)*n,len(times)*n))
        for i,dt in enumerate(h):
            sl=slice(i*n,(i+1)*n);B[sl,sl]=np.eye(n)+dt*K
            if i:B[sl,(i-1)*n:i*n]=-np.eye(n)
        expected=np.linalg.solve(B,(h[:,None]*r).ravel()).reshape(r.shape)
        with jax.enable_x64(True):
            metric=SpaceTimeMetric(RegionalLaplacian(x,v,c),v,mobility=c.physical.mobility,epsilon=c.physical.epsilon)
            actual=metric(jnp.asarray(r),jnp.asarray(times))
            np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-13)
            np.testing.assert_allclose(metric(jnp.zeros_like(actual),jnp.asarray(times)),0,atol=1e-15)
            objective=lambda scale:jnp.sum(jnp.asarray(h[:,None]*v)*metric(jnp.asarray(r)*scale,jnp.asarray(times))**2)/v.sum()
            correct=1.4*np.sum(h[:,None]*v*expected**2)/v.sum()
            np.testing.assert_allclose(jax.jit(jax.grad(objective))(.7),correct,rtol=2e-12)
            # Invert the recurrence to recover every supplied residual.
            previous=np.vstack((np.zeros(n),np.asarray(actual)[:-1]))
            recovered=(np.asarray(actual)-previous)/h[:,None]+np.asarray(actual)@K.T
            np.testing.assert_allclose(recovered,r,rtol=3e-12,atol=3e-12)

    def test_global_physical_loss_and_gradient(self):
        import equinox as eqx
        from jax.flatten_util import ravel_pytree
        from examples.confined_ch2d.configs.train_coarsening_global import get_config
        from examples.confined_ch2d.network import PhaseNetwork,predict_fields
        from examples.confined_ch2d.problem import initial_field,validate_config
        from examples.confined_ch2d.losses import make_loss_fn
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2
        _,x,v=fixture();c.nl.delta=.2;validate_config(c)
        A=native_matrix(x,v,c).toarray();K=c.physical.mobility*c.physical.epsilon**2*(A@A)
        ts=np.array([.006,.10,.4,.9]);nodes=np.r_[ts,c.physical.terminal_time]
        h=np.diff(np.r_[0.,nodes]);epsilon=1e-6
        with jax.enable_x64(True):
            points=jnp.asarray(x);volumes=jnp.asarray(v)
            mean=jnp.dot(volumes,initial_field(points,c,xp=jnp))/volumes.sum()
            net=PhaseNetwork(c,jax.random.PRNGKey(6));D=RegionalLaplacian(x,v,c)
            physical=lambda model,t:predict_fields(model,points,jnp.asarray(t),mean,volumes,D)
            residuals=[]
            for t in nodes:
                phi=np.asarray(physical(net,t))[:,0]
                dt=(np.asarray(physical(net,t+epsilon))[:,0]-np.asarray(physical(net,t-epsilon))[:,0])/(2*epsilon)
                mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(A@phi)
                residuals.append(dt+c.physical.mobility*A@mu)
            error=np.zeros(len(x));expected=0.
            for dt,r in zip(h,residuals,strict=True):
                error=np.linalg.solve(np.eye(len(x))+dt*K,error+dt*r)
                expected+=dt*np.sum(v*error**2)/v.sum()/c.physical.terminal_time
            loss=make_loss_fn(points,volumes,D,None,None,None,c)
            value,parts=eqx.filter_jit(loss)(net,jnp.asarray(ts),mean)
            np.testing.assert_allclose(value,expected,atol=2e-7,rtol=2e-7)
            self.assertEqual(parts.shape,(1,))
            params,static=eqx.partition(net,eqx.is_inexact_array);vector,unravel=ravel_pytree(params)
            f=lambda z:loss(eqx.combine(unravel(z),static),jnp.asarray(ts),mean)[0]
            g=np.asarray(jax.jit(jax.grad(f))(vector))
            direction=np.random.default_rng(11).normal(size=len(vector));direction/=np.linalg.norm(direction)
            fd=(float(f(vector+epsilon*direction))-float(f(vector-epsilon*direction)))/(2*epsilon)
            np.testing.assert_allclose(g@direction,fd,atol=2e-7,rtol=2e-6)
            self.assertTrue(np.isfinite(g).all());self.assertGreater(np.linalg.norm(g),0.)
        c.train.refinement.time_nodes=tuple(np.linspace(.1,c.physical.terminal_time,c.train.refinement.time_points))
        with self.assertRaisesRegex(ValueError,'strictly below T'):validate_config(c)


if __name__=='__main__':unittest.main()
