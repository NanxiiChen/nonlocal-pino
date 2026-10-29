"""General IC lifting: compatibility, derivative paths and legacy arithmetic."""
import unittest
import jax,jax.numpy as jnp,equinox as eqx
import numpy as np
from examples.confined_ch2d.configs.train_coarsening_blend import get_config
from examples.confined_ch2d.network import PhaseNetwork,predict_fields
from examples.confined_ch2d.problem import initial_field,validate_config,boundary_samples
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.losses import make_loss_fn
from tests.test_confined_ch import fixture


class BlendTests(unittest.TestCase):
    def test_logarithmic_increment_initial_feature_and_derivative(self):
        c=get_config();c.phase.lifting='logarithmic_increment';c.phase.initial_time_scale=.01
        c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.initial_feature=True;validate_config(c)
        with jax.enable_x64(True):
            net=PhaseNetwork(c,jax.random.PRNGKey(8));x=jnp.asarray([.52,.6])
            np.testing.assert_allclose(net(x,0.)[0],initial_field(x,c,xp=jnp),atol=1e-14)
            for t in (.003,.1,1.5):
                _,dt=jax.jvp(lambda tt:net(x,tt),(jnp.asarray(t),),(jnp.asarray(1.),))
                np.testing.assert_allclose(dt,(net(x,t+1e-7)-net(x,t-1e-7))/2e-7,atol=2e-6,rtol=2e-6)

    def test_logarithmic_blend_chain_rule(self):
        c=get_config();c.phase.lifting='logarithmic_blend';c.phase.initial_time_scale=.01
        c.arch.hidden_dim=8;c.arch.num_layers=2;validate_config(c)
        with jax.enable_x64(True):
            net=PhaseNetwork(c,jax.random.PRNGKey(8));x=jnp.asarray([.52,.6])
            np.testing.assert_allclose(net(x,0.)[0],initial_field(x,c,xp=jnp),atol=1e-14)
            for t in (.003,.1,1.5):
                _,dt=jax.jvp(lambda tt:net(x,tt),(jnp.asarray(t),),(jnp.asarray(1.),))
                np.testing.assert_allclose(dt,(net(x,t+1e-7)-net(x,t-1e-7))/2e-7,atol=2e-6,rtol=2e-6)

    def test_log_time_coordinate_chain_rule_and_mass(self):
        c=get_config();c.phase.lifting='logarithmic_additive';c.phase.initial_time_scale=.005
        c.arch.hidden_dim=8;c.arch.num_layers=2;validate_config(c)
        _,x,v=fixture()
        with jax.enable_x64(True):
            net=PhaseNetwork(c,jax.random.PRNGKey(7));points=jnp.asarray(x);vol=jnp.asarray(v)
            p0=initial_field(points,c,xp=jnp);mean=jnp.dot(vol,p0)/vol.sum()
            fun=lambda t:predict_fields(net,points,t,mean,vol)
            np.testing.assert_allclose(fun(0.)[:,0],p0,atol=1e-14)
            for t in (.003,.1,1.5):
                value,dt=jax.jvp(fun,(jnp.asarray(t),),(jnp.asarray(1.),))
                np.testing.assert_allclose(dt,(fun(t+1e-7)-fun(t-1e-7))/2e-7,atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(jnp.dot(vol,value[:,0])/vol.sum(),mean,atol=1e-14)
                self.assertLess(abs(float(jnp.dot(vol,dt[:,0]))),1e-12)

    def test_hard_ic_mass_time_and_spatial_derivatives(self):
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;validate_config(c)
        _,x,v=fixture()
        with jax.enable_x64(True):
            net=PhaseNetwork(c,jax.random.PRNGKey(5));p=jnp.asarray(x);v=jnp.asarray(v)
            phi0=initial_field(p,c,xp=jnp);mean=jnp.dot(v,phi0)/v.sum()
            fun=lambda t:predict_fields(net,p,t,mean,v)
            np.testing.assert_allclose(fun(0.)[:,0],phi0,atol=1e-14)
            for t in (.03,.37,1.5):
                value,dt=jax.jvp(fun,(jnp.asarray(t),),(jnp.asarray(1.),))
                np.testing.assert_allclose(jnp.dot(v,value[:,0])/v.sum(),mean,atol=1e-14)
                self.assertLess(abs(float(jnp.dot(v,dt[:,0]))),1e-13)
                np.testing.assert_allclose(dt,(fun(t+1e-6)-fun(t-1e-6))/2e-6,atol=2e-7,rtol=2e-6)
            point=jnp.asarray([.51,.61]);t=.31;step=1e-5
            h=jax.jacfwd(jax.jacrev(net,argnums=0),argnums=0)(point,t)
            fd=sum((net(point+jnp.eye(2)[i]*step,t)-2*net(point,t)+net(point-jnp.eye(2)[i]*step,t))/step**2 for i in range(2))
            np.testing.assert_allclose(jnp.trace(h,axis1=-2,axis2=-1),fd,atol=3e-4,rtol=5e-5)

    def test_validation_and_compiled_coupled_gradient(self):
        c=get_config();c.phase.initial_time_scale=0.
        with self.assertRaises(ValueError):validate_config(c)
        c=get_config();c.phase.lifting='invalid'
        with self.assertRaises(ValueError):validate_config(c)
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;c.nl.delta=.2
        _,x,v=fixture();b,n,bw,_=boundary_samples(c)
        with jax.enable_x64(True):
            points=jnp.asarray(x);vol=jnp.asarray(v);net=PhaseNetwork(c,jax.random.PRNGKey(2))
            mean=jnp.dot(vol,initial_field(points,c,xp=jnp))/vol.sum()
            D=RegionalLaplacian(x,v,c)
            loss=make_loss_fn(points,vol,D,jnp.asarray(b),jnp.asarray(n),jnp.asarray(bw),c)
            f=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))
            (value,parts),g=f(net,jnp.asarray([.02,.5]),mean)
            self.assertTrue(np.isfinite(float(value)));self.assertEqual(parts.shape,(2,))
            self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(eqx.filter(g,eqx.is_array))))


if __name__=='__main__':unittest.main()
