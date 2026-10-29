"""Exact chemical constraint, independent scalar CH equation and gradients."""
import unittest
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from examples.confined_ch2d.configs.train_coarsening_eliminated import get_config
from examples.confined_ch2d.problem import initial_field,validate_config,boundary_samples
from examples.confined_ch2d.network import PhaseNetwork,predict_fields
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.losses import make_loss_fn
from tests.test_confined_ch import fixture


class EliminationTests(unittest.TestCase):
    def test_constraint_independent_equation_and_parameter_gradient(self):
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;c.nl.delta=.2
        # Test the unpreconditioned equation against independent NumPy assembly.
        del c.loss.phase_metric;validate_config(c);_,x,v=fixture()
        A=native_matrix(x,v,c);b,n,bw,_=boundary_samples(c)
        with jax.enable_x64(True):
            D=RegionalLaplacian(x,v,c);p=jnp.asarray(x);vol=jnp.asarray(v)
            mean=jnp.dot(vol,initial_field(p,c,xp=jnp))/vol.sum();net=PhaseNetwork(c,jax.random.PRNGKey(8))
            fun=lambda t:predict_fields(net,p,t,mean,vol,D)
            self.assertEqual(net(p[0],.2).shape,(1,))
            with self.assertRaises(ValueError):predict_fields(net,p,.2,mean,vol)
            value,dt=jax.jvp(fun,(jnp.asarray(.2),),(jnp.asarray(1.),));phi=np.asarray(value[:,0])
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(A@phi)
            np.testing.assert_allclose(value[:,1],mu,atol=2e-13)
            np.testing.assert_allclose(dt,(fun(.2000001)-fun(.1999999))/2e-7,atol=2e-6,rtol=2e-6)
            expected=np.sum(v*(np.asarray(dt[:,0])+c.physical.mobility*(A@mu))**2)/v.sum()
            loss=make_loss_fn(p,vol,D,jnp.asarray(b),jnp.asarray(n),jnp.asarray(bw),c)
            (actual,parts),g=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,jnp.asarray([.2]),mean)
            self.assertEqual(parts.shape,(1,));np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-10)
            leaves=jax.tree.leaves(eqx.filter(g,eqx.is_array))
            self.assertTrue(all(np.isfinite(a).all() for a in leaves));self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)
            c.nl.method='local_ad'
            with self.assertRaises(ValueError):validate_config(c)


if __name__=='__main__':unittest.main()
