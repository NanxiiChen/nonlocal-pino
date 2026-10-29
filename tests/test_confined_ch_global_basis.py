"""Continuous global basis, IC/mass, unaliased chemical integral and gradients."""
import unittest
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebvander
import jax,jax.numpy as jnp,equinox as eqx
from examples.confined_ch2d.configs.train_coarsening_global_basis import get_config
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.problem import initial_field,validate_config
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from tests.test_confined_ch import fixture


class GlobalBasisTests(unittest.TestCase):
    def test_continuous_fields_quadrature_and_training_gradient(self):
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;c.arch.rank=4
        _,x,v=fixture();c.nl.delta=.2
        nodes=integration_nodes(count=16,duration=1.5,time_scale=.005)
        c.sample.Nt=16;c.sample.temporal.nodes=tuple(nodes[1:])
        c.train.refinement.time_points=16;c.train.refinement.time_nodes=tuple(nodes[1:]);validate_config(c)
        A=native_matrix(x,v,c).toarray();W=integration_matrix(nodes,time_scale=.005)
        with jax.enable_x64(True):
            p=jnp.asarray(x);vol=jnp.asarray(v);D=RegionalLaplacian(x,v,c)
            phi0=initial_field(p,c,xp=jnp);mean=jnp.dot(vol,phi0)/vol.sum()
            net=_build_network(c,jax.random.PRNGKey(19));fields=lambda t:predict_fields(net,p,t,mean,vol,D)
            np.testing.assert_allclose(fields(0.)[:,0],phi0,atol=1e-14)
            for t in (0.,.003,.13,1.5):
                actual,dt=jax.jvp(fields,(jnp.asarray(t),),(jnp.asarray(1.),))
                np.testing.assert_allclose(dt,(fields(t+1e-7)-fields(t-1e-7))/2e-7,rtol=2e-6,atol=2e-6)
                np.testing.assert_allclose(np.dot(v,actual[:,0])/v.sum(),mean,atol=1e-14)
            values=np.asarray(jax.vmap(fields)(jnp.asarray(nodes)))
            coefficients=np.asarray(jax.vmap(lambda point:net.net(net.inputs(point)[0]))(p))
            q,w=leggauss(50);L=np.log1p(1.5/.005)
            for i in (5,len(nodes)-1):
                end=np.log1p(nodes[i]/.005)/L;s=.5*end*(q+1)
                ph=np.asarray(phi0)[None,:]+s[:,None]*(chebvander(2*s-1,c.arch.rank-1)@coefficients.T)
                ph+=float(mean)-(ph@v/v.sum())[:,None]
                mu=c.physical.beta*(ph**3-ph)+c.physical.epsilon**2*(ph@A.T)
                direct=(w*.5*end*L*.005*np.exp(L*s))@mu
                np.testing.assert_allclose(W[i]@values[...,1],direct,rtol=2e-10,atol=2e-12)
            loss=make_loss_fn(p,vol,D,None,None,None,c)
            (value,parts),g=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,jnp.asarray(nodes[1:]),mean)
            self.assertTrue(np.isfinite(float(value)));self.assertEqual(parts.shape,(1,))
            leaves=jax.tree.leaves(eqx.filter(g,eqx.is_array))
            self.assertTrue(all(np.isfinite(a).all() for a in leaves));self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)


if __name__=='__main__':unittest.main()
