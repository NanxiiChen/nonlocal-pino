"""Independent mixed residual/scaling, field coupling and global IC/mass checks."""
import unittest
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from examples.confined_ch2d.configs.train_coarsening_global_mixed import get_config
from examples.confined_ch2d.problem import initial_field,validate_config
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from tests.test_confined_ch import fixture


class MixedTests(unittest.TestCase):
    def test_both_layouts_against_dense_residual_and_directional_derivative(self):
        _,x,v=fixture()
        for layout in ('shared','separate'):
            with self.subTest(layout=layout),jax.enable_x64(True):
                c=get_config();c.arch.field_layout=layout
                c.arch.rank=4;c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;c.nl.delta=.2
                validate_config(c);A=native_matrix(x,v,c).toarray()
                vol=jnp.asarray(v);points=jnp.asarray(x);D=RegionalLaplacian(x,v,c)
                phi0=initial_field(points,c,xp=jnp);mean=jnp.dot(vol,phi0)/vol.sum()
                net=_build_network(c,jax.random.PRNGKey(13))
                fields=lambda model,t:predict_fields(model,points,t,mean,vol,D)
                np.testing.assert_allclose(fields(net,0.)[:,0],phi0,atol=1e-14)
                q=np.sqrt(v/v.sum());S=np.sqrt(v[:,None])*A/np.sqrt(v[None,:])
                inverse=np.linalg.inv(S+np.outer(q,q))
                times=jnp.asarray([.002,.1,1.5]);expected=[]
                for t in np.r_[0.,times]:
                    f,dt=jax.jvp(lambda z:fields(net,z),(jnp.asarray(t),),(jnp.asarray(1.),))
                    np.testing.assert_allclose(np.dot(v,f[:,0])/v.sum(),mean,atol=1e-14)
                    np.testing.assert_allclose(np.dot(v,dt[:,0]),0.,atol=1e-12)
                    phi,mu=np.asarray(f).T
                    evolution=(np.asarray(dt[:,0])+c.physical.mobility*(A@mu))
                    scaled=(inverse@(np.sqrt(v)*evolution))/np.sqrt(v)/c.physical.mobility/c.loss.chemical_scale
                    chemistry=(mu-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*(A@phi))/c.loss.chemical_scale
                    expected.append([np.dot(v,scaled**2)/v.sum(),np.dot(v,chemistry**2)/v.sum()])
                loss=make_loss_fn(points,vol,D,None,None,None,c)
                (value,parts),grad=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,times,mean)
                np.testing.assert_allclose(parts,np.mean(expected,axis=0),rtol=2e-10,atol=1e-10)
                np.testing.assert_allclose(value,np.sum(np.mean(expected,axis=0)),rtol=2e-10)
                parameters,static=eqx.partition(net,eqx.is_inexact_array);theta,unravel=ravel_pytree(parameters)
                direction=np.random.default_rng(1).normal(size=theta.shape);direction/=np.linalg.norm(direction)
                objective=eqx.filter_jit(lambda a:loss(eqx.combine(unravel(a),static),times,mean)[0])
                h=1e-5;fd=(objective(theta+h*direction)-objective(theta-h*direction))/(2*h)
                gflat,_=ravel_pytree(eqx.filter(grad,eqx.is_inexact_array))
                np.testing.assert_allclose(jnp.dot(gflat,direction),fd,rtol=2e-5,atol=1e-6)
                if layout=='separate':
                    for module in (grad.net,grad.chemical_net):
                        self.assertGreater(sum(float(jnp.sum(a*a)) for a in jax.tree.leaves(module) if eqx.is_array(a)),0.)


if __name__=='__main__':unittest.main()
