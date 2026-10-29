"""Global mixed inverse versus independently assembled full space-time block."""
import unittest
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from jax.flatten_util import ravel_pytree
from examples.confined_ch2d.configs.train_coarsening_mixed_block import get_config
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.mixed_preconditioning import MixedBlockMetric
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.problem import initial_field,validate_config
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.losses import make_loss_fn


class MixedBlockTests(unittest.TestCase):
    def test_full_block_inverse_and_actual_nonlinear_loss_gradient(self):
        c=get_config();c.arch.rank=4;c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;c.nl.delta=.2
        x=np.array([[.44,.26],[.48,.26],[.52,.26],[.44,.30],[.48,.30],[.52,.30]])
        v=np.linspace(.002,.004,len(x));nodes=integration_nodes(count=16,duration=1.5,time_scale=c.phase.initial_time_scale)
        c.sample.Nt=16;c.sample.temporal.nodes=tuple(nodes[1:]);c.train.refinement.time_points=16;c.train.refinement.time_nodes=tuple(nodes[1:]);validate_config(c)
        W=integration_matrix(nodes,time_scale=c.phase.initial_time_scale);A=native_matrix(x,v,c).toarray()
        N=len(nodes)*len(x);I=np.eye(N)
        L=np.block([[I,c.physical.mobility*np.kron(W,A)],[-c.physical.epsilon**2*np.kron(np.eye(len(nodes)),A),I]])
        with jax.enable_x64(True):
            D=RegionalLaplacian(x,v,c);metric=MixedBlockMetric(D,v,W,mobility=c.physical.mobility,epsilon=c.physical.epsilon)
            errors=np.random.default_rng(2).normal(size=2*N);residual=L@errors
            actual=metric(jnp.asarray(residual[:N].reshape(len(nodes),len(x))),jnp.asarray(residual[N:].reshape(len(nodes),len(x))))
            np.testing.assert_allclose(np.concatenate([np.ravel(a) for a in actual]),errors,rtol=2e-11,atol=2e-11)
            points=jnp.asarray(x);vol=jnp.asarray(v);phi0=initial_field(points,c,xp=jnp);mean=jnp.dot(vol,phi0)/vol.sum()
            net=_build_network(c,jax.random.PRNGKey(7));fields=np.asarray(jax.vmap(lambda t:predict_fields(net,points,t,mean,vol,D))(jnp.asarray(nodes)))
            phi,mu=fields[...,0],fields[...,1]
            R=phi-phi0+c.physical.mobility*(W@mu)@A.T
            C=mu-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*phi@A.T
            dense=np.linalg.solve(L,np.r_[np.ravel(R),np.ravel(C)]).reshape(2,len(nodes),len(x))
            expected=np.mean(np.sum(v*dense**2,axis=-1)/v.sum(),axis=-1)/np.array([1.,c.loss.chemical_scale**2])
            loss=make_loss_fn(points,vol,D,None,None,None,c)
            (value,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,jnp.asarray(nodes[1:]),mean)
            np.testing.assert_allclose(parts,expected,rtol=2e-10,atol=1e-12)
            params,static=eqx.partition(net,eqx.is_inexact_array);theta,unravel=ravel_pytree(params);g,_=ravel_pytree(gradient)
            direction=np.random.default_rng(5).normal(size=theta.shape);direction/=np.linalg.norm(direction)
            fn=eqx.filter_jit(lambda a:loss(eqx.combine(unravel(a),static),jnp.asarray(nodes[1:]),mean)[0])
            h=1e-5;fd=(fn(theta+h*direction)-fn(theta-h*direction))/(2*h)
            np.testing.assert_allclose(jnp.dot(g,direction),fd,rtol=2e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
