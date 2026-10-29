"""Thermodynamic variation and independently assembled global energy loss."""
import unittest
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from examples.confined_ch2d.configs.train_coarsening_global_energy import get_config
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.problem import initial_field,validate_config
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from tests.test_confined_ch import fixture


class EnergyTests(unittest.TestCase):
    def test_variation_global_objective_and_gradient(self):
        c=get_config();c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;c.arch.rank=3
        _,x,v=fixture();c.nl.delta=.2
        nodes=integration_nodes(count=24,duration=1.5,time_scale=.005)
        c.sample.Nt=24;c.sample.temporal.nodes=tuple(nodes[1:])
        c.train.refinement.time_points=24;c.train.refinement.time_nodes=tuple(nodes[1:]);validate_config(c)
        A=native_matrix(x,v,c).toarray();K=c.physical.mobility*c.physical.epsilon**2*(A@A)
        W=integration_matrix(nodes,time_scale=.005)
        energy=lambda phi:np.sum(v*(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*(A@phi)))
        with jax.enable_x64(True):
            p=jnp.asarray(x);vol=jnp.asarray(v);D=RegionalLaplacian(x,v,c)
            phi0=initial_field(p,c,xp=jnp);mean=jnp.dot(vol,phi0)/vol.sum()
            net=_build_network(c,jax.random.PRNGKey(21));times=jnp.asarray(nodes[1:])
            values=np.asarray(jax.vmap(lambda t:predict_fields(net,p,t,mean,vol,D))(jnp.asarray(nodes)))
            phi=values[...,0];mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*(phi@A.T)
            direction=np.random.default_rng(3).normal(size=len(x));step=1e-6
            fd=(energy(phi[7]+step*direction)-energy(phi[7]-step*direction))/(2*step)
            np.testing.assert_allclose(fd,np.sum(v*mu[7]*direction),rtol=1e-8,atol=1e-9)
            E=np.array([energy(a) for a in phi]);diss=np.sum(v*mu*(mu@A.T),axis=1)
            defect=(E[1:]-E[0]+c.physical.mobility*(W[1:]@diss))/E[0]
            expected_energy=np.dot(W[-1,1:],defect**2)/1.5
            residual=phi[1:]-phi[0]+c.physical.mobility*((W[1:]@mu)@A.T)
            errors=np.stack([np.linalg.solve(np.eye(len(x))+t*K,r) for t,r in zip(nodes[1:],residual,strict=True)])
            expected_phase=np.dot(W[-1,1:],np.sum(v*errors**2,axis=1)/v.sum())/1.5
            loss=make_loss_fn(p,vol,D,None,None,None,c)
            (value,parts),g=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(net,times,mean)
            np.testing.assert_allclose(parts,[expected_phase,expected_energy],rtol=1e-10,atol=1e-11)
            np.testing.assert_allclose(value,c.loss.phase_weight*expected_phase+c.loss.energy_weight*expected_energy,rtol=1e-10)
            leaves=jax.tree.leaves(eqx.filter(g,eqx.is_array));self.assertTrue(all(np.isfinite(a).all() for a in leaves))
            self.assertGreater(sum(float(jnp.sum(a*a)) for a in leaves),0.)


if __name__=='__main__':unittest.main()
