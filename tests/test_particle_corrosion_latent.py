"""Known-initial latent coordinates and exact Dirichlet data, without trajectories."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from tests.test_particle_corrosion_losses import fixture
from examples.particle_corrosion2d import network,losses
from examples.particle_corrosion2d.problem import initial_fields


class LatentTests(unittest.TestCase):
    def test_zero_increment_factory_preserves_other_parameters(self):
        from tests.test_particle_corrosion_losses import small_config
        c=small_config();key=jax.random.key(511)
        ordinary=network.build_network(c,key)
        c.arch.zero_output=True
        cold=network.build_network(c,key)
        np.testing.assert_array_equal(cold.output.weight,0.)
        np.testing.assert_array_equal(cold.output.bias,0.)
        restored=eqx.tree_at(lambda net:net.output,cold,ordinary.output)
        for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(ordinary),strict=True):
            if eqx.is_array(a):np.testing.assert_array_equal(a,b)

    def test_initial_and_mouth_identity_full_time_chain_and_two_losses(self):
        with jax.enable_x64(True):
            c,net,data,_=fixture()
            c.geometry.pit_x=.5;c.geometry.pit_radius=.12
            c.geometry.interface_k=29.;c.geometry.mouth_half_width=.055
            c.phase=ConfigDict(dict(initial_lifting=dict(kind='latent_additive',time_scale=.01)))
            c.boundary=ConfigDict(dict(dirichlet='exact_mouth'))
            c.loss.gradnorm_bias=(1.,1.);c.loss.pde_reduction='mean_square'
            xy=jnp.array([[.5,0.],[.445,0.],[.555,0.],[.5,.07],[.5,.12],[.7,.3]])
            expected=initial_fields(c,np.asarray(xy))
            predict=lambda ts:network.predict_fields(net,data,xy,ts,c)
            np.testing.assert_allclose(predict(jnp.zeros(1))[0],expected,atol=1e-14,rtol=1e-14)
            ts=jnp.array([0.,.001,.03,.3,1.])
            prepared=net.prepare_encoded(network.encode(net,data),xy)
            raw,fields,dt=network.time_jets(net,prepared,ts,c,xy)
            np.testing.assert_array_equal(fields[:,:3],0.)
            np.testing.assert_array_equal(dt[:,:3],0.)
            eps=1e-7
            np.testing.assert_allclose(dt,(predict(ts+eps)-predict(ts-eps))/(2*eps),rtol=3e-6,atol=2e-8)
            actual=losses.condition_losses(net,ts,data,c)
            self.assertEqual(actual.shape,(2,));self.assertEqual(len(losses.loss_names(c)),2)
            self.assertTrue(np.isfinite(actual).all())
            changed=dict(data,initial_ic=jnp.full_like(data['initial_ic'],100.),mouth_points=jnp.zeros_like(data['mouth_points']))
            np.testing.assert_array_equal(actual,losses.condition_losses(net,ts,changed,c))
            cold=eqx.tree_at(lambda m:(m.output.weight,m.output.bias),net,
                (jnp.zeros_like(net.output.weight),jnp.zeros_like(net.output.bias)))
            frozen=network.predict_fields(cold,data,xy,ts,c)
            np.testing.assert_allclose(frozen,np.broadcast_to(expected,frozen.shape),atol=1e-14,rtol=1e-14)


if __name__=='__main__':unittest.main()
