"""Independent fixed-linear transpose oracle with repeated sampled cores/padding."""
import unittest

import equinox as eqx

import jax
import jax.numpy as jnp
import numpy as np

from examples.particle_corrosion2d.spatial_adjoint import prepare_adjoint,validate_support,fixed_gather
from examples.particle_corrosion2d.losses import graph_action,residuals
from tests.test_particle_corrosion_losses import small_config


class SpatialAdjointTests(unittest.TestCase):
    def test_zero_padding_cannot_hide_repeated_physical_neighbors(self):
        data=dict(neighbors=np.array([[0,2,0]]),coefficients=np.array([[1.,2.,0.]]))
        validate_support(data)
        data['coefficients'][0,2]=1.
        with self.assertRaisesRegex(ValueError,'unique nonzero'):
            validate_support(data)

    def test_forward_and_parameter_adjoint_equal_original_linear_map(self):
        with jax.enable_x64():
            data=dict(graph_points=jnp.zeros((4,2)),core_indices=jnp.array([1,1,3]),
                      neighbors=jnp.array([[0,2,0],[0,2,0],[0,2,0]]),
                      coefficients=jnp.array([[1.,2.,0.],[1.,2.,0.],[3.,4.,0.]]))
            prepared=prepare_adjoint(data)
            values=jnp.array([[.2,.7],[.4,-.2],[1.,.3],[-.1,.8]])
            cotangent=jnp.array([[1.,-.5],[.4,.2],[-.3,.8]])
            matrix=np.array([[-1.,3.,-2.,0.],[-1.,3.,-2.,0.],[-3.,0.,-4.,7.]])
            np.testing.assert_array_equal(prepared['operator_matrix'],matrix)
            np.testing.assert_array_equal(graph_action(values,data),graph_action(values,prepared))
            gradient=jax.grad(lambda v:jnp.sum(graph_action(v,prepared)*cotangent))(values)
            np.testing.assert_allclose(gradient,matrix.T@np.asarray(cotangent),rtol=1e-13,atol=1e-13)
            gather_gradient=jax.grad(lambda v:jnp.sum(fixed_gather(v,data['core_indices'],
                                               prepared['core_selection'])*cotangent))(values)
            expected=np.zeros((4,2));np.add.at(expected,np.asarray(data['core_indices']),np.asarray(cotangent))
            np.testing.assert_array_equal(gather_gradient,expected)

    def test_complete_network_parameter_gradients_preserve_all_five_losses(self):
        from tests.test_particle_corrosion_losses import fixture
        from examples.particle_corrosion2d.losses import condition_losses
        with jax.enable_x64():
            c,model,data,times=fixture()
            prepared=prepare_adjoint(data)
            np.testing.assert_array_equal(condition_losses(model,times,data,c),
                                          condition_losses(model,times,prepared,c))
            bias=jnp.array([.7,1.3,.8,2.,1.5])
            def objective(net,payload):
                return jnp.sum(condition_losses(net,times,payload,c)*bias)
            original=eqx.filter_grad(objective)(model,data)
            updated=eqx.filter_grad(objective)(model,prepared)
            for a,b in zip(jax.tree.leaves(original),jax.tree.leaves(updated),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-11)

    def test_field_gradient_of_coupled_residual_matches_original(self):
        with jax.enable_x64():
            c=small_config()
            data=dict(graph_points=jnp.zeros((4,2)),core_indices=jnp.array([1,1,3]),
                      neighbors=jnp.array([[0,2],[0,2],[0,2]]),
                      coefficients=jnp.array([[1.,2.],[1.,2.],[3.,4.]]))
            prepared=prepare_adjoint(data)
            raw=jnp.asarray(np.random.default_rng(29).normal(size=(3,4,2)))
            dt=raw*.17
            def loss(r,d,p):
                ac,ch=residuals(r,d,p,c)
                return jnp.sum(ac**2)+jnp.sum(ch**2)
            for original,updated in zip(jax.grad(loss,argnums=(0,1))(raw,dt,data),
                                        jax.grad(loss,argnums=(0,1))(raw,dt,prepared),strict=True):
                np.testing.assert_allclose(original,updated,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
