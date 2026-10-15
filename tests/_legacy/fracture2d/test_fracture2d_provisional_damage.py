"""Provisional advancement must preserve explicit certification and original loss."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.compact_damage_features import Readout
from examples.fracture2d._legacy.compact_damage_loading import JointNetwork
from examples.fracture2d._legacy.core_halo_damage import pack_rows,predict_rows,make_cached_objective
from examples.fracture2d._legacy.provisional_damage import make_dynamic_problem,advancement,mesh_adjacency,crack_extent
jax.config.update('jax_enable_x64',True)


class ProvisionalDamageTest(unittest.TestCase):
    def test_authorization_and_uncertified_ancestry_cannot_be_silently_certified(self):
        for valid in [False,True]:
            row=advancement(user_authorized=True,strict_ready=False,numerically_valid=valid,provisional_ancestry=True)
            self.assertEqual(row['load_advanced'],valid);self.assertFalse(row['certified_path'])
        row=advancement(user_authorized=False,strict_ready=False,numerically_valid=True,provisional_ancestry=False)
        self.assertFalse(row['load_advanced'])
        row=advancement(user_authorized=True,strict_ready=True,numerically_valid=True,provisional_ancestry=True)
        self.assertTrue(row['load_advanced']);self.assertFalse(row['certified_path'])

    def test_dynamic_history_loss_and_parameter_gradient_match_original_cached(self):
        x=jnp.array([[.2,.3],[.8,.1],[.5,.4]])
        B=csr_matrix(np.array([[1.,.2],[.3,.7],[.4,.4]]));bc,bv=pack_rows(B,width=2)
        D=jnp.array([[1.,-1.,0.],[-1.,2.,-1.],[0.,-1.,1.]])
        net=JointNetwork(eqx.nn.MLP(2,1,3,1,dtype=jnp.float64,key=jax.random.PRNGKey(4)),Readout(jnp.array([.1,.2])))
        context=dict(H=jnp.array([.1,.5,.2]),features=x,lower=jnp.zeros(3),upper=jnp.zeros(3),penalty=jnp.asarray(1000.))
        predict,dynamic=make_dynamic_problem(bc,bv,lambda d:D@d,gc0=2.,gc1=.1,power=16,floor=1e-12,trunk_gain=.5)
        old=make_cached_objective(lambda n:predict_rows(n,x,bc,bv,trunk_gain=.5),lambda d:D@d,gc0=2.,gc1=.1,power=16,floor=1e-12)
        v0,g0=eqx.filter_value_and_grad(dynamic)(net,context,jnp.asarray(.0054))
        v1,g1=eqx.filter_value_and_grad(old)(net,context,jnp.asarray(.0054))
        np.testing.assert_allclose(v0,v1,rtol=1e-14)
        for a,b in zip(jax.tree.leaves(g0),jax.tree.leaves(g1)):np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-14)
        np.testing.assert_array_equal(jax.grad(lambda h:dynamic(net,{**context,'H':h},jnp.asarray(.0054)))(context['H']),jnp.zeros(3))
        np.testing.assert_array_equal(jax.grad(lambda f:dynamic(net,{**context,'features':f},jnp.asarray(.0054)))(x),jnp.zeros_like(x))

    def test_crack_requires_connected_path_not_only_damage_at_right_edge(self):
        x=np.array([[.5,0.],[.6,0.],[1.,0.],[.6,.1],[.8,.1]])
        triangles=np.array([[0,1,3],[1,3,4],[1,4,2]])
        graph=mesh_adjacency(triangles,population_size=5)
        evaluate=lambda d:crack_extent(np.asarray(d),x,graph,threshold=.9,tip=[.5,0.],tip_radius=.02,right_edge=1.,tolerance=1e-12)
        self.assertFalse(evaluate([1.,0.,1.,0.,0.])['spans_ligament'])
        self.assertTrue(evaluate([1.,1.,1.,0.,0.])['spans_ligament'])
        self.assertIsNone(evaluate([0.,0.,0.,0.,0.])['tip_component_xmax'])


if __name__=='__main__':unittest.main()
