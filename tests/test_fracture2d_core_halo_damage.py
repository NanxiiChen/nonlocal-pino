"""Sampled NN must retain exact nonlocal rows/gradients and global scales."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.compact_damage_features import Readout, residual_power_loss
from examples.fracture2d.compact_damage_loading import JointNetwork
from examples.fracture2d.damage_residual_weighting import build_core_halo
from examples.fracture2d.core_halo_damage import (
    pack_rows, predict_rows, sampled_power_loss, sampled_augmented_term,
    make_sampled_objective, make_cached_objective, validate_probabilities,
    crack_sampling_probabilities,
)
from examples.fracture2d.range_damage_optimization import augmented_term

jax.config.update('jax_enable_x64', True)


class CoreHaloDamageTest(unittest.TestCase):
    def test_crack_density_symmetric_monotone_positive_and_detached(self):
        x=jnp.array([[.7,.5],[.7,.524],[.7,.476],[.7,.548],[.7,.452],[.7,.9],[.3,.5]])
        probability=lambda pts:crack_sampling_probabilities(pts,start=[.5,.5],end=[1.,.5],
                                                           width=.024,boost=49.,power=2.)
        p=probability(x)
        validate_probabilities(p,population_size=len(x))
        self.assertGreater(p[0],p[1]);self.assertGreater(p[1],p[3]);self.assertGreater(p[3],p[5])
        np.testing.assert_allclose(p[1],p[2],rtol=1e-14)
        np.testing.assert_allclose(p[3],p[4],rtol=1e-14)
        self.assertGreater(p[0],p[6])
        np.testing.assert_array_equal(jax.jacrev(probability)(x),jnp.zeros((len(x),len(x),2)))

    def test_full_core_loss_and_network_gradient_match_full_objective(self):
        n = 4
        features = jnp.array([[.2, .1], [.7, .2], [.3, .8], [.9, .5]])
        net = JointNetwork(eqx.nn.MLP(2, 1, 5, 1, activation=jax.nn.silu, dtype=jnp.float64,
                                      key=jax.random.PRNGKey(2)), Readout(jnp.array([.1, .2, .3])))
        B = csr_matrix(np.array([[1., .2, 0.], [.1, .8, 0.], [0., .4, .6], [.1, 0., .7]]))
        D = csr_matrix(np.array([[2., -1., -1., 0.], [-1., 3., -1., -1.],
                                [-1., -1., 2., 0.], [0., -1., 0., 1.]]))
        bc, bv = pack_rows(B, width=3)
        H = jnp.array([.2, .4, .1, .7]);p = jnp.full(n, 1/n)
        ids = np.arange(n, dtype=np.int32)
        packed = build_core_halo(D, ids, node_capacity=n, row_width=4)
        context = dict(H=H, core_H=H, features=features, basis_columns=bc, basis_values=bv,
            core_local=packed['core_local'], operator_columns=packed['columns'],
            operator_values=packed['values'], p_core=p, lower=jnp.zeros(n), upper=jnp.zeros(n), penalty=jnp.asarray(10.))
        predict = lambda m: predict_rows(m, features, bc, bv, trunk_gain=.7)
        sampled = make_sampled_objective(gc0=2., gc1=.1, power=16, floor=1e-12, trunk_gain=.7, population_size=n)
        cached = make_cached_objective(predict, lambda d:jnp.asarray(D.toarray())@d,
            gc0=2., gc1=.1, power=16, floor=1e-12)
        a, ga = eqx.filter_value_and_grad(sampled)(net, context, jnp.asarray(.0053))
        b, gb = eqx.filter_value_and_grad(cached)(net, context, jnp.asarray(.0053))
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)
        for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-13)

    def test_subset_prediction_preserves_global_basis_ids_and_adjoints(self):
        features = jnp.array([[.2, .3], [.3, .7], [.9, .1], [.4, .8]])
        net = JointNetwork(eqx.nn.MLP(2, 1, 4, 1, dtype=jnp.float64,key=jax.random.PRNGKey(3)),
                           Readout(jnp.array([.2, .3, .4])))
        B = csr_matrix(np.array([[1., 0., .1], [.2, .3, 0.], [0., .8, .1], [.4, 0., .6]]))
        bc,bv=pack_rows(B,width=3);ids=np.array([3,1,3])
        subset=lambda m:predict_rows(m,features[ids],bc[ids],bv[ids],trunk_gain=.6)
        full=lambda m:predict_rows(m,features,bc,bv,trunk_gain=.6)[ids]
        np.testing.assert_array_equal(subset(net),full(net))
        ga=eqx.filter_grad(lambda m:jnp.sum(subset(m)**2))(net)
        gb=eqx.filter_grad(lambda m:jnp.sum(full(m)**2))(net)
        for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
            np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-14)

    def test_global_history_normalization_and_detached_sampling(self):
        H=jnp.array([.1,1.,10.]);rho=jnp.array([.2,.3,.1]);p=jnp.full(3,1/3)
        sampled=lambda h,q:sampled_power_loss(rho,h,q,population_size=3,power=16,floor=1e-12)
        np.testing.assert_allclose(sampled(H,p),residual_power_loss(rho,H,power=16,floor=1e-12),rtol=1e-14)
        np.testing.assert_array_equal(jax.grad(sampled,argnums=0)(H,p),jnp.zeros(3))
        np.testing.assert_array_equal(jax.grad(sampled,argnums=1)(H,p),jnp.zeros(3))
        d=jnp.array([-.01,.3,1.1]);lo=jnp.zeros(3);hi=jnp.zeros(3);pen=jnp.asarray(10.)
        np.testing.assert_allclose(sampled_augmented_term(d,H,lo,hi,pen,p,population_size=3,gc0=2.),
                                   augmented_term(d,H,lo,hi,pen,gc0=2.),rtol=1e-14)
        validate_probabilities(p,population_size=3)
        for bad in [jnp.array([0.,.5,.5]),jnp.array([.2,.2,.2]),jnp.array([.3,jnp.nan,.7])]:
            with self.assertRaises(ValueError):validate_probabilities(bad,population_size=3)
        with self.assertRaises(ValueError):pack_rows(csr_matrix(np.eye(3)),width=0)


if __name__ == '__main__': unittest.main()
