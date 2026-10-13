"""Protect bounded halo propagation, original loss gradients and detached sampling."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.bounded_damage_morphology_trial import make_problem
from examples.fracture2d.configs.sigmoid_damage_accuracy_trial import get_config
from examples.fracture2d.core_halo_damage import pack_rows
from examples.fracture2d.damage_residual_weighting import build_core_halo, core_matvec
from examples.fracture2d.plain_damage_head import fresh_network
from examples.fracture2d.sigmoid_damage_accuracy_trial import make_sampled_sigmoid_loss


class SigmoidAccuracyTest(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64',True)
        c = get_config()
        c.arch.hidden_dim, c.arch.num_layers = 4, 1
        self.D = csr_matrix(np.array([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]]))
        B = csr_matrix(np.eye(3,dtype=np.float64))
        self.bc, self.bv = map(jnp.asarray, pack_rows(B,width=1))
        self.net = fresh_network(c,2,B,head='sigmoid')
        self.net = eqx.tree_at(lambda n:n.readout.coefficients,self.net,jnp.array([.3,-.2,.7]))
        self.H = jnp.array([1.,2.,4.])
        self.X = jnp.array([[1.,2.],[3.,1.],[2.,4.]])
        self.load = jnp.asarray(.0053)
        self.full = dict(H=self.H,features=self.X)
        self.image,self.loss,_ = make_problem(self.bc,self.bv,lambda d:jnp.asarray(self.D.toarray())@d,
            head='sigmoid',gain=1.,gc0=112.5,gc1=.0648,power=16,floor=1e-15,
            volumes=jnp.ones(3),thresholds=(.1,.5,.9))
        self.halo_image,self.sample_loss = make_sampled_sigmoid_loss(gc0=112.5,gc1=.0648,
            gain=1.,power=16,floor=1e-15,population_size=3)

    def context(self,core):
        packed = build_core_halo(self.D,np.asarray(core,np.int32),node_capacity=3,row_width=3)
        ids = packed['global_ids']
        return dict(H=self.H,core_H=self.H[jnp.asarray(core)],features=self.X[ids],
            basis_columns=self.bc[ids],basis_values=self.bv[ids],
            core_local=jnp.asarray(packed['core_local']),operator_columns=jnp.asarray(packed['columns']),
            operator_values=jnp.asarray(packed['values']),p_core=jnp.full(len(core),1/3,jnp.float64))

    def test_full_core_loss_and_parameter_gradient_match(self):
        sample = self.context([0,1,2])
        np.testing.assert_allclose(self.sample_loss(self.net,sample,self.load),
            self.loss(self.net,self.full,self.load),rtol=1e-13)
        gs = eqx.filter_grad(self.sample_loss)(self.net,sample,self.load)
        gf = eqx.filter_grad(self.loss)(self.net,self.full,self.load)
        for a,b in zip(jax.tree.leaves(gs),jax.tree.leaves(gf)):
            np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-12)

    def test_repeated_core_uses_bounded_values_at_every_halo_node(self):
        core = [2,0,2]
        sample = self.context(core)
        d = self.image(self.net,self.full,self.load)
        halo = self.halo_image(self.net,sample)
        np.testing.assert_allclose(halo[sample['core_local']],d[jnp.asarray(core)],rtol=1e-14)
        np.testing.assert_allclose(core_matvec(halo,sample['operator_columns'],sample['operator_values']),
            (self.D@np.asarray(d))[core],rtol=1e-13,atol=1e-14)

    def test_history_features_probabilities_and_row_values_are_detached(self):
        sample = self.context([0,1,2])
        floats = {k:v for k,v in sample.items() if jnp.issubdtype(v.dtype,jnp.inexact)}
        integers = {k:v for k,v in sample.items() if k not in floats}
        gradient = jax.grad(lambda values:self.sample_loss(self.net,{**integers,**values},self.load))(floats)
        for value in gradient.values():
            np.testing.assert_array_equal(value,jnp.zeros_like(value))


if __name__ == '__main__':
    unittest.main()
