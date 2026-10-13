"""Check low-order support, importance gradients, detachment and optimizer memory."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.configs.sigmoid_damage_load_refined import get_config
from examples.fracture2d.core_halo_damage import pack_rows
from examples.fracture2d.damage_residual_weighting import build_core_halo
from examples.fracture2d.low_order_damage import make_loss_fn, make_optimizer, make_step, optimizer_count
from examples.fracture2d.plain_damage_head import fresh_network


class LowOrderDamageTest(unittest.TestCase):
    def setUp(self):
        jax.config.update('jax_enable_x64', True)
        c = get_config(); c.arch.hidden_dim, c.arch.num_layers = 4, 1
        self.D = csr_matrix(np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]]))
        B = csr_matrix(np.eye(3, dtype=np.float64))
        self.bc, self.bv = map(jnp.asarray, pack_rows(B, width=1))
        self.net = fresh_network(c, 2, B, head='sigmoid')
        self.net = eqx.tree_at(lambda n:n.readout.coefficients, self.net, jnp.array([.3, -.2, .7]))
        self.H = jnp.array([1., 2., 4.])
        self.X = jnp.array([[1., 2.], [3., 1.], [2., 4.]])
        self.load = jnp.asarray(.005275)

    def context(self, core, probabilities):
        core = np.asarray(core, np.int32)
        p = build_core_halo(self.D, core, node_capacity=3, row_width=3)
        ids = p['global_ids']
        return dict(H=self.H, core_H=self.H[core], features=self.X[ids],
            basis_columns=self.bc[ids], basis_values=self.bv[ids],
            core_local=jnp.asarray(p['core_local']), operator_columns=jnp.asarray(p['columns']),
            operator_values=jnp.asarray(p['values']), p_core=jnp.asarray(probabilities)[core])

    def test_full_core_matches_original_equation_and_low_order_reduction(self):
        ctx = self.context([0, 1, 2], [1/3]*3)
        for norm in ['l1', 'l2']:
            image, loss = make_loss_fn(gc0=112.5, gc1=.0648, gain=1., population_size=3, norm=norm)
            d = image(self.net, ctx)[ctx['core_local']]
            rho = ((112.5+2*self.H)*d+.0648*(jnp.asarray(self.D.toarray())@d)-2*self.H)/(2*jnp.sqrt(jnp.mean(self.H**2)))
            expected = jnp.mean(abs(rho)) if norm == 'l1' else jnp.mean(rho**2)
            np.testing.assert_allclose(loss(self.net, ctx, self.load), expected, rtol=1e-13)

    def test_nonuniform_expected_loss_and_parameter_gradient_equal_full(self):
        probabilities = np.array([.1, .2, .7])
        full = self.context([0, 1, 2], [1/3]*3)
        for norm in ['l1', 'l2']:
            _, loss = make_loss_fn(gc0=112.5, gc1=.0648, gain=1., population_size=3, norm=norm)
            def expected(net):
                return sum(float(probabilities[i])*loss(net, self.context([i], probabilities), self.load) for i in range(3))
            np.testing.assert_allclose(expected(self.net), loss(self.net, full, self.load), rtol=1e-13)
            a, b = eqx.filter_grad(expected)(self.net), eqx.filter_grad(loss)(self.net, full, self.load)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-12)

    def test_repeated_core_and_frozen_context_gradients(self):
        ctx = self.context([2, 0, 2], [.1, .2, .7])
        for norm in ['l1', 'l2']:
            _, loss = make_loss_fn(gc0=112.5, gc1=.0648, gain=1., population_size=3, norm=norm)
            floats = {k:v for k,v in ctx.items() if jnp.issubdtype(v.dtype, jnp.inexact)}
            integers = {k:v for k,v in ctx.items() if k not in floats}
            gradients = jax.grad(lambda values:loss(self.net, {**integers, **values}, self.load))(floats)
            for value in gradients.values():
                np.testing.assert_array_equal(value, jnp.zeros_like(value))

    def test_adam_and_soap_counts_persist_across_changed_batches(self):
        for kind in ['adam', 'soap']:
            _, loss = make_loss_fn(gc0=112.5, gc1=.0648, gain=1., population_size=3, norm='l2')
            optimizer, _ = make_optimizer(kind=kind, learning_rate=1e-4, decay_steps=4,
                decay_rate=.1, minimum_rate=1e-5, grad_clip=1000., b1=.9, b2=.99,
                eps=1e-8, eps_root=0., shampoo_beta=.99, correct_bias=True,
                precondition_frequency=2, max_precond_dim=8, precondition_1d=False, weight_decay=0.)
            step = make_step(loss, optimizer)
            net = self.net; state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
            for i, core in enumerate([[0, 1], [2, 0], [1, 2]], 1):
                before = net
                net, state, loss_value, _ = step(net, state, self.context(core, [1/3]*3), self.load)
                self.assertEqual(optimizer_count(state, kind=kind), i)
                self.assertTrue(np.isfinite(float(loss_value)))
                if kind == 'soap' and i == 1:
                    self.assertTrue(eqx.tree_equal(net, before))
            self.assertFalse(eqx.tree_equal(net, self.net))


if __name__ == '__main__':
    unittest.main()
