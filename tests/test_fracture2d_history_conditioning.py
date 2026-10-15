"""Experimental input conditioning preserves physics, masks and JIT structure."""
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax
from examples.fracture2d.history_conditioning import ConditionedHistoryOperator, history_inputs
from examples.fracture2d.history_network import HistoryReplay
from examples.fracture2d.history_probe import make_step
from tests.test_fracture2d_history_operator import problem


def model():
    return ConditionedHistoryOperator(width=8, point_depth=1, query_depth=1,
        head_depth=1, heads=2, initial_damage=.1, normalized=False,
        local_history=False, key=jax.random.PRNGKey(3))


class HistoryConditioningTests(unittest.TestCase):
    def test_normalization_preserves_amplitude_and_is_stable_across_scales(self):
        H = jnp.array([0., 1., 3., 8.])
        scale = np.sqrt(np.mean(np.asarray(H)**2))
        encoded, local, amp = history_inputs(H, gc0=2., normalized=jnp.asarray(True))
        np.testing.assert_allclose(encoded, np.asarray(H)/scale, rtol=1e-14)
        np.testing.assert_allclose(local[:,0], np.log1p(np.asarray(H)/scale), rtol=1e-14)
        self.assertAlmostEqual(float(amp), np.log(scale), places=13)
        for factor in (1e-200, 1e-8, 1e8, 1e200):
            other, features, amplitude = history_inputs(factor*H, gc0=2., normalized=jnp.asarray(True))
            np.testing.assert_allclose(other, encoded, rtol=1e-14)
            np.testing.assert_allclose(features, local, rtol=1e-14)
            self.assertAlmostEqual(float(amplitude-amp), np.log(factor), places=12)
        raw, _, amp = history_inputs(H, gc0=2., normalized=jnp.asarray(False))
        np.testing.assert_array_equal(raw, H)
        self.assertEqual(float(amp), 0.)

    def test_query_ordering_padding_and_explicit_index_validation(self):
        points, _, coarse, fine, ctx, _, _, image, _ = problem()
        net = model()
        np.testing.assert_allclose(image(net,ctx), .1, rtol=1e-14)
        net = eqx.tree_at(lambda n:(n.normalized,n.local_history,n.head.layers[-1].weight),
            net,(jnp.asarray(True),jnp.asarray(True),jnp.full_like(net.head.layers[-1].weight,.1)))
        pred = np.asarray(image(net,ctx))
        order = jnp.array([12, 3, 20, 1],jnp.int32)
        actual = net.predict_queries(ctx,jnp.asarray(points)[order],order,coarse,fine,gc0=2.)
        np.testing.assert_allclose(actual,pred[np.asarray(order)],rtol=1e-13,atol=1e-14)
        corrupted = eqx.tree_at(lambda p:p.relative, coarse,
            jnp.where(coarse.mask[...,None],coarse.relative,123.))
        np.testing.assert_allclose(net(ctx,jnp.asarray(points),corrupted,fine,gc0=2.),pred,rtol=1e-13)
        with self.assertRaisesRegex(ValueError,'full-cloud'):
            net(ctx,jnp.asarray(points[:2]),coarse,fine,gc0=2.)
        with self.assertRaisesRegex(Exception,'query index out of range'):
            net.predict_queries(ctx,jnp.asarray(points[:1]),jnp.array([25],jnp.int32),coarse,fine,gc0=2.)

    def test_raw_physical_history_is_used_in_residual_and_inputs_are_detached(self):
        _, _, _, _, ctx, _, D, image, objective = problem()
        net = model()
        net = eqx.tree_at(lambda n:(n.normalized,n.local_history,n.head.layers[-1].weight),
            net,(jnp.asarray(True),jnp.asarray(True),jnp.full_like(net.head.layers[-1].weight,.1)))
        batch = dict(H=ctx['H'][None],fine_ids=ctx['fine_ids'][None],weights=jnp.ones(1))
        d = np.asarray(image(net,ctx));H = np.asarray(ctx['H'])
        expected = np.mean(abs((2+2*H)*d+.3*(D@d)-2*H))/(2*np.sqrt(np.mean(H**2)))
        self.assertAlmostEqual(float(objective(net,batch)),expected,places=13)
        np.testing.assert_array_equal(jax.grad(lambda H:objective(net,dict(batch,H=H)))(batch['H']),0.)
        gradient = eqx.filter_grad(objective)(net,batch)
        self.assertIsNone(gradient.normalized)
        self.assertIsNone(gradient.local_history)
        self.assertGreater(float(optax.global_norm(gradient.point_mlp)),0.)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)))

    def test_all_modes_share_parameters_and_one_trace_with_exact_resume(self):
        _, _, _, _, ctx, _, _, _, objective = problem()
        net = model();parameters = eqx.filter(net,eqx.is_inexact_array)
        optimizer = optax.adam(1e-4);state = optimizer.init(parameters)
        bank = HistoryReplay(capacity=2,nodes=25,fine_count=4)
        traces = []
        def counted(n,b):
            traces.append(1)
            return objective(n,b)
        step = make_step(counted,optimizer,old_count=1,fraction=.25)
        for norm,local in ((False,False),(True,False),(False,True),(True,True)):
            arm = eqx.tree_at(lambda n:(n.normalized,n.local_history),net,
                (jnp.asarray(norm),jnp.asarray(local)))
            for a,b in zip(jax.tree.leaves(parameters),jax.tree.leaves(eqx.filter(arm,eqx.is_inexact_array))):
                np.testing.assert_array_equal(a,b)
            result = step(arm,state,ctx,bank,jax.random.PRNGKey(0),jnp.asarray(False))
            self.assertTrue(np.isfinite(float(result[2])))
        self.assertEqual(len(traces),1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.eqx';eqx.tree_serialise_leaves(path,result[:2])
            restored = eqx.tree_deserialise_leaves(path,(arm,state))
            expected = step(result[0],result[1],ctx,bank,jax.random.PRNGKey(1),jnp.asarray(False))
            actual = step(restored[0],restored[1],ctx,bank,jax.random.PRNGKey(1),jnp.asarray(False))
            for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
                np.testing.assert_array_equal(a,b)
        self.assertEqual(len(traces),1)


if __name__ == '__main__':
    unittest.main()
