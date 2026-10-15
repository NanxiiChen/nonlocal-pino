"""Physical input identity, masking, encoder gradients and JIT replay invariants."""
import json
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import optax
from scipy.sparse import diags
from examples.fracture2d.history_network import (
    HistoryOperator, HistoryReplay, build_patches, make_context)
from examples.fracture2d.history_probe import load_snapshots, make_loss, make_step
from examples.fracture2d.sampling import pack_rows


def problem():
    axis = np.linspace(0, 1, 5)
    points = np.array([(x, y) for y in axis for x in axis], np.float64)
    volumes = np.linspace(.5, 1.5, len(points))
    coarse = build_patches(points, volumes, cells=2)
    fine = build_patches(points, volumes, cells=8)  # Deliberately has empty cells.
    H = 1+np.exp(-40*(points[:, 1]-.5)**2)*(1+points[:, 0])
    context = make_context(H, fine, gc0=2., count=4)
    net = HistoryOperator(width=8, point_depth=1, query_depth=1,
        head_depth=1, heads=2, initial_damage=.1, key=jax.random.PRNGKey(3))
    # A nonconstant head is needed to exercise the encoder, not just initial bias.
    net = eqx.tree_at(lambda n:n.head.layers[-1].weight, net,
                     jnp.full_like(net.head.layers[-1].weight, .1))
    D = diags([-np.ones(24), np.r_[1., np.full(23, 2.), 1.], -np.ones(24)],
              [-1, 0, 1], format='csr')
    columns, values = pack_rows(D, width=3)
    image, objective = make_loss(coarse=coarse, fine=fine, points=jnp.asarray(points),
        columns=jnp.asarray(columns), values=jnp.asarray(values), gc0=2., gc1=.3, floor=1e-15)
    return points, volumes, coarse, fine, context, net, D, image, objective


class HistoryOperatorTests(unittest.TestCase):
    def test_patch_padding_and_empty_tokens_do_not_change_predictions(self):
        points, _, coarse, fine, ctx, net, _, image, _ = problem()
        original = np.asarray(image(net, ctx))
        corrupted = eqx.tree_at(lambda p:p.relative, coarse,
            jnp.where(coarse.mask[..., None], coarse.relative, 123.))
        actual = net(ctx, jnp.asarray(points), corrupted, fine, gc0=2.)
        np.testing.assert_allclose(actual, original, rtol=1e-14, atol=1e-14)
        self.assertTrue(np.isfinite(original).all())
        np.testing.assert_allclose(np.asarray(coarse.weights).sum(1), 1.)
        for i in np.asarray(ctx['fine_ids']):
            self.assertTrue(bool(fine.valid[i]))

    def test_shared_query_coordinates_and_distinct_histories(self):
        points, _, coarse, fine, ctx, net, _, image, _ = problem()
        pred = np.asarray(image(net, ctx))
        order = np.array([12, 3, 20, 1])
        actual = net(ctx, jnp.asarray(points[order]), coarse, fine, gc0=2.)
        np.testing.assert_allclose(actual, pred[order], rtol=1e-13, atol=1e-14)
        changed = make_context(np.asarray(ctx['H'])[::-1].copy(), fine, gc0=2., count=4)
        self.assertGreater(np.max(abs(np.asarray(image(net, changed))-pred)), 1e-10)

    def test_two_history_residual_matches_numpy_and_inputs_are_detached(self):
        _, _, _, fine, ctx, net, D, image, objective = problem()
        other = make_context(3*np.asarray(ctx['H']), fine, gc0=2., count=4)
        batch = dict(H=jnp.stack((ctx['H'], other['H'])),
            fine_ids=jnp.stack((ctx['fine_ids'], other['fine_ids'])), weights=jnp.array([.75, .25]))
        expected = []
        for context in (ctx, other):
            H = np.asarray(context['H'])
            d = np.asarray(image(net, context))
            r = (2+2*H)*d+.3*(D@d)-2*H
            expected.append(np.mean(abs(r))/(2*np.sqrt(np.mean(H**2))))
        self.assertAlmostEqual(float(objective(net, batch)), .75*expected[0]+.25*expected[1], places=13)
        gradient = eqx.filter_grad(objective)(net, batch)
        self.assertGreater(float(optax.global_norm(gradient.point_mlp)), 0.)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradient)))
        np.testing.assert_array_equal(jax.grad(lambda H:objective(net, dict(batch, H=H)))(batch['H']), 0.)
        np.testing.assert_array_equal(jax.grad(lambda w:objective(net, dict(batch, weights=w)))(batch['weights']), 0.)

    def test_buffer_growth_and_masks_use_one_compiled_update(self):
        _, _, _, _, ctx, net, _, _, objective = problem()
        traces = []
        def counted(model, batch):
            traces.append(1)
            return objective(model, batch)
        optimizer = optax.adam(1e-3)
        step = make_step(counted, optimizer, old_count=1, fraction=.25)
        state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
        bank = HistoryReplay(capacity=3, nodes=25, fine_count=4)
        original = bank
        for i in range(4):
            flag = jnp.asarray(i != 0)
            batch = bank.batch(ctx, jax.random.PRNGKey(i), flag, old_count=1, fraction=.25)
            self.assertEqual(batch['H'].shape, (2, 25))
            self.assertEqual(batch['fine_ids'].shape, (2, 4))
            self.assertTrue(np.isfinite(np.asarray(batch['H'])).all())
            np.testing.assert_allclose(batch['weights'], [1., 0.] if i==0 else [.75, .25])
            net, state, loss = step(net, state, ctx, bank, jax.random.PRNGKey(i), flag)
            self.assertTrue(np.isfinite(float(loss)))
            if i < 3:
                bank = bank.append(ctx)
        self.assertEqual(len(traces), 1)
        self.assertEqual(int(original.count), 0)
        self.assertEqual(int(bank.count), 3)
        with self.assertRaisesRegex(ValueError, 'capacity exhausted'):
            bank.append(ctx)
        # Even a nonempty bank can disable replay without a new shape/trace.
        net, state, loss = step(net, state, ctx, bank, jax.random.PRNGKey(8), jnp.asarray(False))
        self.assertEqual(len(traces), 1)
        self.assertEqual(int(state[0].count), 5)

    def test_serialized_replay_and_optimizer_resume_exactly(self):
        _, _, _, _, ctx, net, _, _, objective = problem()
        optimizer = optax.adam(1e-3)
        state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
        bank = HistoryReplay(capacity=3, nodes=25, fine_count=4).append(ctx)
        step = make_step(objective, optimizer, old_count=1, fraction=.25)
        net, state, _ = step(net, state, ctx, bank, jax.random.PRNGKey(0), jnp.asarray(True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.eqx'
            eqx.tree_serialise_leaves(path, (net, state, bank))
            restored = eqx.tree_deserialise_leaves(path, (net, state, bank))
            args = (ctx, bank, jax.random.PRNGKey(9), jnp.asarray(True))
            expected = step(net, state, *args)
            actual = step(restored[0], restored[1], ctx, restored[2], args[2], args[3])
            for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
                np.testing.assert_array_equal(a, b)

    def test_snapshot_collection_uses_only_first_outer_and_physical_endpoint(self):
        from tests.test_fracture2d_main_route import fixture
        from examples.fracture2d.mesh import load_geometry
        from ml_collections import ConfigDict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            m, _, _ = load_geometry(c)
            source = root/'source'
            source.mkdir()
            c.history = ConfigDict(dict(source_root=str(source), loads=[.0001, .0002], held_out_load=.0003))
            (source/'config.json').write_text(json.dumps(c.to_dict()))
            rows = []
            for index, load in [(7, .0001), (9, .0002), (12, .0003)]:
                folder = source/f'state_{index:04d}'
                (folder/'outer_0001').mkdir(parents=True)
                np.savez(folder/'outer_0001/fields.npz', H=np.full(4, float(index)))
                np.savez(folder/'fields.npz', H=np.full(4, float(index+1)), H_used=np.full(4, 999.))
                rows.append(dict(index=index, load=load, q_d_synchronized=True))
            (source/'result.json').write_text(json.dumps(dict(curve=rows)))
            training, held_out = load_snapshots(c, m)
            self.assertEqual(len(training), 4)
            self.assertEqual(len(held_out), 2)
            np.testing.assert_array_equal(training[0]['H'], 7.)
            np.testing.assert_array_equal(training[1]['H'], 8.)
            self.assertEqual([x['phase'] for x in training], ['start', 'end', 'start', 'end'])
            np.testing.assert_array_equal(held_out[0]['H'], 12.)


if __name__ == '__main__':
    unittest.main()
