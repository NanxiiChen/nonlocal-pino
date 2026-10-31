"""Complete6-neighbor sampling, FIFO refresh and exact full objective gradients."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from nonlocal_pinn.operators import build_D
from examples.icemelting3d.configs.train_operator_fixed_pool import get_config
from examples.icemelting3d.operator_losses import make_loss_fn
from examples.icemelting3d.spatial_sampling import SpatialSampler, core_subgraph, make_accumulator, pad_graph


class MeltingSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = np.linspace(0, 1, 4, dtype=np.float32)
        cls.points = np.stack(np.meshgrid(a, a, a, indexing='xy'), -1).reshape(-1, 3)
        cls.op = build_D(jnp.asarray(cls.points), .42, method='sparse', kernel='cubic')

    def config(self, mode):
        c = get_config(); c.sample.Nx = c.sample.Ny = c.sample.Nz = 4
        c.sample.K_core, c.sample.K_pool = 4, 3
        c.sample.spatial.mode = mode
        c.sample.spatial.max_nodes, c.sample.spatial.max_edges = 28, 24
        c.sample.spatial.full_chunk_core = 13
        c.sample.spatial.prefetch = 3
        return c

    def test_complete_rows_fixed_rolling_identity_and_overflow(self):
        values = np.random.default_rng(3).normal(size=64)
        full = np.asarray(self.op._diag)*values
        np.add.at(full, self.op._row_idx_np, self.op._vals_np*values[self.op._col_idx_np])
        traces = []
        for workers in (1, 3):
            c = self.config('rolling_pool'); c.sample.spatial.workers = workers
            with SpatialSampler(c, self.op, self.points) as sampler:
                for k in range(9):
                    points, sub, mask = sampler.take(k)
                    # Match actual padded points to the original fixed grid.
                    ids = np.argmin(np.sum((np.asarray(points)[:, None]-self.points[None])**2, axis=-1), axis=1)
                    row, col, weights, diagonal = map(np.asarray, sub)
                    actual = diagonal*values[ids]
                    np.add.at(actual, row, weights*values[ids[col]])
                    np.testing.assert_allclose(actual[mask], full[ids[mask]], rtol=2e-6, atol=2e-6)
                traces.append(sampler.trace())
        for key in traces[0]:
            np.testing.assert_array_equal(traces[0][key], traces[1][key])
        with SpatialSampler(self.config('fixed_pool'), self.op, self.points) as fixed:
            for k in range(9):
                fixed.take(k)
            np.testing.assert_array_equal(fixed.trace()['core_ids'], traces[0]['core_ids'][:3])
        raw = core_subgraph(self.op, np.array([0, 1, 4, 30]))
        with self.assertRaisesRegex(ValueError, 'overflow'):
            pad_graph(raw, len(raw[0])-1, len(raw[1][0]))

    def test_full_chunks_loss_gradient_and_one_optimizer_update(self):
        with jax.enable_x64(True):
            c = self.config('full')
            net = ConditionalFieldOperator(image_shape=(1, 4, 4, 4), patch_shape=(2, 2, 2),
                extent=(1., 1., 1.), width=12, encoder_depth=1, heads=2, mlp_ratio=2,
                decoder_depth=1, spatial_features=3, spatial_scale=1., time_features=2,
                time_scale=1., local_dim=0, out_dim=1, conditioning='cross_attention',
                norm_eps=1e-5, key=jax.random.key(8))
            # FP64 isolates the algebraic identity from Adam's amplification of
            # FP32 cancellation in near-zero attention bias gradients.
            images = jnp.linspace(-1, 1, 64).reshape(1, 1, 4, 4, 4)
            times = jnp.array([.19, .71])
            initial = lambda x: .3*x[:, 0]-.2*x[:, 1]+.15*x[:, 2]
            raw = core_subgraph(self.op, np.arange(64))
            ids, sub, mask = pad_graph(raw, 64, len(raw[1][0]))
            points = jnp.asarray(self.points[ids])
            loss = make_loss_fn(c)
            expected_value, expected_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))(
                net, times, images, initial(points)[None], points, jax.device_put(sub), jnp.asarray(mask))
            optimizer = optax.adam(1e-3); state = optimizer.init(eqx.filter(net, eqx.is_array))
            accumulate, apply = make_accumulator(c, optimizer)
            total = jax.tree.map(jnp.zeros_like, eqx.filter(net, eqx.is_array))
            value = jnp.asarray(0.)
            covered = []
            with SpatialSampler(c, self.op, self.points) as sampler:
                for (x, sub, mask), weight in sampler.full_chunks():
                    covered.extend(map(tuple, np.asarray(x)[np.asarray(mask)]))
                    total, value = accumulate(net, total, value, times, images,
                        initial(x)[None], x, sub, mask, jnp.asarray(weight))
            self.assertEqual(len(covered), 64)
            self.assertEqual(set(covered), set(map(tuple, self.points)))
            np.testing.assert_allclose(value, expected_value, rtol=2e-12, atol=2e-12)
            for a, b in zip(jax.tree.leaves(total), jax.tree.leaves(expected_grad), strict=True):
                np.testing.assert_allclose(a, b, rtol=2e-10, atol=1e-11)
            actual, actual_state, _, finite = apply(net, state, total, value)
            updates, expected_state = optimizer.update(expected_grad, state, net)
            expected = eqx.apply_updates(net, updates)
            self.assertTrue(bool(finite))
            for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                if eqx.is_array(a):
                    np.testing.assert_allclose(a, b, rtol=2e-9, atol=3e-10)
            np.testing.assert_array_equal(actual_state[0].count, expected_state[0].count)


if __name__ == '__main__':
    unittest.main()
