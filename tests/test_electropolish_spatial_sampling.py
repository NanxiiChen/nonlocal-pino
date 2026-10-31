"""Complete graph rows, strict padding and deterministic background replacement."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import jax.numpy as jnp
from ml_collections import ConfigDict

from examples.electropolish2d.configs.train_operator import get_config
from examples.electropolish2d.online_conditions import make_sampler
from examples.electropolish2d.spatial_sampling import SpatialSampler, pad_subgraph
from nonlocal_pinn.operators import build_D


class SpatialSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, .5, 5), indexing='xy')
        cls.points = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        cls.op = build_D(jnp.asarray(cls.points), .18, method='sparse', kernel='cubic')
        cls.free = (cls.points[:, 1] > 0) & (cls.points[:, 1] < .5)
        cls.bc = np.flatnonzero(~cls.free)

    def config(self, mode, workers):
        c = get_config()
        c.sample.K_core, c.sample.K_pool = 4, 3
        c.sample.spatial = ConfigDict(dict(mode=mode, max_nodes=36,
            max_edges=32, workers=workers, prefetch=3))
        return c

    def sampler(self, mode, workers):
        return SpatialSampler(self.op, self.points, self.free, np.array([5, 15, 25]),
                              self.bc, self.config(mode, workers))

    def test_full_and_sampled_core_rows_match_original_operator(self):
        values = np.random.default_rng(19).normal(size=len(self.points))
        full = np.asarray(self.op._diag)*values
        np.add.at(full, self.op._row_idx_np, self.op._vals_np*values[self.op._col_idx_np])
        for mode in ('full', 'fixed_pool', 'rolling_pool'):
            with self.sampler(mode, 2) as sampler:
                _, geometry = make_sampler(self.config(mode, 2))
                for step in range(8):
                    ids, points, sub, mask, nearest = sampler.take(step)
                    ids, mask = np.asarray(ids), np.asarray(mask)
                    row, col, weights, diagonal = map(np.asarray, sub)
                    u = values[ids[:len(mask)]]
                    actual = diagonal*u
                    np.add.at(actual, row, weights*u[col])
                    np.testing.assert_allclose(actual[mask], full[ids[:len(mask)][mask]], rtol=2e-6, atol=2e-6)
                    np.testing.assert_array_equal(points, self.points[ids])
                    # Reuse the original SDF geometry helper as an independent
                    # check that a replaced entry does not keep stale lookups.
                    np.testing.assert_array_equal(nearest, geometry(points))
                    self.assertTrue(np.all(self.free[ids[:len(mask)][mask]]))

    def test_worker_order_and_initial_pool_do_not_change_draw_identity(self):
        original = SpatialSampler._build
        def delayed(sampler, draw):
            if draw >= 3:
                time.sleep(.003*(draw % 3))
            return original(sampler, draw)
        traces = []
        for workers in (1, 3):
            with patch.object(SpatialSampler, '_build', delayed):
                with self.sampler('rolling_pool', workers) as sampler:
                    entries = [sampler.take(i) for i in range(12)]
                    traces.append(sampler.trace())
            with self.sampler('fixed_pool', 1) as fixed:
                for i in range(3):
                    for a, b in zip(entries[i], fixed.take(i), strict=True):
                        if isinstance(a, tuple):
                            for aa, bb in zip(a, b, strict=True):
                                np.testing.assert_array_equal(aa, bb)
                        else:
                            np.testing.assert_array_equal(a, b)
        for name in traces[0]:
            np.testing.assert_array_equal(traces[0][name], traces[1][name])
        np.testing.assert_array_equal(traces[0]['draw_ids'], np.arange(12))
        self.assertFalse(np.array_equal(traces[0]['core_ids'][0], traces[0]['core_ids'][3]))

    def test_overflow_is_rejected_without_truncation(self):
        raw = self.op.sub_sample(0, 4)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            pad_subgraph(raw, raw[3]-1, len(raw[1][0]))
        with self.assertRaisesRegex(ValueError, 'overflow'):
            pad_subgraph(raw, raw[3], len(raw[1][0])-1)
        c = self.config('rolling_pool', 1)
        c.sample.spatial.max_edges = 1
        with self.assertRaisesRegex(ValueError, 'degree bounds'):
            SpatialSampler(self.op, self.points, self.free, np.array([1]), self.bc, c)

    def test_background_failure_propagates_and_pool_closes(self):
        original = SpatialSampler._build
        def failing(sampler, draw):
            if draw == 3:
                raise RuntimeError('deliberate producer failure')
            return original(sampler, draw)
        with patch.object(SpatialSampler, '_build', failing):
            with self.assertRaisesRegex(RuntimeError, 'deliberate producer failure'):
                with self.sampler('rolling_pool', 1) as sampler:
                    for i in range(4):
                        sampler.take(i)
        self.assertTrue(sampler._executor._shutdown)


if __name__ == '__main__':
    unittest.main()
