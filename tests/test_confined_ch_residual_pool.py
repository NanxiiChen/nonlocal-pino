"""Rolling row identity, geometry invalidation and canonical trainer integration."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.confined_ch2d.operator_residual_pool import RollingResidualRows, validate_residual_rows
from examples.confined_ch2d.operator_problem import validate, sample_geometries
from examples.confined_ch2d.operator_training import train_operator
from tests.test_confined_ch_operator import small_fixture


def fixture():
    c, data = small_fixture()
    c.train.seed = 42
    c.operator.train_conditions = 2
    c.operator.test_conditions = 1
    c.operator.condition_batch = 2
    c.operator.local_validation_indices = ()
    c.operator.metric = 'linear_stable_resolvent'
    c.operator.metric_workers = 1
    c.sample.residual_rows = ConfigDict(dict(count=8, pools=3, refresh_every=2, workers=2))
    return c, {key: np.stack([np.asarray(value)]*3) for key, value in data.items()}


class ResidualPoolTests(unittest.TestCase):
    def test_prefetch_order_replacement_and_epoch(self):
        c, bank = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'a').mkdir()
            (Path(tmp)/'b').mkdir()
            a = RollingResidualRows(c, bank, Path(tmp)/'a')
            b = RollingResidualRows(c, bank, Path(tmp)/'b')
            try:
                # A cached pre-replacement draw may already have completed.
                a.prefetch(np.array([0, 1, 0]), 1)
                replacement = {k: v[:1].copy() for k, v in bank.items()}
                replacement['volumes'][0] = [0., 0., 1., 0.]
                for pool in (a, b):
                    pool.install(np.array([0]), replacement)
                for step, ids in ((1, [0, 1, 0]), (2, [1, 0]), (3, [0, 0]), (5, [1, 0])):
                    x, y = a.batch(np.array(ids), step), b.batch(np.array(ids), step)
                    for key in x:
                        np.testing.assert_array_equal(x[key], y[key])
                    np.testing.assert_array_equal(x['residual_core_indices'][np.array(ids) == 0], 2)
                    np.testing.assert_allclose(x['residual_core_weights'].sum(axis=1), 1.)
                    self.assertEqual(a.cache[0][0], (1, (step-1)//2))
                    a.prefetch(np.array([1, 0]), step+1)
                self.assertEqual(a.summary()['geometry_installations'], 1)
                self.assertLessEqual(sum(len(v[1]) for v in a.cache.values()), 2*3)
                self.assertFalse(a.data[0]['volumes'].flags.writeable)
                self.assertEqual(set(a.data[0]), {'points', 'volumes'})
            finally:
                a.close()
                b.close()
            with self.assertRaises(ValueError):
                a.batch(np.array([0]), 6)

    def test_no_dense_host_copy_and_atomic_invalid_replacement(self):
        c, bank = fixture()
        class ForbiddenConversion:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Dense metric must not be copied to the host pool')
        bank['metric'] = ForbiddenConversion()
        with tempfile.TemporaryDirectory() as tmp:
            pool = RollingResidualRows(c, bank, tmp)
            try:
                before = pool.batch(np.array([0]), 1)
                bad = dict(points=bank['points'][:1], volumes=np.zeros_like(bank['volumes'][:1]))
                with self.assertRaises(ValueError):
                    pool.install(np.array([0]), bad)
                self.assertEqual(pool.versions.sum(), 0)
                after = pool.batch(np.array([0]), 1)
                np.testing.assert_array_equal(before['residual_core_indices'], after['residual_core_indices'])
                wrong_dtype = dict(points=bank['points'][:1].astype(np.float16), volumes=bank['volumes'][:1])
                with self.assertRaises(ValueError):
                    pool.install(np.array([0]), wrong_dtype)
                for ids in ([2], [0.5], []):
                    with self.assertRaises(ValueError):
                        pool.prefetch(np.asarray(ids), 1)
            finally:
                pool.close()

    def test_configuration_rejects_mixed_semantics_and_preserves_default(self):
        from examples.confined_ch2d.configs.train_operator import get_config
        from examples.confined_ch2d.configs.train_operator_residual_rows import get_config as control
        original, changed = get_config(), control()
        validate(changed)
        self.assertNotIn('residual_rows', original.sample)
        del changed.sample.residual_rows
        changed.train.iterations = original.train.iterations
        changed.log.run_name = original.log.run_name
        self.assertEqual(changed.to_dict(), original.to_dict())
        c, _ = fixture()
        for section, key, value in ((c.operator, 'formulation', 'conservative_potential'),
                                    (c.sample, 'spatial', ConfigDict()),
                                    (c.train, 'continuation', ConfigDict()),
                                    (c.operator, 'augmentation', 'axis_reflections')):
            section[key] = value
            with self.assertRaises(ValueError):
                validate_residual_rows(c)
            del section[key]
        c.sample.residual_rows.count = 0
        with self.assertRaises(ValueError):
            validate(c)

    def test_canonical_short_cold_training_records_actual_rows(self):
        # Exercise real loss, optimizer, logging and saving; only bank loading
        # is replaced by a tiny known graph, with no claim about solution accuracy.
        with jax.enable_x64(True):
            c, bank = fixture()
            c.train.iterations = 3
            c.sample.Nt = 2
            c.log.eval_every = 3
            c.log.checkpoint_every = 3
            c.reference.snapshots = 3
            times = np.linspace(0., c.physical.terminal_time, 3)
            holes = sample_geometries(c)
            refs = []
            for i in range(3):
                refs.append(dict(points=bank['points'][i, :3], volumes=bank['volumes'][i, :3],
                    time=times, holes=holes[i], phi=np.tile(bank['initial'][i, :3], (3, 1)),
                    mu=np.full((3, 3), .1), mesh_points=bank['points'][i, :3],
                    mesh_cells=np.array([[0, 1, 2]])))
            manifest = dict(train_indices=[0, 1], test_indices=[2])
            with tempfile.TemporaryDirectory() as tmp:
                c.log.log_dir = tmp
                for label in ('full', 'rows'):
                    run_config = ConfigDict(c.to_dict())
                    run_config.log.run_name = label
                    if label == 'full':
                        del run_config.sample.residual_rows
                    with patch('examples.confined_ch2d.operator_training.load_bank',
                               return_value=(manifest, refs, jax.tree.map(jnp.asarray, bank))), redirect_stdout(io.StringIO()):
                        train_operator(run_config)
                full, rows = Path(tmp)/'full', Path(tmp)/'rows'
                self.assertEqual((full/'model_initial.eqx').read_bytes(), (rows/'model_initial.eqx').read_bytes())
                with np.load(full/'training_draws.npz') as a, np.load(rows/'training_draws.npz') as b:
                    np.testing.assert_array_equal(a['condition_ids'], b['condition_ids'])
                    conditions = b['condition_ids'].copy()
                with np.load(rows/'residual_row_draws.npz') as saved:
                    self.assertEqual(saved['core_ids'].shape, (3, 2, 8))
                    self.assertTrue(np.all(saved['core_ids'] < 3))
                    np.testing.assert_allclose(saved['core_weights'].sum(), 1.)
                recorded = [json.loads(line) for line in (rows/'residual_row_inputs.jsonl').read_text().splitlines()]
                self.assertEqual([r['step'] for r in recorded], [1, 2, 3])
                self.assertEqual(recorded[-1]['samples'][0]['epoch'], 1)
                self.assertEqual(json.loads((rows/'summary.json').read_text())['new_updates'], 3)
                self.assertFalse((full/'residual_row_draws.npz').exists())
                from docs.tasks.pino_spinodal_row_audit import row_replay
                audit = row_replay(rows, json.loads((rows/'config.json').read_text()), conditions, -1-conditions)
                self.assertEqual(audit['scalar_row_indices'], 3*2*8)
                # The independent replay must reject a stale/tampered geometry
                # identity, not merely check that indices are in bounds.
                wrong = -1-conditions
                wrong[0, 0] = int(conditions[0, 0])
                with self.assertRaises(ValueError):
                    row_replay(rows, json.loads((rows/'config.json').read_text()), conditions, wrong)


if __name__ == '__main__':
    unittest.main()
