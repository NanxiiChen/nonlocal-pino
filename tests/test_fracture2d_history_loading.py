"""Online cold loading, two-snapshot lifecycle, monitoring and exact restart."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.fracture2d.history_training import HistoryDamagePINN
from examples.fracture2d.loading import run
from examples.fracture2d.mesh import load_geometry
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_main_route import fixture


def history_config(root):
    c = fixture(root)
    c.arch.type = 'history_operator'
    c.history = ConfigDict(dict(coarse_cells=2, fine_cells=2, fine_tokens=2,
        width=8, heads=2, point_depth=1, query_depth=1, head_depth=1,
        replay_capacity=8, replay_batch=1, replay_fraction=.25, replay_enabled=True,
        replay_every=1, current_only_steps=0))
    c.train.learning_rate = 1e-3
    c.train.minimum_rate = 1e-3
    c.output.reference_root = ''
    c.data.basis_path = str(root/'intentionally_absent_basis.npz')
    return c


class HistoryLoadingTests(unittest.TestCase):
    def test_sparse_replay_finishing_and_endpoint_resume_keep_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = history_config(root)
            c.train.inner_steps = 16
            c.history.replay_every = 3
            c.history.current_only_steps = 4
            m, op, _ = load_geometry(c)
            p = HistoryDamagePINN(c, m, op)
            H = jnp.array([1., 2., 3., 4.])
            p.begin_load(1, .0001)
            p.solve(H, jnp.zeros(4), .0001)
            p.end_load(2*H)
            path = root/'endpoint'
            path.mkdir()
            p.save(path)
            restored = HistoryDamagePINN(c, m, op)
            restored.restore(path, p.count)
            p.begin_load(2, .0002)
            restored.begin_load(2, .0002)
            flags = []
            original = p.step
            def record(*args):
                flags.append(bool(args[-1]))
                return original(*args)
            p.step = record
            expected, _ = p.solve(3*H, jnp.zeros(4), .0002)
            actual, _ = restored.solve(3*H, jnp.zeros(4), .0002)
            self.assertEqual([i for i, flag in enumerate(flags) if flag], [0, 3, 6, 9])
            self.assertFalse(any(flags[-4:]))
            self.assertEqual(p.count, 32)
            self.assertEqual(restored.count, 32)
            np.testing.assert_array_equal(actual, expected)
            for a, b in zip(jax.tree.leaves(p.state), jax.tree.leaves(restored.state)):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(int(p.replay.count), 3)
            c.history.current_only_steps = 17
            with self.assertRaisesRegex(ValueError, 'finishing'):
                HistoryDamagePINN(c, m, op)

    def test_collection_is_once_at_start_and_end_and_store_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = history_config(root)
            m, op, _ = load_geometry(c)
            p = HistoryDamagePINN(c, m, op)
            self.assertEqual(p.count, 0)
            self.assertEqual(int(p.replay.count), 0)
            H1, H2 = jnp.array([1., 2., 3., 4.]), jnp.array([3., 4., 5., 6.])
            with self.assertRaisesRegex(ValueError, 'begin_load'):
                p.solve(H1, jnp.zeros(4), .0001)
            p.begin_load(1, .0001)
            p.solve(H1, jnp.zeros(4), .0001)
            self.assertEqual(int(p.replay.count), 1)
            p.solve(H2, jnp.zeros(4), .0001)
            self.assertEqual(int(p.replay.count), 1)
            np.testing.assert_array_equal(p.replay.histories[0], H1)
            p.end_load(2*H2)
            self.assertEqual(int(p.replay.count), 2)
            np.testing.assert_array_equal(p.replay.histories[1], 2*H2)
            for name in ('first_save', 'second_save'):
                path = root/name
                path.mkdir()
                p.save(path)
            self.assertEqual(len(list(p.store.glob('snapshot_*.npz'))), 2)
            self.assertFalse((root/'first_save/replay.eqx').exists())
            restored = HistoryDamagePINN(c, m, op)
            restored.restore(root/'second_save', p.count)
            for a, b in zip(jax.tree.leaves(restored.replay), jax.tree.leaves(p.replay)):
                np.testing.assert_array_equal(a, b)
            p.begin_load(2, .0002)
            restored.begin_load(2, .0002)
            expected, _ = p.solve(H2, jnp.zeros(4), .0002)
            actual, _ = restored.solve(H2, jnp.zeros(4), .0002)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(p.count, 6)
            with self.assertRaisesRegex(ValueError, 'order/count'):
                p.begin_load(3, .0003)

    def test_shared_loading_cold_state_own_snapshots_monitor_and_exact_resume(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = history_config(root)
            c.output.tensorboard = True
            c.output.event_dir = str(root/'events')
            c.output.figure_dpi = 40
            with contextlib.redirect_stdout(io.StringIO()):
                result = run(c, damage_solver='pinn')
            report = json.loads(result.read_text())
            self.assertTrue(report['complete'])
            self.assertEqual([r['replay_snapshots'] for r in report['curve']], [2, 4])
            self.assertEqual(report['curve'][-1]['optimizer_count'], 12)
            base = Path(c.output.root)
            with np.load(base/'state_0000/fields.npz') as f:
                for name in ('q', 'd', 'H'):
                    np.testing.assert_array_equal(f[name], 0.)
            cold = json.loads((base/'state_0000/replay_state.json').read_text())
            self.assertEqual(cold['snapshots'], 0)
            self.assertEqual(cold['optimizer_count'], 0)
            self.assertFalse((base/'parameterization.npz').exists())
            for load_index in (1, 2):
                with np.load(base/f'state_{load_index:04d}/outer_0001/fields.npz') as stage:
                    with np.load(base/f'replay_store/snapshot_{2*(load_index-1):04d}.npz') as snap:
                        np.testing.assert_array_equal(snap['H'], stage['H'])
                with np.load(base/f'state_{load_index:04d}/fields.npz') as end:
                    with np.load(base/f'replay_store/snapshot_{2*load_index-1:04d}.npz') as snap:
                        np.testing.assert_array_equal(snap['H'], end['H'])
                stage_report = json.loads((base/f'state_{load_index:04d}/outer_0001/result.json').read_text())
                self.assertEqual(stage_report['replay_snapshots'], 2*load_index-1)
            events = EventAccumulator(c.output.event_dir)
            events.Reload()
            self.assertEqual([s.value for s in events.Scalars('replay/snapshots_at_endpoint')], [2., 4.])
            self.assertEqual(set(events.Tags()['images']), {
                'train/damage_field','train/ux_field','train/uy_field',
                'train/history_field','train/load_reaction_curve'})
            with np.load(base/'state_0002/fields.npz') as f:
                final = {key:f[key].copy() for key in ('q','d','H')}
            c.output.tensorboard = False
            c.output.root = str(root/'interrupted')
            def interrupt(path, data):
                atomic_json(path, data)
                if path.name=='result.json' and 'curve' in data and len(data['curve'])==1:
                    raise InterruptedError('after first complete own endpoint')
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.atomic_json', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    run(c, damage_solver='pinn')
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()):
                run(c, damage_solver='pinn')
            resumed = Path(c.output.root)
            with np.load(resumed/'state_0002/fields.npz') as f:
                for name in final:
                    np.testing.assert_array_equal(f[name], final[name])
            for i in range(4):
                with np.load(base/f'replay_store/snapshot_{i:04d}.npz') as a:
                    with np.load(resumed/f'replay_store/snapshot_{i:04d}.npz') as b:
                        for name in ('H','fine_ids'):
                            np.testing.assert_array_equal(a[name], b[name])


if __name__ == '__main__':
    unittest.main()
