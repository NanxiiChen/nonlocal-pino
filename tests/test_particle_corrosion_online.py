"""CPU checks for deterministic rolling geometry and complete spatial pools."""
from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from ml_collections import ConfigDict
from scipy.spatial import cKDTree

from examples.particle_corrosion2d import online
from examples.particle_corrosion2d.configs.train_base import get_config
from examples.particle_corrosion2d.geometry import sample_geometry
from examples.particle_corrosion2d import geometry as geometry_module


def small_config():
    c = get_config()
    c.particles.count = c.particles.minimum_near_pit_count = 1
    c.mesh.spacing, c.mesh.boundary_points = .06, 32
    c.nl.delta, c.nl.visibility_points = .15, 32
    c.nl.max_halo, c.nl.max_neighbors = 512, 128
    c.sample.K_core, c.sample.K_pool = 8, 2
    c.operator.geometry_count = 4
    c.operator.domain_shape = (5, 7)
    c.operator.wall_points_per_particle = c.operator.wall_points_per_side = 8
    c.operator.mouth_points, c.operator.ic_points = 4, 16
    c.operator.online = ConfigDict(dict(workers=2, geometry_every=4, geometry_size=2,
        spatial_every=2, spatial_size=2, prefetch=2, halo=512, neighbors=128, worker_cache=2))
    c.train.iterations = 9
    return c


class ReverseExecutor:
    """Complete submitted work in reverse order when any result is requested."""
    def __init__(self, **kwargs):
        self.pending = []
        self.completion = []
        self.peak = 0

    def submit(self, fn, *args):
        owner = self
        class Ticket(Future):
            def result(self, timeout=None):
                owner.finish()
                return super().result(timeout=timeout)
        future = Ticket()
        self.pending.append((future, fn, args))
        self.peak = max(self.peak, len(self.pending))
        return future

    def finish(self):
        while self.pending:
            future, fn, args = self.pending.pop()
            try:
                future.set_result(fn(*args))
            except Exception as error:
                future.set_exception(error)
            self.completion.append((args[2], args[3]))

    def shutdown(self, wait, cancel_futures):
        for future, _, _ in self.pending:
            future.cancel()
        self.pending.clear()


class RollingInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = small_config()
        cls.seed = 314159
        cls.records = [sample_geometry(cls.c, np.random.default_rng(i+70)) for i in range(4)]
        cls.base, cls.metadata = online.build_input(cls.c.to_dict(), cls.seed, 0, 0, None)

    def test_replay_distinct_geometry_and_independent_spatial_refresh(self):
        c = self.c
        online.build_input(c.to_dict(), self.seed, 0, 0, None)
        same, meta = online.build_input(c.to_dict(), self.seed, 0, 0, None)
        for key in self.base:
            np.testing.assert_array_equal(same[key], self.base[key])
        self.assertEqual(meta['geometry'], self.metadata['geometry'])
        self.assertTrue(meta['cache_hit'])
        with patch.object(online, 'sample_geometry', side_effect=AssertionError('cache must skip prior')):
            new_space, space_meta = online.build_input(c.to_dict(), self.seed, 0, 1, None)
        self.assertEqual(space_meta['geometry'], meta['geometry'])
        for name in ('domain_tokens', 'domain_weights', 'boundary_tokens', 'boundary_weights'):
            np.testing.assert_array_equal(new_space[name], same[name])
        self.assertFalse(np.array_equal(new_space['ic_points'], same['ic_points']))
        self.assertFalse(np.array_equal(new_space['graph_points'], same['graph_points']))
        online._CACHE.clear()
        cold_space, cold_meta = online.build_input(c.to_dict(), self.seed, 0, 1, None)
        self.assertFalse(cold_meta['cache_hit'])
        for key in new_space:np.testing.assert_array_equal(new_space[key], cold_space[key])
        _, other = online.build_input(c.to_dict(), self.seed, 1, 0, None)
        self.assertNotEqual(online.geometry_identity(other['geometry']), online.geometry_identity(meta['geometry']))
        initial, initial_meta = online.build_input(c.to_dict(), self.seed, -1, 1, self.records[0])
        self.assertEqual(initial_meta['geometry'], self.records[0])
        self.assertEqual(set(initial), set(online.FIXED_KEYS+online.GRAPH_KEYS))
        self.assertFalse(initial_meta['evolved_fields_generated'])
        self.assertEqual(initial_meta['pde_solves'], 0)
        with self.assertRaisesRegex(ValueError, 'revision zero'):
            online.build_input(c.to_dict(), self.seed, -1, 0, self.records[0])

    def test_complete_halos_and_bounded_cache(self):
        data, metadata = online.build_input(self.c.to_dict(), self.seed, 3, 0, None)
        _, mesh, matrix, _ = next(reversed(online._CACHE.values()))
        tree = cKDTree(mesh['points'])
        for pool in range(self.c.sample.K_pool):
            distance, halo_ids = tree.query(data['graph_points'][pool])
            self.assertLess(distance.max(), 1e-7)
            for i, core_index in enumerate(data['core_indices'][pool]):
                full_row = matrix.getrow(halo_ids[core_index])
                selected = full_row.data < 0
                positive = data['coefficients'][pool, i] > 0
                actual_ids = halo_ids[data['neighbors'][pool, i, positive]]
                order = np.argsort(actual_ids)
                np.testing.assert_array_equal(actual_ids[order], full_row.indices[selected])
                np.testing.assert_array_equal(data['coefficients'][pool, i, positive][order],
                                              (-full_row.data[selected]).astype(np.float32))
        self.assertLessEqual(len(online._CACHE), self.c.operator.online.worker_cache)
        self.assertLessEqual(metadata['actual_halo'], self.c.operator.online.halo)

    def test_padding_retains_old_values_and_overflow_is_fatal(self):
        h, k = self.metadata['actual_halo'], self.metadata['actual_neighbors']
        old = {name: np.stack([value]*4) for name, value in self.base.items()}
        old['graph_points'] = old['graph_points'][:, :, :h].copy()
        old['neighbors'] = old['neighbors'][..., :k].copy()
        old['coefficients'] = old['coefficients'][..., :k].copy()
        padded = online.pad_initial_bank(old, self.c)
        np.testing.assert_array_equal(padded['graph_points'][:, :, :h], old['graph_points'])
        for name in ('neighbors', 'coefficients'):
            np.testing.assert_array_equal(padded[name][..., :k], old[name])
            self.assertFalse(padded[name][..., k:].any())
        for name in online.FIXED_KEYS+('core_indices', 'pde_weights'):
            np.testing.assert_array_equal(padded[name], old[name])
        c = ConfigDict(self.c.to_dict())
        c.operator.online.halo = 2
        with self.assertRaisesRegex(ValueError, 'truncation'):
            online.pad_initial_bank(old, c)
        with self.assertRaisesRegex(ValueError, 'no truncation/resampling'):
            online.build_input(c.to_dict(), self.seed, 0, 0, None)
        bad = dict(self.base, fields=np.zeros((1,)))
        with self.assertRaisesRegex(ValueError, 'no fields/labels'):
            online.validate_payload(bad, self.c)

    def test_fifo_coinciding_clocks_and_identical_results(self):
        events = online.plan_events(self.c, 4)
        self.assertEqual([e['completed_updates'] for e in events], [2, 4, 6, 8])
        self.assertEqual(len(events[1]['tasks']), 4)  # Distinct slots at coinciding tick4.
        self.assertEqual(events[-1]['spatial_merged_into_geometry'], [2, 3])
        outputs = []
        for repeat in range(2):
            with tempfile.TemporaryDirectory() as directory, patch.object(online, 'ProcessPoolExecutor', ReverseExecutor):
                with online.RollingPool(self.c, directory, {'geometries': self.records}, self.seed, []) as pool:
                    installed = []
                    executor = pool.executor
                    for step in range(self.c.train.iterations):
                        result = pool.before_step(step)
                        if result is not None:
                            slots, values = result
                            installed.append((slots.copy(), values, pool.draw_ids.copy(), pool.spatial_versions.copy()))
                    state = json.loads(pool.save_state(self.c.train.iterations).read_text())
                    self.assertEqual(state['draw_ids'], [0, 1, 2, 3])
                    self.assertEqual(state['spatial_versions'], [1, 1, 0, 0])
                    self.assertLessEqual(executor.peak, 2*(2+2))
                logs = [json.loads(line) for line in (Path(directory)/'online_refresh.jsonl').read_text().splitlines()]
                self.assertEqual([row['completed_updates'] for row in logs], [2, 4, 6, 8])
                self.assertEqual(pool.summary()['installed_geometries'], 4)
                self.assertEqual(pool.summary()['installed_spatial_refreshes'], 6)
                self.assertNotEqual(executor.completion[0], (-1, 1))
                outputs.append(installed)
        for first, second in zip(*outputs, strict=True):
            for a, b in zip(first[2:], second[2:], strict=True):np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(first[0], second[0])
            for name in first[1]:np.testing.assert_array_equal(first[1][name], second[1][name])

    def test_errors_forbidden_and_stale_results_are_not_hidden(self):
        c = self.c
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                online.RollingPool(c, directory, {'geometries': self.records}, self.seed, [self.records[0]])
            with patch.object(online, 'ProcessPoolExecutor', ReverseExecutor):
                with online.RollingPool(c, directory, {'geometries': self.records}, self.seed, []) as pool:
                    with self.assertRaisesRegex(ValueError, 'Missed'):
                        pool.before_step(3)
        with tempfile.TemporaryDirectory() as directory, patch.object(online, 'ProcessPoolExecutor', ReverseExecutor):
            with patch.object(online, 'build_input', side_effect=RuntimeError('construction failure')):
                with online.RollingPool(c, directory, {'geometries': self.records}, self.seed, []) as pool:
                    with self.assertRaisesRegex(RuntimeError, 'construction failure'):
                        pool.before_step(2)
        with tempfile.TemporaryDirectory() as directory, patch.object(online, 'ProcessPoolExecutor', ReverseExecutor):
            with online.RollingPool(c, directory, {'geometries': self.records}, self.seed, []) as pool:
                pool.spatial_versions[0] = 8
                with self.assertRaisesRegex(ValueError, 'Stale spatial'):
                    pool.before_step(2)

    def test_real_spawn_workers_return_cpu_only_known_inputs(self):
        c = ConfigDict(self.c.to_dict())
        c.operator.online.geometry_every = 2
        c.operator.online.spatial_every = 1
        c.train.iterations = 3
        with tempfile.TemporaryDirectory() as directory:
            with online.RollingPool(c, directory, {'geometries': self.records}, self.seed, []) as pool:
                self.assertIsNone(pool.before_step(0))
                for step in (1, 2):
                    slots, values = pool.before_step(step)
                    for slot in range(len(slots)):
                        online.validate_payload({name: value[slot] for name, value in values.items()}, c)
                self.assertEqual(pool.summary()['pde_solves'], 0)
                self.assertEqual(pool.summary()['fem_solves'], 0)

    def test_exhausted_budget_continues_same_rng_and_caches_diagnostics(self):
        online._CACHE.clear()
        real = geometry_module.sample_geometry
        seen = []
        draw = 99

        def force_one_exhaustion(c, rng, *, diagnostics):
            seen.append(rng)
            if len(seen) == 1:
                small = ConfigDict(c.to_dict())
                small.particles.max_geometry_attempts = small.particles.max_particle_attempts = 1
                def reject(_, stream):
                    stream.random()
                    return None
                with patch.object(geometry_module, '_draw_particle', side_effect=reject):
                    return real(small, rng, diagnostics=diagnostics)
            return real(c, rng, diagnostics=diagnostics)

        stdout = io.StringIO()
        with patch.object(online, 'sample_geometry', side_effect=force_one_exhaustion), redirect_stdout(stdout):
            data, meta = online.build_input(self.c.to_dict(), self.seed, draw, 0, None)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], seen[1])
        expected_rng = np.random.default_rng(np.random.SeedSequence([self.seed, online.GEOMETRY_NAMESPACE, draw]))
        expected_rng.random()
        self.assertEqual(meta['geometry'], real(self.c, expected_rng))
        np.testing.assert_array_equal(seen[-1].random(8), expected_rng.random(8))
        generation = meta['generation_diagnostics']
        self.assertEqual(generation['budget_restarts'], 1)
        self.assertEqual(generation['sampling_calls'], 2)
        warning = json.loads(stdout.getvalue())
        self.assertEqual(warning['draw_id'], draw)
        self.assertEqual(warning['warning'], 'geometry_sampling_budget_exhausted_continuing_same_rng')
        with patch.object(online, 'sample_geometry', side_effect=AssertionError('cached generation resampled')):
            _, cached = online.build_input(self.c.to_dict(), self.seed, draw, 1, None)
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(cached['generation_diagnostics'], generation)
        online.validate_payload(data, self.c)

    def test_unexpected_sampler_exceptions_do_not_become_retries(self):
        online._CACHE.clear()
        for error_type in (ValueError, RuntimeError):
            with self.subTest(error=error_type):
                with patch.object(online, 'sample_geometry', side_effect=error_type('unexpected sampler failure')) as sampler:
                    with self.assertRaisesRegex(error_type, 'unexpected sampler failure'):
                        online.build_input(self.c.to_dict(), self.seed, 100, 0, None)
                self.assertEqual(sampler.call_count, 1)


if __name__ == '__main__':
    unittest.main()
