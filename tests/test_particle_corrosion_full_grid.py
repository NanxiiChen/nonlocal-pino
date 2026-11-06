"""CPU validation of complete mesh quadrature and matched rolling IC inputs."""
import json
import unittest

import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from examples.particle_corrosion2d import full_grid, online
from examples.particle_corrosion2d.geometry import sample_geometry
from tests.test_particle_corrosion_online import small_config


def tiny_full():
    c = small_config()
    c.sample.full_grid = ConfigDict(dict(nodes=8, neighbors=3))
    return c


def graph_fixture():
    mesh = dict(points=np.array([[0., .1], [.2, .1], [.4, .2], [.7, .2], [1., .1]]),
        volumes=np.array([.1, .2, .15, .25, .3]), mouth=np.array([True, False, False, False, True]))
    exchange = sparse.diags([np.ones(4), np.ones(4)], [-1, 1], shape=(5, 5)).tocsr()
    matrix = sparse.diags(1/mesh['volumes']) @ (sparse.diags(np.asarray(exchange.sum(axis=1)).ravel())-exchange)
    return mesh, matrix.tocsr()


class FullGraphTests(unittest.TestCase):
    def test_physical_rows_operator_action_and_volume_mean(self):
        import jax
        import jax.numpy as jnp
        from examples.particle_corrosion2d.losses import graph_action, weighted_mean
        self.assertTrue(all(device.platform == 'cpu' for device in jax.devices()))
        c, (mesh, matrix) = tiny_full(), graph_fixture()
        original = matrix.copy()
        graph = full_grid.full_graph(c, mesh, matrix)
        free = np.flatnonzero(~mesh['mouth'])
        active = np.flatnonzero(graph['pde_weights'] > 0)
        np.testing.assert_array_equal(active, free)
        np.testing.assert_array_equal(graph['core_indices'][active], free)
        np.testing.assert_array_equal(graph['pde_weights'][active], mesh['volumes'][free].astype(np.float32))
        np.testing.assert_array_equal(graph['graph_points'][:5], mesh['points'].astype(np.float32))
        np.testing.assert_array_equal(matrix.data, original.data)
        values = np.array([[.8, -.3], [-.5, 2.], [2., 1.], [1., -.7], [3., .9]], dtype=np.float32)
        padded = np.pad(values, ((0, 3), (0, 0)), constant_values=400)
        action = np.asarray(graph_action(jnp.asarray(padded), {key: jnp.asarray(value) for key, value in graph.items()}))
        expected = matrix@values
        np.testing.assert_allclose(action[free], expected[free], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(action[np.r_[0, 4, 5, 6, 7]], 0)
        # Soft mouth predictions participate in neighboring free rows unchanged.
        zero_mouth = values.copy();zero_mouth[mesh['mouth']] = 0
        self.assertFalse(np.allclose(action[free], (matrix@zero_mouth)[free]))
        measured = np.abs(action).T
        measured[:, graph['pde_weights'] == 0] = 1e6
        average = float(weighted_mean(jnp.asarray(measured), jnp.asarray(graph['pde_weights'])))
        target = np.mean(np.sum(np.abs(expected[free]).T*mesh['volumes'][free], axis=-1)/mesh['volumes'][free].sum())
        self.assertAlmostEqual(average, target, places=5)
        constant = np.ones_like(padded)
        np.testing.assert_array_equal(np.asarray(graph_action(jnp.asarray(constant), graph)), 0)

    def test_no_node_or_neighbor_truncation_or_operator_repair(self):
        mesh, matrix = graph_fixture()
        c = tiny_full();c.sample.full_grid.nodes = 4
        with self.assertRaisesRegex(ValueError, 'truncation forbidden'):
            full_grid.full_graph(c, mesh, matrix)
        c = tiny_full();c.sample.full_grid.neighbors = 1
        with self.assertRaisesRegex(ValueError, 'truncation forbidden'):
            full_grid.full_graph(c, mesh, matrix)
        c = tiny_full();bad = matrix.copy();bad[1, 0] = .1
        with self.assertRaisesRegex(ValueError, 'sign convention'):
            full_grid.full_graph(c, mesh, bad)
        bad = matrix.copy();bad[1, 1] += .1
        with self.assertRaisesRegex(ValueError, 'annihilate constants'):
            full_grid.full_graph(c, mesh, bad)
        self.assertFalse(full_grid.is_full(small_config()))
        c.nl.method = 'local_ad'
        with self.assertRaisesRegex(ValueError, 'same nonlocal'):
            full_grid.validate_full(c)


class FullRollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = small_config()
        cls.core.operator.geometry_count = 2
        cls.full = ConfigDict(cls.core.to_dict())
        # Explicit full capacity is independent of the old max_halo=512.
        cls.full.sample.full_grid = ConfigDict(dict(nodes=768, neighbors=128))
        cls.seed = 71213

    def test_online_full_and_core_match_all_fixed_draws(self):
        core, core_meta = online.build_input(self.core.to_dict(), self.seed, 11, 1, None)
        full, full_meta = online.build_input(self.full.to_dict(), self.seed, 11, 1, None)
        for key in online.FIXED_KEYS:
            np.testing.assert_array_equal(full[key], core[key])
        self.assertEqual(full_meta['geometry'], core_meta['geometry'])
        self.assertEqual(full_meta['generation_diagnostics'], core_meta['generation_diagnostics'])
        self.assertEqual(full_meta['spatial_seed_words'], core_meta['spatial_seed_words'])
        self.assertEqual(full['graph_points'].shape, (1, 768, 2))
        self.assertEqual(full['core_indices'].shape, (1, 768))
        self.assertEqual(full_meta['active_pde_nodes'], np.count_nonzero(full['pde_weights']))
        online.validate_payload(full, self.full)
        refreshed, metadata = online.build_input(self.full.to_dict(), self.seed, 11, 2, None)
        for key in online.GRAPH_KEYS:
            np.testing.assert_array_equal(refreshed[key], full[key])
        self.assertFalse(np.array_equal(refreshed['ic_points'], full['ic_points']))
        self.assertEqual(metadata['geometry'], full_meta['geometry'])
        replay, _ = online.build_input(self.core.to_dict(), self.seed, 11, 1, None)
        for key in core:
            np.testing.assert_array_equal(replay[key], core[key])
        bad = {key: value.copy() for key, value in full.items()}
        inactive = np.flatnonzero(bad['pde_weights'][0] == 0)[0]
        bad['coefficients'][0, inactive, 0] = 1
        with self.assertRaisesRegex(ValueError, 'zero inactive edges'):
            online.validate_payload(bad, self.full)
        bad = {key: value.copy() for key, value in full.items()}
        active = np.flatnonzero(bad['pde_weights'][0] > 0)
        bad['core_indices'][0, active[1]] = bad['core_indices'][0, active[0]]
        with self.assertRaisesRegex(ValueError, 'cannot be repeated'):
            online.validate_payload(bad, self.full)

    def test_initial_bank_reconstructs_only_graphs_with_real_cpu_workers(self):
        records = [sample_geometry(self.core, np.random.default_rng(seed)) for seed in (70, 71)]
        slots = [online.build_input(self.core.to_dict(), self.seed, -1-i, 1, record)[0]
                 for i, record in enumerate(records)]
        bank = {key: np.stack([slot[key] for slot in slots]) for key in slots[0]}
        original = {key: value.copy() for key, value in bank.items()}
        manifest = dict(config=json.loads(self.core.to_json()), geometries=records)
        result = full_grid.prepare_initial_full_bank(bank, self.full, manifest, 2)
        for key in online.FIXED_KEYS:
            np.testing.assert_array_equal(result[key], bank[key])
        for key in bank:
            np.testing.assert_array_equal(bank[key], original[key])
        for index, record in enumerate(records):
            exact = full_grid.build_initial_full_slot(self.full.to_dict(), record)
            for key in online.GRAPH_KEYS:
                np.testing.assert_array_equal(result[key][index], exact[key])
            online.validate_payload({key: value[index] for key, value in result.items()}, self.full)
        with self.assertRaisesRegex(ValueError, 'explicit prepare_initial_full_bank'):
            online.pad_initial_bank(bank, self.full)


if __name__ == '__main__':
    unittest.main()
