"""Physical sampling measures and complete, padded nonlocal row payloads."""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse

from examples.particle_corrosion2d.configs.train_base import get_config
from examples.particle_corrosion2d.geometry import boundary, sample_geometry
from examples.particle_corrosion2d.inputs import (
    conditioning, graph_payload, initial_payload, mouth_quadrature, pad_graph,
    wall_quadrature, mesh_mouth_quadrature, mouth_sampling_policy)
from examples.particle_corrosion2d.problem import initial_fields


def small_problem():
    points = np.column_stack((np.linspace(.1, .9, 5), np.full(5, .1)))
    volumes = np.array([.02, .04, .08, .16, .32])
    kernel = 1/(1+np.linalg.norm(points[:, None]-points[None], axis=-1))
    np.fill_diagonal(kernel, 0)
    exchange = kernel*volumes[None]
    matrix = sparse.csr_matrix(np.diag(exchange.sum(axis=1))-exchange)
    mesh = dict(points=points, volumes=volumes, mouth=np.array([True, False, False, False, False]))
    return mesh, matrix


class ParticleInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = get_config()
        cls.record = sample_geometry(cls.c, np.random.default_rng(8326))

    def test_boundary_measures_and_domain_outward_normals(self):
        c, record = self.c, self.record
        points, normals, weights, kinds = wall_quadrature(c, record)
        mouth, mouth_weights = mouth_quadrature(c)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=2e-15)
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(mouth_weights.sum(), 2*c.geometry.mouth_half_width)
        self.assertTrue(np.all(initial_fields(c, mouth) == 0))
        exterior = kinds == 0
        expected_exterior = 2*(c.geometry.length+c.geometry.height)-mouth_weights.sum()
        self.assertAlmostEqual(weights[exterior].sum(), expected_exterior)
        n = c.operator.wall_points_per_particle
        begin = np.count_nonzero(exterior)
        theta = 2*np.pi*(np.arange(n)+.5)/n
        for index, particle in enumerate(record['particles']):
            chosen = slice(begin+index*n, begin+(index+1)*n)
            expected_points, ceramic_normal, _ = boundary(particle, theta)
            np.testing.assert_allclose(points[chosen], expected_points)
            np.testing.assert_allclose(normals[chosen], -ceramic_normal)
            self.assertTrue((np.sum(normals[chosen]*(points[chosen]-particle['center']), axis=1) < 0).all())
            fine = boundary(particle, np.arange(16384)*2*np.pi/16384)[0]
            independent_length = np.linalg.norm(np.roll(fine, -1, axis=0)-fine, axis=1).sum()
            self.assertAlmostEqual(weights[chosen].sum()/independent_length, 1, delta=2e-6)
        data = conditioning(c, record)
        self.assertEqual(data['domain_tokens'].shape, (np.prod(c.operator.domain_shape), 3))
        self.assertEqual(data['boundary_tokens'].shape, (len(points)+len(mouth), 5))
        self.assertTrue((data['domain_weights'] == 0).any())
        self.assertTrue((data['domain_weights'] > 0).any())
        np.testing.assert_allclose(data['boundary_weights'].sum(), weights.sum()+mouth_weights.sum())

    def test_complete_rows_and_padding_are_exact(self):
        c = get_config()
        c.sample.K_core = 17
        mesh, matrix = small_problem()
        data = graph_payload(c, mesh, matrix, np.random.default_rng(904))
        full_values = np.column_stack((np.sin(mesh['points'][:, 0]), mesh['points'][:, 0]**3))
        # Every tiny graph is dense, so complete halo must include the mouth as
        # a neighbor even though it is excluded from the sampled PDE core.
        np.testing.assert_array_equal(data['graph_points'], mesh['points'])
        self.assertFalse(np.any(mesh['mouth'][data['core_indices']]))
        core = data['core_indices']
        actual = np.sum(data['coefficients'][..., None]*(full_values[core, None]
                         -full_values[data['neighbors']]), axis=1)
        np.testing.assert_allclose(actual, (matrix@full_values)[core], atol=1e-16)
        padded = pad_graph(data, 9, 12)
        padded_values = np.tile(full_values[0], (9, 1))
        padded_values[:len(full_values)] = full_values
        padded_action = np.sum(padded['coefficients'][..., None]*(padded_values[core, None]
                                -padded_values[padded['neighbors']]), axis=1)
        np.testing.assert_allclose(padded_action, actual, atol=1e-16)
        np.testing.assert_array_equal(padded['pde_weights'], np.ones(len(core)))

    def test_mass_proportional_core_and_ic_sampling(self):
        c = get_config()
        c.sample.K_core = 8192
        c.operator.ic_points = 8192
        mesh, matrix = small_problem()
        data = graph_payload(c, mesh, matrix, np.random.default_rng(572))
        core_x = data['graph_points'][data['core_indices'], 0]
        free = ~mesh['mouth']
        expected = np.average(mesh['points'][free, 0], weights=mesh['volumes'][free])
        self.assertAlmostEqual(core_x.mean(), expected, delta=.008)
        self.assertGreater(abs(expected-mesh['points'][free, 0].mean()), .1)
        ic = initial_payload(c, mesh, np.random.default_rng(692))
        expected_ic = np.average(mesh['points'][:, 0], weights=mesh['volumes'])
        self.assertAlmostEqual(ic['ic_points'][:, 0].mean(), expected_ic, delta=.008)
        np.testing.assert_array_equal(ic['ic_weights'], np.ones(c.operator.ic_points))
        np.testing.assert_allclose(ic['initial_ic'], initial_fields(c, ic['ic_points']))

    def test_overflow_is_explicit(self):
        c = get_config()
        mesh, matrix = small_problem()
        c.nl.max_halo = 3
        with self.assertRaisesRegex(ValueError, 'overflow'):
            graph_payload(c, mesh, matrix, np.random.default_rng(2))
        c.nl.max_halo, c.nl.max_neighbors = 128, 2
        with self.assertRaisesRegex(ValueError, 'overflow'):
            graph_payload(c, mesh, matrix, np.random.default_rng(2))
        c.nl.max_neighbors = 128
        data = graph_payload(c, mesh, matrix, np.random.default_rng(2))
        with self.assertRaisesRegex(ValueError, 'too small'):
            pad_graph(data, 3, 128)

    def test_mouth_nodes_have_physical_measure_and_no_unconstrained_endpoints(self):
        c = get_config()
        points = np.array([[.445, 0.], [.49, 0.], [.555, 0.], [.5, .1]])
        mesh = dict(points=points, mouth=np.array([True, True, True, False]),
                    boundary_edges=np.array([[0, 1], [1, 2], [2, 3]]),
                    boundary_groups=np.array([1, 1, 0]))
        query, measure = mesh_mouth_quadrature(c, mesh)
        active = measure > 0
        np.testing.assert_array_equal(query[active], points[:3])
        np.testing.assert_allclose(measure[:3], [.0225, .055, .0325], atol=1e-15)
        self.assertAlmostEqual(measure.sum(), .11)
        self.assertAlmostEqual(measure@query[:, 0], .11*.5)
        # A smooth trace can vanish at every old midpoint yet violate both
        # operator boundary endpoints. The nodal loss must detect that trace.
        mid, mid_weights = mouth_quadrature(c)
        def trace(x):
            return np.prod((x[:, None]-mid[:, 0])/(.445-mid[:, 0]), axis=1)
        self.assertEqual(float(mid_weights@trace(mid[:, 0])**2), 0.)
        self.assertGreater(float(measure@trace(query[:, 0])**2), .05)
        self.assertTrue(np.all(measure[3:] == 0))
        c.operator.mouth_points = 2
        with self.assertRaisesRegex(ValueError, 'overflow'):
            mesh_mouth_quadrature(c, mesh)
        c.operator.mouth_points = 16
        mesh['mouth'][1] = False
        with self.assertRaisesRegex(ValueError, 'exactly'):
            mesh_mouth_quadrature(c, mesh)

    def test_mouth_option_preserves_encoder_and_random_ic_draws(self):
        from examples.particle_corrosion2d.mesh import build_mesh
        c = get_config()
        mesh = build_mesh(c, self.record, c.mesh.spacing)
        old_condition = conditioning(c, self.record)
        legacy = initial_payload(c, mesh, np.random.default_rng(94))
        self.assertEqual(mouth_sampling_policy(c), 'midpoint')
        c.operator.mouth_sampling = 'mesh_dirichlet'
        updated = initial_payload(c, mesh, np.random.default_rng(94))
        for name in legacy:
            np.testing.assert_array_equal(legacy[name], updated[name])
        current_condition = conditioning(c, self.record)
        for name in old_condition:
            np.testing.assert_array_equal(old_condition[name], current_condition[name])
        active = updated['mouth_weights'] > 0
        np.testing.assert_array_equal(updated['mouth_points'][active], mesh['points'][mesh['mouth']])
        self.assertTrue(np.all(initial_fields(c, updated['mouth_points'][active]) == 0))
        c.operator.mouth_sampling = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Mouth sampling'):
            initial_payload(c, mesh, np.random.default_rng(94))

    def test_bank_cannot_silently_reuse_midpoint_queries_for_node_policy(self):
        from examples.particle_corrosion2d.train import load_bank
        c = get_config()
        prior = json.loads(c.to_json())
        c.operator.mouth_sampling = 'mesh_dirichlet'
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'manifest.json').write_text(json.dumps(dict(config=prior)))
            with self.assertRaisesRegex(ValueError, 'mouth sampling policy'):
                load_bank(directory, c)


if __name__ == '__main__':
    unittest.main()
