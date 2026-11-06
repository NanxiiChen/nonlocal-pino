"""Independent P1/physical-KKS checks, CPU only and without learned fields."""
import unittest

import numpy as np
from scipy import sparse

from examples.particle_corrosion2d.configs.train_base import get_config
from examples.particle_corrosion2d.geometry import contains, sample_geometry
from examples.particle_corrosion2d.mesh import assemble_p1, build_mesh
from examples.particle_corrosion2d.solver import initial_fields, make_system


class ParticleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_config()
        cls.record = sample_geometry(cls.config, np.random.default_rng(61027))
        cls.mesh = build_mesh(cls.config, cls.record, cls.config.mesh.spacing)

    def test_p1_exact_triangle_and_linear_energy(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.]])
        cells, mass, stiffness = assemble_p1(points, np.array([[0, 2, 1]]))
        np.testing.assert_array_equal(cells, [[0, 1, 2]])
        np.testing.assert_allclose(mass, np.full(3, 1/6), atol=1e-15)
        expected = np.array([[1., -.5, -.5], [-.5, .5, 0.], [-.5, 0., .5]])
        np.testing.assert_allclose(stiffness.toarray(), expected, atol=1e-15)
        mesh = self.mesh
        x, y = mesh['points'].T
        matrix = mesh['stiffness']
        area = mesh['volumes'].sum()
        self.assertAlmostEqual(float(x@matrix@x), area, places=11)
        self.assertAlmostEqual(float(y@matrix@y), area, places=11)
        self.assertAlmostEqual(float(x@matrix@y), 0.0, places=11)
        np.testing.assert_allclose(matrix@np.ones(len(x)), 0.0, atol=2e-12)

    def test_holes_mouth_and_domain_outward_normals(self):
        mesh, c = self.mesh, self.config
        self.assertTrue(np.all(mesh['volumes'] > 0))
        # Membership is a diagnostic on the conforming triangulation, never a
        # centroid-deletion strategy for constructing the physical domain.
        centroids = mesh['points'][mesh['triangles']].mean(axis=1)
        self.assertTrue(contains(centroids, self.record, c).all())
        np.testing.assert_allclose(np.linalg.norm(mesh['boundary_normals'], axis=1), 1.0, atol=1e-14)
        for i, particle in enumerate(self.record['particles']):
            mask = mesh['boundary_groups'] == i+2
            self.assertEqual(int(mask.sum()), c.mesh.boundary_points)
            radial = mesh['boundary_midpoints'][mask]-np.asarray(particle['center'])
            self.assertTrue((np.sum(radial*mesh['boundary_normals'][mask], axis=1) < 0).all())
            nodes = np.unique(mesh['boundary_edges'][mask])
            self.assertEqual(len(nodes), c.mesh.boundary_points)
            distance = np.linalg.norm(mesh['points'][nodes, None]-mesh['hole_polygons'][i][None], axis=-1)
            np.testing.assert_allclose(distance.min(axis=1), 0.0, atol=1e-14)
        mouth_points = mesh['points'][mesh['mouth']]
        np.testing.assert_allclose(mouth_points[:, 1], 0.0, atol=1e-14)
        self.assertAlmostEqual(mouth_points[:, 0].min(), c.geometry.pit_x-c.geometry.mouth_half_width)
        self.assertAlmostEqual(mouth_points[:, 0].max(), c.geometry.pit_x+c.geometry.mouth_half_width)
        mouth_edges = mesh['boundary_groups'] == 1
        self.assertAlmostEqual(mesh['boundary_lengths'][mouth_edges].sum(), 2*c.geometry.mouth_half_width)
        np.testing.assert_allclose(mesh['boundary_normals'][mouth_edges],
            np.tile([0., -1.], (mouth_edges.sum(), 1)), atol=1e-14)
        fields = initial_fields(c, mesh['points'], mesh['mouth'])
        np.testing.assert_array_equal(fields[mesh['mouth']], 0.0)

    def test_physical_jacobian_and_open_boundary_flux(self):
        c, mesh = self.config, self.mesh
        matrix = sparse.diags(1/mesh['volumes'])@mesh['stiffness']
        rhs, jac, initial, unpack = make_system(c, matrix, mesh['points'], mesh['mouth'])
        rng = np.random.default_rng(934)
        state = rng.uniform(.15, .85, size=initial.shape)
        direction = rng.normal(size=state.shape)
        step = 1e-6
        actual = (rhs(0.2, state+step*direction)-rhs(0.2, state-step*direction))/(2*step)
        expected = jac(0.2, state)@direction
        np.testing.assert_allclose(actual, expected, atol=4e-5, rtol=2e-7)
        phi, conc = unpack(state).T
        dc, cle = c.physical.c_se-c.physical.c_le, c.physical.c_le
        q = conc-dc*phi**2*(3-2*phi)-cle
        free = ~mesh['mouth']
        concentration_rate = rhs(0.2, state)[free.sum():]
        expected_open_rate = c.physical.n_ch*np.sum(mesh['volumes'][mesh['mouth']]*(matrix@q)[mesh['mouth']])
        actual_open_rate = mesh['volumes'][free]@concentration_rate
        self.assertAlmostEqual(actual_open_rate, expected_open_rate, places=9)

    def test_closed_transport_and_fem_offdiagonal_sign(self):
        # An obtuse valid triangle has a positive stiffness off-diagonal; this
        # must not be rejected by the physical solver's graph-only assumptions.
        points = np.array([[0., 0.], [1., 0.], [.2, .1]])
        _, volumes, stiffness = assemble_p1(points, np.array([[0, 1, 2]]))
        off = stiffness-sparse.diags(stiffness.diagonal())
        self.assertGreater(off.data.max(), 0)
        matrix = sparse.diags(1/volumes)@stiffness
        rhs, _, _, _ = make_system(self.config, matrix, points, np.zeros(3, dtype=bool))
        state = np.array([.2, .4, .7, .3, .8, .6])
        self.assertAlmostEqual(float(volumes@rhs(.1, state)[3:]), 0.0, places=10)
        np.testing.assert_allclose(rhs(.1, np.ones(6)), 0.0, atol=2e-10)
        with self.assertRaisesRegex(ValueError, 'Positive Laplacian'):
            make_system(self.config, -matrix, points, np.zeros(3, dtype=bool))


if __name__ == '__main__':
    unittest.main()
