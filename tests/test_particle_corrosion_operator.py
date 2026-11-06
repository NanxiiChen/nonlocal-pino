"""Regional pair conservation, full-segment visibility and continuum moments."""
import unittest

import numpy as np
from ml_collections import ConfigDict
from scipy.integrate import quad
from matplotlib.path import Path

from examples.particle_corrosion2d.configs.geometry_base import get_config
from examples.particle_corrosion2d.geometry import boundary
from examples.particle_corrosion2d.operators import (
    build_operator, operator_diagnostics, visible_pairs)


def config():
    c = get_config()
    c.mesh = ConfigDict(dict(boundary_points=128))
    c.nl = ConfigDict(dict(delta=.08, visibility_points=128, visibility_tolerance=1e-10))
    return c


def circle_record():
    return dict(schema='smooth_radial_particles_v1', particles=[dict(
        center=[.5, .3], equivalent_radius=.1, radial_scale=.1, aspect=1., angle=0.,
        modes=[3], cosine=[0.], sine=[0.])])


class ParticleOperatorTests(unittest.TestCase):
    def test_complete_segment_visibility_tangency_and_reversal(self):
        c, record = config(), circle_record()
        # Third bond intersects near its start although both endpoints AND its
        # midpoint are outside. Fourth joins wall points through the particle.
        points = np.array([[.38, .3], [.62, .3], [.38, .4], [.62, .4],
                           [.39, .3], [.95, .3], [.4, .3], [.6, .3]])
        pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [0, 2]])
        expected = np.array([False, True, False, False, True])
        np.testing.assert_array_equal(visible_pairs(c, record, points, pairs), expected)
        np.testing.assert_array_equal(visible_pairs(c, record, points, pairs[:, ::-1]), expected)
        nodes = boundary(record['particles'][0], np.arange(128)*2*np.pi/128)[0]
        self.assertTrue(visible_pairs(c, record, nodes, np.array([[0, 1], [17, 18]])).all())

    def test_weighted_symmetry_conservation_and_energy(self):
        c = config()
        c.nl.delta = .12
        xx, yy = np.meshgrid(np.linspace(0, 1, 31), np.linspace(0, .6, 19))
        points = np.column_stack((xx.ravel(), yy.ravel()))
        radius = np.linalg.norm(points-np.array([.5, .3]), axis=1)
        points = points[radius > .105]
        rng = np.random.default_rng(329)
        volumes = rng.uniform(.0007, .0012, len(points))
        matrix = build_operator(c, circle_record(), points, volumes)
        diagnostics = operator_diagnostics(matrix, volumes)
        self.assertEqual(diagnostics['components'], 1)
        self.assertLess(diagnostics['maximum_row_sum_error'], 2e-12)
        self.assertLess(diagnostics['maximum_weighted_symmetry_error'], 1e-14)
        self.assertLess(diagnostics['maximum_weighted_conservation_error'], 2e-14)
        self.assertLess(diagnostics['maximum_offdiagonal'], 0)
        for _ in range(3):
            values = rng.normal(size=len(points))
            self.assertGreater(np.sum(volumes*values*(matrix@values)), 0)
        coo = matrix.tocoo()
        pairs = np.column_stack((coo.row[coo.row < coo.col], coo.col[coo.row < coo.col]))
        self.assertTrue(visible_pairs(c, circle_record(), points, pairs).all())

    def test_nonconvex_particle_against_independent_dense_segment_probes(self):
        c, record = config(), circle_record()
        record['particles'][0]['cosine'] = [.30]
        polygon = boundary(record['particles'][0], np.arange(128)*2*np.pi/128)[0]
        path = Path(polygon, closed=False)
        rng = np.random.default_rng(742)
        points = rng.uniform([.32, .12], [.68, .48], size=(400, 2))
        points = points[~path.contains_points(points)]
        pairs = np.column_stack((np.arange(0, len(points)-1, 2),
                                 np.arange(1, len(points), 2)))
        start, end = points[pairs[:, 0]], points[pairs[:, 1]]
        probes = start[:, None]+np.linspace(0, 1, 2001)[None, :, None]*(end-start)[:, None]
        expected = ~path.contains_points(probes.reshape(-1, 2)).reshape(len(pairs), -1).any(axis=1)
        actual = visible_pairs(c, record, points, pairs)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.any())
        self.assertTrue((~actual).any())

    def test_kernel_second_moment_and_half_horizon_neumann_limit(self):
        c = config()
        delta = c.nl.delta
        second_moment = np.pi*quad(lambda r: 120/(np.pi*delta**4)*r**3*(1-r/delta)**2,
                                   0, delta, epsabs=1e-12)[0]
        self.assertAlmostEqual(second_moment, 2., places=12)
        # u=x^2 has normal derivative zero at x=0. The regional operator there
        # sees half a disk and converges pointwise to -1, NOT local -2. Interior
        # convergence and weak closed-wall conservation are separate claims.
        h = .005
        xx, yy = np.meshgrid(np.arange(0, .2+h/2, h), np.arange(0, .2+h/2, h))
        points = np.column_stack((xx.ravel(), yy.ravel()))
        weights = np.ones(xx.shape)
        weights[[0, -1], :] *= .5
        weights[:, [0, -1]] *= .5
        record = dict(schema='smooth_radial_particles_v1', particles=[])
        matrix = build_operator(c, record, points, weights.ravel()*h*h)
        values = matrix@(points[:, 0]**2)
        interior = np.flatnonzero(np.all(np.isclose(points, [.1, .1]), axis=1))[0]
        wall = np.flatnonzero(np.all(np.isclose(points, [0., .1]), axis=1))[0]
        self.assertAlmostEqual(values[interior], -2., delta=.005)
        self.assertAlmostEqual(values[wall], -1., delta=.003)

    def test_invalid_geometry_points_and_unmatched_polygon_fail(self):
        c = config()
        points = np.array([[.5, .3], [.51, .3]])
        with self.assertRaisesRegex(ValueError, 'inside'):
            build_operator(c, circle_record(), points, np.ones(2))
        c.nl.visibility_points = 256
        with self.assertRaisesRegex(ValueError, 'same explicit polygon'):
            visible_pairs(c, circle_record(), points, np.array([[0, 1]]))
        c = config()
        with self.assertRaisesRegex(ValueError, 'No nonlocal pairs'):
            build_operator(c, dict(schema='smooth_radial_particles_v1', particles=[]),
                           np.array([[0., 0.], [1., .6]]), np.ones(2))


if __name__ == '__main__':
    unittest.main()
