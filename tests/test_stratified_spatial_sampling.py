"""Measure conservation and geometric coverage of proposed sampling controls."""
import unittest
import numpy as np

from examples.channel_nsch2d.configs.train_stratified_spatial import get_config
from examples.channel_nsch2d.spatial_sampling import (
    draw_stratified_tiles, stratified_partition, validate_spatial)
from nonlocal_pinn.training.spatial_sampling import polyline_band, stratified_indices
from examples.particle_corrosion2d.inputs import make_core_sampler
from ml_collections import ConfigDict


class StratifiedSpatialTest(unittest.TestCase):
    def test_ceramic_wall_weights_and_legacy_draws(self):
        points = np.array([[.25, -.02], [.75, -.02], [1.02, .4], [1.02, .8],
                           [1.3, 1.3], [2., .5], [3., .7], [4., .8], [.5, .5]])
        mass = np.arange(1., 10.)
        mouth = np.array([False]*8+[True])
        mesh = dict(points=points, volumes=mass, mouth=mouth,
                    hole_polygons=np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]]))
        c = ConfigDict(dict(sample=dict(K_core=16)))
        expected = np.random.default_rng(12).choice(8, 16, replace=True, p=mass[:8]/mass[:8].sum())
        rows, weights = make_core_sampler(c, mesh)(np.random.default_rng(12))
        np.testing.assert_array_equal(rows, expected)
        np.testing.assert_array_equal(weights, np.ones(16))
        c.sample.core_draw = 'stratified_cdf'
        c.sample.core_wall = ConfigDict(dict(count=4, distance=.05))
        sampler = make_core_sampler(c, mesh)
        rng = np.random.default_rng(33)
        estimates = []
        function = points[:, 0]**2 + 3*points[:, 1]
        for _ in range(1500):
            rows, weights = sampler(rng)
            self.assertTrue(np.all(rows[:4] < 4))
            self.assertTrue(np.all((rows[4:] >= 4) & (rows[4:] < 8)))
            np.testing.assert_allclose(weights[:4].sum()/weights.sum(), mass[:4].sum()/mass[:8].sum())
            estimates.append(weights@function[rows]/weights.sum())
        expected = mass[:8]@function[:8]/mass[:8].sum()
        self.assertLess(abs(np.mean(estimates)-expected), 6*np.std(estimates)/np.sqrt(len(estimates)))

    def test_equal_mass_intervals_and_zero_mass(self):
        masses = np.array([0., 1., 1., 0., 1., 1.])
        for seed in range(20):
            np.testing.assert_array_equal(stratified_indices(masses, 4, np.random.default_rng(seed)), [1, 2, 4, 5])
        for invalid in ([0., 0.], [1., -1.], [1., np.nan]):
            with self.assertRaises(ValueError):
                stratified_indices(invalid, 4, np.random.default_rng(1))

    def test_band_uses_edges_and_corners(self):
        segment = np.array([[[0., 0.], [2., 0.]], [[2., 0.], [2., 1.]]])
        points = np.array([[1., .04], [1., .1], [2.03, 1.03], [2.08, 1.08]])
        np.testing.assert_array_equal(polyline_band(points, segment, .05), [True, False, True, False])

    def test_all_tiles_cover_the_original_measure_in_expectation(self):
        c = get_config()
        c.sample.Nx, c.sample.Ny = 22, 14
        c.sample.spatial.node_capacity = 23*15
        c.sample.spatial.core_count = 64
        c.sample.spatial.wall_groups = 1
        c.sample.spatial.wall_band = .13
        validate_spatial(c)
        xx, yy = np.meshgrid(np.linspace(0, 2., 23), np.linspace(0, 1., 15), indexing='ij')
        points = np.column_stack((xx.ravel(), yy.ravel()))
        measure = 1.+points[:, 0]**2+points[:, 1]
        free = np.zeros(xx.shape, dtype=bool)
        free[1:-1, 1:-1] = True
        data = dict(points=points, transport_measure=measure, interior=free.ravel())
        for wall_groups in (0, 1, 3):
            c.sample.spatial.wall_groups = wall_groups
            partition = stratified_partition(data, c)
            cells = [ids for stratum in partition[0] for ids in stratum]
            np.testing.assert_array_equal(np.sort(np.concatenate(cells)), np.flatnonzero(free))
            strata_mass = [sum(measure[ids].sum() for ids in stratum) for stratum in partition[0]]
            truth = np.column_stack((points[:, 0]**3, np.sin(8*points[:, 1]), np.exp(-10*points[:, 0])))
            target = measure[free.ravel()] @ truth[free.ravel()] / measure[free.ravel()].sum()
            estimates = []
            rng = np.random.default_rng(17)
            for _ in range(2000):
                rows, weights = draw_stratified_tiles(data, c, rng, partition)
                self.assertEqual(len(rows), 64)
                self.assertTrue(np.all(free.ravel()[rows]))
                np.testing.assert_allclose(weights.sum(), 1., atol=1e-14)
                offset = 0
                for mass, count in zip(strata_mass, partition[1], strict=True):
                    size = count*c.sample.spatial.tile_width**2
                    np.testing.assert_allclose(weights[offset:offset+size].sum(), mass/sum(strata_mass))
                    offset += size
                estimates.append(weights @ truth[rows])
            estimates = np.asarray(estimates)
            self.assertTrue(np.all(abs(estimates.mean(axis=0)-target) < 6*estimates.std(axis=0)/np.sqrt(len(estimates))+1e-12))


if __name__ == '__main__':
    unittest.main()
