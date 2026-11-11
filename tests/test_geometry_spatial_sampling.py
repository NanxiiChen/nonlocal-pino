"""Loss equivalence and physical measures for geometry-specific core sampling."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.configs.train_spatial_control import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.geometry import nominal_parameters
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.spatial_sampling import compact_rows, draw_rows, make_tiles
from examples.particle_corrosion2d.inputs import make_core_sampler, particle_wall_band


class GeometrySpatialSamplingTest(unittest.TestCase):
    def test_channel_full_cover_loss_and_gradient(self):
        c = get_config()
        c.sample.Nx, c.sample.Ny, c.sample.boundary_points = 24, 12, 6
        c.nl.delta, c.nl.neighbors = .6, 192
        c.train.compute_dtype = 'float64'
        c.arch.width = 4
        c.arch.heads = c.arch.condition_depth = c.arch.decoder_depth = 1
        c.arch.spatial_features, c.arch.time_features = 2, 4
        with jax.enable_x64(True):
            raw, _ = geometry_data(nominal_parameters(c), c)
            core = np.flatnonzero(raw['interior'])
            volume = raw['transport_measure'][core]
            compact, _ = compact_rows(raw, core, volume/volume.sum(), len(raw['points']))
            net = build_network(c, jax.random.PRNGKey(8))
            times = jnp.array([.25, 1.])
            fn = eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c), has_aux=True))
            as_batch = lambda data: jax.tree.map(lambda a: jnp.asarray(a)[None], data)
            (full, parts), grad = fn(net, as_batch(raw), times)
            (sampled, sample_parts), sample_grad = fn(net, as_batch(compact), times)
            np.testing.assert_allclose(sampled, full, rtol=3e-10, atol=1e-10)
            np.testing.assert_allclose(sample_parts, parts, rtol=3e-10, atol=1e-10)
            for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(sample_grad), strict=True):
                np.testing.assert_allclose(a, b, rtol=3e-9, atol=1e-9)

    def test_tiles_partition_and_draw_count(self):
        c = get_config()
        nx, ny = c.sample.Nx, c.sample.Ny
        measure = np.arange(1, (nx+1)*(ny+1)+1, dtype=float)
        interior = np.zeros((nx+1, ny+1), dtype=bool)
        interior[1:-1, 1:-1] = True
        tiles = make_tiles(nx, ny)
        np.testing.assert_array_equal(np.sort(np.concatenate(tiles)), np.flatnonzero(interior))
        self.assertGreaterEqual(min(map(len, tiles)), 64)
        data = dict(transport_measure=measure, interior=interior.ravel())
        for mode in ('tiles', 'scatter'):
            c.sample.spatial.mode = mode
            rows, weights = draw_rows(data, c, np.random.default_rng(3), tiles=tiles)
            self.assertEqual(rows.shape, (512,))
            self.assertTrue(np.all(interior.ravel()[rows] & (weights > 0)))
            if mode == 'scatter':
                np.testing.assert_allclose(weights, 1/512, rtol=0, atol=0)

    def test_particle_wall_band_uses_edges_and_stratum_volume(self):
        points = np.array([[0., 0.], [.5, -.02], [1.02, .5], [1.3, 1.3], [3., .5]])
        polygons = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
        near = particle_wall_band(points, polygons, .05)
        np.testing.assert_array_equal(near, [True, True, True, False, False])
        volumes = np.array([100., 1., 2., 4., 8.])
        mesh = dict(points=points, volumes=volumes, mouth=np.array([True, False, False, False, False]),
                    hole_polygons=polygons)
        c = ConfigDict(dict(sample=dict(K_core=8, core_wall=dict(count=4, distance=.05))))
        rows, weights = make_core_sampler(c, mesh)(np.random.default_rng(5))
        self.assertEqual(np.count_nonzero(near[rows]), 4)
        self.assertTrue(np.all(rows != 0))
        np.testing.assert_allclose([weights[:4].sum()/weights.sum(), weights[4:].sum()/weights.sum()], [.2, .8])
