"""Independent geometry, boundary, sampling and graph-moment checks."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.configs.train_thin_interface import get_config
from examples.icemelting3d.thin_interface import (
    build_network, crossing, field_metrics, sample_batch, stencil, validate,
)
from examples.icemelting3d.thin_interface_reference import graph_weight, latent_diffusion, octant_laplacian
from nonlocal_pinn.operators import SparseD
from nonlocal_pinn.training.runtime import with_run_seed


class ThinInterfaceTest(unittest.TestCase):
    def test_compact_stencil_equals_original_graph_rows(self):
        c = with_run_seed(get_config(), 42)
        c.sample.Nx = c.sample.Ny = c.sample.Nz = 8
        c.nl.delta = 1.27/7
        ids = np.array([[0, 0, 0], [3, 4, 2], [7, 6, 7]], dtype=np.int32)
        points, (row, col, vals, diagonal) = stencil(ids, c)
        axis = np.arange(8)/7
        full = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
        with jax.enable_x64(True):
            graph = SparseD(jnp.asarray(full), c.nl.delta, kernel='cubic')
            f = lambda xx: np.sin(xx[:, 0])+xx[:, 1]**3-.2*xx[:, 2]**2
            truth = np.asarray(graph@jnp.asarray(f(full)))
        out = np.bincount(row, weights=vals*f(points)[col], minlength=len(points))+diagonal*f(points)
        np.testing.assert_allclose(out[:len(ids)], truth[np.ravel_multi_index(ids.T, (8,)*3)], rtol=2e-5, atol=2e-5)
        self.assertAlmostEqual(graph_weight(get_config())/127**2, .6858547757362532)

    def test_reflected_neumann_laplacian_converges_at_both_boundary_types(self):
        errors = []
        with jax.enable_x64(True):
            for n in (32, 64):
                h = 1/(n-1)
                x = (np.arange(n//2)+.5)*h
                xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
                phi = np.cos(2*np.pi*xx)+np.cos(2*np.pi*yy)+np.cos(2*np.pi*zz)
                got = np.asarray(octant_laplacian(jnp.asarray(phi), h, local=True))
                errors.append(np.max(np.abs(got+4*np.pi**2*phi)))
                np.testing.assert_array_equal(octant_laplacian(jnp.ones_like(jnp.asarray(phi)), h, local=True), 0.)
        self.assertLess(errors[1], errors[0]/3.9)

    def test_paired_initialization_and_actual_query_stream(self):
        c = with_run_seed(get_config(), 43)
        validate(c)
        other = c.copy_and_resolve_references()
        other.nl.method = 'local'
        validate(other)
        for step in (0, 37):
            for x, y in zip(sample_batch(c, step), sample_batch(other, step), strict=True):
                np.testing.assert_array_equal(x, y)
        a, b = build_network(c), build_network(other)
        self.assertTrue(eqx.tree_equal(a, b))
        x = jnp.asarray([.2, .3, .4])
        for t in (0., .4, 1.):
            np.testing.assert_array_equal(a(x, t), a(x, 0.))

    def test_reference_latent_chain_rule_and_exact_graph_identity(self):
        errors = []
        with jax.enable_x64(True):
            for n in (32, 64):
                h = 1/(n-1)
                x = (np.arange(n//2)+.5)*h
                xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
                u = jnp.asarray(.3*(np.cos(2*np.pi*xx)+np.cos(2*np.pi*yy)+np.cos(2*np.pi*zz)))
                phi = jnp.tanh(u)
                original = octant_laplacian(phi, h, local=False)
                transformed = (1-phi**2)*latent_diffusion(u, h, local=False)
                np.testing.assert_allclose(original, transformed, atol=2e-11, rtol=1e-10)
                direct = octant_laplacian(phi, h, local=True)
                equivalent = (1-phi**2)*latent_diffusion(u, h, local=True)
                errors.append(float(jnp.max(jnp.abs(direct-equivalent))))
        self.assertLess(errors[1], errors[0]/3.8)

    def test_positive_time_metrics_and_crossings(self):
        truth = np.array([[1., 1., 1.], [-1., .2, .8], [-1., .1, .9]])
        pred = truth.copy()
        pred[0] = 1000
        self.assertEqual(field_metrics(pred, truth)['pooled'], 0.)
        radius = np.array([0., .5, 1.])
        np.testing.assert_allclose(crossing(np.array([[1., .25, -.75]]), radius), [.625])
        with self.assertRaises(ValueError):
            crossing(np.ones((1, 3)), radius)
