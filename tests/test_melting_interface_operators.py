"""Independent full-graph, moment, boundary and legacy-stencil checks."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.configs.train_thin_interface import get_config
from examples.icemelting3d.thin_interface import stencil as legacy_stencil
from examples.icemelting3d.thin_interface_operators import compact_stencil, offsets_and_weights
from nonlocal_pinn.operators import SparseD


def config(n, ratio, normalization):
    c = get_config()
    c.sample.Nx = c.sample.Ny = c.sample.Nz = n
    c.nl.delta = ratio/(n-1)
    c.nl.normalization = normalization
    return c


def apply_compact(points, packed, values):
    row, col, weights, diagonal = packed
    field = values(points)
    return np.bincount(row, weights=weights*field[col], minlength=len(points))+diagonal*field


class MeltingInterfaceOperatorTests(unittest.TestCase):
    def test_raw_six_neighbor_arrays_match_historical_stencil_exactly(self):
        for n in (8, 128):
            c = config(n, 1.27, 'raw')
            ids = np.array([[0, 0, 0], [0, n-1, 3], [n//2, n//2-1, 2], [n-1]*3], dtype=np.int32)
            before = legacy_stencil(ids, c)
            after = compact_stencil(ids, c, offsets_and_weights(c))
            np.testing.assert_array_equal(after[0], before[0])
            for x, y in zip(after[1], before[1], strict=True):
                np.testing.assert_array_equal(x, y)

    def test_all_cube_rows_match_independent_sparse_operator_for_three_shells(self):
        n = 8
        ids = np.stack(np.meshgrid(*([np.arange(n)]*3), indexing='ij'), -1).reshape(-1, 3)
        xyz = ids/(n-1)
        value = lambda x: np.sin(1.7*x[:, 0])+.3*x[:, 1]**3-.7*x[:, 2]**2
        for ratio, count in ((1.27, 6), (1.905, 26), (2.54, 80)):
            c = config(n, ratio, 'raw')
            operator = offsets_and_weights(c)
            self.assertEqual(operator[2]['neighbors'], count)
            points, packed = compact_stencil(ids, c, operator)
            with jax.enable_x64(True):
                independent = SparseD(jnp.asarray(xyz), c.nl.delta, kernel='cubic')
                expected = np.asarray(independent@jnp.asarray(value(xyz)))
            actual = apply_compact(points, packed, value)[:len(ids)]
            # The independent operator uses FP64 geometry; training arrays are FP32.
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            self.assertEqual(points.shape, (len(ids)*(count+1), 3))

    def test_full_graph_symmetry_nullspace_and_positive_energy(self):
        n = 8
        ids = np.stack(np.meshgrid(*([np.arange(n)]*3), indexing='ij'), -1).reshape(-1, 3)
        for normalization in ('raw', 'moment'):
            c = config(n, 1.905, normalization)
            offsets, weights, _ = offsets_and_weights(c)
            matrix = np.zeros((n**3, n**3))
            for i, core in enumerate(ids):
                neighbors = core+offsets
                valid = np.all((neighbors >= 0) & (neighbors < n), axis=1)
                cols = np.ravel_multi_index(neighbors[valid].T, (n,)*3)
                matrix[i, cols] = -weights[valid]
                matrix[i, i] = weights[valid].sum()
            np.testing.assert_allclose(matrix, matrix.T, atol=0, rtol=0)
            np.testing.assert_allclose(matrix.sum(axis=1), 0, atol=1e-13)
            self.assertGreaterEqual(np.linalg.eigvalsh(matrix)[0], -1e-12)
            field = np.random.default_rng(623).normal(size=n**3)
            edge_energy = -.5*np.sum(matrix*(field[:, None]-field[None])**2)
            self.assertGreater(edge_energy, 0)
            self.assertAlmostEqual(float(field@matrix@field), float(edge_energy), places=9)

    def test_interior_quadratic_moments_and_boundary_fixed_shape(self):
        for ratio in (1.27, 1.905, 2.54):
            c = config(16, ratio, 'moment')
            operator = offsets_and_weights(c)
            offsets, weights, metadata = operator
            np.testing.assert_allclose(metadata['second_moment_matrix'], np.eye(3), atol=2e-15)
            np.testing.assert_allclose(weights@(offsets/15), 0, atol=1e-13)
            interior = np.array([[7, 7, 7], [8, 7, 8]], dtype=np.int32)
            boundary = np.array([[0, 0, 0], [15, 15, 15]], dtype=np.int32)
            points, packed = compact_stencil(interior, c, operator)
            for axis in range(3):
                actual = apply_compact(points, packed, lambda x: x[:, axis].astype(np.float64)**2)[:2]
                np.testing.assert_allclose(actual, -2, atol=6e-5)
            boundary_points, boundary_packed = compact_stencil(boundary, c, operator)
            self.assertEqual(boundary_points.shape, points.shape)
            for a, b in zip(packed, boundary_packed, strict=True):
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.dtype, b.dtype)
            np.testing.assert_allclose(apply_compact(boundary_points, boundary_packed,
                lambda x: np.ones(len(x))), 0, atol=5e-5)
            self.assertTrue(np.any(boundary_packed[2] == 0))
            self.assertEqual(points.dtype, np.float32)

    def test_required_settings_bad_ids_and_operator_mismatches_fail(self):
        c = config(16, 1.27, 'raw')
        operator = offsets_and_weights(c)
        for field, value in (('delta', 1/15), ('delta', float('nan')),
                             ('kernel', 'quartic'), ('normalization', 'row')):
            bad = c.copy_and_resolve_references()
            bad.nl[field] = value
            with self.assertRaises(ValueError):
                offsets_and_weights(bad)
        bad = c.copy_and_resolve_references()
        bad.sample.Ny = 17
        with self.assertRaises(ValueError):
            offsets_and_weights(bad)
        bad = c.copy_and_resolve_references()
        bad.nl.normalization = 'moment'
        with self.assertRaises(ValueError):
            compact_stencil(np.array([[3, 4, 5]]), bad, operator)
        for ids in (np.array([[3., 4., 5.]]), np.array([[16, 0, 0]]),
                    np.empty((0, 3), dtype=int), np.array([[1, 2]])):
            with self.assertRaises(ValueError):
                compact_stencil(ids, c, operator)


if __name__ == '__main__':
    unittest.main()
