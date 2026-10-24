"""Stage 2 checks for the explicit-volume weighted scalar operator."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from nonlocal_pinn.operators import ExplicitVolumeSparseD


def _star_case(dtype=None):
    if dtype is None:
        dtype = np.float32
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.0, 0.3],
            [0.4, 0.3],
            [0.2, 0.15],
        ],
        dtype=dtype,
    )
    volumes = np.asarray([0.7, 1.3, 0.9, 1.1, 0.8], dtype=dtype)
    pairs = np.asarray([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=np.int64)
    directed_edges = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    return coordinates, volumes, directed_edges, 0.35


def _square_cloud(n, dtype=np.float64):
    coordinate_axis = np.linspace(0.0, 1.0, n, dtype=dtype)
    x_grid, y_grid = np.meshgrid(coordinate_axis, coordinate_axis, indexing="xy")
    coordinates = np.stack((x_grid.ravel(), y_grid.ravel()), axis=-1)

    triangles = []
    for iy in range(n - 1):
        for ix in range(n - 1):
            lower_left = iy * n + ix
            lower_right = lower_left + 1
            upper_left = lower_left + n
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    triangles = np.asarray(triangles, dtype=np.int64)
    p0 = coordinates[triangles[:, 0]].astype(np.float64)
    p1 = coordinates[triangles[:, 1]].astype(np.float64)
    p2 = coordinates[triangles[:, 2]].astype(np.float64)
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    volumes = np.zeros(len(coordinates), dtype=np.float64)
    np.add.at(volumes, triangles[:, 0], areas / 3.0)
    np.add.at(volumes, triangles[:, 1], areas / 3.0)
    np.add.at(volumes, triangles[:, 2], areas / 3.0)
    return coordinates, volumes.astype(dtype), triangles


def _strict_radius_edges(coordinates, delta):
    tree = cKDTree(np.asarray(coordinates, dtype=np.float64))
    pairs = np.asarray(tree.query_pairs(float(delta), output_type="ndarray"))
    if pairs.size == 0:
        raise AssertionError("manufactured square graph has no candidate edges")
    pairs = pairs.reshape((-1, 2)).astype(np.int64, copy=False)
    distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1
    )
    pairs = pairs[distances < delta]
    directed = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return directed[order]


def _build_square_operator(n, delta):
    coordinates, volumes, triangles = _square_cloud(n)
    del triangles
    edges = _strict_radius_edges(coordinates, delta)
    return (
        coordinates,
        volumes,
        ExplicitVolumeSparseD(coordinates, volumes, edges, delta, "cubic"),
    )


class ExplicitVolumeSparseDGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coordinates, cls.volumes, cls.edges, cls.delta = _star_case()
        cls.operator = ExplicitVolumeSparseD(
            cls.coordinates,
            cls.volumes,
            cls.edges[::-1],
            cls.delta,
            "cubic",
        )

    def test_constructor_normalizes_edges_and_exposes_fixed_data(self):
        expected = np.asarray(
            sorted(map(tuple, self.edges.tolist())), dtype=np.int32
        )
        np.testing.assert_array_equal(
            np.asarray(self.operator.directed_edges), expected
        )
        self.assertEqual(self.operator.shape, (len(self.coordinates),) * 2)
        self.assertEqual(self.operator.kernel, "cubic")
        self.assertAlmostEqual(self.operator.delta, self.delta)
        np.testing.assert_array_equal(
            np.asarray(self.operator.volumes), self.volumes
        )

    def test_constants_are_annihilated(self):
        result = np.asarray(
            self.operator @ jnp.ones(len(self.coordinates), dtype=jnp.float32)
        )
        np.testing.assert_allclose(result, 0.0, rtol=0.0, atol=3.0e-6)
        matrix_result = np.asarray(
            self.operator @ jnp.ones((len(self.coordinates), 3), dtype=jnp.float32)
        )
        np.testing.assert_allclose(matrix_result, 0.0, rtol=0.0, atol=3.0e-6)

    def test_exact_entries_and_sign(self):
        matrix = self.operator.to_scipy().toarray()
        expected = np.zeros_like(matrix)
        kernel_scale = 120.0 / (np.pi * self.delta**4)
        for i, j in np.asarray(self.operator.directed_edges):
            distance = np.linalg.norm(self.coordinates[j] - self.coordinates[i])
            weight = kernel_scale * (1.0 - distance / self.delta) ** 2
            expected[i, j] = -weight * self.volumes[j]
            expected[i, i] += weight * self.volumes[j]
        np.testing.assert_allclose(matrix, expected, rtol=2.0e-6, atol=2.0e-5)
        self.assertTrue(np.all(np.diag(matrix) > 0.0))
        self.assertTrue(np.all(matrix[np.nonzero(~np.eye(len(matrix), dtype=bool))] <= 0.0))

    def test_scipy_jax_vector_and_matrix_match(self):
        rng = np.random.default_rng(42)
        vector = jnp.asarray(rng.normal(size=len(self.coordinates)), dtype=jnp.float32)
        matrix_values = jnp.asarray(
            rng.normal(size=(len(self.coordinates), 4)), dtype=jnp.float32
        )
        scipy_matrix = self.operator.to_scipy()
        np.testing.assert_allclose(
            np.asarray(self.operator @ vector),
            scipy_matrix @ np.asarray(vector),
            rtol=3.0e-6,
            atol=3.0e-5,
        )
        np.testing.assert_allclose(
            np.asarray(self.operator @ matrix_values),
            scipy_matrix @ np.asarray(matrix_values),
            rtol=3.0e-6,
            atol=3.0e-5,
        )

    def test_large_offset_and_large_values_use_scale_aware_csr_comparison(self):
        initial_x64 = bool(jax.config.x64_enabled)
        try:
            jax.config.update("jax_enable_x64", True)
            local_coordinates, volumes, edges, delta = _star_case(
                dtype=np.float64
            )
            coordinates = local_coordinates + np.asarray(
                [1.0e8 + 0.125, -2.0e8 - 0.375], dtype=np.float64
            )
            operator = ExplicitVolumeSparseD(
                coordinates, volumes, edges, delta, "cubic"
            )
            values = np.asarray(
                [1.0e8, -2.0e8, 3.0e8, -4.0e8, 5.0e8], dtype=np.float64
            )
            jax_result = np.asarray(operator @ jnp.asarray(values))
            scipy_result = np.asarray(operator.to_scipy() @ values)
            relative_error = np.linalg.norm(jax_result - scipy_result) / np.linalg.norm(
                scipy_result
            )
            self.assertLess(float(relative_error), 2.0e-13)
        finally:
            jax.config.update("jax_enable_x64", initial_x64)

    def test_jit_vector_and_matrix_match(self):
        rng = np.random.default_rng(9)
        vector = jnp.asarray(rng.normal(size=len(self.coordinates)), dtype=jnp.float32)
        matrix_values = jnp.asarray(
            rng.normal(size=(len(self.coordinates), 2)), dtype=jnp.float32
        )
        vector_result = jax.jit(lambda values: self.operator @ values)(vector)
        matrix_result = jax.jit(lambda values: self.operator @ values)(matrix_values)
        np.testing.assert_allclose(vector_result, self.operator @ vector, atol=2.0e-6)
        np.testing.assert_allclose(matrix_result, self.operator @ matrix_values, atol=2.0e-6)

    def test_volume_self_adjoint_and_directed_pair_energy_identity(self):
        matrix = self.operator.to_scipy().toarray()
        mass = np.diag(self.volumes)
        np.testing.assert_allclose(mass @ matrix, (mass @ matrix).T, atol=2.0e-6)

        rng = np.random.default_rng(21)
        kernel_scale = 120.0 / (np.pi * self.delta**4)
        for _ in range(5):
            values = rng.normal(size=len(self.coordinates))
            left = values @ (mass @ matrix @ values)
            right = 0.0
            for i, j in self.edges:
                distance = np.linalg.norm(self.coordinates[j] - self.coordinates[i])
                kernel = kernel_scale * (1.0 - distance / self.delta) ** 2
                right += 0.5 * self.volumes[i] * self.volumes[j] * kernel * (
                    values[i] - values[j]
                ) ** 2
            self.assertGreaterEqual(left, -2.0e-5)
            np.testing.assert_allclose(left, right, rtol=2.0e-6, atol=3.0e-5)

    def test_fixed_history_volume_multiplied_system_is_spd_and_solves(self):
        scipy_matrix = self.operator.to_scipy()
        mass = np.diag(self.volumes)
        gc = 2.7
        l0 = 0.024
        history = np.asarray([0.0, 0.4, 0.2, 0.7, 1.1], dtype=np.float32)
        strong = np.diag(gc / l0 + 2.0 * history) + gc * l0 * scipy_matrix.toarray()
        system = mass @ strong
        rhs = 2.0 * mass @ history
        np.testing.assert_allclose(system, system.T, rtol=0.0, atol=3.0e-6)
        eigenvalues = np.linalg.eigvalsh(system)
        self.assertGreater(float(eigenvalues[0]), 0.0)
        damage = np.linalg.solve(system, rhs)
        residual = system @ damage - rhs
        np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=3.0e-6)

    def test_invalid_matmul_shapes_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.operator @ np.ones((len(self.coordinates), 1, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "leading dimension"):
            self.operator @ np.ones(len(self.coordinates) - 1, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "T > 0"):
            self.operator @ np.ones((len(self.coordinates), 0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "match the operator dtype"):
            self.operator @ np.ones(len(self.coordinates), dtype=np.int32)


class ExplicitVolumeSparseDValidationTest(unittest.TestCase):
    def setUp(self):
        self.coordinates, self.volumes, self.edges, self.delta = _star_case()

    def _build(self, **changes):
        values = {
            "coordinates": self.coordinates,
            "volumes": self.volumes,
            "directed_edges": self.edges,
            "delta": self.delta,
            "kernel": "cubic",
        }
        values.update(changes)
        return ExplicitVolumeSparseD(**values)

    def test_coordinate_shape_finiteness_uniqueness_and_dtype_are_checked(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self._build(coordinates=np.ones(5, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-empty shape"):
            self._build(coordinates=np.empty((0, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self._build(
                coordinates=np.arange(20, dtype=np.float32).reshape(5, 4)
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = self.coordinates.copy()
            bad[0, 0] = np.nan
            self._build(coordinates=bad)
        with self.assertRaisesRegex(ValueError, "unique"):
            bad = self.coordinates.copy()
            bad[1] = bad[0]
            self._build(coordinates=bad)
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            self._build(coordinates=self.coordinates.astype(np.int64))

    def test_volume_shape_range_finiteness_and_dtype_are_checked(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self._build(volumes=self.volumes[:, None])
        with self.assertRaisesRegex(ValueError, "positive"):
            bad = self.volumes.copy()
            bad[0] = 0.0
            self._build(volumes=bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = self.volumes.copy()
            bad[0] = np.inf
            self._build(volumes=bad)
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            self._build(volumes=np.ones(len(self.volumes), dtype=np.int64))

    def test_coordinate_and_volume_dtypes_must_match(self):
        with self.assertRaisesRegex(ValueError, "same dtype"):
            self._build(volumes=self.volumes.astype(np.float64))
        with self.assertRaisesRegex(ValueError, "same dtype"):
            self._build(coordinates=self.coordinates.astype(np.float64))

    def test_edge_shape_integer_graph_and_reciprocity_are_checked(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self._build(directed_edges=np.ones((len(self.edges), 1), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self._build(directed_edges=np.empty((0, 2), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "integer"):
            self._build(directed_edges=self.edges.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "self"):
            self._build(
                directed_edges=np.vstack((self.edges, np.asarray([[0, 0]], dtype=np.int64)))
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self._build(directed_edges=np.vstack((self.edges, self.edges[0])))
        with self.assertRaisesRegex(ValueError, "outside"):
            self._build(
                directed_edges=np.vstack(
                    (self.edges, np.asarray([[0, len(self.coordinates)]], dtype=np.int64))
                )
            )
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            self._build(directed_edges=self.edges[:-1])

    def test_delta_kernel_distance_and_outgoing_family_are_checked(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self._build(delta=0.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self._build(delta=np.nan)
        with self.assertRaisesRegex(ValueError, "scalar"):
            self._build(delta=np.asarray([self.delta]))
        with self.assertRaisesRegex(ValueError, "scalar"):
            self._build(delta="0.2")
        with self.assertRaisesRegex(ValueError, "Unknown kernel"):
            self._build(kernel="not-a-kernel")
        with self.assertRaisesRegex(ValueError, "strict 0 < distance < delta"):
            self._build(delta=0.25)

        isolated_coordinates = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.8, 0.0]], dtype=np.float32
        )
        isolated_edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "outgoing family"):
            ExplicitVolumeSparseD(
                isolated_coordinates,
                np.ones(3, dtype=np.float32),
                isolated_edges,
                0.2,
                "cubic",
            )


class ExplicitVolumeSparseDDtypeTest(unittest.TestCase):
    def test_float32_float64_and_disabled_x64_behavior(self):
        initial_x64 = bool(jax.config.x64_enabled)
        coordinates, volumes, edges, delta = _star_case(dtype=np.float32)
        try:
            jax.config.update("jax_enable_x64", True)
            operator32 = ExplicitVolumeSparseD(
                coordinates, volumes, edges, delta, "cubic"
            )
            self.assertEqual(operator32.diagonal.dtype, jnp.float32)

            coordinates64, volumes64, _, _ = _star_case(dtype=np.float64)
            operator64 = ExplicitVolumeSparseD(
                coordinates64, volumes64, edges, delta, "cubic"
            )
            self.assertEqual(operator64.diagonal.dtype, jnp.float64)
            self.assertEqual((operator64 @ jnp.ones(len(coordinates64))).dtype, jnp.float64)

            jax.config.update("jax_enable_x64", False)
            with self.assertRaisesRegex(ValueError, "64-bit mode"):
                ExplicitVolumeSparseD(
                    coordinates64, volumes64, edges, delta, "cubic"
                )
            with self.assertRaisesRegex(ValueError, "float64 matmul input"):
                operator32 @ np.ones(len(coordinates), dtype=np.float64)
        finally:
            jax.config.update("jax_enable_x64", initial_x64)


class ExplicitVolumeSparseDManufacturedSquareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.initial_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        cls.coordinates, cls.volumes, cls.operator = _build_square_operator(41, 0.1)
        if cls.operator.diagonal.dtype != jnp.float64:
            raise AssertionError("full-square baseline operator must use float64")

    @classmethod
    def tearDownClass(cls):
        jax.config.update("jax_enable_x64", cls.initial_x64)

    def _interior_mask(self, delta=0.1):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        return np.minimum.reduce((x, y, 1.0 - x, 1.0 - y)) > delta

    def test_full_square_constants_affine_and_quadratics(self):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        interior = self._interior_mask()
        cases = (
            ("constant", np.ones_like(x), np.zeros_like(x), 3.0e-6),
            ("affine-x", x, np.zeros_like(x), 3.0e-5),
            ("affine-y", y, np.zeros_like(y), 3.0e-5),
            ("x^2", x**2, -2.0 * np.ones_like(x), 1.0e-2),
            ("y^2", y**2, -2.0 * np.ones_like(y), 1.0e-2),
            ("xy", x * y, np.zeros_like(x), 1.0e-2),
        )
        for name, field, target, interior_tolerance in cases:
            result = np.asarray(self.operator @ jnp.asarray(field))
            full_error = np.abs(result - target)
            interior_error = full_error[interior]
            print(
                "full-square manufactured",
                name,
                "full_mae=%.6e interior_mae=%.6e interior_max=%.6e"
                % (full_error.mean(), interior_error.mean(), interior_error.max()),
            )
            self.assertLessEqual(float(interior_error.max()), interior_tolerance)
            if name != "constant":
                self.assertGreaterEqual(
                    float(full_error.mean()), float(interior_error.mean()) - 1.0e-8
                )

    def test_fixed_delta_refinement_decreases_quadratic_interior_error(self):
        errors = []
        for n in (41, 61, 81):
            coordinates, _, operator = _build_square_operator(n, 0.1)
            values = jnp.asarray(coordinates[:, 0] ** 2)
            result = np.asarray(operator @ values)
            interior = np.minimum.reduce(
                (
                    coordinates[:, 0],
                    coordinates[:, 1],
                    1.0 - coordinates[:, 0],
                    1.0 - coordinates[:, 1],
                )
            ) > 0.1
            error = float(np.mean(np.abs(result[interior] + 2.0)))
            errors.append(error)
        print("full-square fixed-delta refinement errors:", errors)
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])

    def test_coupled_delta_and_spacing_diagnostic_is_finite_only(self):
        cases = ((21, 0.2), (31, 0.13333333333333333), (41, 0.1))
        diagnostics = []
        for n, delta in cases:
            coordinates, _, operator = _build_square_operator(n, delta)
            result = np.asarray(operator @ jnp.asarray(coordinates[:, 0] ** 2))
            interior = np.minimum.reduce(
                (
                    coordinates[:, 0],
                    coordinates[:, 1],
                    1.0 - coordinates[:, 0],
                    1.0 - coordinates[:, 1],
                )
            ) > delta
            error = float(np.mean(np.abs(result[interior] + 2.0)))
            diagnostics.append((n, delta, error))
        print("full-square coupled finite diagnostics:", diagnostics)
        self.assertTrue(np.all(np.isfinite(np.asarray([item[2] for item in diagnostics]))))


if __name__ == "__main__":
    unittest.main()
