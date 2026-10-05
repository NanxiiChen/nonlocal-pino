"""Independent geometry/graph, energy and BC regressions for split radii."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.split_horizon_model import build_model, visibility_graph


def square_cloud():
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    coordinates = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float64)
    triangles = []
    for row in range(4):
        for col in range(4):
            i = row * 5 + col
            triangles.extend(((i, i + 1, i + 6), (i, i + 6, i + 5)))
    triangles = np.asarray(triangles, np.int64)
    _, volumes, _ = direct._g_triangle_quadrature(coordinates, triangles, "test cloud")
    return {"coordinates": coordinates, "triangles": triangles, "volumes": volumes,
            "bottom": coordinates[:, 1] == 0, "top": coordinates[:, 1] == 1,
            "anchor_particle": np.asarray(0, np.int64)}


class SplitHorizonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")

    def make_model(self, delta_u, delta_d):
        return build_model(square_cloud(), delta_u=delta_u, delta_d=delta_d,
            grip_extent=0.6, h_grip=0.2, visibility_tolerance=1e-12,
            E=210000.0, nu=0.3, k_res=1e-8, rank_rtol=1e-12, Gc=2.7, l0=0.024)

    def test_broad_phase_matches_full_predicate_for_nonconvex_polygon(self):
        coords = np.asarray([[0, 0], [0.5, 0], [1, 0], [0, 0.5], [0.5, 0.5], [1, 0.5], [0, 1], [0.5, 1]], np.float64)
        triangles = np.asarray([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6]], np.int64)
        polygon, _, actual, removed = visibility_graph(coords, triangles, 0.8, 1e-12)
        pairs = direct._g_all_strict_pairs(coords, 0.8)
        predicate = direct._g_UnionPolygonPredicate(polygon, 1e-12)
        flags = np.asarray([predicate(coords[i], coords[j]) for i, j in pairs])
        expected = direct._g_directed_edges_from_pairs(pairs[flags])
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(len(removed), 0)

    def test_common_horizon_reproduces_frozen_static_and_constraints(self):
        cloud = square_cloud()
        _, _, edges, _ = visibility_graph(cloud["coordinates"], cloud["triangles"], 0.6, 1e-12)
        geometry = direct.build_union_geometry_from_arrays(cloud["coordinates"], cloud["triangles"], cloud["volumes"],
            cloud["bottom"], cloud["top"], 0, edges, 0.6, 0.2, 1e-12)
        frozen = direct.build_engineering_grip_model(geometry, 210000.0, 0.3, 1e-8, 1e-12, 2.7, 0.024)
        split = self.make_model(0.6, 0.6)
        self.assertEqual(frozen.static_digest, split.static_digest)
        for name in ("coordinates", "material_volumes", "grip_volumes", "directed_edges", "collar_offsets"):
            np.testing.assert_array_equal(getattr(frozen.geometry, name), getattr(split.geometry, name))
        np.testing.assert_array_equal(frozen.constraints.constrained_dofs, split.constraints.constrained_dofs)
        np.testing.assert_allclose(frozen.material_operator.to_scipy().toarray(), split.material_operator.to_scipy().toarray(), rtol=0, atol=0)
        self.assertNotEqual(frozen.model_digest, split.model_digest)

    def test_separate_graphs_keep_grips_and_checked_energy_residual(self):
        model = self.make_model(0.4, 0.6)
        self.assertEqual(model.static.delta, 0.4)
        self.assertEqual(model.material_operator.delta, 0.6)
        self.assertEqual(model.geometry.grip_extent, 0.6)
        self.assertGreater(len(model.geometry.material_damage_edges), len(model.geometry.material_reference_edges))
        collar = np.arange(model.n_union) >= model.n_material
        constrained_y = model.constraints.constrained_dofs[model.constraints.constrained_dofs % 2 == 1] // 2
        np.testing.assert_array_equal(constrained_y, np.flatnonzero(collar))
        self.assertFalse(np.intersect1d(constrained_y, model.geometry.material_top_indices).size)
        backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
        residual = FractureResidual(model, backend)
        points = jnp.asarray(model.geometry.coordinates)
        gradient = jnp.asarray([[1e-4, 2e-5], [-3e-5, 2e-4]])
        q = (points @ gradient.T).reshape(-1)
        damage = jnp.full(model.n_material, 0.3)
        factor = backend._edge_factor(damage)
        F = backend._deformation_gradient(factor, q)
        np.testing.assert_allclose(np.asarray(F), np.broadcast_to(np.eye(2) + np.asarray(gradient), F.shape), rtol=1e-12, atol=1e-13)
        explicit = backend._residual_from_factor(factor, q)
        auto = jax.grad(lambda displacement: backend._energy_from_factor(factor, displacement))(q)
        np.testing.assert_allclose(np.asarray(auto), np.asarray(explicit), rtol=1e-10, atol=1e-10)
        roundoff = 64 * np.finfo(np.float64).eps * max(abs(residual.operator.to_scipy().diagonal()))
        np.testing.assert_allclose(np.asarray(residual.operator.deterministic_matvec(jnp.ones(model.n_material))), 0, atol=roundoff)
        self.assertTrue(np.all(model.static.shape_rank == 2))


if __name__ == "__main__":
    unittest.main()
