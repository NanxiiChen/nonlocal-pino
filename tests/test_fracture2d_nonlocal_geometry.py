"""Focused Stage 1 geometry/quadrature checks for fracture2d."""

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from examples.fracture2d.nonlocal_geometry import (
    ROUNDED_SLOT_TAGS,
    build_particle_cloud,
    segment_is_visible,
)


def _synthetic_mesh():
    """Return a tagged square with a rectangular opening for mesh-data tests."""
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.4],
            [0.5, 0.4],
            [1.0, 0.4],
            [0.0, 0.6],
            [0.5, 0.6],
            [1.0, 0.6],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [4, 5, 7],
            [5, 8, 7],
            [6, 7, 9],
            [7, 10, 9],
            [7, 8, 10],
            [8, 11, 10],
        ],
        dtype=np.int64,
    )
    line_cells = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 5],
            [5, 8],
            [8, 11],
            [11, 10],
            [10, 9],
            [9, 6],
            [6, 7],
            [7, 4],
            [4, 3],
            [3, 0],
        ],
        dtype=np.int64,
    )
    line_tags = np.asarray([1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    mesh = SimpleNamespace(
        points=points,
        cells_dict={
            "triangle": triangles,
            "line": line_cells,
            "vertex": np.asarray([[0]], dtype=np.int64),
        },
        cell_data_dict={
            "gmsh:physical": {
                "triangle": np.full(len(triangles), 10, dtype=np.int64),
                "line": line_tags,
                "vertex": np.asarray([9], dtype=np.int64),
            }
        },
        field_data={
            "bottom": np.asarray([1, 1]),
            "right": np.asarray([2, 1]),
            "top": np.asarray([3, 1]),
            "left_upper": np.asarray([4, 1]),
            "notch_upper": np.asarray([5, 1]),
            "notch_tip": np.asarray([6, 1]),
            "notch_lower": np.asarray([7, 1]),
            "left_lower": np.asarray([8, 1]),
            "anchor": np.asarray([9, 0]),
            "domain": np.asarray([10, 2]),
        },
    )
    return mesh


def _shuffled_mesh(mesh):
    shuffled = copy.deepcopy(mesh)
    permutation = np.asarray([8, 3, 11, 0, 6, 1, 10, 4, 9, 2, 7, 5])
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation))
    shuffled.points = mesh.points[permutation]
    shuffled.cells_dict["triangle"] = inverse[mesh.cells_dict["triangle"]]
    shuffled.cells_dict["line"] = inverse[mesh.cells_dict["line"]]
    shuffled.cells_dict["vertex"] = inverse[mesh.cells_dict["vertex"]]
    # Reordering cell blocks is also harmless because cells_dict represents
    # the normalized meshio view and physical arrays remain aligned.
    shuffled.cells_dict["line"] = shuffled.cells_dict["line"][::-1]
    shuffled.cell_data_dict["gmsh:physical"]["line"] = (
        mesh.cell_data_dict["gmsh:physical"]["line"][::-1]
    )
    shuffled.cells_dict["triangle"] = shuffled.cells_dict["triangle"][::-1]
    shuffled.cell_data_dict["gmsh:physical"]["triangle"] = np.full(
        len(shuffled.cells_dict["triangle"]), 10, dtype=np.int64
    )
    return shuffled


def _cloud(mesh=None):
    return build_particle_cloud(
        _synthetic_mesh() if mesh is None else mesh,
        horizon=0.75,
        boundary_tags=ROUNDED_SLOT_TAGS,
        visibility_tolerance=1.0e-12,
    )


def _particle_index(cloud, coordinate):
    matches = np.flatnonzero(
        np.all(np.isclose(cloud.coordinates, np.asarray(coordinate)), axis=1)
    )
    if len(matches) != 1:
        raise AssertionError(f"Expected one particle at {coordinate}, got {matches}")
    return int(matches[0])


def _multi_chord_slot(scale=1.0):
    angles = np.linspace(0.5 * np.pi, -0.5 * np.pi, 21)
    cap = np.column_stack((1.0 + np.cos(angles), np.sin(angles)))
    polygon = np.vstack(([[0.0, 1.0]], cap, [[0.0, -1.0]])) * scale
    opening = np.asarray([[0.0, 1.0], [0.0, -1.0]], dtype=float) * scale
    return polygon, opening


class Fracture2DNonlocalGeometryTest(unittest.TestCase):
    def test_tag_masks_barycentric_volumes_and_first_moments(self):
        cloud = _cloud()

        self.assertEqual(cloud.coordinates.shape, (12, 2))
        self.assertEqual(cloud.triangles.shape, (10, 3))
        self.assertEqual(cloud.coordinates.dtype, np.float64)
        self.assertEqual(cloud.volumes.dtype, np.float64)
        self.assertTrue(np.all(cloud.volumes > 0.0))
        self.assertTrue(np.all(np.isfinite(cloud.volumes)))
        self.assertAlmostEqual(cloud.material_area, 0.9)
        np.testing.assert_allclose(cloud.first_moments, [0.475, 0.45], atol=1.0e-14)
        np.testing.assert_array_equal(
            cloud.coordinates[cloud.boundary.bottom][:, 1], 0.0
        )
        np.testing.assert_array_equal(cloud.boundary.bottom.sum(), 3)
        np.testing.assert_array_equal(cloud.boundary.top.sum(), 3)
        np.testing.assert_array_equal(cloud.boundary.anchor_mask.sum(), 1)
        np.testing.assert_allclose(
            cloud.coordinates[cloud.boundary.anchor_mask][0], [0.0, 0.0]
        )
        self.assertEqual(cloud.boundary.anchor_particle, 0)
        self.assertEqual(set(cloud.boundary.outer_sides), {"right", "left_upper", "left_lower"})
        self.assertEqual(set(cloud.boundary.slot_faces), {"upper", "lower"})
        self.assertTrue(cloud.boundary.slot_tip.any())

    def test_particle_cloud_rejects_non_float64_direct_baseline_data(self):
        cloud = _cloud()
        with self.assertRaisesRegex(ValueError, "coordinates must use float64"):
            replace(cloud, coordinates=cloud.coordinates.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "volumes must use float64"):
            replace(cloud, volumes=cloud.volumes.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "integer values"):
            replace(cloud, visible_edges=cloud.visible_edges.astype(np.float64))

    def test_vertex_and_triangle_ordering_is_deterministic(self):
        first = _cloud()
        second = _cloud(_shuffled_mesh(_synthetic_mesh()))

        np.testing.assert_array_equal(first.coordinates, second.coordinates)
        np.testing.assert_array_equal(first.triangles, second.triangles)
        np.testing.assert_allclose(first.triangle_areas, second.triangle_areas)
        np.testing.assert_allclose(first.volumes, second.volumes)
        np.testing.assert_array_equal(first.visible_edges, second.visible_edges)
        np.testing.assert_array_equal(first.removed_edges, second.removed_edges)
        np.testing.assert_array_equal(
            first.boundary.bottom, second.boundary.bottom
        )
        self.assertEqual(first.boundary.anchor_particle, second.boundary.anchor_particle)

    def test_graph_is_reciprocal_sorted_and_visibility_removes_pairs(self):
        cloud = _cloud()

        for edge_set in (cloud.candidate_edges, cloud.visible_edges, cloud.removed_edges):
            self.assertEqual(len(edge_set) % 2, 0)
            self.assertEqual(len(np.unique(edge_set, axis=0)), len(edge_set))
            self.assertTrue(np.all(edge_set[:, 0] != edge_set[:, 1]))
            self.assertTrue(
                np.all(
                    np.lexsort((edge_set[:, 1], edge_set[:, 0]))
                    == np.arange(len(edge_set))
                )
            )
            edge_lookup = {tuple(edge) for edge in edge_set.tolist()}
            self.assertTrue(all((j, i) in edge_lookup for i, j in edge_lookup))
        self.assertGreater(cloud.removed_pair_count, 0)
        self.assertLess(cloud.visible_pair_count, cloud.candidate_pair_count)
        candidate_distances = np.linalg.norm(
            cloud.coordinates[cloud.candidate_edges[:, 1]]
            - cloud.coordinates[cloud.candidate_edges[:, 0]],
            axis=1,
        )
        self.assertTrue(np.all(candidate_distances < cloud.horizon))
        self.assertEqual(cloud.neighbor_min, int(np.min(cloud.neighbor_counts)))
        self.assertEqual(cloud.neighbor_max, int(np.max(cloud.neighbor_counts)))

    def test_polygon_visibility_keeps_closure_and_removes_void_segments(self):
        cloud = _cloud()
        visible = {tuple(edge) for edge in cloud.visible_edges.tolist()}
        removed = {tuple(edge) for edge in cloud.removed_edges.tolist()}

        upper_lip = _particle_index(cloud, [0.0, 0.6])
        upper_tip = _particle_index(cloud, [0.5, 0.6])
        lower_tip = _particle_index(cloud, [0.5, 0.4])
        lower_lip = _particle_index(cloud, [0.0, 0.4])
        tangent_target = _particle_index(cloud, [1.0, 1.0])
        outer_target = _particle_index(cloud, [0.5, 0.0])

        def has_pair(edge_set, first, second):
            return (first, second) in edge_set and (second, first) in edge_set

        self.assertTrue(has_pair(visible, upper_tip, lower_tip))
        self.assertTrue(has_pair(visible, upper_lip, upper_tip))
        self.assertTrue(has_pair(visible, lower_lip, lower_tip))
        self.assertTrue(has_pair(visible, upper_tip, tangent_target))
        self.assertTrue(has_pair(visible, _particle_index(cloud, [0.0, 0.0]), outer_target))
        self.assertTrue(has_pair(removed, upper_lip, lower_tip))
        self.assertTrue(has_pair(removed, upper_lip, lower_lip))

    def test_polygon_segment_classifier_is_strict_and_explicit(self):
        polygon = np.asarray(
            [[0.0, 0.6], [0.5, 0.6], [0.5, 0.4], [0.0, 0.4]],
            dtype=float,
        )
        opening = np.asarray([[0.0, 0.6], [0.0, 0.4]], dtype=float)
        tolerance = 1.0e-12

        self.assertFalse(
            segment_is_visible(
                [0.1, 0.5], [0.55, 0.5], polygon, opening, tolerance
            )
        )
        self.assertFalse(
            segment_is_visible(
                [0.0, 0.6], [0.0, 0.4], polygon, opening, tolerance
            )
        )
        self.assertTrue(
            segment_is_visible(
                [0.0, 0.6], [0.5, 0.6], polygon, opening, tolerance
            )
        )
        self.assertTrue(
            segment_is_visible(
                [0.5, 0.6], [0.8, 0.8], polygon, opening, tolerance
            )
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            segment_is_visible(
                [0.1, 0.6005], [0.4, 0.6005], polygon, opening, 1.0e-3
            )

    def test_zero_tolerance_keeps_every_multichord_slot_boundary_edge(self):
        polygon, opening = _multi_chord_slot()
        for tolerance in (0.0, 1.0e-12):
            for start, end in zip(polygon[:-1], polygon[1:]):
                self.assertTrue(
                    segment_is_visible(start, end, polygon, opening, tolerance),
                    msg=f"boundary edge was removed at tolerance={tolerance}",
                )

    def test_shallow_crossing_is_false_or_explicitly_ambiguous(self):
        polygon = np.asarray(
            [[0.0, 0.6], [0.5, 0.6], [0.5, 0.4], [0.0, 0.4]],
            dtype=float,
        )
        opening = np.asarray([[0.0, 0.6], [0.0, 0.4]], dtype=float)
        start = [-0.001, 0.598999999999]
        end = [0.001, 0.601]
        try:
            result = segment_is_visible(start, end, polygon, opening, 1.0e-12)
        except ValueError as error:
            self.assertRegex(str(error), "ambiguous")
        else:
            self.assertFalse(result)

    def test_visibility_is_endpoint_symmetric_and_scale_invariant(self):
        polygon, opening = _multi_chord_slot()
        cases = (
            (np.asarray([0.2, 0.0]), np.asarray([1.8, 0.0]), False),
            (polygon[0], polygon[1], True),
            (np.asarray([1.0, 1.0]), np.asarray([2.0, 1.0]), True),
            (opening[0], opening[1], False),
        )
        for start, end, expected in cases:
            forward = segment_is_visible(start, end, polygon, opening, 1.0e-12)
            reverse = segment_is_visible(end, start, polygon, opening, 1.0e-12)
            self.assertEqual(forward, expected)
            self.assertEqual(reverse, expected)

        for scale in (1.0e-6, 1.0e6):
            scaled_polygon, scaled_opening = _multi_chord_slot(scale)
            for start, end, expected in cases:
                scaled_start = start * scale
                scaled_end = end * scale
                scaled_tolerance = 1.0e-12 * scale
                self.assertEqual(
                    segment_is_visible(
                        scaled_start,
                        scaled_end,
                        scaled_polygon,
                        scaled_opening,
                        scaled_tolerance,
                    ),
                    expected,
                )

    def test_boundary_coverage_internal_and_duplicate_tags_fail(self):
        invalid_slot = _synthetic_mesh()
        invalid_slot.cell_data_dict["gmsh:physical"]["line"][10] = 5
        with self.assertRaisesRegex(ValueError, "slot topology"):
            _cloud(invalid_slot)

        missing = _synthetic_mesh()
        missing.cells_dict["line"] = np.delete(missing.cells_dict["line"], 0, axis=0)
        missing.cell_data_dict["gmsh:physical"]["line"] = np.delete(
            missing.cell_data_dict["gmsh:physical"]["line"], 0
        )
        with self.assertRaisesRegex(ValueError, "untagged"):
            _cloud(missing)

        internal = _synthetic_mesh()
        internal.cells_dict["line"] = np.vstack(
            (internal.cells_dict["line"], np.asarray([[1, 4]], dtype=np.int64))
        )
        internal.cell_data_dict["gmsh:physical"]["line"] = np.append(
            internal.cell_data_dict["gmsh:physical"]["line"], 5
        )
        with self.assertRaisesRegex(ValueError, "internal"):
            _cloud(internal)

        duplicate = _synthetic_mesh()
        duplicate.cells_dict["line"] = np.vstack(
            (duplicate.cells_dict["line"], np.asarray([[0, 1]], dtype=np.int64))
        )
        duplicate.cell_data_dict["gmsh:physical"]["line"] = np.append(
            duplicate.cell_data_dict["gmsh:physical"]["line"], 1
        )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _cloud(duplicate)

        inconsistent = _synthetic_mesh()
        inconsistent.cells_dict["line"] = np.vstack(
            (inconsistent.cells_dict["line"], np.asarray([[0, 1]], dtype=np.int64))
        )
        inconsistent.cell_data_dict["gmsh:physical"]["line"] = np.append(
            inconsistent.cell_data_dict["gmsh:physical"]["line"], 2
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _cloud(inconsistent)

    def test_missing_physical_tag_fails_without_coordinate_fallback(self):
        mesh = _synthetic_mesh()
        del mesh.field_data["top"]
        with self.assertRaisesRegex(ValueError, "physical name 'top'"):
            _cloud(mesh)

    def test_unknown_boundary_tag_and_degenerate_triangle_fail(self):
        unknown = _synthetic_mesh()
        unknown.cell_data_dict["gmsh:physical"]["line"][0] = 99
        with self.assertRaisesRegex(ValueError, "Unknown physical line tag"):
            _cloud(unknown)

        degenerate = _synthetic_mesh()
        degenerate.cells_dict["triangle"] = np.vstack(
            (degenerate.cells_dict["triangle"], np.asarray([[0, 0, 1]]))
        )
        degenerate.cell_data_dict["gmsh:physical"]["triangle"] = np.full(
            len(degenerate.cells_dict["triangle"]), 10, dtype=np.int64
        )
        with self.assertRaisesRegex(ValueError, "degenerate triangle"):
            _cloud(degenerate)

        collinear = _synthetic_mesh()
        collinear.cells_dict["triangle"] = np.vstack(
            (collinear.cells_dict["triangle"], np.asarray([[0, 1, 2]]))
        )
        collinear.cell_data_dict["gmsh:physical"]["triangle"] = np.full(
            len(collinear.cells_dict["triangle"]), 10, dtype=np.int64
        )
        with self.assertRaisesRegex(ValueError, "internal"):
            _cloud(collinear)

        missing_anchor = _synthetic_mesh()
        missing_anchor.cells_dict["vertex"] = np.empty((0, 1), dtype=np.int64)
        missing_anchor.cell_data_dict["gmsh:physical"]["vertex"] = np.empty(
            (0,), dtype=np.int64
        )
        with self.assertRaisesRegex(ValueError, "exactly one mesh point"):
            _cloud(missing_anchor)


if __name__ == "__main__":
    unittest.main()
