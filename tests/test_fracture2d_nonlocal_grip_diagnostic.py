"""Focused executable-contract checks for the isolated Candidate A2 branch."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.fracture2d.nonlocal_grip_diagnostic import (
    AffineConstraints,
    CandidateA2GateFailure,
    FixedDamagePhaseMeasuredMechanics,
    _UnionPolygonPredicate,
    _boundary_polygon_from_triangles,
    _reaction_balance_metrics,
    build_affine_constraints,
    build_union_geometry_from_arrays,
    compute_h_used,
    compute_material_metrics,
    independent_union_ba_reference,
    load_frozen_sources,
    run_manufactured_gates,
    solve_fixed_damage_a2,
    transition_decision_arithmetic,
)


def _material_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_side = 5
    axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    coordinates = np.asarray(
        [(x, y) for x in axis for y in axis], dtype=np.float64
    )
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n_side - 1):
        for iy in range(n_side - 1):
            ll = ix * n_side + iy
            lr = (ix + 1) * n_side + iy
            ul = ix * n_side + iy + 1
            ur = (ix + 1) * n_side + iy + 1
            triangles.extend(((ll, lr, ur), (ll, ur, ul)))
    triangles_array = np.asarray(triangles, dtype=np.int64)
    p0 = coordinates[triangles_array[:, 0]]
    p1 = coordinates[triangles_array[:, 1]]
    p2 = coordinates[triangles_array[:, 2]]
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    volumes = np.zeros(len(coordinates), dtype=np.float64)
    for column in range(3):
        np.add.at(volumes, triangles_array[:, column], areas / 3.0)
    bottom = coordinates[:, 1] == 0.0
    top = coordinates[:, 1] == 1.0
    pairs: list[tuple[int, int]] = []
    delta = 0.48
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
            if 0.0 < distance < delta:
                pairs.extend(((first, second), (second, first)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    return coordinates, triangles_array, volumes, bottom, top, edges


def _geometry(h_g: float = 0.2):
    coordinates, triangles, volumes, bottom, top, edges = _material_case()
    return build_union_geometry_from_arrays(
        coordinates,
        triangles,
        volumes,
        bottom,
        top,
        0,
        edges,
        0.48,
        h_g,
        0.0,
    )


def _slot_geometry():
    """Small concave material mesh for independent slot-visibility checks."""

    coordinates = np.asarray(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.6),
            (0.5, 0.6),
            (0.5, 0.4),
            (0.0, 0.4),
            (0.5, 0.0),
            (0.5, 1.0),
            (1.0, 0.4),
            (1.0, 0.6),
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [
            (8, 1, 10),
            (8, 10, 6),
            (6, 10, 11),
            (6, 11, 5),
            (5, 11, 2),
            (5, 2, 9),
            (0, 8, 6),
            (0, 6, 7),
            (3, 9, 5),
            (3, 5, 4),
        ],
        dtype=np.int64,
    )
    p0 = coordinates[triangles[:, 0]]
    p1 = coordinates[triangles[:, 1]]
    p2 = coordinates[triangles[:, 2]]
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    volumes = np.zeros(len(coordinates), dtype=np.float64)
    for column in range(3):
        np.add.at(volumes, triangles[:, column], areas / 3.0)
    bottom = np.isclose(coordinates[:, 1], 0.0)
    top = np.isclose(coordinates[:, 1], 1.0)
    delta = 0.8
    predicate = _UnionPolygonPredicate(
        _boundary_polygon_from_triangles(coordinates, triangles), 0.0
    )
    pairs: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
            if 0.0 < distance < delta and predicate(coordinates[first], coordinates[second]):
                pairs.append((first, second))
    undirected = np.asarray(pairs, dtype=np.int64)
    directed = np.vstack((undirected, undirected[:, ::-1]))
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    return build_union_geometry_from_arrays(
        coordinates,
        triangles,
        volumes,
        bottom,
        top,
        0,
        directed,
        delta,
        0.2,
        0.0,
    )


class CandidateA2GeometryContractTest(unittest.TestCase):
    def test_prefix_phase_quadrature_and_deterministic_offsets(self) -> None:
        coordinates, triangles, volumes, *_ = _material_case()
        geometry = _geometry(0.2)
        np.testing.assert_array_equal(geometry.coordinates[: len(coordinates)], coordinates)
        np.testing.assert_array_equal(geometry.material_triangles, triangles)
        np.testing.assert_array_equal(geometry.material_volumes[: len(volumes)], volumes)
        np.testing.assert_array_equal(
            geometry.material_volumes[len(volumes) :],
            np.zeros(geometry.n_particles - len(volumes), dtype=np.float64),
        )
        np.testing.assert_array_equal(
            geometry.collar_offsets,
            np.asarray([0.2, 0.4, 0.48], dtype=np.float64),
        )
        self.assertAlmostEqual(geometry.material_area, 1.0, places=14)
        self.assertAlmostEqual(geometry.bottom_grip_area, 0.48, places=14)
        self.assertAlmostEqual(geometry.top_grip_area, 0.48, places=14)
        self.assertAlmostEqual(geometry.union_area, 1.96, places=14)
        np.testing.assert_allclose(
            geometry.union_first_moments,
            np.asarray([0.98, 0.98], dtype=np.float64),
            rtol=0.0,
            atol=5.0e-15,
        )
        self.assertFalse(np.any(np.isnan(geometry.grip_volumes)))
        self.assertTrue(np.all(geometry.union_volumes > 0.0))

        refined = _geometry(0.1)
        np.testing.assert_array_equal(
            refined.collar_offsets,
            np.asarray([0.1, 0.2, 0.1 * 3.0, 0.4, 0.48], dtype=np.float64),
        )
        np.testing.assert_array_equal(refined.coordinates[: len(coordinates)], coordinates)
        np.testing.assert_array_equal(refined.material_triangles, triangles)

    def test_independent_union_graph_and_material_induction(self) -> None:
        _, _, _, _, _, material_edges = _material_case()
        geometry = _geometry()
        self.assertEqual(len(geometry.directed_edges) % 2, 0)
        edge_set = {tuple(edge) for edge in geometry.directed_edges.tolist()}
        self.assertTrue(all((j, i) in edge_set for i, j in edge_set))
        self.assertTrue(
            np.all(
                np.lexsort((geometry.directed_edges[:, 1], geometry.directed_edges[:, 0]))
                == np.arange(len(geometry.directed_edges))
            )
        )
        induced = geometry.directed_edges[
            (geometry.directed_edges[:, 0] < geometry.n_material)
            & (geometry.directed_edges[:, 1] < geometry.n_material)
        ]
        np.testing.assert_array_equal(induced, material_edges)

    def test_slot_crossing_leave_reenter_and_graph_completeness(self) -> None:
        geometry = _slot_geometry()
        predicate = _UnionPolygonPredicate(geometry.boundary_polygon, 0.0)
        lower = np.asarray([0.25, 0.2], dtype=np.float64)
        upper = np.asarray([0.25, 0.8], dtype=np.float64)
        self.assertFalse(predicate(lower, upper), "leave/re-enter path crossed the slot")
        self.assertFalse(
            predicate(lower, np.asarray([0.75, 0.8], dtype=np.float64)),
            "diagonal slot-crossing segment was incorrectly visible",
        )
        self.assertTrue(predicate(lower, np.asarray([0.75, 0.2], dtype=np.float64)))
        self.assertTrue(predicate(upper, np.asarray([0.75, 0.8], dtype=np.float64)))

        candidate = {tuple(edge) for edge in geometry.candidate_edges.tolist()}
        visible = {tuple(edge) for edge in geometry.directed_edges.tolist()}
        removed = {tuple(edge) for edge in geometry.removed_edges.tolist()}
        self.assertEqual(visible & removed, set())
        self.assertEqual(visible | removed, candidate)
        for first, second in candidate:
            self.assertIn((second, first), candidate)
            self.assertIn((second, first), visible | removed)
        for first in range(geometry.n_particles):
            for second in range(first + 1, geometry.n_particles):
                pair = (first, second)
                if pair not in candidate:
                    continue
                expected = predicate(geometry.coordinates[first], geometry.coordinates[second])
                self.assertEqual(pair in visible, expected)
                self.assertEqual(
                    predicate(geometry.coordinates[first], geometry.coordinates[second]),
                    predicate(geometry.coordinates[second], geometry.coordinates[first]),
                )

    def test_geometry_forgeries_fail_loudly(self) -> None:
        coordinates, triangles, volumes, bottom, top, edges = _material_case()
        forged_coordinates = coordinates.copy()
        forged_coordinates[np.flatnonzero(top)[2], 0] += 1.0e-4
        with self.assertRaisesRegex(ValueError, "volumes|uniform|material"):
            build_union_geometry_from_arrays(
                forged_coordinates,
                triangles,
                volumes,
                bottom,
                top,
                0,
                edges,
                0.48,
                0.2,
                0.0,
            )
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            build_union_geometry_from_arrays(
                coordinates,
                triangles,
                volumes,
                bottom,
                top,
                0,
                edges[:-1],
                0.48,
                0.2,
                0.0,
            )


class CandidateA2MechanicsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = _geometry()
        cls.damage = np.asarray(
            0.1 + 0.4 * np.linspace(0.0, 1.0, cls.geometry.n_material),
            dtype=np.float64,
        )
        cls.mechanics = FixedDamagePhaseMeasuredMechanics(
            cls.geometry,
            cls.damage,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
        )
        cls.reference = independent_union_ba_reference(
            cls.geometry, cls.damage, 1.0e-8, 1.0e-12
        )

    def test_phase_factor_reference_and_immutable_state(self) -> None:
        expected = self.reference.phase_center_weights[self.reference.source] * self.reference.eta
        np.testing.assert_allclose(
            self.mechanics.center_factor,
            expected,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            self.reference.phase_center_weights,
            self.geometry.material_volumes
            * np.pad(
                (1.0 - self.damage) ** 2 + 1.0e-8,
                (0, self.geometry.n_particles - self.geometry.n_material),
            )
            + self.geometry.grip_volumes * (1.0 + 1.0e-8),
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertAlmostEqual(self.reference.grip_degradation, 1.0 + 1.0e-8)
        self.assertEqual(
            len(np.flatnonzero((self.geometry.material_volumes > 0.0) & (self.geometry.grip_volumes == 0.0))),
            self.geometry.n_material - 10,
        )
        interface = np.flatnonzero(
            (self.geometry.material_volumes > 0.0) & (self.geometry.grip_volumes > 0.0)
        )
        grip_only = np.flatnonzero(
            (self.geometry.material_volumes == 0.0) & (self.geometry.grip_volumes > 0.0)
        )
        self.assertGreater(len(interface), 0)
        self.assertGreater(len(grip_only), 0)
        material_only = np.flatnonzero(
            (self.geometry.material_volumes > 0.0) & (self.geometry.grip_volumes == 0.0)
        )
        self.assertTrue(np.all(self.reference.phase_center_weights[grip_only] > 0.0))
        self.assertTrue(np.all(self.reference.phase_center_weights[material_only] > 0.0))
        original = self.reference.phase_center_weights.copy()
        forged_damage = self.damage.copy()
        forged_damage[0] = 0.99
        np.testing.assert_array_equal(self.reference.phase_center_weights, original)
        forged_reference = independent_union_ba_reference(
            self.geometry, forged_damage, 1.0e-8, 1.0e-12
        )
        self.assertNotEqual(
            float(forged_reference.phase_center_weights[0]), float(original[0])
        )
        with self.assertRaises(ValueError):
            self.reference.phase_center_weights.setflags(write=True)
        with self.assertRaises(ValueError):
            self.geometry.grip_volumes.setflags(write=True)
        np.testing.assert_allclose(
            self.mechanics.center_factor,
            self.mechanics.geometry.union_volumes[self.reference.source]
            * self.mechanics.effective_center_degradation[self.reference.source]
            * self.reference.eta,
            rtol=0.0,
            atol=2.0e-15,
        )
        self.assertTrue(np.all(np.isfinite(self.mechanics.damage_union)))
        with self.assertRaises(AttributeError):
            self.mechanics.center_factor = np.ones_like(self.mechanics.center_factor)
        with self.assertRaises(ValueError):
            self.mechanics.material_damage.setflags(write=True)

    def test_affine_patch_energy_jvp_and_symmetry(self) -> None:
        affine = np.asarray([[1.12, 0.035], [-0.018, 0.93]], dtype=np.float64)
        translation = np.asarray([0.13, -0.22], dtype=np.float64)
        displacement = (
            self.geometry.coordinates @ (affine - np.eye(2)).T + translation
        ).ravel()
        evaluation = self.mechanics.evaluate(displacement)
        np.testing.assert_allclose(
            evaluation.edge_F,
            np.broadcast_to(affine, evaluation.edge_F.shape),
            rtol=0.0,
            atol=5.0e-14,
        )
        rng = np.random.default_rng(44)
        displacement = np.asarray(1.0e-2 * rng.normal(size=self.geometry.n_dofs), dtype=np.float64)
        direction = np.asarray(rng.normal(size=self.geometry.n_dofs), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        epsilon = 2.0e-5
        energy_fd = (
            self.mechanics.energy(displacement + epsilon * direction)
            - self.mechanics.energy(displacement - epsilon * direction)
        ) / (2.0 * epsilon)
        self.assertLess(
            abs(energy_fd - float(self.mechanics.residual(displacement) @ direction))
            / max(abs(energy_fd), 1.0e-30),
            1.0e-10,
        )
        residual_jvp = (
            self.mechanics.residual(displacement + epsilon * direction)
            - self.mechanics.residual(displacement - epsilon * direction)
        ) / (2.0 * epsilon)
        self.assertLess(
            np.linalg.norm(residual_jvp - self.mechanics.tangent_action(direction))
            / max(np.linalg.norm(residual_jvp), 1.0e-30),
            1.0e-10,
        )
        dense = np.column_stack(
            [
                self.mechanics.tangent_action(np.eye(self.geometry.n_dofs)[:, column])
                for column in range(self.geometry.n_dofs)
            ]
        )
        np.testing.assert_allclose(dense, dense.T, rtol=0.0, atol=5.0e-12)

    def test_outer_constraints_opening_reactions_and_work_chain(self) -> None:
        constraints = build_affine_constraints(
            self.geometry.coordinates,
            self.geometry.outer_bottom_mask,
            self.geometry.outer_top_mask,
            0,
            self.geometry.delta,
        )
        self.assertIsInstance(constraints, AffineConstraints)
        interface_y = np.concatenate(
            (
                2 * self.geometry.material_bottom_indices + 1,
                2 * self.geometry.material_top_indices + 1,
            )
        )
        self.assertFalse(np.any(np.isin(interface_y, constraints.constrained_dofs)))
        unit = solve_fixed_damage_a2(
            self.mechanics, constraints, 1.0, 1.0e-10, 1.0e-12, 3000
        )
        from examples.fracture2d.nonlocal_grip_diagnostic import (
            physical_interface_line_weights,
            physical_interface_opening,
        )

        top_line = physical_interface_line_weights(self.geometry, "top")
        bottom_line = physical_interface_line_weights(self.geometry, "bottom")
        slope = physical_interface_opening(unit.displacement, self.geometry, top_line, bottom_line)
        self.assertGreater(slope, 0.0)
        result = solve_fixed_damage_a2(
            self.mechanics, constraints, 0.08, 1.0e-10, 1.0e-12, 3000
        )
        plus = solve_fixed_damage_a2(
            self.mechanics, constraints, 0.08 + 1.0e-5, 1.0e-10, 1.0e-12, 3000
        )
        minus = solve_fixed_damage_a2(
            self.mechanics, constraints, 0.08 - 1.0e-5, 1.0e-10, 1.0e-12, 3000
        )
        R_fd = (plus.energy - minus.energy) / (2.0e-5)
        D_plus = physical_interface_opening(plus.displacement, self.geometry, top_line, bottom_line)
        D_minus = physical_interface_opening(minus.displacement, self.geometry, top_line, bottom_line)
        R_eff_fd = (plus.energy - minus.energy) / (D_plus - D_minus)
        self.assertLess(abs(R_fd - result.outer_reaction) / max(abs(R_fd), 1.0), 1.0e-7)
        self.assertLess(
            abs(R_eff_fd - result.outer_reaction / slope)
            / max(abs(R_eff_fd), 1.0),
            1.0e-7,
        )
        np.testing.assert_allclose(result.total_force, 0.0, rtol=0.0, atol=1.0e-10)
        self.assertLessEqual(abs(result.total_moment), 1.0e-10)
        balance = _reaction_balance_metrics(result, self.mechanics)
        self.assertAlmostEqual(balance["top_reaction"], result.outer_reaction)
        self.assertAlmostEqual(balance["bottom_reaction"], result.bottom_reaction)
        self.assertAlmostEqual(balance["anchor_reaction"], result.anchor_reaction)
        np.testing.assert_allclose(
            [balance["equilibrium_force_x"], balance["equilibrium_force_y"]],
            result.total_force,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            balance["equilibrium_reference_moment"], result.total_moment, places=12
        )
        self.assertLess(balance["equilibrium_force_error"], 1.0e-10)
        self.assertLess(balance["equilibrium_reference_moment_error"], 1.0e-10)

    def test_constraints_and_reference_forgeries_fail(self) -> None:
        constraints = build_affine_constraints(
            self.geometry.coordinates,
            self.geometry.outer_bottom_mask,
            self.geometry.outer_top_mask,
            0,
            self.geometry.delta,
        )
        with self.assertRaisesRegex(ValueError, "load_direction"):
            replace(
                constraints,
                load_direction=np.asarray(constraints.load_direction).copy()
                + np.eye(1, constraints.n_dofs, 0).reshape(-1),
            )
        bad_bottom = self.geometry.interface_bottom_mask.copy()
        with self.assertRaisesRegex(ValueError, "outer_bottom_mask"):
            build_affine_constraints(
                self.geometry.coordinates,
                bad_bottom,
                self.geometry.outer_top_mask,
                0,
                self.geometry.delta,
            )


class CandidateA2ManufacturedGateTest(unittest.TestCase):
    def test_history_floor_and_transition_decision_arithmetic(self) -> None:
        psi_plus = np.asarray([2.0, 8.0, 1.0], dtype=np.float64)
        history = np.asarray([5.0, 3.0, 1.0], dtype=np.float64)
        np.testing.assert_array_equal(
            compute_h_used(psi_plus, history),
            np.asarray([5.0, 8.0, 1.0], dtype=np.float64),
        )
        with self.assertRaises(ValueError):
            compute_h_used(psi_plus, np.asarray([5.0, -1.0, 1.0], dtype=np.float64))
        decision = transition_decision_arithmetic(0.8, 96.2346, 100.0, True, 0.09)
        self.assertAlmostEqual(decision["top_error_reduction_factor"], 0.962346)
        self.assertAlmostEqual(decision["top_error_reduction_fraction"], 0.037654)
        self.assertFalse(decision["diagnostically_promising"])
        passing = transition_decision_arithmetic(0.8, 49.9, 100.0, True, 0.09)
        self.assertTrue(passing["diagnostically_promising"])

    def test_manufactured_gate_precedes_union_staging(self) -> None:
        metrics = run_manufactured_gates()
        self.assertTrue(metrics["hard_gates_pass"])
        self.assertEqual(metrics["rigid_zero_modes"], 3)
        self.assertLess(metrics["energy_gradient_relative_error"], 1.0e-10)
        self.assertLess(metrics["opening_chain_relative_error"], 1.0e-7)

    @unittest.skipUnless(
        Path("/tmp/opencode/fracture2d_d048_warning_ufinal_0075.raw.npz").is_file()
        and Path("/tmp/opencode/stage3_cloud_float64.npz").is_file()
        and Path("/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz").is_file(),
        "frozen nonproduction provenance artifacts are not installed",
    )
    def test_exact_frozen_provenance_and_raw_forgery_rejection(self) -> None:
        sources = load_frozen_sources(
            "/tmp/opencode/fracture2d_d048_warning_ufinal_0075.raw.npz",
            "/tmp/opencode/stage3_cloud_float64.npz",
            "/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz",
        )
        self.assertFalse(sources.raw_validated)
        self.assertAlmostEqual(sources.committed_load, 0.0055, places=14)
        self.assertEqual(sources.raw_index, 54)
        self.assertEqual(sources.fem_snapshot_index, 56)
        self.assertEqual(sources.raw_sha256, "48d92621955289c51710296569362c50db30e28f63d622ee447c11a2317f66e5")
        self.assertEqual(sources.fem_sha256, "e9634f1f8fdb21116b50ad453d37115746c4a5a09f828e76245a9c59f6107d17")
        with tempfile.TemporaryDirectory(prefix="candidate_a2_forgery_") as directory:
            forged_raw = Path(directory) / "forged.raw.npz"
            payload = bytearray(Path(sources.raw_path).read_bytes())
            payload[-1] ^= 1
            forged_raw.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_frozen_sources(forged_raw, sources.cloud_path, sources.fem_path)


if __name__ == "__main__":
    unittest.main()
