"""Independent Stage 4 checks for the fracture2d fixed subproblems.

The reference helpers below rebuild the directed graph algebra, bond-associated
supports, constitutive response, and damage energy without calling production
energy/residual/tangent methods.  They keep the Stage 4 tests useful against
shared implementation errors rather than only checking internal identities.
"""

from __future__ import annotations

from typing import cast
from unittest import mock
import unittest

import jax
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.nonlocal_baseline import (  # noqa: E402
    amor_positive_energy_from_strain,
    compute_amor_history,
    compute_amor_history_from_edge_strain,
    solve_fixed_damage_mechanics,
    solve_fixed_history_damage,
    update_history_max,
)
from examples.fracture2d.nonlocal_mechanics import (  # noqa: E402
    BondAssociatedMechanics,
    FixedDamageBondAssociatedMechanics,
)
from pinn.operators import ExplicitVolumeSparseD  # noqa: E402


def _edges(coordinates: np.ndarray, delta: float) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[j] - coordinates[i]))
            if 0.0 < distance < delta:
                pairs.extend(((i, j), (j, i)))
    return np.asarray(pairs[::-1], dtype=np.int64)


def _cloud(n_side: int = 5, delta: float = 2.01):
    coordinates = np.asarray(
        [(float(ix), float(iy)) for ix in range(n_side) for iy in range(n_side)],
        dtype=np.float64,
    )
    indices = np.arange(len(coordinates), dtype=np.float64)
    volumes = np.asarray(
        0.8 + 0.025 * (indices % 7.0) + 0.01 * ((indices // 7.0) % 5.0),
        dtype=np.float64,
    )
    return coordinates, volumes, _edges(coordinates, delta), delta


def _reference_ba(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    directed_edges: np.ndarray,
    delta: float,
):
    edges = np.asarray(
        sorted((int(i), int(j)) for i, j in directed_edges.tolist()),
        dtype=np.int64,
    )
    distances = np.linalg.norm(
        coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1
    )
    kernel = 120.0 / (np.pi * delta**4) * (1.0 - distances / delta) ** 2
    adjacency = [set() for _ in range(len(coordinates))]
    weights: dict[tuple[int, int], float] = {}
    for (i, j), value in zip(edges, kernel):
        adjacency[int(i)].add(int(j))
        weights[int(i), int(j)] = float(value)
    center_mass = np.zeros(len(coordinates), dtype=np.float64)
    for (i, j), value in zip(edges, kernel):
        center_mass[int(i)] += float(value) * volumes[int(j)]

    supports: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    xis: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    etas = np.empty(len(edges), dtype=np.float64)
    for edge_index, (i_raw, j_raw) in enumerate(edges):
        i, j = int(i_raw), int(j_raw)
        common = sorted(adjacency[i].intersection(adjacency[j]) - {i, j})
        support = np.asarray([j, *common], dtype=np.int64)
        beta = np.asarray(
            [weights[i, j] * volumes[j]]
            + [weights[i, k] * volumes[k] for k in common],
            dtype=np.float64,
        )
        xi = coordinates[support] - coordinates[i]
        shape = np.einsum("s,sa,sb->ab", beta, xi, xi)
        supports.append(support)
        betas.append(beta)
        xis.append(xi)
        inverses.append(np.linalg.inv(shape))
        etas[edge_index] = beta[0] / center_mass[i]
    return edges, supports, betas, xis, inverses, etas, center_mass, kernel


def _reference_mechanics(
    displacement: np.ndarray,
    coordinates: np.ndarray,
    volumes: np.ndarray,
    reference,
    youngs_modulus: float,
    poisson_ratio: float,
    k_res: float,
    damage: np.ndarray,
):
    edges, supports, betas, xis, inverses, etas, _, _ = reference
    lam = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    values = displacement.reshape(len(coordinates), 2)
    residual = np.zeros_like(displacement)
    energy = 0.0
    edge_f = []
    edge_strain = []
    edge_stress = []
    for edge_index, (i_raw, _) in enumerate(edges):
        i = int(i_raw)
        support = supports[edge_index]
        beta = betas[edge_index]
        xi = xis[edge_index]
        inverse = inverses[edge_index]
        moment = np.zeros((2, 2), dtype=np.float64)
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            current_bond = xi[local] + values[k] - values[i]
            moment += beta[local] * np.outer(current_bond, xi[local])
        deformation_gradient = moment @ inverse
        increment = deformation_gradient - np.eye(2)
        strain = 0.5 * (increment + increment.T)
        trace = float(np.trace(strain))
        stress = lam * trace * np.eye(2) + 2.0 * mu * strain
        g = (1.0 - damage[i]) ** 2 + k_res
        factor = volumes[i] * g * etas[edge_index]
        density = 0.5 * lam * trace**2 + mu * np.sum(strain * strain)
        energy += factor * density
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            endpoint = beta[local] * (stress @ inverse.T @ xi[local])
            residual[2 * k : 2 * k + 2] += factor * endpoint
            residual[2 * i : 2 * i + 2] -= factor * endpoint
        edge_f.append(deformation_gradient)
        edge_strain.append(strain)
        edge_stress.append(stress)
    return (
        float(energy),
        residual,
        np.asarray(edge_f, dtype=np.float64),
        np.asarray(edge_strain, dtype=np.float64),
        np.asarray(edge_stress, dtype=np.float64),
    )


def _reference_damage_system(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    edges: np.ndarray,
    delta: float,
    Gc: float,
    l0: float,
    history: np.ndarray,
):
    n_particles = len(coordinates)
    matrix = np.zeros((n_particles, n_particles), dtype=np.float64)
    distances = np.linalg.norm(
        coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1
    )
    kernel = 120.0 / (np.pi * delta**4) * (1.0 - distances / delta) ** 2
    for (i_raw, j_raw), value in zip(edges, kernel):
        i, j = int(i_raw), int(j_raw)
        coefficient = float(value) * volumes[j]
        matrix[i, i] += coefficient
        matrix[i, j] -= coefficient
    strong = np.diag(Gc / l0 + 2.0 * history) + Gc * l0 * matrix
    mass = np.diag(volumes)
    system = mass @ strong
    rhs = 2.0 * volumes * history
    return matrix, strong, system, rhs, kernel


def _reference_damage_energy(
    damage: np.ndarray,
    coordinates: np.ndarray,
    volumes: np.ndarray,
    edges: np.ndarray,
    delta: float,
    Gc: float,
    l0: float,
    history: np.ndarray,
):
    _, _, _, _, kernel = _reference_damage_system(
        coordinates, volumes, edges, delta, Gc, l0, history
    )
    crack = np.sum(volumes * Gc / (2.0 * l0) * damage**2)
    history_term = np.sum(volumes * history * (1.0 - damage) ** 2)
    graph = Gc * l0 / 4.0 * np.sum(
        volumes[edges[:, 0]]
        * kernel
        * volumes[edges[:, 1]]
        * (damage[edges[:, 1]] - damage[edges[:, 0]]) ** 2
    )
    return float(crack), float(history_term), float(graph), float(
        crack + history_term + graph
    )


class Fracture2DNonlocalBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.initial_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        cls.coordinates, cls.volumes, cls.edges, cls.delta = _cloud()
        cls.youngs_modulus = 2.0
        cls.poisson_ratio = 0.3
        cls.k_res = 1.0e-8
        cls.rank_rtol = 1.0e-12
        cls.reference = _reference_ba(
            cls.coordinates, cls.volumes, cls.edges, cls.delta
        )
        cls.mechanics = BondAssociatedMechanics(
            cls.coordinates,
            cls.volumes,
            cls.edges,
            cls.delta,
            cls.youngs_modulus,
            cls.poisson_ratio,
            cls.k_res,
            cls.rank_rtol,
        )
        cls.operator = ExplicitVolumeSparseD(
            cls.coordinates,
            cls.volumes,
            cls.edges,
            cls.delta,
            "cubic",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.initial_x64)

    def test_fixed_history_euclidean_spd_residual_and_independent_energy(self) -> None:
        history = np.asarray(
            0.03 + 0.01 * (np.arange(len(self.coordinates)) % 4),
            dtype=np.float64,
        )
        Gc, l0 = 1.7, 0.65
        result = solve_fixed_history_damage(self.operator, Gc, l0, history)
        D_ref, strong_ref, system_ref, rhs_ref, _ = _reference_damage_system(
            self.coordinates,
            self.volumes,
            self.edges,
            self.delta,
            Gc,
            l0,
            history,
        )
        expected_damage = np.linalg.solve(system_ref, rhs_ref)
        np.testing.assert_allclose(result.damage, expected_damage, rtol=0.0, atol=3.0e-14)
        np.testing.assert_allclose(
            result.strong_residual,
            strong_ref @ expected_damage - 2.0 * history,
            rtol=0.0,
            atol=3.0e-14,
        )
        np.testing.assert_allclose(
            result.euclidean_residual,
            system_ref @ expected_damage - rhs_ref,
            rtol=0.0,
            atol=3.0e-14,
        )
        eigenvalues = np.linalg.eigvalsh(system_ref)
        self.assertGreater(float(np.min(eigenvalues)), 0.0)
        crack, history_energy, graph, total = _reference_damage_energy(
            expected_damage,
            self.coordinates,
            self.volumes,
            self.edges,
            self.delta,
            Gc,
            l0,
            history,
        )
        self.assertAlmostEqual(result.crack_local_energy, crack, places=14)
        self.assertAlmostEqual(result.history_energy, history_energy, places=14)
        self.assertAlmostEqual(result.graph_energy, graph, places=14)
        self.assertAlmostEqual(result.total_energy, total, places=14)
        self.assertLess(result.euclidean_residual_inf, 3.0e-13)
        self.assertLess(result.symmetry_defect, 3.0e-14)
        np.testing.assert_allclose(
            result.graph_energy,
            0.5 * Gc * l0 * expected_damage @ np.diag(self.volumes) @ D_ref @ expected_damage,
            rtol=0.0,
            atol=3.0e-14,
        )

        direction = np.linspace(-0.3, 0.4, len(history), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        epsilon = 2.0e-6
        energy_plus = _reference_damage_energy(
            expected_damage + epsilon * direction,
            self.coordinates,
            self.volumes,
            self.edges,
            self.delta,
            Gc,
            l0,
            history,
        )[3]
        energy_minus = _reference_damage_energy(
            expected_damage - epsilon * direction,
            self.coordinates,
            self.volumes,
            self.edges,
            self.delta,
            Gc,
            l0,
            history,
        )[3]
        fd = (energy_plus - energy_minus) / (2.0 * epsilon)
        self.assertLess(abs(fd - float((system_ref @ expected_damage - rhs_ref) @ direction)), 3.0e-10)

        zero = solve_fixed_history_damage(
            self.operator,
            Gc,
            l0,
            np.zeros(len(history), dtype=np.float64),
        )
        np.testing.assert_array_equal(zero.damage, 0.0)
        self.assertEqual(zero.damage_min, 0.0)
        self.assertEqual(zero.damage_max, 0.0)
        self.assertEqual(zero.below_zero_count, 0)
        self.assertEqual(zero.above_one_count, 0)

    def test_amor_3d_split_source_average_and_pure_max_update(self) -> None:
        lam = self.mechanics.lame_lambda
        mu = self.mechanics.shear_modulus
        strains = np.asarray(
            [
                [[0.02, 0.0], [0.0, 0.01]],  # tension
                [[-0.02, 0.0], [0.0, -0.01]],  # general in-plane compression
                [[0.0, 0.03], [0.03, 0.0]],  # shear
                [[-0.02, 0.01], [0.01, 0.005]],  # mixed compression/shear
            ],
            dtype=np.float64,
        )
        expected = []
        identity = np.eye(3, dtype=np.float64)
        K = lam + 2.0 * mu / 3.0
        for strain in strains:
            embedded = np.zeros((3, 3), dtype=np.float64)
            embedded[:2, :2] = strain
            trace = np.trace(embedded)
            dev = embedded - trace / 3.0 * identity
            expected.append(
                0.5 * K * max(trace, 0.0) ** 2 + mu * np.sum(dev * dev)
            )
        expected_energy = np.asarray(expected, dtype=np.float64)
        np.testing.assert_allclose(
            amor_positive_energy_from_strain(strains, lam, mu),
            expected_energy,
            rtol=0.0,
            atol=2.0e-16,
        )

        source = np.asarray([0, 0, 1, 1], dtype=np.int64)
        eta = np.asarray([0.25, 0.75, 0.4, 0.6], dtype=np.float64)
        previous = np.asarray([0.001, 0.2], dtype=np.float64)
        previous_before = previous.copy()
        result = compute_amor_history_from_edge_strain(
            strains, source, eta, lam, mu, previous
        )
        expected_driver = np.zeros(2, dtype=np.float64)
        np.add.at(expected_driver, source, eta * expected_energy)
        np.testing.assert_allclose(result.particle_driver, expected_driver, rtol=0.0, atol=2.0e-16)
        np.testing.assert_array_equal(result.trial_history, np.maximum(previous, expected_driver))
        np.testing.assert_array_equal(previous, previous_before)
        self.assertFalse(result.trial_history.flags.writeable)

        direct_update = update_history_max(previous, expected_driver)
        np.testing.assert_array_equal(direct_update, result.trial_history)
        np.testing.assert_array_equal(previous, previous_before)

        deformation_gradient = np.asarray([[1.02, 0.01], [-0.004, 0.98]], dtype=np.float64)
        displacement = (
            self.coordinates @ (deformation_gradient - np.eye(2)).T
        ).ravel()
        mechanics_history = compute_amor_history(
            self.mechanics,
            displacement,
            np.zeros(self.mechanics.n_particles, dtype=np.float64),
        )
        self.assertTrue(
            np.allclose(
                mechanics_history.edge_positive_energy,
                mechanics_history.edge_positive_energy[0],
                rtol=0.0,
                atol=2.0e-14,
            )
        )

    def test_history_and_operator_validation_is_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "float64"):
            solve_fixed_history_damage(
                self.operator,
                1.0,
                0.5,
                np.zeros(len(self.coordinates), dtype=np.float32),
            )
        with self.assertRaises(ValueError):
            solve_fixed_history_damage(
                self.operator,
                0.0,
                0.5,
                np.zeros(len(self.coordinates), dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            solve_fixed_history_damage(
                self.operator,
                1.0,
                0.5,
                -np.ones(len(self.coordinates), dtype=np.float64),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            solve_fixed_history_damage(
                self.operator,
                1.0,
                0.5,
                np.zeros(len(self.coordinates) - 1, dtype=np.float64),
            )
        with self.assertRaises(TypeError):
            solve_fixed_history_damage(
                cast(ExplicitVolumeSparseD, object()),
                1.0,
                0.5,
                np.zeros(len(self.coordinates), dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            update_history_max(
                np.zeros(2, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            update_history_max(
                -np.ones(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            update_history_max(
                np.zeros(2, dtype=np.float64),
                -np.ones(2, dtype=np.float64),
            )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            compute_amor_history_from_edge_strain(
                np.zeros((2, 2, 2), dtype=np.float64),
                np.asarray([0, 0], dtype=np.int64),
                np.ones(2, dtype=np.float64),
                1.0,
                1.0,
                np.zeros(1, dtype=np.float64),
            )
        nonsymmetric_strain = np.zeros((1, 2, 2), dtype=np.float64)
        nonsymmetric_strain[0, 0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "symmetric"):
            amor_positive_energy_from_strain(nonsymmetric_strain, 1.0, 1.0)

        quartic = ExplicitVolumeSparseD(
            self.coordinates,
            self.volumes,
            self.edges,
            self.delta,
            "quartic",
        )
        with self.assertRaisesRegex(ValueError, "cubic"):
            solve_fixed_history_damage(
                quartic,
                1.0,
                0.5,
                np.zeros(len(self.coordinates), dtype=np.float64),
            )
        one_dimensional_coordinates = np.asarray(
            [[0.0], [0.1], [0.2]], dtype=np.float64
        )
        one_dimensional_edges = _edges(one_dimensional_coordinates, 0.25)
        one_dimensional = ExplicitVolumeSparseD(
            one_dimensional_coordinates,
            np.ones(3, dtype=np.float64),
            one_dimensional_edges,
            0.25,
            "cubic",
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            solve_fixed_history_damage(
                one_dimensional,
                1.0,
                0.5,
                np.zeros(3, dtype=np.float64),
            )

        inconsistent_operator = self.operator
        inconsistent_matrix = self.operator.to_scipy().tolil()
        inconsistent_matrix[0, 1] *= 0.75
        with mock.patch.object(
            inconsistent_operator,
            "to_scipy",
            return_value=csr_matrix(inconsistent_matrix),
        ):
            with self.assertRaisesRegex(ValueError, "coefficients"):
                solve_fixed_history_damage(
                    inconsistent_operator,
                    1.0,
                    0.5,
                    np.ones(len(self.coordinates), dtype=np.float64),
                )


class Fracture2DSpatialDamageMechanicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coordinates, cls.volumes, cls.edges, cls.delta = _cloud()
        cls.E = 2.0
        cls.nu = 0.3
        cls.k_res = 1.0e-8
        cls.rank_rtol = 1.0e-12
        cls.reference = _reference_ba(
            cls.coordinates, cls.volumes, cls.edges, cls.delta
        )
        cls.intact = BondAssociatedMechanics(
            cls.coordinates,
            cls.volumes,
            cls.edges,
            cls.delta,
            cls.E,
            cls.nu,
            cls.k_res,
            cls.rank_rtol,
        )
        cls.damage = np.asarray(
            np.linspace(0.0, 0.85, len(cls.coordinates)), dtype=np.float64
        )
        cls.fixed = FixedDamageBondAssociatedMechanics(
            cls.intact,
            cls.damage,
        )

    def test_spatial_damage_reference_effective_fields_tangent_and_psd(self) -> None:
        rng = np.random.default_rng(1704)
        displacement = np.asarray(0.01 * rng.normal(size=self.fixed.n_dofs), dtype=np.float64)
        evaluation = self.fixed.evaluate(displacement)
        expected = _reference_mechanics(
            displacement,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )
        expected_energy, expected_residual, expected_f, expected_strain, expected_stress = expected
        self.assertAlmostEqual(evaluation.energy, expected_energy, places=13)
        np.testing.assert_allclose(evaluation.residual, expected_residual, rtol=0.0, atol=4.0e-14)
        np.testing.assert_allclose(evaluation.edge_deformation_gradient, expected_f, rtol=0.0, atol=4.0e-14)
        np.testing.assert_allclose(evaluation.edge_strain, expected_strain, rtol=0.0, atol=4.0e-14)
        np.testing.assert_allclose(evaluation.edge_stress, expected_stress, rtol=0.0, atol=4.0e-14)
        g_edge = ((1.0 - self.damage[self.fixed.source]) ** 2 + self.k_res)[:, None, None]
        g_particle = ((1.0 - self.damage) ** 2 + self.k_res)[:, None, None]
        np.testing.assert_allclose(
            evaluation.edge_effective_stress,
            g_edge * expected_stress,
            rtol=0.0,
            atol=4.0e-14,
        )
        np.testing.assert_allclose(
            evaluation.particle_effective_stress,
            g_particle * evaluation.particle_stress,
            rtol=0.0,
            atol=4.0e-14,
        )
        np.testing.assert_allclose(
            self.fixed.center_factor,
            self.volumes[self.fixed.source] * g_edge[:, 0, 0] * self.fixed.eta,
            rtol=0.0,
            atol=0.0,
        )

        direction = np.asarray(rng.normal(size=self.fixed.n_dofs), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        epsilon = 2.0e-6
        reference_residual_plus = _reference_mechanics(
            displacement + epsilon * direction,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )[1]
        reference_residual_minus = _reference_mechanics(
            displacement - epsilon * direction,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )[1]
        residual_fd = (reference_residual_plus - reference_residual_minus) / (
            2.0 * epsilon
        )
        tangent_action = self.fixed.tangent_action(direction)
        self.assertLess(
            np.linalg.norm(residual_fd - tangent_action)
            / max(np.linalg.norm(tangent_action), 1.0e-30),
            4.0e-10,
        )
        reference_energy_plus = _reference_mechanics(
            displacement + epsilon * direction,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )[0]
        reference_energy_minus = _reference_mechanics(
            displacement - epsilon * direction,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )[0]
        energy_fd = (reference_energy_plus - reference_energy_minus) / (2.0 * epsilon)
        self.assertLess(
            abs(energy_fd - float(evaluation.residual @ direction))
            / max(abs(energy_fd), abs(float(evaluation.residual @ direction)), 1.0e-30),
            4.0e-10,
        )
        tangent = np.column_stack(
            [
                self.fixed.tangent_action(
                    np.eye(self.fixed.n_dofs, dtype=np.float64)[:, column]
                )
                for column in range(self.fixed.n_dofs)
            ]
        )
        np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=3.0e-13)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(tangent))), -3.0e-12)
        expected_blocks = np.asarray(
            [
                tangent[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                for i in range(self.fixed.n_particles)
            ]
        )
        np.testing.assert_allclose(
            self.fixed.tangent_block_diagonal,
            expected_blocks,
            rtol=0.0,
            atol=4.0e-13,
        )

    def test_fixed_zero_damage_exact_compatibility_and_separate_cache(self) -> None:
        zero_damage = np.zeros(self.intact.n_particles, dtype=np.float64)
        fixed_zero = FixedDamageBondAssociatedMechanics(
            self.intact,
            zero_damage,
        )
        rng = np.random.default_rng(77)
        displacement = np.asarray(rng.normal(size=self.intact.n_dofs), dtype=np.float64)
        direction = np.asarray(rng.normal(size=self.intact.n_dofs), dtype=np.float64)
        intact_evaluation = self.intact.evaluate(displacement)
        fixed_evaluation = fixed_zero.evaluate(displacement)
        self.assertEqual(self.intact.energy(displacement), fixed_zero.energy(displacement))
        np.testing.assert_array_equal(self.intact.residual(displacement), fixed_zero.residual(displacement))
        np.testing.assert_array_equal(self.intact.tangent_action(direction), fixed_zero.tangent_action(direction))
        for name in (
            "edge_deformation_gradient",
            "edge_strain",
            "edge_stress",
            "particle_strain",
            "particle_stress",
            "edge_effective_stress",
            "particle_effective_stress",
        ):
            np.testing.assert_array_equal(
                getattr(intact_evaluation, name), getattr(fixed_evaluation, name)
            )
        np.testing.assert_array_equal(self.intact.center_factor, fixed_zero.center_factor)
        intact_blocks = self.intact.tangent_block_diagonal.copy()
        fixed_blocks = fixed_zero.tangent_block_diagonal
        np.testing.assert_array_equal(intact_blocks, self.intact.tangent_block_diagonal)
        np.testing.assert_array_equal(intact_blocks, fixed_blocks)

        bottom = self.coordinates[:, 1] == 0.0
        top = self.coordinates[:, 1] == 4.0
        intact_constraints = self.intact.build_constraints(bottom, top, 0)
        fixed_constraints = fixed_zero.build_constraints(bottom, top, 0)
        intact_result = self.intact.solve(
            intact_constraints, 0.04, 1.0e-11, 1.0e-13, 3000
        )
        fixed_result = fixed_zero.solve(
            fixed_constraints, 0.04, 1.0e-11, 1.0e-13, 3000
        )
        np.testing.assert_array_equal(intact_result.displacement, fixed_result.displacement)
        np.testing.assert_array_equal(intact_result.residual, fixed_result.residual)
        self.assertEqual(intact_result.top_reaction, fixed_result.top_reaction)
        self.assertEqual(intact_result.energy, fixed_result.energy)

        with self.assertRaises(AttributeError):
            setattr(fixed_zero, "damage", np.ones_like(zero_damage))
        for name, value in (
            ("_damage", np.ones_like(zero_damage)),
            ("center_factor", np.ones_like(fixed_zero.center_factor)),
            ("degradation", np.ones_like(fixed_zero.degradation)),
        ):
            with self.subTest(immutable_field=name):
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    setattr(fixed_zero, name, value)
                with self.assertRaisesRegex(AttributeError, "immutable"):
                    delattr(fixed_zero, name)

    def test_fixed_damage_range_conditioning_and_constraints(self) -> None:
        out_of_range = self.damage.copy()
        out_of_range[0] = -0.2
        out_of_range[1] = 1.2
        ranged = FixedDamageBondAssociatedMechanics(
            self.intact,
            out_of_range,
        )
        self.assertEqual(ranged.damage_min, -0.2)
        self.assertEqual(ranged.damage_max, 1.2)
        self.assertEqual(ranged.damage_below_zero_count, 1)
        self.assertEqual(ranged.damage_above_one_count, 1)
        np.testing.assert_array_equal(ranged.damage, out_of_range)

        low_residual_intact = BondAssociatedMechanics(
            self.coordinates,
            self.volumes,
            self.edges,
            self.delta,
            self.E,
            self.nu,
            1.0e-10,
            self.rank_rtol,
        )
        near_fully_damaged = FixedDamageBondAssociatedMechanics(
            low_residual_intact,
            np.ones(self.intact.n_particles, dtype=np.float64),
        )
        self.assertTrue(np.all(near_fully_damaged.center_factor > 0.0))
        self.assertTrue(
            np.all(
                [
                    np.all(np.linalg.eigvalsh(block) > 0.0)
                    for block in near_fully_damaged.tangent_block_diagonal
                ]
            )
        )

        bottom = self.coordinates[:, 1] == 0.0
        top = self.coordinates[:, 1] == 4.0
        constraints = self.fixed.build_constraints(bottom, top, 0)
        result = solve_fixed_damage_mechanics(
            self.fixed,
            constraints,
            0.04,
            cg_rtol=1.0e-11,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
        )
        self.assertEqual(result.cg_info, 0)
        self.assertLess(result.free_residual_norm, 3.0e-11)
        np.testing.assert_allclose(result.total_force, 0.0, rtol=0.0, atol=3.0e-11)
        self.assertLessEqual(abs(result.total_moment), 3.0e-11)
        self.assertAlmostEqual(
            result.top_reaction,
            np.sum(result.residual[2 * np.flatnonzero(top) + 1]),
            places=13,
        )
        self.assertAlmostEqual(
            result.bottom_reaction,
            np.sum(result.residual[2 * np.flatnonzero(bottom) + 1]),
            places=13,
        )

        dense_tangent = np.column_stack(
            [
                np.asarray(
                    _reference_mechanics(
                        np.eye(self.fixed.n_dofs, dtype=np.float64)[:, column],
                        self.coordinates,
                        self.volumes,
                        self.reference,
                        self.E,
                        self.nu,
                        self.k_res,
                        self.damage,
                    )[1]
                )
                for column in range(self.fixed.n_dofs)
            ]
        )
        free = constraints.free_dofs

        def independent_displacement(load: float) -> np.ndarray:
            prescribed = constraints.prescribed_vector(load)
            free_values = np.linalg.solve(
                dense_tangent[np.ix_(free, free)],
                -dense_tangent[np.ix_(free, np.arange(self.fixed.n_dofs))] @ prescribed,
            )
            displacement = prescribed.copy()
            displacement[free] = free_values
            return displacement

        epsilon = 1.0e-5
        displacement_plus = independent_displacement(result.u_bar + epsilon)
        displacement_minus = independent_displacement(result.u_bar - epsilon)
        energy_plus = _reference_mechanics(
            displacement_plus,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )[0]
        energy_minus = _reference_mechanics(
            displacement_minus,
            self.coordinates,
            self.volumes,
            self.reference,
            self.E,
            self.nu,
            self.k_res,
            self.damage,
        )[0]
        reaction_fd = (energy_plus - energy_minus) / (2.0 * epsilon)
        self.assertLess(
            abs(reaction_fd - result.top_reaction)
            / max(abs(reaction_fd), abs(result.top_reaction), 1.0e-30),
            4.0e-9,
        )

        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.fixed.solve(
                constraints,
                0.04,
                cg_rtol=1.0e-14,
                cg_atol=0.0,
                cg_maxiter=1,
            )

    def test_fixed_damage_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "float64"):
            FixedDamageBondAssociatedMechanics(
                self.intact,
                self.damage.astype(np.float32),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            FixedDamageBondAssociatedMechanics(
                self.intact,
                self.damage[:-1],
            )
        invalid = self.damage.copy()
        invalid[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            FixedDamageBondAssociatedMechanics(
                self.intact,
                invalid,
            )
        with self.assertRaisesRegex(TypeError, "BondAssociatedMechanics"):
            FixedDamageBondAssociatedMechanics(
                object(),
                self.damage,
            )


if __name__ == "__main__":
    unittest.main()
