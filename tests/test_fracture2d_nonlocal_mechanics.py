"""Persistent Stage 3 checks for the intact bond-associated mechanics path.

The reference helpers in this file reconstruct supports, deformation
gradients, stresses, energy, and endpoint residuals independently.  They do
not call the production energy/residual/tangent methods for the assertions
that are intended to catch shared implementation errors.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from examples.fracture2d.nonlocal_mechanics import (
    BondAssociatedMechanics,
    HardBoundaryConstraints,
)


def _strict_edges(coordinates: np.ndarray, delta: float) -> np.ndarray:
    """Build a reciprocal strict graph independently of the production class."""

    pairs: list[tuple[int, int]] = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[j] - coordinates[i]))
            if 0.0 < distance < delta:
                pairs.extend(((i, j), (j, i)))
    # Deliberately return a non-canonical order; construction must normalize it.
    edges = np.asarray(pairs[::-1], dtype=np.int64)
    return edges


def _reference_ba(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    directed_edges: np.ndarray,
    delta: float,
) -> dict[str, object]:
    """Reconstruct target-inclusive BA data without using production methods."""

    edges = np.asarray(
        sorted((int(i), int(j)) for i, j in np.asarray(directed_edges).tolist()),
        dtype=np.int64,
    )
    distance = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    omega = 120.0 / (np.pi * delta**4) * (1.0 - distance / delta) ** 2
    adjacency: list[set[int]] = [set() for _ in range(len(coordinates))]
    weight: dict[tuple[int, int], float] = {}
    for (i, j), value in zip(edges, omega):
        adjacency[int(i)].add(int(j))
        weight[int(i), int(j)] = float(value)
    mass = np.zeros(len(coordinates), dtype=np.float64)
    for (i, j), value in zip(edges, omega):
        mass[int(i)] += float(value) * volumes[int(j)]

    supports: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    xis: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    for i_raw, j_raw in edges:
        i = int(i_raw)
        j = int(j_raw)
        common = sorted(adjacency[i].intersection(adjacency[j]) - {i, j})
        support = np.asarray([j, *common], dtype=np.int64)
        beta = np.asarray(
            [weight[i, j] * volumes[j]]
            + [weight[i, k] * volumes[k] for k in common],
            dtype=np.float64,
        )
        xi = coordinates[support] - coordinates[i]
        shape = np.einsum("s,sa,sb->ab", beta, xi, xi)
        supports.append(support)
        betas.append(beta)
        xis.append(xi)
        inverses.append(np.linalg.inv(shape))
    return {
        "edges": edges,
        "distance": distance,
        "omega": omega,
        "mass": mass,
        "supports": supports,
        "betas": betas,
        "xis": xis,
        "inverses": inverses,
    }


def _reference_response(
    displacement: np.ndarray,
    coordinates: np.ndarray,
    volumes: np.ndarray,
    reference: dict[str, object],
    youngs_modulus: float,
    poisson_ratio: float,
    k_res: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Independent energy/residual/edge-field evaluator for test assertions."""

    edges = reference["edges"]
    supports = reference["supports"]
    betas = reference["betas"]
    xis = reference["xis"]
    inverses = reference["inverses"]
    mass = reference["mass"]
    assert isinstance(edges, np.ndarray)
    assert isinstance(supports, list)
    assert isinstance(betas, list)
    assert isinstance(xis, list)
    assert isinstance(inverses, list)
    assert isinstance(mass, np.ndarray)

    lam = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    g = 1.0 + k_res
    values = displacement.reshape(len(coordinates), 2)
    energy = 0.0
    residual = np.zeros_like(displacement)
    edge_f: list[np.ndarray] = []
    edge_strain: list[np.ndarray] = []
    edge_stress: list[np.ndarray] = []
    for edge_index, (i_raw, _) in enumerate(edges):
        i = int(i_raw)
        support = supports[edge_index]
        beta = betas[edge_index]
        xi = xis[edge_index]
        inverse = inverses[edge_index]
        current = np.zeros((2, 2), dtype=np.float64)
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            y = xi[local] + values[k] - values[i]
            current += beta[local] * np.outer(y, xi[local])
        deformation_gradient = current @ inverse
        gradient_increment = deformation_gradient - np.eye(2)
        strain = 0.5 * (gradient_increment + gradient_increment.T)
        trace = float(np.trace(strain))
        stress = lam * trace * np.eye(2) + 2.0 * mu * strain
        eta = beta[0] / mass[i]
        factor = volumes[i] * g * eta
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
    )


def _regular_case(
    n_side: int = 7,
    delta: float = 2.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    coordinates = np.asarray(
        [(float(ix), float(iy)) for ix in range(n_side) for iy in range(n_side)],
        dtype=np.float64,
    )
    index = np.arange(len(coordinates), dtype=np.float64)
    volumes = 0.8 + 0.025 * (index % 7.0) + 0.01 * ((index // 7.0) % 5.0)
    volumes = np.asarray(volumes, dtype=np.float64)
    edges = _strict_edges(coordinates, delta)
    return coordinates, volumes, edges, delta


class Fracture2DNonlocalMechanicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coordinates, cls.volumes, cls.edges, cls.delta = _regular_case()
        cls.youngs_modulus = 2.0
        cls.poisson_ratio = 0.3
        cls.k_res = 1.0e-8
        cls.rank_rtol = 1.0e-12
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
        cls.reference = _reference_ba(
            cls.coordinates, cls.volumes, cls.edges, cls.delta
        )

    def test_target_inclusive_support_eta_and_affine_patch(self) -> None:
        mechanics = self.mechanics
        reference = self.reference
        supports = reference["supports"]
        betas = reference["betas"]
        mass = reference["mass"]
        assert isinstance(supports, list)
        assert isinstance(betas, list)
        assert isinstance(mass, np.ndarray)
        np.testing.assert_array_equal(
            mechanics.directed_edges, reference["edges"]
        )
        for edge_index, (i_raw, j_raw) in enumerate(mechanics.directed_edges):
            i = int(i_raw)
            j = int(j_raw)
            size = int(mechanics.support_sizes[edge_index])
            np.testing.assert_array_equal(
                mechanics.support_indices[edge_index, :size], supports[edge_index]
            )
            self.assertEqual(int(mechanics.support_indices[edge_index, 0]), j)
            self.assertTrue(np.all(mechanics.support_mask[edge_index, :size]))
            self.assertFalse(np.any(mechanics.support_mask[edge_index, size:]))
            np.testing.assert_allclose(
                mechanics.beta[edge_index, :size], betas[edge_index], rtol=0.0, atol=0.0
            )
            expected_eta = betas[edge_index][0] / mass[i]
            self.assertEqual(float(mechanics.eta[edge_index]), expected_eta)
            self.assertGreater(float(mechanics.target_beta[edge_index]), 0.0)
            self.assertGreater(float(mechanics.target_sensitivity[edge_index]), 0.0)

        prescribed_f = np.asarray(
            [[1.12, 0.035], [-0.018, 0.93]], dtype=np.float64
        )
        translation = np.asarray([0.13, -0.22], dtype=np.float64)
        displacement = (
            self.coordinates @ (prescribed_f - np.eye(2)).T + translation
        ).ravel()
        evaluation = mechanics.evaluate(displacement)
        np.testing.assert_allclose(
            evaluation.edge_deformation_gradient,
            np.broadcast_to(prescribed_f, evaluation.edge_deformation_gradient.shape),
            rtol=0.0,
            atol=4.0e-14,
        )
        expected_strain = 0.5 * (
            prescribed_f - np.eye(2) + (prescribed_f - np.eye(2)).T
        )
        np.testing.assert_allclose(
            evaluation.edge_strain,
            np.broadcast_to(expected_strain, evaluation.edge_strain.shape),
            rtol=0.0,
            atol=4.0e-14,
        )
        np.testing.assert_allclose(
            evaluation.particle_strain,
            np.broadcast_to(expected_strain, evaluation.particle_strain.shape),
            rtol=0.0,
            atol=5.0e-14,
        )
        expected_stress = (
            self.mechanics.lame_lambda * np.trace(expected_strain) * np.eye(2)
            + 2.0 * self.mechanics.shear_modulus * expected_strain
        )
        np.testing.assert_allclose(
            evaluation.edge_stress,
            np.broadcast_to(expected_stress, evaluation.edge_stress.shape),
            rtol=0.0,
            atol=5.0e-14,
        )
        np.testing.assert_allclose(
            evaluation.particle_stress,
            np.broadcast_to(expected_stress, evaluation.particle_stress.shape),
            rtol=0.0,
            atol=5.0e-14,
        )
        independent_energy, independent_residual, independent_f, independent_strain = (
            _reference_response(
                displacement,
                self.coordinates,
                self.volumes,
                self.reference,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
            )
        )
        self.assertAlmostEqual(evaluation.energy, independent_energy, places=13)
        np.testing.assert_allclose(evaluation.residual, independent_residual, atol=2.0e-14)
        np.testing.assert_allclose(evaluation.edge_F, independent_f, atol=4.0e-14)
        np.testing.assert_allclose(evaluation.edge_strain, independent_strain, atol=4.0e-14)

    def test_rigid_translations_and_infinitesimal_rotation_are_null(self) -> None:
        translation_x = np.zeros(self.mechanics.n_dofs, dtype=np.float64)
        translation_x[0::2] = 0.37
        translation_y = np.zeros(self.mechanics.n_dofs, dtype=np.float64)
        translation_y[1::2] = -0.29
        skew = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        rotation = (1.0e-4 * self.coordinates @ skew.T).ravel()
        for mode in (translation_x, translation_y, rotation):
            self.assertLessEqual(self.mechanics.energy(mode), 2.0e-28)
            np.testing.assert_allclose(
                self.mechanics.residual(mode), 0.0, rtol=0.0, atol=2.0e-13
            )

    def _dense_tangent(self) -> np.ndarray:
        n_dofs = self.mechanics.n_dofs
        tangent = np.empty((n_dofs, n_dofs), dtype=np.float64)
        for column in range(n_dofs):
            direction = np.zeros(n_dofs, dtype=np.float64)
            direction[column] = 1.0
            tangent[:, column] = self.mechanics.tangent_action(direction)
        return tangent

    def _dense_reference_tangent(self) -> np.ndarray:
        """Assemble the linear tangent from the independent test residual."""

        n_dofs = self.mechanics.n_dofs
        tangent = np.empty((n_dofs, n_dofs), dtype=np.float64)
        for column in range(n_dofs):
            displacement = np.zeros(n_dofs, dtype=np.float64)
            displacement[column] = 1.0
            _, residual, _, _ = _reference_response(
                displacement,
                self.coordinates,
                self.volumes,
                self.reference,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
            )
            tangent[:, column] = residual
        return tangent

    def test_unconstrained_tangent_has_exactly_three_rigid_modes(self) -> None:
        tangent = self._dense_tangent()
        np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=2.0e-13)
        eigenvalues = np.linalg.eigvalsh(tangent)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        zero_tolerance = 2.0e-11 * scale
        self.assertEqual(int(np.count_nonzero(np.abs(eigenvalues) <= zero_tolerance)), 3)
        self.assertGreater(float(eigenvalues[3]), 20.0 * zero_tolerance)
        self.assertGreaterEqual(float(eigenvalues[0]), -5.0e-12 * scale)

    def test_energy_residual_tangent_and_bilinear_checks(self) -> None:
        rng = np.random.default_rng(401)
        displacement = 1.0e-2 * rng.normal(size=self.mechanics.n_dofs)
        direction = rng.normal(size=self.mechanics.n_dofs)
        direction /= np.linalg.norm(direction)
        epsilon = 2.0e-5
        energy_fd = (
            self.mechanics.energy(displacement + epsilon * direction)
            - self.mechanics.energy(displacement - epsilon * direction)
        ) / (2.0 * epsilon)
        residual = self.mechanics.residual(displacement)
        directional = float(residual @ direction)
        relative_energy_error = abs(energy_fd - directional) / max(
            abs(energy_fd), abs(directional), 1.0e-30
        )
        self.assertLess(relative_energy_error, 3.0e-10)

        residual_plus = self.mechanics.residual(displacement + epsilon * direction)
        residual_minus = self.mechanics.residual(displacement - epsilon * direction)
        residual_jvp = (residual_plus - residual_minus) / (2.0 * epsilon)
        tangent_jvp = self.mechanics.tangent_action(direction)
        self.assertLess(
            np.linalg.norm(residual_jvp - tangent_jvp)
            / max(np.linalg.norm(tangent_jvp), 1.0e-30),
            3.0e-10,
        )
        np.testing.assert_allclose(
            residual, self.mechanics.tangent_action(displacement), rtol=0.0, atol=3.0e-13
        )

        tangent = self._dense_tangent()
        for _ in range(5):
            p = rng.normal(size=self.mechanics.n_dofs)
            q = rng.normal(size=self.mechanics.n_dofs)
            left = float(p @ tangent @ q)
            right = float(q @ tangent @ p)
            self.assertLessEqual(abs(left - right), 3.0e-12 * max(abs(left), abs(right), 1.0))
            self.assertGreaterEqual(float(q @ tangent @ q), -3.0e-11)

    def test_block_jacobi_matches_dense_tangent_diagonal(self) -> None:
        tangent = self._dense_tangent()
        expected = np.asarray(
            [
                tangent[2 * particle : 2 * particle + 2,
                        2 * particle : 2 * particle + 2]
                for particle in range(self.mechanics.n_particles)
            ]
        )
        np.testing.assert_allclose(
            self.mechanics.tangent_block_diagonal,
            expected,
            rtol=0.0,
            atol=3.0e-13,
        )

    def test_linear_and_angular_momentum(self) -> None:
        rng = np.random.default_rng(402)
        displacement = rng.normal(size=self.mechanics.n_dofs)
        residual = self.mechanics.residual(displacement).reshape(-1, 2)
        np.testing.assert_allclose(np.sum(residual, axis=0), 0.0, rtol=0.0, atol=2.0e-12)
        moment = np.sum(
            self.coordinates[:, 0] * residual[:, 1]
            - self.coordinates[:, 1] * residual[:, 0]
        )
        self.assertLessEqual(abs(float(moment)), 3.0e-12)

    def test_hard_constraints_cg_reactions_and_independent_energy_work(self) -> None:
        bottom = self.coordinates[:, 1] == 0.0
        top = self.coordinates[:, 1] == 6.0
        constraints = self.mechanics.build_constraints(bottom, top, 0)
        self.assertIsInstance(constraints, HardBoundaryConstraints)
        expected_free_horizontal = np.concatenate(
            (
                2 * np.flatnonzero(bottom),
                2 * np.flatnonzero(top),
            )
        )
        expected_free_horizontal = expected_free_horizontal[
            expected_free_horizontal != 0
        ]
        self.assertTrue(
            np.all(np.isin(expected_free_horizontal, constraints.free_dofs))
        )
        self.assertEqual(
            len(expected_free_horizontal),
            int(np.count_nonzero(np.isin(expected_free_horizontal, constraints.free_dofs))),
        )

        u_bar = 0.08
        result = self.mechanics.solve(
            constraints,
            u_bar,
            cg_rtol=1.0e-11,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
        )
        np.testing.assert_allclose(
            result.displacement[2 * np.flatnonzero(bottom) + 1], 0.0, atol=2.0e-14
        )
        np.testing.assert_allclose(
            result.displacement[2 * np.flatnonzero(top) + 1], u_bar, atol=2.0e-14
        )
        self.assertEqual(float(result.displacement[0]), 0.0)
        self.assertEqual(result.cg_info, 0)
        self.assertGreater(result.cg_iterations, 0)
        self.assertLess(result.free_residual_norm, 3.0e-11)
        self.assertLess(result.free_residual_relative_norm, 3.0e-10)
        np.testing.assert_allclose(
            result.top_reaction,
            np.sum(result.residual[2 * np.flatnonzero(top) + 1]),
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            result.bottom_reaction,
            np.sum(result.residual[2 * np.flatnonzero(bottom) + 1]),
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertAlmostEqual(result.anchor_reaction, result.residual[0], places=14)
        np.testing.assert_allclose(result.total_force, 0.0, atol=3.0e-12)
        self.assertLessEqual(abs(result.total_moment), 3.0e-12)

        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.mechanics.solve(
                constraints,
                u_bar,
                cg_rtol=1.0e-14,
                cg_atol=0.0,
                cg_maxiter=1,
            )

        # Re-solve the reduced quadratic problem independently from the
        # production CG solve, using a dense tangent assembled only in this
        # test.  This checks the residual-gradient reaction sign and the
        # reduced-energy derivative without reusing result.energy at perturbed
        # loads or a production matrix assembly.
        tangent = self._dense_reference_tangent()
        free = constraints.free_dofs
        reduced = tangent[np.ix_(free, free)]

        def independent_displacement(load: float) -> np.ndarray:
            prescribed = constraints.prescribed_vector(load)
            free_values = np.linalg.solve(reduced, -tangent[np.ix_(free, np.arange(self.mechanics.n_dofs))] @ prescribed)
            displacement = prescribed.copy()
            displacement[free] = free_values
            return displacement

        epsilon = 1.0e-5
        displacement_plus = independent_displacement(u_bar + epsilon)
        displacement_minus = independent_displacement(u_bar - epsilon)
        reduced_energy_fd = (
            self.mechanics.energy(displacement_plus)
            - self.mechanics.energy(displacement_minus)
        ) / (2.0 * epsilon)
        reaction_relative_error = abs(reduced_energy_fd - result.top_reaction) / max(
            abs(reduced_energy_fd), abs(result.top_reaction), 1.0e-30
        )
        self.assertLess(reaction_relative_error, 3.0e-9)

    def test_exact_initial_guess_exits_before_preconditioner(self) -> None:
        bottom = self.coordinates[:, 1] == 0.0
        top = self.coordinates[:, 1] == 6.0
        constraints = self.mechanics.build_constraints(bottom, top, 0)
        cold = self.mechanics.solve(
            constraints,
            0.08,
            cg_rtol=1.0e-11,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
        )
        with mock.patch.object(
            self.mechanics,
            "_get_tangent_block_diagonal",
            side_effect=AssertionError("warm early exit built the preconditioner"),
        ):
            warm = self.mechanics.solve(
                constraints,
                0.08,
                cg_rtol=1.0e-11,
                cg_atol=1.0e-13,
                cg_maxiter=3000,
                initial_guess=cold.displacement,
                allow_initial_early_exit=True,
            )
        self.assertEqual(warm.cg_info, 0)
        self.assertEqual(warm.cg_iterations, 0)
        self.assertTrue(warm.initial_guess_supplied)
        self.assertFalse(warm.cg_initial_guess_used)
        self.assertTrue(warm.early_exit)
        np.testing.assert_array_equal(warm.displacement, cold.displacement)
        np.testing.assert_array_equal(warm.residual, cold.residual)
        self.assertEqual(warm.top_reaction, cold.top_reaction)
        self.assertEqual(warm.free_rhs_norm, cold.free_rhs_norm)

        zero = constraints.prescribed_vector(0.0)
        zero_result = self.mechanics.solve(
            constraints,
            0.0,
            cg_rtol=1.0e-11,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
            initial_guess=zero,
            allow_initial_early_exit=True,
        )
        self.assertEqual(zero_result.cg_iterations, 0)
        self.assertLessEqual(zero_result.free_rhs_norm, 1.0e-13)
        self.assertEqual(
            zero_result.free_residual_relative_norm,
            zero_result.free_residual_norm
            / max(zero_result.free_rhs_norm, 1.0e-300),
        )

    def test_initial_guess_validation_is_loud(self) -> None:
        bottom = self.coordinates[:, 1] == 0.0
        top = self.coordinates[:, 1] == 6.0
        constraints = self.mechanics.build_constraints(bottom, top, 0)
        prescribed = constraints.prescribed_vector(0.08)
        with self.assertRaisesRegex(ValueError, "float64"):
            self.mechanics.solve(
                constraints,
                0.08,
                1.0e-11,
                1.0e-13,
                3000,
                initial_guess=prescribed.astype(np.float32),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            self.mechanics.solve(
                constraints,
                0.08,
                1.0e-11,
                1.0e-13,
                3000,
                initial_guess=prescribed[:-1],
            )
        bad_boundary = prescribed.copy()
        top_dof = int(2 * np.flatnonzero(top)[0] + 1)
        bad_boundary[top_dof] = np.nextafter(bad_boundary[top_dof], np.inf)
        with self.assertRaisesRegex(ValueError, "constrained values"):
            self.mechanics.solve(
                constraints,
                0.08,
                1.0e-11,
                1.0e-13,
                3000,
                initial_guess=bad_boundary,
            )

    def test_invalid_inputs_and_rank_failure_are_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "float64"):
            BondAssociatedMechanics(
                self.coordinates.astype(np.float32),
                self.volumes,
                self.edges,
                self.delta,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
                self.rank_rtol,
            )
        with self.assertRaisesRegex(ValueError, "float64"):
            BondAssociatedMechanics(
                self.coordinates,
                self.volumes.astype(np.float32),
                self.edges,
                self.delta,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
                self.rank_rtol,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            BondAssociatedMechanics(
                self.coordinates,
                self.volumes,
                self.edges,
                0.0,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
                self.rank_rtol,
            )
        for name, value in (
            ("youngs_modulus", 0.0),
            ("poisson_ratio", 0.5),
            ("k_res", 0.0),
            ("rank_rtol", 0.0),
            ("rank_rtol", 1.0),
        ):
            parameters = {
                "youngs_modulus": self.youngs_modulus,
                "poisson_ratio": self.poisson_ratio,
                "k_res": self.k_res,
                "rank_rtol": self.rank_rtol,
            }
            parameters[name] = value
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    BondAssociatedMechanics(
                        self.coordinates,
                        self.volumes,
                        self.edges,
                        self.delta,
                        parameters["youngs_modulus"],
                        parameters["poisson_ratio"],
                        parameters["k_res"],
                        parameters["rank_rtol"],
                    )

        with self.assertRaisesRegex(ValueError, "reciprocal"):
            BondAssociatedMechanics(
                self.coordinates,
                self.volumes,
                self.edges[:-1],
                self.delta,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
                self.rank_rtol,
            )
        with self.assertRaisesRegex(ValueError, "strict"):
            BondAssociatedMechanics(
                self.coordinates,
                self.volumes,
                self.edges,
                2.0,
                self.youngs_modulus,
                self.poisson_ratio,
                self.k_res,
                self.rank_rtol,
            )
        collinear_coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64
        )
        collinear_volumes = np.ones(3, dtype=np.float64)
        collinear_edges = np.asarray(
            [
                [0, 1],
                [1, 0],
                [0, 2],
                [2, 0],
                [1, 2],
                [2, 1],
            ],
            dtype=np.int64,
        )
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            BondAssociatedMechanics(
                collinear_coordinates,
                collinear_volumes,
                collinear_edges,
                2.1,
                1.0,
                0.3,
                1.0e-8,
                1.0e-12,
            )

    def test_displacement_shape_dtype_and_boundary_validation(self) -> None:
        zero = np.zeros(self.mechanics.n_dofs, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "float64"):
            self.mechanics.energy(zero.astype(np.float32))
        with self.assertRaisesRegex(ValueError, "flat shape"):
            self.mechanics.residual(zero.reshape(-1, 2))
        bottom = self.coordinates[:, 1] == 0.0
        top = self.coordinates[:, 1] == 6.0
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.mechanics.build_constraints(bottom.astype(np.int64), top, 0)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.mechanics.build_constraints(bottom[:-1], top[:-1], 0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            overlapping = bottom.copy()
            overlapping[6] = True
            self.mechanics.build_constraints(overlapping, top, 6)
        with self.assertRaisesRegex(ValueError, "bottom_mask"):
            self.mechanics.build_constraints(bottom, top, 1)


class Fracture2DQuadraticInteriorDiagnosticTest(unittest.TestCase):
    """Strictly deep full-square diagnostic for the residual sign and density."""

    @classmethod
    def setUpClass(cls) -> None:
        n_side = 41
        spacing = 1.0 / (n_side - 1)
        axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        cls.coordinates = np.stack((xx.ravel(), yy.ravel()), axis=-1)
        cls.volumes = np.full(len(cls.coordinates), spacing**2, dtype=np.float64)
        # ``0.08`` is explicit diagnostic data.  It avoids a grid-distance
        # equality at the strict cutoff while retaining a deep interior with
        # several horizon layers.
        cls.delta = 0.08
        cls.mechanics = BondAssociatedMechanics(
            cls.coordinates,
            cls.volumes,
            _strict_edges(cls.coordinates, cls.delta),
            cls.delta,
            1.0,
            0.3,
            1.0e-8,
            1.0e-12,
        )

    def test_quadratic_residual_matches_deep_minus_div_sigma(self) -> None:
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        displacement = np.stack((x**2, np.zeros_like(y)), axis=-1).ravel()
        residual_density = (
            self.mechanics.residual(displacement).reshape(-1, 2) / self.volumes[:, None]
        )
        lam = self.mechanics.lame_lambda
        mu = self.mechanics.shear_modulus
        expected = np.asarray(
            [-(1.0 + self.mechanics.k_res) * (2.0 * lam + 4.0 * mu), 0.0],
            dtype=np.float64,
        )
        boundary_distance = np.minimum.reduce(
            (
                self.coordinates[:, 0],
                self.coordinates[:, 1],
                1.0 - self.coordinates[:, 0],
                1.0 - self.coordinates[:, 1],
            )
        )
        deep = boundary_distance > 2.0 * self.delta
        self.assertGreater(int(np.count_nonzero(deep)), 0)
        deep_error = residual_density[deep] - expected
        relative = np.linalg.norm(deep_error, axis=1) / max(np.linalg.norm(expected), 1.0e-30)
        # This is a manufactured full-square evidence check, not a production
        # horizon-selection criterion.  The strict deep mask and the
        # non-cutoff horizon give the observed float64 patch accuracy.
        self.assertLess(float(np.max(relative)), 2.0e-10)
        self.assertLess(float(np.mean(relative)), 5.0e-11)
        self.assertLess(float(np.max(np.abs(residual_density[deep, 1]))), 2.0e-10)


if __name__ == "__main__":
    unittest.main()
