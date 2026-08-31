"""Focused Stage 5 tests for strict fracture2d continuation.

The cloud is a small nonuniform 2D manufactured particle set.  The tests use
the approved Stage 3/4 solvers for clean paths and ``unittest.mock`` to inject
only trial failures; no production test-only hooks are used.
"""

from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from unittest import mock

import jax
import numpy as np

from examples.fracture2d import nonlocal_baseline as baseline
from examples.fracture2d.nonlocal_mechanics import BondAssociatedMechanics
from examples.fracture2d.nonlocal_mechanics_jax import (
    JaxBondAssociatedMechanics,
    JaxCPUReplayMechanicsAdapter,
)
from pinn.operators import ExplicitVolumeSparseD
from tests.test_fracture2d_nonlocal_baseline import (
    _reference_ba,
    _reference_damage_energy,
    _reference_damage_system,
    _reference_mechanics,
)


def _edges(coordinates: np.ndarray, delta: float) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            if 0.0 < np.linalg.norm(coordinates[j] - coordinates[i]) < delta:
                pairs.extend(((i, j), (j, i)))
    return np.asarray(pairs[::-1], dtype=np.int64)


def _system(n_side: int = 4, delta: float = 1.51):
    coordinates = np.asarray(
        [(float(ix), float(iy)) for ix in range(n_side) for iy in range(n_side)],
        dtype=np.float64,
    )
    index = np.arange(len(coordinates), dtype=np.float64)
    volumes = np.asarray(
        0.8 + 0.02 * (index % 5.0) + 0.01 * ((index // 5.0) % 3.0),
        dtype=np.float64,
    )
    edges = _edges(coordinates, delta)
    operator = ExplicitVolumeSparseD(
        coordinates, volumes, edges, delta, "cubic"
    )
    mechanics = BondAssociatedMechanics(
        coordinates,
        volumes,
        edges,
        delta,
        2.0,
        0.3,
        1.0e-8,
        1.0e-12,
    )
    constraints = mechanics.build_constraints(
        coordinates[:, 1] == 0.0,
        coordinates[:, 1] == float(n_side - 1),
        0,
    )
    return coordinates, volumes, edges, operator, mechanics, constraints


def _config(
    *,
    load_final: float = 0.02,
    du_initial: float = 0.01,
    du_min: float = 0.0025,
    max_cutbacks: int = 2,
    max_stagger_iterations: int = 20,
) -> baseline.StrictContinuationConfig:
    gate = baseline.GateTolerance(1.0e-9, 1.0e-8, 1.0)
    residual_gate = baseline.GateTolerance(1.0e-10, 1.0e-8, 1.0)
    return baseline.StrictContinuationConfig(
        load_start=0.0,
        load_final=load_final,
        du_initial=du_initial,
        du_min=du_min,
        cutback_factor=0.5,
        max_cutbacks=max_cutbacks,
        max_stagger_iterations=max_stagger_iterations,
        cg_rtol=1.0e-11,
        cg_atol=1.0e-13,
        cg_maxiter=3000,
        displacement=gate,
        damage=gate,
        history=gate,
        reaction=gate,
        damage_residual=residual_gate,
        force=gate,
        moment=gate,
        healing_atol=1.0e-10,
        damage_range_atol=1.0e-10,
    )


def _archive_triangles(n_side: int) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n_side - 1):
        for iy in range(n_side - 1):
            lower_left = ix * n_side + iy
            upper_left = (ix + 1) * n_side + iy
            upper_right = (ix + 1) * n_side + iy + 1
            lower_right = ix * n_side + iy + 1
            triangles.extend(
                (
                    (lower_left, upper_left, upper_right),
                    (lower_left, upper_right, lower_right),
                )
            )
    return np.asarray(triangles, dtype=np.int64)


def _reference_amor(
    displacement: np.ndarray,
    coordinates: np.ndarray,
    reference,
    youngs_modulus: float,
    poisson_ratio: float,
    previous_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, edge_strain, _ = _reference_mechanics(
        displacement,
        coordinates,
        np.ones(len(coordinates), dtype=np.float64),
        reference,
        youngs_modulus,
        poisson_ratio,
        0.0,
        np.zeros(len(coordinates), dtype=np.float64),
    )
    # The helper above uses unit volumes only for the mechanics kinematics;
    # the BA edge strains are volume-independent.  Rebuild the Amor split
    # directly rather than calling any production history API.
    lam = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    bulk = lam + 2.0 * mu / 3.0
    identity = np.eye(3, dtype=np.float64)
    edge_positive = np.empty(len(edge_strain), dtype=np.float64)
    for index, strain in enumerate(edge_strain):
        embedded = np.zeros((3, 3), dtype=np.float64)
        embedded[:2, :2] = strain
        trace = float(np.trace(embedded))
        deviatoric = embedded - trace / 3.0 * identity
        edge_positive[index] = (
            0.5 * bulk * max(trace, 0.0) ** 2
            + mu * float(np.sum(deviatoric * deviatoric))
        )
    edges, _, _, _, _, eta, _, _ = reference
    driver = np.zeros(len(coordinates), dtype=np.float64)
    np.add.at(driver, edges[:, 0], eta * edge_positive)
    return edge_positive, driver, np.maximum(previous_history, driver)


def _reference_constrained_mechanics(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    reference,
    mechanics,
    constraints,
    load: float,
    damage: np.ndarray,
):
    """Independent dense BA tangent and constrained solve."""

    n_dofs = 2 * len(coordinates)
    tangent_columns = []
    for column in range(n_dofs):
        direction = np.zeros(n_dofs, dtype=np.float64)
        direction[column] = 1.0
        tangent_columns.append(
            _reference_mechanics(
                direction,
                coordinates,
                volumes,
                reference,
                mechanics.youngs_modulus,
                mechanics.poisson_ratio,
                mechanics.k_res,
                damage,
            )[1]
        )
    tangent = np.column_stack(tangent_columns)
    prescribed = np.zeros(n_dofs, dtype=np.float64)
    prescribed[2 * np.flatnonzero(constraints.top_mask) + 1] = load
    free = constraints.free_dofs
    displacement = prescribed.copy()
    displacement[free] = np.linalg.solve(
        tangent[np.ix_(free, free)],
        -tangent[np.ix_(free, np.arange(n_dofs))] @ prescribed,
    )
    evaluation = _reference_mechanics(
        displacement,
        coordinates,
        volumes,
        reference,
        mechanics.youngs_modulus,
        mechanics.poisson_ratio,
        mechanics.k_res,
        damage,
    )
    return tangent, displacement, evaluation


def _reference_clean_step(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    reference,
    mechanics,
    constraints,
    operator,
    Gc: float,
    l0: float,
    target_load: float,
    stagger_iterations: int,
):
    if stagger_iterations < 1:
        raise ValueError("reference continuation requires at least one stagger sweep")
    committed_damage = np.zeros(len(coordinates), dtype=np.float64)
    committed_history = np.zeros(len(coordinates), dtype=np.float64)
    d_iter = committed_damage.copy()
    d_next = d_iter.copy()
    H_used = committed_history.copy()
    H_next = committed_history.copy()
    edge_positive = np.zeros(len(reference[0]), dtype=np.float64)
    driver_before = committed_history.copy()
    driver_after = committed_history.copy()
    strong = np.eye(len(coordinates), dtype=np.float64)
    system = np.eye(len(coordinates), dtype=np.float64)
    rhs = committed_history.copy()
    tangent_after = np.zeros((2 * len(coordinates), 2 * len(coordinates)), dtype=np.float64)
    u_after = np.zeros(2 * len(coordinates), dtype=np.float64)
    before = None
    after = None
    for _ in range(stagger_iterations):
        _, u_before, before = _reference_constrained_mechanics(
            coordinates,
            volumes,
            reference,
            mechanics,
            constraints,
            target_load,
            d_iter,
        )
        _, driver_before, H_used = _reference_amor(
            u_before,
            coordinates,
            reference,
            mechanics.youngs_modulus,
            mechanics.poisson_ratio,
            committed_history,
        )
        matrix, strong, system, rhs, _ = _reference_damage_system(
            coordinates,
            volumes,
            reference[0],
            mechanics.delta,
            Gc,
            l0,
            H_used,
        )
        d_next = np.linalg.solve(system, rhs)
        tangent_after, u_after, after = _reference_constrained_mechanics(
            coordinates,
            volumes,
            reference,
            mechanics,
            constraints,
            target_load,
            d_next,
        )
        edge_positive, driver_after, H_next = _reference_amor(
            u_after,
            coordinates,
            reference,
            mechanics.youngs_modulus,
            mechanics.poisson_ratio,
            committed_history,
        )
        d_iter = d_next
    crack, history_energy, graph, total = _reference_damage_energy(
        d_next,
        coordinates,
        volumes,
        reference[0],
        mechanics.delta,
        Gc,
        l0,
        H_used,
    )
    committed_history_energy = float(
        np.sum(volumes * H_next * (1.0 - d_next) ** 2)
    )
    if after is None:
        raise RuntimeError("reference continuation did not produce mechanics")
    return {
        "tangent": tangent_after,
        "displacement": u_after,
        "evaluation": after,
        "damage": d_next,
        "H_used": H_used,
        "H_next": H_next,
        "edge_positive": edge_positive,
        "driver": driver_after,
        "driver_before": driver_before,
        "strong_residual": strong @ d_next - 2.0 * H_used,
        "euclidean_residual": system @ d_next - rhs,
        "damage_energies": (crack, history_energy, graph, total),
        "committed_history_energy": committed_history_energy,
        "committed_total_energy": crack + graph + committed_history_energy,
        "before": before,
    }


class Fracture2DStrictContinuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.initial_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        (
            cls.coordinates,
            cls.volumes,
            cls.edges,
            cls.operator,
            cls.mechanics,
            cls.constraints,
        ) = _system()

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.initial_x64)

    def _run(
        self,
        config: baseline.StrictContinuationConfig | None = None,
        *,
        mechanics_solver=None,
    ):
        return baseline.run_strict_continuation(
            self.operator,
            self.mechanics,
            self.constraints,
            1.7,
            0.65,
            _config() if config is None else config,
            mechanics_solver=mechanics_solver,
        )

    def _cpu_replay_adapter(self) -> JaxCPUReplayMechanicsAdapter:
        backend = JaxBondAssociatedMechanics(
            self.mechanics,
            self.constraints,
            jax.devices("cpu")[0],
        )
        return JaxCPUReplayMechanicsAdapter(backend)

    def _archive_metadata(self) -> baseline.SupplementalArchiveMetadata:
        return baseline.SupplementalArchiveMetadata(
            triangles=_archive_triangles(4),
            first_moments=np.sum(self.volumes[:, None] * self.coordinates, axis=0),
            visibility_tolerance=0.0,
            visibility_rule="manufactured reciprocal graph",
            quadrature_rule="manufactured explicit particle weights",
            study_label="manufactured-cloud-strict-continuation",
            model_provenance="independent fixed-subproblem reference helpers",
        )

    def test_ulp_scale_terminal_remainder_is_snapped_to_endpoint(self) -> None:
        load_final = 3.0e-3
        increment = 1.0e-4
        committed = 0.0
        targets = []
        while committed < load_final:
            committed = baseline._next_load_target(
                committed, increment, load_final
            )
            targets.append(committed)

        self.assertEqual(len(targets), 30)
        self.assertEqual(targets[-1], load_final)
        self.assertGreater(targets[-1] - targets[-2], 0.5 * increment)

    def test_scale_aware_replay_allows_only_float64_reconstruction_noise(self) -> None:
        expected = np.asarray([0.0, 1.0], dtype=np.float64)
        baseline._require_scale_aware_close(
            expected + 256.0 * np.finfo(np.float64).eps,
            expected,
            "reconstruction noise",
        )
        with self.assertRaisesRegex(ValueError, "authoritative mechanics value"):
            baseline._require_scale_aware_close(
                expected + 1024.0 * np.finfo(np.float64).eps,
                expected,
                "forged residual",
            )

    def test_fixed_history_replay_uses_term_scaled_residual_bound(self) -> None:
        high_scale_mechanics = BondAssociatedMechanics(
            self.coordinates,
            self.volumes,
            self.edges,
            self.mechanics.delta,
            210.0e3,
            self.mechanics.poisson_ratio,
            self.mechanics.k_res,
            self.mechanics.rank_rtol,
        )
        high_scale_constraints = high_scale_mechanics.build_constraints(
            self.coordinates[:, 1] == 0.0,
            self.coordinates[:, 1] == float(self.coordinates[:, 1].max()),
            0,
        )
        result = baseline.run_strict_continuation(
            self.operator,
            high_scale_mechanics,
            high_scale_constraints,
            1.7,
            0.65,
            _config(load_final=0.005, du_initial=0.005, du_min=0.005),
        )
        step = result.accepted_steps[0]
        fixed = step.fixed_history_result
        damage_matrix = result.metadata.damage_operator()
        terms = (
            np.asarray(
                (result.metadata.Gc / result.metadata.l0) * fixed.damage,
                dtype=np.float64,
            ),
            np.asarray(
                result.metadata.Gc
                * result.metadata.l0
                * (damage_matrix @ fixed.damage),
                dtype=np.float64,
            ),
            np.asarray(2.0 * fixed.history * fixed.damage, dtype=np.float64),
            np.asarray(-2.0 * fixed.history, dtype=np.float64),
        )
        term_scale = np.sum(np.abs(np.asarray(terms)), axis=0)
        index = int(np.argmax(term_scale))
        cancellation_defect = 1.4210854715202004e-13
        term_bound = baseline._RESIDUAL_RECONSTRUCTION_EPS * term_scale[index]
        self.assertGreater(term_bound, cancellation_defect)
        volume_term_scale = self.volumes[index] * term_scale[index]
        self.assertGreater(
            baseline._RESIDUAL_RECONSTRUCTION_EPS * volume_term_scale,
            self.volumes[index] * cancellation_defect,
        )

        def forge_residuals(defect: float):
            strong = np.array(fixed.strong_residual, copy=True)
            strong[index] += defect
            euclidean = np.array(fixed.euclidean_residual, copy=True)
            euclidean[index] += self.volumes[index] * defect
            forged_fixed = replace(
                fixed,
                strong_residual=strong,
                strong_residual_norm=float(np.linalg.norm(strong)),
                strong_residual_inf=float(np.max(np.abs(strong))),
                euclidean_residual=euclidean,
                euclidean_residual_norm=float(np.linalg.norm(euclidean)),
                euclidean_residual_inf=float(np.max(np.abs(euclidean))),
            )
            forged_diagnostics = replace(
                step.diagnostics,
                rho_used=strong,
                rho_used_volume=euclidean,
            )
            forged_step = replace(
                step,
                fixed_history_result=forged_fixed,
                diagnostics=forged_diagnostics,
            )
            forged_attempt = replace(
                result.attempts[0], diagnostics=forged_diagnostics
            )
            return replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        accepted = forge_residuals(cancellation_defect)
        self.assertEqual(accepted.final_state.u_bar, result.final_state.u_bar)
        with self.assertRaisesRegex(
            ValueError, "term-scaled float64 reconstruction bound"
        ):
            forge_residuals(10.0 * term_bound)

    def test_clean_step_matches_independent_dense_ba_damage_and_amor_reference(self) -> None:
        config = _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        result = self._run(config)
        step = result.accepted_steps[0]
        attempt = next(attempt for attempt in result.attempts if attempt.accepted)
        reference = _reference_ba(
            self.coordinates, self.volumes, self.edges, 1.51
        )
        expected = _reference_clean_step(
            self.coordinates,
            self.volumes,
            reference,
            self.mechanics,
            self.constraints,
            self.operator,
            1.7,
            0.65,
            0.01,
            attempt.stagger_iterations,
        )
        np.testing.assert_allclose(
            step.state.displacement, expected["displacement"], rtol=0.0, atol=3.0e-12
        )
        np.testing.assert_allclose(
            step.state.damage, expected["damage"], rtol=0.0, atol=3.0e-14
        )
        np.testing.assert_allclose(
            step.state.history, expected["H_next"], rtol=0.0, atol=3.0e-14
        )
        expected_energy, expected_residual, expected_F, expected_strain, expected_stress = (
            expected["evaluation"]
        )
        production_evaluation = step.mechanics_result.evaluation
        self.assertAlmostEqual(production_evaluation.energy, expected_energy, places=13)
        np.testing.assert_allclose(
            production_evaluation.residual, expected_residual, rtol=0.0, atol=4.0e-12
        )
        np.testing.assert_allclose(
            production_evaluation.edge_deformation_gradient,
            expected_F,
            rtol=0.0,
            atol=4.0e-12,
        )
        np.testing.assert_allclose(
            production_evaluation.edge_strain,
            expected_strain,
            rtol=0.0,
            atol=4.0e-12,
        )
        np.testing.assert_allclose(
            production_evaluation.edge_stress,
            expected_stress,
            rtol=0.0,
            atol=4.0e-11,
        )
        top_dofs = 2 * np.flatnonzero(self.constraints.top_mask) + 1
        expected_reaction = float(np.sum(expected_residual[top_dofs]))
        self.assertAlmostEqual(step.mechanics_result.top_reaction, expected_reaction, places=11)
        np.testing.assert_allclose(
            step.amor_driver.edge_positive_energy,
            expected["edge_positive"],
            rtol=0.0,
            atol=4.0e-13,
        )
        np.testing.assert_allclose(
            step.amor_driver.particle_driver,
            expected["driver"],
            rtol=0.0,
            atol=4.0e-13,
        )

        edges, _, _, _, _, eta, _, _ = reference
        particle_strain = np.zeros((len(self.coordinates), 2, 2), dtype=np.float64)
        particle_stress = np.zeros_like(particle_strain)
        np.add.at(particle_strain, edges[:, 0], eta[:, None, None] * expected_strain)
        np.add.at(particle_stress, edges[:, 0], eta[:, None, None] * expected_stress)
        damage = expected["damage"]
        edge_effective = ((1.0 - damage[edges[:, 0]]) ** 2 + self.mechanics.k_res)[
            :, None, None
        ] * expected_stress
        particle_effective = (
            ((1.0 - damage) ** 2 + self.mechanics.k_res)[:, None, None]
            * particle_stress
        )
        np.testing.assert_allclose(
            production_evaluation.particle_strain,
            particle_strain,
            rtol=0.0,
            atol=4.0e-12,
        )
        np.testing.assert_allclose(
            production_evaluation.particle_effective_stress,
            particle_effective,
            rtol=0.0,
            atol=4.0e-11,
        )
        np.testing.assert_allclose(
            production_evaluation.edge_effective_stress,
            edge_effective,
            rtol=0.0,
            atol=4.0e-11,
        )

        crack, history_energy, graph, total = expected["damage_energies"]
        self.assertAlmostEqual(step.fixed_history_result.crack_local_energy, crack, places=13)
        self.assertAlmostEqual(step.fixed_history_result.history_energy, history_energy, places=13)
        self.assertAlmostEqual(step.fixed_history_result.graph_energy, graph, places=13)
        self.assertAlmostEqual(step.fixed_history_result.total_energy, total, places=13)
        np.testing.assert_allclose(
            step.fixed_history_result.strong_residual,
            expected["strong_residual"],
            rtol=0.0,
            atol=4.0e-14,
        )
        np.testing.assert_array_equal(
            step.fixed_history_result.strong_residual, step.diagnostics.rho_used
        )
        np.testing.assert_allclose(
            step.diagnostics.rho_next,
            1.7 / 0.65 * damage
            + 1.7 * 0.65 * (_reference_damage_system(
                self.coordinates,
                self.volumes,
                self.edges,
                1.51,
                1.7,
                0.65,
                expected["H_next"],
            )[0] @ damage)
            + 2.0 * expected["H_next"] * damage
            - 2.0 * expected["H_next"],
            rtol=0.0,
            atol=4.0e-14,
        )
        np.testing.assert_allclose(
            expected["tangent"], expected["tangent"].T, rtol=0.0, atol=2.0e-10
        )
        free_tangent = expected["tangent"][
            np.ix_(self.constraints.free_dofs, self.constraints.free_dofs)
        ]
        self.assertGreater(np.min(np.linalg.eigvalsh(free_tangent)), 0.0)

    def test_clean_two_step_convergence_and_exact_final_pairing(self) -> None:
        result = self._run()
        self.assertEqual(len(result.accepted_steps), 2)
        self.assertEqual([state.u_bar for state in result.states], [0.0, 0.01, 0.02])
        self.assertEqual(result.final_state.u_bar, 0.02)
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.metadata.n_particles, len(self.coordinates))
        self.assertEqual(result.metadata.n_dofs, 2 * len(self.coordinates))
        self.assertIs(result.metadata.config, result.metadata.config)
        for values in (
            result.metadata.coordinates,
            result.metadata.volumes,
            result.metadata.directed_edges,
            result.metadata.bottom_mask,
            result.metadata.top_mask,
            result.metadata.constrained_dofs,
            result.metadata.free_dofs,
        ):
            self.assertFalse(values.flags.writeable)
            with self.assertRaises(ValueError):
                values.setflags(write=True)
        for step in result.accepted_steps:
            state = step.state
            np.testing.assert_array_equal(state.displacement, step.mechanics_result.displacement)
            np.testing.assert_array_equal(state.damage, step.fixed_history_result.damage)
            np.testing.assert_array_equal(state.history, step.amor_driver.trial_history)
            self.assertTrue(step.diagnostics.passed)
            self.assertLessEqual(
                step.diagnostics.e_rho,
                step.diagnostics.rho_limit,
            )
            rho = (
                1.7 / 0.65 * state.damage
                + 1.7 * 0.65 * (self.operator.to_scipy() @ state.damage)
                + 2.0 * state.history * state.damage
                - 2.0 * state.history
            )
            np.testing.assert_array_equal(step.diagnostics.rho_next, rho)
            np.testing.assert_array_equal(
                step.diagnostics.rho_next_volume,
                self.volumes * rho,
            )

    def test_committed_state_is_bytes_backed_and_setflags_cannot_reenable_write(self) -> None:
        result = self._run()
        for state in result.states:
            for values in (state.displacement, state.damage, state.history):
                self.assertFalse(values.flags.writeable)
                with self.assertRaises(ValueError):
                    values.setflags(write=True)
                values.setflags(write=False)
        # Inputs and trial buffers are not aliases of any committed field.
        self.assertFalse(np.shares_memory(result.final_state.damage, self.operator.volumes))
        self.assertIsNot(result.final_state.damage, result.accepted_steps[-1].fixed_history_result.damage)

        for step in result.accepted_steps:
            arrays = (
                step.history_used,
                step.amor_driver.edge_positive_energy,
                step.amor_driver.particle_driver,
                step.amor_driver.previous_history,
                step.amor_driver.trial_history,
                step.fixed_history_result.damage,
                step.fixed_history_result.history,
                step.fixed_history_result.strong_residual,
                step.fixed_history_result.euclidean_residual,
                step.mechanics_result.displacement,
                step.mechanics_result.free_residual,
                step.mechanics_result.total_force,
                step.mechanics_result.constraints.bottom_mask,
                step.mechanics_result.constraints.top_mask,
                step.mechanics_result.constraints.constrained_dofs,
                step.mechanics_result.constraints.free_dofs,
                step.mechanics_result.evaluation.residual,
                step.mechanics_result.evaluation.internal_force,
                step.mechanics_result.evaluation.edge_deformation_gradient,
                step.mechanics_result.evaluation.edge_strain,
                step.mechanics_result.evaluation.edge_stress,
                step.mechanics_result.evaluation.particle_strain,
                step.mechanics_result.evaluation.particle_stress,
                step.mechanics_result.evaluation.edge_effective_stress,
                step.mechanics_result.evaluation.particle_effective_stress,
                step.mechanics_result.evaluation.edge_energy_density,
                step.diagnostics.rho_used,
                step.diagnostics.rho_next,
                step.diagnostics.force_before,
                step.diagnostics.force_after,
            )
            for values in arrays:
                with self.subTest(array=values.shape):
                    self.assertFalse(values.flags.writeable)
                    with self.assertRaises(ValueError):
                        values.setflags(write=True)

    def test_trial_history_always_uses_last_committed_history(self) -> None:
        observed: list[np.ndarray] = []
        original = baseline.compute_amor_history

        def observe(mechanics, displacement, previous_history):
            observed.append(np.array(previous_history, copy=True))
            return original(mechanics, displacement, previous_history)

        with mock.patch.object(baseline, "compute_amor_history", side_effect=observe):
            result = self._run()
        self.assertGreaterEqual(len(observed), 4)
        first_step_calls = len(observed) // 2
        for index, history in enumerate(observed):
            expected = result.states[0 if index < first_step_calls else 1].history
            np.testing.assert_array_equal(history, expected)

    def test_complete_sweep_has_no_standalone_final_damage_resolve(self) -> None:
        original_damage = baseline.solve_fixed_history_damage
        original_mechanics = baseline.solve_fixed_damage_mechanics
        damage_calls = 0
        mechanics_calls = 0

        def count_damage(*args, **kwargs):
            nonlocal damage_calls
            damage_calls += 1
            return original_damage(*args, **kwargs)

        def count_mechanics(*args, **kwargs):
            nonlocal mechanics_calls
            mechanics_calls += 1
            return original_mechanics(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_history_damage", side_effect=count_damage
        ), mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=count_mechanics
        ):
            result = self._run(
                _config(load_final=0.01, du_initial=0.01, du_min=0.01)
            )
        stagger_sweeps = sum(
            attempt.stagger_iterations
            for attempt in result.attempts
            if attempt.accepted
        )
        self.assertEqual(damage_calls, stagger_sweeps)
        self.assertEqual(mechanics_calls, 2 * stagger_sweeps)

    def test_trial_mechanics_seeds_are_exact_and_views_remain_fresh(self) -> None:
        original = baseline.solve_fixed_damage_mechanics
        calls = []

        def observe(*args, **kwargs):
            seed = np.array(kwargs["initial_guess"], copy=True)
            result = original(*args, **kwargs)
            calls.append(
                {
                    "view": args[0],
                    "damage": np.array(args[0].damage, copy=True),
                    "load": float(args[2]),
                    "seed": seed,
                    "result": np.array(result.displacement, copy=True),
                    "cg_iterations": result.cg_iterations,
                    "allow_early_exit": kwargs["allow_initial_early_exit"],
                    "early_exit": result.early_exit,
                }
            )
            return result

        config = _config(load_final=0.02, du_initial=0.01, du_min=0.01)
        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=observe
        ):
            result = self._run(config)

        self.assertEqual(len({id(call["view"]) for call in calls}), len(calls))
        first_seed = calls[0]["seed"]
        np.testing.assert_array_equal(
            first_seed, self.constraints.prescribed_vector(0.01)
        )
        for index in range(1, len(calls)):
            previous = calls[index - 1]
            current = calls[index]
            if current["load"] == previous["load"]:
                np.testing.assert_array_equal(current["seed"], previous["result"])
        self.assertFalse(calls[0]["allow_early_exit"])
        for index, call in enumerate(calls):
            if bool(call["allow_early_exit"]):
                self.assertGreaterEqual(index, 2)
                self.assertEqual(call["load"], calls[index - 1]["load"])
                np.testing.assert_array_equal(call["damage"], calls[index - 1]["damage"])
                self.assertTrue(call["early_exit"])

        second_load_index = next(
            index for index, call in enumerate(calls) if call["load"] == 0.02
        )
        committed = result.states[1]
        expected = self.constraints.prescribed_vector(0.02)
        expected[self.constraints.free_dofs] += 2.0 * (
            committed.displacement[self.constraints.free_dofs]
            - self.constraints.prescribed_vector(0.01)[self.constraints.free_dofs]
        )
        np.testing.assert_array_equal(calls[second_load_index]["seed"], expected)
        self.assertTrue(
            any(int(call["cg_iterations"]) == 0 for call in calls[1:])
        )

    def test_warm_and_cold_continuation_match(self) -> None:
        config = _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        warm = self._run(config)
        original = baseline.solve_fixed_damage_mechanics

        def cold(*args, **kwargs):
            kwargs["initial_guess"] = None
            return original(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=cold
        ):
            cold_result = self._run(config)
        np.testing.assert_allclose(
            warm.final_state.displacement,
            cold_result.final_state.displacement,
            rtol=0.0,
            atol=3.0e-12,
        )
        np.testing.assert_allclose(
            warm.final_state.damage,
            cold_result.final_state.damage,
            rtol=0.0,
            atol=3.0e-14,
        )
        np.testing.assert_allclose(
            warm.final_state.history,
            cold_result.final_state.history,
            rtol=0.0,
            atol=3.0e-14,
        )
        self.assertEqual(
            [attempt.accepted for attempt in warm.attempts],
            [attempt.accepted for attempt in cold_result.attempts],
        )

    def test_cpu_replay_adapter_matches_cpu_authoritative_continuation(self) -> None:
        config = _config(load_final=0.02, du_initial=0.01, du_min=0.01)
        cpu_result = self._run(config)
        adapter = self._cpu_replay_adapter()
        jax_result = self._run(config, mechanics_solver=adapter)

        self.assertEqual(len(jax_result.states), len(cpu_result.states))
        self.assertEqual(
            [attempt.accepted for attempt in jax_result.attempts],
            [attempt.accepted for attempt in cpu_result.attempts],
        )
        for actual, expected in zip(jax_result.states, cpu_result.states, strict=True):
            self.assertEqual(actual.u_bar, expected.u_bar)
            np.testing.assert_allclose(
                actual.displacement, expected.displacement, rtol=0.0, atol=1.0e-10
            )
            np.testing.assert_allclose(
                actual.damage, expected.damage, rtol=0.0, atol=1.0e-12
            )
            np.testing.assert_allclose(
                actual.history, expected.history, rtol=0.0, atol=1.0e-12
            )
        for step in jax_result.accepted_steps:
            self.assertIsInstance(step.mechanics_result, baseline.MechanicsSolveResult)
            self.assertEqual(step.mechanics_result.cg_info, 0)
        self.assertEqual(
            jax_result.metadata.mechanics_backend, "jax_cpu_cpu_replay"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/jax-continuation.npz"
            baseline.write_continuation_archive(
                path, jax_result, self._archive_metadata()
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(
                    str(archive["mechanics_backend"]), "jax_cpu_cpu_replay"
                )
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")

    def test_cpu_replay_adapter_failure_rolls_back_and_matches_cutback_path(self) -> None:
        clean_adapter = self._cpu_replay_adapter()
        clean = self._run(
            _config(load_final=0.005, du_initial=0.0025, du_min=0.0025),
            mechanics_solver=clean_adapter,
        )
        delegate = self._cpu_replay_adapter()
        original = delegate.backend.solve_strict
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("manufactured JAX adapter failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            delegate.backend, "solve_strict", side_effect=fail_once
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025),
                mechanics_solver=delegate,
            )
        self.assertFalse(result.attempts[0].accepted)
        self.assertEqual(result.attempts[0].failure_stage, "mechanics_before")
        self.assertEqual(
            result.attempts[0].committed_before_digest,
            result.attempts[0].committed_after_digest,
        )
        np.testing.assert_allclose(
            result.final_state.displacement,
            clean.final_state.displacement,
            rtol=0.0,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            result.final_state.damage,
            clean.final_state.damage,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.final_state.history,
            clean.final_state.history,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_cpu_replay_adapter_inputs_are_bytes_backed_and_after_failure_rolls_back(self) -> None:
        adapter = self._cpu_replay_adapter()
        original = adapter.backend.solve_strict
        calls = 0

        def inspect_and_fail_after(damage, *args, **kwargs):
            nonlocal calls
            calls += 1
            seed = kwargs["initial_guess"]
            for values in (damage, seed):
                owner = values
                while isinstance(owner, np.ndarray) and owner.base is not None:
                    owner = owner.base
                self.assertIsInstance(owner, bytes)
                self.assertFalse(values.flags.writeable)
                with self.assertRaises(ValueError):
                    values.setflags(write=True)
            if calls == 2:
                raise RuntimeError("manufactured JAX mechanics-after failure")
            return original(damage, *args, **kwargs)

        with mock.patch.object(
            adapter.backend, "solve_strict", side_effect=inspect_and_fail_after
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025),
                mechanics_solver=adapter,
            )
        self.assertFalse(result.attempts[0].accepted)
        self.assertEqual(result.attempts[0].failure_stage, "mechanics_after")
        self.assertEqual(
            result.attempts[0].committed_before_digest,
            result.attempts[0].committed_after_digest,
        )

    def test_cpu_replay_adapter_structural_replay_error_propagates(self) -> None:
        adapter = self._cpu_replay_adapter()
        with mock.patch.object(
            adapter.backend,
            "replay_cpu",
            side_effect=ValueError("manufactured structural JAX replay error"),
        ):
            with self.assertRaisesRegex(ValueError, "structural JAX replay"):
                self._run(
                    _config(load_final=0.005, du_initial=0.005, du_min=0.0025),
                    mechanics_solver=adapter,
                )

    @unittest.skipUnless(
        any(device.platform == "gpu" for device in jax.devices()),
        "GPU CPU-replay adapter test requires a visible GPU",
    )
    def test_jax_gpu_adapter_runs_cpu_authoritative_continuation(self) -> None:
        gpu_backend = JaxBondAssociatedMechanics(
            self.mechanics,
            self.constraints,
            next(device for device in jax.devices() if device.platform == "gpu"),
        )
        adapter = JaxCPUReplayMechanicsAdapter(gpu_backend)
        result = self._run(
            _config(load_final=0.005, du_initial=0.005, du_min=0.005),
            mechanics_solver=adapter,
        )
        self.assertEqual(result.final_state.u_bar, 0.005)
        self.assertEqual(result.metadata.mechanics_backend, "jax_gpu_cpu_replay")
        self.assertTrue(all(step.diagnostics.passed for step in result.accepted_steps))

    def test_mechanics_failure_rolls_back_and_retries_equivalent_cutback(self) -> None:
        clean = self._run(
            _config(load_final=0.005, du_initial=0.0025, du_min=0.0025)
        )
        original = baseline.solve_fixed_damage_mechanics
        calls = 0
        seeds: list[np.ndarray] = []

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            seeds.append(np.array(kwargs["initial_guess"], copy=True))
            if calls == 1:
                raise RuntimeError("manufactured mechanics failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=fail_once
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
            )
        self.assertFalse(result.attempts[0].accepted)
        self.assertEqual(result.attempts[0].failure_stage, "mechanics_before")
        self.assertEqual(result.attempts[0].committed_before_digest, result.attempts[0].committed_after_digest)
        np.testing.assert_array_equal(
            seeds[0], self.constraints.prescribed_vector(0.005)
        )
        np.testing.assert_array_equal(
            seeds[1], self.constraints.prescribed_vector(0.0025)
        )
        np.testing.assert_array_equal(
            result.accepted_steps[0].state.displacement,
            clean.accepted_steps[0].state.displacement,
        )
        np.testing.assert_array_equal(
            result.accepted_steps[0].state.damage,
            clean.accepted_steps[0].state.damage,
        )
        self.assertEqual(result.accepted_steps[0].cutback_count, 1)
        self.assertEqual(
            [attempt.load_target for attempt in result.attempts],
            [0.005, 0.0025, 0.005],
        )
        self.assertEqual(
            [attempt.cutback for attempt in result.attempts],
            [0, 1, 0],
        )
        self.assertEqual(
            [step.increment for step in result.accepted_steps],
            [0.0025, 0.0025],
        )

    def test_mechanics_validator_rejects_forged_tiny_norms_and_bad_free_residual(self) -> None:
        original = baseline.solve_fixed_damage_mechanics

        def forge_result(*args, **kwargs):
            result = original(*args, **kwargs)
            bad_free_residual = np.array(result.free_residual, copy=True)
            bad_free_residual[0] += 1.0
            return replace(
                result,
                free_residual=bad_free_residual,
                free_residual_norm=1.0e-300,
                free_residual_relative_norm=1.0e-300,
            )

        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.005,
            max_cutbacks=1,
            max_stagger_iterations=1,
        )
        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=forge_result
        ):
            with self.assertRaisesRegex(ValueError, "free.?residual"):
                self._run(config)

    def test_mechanics_after_failure_rolls_back_and_retries(self) -> None:
        original = baseline.solve_fixed_damage_mechanics
        calls = 0

        def fail_after(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("manufactured mechanics-after failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=fail_after
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
            )
        self.assertEqual(result.attempts[0].failure_stage, "mechanics_after")
        self.assertEqual(
            result.attempts[0].committed_before_digest,
            result.attempts[0].committed_after_digest,
        )
        self.assertEqual(result.accepted_steps[0].cutback_count, 1)
        self.assertEqual(result.final_state.u_bar, 0.005)

    def test_after_history_failure_rolls_back_and_retries(self) -> None:
        original = baseline.compute_amor_history
        calls = 0

        def fail_after_history(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("manufactured after-history failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            baseline, "compute_amor_history", side_effect=fail_after_history
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
            )
        self.assertEqual(result.attempts[0].failure_stage, "after_history")
        self.assertEqual(
            result.attempts[0].committed_before_digest,
            result.attempts[0].committed_after_digest,
        )
        self.assertEqual(result.accepted_steps[0].cutback_count, 1)
        diagnostics = result.attempts[0].diagnostics
        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertGreaterEqual(diagnostics.moment_scale, 0.0)
        self.assertEqual(result.final_state.u_bar, 0.005)

    def test_damage_failure_rolls_back_and_cutback_is_recorded(self) -> None:
        original = baseline.solve_fixed_history_damage
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise FloatingPointError("manufactured damage failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_history_damage", side_effect=fail_once
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
            )
        self.assertFalse(result.attempts[0].accepted)
        self.assertEqual(result.attempts[0].failure_stage, "fixed_history_damage")
        self.assertEqual(result.attempts[0].exception_type, "FloatingPointError")
        self.assertEqual(result.attempts[0].committed_before_digest, result.attempts[0].committed_after_digest)
        self.assertEqual(result.accepted_steps[0].cutback_count, 1)

    def test_force_gate_rejects_and_cutbacks(self) -> None:
        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.0025,
            max_cutbacks=1,
            max_stagger_iterations=1,
        )
        config = replace(
            config,
            force=baseline.GateTolerance(np.nextafter(0.0, 1.0), 0.0, 1.0),
        )
        with self.assertRaises(baseline.ContinuationFailure) as context:
            self._run(config)
        failure = context.exception
        self.assertEqual(
            [attempt.load_target for attempt in failure.attempts],
            [0.005, 0.0025],
        )
        for attempt in failure.attempts:
            self.assertEqual(attempt.failure_stage, "gates")
            diagnostics = attempt.diagnostics
            self.assertIsNotNone(diagnostics)
            assert diagnostics is not None
            self.assertFalse(diagnostics.force_pass)
            self.assertIn("force", diagnostics.rejection_reason)

    def test_moment_gate_rejects_and_cutbacks(self) -> None:
        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.0025,
            max_cutbacks=1,
            max_stagger_iterations=1,
        )
        config = replace(
            config,
            moment=baseline.GateTolerance(np.nextafter(0.0, 1.0), 0.0, 1.0),
        )
        with self.assertRaises(baseline.ContinuationFailure) as context:
            self._run(config)
        failure = context.exception
        self.assertEqual(
            [attempt.load_target for attempt in failure.attempts],
            [0.005, 0.0025],
        )
        for attempt in failure.attempts:
            self.assertEqual(attempt.failure_stage, "gates")
            diagnostics = attempt.diagnostics
            self.assertIsNotNone(diagnostics)
            assert diagnostics is not None
            self.assertFalse(diagnostics.moment_pass)
            self.assertIn("moment", diagnostics.rejection_reason)

    def test_raw_damage_range_is_rejected_without_clipping(self) -> None:
        original = baseline.solve_fixed_history_damage
        calls = 0
        bad_damage = None

        def out_of_range(*args, **kwargs):
            nonlocal calls, bad_damage
            result = original(*args, **kwargs)
            if calls == 0:
                bad_damage = result.damage.copy()
                bad_damage[0] = -0.2
                result = replace(
                    result,
                    damage=bad_damage,
                    damage_min=-0.2,
                    below_zero_count=1,
                )
            calls += 1
            return result

        with mock.patch.object(
            baseline, "solve_fixed_history_damage", side_effect=out_of_range
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
            )
        self.assertIsNotNone(bad_damage)
        self.assertFalse(result.attempts[0].accepted)
        self.assertEqual(result.attempts[0].failure_stage, "damage_acceptance")
        self.assertIsNotNone(result.attempts[0].diagnostics)
        diagnostics = result.attempts[0].diagnostics
        assert diagnostics is not None
        self.assertLess(diagnostics.damage_min, 0.0)
        self.assertTrue(np.all(result.final_state.damage >= 0.0))

    def test_committed_residual_gate_rejects_and_minimum_failure_preserves_state(self) -> None:
        original = baseline.compute_amor_history
        calls = 0

        def inflated_after_history(mechanics, displacement, previous_history):
            nonlocal calls
            result = original(mechanics, displacement, previous_history)
            calls += 1
            if calls % 2 == 0:
                inflated = result.trial_history + 0.5
                return replace(result, trial_history=inflated)
            return result

        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.005,
            max_cutbacks=1,
            max_stagger_iterations=1,
        )
        with mock.patch.object(
            baseline, "compute_amor_history", side_effect=inflated_after_history
        ):
            with self.assertRaises(baseline.ContinuationFailure) as context:
                self._run(config)
        failure = context.exception
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        self.assertEqual(len(failure.attempts), 1)
        attempt = failure.attempts[0]
        self.assertEqual(attempt.failure_stage, "gates")
        self.assertIsNotNone(attempt.diagnostics)
        diagnostics = attempt.diagnostics
        assert diagnostics is not None
        self.assertFalse(diagnostics.rho_pass)
        self.assertFalse(diagnostics.passed)
        self.assertEqual(attempt.committed_before_digest, attempt.committed_after_digest)
        self.assertEqual(calls, 2)

    def test_unknown_exception_propagates(self) -> None:
        with mock.patch.object(
            baseline,
            "solve_fixed_history_damage",
            side_effect=KeyError("unknown trial error"),
        ):
            with self.assertRaises(KeyError):
                self._run(_config(load_final=0.005, du_initial=0.005, du_min=0.0025))

    def test_lazy_fixed_views_do_not_leak_into_committed_result(self) -> None:
        result = self._run(_config(load_final=0.01, du_initial=0.01, du_min=0.01))
        state = result.final_state
        self.assertNotIn("_tangent_block_diagonal_cache", state.__dict__)
        self.assertNotIn("_tangent_diagonal_cache", state.__dict__)
        self.assertFalse(hasattr(result.accepted_steps[0].mechanics_result, "intact_mechanics"))
        self.assertIsInstance(result.attempts, tuple)
        self.assertIsInstance(result.accepted_steps, tuple)

    def test_forged_accepted_step_load_and_residual_invariants_fail(self) -> None:
        result = self._run(_config(load_final=0.01, du_initial=0.01, du_min=0.01))
        step = result.accepted_steps[0]
        with self.assertRaisesRegex(ValueError, "increment|u_bar.*load_target"):
            replace(step, load_target=step.load_target + 1.0e-4)
        bad_mechanics = replace(
            step.mechanics_result,
            u_bar=step.load_target + 1.0e-4,
        )
        with self.assertRaisesRegex(ValueError, "mechanics_result.u_bar"):
            replace(step, mechanics_result=bad_mechanics)
        bad_diagnostics = replace(step.diagnostics, passed=False)
        with self.assertRaisesRegex(ValueError, "diagnostics.*passed"):
            replace(step, diagnostics=bad_diagnostics)
        bad_strong = np.array(step.fixed_history_result.strong_residual, copy=True)
        bad_strong[0] += 1.0
        bad_fixed = replace(step.fixed_history_result, strong_residual=bad_strong)
        with self.assertRaisesRegex(ValueError, "strong residual"):
            replace(step, fixed_history_result=bad_fixed)

        bad_euclidean = np.array(
            step.fixed_history_result.euclidean_residual, copy=True
        )
        bad_euclidean[0] += 1.0
        bad_fixed = replace(
            step.fixed_history_result,
            euclidean_residual=bad_euclidean,
            euclidean_residual_norm=float(np.linalg.norm(bad_euclidean)),
            euclidean_residual_inf=float(np.max(np.abs(bad_euclidean))),
        )
        forged_step = replace(step, fixed_history_result=bad_fixed)
        forged_attempt = replace(
            result.attempts[0], diagnostics=forged_step.diagnostics
        )
        with self.assertRaisesRegex(ValueError, "Euclidean residual"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        with self.assertRaisesRegex(ValueError, "relative.?norm"):
            replace(
                step.mechanics_result,
                free_residual_relative_norm=(
                    step.mechanics_result.free_residual_relative_norm + 1.0
                ),
            )

    def test_amor_previous_history_is_paired_with_preceding_committed_state(self) -> None:
        result = self._run()
        step = result.accepted_steps[1]
        forged_amor = replace(
            step.amor_driver,
            previous_history=np.zeros_like(step.amor_driver.previous_history),
        )
        forged_step = replace(step, amor_driver=forged_amor)
        forged_steps = (result.accepted_steps[0], forged_step)
        forged_attempts = (
            result.attempts[0],
            replace(result.attempts[1], diagnostics=forged_step.diagnostics),
        )
        with self.assertRaisesRegex(ValueError, "previous history"):
            replace(
                result,
                accepted_steps=forged_steps,
                attempts=forged_attempts,
            )

    def test_structural_error_while_building_failure_diagnostics_propagates(self) -> None:
        original_mechanics = baseline.solve_fixed_damage_mechanics
        calls = 0

        def fail_after(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("manufactured mechanics-after failure")
            return original_mechanics(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=fail_after
        ), mock.patch.object(
            baseline,
            "_damage_residuals",
            side_effect=ValueError("manufactured structural diagnostic error"),
        ):
            with self.assertRaisesRegex(ValueError, "structural diagnostic"):
                self._run(
                    _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
                )

    def test_structural_sparse_solve_error_propagates(self) -> None:
        with mock.patch.object(
            baseline,
            "spsolve",
            side_effect=ValueError("manufactured structural sparse-solve error"),
        ):
            with self.assertRaisesRegex(ValueError, "structural sparse-solve"):
                self._run(
                    _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
                )

    def test_result_replay_rejects_coherently_forged_mechanics(self) -> None:
        result = self._run(
            _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        )
        step = result.accepted_steps[0]
        forged_evaluation = replace(
            step.mechanics_result.evaluation,
            energy=step.mechanics_result.evaluation.energy + 1.0,
        )
        forged_mechanics = replace(
            step.mechanics_result,
            evaluation=forged_evaluation,
        )
        forged_step = replace(step, mechanics_result=forged_mechanics)
        forged_attempt = replace(
            result.attempts[0], diagnostics=forged_step.diagnostics
        )
        with self.assertRaisesRegex(ValueError, "evaluation energy"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        forged_rhs_norm = step.mechanics_result.free_rhs_norm + 1.0
        forged_mechanics = replace(
            step.mechanics_result,
            free_rhs_norm=forged_rhs_norm,
            free_residual_relative_norm=(
                step.mechanics_result.free_residual_norm / forged_rhs_norm
            ),
        )
        forged_diagnostics = replace(
            step.diagnostics,
            mechanics_after_free_residual_relative_norm=(
                forged_mechanics.free_residual_relative_norm
            ),
        )
        forged_step = replace(
            step,
            mechanics_result=forged_mechanics,
            diagnostics=forged_diagnostics,
        )
        forged_attempt = replace(
            result.attempts[0], diagnostics=forged_diagnostics
        )
        with self.assertRaisesRegex(ValueError, "right-hand-side norm"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            replace(
                step.mechanics_result,
                free_rhs_norm=-1.0,
                free_residual_relative_norm=(
                    step.mechanics_result.free_residual_norm / 1.0e-300
                ),
            )

        stricter_config = replace(
            result.metadata.config,
            cg_rtol=0.0,
            cg_atol=step.mechanics_result.free_residual_norm / 2.0,
        )
        stricter_metadata = replace(result.metadata, config=stricter_config)
        with self.assertRaisesRegex(
            ValueError, "CG convergence limit|configured CG tolerance"
        ):
            replace(result, metadata=stricter_metadata)

    def test_model_operator_and_constraint_mismatches_fail_loudly(self) -> None:
        bad_coordinates = self.coordinates.copy()
        bad_coordinates[0, 0] += 1.0e-3
        bad_operator = ExplicitVolumeSparseD(
            bad_coordinates,
            self.volumes,
            self.edges,
            1.51,
            "cubic",
        )
        with self.assertRaisesRegex(ValueError, "coordinates"):
            baseline.run_strict_continuation(
                bad_operator,
                self.mechanics,
                self.constraints,
                1.7,
                0.65,
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025),
            )

        _, _, _, _, other_mechanics, other_constraints = _system(n_side=3)
        del other_mechanics
        with self.assertRaisesRegex(ValueError, "particle count"):
            baseline.run_strict_continuation(
                self.operator,
                self.mechanics,
                other_constraints,
                1.7,
                0.65,
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025),
            )

    def test_metadata_graph_validation_rejects_invalid_distance_and_family_mass(self) -> None:
        result = self._run(
            _config(load_final=0.005, du_initial=0.005, du_min=0.005)
        )
        metadata = result.metadata
        invalid_edge = np.asarray([[0, 8], [8, 0]], dtype=np.int64)
        distance_edges = np.concatenate((metadata.directed_edges, invalid_edge), axis=0)
        distance_edges = distance_edges[
            np.lexsort((distance_edges[:, 1], distance_edges[:, 0]))
        ]
        with self.assertRaisesRegex(ValueError, "strict 0 < distance < delta"):
            replace(metadata, directed_edges=distance_edges)

        without_particle_zero = metadata.directed_edges[
            (metadata.directed_edges[:, 0] != 0)
            & (metadata.directed_edges[:, 1] != 0)
        ]
        with self.assertRaisesRegex(ValueError, "family mass"):
            replace(metadata, directed_edges=without_particle_zero)
        with self.assertRaisesRegex(ValueError, "rank_rtol"):
            replace(metadata, rank_rtol=0.0)

    def test_continuation_result_rejects_empty_and_mismatched_attempt_logs(self) -> None:
        result = self._run(
            _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        )
        with self.assertRaisesRegex(ValueError, "attempts.*empty|accepted-attempt"):
            replace(result, attempts=())

        forged_stagger = replace(
            result.attempts[0],
            stagger_iterations=result.attempts[0].stagger_iterations + 1,
        )
        with self.assertRaisesRegex(ValueError, "stagger iterations"):
            replace(result, attempts=(forged_stagger,))

        forged_digest = replace(
            result.attempts[0],
            committed_after_digest="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "committed-after digest"):
            replace(result, attempts=(forged_digest,))

        attempt_diagnostics = result.attempts[0].diagnostics
        self.assertIsNotNone(attempt_diagnostics)
        assert attempt_diagnostics is not None
        forged_diagnostics = replace(
            attempt_diagnostics,
            e_u=attempt_diagnostics.e_u + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            replace(
                result,
                attempts=(replace(result.attempts[0], diagnostics=forged_diagnostics),),
            )

        object.__setattr__(result, "attempts", (forged_stagger,))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "stagger iterations"):
                baseline.write_continuation_archive(
                    f"{directory}/corrupt.npz", result, self._archive_metadata()
                )

    def test_zero_progress_after_cutback_is_typed_and_audited(self) -> None:
        original = baseline.solve_fixed_damage_mechanics

        def fail_after_first_step(*args, **kwargs):
            if args[2] > 0.015:
                raise RuntimeError("manufactured second-step failure")
            return original(*args, **kwargs)

        config = _config(
            load_final=0.02,
            du_initial=0.01,
            du_min=1.0e-300,
            max_cutbacks=1,
        )
        config = replace(config, cutback_factor=1.0e-300)
        with mock.patch.object(
            baseline,
            "solve_fixed_damage_mechanics",
            side_effect=fail_after_first_step,
        ):
            with self.assertRaises(baseline.ContinuationFailure) as context:
                self._run(config)
        failure = context.exception
        self.assertIn("no floating-point load progress", failure.message)
        self.assertEqual(failure.last_committed_state.u_bar, 0.01)
        self.assertEqual(len(failure.attempts), 2)
        self.assertEqual(failure.attempts[-1].failure_stage, "mechanics_before")
        self.assertEqual(
            failure.attempts[-1].committed_before_digest,
            failure.attempts[-1].committed_after_digest,
        )

    def test_gate_and_continuation_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            baseline.GateTolerance(-1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            baseline.GateTolerance(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            _config(du_initial=0.001, du_min=0.0025)
        with self.assertRaises(ValueError):
            base = _config()
            baseline.StrictContinuationConfig(
                0.0,
                0.01,
                0.01,
                0.001,
                1.0,
                1,
                1,
                1.0e-8,
                1.0e-10,
                10,
                base.displacement,
                base.damage,
                base.history,
                base.reaction,
                base.damage_residual,
                base.force,
                base.moment,
                base.healing_atol,
                base.damage_range_atol,
            )

    def test_gate_limit_and_attempt_archive_audit(self) -> None:
        tolerance = baseline.GateTolerance(2.0e-3, 0.1, 0.5)
        self.assertEqual(tolerance.limit(0.1), 0.05)
        self.assertEqual(tolerance.limit(10.0), 1.0)

        original = baseline.solve_fixed_damage_mechanics
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("audit failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=fail_once
        ):
            result = self._run(
                _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
            )
        self.assertEqual(
            result.attempts[0].committed_before_digest,
            result.attempts[0].committed_after_digest,
        )
        self.assertNotEqual(
            result.attempts[-1].committed_before_digest,
            result.attempts[-1].committed_after_digest,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/continuation.npz"
            archive_metadata = self._archive_metadata()
            baseline.write_continuation_archive(path, result, archive_metadata)
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "coordinates",
                    "volumes",
                    "directed_edges",
                    "delta",
                    "kernel",
                    "bottom_mask",
                    "top_mask",
                    "anchor_particle",
                    "constrained_dofs",
                    "free_dofs",
                    "E",
                    "nu",
                    "k_res",
                    "rank_rtol",
                    "Gc",
                    "l0",
                    "plane_strain",
                    "stabilization",
                    "mechanics_backend",
                    "dtype",
                    "triangles",
                    "first_moments",
                    "visibility_tolerance",
                    "visibility_rule",
                    "quadrature_rule",
                    "study_label",
                    "model_provenance",
                    "config_json",
                    "state_u_bar",
                    "state_displacement",
                    "state_damage",
                    "state_history",
                    "accepted_mechanics_energy",
                    "accepted_mechanics_residual",
                    "accepted_mechanics_internal_force",
                    "accepted_fixed_history_crack_energy",
                    "accepted_fixed_history_history_energy",
                    "accepted_fixed_history_graph_energy",
                    "accepted_fixed_history_total_energy",
                    "accepted_fixed_history_strong_residual_norm",
                    "accepted_fixed_history_strong_residual_inf",
                    "accepted_fixed_history_euclidean_residual_norm",
                    "accepted_fixed_history_euclidean_residual_inf",
                    "accepted_fixed_history_damage_min",
                    "accepted_fixed_history_damage_max",
                    "accepted_fixed_history_below_zero_count",
                    "accepted_fixed_history_above_one_count",
                    "accepted_fixed_history_symmetry_defect",
                    "accepted_fixed_history_symmetry_relative_defect",
                    "accepted_committed_history_energy",
                    "accepted_committed_damage_total_energy",
                    "accepted_edge_F",
                    "accepted_edge_strain",
                    "accepted_edge_stress",
                    "accepted_edge_effective_stress",
                    "accepted_particle_strain",
                    "accepted_particle_stress",
                    "accepted_particle_effective_stress",
                    "accepted_amor_edge_positive_energy",
                    "accepted_amor_particle_driver",
                    "accepted_H_used",
                    "accepted_H_next",
                    "accepted_rho_next_volume",
                    "accepted_cg_iterations",
                    "accepted_free_rhs_norm",
                    "accepted_initial_free_residual_norm",
                    "accepted_cg_convergence_limit",
                    "accepted_initial_guess_supplied",
                    "accepted_cg_initial_guess_used",
                    "accepted_early_exit",
                    "cumulative_top_reaction_work",
                    "attempt_audit_json",
                }
                self.assertTrue(required.issubset(set(archive.files)))
                self.assertEqual(archive["state_u_bar"].shape[0], 3)
                self.assertEqual(archive["accepted_rho_next_volume"].shape[0], 2)
                np.testing.assert_array_equal(archive["coordinates"], result.metadata.coordinates)
                np.testing.assert_array_equal(archive["triangles"], archive_metadata.triangles)
                self.assertEqual(str(archive["dtype"]), "float64")
                self.assertEqual(str(archive["stabilization"]), "none")
                self.assertEqual(str(archive["mechanics_backend"]), "numpy_scipy")
                self.assertEqual(float(archive["rank_rtol"]), self.mechanics.rank_rtol)
                np.testing.assert_array_equal(
                    archive["accepted_cg_iterations"],
                    np.asarray(
                        [
                            step.mechanics_result.cg_iterations
                            for step in result.accepted_steps
                        ],
                        dtype=np.int64,
                    ),
                )
                np.testing.assert_array_equal(
                    archive["accepted_fixed_history_strong_residual_norm"],
                    np.asarray(
                        [
                            step.fixed_history_result.strong_residual_norm
                            for step in result.accepted_steps
                        ],
                        dtype=np.float64,
                    ),
                )
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")
                config_json = json.loads(str(archive["config_json"]))
                self.assertEqual(config_json["load_final"], 0.005)
                metadata_json = json.loads(str(archive["metadata_json"]))
                self.assertEqual(metadata_json["rank_rtol"], self.mechanics.rank_rtol)
                self.assertIs(metadata_json["plane_strain"], True)
                self.assertEqual(metadata_json["mechanics_backend"], "numpy_scipy")
                audit = json.loads(str(archive["attempt_audit_json"]))
                self.assertEqual(len(audit), len(result.attempts))
                self.assertIsInstance(audit[0]["accepted"], bool)
                self.assertEqual(
                    audit[0]["committed_before_digest"],
                    result.attempts[0].committed_before_digest,
                )
                self.assertIn("errors", audit[0]["diagnostics"])
                self.assertIn("scales", audit[0]["diagnostics"])
                self.assertIn("limits", audit[0]["diagnostics"])
                self.assertIn("rho_next", audit[0]["diagnostics"])
                self.assertTrue(
                    all(
                        isinstance(value, bool)
                        for value in audit[0]["diagnostics"]["passes"].values()
                    )
                )
                self.assertEqual(audit[0]["failure_stage"], "mechanics_before")

            corrupt_archive_metadata = self._archive_metadata()
            object.__setattr__(
                corrupt_archive_metadata,
                "triangles",
                np.asarray(corrupt_archive_metadata.triangles, dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "triangles"):
                baseline.write_continuation_archive(
                    f"{directory}/object-dtype.npz",
                    result,
                    corrupt_archive_metadata,
                )

    def test_conservation_gate_scales_use_reactions_and_geometry(self) -> None:
        result = self._run(
            _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        )
        step = result.accepted_steps[0]
        diagnostics = step.diagnostics
        expected_reaction_scale = max(
            abs(step.mechanics_result.top_reaction),
            abs(step.mechanics_result.bottom_reaction),
            abs(step.mechanics_result.anchor_reaction),
            abs(diagnostics.reaction_before),
            abs(diagnostics.reaction_after),
        )
        geometry_length = max(
            float(np.max(np.abs(self.coordinates))),
            float(np.max(np.abs(np.ptp(self.coordinates, axis=0)))),
        )
        self.assertEqual(diagnostics.R_scale, expected_reaction_scale)
        self.assertEqual(diagnostics.force_scale, expected_reaction_scale)
        self.assertEqual(
            diagnostics.moment_scale,
            expected_reaction_scale * geometry_length,
        )
        self.assertEqual(
            diagnostics.force_limit,
            _config(
                load_final=0.01, du_initial=0.01, du_min=0.01
            ).force.limit(expected_reaction_scale),
        )
        self.assertEqual(
            diagnostics.moment_limit,
            _config(
                load_final=0.01, du_initial=0.01, du_min=0.01
            ).moment.limit(expected_reaction_scale * geometry_length),
        )


class Fracture2DWarningContinuationTest(unittest.TestCase):
    """All-gate warning diagnostic mode: commit, audit, archive contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.initial_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        (
            cls.coordinates,
            cls.volumes,
            cls.edges,
            cls.operator,
            cls.mechanics,
            cls.constraints,
        ) = _system()

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.initial_x64)

    def _run_warning(
        self,
        config: baseline.StrictContinuationConfig | None = None,
        *,
        mechanics_solver=None,
    ):
        return baseline.run_warning_continuation(
            self.operator,
            self.mechanics,
            self.constraints,
            1.7,
            0.65,
            _config() if config is None else config,
            mechanics_solver=mechanics_solver,
        )

    def _run_strict(
        self,
        config: baseline.StrictContinuationConfig | None = None,
        *,
        mechanics_solver=None,
    ):
        return baseline.run_strict_continuation(
            self.operator,
            self.mechanics,
            self.constraints,
            1.7,
            0.65,
            _config() if config is None else config,
            mechanics_solver=mechanics_solver,
        )

    def _forced_force_moment_config(
        self,
        *,
        load_final: float = 0.005,
        du: float = 0.005,
        max_stagger_iterations: int = 20,
    ) -> baseline.StrictContinuationConfig:
        config = _config(
            load_final=load_final,
            du_initial=du,
            du_min=du,
            max_stagger_iterations=max_stagger_iterations,
        )
        tiny = baseline.GateTolerance(np.nextafter(0.0, 1.0), 0.0, 1.0)
        return replace(config, force=tiny, moment=tiny)

    def _cpu_replay_adapter(self) -> JaxCPUReplayMechanicsAdapter:
        backend = JaxBondAssociatedMechanics(
            self.mechanics,
            self.constraints,
            jax.devices("cpu")[0],
        )
        return JaxCPUReplayMechanicsAdapter(backend)

    def _archive_metadata(self) -> baseline.SupplementalArchiveMetadata:
        return baseline.SupplementalArchiveMetadata(
            triangles=_archive_triangles(4),
            first_moments=np.sum(self.volumes[:, None] * self.coordinates, axis=0),
            visibility_tolerance=0.0,
            visibility_rule="manufactured reciprocal graph",
            quadrature_rule="manufactured explicit particle weights",
            study_label="manufactured-cloud-warning-continuation",
            model_provenance="independent Stage 0-4 reference helpers",
        )

    def test_warning_commit_runs_full_budget_with_exact_bypass_list(self) -> None:
        config = self._forced_force_moment_config()
        result = self._run_warning(config)
        self.assertEqual(result.metadata.acceptance_mode, "warning")
        self.assertEqual(result.final_state.u_bar, 0.005)
        self.assertEqual(len(result.accepted_steps), 1)
        step = result.accepted_steps[0]
        diagnostics = step.diagnostics
        self.assertEqual(step.acceptance_status, "warning_accepted")
        self.assertFalse(diagnostics.passed)
        self.assertTrue(diagnostics.complete_sweep)
        self.assertTrue(diagnostics.mechanics_pass)
        self.assertEqual(
            step.stagger_iterations, config.max_stagger_iterations
        )
        # Every other gate converges before the budget, so the canonical
        # bypass list is exactly the two forced failures, in canonical order.
        self.assertEqual(step.bypassed_gates, ("force", "moment"))
        expected = tuple(
            name
            for name in baseline.WARNING_GATE_NAMES
            if not diagnostics.gate_passes[name]
        )
        self.assertEqual(step.bypassed_gates, expected)
        self.assertFalse(diagnostics.force_pass)
        self.assertFalse(diagnostics.moment_pass)
        attempt = result.attempts[0]
        self.assertTrue(attempt.accepted)
        self.assertEqual(attempt.acceptance_status, "warning_accepted")
        self.assertTrue(attempt.sweep_complete)
        self.assertEqual(attempt.bypassed_gates, ("force", "moment"))
        self.assertEqual(attempt.stagger_iterations, config.max_stagger_iterations)
        # The warning commit pairs the exact unmodified candidate fields.
        np.testing.assert_array_equal(
            step.state.displacement, step.mechanics_result.displacement
        )
        np.testing.assert_array_equal(
            step.state.damage, step.fixed_history_result.damage
        )
        np.testing.assert_array_equal(step.state.history, step.amor_driver.trial_history)
        np.testing.assert_array_equal(step.history_used, step.fixed_history_result.history)

    def test_warning_commit_endpoint_and_load_pairing(self) -> None:
        config = self._forced_force_moment_config(
            load_final=0.01, du=0.005, max_stagger_iterations=3
        )
        result = self._run_warning(config)
        self.assertEqual(
            [state.u_bar for state in result.states], [0.0, 0.005, 0.01]
        )
        self.assertEqual(result.final_state.u_bar, config.load_final)
        for index, step in enumerate(result.accepted_steps, start=1):
            self.assertIs(step.state, result.states[index])
            self.assertEqual(step.load_start, result.states[index - 1].u_bar)
            self.assertEqual(step.load_target, result.states[index].u_bar)
            self.assertEqual(step.acceptance_status, "warning_accepted")
        for attempt in result.attempts:
            self.assertTrue(attempt.accepted)
            self.assertEqual(attempt.stagger_iterations, 3)
        # Exact accepted attempt/step load pairing.
        for attempt, step in zip(
            result.attempts, result.accepted_steps, strict=True
        ):
            self.assertEqual(attempt.load_start, step.load_start)
            self.assertEqual(attempt.load_target, step.load_target)

    def test_warning_mode_commits_raw_out_of_range_damage_without_clipping(
        self,
    ) -> None:
        original = baseline.solve_fixed_history_damage
        D = self.operator.to_scipy()
        Gc, l0 = 1.7, 0.65
        row, col = self.edges[:, 0], self.edges[:, 1]
        directed_weight = -np.asarray(D[row, col]).reshape(-1)
        mechanics_calls = 0
        watcher = baseline.solve_fixed_damage_mechanics

        def count_mechanics(*args, **kwargs):
            nonlocal mechanics_calls
            mechanics_calls += 1
            return watcher(*args, **kwargs)

        def out_of_range(operator, gc, l0_value, history):
            result = original(operator, gc, l0_value, history)
            damage = np.array(result.damage, copy=True)
            damage[0] = -0.2
            strong = (
                (gc / l0_value) * damage
                + gc * l0_value * np.asarray(D @ damage)
                + 2.0 * history * damage
                - 2.0 * history
            )
            euclidean = self.volumes * strong
            crack = float(
                np.sum(self.volumes * (gc / (2.0 * l0_value)) * damage**2)
            )
            history_energy = float(
                np.sum(self.volumes * history * (1.0 - damage) ** 2)
            )
            graph = float(
                gc
                * l0_value
                / 4.0
                * np.sum(
                    self.volumes[row]
                    * directed_weight
                    * (damage[col] - damage[row]) ** 2
                )
            )
            return replace(
                result,
                damage=damage,
                crack_local_energy=crack,
                history_energy=history_energy,
                graph_energy=graph,
                total_energy=crack + history_energy + graph,
                strong_residual=strong,
                euclidean_residual=euclidean,
                strong_residual_norm=float(np.linalg.norm(strong)),
                strong_residual_inf=float(np.max(np.abs(strong))),
                euclidean_residual_norm=float(np.linalg.norm(euclidean)),
                euclidean_residual_inf=float(np.max(np.abs(euclidean))),
                damage_min=float(np.min(damage)),
                damage_max=float(np.max(damage)),
                below_zero_count=int(np.count_nonzero(damage < 0.0)),
                above_one_count=int(np.count_nonzero(damage > 1.0)),
            )

        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.005,
            max_stagger_iterations=3,
        )
        with mock.patch.object(
            baseline, "solve_fixed_history_damage", side_effect=out_of_range
        ), mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=count_mechanics
        ):
            result = self._run_warning(config)
        step = result.accepted_steps[0]
        self.assertEqual(step.acceptance_status, "warning_accepted")
        self.assertEqual(step.stagger_iterations, 3)
        # Range/healing failures were bypassed, not short-circuited: the
        # sweep always executed the mechanics-after solve, and the committed
        # damage is the exact unmodified raw candidate.
        self.assertEqual(mechanics_calls, 2 * step.stagger_iterations)
        self.assertIn("damage_range", step.bypassed_gates)
        self.assertIn("healing", step.bypassed_gates)
        self.assertEqual(step.diagnostics.damage_min, -0.2)
        self.assertLess(result.final_state.damage[0], 0.0)
        self.assertEqual(result.final_state.damage[0], step.fixed_history_result.damage[0])
        np.testing.assert_array_equal(
            result.final_state.damage, step.fixed_history_result.damage
        )
        expected = tuple(
            name
            for name in baseline.WARNING_GATE_NAMES
            if not step.diagnostics.gate_passes[name]
        )
        self.assertEqual(step.bypassed_gates, expected)
        # Healing is bypassed as a gate only; the raw value is retained.
        self.assertLess(step.diagnostics.healing_min, 0.0)

    def test_warning_mode_rejects_incomplete_candidates_via_cutback(self) -> None:
        original = baseline.solve_fixed_damage_mechanics
        calls = 0

        def fail_first_attempt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("manufactured warning mechanics failure")
            return original(*args, **kwargs)

        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.0025,
            max_cutbacks=2,
        )
        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=fail_first_attempt
        ):
            result = self._run_warning(config)
        rejected = result.attempts[0]
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.failure_stage, "mechanics_before")
        self.assertIsNone(rejected.acceptance_status)
        self.assertFalse(rejected.sweep_complete)
        self.assertEqual(rejected.bypassed_gates, ())
        self.assertEqual(
            rejected.committed_before_digest, rejected.committed_after_digest
        )
        accepted_attempt = result.attempts[1]
        self.assertTrue(accepted_attempt.accepted)
        self.assertTrue(accepted_attempt.sweep_complete)
        self.assertEqual(result.accepted_steps[0].cutback_count, 1)

    def test_warning_mode_typed_failure_at_minimum_never_warns(self) -> None:
        def always_fail(*args, **kwargs):
            raise RuntimeError("manufactured persistent mechanics failure")

        config = _config(
            load_final=0.005,
            du_initial=0.005,
            du_min=0.005,
            max_cutbacks=1,
        )
        with mock.patch.object(
            baseline, "solve_fixed_damage_mechanics", side_effect=always_fail
        ):
            with self.assertRaises(baseline.ContinuationFailure) as context:
                self._run_warning(config)
        failure = context.exception
        self.assertIn("warning continuation failed", failure.message)
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        for attempt in failure.attempts:
            self.assertFalse(attempt.accepted)
            self.assertIsNone(attempt.acceptance_status)
            self.assertEqual(attempt.failure_stage, "mechanics_before")

    def test_warning_mode_structural_error_propagates(self) -> None:
        with mock.patch.object(
            baseline,
            "solve_fixed_history_damage",
            side_effect=ValueError("manufactured structural warning error"),
        ):
            with self.assertRaisesRegex(ValueError, "structural warning"):
                self._run_warning(
                    _config(load_final=0.005, du_initial=0.005, du_min=0.0025)
                )

    def test_strict_mode_rejects_the_warning_configuration(self) -> None:
        config = self._forced_force_moment_config(
            load_final=0.005,
            du=0.005,
            max_stagger_iterations=2,
        )
        with self.assertRaises(baseline.ContinuationFailure) as context:
            self._run_strict(config)
        failure = context.exception
        self.assertIn("strict continuation failed", failure.message)
        for attempt in failure.attempts:
            self.assertFalse(attempt.accepted)
            self.assertEqual(attempt.failure_stage, "gates")
            self.assertIsNone(attempt.acceptance_status)
            diagnostics = attempt.diagnostics
            self.assertIsNotNone(diagnostics)
            assert diagnostics is not None
            self.assertTrue(diagnostics.complete_sweep)
            self.assertTrue(attempt.sweep_complete)
            self.assertIn("force", diagnostics.rejection_reason)

    def test_strict_clean_path_statuses_are_unchanged(self) -> None:
        result = self._run_strict(
            _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        )
        self.assertEqual(result.metadata.acceptance_mode, "strict")
        for step in result.accepted_steps:
            self.assertEqual(step.acceptance_status, "strict_passed")
            self.assertTrue(step.diagnostics.passed)
            self.assertTrue(step.diagnostics.complete_sweep)
            self.assertEqual(step.bypassed_gates, ())
        for attempt in result.attempts:
            self.assertEqual(attempt.acceptance_status, "strict_passed")
            self.assertEqual(attempt.bypassed_gates, ())

    def test_warning_clean_path_matches_strict_states(self) -> None:
        config = _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        strict = self._run_strict(config)
        warning = self._run_warning(config)
        self.assertEqual(warning.metadata.acceptance_mode, "warning")
        self.assertEqual(
            [state.u_bar for state in warning.states],
            [state.u_bar for state in strict.states],
        )
        # All gates passed before the budget, so the warning controller
        # accepts early with strict-passed semantics and no bypassed gates.
        for step in warning.accepted_steps:
            self.assertEqual(step.acceptance_status, "strict_passed")
            self.assertTrue(step.diagnostics.passed)
            self.assertEqual(step.bypassed_gates, ())
        for actual, expected in zip(
            warning.states, strict.states, strict=True
        ):
            np.testing.assert_array_equal(actual.displacement, expected.displacement)
            np.testing.assert_array_equal(actual.damage, expected.damage)
            np.testing.assert_array_equal(actual.history, expected.history)

    def test_warning_cpu_replay_adapter_matches_cpu_warning(self) -> None:
        config = self._forced_force_moment_config(max_stagger_iterations=3)
        cpu_result = self._run_warning(config)
        jax_result = self._run_warning(
            config, mechanics_solver=self._cpu_replay_adapter()
        )
        self.assertEqual(
            jax_result.metadata.mechanics_backend, "jax_cpu_cpu_replay"
        )
        self.assertEqual(
            [step.acceptance_status for step in jax_result.accepted_steps],
            [step.acceptance_status for step in cpu_result.accepted_steps],
        )
        for jax_step, cpu_step in zip(
            jax_result.accepted_steps, cpu_result.accepted_steps, strict=True
        ):
            self.assertEqual(jax_step.bypassed_gates, cpu_step.bypassed_gates)
            self.assertEqual(jax_step.stagger_iterations, cpu_step.stagger_iterations)
            np.testing.assert_allclose(
                jax_step.state.displacement,
                cpu_step.state.displacement,
                rtol=0.0,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                jax_step.state.damage,
                cpu_step.state.damage,
                rtol=0.0,
                atol=1.0e-12,
            )
        self.assertTrue(
            all(step.diagnostics.complete_sweep for step in jax_result.accepted_steps)
        )

    @unittest.skipUnless(
        any(device.platform == "gpu" for device in jax.devices()),
        "GPU warning CPU-replay adapter test requires a visible GPU",
    )
    def test_warning_jax_gpu_adapter_runs_cpu_authoritative(self) -> None:
        gpu_backend = JaxBondAssociatedMechanics(
            self.mechanics,
            self.constraints,
            next(device for device in jax.devices() if device.platform == "gpu"),
        )
        adapter = JaxCPUReplayMechanicsAdapter(gpu_backend)
        config = self._forced_force_moment_config(max_stagger_iterations=2)
        result = self._run_warning(config, mechanics_solver=adapter)
        self.assertEqual(result.final_state.u_bar, 0.005)
        self.assertEqual(result.metadata.mechanics_backend, "jax_gpu_cpu_replay")
        for step in result.accepted_steps:
            self.assertEqual(step.acceptance_status, "warning_accepted")
            self.assertFalse(step.diagnostics.passed)
            self.assertIsInstance(
                step.mechanics_result, baseline.MechanicsSolveResult
            )

    def test_forged_warning_metadata_and_statuses_are_rejected(self) -> None:
        config = self._forced_force_moment_config()
        result = self._run_warning(config)
        step = result.accepted_steps[0]
        # A warning-accepted step cannot be relabeled strict metadata.
        strict_metadata = replace(result.metadata, acceptance_mode="strict")
        with self.assertRaisesRegex(ValueError, "strict metadata"):
            replace(result, metadata=strict_metadata)
        # The bypass list must stay exact, ordered, and nonempty.
        with self.assertRaisesRegex(ValueError, "exactly match"):
            replace(step, bypassed_gates=("displacement", "force", "moment"))
        with self.assertRaisesRegex(ValueError, "bypassed_gates"):
            replace(step, bypassed_gates=())
        with self.assertRaisesRegex(ValueError, "unknown|not a canonical"):
            replace(step, bypassed_gates=("not_a_gate",))
        with self.assertRaisesRegex(ValueError, "in order"):
            replace(step, bypassed_gates=("moment", "force"))
        # Status flipping must stay consistent with the diagnostics.
        with self.assertRaisesRegex(ValueError, "passed sweep"):
            replace(step, acceptance_status="strict_passed")
        bad_diagnostics = replace(step.diagnostics, passed=True)
        with self.assertRaisesRegex(ValueError, "retain passed=False"):
            replace(step, diagnostics=bad_diagnostics)
        # An accepted attempt cannot carry a mismatched status.
        with self.assertRaisesRegex(ValueError, "carry passing diagnostics"):
            replace(
                result.attempts[0],
                acceptance_status="strict_passed",
            )
        # Strict results cannot contain warning-accepted steps.
        strict_result = self._run_strict(
            _config(load_final=0.01, du_initial=0.01, du_min=0.01)
        )
        strict_step = strict_result.accepted_steps[0]
        with self.assertRaisesRegex(ValueError, "retain passed=False"):
            replace(strict_step, acceptance_status="warning_accepted")
        with self.assertRaisesRegex(ValueError, "cannot bypass"):
            replace(strict_step, bypassed_gates=("force",))
        strict_attempt = strict_result.attempts[0]
        with self.assertRaisesRegex(ValueError, "passed=False"):
            replace(strict_attempt, acceptance_status="warning_accepted")

    def test_warning_archive_reload_and_provenance(self) -> None:
        config = self._forced_force_moment_config()
        result = self._run_warning(config)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/warning-continuation.npz"
            baseline.write_continuation_archive(
                path, result, self._archive_metadata()
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(int(archive["archive_schema_version"]), 2)
                self.assertEqual(str(archive["acceptance_mode"]), "warning")
                np.testing.assert_array_equal(
                    archive["gate_names"],
                    np.asarray(baseline.WARNING_GATE_NAMES, dtype=np.str_),
                )
                np.testing.assert_array_equal(
                    archive["state_acceptance_status"],
                    np.asarray(["initial", "warning_accepted"], dtype=np.str_),
                )
                np.testing.assert_array_equal(
                    archive["accepted_acceptance_status"],
                    np.asarray(["warning_accepted"], dtype=np.str_),
                )
                np.testing.assert_array_equal(
                    archive["accepted_warning"], np.asarray([True])
                )
                expected_mask = np.zeros((1, len(baseline.WARNING_GATE_NAMES)), dtype=bool)
                expected_mask[
                    0, baseline.WARNING_GATE_NAMES.index("force")
                ] = True
                expected_mask[
                    0, baseline.WARNING_GATE_NAMES.index("moment")
                ] = True
                np.testing.assert_array_equal(
                    archive["accepted_bypassed_gate_mask"], expected_mask
                )
                # Existing gate pass arrays stay false for warning states.
                self.assertFalse(bool(archive["accepted_force_pass"][0]))
                self.assertFalse(bool(archive["accepted_moment_pass"][0]))
                self.assertFalse(bool(archive["accepted_gate_pass"][0]))
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")
                audit = json.loads(str(archive["attempt_audit_json"]))
                self.assertEqual(len(audit), 1)
                self.assertEqual(audit[0]["acceptance_status"], "warning_accepted")
                self.assertTrue(audit[0]["sweep_complete"])
                self.assertEqual(audit[0]["bypassed_gates"], ["force", "moment"])
                accepted_audit = json.loads(str(archive["accepted_audit_json"]))
                self.assertEqual(
                    accepted_audit[0]["acceptance_status"], "warning_accepted"
                )
                self.assertEqual(
                    accepted_audit[0]["bypassed_gates"], ["force", "moment"]
                )
                metadata_json = json.loads(str(archive["metadata_json"]))
                self.assertEqual(metadata_json["acceptance_mode"], "warning")
                self.assertEqual(
                    metadata_json["archive_schema_version"], 2
                )
                self.assertEqual(
                    metadata_json["gate_names"], list(baseline.WARNING_GATE_NAMES)
                )
                self.assertTrue(
                    audit[0]["diagnostics"]["complete_sweep"]
                )

        strict_result = self._run_strict(
            _config(load_final=0.005, du_initial=0.005, du_min=0.005)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/strict-continuation.npz"
            baseline.write_continuation_archive(
                path, strict_result, self._archive_metadata()
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(int(archive["archive_schema_version"]), 2)
                self.assertEqual(str(archive["acceptance_mode"]), "strict")
                np.testing.assert_array_equal(
                    archive["state_acceptance_status"],
                    np.asarray(["initial", "strict_passed"], dtype=np.str_),
                )
                np.testing.assert_array_equal(
                    archive["accepted_warning"], np.asarray([False])
                )
                np.testing.assert_array_equal(
                    archive["accepted_bypassed_gate_mask"],
                    np.zeros((1, len(baseline.WARNING_GATE_NAMES)), dtype=bool),
                )
                audit = json.loads(str(archive["attempt_audit_json"]))
                self.assertEqual(audit[0]["acceptance_status"], "strict_passed")
                self.assertTrue(audit[0]["sweep_complete"])
                self.assertEqual(audit[0]["bypassed_gates"], [])

    def test_forged_final_sweep_gate_summaries_are_rejected(self) -> None:
        config = self._forced_force_moment_config()
        result = self._run_warning(config)
        step = result.accepted_steps[0]
        attempt = result.attempts[0]

        # A coherently forged force pass with a matching shrunken bypass list
        # is self-consistent at the step level but contradicts the
        # authoritative final-sweep replay.
        forged_diag = replace(step.diagnostics, force_pass=True)
        forged_step = replace(
            step, diagnostics=forged_diag, bypassed_gates=("moment",)
        )
        forged_attempt = replace(
            attempt, diagnostics=forged_diag, bypassed_gates=("moment",)
        )
        with self.assertRaisesRegex(
            ValueError, "force_pass.*authoritative final-sweep replay"
        ):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        # A forged volume-weighted used residual is caught by the replay even
        # though every other stored summary is genuine.
        bad_volume = np.array(step.diagnostics.rho_used_volume, copy=True)
        bad_volume[0] += 0.1
        forged_diag = replace(step.diagnostics, rho_used_volume=bad_volume)
        forged_step = replace(step, diagnostics=forged_diag)
        forged_attempt = replace(attempt, diagnostics=forged_diag)
        with self.assertRaisesRegex(ValueError, "rho_used_volume"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        # A forged H_used that is internally consistent with the stored
        # fixed-history result still contradicts the replayed mechanics-before
        # Amor driver.
        bad_history = np.array(step.history_used, copy=True)
        bad_history[3] += 0.05
        bad_fixed = replace(step.fixed_history_result, history=bad_history)
        forged_step = replace(
            step, history_used=bad_history, fixed_history_result=bad_fixed
        )
        forged_attempt = replace(attempt, diagnostics=forged_step.diagnostics)
        with self.assertRaisesRegex(ValueError, "H_used"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        # Forging the stored before-force coherently with the diagnostics is
        # caught by the authoritative mechanics-before validation.
        bad_force = np.array(step.mechanics_before_result.total_force, copy=True)
        bad_force[0] += 1.0
        bad_before = replace(step.mechanics_before_result, total_force=bad_force)
        forged_diag = replace(
            step.diagnostics, force_before=bad_force, force_error=1.0
        )
        forged_step = replace(
            step,
            mechanics_before_result=bad_before,
            diagnostics=forged_diag,
        )
        forged_attempt = replace(attempt, diagnostics=forged_diag)
        with self.assertRaisesRegex(ValueError, "total force"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

    def test_strict_replay_rejects_forged_before_summaries(self) -> None:
        result = self._run_strict(
            _config(load_final=0.005, du_initial=0.005, du_min=0.005)
        )
        step = result.accepted_steps[0]
        attempt = result.attempts[0]
        self.assertEqual(step.acceptance_status, "strict_passed")
        # Strict steps also carry authoritative final-sweep before data.
        self.assertEqual(step.mechanics_before_result.u_bar, step.load_target)
        self.assertFalse(step.damage_before.flags.writeable)
        prescribed = self.constraints.prescribed_vector(step.load_target)
        np.testing.assert_array_equal(
            step.mechanics_before_result.displacement[
                self.constraints.constrained_dofs
            ],
            prescribed[self.constraints.constrained_dofs],
        )

        bad_volume = np.array(step.diagnostics.rho_used_volume, copy=True)
        bad_volume[0] += 0.1
        forged_diag = replace(step.diagnostics, rho_used_volume=bad_volume)
        forged_step = replace(step, diagnostics=forged_diag)
        forged_attempt = replace(attempt, diagnostics=forged_diag)
        with self.assertRaisesRegex(ValueError, "rho_used_volume"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

        bad_e_d = step.diagnostics.e_d + 1.0
        forged_diag = replace(step.diagnostics, e_d=bad_e_d)
        with self.assertRaisesRegex(ValueError, "damage error"):
            replace(step, diagnostics=forged_diag)

    def test_complete_sweep_diagnostics_must_be_finite(self) -> None:
        config = self._forced_force_moment_config()
        result = self._run_warning(config)
        step = result.accepted_steps[0]
        for name in ("e_u", "e_H", "e_R", "e_rho", "force_error", "moment_error"):
            with self.subTest(scalar=name):
                with self.assertRaisesRegex(ValueError, "finite for a complete sweep"):
                    replace(step.diagnostics, **{name: np.inf})
        with self.assertRaisesRegex(ValueError, "must not be NaN"):
            replace(step.diagnostics, e_u=np.nan)
        # Incomplete failure diagnostics retain their inf sentinels, but they
        # can never be paired with an accepted step, so an inf sentinel can
        # never reach an archived accepted state.
        incomplete = replace(
            step.diagnostics, complete_sweep=False, e_u=np.inf
        )
        self.assertTrue(np.isinf(incomplete.e_u))
        with self.assertRaisesRegex(ValueError, "complete stagger sweep"):
            replace(step, diagnostics=incomplete)

        corrupted = replace(step.diagnostics)
        object.__setattr__(corrupted, "force_error", np.inf)
        forged_step = replace(step, diagnostics=corrupted)
        forged_attempt = replace(result.attempts[0], diagnostics=corrupted)
        with self.assertRaisesRegex(ValueError, "finite values"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

    def test_final_sweep_before_data_and_archive_reconstruction_fields(self) -> None:
        config = self._forced_force_moment_config(
            load_final=0.01, du=0.005, max_stagger_iterations=2
        )
        result = self._run_warning(config)
        for index, step in enumerate(result.accepted_steps, start=1):
            self.assertFalse(step.damage_before.flags.writeable)
            with self.assertRaises(ValueError):
                step.damage_before.setflags(write=True)
            self.assertEqual(
                step.mechanics_before_result.u_bar, step.load_target
            )
            np.testing.assert_array_equal(
                step.diagnostics.force_before,
                step.mechanics_before_result.total_force,
            )
            self.assertEqual(
                step.diagnostics.e_d,
                float(np.max(np.abs(step.state.damage - step.damage_before))),
            )
            if step.stagger_iterations == 1:
                np.testing.assert_array_equal(
                    step.damage_before, result.states[index - 1].damage
                )

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/warning-before.npz"
            baseline.write_continuation_archive(
                path, result, self._archive_metadata()
            )
            with np.load(path, allow_pickle=False) as archive:
                expected_damage_before = np.stack(
                    [step.damage_before for step in result.accepted_steps]
                )
                expected_before_u = np.stack(
                    [
                        step.mechanics_before_result.displacement
                        for step in result.accepted_steps
                    ]
                )
                expected_before_free = np.stack(
                    [
                        step.mechanics_before_result.free_residual
                        for step in result.accepted_steps
                    ]
                )
                np.testing.assert_array_equal(
                    archive["accepted_damage_before"], expected_damage_before
                )
                np.testing.assert_array_equal(
                    archive["accepted_mechanics_before_displacement"],
                    expected_before_u,
                )

                np.testing.assert_array_equal(
                    archive["accepted_mechanics_before_free_residual"],
                    expected_before_free,
                )
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")
                # The archived before data independently reconstructs the
                # force conservation gate inputs.
                u_before = archive["accepted_mechanics_before_displacement"][0]
                u_after = archive["accepted_mechanics_displacement"][0]
                self.assertGreater(
                    float(np.max(np.abs(u_after - u_before))), 0.0
                )

        strict_result = self._run_strict(
            _config(load_final=0.005, du_initial=0.005, du_min=0.005)
        )
        strict_step = strict_result.accepted_steps[0]
        if strict_step.stagger_iterations == 1:
            np.testing.assert_array_equal(
                strict_step.damage_before,
                strict_result.states[0].damage,
            )
        prescribed = self.constraints.prescribed_vector(strict_step.load_target)
        np.testing.assert_array_equal(
            strict_step.mechanics_before_result.displacement[
                self.constraints.constrained_dofs
            ],
            prescribed[self.constraints.constrained_dofs],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/strict-before.npz"
            baseline.write_continuation_archive(
                path, strict_result, self._archive_metadata()
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("accepted_damage_before", archive.files)
                self.assertIn(
                    "accepted_mechanics_before_displacement", archive.files
                )
                self.assertIn(
                    "accepted_mechanics_before_free_residual", archive.files
                )
                np.testing.assert_array_equal(
                    archive["accepted_damage_before"],
                    np.stack(
                        [s.damage_before for s in strict_result.accepted_steps]
                    ),
                )

    def test_mechanics_backend_provenance_is_replay_anchored(self) -> None:
        result = self._run_strict(
            _config(load_final=0.005, du_initial=0.005, du_min=0.005)
        )
        self.assertEqual(result.metadata.mechanics_backend, "numpy_scipy")
        forged_metadata = replace(
            result.metadata, mechanics_backend="jax_cpu_cpu_replay"
        )
        with self.assertRaisesRegex(ValueError, "backend does not match metadata"):
            replace(result, metadata=forged_metadata)

        forged_after = replace(result.accepted_steps[0].mechanics_result)
        forged_before = replace(
            result.accepted_steps[0].mechanics_before_result
        )
        object.__setattr__(
            forged_after, "mechanics_backend", "jax_cpu_cpu_replay"
        )
        object.__setattr__(
            forged_before, "mechanics_backend", "jax_cpu_cpu_replay"
        )
        forged_step = replace(
            result.accepted_steps[0],
            mechanics_result=forged_after,
            mechanics_before_result=forged_before,
        )
        forged_attempt = replace(
            result.attempts[0], diagnostics=forged_step.diagnostics
        )
        with self.assertRaisesRegex(ValueError, "backend does not match metadata"):
            replace(
                result,
                metadata=forged_metadata,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

    def test_post_construction_diagnostic_dtype_corruption_is_rejected(self) -> None:
        result = self._run_warning(self._forced_force_moment_config())
        step = result.accepted_steps[0]
        corrupted = replace(step.diagnostics)
        object.__setattr__(
            corrupted,
            "force_before",
            np.asarray(corrupted.force_before, dtype=np.float32),
        )
        forged_step = replace(step, diagnostics=corrupted)
        forged_attempt = replace(result.attempts[0], diagnostics=corrupted)
        with self.assertRaisesRegex(ValueError, "dtype float32"):
            replace(
                result,
                accepted_steps=(forged_step,),
                attempts=(forged_attempt,),
            )

    def test_warning_gate_name_contract(self) -> None:
        self.assertEqual(
            baseline.WARNING_GATE_NAMES,
            (
                "displacement",
                "damage",
                "history",
                "reaction",
                "damage_residual",
                "damage_range",
                "healing",
                "history_monotonic",
                "force",
                "moment",
            ),
        )
        self.assertEqual(baseline.ACCEPTANCE_MODES, ("strict", "warning"))
        self.assertEqual(
            baseline.ACCEPTANCE_STATUSES,
            ("strict_passed", "warning_accepted"),
        )
        self.assertEqual(baseline.ARCHIVE_SCHEMA_VERSION, 2)
        strict = self._run_strict(_config(load_final=0.005, du_initial=0.005, du_min=0.005))
        with self.assertRaisesRegex(ValueError, "acceptance mode"):
            replace(strict.metadata, acceptance_mode="permissive")
        with self.assertRaisesRegex(ValueError, "acceptance mode"):
            replace(strict.metadata, acceptance_mode="")


if __name__ == "__main__":
    unittest.main()
