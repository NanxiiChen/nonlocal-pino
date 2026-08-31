"""Focused contract tests for the direct engineering-grip baseline."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import jax
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

from examples.fracture2d.nonlocal_grip_diagnostic import (
    build_union_geometry_from_arrays,
    build_volume_grip_constraints,
)
from examples.fracture2d import nonlocal_engineering_grip as engineering


def _material_cloud():
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
    delta = 0.48
    pairs: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
            if 0.0 < distance < delta:
                pairs.extend(((first, second), (second, first)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    return coordinates, triangles_array, volumes, bottom, top, edges, delta


class EngineeringGripBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        coordinates, triangles, volumes, bottom, top, edges, delta = _material_cloud()
        cls.geometry = build_union_geometry_from_arrays(
            coordinates,
            triangles,
            volumes,
            bottom,
            top,
            0,
            edges,
            delta,
            0.2,
            0.0,
        )
        cls.model = engineering.build_engineering_grip_model(
            cls.geometry,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
            1.7,
            0.65,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    def _write_valid_archive(self, directory: str, name: str = "accepted.npz") -> Path:
        path = Path(directory) / name
        with mock.patch.object(engineering, "_localization_pass", return_value=True):
            result = engineering.run_strict_engineering_grip(
                self.model, engineering.EngineeringGripConfig.smoke()
            )
            engineering.write_engineering_grip_archive(path, result)
        return path

    def _warning_result(self) -> engineering.EngineeringGripResult:
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            max_stagger_iterations=1,
            damage=engineering.GateTolerance(1.0e-20, 0.0, 1.0),
        )
        return engineering.run_warning_engineering_grip(
            self.model, config, 1.0e-3
        )

    def _forced_damage_solver(
        self, values: tuple[float, ...]
    ) -> tuple[Callable[..., Any], list[int]]:
        original = engineering.solve_fixed_history_damage
        calls = [0]

        def solve(*args: Any, **kwargs: Any) -> Any:
            if calls[0] >= len(values):
                raise AssertionError("manufactured damage solver was called too often")
            result = original(*args, **kwargs)
            damage = np.array(result.damage, copy=True)
            damage[0] = values[calls[0]]
            history = np.asarray(args[3], dtype=np.float64)
            rho_used, _, rho_used_volume, _, _ = engineering._damage_residuals(
                self.model, damage, history, history
            )
            calls[0] += 1
            return replace(
                result,
                damage=damage,
                strong_residual=rho_used,
                euclidean_residual=rho_used_volume,
            )

        return solve, calls

    def _forced_replay_damage(
        self, values: tuple[float, ...]
    ) -> tuple[Callable[..., Any], list[int]]:
        original = engineering._replay_damage
        calls = [0]

        def replay(*args: Any, **kwargs: Any) -> Any:
            if calls[0] >= len(values):
                raise AssertionError("manufactured replay damage solver was called too often")
            history, coordinates, volumes, edges, delta, Gc, l0 = args
            damage, _, _, matrix, scale = original(*args, **kwargs)
            damage = np.array(damage, copy=True)
            damage[0] = values[calls[0]]
            history = np.asarray(history, dtype=np.float64)
            rho = (
                Gc / l0 * damage
                + Gc * l0 * (matrix @ damage)
                + 2.0 * history * damage
                - 2.0 * history
            )
            calls[0] += 1
            return (
                damage,
                np.asarray(rho, dtype=np.float64),
                np.asarray(volumes * rho, dtype=np.float64),
                matrix,
                scale,
            )

        return replay, calls

    def test_volume_grip_masks_and_component_ownership(self) -> None:
        constraints = build_volume_grip_constraints(self.geometry, 0)
        coordinates = self.geometry.coordinates
        bottom_particles = np.flatnonzero(constraints.outer_bottom_mask)
        top_particles = np.flatnonzero(constraints.outer_top_mask)
        self.assertTrue(np.all(coordinates[bottom_particles, 1] < 0.0))
        self.assertTrue(np.all(coordinates[top_particles, 1] > 1.0))
        interface_dofs = np.concatenate(
            (
                2 * self.geometry.material_bottom_indices + 1,
                2 * self.geometry.material_top_indices + 1,
            )
        )
        self.assertFalse(np.any(np.isin(interface_dofs, constraints.constrained_dofs)))
        expected_ux = np.asarray([0], dtype=np.int64)
        constrained_ux = constraints.constrained_dofs[constraints.constrained_dofs % 2 == 0]
        np.testing.assert_array_equal(constrained_ux, expected_ux)
        all_grip_y = np.concatenate(
            (2 * bottom_particles + 1, 2 * top_particles + 1)
        )
        np.testing.assert_array_equal(
            np.sort(all_grip_y),
            np.sort(constraints.constrained_dofs[constraints.constrained_dofs % 2 == 1]),
        )

    def test_static_identity_and_material_only_damage_operator(self) -> None:
        second = engineering.build_engineering_grip_model(
            self.geometry,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
            1.7,
            0.65,
        )
        self.assertEqual(self.model.static_digest, second.static_digest)
        self.assertIsNot(self.model.static.runtime_identity, second.static.runtime_identity)
        self.assertEqual(self.model.material_operator.shape, (25, 25))
        self.assertEqual(self.model.static.material_count, 25)
        np.testing.assert_array_equal(
            self.model.static.union_material_phase_volumes[:25],
            self.geometry.material_reference_volumes,
        )
        np.testing.assert_array_equal(
            self.model.static.union_material_phase_volumes[25:], 0.0
        )
        self.assertFalse(self.model.static.coordinates.flags.writeable)
        with self.assertRaises(ValueError):
            self.model.static.coordinates.setflags(write=True)

    def test_energy_tangent_and_direct_top_volume_reaction(self) -> None:
        rng = np.random.default_rng(81)
        damage = np.asarray(
            np.linspace(0.05, 0.35, self.model.n_material), dtype=np.float64
        )
        view = self.model.fixed_damage_view(damage)
        q = np.asarray(1.0e-2 * rng.normal(size=self.model.n_dofs), dtype=np.float64)
        direction = np.asarray(rng.normal(size=self.model.n_dofs), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        epsilon = 2.0e-6
        energy_fd = (
            view.energy(q + epsilon * direction)
            - view.energy(q - epsilon * direction)
        ) / (2.0 * epsilon)
        self.assertLess(
            abs(energy_fd - float(view.residual(q) @ direction))
            / max(abs(energy_fd), 1.0),
            3.0e-10,
        )
        residual_fd = (
            view.residual(q + epsilon * direction)
            - view.residual(q - epsilon * direction)
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            residual_fd, view.tangent_action(direction), rtol=0.0, atol=3.0e-9
        )
        result = engineering.solve_fixed_outer_displacement(
            view,
            self.model.constraints,
            0.08,
            1.0e-10,
            1.0e-13,
            3000,
        )
        self.assertEqual(result.U_ext, 0.08)
        self.assertAlmostEqual(
            result.outer_reaction,
            float(self.model.constraints.outer_top_selector @ result.residual),
            places=14,
        )
        np.testing.assert_allclose(result.total_force, 0.0, rtol=0.0, atol=2.0e-10)
        self.assertLessEqual(abs(result.total_moment), 2.0e-10)

    def test_independent_vectorized_replay_matches_production_primitives(self) -> None:
        ba = engineering._replay_ba(
            self.geometry.coordinates,
            self.geometry.union_volumes,
            self.geometry.directed_edges,
            self.geometry.delta,
            self.model.static.rank_rtol,
        )
        rng = np.random.default_rng(181)
        damage = np.asarray(
            np.linspace(0.05, 0.35, self.model.n_material), dtype=np.float64
        )
        q = np.asarray(rng.normal(size=self.model.n_dofs) * 1.0e-2, dtype=np.float64)
        direction = np.asarray(rng.normal(size=self.model.n_dofs), dtype=np.float64)
        center = engineering._replay_center_weights(
            damage,
            self.geometry.material_volumes,
            self.geometry.grip_volumes,
            self.model.static.k_res,
            self.model.n_material,
        )
        replay = engineering._replay_response(
            q,
            damage,
            ba,
            center,
            self.model.static.youngs_modulus,
            self.model.static.poisson_ratio,
            self.model.n_material,
        )
        production = self.model.fixed_damage_view(damage).evaluate(q)
        self.assertEqual(replay["energy"], production.energy)
        np.testing.assert_array_equal(replay["residual"], production.residual)
        np.testing.assert_array_equal(
            replay["edge_f"], production.edge_deformation_gradient
        )
        np.testing.assert_array_equal(replay["edge_strain"], production.edge_strain)
        np.testing.assert_array_equal(
            replay["particle_strain"], production.particle_strain
        )
        np.testing.assert_array_equal(
            engineering._replay_tangent_action(
                direction,
                ba,
                center,
                self.model.static.youngs_modulus,
                self.model.static.poisson_ratio,
            ),
            self.model.fixed_damage_view(damage).tangent_action(direction),
        )

    def test_replay_damage_is_sparse_and_matches_material_solver(self) -> None:
        H_used = np.asarray(
            np.linspace(1.0e-3, 0.2, self.model.n_material), dtype=np.float64
        )
        damage, rho, volume_rho, matrix, _ = engineering._replay_damage(
            H_used,
            self.geometry.material_reference_coordinates,
            self.geometry.material_reference_volumes,
            self.geometry.material_reference_edges,
            self.geometry.delta,
            self.model.Gc,
            self.model.l0,
        )
        self.assertTrue(issparse(matrix))
        self.assertEqual(matrix.shape, (self.model.n_material, self.model.n_material))
        production = engineering.solve_fixed_history_damage(
            self.model.material_operator, self.model.Gc, self.model.l0, H_used
        )
        np.testing.assert_allclose(damage, production.damage, rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(
            rho, production.strong_residual, rtol=0.0, atol=2.0e-14
        )
        np.testing.assert_allclose(
            volume_rho, production.euclidean_residual, rtol=0.0, atol=2.0e-14
        )

    def test_replay_direct_path_has_no_dense_global_solve(self) -> None:
        ba = engineering._replay_ba(
            self.geometry.coordinates,
            self.geometry.union_volumes,
            self.geometry.directed_edges,
            self.geometry.delta,
            self.model.static.rank_rtol,
        )
        direct_source = inspect.getsource(engineering._replay_direct_solve)
        damage_source = inspect.getsource(engineering._replay_damage)
        for source in (direct_source, damage_source):
            self.assertNotIn("np.linalg.solve", source)
        self.assertNotIn("np.eye(n_dofs", direct_source)
        self.assertNotIn("np.zeros((n, n)", damage_source)
        with mock.patch.object(
            engineering.np.linalg,
            "solve",
            side_effect=AssertionError("dense replay solve is forbidden"),
        ):
            direct = engineering._replay_direct_solve(
                np.zeros(self.model.n_material, dtype=np.float64),
                0.08,
                ba,
                self.geometry.material_volumes,
                self.geometry.grip_volumes,
                self.model.constraints.outer_bottom_mask,
                self.model.constraints.outer_top_mask,
                0,
                self.model.static.youngs_modulus,
                self.model.static.poisson_ratio,
                self.model.static.k_res,
                self.model.n_material,
                cg_rtol=1.0e-10,
                cg_atol=1.0e-13,
                cg_maxiter=3000,
            )
            _, _, _, matrix, _ = engineering._replay_damage(
                np.linspace(1.0e-3, 0.2, self.model.n_material, dtype=np.float64),
                self.geometry.material_reference_coordinates,
                self.geometry.material_reference_volumes,
                self.geometry.material_reference_edges,
                self.geometry.delta,
                self.model.Gc,
                self.model.l0,
            )
        self.assertIsInstance(direct["tangent"], LinearOperator)
        self.assertEqual(direct["tangent"].shape, (self.model.n_dofs, self.model.n_dofs))
        self.assertTrue(issparse(matrix))
        self.assertEqual(direct["cg_info"], 0)
        self.assertLess(direct["free_residual_norm"], direct["cg_convergence_limit"])

    def test_exact_zero_replay_is_analytic(self) -> None:
        ba = engineering._replay_ba(
            self.geometry.coordinates,
            self.geometry.union_volumes,
            self.geometry.directed_edges,
            self.geometry.delta,
            self.model.static.rank_rtol,
        )
        zero_q = np.zeros(self.model.n_dofs, dtype=np.float64)
        zero_damage = np.zeros(self.model.n_material, dtype=np.float64)
        center = engineering._replay_center_weights(
            zero_damage,
            self.geometry.material_volumes,
            self.geometry.grip_volumes,
            self.model.static.k_res,
            self.model.n_material,
        )
        response = engineering._replay_response(
            zero_q,
            zero_damage,
            ba,
            center,
            self.model.static.youngs_modulus,
            self.model.static.poisson_ratio,
            self.model.n_material,
        )
        self.assertEqual(response["energy"], 0.0)
        self.assertEqual(response["energy_constituent_scale"], 0.0)
        self.assertEqual(response["residual"].max(), 0.0)
        self.assertEqual(response["edge_strain"].max(), 0.0)
        driver, trial, strain = engineering._replay_amor(
            zero_q,
            np.linspace(0.0, 0.2, self.model.n_material, dtype=np.float64),
            ba,
            self.model.static.youngs_modulus,
            self.model.static.poisson_ratio,
            self.model.n_material,
        )
        self.assertEqual(driver.max(), 0.0)
        np.testing.assert_array_equal(
            trial, np.linspace(0.0, 0.2, self.model.n_material, dtype=np.float64)
        )
        self.assertEqual(strain.max(), 0.0)

    def test_material_source_amor_includes_grip_targets_but_no_grip_history(self) -> None:
        view = self.model.fixed_damage_view(
            np.zeros(self.model.n_material, dtype=np.float64)
        )
        # Use the public helper's real mechanics evaluation for its shape
        # contract, then verify source ownership from the resulting edge data.
        q = np.zeros(self.model.n_dofs, dtype=np.float64)
        result = engineering.compute_material_source_amor(
            view, q, np.zeros(self.model.n_material, dtype=np.float64)
        )
        self.assertEqual(result.particle_driver.shape, (self.model.n_material,))
        self.assertEqual(result.trial_history.shape, (self.model.n_material,))
        retained = view.static.source < self.model.n_material
        self.assertTrue(np.any(view.static.target[retained] >= self.model.n_material))
        expected = np.zeros(self.model.n_material, dtype=np.float64)
        np.add.at(
            expected,
            view.static.source[retained],
            view.static.eta[retained] * result.edge_positive_energy[retained],
        )
        np.testing.assert_allclose(result.particle_driver, expected, rtol=0.0, atol=1.0e-14)
        self.assertEqual(result.particle_driver.size, self.model.n_material)
        self.assertFalse(hasattr(result, "grip_history"))

    def test_one_time_work_check_and_strict_history_order(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        check = engineering.run_zero_damage_work_check(
            self.model,
            config.work_check_load,
            config.work_epsilon,
            config.work_relative_limit,
            config.cg_rtol,
            config.cg_atol,
            config.cg_maxiter,
        )
        self.assertTrue(check.passed)
        self.assertLessEqual(check.relative_error, config.work_relative_limit)
        # This tiny full-square manufactured cloud is intentionally not a
        # localized notch solution.  Bypass only the final localization gate
        # here so the stagger/work contract remains independently exercised;
        # the hard gate itself is tested below and in archive replay.
        with mock.patch.object(engineering, "_localization_pass", return_value=True):
            result = engineering.run_strict_engineering_grip(self.model, config)
        self.assertEqual(len(result.states), 2)
        step = result.accepted_steps[0]
        np.testing.assert_array_equal(step.H_used, np.maximum(step.damage_before * 0.0, step.psi_before))
        np.testing.assert_array_equal(step.H_next, np.maximum(step.damage_before * 0.0, step.psi_after))
        self.assertTrue(all(step.diagnostics.gate_passes.values()))
        self.assertEqual(result.final_state.u_bar, config.load_final)
        floor = float(np.finfo(np.float64).tiny)
        self.assertAlmostEqual(engineering._relative(2.0 * floor, floor), 0.5)

    def test_minimum_increment_roundoff_classification(self) -> None:
        du_min = np.float64(5.0e-5)
        load_start = np.float64(0.0031183145201048547)
        load_target = np.float64(load_start + du_min)
        trial_increment = np.float64(load_target - load_start)
        self.assertGreater(trial_increment, du_min)
        self.assertTrue(
            engineering._is_minimum_increment(
                trial_increment, du_min, load_start, load_target
            )
        )
        self.assertTrue(
            engineering._is_minimum_increment(
                np.nextafter(du_min, np.inf), du_min, load_start, load_target
            )
        )
        self.assertFalse(
            engineering._is_minimum_increment(
                du_min + np.float64(1.0e-8), du_min, load_start, load_target
            )
        )

    def test_minimum_increment_roundoff_does_not_retry_duplicate_target(self) -> None:
        du_min = np.float64(5.0e-5)
        load_start = np.float64(0.0031183145201048547)
        load_target = np.float64(load_start + du_min)
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            load_final=float(load_target),
            du_initial=float(load_start),
            du_min=float(du_min),
        )
        accepted_step = SimpleNamespace(
            state=SimpleNamespace(u_bar=float(load_start)),
            increment=float(du_min),
        )
        accepted_attempt = SimpleNamespace(
            committed_before_digest="0" * 64,
            committed_after_digest="1" * 64,
        )
        rejected_attempt = SimpleNamespace(
            committed_before_digest="0" * 64,
            committed_after_digest="0" * 64,
            failure_reason="manufactured rejection",
        )
        targets: list[float] = []

        def mocked_attempt(*args: Any, **kwargs: Any):
            targets.append(float(args[3]))
            if len(targets) == 1:
                return accepted_step, accepted_attempt
            return None, rejected_attempt

        with mock.patch.object(
            engineering,
            "run_zero_damage_work_check",
            return_value=SimpleNamespace(),
        ), mock.patch.object(
            engineering, "_run_engineering_attempt", side_effect=mocked_attempt
        ):
            with self.assertRaises(engineering.EngineeringGripContinuationFailure):
                engineering.run_strict_engineering_grip(self.model, config)
        self.assertEqual(targets, [float(load_start), float(load_target)])
        self.assertEqual(len(targets), len(set(targets)))

    def test_warning_acceptance_is_final_complete_and_finite_only(self) -> None:
        passes = {
            name: True
            for name in (
                *engineering._WARNING_HARD_GATE_NAMES,
                *engineering.WARNING_GATE_NAMES,
            )
        }
        passes["damage"] = False
        diagnostic = SimpleNamespace(
            complete_sweep=True,
            passed=False,
            d_next=np.asarray([0.25, 0.5], dtype=np.float64),
            gate_passes=passes,
        )

        def changed_diagnostic(**updates: Any) -> SimpleNamespace:
            return SimpleNamespace(**{**vars(diagnostic), **updates})

        accepted, bypassed, range_excess, healing = (
            engineering._warning_acceptance_decision(
                diagnostic,
                np.zeros(2, dtype=np.float64),
                1.0e-3,
            )
        )
        self.assertTrue(accepted)
        self.assertEqual(bypassed, ("damage",))
        self.assertEqual(range_excess, 0.0)
        self.assertEqual(healing, 0.0)

        incomplete = changed_diagnostic(complete_sweep=False)
        self.assertFalse(
            engineering._warning_acceptance_decision(
                incomplete, np.zeros(2, dtype=np.float64), 1.0e-3
            )[0]
        )
        for hard_gate in engineering._WARNING_HARD_GATE_NAMES:
            hard_failed = dict(passes)
            hard_failed[hard_gate] = False
            with self.subTest(hard_gate=hard_gate):
                self.assertFalse(
                    engineering._warning_acceptance_decision(
                        changed_diagnostic(gate_passes=hard_failed),
                        np.zeros(2, dtype=np.float64),
                        1.0e-3,
                    )[0]
                )
        with self.assertRaises(FloatingPointError):
                engineering._warning_acceptance_decision(
                changed_diagnostic(
                    d_next=np.asarray([np.nan, 0.0], dtype=np.float64)
                ),
                np.zeros(2, dtype=np.float64),
                1.0e-3,
            )
        over_limit = changed_diagnostic(
            d_next=np.asarray([1.002, 0.0], dtype=np.float64),
        )
        accepted, bypassed, range_excess, healing = (
            engineering._warning_acceptance_decision(
                over_limit,
                np.zeros(2, dtype=np.float64),
                1.0e-3,
            )
        )
        self.assertFalse(accepted)
        self.assertEqual(bypassed, ())
        self.assertAlmostEqual(range_excess, 2.0e-3)
        self.assertEqual(healing, 0.0)

    def test_warning_intermediate_range_violation_can_recover(self) -> None:
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            max_stagger_iterations=2,
        )
        original_damage = engineering.solve_fixed_history_damage
        original_diagnostics = engineering._full_diagnostics
        calls = 0

        def damaged_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            result = original_damage(*args, **kwargs)
            if calls != 1:
                return result
            damage = np.array(result.damage, copy=True)
            damage[0] = 1.1
            history = np.asarray(args[3], dtype=np.float64)
            rho_used, _, rho_used_volume, _, _ = engineering._damage_residuals(
                self.model, damage, history, history
            )
            return replace(
                result,
                damage=damage,
                strong_residual=rho_used,
                euclidean_residual=rho_used_volume,
            )

        def diagnostics_with_recovery(*args: Any, **kwargs: Any) -> Any:
            result = original_diagnostics(*args, **kwargs)
            if result.sweep_index != 1:
                return result
            return replace(
                result,
                passed=True,
                failure_stage="",
                failure_reason="",
                finite_pass=True,
                mechanics_before_pass=True,
                mechanics_after_pass=True,
                displacement_pass=True,
                damage_pass=True,
                history_pass=True,
                reaction_pass=True,
                damage_residual_pass=True,
                damage_range_pass=True,
                healing_pass=True,
                history_monotonic_pass=True,
                force_pass=True,
                moment_pass=True,
            )

        result: engineering.EngineeringGripResult
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", side_effect=damaged_once
        ), mock.patch.object(
            engineering, "_full_diagnostics", side_effect=diagnostics_with_recovery
        ):
            result = engineering.run_warning_engineering_grip(
                self.model, config, 0.05
            )
        self.assertEqual(calls, 2)
        self.assertEqual(len(result.accepted_steps), 1)
        step = result.accepted_steps[0]
        self.assertEqual(step.stagger_iterations, 2)
        self.assertEqual(step.acceptance_status, "strict_passed")
        self.assertFalse(result.attempts[0].sweeps[0].damage_range_pass)
        self.assertTrue(result.attempts[0].sweeps[-1].passed)
        self.assertTrue(step.diagnostics.damage_range_pass)
        self.assertEqual(result.final_state.u_bar, config.load_final)

    def test_warning_bounded_negative_damage_keeps_strict_flags_and_replays(self) -> None:
        permissive = engineering.GateTolerance(1.0e10, 0.0, 1.0)
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            max_stagger_iterations=1,
            displacement=permissive,
            damage=permissive,
            history=permissive,
            reaction=permissive,
            damage_residual=permissive,
        )
        damage_solver, damage_calls = self._forced_damage_solver((-5.0e-4,))
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", side_effect=damage_solver
        ):
            result = engineering.run_warning_engineering_grip(
                self.model, config, 1.0e-3
            )
        self.assertEqual(damage_calls[0], 1)
        step = result.accepted_steps[0]
        self.assertEqual(step.acceptance_status, "warning_accepted")
        self.assertEqual(step.bypassed_gates, ("damage_range", "healing"))
        self.assertFalse(step.diagnostics.damage_range_pass)
        self.assertFalse(step.diagnostics.healing_pass)
        assert step.warning_range_excess is not None
        assert step.warning_healing_magnitude is not None
        self.assertAlmostEqual(step.warning_range_excess, 5.0e-4)
        self.assertAlmostEqual(step.warning_healing_magnitude, 5.0e-4)
        self.assertAlmostEqual(result.final_state.material_damage[0], -5.0e-4)

        with tempfile.TemporaryDirectory(prefix="engineering-grip-negative-") as directory:
            path = Path(directory) / "negative.npz"
            engineering.write_engineering_grip_archive(path, result)
            payload = engineering.load_engineering_grip_archive(path)
            accepted_diag = json.loads(str(payload["accepted_diagnostics_json"][0]))
            self.assertFalse(accepted_diag["damage_range_pass"])
            self.assertFalse(accepted_diag["healing_pass"])
            self.assertAlmostEqual(float(payload["state_material_damage"][1, 0]), -5.0e-4)

            replay_damage, replay_calls = self._forced_replay_damage((-5.0e-4,))
            with mock.patch.object(
                engineering, "_replay_damage", side_effect=replay_damage
            ):
                replay = engineering.replay_engineering_grip_archive(path)
            self.assertTrue(replay["replayed"])
            self.assertFalse(replay["validated"])
            self.assertEqual(replay_calls[0], 1)

    def test_warning_bounded_healing_keeps_strict_flag_and_replays(self) -> None:
        permissive = engineering.GateTolerance(1.0e10, 0.0, 1.0)
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            load_final=2.0e-4,
            du_initial=1.0e-4,
            du_min=1.0e-4,
            max_stagger_iterations=1,
            displacement=permissive,
            damage=permissive,
            history=permissive,
            reaction=permissive,
            damage_residual=permissive,
        )
        values = (0.5, 0.4995)
        damage_solver, damage_calls = self._forced_damage_solver(values)
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", side_effect=damage_solver
        ):
            result = engineering.run_warning_engineering_grip(
                self.model, config, 1.0e-3
            )
        self.assertEqual(damage_calls[0], 2)
        self.assertEqual(result.accepted_steps[0].acceptance_status, "strict_passed")
        step = result.accepted_steps[1]
        self.assertEqual(step.acceptance_status, "warning_accepted")
        self.assertEqual(step.bypassed_gates, ("healing",))
        self.assertTrue(step.diagnostics.damage_range_pass)
        self.assertFalse(step.diagnostics.healing_pass)
        self.assertEqual(step.warning_range_excess, 0.0)
        assert step.warning_healing_magnitude is not None
        self.assertAlmostEqual(step.warning_healing_magnitude, 5.0e-4)

        with tempfile.TemporaryDirectory(prefix="engineering-grip-healing-") as directory:
            path = Path(directory) / "healing.npz"
            engineering.write_engineering_grip_archive(path, result)
            healing_solver, replay_calls = self._forced_replay_damage(values)
            with mock.patch.object(
                engineering, "_replay_damage", side_effect=healing_solver
            ):
                replay = engineering.replay_engineering_grip_archive(path)
            self.assertTrue(replay["replayed"])
            self.assertFalse(replay["validated"])
            self.assertEqual(replay_calls[0], 2)

    def test_warning_unbounded_keeps_large_raw_range_and_healing_and_replays(self) -> None:
        permissive = engineering.GateTolerance(1.0e10, 0.0, 1.0)
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            max_stagger_iterations=1,
            displacement=permissive,
            damage=permissive,
            history=permissive,
            reaction=permissive,
            damage_residual=permissive,
        )
        damage_solver, damage_calls = self._forced_damage_solver((-2.0,))
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", side_effect=damage_solver
        ):
            result = engineering.run_warning_engineering_grip(
                self.model,
                config,
                warning_safety_policy="unbounded",
            )
        self.assertEqual(damage_calls[0], 1)
        self.assertEqual(result.warning_safety_policy, "unbounded")
        self.assertIsNone(result.warning_safety_limit)
        self.assertFalse(result.validated)
        step = result.accepted_steps[0]
        self.assertEqual(step.acceptance_status, "warning_accepted")
        self.assertIn("damage_range", step.bypassed_gates)
        self.assertIn("healing", step.bypassed_gates)
        self.assertAlmostEqual(step.warning_range_excess or 0.0, 2.0)
        self.assertAlmostEqual(step.warning_healing_magnitude or 0.0, 2.0)
        self.assertEqual(result.final_state.material_damage[0], -2.0)

        with tempfile.TemporaryDirectory(prefix="engineering-grip-unbounded-") as directory:
            path = Path(directory) / "unbounded.npz"
            engineering.write_engineering_grip_archive(path, result)
            payload = engineering.load_engineering_grip_archive(path)
            self.assertEqual(
                str(payload["archive_schema"]),
                engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA,
            )
            self.assertEqual(str(payload["warning_safety_policy"]), "unbounded")
            self.assertFalse(bool(payload["warning_safety_limit_present"]))
            self.assertNotIn("warning_safety_limit", payload)
            self.assertTrue(all(value.dtype.kind != "O" for value in payload.values()))

            replay_damage, replay_calls = self._forced_replay_damage((-2.0,))
            with mock.patch.object(
                engineering, "_replay_damage", side_effect=replay_damage
            ):
                replay = engineering.replay_engineering_grip_archive(path)
            self.assertTrue(replay["replayed"])
            self.assertFalse(replay["validated"])
            self.assertEqual(replay["warning_safety_policy"], "unbounded")
            self.assertIsNone(replay["warning_safety_limit"])
            self.assertEqual(replay_calls[0], 1)

    def test_warning_force_and_moment_failures_are_independent_and_replayable(self) -> None:
        tiny = engineering.GateTolerance(float(np.nextafter(0.0, 1.0)), 0.0, 1.0)
        base = engineering.EngineeringGripConfig.smoke()
        for failed_gate in ("force", "moment"):
            with self.subTest(failed_gate=failed_gate):
                config = replace(base, **{failed_gate: tiny})
                result = engineering.run_warning_engineering_grip(
                    self.model, config, 1.0e-3
                )
                step = result.accepted_steps[0]
                self.assertEqual(step.acceptance_status, "warning_accepted")
                self.assertEqual(step.bypassed_gates, (failed_gate,))
                self.assertFalse(step.diagnostics.gate_passes[failed_gate])
                other_gate = "moment" if failed_gate == "force" else "force"
                self.assertTrue(step.diagnostics.gate_passes[other_gate])
                with tempfile.TemporaryDirectory(
                    prefix=f"engineering-grip-{failed_gate}-"
                ) as directory:
                    path = Path(directory) / f"{failed_gate}.npz"
                    engineering.write_engineering_grip_archive(path, result)
                    replay = engineering.replay_engineering_grip_archive(path)
                self.assertTrue(replay["replayed"])
                self.assertFalse(replay["validated"])

    def test_warning_over_limit_rejects_without_clipping_or_commit(self) -> None:
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            load_final=5.0e-5,
            du_initial=5.0e-5,
            du_min=5.0e-5,
            max_stagger_iterations=2,
            damage=engineering.GateTolerance(1.0e-20, 0.0, 1.0),
            work_check_load=4.0e-5,
        )
        original_damage = engineering.solve_fixed_history_damage

        def over_limit_damage(*args: Any, **kwargs: Any) -> Any:
            result = original_damage(*args, **kwargs)
            damage = np.array(result.damage, copy=True)
            damage[0] = 1.2
            history = np.asarray(args[3], dtype=np.float64)
            rho_used, _, rho_used_volume, _, _ = engineering._damage_residuals(
                self.model, damage, history, history
            )
            return replace(
                result,
                damage=damage,
                strong_residual=rho_used,
                euclidean_residual=rho_used_volume,
            )

        with mock.patch.object(
            engineering, "solve_fixed_history_damage", side_effect=over_limit_damage
        ):
            with self.assertRaises(engineering.EngineeringGripContinuationFailure) as context:
                engineering.run_warning_engineering_grip(
                    self.model, config, 0.1
                )
        failure = context.exception
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        self.assertEqual(len(failure.attempts), 1)
        attempt = failure.attempts[0]
        self.assertFalse(attempt.accepted)
        self.assertEqual(len(attempt.sweeps), config.max_stagger_iterations)
        self.assertEqual(
            attempt.committed_before_digest, attempt.committed_after_digest
        )
        assert attempt.warning_range_excess is not None
        assert attempt.warning_healing_magnitude is not None
        assert attempt.diagnostics.d_next is not None
        self.assertAlmostEqual(attempt.warning_range_excess, 0.2)
        self.assertEqual(attempt.warning_healing_magnitude, 0.0)
        self.assertAlmostEqual(float(attempt.diagnostics.d_next[0]), 1.2)

    def test_warning_replay_rejects_forged_sweep_count(self) -> None:
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            max_stagger_iterations=2,
            damage=engineering.GateTolerance(1.0e-20, 0.0, 1.0),
        )
        result = engineering.run_warning_engineering_grip(
            self.model, config, 1.0e-3
        )
        self.assertEqual(result.accepted_steps[0].stagger_iterations, 2)
        with tempfile.TemporaryDirectory(prefix="engineering-grip-sweep-count-") as directory:
            path = Path(directory) / "warning.npz"
            engineering.write_engineering_grip_archive(path, result)
            payload = {
                name: np.array(value, copy=True)
                for name, value in engineering.load_engineering_grip_archive(path).items()
            }
            payload["attempt_sweep_count"][0] = 1
            forged_path = Path(directory) / "forged-count.npz"
            np.savez_compressed(file=str(forged_path), **payload)
            with self.assertRaisesRegex(ValueError, "sweep count is forged"):
                engineering.replay_engineering_grip_archive(forged_path)

    def test_warning_replay_rejects_sweeps_after_strict_pass(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict-passing sweep"):
            engineering._validate_replay_sweep_termination(
                SimpleNamespace(passed=True), 0, 2
            )
        engineering._validate_replay_sweep_termination(
            SimpleNamespace(passed=False), 0, 2
        )
        engineering._validate_replay_sweep_termination(
            SimpleNamespace(passed=True), 1, 2
        )

    def test_warning_strict_path_matches_strict_result_before_acceptance_metadata(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        with mock.patch.object(engineering, "_localization_pass", return_value=True):
            strict = engineering.run_strict_engineering_grip(self.model, config)
        warning = engineering.run_warning_engineering_grip(
            self.model, config, 1.0e-3
        )
        self.assertEqual(warning.acceptance_status, "strict_passed")
        self.assertEqual(warning.accepted_steps[0].acceptance_status, "strict_passed")
        self.assertFalse(warning.validated)
        strict_step = strict.accepted_steps[0]
        warning_step = warning.accepted_steps[0]
        self.assertEqual(strict_step.diagnostics.gate_passes, warning_step.diagnostics.gate_passes)
        self.assertEqual(strict_step.diagnostics.passed, warning_step.diagnostics.passed)
        np.testing.assert_array_equal(strict_step.q_after, warning_step.q_after)
        np.testing.assert_array_equal(strict_step.d_next, warning_step.d_next)
        np.testing.assert_array_equal(strict_step.H_next, warning_step.H_next)
        np.testing.assert_array_equal(strict_step.rho_next, warning_step.rho_next)

    def test_warning_result_archive_and_replay_are_never_validated(self) -> None:
        result = self._warning_result()
        self.assertEqual(result.acceptance_mode, "warning")
        self.assertEqual(result.acceptance_status, "warning_accepted")
        self.assertEqual(result.completion_status, "warning_completed")
        self.assertFalse(result.validated)
        self.assertTrue(result.accepted_steps[0].diagnostics.complete_sweep)
        self.assertFalse(result.accepted_steps[0].diagnostics.passed)
        self.assertEqual(
            result.accepted_steps[0].bypassed_gates,
            ("damage",),
        )

        with tempfile.TemporaryDirectory(prefix="engineering-grip-warning-") as directory:
            path = Path(directory) / "warning.npz"
            engineering.write_engineering_grip_archive(path, result)
            payload = engineering.load_engineering_grip_archive(path)
            self.assertEqual(
                str(payload["archive_schema"]), engineering.WARNING_ARCHIVE_SCHEMA
            )
            self.assertFalse(bool(payload["validated"]))
            self.assertTrue(all(value.dtype.kind != "O" for value in payload.values()))
            replay = engineering.replay_engineering_grip_archive(path)
            self.assertTrue(replay["replayed"])
            self.assertFalse(replay["validated"])

    def test_warning_replay_rejects_status_bypass_limit_and_source_forgery(self) -> None:
        result = self._warning_result()
        with tempfile.TemporaryDirectory(prefix="engineering-grip-warning-forgery-") as directory:
            path = Path(directory) / "warning.npz"
            engineering.write_engineering_grip_archive(path, result)
            original = engineering.load_engineering_grip_archive(path)

            def write(name: str, values: dict[str, np.ndarray]) -> Path:
                target = Path(directory) / name
                np.savez_compressed(file=str(target), **values)
                return target

            status = {name: np.array(value, copy=True) for name, value in original.items()}
            status["acceptance_status"] = np.asarray("strict_passed", dtype=np.str_)
            with self.assertRaisesRegex(ValueError, "acceptance status"):
                engineering.replay_engineering_grip_archive(write("status.npz", status))

            bypass = {name: np.array(value, copy=True) for name, value in original.items()}
            bypass["accepted_bypassed_gate_mask"][0, engineering._WARNING_GATE_INDEX["damage"]] = False
            with self.assertRaisesRegex(ValueError, "warning accepted-state audit"):
                engineering.replay_engineering_grip_archive(write("bypass.npz", bypass))

            limit = {name: np.array(value, copy=True) for name, value in original.items()}
            limit["warning_safety_limit"] = np.asarray(2.0e-3, dtype=np.float64)
            with self.assertRaisesRegex(ValueError, "safety limit"):
                engineering.replay_engineering_grip_archive(write("limit.npz", limit))

            source = {name: np.array(value, copy=True) for name, value in original.items()}
            metadata = json.loads(str(source["metadata_json"]))
            metadata["code_hash"] = "0" * 64
            source["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True), dtype=np.str_
            )
            with self.assertRaisesRegex(ValueError, "source-bundle code hash"):
                engineering.replay_engineering_grip_archive(write("source.npz", source))

    def test_warning_hard_failures_are_not_bypassable(self) -> None:
        passes = {
            name: True
            for name in (
                *engineering._WARNING_HARD_GATE_NAMES,
                *engineering.WARNING_GATE_NAMES,
            )
        }
        passes["damage"] = False
        diagnostic = SimpleNamespace(
            complete_sweep=True,
            passed=False,
            d_next=np.asarray([0.25, 0.5], dtype=np.float64),
            gate_passes=passes,
        )
        malformed = dict(passes)
        del malformed["finite"]
        with self.assertRaisesRegex(ValueError, "canonical gates"):
            engineering._warning_acceptance_decision(
                SimpleNamespace(**{**vars(diagnostic), "gate_passes": malformed}),
                np.zeros(2, dtype=np.float64),
                1.0e-3,
            )
        with mock.patch.object(
            engineering,
            "run_zero_damage_work_check",
            side_effect=RuntimeError("manufactured work failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "manufactured work failure"):
                engineering.run_warning_engineering_grip(
                    self.model,
                    engineering.EngineeringGripConfig.smoke(),
                    1.0e-3,
                )

    def test_incomplete_mechanics_gate_preserves_independent_conservation_status(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        committed = engineering.EngineeringGripState(
            u_bar=0.0,
            union_displacement=np.zeros(self.model.n_dofs, dtype=np.float64),
            material_damage=np.zeros(self.model.n_material, dtype=np.float64),
            material_history=np.zeros(self.model.n_material, dtype=np.float64),
        )
        before: Any = SimpleNamespace(
            displacement=np.zeros(self.model.n_dofs, dtype=np.float64),
            residual=np.zeros(self.model.n_dofs, dtype=np.float64),
            total_force=np.asarray([1.0, 0.0], dtype=np.float64),
            total_moment=0.0,
            energy=0.0,
            outer_reaction=0.0,
            bottom_reaction=0.0,
            anchor_reaction=0.0,
            free_residual_norm=0.0,
            free_residual_relative_norm=0.0,
            cg_info=0,
            cg_convergence_limit=1.0,
        )
        with mock.patch.object(
            engineering,
            "_mechanics_gate",
            return_value=(False, True, 1.0, 0.0, False, True),
        ):
            diagnostic = engineering._incomplete_diagnostics(
                config,
                committed,
                np.zeros(self.model.n_material, dtype=np.float64),
                1.0e-4,
                "mechanics_before_gate",
                "mechanics-before force gate failed",
                before=before,
                model=self.model,
                sweep_index=0,
            )
        self.assertFalse(diagnostic.mechanics_before_pass)
        self.assertFalse(diagnostic.force_pass)
        self.assertTrue(diagnostic.moment_pass)
        self.assertFalse(diagnostic.gate_passes["force"])
        self.assertTrue(diagnostic.gate_passes["moment"])

    def test_incomplete_mechanics_gate_status_replays_independently(self) -> None:
        config = replace(
            engineering.EngineeringGripConfig.smoke(),
            load_final=2.0e-4,
            du_initial=2.0e-4,
            du_min=1.0e-4,
            max_stagger_iterations=1,
        )
        original_solver = engineering._solve_direct_mechanics
        calls = [0]

        def fail_first_continuation_gate(*args: Any, **kwargs: Any) -> Any:
            result = original_solver(*args, **kwargs)
            # The first three solves belong to the one-time work check.  Make
            # only the first continuation mechanics result carry a nonzero CG
            # status; the subsequent cutback and endpoint attempts are real.
            if calls[0] == 3:
                result = replace(result, cg_info=1)
            calls[0] += 1
            return result

        with mock.patch.object(
            engineering,
            "_solve_direct_mechanics",
            side_effect=fail_first_continuation_gate,
        ):
            result = engineering.run_warning_engineering_grip(
                self.model, config, 1.0e-3
            )
        self.assertGreaterEqual(calls[0], 5)
        rejected = result.attempts[0]
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.failure_stage, "mechanics_before_gate")
        self.assertFalse(rejected.diagnostics.mechanics_before_pass)
        self.assertTrue(rejected.diagnostics.force_pass)
        self.assertTrue(rejected.diagnostics.moment_pass)

        with tempfile.TemporaryDirectory(prefix="engineering-grip-incomplete-gates-") as directory:
            path = Path(directory) / "warning.npz"
            engineering.write_engineering_grip_archive(path, result)
            replay = engineering.replay_engineering_grip_archive(path)
        self.assertTrue(replay["replayed"])
        self.assertFalse(replay["validated"])

    def test_incomplete_failure_rolls_back_and_cutback_is_typed(self) -> None:
        original = engineering.solve_fixed_outer_displacement
        calls = 0

        def fail_after_work(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls >= 4:
                raise RuntimeError("manufactured direct mechanics failure")
            return original(*args, **kwargs)

        with mock.patch.object(
            engineering, "solve_fixed_outer_displacement", side_effect=fail_after_work
        ):
            with self.assertRaises(engineering.EngineeringGripContinuationFailure) as context:
                engineering.run_strict_engineering_grip(
                    self.model, engineering.EngineeringGripConfig.smoke()
                )
        failure = context.exception
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        self.assertGreaterEqual(len(failure.attempts), 1)
        self.assertFalse(failure.attempts[0].accepted)
        self.assertEqual(
            failure.attempts[0].committed_before_digest,
            failure.attempts[0].committed_after_digest,
        )
        self.assertEqual(failure.attempts[0].failure_stage, "mechanics_before")

    def test_controller_has_no_matched_opening_or_per_sweep_work_path(self) -> None:
        self.assertFalse(hasattr(engineering, "solve_a2_opening_controlled"))
        self.assertFalse(hasattr(engineering, "solve_a2_unit_load"))
        self.assertFalse(hasattr(engineering, "scale_a2_unit_solution"))
        source = Path(engineering.__file__).read_text(encoding="utf-8")
        self.assertNotIn("physical_interface_opening(", source)
        self.assertNotIn("R_eff", source)

    def test_compact_archive_crack_replay_and_forgery_rejection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-") as directory:
            path = Path(directory) / "accepted.npz"
            # See the corresponding note in the stagger test: this fixture is
            # for archive algebra, while the real final localization gate is
            # checked after the override is removed.
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                result = engineering.run_strict_engineering_grip(
                    self.model, engineering.EngineeringGripConfig.smoke()
                )
                engineering.write_engineering_grip_archive(path, result)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(value.dtype.kind != "O" for value in archive.values()))
                self.assertIn("state_crack_particle_strain", archive.files)
                self.assertIn("state_material_amor_driver", archive.files)
                self.assertIn("state_physical_interface_opening", archive.files)
                self.assertIn("state_interface_damage_top_max", archive.files)
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                original_replay_mechanics = engineering._replay_mechanics_at_state

                def reject_exact_zero_state(*args, **kwargs):
                    if float(args[2]) == 0.0:
                        raise AssertionError(
                            "exact zero state must use analytic replay, not mechanics reconstruction"
                        )
                    return original_replay_mechanics(*args, **kwargs)

                with mock.patch.object(
                    engineering,
                    "_replay_mechanics_at_state",
                    side_effect=reject_exact_zero_state,
                ):
                    replay = engineering.replay_engineering_grip_archive(path)
                self.assertTrue(replay["replayed"])
            payload = engineering.load_engineering_grip_archive(path)
            forged = {name: np.array(value, copy=True) for name, value in payload.items()}
            forged["accepted_top_reaction_after"][0] += 1.0e-3
            forged_path = Path(directory) / "forged.npz"
            np.savez_compressed(file=str(forged_path), **forged)
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "reaction"):
                    engineering.replay_engineering_grip_archive(forged_path)
            forged_opening = {
                name: np.array(value, copy=True) for name, value in payload.items()
            }
            forged_opening["state_physical_interface_opening"][1] += 1.0e-3
            opening_path = Path(directory) / "forged-opening.npz"
            np.savez_compressed(file=str(opening_path), **forged_opening)
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "interface opening"):
                    engineering.replay_engineering_grip_archive(opening_path)

            with self.assertRaisesRegex(ValueError, "crack localization"):
                engineering.replay_engineering_grip_archive(path)

    def _damage_residual_reconstruction_fixture(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        damage = np.linspace(0.2, 0.8, self.model.n_material, dtype=np.float64)
        history = np.full(self.model.n_material, 1.0e3, dtype=np.float64)
        matrix = self.model.material_operator.to_scipy().tocsr()
        rho_used, _, rho_used_volume, _, _ = engineering._damage_residuals(
            self.model, damage, history, history
        )
        graph = np.asarray(matrix @ damage, dtype=np.float64)
        terms = (
            self.model.Gc / self.model.l0 * damage,
            self.model.Gc * self.model.l0 * graph,
            2.0 * history * damage,
            2.0 * history,
        )
        bound = engineering._REPLAY_BOUND * np.sum(
            np.abs(np.asarray(terms, dtype=np.float64)), axis=0
        )
        return (
            damage,
            history,
            matrix,
            np.asarray(self.model.material_operator.volumes, dtype=np.float64),
            rho_used,
            rho_used_volume,
            np.asarray(bound, dtype=np.float64),
        )

    def test_damage_residual_reconstruction_accepts_observed_cancellation_scale(
        self,
    ) -> None:
        damage, history, matrix, volumes, rho, volume_rho, bound = (
            self._damage_residual_reconstruction_fixture()
        )
        index = int(np.argmax(bound))
        observed_defect = 2.3283064365386963e-10
        self.assertLess(observed_defect, bound[index])
        forged_rho = np.array(rho, copy=True)
        forged_volume_rho = np.array(volume_rho, copy=True)
        forged_rho[index] += observed_defect
        forged_volume_rho[index] += volumes[index] * observed_defect
        engineering._compare_damage_residual_reconstruction(
            forged_rho,
            forged_volume_rho,
            damage,
            history,
            matrix,
            volumes,
            self.model.Gc,
            self.model.l0,
            "manufactured cancellation",
        )

    def test_damage_residual_reconstruction_rejects_strong_signal_above_bound(
        self,
    ) -> None:
        damage, history, matrix, volumes, rho, volume_rho, bound = (
            self._damage_residual_reconstruction_fixture()
        )
        index = int(np.argmax(bound))
        forged_rho = np.array(rho, copy=True)
        forged_rho[index] += 2.0 * bound[index]
        with self.assertRaisesRegex(ValueError, "512eps constituent bound"):
            engineering._compare_damage_residual_reconstruction(
                forged_rho,
                volume_rho,
                damage,
                history,
                matrix,
                volumes,
                self.model.Gc,
                self.model.l0,
                "manufactured strong signal",
            )

    def test_damage_residual_reconstruction_checks_volume_bound(self) -> None:
        damage, history, matrix, volumes, rho, volume_rho, bound = (
            self._damage_residual_reconstruction_fixture()
        )
        index = int(np.argmax(bound))
        accepted_volume_rho = np.array(volume_rho, copy=True)
        accepted_volume_rho[index] += 0.5 * volumes[index] * bound[index]
        engineering._compare_damage_residual_reconstruction(
            rho,
            accepted_volume_rho,
            damage,
            history,
            matrix,
            volumes,
            self.model.Gc,
            self.model.l0,
            "manufactured volume roundoff",
        )
        rejected_volume_rho = np.array(volume_rho, copy=True)
        rejected_volume_rho[index] += 2.0 * volumes[index] * bound[index]
        with self.assertRaisesRegex(ValueError, "512eps constituent bound"):
            engineering._compare_damage_residual_reconstruction(
                rho,
                rejected_volume_rho,
                damage,
                history,
                matrix,
                volumes,
                self.model.Gc,
                self.model.l0,
                "manufactured volume signal",
            )

    def test_damage_residual_reconstruction_validates_float64_and_positive_volumes(
        self,
    ) -> None:
        damage, history, matrix, volumes, rho, volume_rho, _ = (
            self._damage_residual_reconstruction_fixture()
        )
        with self.assertRaisesRegex(ValueError, "exact float64 dtype"):
            engineering._compare_damage_residual_reconstruction(
                rho.astype(np.float32),
                volume_rho,
                damage,
                history,
                matrix,
                volumes,
                self.model.Gc,
                self.model.l0,
                "invalid dtype",
            )
        invalid_volumes = np.array(volumes, copy=True)
        invalid_volumes[0] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            engineering._compare_damage_residual_reconstruction(
                rho,
                volume_rho,
                damage,
                history,
                matrix,
                invalid_volumes,
                self.model.Gc,
                self.model.l0,
                "invalid volumes",
            )

    def test_cancellation_aware_reconstruction_preserves_manufactured_paths(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        with mock.patch.object(engineering, "_localization_pass", return_value=True):
            strict = engineering.run_strict_engineering_grip(self.model, config)
        warning = engineering.run_warning_engineering_grip(
            self.model, config, 1.0e-3
        )
        self.assertEqual(strict.acceptance_status, "strict_passed")
        self.assertEqual(warning.acceptance_status, "strict_passed")
        self.assertTrue(strict.accepted_steps[0].diagnostics.passed)
        self.assertTrue(warning.accepted_steps[0].diagnostics.passed)

    def test_replay_comparators_accept_roundoff_and_reject_signal(self) -> None:
        scalar_expected = 0.0
        scalar_scale = 3.0
        scalar_atol = engineering._replay_reduction_atol(
            scalar_scale, "test scalar"
        )
        engineering._compare_replay_scalar(
            0.5 * scalar_atol,
            scalar_expected,
            "test scalar",
            limit=0.0,
            absolute_atol=scalar_atol,
            scale_floor=0.0,
        )
        with self.assertRaisesRegex(ValueError, "test scalar"):
            engineering._compare_replay_scalar(
                2.0 * scalar_atol,
                scalar_expected,
                "test scalar",
                limit=0.0,
                absolute_atol=scalar_atol,
                scale_floor=0.0,
            )

        vector_expected = np.zeros(2, dtype=np.float64)
        vector_scale = np.asarray([2.0, 4.0], dtype=np.float64)
        vector_tolerance = engineering._REPLAY_COMPARISON_BOUND * vector_scale
        engineering._compare_replay_vector(
            0.5 * vector_tolerance,
            vector_expected,
            "test vector",
            vector_scale,
        )
        meaningful = 0.5 * vector_tolerance
        meaningful[0] += 1.0e-9
        with self.assertRaisesRegex(ValueError, "test vector"):
            engineering._compare_replay_vector(
                meaningful,
                vector_expected,
                "test vector",
                vector_scale,
            )

        collar_coordinates = np.asarray([1.2, 1.4], dtype=np.float64)
        collar_offsets = np.asarray([0.2, 0.4], dtype=np.float64)
        collar_bound = engineering._top_collar_coordinate_roundoff_bound(
            collar_coordinates,
            np.asarray(1.0, dtype=np.float64),
            collar_offsets,
        )
        np.testing.assert_array_equal(
            collar_bound,
            4.0
            * (np.finfo(np.float64).eps / 2.0)
            * (collar_coordinates + 1.0 + collar_offsets),
        )
        self.assertGreater(collar_bound[1], collar_bound[0])
        with self.assertRaisesRegex(ValueError, "exact float64"):
            engineering._top_collar_coordinate_roundoff_bound(
                collar_coordinates.astype(np.float32),
                np.asarray(1.0, dtype=np.float64),
                collar_offsets,
            )

    def test_source_bundle_hash_is_framed_and_sensitive(self) -> None:
        entries = (("a/module.py", b"ab"), ("b/module.py", b"c"))
        self.assertEqual(
            engineering._source_bundle_hash(entries),
            engineering._source_bundle_hash(tuple(reversed(entries))),
        )
        self.assertNotEqual(
            engineering._source_bundle_hash(entries),
            engineering._source_bundle_hash((("a/module.py", b"a"), ("b/module.py", b"bc"))),
        )
        self.assertNotEqual(
            engineering._source_bundle_hash(entries),
            engineering._source_bundle_hash((("a/other.py", b"ab"), ("b/module.py", b"c"))),
        )

    def test_replay_requires_recomputed_work_check_to_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-work-") as directory:
            path = self._write_valid_archive(directory)
            payload = {
                name: np.array(value, copy=True)
                for name, value in engineering.load_engineering_grip_archive(path).items()
            }
            tiny_limit = float(np.nextafter(0.0, 1.0))
            metadata = json.loads(str(payload["metadata_json"]))
            metadata["config"]["work_relative_limit"] = tiny_limit
            payload["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True), dtype=np.str_
            )
            payload["work_check_relative_limit"] = np.asarray(
                tiny_limit, dtype=np.float64
            )
            payload["work_check_passed"] = np.asarray(False, dtype=np.bool_)
            forged_path = Path(directory) / "failed-work.npz"
            np.savez_compressed(file=str(forged_path), **payload)
            with mock.patch.object(engineering, "_localization_pass", return_value=True):
                with self.assertRaisesRegex(ValueError, "recomputed work-check failed"):
                    engineering.replay_engineering_grip_archive(forged_path)

    def test_replay_requires_material_masks_and_rejects_dtype_shape_content_forgeries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-material-mask-") as directory:
            path = self._write_valid_archive(directory)
            original = engineering.load_engineering_grip_archive(path)

            for field in ("material_bottom_mask", "material_top_mask"):
                missing = {
                    name: np.array(value, copy=True) for name, value in original.items()
                }
                del missing[field]
                missing_path = Path(directory) / f"missing-{field}.npz"
                np.savez_compressed(file=str(missing_path), **missing)
                with self.assertRaisesRegex(ValueError, f"missing fields.*{field}"):
                    engineering.replay_engineering_grip_archive(missing_path)

                bad_dtype = {
                    name: np.array(value, copy=True) for name, value in original.items()
                }
                bad_dtype[field] = bad_dtype[field].astype(np.int8)
                bad_dtype_path = Path(directory) / f"dtype-{field}.npz"
                np.savez_compressed(file=str(bad_dtype_path), **bad_dtype)
                with self.assertRaisesRegex(ValueError, f"{field}.*exact boolean dtype"):
                    engineering.replay_engineering_grip_archive(bad_dtype_path)

                bad_shape = {
                    name: np.array(value, copy=True) for name, value in original.items()
                }
                bad_shape[field] = bad_shape[field].reshape(1, -1)
                bad_shape_path = Path(directory) / f"shape-{field}.npz"
                np.savez_compressed(file=str(bad_shape_path), **bad_shape)
                with self.assertRaisesRegex(ValueError, f"{field}.*shape"):
                    engineering.replay_engineering_grip_archive(bad_shape_path)

                bad_content = {
                    name: np.array(value, copy=True) for name, value in original.items()
                }
                bad_content[field].flat[0] = not bool(bad_content[field].flat[0])
                bad_content_path = Path(directory) / f"content-{field}.npz"
                np.savez_compressed(file=str(bad_content_path), **bad_content)
                with self.assertRaisesRegex(ValueError, f"{field}.*coordinates"):
                    engineering.replay_engineering_grip_archive(bad_content_path)

    def test_replay_requires_canonical_static_integer_and_boolean_dtypes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-static-dtype-") as directory:
            path = self._write_valid_archive(directory)
            original = engineering.load_engineering_grip_archive(path)
            fields = (
                ("static_support_mask", np.int8, "exact boolean dtype"),
                ("static_support_sizes", np.float64, "exact int64 dtype"),
                ("static_shape_rank", np.float64, "exact int64 dtype"),
            )
            for field, dtype, message in fields:
                forged = {
                    name: np.array(value, copy=True) for name, value in original.items()
                }
                forged[field] = forged[field].astype(dtype)
                forged_path = Path(directory) / f"{field}.npz"
                np.savez_compressed(file=str(forged_path), **forged)
                with self.assertRaisesRegex(ValueError, f"{field}.*{message}"):
                    engineering.replay_engineering_grip_archive(forged_path)

    def test_replay_rejects_stale_v1_schema_explicitly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-schema-") as directory:
            path = self._write_valid_archive(directory)
            payload = {
                name: np.array(value, copy=True)
                for name, value in engineering.load_engineering_grip_archive(path).items()
            }
            payload["archive_schema"] = np.asarray(
                "fracture2d-engineering-grip-v1", dtype=np.str_
            )
            stale_path = Path(directory) / "stale-v1.npz"
            np.savez_compressed(file=str(stale_path), **payload)
            with self.assertRaisesRegex(ValueError, "unsupported stale schema"):
                engineering.replay_engineering_grip_archive(stale_path)

    def test_replay_rejects_meaningful_top_collar_coordinate_forgery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-top-coordinate-") as directory:
            path = self._write_valid_archive(directory)
            payload = {
                name: np.array(value, copy=True)
                for name, value in engineering.load_engineering_grip_archive(path).items()
            }
            top_particle = int(np.flatnonzero(payload["volume_top_mask"])[0])
            payload["union_coordinates"][top_particle, 1] += 1.0e-6
            forged_path = Path(directory) / "forged-top-coordinate.npz"
            np.savez_compressed(file=str(forged_path), **payload)
            with mock.patch.object(engineering, "_localization_pass", return_value=True):
                with self.assertRaisesRegex(ValueError, "collar|grip phase quadrature"):
                    engineering.replay_engineering_grip_archive(forged_path)

    def test_replay_accepts_top_interface_ulp_roundoff(self) -> None:
        coordinates, triangles, volumes, bottom, top, edges, delta = _material_cloud()
        top_particles = np.flatnonzero(top)
        coordinates = np.array(coordinates, copy=True)
        coordinates[top_particles, 1] = np.nextafter(
            np.float64(1.0), np.float64(0.0)
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
        geometry = build_union_geometry_from_arrays(
            coordinates,
            triangles,
            volumes,
            bottom,
            top,
            0,
            edges,
            delta,
            0.2,
            0.0,
        )
        model = engineering.build_engineering_grip_model(
            geometry, 2.0, 0.3, 1.0e-8, 1.0e-12, 1.7, 0.65
        )
        with tempfile.TemporaryDirectory(prefix="engineering-grip-top-ulp-") as directory:
            path = Path(directory) / "ulp-top-interface.npz"
            with mock.patch.object(engineering, "_localization_pass", return_value=True):
                result = engineering.run_strict_engineering_grip(
                    model, engineering.EngineeringGripConfig.smoke()
                )
                engineering.write_engineering_grip_archive(path, result)
                replay = engineering.replay_engineering_grip_archive(path)
            self.assertTrue(replay["replayed"])

    def test_replay_rejects_work_graph_metadata_and_cg_forgeries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-provenance-") as directory:
            path = Path(directory) / "accepted.npz"
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                result = engineering.run_strict_engineering_grip(
                    self.model, engineering.EngineeringGripConfig.smoke()
                )
                engineering.write_engineering_grip_archive(path, result)

            original = engineering.load_engineering_grip_archive(path)

            def copy_payload() -> dict[str, np.ndarray]:
                return {
                    name: np.array(value, copy=True)
                    for name, value in original.items()
                }

            def write_payload(
                name: str, payload: dict[str, np.ndarray]
            ) -> Path:
                forged_path = Path(directory) / name
                np.savez_compressed(file=str(forged_path), **payload)
                return forged_path

            def update_metadata(
                payload: dict[str, np.ndarray],
                update: Callable[[dict[str, Any]], None],
            ) -> None:
                metadata = json.loads(str(payload["metadata_json"]))
                update(metadata)
                payload["metadata_json"] = np.asarray(
                    json.dumps(metadata, sort_keys=True), dtype=np.str_
                )

            def assert_rejected(
                name: str,
                payload: dict[str, np.ndarray],
                message: str,
            ) -> None:
                with mock.patch.object(
                    engineering, "_localization_pass", return_value=True
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        engineering.replay_engineering_grip_archive(
                            write_payload(name, payload)
                        )

            work_config = copy_payload()
            update_metadata(
                work_config,
                lambda metadata: metadata["config"].update(
                    {"work_relative_limit": 0.0}
                ),
            )
            assert_rejected(
                "forged-work-config.npz",
                work_config,
                "work-check settings",
            )

            visibility_rule = copy_payload()
            update_metadata(
                visibility_rule,
                lambda metadata: metadata.update({"visibility_rule": "forged"}),
            )
            assert_rejected(
                "forged-visibility-rule.npz",
                visibility_rule,
                "visibility rule",
            )

            static_digest = copy_payload()
            update_metadata(
                static_digest,
                lambda metadata: metadata.update({"static_digest": "0" * 64}),
            )
            assert_rejected(
                "forged-static-digest.npz",
                static_digest,
                "static digest",
            )

            partition = copy_payload()
            partition["candidate_edges"] = partition["candidate_edges"][::-1].copy()
            assert_rejected(
                "forged-edge-partition.npz",
                partition,
                "edge partition",
            )

            cg_status = copy_payload()
            sweeps = json.loads(str(cg_status["attempt_sweeps_json"][0]))
            sweeps[0]["cg_info_after"] = 1
            cg_status["attempt_sweeps_json"] = np.asarray(
                [json.dumps(sweeps, sort_keys=True)], dtype=np.str_
            )
            assert_rejected(
                "forged-cg-status.npz",
                cg_status,
                "production cg_info_after",
            )

            accepted_cg_status = copy_payload()
            accepted_cg_status["accepted_cg_info_after"][0] = 1
            assert_rejected(
                "forged-accepted-cg-status.npz",
                accepted_cg_status,
                "accepted cg_info_after",
            )

            duplicate_edges = np.vstack(
                (
                    self.geometry.directed_edges,
                    self.geometry.directed_edges[-1],
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate directed edges"):
                engineering._replay_ba(
                    self.geometry.coordinates,
                    self.geometry.union_volumes,
                    duplicate_edges,
                    self.geometry.delta,
                    self.model.static.rank_rtol,
                )

            material_edges = self.geometry.material_reference_edges
            missing_family = material_edges[
                (material_edges[:, 0] != 0) & (material_edges[:, 1] != 0)
            ]
            with self.assertRaisesRegex(ValueError, "outgoing family"):
                engineering._replay_damage(
                    np.zeros(self.model.n_material, dtype=np.float64),
                    self.geometry.material_reference_coordinates,
                    self.geometry.material_reference_volumes,
                    missing_family,
                    self.geometry.delta,
                    self.model.Gc,
                    self.model.l0,
                )

    def test_replay_rejects_attempt_schedule_and_stage_prerequisite_forgeries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-attempt-") as directory:
            accepted_path = Path(directory) / "accepted.npz"
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                result = engineering.run_strict_engineering_grip(
                    self.model, engineering.EngineeringGripConfig.smoke()
                )
                engineering.write_engineering_grip_archive(accepted_path, result)

            accepted_payload = engineering.load_engineering_grip_archive(accepted_path)
            forged_schedule = {
                name: np.array(value, copy=True)
                for name, value in accepted_payload.items()
            }
            forged_schedule["attempt_load_target"][0] = 9.0e-5
            schedule_path = Path(directory) / "forged-schedule.npz"
            np.savez_compressed(file=str(schedule_path), **forged_schedule)
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "target schedule"):
                    engineering.replay_engineering_grip_archive(schedule_path)

            original_solver = engineering.solve_fixed_outer_displacement
            calls = 0

            def fail_once(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise RuntimeError("manufactured one-time mechanics failure")
                return original_solver(*args, **kwargs)

            incomplete_path = Path(directory) / "incomplete.npz"
            with mock.patch.object(
                engineering,
                "solve_fixed_outer_displacement",
                side_effect=fail_once,
            ), mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                incomplete_result = engineering.run_strict_engineering_grip(
                    self.model, engineering.EngineeringGripConfig.smoke()
                )
                self.assertEqual(
                    incomplete_result.attempts[0].failure_stage,
                    "mechanics_after",
                )
                engineering.write_engineering_grip_archive(
                    incomplete_path, incomplete_result
                )

            incomplete_payload = engineering.load_engineering_grip_archive(
                incomplete_path
            )

            bad_stage = {
                name: np.array(value, copy=True)
                for name, value in incomplete_payload.items()
            }
            bad_stage["attempt_failure_stage"][0] = "damage"
            bad_stage_path = Path(directory) / "forged-stage.npz"
            np.savez_compressed(file=str(bad_stage_path), **bad_stage)
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "failure classification"):
                    engineering.replay_engineering_grip_archive(bad_stage_path)

            missing_prerequisite = {
                name: np.array(value, copy=True)
                for name, value in incomplete_payload.items()
            }
            sweeps = json.loads(
                str(missing_prerequisite["attempt_sweeps_json"][0])
            )
            sweeps[0]["d_next"] = None
            sweeps[0]["rho_used"] = None
            sweeps[0]["rho_used_volume"] = None
            sweep_json = np.array(
                missing_prerequisite["attempt_sweeps_json"], copy=True
            )
            sweep_json[0] = json.dumps(sweeps, sort_keys=True)
            missing_prerequisite["attempt_sweeps_json"] = sweep_json
            missing_path = Path(directory) / "forged-stage-prerequisite.npz"
            np.savez_compressed(file=str(missing_path), **missing_prerequisite)
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "inconsistent primitives"):
                    engineering.replay_engineering_grip_archive(missing_path)

    def test_archive_audit_rejects_state_pairing_free_gate_and_rollback_forgeries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-grip-audit-") as directory:
            path = Path(directory) / "accepted.npz"
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                result = engineering.run_strict_engineering_grip(
                    self.model, engineering.EngineeringGripConfig.smoke()
                )
                engineering.write_engineering_grip_archive(path, result)

            def write_payload(name: str, payload: dict[str, np.ndarray]) -> Path:
                forged_path = Path(directory) / name
                np.savez_compressed(file=str(forged_path), **payload)
                return forged_path

            def copy_payload() -> dict[str, np.ndarray]:
                return {
                    name: np.array(value, copy=True)
                    for name, value in engineering.load_engineering_grip_archive(path).items()
                }

            state_q = copy_payload()
            state_q["state_union_displacement"][1, 2] += 1.0e-3
            state_q["state_digest"][1] = engineering._state_digest(
                state_q["state_u_bar"][1],
                state_q["state_union_displacement"][1],
                state_q["state_material_damage"][1],
                state_q["state_material_history"][1],
            )
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(
                    ValueError, "state material displacement|q_after|free residual"
                ):
                    engineering.replay_engineering_grip_archive(
                        write_payload("forged-state-q.npz", state_q)
                    )

            for field, amount in (
                ("state_material_damage", 1.0e-3),
                ("state_material_history", 1.0e-3),
            ):
                payload = copy_payload()
                payload[field][1, 12] += amount
                payload["state_digest"][1] = engineering._state_digest(
                    payload["state_u_bar"][1],
                    payload["state_union_displacement"][1],
                    payload["state_material_damage"][1],
                    payload["state_material_history"][1],
                )
                with mock.patch.object(
                    engineering, "_localization_pass", return_value=True
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "crack diagnostics|accepted .*state pairing|damage/history",
                    ):
                        engineering.replay_engineering_grip_archive(
                            write_payload(f"forged-{field}.npz", payload)
                        )

            accepted_pair = copy_payload()
            accepted_pair["accepted_q_after"][0, 2] += 1.0e-3
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "accepted q_after"):
                    engineering.replay_engineering_grip_archive(
                        write_payload("forged-accepted-pair.npz", accepted_pair)
                    )

            free_gate = copy_payload()

            def forge_free_residual(raw: str) -> str:
                payload = json.loads(raw)
                payload["free_residual_norm_after"] += 1.0e-6
                return json.dumps(payload, sort_keys=True)

            free_gate["accepted_diagnostics_json"] = np.asarray(
                [
                    forge_free_residual(
                        str(free_gate["accepted_diagnostics_json"][0])
                    )
                ],
                dtype=np.str_,
            )
            free_gate["attempt_diagnostics_json"] = np.asarray(
                [
                    forge_free_residual(
                        str(free_gate["attempt_diagnostics_json"][0])
                    )
                ],
                dtype=np.str_,
            )
            sweeps = json.loads(str(free_gate["attempt_sweeps_json"][0]))
            sweeps[0] = json.loads(forge_free_residual(json.dumps(sweeps[0])))
            free_gate["attempt_sweeps_json"] = np.asarray(
                [json.dumps(sweeps, sort_keys=True)], dtype=np.str_
            )
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "free_residual"):
                    engineering.replay_engineering_grip_archive(
                        write_payload("forged-free-gate.npz", free_gate)
                    )

            rejected_config = replace(
                engineering.EngineeringGripConfig.smoke(),
                max_stagger_iterations=1,
                damage=engineering.GateTolerance(3.5e-9, 0.0, 1.0),
            )
            rejected_path = Path(directory) / "rejected.npz"
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                rejected_result = engineering.run_strict_engineering_grip(
                    self.model, rejected_config
                )
                self.assertFalse(rejected_result.attempts[0].accepted)
                engineering.write_engineering_grip_archive(
                    rejected_path, rejected_result
                )
            rollback = {
                name: np.array(value, copy=True)
                for name, value in engineering.load_engineering_grip_archive(
                    rejected_path
                ).items()
            }
            rollback["attempt_after_digest"][0] = rollback["state_digest"][1]
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                with self.assertRaisesRegex(ValueError, "rollback"):
                    engineering.replay_engineering_grip_archive(
                        write_payload("forged-rollback.npz", rollback)
                    )

    def test_crack_region_mask_and_diagnostics(self) -> None:
        mask = engineering.build_crack_region_mask(
            self.geometry.material_reference_coordinates, self.geometry.delta
        )
        self.assertTrue(mask[12])
        self.assertGreater(np.count_nonzero(mask), 0)
        damage = np.zeros(self.model.n_material, dtype=np.float64)
        history = np.zeros_like(damage)
        damage[mask] = 0.4
        history[mask] = 2.0
        strain = np.zeros((self.model.n_material, 2, 2), dtype=np.float64)
        strain[mask, 0, 0] = 0.2
        amor = np.zeros_like(damage)
        amor[mask] = 1.5
        diagnostics = engineering.compute_crack_region_diagnostics(
            self.geometry.material_reference_coordinates,
            self.geometry.delta,
            damage,
            history,
            strain,
            amor,
        )
        self.assertTrue(diagnostics.localization_pass)
        self.assertAlmostEqual(diagnostics.damage_inside_max, 0.4)
        self.assertAlmostEqual(diagnostics.damage_outside_max, 0.0)
        zero = engineering.compute_crack_region_diagnostics(
            self.geometry.material_reference_coordinates,
            self.geometry.delta,
            np.zeros_like(damage),
            np.zeros_like(history),
            np.zeros_like(strain),
            np.zeros_like(amor),
        )
        self.assertTrue(zero.localization_pass)
        forged = damage.copy()
        forged[~mask][0] = 0.9
        # Advanced indexing above intentionally creates a copy; edit an
        # explicit outside index for the negative localization diagnostic.
        outside = int(np.flatnonzero(~mask)[0])
        forged[outside] = 0.9
        failed = engineering.compute_crack_region_diagnostics(
            self.geometry.material_reference_coordinates,
            self.geometry.delta,
            forged,
            history,
            strain,
            amor,
        )
        self.assertFalse(failed.localization_pass)

        interface_forged = damage.copy()
        interface_forged[0] = 0.4
        interface_failed = engineering.compute_crack_region_diagnostics(
            self.geometry.material_reference_coordinates,
            self.geometry.delta,
            interface_forged,
            history,
            strain,
            amor,
        )
        self.assertFalse(interface_failed.localization_pass)

    def test_result_localization_is_a_final_hard_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "final crack localization"):
            engineering.run_strict_engineering_grip(
                self.model, engineering.EngineeringGripConfig.smoke()
            )


if __name__ == "__main__":
    unittest.main()
