"""Focused phase-1--4 tests for the strict unequal-cloud A2 coupling."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from dataclasses import replace

import jax
import numpy as np

from examples.fracture2d import nonlocal_a2_coupled as a2
from examples.fracture2d.nonlocal_grip_diagnostic import build_union_geometry_from_arrays


def _material_cloud(n_side: int = 3):
    axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    coordinates = np.asarray([(x, y) for x in axis for y in axis], dtype=np.float64)
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
    delta = 0.8
    pairs: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            if 0.0 < np.linalg.norm(coordinates[second] - coordinates[first]) < delta:
                pairs.extend(((first, second), (second, first)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    return coordinates, triangles_array, volumes, bottom, top, edges, delta


def _strict_config(
    *,
    opening_final: float = 1.0e-4,
    d_opening_initial: float = 1.0e-4,
    d_opening_min: float = 5.0e-5,
    max_cutbacks: int = 2,
    max_stagger_iterations: int = 20,
) -> a2.A2StrictConfig:
    gate = a2.GateTolerance(1.0e-7, 1.0e-5, 1.0)
    residual_gate = a2.GateTolerance(1.0e-9, 1.0e-7, 1.0)
    return a2.A2StrictConfig(
        opening_start=0.0,
        opening_final=opening_final,
        d_opening_initial=d_opening_initial,
        d_opening_min=d_opening_min,
        cutback_factor=0.5,
        max_cutbacks=max_cutbacks,
        max_stagger_iterations=max_stagger_iterations,
        cg_rtol=1.0e-10,
        cg_atol=1.0e-12,
        cg_maxiter=3000,
        displacement_material=gate,
        displacement_union=gate,
        damage=gate,
        history=gate,
        reaction_effective=gate,
        damage_residual=residual_gate,
        force=gate,
        moment=gate,
        opening_atol=1.0e-12,
        opening_slope_min=1.0e-6,
        healing_atol=1.0e-10,
        damage_range_atol=1.0e-10,
        work_epsilon=1.0e-5,
        work_relative_limit=1.0e-7,
        work_cg_rtol=1.0e-10,
        work_cg_atol=1.0e-13,
    )


class A2CoupledContractTest(unittest.TestCase):
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
        cls.model = a2.build_a2_coupled_model(
            cls.geometry, 2.0, 0.3, 1.0e-8, 1.0e-12, 1.7, 0.65
        )
        cls.gate = a2.GateTolerance(1.0e-7, 1.0e-5, 1.0)
        cls.residual_gate = a2.GateTolerance(1.0e-9, 1.0e-7, 1.0)
        cls.config = _strict_config()
        # Build the expensive clean-start result once.  Archive and replay
        # tests below deliberately mutate copies of this cached payload.
        cls.result = a2.run_a2_strict_continuation(cls.model, cls.config)
        cls.archive_directory = tempfile.TemporaryDirectory(prefix="a2-coupled-fixture-")
        cls.archive_path = Path(cls.archive_directory.name) / "accepted.npz"
        a2.write_a2_archive(cls.archive_path, cls.result)
        cls.archive_payload = a2.load_a2_archive(cls.archive_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.archive_directory.cleanup()
        jax.config.update("jax_enable_x64", cls.old_x64)

    def _config(self, *, max_stagger_iterations: int = 20, max_cutbacks: int = 2):
        return _strict_config(
            max_stagger_iterations=max_stagger_iterations,
            max_cutbacks=max_cutbacks,
        )

    def test_static_identity_whitelist_and_phase_volumes(self) -> None:
        static = self.model.static
        self.assertTrue(set(static.__dict__) <= static.SHARED_WHITELIST)
        self.assertEqual(static.material_prefix_count, self.geometry.n_material)
        self.assertEqual(static.n_particles, self.geometry.n_particles)
        np.testing.assert_array_equal(
            static.union_material_phase_volumes[: static.material_count],
            static.material_damage_volumes,
        )
        np.testing.assert_array_equal(
            static.union_material_phase_volumes[static.material_count :],
            0.0,
        )
        np.testing.assert_allclose(
            static.union_volumes,
            static.union_material_phase_volumes + static.grip_phase_volumes,
            rtol=0.0,
            atol=512.0 * np.finfo(np.float64).eps,
        )
        interface = (static.union_material_phase_volumes > 0.0) & (
            static.grip_phase_volumes > 0.0
        )
        self.assertGreater(np.count_nonzero(interface), 0)
        self.assertFalse(static.coordinates.flags.writeable)
        with self.assertRaises(ValueError):
            static.coordinates.setflags(write=True)
        second = a2.A2StaticMechanics.from_geometry(
            self.geometry,
            self.model.constraints,
            self.model.top_line,
            self.model.bottom_line,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
        )
        self.assertEqual(static.deterministic_digest, second.deterministic_digest)
        self.assertIsNot(static.runtime_identity, second.runtime_identity)

    def test_fresh_fixed_damage_view_matches_independent_constructor(self) -> None:
        damage_a = np.linspace(0.05, 0.55, self.geometry.n_material, dtype=np.float64)
        damage_b = np.linspace(0.15, 0.75, self.geometry.n_material, dtype=np.float64)
        view_a = a2.A2FixedDamageView(self.model.static, damage_a)
        view_b = a2.A2FixedDamageView(self.model.static, damage_b)
        independent_a = __import__(
            "examples.fracture2d.nonlocal_grip_diagnostic",
            fromlist=["FixedDamagePhaseMeasuredMechanics"],
        ).FixedDamagePhaseMeasuredMechanics(
            self.geometry, damage_a, 2.0, 0.3, 1.0e-8, 1.0e-12
        )
        rng = np.random.default_rng(100)
        q = np.asarray(rng.normal(size=self.geometry.n_dofs) * 1.0e-2, dtype=np.float64)
        direction = np.asarray(rng.normal(size=self.geometry.n_dofs), dtype=np.float64)
        for name in ("energy", "residual", "tangent_action"):
            actual = getattr(view_a, name)(q if name != "tangent_action" else direction)
            expected = getattr(independent_a, name)(q if name != "tangent_action" else direction)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-13)
        np.testing.assert_allclose(
            view_a.center_factor, independent_a.center_factor, rtol=0.0, atol=5.0e-13
        )
        self.assertIsNot(view_a.runtime_identity, view_b.runtime_identity)
        _ = view_a.tangent_block_diagonal
        _ = view_b.tangent_block_diagonal
        self.assertIsNot(view_a.tangent_block_diagonal, view_b.tangent_block_diagonal)
        with self.assertRaises(AttributeError):
            view_a.center_factor = np.ones_like(view_a.center_factor)
        with self.assertRaises(ValueError):
            view_a.material_damage.setflags(write=True)

    def test_material_source_amor_invariant_includes_grip_targets_only(self) -> None:
        view = a2.A2FixedDamageView(
            self.model.static,
            np.zeros(self.model.static.material_count, dtype=np.float64),
        )
        edge_strain = np.broadcast_to(
            np.asarray([[0.02, 0.0], [0.0, 0.01]], dtype=np.float64),
            (self.model.static.n_edges, 2, 2),
        ).copy()
        result = a2.compute_a2_amor_history_from_edge_strain(
            edge_strain,
            self.model.static.source,
            self.model.static.target,
            self.model.static.eta,
            self.model.static.material_count,
            view.lame_lambda,
            view.shear_modulus,
            np.zeros(self.model.static.material_count, dtype=np.float64),
        )
        self.assertEqual(result.particle_driver.shape, (self.geometry.n_material,))
        self.assertEqual(result.trial_history.shape, (self.geometry.n_material,))
        retained = self.model.static.source < self.geometry.n_material
        expected = np.zeros(self.geometry.n_material, dtype=np.float64)
        np.add.at(expected, self.model.static.source[retained], self.model.static.eta[retained] * result.edge_positive_energy[retained])
        np.testing.assert_allclose(result.particle_driver, expected, rtol=0.0, atol=1.0e-14)
        with self.assertRaises(ValueError):
            a2.compute_a2_amor_history_from_edge_strain(
                edge_strain,
                self.model.static.source,
                self.model.static.target,
                self.model.static.eta * 1.01,
                self.model.static.material_count,
                view.lame_lambda,
                view.shear_modulus,
                np.zeros(self.model.static.material_count, dtype=np.float64),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            a2.compute_a2_amor_history(view, np.zeros(self.geometry.n_dofs, dtype=np.float64), np.zeros(self.geometry.n_particles, dtype=np.float64))

    def test_opening_scaling_work_and_safe_seed_identity(self) -> None:
        view = a2.A2FixedDamageView(
            self.model.static,
            np.linspace(0.05, 0.35, self.geometry.n_material, dtype=np.float64),
        )
        unit = a2.solve_a2_unit_load(
            view, self.model.constraints, 1.0e-10, 1.0e-12, 3000,
            initial_guess=None, source_state="test", source_step=0, source_sweep=0,
            source_damage_digest=view.damage_digest,
        )
        np.testing.assert_array_equal(
            unit.displacement[self.model.constraints.constrained_dofs],
            self.model.constraints.prescribed_vector(1.0)[self.model.constraints.constrained_dofs],
        )
        seed = a2.build_a2_unit_seed(self.model.constraints, unit.displacement, unit.outer_displacement)
        np.testing.assert_array_equal(seed[self.model.constraints.constrained_dofs], self.model.constraints.prescribed_vector(1.0)[self.model.constraints.constrained_dofs])
        zero_seed = a2.build_a2_unit_seed(self.model.constraints, np.zeros(self.geometry.n_dofs, dtype=np.float64), 0.0)
        np.testing.assert_array_equal(zero_seed, self.model.constraints.prescribed_vector(1.0))
        result = a2.solve_a2_opening_controlled(
            view,
            self.model.constraints,
            1.0e-3,
            1.0e-10,
            1.0e-12,
            3000,
            1.0e-6,
            1.0e-12,
            1.0e-5,
            1.0e-7,
            1.0e-10,
            1.0e-13,
            initial_guess=seed,
            source_state="test",
            source_step=0,
            source_sweep=0,
            source_damage_digest=view.damage_digest,
        )
        self.assertTrue(result.opening_pass)
        self.assertTrue(result.work_pass)
        self.assertAlmostEqual(result.achieved_opening, 1.0e-3, places=12)
        self.assertGreater(result.opening_slope, 0.0)
        self.assertAlmostEqual(result.R_eff, result.R_ext / result.opening_slope, places=12)
        changed = a2.A2FixedDamageView(
            self.model.static,
            np.linspace(0.06, 0.36, self.geometry.n_material, dtype=np.float64),
        )
        with self.assertRaisesRegex(ValueError, "different fixed-damage|damage digest"):
            a2.scale_a2_unit_solution(
                unit, changed, self.model.constraints, 1.0e-3, result.opening_slope, 1.0e-10, 1.0e-12
            )

    def test_strict_one_step_zero_state_residuals_and_complete_replay(self) -> None:
        result = self.result
        self.assertEqual(result.metadata.schema, "fracture2d-a2-coupled-v1")
        self.assertFalse(result.metadata.raw_source_used)
        self.assertEqual(result.metadata.n_material, self.geometry.n_material)
        self.assertEqual(result.metadata.n_union, self.geometry.n_particles)
        initial = result.initial_state
        np.testing.assert_array_equal(initial.union_displacement, 0.0)
        np.testing.assert_array_equal(initial.material_damage, 0.0)
        np.testing.assert_array_equal(initial.material_history, 0.0)
        step = result.accepted_steps[0]
        self.assertTrue(step.diagnostics.passed)
        self.assertIsNotNone(step.diagnostics.rho_used)
        self.assertIsNotNone(step.diagnostics.rho_next)
        self.assertEqual(step.diagnostics.rho_used.shape, (self.geometry.n_material,))
        self.assertEqual(step.diagnostics.rho_next.shape, (self.geometry.n_material,))
        self.assertIn("displacement_material", step.diagnostics.gate_passes)
        self.assertIn("displacement_union", step.diagnostics.gate_passes)
        self.assertGreaterEqual(len(result.attempts[0].sweeps), 1)
        self.assertEqual(a2.independent_replay(result)["accepted_steps"], 1)
        self.assertFalse(hasattr(a2, "run_warning_continuation"))

    def test_complete_rejected_sweep_is_replayable_without_commit(self) -> None:
        tight = replace(
            self._config(max_stagger_iterations=1),
            damage=a2.GateTolerance(1.0e-12, 1.0e-12, 1.0),
        )
        committed = a2.A2CommittedState(
            0.0,
            0.0,
            np.zeros(self.model.static.n_dofs, dtype=np.float64),
            np.zeros(self.model.static.n_material, dtype=np.float64),
            np.zeros(self.model.static.n_material, dtype=np.float64),
        )
        step, attempt = a2._run_a2_attempt(
            self.model, tight, committed, 1.0e-4, 0, 1
        )
        self.assertIsNone(step)
        self.assertTrue(attempt.diagnostics.complete_sweep)
        self.assertFalse(attempt.diagnostics.passed)
        self.assertEqual(attempt.committed_before_digest, attempt.committed_after_digest)
        metadata = a2.A2ContinuationMetadata.from_model(self.model, tight)
        ba = a2._replay_ba(
            metadata.union_coordinates,
            metadata.union_volumes,
            metadata.visible_edges,
            metadata.delta,
            metadata.rank_rtol,
        )
        a2._replay_complete_diagnostics(
            attempt.diagnostics,
            committed.material_damage,
            committed.material_history,
            metadata,
            ba,
            committed.union_displacement,
            committed.outer_displacement,
            "committed",
            a2._damage_digest(committed.material_damage),
            1,
        )

    def test_incomplete_failure_rolls_back_and_cutback_is_strict(self) -> None:
        original = a2.solve_a2_opening_controlled
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("manufactured incomplete A2 mechanics failure")
            return original(*args, **kwargs)

        with mock.patch.object(a2, "solve_a2_opening_controlled", side_effect=fail_once):
            result = a2.run_a2_strict_continuation(self.model, self._config())
        rejected = result.attempts[0]
        self.assertFalse(rejected.accepted)
        self.assertFalse(rejected.complete_sweep)
        self.assertEqual(rejected.committed_before_digest, rejected.committed_after_digest)
        self.assertIsNone(rejected.diagnostics.H_next)
        self.assertIsNone(rejected.diagnostics.rho_next)
        self.assertEqual(result.accepted_steps[0].cutback_count, 1)
        self.assertEqual(result.final_state.physical_opening, 1.0e-4)

    def test_actual_healing_rejection_with_nonzero_committed_damage_is_typed(self) -> None:
        committed = replace(
            self.result.final_state,
            material_damage=np.full(
                self.model.static.material_count, 0.5, dtype=np.float64
            ),
            material_history=np.zeros(
                self.model.static.material_count, dtype=np.float64
            ),
        )
        self.assertGreater(float(np.max(committed.material_damage)), 0.0)
        config = _strict_config(opening_final=2.0e-4)
        step, attempt = a2._run_a2_attempt(
            self.model, config, committed, 2.0e-4, 1, 2
        )
        self.assertIsNone(step)
        self.assertFalse(attempt.accepted)
        self.assertEqual(attempt.exception_type, a2.A2SweepFailure.__name__)
        self.assertEqual(attempt.failure_stage, "damage_acceptance")
        self.assertEqual(attempt.failure_reason, "healing")
        self.assertEqual(attempt.cutback_count, 1)
        self.assertEqual(attempt.committed_before_digest, committed.digest)
        self.assertEqual(attempt.committed_after_digest, committed.digest)
        self.assertEqual(
            attempt.diagnostics.damage_before.tobytes(),
            committed.material_damage.tobytes(),
        )
        np.testing.assert_array_equal(attempt.diagnostics.d_iter, committed.material_damage)
        self.assertIsNone(attempt.diagnostics.H_next)
        self.assertIsNone(attempt.diagnostics.rho_next)
        self.assertIsNotNone(attempt.diagnostics.seed_before)
        self.assertIsNone(attempt.diagnostics.seed_after)
        self.assertIsNotNone(attempt.diagnostics.d_next)
        self.assertIsNotNone(attempt.diagnostics.rho_used)
        metadata = a2.A2ContinuationMetadata.from_model(self.model, config)
        ba = a2._replay_ba(
            metadata.union_coordinates,
            metadata.union_volumes,
            metadata.visible_edges,
            metadata.delta,
            metadata.rank_rtol,
        )
        serialized = a2._diag_namespace(
            json.loads(json.dumps(a2._json_ready(attempt.diagnostics), sort_keys=True))
        )
        a2._replay_incomplete_diagnostics(
            serialized,
            committed.material_damage,
            committed.material_history,
            metadata,
            ba,
            committed.union_displacement,
            committed.outer_displacement,
            "committed",
            a2._damage_digest(committed.material_damage),
            2,
        )
        forged = a2._diag_namespace(
            json.loads(
                json.dumps(
                    a2._json_ready(replace(attempt.diagnostics, failure_reason="damage_range")),
                    sort_keys=True,
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "damage-gate failure reason"):
            a2._replay_incomplete_diagnostics(
                forged,
                committed.material_damage,
                committed.material_history,
                metadata,
                ba,
                committed.union_displacement,
                committed.outer_displacement,
                "committed",
                a2._damage_digest(committed.material_damage),
                2,
            )

        forged_flag = a2._diag_namespace(
            json.loads(
                json.dumps(
                    a2._json_ready(
                        replace(attempt.diagnostics, work_after_pass=True)
                    ),
                    sort_keys=True,
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "work_after_pass"):
            a2._replay_incomplete_diagnostics(
                forged_flag,
                committed.material_damage,
                committed.material_history,
                metadata,
                ba,
                committed.union_displacement,
                committed.outer_displacement,
                "committed",
                a2._damage_digest(committed.material_damage),
                2,
            )

    def test_actual_work_gate_rejection_cutbacks_without_seed_commit(self) -> None:
        config = replace(
            _strict_config(opening_final=1.0e-4),
            # The independently replayed work error at this deliberately
            # larger diagnostic target exceeds this tighter, finite limit,
            # while the preceding accepted step remains below it.
            work_relative_limit=5.0e-14,
        )
        committed = self.result.final_state
        step, rejected = a2._run_a2_attempt(
            self.model, config, committed, 5.0e-3, 0, 2
        )
        self.assertIsNone(step)
        self.assertFalse(rejected.accepted)
        self.assertFalse(rejected.diagnostics.complete_sweep)
        self.assertEqual(rejected.exception_type, a2.A2SweepFailure.__name__)
        self.assertEqual(rejected.failure_stage, "mechanics_before_gate")
        self.assertEqual(rejected.failure_reason, "mechanics-before work gate failed")
        self.assertEqual(rejected.committed_before_digest, committed.digest)
        self.assertEqual(rejected.committed_after_digest, committed.digest)
        self.assertEqual(rejected.cutback_count, 0)
        self.assertEqual(rejected.diagnostics.damage_before.tobytes(), committed.material_damage.tobytes())
        self.assertIsNone(rejected.diagnostics.seed_after)
        retry_step, retry_attempt = a2._run_a2_attempt(
            self.model,
            _strict_config(opening_final=2.0e-4),
            committed,
            1.5e-4,
            1,
            2,
        )
        self.assertIsNotNone(retry_step)
        self.assertTrue(retry_attempt.accepted)
        assert retry_step is not None
        self.assertEqual(retry_step.cutback_count, 1)
        retry_seed = retry_step.diagnostics.seed_before
        self.assertIsNotNone(retry_seed)
        assert retry_seed is not None
        self.assertEqual(retry_seed.source_state, "committed")
        self.assertEqual(
            retry_seed.source_damage_digest,
            a2._damage_digest(committed.material_damage),
        )
        accepted_diag = replace(
            self.result.accepted_steps[0].diagnostics,
            work_limit=config.work_relative_limit,
        )
        accepted_step = replace(
            self.result.accepted_steps[0], diagnostics=accepted_diag
        )
        accepted_attempt = replace(
            self.result.attempts[0],
            sweeps=(accepted_diag,),
            diagnostics=accepted_diag,
        )
        archive_result = a2.A2ContinuationResult(
            initial_state=self.result.initial_state,
            metadata=a2.A2ContinuationMetadata.from_model(self.model, config),
            states=(self.result.initial_state, self.result.final_state),
            accepted_steps=(accepted_step,),
            attempts=(accepted_attempt, rejected),
            model=self.model,
            acceptance_mode="strict",
            acceptance_status="strict_passed",
            completion_status="completed",
        )
        gate_archive = self._archive_path("work-gate-rejected.npz")
        a2.write_a2_archive(gate_archive, archive_result)
        self.assertTrue(a2.replay_a2_archive(gate_archive)["replayed"])
        forged_payload = a2.load_a2_archive(gate_archive)
        forged_payload["attempt_failure_reason"][1] = "forged gate reason"
        forged_archive = self._archive_path("forged-work-gate-reason.npz")
        np.savez_compressed(forged_archive, **forged_payload)
        with self.assertRaisesRegex(ValueError, "failure classification|reason"):
            a2.replay_a2_archive(forged_archive)

    def test_actual_opening_gate_rejection_replays_off_target_mechanics(self) -> None:
        config = _strict_config(opening_final=1.0e-4)
        committed = self.result.final_state
        target = 2.0e-4
        offset = 1.0e-4
        original_solver = a2.solve_a2_opening_controlled
        returned: list[a2.A2OpeningMechanicsResult] = []

        def solve_at_off_target(*args, **kwargs):
            forwarded = list(args)
            forwarded[2] = float(forwarded[2]) + offset
            mechanics = original_solver(*forwarded, **kwargs)
            returned.append(mechanics)
            return mechanics

        with mock.patch.object(
            a2, "solve_a2_opening_controlled", side_effect=solve_at_off_target
        ):
            step, rejected = a2._run_a2_attempt(
                self.model, config, committed, target, 0, 2
            )

        self.assertIsNone(step)
        self.assertFalse(rejected.accepted)
        self.assertFalse(rejected.diagnostics.complete_sweep)
        self.assertEqual(rejected.exception_type, a2.A2SweepFailure.__name__)
        self.assertEqual(rejected.failure_stage, "mechanics_before_gate")
        self.assertEqual(rejected.failure_reason, "mechanics-before opening gate failed")
        self.assertEqual(rejected.committed_before_digest, committed.digest)
        self.assertEqual(rejected.committed_after_digest, committed.digest)
        self.assertEqual(rejected.cutback_count, 0)
        self.assertEqual(len(returned), 1)
        off_target = returned[0]
        # The test double returns a real equilibrated solve at target+offset;
        # its own opening flag is therefore true.  The production gate must
        # compare the retained achieved opening with the attempted target.
        self.assertTrue(off_target.opening_pass)
        self.assertEqual(off_target.target_opening, target + offset)
        diagnostics = rejected.diagnostics
        self.assertEqual(diagnostics.target_opening, target)
        self.assertEqual(
            diagnostics.mechanics_target_opening_before, target + offset
        )
        self.assertAlmostEqual(
            diagnostics.achieved_opening_before, target + offset, places=12
        )
        self.assertFalse(diagnostics.opening_before_pass)
        self.assertTrue(diagnostics.slope_before_pass)
        self.assertTrue(diagnostics.work_before_pass)
        self.assertTrue(diagnostics.cg_before_pass)
        self.assertIsNotNone(diagnostics.q_before)
        self.assertIsNotNone(diagnostics.seed_before)
        self.assertIsNone(diagnostics.seed_after)
        self.assertIsNone(diagnostics.H_next)
        self.assertIsNone(diagnostics.rho_next)

        metadata = a2.A2ContinuationMetadata.from_model(self.model, config)
        ba = a2._replay_ba(
            metadata.union_coordinates,
            metadata.union_volumes,
            metadata.visible_edges,
            metadata.delta,
            metadata.rank_rtol,
        )
        a2._replay_incomplete_diagnostics(
            diagnostics,
            committed.material_damage,
            committed.material_history,
            metadata,
            ba,
            committed.union_displacement,
            committed.outer_displacement,
            "committed",
            a2._damage_digest(committed.material_damage),
            2,
        )
        omitted_target = replace(
            diagnostics,
            mechanics_target_opening_before=None,
            available_fields=tuple(
                name
                for name in diagnostics.available_fields
                if name != "mechanics_target_opening_before"
            ),
        )
        with self.assertRaisesRegex(ValueError, "target provenance"):
            a2._replay_incomplete_diagnostics(
                omitted_target,
                committed.material_damage,
                committed.material_history,
                metadata,
                ba,
                committed.union_displacement,
                committed.outer_displacement,
                "committed",
                a2._damage_digest(committed.material_damage),
                2,
            )

        accepted_diag = self.result.accepted_steps[0].diagnostics
        accepted_step = self.result.accepted_steps[0]
        accepted_attempt = self.result.attempts[0]
        archive_result = a2.A2ContinuationResult(
            initial_state=self.result.initial_state,
            metadata=metadata,
            states=(self.result.initial_state, self.result.final_state),
            accepted_steps=(accepted_step,),
            attempts=(accepted_attempt, rejected),
            model=self.model,
            acceptance_mode="strict",
            acceptance_status="strict_passed",
            completion_status="completed",
        )
        archive = self._archive_path("opening-gate-rejected.npz")
        a2.write_a2_archive(archive, archive_result)
        self.assertTrue(a2.replay_a2_archive(archive)["replayed"])

        forged_reason = a2.load_a2_archive(archive)
        forged_reason["attempt_failure_reason"][1] = "forged opening reason"
        forged_reason_path = self._archive_path("forged-opening-reason.npz")
        np.savez_compressed(forged_reason_path, **forged_reason)
        with self.assertRaisesRegex(ValueError, "failure classification|reason"):
            a2.replay_a2_archive(forged_reason_path)

        forged_pass = a2.load_a2_archive(archive)
        sweep_payload = json.loads(str(forged_pass["attempt_sweeps_json"][1]))
        sweep_payload[0]["opening_before_pass"] = True
        forged_pass["attempt_sweeps_json"][1] = json.dumps(
            sweep_payload, sort_keys=True
        )
        forged_pass_path = self._archive_path("forged-opening-pass.npz")
        np.savez_compressed(forged_pass_path, **forged_pass)
        with self.assertRaisesRegex(ValueError, "opening_before_pass|primitive gates"):
            a2.replay_a2_archive(forged_pass_path)

    def test_configured_work_gate_failure_is_typed_and_rolls_back(self) -> None:
        config = replace(self.config, work_relative_limit=0.0)
        committed = self.result.initial_state
        step, attempt = a2._run_a2_attempt(
            self.model, config, committed, 1.0e-4, 0, 1
        )
        self.assertIsNone(step)
        self.assertFalse(attempt.accepted)
        self.assertFalse(attempt.complete_sweep)
        self.assertEqual(attempt.exception_type, a2.A2SweepFailure.__name__)
        self.assertEqual(attempt.failure_stage, "mechanics_before_gate")
        self.assertEqual(attempt.failure_reason, "mechanics-before work gate failed")
        self.assertEqual(attempt.committed_before_digest, committed.digest)
        self.assertEqual(attempt.committed_after_digest, committed.digest)
        self.assertIsNotNone(attempt.diagnostics.q_before)
        self.assertIsNotNone(attempt.diagnostics.seed_before)
        self.assertIsNone(attempt.diagnostics.seed_after)
        self.assertIsNone(attempt.diagnostics.H_next)
        self.assertIsNone(attempt.diagnostics.rho_next)
        metadata = a2.A2ContinuationMetadata.from_model(self.model, config)
        ba = a2._replay_ba(
            metadata.union_coordinates,
            metadata.union_volumes,
            metadata.visible_edges,
            metadata.delta,
            metadata.rank_rtol,
        )
        a2._replay_incomplete_diagnostics(
            attempt.diagnostics,
            committed.material_damage,
            committed.material_history,
            metadata,
            ba,
            committed.union_displacement,
            committed.outer_displacement,
            "committed",
            a2._damage_digest(committed.material_damage),
            1,
        )
        with self.assertRaisesRegex(ValueError, "reason does not match replay|forged"):
            a2._replay_incomplete_diagnostics(
                replace(attempt.diagnostics, failure_reason="forged"),
                committed.material_damage,
                committed.material_history,
                metadata,
                ba,
                committed.union_displacement,
                committed.outer_displacement,
                "committed",
                a2._damage_digest(committed.material_damage),
                1,
            )

    def test_structural_solver_errors_propagate_instead_of_cutting_back(self) -> None:
        committed = self.result.initial_state
        with mock.patch.object(
            a2,
            "solve_a2_opening_controlled",
            side_effect=ValueError("structural A2 solver contract error"),
        ):
            with self.assertRaisesRegex(ValueError, "structural A2 solver contract"):
                a2._run_a2_attempt(
                    self.model, self.config, committed, 1.0e-4, 0, 1
                )

    def test_compact_archive_replay_and_forged_field_rejection(self) -> None:
        path = self.archive_path
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(str(archive["archive_schema"]), "fracture2d-a2-coupled-v1")
            self.assertEqual(str(archive["acceptance_mode"]), "strict")
            self.assertEqual(str(archive["acceptance_status"]), "strict_passed")
            self.assertEqual(str(archive["completion_status"]), "completed")
            self.assertTrue(np.all(archive["accepted_acceptance_mode"] == "strict"))
            self.assertTrue(np.all(archive["accepted_acceptance_status"] == "strict_passed"))
            self.assertTrue(np.all(archive["attempt_acceptance_mode"] == "strict"))
            self.assertFalse(bool(archive["raw_source_used"]))
            self.assertIn("union_material_phase_volumes", archive.files)
            self.assertIn("union_grip_phase_volumes", archive.files)
            self.assertIn("material_prefix_map", archive.files)
            for name in archive.files:
                self.assertNotEqual(archive[name].dtype.kind, "O")
        replay = a2.replay_a2_archive(path)
        self.assertTrue(replay["replayed"])

        payload = {name: np.array(value, copy=True) for name, value in self.archive_payload.items()}
        forged = np.array(payload["accepted_rho_next"], copy=True)
        forged[0, 0] += 1.0e-3
        payload["accepted_rho_next"] = forged
        forged_path = Path(self.archive_directory.name) / "forged-rho.npz"
        np.savez_compressed(forged_path, **payload)
        with self.assertRaisesRegex(ValueError, "accepted rho_next|rho_next"):
            a2.replay_a2_archive(forged_path)

    def test_strict_mode_status_and_completion_are_required(self) -> None:
        self.assertEqual(self.result.acceptance_mode, "strict")
        self.assertEqual(self.result.acceptance_status, "strict_passed")
        self.assertEqual(self.result.completion_status, "completed")
        for field, value in (
            ("acceptance_mode", "warning"),
            ("acceptance_status", "warning_accepted"),
            ("completion_status", "truncated"),
            ("accepted_acceptance_status", "warning_accepted"),
            ("attempt_acceptance_status", "warning_accepted"),
        ):
            with self.subTest(field=field):
                self._assert_archive_forgery_rejected(
                    f"forged-result-{field}.npz",
                    lambda payload, field=field, value=value: payload[field].__setitem__((), value),
                )
        for field in ("acceptance_mode", "acceptance_status", "completion_status"):
            with self.subTest(missing=field):
                payload = self._archive_copy()
                del payload[field]
                path = self._archive_path(f"missing-result-{field}.npz")
                np.savez_compressed(path, **payload)
                with self.assertRaises(ValueError):
                    a2.replay_a2_archive(path)

        truncated = self._archive_copy()
        truncated["state_physical_opening"][-1] = self.result.states[-2].physical_opening
        truncated_path = self._archive_path("truncated-final-state.npz")
        np.savez_compressed(truncated_path, **truncated)
        with self.assertRaisesRegex(ValueError, "final opening|opening_final"):
            a2.replay_a2_archive(truncated_path)

        config = json.loads(str(self.archive_payload["config_json"]))
        for forged in ("warning", "warning_accepted", None):
            payload = dict(config)
            if forged is None:
                del payload["acceptance_mode"]
            else:
                payload["acceptance_mode"] = forged
            with self.subTest(config_mode=forged):
                with self.assertRaises(ValueError):
                    a2._config_from_payload(payload)

    def _archive_copy(self) -> dict[str, np.ndarray]:
        return {
            name: np.array(value, copy=True) for name, value in self.archive_payload.items()
        }

    def _archive_path(self, name: str) -> Path:
        return Path(self.archive_directory.name) / name

    def _assert_archive_forgery_rejected(self, name: str, mutate) -> None:
        payload = self._archive_copy()
        mutate(payload)
        path = self._archive_path(name)
        np.savez_compressed(path, **payload)
        with self.assertRaises(ValueError):
            a2.replay_a2_archive(path)

    @staticmethod
    def _mutate_diagnostic_json(
        payload: dict[str, np.ndarray], field: str, mutate
    ) -> None:
        records = [json.loads(str(value)) for value in payload["accepted_diagnostics_json"]]
        mutate(records[0], field)
        payload["accepted_diagnostics_json"] = np.asarray(
            [json.dumps(record, sort_keys=True) for record in records], dtype=np.str_
        )

    @staticmethod
    def _mutate_attempt_sweep_json(
        payload: dict[str, np.ndarray], mutate
    ) -> None:
        records = [json.loads(str(value)) for value in payload["attempt_sweeps_json"]]
        mutate(records[0][0])
        payload["attempt_sweeps_json"] = np.asarray(
            [json.dumps(record, sort_keys=True) for record in records], dtype=np.str_
        )

    def test_accepted_archive_primitive_and_diagnostic_forgeries_are_rejected(self) -> None:
        # These are intentionally separate payloads: each accepted duplicate is
        # independently authoritative and must not be bypassed by changing a
        # different copy of the same sweep.
        array_fields = (
            "accepted_target_opening",
            "accepted_outer_displacement",
            "accepted_H_used",
            "accepted_rho_used",
            "accepted_rho_used_volume",
            "accepted_damage_before",
            "accepted_d_iter",
            "accepted_q_before",
            "accepted_Psi_before",
            "accepted_Psi_after",
            "accepted_d_next",
            "accepted_q_after",
            "accepted_H_next",
            "accepted_rho_next",
            "accepted_rho_next_volume",
        )
        for index, field in enumerate(array_fields):
            def mutate(payload, field=field):
                if payload[field].ndim == 1:
                    payload[field][0] += 1.0e-5
                else:
                    payload[field][0].flat[0] += 1.0e-5

            with self.subTest(field=field):
                self._assert_archive_forgery_rejected(
                    f"forged-accepted-{index}.npz", mutate
                )

        scalar_fields = (
            "U_before",
            "U_after",
            "slope_before",
            "slope_after",
            "R_ext_before",
            "R_ext_after",
            "R_eff_before",
            "R_eff_after",
        )
        for index, field in enumerate(scalar_fields):
            def mutate(payload, field=field):
                self._mutate_diagnostic_json(
                    payload,
                    field,
                    lambda record, name: record.__setitem__(name, float(record[name]) + 1.0e-5),
                )

            with self.subTest(field=field):
                self._assert_archive_forgery_rejected(
                    f"forged-accepted-scalar-{index}.npz", mutate
                )

        self._assert_archive_forgery_rejected(
            "forged-diagnostics-json.npz",
            lambda payload: self._mutate_diagnostic_json(
                payload,
                "damage_residual_pass",
                lambda record, name: record.__setitem__(name, False),
            ),
        )
        self._assert_archive_forgery_rejected(
            "forged-sweep-json.npz",
            lambda payload: self._mutate_attempt_sweep_json(
                payload,
                lambda record: record.__setitem__("rho_next_inf", float(record["rho_next_inf"]) + 1.0e-3),
            ),
        )

    def test_attempt_target_cutback_status_and_failure_fields_are_authoritative(self) -> None:
        def mutate_target(payload):
            payload["attempt_load_target"][0] += 1.0e-5

        def mutate_load_start(payload):
            payload["attempt_load_start"][0] += 1.0e-5

        def mutate_cutback(payload):
            payload["attempt_cutback_count"][0] += 1

        def mutate_status(payload):
            payload["attempt_accepted"][0] = False

        def mutate_before(payload):
            payload["attempt_before_digest"][0] = "0" * 64

        def mutate_after(payload):
            payload["attempt_after_digest"][0] = "0" * 64

        def mutate_failure_stage(payload):
            payload["attempt_failure_stage"][0] = "forged_stage"

        def mutate_failure_reason(payload):
            payload["attempt_failure_reason"][0] = "forged_reason"

        def mutate_exception(payload):
            payload["attempt_exception_type"][0] = "RuntimeError"

        for name, mutate in (
            ("target", mutate_target),
            ("load-start", mutate_load_start),
            ("cutback", mutate_cutback),
            ("status", mutate_status),
            ("before", mutate_before),
            ("after", mutate_after),
            ("failure-stage", mutate_failure_stage),
            ("failure-reason", mutate_failure_reason),
            ("exception", mutate_exception),
        ):
            with self.subTest(field=name):
                self._assert_archive_forgery_rejected(
                    f"forged-attempt-{name}.npz", mutate
                )

    def test_incomplete_archive_failure_fields_and_cutback_identity_are_rejected(self) -> None:
        committed = self.result.initial_state
        with mock.patch.object(
            a2,
            "solve_a2_opening_controlled",
            side_effect=RuntimeError("manufactured mechanics-before failure"),
        ):
            _, attempt = a2._run_a2_attempt(
                self.model, self.config, committed, 1.0e-4, 0, 1
            )
        accepted_step = replace(self.result.accepted_steps[0], cutback_count=1)
        accepted_attempt = replace(self.result.attempts[0], cutback_count=1)
        failure_result = a2.A2ContinuationResult(
            initial_state=self.result.initial_state,
            metadata=a2.A2ContinuationMetadata.from_model(self.model, self.config),
            states=self.result.states,
            accepted_steps=(accepted_step,),
            attempts=(attempt, accepted_attempt),
            model=self.model,
            acceptance_mode="strict",
            acceptance_status="strict_passed",
            completion_status="completed",
        )
        path = self._archive_path("incomplete-fixture.npz")
        a2.write_a2_archive(path, failure_result)
        base_payload = a2.load_a2_archive(path)

        def assert_forged(mutate, name):
            payload = {key: np.array(value, copy=True) for key, value in base_payload.items()}
            mutate(payload)
            forged_path = self._archive_path(name)
            np.savez_compressed(forged_path, **payload)
            with self.assertRaises(ValueError):
                a2.replay_a2_archive(forged_path)

        assert_forged(
            lambda payload: payload["attempt_exception_type"].__setitem__(0, ""),
            "forged-incomplete-exception.npz",
        )
        assert_forged(
            lambda payload: payload["attempt_exception_type"].__setitem__(0, "RuntimeError"),
            "forged-incomplete-gate-exception-type.npz",
        )
        assert_forged(
            lambda payload: payload["attempt_cutback_count"].__setitem__(0, 1),
            "forged-incomplete-cutback.npz",
        )
        assert_forged(
            lambda payload: payload["attempt_status"].__setitem__(0, True)
            if "attempt_status" in payload
            else payload["attempt_accepted"].__setitem__(0, True),
            "forged-incomplete-status.npz",
        )
        assert_forged(
            lambda payload: payload["attempt_failure_reason"].__setitem__(0, "forged"),
            "forged-incomplete-reason.npz",
        )

        def forge_stage(payload):
            payload["attempt_failure_stage"][0] = "amor_before"
            records = [json.loads(str(value)) for value in payload["attempt_sweeps_json"]]
            records[0][0]["failure_stage"] = "amor_before"
            payload["attempt_sweeps_json"] = np.asarray(
                [json.dumps(record, sort_keys=True) for record in records], dtype=np.str_
            )

        assert_forged(forge_stage, "forged-incomplete-stage.npz")

    def test_seed_source_target_digests_and_stale_unit_system_identity(self) -> None:
        step = self.result.accepted_steps[0]
        seed_before = step.diagnostics.seed_before
        seed_after = step.diagnostics.seed_after
        self.assertIsNotNone(seed_before)
        self.assertIsNotNone(seed_after)
        assert seed_before is not None and seed_after is not None
        self.assertNotEqual(
            seed_after.source_damage_digest, seed_after.target_damage_digest
        )

        for seed_name, digest_field in (
            ("seed_before", "source_damage_digest"),
            ("seed_before", "target_damage_digest"),
            ("seed_after", "source_damage_digest"),
            ("seed_after", "target_damage_digest"),
            ("seed_after", "unit_system_digest"),
        ):
            def mutate(payload, seed_name=seed_name, digest_field=digest_field):
                def edit(record):
                    record[seed_name][digest_field] = "0" * 64

                self._mutate_attempt_sweep_json(payload, edit)

            with self.subTest(seed=seed_name, field=digest_field):
                self._assert_archive_forgery_rejected(
                    f"forged-seed-{seed_name}-{digest_field}.npz", mutate
                )

        view = a2.A2FixedDamageView(self.model.static, step.damage_before)
        unit = a2.solve_a2_unit_load(
            view,
            self.model.constraints,
            self.config.cg_rtol,
            self.config.cg_atol,
            self.config.cg_maxiter,
            initial_guess=None,
            source_state="test",
            source_step=0,
            source_sweep=0,
            source_damage_digest=view.damage_digest,
        )
        forged_unit = copy.copy(unit)
        object.__setattr__(forged_unit, "unit_system_digest", "0" * 64)
        with self.assertRaisesRegex(ValueError, "unit-system|unit system"):
            a2.scale_a2_unit_solution(
                forged_unit,
                view,
                self.model.constraints,
                step.mechanics_before.target_opening,
                step.mechanics_before.opening_slope,
                self.config.cg_rtol,
                self.config.cg_atol,
            )
        forged_seed = copy.copy(unit.seed_provenance)
        object.__setattr__(forged_seed, "unit_system_digest", "0" * 64)
        forged_unit_seed = copy.copy(unit)
        object.__setattr__(forged_unit_seed, "seed_provenance", forged_seed)
        with self.assertRaisesRegex(ValueError, "seed unit-system"):
            a2.scale_a2_unit_solution(
                forged_unit_seed,
                view,
                self.model.constraints,
                step.mechanics_before.target_opening,
                step.mechanics_before.opening_slope,
                self.config.cg_rtol,
                self.config.cg_atol,
            )

    def test_exact_512eps_residual_reconstruction_threshold_for_strong_and_volume(self) -> None:
        damage = np.asarray([0.25], dtype=np.float64)
        history = np.asarray([3.0], dtype=np.float64)
        matrix = np.asarray([[1.5]], dtype=np.float64)
        volumes = np.asarray([2.0], dtype=np.float64)
        Gc = 1.7
        l0 = 0.65
        local = Gc / l0 * damage
        graph = Gc * l0 * (matrix @ damage)
        history_damage = 2.0 * history * damage
        history_constant = 2.0 * history
        expected = local + graph + history_damage - history_constant
        bound = 512.0 * np.finfo(np.float64).eps * (
            np.abs(local) + np.abs(graph) + np.abs(history_damage) + np.abs(history_constant)
        )
        strong_at_bound = expected + bound
        volume_at_bound = volumes * expected + volumes * bound
        a2._compare_residual_reconstruction(
            strong_at_bound,
            volume_at_bound,
            damage,
            history,
            matrix,
            volumes,
            Gc,
            l0,
            "threshold",
        )
        with self.assertRaisesRegex(ValueError, "512eps"):
            a2._compare_residual_reconstruction(
                np.nextafter(expected + bound, np.inf),
                volume_at_bound,
                damage,
                history,
                matrix,
                volumes,
                Gc,
                l0,
                "strong-above-threshold",
            )
        with self.assertRaisesRegex(ValueError, "512eps"):
            a2._compare_residual_reconstruction(
                strong_at_bound,
                np.nextafter(volumes * expected + volumes * bound, np.inf),
                damage,
                history,
                matrix,
                volumes,
                Gc,
                l0,
                "volume-above-threshold",
            )

    def test_frozen_first_pilot_configuration_is_exact_and_strict(self) -> None:
        gate = a2.GateTolerance(1.0e-9, 1.0e-8, 1.0)
        residual_gate = a2.GateTolerance(1.0e-10, 1.0e-8, 1.0)
        pilot = a2.A2StrictConfig(
            opening_start=0.0,
            opening_final=1.0e-4,
            d_opening_initial=1.0e-4,
            d_opening_min=5.0e-5,
            cutback_factor=0.5,
            max_cutbacks=8,
            max_stagger_iterations=20,
            cg_rtol=1.0e-10,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
            displacement_material=gate,
            displacement_union=gate,
            damage=gate,
            history=gate,
            reaction_effective=gate,
            damage_residual=residual_gate,
            force=gate,
            moment=gate,
            opening_atol=1.0e-12,
            opening_slope_min=1.0e-6,
            healing_atol=1.0e-10,
            damage_range_atol=1.0e-10,
            work_epsilon=1.0e-5,
            work_relative_limit=1.0e-7,
            work_cg_rtol=1.0e-10,
            work_cg_atol=1.0e-13,
        )
        self.assertEqual(pilot.acceptance_mode, "strict")
        self.assertEqual(pilot.opening_start, 0.0)
        self.assertEqual(pilot.opening_final, 1.0e-4)
        self.assertEqual(pilot.d_opening_initial, 1.0e-4)
        self.assertEqual(pilot.d_opening_min, 5.0e-5)
        self.assertEqual(pilot.cutback_factor, 0.5)
        self.assertEqual(pilot.max_cutbacks, 8)
        self.assertEqual(pilot.max_stagger_iterations, 20)
        self.assertEqual((pilot.cg_rtol, pilot.cg_atol, pilot.cg_maxiter), (1.0e-10, 1.0e-13, 3000))
        self.assertEqual(
            (pilot.cg_rtol, pilot.cg_atol),
            (pilot.work_cg_rtol, pilot.work_cg_atol),
        )
        for name in (
            "displacement_material",
            "displacement_union",
            "damage",
            "history",
            "reaction_effective",
            "force",
            "moment",
        ):
            self.assertEqual(getattr(pilot, name), gate)
        self.assertEqual(pilot.damage_residual, residual_gate)
        self.assertEqual(
            (
                pilot.opening_atol,
                pilot.opening_slope_min,
                pilot.healing_atol,
                pilot.damage_range_atol,
                pilot.work_epsilon,
                pilot.work_relative_limit,
                pilot.work_cg_rtol,
                pilot.work_cg_atol,
            ),
            (1.0e-12, 1.0e-6, 1.0e-10, 1.0e-10, 1.0e-5, 1.0e-7, 1.0e-10, 1.0e-13),
        )

    def _full_diagnostics_for(self, config, before, after):
        step = self.result.accepted_steps[0]
        return a2._full_diagnostics(
            config,
            before.target_opening,
            self.result.initial_state,
            step.d_iter,
            step.d_next,
            step.H_used,
            step.H_next,
            before,
            after,
            step.diagnostics.rho_used,
            step.diagnostics.rho_next,
            step.diagnostics.rho_used_volume,
            step.diagnostics.rho_next_volume,
            step.diagnostics.damage_residual_scale,
            step.Psi_before,
            step.Psi_after,
            a2._model_geometry_length(self.model),
            before.unit_result.seed_provenance,
            after.unit_result.seed_provenance,
            0,
        )

    def _loose_config(self, **changes):
        loose = a2.GateTolerance(1.0e6, 1.0e6, 1.0)
        values = {
            "displacement_material": loose,
            "displacement_union": loose,
            "damage": loose,
            "history": loose,
            "reaction_effective": loose,
            "damage_residual": loose,
            "force": loose,
            "moment": loose,
        }
        values.update(changes)
        return replace(self.config, **values)

    def test_unequal_opening_slopes_gate_reaction_by_R_eff(self) -> None:
        base = self.result.accepted_steps[0].mechanics_before
        forged_mechanics = replace(
            base.mechanics_result,
            outer_reaction=2.0 * base.R_ext,
        )
        forged_after = replace(
            base,
            mechanics_result=forged_mechanics,
            opening_slope=2.0 * base.opening_slope,
            effective_reaction=base.R_eff,
        )
        config = self._loose_config(
            reaction_effective=a2.GateTolerance(0.0, 0.0, 1.0)
        )
        diagnostics = self._full_diagnostics_for(config, base, forged_after)
        self.assertNotEqual(base.R_ext, forged_after.R_ext)
        self.assertNotEqual(base.opening_slope, forged_after.opening_slope)
        self.assertEqual(base.R_eff, forged_after.R_eff)
        self.assertTrue(diagnostics.reaction_effective_pass)

    def test_configured_force_and_moment_gates_are_authoritative(self) -> None:
        step = self.result.accepted_steps[0]
        base_before = step.mechanics_before
        base_after = step.mechanics_after
        force_before = replace(
            base_before.mechanics_result,
            total_force=np.asarray([1.0, 0.0], dtype=np.float64),
        )
        force_after = replace(
            base_after.mechanics_result,
            total_force=np.asarray([1.0, 0.0], dtype=np.float64),
        )
        force_before_result = replace(base_before, mechanics_result=force_before)
        force_after_result = replace(base_after, mechanics_result=force_after)
        force_config = self._loose_config(
            force=a2.GateTolerance(0.0, 0.0, 1.0)
        )
        force_diagnostics = self._full_diagnostics_for(
            force_config, force_before_result, force_after_result
        )
        self.assertFalse(force_diagnostics.force_pass)
        self.assertFalse(force_diagnostics.passed)

        moment_before = replace(base_before.mechanics_result, total_moment=1.0)
        moment_after = replace(base_after.mechanics_result, total_moment=1.0)
        moment_before_result = replace(base_before, mechanics_result=moment_before)
        moment_after_result = replace(base_after, mechanics_result=moment_after)
        moment_config = self._loose_config(
            moment=a2.GateTolerance(0.0, 0.0, 1.0)
        )
        moment_diagnostics = self._full_diagnostics_for(
            moment_config, moment_before_result, moment_after_result
        )
        self.assertFalse(moment_diagnostics.moment_pass)
        self.assertFalse(moment_diagnostics.passed)

    def test_force_and_moment_before_after_scales_are_independent(self) -> None:
        step = self.result.accepted_steps[0]
        geometry_length = a2._model_geometry_length(self.model)
        before_mechanics = replace(
            step.mechanics_before.mechanics_result,
            outer_reaction=10.0,
            bottom_reaction=3.0,
            anchor_reaction=2.0,
            total_force=np.asarray([0.05, 0.0], dtype=np.float64),
            total_moment=0.5,
        )
        after_mechanics = replace(
            step.mechanics_after.mechanics_result,
            outer_reaction=1.0,
            bottom_reaction=2.0,
            anchor_reaction=3.0,
            total_force=np.asarray([0.2, 0.0], dtype=np.float64),
            total_moment=1.1,
        )
        before = replace(step.mechanics_before, mechanics_result=before_mechanics)
        after = replace(step.mechanics_after, mechanics_result=after_mechanics)
        config = self._loose_config(
            force=a2.GateTolerance(0.1, 0.0, 1.0),
            moment=a2.GateTolerance(1.0, 0.0, 1.0),
        )
        diagnostics = self._full_diagnostics_for(config, before, after)
        self.assertEqual(diagnostics.force_before_scale, 10.0)
        self.assertEqual(diagnostics.force_after_scale, 3.0)
        self.assertEqual(diagnostics.moment_before_scale, 10.0 * geometry_length)
        self.assertEqual(diagnostics.moment_after_scale, 3.0 * geometry_length)
        self.assertTrue(diagnostics.force_before_pass)
        self.assertFalse(diagnostics.force_after_pass)
        self.assertFalse(diagnostics.force_pass)
        self.assertTrue(diagnostics.moment_before_pass)
        self.assertFalse(diagnostics.moment_after_pass)
        self.assertFalse(diagnostics.moment_pass)

    def test_deterministic_static_digest_rejects_direct_and_archive_forgeries(self) -> None:
        forged_static = copy.copy(self.model.static)
        object.__setattr__(forged_static, "deterministic_digest", "0" * 64)
        with self.assertRaisesRegex(ValueError, "digest|identity"):
            a2.A2FixedDamageView(
                forged_static,
                np.zeros(self.model.static.material_count, dtype=np.float64),
            )
        self._assert_archive_forgery_rejected(
            "forged-static-digest.npz",
            lambda payload: payload["static_edge_omega"].__setitem__(
                0, payload["static_edge_omega"].flat[0] * (1.0 + 1.0e-6)
            ),
        )

    def test_distinct_amor_failures_retain_only_available_mechanics_primitives(self) -> None:
        committed = self.result.initial_state
        original = a2.compute_a2_amor_history
        with mock.patch.object(
            a2, "compute_a2_amor_history", side_effect=RuntimeError("amor before failure")
        ):
            step_before, attempt_before = a2._run_a2_attempt(
                self.model, self.config, committed, 1.0e-4, 0, 1
            )
        self.assertIsNone(step_before)
        diagnostics_before = attempt_before.diagnostics
        self.assertEqual(diagnostics_before.failure_stage, "amor_before")
        self.assertIsNotNone(diagnostics_before.q_before)
        self.assertIsNotNone(diagnostics_before.seed_before)
        self.assertIsNone(diagnostics_before.Psi_before)
        self.assertIsNone(diagnostics_before.q_after)
        a2._validate_incomplete_diagnostics(diagnostics_before)

        calls = 0

        def fail_after(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("amor after failure")
            return original(*args, **kwargs)

        with mock.patch.object(a2, "compute_a2_amor_history", side_effect=fail_after):
            step_after, attempt_after = a2._run_a2_attempt(
                self.model, self.config, committed, 1.0e-4, 0, 1
            )
        self.assertIsNone(step_after)
        diagnostics_after = attempt_after.diagnostics
        self.assertEqual(diagnostics_after.failure_stage, "amor_after")
        self.assertIsNotNone(diagnostics_after.q_before)
        self.assertIsNotNone(diagnostics_after.q_after)
        self.assertIsNotNone(diagnostics_after.d_next)
        self.assertIsNotNone(diagnostics_after.seed_before)
        self.assertIsNotNone(diagnostics_after.seed_after)
        self.assertIsNone(diagnostics_after.Psi_after)
        self.assertIsNone(diagnostics_after.H_next)
        a2._validate_incomplete_diagnostics(diagnostics_after)

        metadata = a2.A2ContinuationMetadata.from_model(self.model, self.config)
        ba = a2._replay_ba(
            metadata.union_coordinates,
            metadata.union_volumes,
            metadata.visible_edges,
            metadata.delta,
            metadata.rank_rtol,
        )
        for attempt in (attempt_before, attempt_after):
            a2._replay_incomplete_diagnostics(
                attempt.diagnostics,
                committed.material_damage,
                committed.material_history,
                metadata,
                ba,
                committed.union_displacement,
                committed.outer_displacement,
                "committed",
                a2._damage_digest(committed.material_damage),
                1,
            )

    def test_incomplete_available_fields_cannot_fabricate_later_primitives(self) -> None:
        committed = self.result.initial_state
        original = a2.compute_a2_amor_history
        calls = 0

        def fail_after(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("manufactured after-history failure")
            return original(*args, **kwargs)

        with mock.patch.object(a2, "compute_a2_amor_history", side_effect=fail_after):
            _, attempt = a2._run_a2_attempt(
                self.model, self.config, committed, 1.0e-4, 0, 1
            )
        diagnostics = attempt.diagnostics
        a2._validate_incomplete_diagnostics(diagnostics)
        missing_q = tuple(
            name for name in diagnostics.available_fields if name != "q_before"
        )
        with self.assertRaisesRegex(ValueError, "availability|failure stage"):
            a2._validate_incomplete_diagnostics(
                replace(diagnostics, available_fields=missing_q)
            )
        missing_d_iter = tuple(
            name for name in diagnostics.available_fields if name != "d_iter"
        )
        with self.assertRaisesRegex(ValueError, "availability"):
            a2._validate_incomplete_diagnostics(
                replace(diagnostics, available_fields=missing_d_iter)
            )
        with self.assertRaisesRegex(ValueError, "availability"):
            a2._validate_incomplete_diagnostics(
                replace(
                    diagnostics,
                    available_fields=diagnostics.available_fields + ("H_next",),
                )
            )

    def test_minimum_increment_and_cutback_exhaustion_retain_attempt_audit(self) -> None:
        def failed_attempt(target, cutback_count):
            diagnostic = replace(
                self.result.accepted_steps[0].diagnostics,
                target_opening=target,
                passed=False,
                failure_stage="gates",
                failure_reason="manufactured gate failure",
            )
            return a2.A2AttemptRecord(
                cutback_count=cutback_count,
                load_start=0.0,
                load_target=target,
                accepted=False,
                acceptance_mode="strict",
                acceptance_status=None,
                failure_stage="gates",
                failure_reason="manufactured gate failure",
                exception_type=None,
                committed_before_digest=self.result.initial_state.digest,
                committed_after_digest=self.result.initial_state.digest,
                sweeps=(diagnostic,),
                diagnostics=diagnostic,
            )

        def always_fail(*args):
            return None, failed_attempt(float(args[3]), int(args[4]))

        minimum_config = _strict_config(
            d_opening_initial=1.0e-4,
            d_opening_min=1.0e-4,
            max_cutbacks=8,
        )
        with mock.patch.object(a2, "_run_a2_attempt", side_effect=always_fail):
            with self.assertRaises(a2.A2ContinuationFailure) as minimum_context:
                a2.run_a2_strict_continuation(self.model, minimum_config)
        minimum_failure = minimum_context.exception
        self.assertIn("d_opening_min", str(minimum_failure))
        self.assertEqual(len(minimum_failure.attempts), 1)
        self.assertEqual(minimum_failure.last_committed_state.digest, self.result.initial_state.digest)

        exhaustion_config = _strict_config(
            d_opening_initial=1.0e-4,
            d_opening_min=1.0e-5,
            max_cutbacks=1,
        )
        with mock.patch.object(a2, "_run_a2_attempt", side_effect=always_fail):
            with self.assertRaises(a2.A2ContinuationFailure) as exhaustion_context:
                a2.run_a2_strict_continuation(self.model, exhaustion_config)
        exhaustion_failure = exhaustion_context.exception
        self.assertIn("max_cutbacks", str(exhaustion_failure))
        self.assertEqual(
            [attempt.load_target for attempt in exhaustion_failure.attempts],
            [1.0e-4, 5.0e-5],
        )
        self.assertEqual(
            [attempt.cutback_count for attempt in exhaustion_failure.attempts], [0, 1]
        )

    def test_later_nonzero_committed_damage_is_used_by_healing_replay(self) -> None:
        second_config = _strict_config(opening_final=2.0e-4)
        second_step, second_attempt = a2._run_a2_attempt(
            self.model,
            second_config,
            self.result.final_state,
            2.0e-4,
            0,
            2,
        )
        self.assertIsNotNone(second_step)
        self.assertIsNotNone(second_attempt)
        assert second_step is not None
        assert second_attempt is not None
        self.assertGreater(float(np.max(second_step.damage_before)), 0.0)
        self.assertEqual(
            second_step.diagnostics.healing_min,
            float(np.min(second_step.d_next - second_step.damage_before)),
        )
        combined = a2.A2ContinuationResult(
            initial_state=self.result.initial_state,
            metadata=a2.A2ContinuationMetadata.from_model(self.model, second_config),
            states=(self.result.initial_state, self.result.final_state, second_step.state),
            accepted_steps=(self.result.accepted_steps[0], second_step),
            attempts=(self.result.attempts[0], second_attempt),
            model=self.model,
            acceptance_mode="strict",
            acceptance_status="strict_passed",
            completion_status="completed",
        )
        self.assertEqual(a2.independent_replay(combined)["accepted_steps"], 2)
        two_step_path = self._archive_path("two-step-healing.npz")
        a2.write_a2_archive(two_step_path, combined)
        forged_payload = a2.load_a2_archive(two_step_path)
        records = [json.loads(str(value)) for value in forged_payload["accepted_diagnostics_json"]]
        records[1]["healing_min"] = float(np.min(second_step.d_next)) + 1.0
        forged_payload["accepted_diagnostics_json"] = np.asarray(
            [json.dumps(record, sort_keys=True) for record in records], dtype=np.str_
        )
        forged_path = self._archive_path("forged-two-step-healing.npz")
        np.savez_compressed(forged_path, **forged_payload)
        with self.assertRaisesRegex(ValueError, "healing_min|healing"):
            a2.replay_a2_archive(forged_path)

    def test_complete_failed_trial_seed_is_not_committed_to_the_next_attempt(self) -> None:
        original = a2._full_diagnostics
        calls = 0

        def fail_first(*args, **kwargs):
            nonlocal calls
            diagnostics = original(*args, **kwargs)
            calls += 1
            if calls == 1:
                return replace(
                    diagnostics,
                    passed=False,
                    failure_stage="gates",
                    failure_reason="manufactured complete gate failure",
                )
            return diagnostics

        config = _strict_config(max_stagger_iterations=1)
        committed = self.result.initial_state
        with mock.patch.object(a2, "_full_diagnostics", side_effect=fail_first):
            rejected_step, rejected_attempt = a2._run_a2_attempt(
                self.model, config, committed, 1.0e-4, 0, 1
            )
        self.assertIsNone(rejected_step)
        self.assertTrue(rejected_attempt.diagnostics.complete_sweep)
        self.assertEqual(
            rejected_attempt.committed_before_digest,
            rejected_attempt.committed_after_digest,
        )
        self.assertIsNotNone(rejected_attempt.diagnostics.seed_after)
        assert rejected_attempt.diagnostics.seed_after is not None
        self.assertNotEqual(
            rejected_attempt.diagnostics.seed_after.target_damage_digest,
            a2._damage_digest(committed.material_damage),
        )

        accepted_step, accepted_attempt = a2._run_a2_attempt(
            self.model, config, committed, 1.0e-4, 1, 1
        )
        self.assertIsNotNone(accepted_step)
        self.assertTrue(accepted_attempt.accepted)
        assert accepted_step is not None
        assert accepted_step.diagnostics.seed_before is not None
        self.assertEqual(accepted_step.diagnostics.seed_before.source_state, "committed")
        self.assertEqual(
            accepted_step.diagnostics.seed_before.source_damage_digest,
            a2._damage_digest(committed.material_damage),
        )


if __name__ == "__main__":
    unittest.main()
