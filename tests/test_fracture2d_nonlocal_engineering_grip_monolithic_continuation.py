"""Manufactured Phase-2C adaptive strict-continuation tests.

The tests use only the existing 5x5 engineering-grip cloud.  The smoke
success path runs the actual controller and direct predictor; rejected-attempt
objects are manufactured only for isolated cutback/failure audits.  No
workflow, CLI, archive, callback, or physical execution is invoked.
"""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from unittest import mock

import jax
import numpy as np

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import nonlocal_engineering_grip_monolithic as monolithic
from examples.fracture2d.nonlocal_engineering_grip_monolithic import (
    MonolithicContinuationConfig,
    MonolithicContinuationFailure,
    MonolithicContinuationResult,
    MonolithicGripResidual,
    MonolithicNewtonConfig,
    MonolithicNewtonFailureAudit,
    MonolithicNewtonGateConfig,
    MonolithicVerifiedAttempt,
    run_monolithic_strict_continuation,
)
from examples.fracture2d.nonlocal_grip_diagnostic import build_union_geometry_from_arrays
from tests.test_fracture2d_nonlocal_engineering_grip_monolithic import _material_cloud


_REAL_PHASE2B = monolithic.run_verified_monolithic_attempt


class MonolithicContinuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        coordinates, triangles, volumes, bottom, top, edges, delta = _material_cloud()
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
        cls.model = engineering.build_engineering_grip_model(
            geometry,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
            1.7,
            0.65,
        )
        cls.base = engineering.EngineeringGripConfig.smoke()
        cls.newton = MonolithicNewtonConfig(
            max_newton_iterations=30,
            gmres_rtol=1.0e-10,
            gmres_atol=1.0e-13,
            gmres_restart=50,
            gmres_maxiter=200,
            armijo_c=1.0e-4,
            line_search_reduction=0.5,
            line_search_max_backtracks=12,
            line_search_alpha_min=1.0e-8,
        )
        cls.work_check = engineering.run_zero_damage_work_check(
            cls.model,
            cls.base.work_check_load,
            cls.base.work_epsilon,
            cls.base.work_relative_limit,
            cls.base.cg_rtol,
            cls.base.cg_atol,
            cls.base.cg_maxiter,
            mechanics_adapter=None,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    def _zero(self) -> engineering.EngineeringGripState:
        return engineering.EngineeringGripState(
            u_bar=0.0,
            union_displacement=np.zeros(self.model.n_dofs, dtype=np.float64),
            material_damage=np.zeros(self.model.n_material, dtype=np.float64),
            material_history=np.zeros(self.model.n_material, dtype=np.float64),
        )

    def _config(
        self,
        base: engineering.EngineeringGripConfig | None = None,
    ) -> MonolithicContinuationConfig:
        selected = self.base if base is None else base
        return MonolithicContinuationConfig(
            selected.load_start,
            selected.load_final,
            selected.du_initial,
            selected.du_min,
            selected.cutback_factor,
            selected.max_cutbacks,
            selected,
            self.newton,
        )

    def _direct_initial(
        self,
        committed: engineering.EngineeringGripState,
        target: float,
    ) -> np.ndarray:
        full = engineering._direct_seed(
            committed.union_displacement,
            committed.u_bar,
            target,
            self.model.constraints,
        )
        return np.concatenate(
            (
                full[self.model.constraints.free_dofs],
                committed.material_damage,
                committed.material_history,
            )
        )

    def _actual_success(
        self,
        committed: engineering.EngineeringGripState,
        target: float,
        base: engineering.EngineeringGripConfig | None = None,
    ) -> MonolithicVerifiedAttempt:
        selected = self.base if base is None else base
        initial = self._direct_initial(committed, target)
        result = _REAL_PHASE2B(
            self.model,
            selected,
            committed,
            target,
            None,
            initial,
            self.newton,
        )
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(result.initial_z, initial)
        return result

    def _rejected(
        self,
        committed: engineering.EngineeringGripState,
        target: float,
        initial_z: np.ndarray,
        base: engineering.EngineeringGripConfig | None = None,
    ) -> MonolithicVerifiedAttempt:
        selected = self.base if base is None else base
        gate_config = MonolithicNewtonGateConfig.from_engineering_config(selected)
        residual = MonolithicGripResidual(self.model, target, committed.material_history)
        audit = MonolithicNewtonFailureAudit(
            stage="max_newton_iterations",
            target_u_bar=target,
            model_identity=self.model,
            static_identity=self.model.static,
            config=self.newton,
            gate_config=gate_config,
            reason="manufactured continuation rejection",
            iteration=0,
            current_z=initial_z,
            active_mask=residual.active_mask(initial_z),
            linearization_scales=None,
            gmres_info=None,
            gmres_callback_residuals=(),
            gmres_unpreconditioned_residual=None,
            gmres_residual_limit=None,
            attempted_alphas=(),
            rejection_reasons=(),
            completed_diagnostics=(),
        )
        return monolithic._failure_attempt(
            model=self.model,
            config=selected,
            committed=committed,
            target=target,
            adapter_kind="cpu_reference",
            initial_z=initial_z,
            newton_config=self.newton,
            newton_gate_config=gate_config,
            newton_result=None,
            auxiliary=None,
            current_z=initial_z,
            stage="newton",
            reason="manufactured continuation rejection",
            exception_type="MonolithicNewtonFailure",
            newton_audit=audit,
            canonical_z=None,
            canonical_active_mask=None,
            canonical_scales=None,
            canonical_scaled_residual=None,
            canonical_gate_passes=tuple(
                (name, False) for name in monolithic._MONOLITHIC_CANONICAL_GATE_NAMES
            ),
        )

    def _run_with_work_check(self, config, side_effect):
        with mock.patch.object(
            engineering,
            "run_zero_damage_work_check",
            return_value=self.work_check,
        ) as work_probe, mock.patch.object(
            monolithic,
            "run_verified_monolithic_attempt",
            side_effect=side_effect,
        ) as attempt_probe:
            result = run_monolithic_strict_continuation(self.model, config, None)
        self.assertEqual(work_probe.call_count, 1)
        return result, attempt_probe

    def test_api_and_config_are_explicit_and_schedule_bound(self) -> None:
        parameters = inspect.signature(MonolithicContinuationConfig).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "load_start",
                "load_final",
                "du_initial",
                "du_min",
                "cutback_factor",
                "max_cutbacks",
                "engineering_config",
                "newton_config",
            ),
        )
        self.assertTrue(
            all(parameter.default is inspect.Parameter.empty for parameter in parameters.values())
        )
        run_parameters = inspect.signature(run_monolithic_strict_continuation).parameters
        self.assertEqual(tuple(run_parameters), ("model", "config", "adapter"))
        self.assertTrue(
            all(parameter.default is inspect.Parameter.empty for parameter in run_parameters.values())
        )
        with self.assertRaises(ValueError):
            MonolithicContinuationConfig(
                0.0,
                self.base.load_final,
                self.base.du_initial,
                self.base.du_min,
                self.base.cutback_factor,
                self.base.max_cutbacks,
                replace(self.base, du_initial=5.0e-5),
                self.newton,
            )

    def test_success_snaps_endpoint_and_preserves_immutable_audit(self) -> None:
        final = self.base.load_final
        endpoint_base = replace(
            self.base,
            du_initial=float(np.nextafter(np.float64(final), np.float64(0.0))),
        )
        config = self._config(endpoint_base)
        zero = self._zero()
        q_before = zero.q.copy()
        d_before = zero.d.copy()
        H_before = zero.H.copy()
        result = run_monolithic_strict_continuation(self.model, config, None)
        self.assertIsInstance(result, MonolithicContinuationResult)
        self.assertTrue(result.validated)
        self.assertEqual(result.final_state.u_bar, final)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].load_target, final)
        self.assertEqual(result.attempts[0].cutback_count, 0)
        np.testing.assert_array_equal(
            result.attempts[0].initial_z,
            self._direct_initial(zero, final),
        )
        np.testing.assert_array_equal(zero.q, q_before)
        np.testing.assert_array_equal(zero.d, d_before)
        np.testing.assert_array_equal(zero.H, H_before)
        self.assertFalse(result.attempts[0].initial_z.flags.writeable)
        with self.assertRaises(ValueError):
            result.attempts[0].initial_z[0] = result.attempts[0].initial_z[0]

    def test_owner_replay_accepts_legitimate_direct_predictor_result(self) -> None:
        result, _ = self._run_with_work_check(
            self._config(),
            lambda *args: self._actual_success(args[2], args[3], args[1]),
        )
        with mock.patch.object(
            monolithic,
            "_AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT",
            wraps=_REAL_PHASE2B,
        ) as replay_probe:
            result.validate_with_owner(self.model)
        replay_probe.assert_called_once()

    def test_owner_replay_recurses_into_forged_work_check(self) -> None:
        result, _ = self._run_with_work_check(
            self._config(),
            lambda *args: self._actual_success(args[2], args[3], args[1]),
        )
        center = result.work_check.center_result
        forged_center = replace(
            center,
            evaluation=replace(
                center.evaluation,
                energy=center.evaluation.energy + 1.0e-9,
            ),
        )
        forged_work_check = replace(
            result.work_check,
            center_result=forged_center,
            center_result_digest=engineering._mechanics_result_digest(forged_center),
        )
        forged = replace(result, work_check=forged_work_check)
        forged.validate()
        with self.assertRaises(ValueError):
            forged.validate_with_owner(self.model)

    def test_controller_no_progress_retains_exact_failed_attempt(self) -> None:
        with mock.patch.object(
            engineering, "run_zero_damage_work_check", return_value=self.work_check
        ), mock.patch.object(
            engineering, "_next_target", return_value=0.0
        ), mock.patch.object(
            monolithic, "run_verified_monolithic_attempt"
        ) as attempt_probe:
            with self.assertRaises(MonolithicContinuationFailure) as raised:
                run_monolithic_strict_continuation(self.model, self._config(), None)
        attempt_probe.assert_not_called()
        failure = raised.exception
        self.assertEqual(len(failure.attempts), 1)
        failed_attempt = failure.attempts[-1]
        self.assertIs(failure.failed_attempt, failed_attempt)
        self.assertEqual(failed_attempt.failure_stage, "controller")
        self.assertEqual(failed_attempt.load_target, 0.0)
        self.assertEqual(failed_attempt.increment, 0.0)
        self.assertEqual(failure.failed_target, 0.0)
        self.assertEqual(
            failure.result.failure_reason,
            "no_progress: continuation made no floating-point load progress",
        )

    def test_controller_no_progress_owner_validation_skips_phase2b_replay(self) -> None:
        with mock.patch.object(
            engineering, "run_zero_damage_work_check", return_value=self.work_check
        ), mock.patch.object(
            engineering, "_next_target", return_value=0.0
        ), mock.patch.object(
            monolithic, "run_verified_monolithic_attempt"
        ) as attempt_probe:
            with self.assertRaises(MonolithicContinuationFailure) as raised:
                run_monolithic_strict_continuation(self.model, self._config(), None)
            attempt_probe.assert_not_called()
            with mock.patch.object(
                monolithic,
                "_AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT",
                side_effect=AssertionError("controller-only record must not replay Phase2B"),
            ):
                raised.exception.result.validate_with_owner(self.model)

    def test_controller_only_stage_target_increment_and_reason_forgeries_fail_owner_boundary(
        self,
    ) -> None:
        for forgery in ("stage", "target", "increment", "reason"):
            with self.subTest(forgery=forgery), mock.patch.object(
                engineering, "run_zero_damage_work_check", return_value=self.work_check
            ), mock.patch.object(
                engineering, "_next_target", return_value=0.0
            ):
                with self.assertRaises(MonolithicContinuationFailure) as raised:
                    run_monolithic_strict_continuation(self.model, self._config(), None)
                result = raised.exception.result
                attempt = result.attempts[-1]
                assert attempt.phase2b.failure is not None
                if forgery == "stage":
                    object.__setattr__(attempt.phase2b.failure, "stage", "newton")
                elif forgery == "target":
                    object.__setattr__(attempt, "load_target", 1.0e-4)
                elif forgery == "increment":
                    object.__setattr__(attempt, "increment", 1.0e-4)
                else:
                    object.__setattr__(
                        attempt.phase2b.failure,
                        "reason",
                        "forged controller reason",
                    )
                with self.assertRaises(ValueError):
                    result.validate_with_owner(self.model)

    def test_failed_result_without_a_terminal_attempt_is_rejected(self) -> None:
        result, _ = self._run_with_work_check(
            self._config(),
            lambda *args: self._actual_success(args[2], args[3], args[1]),
        )
        with self.assertRaises(ValueError):
            replace(
                result,
                states=(result.initial_state,),
                accepted_steps=(),
                attempts=(),
                completion_status="failed",
                failed_target=0.0,
                failure_reason="arbitrary forged failure",
                failed_attempt=None,
            )

    def test_rejection_cutback_rebuilds_predictor_from_unchanged_commit(self) -> None:
        zero = self._zero()
        first_target = self.base.load_final
        first_initial = self._direct_initial(zero, first_target)
        config = self._config()
        seeds: list[np.ndarray] = []

        def probe(*args):
            seeds.append(np.array(args[5], copy=True))
            if len(seeds) == 1:
                return self._rejected(args[2], args[3], args[5], args[1])
            return self._actual_success(args[2], args[3], args[1])

        result, probes = self._run_with_work_check(config, probe)
        self.assertEqual(probes.call_count, 3)
        self.assertEqual(len(result.attempts), 3)
        self.assertFalse(result.attempts[0].accepted)
        self.assertEqual(result.attempts[0].load_target, first_target)
        self.assertEqual(result.attempts[1].load_target, 5.0e-5)
        self.assertEqual(result.attempts[1].cutback_count, 1)
        self.assertEqual(result.attempts[2].cutback_count, 0)
        accepted_half = result.attempts[1].state
        assert accepted_half is not None
        np.testing.assert_array_equal(seeds[0], first_initial)
        np.testing.assert_array_equal(seeds[1], self._direct_initial(zero, 5.0e-5))
        np.testing.assert_array_equal(
            seeds[2], self._direct_initial(accepted_half, first_target)
        )
        self.assertEqual(result.attempts[0].committed_before_digest, zero.digest)
        self.assertEqual(result.attempts[0].committed_after_digest, zero.digest)

    def test_minimum_increment_failure_retains_last_state_and_failed_target(self) -> None:
        config = self._config(replace(self.base, du_min=self.base.du_initial))

        def reject(*args):
            return self._rejected(args[2], args[3], args[5], args[1])

        with mock.patch.object(engineering, "run_zero_damage_work_check", return_value=self.work_check), mock.patch.object(
            monolithic, "run_verified_monolithic_attempt", side_effect=reject
        ):
            with self.assertRaises(MonolithicContinuationFailure) as raised:
                run_monolithic_strict_continuation(self.model, config, None)
        failure = raised.exception
        self.assertIs(failure.result, failure.audit)
        self.assertEqual(failure.result.completion_status, "failed")
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        self.assertEqual(failure.failed_target, self.base.load_final)
        self.assertEqual(len(failure.attempts), 1)
        self.assertIs(failure.failed_attempt, failure.attempts[-1])
        self.assertEqual(failure.attempts[-1].committed_before_digest, failure.attempts[-1].committed_after_digest)

    def test_cutback_limit_failure_retains_every_rejected_attempt(self) -> None:
        base = replace(self.base, du_min=2.5e-5, max_cutbacks=1)
        config = self._config(base)

        def reject(*args):
            return self._rejected(args[2], args[3], args[5], base)

        with mock.patch.object(engineering, "run_zero_damage_work_check", return_value=self.work_check), mock.patch.object(
            monolithic, "run_verified_monolithic_attempt", side_effect=reject
        ):
            with self.assertRaises(MonolithicContinuationFailure) as raised:
                run_monolithic_strict_continuation(self.model, config, None)
        attempts = raised.exception.attempts
        self.assertEqual(len(attempts), 2)
        self.assertEqual([attempt.cutback_count for attempt in attempts], [0, 1])
        self.assertEqual([attempt.load_target for attempt in attempts], [1.0e-4, 5.0e-5])
        self.assertTrue(all(not attempt.accepted for attempt in attempts))
        self.assertEqual(attempts[0].committed_after_digest, attempts[1].committed_after_digest)

    def test_work_check_failure_prevents_any_phase2b_attempt(self) -> None:
        failed_base = replace(self.base, work_relative_limit=0.0)
        with self.assertRaises(engineering.EngineeringGripWorkCheckFailure) as work_failure:
            engineering.run_zero_damage_work_check(
                self.model,
                failed_base.work_check_load,
                failed_base.work_epsilon,
                failed_base.work_relative_limit,
                failed_base.cg_rtol,
                failed_base.cg_atol,
                failed_base.cg_maxiter,
                mechanics_adapter=None,
            )
        failed_check = work_failure.exception.check
        self.assertFalse(failed_check.passed)
        failed_check.validate()
        with mock.patch.object(
            engineering,
            "run_zero_damage_work_check",
            return_value=failed_check,
        ), mock.patch.object(
            monolithic,
            "run_verified_monolithic_attempt",
        ) as attempt_probe:
            with self.assertRaises(engineering.EngineeringGripWorkCheckFailure) as raised:
                run_monolithic_strict_continuation(self.model, self._config(failed_base), None)
        self.assertIs(raised.exception.check, failed_check)
        attempt_probe.assert_not_called()

    def test_structural_phase2b_exception_propagates_without_failure_reclassification(self) -> None:
        with mock.patch.object(engineering, "run_zero_damage_work_check", return_value=self.work_check), mock.patch.object(
            monolithic,
            "run_verified_monolithic_attempt",
            side_effect=TypeError("foreign structural object"),
        ) as attempt_probe:
            with self.assertRaisesRegex(TypeError, "foreign structural object"):
                run_monolithic_strict_continuation(self.model, self._config(), None)
        self.assertEqual(attempt_probe.call_count, 1)

    def test_result_and_attempt_constructors_reject_forged_pairings(self) -> None:
        result, _ = self._run_with_work_check(
            self._config(),
            lambda *args: self._actual_success(args[2], args[3], args[1]),
        )
        with self.assertRaises(ValueError):
            replace(result, static_identity=object())
        with self.assertRaises(ValueError):
            replace(result.attempts[0], load_target=5.0e-5, increment=5.0e-5)
        with self.assertRaises(ValueError):
            replace(result, accepted_steps=())
        with self.assertRaises(ValueError):
            replace(result, work_check=replace(result.work_check, u_bar=5.0e-5))
        with self.assertRaises(ValueError):
            replace(result, work_check=replace(result.work_check, epsilon=5.0e-6))
        with self.assertRaises(ValueError):
            replace(result, work_check=replace(result.work_check, relative_limit=1.0e-6))
        with self.assertRaises(ValueError):
            replace(result, work_check=replace(result.work_check, passed=False))
        with self.assertRaises(ValueError):
            replace(
                result,
                attempts=(replace(result.attempts[0], cutback_count=1),),
            )


if __name__ == "__main__":
    unittest.main()
