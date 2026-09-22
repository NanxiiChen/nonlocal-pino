"""Manufactured tests for the isolated hookstep outer controller."""

from __future__ import annotations

import unittest
import inspect
from typing import Any
from unittest import mock

import jax
import numpy as np

from examples.fracture2d.nonlocal_baseline import GateTolerance
from examples.fracture2d.nonlocal_engineering_grip_monolithic import (
    MonolithicGripResidual,
    MonolithicHookstepArnoldi,
    MonolithicHookstepConfig,
    MonolithicHookstepControllerFailure,
    MonolithicHookstepDirection,
    MonolithicHookstepFailure,
    MonolithicHookstepFailureAudit,
    MonolithicNewtonGateConfig,
    MonolithicProjectedHookstep,
    build_monolithic_trust_metric,
    run_monolithic_hookstep_correction,
)
from tests.test_fracture2d_nonlocal_engineering_grip_monolithic_hookstep import (
    _config,
    _model_from_cloud,
)


class MonolithicHookstepControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        cls.model = _model_from_cloud()

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    @staticmethod
    def _gates() -> MonolithicNewtonGateConfig:
        return MonolithicNewtonGateConfig(
            mechanics_cg_rtol=1.0e-10,
            mechanics_cg_atol=1.0e-13,
            displacement=GateTolerance(1.0e-7, 1.0e-5, 1.0),
            damage=GateTolerance(1.0e-7, 1.0e-5, 1.0),
            history=GateTolerance(1.0e-7, 1.0e-5, 1.0),
            damage_residual=GateTolerance(1.0e-9, 1.0e-7, 1.0),
        )

    def _state(self) -> tuple[MonolithicGripResidual, np.ndarray, Any, np.ndarray]:
        owner = MonolithicGripResidual(
            self.model,
            1.0e-4,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        rng = np.random.default_rng(923)
        state = owner.pack(
            rng.standard_normal(owner.n_free) * 1.0e-5,
            rng.random(owner.n_material) * 1.0e-3,
            rng.random(owner.n_material) * 1.0e-3,
        )
        scales = owner.block_scales(state, self._gates())
        return owner, state, scales, owner.active_mask(state)

    @staticmethod
    def _arnoldi(n_packed: int, beta: float) -> MonolithicHookstepArnoldi:
        basis = np.zeros((n_packed, 1), dtype=np.float64)
        basis[0, 0] = 1.0
        return MonolithicHookstepArnoldi(
            beta=beta,
            basis=basis,
            preconditioned_basis=basis,
            hessenberg=np.asarray([[1.0], [0.0]], dtype=np.float64),
            metric_basis=basis,
            terminal_basis_vector=None,
            happy_breakdown=True,
            orthogonality_defect=0.0,
        )

    @staticmethod
    def _direction(
        arnoldi: MonolithicHookstepArnoldi,
        radius: float,
        config: MonolithicHookstepConfig,
        prediction: float = 0.5,
    ) -> MonolithicHookstepDirection:
        coefficients = np.asarray([-1.0], dtype=np.float64)
        step = arnoldi.preconditioned_basis @ coefficients
        projected = MonolithicProjectedHookstep(
            coefficients=coefficients,
            trust_multiplier=1.0,
            radius=radius,
            metric_norm=radius,
            projected_residual_norm=0.0,
            predicted_reduction=prediction,
            metric_rank=1,
            projected_rank=1,
            on_boundary=True,
            secular_iterations=1,
            radius_tolerance=config.secular_rtol * radius,
        )
        return MonolithicHookstepDirection(
            step=step,
            metric_norm=radius,
            full_predicted_reduction=prediction,
            projected_prediction_defect=0.0,
            metric_consistency_defect=0.0,
            prediction_consistency_rtol=config.prediction_consistency_rtol,
            metric_consistency_rtol=config.metric_consistency_rtol,
            arnoldi=arnoldi,
            projected=projected,
        )

    def test_rejection_reuses_arnoldi_then_acceptance_requires_rebuild(self) -> None:
        owner, state, scales, active = self._state()
        config = _config()
        state_before = state.copy()
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        changed = active.copy()
        changed[0] = ~changed[0]

        def direction_at_radius(
            _owner: MonolithicGripResidual,
            _state: np.ndarray,
            _scales: Any,
            _active: np.ndarray,
            frozen_config: MonolithicHookstepConfig,
            radius: float,
            _metric: Any,
            frozen_arnoldi: MonolithicHookstepArnoldi,
            _reference: np.ndarray,
        ) -> MonolithicHookstepDirection:
            self.assertIs(frozen_arnoldi, arnoldi)
            return self._direction(arnoldi, radius, frozen_config)

        with mock.patch.object(
            type(owner),
            "active_mask",
            autospec=True,
            side_effect=[active, changed],
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, reference, np.zeros_like(reference)],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ) as build, mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_trust_metric",
            wraps=build_monolithic_trust_metric,
        ) as build_metric, mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            side_effect=direction_at_radius,
        ) as materialize:
            result = run_monolithic_hookstep_correction(
                owner, state, self._gates(), active, config
            )

        self.assertEqual(result.status, "accepted")
        self.assertEqual([row.outcome for row in result.attempts], [
            "rejected_ratio",
            "accepted",
        ])
        self.assertEqual(
            [row.radius for row in result.attempts],
            [config.trust_radius_initial, config.gamma_shrink],
        )
        self.assertTrue(all(row.direction.arnoldi is arnoldi for row in result.attempts))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(build_metric.call_count, 1)
        self.assertEqual(materialize.call_count, 2)
        self.assertTrue(result.active_set_changed)
        self.assertTrue(result.rebuild_required)
        self.assertEqual(result.next_radius, 2.0 * config.gamma_shrink)
        self.assertTrue(np.array_equal(result.z_before, state))
        self.assertTrue(np.array_equal(state, state_before))
        self.assertFalse(result.z_before.flags.writeable)
        self.assertFalse(result.z_after.flags.writeable)

    def test_nonfinite_trial_is_rejected_then_finite_trial_can_accept(self) -> None:
        owner, state, scales, active = self._state()
        config = _config()
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        directions = [
            self._direction(arnoldi, config.trust_radius_initial, config),
            self._direction(
                arnoldi,
                config.gamma_shrink * config.trust_radius_initial,
                config,
            ),
        ]
        with mock.patch.object(
            type(owner),
            "active_mask",
            autospec=True,
            side_effect=[active, active],
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[
                reference,
                FloatingPointError("manufactured non-finite trial"),
                np.zeros_like(reference),
            ],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            side_effect=directions,
        ):
            result = run_monolithic_hookstep_correction(
                owner, state, self._gates(), active, config
            )
        self.assertEqual(result.attempts[0].outcome, "rejected_nonfinite_trial")
        self.assertIsNone(result.attempts[0].rho)
        failure_reason = result.attempts[0].failure_reason
        if failure_reason is None:
            self.fail("non-finite trial omitted its failure reason")
        self.assertIn("manufactured non-finite trial", failure_reason)
        self.assertEqual(result.attempts[1].outcome, "accepted")

    def test_trial_linalg_failure_is_hard_and_not_a_radius_rejection(self) -> None:
        owner, state, _scales, active = self._state()
        config = _config()
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        direction = self._direction(arnoldi, config.trust_radius_initial, config)
        with mock.patch.object(
            type(owner), "active_mask", autospec=True, return_value=active
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, np.linalg.LinAlgError("manufactured linalg")],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            return_value=direction,
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepControllerFailure, "trial evaluation failed"
            ) as caught:
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), active, config
                )
        self.assertEqual(caught.exception.audit.attempts, ())
        self.assertEqual(caught.exception.audit.inner_audit.stage, "prediction")

    def test_unrepresentable_finite_ratio_is_an_audited_hard_failure(self) -> None:
        owner, state, _scales, active = self._state()
        config = _config()
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        trial_residual = np.zeros_like(reference)
        trial_residual[0] = 1.0e154
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        direction = self._direction(
            arnoldi,
            config.trust_radius_initial,
            config,
            prediction=np.nextafter(0.0, 1.0),
        )
        with mock.patch.object(
            type(owner), "active_mask", autospec=True, return_value=active
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, trial_residual],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            return_value=direction,
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepControllerFailure, "ratio is non-finite"
            ) as caught:
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), active, config
                )
        self.assertEqual(caught.exception.audit.inner_audit.stage, "radius")
        self.assertEqual(caught.exception.audit.attempts, ())

    def test_rejection_budget_failure_retains_all_rejected_trials(self) -> None:
        owner, state, scales, active = self._state()
        config = _config(max_radius_rejections=1)
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)

        def direction_at_radius(*args: Any) -> MonolithicHookstepDirection:
            return self._direction(arnoldi, float(args[5]), config)

        with mock.patch.object(
            type(owner), "active_mask", autospec=True, return_value=active
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, reference, reference],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            side_effect=direction_at_radius,
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepControllerFailure, "rejection budget"
            ) as caught:
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), active, config
                )
        audit = caught.exception.audit
        self.assertEqual(len(audit.attempts), 2)
        self.assertTrue(all(row.outcome == "rejected_ratio" for row in audit.attempts))
        self.assertTrue(all(row.direction.arnoldi is audit.arnoldi for row in audit.attempts))
        self.assertEqual(audit.inner_audit.stage, "radius")
        if audit.inner_audit.current_z is None:
            self.fail("controller failure omitted the frozen state")
        self.assertTrue(np.array_equal(audit.inner_audit.current_z, state))

    def test_radius_floor_failure_retains_the_triggering_rejection(self) -> None:
        owner, state, scales, active = self._state()
        config = _config(trust_radius_min=0.5)
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        direction = self._direction(arnoldi, config.trust_radius_initial, config)
        with mock.patch.object(
            type(owner), "active_mask", autospec=True, return_value=active
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, reference],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            return_value=direction,
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepControllerFailure, "trust_radius_min"
            ) as caught:
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), active, config
                )
        self.assertEqual(len(caught.exception.audit.attempts), 1)
        self.assertEqual(
            caught.exception.audit.attempts[0].radius_decision.next_radius,
            config.gamma_shrink,
        )

    def test_inner_failure_after_rejection_retains_prior_trial(self) -> None:
        owner, state, scales, active = self._state()
        config = _config()
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        first = self._direction(arnoldi, config.trust_radius_initial, config)
        inner = MonolithicHookstepFailure(
            "manufactured projected failure",
            MonolithicHookstepFailureAudit(
                stage="projected",
                reason="manufactured projected failure",
                radius=config.gamma_shrink,
                beta=1.0,
                krylov_dimension=1,
                config=config,
                model_identity=owner._model,
                current_z=state,
                active_mask=active,
                scale_state_digest=scales.state_digest,
                hessenberg=arnoldi.hessenberg,
                metric_singular_values=None,
                projected_singular_values=None,
            ),
        )
        with mock.patch.object(
            type(owner), "active_mask", autospec=True, return_value=active
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, reference],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            side_effect=[first, inner],
        ):
            with self.assertRaises(MonolithicHookstepControllerFailure) as caught:
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), active, config
                )
        self.assertEqual(len(caught.exception.audit.attempts), 1)
        self.assertEqual(caught.exception.audit.attempts[0].outcome, "rejected_ratio")
        self.assertIs(caught.exception.__cause__, inner)

    def test_accepted_active_rebuild_failure_retains_pending_trial(self) -> None:
        owner, state, scales, active = self._state()
        config = _config()
        reference = np.zeros(owner.n_packed, dtype=np.float64)
        reference[0] = 1.0
        arnoldi = self._arnoldi(owner.n_packed, 1.0)
        direction = self._direction(arnoldi, config.trust_radius_initial, config)
        with mock.patch.object(
            type(owner),
            "active_mask",
            autospec=True,
            side_effect=[active, FloatingPointError("manufactured active failure")],
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            side_effect=[reference, np.zeros_like(reference)],
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "_materialize_monolithic_hookstep_direction",
            return_value=direction,
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepControllerFailure, "active-mask rebuild"
            ) as caught:
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), active, config
                )
        audit = caught.exception.audit
        self.assertEqual(len(audit.attempts), 1)
        self.assertEqual(audit.attempts[0].outcome, "accepted_pending_rebuild")
        self.assertTrue(audit.attempts[0].radius_decision.accepted)

    def test_exact_root_bypasses_metric_and_arnoldi(self) -> None:
        owner, state, scales, active = self._state()
        with mock.patch.object(
            type(owner), "active_mask", autospec=True, return_value=active
        ), mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            return_value=np.zeros(owner.n_packed, dtype=np.float64),
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi"
        ) as build:
            result = run_monolithic_hookstep_correction(
                owner, state, self._gates(), active, _config()
            )
        build.assert_not_called()
        self.assertEqual(result.status, "exact_root")
        self.assertEqual(result.attempts, ())
        self.assertFalse(result.rebuild_required)
        self.assertTrue(np.array_equal(result.z_before, result.z_after))

    def test_stale_active_mask_is_rejected_before_trial_evaluation(self) -> None:
        owner, state, scales, active = self._state()
        stale = ~active
        with mock.patch.object(
            type(owner), "scaled_residual", autospec=True
        ) as residual:
            with self.assertRaisesRegex(ValueError, "active mask is stale"):
                run_monolithic_hookstep_correction(
                    owner, state, self._gates(), stale, _config()
                )
        residual.assert_not_called()

    def test_controller_requires_gate_config_instead_of_external_scales(self) -> None:
        parameters = inspect.signature(run_monolithic_hookstep_correction).parameters
        self.assertEqual(
            tuple(parameters),
            ("monolithic", "z", "gate_config", "active_mask", "config"),
        )
        owner, state, scales, active = self._state()
        with self.assertRaisesRegex(TypeError, "gate_config"):
            run_monolithic_hookstep_correction(
                owner, state, scales, active, _config()
            )

    def test_real_manufactured_controller_smoke(self) -> None:
        owner, state, scales, active = self._state()
        result = run_monolithic_hookstep_correction(
            owner, state, self._gates(), active, _config()
        )
        self.assertEqual(result.status, "accepted")
        self.assertEqual(len(result.attempts), 1)
        actual = result.attempts[0].actual_reduction
        ratio = result.attempts[0].rho
        if actual is None or ratio is None:
            self.fail("accepted controller smoke omitted reduction evidence")
        self.assertGreater(actual, 0.0)
        self.assertGreater(ratio, 0.0)
        trial_residual = result.attempts[0].trial_scaled_residual
        if trial_residual is None:
            self.fail("accepted controller smoke omitted its trial residual")
        residual_change = trial_residual - result.reference_scaled_residual
        expected_actual = -float(
            np.dot(result.reference_scaled_residual, residual_change)
        ) - 0.5 * float(np.dot(residual_change, residual_change))
        self.assertEqual(actual, expected_actual)
        self.assertEqual(
            ratio,
            expected_actual / result.attempts[0].direction.full_predicted_reduction,
        )
        self.assertTrue(result.rebuild_required)


if __name__ == "__main__":
    unittest.main()
