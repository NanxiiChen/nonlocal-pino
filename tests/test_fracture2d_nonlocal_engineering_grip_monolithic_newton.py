"""Manufactured Phase-2A scaled semismooth Newton/GMRES tests.

These tests use only the existing 5x5 engineering-grip cloud.  They exercise
the fixed-load core and never invoke a workflow, continuation controller, or
rounded-slot physical run.
"""

from __future__ import annotations

import hashlib
import inspect
import unittest
from dataclasses import replace
from typing import cast
from unittest import mock

import jax
import numpy as np
import scipy.sparse.linalg

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import nonlocal_engineering_grip_monolithic as monolithic_module
from examples.fracture2d.nonlocal_baseline import GateTolerance
from examples.fracture2d.nonlocal_engineering_grip_monolithic import (
    MonolithicBlockDiagonalPreconditioner,
    MonolithicGripResidual,
    MonolithicNewtonConfig,
    MonolithicNewtonFailure,
    MonolithicNewtonFailureAudit,
    MonolithicNewtonGateConfig,
    MonolithicNewtonScales,
    solve_monolithic_newton,
)
from examples.fracture2d.nonlocal_grip_diagnostic import build_union_geometry_from_arrays
from tests.test_fracture2d_nonlocal_engineering_grip_monolithic import (
    _IndependentMonolithicOracle,
    _material_cloud,
)


class MonolithicNewtonCoreTest(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    def _gates(self, *, mechanics_atol: float) -> MonolithicNewtonGateConfig:
        return MonolithicNewtonGateConfig(
            mechanics_cg_rtol=1.0e-10,
            mechanics_cg_atol=mechanics_atol,
            displacement=GateTolerance(1.0e-7, 1.0e-5, 1.0),
            damage=GateTolerance(1.0e-7, 1.0e-5, 1.0),
            history=GateTolerance(1.0e-7, 1.0e-5, 1.0),
            damage_residual=GateTolerance(1.0e-9, 1.0e-7, 1.0),
        )

    def _config(
        self,
        *,
        max_newton_iterations: int,
        armijo_c: float,
        line_search_reduction: float,
        line_search_max_backtracks: int,
        line_search_alpha_min: float,
        gmres_maxiter: int,
    ) -> MonolithicNewtonConfig:
        # Every value is supplied explicitly to the production constructor;
        # these keyword-only test controls are not production defaults.
        return MonolithicNewtonConfig(
            max_newton_iterations=max_newton_iterations,
            gmres_rtol=1.0e-10,
            gmres_atol=1.0e-13,
            gmres_restart=50,
            gmres_maxiter=gmres_maxiter,
            armijo_c=armijo_c,
            line_search_reduction=line_search_reduction,
            line_search_max_backtracks=line_search_max_backtracks,
            line_search_alpha_min=line_search_alpha_min,
        )

    def _perturbed_state(self, seed: int) -> tuple[MonolithicGripResidual, np.ndarray]:
        rng = np.random.default_rng(seed)
        monolithic = MonolithicGripResidual(
            self.model,
            1.0e-4,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        return monolithic, monolithic.pack(
            rng.standard_normal(monolithic.n_free) * 1.0e-4,
            rng.random(monolithic.n_material) * 1.0e-2,
            rng.random(monolithic.n_material) * 1.0e-2,
        )

    @staticmethod
    def _digest(values: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()

    def _independent_damage_history_scales(
        self,
        monolithic: MonolithicGripResidual,
        z: np.ndarray,
        gates: MonolithicNewtonGateConfig,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Rebuild the six-term damage/history scale without production D."""

        _, damage, H_used = monolithic.unpack(z)
        oracle = _IndependentMonolithicOracle(
            self.model,
            monolithic.u_bar,
            monolithic.committed_history,
        )
        displacement = monolithic.reconstruct_displacement(z)
        strain = oracle._strain(oracle.deformation_gradient(displacement))
        driver, _ = oracle._amor(strain)
        H_next = np.maximum(monolithic.committed_history, driver)
        graph = oracle._damage_graph(damage)
        terms = (
            self.model.Gc / self.model.l0 * damage,
            self.model.Gc * self.model.l0 * graph,
            2.0 * H_used * damage,
            2.0 * H_used,
            2.0 * H_next * damage,
            2.0 * H_next,
        )
        residual_scale = max(float(np.max(np.abs(term))) for term in terms)
        history_scale = max(
            float(np.max(np.abs(monolithic.committed_history))),
            float(np.max(np.abs(H_used))),
            float(np.max(np.abs(H_next))),
        )
        return (
            H_used.copy(),
            H_next.copy(),
            residual_scale,
            gates.damage_residual.limit(residual_scale),
            gates.history.limit(history_scale),
        )

    def test_all_newton_controls_are_required_and_strictly_validated(self) -> None:
        fields = inspect.signature(MonolithicNewtonConfig).parameters
        required = (
            "max_newton_iterations",
            "gmres_rtol",
            "gmres_atol",
            "gmres_restart",
            "gmres_maxiter",
            "armijo_c",
            "line_search_reduction",
            "line_search_max_backtracks",
            "line_search_alpha_min",
        )
        self.assertEqual(tuple(fields), required)
        self.assertTrue(all(parameter.default is inspect.Parameter.empty for parameter in fields.values()))
        valid = self._config(
            max_newton_iterations=30,
            armijo_c=1.0e-4,
            line_search_reduction=0.5,
            line_search_max_backtracks=12,
            line_search_alpha_min=1.0e-8,
            gmres_maxiter=200,
        )
        invalid = (
            ("max_newton_iterations", 0),
            ("gmres_rtol", -1.0),
            ("gmres_atol", -1.0),
            ("gmres_restart", 0),
            ("gmres_maxiter", 0),
            ("armijo_c", 0.0),
            ("line_search_reduction", 1.0),
            ("line_search_max_backtracks", -1),
            ("line_search_alpha_min", 0.0),
        )
        for name, value in invalid:
            values = {field: getattr(valid, field) for field in required}
            values[name] = value
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    MonolithicNewtonConfig(**values)
        with self.assertRaises(ValueError):
            MonolithicNewtonConfig(
                max_newton_iterations=1,
                gmres_rtol=0.0,
                gmres_atol=0.0,
                gmres_restart=1,
                gmres_maxiter=1,
                armijo_c=1.0e-4,
                line_search_reduction=0.5,
                line_search_max_backtracks=0,
                line_search_alpha_min=1.0,
            )

    def test_scales_and_scaled_jvp_use_the_declared_row_transform(self) -> None:
        monolithic, z = self._perturbed_state(seed=3)
        gates = self._gates(mechanics_atol=1.0e-13)
        scales = monolithic.block_scales(z, gates)
        H_used, H_next, residual_scale, damage_limit, history_limit = (
            self._independent_damage_history_scales(monolithic, z, gates)
        )
        np.testing.assert_array_equal(scales.H_used, H_used)
        np.testing.assert_allclose(scales.H_next, H_next, rtol=1.0e-10, atol=1.0e-20)
        self.assertGreater(float(np.max(np.abs(H_next - H_used))), 0.0)
        np.testing.assert_allclose(scales.damage_residual_scale, residual_scale, rtol=1.0e-10, atol=1.0e-20)
        np.testing.assert_allclose(scales.Lrho, damage_limit, rtol=1.0e-10, atol=1.0e-20)
        np.testing.assert_allclose(scales.LH, history_limit, rtol=1.0e-10, atol=1.0e-20)
        residual = monolithic.residual(z)
        q, damage_volume, history = monolithic.unpack(residual)
        expected = np.concatenate(
            (q / scales.Lq, damage_volume / monolithic._damage_volumes / scales.Lrho, history / scales.LH)
        )
        np.testing.assert_array_equal(monolithic.scaled_residual(z, scales), expected)
        rng = np.random.default_rng(5)
        direction = rng.standard_normal(monolithic.n_packed) * 1.0e-4
        active = monolithic.active_mask(z)
        step = 1.0e-5
        finite_difference = (
            monolithic.scaled_residual(z + step * direction, scales)
            - monolithic.scaled_residual(z - step * direction, scales)
        ) / (2.0 * step)
        analytic = monolithic.scaled_jvp(z, direction, scales, active)
        defect = np.max(np.abs(finite_difference - analytic)) / max(
            np.max(np.abs(finite_difference)), np.max(np.abs(analytic)), 1.0
        )
        self.assertLess(float(defect), 1.0e-6)

    def test_active_mask_can_switch_while_the_linearization_stays_frozen(self) -> None:
        monolithic, z = self._perturbed_state(seed=7)
        driver = monolithic.psi_plus(z)
        committed = np.full(monolithic.n_material, float(np.median(driver)), dtype=np.float64)
        monolithic = MonolithicGripResidual(self.model, monolithic.u_bar, committed)
        z = monolithic.pack(*monolithic.unpack(z))
        direction = np.zeros(monolithic.n_packed, dtype=np.float64)
        direction[: monolithic.n_free] = 1.0e-3
        switched = z + direction
        old_active = monolithic.active_mask(z)
        new_active = monolithic.active_mask(switched)
        self.assertTrue(np.any(old_active != new_active))
        frozen = monolithic.jvp_with_active_set(switched, direction, old_active)
        current = monolithic.jvp(switched, direction)
        history_start = monolithic.n_free + monolithic.n_material
        self.assertTrue(np.any(frozen[history_start:] != current[history_start:]))

    def test_preconditioner_matches_exact_three_block_definition(self) -> None:
        monolithic, z = self._perturbed_state(seed=11)
        gates = self._gates(mechanics_atol=1.0e-13)
        scales = monolithic.block_scales(z, gates)
        preconditioner = MonolithicBlockDiagonalPreconditioner(monolithic, z, scales)
        rng = np.random.default_rng(13)
        vector = rng.standard_normal(monolithic.n_packed)
        actual = preconditioner.apply(vector)
        expected = np.zeros_like(vector)
        oracle = _IndependentMonolithicOracle(
            self.model,
            monolithic.u_bar,
            monolithic.committed_history,
        )
        independent_mechanics = np.empty(
            (monolithic.n_free, monolithic.n_free), dtype=np.float64
        )
        for column in range(monolithic.n_free):
            direction = np.zeros(monolithic.n_packed, dtype=np.float64)
            direction[column] = 1.0
            independent_mechanics[:, column] = _IndependentMonolithicOracle.jvp(
                oracle, z, direction
            )[: monolithic.n_free]
        expected[: monolithic.n_free] = scales.Lq * np.linalg.solve(
            independent_mechanics, vector[: monolithic.n_free]
        )
        damage_start = monolithic.n_free
        damage_stop = damage_start + monolithic.n_material
        _, damage, H_used = monolithic.unpack(z)
        row = oracle.damage_edges[:, 0]
        col = oracle.damage_edges[:, 1]
        independent_damage_graph = np.zeros(
            (monolithic.n_material, monolithic.n_material), dtype=np.float64
        )
        np.add.at(independent_damage_graph, (row, row), oracle.damage_weights)
        np.add.at(independent_damage_graph, (row, col), -oracle.damage_weights)
        independent_strong_damage = np.diag(
            self.model.Gc / self.model.l0 + 2.0 * H_used
        ) + self.model.Gc * self.model.l0 * independent_damage_graph
        expected[damage_start:damage_stop] = scales.Lrho * np.linalg.solve(
            independent_strong_damage,
            vector[damage_start:damage_stop],
        )
        expected[damage_stop:] = scales.LH * vector[damage_stop:]
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_model_bound_scale_consumers_reject_forged_history_shape(self) -> None:
        monolithic, z = self._perturbed_state(seed=12)
        gates = self._gates(mechanics_atol=1.0e-13)
        scales = monolithic.block_scales(z, gates)
        self.assertGreater(monolithic.n_material, 1)

        # Preserve the packed-state digest and make the one-entry H_used tail
        # agree with z so only the missing model-bound shape contract can reject
        # this generic dataclass bundle.
        forged = replace(
            scales,
            H_used=np.array(z[-1:], dtype=np.float64, copy=True),
            H_next=np.array(scales.H_next[-1:], dtype=np.float64, copy=True),
        )
        self.assertEqual(forged.state_digest, scales.state_digest)
        self.assertEqual(forged.H_used.shape, (1,))

        with self.assertRaisesRegex(ValueError, "monolithic.n_material"):
            MonolithicBlockDiagonalPreconditioner(monolithic, z, forged)
        with self.assertRaisesRegex(ValueError, "monolithic.n_material"):
            monolithic.scaled_residual(z, forged)
        with self.assertRaisesRegex(ValueError, "monolithic.n_material"):
            monolithic.scaled_jvp(
                z,
                np.zeros(monolithic.n_packed, dtype=np.float64),
                forged,
                monolithic.active_mask(z),
            )

        with mock.patch.object(monolithic, "block_scales", return_value=forged):
            with self.assertRaisesRegex(ValueError, "monolithic.n_material"):
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    gates,
                )

    def test_bitwise_zero_packed_residual_is_returned_without_a_gmres_call(self) -> None:
        monolithic = MonolithicGripResidual(
            self.model,
            0.0,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        z = monolithic.pack(
            np.zeros(monolithic.n_free, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
        )
        with mock.patch.object(
            monolithic,
            "residual",
            return_value=np.zeros(monolithic.n_packed, dtype=np.float64),
        ), mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            side_effect=AssertionError("exact root must not enter GMRES"),
        ):
            result = solve_monolithic_newton(
                monolithic,
                z,
                self._config(
                    max_newton_iterations=30,
                    armijo_c=1.0e-4,
                    line_search_reduction=0.5,
                    line_search_max_backtracks=12,
                    line_search_alpha_min=1.0e-8,
                    gmres_maxiter=200,
                ),
                self._gates(mechanics_atol=1.0e-13),
            )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.diagnostics, ())

    def test_initial_active_set_failure_is_retained_as_an_immutable_audit(self) -> None:
        monolithic, z = self._perturbed_state(seed=14)
        config = self._config(
            max_newton_iterations=30,
            armijo_c=1.0e-4,
            line_search_reduction=0.5,
            line_search_max_backtracks=12,
            line_search_alpha_min=1.0e-8,
            gmres_maxiter=200,
        )
        gates = self._gates(mechanics_atol=1.0e-13)

        with mock.patch.object(
            monolithic,
            "active_mask",
            side_effect=FloatingPointError("manufactured initial Amor failure"),
        ):
            with self.assertRaises(MonolithicNewtonFailure) as failure:
                solve_monolithic_newton(monolithic, z, config, gates)

        audit = failure.exception.audit
        self.assertEqual(audit.stage, "initial")
        self.assertIn("initial active-set evaluation failed", audit.reason)
        self.assertIn("manufactured initial Amor failure", audit.reason)
        np.testing.assert_array_equal(audit.current_z, z)
        self.assertFalse(audit.current_z.flags.writeable)
        self.assertEqual(audit.active_mask.shape, (monolithic.n_material,))
        self.assertFalse(audit.active_mask.flags.writeable)
        self.assertIsNone(audit.linearization_scales)
        self.assertEqual(audit.completed_diagnostics, ())

    def test_initial_active_set_structural_failure_propagates(self) -> None:
        monolithic, z = self._perturbed_state(seed=14)
        with mock.patch.object(
            monolithic,
            "active_mask",
            side_effect=TypeError("manufactured structural failure"),
        ):
            with self.assertRaisesRegex(TypeError, "manufactured structural failure"):
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )

    def test_line_search_structural_failure_propagates(self) -> None:
        monolithic, z = self._perturbed_state(seed=14)
        real_scaled_residual = monolithic.scaled_residual
        calls = 0

        def fail_trial(values, scales):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError("manufactured line-search structural failure")
            return real_scaled_residual(values, scales)

        with mock.patch.object(
            monolithic,
            "scaled_residual",
            side_effect=fail_trial,
        ):
            with self.assertRaisesRegex(
                ValueError, "manufactured line-search structural failure"
            ):
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )

    def test_failure_audit_structural_validation_failure_propagates(self) -> None:
        monolithic, z = self._perturbed_state(seed=14)
        real_validate = monolithic_module._validate_model_scale_pair

        def fail_audit_validation(owner, scales, packed, name):
            if name == "Newton failure linearization":
                raise ValueError("manufactured failure-audit structural failure")
            return real_validate(owner, scales, packed, name)

        with mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            return_value=(np.zeros(monolithic.n_packed, dtype=np.float64), 0),
        ), mock.patch.object(
            monolithic_module,
            "_validate_model_scale_pair",
            side_effect=fail_audit_validation,
        ):
            with self.assertRaisesRegex(
                ValueError, "manufactured failure-audit structural failure"
            ):
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )

    def test_tolerance_level_nonroot_executes_newton_and_is_audited(self) -> None:
        monolithic = MonolithicGripResidual(
            self.model,
            0.0,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        z = monolithic.pack(
            np.zeros(monolithic.n_free, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
        )
        raw_residual = monolithic.residual(z)
        self.assertGreater(float(np.max(np.abs(raw_residual))), 0.0)
        real_gmres = scipy.sparse.linalg.gmres
        with mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            wraps=real_gmres,
        ) as gmres_mock:
            result = solve_monolithic_newton(
                monolithic,
                z,
                self._config(
                    max_newton_iterations=30,
                    armijo_c=1.0e-4,
                    line_search_reduction=0.5,
                    line_search_max_backtracks=12,
                    line_search_alpha_min=1.0e-8,
                    gmres_maxiter=200,
                ),
                self._gates(mechanics_atol=1.0e-13),
            )
        self.assertGreaterEqual(gmres_mock.call_count, 1)
        kwargs = gmres_mock.call_args.kwargs
        self.assertEqual(kwargs["callback_type"], "pr_norm")
        self.assertEqual(kwargs["restart"], 50)
        self.assertEqual(kwargs["maxiter"], 200)
        self.assertGreaterEqual(result.iterations, 1)

    def test_perturbed_nonlinear_state_converges_and_records_gmres_audit(self) -> None:
        monolithic, z = self._perturbed_state(seed=1)
        z_before = z.copy()
        gates = self._gates(mechanics_atol=1.0e-13)
        real_gmres = scipy.sparse.linalg.gmres
        with mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            wraps=real_gmres,
        ) as gmres_mock:
            result = solve_monolithic_newton(
                monolithic,
                z,
                self._config(
                    max_newton_iterations=30,
                    armijo_c=1.0e-4,
                    line_search_reduction=0.5,
                    line_search_max_backtracks=12,
                    line_search_alpha_min=1.0e-8,
                    gmres_maxiter=200,
                ),
                gates,
            )
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 0)
        self.assertGreaterEqual(gmres_mock.call_count, 1)
        self.assertEqual(gmres_mock.call_args.kwargs["callback_type"], "pr_norm")
        self.assertEqual(gmres_mock.call_args.kwargs["restart"], 50)
        self.assertEqual(gmres_mock.call_args.kwargs["maxiter"], 200)
        self.assertTrue(np.all(np.isfinite(result.z)))
        self.assertTrue(np.array_equal(z, z_before))
        self.assertTrue(np.all(result.z[monolithic.n_free + monolithic.n_material :] >= 0.0))
        self.assertFalse(result.z.flags.writeable)
        with self.assertRaises(ValueError):
            result.z[0] = result.z[0]
        final_scales = result.final_accepted_trial_scales
        final_H_used, final_H_next, final_residual_scale, final_damage_limit, final_history_limit = (
            self._independent_damage_history_scales(monolithic, result.z, gates)
        )
        np.testing.assert_array_equal(final_scales.H_used, final_H_used)
        np.testing.assert_allclose(final_scales.H_next, final_H_next, rtol=1.0e-10, atol=1.0e-20)
        np.testing.assert_allclose(final_scales.damage_residual_scale, final_residual_scale, rtol=1.0e-10, atol=1.0e-20)
        np.testing.assert_allclose(final_scales.Lrho, final_damage_limit, rtol=1.0e-10, atol=1.0e-20)
        np.testing.assert_allclose(final_scales.LH, final_history_limit, rtol=1.0e-10, atol=1.0e-20)
        self.assertEqual(final_scales.state_digest, result.diagnostics[-1].accepted_trial_scales.state_digest)
        expected_final_scaled = monolithic.scaled_residual(result.z, final_scales)
        np.testing.assert_array_equal(result.final_scaled_residual, expected_final_scaled)
        np.testing.assert_array_equal(result.final_active_mask, monolithic.active_mask(result.z))
        expected_initial, expected_rows, expected_final = result.recompute_with_owner(monolithic)
        np.testing.assert_array_equal(result.initial_residual, expected_initial)
        np.testing.assert_array_equal(result.final_scaled_residual, expected_final)
        for diagnostic, expected in zip(result.diagnostics, expected_rows, strict=True):
            np.testing.assert_array_equal(diagnostic.reference_scaled_residual, expected[0])
            np.testing.assert_array_equal(diagnostic.line_search_scaled_residual, expected[1])
            np.testing.assert_array_equal(diagnostic.accepted_trial_scaled_residual, expected[2])
        for diagnostic in result.diagnostics:
            self.assertFalse(diagnostic.z_before.flags.writeable)
            self.assertFalse(diagnostic.z_after.flags.writeable)
            self.assertFalse(diagnostic.delta.flags.writeable)
            self.assertFalse(diagnostic.active_mask_before.flags.writeable)
            self.assertFalse(diagnostic.active_mask_after.flags.writeable)
            before_H_used, before_H_next, before_residual_scale, before_damage_limit, before_history_limit = (
                self._independent_damage_history_scales(monolithic, diagnostic.z_before, gates)
            )
            after_H_used, after_H_next, after_residual_scale, after_damage_limit, after_history_limit = (
                self._independent_damage_history_scales(monolithic, diagnostic.z_after, gates)
            )
            np.testing.assert_array_equal(diagnostic.linearization_scales.H_used, before_H_used)
            np.testing.assert_allclose(
                diagnostic.linearization_scales.H_next,
                before_H_next,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_array_equal(diagnostic.accepted_trial_scales.H_used, after_H_used)
            np.testing.assert_allclose(
                diagnostic.accepted_trial_scales.H_next,
                after_H_next,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                diagnostic.linearization_scales.damage_residual_scale,
                before_residual_scale,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                diagnostic.linearization_scales.Lrho,
                before_damage_limit,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                diagnostic.linearization_scales.LH,
                before_history_limit,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                diagnostic.accepted_trial_scales.damage_residual_scale,
                after_residual_scale,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                diagnostic.accepted_trial_scales.Lrho,
                after_damage_limit,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            np.testing.assert_allclose(
                diagnostic.accepted_trial_scales.LH,
                after_history_limit,
                rtol=1.0e-10,
                atol=1.0e-20,
            )
            final_raw = monolithic.residual(diagnostic.z_after)
            final_q, final_damage_volume, final_history = monolithic.unpack(final_raw)
            expected_norms = (
                float(np.linalg.norm(final_q / diagnostic.accepted_trial_scales.Lq)),
                float(
                    np.max(
                        np.abs(
                            final_damage_volume
                            / monolithic._damage_volumes
                            / diagnostic.accepted_trial_scales.Lrho
                        )
                    )
                ),
                float(np.max(np.abs(final_history / diagnostic.accepted_trial_scales.LH))),
            )
            np.testing.assert_allclose(diagnostic.scaled_block_norms, expected_norms)
            self.assertEqual(
                diagnostic.active_set_stable,
                bool(np.array_equal(diagnostic.active_mask_before, diagnostic.active_mask_after)),
            )
            q_before = monolithic.reconstruct_displacement(diagnostic.z_before)
            q_after = monolithic.reconstruct_displacement(diagnostic.z_after)
            q_delta = q_after - q_before
            material_dofs = 2 * monolithic.n_material
            independent_correction_values = (
                float(np.max(np.abs(q_delta[:material_dofs]))),
                float(np.max(np.abs(q_delta))),
            )
            independent_displacement_pass = bool(
                independent_correction_values[0]
                <= gates.displacement.limit(
                    max(
                        float(np.max(np.abs(q_before[:material_dofs]))),
                        float(np.max(np.abs(q_after[:material_dofs]))),
                        abs(monolithic.u_bar),
                    )
                )
                and independent_correction_values[1]
                <= gates.displacement.limit(
                    max(
                        float(np.max(np.abs(q_before))),
                        float(np.max(np.abs(q_after))),
                        abs(monolithic.u_bar),
                    )
                )
            )
            _, damage_before, history_before = monolithic.unpack(diagnostic.z_before)
            _, damage_after, history_after = monolithic.unpack(diagnostic.z_after)
            independent_damage = float(np.max(np.abs(damage_after - damage_before)))
            independent_history = float(np.max(np.abs(history_after - history_before)))
            independent_damage_pass = independent_damage <= gates.damage.limit(
                max(float(np.max(np.abs(damage_before))), float(np.max(np.abs(damage_after))))
            )
            independent_history_pass = independent_history <= gates.history.limit(
                max(
                    float(np.max(np.abs(monolithic.committed_history))),
                    float(np.max(np.abs(history_before))),
                    float(np.max(np.abs(history_after))),
                )
            )
            np.testing.assert_allclose(
                (
                    diagnostic.correction_mechanics_material_inf,
                    diagnostic.correction_mechanics_union_inf,
                    diagnostic.correction_damage_inf,
                    diagnostic.correction_history_inf,
                ),
                (
                    independent_correction_values[0],
                    independent_correction_values[1],
                    independent_damage,
                    independent_history,
                ),
            )
            self.assertEqual(diagnostic.correction_displacement_pass, independent_displacement_pass)
            self.assertEqual(diagnostic.correction_damage_pass, independent_damage_pass)
            self.assertEqual(diagnostic.correction_history_pass, independent_history_pass)
            self.assertEqual(
                diagnostic.correction_pass,
                bool(
                    independent_displacement_pass
                    and independent_damage_pass
                    and independent_history_pass
                ),
            )
            self.assertEqual(diagnostic.gmres_callback_type, "pr_norm")
            self.assertTrue(np.all(np.isfinite(diagnostic.gmres_callback_residuals)))
            self.assertLessEqual(
                diagnostic.gmres_unpreconditioned_residual,
                diagnostic.gmres_residual_limit,
            )
            self.assertEqual(diagnostic.trial_alphas[0], 1.0)
            self.assertEqual(len(diagnostic.trial_alphas), len(diagnostic.line_search_rejections))
            self.assertEqual(diagnostic.linearization_scales.state_digest, self._digest(diagnostic.z_before))
            self.assertEqual(diagnostic.accepted_trial_scales.state_digest, self._digest(diagnostic.z_after))
        with self.assertRaises(ValueError):
            replace(
                result.diagnostics[0],
                accepted_trial_scales=result.diagnostics[0].linearization_scales,
            )
        with self.assertRaises(ValueError):
            replace(
                result,
                final_accepted_trial_scales=result.diagnostics[-1].linearization_scales,
            )
        forged_active = result.final_active_mask.copy()
        forged_active[0] = ~forged_active[0]
        forged_block_norms = list(result.final_scaled_block_norms)
        forged_block_norms[0] += 1.0
        for name, value in (
            ("final_scaled_residual", result.final_scaled_residual + 1.0),
            ("final_scaled_block_norms", tuple(forged_block_norms)),
            ("final_merit", result.final_merit + 1.0),
            ("final_active_mask", forged_active),
            ("convergence_status", "forged_status"),
            ("convergence_reason", "exact_initial_residual"),
        ):
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    replace(result, **{name: value})
        self.assertLessEqual(result.final_merit, 0.5)

    def test_armijo_rejection_is_recorded_before_a_reduced_trial_is_accepted(self) -> None:
        monolithic = MonolithicGripResidual(
            self.model,
            0.0,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        z = monolithic.pack(
            np.zeros(monolithic.n_free, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
        )
        original_scaled_residual = monolithic.scaled_residual
        original_scaled_jvp = monolithic.scaled_jvp
        calls = 0
        manufactured_responses: dict[tuple[bytes, str], float] = {}

        def manufactured_merit(
            values: np.ndarray, scales: MonolithicNewtonScales
        ) -> np.ndarray:
            nonlocal calls
            calls += 1
            # The first raw trial has merit 0.40 and fails c=0.50 against
            # reference merit 0.60; the reduced trial has merit 0.44 and
            # passes its Armijo bound 0.45.
            key = (np.asarray(values, dtype=np.float64).tobytes(), scales.state_digest)
            if key not in manufactured_responses:
                unique_response = len(manufactured_responses)
                merit = 0.60 if unique_response == 0 else 0.40 if unique_response == 1 else 0.44
                manufactured_responses[key] = merit
            merit = manufactured_responses[key]
            result = np.zeros(monolithic.n_packed, dtype=np.float64)
            result[0] = np.sqrt(2.0 * merit)
            return result

        def manufactured_jvp(
            values: np.ndarray,
            direction: np.ndarray,
            scales: object,
            active_mask: np.ndarray,
        ) -> np.ndarray:
            return np.array(direction, dtype=np.float64, copy=True)

        config = self._config(
            max_newton_iterations=1,
            armijo_c=0.5,
            line_search_reduction=0.5,
            line_search_max_backtracks=2,
            line_search_alpha_min=0.25,
            gmres_maxiter=5,
        )
        broad_gates = MonolithicNewtonGateConfig(
            mechanics_cg_rtol=1.0e-10,
            mechanics_cg_atol=1.0e-13,
            displacement=GateTolerance(1.0, 0.0, 1.0),
            damage=GateTolerance(1.0, 0.0, 1.0),
            history=GateTolerance(1.0, 0.0, 1.0),
            damage_residual=GateTolerance(1.0, 0.0, 1.0),
        )
        with mock.patch.object(
            monolithic,
            "scaled_residual",
            side_effect=manufactured_merit,
        ), mock.patch.object(
            monolithic,
            "scaled_jvp",
            side_effect=manufactured_jvp,
        ):
            result = solve_monolithic_newton(monolithic, z, config, broad_gates)
        self.assertGreaterEqual(calls, 5)
        self.assertEqual(result.diagnostics[0].accepted_alpha, 0.5)
        self.assertEqual(result.diagnostics[0].line_search_backtracks, 1)
        self.assertIn("Armijo insufficient decrease", result.diagnostics[0].line_search_rejections[0])
        # Retain a reference to the unpatched methods so this test also makes
        # clear that the production object remains usable after the branch.
        self.assertIs(monolithic.scaled_residual.__func__, original_scaled_residual.__func__)
        self.assertIs(monolithic.scaled_jvp.__func__, original_scaled_jvp.__func__)

    def test_raw_damage_outside_unit_interval_is_not_projected(self) -> None:
        monolithic = MonolithicGripResidual(
            self.model,
            0.0,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        damage = np.zeros(monolithic.n_material, dtype=np.float64)
        damage[0] = -0.25
        damage[1] = 1.25
        history = np.ones(monolithic.n_material, dtype=np.float64)
        z = monolithic.pack(
            np.zeros(monolithic.n_free, dtype=np.float64), damage, history
        )
        z_before = z.copy()
        scales = monolithic.block_scales(z, self._gates(mechanics_atol=1.0))
        self.assertTrue(np.all(np.isfinite(monolithic.scaled_residual(z, scales))))
        np.testing.assert_array_equal(monolithic.unpack(z)[1], damage)
        np.testing.assert_array_equal(monolithic.unpack(z)[2], history)
        np.testing.assert_array_equal(z, z_before)

    def test_forced_gmres_and_line_search_failures_are_hard(self) -> None:
        monolithic, z = self._perturbed_state(seed=17)
        with self.assertRaises(MonolithicNewtonFailure) as gmres_failure:
            solve_monolithic_newton(
                monolithic,
                z,
                self._config(
                    max_newton_iterations=30,
                    armijo_c=1.0e-4,
                    line_search_reduction=0.5,
                    line_search_max_backtracks=12,
                    line_search_alpha_min=1.0e-8,
                    gmres_maxiter=1,
                ),
                self._gates(mechanics_atol=1.0e-13),
            )
        gmres_audit = gmres_failure.exception.audit
        self.assertIsInstance(gmres_audit, MonolithicNewtonFailureAudit)
        self.assertEqual(gmres_audit.stage, "gmres")
        self.assertEqual(gmres_audit.gmres_info, 1)
        self.assertIsNotNone(gmres_audit.linearization_scales)
        self.assertEqual(gmres_audit.completed_diagnostics, ())
        self.assertFalse(gmres_audit.current_z.flags.writeable)
        self.assertFalse(gmres_audit.active_mask.flags.writeable)
        with self.assertRaises(ValueError):
            gmres_audit.current_z[0] = gmres_audit.current_z[0]
        with self.assertRaises((AttributeError, TypeError)):
            setattr(gmres_audit, "reason", "forged")

        with mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            return_value=(np.zeros(monolithic.n_packed, dtype=np.float64), 0),
        ):
            with self.assertRaises(MonolithicNewtonFailure) as bad_residual_failure:
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )
        bad_residual_audit = bad_residual_failure.exception.audit
        self.assertEqual(bad_residual_audit.stage, "gmres_residual")
        self.assertEqual(bad_residual_audit.gmres_info, 0)
        self.assertIsNotNone(bad_residual_audit.gmres_unpreconditioned_residual)
        self.assertIsNotNone(bad_residual_audit.gmres_residual_limit)
        bad_residual_value = float(cast(float, bad_residual_audit.gmres_unpreconditioned_residual))
        bad_residual_limit = float(cast(float, bad_residual_audit.gmres_residual_limit))
        self.assertGreater(
            bad_residual_value,
            bad_residual_limit,
        )

        with mock.patch.object(
            scipy.sparse.linalg,
            "splu",
            side_effect=RuntimeError("manufactured sparse factorization failure"),
        ):
            with self.assertRaises(MonolithicNewtonFailure) as preconditioner_failure:
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )
        preconditioner_audit = preconditioner_failure.exception.audit
        self.assertEqual(preconditioner_audit.stage, "preconditioner")
        self.assertEqual(preconditioner_audit.completed_diagnostics, ())
        self.assertIsNotNone(preconditioner_audit.linearization_scales)

        with mock.patch.object(
            scipy.sparse.linalg,
            "splu",
            side_effect=ValueError("manufactured structural factorization failure"),
        ):
            with self.assertRaisesRegex(
                ValueError, "manufactured structural factorization failure"
            ):
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=12,
                        line_search_alpha_min=1.0e-8,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )

        max_iteration_monolithic, max_iteration_z = self._perturbed_state(seed=1)
        with self.assertRaises(MonolithicNewtonFailure) as max_iteration_failure:
            solve_monolithic_newton(
                max_iteration_monolithic,
                max_iteration_z,
                self._config(
                    max_newton_iterations=1,
                    armijo_c=1.0e-4,
                    line_search_reduction=0.5,
                    line_search_max_backtracks=12,
                    line_search_alpha_min=1.0e-8,
                    gmres_maxiter=200,
                ),
                self._gates(mechanics_atol=1.0e-13),
            )
        max_iteration_audit = max_iteration_failure.exception.audit
        self.assertEqual(max_iteration_audit.stage, "max_newton_iterations")
        self.assertEqual(len(max_iteration_audit.completed_diagnostics), 1)
        self.assertTrue(
            np.array_equal(
                max_iteration_audit.current_z,
                max_iteration_audit.completed_diagnostics[-1].z_after,
            )
        )
        self.assertIsNotNone(max_iteration_audit.linearization_scales)

        constant_scaled_residual = np.zeros(monolithic.n_packed, dtype=np.float64)
        constant_scaled_residual[0] = 2.0

        def constant_jvp(
            values: np.ndarray,
            direction: np.ndarray,
            scales: object,
            active_mask: np.ndarray,
        ) -> np.ndarray:
            result = np.zeros_like(direction)
            result[0] = -2.0
            return result

        with mock.patch.object(
            monolithic,
            "scaled_residual",
            return_value=constant_scaled_residual,
        ), mock.patch.object(
            monolithic,
            "scaled_jvp",
            side_effect=constant_jvp,
        ), mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            return_value=(np.zeros(monolithic.n_packed, dtype=np.float64), 0),
        ):
            with self.assertRaises(MonolithicNewtonFailure) as minimum_alpha_failure:
                solve_monolithic_newton(
                    monolithic,
                    z,
                    self._config(
                        max_newton_iterations=1,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=1,
                        line_search_alpha_min=1.0,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )
        minimum_alpha_audit = minimum_alpha_failure.exception.audit
        self.assertEqual(minimum_alpha_audit.stage, "line_search")
        self.assertIn("line_search_alpha_min", minimum_alpha_failure.exception.audit.reason)

        rng = np.random.default_rng(1)
        hard_history = np.full(monolithic.n_material, 1.0e-2, dtype=np.float64)
        raw_trial_state = monolithic.pack(
            rng.standard_normal(monolithic.n_free) * 1.0e-4,
            rng.random(monolithic.n_material) * 1.0e-4,
            hard_history,
        )
        with self.assertRaises(MonolithicNewtonFailure) as line_search_failure:
            solve_monolithic_newton(
                monolithic,
                raw_trial_state,
                self._config(
                    max_newton_iterations=1,
                    armijo_c=1.0e-4,
                    line_search_reduction=0.5,
                    line_search_max_backtracks=0,
                    line_search_alpha_min=1.0,
                    gmres_maxiter=200,
                ),
                self._gates(mechanics_atol=1.0e-13),
            )
        line_search_audit = line_search_failure.exception.audit
        self.assertEqual(line_search_audit.stage, "max_newton_iterations")
        self.assertEqual(len(line_search_audit.completed_diagnostics), 1)
        intermediate = line_search_audit.completed_diagnostics[0]
        self.assertLess(intermediate.z_after[-monolithic.n_material :].min(), 0.0)
        self.assertFalse(
            any("H<0" in reason for reason in intermediate.line_search_rejections)
        )
        self.assertEqual(intermediate.line_search_rejections, ("",))
        self.assertIsNotNone(intermediate.gmres_info)

        successful = solve_monolithic_newton(
            monolithic,
            raw_trial_state,
            self._config(
                max_newton_iterations=30,
                armijo_c=1.0e-4,
                line_search_reduction=0.5,
                line_search_max_backtracks=0,
                line_search_alpha_min=1.0,
                gmres_maxiter=200,
            ),
            self._gates(mechanics_atol=1.0e-13),
        )
        self.assertTrue(successful.converged)
        self.assertGreaterEqual(successful.z[-monolithic.n_material :].min(), 0.0)

        fake_scaled_initial = np.zeros(monolithic.n_packed, dtype=np.float64)
        fake_scaled_initial[0] = 2.0
        huge_direction = np.full(monolithic.n_packed, 1.0e308, dtype=np.float64)
        overflow_state = monolithic.pack(
            np.zeros(monolithic.n_free, dtype=np.float64),
            np.zeros(monolithic.n_material, dtype=np.float64),
            np.full(monolithic.n_material, 8.0e307, dtype=np.float64),
        )

        def tiny_jvp(
            values: np.ndarray,
            direction: np.ndarray,
            scales: object,
            active_mask: np.ndarray,
        ) -> np.ndarray:
            result = np.zeros_like(direction)
            result[0] = -2.0e-308 * direction[0]
            return result

        with mock.patch.object(
            monolithic,
            "scaled_residual",
            return_value=fake_scaled_initial,
        ), mock.patch.object(
            monolithic,
            "scaled_jvp",
            side_effect=tiny_jvp,
        ), mock.patch.object(
            scipy.sparse.linalg,
            "gmres",
            return_value=(huge_direction, 0),
        ):
            with self.assertRaises(MonolithicNewtonFailure) as nonfinite_failure:
                solve_monolithic_newton(
                    monolithic,
                    overflow_state,
                    self._config(
                        max_newton_iterations=30,
                        armijo_c=1.0e-4,
                        line_search_reduction=0.5,
                        line_search_max_backtracks=0,
                        line_search_alpha_min=1.0,
                        gmres_maxiter=200,
                    ),
                    self._gates(mechanics_atol=1.0e-13),
                )
        nonfinite_audit = nonfinite_failure.exception.audit
        self.assertEqual(nonfinite_audit.stage, "line_search")
        self.assertIn("non-finite raw trial", nonfinite_audit.rejection_reasons[0])


if __name__ == "__main__":
    unittest.main()
