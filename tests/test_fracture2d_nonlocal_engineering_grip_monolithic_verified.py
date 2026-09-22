"""Manufactured Phase-2B single-target verified-attempt tests.

The tests use the existing 5x5 engineering-grip cloud only.  They exercise
the explicit fixed-target API, the authoritative auxiliary sweep, canonical
state pairing, and immutable numerical failure audits.  No continuation,
workflow, CLI, archive, or rounded-slot physical run is invoked.
"""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

import jax
import numpy as np

from examples.fracture2d import nonlocal_a2_coupled as a2
from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import nonlocal_engineering_grip_monolithic as monolithic_module
from examples.fracture2d.nonlocal_engineering_grip_monolithic import (
    MonolithicAuxiliarySweep,
    MonolithicGripResidual,
    MonolithicNewtonConfig,
    MonolithicVerifiedAttempt,
    run_verified_monolithic_attempt,
)
from examples.fracture2d.nonlocal_grip_diagnostic import build_union_geometry_from_arrays
from tests.test_fracture2d_nonlocal_engineering_grip_monolithic import _material_cloud


class MonolithicVerifiedAttemptTest(unittest.TestCase):
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
        cls.config = engineering.EngineeringGripConfig.smoke()

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    def _committed(self) -> engineering.EngineeringGripState:
        return engineering.EngineeringGripState(
            u_bar=0.0,
            union_displacement=np.zeros(self.model.n_dofs, dtype=np.float64),
            material_damage=np.zeros(self.model.n_material, dtype=np.float64),
            material_history=np.zeros(self.model.n_material, dtype=np.float64),
        )

    def _newton_config(self, *, max_iterations: int = 30) -> MonolithicNewtonConfig:
        return MonolithicNewtonConfig(
            max_newton_iterations=max_iterations,
            gmres_rtol=1.0e-10,
            gmres_atol=1.0e-13,
            gmres_restart=50,
            gmres_maxiter=200,
            armijo_c=1.0e-4,
            line_search_reduction=0.5,
            line_search_max_backtracks=12,
            line_search_alpha_min=1.0e-8,
        )

    def _initial(self, seed: int = 1) -> np.ndarray:
        committed = self._committed()
        monolithic = MonolithicGripResidual(
            self.model,
            1.0e-4,
            committed.material_history,
        )
        rng = np.random.default_rng(seed)
        return monolithic.pack(
            rng.standard_normal(monolithic.n_free) * 1.0e-4,
            rng.random(monolithic.n_material) * 1.0e-2,
            rng.random(monolithic.n_material) * 1.0e-2,
        )

    def test_api_requires_all_explicit_inputs_without_defaults(self) -> None:
        parameters = inspect.signature(run_verified_monolithic_attempt).parameters
        self.assertEqual(
            tuple(parameters),
            ("model", "config", "committed", "target", "adapter", "initial_z", "newton_config"),
        )
        self.assertTrue(
            all(parameter.default is inspect.Parameter.empty for parameter in parameters.values())
        )

    def test_success_is_canonical_auxiliary_state_and_immutable(self) -> None:
        initial = self._initial()
        initial_before = initial.copy()
        committed = self._committed()
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            committed,
            1.0e-4,
            None,
            initial,
            self._newton_config(),
        )
        self.assertIsInstance(result, MonolithicVerifiedAttempt)
        self.assertTrue(result.accepted)
        self.assertIsNone(result.failure)
        self.assertIsNotNone(result.state)
        self.assertIsNotNone(result.newton_result)
        self.assertIsNotNone(result.auxiliary)
        newton_result = result.newton_result
        assert newton_result is not None
        auxiliary = result.auxiliary
        assert auxiliary is not None
        self.assertIsInstance(auxiliary, MonolithicAuxiliarySweep)
        self.assertTrue(auxiliary.complete_sweep)
        self.assertTrue(auxiliary.passed)
        self.assertEqual(
            set(auxiliary.gate_passes),
            {
                "finite",
                "mechanics_before",
                "mechanics_after",
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
                "coupled",
            },
        )
        self.assertTrue(all(auxiliary.gate_passes.values()))
        self.assertTrue(all(result.gate_passes.values()))
        self.assertTrue(all(result.extended_gate_passes.values()))
        assert result.state is not None
        assert result.canonical_z is not None
        expected_canonical = np.concatenate(
            (
                result.state.q[self.model.constraints.free_dofs],
                result.state.d,
                result.state.H,
            )
        )
        np.testing.assert_array_equal(result.canonical_z, expected_canonical)
        np.testing.assert_array_equal(auxiliary.raw_z, newton_result.z)
        np.testing.assert_array_equal(auxiliary.d_solved, result.state.d)
        np.testing.assert_array_equal(auxiliary.H_next, result.state.H)
        np.testing.assert_array_equal(initial, initial_before)
        self.assertEqual(result.committed_before_digest, committed.digest)
        self.assertEqual(result.committed_after_digest, result.state.digest)
        for array in (
            result.initial_z,
            result.canonical_z,
            auxiliary.raw_z,
            auxiliary.raw_history,
            auxiliary.d_solved,
            auxiliary.H_used,
            auxiliary.H_next,
        ):
            assert array is not None
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0] = array[0]

    def test_rejected_newton_retains_failure_audit_and_rolls_back_digest(self) -> None:
        committed = self._committed()
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            committed,
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(max_iterations=1),
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.failure_stage, "newton")
        self.assertEqual(result.committed_before_digest, committed.digest)
        self.assertEqual(result.committed_after_digest, committed.digest)
        self.assertIsNone(result.state)
        self.assertIsNotNone(result.failure)
        assert result.failure is not None
        self.assertIsNotNone(result.failure.newton_audit)
        assert result.newton_audit is not None
        self.assertIs(result.failure.newton_audit, result.newton_audit)
        self.assertEqual(result.newton_audit.target_u_bar, result.target_u_bar)
        self.assertIs(result.newton_audit.model_identity, result.model)
        self.assertIs(result.newton_audit.static_identity, result.model.static)
        self.assertIs(result.newton_audit.config, result.newton_config)
        self.assertIs(result.newton_audit.gate_config, result.newton_gate_config)
        self.assertFalse(result.failure.current_z.flags.writeable)
        with self.assertRaises(ValueError):
            result.failure.current_z[0] = result.failure.current_z[0]

    def test_auxiliary_numerical_failure_is_distinct_from_structural_failure(self) -> None:
        committed = self._committed()
        with mock.patch.object(
            engineering,
            "_solve_direct_mechanics",
            side_effect=RuntimeError("manufactured auxiliary mechanics failure"),
        ):
            result = run_verified_monolithic_attempt(
                self.model,
                self.config,
                committed,
                1.0e-4,
                None,
                self._initial(),
                self._newton_config(),
            )
        self.assertFalse(result.accepted)
        self.assertEqual(result.failure_stage, "auxiliary")
        self.assertEqual(result.exception_type, "RuntimeError")
        self.assertIsNotNone(result.auxiliary)
        assert result.auxiliary is not None
        assert result.failure is not None
        self.assertIs(result.failure.auxiliary, result.auxiliary)
        self.assertEqual(result.auxiliary.failure_stage, "mechanics_before")
        self.assertIn("manufactured auxiliary mechanics failure", result.failure_reason or "")
        with self.assertRaises(TypeError):
            run_verified_monolithic_attempt(
                self.model,
                self.config,
                committed,
                1.0e-4,
                object(),
                self._initial(),
                self._newton_config(),
            )

    def test_fixed_history_residual_failures_retain_distinct_stages(self) -> None:
        committed = self._committed()
        for stage, side_effect in (
            ("rho_used", RuntimeError("manufactured rho_used failure")),
            (
                "rho_next",
                [None, RuntimeError("manufactured rho_next failure")],
            ),
        ):
            with self.subTest(stage=stage), mock.patch.object(
                engineering,
                "_compare_damage_residual_reconstruction",
                side_effect=side_effect,
            ):
                result = run_verified_monolithic_attempt(
                    self.model,
                    self.config,
                    committed,
                    1.0e-4,
                    None,
                    self._initial(),
                    self._newton_config(),
                )
            self.assertFalse(result.accepted)
            self.assertEqual(result.failure_stage, "auxiliary")
            self.assertIsNotNone(result.auxiliary)
            assert result.auxiliary is not None
            self.assertEqual(result.auxiliary.failure_stage, stage)
            self.assertIsNotNone(result.failure)
            assert result.failure is not None
            self.assertIs(result.failure.auxiliary, result.auxiliary)
            self.assertEqual(result.failure.exception_type, "RuntimeError")

    def test_target_and_initial_history_flow_validation_is_loud(self) -> None:
        committed = self._committed()
        with self.assertRaises(ValueError):
            run_verified_monolithic_attempt(
                self.model,
                self.config,
                committed,
                0.0,
                None,
                self._initial(),
                self._newton_config(),
            )
        negative = self._initial()
        negative[-1] = -1.0
        with self.assertRaises(ValueError):
            run_verified_monolithic_attempt(
                self.model,
                self.config,
                committed,
                1.0e-4,
                None,
                negative,
                self._newton_config(),
            )

    def test_forged_accepted_digest_pair_is_rejected(self) -> None:
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            self._committed(),
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(),
        )
        self.assertTrue(result.accepted)
        with self.assertRaises(ValueError):
            replace(result, committed_after_digest=result.committed_before_digest)

    def test_owner_replay_authenticates_cpu_result_without_public_entrypoint(self) -> None:
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            self._committed(),
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(),
        )
        self.assertTrue(result.accepted)
        with mock.patch.object(
            monolithic_module,
            "run_verified_monolithic_attempt",
            side_effect=AssertionError("public replay entry point was replaced"),
        ):
            result.validate_with_owner(self.model, self.config)

    def test_owner_replay_rejects_self_consistent_forged_auxiliary_value(self) -> None:
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            self._committed(),
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(),
        )
        self.assertTrue(result.accepted)
        auxiliary = result.auxiliary
        assert auxiliary is not None
        mechanics = auxiliary.mechanics_before
        assert mechanics is not None
        forged_residual = mechanics.evaluation.residual.copy()
        top_dof = int(np.flatnonzero(mechanics.constraints.outer_top_selector)[0])
        forged_residual[top_dof] += 1.0e-6
        forged_evaluation = replace(
            mechanics.evaluation,
            energy=mechanics.evaluation.energy + 1.0e-9,
            residual=forged_residual,
        )
        forged_free = forged_residual[mechanics.constraints.free_dofs]
        forged_matrix = forged_residual.reshape(self.model.n_union, 2)
        forged_mechanics = replace(
            mechanics,
            evaluation=forged_evaluation,
            free_residual=forged_free,
            free_residual_norm=float(np.linalg.norm(forged_free)),
            free_residual_relative_norm=(
                float(np.linalg.norm(forged_free)) / max(mechanics.free_rhs_norm, 1.0e-300)
            ),
            outer_reaction=float(
                mechanics.constraints.outer_top_selector @ forged_residual
            ),
            bottom_reaction=float(
                mechanics.constraints.outer_bottom_selector @ forged_residual
            ),
            anchor_reaction=float(
                mechanics.constraints.anchor_selector @ forged_residual
            ),
            total_force=np.sum(forged_matrix, axis=0),
            total_moment=float(
                np.sum(
                    self.model.static.coordinates[:, 0] * forged_matrix[:, 1]
                    - self.model.static.coordinates[:, 1] * forged_matrix[:, 0]
                )
            ),
        )
        forged_auxiliary = replace(auxiliary, mechanics_before=forged_mechanics)
        forged = replace(result, auxiliary=forged_auxiliary)
        forged.validate()
        with self.assertRaises(ValueError):
            forged.validate_with_owner(self.model, self.config)

    def test_forged_auxiliary_or_canonical_pair_is_rejected(self) -> None:
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            self._committed(),
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(),
        )
        self.assertTrue(result.accepted)
        assert result.auxiliary is not None
        assert result.canonical_active_mask is not None
        forged_damage = result.auxiliary.raw_damage.copy()
        forged_damage[0] += 1.0e-6
        with self.assertRaises(ValueError):
            replace(result.auxiliary, raw_damage=forged_damage)
        forged_active = result.canonical_active_mask.copy()
        forged_active[0] = ~forged_active[0]
        with self.assertRaises(ValueError):
            replace(result, canonical_active_mask=forged_active)
        assert result.newton_result is not None
        forged_newton = result.newton_result.z.copy()
        forged_newton[-1] = -1.0
        with self.assertRaises(ValueError):
            replace(result.newton_result, z=forged_newton)
        assert result.canonical_z is not None
        forged_canonical = result.canonical_z.copy()
        forged_canonical[-1] = -1.0
        with self.assertRaises(ValueError):
            replace(result, canonical_z=forged_canonical)

    def test_auxiliary_gate_conjunction_is_independent_and_each_gate_forgery_fails(self) -> None:
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            self._committed(),
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(),
        )
        self.assertTrue(result.accepted)
        assert result.auxiliary is not None
        diagnostics = result.auxiliary.diagnostics
        gate_fields = {
            "finite": "finite_pass",
            "mechanics_before": "mechanics_before_pass",
            "mechanics_after": "mechanics_after_pass",
            "displacement": "displacement_pass",
            "damage": "damage_pass",
            "history": "history_pass",
            "reaction": "reaction_pass",
            "damage_residual": "damage_residual_pass",
            "damage_range": "damage_range_pass",
            "healing": "healing_pass",
            "history_monotonic": "history_monotonic_pass",
            "force": "force_pass",
            "moment": "moment_pass",
        }
        expected = {name: bool(getattr(diagnostics, field)) for name, field in gate_fields.items()}
        self.assertEqual(diagnostics.passed, all(expected.values()))
        expected["coupled"] = bool(diagnostics.complete_sweep and all(expected.values()))
        self.assertEqual(result.auxiliary.gate_passes, expected)
        self.assertEqual(result.auxiliary.passed, all(expected.values()))
        for field in gate_fields.values():
            forged_diagnostics = replace(diagnostics, **{field: False})
            with self.assertRaises(ValueError):
                replace(result.auxiliary, diagnostics=forged_diagnostics)
        for gate_name, gate_value in result.canonical_gate_passes:
            if gate_name == "coupled":
                continue
            forged_canonical = tuple(
                (name, False if name in (gate_name, "coupled") else value)
                for name, value in result.canonical_gate_passes
            )
            with self.assertRaises(ValueError):
                replace(result, canonical_gate_passes=forged_canonical)

    def test_authoritative_call_order_and_seed_provenance_are_independent(self) -> None:
        events: list[str] = []
        seed_records: list[tuple[str, np.ndarray, str, str]] = []
        original_mechanics = engineering._solve_direct_mechanics
        original_amor = monolithic_module.compute_material_source_amor
        original_damage = engineering.solve_fixed_history_damage
        original_residuals = engineering._damage_residuals
        original_full = engineering._full_diagnostics
        original_mechanics_gate = engineering._mechanics_gate

        def mechanics_probe(*args: Any, **kwargs: Any) -> Any:
            seed_kind = str(kwargs["seed_kind"])
            stage = "mechanics_before" if seed_kind == "monolithic_newton_candidate" else "mechanics_after"
            view = args[1]
            assert isinstance(view, a2.A2FixedDamageView)
            initial_guess = kwargs["initial_guess"]
            assert isinstance(initial_guess, np.ndarray)
            seed_records.append(
                (
                    stage,
                    np.array(initial_guess, dtype=np.float64, copy=True),
                    str(kwargs["source_damage_digest"]),
                    view.damage_digest,
                )
            )
            events.append(stage)
            return original_mechanics(*args, **kwargs)

        def amor_probe(*args: Any, **kwargs: Any) -> Any:
            del kwargs
            if "mechanics_before" not in events:
                events.append("newton_amor")
            else:
                events.append(
                    "amor_before"
                    if "amor_before" not in events
                    else "amor_after"
                )
            return original_amor(*args)

        def damage_probe(*args: Any, **kwargs: Any) -> Any:
            del kwargs
            events.append("damage")
            return original_damage(*args)

        def residual_probe(*args: Any, **kwargs: Any) -> Any:
            del kwargs
            events.append("rho_next")
            return original_residuals(*args)

        def full_probe(*args: Any, **kwargs: Any) -> Any:
            del kwargs
            events.append("full_diagnostics")
            return original_full(*args)

        def mechanics_gate_probe(*args: Any, **kwargs: Any) -> Any:
            del kwargs
            events.append("force_moment_gates")
            return original_mechanics_gate(*args)

        with mock.patch.object(engineering, "_solve_direct_mechanics", side_effect=mechanics_probe), mock.patch.object(
            monolithic_module, "compute_material_source_amor", side_effect=amor_probe
        ), mock.patch.object(
            engineering, "solve_fixed_history_damage", side_effect=damage_probe
        ), mock.patch.object(
            engineering, "_damage_residuals", side_effect=residual_probe
        ), mock.patch.object(
            engineering, "_full_diagnostics", side_effect=full_probe
        ), mock.patch.object(
            engineering, "_mechanics_gate", side_effect=mechanics_gate_probe
        ):
            result = run_verified_monolithic_attempt(
                self.model,
                self.config,
                self._committed(),
                1.0e-4,
                None,
                self._initial(),
                self._newton_config(),
            )
        self.assertTrue(result.accepted)
        full_index = events.index("full_diagnostics")
        self.assertEqual(
            events[events.index("mechanics_before") : full_index + 3],
            [
                "mechanics_before",
                "amor_before",
                "damage",
                "mechanics_after",
                "amor_after",
                "rho_next",
                "full_diagnostics",
                "force_moment_gates",
                "force_moment_gates",
            ],
        )
        assert result.auxiliary is not None
        assert result.newton_result is not None
        assert result.auxiliary.d_solved is not None
        mechanics_before = result.auxiliary.mechanics_before
        mechanics_after = result.auxiliary.mechanics_after
        assert mechanics_before is not None
        assert mechanics_after is not None
        expected_raw_seed = result.auxiliary.reconstruct_raw_displacement(result.auxiliary.raw_z)
        expected_after_seed = mechanics_before
        self.assertEqual(len(seed_records), 2)
        np.testing.assert_array_equal(seed_records[0][1], expected_raw_seed)
        np.testing.assert_array_equal(seed_records[1][1], expected_after_seed.displacement)
        raw_digest = engineering._damage_digest(result.auxiliary.raw_damage)
        self.assertEqual(seed_records[0][2], raw_digest)
        self.assertEqual(seed_records[1][2], raw_digest)
        self.assertEqual(seed_records[0][3], raw_digest)
        self.assertEqual(seed_records[1][3], engineering._damage_digest(result.auxiliary.d_solved))
        self.assertEqual(mechanics_before.seed_provenance.source_damage_digest, raw_digest)
        self.assertEqual(mechanics_after.seed_provenance.source_damage_digest, raw_digest)
        self.assertEqual(
            mechanics_before.seed_provenance.seed_digest,
            a2._digest_primitives((("seed", expected_raw_seed),)),
        )
        self.assertEqual(
            mechanics_after.seed_provenance.seed_digest,
            a2._digest_primitives((("seed", expected_after_seed.displacement),)),
        )

    def test_foreign_target_identity_and_rejected_foreign_auxiliary_fail(self) -> None:
        committed = self._committed()
        result = run_verified_monolithic_attempt(
            self.model,
            self.config,
            committed,
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(),
        )
        self.assertTrue(result.accepted)
        assert result.auxiliary is not None
        with self.assertRaises(ValueError):
            replace(result.auxiliary, target_u_bar=5.0e-5)
        with self.assertRaises(ValueError):
            replace(result.auxiliary, q_before=None)
        with self.assertRaises(ValueError):
            replace(
                result.auxiliary,
                diagnostics=replace(result.auxiliary.diagnostics, q_before=None),
            )
        for field in (
            "q_before",
            "q_after",
            "psi_before",
            "H_used",
            "d_solved",
            "psi_after",
            "H_next",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        ):
            value = getattr(result.auxiliary, field)
            assert value is not None
            with self.assertRaises(ValueError):
                replace(result.auxiliary, **{field: None})
            forged_value = value.copy()
            forged_value.flat[0] += 1.0e-12
            with self.assertRaises(ValueError):
                replace(result.auxiliary, **{field: forged_value})
        with self.assertRaises(ValueError):
            replace(result, target_u_bar=5.0e-5)
        with self.assertRaises(ValueError):
            replace(result, auxiliary=replace(result.auxiliary, config=engineering.EngineeringGripConfig.smoke()))
        with self.assertRaises((TypeError, ValueError)):
            replace(result.auxiliary, model=object())
        assert result.newton_result is not None
        with self.assertRaises(ValueError):
            replace(
                result,
                newton_result=replace(result.newton_result, target_u_bar=5.0e-5),
            )

        foreign = run_verified_monolithic_attempt(
            self.model,
            self.config,
            committed,
            5.0e-5,
            None,
            self._initial(seed=2),
            self._newton_config(),
        )
        self.assertTrue(foreign.accepted)
        assert foreign.auxiliary is not None
        with self.assertRaises(ValueError):
            replace(result, auxiliary=foreign.auxiliary)

        rejected = run_verified_monolithic_attempt(
            self.model,
            self.config,
            committed,
            1.0e-4,
            None,
            self._initial(),
            self._newton_config(max_iterations=1),
        )
        self.assertFalse(rejected.accepted)
        assert rejected.failure is not None
        forged_failure = replace(rejected.failure, auxiliary=result.auxiliary)
        with self.assertRaises(ValueError):
            replace(rejected, failure=forged_failure)


if __name__ == "__main__":
    unittest.main()
