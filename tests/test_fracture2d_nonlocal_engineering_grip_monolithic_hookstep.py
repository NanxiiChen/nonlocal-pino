"""Manufactured-only tests for the isolated monolithic hookstep algebra."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

import jax
import numpy as np

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d.nonlocal_baseline import GateTolerance
from examples.fracture2d.nonlocal_engineering_grip_monolithic import (
    MonolithicBlockDiagonalPreconditioner,
    MonolithicGripResidual,
    MonolithicHookstepArnoldi,
    MonolithicHookstepConfig,
    MonolithicHookstepFailure,
    MonolithicNewtonGateConfig,
    build_monolithic_hookstep_arnoldi,
    build_monolithic_trust_metric,
    compute_monolithic_hookstep_direction,
    decide_monolithic_hookstep_radius,
    solve_projected_monolithic_hookstep,
)
from examples.fracture2d.nonlocal_grip_diagnostic import (
    build_union_geometry_from_arrays,
)
from tests.test_fracture2d_nonlocal_engineering_grip_monolithic import (
    _IndependentMonolithicOracle,
    _material_cloud,
)


def _config(**overrides: object) -> MonolithicHookstepConfig:
    values: dict[str, Any] = {
        "max_krylov_dimension": 8,
        "arnoldi_breakdown_rtol": 1.0e-13,
        "arnoldi_orthogonality_rtol": 1.0e-11,
        "metric_rank_rtol": 1.0e-12,
        "projected_rank_rtol": 1.0e-12,
        "secular_rtol": 1.0e-12,
        "prediction_consistency_rtol": 1.0e-10,
        "metric_consistency_rtol": 1.0e-10,
        "secular_max_iterations": 100,
        "trust_radius_initial": 1.0,
        "trust_radius_min": 1.0e-8,
        "trust_radius_max": 10.0,
        "max_radius_rejections": 8,
        "eta_accept": 0.1,
        "eta_expand": 0.75,
        "gamma_shrink": 0.25,
        "gamma_expand": 2.0,
        "boundary_fraction": 0.9,
    }
    values.update(overrides)
    return MonolithicHookstepConfig(**values)


def _model_from_cloud() -> engineering.EngineeringGripModel:
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
    return engineering.build_engineering_grip_model(
        geometry,
        2.0,
        0.3,
        1.0e-8,
        1.0e-12,
        1.7,
        0.65,
    )


def _regular_square_model(n_side: int) -> engineering.EngineeringGripModel:
    axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    coordinates = np.asarray([(x, y) for x in axis for y in axis], dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n_side - 1):
        for iy in range(n_side - 1):
            lower_left = ix * n_side + iy
            lower_right = (ix + 1) * n_side + iy
            upper_left = lower_left + 1
            upper_right = lower_right + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    triangle_array = np.asarray(triangles, dtype=np.int64)
    first = coordinates[triangle_array[:, 0]]
    second = coordinates[triangle_array[:, 1]]
    third = coordinates[triangle_array[:, 2]]
    areas = 0.5 * np.abs(
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    )
    volumes = np.zeros(len(coordinates), dtype=np.float64)
    for column in range(3):
        np.add.at(volumes, triangle_array[:, column], areas / 3.0)
    delta = 2.1 / float(n_side - 1)
    pairs: list[tuple[int, int]] = []
    for source in range(len(coordinates)):
        for target in range(len(coordinates)):
            distance = float(np.linalg.norm(coordinates[target] - coordinates[source]))
            if 0.0 < distance < delta:
                pairs.append((source, target))
    geometry = build_union_geometry_from_arrays(
        coordinates,
        triangle_array,
        volumes,
        coordinates[:, 1] == 0.0,
        coordinates[:, 1] == 1.0,
        0,
        np.asarray(pairs, dtype=np.int64),
        delta,
        0.2,
        0.0,
    )
    return engineering.build_engineering_grip_model(
        geometry,
        2.0,
        0.3,
        1.0e-8,
        1.0e-12,
        1.7,
        0.65,
    )


class MonolithicHookstepCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        cls.model = _model_from_cloud()

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    def _owner_and_state(self) -> tuple[MonolithicGripResidual, np.ndarray]:
        owner = MonolithicGripResidual(
            self.model,
            1.0e-4,
            np.zeros(self.model.n_material, dtype=np.float64),
        )
        rng = np.random.default_rng(712)
        state = owner.pack(
            rng.standard_normal(owner.n_free) * 1.0e-5,
            rng.random(owner.n_material) * 1.0e-3,
            rng.random(owner.n_material) * 1.0e-3,
        )
        return owner, state

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

    def test_all_hookstep_controls_are_required_and_strictly_validated(self) -> None:
        fields = inspect.signature(MonolithicHookstepConfig).parameters
        self.assertEqual(
            tuple(fields),
            (
                "max_krylov_dimension",
                "arnoldi_breakdown_rtol",
                "arnoldi_orthogonality_rtol",
                "metric_rank_rtol",
                "projected_rank_rtol",
                "secular_rtol",
                "prediction_consistency_rtol",
                "metric_consistency_rtol",
                "secular_max_iterations",
                "trust_radius_initial",
                "trust_radius_min",
                "trust_radius_max",
                "max_radius_rejections",
                "eta_accept",
                "eta_expand",
                "gamma_shrink",
                "gamma_expand",
                "boundary_fraction",
            ),
        )
        self.assertTrue(
            all(parameter.default is inspect.Parameter.empty for parameter in fields.values())
        )
        _config()
        with self.assertRaisesRegex(ValueError, "eta controls"):
            _config(eta_accept=0.8, eta_expand=0.75)
        with self.assertRaisesRegex(ValueError, "trust radii"):
            _config(trust_radius_min=2.0)
        with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
            _config(metric_rank_rtol=1.0)
        for name in ("prediction_consistency_rtol", "metric_consistency_rtol"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "strictly between zero and one"):
                    _config(**{name: 1.0})

    def test_metric_matches_independent_volume_formula_and_is_immutable(self) -> None:
        owner, _ = self._owner_and_state()
        metric = build_monolithic_trust_metric(owner)
        rng = np.random.default_rng(91)
        step = rng.standard_normal(owner.n_packed)
        q, damage, history = owner.unpack(step)
        full_q = np.zeros(self.model.n_dofs, dtype=np.float64)
        full_q[self.model.constraints.free_dofs] = q
        full_q = full_q.reshape(-1, 2)
        union = self.model.static.union_volumes
        material = self.model.material_operator.volumes
        expected_squared = (
            float(np.sum(union[:, None] * full_q**2))
            / (float(np.sum(union)) * self.model.l0**2)
            + float(np.sum(material * damage**2)) / float(np.sum(material))
            + float(np.sum(material * history**2))
            / (
                float(np.sum(material))
                * (self.model.Gc / self.model.l0) ** 2
            )
        )
        self.assertAlmostEqual(metric.norm(step) ** 2, expected_squared, places=12)
        damage_constant = owner.pack(
            np.zeros(owner.n_free),
            np.ones(owner.n_material),
            np.zeros(owner.n_material),
        )
        history_constant = owner.pack(
            np.zeros(owner.n_free),
            np.zeros(owner.n_material),
            np.full(owner.n_material, self.model.Gc / self.model.l0),
        )
        self.assertAlmostEqual(metric.norm(damage_constant), 1.0, places=14)
        self.assertAlmostEqual(metric.norm(history_constant), 1.0, places=14)
        with self.assertRaises(ValueError):
            metric.diagonal.setflags(write=True)

    def test_metric_constant_damage_and_history_norms_are_refinement_invariant(self) -> None:
        for n_side in (5, 7):
            model = _regular_square_model(n_side)
            owner = MonolithicGripResidual(
                model,
                1.0e-4,
                np.zeros(model.n_material, dtype=np.float64),
            )
            metric = build_monolithic_trust_metric(owner)
            damage = owner.pack(
                np.zeros(owner.n_free),
                np.ones(owner.n_material),
                np.zeros(owner.n_material),
            )
            history = owner.pack(
                np.zeros(owner.n_free),
                np.zeros(owner.n_material),
                np.full(owner.n_material, model.Gc / model.l0),
            )
            self.assertAlmostEqual(metric.norm(damage), 1.0, places=13)
            self.assertAlmostEqual(metric.norm(history), 1.0, places=13)

    def test_displacement_metric_is_stable_across_two_cloud_refinements(self) -> None:
        norms: list[float] = []
        for n_side in (7, 9):
            model = _regular_square_model(n_side)
            owner = MonolithicGripResidual(
                model,
                1.0e-4,
                np.zeros(model.n_material, dtype=np.float64),
            )
            coordinates = model.static.coordinates
            anchor = int(model.static.anchor_particle)
            full = np.zeros(model.n_dofs, dtype=np.float64)
            full[0::2] = coordinates[:, 0] - coordinates[anchor, 0]
            y_min = float(np.min(coordinates[:, 1]))
            y_max = float(np.max(coordinates[:, 1]))
            full[1::2] = (
                (coordinates[:, 1] - y_min) * (y_max - coordinates[:, 1])
            )
            full[model.constraints.constrained_dofs] = 0.0
            self.assertTrue(
                np.array_equal(
                    full[model.constraints.constrained_dofs],
                    np.zeros(len(model.constraints.constrained_dofs)),
                )
            )
            step = owner.pack(
                full[model.constraints.free_dofs],
                np.zeros(model.n_material),
                np.zeros(model.n_material),
            )
            norms.append(build_monolithic_trust_metric(owner).norm(step))
        relative_difference = abs(norms[1] - norms[0]) / max(norms)
        self.assertLess(relative_difference, 0.08)

    def test_metric_from_an_equal_sized_different_model_is_rejected(self) -> None:
        owner, state = self._owner_and_state()
        other_model = _model_from_cloud()
        other_owner = MonolithicGripResidual(
            other_model,
            owner.u_bar,
            np.zeros(other_model.n_material, dtype=np.float64),
        )
        metric = build_monolithic_trust_metric(other_owner)
        scales = owner.block_scales(state, self._gates())
        with self.assertRaisesRegex(ValueError, "not owner-paired"):
            build_monolithic_hookstep_arnoldi(
                owner,
                state,
                scales,
                owner.active_mask(state),
                _config(),
                1.0,
                metric,
            )

    def test_same_owner_forged_metric_contents_are_rejected(self) -> None:
        owner, state = self._owner_and_state()
        metric = build_monolithic_trust_metric(owner)
        forged = replace(metric, diagonal=2.0 * metric.diagonal)
        scales = owner.block_scales(state, self._gates())
        with self.assertRaisesRegex(ValueError, "contents are not owner-paired"):
            build_monolithic_hookstep_arnoldi(
                owner,
                state,
                scales,
                owner.active_mask(state),
                _config(),
                1.0,
                forged,
            )

    @staticmethod
    def _projected_fixture(metric_basis: np.ndarray) -> MonolithicHookstepArnoldi:
        basis = np.eye(2, dtype=np.float64)
        hessenberg = np.asarray(
            [[2.0, 0.25], [0.0, 0.5], [0.0, 0.0]], dtype=np.float64
        )
        return MonolithicHookstepArnoldi(
            beta=1.25,
            basis=basis,
            preconditioned_basis=basis,
            hessenberg=hessenberg,
            metric_basis=np.asarray(metric_basis, dtype=np.float64),
            terminal_basis_vector=None,
            happy_breakdown=True,
            orthogonality_defect=0.0,
        )

    def test_projected_interior_step_matches_independent_lstsq(self) -> None:
        arnoldi = self._projected_fixture(np.diag([2.0, 0.5]))
        result = solve_projected_monolithic_hookstep(arnoldi, 10.0, _config())
        rhs = np.asarray([arnoldi.beta, 0.0, 0.0])
        expected = np.linalg.lstsq(arnoldi.hessenberg, -rhs, rcond=1.0e-12)[0]
        self.assertTrue(np.allclose(result.coefficients, expected, rtol=1.0e-12, atol=1.0e-13))
        self.assertFalse(result.on_boundary)
        self.assertEqual(result.trust_multiplier, 0.0)
        self.assertLess(result.metric_norm, result.radius)

    def test_projected_boundary_step_satisfies_kkt_and_radius(self) -> None:
        arnoldi = self._projected_fixture(np.asarray([[2.0, 0.0], [0.4, 0.75]]))
        radius = 0.1
        result = solve_projected_monolithic_hookstep(arnoldi, radius, _config())
        hessian = arnoldi.hessenberg.T @ arnoldi.hessenberg
        metric_gram = arnoldi.metric_basis.T @ arnoldi.metric_basis
        rhs = np.zeros(3, dtype=np.float64)
        rhs[0] = arnoldi.beta
        kkt = (
            hessian + result.trust_multiplier * metric_gram
        ) @ result.coefficients + arnoldi.hessenberg.T @ rhs
        self.assertTrue(result.on_boundary)
        self.assertGreater(result.trust_multiplier, 0.0)
        self.assertLessEqual(abs(result.metric_norm - radius), 2.0e-12)
        self.assertLess(np.linalg.norm(kkt), 2.0e-10)

    def test_projected_metric_rank_loss_and_stationary_nonroot_fail_loudly(self) -> None:
        rank_deficient = self._projected_fixture(
            np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        )
        with self.assertRaisesRegex(MonolithicHookstepFailure, "metric basis is rank deficient"):
            solve_projected_monolithic_hookstep(rank_deficient, 1.0, _config())
        stationary = MonolithicHookstepArnoldi(
            beta=1.0,
            basis=np.eye(2, dtype=np.float64),
            preconditioned_basis=np.eye(2, dtype=np.float64),
            hessenberg=np.zeros((3, 2), dtype=np.float64),
            metric_basis=np.eye(2, dtype=np.float64),
            terminal_basis_vector=None,
            happy_breakdown=True,
            orthogonality_defect=0.0,
        )
        with self.assertRaisesRegex(MonolithicHookstepFailure, "no positive model reduction"):
            solve_projected_monolithic_hookstep(stationary, 1.0, _config())

    def test_rank_deficient_projected_jacobian_can_have_positive_reduction(self) -> None:
        arnoldi = MonolithicHookstepArnoldi(
            beta=1.0,
            basis=np.eye(2, dtype=np.float64),
            preconditioned_basis=np.eye(2, dtype=np.float64),
            hessenberg=np.asarray(
                [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64
            ),
            metric_basis=np.eye(2, dtype=np.float64),
            terminal_basis_vector=None,
            happy_breakdown=True,
            orthogonality_defect=0.0,
        )
        result = solve_projected_monolithic_hookstep(arnoldi, 1.0, _config())
        self.assertEqual(result.projected_rank, 1)
        self.assertGreater(result.predicted_reduction, 0.0)
        self.assertTrue(np.allclose(result.coefficients, [-0.5, 0.0]))

    def test_tiny_boundary_step_keeps_positive_prediction(self) -> None:
        arnoldi = MonolithicHookstepArnoldi(
            beta=1.0,
            basis=np.ones((1, 1), dtype=np.float64),
            preconditioned_basis=np.ones((1, 1), dtype=np.float64),
            hessenberg=np.asarray([[1.0], [0.0]], dtype=np.float64),
            metric_basis=np.ones((1, 1), dtype=np.float64),
            terminal_basis_vector=None,
            happy_breakdown=True,
            orthogonality_defect=0.0,
        )
        config = _config(
            trust_radius_initial=1.0e-18,
            trust_radius_min=1.0e-20,
            secular_rtol=1.0e-8,
            secular_max_iterations=200,
        )
        result = solve_projected_monolithic_hookstep(arnoldi, 1.0e-18, config)
        self.assertGreater(result.predicted_reduction, 0.0)

    def test_real_manufactured_arnoldi_and_full_prediction_are_consistent(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        active = owner.active_mask(state)
        direction = compute_monolithic_hookstep_direction(
            owner,
            state,
            scales,
            active,
            _config(max_krylov_dimension=12),
            1.0,
        )
        self.assertLessEqual(direction.metric_norm, 1.0 * (1.0 + 2.0e-12))
        self.assertLess(direction.projected_prediction_defect, 2.0e-11)
        self.assertGreater(direction.full_predicted_reduction, 0.0)
        self.assertFalse(direction.step.flags.writeable)
        preconditioner = MonolithicBlockDiagonalPreconditioner(owner, state, scales)
        terminal = direction.arnoldi.terminal_basis_vector
        if terminal is None:
            self.fail("maximum-dimension Arnoldi result omitted its terminal vector")
        augmented_basis = np.column_stack((direction.arnoldi.basis, terminal))
        oracle = _IndependentMonolithicOracle(
            owner._model,
            owner.u_bar,
            owner.committed_history,
        )
        for column in range(direction.arnoldi.dimension):
            action = owner.scaled_jvp(
                state,
                preconditioner.apply(direction.arnoldi.basis[:, column]),
                scales,
                active,
            )
            projected = augmented_basis @ direction.arnoldi.hessenberg[:, column]
            self.assertTrue(np.allclose(action, projected, rtol=2.0e-12, atol=2.0e-12))
            raw_oracle = oracle.jvp(
                state, direction.arnoldi.preconditioned_basis[:, column]
            )
            oracle_q, oracle_d, oracle_h = owner.unpack(raw_oracle)
            scaled_oracle = np.concatenate(
                (
                    oracle_q / scales.Lq,
                    oracle_d / owner._damage_volumes / scales.Lrho,
                    oracle_h / scales.LH,
                )
            )
            self.assertTrue(
                np.allclose(scaled_oracle, projected, rtol=2.0e-11, atol=2.0e-11)
            )

    def test_exact_root_is_an_explicit_caller_shortcut(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        metric = build_monolithic_trust_metric(owner)
        with mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            return_value=np.zeros(owner.n_packed, dtype=np.float64),
        ):
            with self.assertRaisesRegex(ValueError, "exact root"):
                build_monolithic_hookstep_arnoldi(
                    owner,
                    state,
                    scales,
                    owner.active_mask(state),
                    _config(),
                    1.0,
                    metric,
                )

    def test_full_prediction_mismatch_is_rejected(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        active = owner.active_mask(state)
        config = _config(max_krylov_dimension=5, prediction_consistency_rtol=1.0e-12)
        metric = build_monolithic_trust_metric(owner)
        arnoldi = build_monolithic_hookstep_arnoldi(
            owner, state, scales, active, config, 1.0, metric
        )
        original = type(owner).scaled_jvp

        def inconsistent_action(
            instance: MonolithicGripResidual,
            packed: np.ndarray,
            values: np.ndarray,
            frozen_scales: Any,
            frozen_active: np.ndarray,
        ) -> np.ndarray:
            return 0.9 * original(
                instance,
                packed,
                values,
                frozen_scales,
                frozen_active,
            )

        with mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "build_monolithic_hookstep_arnoldi",
            return_value=arnoldi,
        ), mock.patch.object(
            type(owner),
            "scaled_jvp",
            autospec=True,
            side_effect=inconsistent_action,
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepFailure, "prediction exceeds its tolerance"
            ):
                compute_monolithic_hookstep_direction(
                    owner, state, scales, active, config, 1.0
                )

    def test_identity_action_records_happy_breakdown(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        metric = build_monolithic_trust_metric(owner)
        active = owner.active_mask(state)
        with mock.patch.object(
            type(owner),
            "scaled_jvp",
            autospec=True,
            side_effect=lambda _self, _z, values, _scales, _active: np.asarray(
                values, dtype=np.float64
            ),
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "MonolithicBlockDiagonalPreconditioner.apply",
            autospec=True,
            side_effect=lambda _self, values: np.asarray(values, dtype=np.float64),
        ):
            arnoldi = build_monolithic_hookstep_arnoldi(
                owner, state, scales, active, _config(), 1.0, metric
            )
        self.assertTrue(arnoldi.happy_breakdown)
        self.assertEqual(arnoldi.dimension, 1)
        self.assertLessEqual(
            arnoldi.hessenberg[1, 0],
            _config().arnoldi_breakdown_rtol * arnoldi.hessenberg[0, 0],
        )
        if arnoldi.hessenberg[1, 0] == 0.0:
            self.assertIsNone(arnoldi.terminal_basis_vector)
        else:
            self.assertIsNotNone(arnoldi.terminal_basis_vector)

    def test_nonzero_subnormal_residual_is_not_misclassified_as_exact_root(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        metric = build_monolithic_trust_metric(owner)
        active = owner.active_mask(state)
        residual = np.zeros(owner.n_packed, dtype=np.float64)
        residual[0] = np.nextafter(0.0, 1.0)
        with mock.patch.object(
            type(owner),
            "scaled_residual",
            autospec=True,
            return_value=residual,
        ), mock.patch.object(
            type(owner),
            "scaled_jvp",
            autospec=True,
            side_effect=lambda _self, _z, values, _scales, _active: np.asarray(
                values, dtype=np.float64
            ),
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "MonolithicBlockDiagonalPreconditioner.apply",
            autospec=True,
            side_effect=lambda _self, values: np.asarray(values, dtype=np.float64),
        ):
            arnoldi = build_monolithic_hookstep_arnoldi(
                owner, state, scales, active, _config(), 1.0, metric
            )
        self.assertEqual(arnoldi.beta, residual[0])
        self.assertGreater(arnoldi.beta, 0.0)

    def test_tolerance_breakdown_retains_nonzero_terminal_vector(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        metric = build_monolithic_trust_metric(owner)
        active = owner.active_mask(state)

        def nearly_identity_action(
            _self: MonolithicGripResidual,
            _z: np.ndarray,
            values: np.ndarray,
            _scales: Any,
            _active: np.ndarray,
        ) -> np.ndarray:
            vector = np.asarray(values, dtype=np.float64)
            index = int(np.argmin(np.abs(vector)))
            transverse = np.zeros_like(vector)
            transverse[index] = 1.0
            transverse -= np.dot(vector, transverse) * vector
            transverse /= np.linalg.norm(transverse)
            return vector + 1.0e-10 * transverse

        with mock.patch.object(
            type(owner),
            "scaled_jvp",
            autospec=True,
            side_effect=nearly_identity_action,
        ), mock.patch(
            "examples.fracture2d.nonlocal_engineering_grip_monolithic."
            "MonolithicBlockDiagonalPreconditioner.apply",
            autospec=True,
            side_effect=lambda _self, values: np.asarray(values, dtype=np.float64),
        ):
            arnoldi = build_monolithic_hookstep_arnoldi(
                owner,
                state,
                scales,
                active,
                _config(arnoldi_breakdown_rtol=1.0e-6),
                1.0,
                metric,
            )
        terminal = arnoldi.terminal_basis_vector
        if terminal is None:
            self.fail("nonzero tolerance breakdown omitted its terminal vector")
        reconstructed = np.column_stack((arnoldi.basis, terminal)) @ arnoldi.hessenberg
        expected = nearly_identity_action(owner, state, arnoldi.basis[:, 0], scales, active)
        self.assertTrue(arnoldi.happy_breakdown)
        self.assertGreater(arnoldi.hessenberg[1, 0], 0.0)
        self.assertTrue(np.allclose(reconstructed[:, 0], expected, atol=2.0e-16))

    def test_nonfinite_arnoldi_action_is_a_numerical_failure(self) -> None:
        owner, state = self._owner_and_state()
        scales = owner.block_scales(state, self._gates())
        metric = build_monolithic_trust_metric(owner)
        active = owner.active_mask(state)
        with mock.patch.object(
            type(owner),
            "scaled_jvp",
            autospec=True,
            return_value=np.full(owner.n_packed, np.inf, dtype=np.float64),
        ):
            with self.assertRaisesRegex(
                MonolithicHookstepFailure, "action failed.*non-finite"
            ) as caught:
                build_monolithic_hookstep_arnoldi(
                    owner, state, scales, active, _config(), 1.0, metric
                )
        self.assertEqual(caught.exception.audit.stage, "arnoldi")

    def test_radius_decision_consumes_acceptance_and_rejection_controls(self) -> None:
        config = _config()
        rejected = decide_monolithic_hookstep_radius(0.05, 0.8, 1.0, 0, config)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.next_radius, 0.25)
        self.assertEqual(rejected.rejection_count, 1)
        retained = decide_monolithic_hookstep_radius(0.5, 0.95, 1.0, 2, config)
        self.assertTrue(retained.accepted)
        self.assertFalse(retained.expanded)
        self.assertEqual(retained.next_radius, 1.0)
        expanded = decide_monolithic_hookstep_radius(0.9, 0.95, 1.0, 2, config)
        self.assertTrue(expanded.expanded)
        self.assertEqual(expanded.next_radius, 2.0)
        threshold_accept = decide_monolithic_hookstep_radius(
            config.eta_accept, 0.1, 1.0, 0, config
        )
        self.assertTrue(threshold_accept.accepted)
        self.assertFalse(threshold_accept.expanded)
        threshold_expand = decide_monolithic_hookstep_radius(
            config.eta_expand,
            config.boundary_fraction,
            1.0,
            0,
            config,
        )
        self.assertTrue(threshold_expand.expanded)
        floor_equal = decide_monolithic_hookstep_radius(
            0.0,
            0.1,
            1.0,
            0,
            _config(trust_radius_min=0.25),
        )
        self.assertEqual(floor_equal.next_radius, 0.25)
        with self.assertRaisesRegex(MonolithicHookstepFailure, "rejection budget"):
            decide_monolithic_hookstep_radius(
                0.0,
                0.1,
                1.0,
                config.max_radius_rejections,
                config,
            )
        for ratio, step_norm in ((np.nan, 0.8), (0.5, np.inf)):
            with self.subTest(ratio=ratio, step_norm=step_norm):
                with self.assertRaisesRegex(MonolithicHookstepFailure, "non-finite"):
                    decide_monolithic_hookstep_radius(
                        ratio, step_norm, 1.0, 0, config
                    )
        with self.assertRaisesRegex(MonolithicHookstepFailure, "rejection budget"):
            decide_monolithic_hookstep_radius(
                0.0, 0.1, 1.0, 0, _config(max_radius_rejections=0)
            )


if __name__ == "__main__":
    unittest.main()
