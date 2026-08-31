"""Core tests for the lean v5 stabilized engineering-grip archive."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from typing import Any, Callable

import jax
import numpy as np

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import (
    nonlocal_engineering_grip_stabilized as stabilized,
)
from examples.fracture2d.nonlocal_grip_diagnostic import (
    build_union_geometry_from_arrays,
)


def _source_cloud() -> dict[str, np.ndarray]:
    """Construct the smallest regular square fixture with exact d=0.048 support."""

    n_side = 31
    axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    coordinates = np.asarray(
        [(x, y) for x in axis for y in axis], dtype=np.float64
    )
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n_side - 1):
        for iy in range(n_side - 1):
            lower_left = ix * n_side + iy
            lower_right = (ix + 1) * n_side + iy
            upper_left = ix * n_side + iy + 1
            upper_right = (ix + 1) * n_side + iy + 1
            triangles.extend(
                ((lower_left, lower_right, upper_right),
                 (lower_left, upper_right, upper_left))
            )
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
    delta = 0.048
    pairs: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
            if 0.0 < distance < delta:
                pairs.extend(((first, second), (second, first)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    return {
        "coordinates": coordinates,
        "triangles": triangles_array,
        "volumes": volumes,
        "bottom": bottom,
        "top": top,
        "anchor_particle": np.asarray(0, dtype=np.int64),
        "d048": edges,
    }



def _source_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        fields = {
            name: {
                "dtype": str(archive[name].dtype),
                "shape": list(archive[name].shape),
            }
            for name in stabilized._V5_SOURCE_FIELDS
        }
    return {
        "path": str(path.resolve()),
        "sha256": stabilized._archive_sha256(path),
        "size_bytes": path.stat().st_size,
        "required_fields": list(stabilized._V5_SOURCE_FIELDS),
        "fields": fields,
    }


def _assert_scalar_tree(test: unittest.TestCase, value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_scalar_tree(test, item)
    elif isinstance(value, list):
        test.assertTrue(
            all(isinstance(item, str) for item in value),
            "v5 scalar logs must not contain numerical vectors",
        )
        for item in value:
            _assert_scalar_tree(test, item)


def _warning_diagnostic(
    *,
    failed_gates: tuple[str, ...] = ("damage", "damage_residual", "coupled"),
    hard_failed_gates: tuple[str, ...] = (),
    complete_sweep: bool = True,
) -> Any:
    gates = {
        name: True
        for name in (
            *stabilized.STABILIZED_WARNING_HARD_GATE_NAMES,
            *stabilized.STABILIZED_WARNING_GATE_NAMES,
        )
    }
    for name in (*failed_gates, *hard_failed_gates):
        gates[name] = False
    previous = np.asarray([1.0, 0.201, 0.0], dtype=np.float64)
    trial = np.asarray([1.0005, 0.2, -0.0002], dtype=np.float64)
    return SimpleNamespace(
        complete_sweep=complete_sweep,
        passed=False,
        gate_passes=gates,
        d_trial=trial,
        base_diagnostics=SimpleNamespace(damage_before=previous),
    )


class StabilizedProgressContractTest(unittest.TestCase):
    def test_progress_settings_require_a_positive_explicit_cadence(self) -> None:
        callback = lambda event: None
        self.assertEqual(
            stabilized._validate_progress_settings(callback, 3),
            3,
        )
        with self.assertRaisesRegex(ValueError, "requires progress_callback"):
            stabilized._validate_progress_settings(None, 3)
        with self.assertRaisesRegex(ValueError, "required when"):
            stabilized._validate_progress_settings(callback, None)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            stabilized._validate_progress_settings(callback, 0)

    def test_progress_cadence_keeps_first_periodic_final_and_passing_sweeps(self) -> None:
        self.assertEqual(
            [
                stabilized._progress_sweep_is_due(index, 3, 7, False)
                for index in range(7)
            ],
            [True, False, True, False, False, True, True],
        )
        self.assertTrue(stabilized._progress_sweep_is_due(1, 3, 7, True))

    def test_progress_events_are_strict_json_scalar_payloads(self) -> None:
        events: list[dict[str, Any]] = []
        with mock.patch("builtins.print") as printer:
            stabilized._emit_progress_event(
                events.append,
                {
                    "event": "sweep",
                    "integer": np.int64(4),
                    "finite": np.float64(1.25),
                    "unavailable": np.float64(np.nan),
                    "gates": ["damage"],
                },
                0.0,
            )
        printer.assert_not_called()
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["unavailable"])
        json.dumps(events[0], sort_keys=True, allow_nan=False)
        with self.assertRaises(TypeError):
            stabilized._emit_progress_event(
                events.append,
                {"event": "invalid", "array": np.zeros(2)},
                0.0,
            )


class StabilizedWarningPolicyTest(unittest.TestCase):
    def test_final_complete_sweep_classifies_only_nonhard_failures(self) -> None:
        diagnostic = _warning_diagnostic()

        accepted, bypassed, range_excess, healing = (
            stabilized._stabilized_warning_acceptance_decision(diagnostic)
        )

        self.assertTrue(accepted)
        self.assertEqual(
            bypassed,
            ("damage", "damage_residual", "coupled"),
        )
        self.assertEqual(
            bypassed,
            stabilized._stabilized_warning_failed_gates(diagnostic),
        )
        self.assertAlmostEqual(range_excess, 5.0e-4)
        self.assertAlmostEqual(healing, 1.0e-3)

        for name, invalid in (
            ("all-gates-pass", _warning_diagnostic(failed_gates=())),
            (
                "hard-gate-failure",
                _warning_diagnostic(hard_failed_gates=("mechanics_after",)),
            ),
            (
                "incomplete-sweep",
                _warning_diagnostic(complete_sweep=False),
            ),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    stabilized._stabilized_warning_acceptance_decision(invalid),
                    (False, (), 0.0, 0.0),
                )


class StabilizedEngineeringGripArchiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        cls._directory = tempfile.TemporaryDirectory(prefix="stabilized-v5-source-")
        root = Path(cls._directory.name)
        cls.source_path = root / "source.npz"
        source = _source_cloud()
        np.savez_compressed(str(cls.source_path), **source)  # type: ignore[arg-type]
        cls.source_metadata = _source_metadata(cls.source_path)
        geometry = build_union_geometry_from_arrays(
            source["coordinates"],
            source["triangles"],
            source["volumes"],
            source["bottom"],
            source["top"],
            int(source["anchor_particle"]),
            source["d048"],
            0.048,
            0.024,
            0.0,
        )
        cls.model = engineering.build_engineering_grip_model(
            geometry,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
            2.7,
            0.024,
        )
        cls.cache_model = cls.model
        cls.config = stabilized.StabilizedEngineeringGripConfig.smoke()
        cls.result = stabilized.run_stabilized_engineering_grip(
            cls.model, cls.config
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()
        jax.config.update("jax_enable_x64", cls._old_x64)

    @staticmethod
    def _copy_payload(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: np.array(value, copy=True) for name, value in values.items()}

    def _write_archive(
        self,
        directory: Path,
        *,
        result: stabilized.StabilizedEngineeringGripResult | None = None,
    ) -> tuple[Path, dict[str, np.ndarray]]:
        path = directory / "stabilized.npz"
        stabilized.write_stabilized_engineering_grip_archive(
            path,
            self.result if result is None else result,
            source_metadata=self.source_metadata,
        )
        return path, stabilized.load_stabilized_engineering_grip_archive(path)

    @staticmethod
    def _write_payload(
        directory: Path, name: str, payload: dict[str, np.ndarray]
    ) -> Path:
        path = directory / name
        np.savez_compressed(str(path), **payload)  # type: ignore[arg-type]
        return path

    @staticmethod
    def _update_metadata(
        payload: dict[str, np.ndarray], update: Callable[[dict[str, Any]], None]
    ) -> None:
        metadata = json.loads(str(payload["metadata_json"]))
        update(metadata)
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False), dtype=np.str_
        )

    @staticmethod
    def _update_summary(
        payload: dict[str, np.ndarray],
        index: int,
        update: Callable[[dict[str, Any]], None],
    ) -> None:
        records = [json.loads(str(item)) for item in payload["solver_summary_json"]]
        update(records[index])
        payload["solver_summary_json"] = np.asarray(
            [json.dumps(item, sort_keys=True, allow_nan=False) for item in records],
            dtype=np.str_,
        )

    def _cache_config(
        self, policy: str
    ) -> stabilized.StabilizedEngineeringGripConfig:
        base = replace(
            self.config.base,
            max_stagger_iterations=2,
        )
        if policy == "fixed":
            return replace(
                self.config,
                base=base,
                relaxation_policy="fixed",
                omega_initial=0.5,
                omega_min=0.5,
                omega_reduction=None,
                sufficient_decrease=None,
                max_backtracks=0,
            )
        if policy == "sufficient_decrease":
            return replace(self.config, base=base)
        raise AssertionError(f"unknown test relaxation policy: {policy}")

    def _warning_result(self) -> stabilized.StabilizedEngineeringGripResult:
        """Reclassify the manufactured accepted state as a warning row."""

        strict_step = self.result.accepted_steps[0]
        strict_diagnostic = strict_step.diagnostics
        warning_base = replace(
            strict_diagnostic.base_diagnostics,
            passed=False,
            failure_stage="gates",
            failure_reason="damage, damage_residual, coupled",
            damage_pass=False,
            damage_residual_pass=False,
        )
        fixed_trial = replace(
            strict_diagnostic.trials[-1],
            sufficient_decrease_limit=None,
        )
        warning_diagnostic = replace(
            strict_diagnostic,
            base_diagnostics=warning_base,
            passed=False,
            failure_stage="gates",
            failure_reason="damage, damage_residual, coupled",
            sufficient_decrease_limit=None,
            coupled_pass=False,
            trials=(fixed_trial,),
        )
        bypassed = stabilized._stabilized_warning_failed_gates(
            warning_diagnostic
        )
        range_excess, healing = stabilized._stabilized_warning_excesses(
            warning_diagnostic
        )
        config = replace(
            self.config,
            relaxation_policy="fixed",
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=None,
            max_backtracks=0,
            acceptance_mode=stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
        )
        warning_step = replace(
            strict_step,
            diagnostics=warning_diagnostic,
            acceptance_status=stabilized.STABILIZED_WARNING_ACCEPTANCE_STATUS,
            bypassed_gates=bypassed,
            warning_range_excess=range_excess,
            warning_healing_magnitude=healing,
            acceptance_mode=stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
        )
        strict_attempt = self.result.attempts[0]
        warning_attempt = replace(
            strict_attempt,
            acceptance_status=stabilized.STABILIZED_WARNING_ACCEPTANCE_STATUS,
            sweeps=(*strict_attempt.sweeps[:-1], warning_diagnostic),
            diagnostics=warning_diagnostic,
            bypassed_gates=bypassed,
            warning_range_excess=range_excess,
            warning_healing_magnitude=healing,
            acceptance_mode=stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
        )
        return stabilized.StabilizedEngineeringGripResult(
            model=self.model,
            config=config,
            work_check=self.result.work_check,
            initial_state=self.result.initial_state,
            states=self.result.states,
            accepted_steps=(warning_step,),
            attempts=(warning_attempt,),
            completion_status=stabilized.STABILIZED_WARNING_COMPLETION_STATUS,
            _acceptance_mode=stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
        )

    def _run_cache_case(
        self,
        policy: str,
        *,
        reuse: bool,
    ) -> tuple[stabilized.StabilizedEngineeringGripResult, int]:
        """Run a two-sweep rejection/cutback path with a counting solve seam."""

        config = self._cache_config(policy)
        # Attempt 1 rejects after two selected-but-not-coupled sweeps.  The
        # cutback and final attempts then accept after two sweeps each.  This
        # also proves that an attempt-local cache cannot cross a cutback.
        fixed_point_outcomes = [False, False, False, True, False, True]
        real_solver = engineering._solve_direct_mechanics
        with mock.patch.object(
            engineering,
            "_solve_direct_mechanics",
            wraps=real_solver,
        ) as solve_mock, mock.patch.object(
            stabilized,
            "_fixed_point_pass",
            side_effect=fixed_point_outcomes,
        ), mock.patch.object(
            stabilized,
            "_candidate_merit",
            return_value=0.0,
        ):
            result = stabilized.run_stabilized_engineering_grip(
                self.cache_model,
                config,
                _reuse_selected_mechanics=reuse,
            )
        return result, solve_mock.call_count

    def _write_cache_archive(
        self,
        directory: Path,
        name: str,
        result: stabilized.StabilizedEngineeringGripResult,
    ) -> tuple[Path, dict[str, np.ndarray]]:
        path = directory / name
        stabilized.write_stabilized_engineering_grip_archive(
            path,
            result,
            source_metadata=self.source_metadata,
        )
        return path, stabilized.load_stabilized_engineering_grip_archive(path)

    def _assert_cache_results_equal(
        self,
        cached: stabilized.StabilizedEngineeringGripResult,
        uncached: stabilized.StabilizedEngineeringGripResult,
        policy: str,
    ) -> None:
        self.assertEqual(len(cached.states), len(uncached.states))
        self.assertEqual(len(cached.accepted_steps), len(uncached.accepted_steps))
        self.assertEqual(len(cached.attempts), len(uncached.attempts))
        for cached_state, uncached_state in zip(cached.states, uncached.states):
            self.assertEqual(cached_state.u_bar, uncached_state.u_bar)
            np.testing.assert_array_equal(cached_state.q, uncached_state.q)
            np.testing.assert_array_equal(cached_state.d, uncached_state.d)
            np.testing.assert_array_equal(cached_state.H, uncached_state.H)
            self.assertEqual(cached_state.digest, uncached_state.digest)
        for index, (cached_step, uncached_step) in enumerate(zip(
            cached.accepted_steps, uncached.accepted_steps
        ), start=1):
            self.assertEqual(cached_step.load_start, uncached_step.load_start)
            self.assertEqual(cached_step.load_target, uncached_step.load_target)
            self.assertEqual(cached_step.increment, uncached_step.increment)
            self.assertEqual(cached_step.stagger_iterations, uncached_step.stagger_iterations)
            self.assertEqual(cached_step.cutback_count, uncached_step.cutback_count)
            self.assertEqual(
                cached_step.diagnostics.gate_passes,
                uncached_step.diagnostics.gate_passes,
            )
            self.assertEqual(
                stabilized._v5_solver_summary_payload(
                    index,
                    cached_step,
                    relaxation_policy=policy,
                ),
                stabilized._v5_solver_summary_payload(
                    index,
                    uncached_step,
                    relaxation_policy=policy,
                ),
            )
        self.assertEqual(cached.work_check.u_bar, uncached.work_check.u_bar)
        self.assertEqual(cached.work_check.epsilon, uncached.work_check.epsilon)
        self.assertEqual(cached.work_check.reaction, uncached.work_check.reaction)
        self.assertEqual(cached.work_check.reaction_fd, uncached.work_check.reaction_fd)
        self.assertEqual(cached.work_check.relative_error, uncached.work_check.relative_error)
        self.assertEqual(cached.work_check.passed, uncached.work_check.passed)

    def test_selected_mechanics_cache_matches_uncached_both_policies_and_cutback(self) -> None:
        for policy in stabilized.STABILIZED_RELAXATION_POLICIES:
            with self.subTest(policy=policy):
                uncached, uncached_calls = self._run_cache_case(policy, reuse=False)
                cached, cached_calls = self._run_cache_case(policy, reuse=True)

                self.assertEqual(uncached_calls, 3 + 3 * 4)
                self.assertEqual(cached_calls, 3 + 3 * 3)
                self._assert_cache_results_equal(cached, uncached, policy)
                self.assertEqual(
                    [attempt.accepted for attempt in cached.attempts],
                    [False, True, True],
                )
                self.assertEqual(
                    [attempt.cutback_count for attempt in cached.attempts],
                    [0, 1, 0],
                )
                self.assertEqual(
                    [state.u_bar for state in cached.states],
                    [0.0, 5.0e-5, 1.0e-4],
                )

                with tempfile.TemporaryDirectory(
                    prefix=f"stabilized-v5-cache-{policy}-"
                ) as name:
                    directory = Path(name)
                    uncached_path, uncached_payload = self._write_cache_archive(
                        directory, "uncached.npz", uncached
                    )
                    cached_path, cached_payload = self._write_cache_archive(
                        directory, "cached.npz", cached
                    )
                    uncached_size = uncached_path.stat().st_size
                    cached_size = cached_path.stat().st_size
                self.assertEqual(
                    set(uncached_payload), set(stabilized._V5_ARCHIVE_BASE_FIELDS)
                )
                self.assertEqual(set(cached_payload), set(uncached_payload))
                self.assertEqual(len(cached_payload), len(uncached_payload))
                for field in uncached_payload:
                    np.testing.assert_array_equal(
                        cached_payload[field], uncached_payload[field]
                    )
                self.assertEqual(
                    cached_size, uncached_size
                )

    def test_progress_callback_reports_transient_ordered_events(self) -> None:
        events: list[dict[str, Any]] = []
        result = stabilized.run_stabilized_engineering_grip(
            self.model,
            self.config,
            progress_callback=events.append,
            progress_every_sweeps=2,
        )
        self.assertEqual(events[0]["event"], "attempt_start")
        self.assertEqual(events[-2]["event"], "accepted")
        self.assertEqual(events[-1]["event"], "completed")
        sweep_events = [event for event in events if event["event"] == "sweep"]
        self.assertEqual([event["sweep"] for event in sweep_events], [1, 2])
        self.assertEqual(events[0]["step_index"], 1)
        self.assertFalse(sweep_events[0]["cached_q_before"])
        self.assertTrue(sweep_events[1]["cached_q_before"])
        self.assertEqual(events[-2]["load"], result.final_state.u_bar)
        self.assertEqual(events[-1]["load"], result.final_state.u_bar)
        elapsed = [float(event["elapsed"]) for event in events]
        self.assertTrue(all(later >= earlier for earlier, later in zip(elapsed, elapsed[1:])))
        for event in events:
            json.dumps(event, sort_keys=True, allow_nan=False)
            self.assertFalse(any(isinstance(value, np.ndarray) for value in event.values()))

    def test_selected_mechanics_cache_owns_immutable_cpu_result(self) -> None:
        view = self.cache_model.fixed_damage_view(
            np.zeros(self.cache_model.n_material, dtype=np.float64)
        )
        source_damage = np.array(view.material_damage, copy=True)
        result = engineering._solve_direct_mechanics(
            None,
            view,
            self.cache_model.constraints,
            1.0e-4,
            self.config.base.cg_rtol,
            self.config.base.cg_atol,
            self.config.base.cg_maxiter,
            initial_guess=None,
            seed_kind="cache_test",
            source_state="cache_test",
            source_step=0,
            source_sweep=0,
            source_damage_digest=view.damage_digest,
        )
        cache = stabilized._StabilizedMechanicsReuse.from_selected(
            self.cache_model,
            1.0e-4,
            source_damage,
            result,
        )
        source_damage[0] = 1.0
        np.testing.assert_array_equal(cache.damage, view.material_damage)
        self.assertIsInstance(cache.result, type(result))
        self.assertFalse(cache.result.displacement.flags.writeable)
        self.assertFalse(cache.result.evaluation.residual.flags.writeable)
        with self.assertRaises(ValueError):
            cache.damage[0] = 1.0
        with self.assertRaises(ValueError):
            cache.result.displacement[0] = 1.0

    def test_selected_mechanics_cache_requires_exact_target_and_damage(self) -> None:
        damage = np.zeros(self.cache_model.n_material, dtype=np.float64)
        view = self.cache_model.fixed_damage_view(damage)
        result = engineering._solve_direct_mechanics(
            None,
            view,
            self.cache_model.constraints,
            1.0e-4,
            self.config.base.cg_rtol,
            self.config.base.cg_atol,
            self.config.base.cg_maxiter,
            initial_guess=None,
            seed_kind="cache_identity_test",
            source_state="cache_identity_test",
            source_step=0,
            source_sweep=0,
            source_damage_digest=view.damage_digest,
        )
        cache = stabilized._StabilizedMechanicsReuse.from_selected(
            self.cache_model,
            1.0e-4,
            damage,
            result,
        )
        self.assertIs(cache.take(self.cache_model, view, 1.0e-4), result)
        self.assertIsNone(cache.take(self.cache_model, view, np.nextafter(1.0e-4, 1.0)))
        changed_damage = np.array(damage, copy=True)
        changed_damage[0] = 1.0e-8
        changed_view = self.cache_model.fixed_damage_view(changed_damage)
        self.assertIsNone(cache.take(self.cache_model, changed_view, 1.0e-4))

    def test_completed_archive_is_lean_v5_and_replays_without_controller(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-complete-") as name:
            directory = Path(name)
            path, payload = self._write_archive(directory)
            self.assertEqual(set(payload), set(stabilized._V5_ARCHIVE_BASE_FIELDS))
            self.assertTrue(all(value.dtype.kind != "O" for value in payload.values()))
            self.assertEqual(payload["state_union_displacement"].shape[0], 2)
            self.assertEqual(payload["state_material_damage"].shape[0], 2)
            self.assertNotIn("static_arrays", payload)
            for forbidden in (
                "trial_q_after",
                "trial_d_trial",
                "trial_candidate_diagnostics_json",
                "sweep_trial_offsets",
            ):
                self.assertNotIn(forbidden, payload)
            metadata = json.loads(str(payload["metadata_json"]))
            self.assertEqual(metadata["source"]["path"], self.source_metadata["path"])
            self.assertEqual(metadata["source_sha256"], self.source_metadata["sha256"])
            self.assertIn("config", metadata)
            self.assertIn("stabilized_config", metadata)
            self.assertEqual(len(payload["solver_summary_json"]), 2)
            for raw in payload["solver_summary_json"]:
                _assert_scalar_tree(self, json.loads(str(raw)))
            self.assertEqual(payload["state_external_work"].shape, (2,))
            self.assertEqual(json.loads(str(payload["solver_summary_json"][0]))["summary_kind"], "initial")
            with mock.patch.object(
                stabilized,
                "run_stabilized_engineering_grip",
                side_effect=AssertionError("replay called the controller"),
            ), mock.patch.object(
                engineering,
                "_solve_direct_mechanics",
                side_effect=AssertionError("replay called the production solver"),
            ):
                replay = stabilized.replay_stabilized_engineering_grip_archive(path)
        self.assertTrue(replay["replayed"])
        self.assertFalse(replay["validated"])
        self.assertTrue(replay["field_equations_validated"])
        self.assertFalse(replay["path_replayed"])
        self.assertEqual(replay["archive_schema"], stabilized.STABILIZED_ARCHIVE_SCHEMA)
        self.assertEqual(replay["accepted_steps"], 1)

    def test_warning_archive_is_distinct_and_tamper_checked(self) -> None:
        warning_result = self._warning_result()
        with tempfile.TemporaryDirectory(prefix="stabilized-warning-v1-") as name:
            directory = Path(name)
            _, strict_payload = self._write_archive(directory)
            warning_path = directory / "warning.npz"
            stabilized.write_stabilized_engineering_grip_archive(
                warning_path,
                warning_result,
                source_metadata=self.source_metadata,
            )
            warning_payload = stabilized.load_stabilized_engineering_grip_archive(
                warning_path
            )

            self.assertEqual(
                strict_payload["archive_schema"],
                np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA),
            )
            self.assertEqual(
                warning_payload["archive_schema"],
                np.asarray(stabilized.STABILIZED_WARNING_ARCHIVE_SCHEMA),
            )
            self.assertNotEqual(
                str(strict_payload["archive_schema"]),
                str(warning_payload["archive_schema"]),
            )
            self.assertNotIn("warning_gate_names", strict_payload)
            self.assertEqual(
                set(warning_payload), set(stabilized._WARNING_ARCHIVE_BASE_FIELDS)
            )
            metadata = json.loads(str(warning_payload["metadata_json"]))
            self.assertEqual(
                metadata["schema"], stabilized.STABILIZED_WARNING_ARCHIVE_SCHEMA
            )
            self.assertEqual(
                metadata["acceptance_mode"],
                stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
            )
            self.assertEqual(
                metadata["warning_hard_gate_names"],
                list(stabilized.STABILIZED_WARNING_HARD_GATE_NAMES),
            )
            self.assertEqual(
                metadata["warning_bypassable_gate_names"],
                list(stabilized.STABILIZED_WARNING_GATE_NAMES),
            )
            self.assertEqual(
                tuple(str(name) for name in warning_payload["warning_gate_names"]),
                stabilized.STABILIZED_WARNING_GATE_NAMES,
            )
            self.assertEqual(
                warning_payload["state_bypassed_gate_mask"].shape,
                (1, len(stabilized.STABILIZED_WARNING_GATE_NAMES)),
            )
            expected_mask = np.asarray(
                [
                    name in set(warning_result.accepted_steps[0].bypassed_gates)
                    for name in stabilized.STABILIZED_WARNING_GATE_NAMES
                ],
                dtype=bool,
            )
            np.testing.assert_array_equal(
                warning_payload["state_bypassed_gate_mask"][0], expected_mask
            )
            self.assertTrue(bool(warning_payload["accepted_warning_present"][0]))
            self.assertEqual(
                warning_payload["accepted_warning_range_excess"][0],
                warning_result.accepted_steps[0].warning_range_excess,
            )
            self.assertEqual(
                warning_payload["accepted_warning_healing_magnitude"][0],
                warning_result.accepted_steps[0].warning_healing_magnitude,
            )
            warning_summary = json.loads(
                str(warning_payload["solver_summary_json"][1])
            )
            self.assertEqual(
                warning_summary["acceptance_status"],
                stabilized.STABILIZED_WARNING_ACCEPTANCE_STATUS,
            )
            self.assertFalse(warning_summary["passed"])
            self.assertEqual(warning_summary["failure_stage"], "")
            self.assertEqual(warning_summary["failure_reason"], "")
            self.assertFalse(warning_summary["gate_passes"]["damage"])
            self.assertFalse(warning_summary["gate_passes"]["damage_residual"])
            self.assertFalse(warning_summary["gate_passes"]["coupled"])

            replay = stabilized.replay_stabilized_engineering_grip_archive(
                warning_path
            )
            self.assertTrue(replay["replayed"])
            self.assertFalse(replay["validated"])
            self.assertFalse(replay["field_equations_validated"])
            self.assertFalse(replay["path_replayed"])
            self.assertEqual(
                replay["acceptance_mode"],
                stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
            )
            self.assertEqual(
                replay["warning_acceptance"]["warning_accepted_steps"], 1
            )

            mutations = {
                "mask": lambda payload: payload["state_bypassed_gate_mask"].__setitem__(
                    (0, 0), True
                ),
                "present": lambda payload: payload["accepted_warning_present"].__setitem__(
                    0, False
                ),
                "range": lambda payload: payload[
                    "accepted_warning_range_excess"
                ].__setitem__(0, 1.0e-6),
                "metadata": lambda payload: self._update_metadata(
                    payload,
                    lambda metadata: metadata[
                        "warning_bypassable_gate_names"
                    ].reverse(),
                ),
            }
            for field, mutate in mutations.items():
                with self.subTest(field=field):
                    forged = self._copy_payload(warning_payload)
                    mutate(forged)
                    forged_path = self._write_payload(
                        directory, f"forged-warning-{field}.npz", forged
                    )
                    with self.assertRaises(ValueError):
                        stabilized.replay_stabilized_engineering_grip_archive(
                            forged_path
                        )

    def test_fixed_policy_uses_one_candidate_without_merit_decrease(self) -> None:
        config = replace(
            self.config,
            relaxation_policy="fixed",
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=None,
            max_backtracks=0,
        )
        with mock.patch.object(stabilized, "_candidate_merit", return_value=1.0e30):
            result = stabilized.run_stabilized_engineering_grip(self.model, config)

        diagnostic = result.accepted_steps[0].diagnostics
        self.assertEqual(config.relaxation_policy, "fixed")
        self.assertEqual(len(diagnostic.trials), 1)
        self.assertEqual(diagnostic.backtrack_count, 0)
        self.assertEqual(diagnostic.omega, 0.5)
        self.assertIsNone(diagnostic.sufficient_decrease_limit)
        self.assertTrue(diagnostic.globalization_pass)
        expected_trial = diagnostic.d_iter + 0.5 * (
            diagnostic.d_solved - diagnostic.d_iter  # type: ignore[operator]
        )
        assert diagnostic.d_trial is not None
        np.testing.assert_array_equal(diagnostic.d_trial, expected_trial)
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-fixed-") as name:
            path, payload = self._write_archive(Path(name), result=result)
            metadata = json.loads(str(payload["metadata_json"]))
            self.assertEqual(
                metadata["stabilized_config"]["relaxation_policy"], "fixed"
            )
            summary = json.loads(str(payload["solver_summary_json"][1]))
            self.assertEqual(summary["relaxation_policy"], "fixed")
            self.assertIsNone(summary["sufficient_decrease_limit"])
            self.assertEqual(summary["backtrack_count"], 0)
            replay = stabilized.replay_stabilized_engineering_grip_archive(path)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["accepted_steps"], 1)

    def test_fixed_policy_unconverged_attempt_is_rejected_and_cut_back(self) -> None:
        config = replace(
            self.config,
            relaxation_policy="fixed",
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=None,
            max_backtracks=0,
            base=replace(
                self.config.base,
                max_stagger_iterations=1,
                damage=engineering.GateTolerance(1.0e-20, 0.0, 1.0),
            ),
        )
        with self.assertRaises(
            stabilized.StabilizedEngineeringGripContinuationFailure
        ) as context:
            stabilized.run_stabilized_engineering_grip(self.model, config)

        failure = context.exception
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        self.assertGreaterEqual(len(failure.attempts), 2)
        self.assertFalse(failure.attempts[-1].accepted)
        self.assertEqual(failure.attempts[0].cutback_count, 0)
        self.assertEqual(failure.attempts[-1].cutback_count, 1)
        self.assertEqual(failure.attempts[-1].committed_before_digest,
                         failure.attempts[-1].committed_after_digest)

    def test_public_writer_requires_provenance_and_loader_dispatches_only_v5(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-dispatch-") as name:
            directory = Path(name)
            with self.assertRaises(TypeError):
                writer: Any = stabilized.write_stabilized_engineering_grip_archive
                writer(directory / "missing-source.npz", self.result)
            legacy = directory / "legacy.npz"
            np.savez_compressed(legacy, archive_schema=np.asarray("legacy"))
            with self.assertRaisesRegex(ValueError, "only accepts v5"):
                stabilized.load_stabilized_engineering_grip_archive(legacy)

    def test_source_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-source-tamper-") as name:
            directory = Path(name)
            path, original = self._write_archive(directory)
            forged = self._copy_payload(original)

            def forge_source_metadata(metadata: dict[str, Any]) -> None:
                metadata["source"].update({"sha256": "0" * 64})
                metadata.update({"source_sha256": "0" * 64})

            self._update_metadata(
                forged,
                forge_source_metadata,
            )
            forged_path = self._write_payload(directory, "forged-metadata.npz", forged)
            with self.assertRaises(ValueError):
                stabilized.replay_stabilized_engineering_grip_archive(forged_path)

            source_bytes = self.source_path.read_bytes()
            try:
                with np.load(self.source_path, allow_pickle=False) as archive:
                    source = {name: np.array(archive[name], copy=True) for name in archive.files}
                source["volumes"][0] *= 2.0
                np.savez_compressed(  # type: ignore[arg-type]
                    str(self.source_path), **source  # type: ignore[arg-type]
                )
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    stabilized.replay_stabilized_engineering_grip_archive(path)
            finally:
                self.source_path.write_bytes(source_bytes)

    def test_state_reaction_work_and_solver_summary_tampering_are_rejected(self) -> None:
        mutations: dict[str, Callable[[dict[str, np.ndarray]], None]] = {
            "q": lambda payload: payload["state_union_displacement"].__setitem__(
                (1, 0), payload["state_union_displacement"][1, 0] + 1.0e-6
            ),
            "d": lambda payload: payload["state_material_damage"].__setitem__(
                (1, 0), payload["state_material_damage"][1, 0] + 1.0e-6
            ),
            "H": lambda payload: payload["state_material_history"].__setitem__(
                (1, 0), payload["state_material_history"][1, 0] + 1.0e-6
            ),
            "load": lambda payload: payload["state_u_bar"].__setitem__(
                1, payload["state_u_bar"][1] + 1.0e-6
            ),
            "reaction": lambda payload: payload["accepted_top_reaction_after"].__setitem__(
                0, payload["accepted_top_reaction_after"][0] + 1.0e-6
            ),
            "work": lambda payload: payload["work_check_reaction"].__setitem__(
                (), payload["work_check_reaction"].item() + 1.0e-6
            ),
            "state_work": lambda payload: payload["state_external_work"].__setitem__(
                1, payload["state_external_work"][1] + 1.0e-6
            ),
            "summary": lambda payload: self._update_summary(
                payload,
                1,
                lambda record: record.update(
                    {"top_reaction_after": record["top_reaction_after"] + 1.0e-6}
                ),
            ),
        }
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-state-tamper-") as name:
            directory = Path(name)
            _, original = self._write_archive(directory)
            for field, mutate in mutations.items():
                with self.subTest(field=field):
                    forged = self._copy_payload(original)
                    mutate(forged)
                    forged_path = self._write_payload(directory, f"forged-{field}.npz", forged)
                    with self.assertRaises(ValueError):
                        stabilized.replay_stabilized_engineering_grip_archive(forged_path)

    def test_failure_archive_marks_unvalidated_committed_state_and_replays(self) -> None:
        config = replace(
            self.config,
            base=replace(
                self.config.base,
                max_stagger_iterations=1,
                damage=engineering.GateTolerance(1.0e-20, 0.0, 1.0),
            ),
        )
        events: list[dict[str, Any]] = []
        with self.assertRaises(stabilized.StabilizedEngineeringGripContinuationFailure) as context:
            stabilized.run_stabilized_engineering_grip(
                self.model,
                config,
                progress_callback=events.append,
                progress_every_sweeps=1,
            )
        failure = context.exception
        self.assertEqual(failure.last_committed_state.u_bar, 0.0)
        self.assertEqual(events[-1]["event"], "failure")
        self.assertEqual(events[-1]["accepted_steps"], 0)
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-failure-") as name:
            directory = Path(name)
            path = directory / "failure.npz"
            stabilized.write_stabilized_engineering_grip_failure_archive(
                path,
                failure,
                self.model,
                config,
                source_metadata=self.source_metadata,
            )
            payload = stabilized.load_stabilized_engineering_grip_archive(path)
            self.assertEqual(
                set(payload),
                set(stabilized._V5_ARCHIVE_BASE_FIELDS)
                | set(stabilized._V5_FAILURE_FIELDS),
            )
            replay = stabilized.replay_stabilized_engineering_grip_archive(path)
        self.assertTrue(replay["replayed"])
        self.assertFalse(replay["validated"])
        self.assertFalse(replay["field_equations_validated"])
        self.assertFalse(replay["path_replayed"])
        self.assertEqual(replay["accepted_steps"], 0)
        self.assertEqual(replay["completion_status"], stabilized.STABILIZED_FAILURE_STATUS)

    def test_budget_failure_after_acceptance_has_no_fake_rejected_candidate(self) -> None:
        config = replace(
            self.config,
            base=replace(self.config.base, load_final=2.0e-4),
        )
        budget = int(
            stabilized.estimate_stabilized_archive_bytes_for_counts(
                self.model.n_material,
                self.model.n_dofs,
                config,
                accepted_steps=1,
            )["archive_bytes"]
        )
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-budget-failure-") as name:
            directory = Path(name)
            archive = directory / "budget-failure.npz"
            with self.assertRaises(
                stabilized.StabilizedEngineeringGripArchiveBudgetFailure
            ) as context:
                stabilized.run_stabilized_engineering_grip(
                    self.model,
                    config,
                    archive_budget_bytes=budget,
                    archive_output_path=archive,
                )
            failure = context.exception
            self.assertTrue(failure.attempts)
            self.assertTrue(failure.attempts[-1].accepted)
            stabilized.write_stabilized_engineering_grip_failure_archive(
                archive,
                failure,
                self.model,
                config,
                source_metadata=self.source_metadata,
                archive_budget_bytes=budget,
                free_bytes=budget + stabilized.STABILIZED_FILESYSTEM_RESERVE_BYTES,
            )
            replay = stabilized.replay_stabilized_engineering_grip_archive(archive)
        self.assertEqual(replay["accepted_steps"], 1)
        self.assertEqual(replay["completion_status"], stabilized.STABILIZED_FAILURE_STATUS)

    def test_size_estimate_covers_materialized_lean_archive_without_static_bytes(self) -> None:
        estimate = stabilized.estimate_stabilized_archive_bytes_for_counts(
            self.model.n_material,
            self.model.n_dofs,
            self.config,
            accepted_steps=1,
        )
        self.assertEqual(estimate["static_bound_kind"], "rebuilt_from_source")
        self.assertNotIn("static_bytes", estimate)
        self.assertGreater(estimate["archive_bytes"], 0)
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-size-") as name:
            path, payload = self._write_archive(Path(name))
            del path
            materialized = stabilized._uncompressed_archive_bytes(payload)
        self.assertGreaterEqual(estimate["archive_bytes"], materialized)
        larger = stabilized.estimate_stabilized_archive_bytes_for_counts(
            self.model.n_material,
            self.model.n_dofs,
            self.config,
            accepted_steps=2,
        )
        self.assertGreater(larger["archive_bytes"], estimate["archive_bytes"])


if __name__ == "__main__":
    unittest.main()
