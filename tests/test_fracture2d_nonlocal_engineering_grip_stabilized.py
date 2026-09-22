"""Core tests for the lean v5 stabilized engineering-grip archive."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from typing import Any, Callable, Sequence, cast

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
            all(isinstance(item, (str, dict)) for item in value),
            "v5 scalar logs must not contain numerical vectors",
        )
        for item in value:
            _assert_scalar_tree(test, item)


def _residual_gate_record(
    *,
    failed: tuple[str, ...] = (),
    unavailable: tuple[str, ...] = (),
) -> dict[str, Any]:
    details: dict[str, dict[str, Any]] = {}
    for name in stabilized.ANDERSON_RESIDUAL_HARD_GATE_NAMES:
        is_available = name not in unavailable and not (
            name in {"force", "moment"}
            and bool({"mechanics_before", "mechanics_after"} & set(unavailable))
        )
        if not is_available:
            stage_reason = {
                "finite": "candidate_not_fully_evaluated",
                "mechanics_before": "mechanics_before_not_evaluated",
                "mechanics_after": "mechanics_after_not_evaluated",
                "force": "mechanics_after_not_evaluated",
                "moment": "mechanics_after_not_evaluated",
                "damage_range_lower": "damage_not_evaluated",
                "damage_range_upper": "damage_not_evaluated",
                "healing": "damage_not_evaluated",
                "history_ordering": "history_not_evaluated",
            }[name]
            details[name] = {
                "available": False,
                "passed": False,
                "raw": None,
                "scale": None,
                "limit": None,
                "normalized": None,
                "ratio_available": False,
                "metric_reason": "unavailable",
                "stage_reason": stage_reason,
                "cg_info": None,
                "cg_status_pass": None,
                "residual_limit_pass": None,
                "stage_metrics": None,
            }
            continue
        failed_here = name in failed
        scale = stage_metrics = None
        if name == "finite":
            raw = limit = normalized = None
            ratio_available = False
            cg_info = cg_status_pass = residual_limit_pass = None
            passed = not failed_here
            metric_reason = "boolean_gate"
        elif name in {"mechanics_before", "mechanics_after"}:
            cg_info = 1 if failed_here else 0
            cg_status_pass = cg_info == 0
            raw = 0.0
            limit = 1.0
            residual_limit_pass = raw < limit
            normalized = raw / limit
            ratio_available = True
            passed = cg_status_pass and residual_limit_pass
            metric_reason = "cg_info_and_free_residual_limit"
        elif name in {"force", "moment"}:
            raw = 2.0 if failed_here else 0.0
            stage_metrics = {
                stage: {
                    "error": raw,
                    "scale": 1.0,
                    "limit": 1.0,
                    "normalized": raw,
                    "ratio_available": True,
                    "passed": raw <= 1.0,
                }
                for stage in ("before", "after")
            }
            worst = stage_metrics["before"]
            scale = worst["scale"]
            limit = worst["limit"]
            normalized = worst["normalized"]
            ratio_available = True
            cg_info = cg_status_pass = residual_limit_pass = None
            passed = all(item["passed"] for item in stage_metrics.values())
            metric_reason = "stage_worst_ratio_and_conjunction"
        elif name in {
            "damage_range_lower",
            "damage_range_upper",
            "healing",
        }:
            raw = 2.0 if failed_here else 0.0
            limit = 1.0
            normalized = raw / limit
            ratio_available = True
            cg_info = cg_status_pass = residual_limit_pass = None
            passed = raw <= limit
            metric_reason = (
                "damage_range_excess_and_limit"
                if name.startswith("damage_range")
                else "healing_excess_and_limit"
            )
        else:
            raw = 1.0 if failed_here else 0.0
            limit = 0.0
            normalized = None
            ratio_available = False
            cg_info = cg_status_pass = residual_limit_pass = None
            passed = raw <= limit
            metric_reason = "history_ordering_excess_and_zero_limit"
        details[name] = {
            "available": True,
            "passed": passed,
            "raw": raw,
            "scale": scale,
            "limit": limit,
            "normalized": normalized,
            "ratio_available": ratio_available,
            "metric_reason": metric_reason,
            "stage_reason": None,
            "cg_info": cg_info,
            "cg_status_pass": cg_status_pass,
            "residual_limit_pass": residual_limit_pass,
            "stage_metrics": stage_metrics,
        }
    failed_names = [
        name
        for name in stabilized.ANDERSON_RESIDUAL_HARD_GATE_NAMES
        if not details[name]["passed"]
    ]
    return {
        "record_type": "proposal",
        "schema": stabilized.ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
        "candidate_kind": "anderson_accelerated",
        "candidate_map_status": "completed",
        "failure_stage": None,
        "complete": True,
        "hard_pass": not failed_names,
        "hard_gate_status": "evaluated",
        "hard_gate_attribution_schema": (
            stabilized.ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA
        ),
        "hard_gate_diagnostics": details,
        "failed_hard_gates": failed_names,
    }


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


class StabilizedLoadScheduleArtifactTest(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _schedule(root: Path) -> Any:
        targets = np.asarray([0.25, 0.5], dtype=np.dtype("<f8"))
        source = root / (
            "engineering_grip_anderson_u5p9875e3_dumin6p25e6_s144.npz"
        )
        np.savez_compressed(
            str(source),
            archive_schema=np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "schema": stabilized.STABILIZED_ARCHIVE_SCHEMA,
                        "kind": "completed",
                        "completion_status": stabilized.STABILIZED_COMPLETION_STATUS,
                        "stabilized_config": {"relaxation_policy": "anderson"},
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                dtype=np.str_,
            ),
            state_u_bar=np.asarray([0.0, *targets], dtype=np.dtype("<f8")),
        )
        output = root / "schedule.npz"
        source_sha256 = stabilized._archive_sha256(source)
        target_sha256 = stabilized._schedule_target_sha256(targets)
        with (
            mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256",
                source_sha256,
            ),
            mock.patch.object(
                stabilized, "STABILIZED_SCHEDULE_TARGET_SHA256", target_sha256
            ),
            mock.patch.object(stabilized, "STABILIZED_SCHEDULE_TARGET_COUNT", 2),
        ):
            schedule = stabilized.extract_stabilized_load_schedule(source, output)
            yield schedule

    def test_replay_sequence_reports_completion_and_exact_next_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="matched-load-replay-") as directory:
            root = Path(directory)
            with self._schedule(root) as schedule:
                base = replace(
                    engineering.EngineeringGripConfig.smoke(),
                    load_final=0.5,
                )
                config = stabilized.StabilizedEngineeringGripConfig(
                    base=base,
                    omega_initial=0.5,
                    omega_min=0.5,
                    omega_reduction=None,
                    sufficient_decrease=1.0e-2,
                    max_backtracks=0,
                    relaxation_policy="anderson_residual",
                    anderson_history_depth=5,
                    anderson_beta=0.5,
                    load_schedule=schedule,
                )
                completed = stabilized._v5_validate_schedule_sequence(
                    {},
                    {},
                    config,
                    {"u": np.asarray([0.0, 0.25, 0.5], dtype=np.float64)},
                    {"failure_summary": None},
                    failure=False,
                )
                self.assertEqual(completed["target_count"], 2)
                self.assertTrue(completed["completed"])
                self.assertIsNone(completed["target_index"])
                failure = stabilized._v5_validate_schedule_sequence(
                    {},
                    {},
                    config,
                    {"u": np.asarray([0.0, 0.25], dtype=np.float64)},
                    {
                        "failure_summary": {
                            "available": True,
                            "load_start": 0.25,
                            "load_target": 0.5,
                            "increment": 0.25,
                            "cutback_count": 0,
                        }
                    },
                    failure=True,
                )
                self.assertEqual(failure["target_index"], 1)
                self.assertEqual(failure["next_target"], 0.5)
                self.assertFalse(failure["completed"])
                with self.assertRaisesRegex(ValueError, "exact zero-plus-schedule prefix"):
                    stabilized._v5_validate_schedule_sequence(
                        {},
                        {},
                        config,
                        {"u": np.asarray([0.0, 0.3], dtype=np.float64)},
                        {"failure_summary": None},
                        failure=True,
                    )

    def test_public_constructor_rejects_unverified_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="matched-load-unverified-") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                stabilized.StabilizedLoadSchedule(
                    path=str((root / "missing.npz").resolve()),
                    file_sha256="1" * 64,
                    targets=np.asarray([0.25], dtype=np.dtype("<f8")),
                    canonical_target_sha256="2" * 64,
                    source_archive_path=str((root / "historical.npz").resolve()),
                    source_archive_sha256="3" * 64,
                    target_count=1,
                    load_start=0.0,
                    load_final=0.25,
                    source_state_count=2,
                )

    def test_extraction_and_loader_require_the_exact_numeric_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="matched-load-schedule-") as directory:
            root = Path(directory)
            source = root / "historical.npz"
            state_u = np.asarray([0.0, 0.25, 0.5], dtype=np.dtype("<f8"))
            source_metadata = {
                "schema": stabilized.STABILIZED_ARCHIVE_SCHEMA,
                "kind": "completed",
                "completion_status": stabilized.STABILIZED_COMPLETION_STATUS,
                "stabilized_config": {"relaxation_policy": "anderson"},
            }
            np.savez_compressed(
                str(source),
                archive_schema=np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA),
                metadata_json=np.asarray(
                    json.dumps(source_metadata, sort_keys=True, allow_nan=False),
                    dtype=np.str_,
                ),
                state_u_bar=state_u,
            )
            source_sha256 = stabilized._archive_sha256(source)
            target_sha256 = stabilized._schedule_target_sha256(state_u[1:])
            output = root / "schedule.npz"
            with mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256",
                source_sha256,
            ), mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_TARGET_SHA256",
                target_sha256,
            ), mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_TARGET_COUNT",
                2,
            ):
                schedule = stabilized.extract_stabilized_load_schedule(source, output)
                self.assertEqual(schedule.targets.dtype.str, "<f8")
                np.testing.assert_array_equal(schedule.targets, state_u[1:])
                self.assertEqual(schedule.source_state_count, 3)
                with np.load(output, allow_pickle=False) as archive:
                    self.assertEqual(set(archive.files), {"targets", "metadata_json"})
                    self.assertEqual(archive["targets"].dtype.str, "<f8")
                    self.assertEqual(archive["metadata_json"].shape, ())
                loaded = stabilized.load_stabilized_load_schedule(
                    output,
                    load_start=0.0,
                    load_final=0.5,
                )
                self.assertEqual(loaded.payload(), schedule.payload())
                with self.assertRaises(FileExistsError):
                    stabilized.extract_stabilized_load_schedule(source, output)

                forged = root / "forged.npz"
                with np.load(output, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"]))
                metadata["target_count"] = 1
                np.savez_compressed(
                    str(forged),
                    targets=np.asarray(schedule.targets, dtype=np.dtype("<f8")),
                    metadata_json=np.asarray(
                        json.dumps(metadata, sort_keys=True, allow_nan=False),
                        dtype=np.str_,
                    ),
                )
                with self.assertRaisesRegex(ValueError, "target shape/count"):
                    stabilized.load_stabilized_load_schedule(
                        forged,
                        load_start=0.0,
                        load_final=0.5,
                    )


class AndersonScalarDiagnosticContractTest(unittest.TestCase):
    def test_durable_boundaries_reject_schema_less_attribution(self) -> None:
        record = _residual_gate_record()
        record.pop("schema")
        with tempfile.TemporaryDirectory(prefix="anderson-schema-less-") as directory:
            root = Path(directory)
            archive = root / "result.npz"
            archive.write_bytes(b"exact archive bytes")
            diagnostic = root / "result.anderson.jsonl"
            writer = stabilized.AndersonScalarDiagnosticWriter(
                diagnostic,
                binding={
                    "source_sha256": "1" * 64,
                    "configuration_sha256": "2" * 64,
                    "source_bundle_hash": "3" * 64,
                    "source_bundle_files": ["source.py"],
                },
            )
            with self.assertRaisesRegex(ValueError, "current scalar schema"):
                writer.emit(record)
            writer.finalize(archive, status="completed")
            writer.close()

            lines = diagnostic.read_text(encoding="utf-8").splitlines()
            lines.insert(1, json.dumps(record, separators=(",", ":")))
            diagnostic.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current scalar schema"):
                stabilized.load_anderson_scalar_diagnostic(diagnostic)

    def test_writer_binds_final_archive_and_rejects_vectors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anderson-scalar-diagnostic-") as directory:
            root = Path(directory)
            archive = root / "result.npz"
            archive.write_bytes(b"exact archive bytes")
            diagnostic = root / "result.anderson.jsonl"
            writer = stabilized.AndersonScalarDiagnosticWriter(
                diagnostic,
                binding={
                    "source_sha256": "1" * 64,
                    "configuration_sha256": "2" * 64,
                    "source_bundle_hash": "3" * 64,
                    "source_bundle_files": ["source.py"],
                },
            )
            writer.emit(
                {
                    "record_type": "sweep",
                    "schema": stabilized.ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
                    "candidate_merit": 0.25,
                    "candidate_merit_components": {"history_transition": 0.25},
                }
            )
            with self.assertRaises(TypeError):
                writer.emit(
                    {
                        "record_type": "invalid",
                        "field_vector": np.zeros(2, dtype=np.float64),
                    }
                )
            with self.assertRaises(TypeError):
                writer.emit(
                    {"record_type": "invalid", "numerical_vector": [1.0, 2.0]}
                )
            archive_hash = writer.finalize(archive, status="completed")
            writer.close()

            records = [
                json.loads(line)
                for line in diagnostic.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["record_type"], "header")
            self.assertEqual(records[-1]["record_type"], "final")
            self.assertEqual(records[-1]["final_archive_sha256"], archive_hash)
            self.assertEqual(
                archive_hash,
                stabilized._archive_sha256(archive),
            )
            self.assertTrue(records[-1]["finalized"])

    def test_writer_accepts_bound_completed_or_failure_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anderson-scalar-paths-") as directory:
            root = Path(directory)
            archive = root / "result.npz"
            failure_archive = root / "result.failure.npz"
            archive.write_bytes(b"completed archive bytes")
            failure_archive.write_bytes(b"failure archive bytes")
            diagnostic = root / "result.anderson.jsonl"
            writer = stabilized.AndersonScalarDiagnosticWriter(
                diagnostic,
                binding={
                    "expected_archive_path": str(archive),
                    "expected_archive_paths": [
                        str(archive),
                        str(failure_archive),
                    ],
                },
            )
            try:
                writer.finalize(failure_archive, status="failed")
            finally:
                writer.close()
            parsed = stabilized.load_anderson_scalar_diagnostic(
                diagnostic,
                archive_path=failure_archive,
            )
            self.assertEqual(
                parsed["final_archive_sha256"],
                stabilized._archive_sha256(failure_archive),
            )
            forged_path = root / "unbound.npz"
            forged_path.write_bytes(failure_archive.read_bytes())
            with self.assertRaisesRegex(ValueError, "path does not match archive"):
                stabilized.load_anderson_scalar_diagnostic(
                    diagnostic,
                    archive_path=forged_path,
                )

    def test_archive_binding_pairs_path_kind_status_and_schema(self) -> None:
        def write_identity_archive(
            path: Path,
            *,
            schema: str,
            status: str,
            kind: str,
            schedule: dict[str, Any] | None = None,
        ) -> None:
            metadata: dict[str, Any] = {
                "schema": schema,
                "completion_status": status,
                "kind": kind,
            }
            if schedule is not None:
                metadata["load_schedule"] = dict(schedule)
            np.savez_compressed(
                str(path),
                archive_schema=np.asarray(schema, dtype=np.str_),
                completion_status=np.asarray(status, dtype=np.str_),
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, allow_nan=False),
                    dtype=np.str_,
                ),
            )

        for label, completed_schema, failure_schema, schedule in (
            (
                "adaptive",
                stabilized.STABILIZED_ARCHIVE_SCHEMA,
                stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA,
                None,
            ),
            (
                "schedule",
                stabilized.STABILIZED_SCHEDULE_ARCHIVE_SCHEMA,
                stabilized.STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA,
                {"canonical_target_sha256": "a" * 64, "target_count": 1},
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"anderson-bound-{label}-"
            ) as directory:
                root = Path(directory)
                completed = root / f"{label}.npz"
                failure = root / f"{label}.failure.npz"
                write_identity_archive(
                    completed,
                    schema=completed_schema,
                    status=stabilized.STABILIZED_COMPLETION_STATUS,
                    kind="completed",
                    schedule=schedule,
                )
                write_identity_archive(
                    failure,
                    schema=failure_schema,
                    status=stabilized.STABILIZED_FAILURE_STATUS,
                    kind="failure",
                    schedule=schedule,
                )
                binding: dict[str, Any] = {
                    "expected_archive_path": str(completed),
                    "expected_archive_paths": [str(completed), str(failure)],
                    "expected_archive_bindings": [
                        {
                            "path": str(completed),
                            "path_suffix": ".npz",
                            "kind": "completed",
                            "completion_status": stabilized.STABILIZED_COMPLETION_STATUS,
                            "archive_schema": completed_schema,
                        },
                        {
                            "path": str(failure),
                            "path_suffix": ".failure.npz",
                            "kind": "failure",
                            "completion_status": stabilized.STABILIZED_FAILURE_STATUS,
                            "archive_schema": failure_schema,
                        },
                    ],
                    "expected_archive_schemas": [completed_schema, failure_schema],
                    "expected_completion_statuses": [
                        stabilized.STABILIZED_COMPLETION_STATUS,
                        stabilized.STABILIZED_FAILURE_STATUS,
                    ],
                }
                if schedule is not None:
                    binding["load_schedule"] = schedule
                diagnostic = root / f"{label}.anderson.jsonl"
                writer = stabilized.AndersonScalarDiagnosticWriter(
                    diagnostic, binding=binding
                )
                try:
                    writer.finalize(
                        completed,
                        status=stabilized.STABILIZED_COMPLETION_STATUS,
                    )
                finally:
                    writer.close()
                parsed = stabilized.load_anderson_scalar_diagnostic(
                    diagnostic, archive_path=completed
                )
                final = parsed["final"]
                self.assertEqual(final["final_archive_kind"], "completed")
                self.assertEqual(final["final_archive_path_suffix"], ".npz")
                self.assertEqual(final["final_archive_schema"], completed_schema)

                cross_status = root / f"{label}.cross-status.jsonl"
                writer = stabilized.AndersonScalarDiagnosticWriter(
                    cross_status, binding=binding
                )
                try:
                    with self.assertRaisesRegex(ValueError, "does not match its bound archive"):
                        writer.finalize(
                            completed,
                            status=stabilized.STABILIZED_FAILURE_STATUS,
                        )
                finally:
                    writer.close()

                forged_failure = root / f"{label}.forged.failure.npz"
                write_identity_archive(
                    forged_failure,
                    schema=completed_schema,
                    status=stabilized.STABILIZED_COMPLETION_STATUS,
                    kind="completed",
                    schedule=schedule,
                )
                forged_binding = dict(binding)
                forged_binding["expected_archive_path"] = str(completed)
                forged_binding["expected_archive_paths"] = [
                    str(completed),
                    str(forged_failure),
                ]
                forged_entries = [dict(entry) for entry in binding["expected_archive_bindings"]]
                forged_entries[1]["path"] = str(forged_failure)
                forged_binding["expected_archive_bindings"] = forged_entries
                forged_diagnostic = root / f"{label}.forged.jsonl"
                writer = stabilized.AndersonScalarDiagnosticWriter(
                    forged_diagnostic, binding=forged_binding
                )
                try:
                    with self.assertRaisesRegex(ValueError, "not paired"):
                        writer.finalize(
                            forged_failure,
                            status=stabilized.STABILIZED_FAILURE_STATUS,
                        )
                finally:
                    writer.close()

    def test_writer_io_errors_propagate_for_write_flush_and_fsync(self) -> None:
        binding = {
            "source_sha256": "1" * 64,
            "configuration_sha256": "2" * 64,
            "source_bundle_hash": "3" * 64,
            "source_bundle_files": ["source.py"],
        }
        event = {"record_type": "sweep", "value": 1.0}
        for operation in ("write", "flush"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory(
                prefix=f"anderson-scalar-{operation}-error-"
            ) as directory:
                writer = stabilized.AndersonScalarDiagnosticWriter(
                    Path(directory) / "diagnostic.jsonl",
                    binding=binding,
                )
                handle = mock.Mock()
                if operation == "write":
                    handle.write.side_effect = OSError("manufactured write failure")
                else:
                    handle.write.return_value = 0
                    handle.flush.side_effect = OSError("manufactured flush failure")
                original_handle = writer._handle
                writer._handle = handle
                try:
                    with self.assertRaisesRegex(OSError, "cannot write Anderson diagnostic"):
                        writer.emit(event)
                finally:
                    writer._handle = original_handle
                    writer.close()

        with tempfile.TemporaryDirectory(prefix="anderson-scalar-fsync-error-") as directory:
            root = Path(directory)
            archive = root / "result.npz"
            archive.write_bytes(b"exact archive bytes")
            writer = stabilized.AndersonScalarDiagnosticWriter(
                root / "diagnostic.jsonl",
                binding=binding,
            )
            try:
                with mock.patch.object(
                    stabilized.os,
                    "fsync",
                    side_effect=OSError("manufactured fsync failure"),
                ), self.assertRaisesRegex(
                    OSError, "cannot finalize Anderson diagnostic"
                ):
                    writer.finalize(archive, status="completed")
            finally:
                writer.close()

    def test_scheduled_scalar_header_and_final_bind_the_schedule_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anderson-scalar-schedule-") as name:
            root = Path(name)
            with StabilizedLoadScheduleArtifactTest._schedule(root) as schedule:
                archive = root / "scheduled-result.npz"
                np.savez_compressed(
                    str(archive),
                    archive_schema=np.asarray(
                        stabilized.STABILIZED_SCHEDULE_ARCHIVE_SCHEMA
                    ),
                    metadata_json=np.asarray(
                        json.dumps(
                            {"load_schedule": schedule.payload()},
                            sort_keys=True,
                            allow_nan=False,
                        ),
                        dtype=np.str_,
                    ),
                )
                diagnostic = root / "scheduled.anderson.jsonl"
                completed_contract = stabilized._stabilized_contract(
                    stabilized.STABILIZED_ACCEPTANCE_MODE,
                    failure=False,
                    schedule=True,
                )
                failure_contract = stabilized._stabilized_contract(
                    stabilized.STABILIZED_ACCEPTANCE_MODE,
                    failure=True,
                    schedule=True,
                )
                writer = stabilized.AndersonScalarDiagnosticWriter(
                    diagnostic,
                    binding={
                        "source_sha256": "1" * 64,
                        "configuration_sha256": "2" * 64,
                        "source_bundle_hash": "3" * 64,
                        "source_bundle_files": ["source.py"],
                        "expected_archive_schemas": [
                            completed_contract["archive_schema"],
                            failure_contract["archive_schema"],
                        ],
                        "expected_completion_statuses": [
                            completed_contract["completion_status"],
                            failure_contract["completion_status"],
                        ],
                        "load_schedule": schedule.payload(),
                    },
                )
                try:
                    writer.finalize(
                        archive,
                        status=completed_contract["completion_status"],
                    )
                finally:
                    writer.close()
                records = [
                    json.loads(line)
                    for line in diagnostic.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(records[0]["load_schedule"], schedule.payload())
                self.assertEqual(records[-1]["load_schedule"], schedule.payload())
                self.assertEqual(
                    records[-1]["schedule_target_sha256"],
                    schedule.canonical_target_sha256,
                )
                self.assertEqual(
                    records[-1]["schedule_target_count"], schedule.target_count
                )
                self.assertEqual(
                    records[-1]["schedule_endpoint"], schedule.load_final
                )

    def test_residual_hard_gate_attribution_counts_overlaps_and_missing_stage(self) -> None:
        record = _residual_gate_record(
            failed=("force", "moment"),
            unavailable=("mechanics_after",),
        )
        stabilized._validate_anderson_scalar_event(record)
        with tempfile.TemporaryDirectory(prefix="anderson-hard-gates-") as directory:
            root = Path(directory)
            archive = root / "result.npz"
            archive.write_bytes(b"exact archive bytes")
            diagnostic = root / "result.anderson.jsonl"
            writer = stabilized.AndersonScalarDiagnosticWriter(
                diagnostic,
                binding={
                    "source_sha256": "1" * 64,
                    "configuration_sha256": "2" * 64,
                    "source_bundle_hash": "3" * 64,
                    "source_bundle_files": ["source.py"],
                },
            )
            try:
                writer.emit(record)
                writer.finalize(archive, status="completed")
            finally:
                writer.close()
            parsed = stabilized.load_anderson_scalar_diagnostic(
                diagnostic,
                archive_path=archive,
            )
            report = stabilized.summarize_anderson_scalar_diagnostic(parsed)
            self.assertEqual(report["attributed_proposal_records"], 1)
            self.assertEqual(report["hard_gate_failure_counts"]["force"], 1)
            self.assertEqual(
                report["hard_gate_unavailable_counts"]["mechanics_after"], 1
            )
            self.assertEqual(
                report["hard_gate_overlap_counts"]["mechanics_after+force"], 1
            )
            self.assertEqual(
                report["hard_gate_overlap_counts"]["force+moment"], 1
            )

    def test_residual_hard_gate_validator_rejects_forged_flags_lists_and_metrics(self) -> None:
        valid = _residual_gate_record(failed=("force",))
        for label, forged in (
            (
                "failed list",
                {**valid, "failed_hard_gates": ["moment"]},
            ),
            (
                "boolean",
                {
                    **valid,
                    "hard_gate_diagnostics": {
                        **valid["hard_gate_diagnostics"],
                        "force": {
                            **valid["hard_gate_diagnostics"]["force"],
                            "passed": "false",
                        },
                    },
                },
            ),
            (
                "metric",
                {
                    **valid,
                    "hard_gate_diagnostics": {
                        **valid["hard_gate_diagnostics"],
                        "force": {
                            **valid["hard_gate_diagnostics"]["force"],
                            "raw": [1.0, 2.0],
                        },
                    },
                },
            ),
            (
                "schema",
                {
                    key: value
                    for key, value in valid.items()
                    if key != "hard_gate_attribution_schema"
                },
            ),
        ):
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                stabilized._validate_anderson_scalar_event(forged)

    def test_backtracking_scalar_trial_alpha_is_required_and_canonical(self) -> None:
        valid = {
            "record_type": "proposal",
            "selection_merit_identity": (
                stabilized.STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY
            ),
            "candidate_kind": "anderson_accelerated",
            "trial_alpha": 0.5,
            "trial_alpha_index": 1,
        }
        stabilized._validate_anderson_scalar_event(valid)
        for label, forged, pattern in (
            (
                "missing",
                {key: value for key, value in valid.items() if key != "trial_alpha"},
                "missing trial_alpha",
            ),
            (
                "off schedule",
                {**valid, "trial_alpha": 0.3},
                "configured schedule",
            ),
            (
                "on schedule but wrong index",
                {**valid, "trial_alpha": 0.25},
                "paired with trial_alpha_index",
            ),
            (
                "forged trial index",
                {**valid, "trial_alpha_index": 0},
                "paired with trial_alpha_index",
            ),
            (
                "old identity",
                {
                    **valid,
                    "selection_merit_identity": (
                        stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY
                    ),
                },
                "valid only",
            ),
            (
                "fallback alpha",
                {**valid, "candidate_kind": "picard_fallback"},
                "fallback retains",
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, pattern):
                stabilized._validate_anderson_scalar_event(forged)

        first_trial = {**valid, "trial_alpha": 1.0, "trial_alpha_index": 0}
        stabilized._validate_anderson_scalar_event(first_trial)
        with self.assertRaisesRegex(ValueError, "paired with trial_alpha_index"):
            stabilized._validate_anderson_scalar_event(
                {**first_trial, "trial_alpha": 0.25}
            )

        sweep = {
            "record_type": "sweep",
            "selection_merit_identity": (
                stabilized.STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY
            ),
            "anderson_selection": "accelerated",
            "trial_alpha": 0.5,
            "backtrack_count": 1,
        }
        stabilized._validate_anderson_scalar_event(sweep)
        with self.assertRaisesRegex(ValueError, "backtrack_count"):
            stabilized._validate_anderson_scalar_event(
                {**sweep, "trial_alpha": 0.25}
            )

    def test_regularized_scalar_fields_are_explicit_and_lambda_is_authoritative(self) -> None:
        valid = {
            "record_type": "proposal",
            "selection_merit_identity": (
                stabilized.STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY
            ),
            "candidate_kind": "anderson_accelerated",
            "trial_alpha": 1.0,
            "trial_alpha_index": 0,
            "anderson_regularization_mu": 2.0,
            "anderson_current_residual_l2": 3.0,
            "anderson_regularization_lambda": 18.0,
            "anderson_unregularized_ls_residual_l2": 0.25,
            "anderson_coefficient_l2": 0.5,
        }
        stabilized._validate_anderson_scalar_event(valid)
        for label, forged, pattern in (
            (
                "missing current residual",
                {
                    key: value
                    for key, value in valid.items()
                    if key != "anderson_current_residual_l2"
                },
                "missing",
            ),
            (
                "wrong lambda",
                {**valid, "anderson_regularization_lambda": 17.0},
                "paired with mu and current residual",
            ),
            (
                "wrong mu",
                {**valid, "anderson_regularization_mu": 1.0},
                "paired with mu and current residual",
            ),
            (
                "ambiguous old field",
                {**valid, "unregularized_residual_norm": 3.0},
                "regularization diagnostics",
            ),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, pattern):
                stabilized._validate_anderson_scalar_event(forged)

    def test_residual_hard_gate_validator_rejects_inconsistent_metric_provenance(self) -> None:
        valid = _residual_gate_record(failed=("force",))

        def with_detail(name: str, **updates: Any) -> dict[str, Any]:
            details = {
                gate: dict(item)
                for gate, item in valid["hard_gate_diagnostics"].items()
            }
            details[name].update(updates)
            result = dict(valid)
            result["hard_gate_diagnostics"] = details
            result["failed_hard_gates"] = [
                gate
                for gate in stabilized.ANDERSON_RESIDUAL_HARD_GATE_NAMES
                if not details[gate]["passed"]
            ]
            result["hard_pass"] = not result["failed_hard_gates"]
            return result

        all_null = {
            gate: {
                **item,
                "raw": None,
                "limit": None,
                "normalized": None,
            }
            for gate, item in valid["hard_gate_diagnostics"].items()
        }
        all_null_record = dict(valid)
        all_null_record["hard_gate_diagnostics"] = all_null
        missing_status = dict(valid)
        missing_status.pop("hard_gate_status")
        missing_map_status = dict(valid)
        missing_map_status.pop("candidate_map_status")
        for label, forged in (
            ("all metrics null", all_null_record),
            ("missing v3 status", missing_status),
            ("missing candidate-map status", missing_map_status),
            (
                "failed map with evaluated status",
                {**valid, "candidate_map_status": "failed"},
            ),
            (
                "candidate-map failure on Picard proposal",
                {
                    **valid,
                    "candidate_kind": "picard_fallback",
                    "candidate_map_status": "failed",
                    "failure_stage": "anderson_candidate_damage",
                    "complete": False,
                    "hard_gate_status": "candidate_map_failure",
                    "hard_pass": False,
                },
            ),
            (
                "fake metric reason",
                with_detail("force", metric_reason="forged_reason"),
            ),
            (
                "pass mismatch",
                with_detail("force", passed=True),
            ),
            (
                "raw limit ratio mismatch",
                with_detail("force", normalized=0.0),
            ),
            (
                "stage ratio mismatch",
                with_detail(
                    "force",
                    stage_metrics={
                        **valid["hard_gate_diagnostics"]["force"]["stage_metrics"],
                        "before": {
                            **valid["hard_gate_diagnostics"]["force"][
                                "stage_metrics"
                            ]["before"],
                            "normalized": 0.0,
                        },
                    },
                ),
            ),
            (
                "stage pass mismatch",
                with_detail(
                    "force",
                    stage_metrics={
                        **valid["hard_gate_diagnostics"]["force"]["stage_metrics"],
                        "before": {
                            **valid["hard_gate_diagnostics"]["force"][
                                "stage_metrics"
                            ]["before"],
                            "passed": True,
                        },
                    },
                ),
            ),
            (
                "aggregate representative mismatch",
                with_detail("force", scale=2.0),
            ),
            (
                "nonzero cg claimed pass",
                with_detail(
                    "mechanics_before",
                    cg_info=1,
                    cg_status_pass=True,
                    passed=True,
                ),
            ),
            (
                "residual over limit claimed pass",
                with_detail(
                    "mechanics_before",
                    raw=2.0,
                    normalized=2.0,
                    residual_limit_pass=True,
                    passed=True,
                ),
            ),
        ):
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                stabilized._validate_anderson_scalar_event(forged)

        valid = _residual_gate_record()
        zero_limit = with_detail(
            "mechanics_before",
            raw=0.0,
            limit=0.0,
            normalized=None,
            ratio_available=False,
            cg_info=0,
            cg_status_pass=True,
            residual_limit_pass=False,
            passed=False,
        )
        zero_limit["failed_hard_gates"] = ["mechanics_before"]
        zero_limit["hard_pass"] = False
        stabilized._validate_anderson_scalar_event(zero_limit)

    def test_zero_force_and_moment_limits_use_raw_pass_and_unavailable_ratio(self) -> None:
        zero_gate = engineering.GateTolerance(0.0, 0.0, 1.0)
        result: Any = SimpleNamespace(
            outer_reaction=0.0,
            bottom_reaction=0.0,
            anchor_reaction=0.0,
        )

        for component, gate_index in (("force", 4), ("moment", 5)):
            with self.subTest(component=component):
                passing_stage = stabilized._anderson_force_moment_stage_metric(
                    result,
                    (True, True, 0.0, 0.0, True, True),
                    zero_gate,
                    component=component,
                    geometry_length=1.0,
                )
                failing_stage = stabilized._anderson_force_moment_stage_metric(
                    result,
                    (
                        True,
                        True,
                        1.0 if gate_index == 4 else 0.0,
                        1.0 if gate_index == 5 else 0.0,
                        False if gate_index == 4 else True,
                        False if gate_index == 5 else True,
                    ),
                    zero_gate,
                    component=component,
                    geometry_length=1.0,
                )
                self.assertEqual(
                    (passing_stage["limit"], passing_stage["normalized"], passing_stage["ratio_available"]),
                    (0.0, None, False),
                )
                self.assertTrue(passing_stage["passed"])
                self.assertFalse(failing_stage["passed"])

                passing_aggregate = stabilized._anderson_force_moment_aggregate_detail(
                    {"before": passing_stage, "after": passing_stage},
                    passed=True,
                )
                self.assertEqual(
                    (passing_aggregate["raw"], passing_aggregate["limit"]),
                    (0.0, 0.0),
                )
                self.assertIsNone(passing_aggregate["normalized"])
                self.assertFalse(passing_aggregate["ratio_available"])

                failing_aggregate = stabilized._anderson_force_moment_aggregate_detail(
                    {"before": failing_stage, "after": passing_stage},
                    passed=False,
                )
                self.assertEqual(
                    (failing_aggregate["raw"], failing_aggregate["limit"]),
                    (1.0, 0.0),
                )
                self.assertIsNone(failing_aggregate["normalized"])
                self.assertFalse(failing_aggregate["ratio_available"])

                finite_failing_stage = {
                    **passing_stage,
                    "error": 2.0,
                    "limit": 1.0,
                    "normalized": 2.0,
                    "ratio_available": True,
                    "passed": False,
                }
                mixed_aggregate = stabilized._anderson_force_moment_aggregate_detail(
                    {"before": passing_stage, "after": finite_failing_stage},
                    passed=False,
                )
                self.assertEqual(
                    (mixed_aggregate["raw"], mixed_aggregate["limit"]),
                    (2.0, 1.0),
                )
                self.assertEqual(mixed_aggregate["normalized"], 2.0)
                self.assertTrue(mixed_aggregate["ratio_available"])

                record = _residual_gate_record()
                details = {
                    name: dict(item)
                    for name, item in record["hard_gate_diagnostics"].items()
                }
                details[component] = failing_aggregate
                forged = {
                    **record,
                    "hard_gate_diagnostics": details,
                    "hard_pass": False,
                    "failed_hard_gates": [component],
                }
                stabilized._validate_anderson_scalar_event(forged)

                forged_detail = {
                    **failing_aggregate,
                    "normalized": 0.0,
                    "ratio_available": True,
                }
                details[component] = forged_detail
                with self.assertRaises((TypeError, ValueError)):
                    stabilized._validate_anderson_scalar_event(
                        {
                            **forged,
                            "hard_gate_diagnostics": details,
                        }
                    )

    def test_old_scalar_stream_remains_readable_without_attribution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="anderson-old-stream-") as directory:
            root = Path(directory)
            archive = root / "result.npz"
            archive.write_bytes(b"exact archive bytes")
            diagnostic = root / "result.anderson.jsonl"
            archive_hash = stabilized._archive_sha256(archive)
            lines = [
                {
                    "record_type": "header",
                    "schema": stabilized.ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
                    "final_archive_sha256": None,
                    "final_archive_path": None,
                    "finalized": False,
                    "source_sha256": "1" * 64,
                },
                {
                    "record_type": "proposal",
                    "candidate_kind": "anderson_accelerated",
                    "hard_pass": True,
                },
                {
                    "record_type": "final",
                    "schema": stabilized.ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
                    "finalized": True,
                    "final_archive_path": str(archive),
                    "final_archive_sha256": archive_hash,
                    "final_archive_size_bytes": archive.stat().st_size,
                    "status": "completed",
                },
            ]
            diagnostic.write_text(
                "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
                encoding="utf-8",
            )
            parsed = stabilized.load_anderson_scalar_diagnostic(
                diagnostic,
                archive_path=archive,
            )
            report = stabilized.summarize_anderson_scalar_diagnostic(parsed)
            self.assertEqual(report["proposal_records"], 1)
            self.assertEqual(report["unattributed_proposal_records"], 1)
            self.assertFalse(report["hard_gate_attribution_available"])

    def test_manufactured_history_transition_exceeds_anderson_decrease_limit(self) -> None:
        config = stabilized.StabilizedEngineeringGripConfig.smoke()
        reference_merit = 1.0
        transition = 3.0e-6
        history_scale = 282.0
        history_ratio = stabilized._normalized(
            transition,
            config.base.history,
            history_scale,
            "manufactured history transition",
        )
        sufficient_limit = (1.0 - 0.01 * 0.5) * reference_merit

        self.assertAlmostEqual(history_ratio, 1.0638297872340425)
        self.assertGreater(history_ratio, sufficient_limit)
        damage_before = 0.25
        delta_history = transition
        residual_difference = -2.0 * (1.0 - damage_before) * delta_history
        self.assertAlmostEqual(residual_difference, -4.5e-6)


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


class AndersonProposalContractTest(unittest.TestCase):
    def test_type_ii_qr_proposal_matches_forward_difference_formula(self) -> None:
        history_x = (
            np.asarray([0.1, 0.2], dtype=np.float64),
            np.asarray([0.2, 0.35], dtype=np.float64),
        )
        history_f = (
            np.asarray([0.3, 0.4], dtype=np.float64),
            np.asarray([0.25, 0.45], dtype=np.float64),
        )
        x_current = np.asarray([0.3, 0.4], dtype=np.float64)
        g_current = np.asarray([0.5, 0.55], dtype=np.float64)
        proposal = stabilized._anderson_propose(
            history_x,
            history_f,
            x_current,
            g_current,
            history_depth=5,
            beta=0.5,
            damage_range_atol=1.0e-12,
        )

        current_f = g_current - x_current
        delta_x = np.column_stack(
            [history_x[1] - history_x[0], x_current - history_x[1]]
        )
        delta_f = np.column_stack(
            [history_f[1] - history_f[0], current_f - history_f[1]]
        )
        gamma = np.linalg.solve(delta_f, current_f)
        expected = x_current + 0.5 * current_f - (
            delta_x + 0.5 * delta_f
        ) @ gamma

        self.assertIsNotNone(proposal.candidate)
        assert proposal.candidate is not None
        np.testing.assert_allclose(proposal.candidate, expected, rtol=0.0, atol=1.0e-14)
        self.assertEqual(proposal.history_size, 2)
        self.assertEqual(proposal.qr_rank, 2)
        self.assertIsNone(proposal.fallback_reason)
        self.assertIsNone(proposal.restart_reason)
        self.assertIsNotNone(proposal.anderson_coefficient_l2)

    def test_insufficient_history_is_an_explicit_picard_fallback(self) -> None:
        proposal = stabilized._anderson_propose(
            (),
            (),
            np.asarray([0.2], dtype=np.float64),
            np.asarray([0.4], dtype=np.float64),
            history_depth=5,
            beta=0.5,
            damage_range_atol=1.0e-12,
        )

        self.assertIsNone(proposal.candidate)
        self.assertEqual(proposal.history_size, 0)
        self.assertEqual(proposal.qr_rank, 0)
        self.assertEqual(proposal.fallback_reason, "insufficient_history")
        self.assertIsNone(proposal.restart_reason)

    def test_rank_deficiency_and_raw_range_are_not_repaired(self) -> None:
        rank_deficient = stabilized._anderson_propose(
            (
                np.asarray([0.0, 0.0], dtype=np.float64),
                np.asarray([0.05, 0.05], dtype=np.float64),
            ),
            (
                np.asarray([0.2, 0.2], dtype=np.float64),
                np.asarray([0.3, 0.3], dtype=np.float64),
            ),
            np.asarray([0.1, 0.1], dtype=np.float64),
            np.asarray([0.5, 0.5], dtype=np.float64),
            history_depth=5,
            beta=0.5,
            damage_range_atol=1.0e-12,
        )
        self.assertIsNone(rank_deficient.candidate)
        self.assertEqual(rank_deficient.fallback_reason, "rank_deficient")
        self.assertEqual(rank_deficient.restart_reason, "rank_deficient")
        self.assertEqual(rank_deficient.qr_rank, 1)

        out_of_range = stabilized._anderson_propose(
            (np.asarray([0.2], dtype=np.float64),),
            (np.asarray([0.4], dtype=np.float64),),
            np.asarray([0.3], dtype=np.float64),
            np.asarray([0.8], dtype=np.float64),
            history_depth=5,
            beta=0.5,
            damage_range_atol=1.0e-12,
        )
        self.assertIsNone(out_of_range.candidate)
        self.assertEqual(out_of_range.fallback_reason, "out_of_range_candidate")
        self.assertEqual(out_of_range.restart_reason, "out_of_range_candidate")

    def test_nonfinite_residual_is_reported_without_clipping(self) -> None:
        with np.errstate(over="ignore", invalid="ignore"):
            proposal = stabilized._anderson_propose(
                (),
                (),
                np.asarray([1.0e308], dtype=np.float64),
                np.asarray([-1.0e308], dtype=np.float64),
                history_depth=5,
                beta=0.5,
                damage_range_atol=1.0e-12,
            )
        self.assertIsNone(proposal.candidate)
        self.assertEqual(proposal.fallback_reason, "nonfinite_residual")
        self.assertEqual(proposal.restart_reason, "nonfinite_residual")

    @staticmethod
    def _manufactured_history(
        delta_f: np.ndarray,
        current_f: np.ndarray,
        *,
        delta_x: np.ndarray | None = None,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """Construct histories whose final forward differences are prescribed."""

        differences = np.asarray(delta_f, dtype=np.float64)
        residual = np.asarray(current_f, dtype=np.float64)
        if differences.ndim != 2 or residual.ndim != 1:
            raise AssertionError("manufactured Anderson data has the wrong rank")
        if differences.shape[0] != residual.shape[0] or differences.shape[1] == 0:
            raise AssertionError("manufactured Anderson data has incompatible shapes")
        depth = differences.shape[1]
        if delta_x is None:
            state_differences = np.zeros_like(differences)
        else:
            state_differences = np.asarray(delta_x, dtype=np.float64)
            if state_differences.shape != differences.shape:
                raise AssertionError("manufactured state differences have the wrong shape")
        states = np.zeros((depth, residual.size), dtype=np.float64)
        residuals = np.zeros_like(states)
        residuals[0] = residual - np.sum(differences, axis=1)
        for index in range(1, depth):
            states[index] = states[index - 1] + state_differences[:, index - 1]
            residuals[index] = residuals[index - 1] + differences[:, index - 1]
        x_current = states[-1] + state_differences[:, -1]
        g_current = x_current + residual
        return (
            tuple(states),
            tuple(residuals),
            x_current,
            g_current,
        )

    def test_regularized_augmented_qr_matches_independent_lstsq(self) -> None:
        delta_f = np.asarray(
            [[1.0, 0.2], [0.0, 1.0], [0.5, -0.3]], dtype=np.float64
        )
        delta_x = np.asarray(
            [[0.1, -0.2], [0.3, 0.4], [-0.1, 0.2]], dtype=np.float64
        )
        current_f = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
        history_x, history_f, x_current, g_current = self._manufactured_history(
            delta_f, current_f, delta_x=delta_x
        )
        mu = 0.75
        proposal = stabilized._anderson_propose_regularized(
            history_x,
            history_f,
            x_current,
            g_current,
            history_depth=5,
            beta=0.5,
            regularization_mu=mu,
            damage_range_atol=1.0e-12,
        )

        current_residual_l2 = float(np.linalg.norm(current_f))
        regularization_lambda = mu * current_residual_l2**2
        augmented = np.vstack(
            (
                delta_f,
                np.sqrt(regularization_lambda) * np.eye(2, dtype=np.float64),
            )
        )
        rhs = np.concatenate((current_f, np.zeros(2, dtype=np.float64)))
        gamma, _, _, _ = np.linalg.lstsq(augmented, rhs, rcond=None)
        expected_candidate = x_current + 0.5 * current_f - (
            delta_x + 0.5 * delta_f
        ) @ gamma

        self.assertIsNone(proposal.fallback_reason)
        self.assertEqual(proposal.qr_rank, 2)
        self.assertIsNotNone(proposal.candidate)
        self.assertIsNotNone(proposal.anderson_current_residual_l2)
        self.assertIsNotNone(proposal.anderson_regularization_lambda)
        self.assertIsNotNone(proposal.anderson_coefficient_l2)
        self.assertIsNotNone(proposal.anderson_unregularized_ls_residual_l2)
        assert proposal.candidate is not None
        assert proposal.anderson_current_residual_l2 is not None
        assert proposal.anderson_regularization_lambda is not None
        assert proposal.anderson_coefficient_l2 is not None
        assert proposal.anderson_unregularized_ls_residual_l2 is not None
        np.testing.assert_allclose(
            proposal.candidate, expected_candidate, rtol=0.0, atol=2.0e-14
        )
        self.assertAlmostEqual(
            proposal.anderson_current_residual_l2, current_residual_l2
        )
        self.assertAlmostEqual(
            proposal.anderson_regularization_lambda,
            mu * proposal.anderson_current_residual_l2**2,
        )
        self.assertAlmostEqual(
            proposal.anderson_coefficient_l2, float(np.linalg.norm(gamma))
        )
        self.assertAlmostEqual(
            proposal.anderson_unregularized_ls_residual_l2,
            float(np.linalg.norm(current_f - delta_f @ gamma)),
        )

    def test_regularized_augmented_qr_handles_rank_deficiency_and_zero_residual(self) -> None:
        rank_deficient_delta = np.asarray(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64
        )
        zero_f = np.zeros(3, dtype=np.float64)
        history_x, history_f, x_current, g_current = self._manufactured_history(
            rank_deficient_delta, zero_f
        )
        rank_deficient = stabilized._anderson_propose_regularized(
            history_x,
            history_f,
            x_current,
            g_current,
            history_depth=5,
            beta=0.5,
            regularization_mu=1.0,
            damage_range_atol=0.0,
        )
        self.assertIsNone(rank_deficient.candidate)
        self.assertEqual(rank_deficient.fallback_reason, "rank_deficient")
        self.assertLess(rank_deficient.qr_rank, rank_deficient.history_size)
        self.assertEqual(rank_deficient.anderson_current_residual_l2, 0.0)
        self.assertEqual(rank_deficient.anderson_regularization_lambda, 0.0)

        full_delta = np.asarray(
            [[1.0, 0.1], [0.0, 1.0], [0.3, -0.2]], dtype=np.float64
        )
        full_history_x, full_history_f, full_x, full_g = self._manufactured_history(
            full_delta, zero_f
        )
        zero_residual = stabilized._anderson_propose_regularized(
            full_history_x,
            full_history_f,
            full_x,
            full_g,
            history_depth=5,
            beta=0.5,
            regularization_mu=2.0,
            damage_range_atol=0.0,
        )
        self.assertIsNone(zero_residual.fallback_reason)
        self.assertEqual(zero_residual.anderson_regularization_lambda, 0.0)
        self.assertEqual(zero_residual.anderson_current_residual_l2, 0.0)
        self.assertEqual(zero_residual.anderson_coefficient_l2, 0.0)
        self.assertEqual(zero_residual.anderson_unregularized_ls_residual_l2, 0.0)

    def test_regularized_augmented_qr_resolves_small_rank_deficient_residual(self) -> None:
        delta_f = np.asarray(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64
        )
        current_f = np.asarray([1.0e-8, -2.0e-8, 3.0e-8], dtype=np.float64)
        history_x, history_f, x_current, g_current = self._manufactured_history(
            delta_f, current_f
        )
        mu = 2.0
        proposal = stabilized._anderson_propose_regularized(
            history_x,
            history_f,
            x_current,
            g_current,
            history_depth=5,
            beta=0.5,
            regularization_mu=mu,
            damage_range_atol=0.0,
        )
        self.assertIsNone(proposal.fallback_reason)
        self.assertEqual(proposal.qr_rank, 2)
        self.assertIsNotNone(proposal.anderson_coefficient_l2)
        assert proposal.anderson_coefficient_l2 is not None
        actual_delta_f = np.column_stack(
            [
                history_f[index + 1] - history_f[index]
                for index in range(len(history_f) - 1)
            ]
            + [g_current - x_current - history_f[-1]]
        )
        actual_current_f = g_current - x_current
        augmented = np.vstack(
            (
                actual_delta_f,
                np.sqrt(mu * np.linalg.norm(actual_current_f) ** 2)
                * np.eye(2, dtype=np.float64),
            )
        )
        rhs = np.concatenate((actual_current_f, np.zeros(2, dtype=np.float64)))
        gamma, _, _, _ = np.linalg.lstsq(augmented, rhs, rcond=None)
        self.assertTrue(
            np.isclose(
                proposal.anderson_coefficient_l2,
                float(np.linalg.norm(gamma)),
                rtol=1.0e-8,
                atol=1.0e-20,
            )
        )

    def test_regularization_suppresses_coefficients_without_normal_equations(self) -> None:
        delta_f = np.asarray(
            [[1.0, 1.0], [1.0, 1.0 + 1.0e-6], [0.0, 0.1]], dtype=np.float64
        )
        current_f = np.asarray([1.0, 1.0, 0.2], dtype=np.float64)
        history_x, history_f, x_current, g_current = self._manufactured_history(
            delta_f, current_f
        )
        unregularized_gamma, _, _, _ = np.linalg.lstsq(
            delta_f, current_f, rcond=None
        )
        proposal = stabilized._anderson_propose_regularized(
            history_x,
            history_f,
            x_current,
            g_current,
            history_depth=5,
            beta=0.5,
            regularization_mu=10.0,
            damage_range_atol=0.0,
        )
        self.assertIsNotNone(proposal.anderson_coefficient_l2)
        assert proposal.anderson_coefficient_l2 is not None
        self.assertLess(
            proposal.anderson_coefficient_l2,
            float(np.linalg.norm(unregularized_gamma)),
        )
        source = inspect.getsource(stabilized._anderson_propose_regularized)
        self.assertIn("_pivoted_qr", source)
        self.assertNotIn("np.linalg.solve", source)
        self.assertNotIn("delta_f.T @ delta_f", source)

    def test_anderson_merit_uses_candidate_residual_not_rho_used(self) -> None:
        diagnostics: Any = SimpleNamespace(
            e_u_material=0.0,
            e_u_union=0.0,
            e_damage=0.0,
            e_history=0.0,
            e_reaction=0.0,
            rho_used_inf=1.0e300,
            rho_next_inf=0.0,
            force_error=0.0,
            moment_error=0.0,
            displacement_scale=1.0,
            damage_scale=1.0,
            history_scale=1.0,
            reaction_scale=1.0,
            force_scale=1.0,
            moment_scale=1.0,
        )
        config = stabilized.StabilizedEngineeringGripConfig.smoke()
        accelerated = stabilized._anderson_candidate_merit(
            diagnostics,
            candidate_fp_error=0.1,
            candidate_fp_scale=1.0,
            candidate_damage_residual_scale=1.0,
            config=config,
        )
        diagnostics.rho_used_inf = 0.0
        self.assertEqual(
            accelerated,
            stabilized._anderson_candidate_merit(
                diagnostics,
                candidate_fp_error=0.1,
                candidate_fp_scale=1.0,
                candidate_damage_residual_scale=1.0,
                config=config,
            ),
        )

    def test_anderson_config_is_explicit_and_round_trips(self) -> None:
        base = engineering.EngineeringGripConfig.smoke()
        config = stabilized.StabilizedEngineeringGripConfig(
            base=base,
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            relaxation_policy="anderson",
            anderson_history_depth=5,
            anderson_beta=0.5,
        )

        self.assertEqual(
            config.payload(),
            {
                "relaxation_policy": "anderson",
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": 1.0e-2,
                "max_backtracks": 0,
                "anderson_history_depth": 5,
                "anderson_beta": 0.5,
            },
        )
        restored = stabilized.StabilizedEngineeringGripConfig.from_payload(
            config.payload(), base
        )
        self.assertEqual(restored.payload(), config.payload())

        with self.assertRaisesRegex(ValueError, "history depth must equal"):
            replace(config, anderson_history_depth=4)
        with self.assertRaisesRegex(ValueError, "requires omega_initial"):
            replace(config, omega_min=0.25)
        with self.assertRaisesRegex(ValueError, "warning acceptance"):
            replace(
                config,
                acceptance_mode=stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
            )

    def test_residual_anderson_config_has_distinct_merit_identity(self) -> None:
        base = engineering.EngineeringGripConfig.smoke()
        legacy = stabilized.StabilizedEngineeringGripConfig(
            base=base,
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            relaxation_policy="anderson",
            anderson_history_depth=5,
            anderson_beta=0.5,
        )
        config = replace(legacy, relaxation_policy="anderson_residual")
        payload = config.payload()
        self.assertEqual(
            payload["anderson_merit_identity"],
            stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY,
        )
        self.assertEqual(
            stabilized.StabilizedEngineeringGripConfig.from_payload(
                payload, base
            ).payload(),
            payload,
        )
        self.assertEqual(
            legacy.payload().get("anderson_merit_identity"),
            None,
        )
        self.assertNotEqual(
            stabilized._anderson_merit_identity("anderson"),
            stabilized._anderson_merit_identity("anderson_residual"),
        )

        forged = dict(payload)
        forged["anderson_merit_identity"] = (
            stabilized.STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY
        )
        with self.assertRaisesRegex(ValueError, "merit identity is invalid"):
            stabilized.StabilizedEngineeringGripConfig.from_payload(forged, base)

    def test_backtracking_config_requires_and_serializes_only_new_fields(self) -> None:
        base = engineering.EngineeringGripConfig.smoke()
        config = stabilized.StabilizedEngineeringGripConfig(
            base=base,
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=stabilized.STABILIZED_ANDERSON_MAX_BACKTRACKS,
            relaxation_policy="anderson_residual_backtracking",
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
            anderson_alpha_reduction=stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
        )
        payload = config.payload()
        self.assertEqual(
            payload["anderson_merit_identity"],
            stabilized.STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY,
        )
        self.assertEqual(
            payload["anderson_alpha_reduction"],
            stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
        )
        self.assertEqual(
            stabilized.StabilizedEngineeringGripConfig.from_payload(
                payload, base
            ).payload(),
            payload,
        )

        historical = dict(payload)
        historical["relaxation_policy"] = "anderson_residual"
        historical.pop("anderson_alpha_reduction")
        historical["max_backtracks"] = 0
        historical["anderson_merit_identity"] = (
            stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY
        )
        restored_historical = stabilized.StabilizedEngineeringGripConfig.from_payload(
            historical, base
        )
        self.assertNotIn("anderson_alpha_reduction", restored_historical.payload())

        missing = dict(payload)
        missing.pop("anderson_alpha_reduction")
        with self.assertRaisesRegex(ValueError, "alpha_reduction"):
            stabilized.StabilizedEngineeringGripConfig.from_payload(missing, base)
        forged_old = dict(historical)
        forged_old["anderson_alpha_reduction"] = 0.5
        with self.assertRaisesRegex(ValueError, "only valid"):
            stabilized.StabilizedEngineeringGripConfig.from_payload(forged_old, base)

    def test_regularized_backtracking_config_requires_fixed_positive_mu(self) -> None:
        base = engineering.EngineeringGripConfig.smoke()
        config = stabilized.StabilizedEngineeringGripConfig(
            base=base,
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=stabilized.STABILIZED_ANDERSON_MAX_BACKTRACKS,
            relaxation_policy=(
                "anderson_residual_regularized_backtracking"
            ),
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
            anderson_alpha_reduction=stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
            anderson_regularization_mu=1.25,
        )
        payload = config.payload()
        self.assertEqual(
            payload,
            {
                "relaxation_policy": (
                    "anderson_residual_regularized_backtracking"
                ),
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": 1.0e-2,
                 "max_backtracks": stabilized.STABILIZED_ANDERSON_MAX_BACKTRACKS,
                "anderson_history_depth": 5,
                "anderson_beta": 0.5,
                "anderson_merit_identity": (
                    stabilized.STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY
                ),
                "anderson_alpha_reduction": 0.5,
                "anderson_regularization_mu": 1.25,
            },
        )
        self.assertEqual(
            stabilized.StabilizedEngineeringGripConfig.from_payload(
                payload, base
            ).payload(),
            payload,
        )
        for invalid in (None, 0.0, -1.0, np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(config, anderson_regularization_mu=invalid)
        with self.assertRaisesRegex(ValueError, "only valid"):
            replace(
                config,
                relaxation_policy="anderson_residual_backtracking",
            )
        forged = dict(payload)
        forged.pop("anderson_regularization_mu")
        with self.assertRaisesRegex(ValueError, "regularization_mu"):
            stabilized.StabilizedEngineeringGripConfig.from_payload(forged, base)

    def test_regularized_v5_summary_fields_and_lambda_provenance_are_strict(self) -> None:
        policy = "anderson_residual_regularized_backtracking"
        summary = stabilized._v5_initial_solver_summary(
            policy,
            anderson_regularization_mu=2.0,
        )
        summary.update(
            {
                "summary_kind": "accepted",
                "state_index": 1,
                "load_start": 0.0,
                "load_target": 1.0e-4,
                "increment": 1.0e-4,
                "stagger_iterations": 1,
                "acceptance_status": stabilized.STABILIZED_ACCEPTANCE_STATUS,
                "accepted": True,
                "sweep_index": 0,
                "target_u_bar": 1.0e-4,
                "complete_sweep": True,
                "passed": True,
                "failure_stage": "",
                "failure_reason": "",
                "gate_passes": {
                    name: True for name in stabilized._V5_SOLVER_SUMMARY_GATE_NAMES
                },
                "reference_merit": 1.0,
                "merit": 0.5,
                "sufficient_decrease_limit": 0.5,
                "fixed_point_error": 0.1,
                "fixed_point_scale": 1.0,
                "anderson_selection": "accelerated",
                "anderson_fallback_reason": None,
                "anderson_restart_reason": None,
                "candidate_fp_error": 0.1,
                "candidate_fp_scale": 1.0,
                "candidate_damage_residual_scale": 1.0,
                "trial_alpha": 1.0,
                "backtrack_count": 0,
                "anderson_regularization_mu": 2.0,
                "anderson_current_residual_l2": 2.0,
                "anderson_regularization_lambda": 8.0,
                "anderson_unregularized_ls_residual_l2": 0.1,
                "anderson_coefficient_l2": 0.5,
            }
        )
        for field in stabilized._V5_SOLVER_SUMMARY_BOOL_FIELDS:
            summary[field] = True
        stabilized._v5_validate_solver_summary(
            summary,
            "valid regularized summary",
            relaxation_policy=policy,
            anderson_regularization_mu=2.0,
        )
        for label, update, pattern in (
            (
                "wrong lambda",
                {"anderson_regularization_lambda": 7.0},
                "current residual",
            ),
            (
                "wrong current residual",
                {"anderson_current_residual_l2": 3.0},
                "current residual",
            ),
            (
                "wrong configured mu",
                {"anderson_regularization_mu": 1.0},
                "does not match configuration",
            ),
            (
                "missing strict field",
                {"anderson_coefficient_l2": None},
                "selected regularized row lacks",
            ),
        ):
            with self.subTest(label=label):
                forged = dict(summary)
                forged.update(update)
                if label == "missing strict field":
                    forged.pop("anderson_coefficient_l2")
                with self.assertRaises(ValueError):
                    stabilized._v5_validate_solver_summary(
                        forged,
                        f"forged regularized summary {label}",
                        relaxation_policy=policy,
                        anderson_regularization_mu=2.0,
                    )

    def test_residual_merit_uses_own_normalizer_and_map_only_component(self) -> None:
        config = stabilized.StabilizedEngineeringGripConfig.smoke()
        reference = stabilized._anderson_residual_reference_merit_components(
            np.asarray([0.2, 0.4], dtype=np.float64),
            np.asarray([0.3, 0.4], dtype=np.float64),
            config,
        )
        candidate = stabilized._anderson_residual_candidate_merit_components(
            0.1,
            0.8,
            config,
        )

        self.assertEqual(set(reference), {"committed_history_map_residual"})
        self.assertEqual(set(candidate), {"committed_history_map_residual"})
        self.assertAlmostEqual(
            reference["committed_history_map_residual"],
            stabilized._normalized(
                0.1,
                config.base.damage,
                0.4,
                "test reference",
            ),
        )
        self.assertAlmostEqual(
            candidate["committed_history_map_residual"],
            stabilized._normalized(
                0.1,
                config.base.damage,
                0.8,
                "test candidate",
            ),
        )
        self.assertNotEqual(reference, candidate)

    def test_backtracking_alpha_schedule_and_threshold_are_exact(self) -> None:
        config = replace(
            stabilized.StabilizedEngineeringGripConfig.smoke(),
            relaxation_policy="anderson_residual_backtracking",
            omega_initial=0.5,
            omega_min=0.5,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=4,
            anderson_history_depth=5,
            anderson_beta=0.5,
            anderson_alpha_reduction=0.5,
        )
        alphas = [
            stabilized._anderson_residual_backtracking_alpha(config, index)
            for index in range(config.max_backtracks + 1)
        ]
        self.assertEqual(alphas, [1.0, 0.5, 0.25, 0.125, 0.0625])
        with self.assertRaisesRegex(ValueError, "outside its budget"):
            stabilized._anderson_residual_backtracking_alpha(config, 5)
        self.assertEqual(
            stabilized._anderson_residual_sufficient_decrease_limit(
                2.0,
                config,
                trial_alpha=0.25,
            ),
            (1.0 - 1.0e-2 * 0.5 * 0.25) * 2.0,
        )

    def test_residual_hard_pass_rejects_each_protected_gate(self) -> None:
        gate_names = (
            "finite_pass",
            "mechanics_before_pass",
            "mechanics_after_pass",
            "force_pass",
            "moment_pass",
            "damage_range_pass",
            "healing_pass",
            "history_monotonic_pass",
        )
        diagnostics = SimpleNamespace(**{name: True for name in gate_names})
        self.assertTrue(
            stabilized._anderson_residual_hard_pass(cast(Any, diagnostics))
        )
        for name in gate_names:
            with self.subTest(gate=name):
                failed = SimpleNamespace(
                    **{item: item != name for item in gate_names}
                )
                self.assertFalse(
                    stabilized._anderson_residual_hard_pass(cast(Any, failed))
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
        if policy in ("anderson", "anderson_residual"):
            return replace(
                self.config,
                base=base,
                relaxation_policy=policy,
                omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
                omega_min=stabilized.STABILIZED_ANDERSON_BETA,
                omega_reduction=None,
                sufficient_decrease=1.0e-2,
                max_backtracks=0,
                anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
                anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
            )
        if policy == "anderson_residual_backtracking":
            return replace(
                self.config,
                base=base,
                relaxation_policy=policy,
                omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
                omega_min=stabilized.STABILIZED_ANDERSON_BETA,
                omega_reduction=None,
                sufficient_decrease=1.0e-2,
                max_backtracks=stabilized.STABILIZED_ANDERSON_MAX_BACKTRACKS,
                anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
                anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
                anderson_alpha_reduction=stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
            )
        if policy == "anderson_residual_regularized_backtracking":
            return replace(
                self.config,
                base=base,
                relaxation_policy=policy,
                omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
                omega_min=stabilized.STABILIZED_ANDERSON_BETA,
                omega_reduction=None,
                sufficient_decrease=1.0e-2,
                max_backtracks=stabilized.STABILIZED_ANDERSON_MAX_BACKTRACKS,
                anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
                anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
                anderson_alpha_reduction=stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
                anderson_regularization_mu=1.25,
            )
        raise AssertionError(f"unknown test relaxation policy: {policy}")

    def test_residual_force_moment_attribution_uses_stage_ratios(self) -> None:
        diagnostic = self.result.attempts[0].sweeps[0].base_diagnostics

        def mechanics(*, reaction_scale: float, force_error: float, moment_error: float) -> Any:
            return SimpleNamespace(
                displacement=np.zeros_like(diagnostic.q_before),
                residual=np.zeros_like(diagnostic.q_before),
                total_force=np.asarray([force_error, 0.0], dtype=np.float64),
                total_moment=moment_error,
                cg_info=0,
                free_residual_norm=0.0,
                cg_convergence_limit=1.0,
                outer_reaction=reaction_scale,
                bottom_reaction=0.0,
                anchor_reaction=0.0,
            )

        before = mechanics(
            reaction_scale=0.25,
            force_error=2.0e-8,
            moment_error=2.0e-8,
        )
        after = mechanics(
            reaction_scale=1.0e8,
            force_error=0.5,
            moment_error=0.5,
        )
        details, _ = stabilized._anderson_residual_hard_gate_attribution(
            diagnostic,
            self.config,
            mechanics_before=before,
            mechanics_after=after,
            geometry_length=1.0,
        )
        for name in ("force", "moment"):
            with self.subTest(gate=name):
                aggregate = details[name]
                stages = aggregate["stage_metrics"]
                self.assertEqual(tuple(stages), ("before", "after"))
                self.assertEqual(
                    aggregate["normalized"],
                    max(
                        stages[stage]["normalized"]
                        for stage in ("before", "after")
                    ),
                )
                self.assertFalse(stages["before"]["passed"])
                self.assertTrue(stages["after"]["passed"])
                self.assertEqual(stages["before"]["scale"], 1.0)
                self.assertEqual(stages["after"]["scale"], 1.0e8)
                self.assertFalse(aggregate["passed"])
                self.assertAlmostEqual(
                    aggregate["raw"] / aggregate["limit"],
                    aggregate["normalized"],
                )
                self.assertEqual(aggregate["raw"], stages["before"]["error"])
                self.assertEqual(aggregate["scale"], stages["before"]["scale"])
                self.assertEqual(aggregate["limit"], stages["before"]["limit"])

    @contextmanager
    def _valid_schedule_artifact(
        self,
        directory: Path,
        targets: Sequence[float] = (5.0e-5, 1.0e-4),
    ) -> Any:
        """Create a temporary canonical-shaped source/schedule pair.

        The reviewed constants are patched only while this valid fixture is
        built and exercised; no constructor bypass or forged provenance field
        is used.
        """

        target_array = np.asarray(targets, dtype=np.dtype("<f8"))
        source = directory / (
            "engineering_grip_anderson_u5p9875e3_dumin6p25e6_s144.npz"
        )
        source_metadata = {
            "schema": stabilized.STABILIZED_ARCHIVE_SCHEMA,
            "kind": "completed",
            "completion_status": stabilized.STABILIZED_COMPLETION_STATUS,
            "stabilized_config": {"relaxation_policy": "anderson"},
        }
        np.savez_compressed(
            str(source),
            archive_schema=np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA),
            metadata_json=np.asarray(
                json.dumps(source_metadata, sort_keys=True, allow_nan=False),
                dtype=np.str_,
            ),
            state_u_bar=np.asarray(
                [0.0, *target_array], dtype=np.dtype("<f8")
            ),
        )
        output = directory / "matched-load-schedule.npz"
        source_sha256 = stabilized._archive_sha256(source)
        target_sha256 = stabilized._schedule_target_sha256(target_array)
        with (
            mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256",
                source_sha256,
            ),
            mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_TARGET_SHA256",
                target_sha256,
            ),
            mock.patch.object(
                stabilized,
                "STABILIZED_SCHEDULE_TARGET_COUNT",
                len(target_array),
            ),
        ):
            yield stabilized.extract_stabilized_load_schedule(source, output)

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

    def test_scheduled_controller_success_archive_round_trip_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stabilized-schedule-success-") as name:
            directory = Path(name)
            with self._valid_schedule_artifact(directory) as schedule:
                base = replace(
                    self.config.base,
                    load_final=1.0e-4,
                    max_stagger_iterations=2,
                )
                config = replace(
                    self.config,
                    base=base,
                    relaxation_policy="anderson_residual",
                    omega_initial=0.5,
                    omega_min=0.5,
                    omega_reduction=None,
                    sufficient_decrease=1.0e-2,
                    max_backtracks=0,
                    anderson_history_depth=5,
                    anderson_beta=0.5,
                    load_schedule=schedule,
                )
                with mock.patch.object(
                    stabilized, "_fixed_point_pass", return_value=True
                ):
                    result = stabilized.run_stabilized_engineering_grip(
                        self.model, config
                    )

                self.assertEqual(
                    [state.u_bar for state in result.states],
                    [0.0, 5.0e-5, 1.0e-4],
                )
                self.assertEqual(len(result.attempts), 2)
                self.assertEqual([attempt.accepted for attempt in result.attempts], [True, True])
                self.assertEqual(
                    [attempt.cutback_count for attempt in result.attempts], [0, 0]
                )
                self.assertEqual(
                    [attempt.load_target for attempt in result.attempts],
                    [5.0e-5, 1.0e-4],
                )

                archive, payload = self._write_archive(directory, result=result)
                self.assertIn(
                    "schedule_target_sha256", payload
                )
                metadata = json.loads(str(payload["metadata_json"]))
                self.assertEqual(
                    metadata["load_schedule"], schedule.payload()
                )
                replay = stabilized.replay_stabilized_engineering_grip_archive(
                    archive
                )
                self.assertTrue(replay["schedule_completed"])
                self.assertEqual(replay["schedule_target_count"], 2)
                self.assertIsNone(replay["schedule_next_target"])
                self.assertEqual(replay["load_schedule"], schedule.payload())

    def test_scheduled_controller_failure_rolls_back_and_binds_rejected_next_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="stabilized-schedule-failure-") as name:
            directory = Path(name)
            with self._valid_schedule_artifact(directory) as schedule:
                config = replace(
                    self.config,
                    base=replace(
                        self.config.base,
                        load_final=1.0e-4,
                        max_stagger_iterations=1,
                    ),
                    relaxation_policy="anderson_residual",
                    omega_initial=0.5,
                    omega_min=0.5,
                    omega_reduction=None,
                    sufficient_decrease=1.0e-2,
                    max_backtracks=0,
                    anderson_history_depth=5,
                    anderson_beta=0.5,
                    load_schedule=schedule,
                )
                with mock.patch.object(
                    stabilized,
                    "_fixed_point_pass",
                    side_effect=[True, False],
                ), self.assertRaises(
                    stabilized.StabilizedEngineeringGripContinuationFailure
                ) as context:
                    stabilized.run_stabilized_engineering_grip(self.model, config)

                failure = context.exception
                self.assertEqual(len(failure.attempts), 2)
                self.assertTrue(failure.attempts[0].accepted)
                self.assertFalse(failure.attempts[1].accepted)
                self.assertEqual(
                    [attempt.cutback_count for attempt in failure.attempts], [0, 0]
                )
                self.assertEqual(
                    [attempt.load_target for attempt in failure.attempts],
                    [5.0e-5, 1.0e-4],
                )
                self.assertEqual(failure.last_committed_state.u_bar, 5.0e-5)
                rejected = failure.attempts[-1]
                self.assertEqual(
                    rejected.committed_before_digest,
                    rejected.committed_after_digest,
                )

                archive = directory / "schedule.failure.npz"
                stabilized.write_stabilized_engineering_grip_failure_archive(
                    archive,
                    failure,
                    self.model,
                    config,
                    source_metadata=self.source_metadata,
                )
                payload = stabilized.load_stabilized_engineering_grip_archive(archive)
                failure_summary = json.loads(str(payload["failure_summary_json"]))
                self.assertEqual(failure_summary["cutback_count"], 0)
                self.assertEqual(failure_summary["load_target"], 1.0e-4)
                replay = stabilized.replay_stabilized_engineering_grip_archive(
                    archive
                )
                self.assertFalse(replay["schedule_completed"])
                self.assertEqual(replay["schedule_target_index"], 1)
                self.assertEqual(replay["schedule_next_target"], 1.0e-4)

                forged_sequence = self._copy_payload(payload)
                forged_sequence["state_u_bar"][1] = 7.5e-5
                forged_sequence_path = self._write_payload(
                    directory, "forged-schedule-sequence.npz", forged_sequence
                )
                with self.assertRaisesRegex(
                    ValueError, "state digests|zero-plus-schedule prefix"
                ):
                    stabilized.replay_stabilized_engineering_grip_archive(
                        forged_sequence_path
                    )

                forged_failure = self._copy_payload(payload)
                forged_failure_summary = json.loads(
                    str(forged_failure["failure_summary_json"])
                )
                forged_failure_summary["load_target"] = 1.25e-4
                forged_failure["failure_summary_json"] = np.asarray(
                    json.dumps(
                        forged_failure_summary,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    dtype=np.str_,
                )
                forged_failure_path = self._write_payload(
                    directory, "forged-schedule-next-target.npz", forged_failure
                )
                with self.assertRaisesRegex(
                    ValueError, "load interval|next target binding"
                ):
                    stabilized.replay_stabilized_engineering_grip_archive(
                        forged_failure_path
                    )

                forged_header = self._copy_payload(payload)
                forged_header["schedule_target_sha256"] = np.asarray(
                    "0" * 64, dtype=np.str_
                )
                forged_header_path = self._write_payload(
                    directory, "forged-schedule-header.npz", forged_header
                )
                with self.assertRaisesRegex(ValueError, "schedule_target_sha256"):
                    stabilized.replay_stabilized_engineering_grip_archive(
                        forged_header_path
                    )

    def test_selected_mechanics_cache_matches_uncached_both_policies_and_cutback(self) -> None:
        for policy in (
            "fixed",
            "sufficient_decrease",
            "anderson",
            "anderson_residual",
        ):
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

    def test_backtracking_fallback_archive_requires_new_summary_fields(self) -> None:
        historical_payload = self.result.attempts[0].diagnostics.payload()
        self.assertNotIn("trial_alpha", historical_payload)
        self.assertTrue(
            all(
                "trial_alpha" not in trial
                for trial in historical_payload["relaxation_trials"]
            )
        )
        result, _ = self._run_cache_case(
            "anderson_residual_backtracking", reuse=False
        )
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-backtracking-") as name:
            directory = Path(name)
            archive, payload = self._write_cache_archive(
                directory, "backtracking.npz", result
            )
            metadata = json.loads(str(payload["metadata_json"]))
            stabilized_config = metadata["stabilized_config"]
            self.assertEqual(
                stabilized_config["relaxation_policy"],
                "anderson_residual_backtracking",
            )
            self.assertEqual(
                stabilized_config["anderson_alpha_reduction"],
                stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
            )
            summaries = [
                json.loads(str(item)) for item in payload["solver_summary_json"]
            ]
            self.assertIn("trial_alpha", summaries[0])
            self.assertIsNone(summaries[0]["trial_alpha"])
            self.assertTrue(
                all(
                    "trial_alpha" in summary
                    for summary in summaries
                    if summary["summary_kind"] == "accepted"
                )
            )
            replay = stabilized.replay_stabilized_engineering_grip_archive(archive)
            self.assertEqual(
                replay["selection_merit_identity"],
                stabilized.STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY,
            )
            for label, update, pattern in (
                (
                    "missing alpha",
                    lambda summary: summary.pop("trial_alpha"),
                    "summary fields differ",
                ),
                (
                    "forged alpha",
                    lambda summary: summary.__setitem__("trial_alpha", 0.3),
                    "configured schedule",
                ),
                (
                    "on-schedule alpha bound to the wrong backtrack count",
                    lambda summary: summary.__setitem__("trial_alpha", 0.25),
                    "backtrack_count",
                ),
            ):
                with self.subTest(label=label):
                    forged = self._copy_payload(payload)
                    records = [
                        json.loads(str(item))
                        for item in forged["solver_summary_json"]
                    ]
                    update(records[1])
                    forged["solver_summary_json"] = np.asarray(
                        [
                            json.dumps(record, sort_keys=True, allow_nan=False)
                            for record in records
                        ],
                        dtype=np.str_,
                    )
                    forged_path = self._write_payload(
                        directory, f"forged-{label}.npz", forged
                    )
                    with self.assertRaisesRegex(ValueError, pattern):
                        stabilized.replay_stabilized_engineering_grip_archive(
                            forged_path
                        )

    def test_residual_threshold_selects_acceleration_and_unmerited_fallback(self) -> None:
        for candidate_merit, expected_selection in (
            (0.995, "accelerated"),
            (0.996, "picard_fallback"),
        ):
            with self.subTest(candidate_merit=candidate_merit):
                config = self._cache_config("anderson_residual")
                events: list[dict[str, Any]] = []
                with (
                    mock.patch.object(
                        stabilized, "_fixed_point_pass", return_value=True
                    ),
                    mock.patch.object(
                        stabilized,
                        "_anderson_residual_reference_merit",
                        return_value=1.0,
                    ),
                    mock.patch.object(
                        stabilized,
                        "_anderson_residual_candidate_merit",
                        return_value=candidate_merit,
                    ),
                    mock.patch.object(
                        stabilized,
                        "_validate_anderson_candidate_map",
                        return_value=(
                            np.zeros(self.model.n_material, dtype=np.float64),
                            1.0,
                        ),
                    ),
                ):
                    step, attempt = stabilized._run_stabilized_attempt(
                        self.model,
                        config,
                        self.result.initial_state,
                        1.0e-4,
                        0,
                        1,
                        None,
                        anderson_diagnostic_callback=lambda event: events.append(
                            dict(event)
                        ),
                    )

                self.assertIsNotNone(step)
                final = attempt.diagnostics
                self.assertEqual(final.anderson_selection, expected_selection)
                if expected_selection == "accelerated":
                    self.assertEqual(final.merit, candidate_merit)
                    limit = final.sufficient_decrease_limit
                    self.assertIsNotNone(limit)
                    assert limit is not None
                    self.assertAlmostEqual(limit, 0.995)
                    self.assertIsNone(final.omega)
                else:
                    self.assertIsNone(final.merit)
                    self.assertIsNone(final.sufficient_decrease_limit)
                    self.assertEqual(final.omega, 0.5)
                    self.assertEqual(
                        final.anderson_fallback_reason, "insufficient_decrease"
                    )
                    self.assertEqual(
                        final.anderson_restart_reason, "insufficient_decrease"
                    )
                    fallback_records = [
                        event
                        for event in events
                        if event.get("record_type") == "proposal"
                        and event.get("candidate_kind") == "picard_fallback"
                    ]
                    self.assertTrue(fallback_records)
                    self.assertTrue(
                        all(
                            event["candidate_merit"] is None
                            and event["candidate_map_status"] == "not_requested"
                            and event["selection_merit_identity"]
                            == stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY
                            for event in fallback_records
                        )
                    )
                accelerated_records = [
                    event
                    for event in events
                    if event.get("record_type") == "proposal"
                    and event.get("candidate_kind") == "anderson_accelerated"
                ]
                self.assertTrue(accelerated_records)
                self.assertTrue(
                    all(
                        event["candidate_map_status"] == "completed"
                        and event["evaluation_counts"]["candidate_map"] == 1
                        and event["evaluation_counts"]["mechanics_after"] == 1
                        and event["hard_gate_attribution_schema"]
                        == stabilized.ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA
                        for event in accelerated_records
                    )
                )

    def _backtracking_proposal(self) -> stabilized._AndersonProposal:
        return stabilized._AndersonProposal(
            candidate=np.full(self.model.n_material, 0.2, dtype=np.float64),
            history_size=1,
            qr_rank=1,
            fallback_reason=None,
            restart_reason=None,
            anderson_coefficient_l2=1.0,
        )

    def _regularized_backtracking_proposal(self) -> stabilized._AndersonProposal:
        return stabilized._AndersonProposal(
            candidate=np.full(self.model.n_material, 0.2, dtype=np.float64),
            history_size=1,
            qr_rank=1,
            fallback_reason=None,
            restart_reason=None,
            anderson_coefficient_l2=1.0,
            anderson_regularization_mu=1.25,
            anderson_current_residual_l2=0.8,
            anderson_regularization_lambda=0.8,
            anderson_unregularized_ls_residual_l2=0.2,
        )

    def test_regularized_backtracking_matches_controller_alpha_gates_and_solve_counts(
        self,
    ) -> None:
        def run_case(
            policy: str,
            proposal: stabilized._AndersonProposal,
        ) -> tuple[Any, list[dict[str, Any]], int, int]:
            config = self._cache_config(policy)
            events: list[dict[str, Any]] = []
            proposer = (
                "_anderson_propose_regularized"
                if policy == "anderson_residual_regularized_backtracking"
                else "_anderson_propose"
            )
            with (
                mock.patch.object(stabilized, proposer, return_value=proposal),
                mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
                mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
                mock.patch.object(stabilized, "_coupled_pass", return_value=True),
                mock.patch.object(
                    stabilized, "_anderson_residual_hard_pass", return_value=True
                ),
                mock.patch.object(
                    stabilized,
                    "_anderson_residual_reference_merit",
                    return_value=1.0,
                ),
                mock.patch.object(
                    stabilized,
                    "_anderson_residual_candidate_merit",
                    side_effect=[0.996, 0.996],
                ),
                mock.patch.object(
                    stabilized,
                    "_validate_anderson_candidate_map",
                    return_value=(
                        np.zeros(self.model.n_material, dtype=np.float64),
                        1.0,
                    ),
                ) as candidate_map,
                mock.patch.object(
                    engineering,
                    "_solve_direct_mechanics",
                    wraps=engineering._solve_direct_mechanics,
                ) as mechanics_solve,
            ):
                step, attempt = stabilized._run_stabilized_attempt(
                    self.model,
                    config,
                    self.result.initial_state,
                    1.0e-4,
                    0,
                    1,
                    None,
                    anderson_diagnostic_callback=lambda event: events.append(dict(event)),
                )
            self.assertIsNotNone(step)
            return attempt, events, candidate_map.call_count, mechanics_solve.call_count

        historical, historical_events, historical_maps, historical_solves = run_case(
            "anderson_residual_backtracking", self._backtracking_proposal()
        )
        regularized, regularized_events, regularized_maps, regularized_solves = run_case(
            "anderson_residual_regularized_backtracking",
            self._regularized_backtracking_proposal(),
        )

        self.assertEqual(historical_maps, 2)
        self.assertEqual(regularized_maps, historical_maps)
        self.assertEqual(regularized_solves, historical_solves)
        self.assertEqual(historical_solves, 3)
        for attempt in (historical, regularized):
            diagnostic = attempt.diagnostics
            self.assertEqual(diagnostic.anderson_selection, "accelerated")
            self.assertEqual(diagnostic.trial_alpha, 0.5)
            self.assertEqual(diagnostic.backtrack_count, 1)
            self.assertEqual(
                [trial.trial_alpha for trial in diagnostic.trials], [1.0, 0.5]
            )
            self.assertFalse(diagnostic.trials[0].selected)
            self.assertEqual(diagnostic.trials[0].failure_stage, "globalization")
            self.assertTrue(diagnostic.trials[1].selected)
        historical_trial_records = [
            event
            for event in historical_events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        regularized_trial_records = [
            event
            for event in regularized_events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        self.assertEqual(
            [event["trial_alpha"] for event in historical_trial_records], [1.0, 0.5]
        )
        self.assertEqual(
            [event["trial_alpha"] for event in regularized_trial_records], [1.0, 0.5]
        )
        self.assertEqual(
            [event["hard_pass"] for event in historical_trial_records],
            [event["hard_pass"] for event in regularized_trial_records],
        )
        self.assertEqual(
            [event["candidate_map_status"] for event in historical_trial_records],
            [event["candidate_map_status"] for event in regularized_trial_records],
        )
        self.assertEqual(
            [event["failure_stage"] for event in historical_trial_records],
            [event["failure_stage"] for event in regularized_trial_records],
        )
        self.assertTrue(
            all(
                event["anderson_regularization_mu"] == 1.25
                and event["anderson_current_residual_l2"] == 0.8
                and event["anderson_regularization_lambda"] == 0.8
                and event["anderson_coefficient_l2"] == 1.0
                for event in regularized_trial_records
            )
        )

    def test_backtracking_first_passing_alpha_is_selected_and_reported(self) -> None:
        config = self._cache_config("anderson_residual_backtracking")
        events: list[dict[str, Any]] = []
        with (
            mock.patch.object(
                stabilized, "_anderson_propose", return_value=self._backtracking_proposal()
            ),
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
            mock.patch.object(stabilized, "_coupled_pass", return_value=True),
            mock.patch.object(stabilized, "_anderson_residual_hard_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_reference_merit",
                return_value=1.0,
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=0.5,
            ),
            mock.patch.object(
                stabilized,
                "_validate_anderson_candidate_map",
                return_value=(
                    np.zeros(self.model.n_material, dtype=np.float64),
                    1.0,
                ),
            ) as candidate_map,
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
                anderson_diagnostic_callback=lambda event: events.append(dict(event)),
            )

        self.assertIsNotNone(step)
        diagnostic = attempt.diagnostics
        self.assertEqual(diagnostic.anderson_selection, "accelerated")
        self.assertEqual(diagnostic.trial_alpha, 1.0)
        self.assertEqual(diagnostic.backtrack_count, 0)
        self.assertEqual(len(diagnostic.trials), 1)
        self.assertEqual(diagnostic.trials[0].trial_alpha, 1.0)
        self.assertTrue(diagnostic.trials[0].selected)
        self.assertIsNone(diagnostic.anderson_fallback_reason)
        self.assertIsNone(diagnostic.anderson_restart_reason)
        self.assertEqual(candidate_map.call_count, 1)
        accelerated = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        self.assertEqual(len(accelerated), 1)
        self.assertEqual(accelerated[0]["trial_alpha"], 1.0)

    def test_backtracking_threshold_uses_alpha_specific_limit_end_to_end(self) -> None:
        config = self._cache_config("anderson_residual_backtracking")
        events: list[dict[str, Any]] = []
        with (
            mock.patch.object(
                stabilized, "_anderson_propose", return_value=self._backtracking_proposal()
            ),
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
            mock.patch.object(stabilized, "_coupled_pass", return_value=True),
            mock.patch.object(
                stabilized, "_anderson_residual_hard_pass", return_value=True
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_reference_merit",
                return_value=1.0,
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                side_effect=[0.996, 0.996],
            ),
            mock.patch.object(
                stabilized,
                "_validate_anderson_candidate_map",
                return_value=(
                    np.zeros(self.model.n_material, dtype=np.float64),
                    1.0,
                ),
            ) as candidate_map,
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
                anderson_diagnostic_callback=lambda event: events.append(dict(event)),
            )

        self.assertIsNotNone(step)
        diagnostic = attempt.diagnostics
        self.assertEqual(diagnostic.anderson_selection, "accelerated")
        self.assertEqual(diagnostic.trial_alpha, 0.5)
        self.assertEqual(diagnostic.backtrack_count, 1)
        self.assertEqual([trial.trial_alpha for trial in diagnostic.trials], [1.0, 0.5])
        self.assertFalse(diagnostic.trials[0].selected)
        self.assertEqual(diagnostic.trials[0].failure_stage, "globalization")
        first_limit = diagnostic.trials[0].sufficient_decrease_limit
        self.assertIsNotNone(first_limit)
        self.assertAlmostEqual(cast(float, first_limit), 0.995)
        self.assertTrue(diagnostic.trials[1].selected)
        second_limit = diagnostic.trials[1].sufficient_decrease_limit
        self.assertIsNotNone(second_limit)
        self.assertAlmostEqual(cast(float, second_limit), 0.9975)
        self.assertEqual(candidate_map.call_count, 2)
        accelerated = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        self.assertEqual([event["trial_alpha"] for event in accelerated], [1.0, 0.5])
        self.assertEqual(
            [event["trial_alpha_index"] for event in accelerated], [0, 1]
        )
        sweep = [event for event in events if event.get("record_type") == "sweep"][-1]
        self.assertEqual(sweep["backtrack_count"], 1)
        self.assertEqual(sweep["trial_alpha"], 0.5)

    def test_backtracking_hard_gate_rejects_without_map_and_advances_alpha(self) -> None:
        config = self._cache_config("anderson_residual_backtracking")
        with (
            mock.patch.object(
                stabilized, "_anderson_propose", return_value=self._backtracking_proposal()
            ),
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
            mock.patch.object(stabilized, "_coupled_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_hard_pass",
                side_effect=[False, True],
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_reference_merit",
                return_value=1.0,
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=0.5,
            ),
            mock.patch.object(
                stabilized,
                "_validate_anderson_candidate_map",
                return_value=(
                    np.zeros(self.model.n_material, dtype=np.float64),
                    1.0,
                ),
            ) as candidate_map,
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )

        self.assertIsNotNone(step)
        trials = attempt.diagnostics.trials
        self.assertEqual([trial.trial_alpha for trial in trials], [1.0, 0.5])
        self.assertFalse(trials[0].hard_pass)
        self.assertEqual(trials[0].failure_stage, "anderson_candidate_hard_gate")
        self.assertEqual(trials[1].trial_alpha, 0.5)
        self.assertTrue(trials[1].selected)
        self.assertEqual(candidate_map.call_count, 1)

    def test_backtracking_candidate_map_failure_advances_alpha_without_restart(self) -> None:
        config = self._cache_config("anderson_residual_backtracking")

        def run_case(
            callback_enabled: bool,
        ) -> tuple[stabilized.StabilizedEngineeringGripAttempt, int, list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            with (
                mock.patch.object(
                    stabilized,
                    "_anderson_propose",
                    return_value=self._backtracking_proposal(),
                ),
                mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
                mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
                mock.patch.object(stabilized, "_coupled_pass", return_value=True),
                mock.patch.object(
                    stabilized, "_anderson_residual_hard_pass", return_value=True
                ),
                mock.patch.object(
                    stabilized,
                    "_anderson_residual_reference_merit",
                    return_value=1.0,
                ),
                mock.patch.object(
                    stabilized,
                    "_anderson_residual_candidate_merit",
                    return_value=0.5,
                ),
                mock.patch.object(
                    stabilized,
                    "_validate_anderson_candidate_map",
                    side_effect=[
                        FloatingPointError("manufactured candidate-map failure"),
                        (
                            np.zeros(self.model.n_material, dtype=np.float64),
                            1.0,
                        ),
                    ],
                ) as candidate_map,
            ):
                _, attempt = stabilized._run_stabilized_attempt(
                    self.model,
                    config,
                    self.result.initial_state,
                    1.0e-4,
                    0,
                    1,
                    None,
                    anderson_diagnostic_callback=(
                        None
                        if not callback_enabled
                        else lambda event: events.append(dict(event))
                    ),
                )
            return attempt, candidate_map.call_count, events

        without_callback, maps_without, _ = run_case(False)
        with_callback, maps_with, events = run_case(True)
        self.assertEqual(maps_without, maps_with)
        self.assertEqual(maps_with, 2)
        self.assertEqual(
            [trial.trial_alpha for trial in with_callback.diagnostics.trials],
            [1.0, 0.5],
        )
        self.assertEqual(
            [trial.failure_stage for trial in with_callback.diagnostics.trials],
            ["anderson_candidate_damage", ""],
        )
        self.assertEqual(with_callback.diagnostics.anderson_selection, "accelerated")
        self.assertIsNone(with_callback.diagnostics.anderson_restart_reason)
        self.assertIsNone(with_callback.diagnostics.anderson_fallback_reason)
        self.assertEqual(
            [
                (trial.candidate_kind, trial.trial_alpha, trial.failure_stage)
                for trial in with_callback.diagnostics.trials
            ],
            [
                (trial.candidate_kind, trial.trial_alpha, trial.failure_stage)
                for trial in without_callback.diagnostics.trials
            ],
        )
        accelerated = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        self.assertEqual([event["trial_alpha"] for event in accelerated], [1.0, 0.5])
        self.assertEqual(accelerated[0]["candidate_map_status"], "failed")
        self.assertEqual(accelerated[1]["candidate_map_status"], "completed")

    def test_backtracking_outer_candidate_failure_is_trial_local_without_restart(self) -> None:
        config = self._cache_config("anderson_residual_backtracking")
        real_mechanics = engineering._solve_direct_mechanics
        call_count = 0

        def mechanics_side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if 2 <= call_count <= 6:
                raise RuntimeError("manufactured candidate mechanics failure")
            return real_mechanics(*args, **kwargs)

        with (
            mock.patch.object(
                stabilized, "_anderson_propose", return_value=self._backtracking_proposal()
            ),
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
            mock.patch.object(stabilized, "_coupled_pass", return_value=True),
            mock.patch.object(stabilized, "_anderson_residual_hard_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_reference_merit",
                return_value=1.0,
            ),
            mock.patch.object(
                engineering,
                "_solve_direct_mechanics",
                side_effect=mechanics_side_effect,
            ),
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )

        self.assertIsNotNone(step)
        self.assertEqual(call_count, 7)
        trials = attempt.diagnostics.trials
        self.assertEqual(
            [trial.trial_alpha for trial in trials],
            [1.0, 0.5, 0.25, 0.125, 0.0625, None],
        )
        self.assertTrue(all(not trial.complete for trial in trials[:-1]))
        self.assertTrue(trials[-1].selected)
        self.assertEqual(trials[-1].candidate_kind, "picard_fallback")
        self.assertEqual(attempt.diagnostics.anderson_selection, "picard_fallback")
        self.assertIsNone(attempt.diagnostics.anderson_restart_reason)
        self.assertEqual(
            attempt.diagnostics.anderson_fallback_reason,
            "accelerated_candidate_mechanics_after",
        )

    def test_backtracking_finite_trial_is_unavailable_not_rejected_and_callback_invariant(
        self,
    ) -> None:
        config = self._cache_config("anderson_residual_backtracking")
        proposal = stabilized._AndersonProposal(
            candidate=np.full(self.model.n_material, np.inf, dtype=np.float64),
            history_size=1,
            qr_rank=1,
            fallback_reason=None,
            restart_reason=None,
            anderson_coefficient_l2=1.0,
        )

        def run_case(
            callback_enabled: bool,
        ) -> tuple[stabilized.StabilizedEngineeringGripAttempt, int, list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            real_solver = engineering._solve_direct_mechanics
            with (
                mock.patch.object(stabilized, "_anderson_propose", return_value=proposal),
                mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
                mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
                mock.patch.object(stabilized, "_coupled_pass", return_value=True),
                mock.patch.object(
                    stabilized, "_anderson_residual_hard_pass", return_value=True
                ),
                mock.patch.object(
                    engineering,
                    "_solve_direct_mechanics",
                    wraps=real_solver,
                ) as mechanics,
            ):
                step, attempt = stabilized._run_stabilized_attempt(
                    self.model,
                    config,
                    self.result.initial_state,
                    1.0e-4,
                    0,
                    1,
                    None,
                    anderson_diagnostic_callback=(
                        None
                        if not callback_enabled
                        else lambda event: events.append(dict(event))
                    ),
                )
            self.assertIsNotNone(step)
            return attempt, mechanics.call_count, events

        without_callback, solves_without, _ = run_case(False)
        with_callback, solves_with, events = run_case(True)

        self.assertEqual(solves_without, solves_with)
        self.assertEqual(solves_with, 2)
        self.assertEqual(
            [trial.trial_alpha for trial in with_callback.diagnostics.trials],
            [1.0, 0.5, 0.25, 0.125, 0.0625, None],
        )
        self.assertTrue(
            all(
                trial.failure_stage == "anderson_candidate_finite"
                for trial in with_callback.diagnostics.trials[:-1]
            )
        )
        self.assertEqual(
            with_callback.diagnostics.anderson_selection, "picard_fallback"
        )
        self.assertIsNone(with_callback.diagnostics.anderson_restart_reason)
        self.assertEqual(
            with_callback.diagnostics.anderson_fallback_reason,
            "anderson_candidate_finite",
        )
        finite_events = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        self.assertEqual(len(finite_events), 5)
        for index, event in enumerate(finite_events):
            with self.subTest(index=index):
                self.assertEqual(event["trial_alpha_index"], index)
                self.assertEqual(event["failure_stage"], "anderson_candidate_finite")
                self.assertEqual(event["candidate_map_status"], "not_requested")
                self.assertEqual(
                    event["evaluation_counts"],
                    {"mechanics_before": 1, "mechanics_after": 0, "candidate_map": 0},
                )
                self.assertEqual(event["hard_gate_status"], "not_evaluated")
                self.assertFalse(event["hard_pass"])
                self.assertIn("finite", event["failed_hard_gates"])
                self.assertIn("mechanics_after", event["failed_hard_gates"])
                self.assertFalse(
                    event["hard_gate_diagnostics"]["mechanics_after"]["available"]
                )
        report = stabilized.summarize_anderson_scalar_diagnostic(
            {
                "schema": stabilized.ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
                "path": None,
                "header": {},
                "events": finite_events,
                "final": {},
                "final_archive_sha256": None,
            }
        )
        self.assertTrue(
            all(count == 0 for count in report["hard_gate_failure_counts"].values())
        )
        self.assertEqual(report["hard_gate_overlap_counts"], {})
        self.assertEqual(
            report["hard_gate_unavailable_counts"]["mechanics_after"], 5
        )
        fallback_events = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "picard_fallback"
        ]
        self.assertEqual(len(fallback_events), 1)
        self.assertEqual(
            fallback_events[0]["evaluation_counts"],
            {"mechanics_before": 1, "mechanics_after": 1, "candidate_map": 0},
        )

    def test_backtracking_proposal_failure_retains_restart_semantics(self) -> None:
        config = self._cache_config("anderson_residual_backtracking")
        proposal = stabilized._AndersonProposal(
            candidate=None,
            history_size=1,
            qr_rank=0,
            fallback_reason="rank_deficient",
            restart_reason="rank_deficient",
            anderson_coefficient_l2=None,
        )
        with (
            mock.patch.object(stabilized, "_anderson_propose", return_value=proposal),
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(stabilized, "_damage_equation_pass", return_value=True),
            mock.patch.object(stabilized, "_coupled_pass", return_value=True),
            mock.patch.object(stabilized, "_anderson_residual_hard_pass", return_value=True),
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )

        self.assertIsNotNone(step)
        self.assertEqual(attempt.diagnostics.anderson_selection, "picard_fallback")
        self.assertEqual(attempt.diagnostics.anderson_restart_reason, "rank_deficient")
        self.assertEqual(attempt.diagnostics.backtrack_count, 0)
        self.assertEqual(len(attempt.diagnostics.trials), 1)
        self.assertIsNone(attempt.diagnostics.trials[0].trial_alpha)

    def test_residual_improvement_cannot_bypass_transition_gate(self) -> None:
        config = self._cache_config("anderson_residual")
        with (
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_reference_merit",
                return_value=1.0,
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=0.0,
            ),
            mock.patch.object(
                stabilized,
                "_validate_anderson_candidate_map",
                return_value=(
                    np.zeros(self.model.n_material, dtype=np.float64),
                    1.0,
                ),
            ),
            mock.patch.object(stabilized, "_coupled_pass", return_value=False),
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )

        self.assertIsNone(step)
        self.assertFalse(attempt.accepted)
        self.assertFalse(attempt.diagnostics.coupled_pass)
        self.assertTrue(attempt.diagnostics.globalization_pass)
        self.assertEqual(attempt.diagnostics.merit, 0.0)
        failure_reason = attempt.failure_reason
        self.assertIsNotNone(failure_reason)
        assert failure_reason is not None
        self.assertIn("coupled", failure_reason)

    def test_residual_picard_fallback_is_hard_gated(self) -> None:
        config = self._cache_config("anderson_residual")
        with mock.patch.object(
            stabilized, "_anderson_residual_hard_pass", return_value=False
        ):
            step, attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )

        self.assertIsNone(step)
        self.assertFalse(attempt.accepted)
        trial = attempt.diagnostics.trials[-1]
        self.assertEqual(trial.candidate_kind, "picard_fallback")
        self.assertFalse(trial.hard_pass)
        self.assertFalse(trial.selected)
        self.assertIsNone(trial.merit)
        self.assertEqual(trial.omega, 0.5)
        self.assertEqual(trial.failure_stage, "anderson_candidate_hard_gate")

    def test_residual_candidate_structural_errors_propagate(self) -> None:
        config = self._cache_config("anderson_residual")
        with (
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(
                stabilized, "_anderson_residual_hard_pass", return_value=True
            ),
            mock.patch.object(
                stabilized,
                "_validate_anderson_candidate_map",
                side_effect=ValueError("manufactured structural candidate error"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "structural candidate error"):
                stabilized._run_stabilized_attempt(
                    self.model,
                    config,
                    self.result.initial_state,
                    1.0e-4,
                    0,
                    1,
                    None,
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

    def test_anderson_diagnostic_callback_preserves_path_fields_selection_and_solves(self) -> None:
        config = self._cache_config("anderson")
        fixed_point_outcomes = [False, False, False, True, False, True]

        def run_case(
            callback_enabled: bool,
        ) -> tuple[stabilized.StabilizedEngineeringGripResult, int, int, list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            diagnostic_callback: Callable[[Any], None] | None = None
            if callback_enabled:
                diagnostic_callback = lambda event: events.append(dict(event))
            real_mechanics = engineering._solve_direct_mechanics
            real_damage = engineering.solve_fixed_history_damage
            with mock.patch.object(
                engineering,
                "_solve_direct_mechanics",
                wraps=real_mechanics,
            ) as mechanics_mock, mock.patch.object(
                engineering,
                "solve_fixed_history_damage",
                wraps=real_damage,
            ) as damage_mock, mock.patch.object(
                stabilized,
                "_fixed_point_pass",
                side_effect=list(fixed_point_outcomes),
            ):
                result = stabilized.run_stabilized_engineering_grip(
                    self.model,
                    config,
                    anderson_diagnostic_callback=diagnostic_callback,
                )
            return result, mechanics_mock.call_count, damage_mock.call_count, events

        without_callback, mechanics_without, damage_without, _ = run_case(False)
        with_callback, mechanics_with, damage_with, events = run_case(True)

        self._assert_cache_results_equal(with_callback, without_callback, "anderson")
        self.assertEqual(mechanics_with, mechanics_without)
        self.assertEqual(damage_with, damage_without)
        self.assertTrue(events)
        self.assertEqual(
            [
                (
                    attempt.accepted,
                    attempt.cutback_count,
                    tuple(
                        (
                            sweep.anderson_selection,
                            sweep.anderson_fallback_reason,
                            sweep.anderson_restart_reason,
                            tuple(
                                (
                                    trial.candidate_kind,
                                    trial.complete,
                                    trial.globalization_pass,
                                    trial.coupled_pass,
                                    trial.selected,
                                    trial.failure_stage,
                                    trial.failure_reason,
                                )
                                for trial in sweep.trials
                            ),
                        )
                        for sweep in attempt.sweeps
                    ),
                )
                for attempt in with_callback.attempts
            ],
            [
                (
                    attempt.accepted,
                    attempt.cutback_count,
                    tuple(
                        (
                            sweep.anderson_selection,
                            sweep.anderson_fallback_reason,
                            sweep.anderson_restart_reason,
                            tuple(
                                (
                                    trial.candidate_kind,
                                    trial.complete,
                                    trial.globalization_pass,
                                    trial.coupled_pass,
                                    trial.selected,
                                    trial.failure_stage,
                                    trial.failure_reason,
                                )
                                for trial in sweep.trials
                            ),
                        )
                        for sweep in attempt.sweeps
                    ),
                )
                for attempt in without_callback.attempts
            ],
        )
        self.assertTrue(
            any(event.get("record_type") == "proposal" for event in events)
        )

    def test_residual_hard_rejection_reports_cached_and_zero_map_counts(self) -> None:
        config = self._cache_config("anderson_residual")

        def run_case(
            callback_enabled: bool,
        ) -> tuple[stabilized.StabilizedEngineeringGripAttempt, int, int, int, list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            full_diagnostics_calls = 0
            real_full_diagnostics = engineering._full_diagnostics
            real_mechanics = engineering._solve_direct_mechanics
            real_damage = engineering.solve_fixed_history_damage
            real_candidate_map = stabilized._validate_anderson_candidate_map

            def hard_fail_after_first_sweep(*args: Any, **kwargs: Any) -> Any:
                nonlocal full_diagnostics_calls
                full_diagnostics_calls += 1
                if full_diagnostics_calls >= 2:
                    # Mutate only this manufactured mechanics result before the
                    # real diagnostic builder runs.  The stage gate and the
                    # attribution therefore see the same force defect; merely
                    # forging the aggregate diagnostic would be invalid under
                    # the stage-specific contract.
                    after = args[8]
                    force_scale = max(
                        1.0,
                        abs(float(after.outer_reaction)),
                        abs(float(after.bottom_reaction)),
                        abs(float(after.anchor_reaction)),
                    )
                    force_limit = config.base.force.limit(force_scale)
                    object.__setattr__(
                        after,
                        "total_force",
                        np.asarray([2.0 * force_limit, 0.0], dtype=np.float64),
                    )
                diagnostic = real_full_diagnostics(*args, **kwargs)
                return diagnostic

            with (
                mock.patch.object(
                    engineering,
                    "_full_diagnostics",
                    side_effect=hard_fail_after_first_sweep,
                ),
                mock.patch.object(
                    engineering,
                    "_solve_direct_mechanics",
                    wraps=real_mechanics,
                ) as mechanics_mock,
                mock.patch.object(
                    engineering,
                    "solve_fixed_history_damage",
                    wraps=real_damage,
                ) as damage_mock,
                mock.patch.object(
                    stabilized,
                    "_fixed_point_pass",
                    side_effect=[False, True, True],
                ),
                mock.patch.object(
                    stabilized,
                    "_damage_equation_pass",
                    return_value=True,
                ),
                mock.patch.object(
                    stabilized,
                    "_validate_anderson_candidate_map",
                    wraps=real_candidate_map,
                ) as candidate_map_mock,
            ):
                _, attempt = stabilized._run_stabilized_attempt(
                    self.model,
                    config,
                    self.result.initial_state,
                    1.0e-4,
                    0,
                    1,
                    None,
                    anderson_diagnostic_callback=(
                        None
                        if not callback_enabled
                        else lambda event: events.append(dict(event))
                    ),
                )
            return (
                attempt,
                mechanics_mock.call_count,
                damage_mock.call_count,
                candidate_map_mock.call_count,
                events,
            )

        without_callback, mechanics_without, damage_without, maps_without, _ = run_case(
            False
        )
        with_callback, mechanics_with, damage_with, maps_with, events = run_case(True)

        self.assertEqual(mechanics_with, mechanics_without)
        self.assertEqual(damage_with, damage_without)
        self.assertEqual(maps_with, maps_without)
        self.assertEqual(mechanics_with, 4)
        self.assertEqual(damage_with, 2)
        self.assertEqual(maps_with, 0)

        def diagnostic_signature(
            diagnostic: stabilized.StabilizedRelaxationSweepDiagnostics,
        ) -> tuple[Any, ...]:
            base = diagnostic.base_diagnostics
            return (
                diagnostic.complete_sweep,
                diagnostic.passed,
                diagnostic.failure_stage,
                diagnostic.failure_reason,
                diagnostic.omega,
                diagnostic.reference_merit,
                diagnostic.merit,
                diagnostic.sufficient_decrease_limit,
                diagnostic.fixed_point_error,
                diagnostic.fixed_point_scale,
                diagnostic.fixed_point_pass,
                diagnostic.damage_equation_pass,
                diagnostic.globalization_pass,
                diagnostic.coupled_pass,
                diagnostic.anderson_selection,
                diagnostic.anderson_fallback_reason,
                diagnostic.anderson_restart_reason,
                tuple(sorted(base.gate_passes.items())),
                base.complete_sweep,
                base.passed,
                base.failure_stage,
                base.failure_reason,
                tuple(
                    (
                        trial.candidate_kind,
                        trial.complete,
                        trial.hard_pass,
                        trial.globalization_pass,
                        trial.coupled_pass,
                        trial.selected,
                        trial.failure_stage,
                        trial.failure_reason,
                        trial.d_trial_digest,
                        trial.q_after_digest,
                    )
                    for trial in diagnostic.trials
                ),
            )

        def attempt_signature(
            attempt: stabilized.StabilizedEngineeringGripAttempt,
        ) -> tuple[Any, ...]:
            return (
                attempt.accepted,
                attempt.acceptance_status,
                attempt.load_start,
                attempt.load_target,
                attempt.cutback_count,
                attempt.failure_stage,
                attempt.failure_reason,
                attempt.exception_type,
                attempt.committed_before_digest,
                attempt.committed_after_digest,
                attempt.bypassed_gates,
                tuple(diagnostic_signature(sweep) for sweep in attempt.sweeps),
            )

        self.assertEqual(
            attempt_signature(with_callback), attempt_signature(without_callback)
        )
        for with_sweep, without_sweep in zip(
            with_callback.sweeps, without_callback.sweeps
        ):
            for name in ("d_solved", "d_trial"):
                with_value = getattr(with_sweep, name)
                without_value = getattr(without_sweep, name)
                if with_value is None or without_value is None:
                    self.assertIs(with_value, without_value)
                else:
                    np.testing.assert_array_equal(with_value, without_value)
            for name in ("d_iter", "q_before", "q_after"):
                with_value = getattr(with_sweep.base_diagnostics, name)
                without_value = getattr(without_sweep.base_diagnostics, name)
                if with_value is None or without_value is None:
                    self.assertIs(with_value, without_value)
                else:
                    np.testing.assert_array_equal(with_value, without_value)
        self.assertEqual(
            [
                tuple(
                    (trial.candidate_kind, trial.failure_stage, trial.hard_pass)
                    for trial in sweep.trials
                )
                for sweep in with_callback.sweeps
            ],
            [
                tuple(
                    (trial.candidate_kind, trial.failure_stage, trial.hard_pass)
                    for trial in sweep.trials
                )
                for sweep in without_callback.sweeps
            ],
        )
        accelerated = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        self.assertEqual(len(accelerated), 1)
        self.assertEqual(
            accelerated[0]["evaluation_counts"],
            {"mechanics_before": 0, "mechanics_after": 1, "candidate_map": 0},
        )
        self.assertEqual(accelerated[0]["candidate_map_status"], "not_requested")
        self.assertFalse(accelerated[0]["hard_pass"])
        self.assertEqual(accelerated[0]["failed_hard_gates"], ["force"])
        self.assertFalse(
            accelerated[0]["hard_gate_diagnostics"]["force"]["passed"]
        )
        force_detail = accelerated[0]["hard_gate_diagnostics"]["force"]
        self.assertEqual(tuple(force_detail["stage_metrics"]), ("before", "after"))
        self.assertAlmostEqual(
            force_detail["normalized"],
            max(
                force_detail["stage_metrics"][stage]["normalized"]
                for stage in ("before", "after")
            ),
        )
        self.assertAlmostEqual(
            force_detail["raw"] / force_detail["limit"],
            force_detail["normalized"],
        )
        self.assertTrue(
            all(
                event["evaluation_counts"]["candidate_map"] == 0
                for event in events
                if event.get("record_type") == "proposal"
            )
        )

    def test_candidate_map_failure_attribution_is_distinct_and_callback_invariant(self) -> None:
        config = self._cache_config("anderson_residual")

        def run_case(
            callback_enabled: bool,
        ) -> tuple[stabilized.StabilizedEngineeringGripResult, int, int, int, list[dict[str, Any]]]:
            events: list[dict[str, Any]] = []
            real_mechanics = engineering._solve_direct_mechanics
            real_damage = engineering.solve_fixed_history_damage
            with (
                mock.patch.object(
                    engineering,
                    "_solve_direct_mechanics",
                    wraps=real_mechanics,
                ) as mechanics_mock,
                mock.patch.object(
                    engineering,
                    "solve_fixed_history_damage",
                    wraps=real_damage,
                ) as damage_mock,
                mock.patch.object(
                    stabilized,
                    "_fixed_point_pass",
                    return_value=True,
                ),
                mock.patch.object(
                    stabilized,
                    "_validate_anderson_candidate_map",
                    side_effect=FloatingPointError("manufactured candidate-map failure"),
                ) as candidate_map_mock,
            ):
                result = stabilized.run_stabilized_engineering_grip(
                    self.model,
                    config,
                    anderson_diagnostic_callback=(
                        None
                        if not callback_enabled
                        else lambda event: events.append(dict(event))
                    ),
                )
            return (
                result,
                mechanics_mock.call_count,
                damage_mock.call_count,
                candidate_map_mock.call_count,
                events,
            )

        without_callback, mechanics_without, damage_without, maps_without, _ = run_case(
            False
        )
        with_callback, mechanics_with, damage_with, maps_with, events = run_case(True)

        self._assert_cache_results_equal(
            with_callback, without_callback, "anderson_residual"
        )
        self.assertEqual((mechanics_with, damage_with, maps_with), (7, 2, 1))
        self.assertEqual(
            (mechanics_with, damage_with, maps_with),
            (mechanics_without, damage_without, maps_without),
        )

        accelerated = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "anderson_accelerated"
        ]
        fallback = [
            event
            for event in events
            if event.get("record_type") == "proposal"
            and event.get("candidate_kind") == "picard_fallback"
        ]
        self.assertEqual(len(accelerated), 1)
        self.assertEqual(len(fallback), 2)
        failed_candidate = accelerated[0]
        self.assertEqual(failed_candidate["candidate_map_status"], "failed")
        self.assertEqual(failed_candidate["hard_gate_status"], "candidate_map_failure")
        self.assertFalse(failed_candidate["hard_pass"])
        self.assertEqual(failed_candidate["failed_hard_gates"], [])
        self.assertTrue(
            all(
                detail["passed"]
                for detail in failed_candidate["hard_gate_diagnostics"].values()
            )
        )
        self.assertEqual(fallback[-1]["hard_gate_status"], "evaluated")
        self.assertTrue(fallback[-1]["hard_pass"])
        self.assertEqual(fallback[-1]["failed_hard_gates"], [])

    def test_anderson_diagnostic_callback_oserror_is_not_a_numerical_fallback(self) -> None:
        config = self._cache_config("anderson")
        events: list[Any] = []

        def failing_callback(event: Any) -> None:
            events.append(event)
            raise OSError("manufactured diagnostic callback failure")

        with mock.patch.object(stabilized, "_fixed_point_pass", return_value=True):
            with self.assertRaisesRegex(OSError, "manufactured diagnostic callback"):
                stabilized.run_stabilized_engineering_grip(
                    self.model,
                    config,
                    anderson_diagnostic_callback=failing_callback,
                )
        self.assertTrue(events)

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

    def test_anderson_candidate_map_validates_type_pairing_and_residuals(self) -> None:
        history = np.zeros(self.model.n_material, dtype=np.float64)
        result = engineering.solve_fixed_history_damage(
            self.model.material_operator, self.model.Gc, self.model.l0, history
        )
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", return_value=object()
        ):
            with self.assertRaisesRegex(TypeError, "wrong type"):
                stabilized._validate_anderson_candidate_map(self.model, history)

        mismatched_history = replace(
            result,
            history=np.asarray(result.history, dtype=np.float64) + 1.0e-6,
        )
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", return_value=mismatched_history
        ):
            with self.assertRaisesRegex(ValueError, "not paired"):
                stabilized._validate_anderson_candidate_map(self.model, history)

        mismatched_residual = replace(
            result,
            strong_residual=np.asarray(result.strong_residual, dtype=np.float64)
            + 1.0,
        )
        with mock.patch.object(
            engineering, "solve_fixed_history_damage", return_value=mismatched_residual
        ):
            with self.assertRaisesRegex(FloatingPointError, "reconstruction failed"):
                stabilized._validate_anderson_candidate_map(self.model, history)

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
        self.assertEqual(
            replay["anderson_provenance"],
            stabilized._v5_anderson_provenance("fixed"),
        )

    def test_anderson_selection_and_fallback_provenance_round_trip(self) -> None:
        config = replace(
            self.config,
            base=replace(self.config.base, max_stagger_iterations=2),
            relaxation_policy="anderson",
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
        )
        with mock.patch.object(
            stabilized, "_fixed_point_pass", return_value=True
        ), mock.patch.object(stabilized, "_candidate_merit", return_value=0.0), mock.patch.object(
            stabilized,
            "_validate_anderson_candidate_map",
            wraps=stabilized._validate_anderson_candidate_map,
        ) as candidate_map_mock:
            result = stabilized.run_stabilized_engineering_grip(self.model, config)

        self.assertEqual(len(result.accepted_steps), 1)
        attempt = result.attempts[0]
        self.assertEqual(len(attempt.sweeps), 2)
        first, final = attempt.sweeps
        self.assertEqual(first.anderson_selection, "picard_fallback")
        self.assertEqual(first.anderson_history_size, 0)
        self.assertEqual(first.anderson_qr_rank, 0)
        self.assertEqual(first.anderson_fallback_reason, "insufficient_history")
        self.assertEqual(first.trials[-1].candidate_kind, "picard_fallback")
        self.assertEqual(first.omega, stabilized.STABILIZED_ANDERSON_BETA)
        self.assertEqual(final.anderson_selection, "accelerated")
        self.assertEqual(final.anderson_history_size, 1)
        self.assertEqual(final.anderson_qr_rank, 1)
        self.assertIsNone(final.anderson_fallback_reason)
        self.assertIsNone(final.anderson_restart_reason)
        self.assertIsNone(final.omega)
        self.assertEqual(final.trials[-1].candidate_kind, "anderson_accelerated")
        candidate_map_mock.assert_called_once()
        self.assertIsNotNone(final.candidate_fp_error)
        self.assertIsNotNone(final.candidate_fp_scale)
        self.assertIsNotNone(final.candidate_damage_residual_scale)

        with tempfile.TemporaryDirectory(prefix="stabilized-v5-anderson-") as name:
            path, payload = self._write_archive(Path(name), result=result)
            metadata = json.loads(str(payload["metadata_json"]))
            self.assertEqual(
                metadata["stabilized_config"]["relaxation_policy"], "anderson"
            )
            summary = json.loads(str(payload["solver_summary_json"][1]))
            self.assertEqual(summary["anderson_selection"], "accelerated")
            self.assertIsNone(summary["omega"])
            self.assertEqual(summary["anderson_history_size"], 1)
            self.assertEqual(summary["anderson_qr_rank"], 1)
            self.assertIsNotNone(summary["candidate_fp_error"])
            self.assertIsNotNone(summary["candidate_fp_scale"])
            self.assertIsNotNone(summary["candidate_damage_residual_scale"])
            replay = stabilized.replay_stabilized_engineering_grip_archive(path)
            for gate_name in ("damage_range", "healing"):
                forged = self._copy_payload(payload)
                self._update_summary(
                    forged,
                    1,
                    lambda record, name=gate_name: record["gate_passes"].__setitem__(
                        name, False
                    ),
                )
                forged_path = Path(name) / f"forged-anderson-{gate_name}.npz"
                self._write_payload(Path(name), forged_path.name, forged)
                with self.assertRaises(ValueError):
                    stabilized.replay_stabilized_engineering_grip_archive(
                        forged_path
                    )
        self.assertTrue(replay["replayed"])
        self.assertTrue(replay["field_equations_validated"])
        self.assertEqual(
            replay["anderson_provenance"],
            {
                "status": "recorded_not_reconstructed",
                "selection_recorded": True,
                "selection_reconstructed": False,
                "fallback_recorded": True,
                "fallback_reconstructed": False,
                "restart_recorded": True,
                "restart_reconstructed": False,
                "history_rank_recorded": True,
                "history_rank_reconstructed": False,
            },
        )

    def test_residual_anderson_archive_round_trip_and_policy_mismatch_rejection(self) -> None:
        config = self._cache_config("anderson_residual")
        with (
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=0.0,
            ),
        ):
            result = stabilized.run_stabilized_engineering_grip(self.model, config)

        self.assertEqual(
            result.config.relaxation_policy,
            "anderson_residual",
        )
        self.assertEqual(
            result.accepted_steps[-1].diagnostics.anderson_selection,
            "accelerated",
        )
        with tempfile.TemporaryDirectory(prefix="stabilized-v5-residual-") as name:
            directory = Path(name)
            path, payload = self._write_archive(directory, result=result)
            metadata = json.loads(str(payload["metadata_json"]))
            stable_payload = metadata["stabilized_config"]
            self.assertEqual(
                stable_payload["relaxation_policy"], "anderson_residual"
            )
            self.assertEqual(
                stable_payload["anderson_merit_identity"],
                stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY,
            )
            summary = json.loads(str(payload["solver_summary_json"][1]))
            self.assertEqual(
                summary["anderson_merit_identity"],
                stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY,
            )
            replay = stabilized.replay_stabilized_engineering_grip_archive(path)
            self.assertEqual(replay["relaxation_policy"], "anderson_residual")
            self.assertEqual(
                replay["selection_merit_identity"],
                stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY,
            )

            forged = self._copy_payload(payload)
            self._update_summary(
                forged,
                1,
                lambda record: record.__setitem__("relaxation_policy", "anderson"),
            )
            forged_path = self._write_payload(
                directory, "forged-residual-policy.npz", forged
            )
            with self.assertRaises(ValueError):
                stabilized.replay_stabilized_engineering_grip_archive(forged_path)

    def test_residual_summary_rejects_forged_accepted_selection_fields(self) -> None:
        accelerated_config = self._cache_config("anderson_residual")
        with (
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=0.0,
            ),
        ):
            accelerated_result = stabilized.run_stabilized_engineering_grip(
                self.model, accelerated_config
            )
        accelerated_summary = stabilized._v5_solver_summary_payload(
            1,
            accelerated_result.accepted_steps[0],
            relaxation_policy="anderson_residual",
        )
        stabilized._v5_validate_solver_summary(
            accelerated_summary,
            "valid residual accelerated summary",
            relaxation_policy="anderson_residual",
        )
        for field in ("merit", "reference_merit", "candidate_fp_error"):
            with self.subTest(selection="accelerated", field=field):
                forged = dict(accelerated_summary)
                forged[field] = None
                with self.assertRaises(ValueError):
                    stabilized._v5_validate_solver_summary(
                        forged,
                        f"forged residual accelerated {field}",
                        relaxation_policy="anderson_residual",
                    )

        fallback_config = self._cache_config("anderson_residual")
        with (
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=1.0e30,
            ),
        ):
            fallback_result = stabilized.run_stabilized_engineering_grip(
                self.model, fallback_config
            )
        fallback_summary = stabilized._v5_solver_summary_payload(
            1,
            fallback_result.accepted_steps[0],
            relaxation_policy="anderson_residual",
        )
        self.assertEqual(fallback_summary["anderson_selection"], "picard_fallback")
        stabilized._v5_validate_solver_summary(
            fallback_summary,
            "valid residual fallback summary",
            relaxation_policy="anderson_residual",
        )
        for field, value in (
            ("merit", 123.0),
            ("candidate_fp_error", 0.1),
            ("candidate_fp_scale", 1.0),
            ("candidate_damage_residual_scale", 1.0),
        ):
            with self.subTest(selection="picard_fallback", field=field):
                forged = dict(fallback_summary)
                forged[field] = value
                with self.assertRaisesRegex(ValueError, "selection metrics"):
                    stabilized._v5_validate_solver_summary(
                        forged,
                        f"forged residual fallback {field}",
                        relaxation_policy="anderson_residual",
                    )

    def test_residual_failure_summary_requires_complete_accelerated_metrics(self) -> None:
        config = self._cache_config("anderson_residual")
        with (
            mock.patch.object(stabilized, "_fixed_point_pass", return_value=True),
            mock.patch.object(
                stabilized,
                "_anderson_residual_reference_merit",
                return_value=1.0,
            ),
            mock.patch.object(
                stabilized,
                "_anderson_residual_candidate_merit",
                return_value=0.0,
            ),
            mock.patch.object(
                stabilized,
                "_validate_anderson_candidate_map",
                return_value=(
                    np.zeros(self.model.n_material, dtype=np.float64),
                    1.0,
                ),
            ),
            mock.patch.object(stabilized, "_coupled_pass", return_value=False),
        ):
            step, rejected_attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )
        self.assertIsNone(step)
        failure = stabilized.StabilizedEngineeringGripContinuationFailure(
            "manufactured residual failure",
            self.result.initial_state,
            (rejected_attempt,),
            self.result.work_check,
        )
        failure_summary = stabilized._v5_failure_summary(
            failure,
            relaxation_policy="anderson_residual",
        )
        solver_summary = failure_summary["solver_summary"]
        self.assertIsInstance(solver_summary, dict)
        assert isinstance(solver_summary, dict)
        self.assertEqual(solver_summary["anderson_selection"], "accelerated")
        stabilized._v5_validate_solver_summary(
            solver_summary,
            "valid residual failure summary",
            relaxation_policy="anderson_residual",
        )
        for field in ("merit", "reference_merit", "candidate_fp_scale"):
            with self.subTest(field=field):
                forged = dict(solver_summary)
                forged[field] = None
                with self.assertRaises(ValueError):
                    stabilized._v5_validate_solver_summary(
                        forged,
                        f"forged residual failure {field}",
                        relaxation_policy="anderson_residual",
                    )

        with mock.patch.object(
            stabilized, "_anderson_residual_hard_pass", return_value=False
        ):
            _, incomplete_attempt = stabilized._run_stabilized_attempt(
                self.model,
                config,
                self.result.initial_state,
                1.0e-4,
                0,
                1,
                None,
            )
        incomplete_failure = stabilized.StabilizedEngineeringGripContinuationFailure(
            "manufactured incomplete residual failure",
            self.result.initial_state,
            (incomplete_attempt,),
            self.result.work_check,
        )
        incomplete_summary = stabilized._v5_failure_summary(
            incomplete_failure,
            relaxation_policy="anderson_residual",
        )["solver_summary"]
        self.assertIsInstance(incomplete_summary, dict)
        assert isinstance(incomplete_summary, dict)
        self.assertFalse(incomplete_summary["complete_sweep"])
        self.assertIsNone(incomplete_summary["merit"])
        self.assertIsNone(incomplete_summary["candidate_fp_error"])
        stabilized._v5_validate_solver_summary(
            incomplete_summary,
            "valid incomplete residual failure summary",
            relaxation_policy="anderson_residual",
        )

    def test_anderson_merit_rejection_restarts_and_uses_picard_fallback(self) -> None:
        config = replace(
            self.config,
            base=replace(self.config.base, max_stagger_iterations=2),
            relaxation_policy="anderson",
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
        )
        with mock.patch.object(
            stabilized, "_fixed_point_pass", return_value=True
        ), mock.patch.object(
            stabilized,
            "_candidate_merit",
            side_effect=[0.0, 0.0],
        ), mock.patch.object(
            stabilized,
            "_anderson_candidate_merit",
            return_value=1.0e30,
        ):
            result = stabilized.run_stabilized_engineering_grip(self.model, config)

        final = result.attempts[0].sweeps[-1]
        self.assertEqual(final.anderson_selection, "picard_fallback")
        self.assertEqual(final.anderson_fallback_reason, "insufficient_decrease")
        self.assertEqual(final.anderson_restart_reason, "insufficient_decrease")
        self.assertEqual(final.omega, stabilized.STABILIZED_ANDERSON_BETA)
        self.assertEqual(
            [trial.candidate_kind for trial in final.trials],
            ["anderson_accelerated", "picard_fallback"],
        )
        self.assertFalse(final.trials[0].globalization_pass)
        self.assertTrue(final.trials[1].selected)

    def test_anderson_numerical_candidate_map_failure_falls_back_and_restarts(self) -> None:
        config = replace(
            self.config,
            base=replace(self.config.base, max_stagger_iterations=2),
            relaxation_policy="anderson",
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
        )
        with mock.patch.object(
            stabilized, "_fixed_point_pass", return_value=True
        ), mock.patch.object(
            stabilized, "_candidate_merit", return_value=0.0
        ), mock.patch.object(
            stabilized,
            "_validate_anderson_candidate_map",
            side_effect=FloatingPointError("manufactured candidate-map failure"),
        ):
            result = stabilized.run_stabilized_engineering_grip(self.model, config)

        final = result.attempts[0].sweeps[-1]
        self.assertEqual(final.anderson_selection, "picard_fallback")
        self.assertEqual(final.anderson_fallback_reason, "anderson_candidate_damage")
        self.assertEqual(final.anderson_restart_reason, "anderson_candidate_damage")
        self.assertEqual(
            [trial.candidate_kind for trial in final.trials],
            ["anderson_accelerated", "picard_fallback"],
        )
        self.assertFalse(final.trials[0].complete)
        self.assertIn("manufactured candidate-map failure", final.trials[0].failure_reason)

    def test_anderson_structural_candidate_map_error_propagates(self) -> None:
        config = replace(
            self.config,
            base=replace(self.config.base, max_stagger_iterations=2),
            relaxation_policy="anderson",
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
        )
        with mock.patch.object(
            stabilized, "_fixed_point_pass", return_value=True
        ), mock.patch.object(
            stabilized, "_candidate_merit", return_value=0.0
        ), mock.patch.object(
            stabilized,
            "_validate_anderson_candidate_map",
            side_effect=ValueError("manufactured structural candidate-map error"),
        ), self.assertRaisesRegex(ValueError, "structural candidate-map"):
            stabilized.run_stabilized_engineering_grip(self.model, config)

    def test_anderson_retains_original_unrelaxed_fixed_point_gate(self) -> None:
        config = replace(
            self.config,
            base=replace(
                self.config.base,
                load_final=1.0e-4,
                du_initial=1.0e-4,
                du_min=1.0e-4,
                max_cutbacks=1,
                max_stagger_iterations=1,
            ),
            relaxation_policy="anderson",
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
        )
        with mock.patch.object(stabilized, "_fixed_point_pass", return_value=False), self.assertRaises(
            stabilized.StabilizedEngineeringGripContinuationFailure
        ) as context:
            stabilized.run_stabilized_engineering_grip(self.model, config)

        diagnostic = context.exception.attempts[-1].diagnostics
        self.assertFalse(diagnostic.fixed_point_pass)
        self.assertIn("damage_fixed_point", diagnostic.failure_reason)

    def test_anderson_strict_acceptance_requires_range_and_healing(self) -> None:
        config = replace(
            self.config,
            base=replace(
                self.config.base,
                load_final=1.0e-4,
                du_initial=1.0e-4,
                du_min=1.0e-4,
                max_cutbacks=1,
                max_stagger_iterations=1,
            ),
            relaxation_policy="anderson",
            omega_initial=stabilized.STABILIZED_ANDERSON_BETA,
            omega_min=stabilized.STABILIZED_ANDERSON_BETA,
            omega_reduction=None,
            sufficient_decrease=1.0e-2,
            max_backtracks=0,
            anderson_history_depth=stabilized.STABILIZED_ANDERSON_HISTORY_DEPTH,
            anderson_beta=stabilized.STABILIZED_ANDERSON_BETA,
        )
        real_full_diagnostics = engineering._full_diagnostics

        def forged_range_healing(*args: Any, **kwargs: Any) -> Any:
            return replace(
                real_full_diagnostics(*args, **kwargs),
                damage_range_pass=False,
                healing_pass=False,
            )

        with mock.patch.object(
            stabilized, "_fixed_point_pass", return_value=True
        ), mock.patch.object(
            stabilized, "_damage_equation_pass", return_value=True
        ), mock.patch.object(
            stabilized, "_coupled_pass", return_value=True
        ), mock.patch.object(
            engineering, "_full_diagnostics", side_effect=forged_range_healing
        ), self.assertRaises(
            stabilized.StabilizedEngineeringGripContinuationFailure
        ) as context:
            stabilized.run_stabilized_engineering_grip(self.model, config)

        diagnostic = context.exception.attempts[-1].diagnostics
        self.assertFalse(diagnostic.passed)
        self.assertIn("damage_range", diagnostic.failure_reason)
        self.assertIn("healing", diagnostic.failure_reason)

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
