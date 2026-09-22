"""Focused tests for the durable engineering-grip orchestration layer."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
import unittest
from unittest import mock

import jax
import numpy as np

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import engineering_grip_workflow as workflow
from examples.fracture2d import nonlocal_engineering_grip_stabilized as stabilized
from examples.fracture2d.nonlocal_grip_diagnostic import (
    build_union_geometry_from_arrays,
)


def _source_arrays() -> dict[str, np.ndarray]:
    return {
        "coordinates": np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [0.01, 0.0]], dtype=np.float64
        ),
        "triangles": np.asarray([[0, 1, 2]], dtype=np.int64),
        "volumes": np.asarray([1.0 / 6.0] * 3, dtype=np.float64),
        "bottom": np.asarray([True, False, True], dtype=bool),
        "top": np.asarray([False, True, False], dtype=bool),
        "anchor_particle": np.asarray(0, dtype=np.int64),
        "d048": np.asarray([[0, 2], [2, 0]], dtype=np.int64),
    }


def _stabilized_source_arrays() -> dict[str, np.ndarray]:
    """Small source cloud with the exact crack-tip particle used by reports."""

    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.49, 0.5],
            [0.51, 0.5],
        ],
        dtype=np.float64,
    )
    return {
        "coordinates": coordinates,
        "triangles": np.asarray(
            [[0, 3, 2], [0, 2, 1], [3, 4, 2], [4, 1, 2]], dtype=np.int64
        ),
        "volumes": np.full(len(coordinates), 1.0 / len(coordinates), dtype=np.float64),
        "bottom": np.asarray([True, False, False, True, False, False, False], dtype=bool),
        "top": np.asarray([False, True, False, False, True, False, False], dtype=bool),
        "anchor_particle": np.asarray(0, dtype=np.int64),
        "d048": np.asarray(
            [[2, 5], [2, 6], [5, 2], [5, 6], [6, 2], [6, 5]], dtype=np.int64
        ),
    }


def _fem_arrays() -> dict[str, np.ndarray]:
    return {
        "coords": np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64
        ),
        # The production FEM writer uses int32 connectivity.
        "triangles": np.asarray([[0, 1, 2]], dtype=np.int32),
        "u_bar": np.asarray([0.0, 0.1], dtype=np.float64),
        "reaction_y": np.asarray([0.0, 2.0], dtype=np.float64),
        "nominal_strain": np.asarray([0.0, 0.1], dtype=np.float64),
        "nominal_stress": np.asarray([0.0, 2.0], dtype=np.float64),
        "snapshot_u_bar": np.asarray([0.0, 0.1], dtype=np.float64),
        "u": np.zeros((2, 3, 2), dtype=np.float64),
        "d": np.zeros((2, 3), dtype=np.float64),
        "gate_mode": np.asarray("warning"),
        "boundary_mode": np.asarray("bottom_vertical_point_horizontal"),
        "plane_assumption": np.asarray("plane_strain"),
    }


def _save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    # NumPy's stubs reserve a keyword named ``allow_pickle``; an Any-valued
    # mapping keeps arbitrary numerical field names type-safe in these tests.
    np.savez_compressed(str(path), **dict(arrays))


def _assert_scalar_tree(test: unittest.TestCase, value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_scalar_tree(test, item)
    elif isinstance(value, list):
        test.assertTrue(all(isinstance(item, (str, dict)) for item in value))
        for item in value:
            _assert_scalar_tree(test, item)


_STABILIZED_SOURCE_FIXTURE: tuple[Path, dict[str, Any]] | None = None


def _stabilized_source_fixture() -> tuple[Path, dict[str, Any]]:
    """Return one real temporary v5 source cloud and its exact provenance."""

    global _STABILIZED_SOURCE_FIXTURE
    if _STABILIZED_SOURCE_FIXTURE is None:
        directory = tempfile.TemporaryDirectory(prefix="engineering-workflow-source-")
        path = Path(directory.name) / "source.npz"
        _save_npz(path, _stabilized_source_arrays())
        _, metadata = workflow._load_source_cloud(path, 0.048)
        # Keep the owner alive for the duration of this test process.  The
        # source itself is deliberately external to every archive fixture.
        _STABILIZED_SOURCE_FIXTURE = (path, metadata)
        _STABILIZED_SOURCE_FIXTURE_OWNER = directory
        globals()["_STABILIZED_SOURCE_FIXTURE_OWNER"] = _STABILIZED_SOURCE_FIXTURE_OWNER
    return _STABILIZED_SOURCE_FIXTURE


@contextlib.contextmanager
def _schedule_fixture(root: Path) -> Any:
    """Create a valid temporary schedule for workflow binding tests."""

    targets = np.asarray([0.25, 0.5], dtype=np.dtype("<f8"))
    source = root / (
        "engineering_grip_anderson_u5p9875e3_dumin6p25e6_s144.npz"
    )
    metadata = {
        "schema": stabilized.STABILIZED_ARCHIVE_SCHEMA,
        "kind": "completed",
        "completion_status": stabilized.STABILIZED_COMPLETION_STATUS,
        "stabilized_config": {"relaxation_policy": "anderson"},
    }
    _save_npz(
        source,
        {
            "archive_schema": np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA),
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False),
                dtype=np.str_,
            ),
            "state_u_bar": np.asarray([0.0, *targets], dtype=np.dtype("<f8")),
        },
    )
    schedule_path = root / "schedule.npz"
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
        yield stabilized.extract_stabilized_load_schedule(source, schedule_path)


def _bound_stabilized_replay(
    archive: Path, values: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    metadata = json.loads(str(values["metadata_json"]))
    valid, binding, error = workflow._stabilized_replay_binding(
        archive, values, metadata, {}, stabilized
    )
    if binding == {}:
        raise AssertionError(f"test replay binding could not be derived: {error}")
    replay = {
        "schema": binding["archive_schema"],
        "experimental": True,
        "field_equations_validated": True,
        "path_replayed": False,
        "rejected_trial_path_available": False,
        **binding,
    }
    if not valid:
        valid, _, error = workflow._stabilized_replay_binding(
            archive, values, metadata, replay, stabilized
        )
    if not valid:
        raise AssertionError(f"test replay binding is not valid: {error}")
    return replay


def _stabilized_metadata(
    kind: str = "completed",
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    if source_metadata is None:
        _, source_metadata = _stabilized_source_fixture()
    config = engineering.EngineeringGripConfig.smoke()
    stabilized_config = stabilized.StabilizedEngineeringGripConfig.smoke()
    schema = (
        stabilized.STABILIZED_ARCHIVE_SCHEMA
        if kind == "completed"
        else stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA
    )
    status = (
        stabilized.STABILIZED_COMPLETION_STATUS
        if kind == "completed"
        else stabilized.STABILIZED_FAILURE_STATUS
    )
    return {
        "schema": schema,
        "layout": stabilized.STABILIZED_ARCHIVE_LAYOUT,
        "replay_contract": stabilized.STABILIZED_REPLAY_CONTRACT,
        "kind": kind,
        "validated": False,
        "experimental": True,
        "field_equations_validated": False,
        "path_replayed": False,
        "rejected_trial_path_available": False,
        "acceptance_mode": stabilized.STABILIZED_ACCEPTANCE_MODE,
        "acceptance_status": stabilized.STABILIZED_ACCEPTANCE_STATUS,
        "completion_status": status,
        "config": engineering._config_payload(config),
        "stabilized_config": stabilized_config.payload(),
        "code_hash": stabilized._current_stabilized_source_bundle_hash(),
        "delta": 0.048,
        "source": dict(source_metadata),
        "source_sha256": source_metadata["sha256"],
    }


def _stabilized_summary_rows(
    load_target: float,
    *,
    sweep_index: int = 0,
) -> np.ndarray:
    initial = stabilized._v5_initial_solver_summary()
    accepted = dict(initial)
    accepted.update(
        {
            "summary_kind": "accepted",
            "state_index": 1,
            "load_start": 0.0,
            "load_target": load_target,
            "increment": load_target,
            "cutback_count": 0,
            "stagger_iterations": sweep_index + 1,
            "acceptance_status": stabilized.STABILIZED_ACCEPTANCE_STATUS,
            "accepted": True,
            "sweep_index": sweep_index,
            "target_u_bar": load_target,
            "complete_sweep": True,
            "passed": True,
            "failure_stage": "",
            "failure_reason": "",
            "gate_passes": {
                name: True for name in stabilized._V5_SOLVER_SUMMARY_GATE_NAMES
            },
            "omega": 1.0,
            "backtrack_count": 0,
            "reference_merit": 0.0,
            "merit": 0.0,
            "sufficient_decrease_limit": 0.0,
            "fixed_point_error": 0.0,
            "fixed_point_scale": 1.0,
            "fixed_point_pass": True,
            "damage_equation_pass": True,
            "globalization_pass": True,
            "coupled_pass": True,
        }
    )
    for name in stabilized._V5_SOLVER_SUMMARY_SCALAR_FIELDS:
        if name.startswith("cg_info_"):
            accepted[name] = 0
        elif name.endswith("_scale"):
            accepted[name] = 1.0
        else:
            accepted[name] = 0.0
    accepted.update(
        {
            "top_reaction_before": 1.0,
            "top_reaction_after": 1.0,
            "bottom_reaction_before": -0.5,
            "bottom_reaction_after": -0.5,
            "anchor_reaction_before": -0.5,
            "anchor_reaction_after": -0.5,
        }
    )
    for name in stabilized._V5_DIAGNOSTIC_BOOL_FIELDS:
        accepted[name] = True
    return np.asarray(
        [
            json.dumps(initial, sort_keys=True, allow_nan=False),
            json.dumps(accepted, sort_keys=True, allow_nan=False),
        ],
        dtype=np.str_,
    )


def _stabilized_completed_arrays(
    source_metadata: Mapping[str, Any] | None = None,
    *,
    sweep_index: int = 0,
) -> dict[str, np.ndarray]:
    metadata = _stabilized_metadata(source_metadata=source_metadata)
    n_material = len(_stabilized_source_arrays()["coordinates"])
    load_target = 1.0e-4
    return {
        "archive_schema": np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA, dtype=np.str_),
        "acceptance_mode": np.asarray(stabilized.STABILIZED_ACCEPTANCE_MODE, dtype=np.str_),
        "acceptance_status": np.asarray(stabilized.STABILIZED_ACCEPTANCE_STATUS, dtype=np.str_),
        "completion_status": np.asarray(stabilized.STABILIZED_COMPLETION_STATUS, dtype=np.str_),
        "validated": np.asarray(False, dtype=bool),
        "experimental": np.asarray(True, dtype=bool),
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False), dtype=np.str_
        ),
        "state_u_bar": np.asarray([0.0, 1.0e-4], dtype=np.float64),
        "state_material_damage": np.zeros((2, n_material), dtype=np.float64),
        "state_material_history": np.zeros((2, n_material), dtype=np.float64),
        # v5 stores the full union displacement; plotting must reduce it to the
        # material prefix before using source-rebuilt material triangles.
        "state_union_displacement": np.zeros(
            (2, 2 * (n_material + 2)), dtype=np.float64
        ),
        "state_digest": np.asarray(["0" * 64, "1" * 64], dtype=np.str_),
        "state_acceptance_status": np.asarray(
            ["initial", stabilized.STABILIZED_ACCEPTANCE_STATUS], dtype=np.str_
        ),
        "accepted_top_reaction_after": np.asarray([1.0], dtype=np.float64),
        "accepted_bottom_reaction_after": np.asarray([-0.5], dtype=np.float64),
        "accepted_anchor_reaction_after": np.asarray([-0.5], dtype=np.float64),
        "state_external_work": np.asarray([0.0, 5.0e-5], dtype=np.float64),
        "solver_summary_json": _stabilized_summary_rows(
            load_target, sweep_index=sweep_index
        ),
        "work_check_u_bar": np.asarray(load_target, dtype=np.float64),
        "work_check_epsilon": np.asarray(1.0e-5, dtype=np.float64),
        "work_check_reaction": np.asarray(1.0, dtype=np.float64),
        "work_check_reaction_fd": np.asarray(1.0, dtype=np.float64),
        "work_check_relative_error": np.asarray(0.0, dtype=np.float64),
        "work_check_relative_limit": np.asarray(1.0e-7, dtype=np.float64),
        "work_check_passed": np.asarray(True, dtype=bool),
    }


def _stabilized_failure_summary_json(
    load_target: float,
    *,
    sweep_index: int,
) -> np.ndarray:
    rows = [
        json.loads(str(raw))
        for raw in _stabilized_summary_rows(load_target, sweep_index=7)
    ]
    failure_row = dict(rows[-1])
    failure_row.update(
        {
            "summary_kind": "failure",
            "state_index": -1,
            "sweep_index": sweep_index,
            "stagger_iterations": sweep_index + 1,
            "accepted": False,
            "complete_sweep": False,
            "passed": False,
            "failure_stage": "gates",
            "failure_reason": "manufactured failure",
            "e_damage": 1.0,
        }
    )
    return np.asarray(
        json.dumps(
            {
                "available": True,
                "failure_code": stabilized.STABILIZED_FAILURE_CODE,
                "failure_type": "StabilizedEngineeringGripContinuationFailure",
                "failure_message": "manufactured failure",
                "last_committed_u_bar": load_target,
                "failure_stage": "gates",
                "failure_reason": "manufactured failure",
                "load_start": 0.0,
                "load_target": load_target,
                "increment": load_target,
                "cutback_count": 0,
                "solver_summary": failure_row,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        dtype=np.str_,
    )


class EngineeringGripWorkflowTest(unittest.TestCase):
    def test_run_parser_uses_current_numerical_defaults(self) -> None:
        arguments = workflow.build_parser().parse_args(["run"])

        self.assertEqual(arguments.source_cloud, workflow.DEFAULT_SOURCE_CLOUD)
        self.assertEqual(arguments.fem_archive, workflow.DEFAULT_FEM_ARCHIVE)
        self.assertEqual(arguments.load_final, 0.0075)
        self.assertEqual(arguments.max_stagger_iterations, 144)
        self.assertEqual(arguments.gpu_device_index, 0)
        self.assertEqual(arguments.delta, 0.048)
        self.assertEqual(arguments.h_g, 0.02)
        self.assertEqual(arguments.du_initial, 1.0e-4)
        self.assertEqual(arguments.du_min, 6.25e-6)
        self.assertEqual(arguments.visibility_tolerance, 1.0e-12)
        self.assertEqual(arguments.youngs_modulus, 210000.0)
        self.assertEqual(arguments.poisson_ratio, 0.3)
        self.assertEqual(arguments.k_res, 1.0e-8)
        self.assertEqual(arguments.rank_rtol, 1.0e-12)
        self.assertEqual(arguments.Gc, 2.7)
        self.assertEqual(arguments.l0, 0.024)
        self.assertEqual(arguments.relaxation_policy, "fixed")
        self.assertEqual(arguments.omega_initial, 0.5)
        self.assertEqual(arguments.omega_min, 0.5)
        self.assertEqual(arguments.max_backtracks, 0)
        self.assertEqual(arguments.acceptance_mode, "strict")
        self.assertIsNone(arguments.warning_safety_policy)
        self.assertFalse(arguments.anderson_scalar_diagnostic)

    def test_extract_schedule_parser_is_read_only_and_has_aliases(self) -> None:
        parser = workflow.build_parser()
        arguments = parser.parse_args(
            [
                "extract_schedule",
                "--source-archive",
                "/tmp/historical.npz",
                "--schedule-output",
                "/tmp/reviewed-schedule.npz",
            ]
        )

        self.assertEqual(arguments.command, "extract_schedule")
        self.assertEqual(arguments.source_archive, Path("/tmp/historical.npz"))
        self.assertEqual(arguments.output_path, Path("/tmp/reviewed-schedule.npz"))
        self.assertFalse(hasattr(arguments, "overwrite"))

    def test_source_cloud_contract_preserves_exact_arrays(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-") as directory:
            path = Path(directory) / "cloud.npz"
            source = _source_arrays()
            _save_npz(path, source)

            loaded, metadata = workflow._load_source_cloud(path, 0.048)

            for name, expected in source.items():
                np.testing.assert_array_equal(loaded[name], expected)
                self.assertEqual(loaded[name].dtype, expected.dtype)
            self.assertEqual(metadata["sha256"], workflow._sha256_file(path))
            self.assertFalse(metadata["copied_into_output"])

    def test_source_cloud_rejects_missing_fields_and_wrong_delta(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-") as directory:
            path = Path(directory) / "cloud.npz"
            source = _source_arrays()
            incomplete = dict(source)
            del incomplete["d048"]
            _save_npz(path, incomplete)
            with self.assertRaisesRegex(ValueError, "missing required fields.*d048"):
                workflow._load_source_cloud(path, 0.048)

            _save_npz(path, source)
            with self.assertRaisesRegex(ValueError, "only compatible with delta=0.048"):
                workflow._load_source_cloud(path, 0.04)

    def test_fem_archive_accepts_production_int32_connectivity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-") as directory:
            path = Path(directory) / "fem.npz"
            _save_npz(path, _fem_arrays())

            loaded = workflow._load_fem_archive(path)

            self.assertEqual(loaded["triangles"].dtype, np.dtype(np.int64))
            np.testing.assert_array_equal(loaded["triangles"], [[0, 1, 2]])
            self.assertEqual(loaded["gate_mode"], "warning")
            self.assertEqual(loaded["boundary_mode"], "bottom_vertical_point_horizontal")
            self.assertEqual(loaded["plane_assumption"], "plane_strain")

    def test_parser_rejects_out_of_range_poisson_ratio(self) -> None:
        parser = workflow.build_parser()
        common = [
            "run",
            "--source-cloud",
            "cloud.npz",
            "--fem-archive",
            "fem.npz",
            "--load-final",
            "1e-4",
            "--max-stagger-iterations",
            "20",
            "--gpu-device-index",
            "0",
            "--delta",
            "0.048",
            "--h-g",
            "0.02",
            "--du-initial",
            "1e-4",
            "--du-min",
            "5e-5",
            "--visibility-tolerance",
            "1e-12",
            "--youngs-modulus",
            "210000",
            "--poisson-ratio",
            "0.3",
            "--k-res",
            "1e-8",
            "--rank-rtol",
            "1e-12",
            "--Gc",
            "2.7",
            "--l0",
            "0.024",
        ]
        arguments = parser.parse_args(common)
        self.assertEqual(arguments.poisson_ratio, 0.3)
        for invalid in ("-1", "0.5", "nan"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                parser.parse_args(common[: common.index("0.3")] + [invalid] + common[common.index("0.3") + 1 :])

        warning = parser.parse_args(
            common
            + [
                "--acceptance-mode",
                "warning",
                "--warning-safety-limit",
                "1e-3",
            ]
        )
        self.assertEqual(
            workflow._acceptance_from_arguments(warning, engineering),
            ("warning", 1.0e-3),
        )
        missing_limit = parser.parse_args(
            common + ["--acceptance-mode", "warning"]
        )
        with self.assertRaisesRegex(ValueError, "safety_limit is required"):
            workflow._acceptance_from_arguments(missing_limit, engineering)

        unbounded = parser.parse_args(
            common
            + [
                "--acceptance-mode",
                "warning",
                "--warning-safety-policy",
                "unbounded",
            ]
        )
        self.assertEqual(
            workflow._acceptance_details_from_arguments(unbounded, engineering),
            ("warning", "unbounded", None),
        )
        with self.assertRaisesRegex(ValueError, "forbidden for unbounded"):
            workflow._acceptance_details_from_arguments(
                parser.parse_args(
                    common
                    + [
                        "--acceptance-mode",
                        "warning",
                        "--warning-safety-policy",
                        "unbounded",
                        "--warning-safety-limit",
                        "1e-3",
                    ]
                ),
                engineering,
            )

        strict_with_limit = parser.parse_args(
            common + ["--warning-safety-limit", "1e-3"]
        )
        with self.assertRaisesRegex(ValueError, "only valid with"):
            workflow._acceptance_from_arguments(strict_with_limit, engineering)

    def test_artifact_paths_reject_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "simple filename stem"):
            workflow.artifact_paths(Path("/tmp/output"), "../escape")

        legacy_paths = workflow.artifact_paths(Path("/tmp/output"), "legacy")
        self.assertNotIn("anderson_diagnostic", legacy_paths)
        diagnostic_paths = workflow.artifact_paths(
            Path("/tmp/output"),
            "anderson",
            anderson_scalar_diagnostic=True,
        )
        self.assertEqual(
            diagnostic_paths["anderson_diagnostic"],
            Path("/tmp/output/anderson.anderson.jsonl"),
        )

    def test_anderson_scalar_binding_is_source_and_configuration_bound(self) -> None:
        source_path, source_metadata = _stabilized_source_fixture()
        del source_path
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
        binding = workflow._anderson_scalar_diagnostic_binding(
            stabilized,
            engineering,
            config,
            source_metadata,
            run_id="diagnostic-test",
            archive_path=Path("/tmp/diagnostic-test.npz"),
        )

        self.assertEqual(binding["source_sha256"], source_metadata["sha256"])
        self.assertEqual(
            binding["expected_archive_paths"],
            [
                "/tmp/diagnostic-test.npz",
                "/tmp/diagnostic-test.failure.npz",
            ],
        )
        self.assertEqual(
            binding["expected_archive_bindings"],
            [
                {
                    "path": "/tmp/diagnostic-test.npz",
                    "path_suffix": ".npz",
                    "kind": "completed",
                    "completion_status": stabilized.STABILIZED_COMPLETION_STATUS,
                    "archive_schema": stabilized.STABILIZED_ARCHIVE_SCHEMA,
                },
                {
                    "path": "/tmp/diagnostic-test.failure.npz",
                    "path_suffix": ".failure.npz",
                    "kind": "failure",
                    "completion_status": stabilized.STABILIZED_FAILURE_STATUS,
                    "archive_schema": stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA,
                },
            ],
        )
        self.assertEqual(
            binding["source_bundle_hash"],
            stabilized._current_stabilized_source_bundle_hash(),
        )
        self.assertEqual(
            binding["configuration_sha256"],
            stabilized._v5_json_sha256(binding["configuration"]),
        )
        self.assertEqual(
            binding["selection_merit_identity"],
            stabilized.STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY,
        )
        _assert_scalar_tree(self, binding)

        residual_config = replace(config, relaxation_policy="anderson_residual")
        residual_binding = workflow._anderson_scalar_diagnostic_binding(
            stabilized,
            engineering,
            residual_config,
            source_metadata,
            run_id="diagnostic-residual-test",
            archive_path=Path("/tmp/diagnostic-residual-test.npz"),
        )
        self.assertEqual(
            residual_binding["selection_merit_identity"],
            stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY,
        )
        self.assertEqual(
            residual_binding["expected_archive_paths"],
            [
                "/tmp/diagnostic-residual-test.npz",
                "/tmp/diagnostic-residual-test.failure.npz",
            ],
        )
        self.assertTrue(
            residual_binding["instrumentation_contract"]["hard_gate_attribution"]
        )
        self.assertTrue(
            {
                "hard_gate_attribution_schema",
                "hard_gate_status",
                "hard_gate_diagnostics",
                "failed_hard_gates",
            }.issubset(residual_binding["required_scalar_fields"])
        )
        self.assertNotEqual(
            binding["selection_merit_identity"],
            residual_binding["selection_merit_identity"],
        )
        _assert_scalar_tree(self, residual_binding)

        regularized_config = replace(
            config,
            relaxation_policy="anderson_residual_regularized_backtracking",
            max_backtracks=stabilized.STABILIZED_ANDERSON_MAX_BACKTRACKS,
            anderson_alpha_reduction=stabilized.STABILIZED_ANDERSON_ALPHA_REDUCTION,
            anderson_regularization_mu=1.25,
        )
        regularized_binding = workflow._anderson_scalar_diagnostic_binding(
            stabilized,
            engineering,
            regularized_config,
            source_metadata,
            run_id="diagnostic-regularized-test",
            archive_path=Path("/tmp/diagnostic-regularized-test.npz"),
        )
        regularized_stabilized_payload = regularized_binding["configuration"][
            "stabilized"
        ]
        self.assertEqual(
            regularized_stabilized_payload["anderson_regularization_mu"], 1.25
        )
        self.assertEqual(
            regularized_binding["selection_merit_identity"],
            stabilized.STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY,
        )
        self.assertIn(
            "anderson_regularization_mu",
            regularized_binding["required_scalar_fields"],
        )
        for field in (
            "anderson_current_residual_l2",
            "anderson_regularization_lambda",
            "anderson_unregularized_ls_residual_l2",
            "anderson_coefficient_l2",
        ):
            self.assertIn(field, regularized_binding["required_scalar_fields"])
        self.assertNotIn(
            "anderson_coefficient_norm",
            regularized_binding["required_scalar_fields"],
        )
        _assert_scalar_tree(self, regularized_binding)

    def test_workflow_reports_residual_hard_gate_marginals_and_overlaps(self) -> None:
        details: dict[str, dict[str, Any]] = {}
        failed = {"force", "moment"}
        for name in stabilized.ANDERSON_RESIDUAL_HARD_GATE_NAMES:
            if name == "finite":
                raw = limit = normalized = None
                scale = stage_metrics = None
                ratio_available = False
                cg_info = cg_status_pass = residual_limit_pass = None
                metric_reason = "boolean_gate"
                passed = name not in failed
            elif name in {"mechanics_before", "mechanics_after"}:
                cg_info = 1 if name in failed else 0
                cg_status_pass = cg_info == 0
                raw = 0.0
                scale = stage_metrics = None
                limit = 1.0
                residual_limit_pass = raw < limit
                normalized = raw / limit
                ratio_available = True
                passed = cg_status_pass and residual_limit_pass
                metric_reason = "cg_info_and_free_residual_limit"
            elif name in {"force", "moment"}:
                raw = 2.0 if name in failed else 0.0
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
                scale = stage_metrics["before"]["scale"]
                limit = stage_metrics["before"]["limit"]
                normalized = stage_metrics["before"]["normalized"]
                ratio_available = True
                cg_info = cg_status_pass = residual_limit_pass = None
                passed = all(item["passed"] for item in stage_metrics.values())
                metric_reason = "stage_worst_ratio_and_conjunction"
            elif name in {
                "damage_range_lower",
                "damage_range_upper",
                "healing",
            }:
                raw = 2.0 if name in failed else 0.0
                scale = stage_metrics = None
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
                raw = 1.0 if name in failed else 0.0
                scale = stage_metrics = None
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
        record = {
            "record_type": "proposal",
            "schema": stabilized.ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
            "candidate_kind": "anderson_accelerated",
            "candidate_map_status": "not_requested",
            "failure_stage": None,
            "complete": True,
            "hard_pass": False,
            "hard_gate_status": "evaluated",
            "hard_gate_attribution_schema": (
                stabilized.ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA
            ),
            "hard_gate_diagnostics": details,
            "failed_hard_gates": [
                name
                for name in stabilized.ANDERSON_RESIDUAL_HARD_GATE_NAMES
                if name in failed
            ],
        }
        with tempfile.TemporaryDirectory(prefix="workflow-hard-gates-") as directory:
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
            report = workflow.analyze_anderson_scalar_diagnostic(
                diagnostic,
                archive_path=archive,
            )
            self.assertEqual(report["proposal_records"], 1)
            self.assertEqual(report["hard_gate_failure_counts"]["force"], 1)
            self.assertEqual(report["hard_gate_overlap_counts"]["force+moment"], 1)

    def test_stabilized_parser_requires_relaxation_flags_and_forbids_warning_flags(self) -> None:
        parser = workflow.build_parser()
        common = [
            "run",
            "--source-cloud", "cloud.npz",
            "--fem-archive", "fem.npz",
            "--continuation-strategy", "stabilized",
            "--relaxation-policy", "sufficient_decrease",
            "--archive-budget-bytes", "500000000",
            "--load-final", "1e-4",
            "--max-stagger-iterations", "20",
            "--gpu-device-index", "0",
            "--delta", "0.048",
            "--h-g", "0.02",
            "--du-initial", "1e-4",
            "--du-min", "5e-5",
            "--visibility-tolerance", "1e-12",
            "--youngs-modulus", "210000",
            "--poisson-ratio", "0.3",
            "--k-res", "1e-8",
            "--rank-rtol", "1e-12",
            "--Gc", "2.7",
            "--l0", "0.024",
            "--omega-initial", "0.5",
            "--omega-min", "0.03125",
            "--omega-reduction", "0.5",
            "--sufficient-decrease", "1e-2",
            "--max-backtracks", "5",
        ]
        arguments = parser.parse_args(common)
        base = workflow._config_from_arguments(arguments, engineering)
        config = workflow._stabilized_config_from_arguments(
            arguments, base, stabilized
        )
        self.assertEqual(
            config.payload(),
            {
                "relaxation_policy": "sufficient_decrease",
                "omega_initial": 0.5,
                "omega_min": 0.03125,
                "omega_reduction": 0.5,
                "sufficient_decrease": 1.0e-2,
                "max_backtracks": 5,
            },
        )

        anderson_common = list(common)
        anderson_common[
            anderson_common.index("--relaxation-policy") + 1
        ] = "anderson"
        anderson_common[
            anderson_common.index("--omega-min") + 1
        ] = "0.5"
        anderson_common[
            anderson_common.index("--max-backtracks") + 1
        ] = "0"
        reduction_index = anderson_common.index("--omega-reduction")
        del anderson_common[reduction_index : reduction_index + 2]
        anderson_common.extend(
            [
                "--anderson-history-depth", "5",
                "--anderson-beta", "0.5",
            ]
        )
        anderson_arguments = parser.parse_args(anderson_common)
        anderson_base = workflow._config_from_arguments(
            anderson_arguments, engineering
        )
        anderson_config = workflow._stabilized_config_from_arguments(
            anderson_arguments, anderson_base, stabilized
        )
        self.assertEqual(
            anderson_config.payload(),
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

        residual_common = list(anderson_common)
        residual_common[
            residual_common.index("--relaxation-policy") + 1
        ] = "anderson_residual"
        residual_arguments = parser.parse_args(residual_common)
        residual_base = workflow._config_from_arguments(
            residual_arguments, engineering
        )
        residual_config = workflow._stabilized_config_from_arguments(
            residual_arguments, residual_base, stabilized
        )
        self.assertEqual(
            residual_config.payload(),
            {
                "relaxation_policy": "anderson_residual",
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": 1.0e-2,
                "max_backtracks": 0,
                "anderson_history_depth": 5,
                "anderson_beta": 0.5,
                "anderson_merit_identity": (
                    stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY
                ),
            },
        )

        backtracking_common = list(residual_common)
        backtracking_common[
            backtracking_common.index("--relaxation-policy") + 1
        ] = "anderson_residual_backtracking"
        backtracking_common[
            backtracking_common.index("--max-backtracks") + 1
        ] = "4"
        backtracking_common.extend(["--anderson-alpha-reduction", "0.5"])
        backtracking_arguments = parser.parse_args(backtracking_common)
        backtracking_base = workflow._config_from_arguments(
            backtracking_arguments, engineering
        )
        backtracking_config = workflow._stabilized_config_from_arguments(
            backtracking_arguments, backtracking_base, stabilized
        )
        self.assertEqual(
            backtracking_config.payload(),
            {
                "relaxation_policy": "anderson_residual_backtracking",
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": 1.0e-2,
                "max_backtracks": 4,
                "anderson_history_depth": 5,
                "anderson_beta": 0.5,
                "anderson_merit_identity": (
                    stabilized.STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY
                ),
                "anderson_alpha_reduction": 0.5,
            },
        )

        regularized_common = list(backtracking_common)
        regularized_common[
            regularized_common.index("--relaxation-policy") + 1
        ] = "anderson_residual_regularized_backtracking"
        regularized_common.extend(["--anderson-regularization-mu", "1.25"])
        regularized_arguments = parser.parse_args(regularized_common)
        regularized_base = workflow._config_from_arguments(
            regularized_arguments, engineering
        )
        regularized_config = workflow._stabilized_config_from_arguments(
            regularized_arguments, regularized_base, stabilized
        )
        self.assertEqual(
            regularized_config.payload(),
            {
                "relaxation_policy": (
                    "anderson_residual_regularized_backtracking"
                ),
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": 1.0e-2,
                "max_backtracks": 4,
                "anderson_history_depth": 5,
                "anderson_beta": 0.5,
                "anderson_merit_identity": (
                    stabilized.STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY
                ),
                "anderson_alpha_reduction": 0.5,
                "anderson_regularization_mu": 1.25,
            },
        )
        missing_mu = [
            value
            for index, value in enumerate(regularized_common)
            if value != "--anderson-regularization-mu"
            and not (
                index > 0
                and regularized_common[index - 1] == "--anderson-regularization-mu"
            )
        ]
        missing_mu_arguments = parser.parse_args(missing_mu)
        missing_mu_base = workflow._config_from_arguments(
            missing_mu_arguments, engineering
        )
        with self.assertRaisesRegex(ValueError, "anderson-regularization-mu"):
            workflow._stabilized_config_from_arguments(
                missing_mu_arguments, missing_mu_base, stabilized
            )
        with self.assertRaises(SystemExit):
            parser.parse_args(regularized_common[:-1] + ["0"])
        with self.assertRaisesRegex(ValueError, "only valid with"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(
                    backtracking_common + ["--anderson-regularization-mu", "1.25"]
                ),
                backtracking_base,
                stabilized,
            )
        missing_alpha = list(backtracking_common)
        alpha_index = missing_alpha.index("--anderson-alpha-reduction")
        del missing_alpha[alpha_index : alpha_index + 2]
        missing_alpha_arguments = parser.parse_args(missing_alpha)
        missing_alpha_base = workflow._config_from_arguments(
            missing_alpha_arguments, engineering
        )
        with self.assertRaisesRegex(ValueError, "anderson-alpha-reduction"):
            workflow._stabilized_config_from_arguments(
                missing_alpha_arguments, missing_alpha_base, stabilized
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(
                    residual_common + ["--anderson-alpha-reduction", "0.5"]
                ),
                residual_base,
                stabilized,
            )
        missing_anderson = list(anderson_common)
        for flag in ("--anderson-history-depth", "--anderson-beta"):
            index = missing_anderson.index(flag)
            del missing_anderson[index : index + 2]
        missing_arguments = parser.parse_args(missing_anderson)
        missing_base = workflow._config_from_arguments(
            missing_arguments, engineering
        )
        with self.assertRaisesRegex(ValueError, "anderson-history-depth"):
            workflow._stabilized_config_from_arguments(
                missing_arguments, missing_base, stabilized
            )
        with self.assertRaisesRegex(ValueError, "omega-reduction.*[Aa]nderson"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(anderson_common + ["--omega-reduction", "0.5"]),
                anderson_base,
                stabilized,
            )

        without_budget = [
            value
            for index, value in enumerate(common)
            if value not in {"--archive-budget-bytes", "500000000"}
            and not (
                index > 0
                and common[index - 1] == "--archive-budget-bytes"
            )
        ]
        missing_budget_arguments = parser.parse_args(without_budget)
        missing_budget_base = workflow._config_from_arguments(
            missing_budget_arguments, engineering
        )
        with self.assertRaisesRegex(ValueError, "archive-budget-bytes"):
            workflow._stabilized_config_from_arguments(
                missing_budget_arguments, missing_budget_base, stabilized
            )
        with self.assertRaisesRegex(ValueError, "warning acceptance"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(common + ["--acceptance-mode", "warning"]),
                base,
                stabilized,
            )
        with self.assertRaisesRegex(ValueError, "warning safety"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(common + ["--warning-safety-limit", "1e-3"]),
                base,
                stabilized,
            )

        progress_arguments = parser.parse_args(
            common + ["--progress-every-sweeps", "4"]
        )
        self.assertEqual(progress_arguments.progress_every_sweeps, 4)

    def test_schedule_cli_rejects_legacy_warning_and_nonresidual_policies(self) -> None:
        parser = workflow.build_parser()
        legacy = parser.parse_args(
            ["run", "--load-schedule-npz", "/tmp/unverified-schedule.npz"]
        )
        with self.assertRaisesRegex(ValueError, "only valid.*stabilized"):
            workflow.run_workflow(legacy)

        common = [
            "run",
            "--source-cloud", "cloud.npz",
            "--fem-archive", "fem.npz",
            "--continuation-strategy", "stabilized",
            "--archive-budget-bytes", "500000000",
            "--load-final", "1e-4",
            "--max-stagger-iterations", "20",
            "--gpu-device-index", "0",
            "--delta", "0.048",
            "--h-g", "0.02",
            "--du-initial", "1e-4",
            "--du-min", "5e-5",
            "--visibility-tolerance", "1e-12",
            "--youngs-modulus", "210000",
            "--poisson-ratio", "0.3",
            "--k-res", "1e-8",
            "--rank-rtol", "1e-12",
            "--Gc", "2.7",
            "--l0", "0.024",
            "--omega-initial", "0.5",
            "--omega-min", "0.5",
            "--max-backtracks", "0",
            "--load-schedule-npz", "/tmp/unverified-schedule.npz",
        ]
        for policy in ("fixed", "anderson"):
            with self.subTest(policy=policy):
                arguments = parser.parse_args(
                    common[:]
                    + ["--relaxation-policy", policy]
                    + ([] if policy == "fixed" else [
                        "--sufficient-decrease", "0.01",
                        "--anderson-history-depth", "5",
                        "--anderson-beta", "0.5",
                    ])
                )
                base = workflow._config_from_arguments(arguments, engineering)
                with self.assertRaisesRegex(ValueError, "load-schedule-npz"):
                    workflow._stabilized_config_from_arguments(
                        arguments, base, stabilized
                    )

        warning_arguments = parser.parse_args(
            common
            + [
                "--relaxation-policy", "fixed",
                "--acceptance-mode", "warning",
                "--warning-safety-policy", "unbounded",
            ]
        )
        warning_base = workflow._config_from_arguments(warning_arguments, engineering)
        with self.assertRaisesRegex(ValueError, "load-schedule-npz"):
            workflow._stabilized_config_from_arguments(
                warning_arguments, warning_base, stabilized
            )

    def test_workflow_summary_retains_schedule_binding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="workflow-schedule-summary-") as name:
            root = Path(name)
            source_path, source_metadata = _stabilized_source_fixture()
            del source_path
            with _schedule_fixture(root) as schedule:
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
                config_payload = workflow._workflow_config_payload(
                    config, "stabilized", engineering
                )
                summary = workflow._run_summary(
                    validated=False,
                    status=stabilized.STABILIZED_COMPLETION_STATUS,
                    elapsed_seconds=0.0,
                    archive_path=root / "missing.npz",
                    replay_path=None,
                    analysis_path=None,
                    plot_report_path=None,
                    state_summary={"state_u_bar": [0.0, 0.5]},
                    source_metadata=source_metadata,
                    device="cpu",
                    config_payload=config_payload,
                    acceptance_mode=stabilized.STABILIZED_ACCEPTANCE_MODE,
                    warning_safety_limit=None,
                    warning_safety_policy=None,
                    field_equations_validated=True,
                )
                self.assertEqual(summary["load_schedule"], schedule.payload())
                self.assertEqual(
                    summary["configuration"]["stabilized"]["load_schedule"],
                    schedule.payload(),
                )

    def test_write_manifest_binds_exact_scheduled_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="workflow-schedule-manifest-") as name:
            root = Path(name)
            _, source_metadata = _stabilized_source_fixture()
            with _schedule_fixture(root) as schedule:
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
                artifact_names = (
                    "archive",
                    "summary",
                    "replay",
                    "failure_archive",
                    "failure_summary",
                    "failure_replay",
                    "analysis",
                    "fields",
                    "stress_strain",
                    "curve_data",
                    "plot_report",
                    "failure_analysis",
                    "failure_fields",
                    "failure_stress_strain",
                    "failure_curve_data",
                    "failure_plot_report",
                )
                paths = {
                    artifact: root / f"{artifact}.artifact"
                    for artifact in artifact_names
                }
                for artifact in ("archive", "summary", "replay"):
                    paths[artifact].write_bytes(b"temporary scheduled artifact")
                manifest_path = root / "scheduled.manifest.json"
                workflow._write_manifest(
                    manifest_path,
                    paths,
                    source_metadata,
                    engineering,
                    config,
                    device="cpu",
                    command=["test-scheduled-manifest"],
                    validated=False,
                    field_equations_validated=True,
                    acceptance_mode=stabilized.STABILIZED_ACCEPTANCE_MODE,
                    warning_safety_limit=None,
                    warning_safety_policy=None,
                    strategy="stabilized",
                    solver_module=stabilized,
                    overwrite=False,
                )

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["load_schedule"], schedule.payload())
                self.assertEqual(
                    manifest["configuration"]["stabilized"]["load_schedule"],
                    schedule.payload(),
                )
                self.assertEqual(
                    manifest["archive_layout"],
                    stabilized.STABILIZED_SCHEDULE_ARCHIVE_LAYOUT,
                )
                self.assertEqual(
                    manifest["replay_contract"],
                    stabilized.STABILIZED_SCHEDULE_REPLAY_CONTRACT,
                )

    def test_fixed_relaxation_policy_requires_one_explicit_candidate(self) -> None:
        parser = workflow.build_parser()
        common = [
            "run",
            "--source-cloud", "cloud.npz",
            "--fem-archive", "fem.npz",
            "--continuation-strategy", "stabilized",
            "--relaxation-policy", "fixed",
            "--archive-budget-bytes", "500000000",
            "--load-final", "1e-4",
            "--max-stagger-iterations", "20",
            "--gpu-device-index", "0",
            "--delta", "0.048",
            "--h-g", "0.02",
            "--du-initial", "1e-4",
            "--du-min", "5e-5",
            "--visibility-tolerance", "1e-12",
            "--youngs-modulus", "210000",
            "--poisson-ratio", "0.3",
            "--k-res", "1e-8",
            "--rank-rtol", "1e-12",
            "--Gc", "2.7",
            "--l0", "0.024",
            "--omega-initial", "0.5",
            "--omega-min", "0.5",
            "--max-backtracks", "0",
        ]
        arguments = parser.parse_args(common)
        base = workflow._config_from_arguments(arguments, engineering)
        config = workflow._stabilized_config_from_arguments(
            arguments, base, stabilized
        )
        self.assertEqual(config.relaxation_policy, "fixed")
        self.assertEqual(config.omega_initial, 0.5)
        self.assertEqual(config.omega_min, 0.5)
        self.assertEqual(config.max_backtracks, 0)
        self.assertIsNone(config.omega_reduction)
        self.assertIsNone(config.sufficient_decrease)

        with self.assertRaisesRegex(ValueError, "omega-reduction.*forbidden"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(common + ["--omega-reduction", "0.5"]),
                base,
                stabilized,
            )
        with self.assertRaisesRegex(ValueError, "sufficient-decrease.*forbidden"):
            workflow._stabilized_config_from_arguments(
                parser.parse_args(common + ["--sufficient-decrease", "0.01"]),
                base,
                stabilized,
            )
        with self.assertRaisesRegex(ValueError, "omega_min == omega_initial"):
            unequal = list(common)
            unequal[unequal.index("--omega-min") + 1] = "0.25"
            workflow._stabilized_config_from_arguments(
                parser.parse_args(unequal),
                base,
                stabilized,
            )

    def test_stabilized_warning_requires_unbounded_fixed_policy(self) -> None:
        parser = workflow.build_parser()
        common = [
            "run",
            "--source-cloud", "cloud.npz",
            "--fem-archive", "fem.npz",
            "--continuation-strategy", "stabilized",
            "--relaxation-policy", "fixed",
            "--archive-budget-bytes", "500000000",
            "--load-final", "1e-4",
            "--max-stagger-iterations", "20",
            "--gpu-device-index", "0",
            "--delta", "0.048",
            "--h-g", "0.02",
            "--du-initial", "1e-4",
            "--du-min", "5e-5",
            "--visibility-tolerance", "1e-12",
            "--youngs-modulus", "210000",
            "--poisson-ratio", "0.3",
            "--k-res", "1e-8",
            "--rank-rtol", "1e-12",
            "--Gc", "2.7",
            "--l0", "0.024",
            "--omega-initial", "0.5",
            "--omega-min", "0.5",
            "--max-backtracks", "0",
        ]

        def build(arguments: list[str]) -> Any:
            parsed = parser.parse_args(arguments)
            base = workflow._config_from_arguments(parsed, engineering)
            return workflow._stabilized_config_from_arguments(
                parsed, base, stabilized
            )

        warning = parser.parse_args(
            common
            + [
                "--acceptance-mode", "warning",
                "--warning-safety-policy", "unbounded",
            ]
        )
        warning_base = workflow._config_from_arguments(warning, engineering)
        warning_config = workflow._stabilized_config_from_arguments(
            warning, warning_base, stabilized
        )
        self.assertEqual(
            warning_config.payload(),
            {
                "relaxation_policy": "fixed",
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": None,
                "max_backtracks": 0,
                "acceptance_mode": stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
            },
        )
        self.assertEqual(
            workflow._acceptance_details_from_arguments(warning, engineering),
            ("warning", "unbounded", None),
        )
        self.assertEqual(
            workflow._acceptance_payload(
                stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
                None,
                "unbounded",
            ),
            {
                "mode": stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
                "warning_safety_policy": "unbounded",
                "warning_safety_limit_present": False,
            },
        )

        for extra, message in (
            ([], "explicit.*unbounded"),
            (["--warning-safety-policy", "bounded"], "explicit.*unbounded"),
            (
                [
                    "--warning-safety-policy", "unbounded",
                    "--warning-safety-limit", "1e-3",
                ],
                "incompatible with unbounded",
            ),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(
                ValueError, message
            ):
                build(common + ["--acceptance-mode", "warning"] + extra)

        sufficient = list(common)
        sufficient[
            sufficient.index("--relaxation-policy") + 1
        ] = "sufficient_decrease"
        sufficient[
            sufficient.index("--omega-min") + 1
        ] = "0.03125"
        sufficient[
            sufficient.index("--max-backtracks") + 1
        ] = "5"
        sufficient.extend(
            [
                "--omega-reduction", "0.5",
                "--sufficient-decrease", "0.01",
                "--acceptance-mode", "warning",
                "--warning-safety-policy", "unbounded",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires.*fixed"):
            build(sufficient)

    def test_legacy_rejects_archive_budget(self) -> None:
        parser = workflow.build_parser()
        arguments = parser.parse_args(
            [
                "run",
                "--source-cloud", "cloud.npz",
                "--fem-archive", "fem.npz",
                "--archive-budget-bytes", "500000000",
                "--load-final", "1e-4",
                "--max-stagger-iterations", "20",
                "--gpu-device-index", "0",
                "--delta", "0.048",
                "--h-g", "0.02",
                "--du-initial", "1e-4",
                "--du-min", "5e-5",
                "--visibility-tolerance", "1e-12",
                "--youngs-modulus", "210000",
                "--poisson-ratio", "0.3",
                "--k-res", "1e-8",
                "--rank-rtol", "1e-12",
                "--Gc", "2.7",
                "--l0", "0.024",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only valid.*stabilized"):
            workflow.run_workflow(arguments)

        arguments.archive_budget_bytes = None
        arguments.progress_every_sweeps = 2
        with self.assertRaisesRegex(ValueError, "progress-every-sweeps.*stabilized"):
            workflow.run_workflow(arguments)

    def test_stabilized_progress_printer_is_compact_sorted_and_flushed(self) -> None:
        with mock.patch("builtins.print") as printer:
            workflow._print_stabilized_progress(
                {
                    "z": np.float64(1.5),
                    "a": np.int64(2),
                    "unavailable": np.float64(np.nan),
                }
            )
        printer.assert_called_once()
        args, kwargs = printer.call_args
        self.assertEqual(
            args,
            ('STABILIZED_PROGRESS {"a":2,"unavailable":null,"z":1.5}',),
        )
        self.assertTrue(kwargs["flush"])

    def test_stabilized_failure_plot_inputs_slice_union_q_to_material_prefix(self) -> None:
        values = _stabilized_completed_arrays()
        n_material = len(_stabilized_source_arrays()["coordinates"])
        metadata = _stabilized_metadata("failure")
        values["archive_schema"] = np.asarray(
            stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA, dtype=np.str_
        )
        values["completion_status"] = np.asarray(
            stabilized.STABILIZED_FAILURE_STATUS, dtype=np.str_
        )
        values["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False), dtype=np.str_
        )
        values["state_union_displacement"][1, 2 * n_material :] = np.arange(
            values["state_union_displacement"].shape[1] - 2 * n_material,
            dtype=np.float64,
        )

        plot_data = workflow._plot_inputs(
            values,
            "failure",
            1.0e-4,
            engineering,
            strategy="stabilized",
        )

        self.assertEqual(plot_data["coordinates"].shape, (n_material, 2))
        self.assertEqual(plot_data["triangles"].shape, (4, 3))
        self.assertEqual(plot_data["q"].shape, (2, 2 * n_material))
        np.testing.assert_array_equal(
            plot_data["q"], values["state_union_displacement"][:, : 2 * n_material]
        )

    def test_stabilized_archive_classification_requires_explicit_acceptance_fields(self) -> None:
        completed = _stabilized_completed_arrays()
        failure = {
            "archive_schema": np.asarray(
                stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA, dtype=np.str_
            ),
            "acceptance_mode": np.asarray(
                stabilized.STABILIZED_ACCEPTANCE_MODE, dtype=np.str_
            ),
            "acceptance_status": np.asarray(
                stabilized.STABILIZED_ACCEPTANCE_STATUS, dtype=np.str_
            ),
            "completion_status": np.asarray(
                stabilized.STABILIZED_FAILURE_STATUS, dtype=np.str_
            ),
            "validated": np.asarray(False, dtype=bool),
            "experimental": np.asarray(True, dtype=bool),
        }
        failure_metadata = _stabilized_metadata("failure")
        self.assertEqual(workflow._archive_kind(completed, engineering), "completed")
        self.assertEqual(workflow._archive_strategy(completed, engineering), "stabilized")
        self.assertEqual(workflow._archive_kind(failure, engineering), "failure")
        self.assertEqual(
            workflow._archive_acceptance_details(
                completed,
                _stabilized_metadata(),
                "completed",
                engineering,
            ),
            ("stabilized", False, None, None),
        )
        with self.assertRaisesRegex(ValueError, "acceptance status"):
            workflow._archive_acceptance_details(
                {key: value for key, value in completed.items() if key != "acceptance_status"},
                _stabilized_metadata(),
                "completed",
                engineering,
            )
        self.assertEqual(
            workflow._archive_acceptance_details(
                failure, failure_metadata, "failure", engineering
            ),
            ("stabilized", False, None, None),
        )

    def test_stabilized_replay_binding_requires_exact_archive_identity_and_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-binding-") as directory:
            archive = Path(directory) / "stabilized.npz"
            values = _stabilized_completed_arrays()
            _save_npz(archive, values)
            metadata = json.loads(str(values["metadata_json"]))
            replay = _bound_stabilized_replay(archive, values)
            valid, expected, error = workflow._stabilized_replay_binding(
                archive, values, metadata, replay, stabilized
            )
            self.assertTrue(valid, error)
            self.assertEqual(replay["archive_sha256"], workflow._sha256_file(archive))
            self.assertEqual(replay["archive_schema"], stabilized.STABILIZED_ARCHIVE_SCHEMA)
            self.assertEqual(
                replay["replay_contract"], stabilized.STABILIZED_REPLAY_CONTRACT
            )
            self.assertEqual(replay["accepted_steps"], 1)
            self.assertEqual(replay["states"], 2)
            self.assertEqual(replay["solver_summary_rows"], 2)
            self.assertEqual(replay["source_bundle_hash"], metadata["code_hash"])
            self.assertEqual(expected, {name: replay[name] for name in expected})

            mutations: dict[str, object] = {
                "archive_sha256": "0" * 64,
                "archive_schema": "forged-schema",
                "replay_contract": "forged-contract",
                "accepted_steps": 2,
                "states": 3,
                "solver_summary_rows": 3,
            }
            for field, value in mutations.items():
                with self.subTest(field=field, kind="wrong"):
                    forged = dict(replay)
                    forged[field] = value
                    valid, _, error = workflow._stabilized_replay_binding(
                        archive, values, metadata, forged, stabilized
                    )
                    self.assertFalse(valid, error)
            for field in (
                "archive_sha256",
                "archive_schema",
                "replay_contract",
                "accepted_steps",
                "states",
                "solver_summary_rows",
            ):
                with self.subTest(field=field, kind="missing"):
                    forged = dict(replay)
                    del forged[field]
                    valid, _, error = workflow._stabilized_replay_binding(
                        archive, values, metadata, forged, stabilized
                    )
                    self.assertFalse(valid, error)

            minimal = {"experimental": True}
            valid, _, error = workflow._stabilized_replay_binding(
                archive, values, metadata, minimal, stabilized
            )
            self.assertFalse(valid, error)

    def test_stabilized_replay_binding_checks_optional_anderson_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-anderson-binding-") as directory:
            root = Path(directory)
            archive = root / "legacy-anderson.npz"
            values = _stabilized_completed_arrays()
            metadata = json.loads(str(values["metadata_json"]))
            metadata["stabilized_config"] = {
                "relaxation_policy": "anderson",
                "omega_initial": 0.5,
                "omega_min": 0.5,
                "omega_reduction": None,
                "sufficient_decrease": 1.0e-2,
                "max_backtracks": 0,
                "anderson_history_depth": 5,
                "anderson_beta": 0.5,
            }
            values["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False),
                dtype=np.str_,
            )
            _save_npz(archive, values)
            report_without_identity = _bound_stabilized_replay(archive, values)
            report_without_identity.pop("relaxation_policy")
            report_without_identity.pop("selection_merit_identity")

            valid, expected, error = workflow._stabilized_replay_binding(
                archive,
                values,
                metadata,
                report_without_identity,
                stabilized,
            )
            self.assertTrue(valid, error)
            self.assertEqual(expected["relaxation_policy"], "anderson")
            self.assertEqual(
                expected["selection_merit_identity"],
                stabilized.STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY,
            )

            matching_report = dict(report_without_identity)
            matching_report.update(
                {
                    "relaxation_policy": "anderson",
                    "selection_merit_identity": (
                        stabilized.STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY
                    ),
                }
            )
            valid, _, error = workflow._stabilized_replay_binding(
                archive, values, metadata, matching_report, stabilized
            )
            self.assertTrue(valid, error)

            residual_report = dict(report_without_identity)
            residual_report.update(
                {
                    "relaxation_policy": "anderson_residual",
                    "selection_merit_identity": (
                        stabilized.STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY
                    ),
                }
            )
            valid, _, error = workflow._stabilized_replay_binding(
                archive, values, metadata, residual_report, stabilized
            )
            self.assertFalse(valid)
            self.assertIn("does not match archive", error or "")

    def test_stabilized_replay_report_preserves_lean_replay_semantics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-stabilized-") as directory:
            root = Path(directory)
            archive = root / "stabilized.npz"
            report = root / "stabilized.replay.json"
            values = _stabilized_completed_arrays()
            _save_npz(archive, values)
            replay = _bound_stabilized_replay(archive, values)
            metadata = json.loads(str(values["metadata_json"]))
            with mock.patch.object(
                stabilized,
                "replay_stabilized_engineering_grip_archive",
                return_value=replay,
            ):
                payload = workflow.replay_archive(archive, report, overwrite=False)
            self.assertEqual(payload["schema"], workflow.STABILIZED_REPLAY_REPORT_SCHEMA)
            self.assertTrue(payload["experimental"])
            self.assertTrue(payload["field_equations_validated"])
            self.assertFalse(payload["path_replayed"])
            self.assertFalse(payload["rejected_trial_path_available"])
            self.assertNotIn("validated", payload)
            self.assertEqual(payload["continuation_strategy"], "stabilized")
            self.assertTrue(payload["replay_binding_valid"])
            self.assertEqual(
                payload["anderson_provenance"],
                stabilized._v5_anderson_provenance("sufficient_decrease"),
            )
            self.assertEqual(payload["archive_schema"], stabilized.STABILIZED_ARCHIVE_SCHEMA)
            self.assertEqual(
                payload["replay_contract"], stabilized.STABILIZED_REPLAY_CONTRACT
            )
            self.assertEqual(payload["source_bundle_hash"], metadata["code_hash"])
            written = json.loads(report.read_text(encoding="utf-8"))
            self.assertNotIn("validated", written)
            self.assertFalse(written["path_replayed"])

            minimal_report = root / "stabilized-minimal.replay.json"
            with mock.patch.object(
                stabilized,
                "replay_stabilized_engineering_grip_archive",
                return_value={
                    "experimental": True,
                    "field_equations_validated": False,
                    "path_replayed": False,
                    "rejected_trial_path_available": False,
                },
            ):
                minimal_payload = workflow.replay_archive(
                    archive, minimal_report, overwrite=False
                )
            self.assertFalse(minimal_payload["field_equations_validated"])
            self.assertNotIn("validated", minimal_payload)
            self.assertFalse(minimal_payload["replay_binding_valid"])

            with self.assertRaisesRegex(ValueError, "provenance"):
                workflow._stabilized_replay_semantics(
                    {
                        "anderson_provenance": {
                            "status": "recorded_not_reconstructed",
                            "selection_recorded": True,
                        }
                    }
                )

            failed = _stabilized_completed_arrays()
            failed_archive = root / "stabilized-failed-gate.npz"
            failed_report = root / "stabilized-failed-gate.replay.json"
            _save_npz(failed_archive, failed)
            failed_replay = _bound_stabilized_replay(failed_archive, failed)
            failed_replay["field_equations_validated"] = False
            with mock.patch.object(
                stabilized,
                "replay_stabilized_engineering_grip_archive",
                return_value=failed_replay,
            ):
                failed_payload = workflow.replay_archive(
                    failed_archive, failed_report, overwrite=False
                )
            self.assertFalse(failed_payload["field_equations_validated"])
            self.assertNotIn("validated", failed_payload)

            failure_values = _stabilized_completed_arrays()
            failure_metadata = _stabilized_metadata("failure")
            failure_values.update(
                {
                    "archive_schema": np.asarray(
                        stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA, dtype=np.str_
                    ),
                    "completion_status": np.asarray(
                        stabilized.STABILIZED_FAILURE_STATUS, dtype=np.str_
                    ),
                    "metadata_json": np.asarray(
                        json.dumps(
                            failure_metadata, sort_keys=True, allow_nan=False
                        ),
                        dtype=np.str_,
                    ),
                }
            )
            failure_archive = root / "stabilized.failure.npz"
            failure_report = root / "stabilized.failure.replay.json"
            _save_npz(failure_archive, failure_values)
            failure_replay = _bound_stabilized_replay(failure_archive, failure_values)
            with mock.patch.object(
                stabilized,
                "replay_stabilized_engineering_grip_archive",
                return_value=failure_replay,
            ):
                failure_payload = workflow.replay_archive(
                    failure_archive, failure_report, overwrite=False
                )
            self.assertTrue(failure_payload["field_equations_validated"])
            self.assertNotIn("validated", failure_payload)

    def test_stabilized_analyze_and_plot_reports_preserve_replay_provenance(self) -> None:
        values = _stabilized_completed_arrays(sweep_index=7)
        coordinates = _stabilized_source_arrays()["coordinates"]
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-stabilized-report-") as directory:
            root = Path(directory)
            archive = root / "stabilized.npz"
            analysis_path = root / "stabilized.analysis.json"
            _save_npz(archive, values)
            replay = _bound_stabilized_replay(archive, values)
            with mock.patch.object(
                stabilized,
                "replay_stabilized_engineering_grip_archive",
                return_value=replay,
            ), mock.patch.object(workflow, "_enable_stabilized_x64") as enable_x64:
                analysis = workflow.analyze_archive(
                    archive, analysis_path, overwrite=False, replay_result=None
                )
            enable_x64.assert_called_once_with()
            self.assertEqual(analysis["continuation_strategy"], "stabilized")
            self.assertTrue(analysis["experimental"])
            self.assertTrue(analysis["field_equations_validated"])
            self.assertFalse(analysis["path_replayed"])
            self.assertNotIn("validated", analysis)
            self.assertEqual(analysis["independent_replay"], replay)
            self.assertTrue(analysis["replay_binding_valid"])
            self.assertEqual(
                analysis["gate_ratios"]["damage"]["sweep_index"], 7
            )
            self.assertEqual(
                analysis["gate_ratios"]["damage"]["target_u_bar"], 1.0e-4
            )
            self.assertEqual(analysis["replay_binding"], {
                name: replay[name]
                for name in (
                    "archive_sha256",
                    "archive_schema",
                    "archive_layout",
                    "replay_contract",
                    "acceptance_mode",
                    "acceptance_status",
                    "completion_status",
                    "source_bundle_hash",
                    "source_path",
                    "source_sha256",
                    "accepted_steps",
                    "states",
                    "solver_summary_rows",
                    "anderson_provenance",
                )
            })

            plot_data = {
                "coordinates": coordinates,
                "triangles": np.asarray([[0, 1, 2]], dtype=np.int64),
                "loads": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "q": np.zeros((2, 2 * len(coordinates)), dtype=np.float64),
                "damage": np.zeros((2, len(coordinates)), dtype=np.float64),
                "reaction": np.asarray([0.0, 1.0], dtype=np.float64),
                "target_index": 1,
                "field_equations_validated": False,
                "kind": "completed",
                "n_material": len(coordinates),
            }
            fem = {
                **_fem_arrays(),
                "coords": coordinates,
                "triangles": np.asarray([[0, 1, 2]], dtype=np.int32),
                "u": np.zeros((2, len(coordinates), 2), dtype=np.float64),
                "d": np.zeros((2, len(coordinates)), dtype=np.float64),
                "response_load": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "nominal_strain": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "nominal_stress": np.asarray([0.0, 1.0], dtype=np.float64),
                "snapshot_u_bar": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "snapshot_load": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "reaction": np.asarray([0.0, 1.0], dtype=np.float64),
                "path": "fem.npz",
                "sha256": "fem-sha256",
                "Lx": 1.0,
                "Ly": 1.0,
            }
            plot_report_path = root / "stabilized.plot.json"
            with (
                mock.patch.object(workflow, "_plot_inputs", return_value=plot_data),
                mock.patch.object(workflow, "_load_fem_archive", return_value=fem),
                mock.patch.object(workflow, "_write_figure"),
            ):
                plot = workflow.plot_archive(
                    archive,
                    root / "fem.npz",
                    target_load=1.0e-4,
                    fields_path=root / "fields.png",
                    curve_path=root / "curve.png",
                    curve_data_path=root / "curve.npz",
                    report_path=plot_report_path,
                    overwrite=False,
                    replay_result=replay,
                )
            self.assertEqual(plot["continuation_strategy"], "stabilized")
            self.assertTrue(plot["experimental"])
            self.assertTrue(plot["field_equations_validated"])
            self.assertFalse(plot["path_replayed"])
            self.assertNotIn("validated", plot)
            self.assertEqual(plot["independent_replay"], replay)
            self.assertTrue(plot["replay_binding_valid"])

    def test_gate_ratio_provenance_uses_stored_failure_summary_location(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        values = _stabilized_completed_arrays(sweep_index=7)
        values["failure_summary_json"] = _stabilized_failure_summary_json(
            1.0e-4, sweep_index=13
        )
        failure_row = json.loads(str(values["failure_summary_json"]))["solver_summary"]

        diagnostics = workflow._failure_diagnostic_payloads(values)
        ratios = workflow._analysis_gate_ratios(diagnostics, config)

        self.assertEqual(ratios["damage"]["sweep_index"], 13)
        self.assertEqual(ratios["damage"]["target_u_bar"], 1.0e-4)
        self.assertNotEqual(ratios["damage"]["sweep_index"], 2)

        malformed = dict(failure_row)
        malformed["sweep_index"] = True
        with self.assertRaisesRegex(ValueError, "sweep_index must be an integer"):
            workflow._analysis_gate_ratios([malformed], config)

        malformed = dict(failure_row)
        malformed["sweep_index"] = config.max_stagger_iterations
        with self.assertRaisesRegex(ValueError, "sweep_index is out of range"):
            workflow._analysis_gate_ratios([malformed], config)

        malformed = dict(failure_row)
        malformed["target_u_bar"] = 2.0e-4
        with self.assertRaisesRegex(ValueError, "target/load fields disagree"):
            workflow._analysis_gate_ratios([malformed], config)

    def test_stabilized_failure_uses_state_arrays_for_reports(self) -> None:
        values = _stabilized_completed_arrays(sweep_index=7)
        coordinates = _stabilized_source_arrays()["coordinates"]
        values.update(
            {
                "archive_schema": np.asarray(
                    stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA, dtype=np.str_
                ),
                "completion_status": np.asarray(
                    stabilized.STABILIZED_FAILURE_STATUS, dtype=np.str_
                ),
                "metadata_json": np.asarray(
                    json.dumps(_stabilized_metadata("failure"), sort_keys=True),
                    dtype=np.str_,
                ),
                "failure_type": np.asarray(
                    "StabilizedEngineeringGripContinuationFailure", dtype=np.str_
                ),
                "last_committed_u_bar": np.asarray(1.0e-4, dtype=np.float64),
                "last_committed_damage_max": np.asarray(0.0, dtype=np.float64),
                "last_committed_history_max": np.asarray(0.0, dtype=np.float64),
                "failure_message": np.asarray("stabilized failure", dtype=np.str_),
                "failure_summary_json": _stabilized_failure_summary_json(
                    1.0e-4, sweep_index=13
                ),
            }
        )
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-stabilized-failure-") as directory:
            root = Path(directory)
            archive = root / "stabilized.failure.npz"
            _save_npz(archive, values)
            summary = workflow._failure_state_summary(values)
            self.assertEqual(summary["state_u_bar"], [0.0, 1.0e-4])
            self.assertEqual(summary["state_top_reaction"], [0.0, 1.0])

            replay = _bound_stabilized_replay(archive, values)
            with mock.patch.object(
                stabilized,
                "replay_stabilized_engineering_grip_archive",
                return_value=replay,
            ):
                analysis = workflow.analyze_archive(
                    archive,
                    root / "failure.analysis.json",
                    overwrite=False,
                    replay_result=replay,
                )
            self.assertEqual(analysis["archive_kind"], "failure")
            self.assertEqual(analysis["continuation_strategy"], "stabilized")
            self.assertTrue(analysis["field_equations_validated"])
            self.assertFalse(analysis["path_replayed"])
            self.assertNotIn("validated", analysis)
            self.assertEqual(
                analysis["gate_ratios"]["damage"]["sweep_index"], 13
            )
            self.assertEqual(
                analysis["gate_ratios"]["damage"]["target_u_bar"], 1.0e-4
            )

            plot_data = {
                "coordinates": coordinates,
                "triangles": np.asarray([[0, 1, 2]], dtype=np.int64),
                "loads": values["state_u_bar"],
                "q": values["state_union_displacement"],
                "damage": values["state_material_damage"],
                "reaction": np.asarray([0.0, 1.0], dtype=np.float64),
                "target_index": 1,
                "field_equations_validated": False,
                "kind": "failure",
                "n_material": len(coordinates),
            }
            fem = {
                **_fem_arrays(),
                "coords": coordinates,
                "triangles": np.asarray([[0, 1, 2]], dtype=np.int32),
                "u": np.zeros((2, len(coordinates), 2), dtype=np.float64),
                "d": np.zeros((2, len(coordinates)), dtype=np.float64),
                "response_load": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "nominal_strain": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "nominal_stress": np.asarray([0.0, 1.0], dtype=np.float64),
                "snapshot_u_bar": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "snapshot_load": np.asarray([0.0, 1.0e-4], dtype=np.float64),
                "path": "fem.npz",
                "sha256": "fem-sha256",
                "Lx": 1.0,
                "Ly": 1.0,
            }
            with (
                mock.patch.object(workflow, "_plot_inputs", return_value=plot_data),
                mock.patch.object(workflow, "_load_fem_archive", return_value=fem),
                mock.patch.object(workflow, "_write_figure"),
            ):
                plot = workflow.plot_archive(
                    archive,
                    root / "fem.npz",
                    target_load=1.0e-4,
                    fields_path=root / "failure.fields.png",
                    curve_path=root / "failure.curve.png",
                    curve_data_path=root / "failure.curve.npz",
                    report_path=root / "failure.plot.json",
                    overwrite=False,
                    replay_result=replay,
                )
            self.assertEqual(plot["archive_kind"], "failure")
            self.assertEqual(plot["continuation_strategy"], "stabilized")
            self.assertTrue(plot["field_equations_validated"])
            self.assertFalse(plot["path_replayed"])
            self.assertNotIn("validated", plot)

    def test_stabilized_archive_loader_rejects_object_arrays(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-object-") as directory:
            archive = Path(directory) / "object.npz"
            _save_npz(
                archive,
                {
                    "archive_schema": np.asarray(stabilized.STABILIZED_ARCHIVE_SCHEMA),
                    "forbidden": np.asarray([object()], dtype=object),
                },
            )
            with self.assertRaisesRegex(ValueError, "Object arrays cannot be loaded"):
                workflow._load_npz(archive, "stabilized archive")

    def test_archive_classification_keeps_warning_failure_audits_nonvalidated(self) -> None:
        completed = {
            "archive_schema": np.asarray(engineering.WARNING_ARCHIVE_SCHEMA, dtype=np.str_),
            "acceptance_mode": np.asarray("warning", dtype=np.str_),
        }
        failure = {
            "archive_schema": np.asarray(workflow.FAILURE_ARCHIVE_SCHEMA, dtype=np.str_),
            "acceptance_mode": np.asarray("warning", dtype=np.str_),
        }
        strict_failure = {
            "archive_schema": np.asarray(workflow.FAILURE_ARCHIVE_SCHEMA, dtype=np.str_),
            "acceptance_mode": np.asarray("strict", dtype=np.str_),
        }
        self.assertEqual(workflow._archive_kind(completed, engineering), "completed")
        self.assertEqual(workflow._archive_kind(failure, engineering), "failure")
        self.assertEqual(workflow._archive_kind(strict_failure, engineering), "failure")

        completed_mode = workflow._archive_acceptance(
            completed,
            {"warning_safety_limit": 1.0e-3},
            "completed",
            engineering,
        )
        self.assertEqual(completed_mode, ("warning", True, 1.0e-3))
        failure_mode = workflow._archive_acceptance(
            failure,
            {"warning_safety_limit": 1.0e-3},
            "failure",
            engineering,
        )
        self.assertEqual(failure_mode, ("warning", True, 1.0e-3))
        self.assertEqual(
            workflow._archive_acceptance(
                strict_failure, {}, "failure", engineering
            ),
            ("strict", False, None),
        )

    def test_warning_replay_report_is_not_validated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-warning-") as directory:
            archive = Path(directory) / "warning.npz"
            report = Path(directory) / "warning.replay.json"
            _save_npz(
                archive,
                {
                    "archive_schema": np.asarray(
                        engineering.WARNING_ARCHIVE_SCHEMA, dtype=np.str_
                    ),
                    "acceptance_mode": np.asarray("warning", dtype=np.str_),
                    "metadata_json": np.asarray(
                        json.dumps({"warning_safety_limit": 1.0e-3}), dtype=np.str_
                    ),
                },
            )
            with mock.patch.object(
                engineering,
                "replay_engineering_grip_archive",
                return_value={"replayed": True, "validated": False},
            ):
                payload = workflow.replay_archive(
                    archive, report, overwrite=False
                )
            self.assertTrue(payload["replay"]["replayed"])
            self.assertFalse(payload["validated"])
            written = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(written["validated"])

    def test_unbounded_archive_acceptance_payload_is_explicit_and_unvalidated(self) -> None:
        values = {
            "archive_schema": np.asarray(
                engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA, dtype=np.str_
            ),
            "acceptance_mode": np.asarray("warning", dtype=np.str_),
            "warning_safety_policy": np.asarray("unbounded", dtype=np.str_),
            "warning_safety_limit_present": np.asarray(False, dtype=bool),
        }
        metadata = {
            "schema": engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA,
            "acceptance_mode": "warning",
            "warning_safety_policy": "unbounded",
            "warning_safety_limit_present": False,
        }
        self.assertEqual(
            workflow._archive_kind(values, engineering), "completed"
        )
        self.assertEqual(
            workflow._archive_acceptance_details(
                values, metadata, "completed", engineering
            ),
            ("warning", True, "unbounded", None),
        )
        self.assertEqual(
            workflow._acceptance_payload("warning", None, "unbounded"),
            {
                "mode": "warning",
                "warning_safety_policy": "unbounded",
                "warning_safety_limit_present": False,
            },
        )

        forged = dict(values)
        forged["warning_safety_limit_present"] = np.asarray(True, dtype=bool)
        with self.assertRaisesRegex(ValueError, "limit presence"):
            workflow._archive_acceptance_details(
                forged, metadata, "completed", engineering
            )

    def test_plot_report_uses_response_index_not_snapshot_index(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        target_load = 0.0067
        plot_data = {
            "coordinates": coordinates,
            "triangles": triangles,
            "loads": np.asarray([0.0, target_load], dtype=np.float64),
            "q": np.zeros((2, 6), dtype=np.float64),
            "damage": np.zeros((2, 3), dtype=np.float64),
            "reaction": np.asarray([0.0, 12.0], dtype=np.float64),
            "target_index": 1,
            "field_equations_validated": True,
            "kind": "completed",
            "n_material": 3,
        }
        fem = {
            "coords": coordinates,
            "triangles": triangles,
            # The response grid deliberately has a different index from the
            # exact target field snapshot below.
            "response_load": np.asarray([0.0, 0.003, target_load], dtype=np.float64),
            "nominal_strain": np.asarray([0.0, 0.003, target_load], dtype=np.float64),
            "nominal_stress": np.asarray([0.0, 535.537, 5.55723], dtype=np.float64),
            "snapshot_load": np.asarray([0.0, target_load], dtype=np.float64),
            "u": np.zeros((2, 3, 2), dtype=np.float64),
            "d": np.zeros((2, 3), dtype=np.float64),
            "Lx": 1.0,
            "Ly": 1.0,
            "path": "fem.npz",
            "sha256": "fem-sha256",
            "gate_mode": "warning",
            "boundary_mode": "bottom_vertical_point_horizontal",
            "plane_assumption": "plane_strain",
        }
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-plot-") as directory:
            root = Path(directory)
            archive = root / "archive.npz"
            _save_npz(
                archive,
                {
                    "archive_schema": np.asarray(
                        engineering.ARCHIVE_SCHEMA, dtype=np.str_
                    ),
                    "acceptance_mode": np.asarray("strict", dtype=np.str_),
                    "metadata_json": np.asarray("{}", dtype=np.str_),
                },
            )
            report_path = root / "plot.json"
            curve_data_path = root / "curve.npz"
            with (
                mock.patch.object(workflow, "_plot_inputs", return_value=plot_data),
                mock.patch.object(workflow, "_load_fem_archive", return_value=fem),
            ):
                report = workflow.plot_archive(
                    archive,
                    root / "fem.npz",
                    target_load=target_load,
                    fields_path=root / "fields.png",
                    curve_path=root / "curve.png",
                    curve_data_path=curve_data_path,
                    report_path=report_path,
                    overwrite=False,
                )

            self.assertEqual(report["fem_snapshot_index"], 1)
            self.assertEqual(report["fem_response_index"], 2)
            self.assertEqual(
                report["fem_response_provenance"],
                {
                    "method": "exact_response_load",
                    "response_index": 2,
                    "response_load": target_load,
                },
            )
            self.assertAlmostEqual(
                report["fem_nominal_stress_at_target"], 5.55723, places=12
            )
            self.assertAlmostEqual(
                report["final_stress_relative_difference"],
                abs(12.0 - 5.55723) / 5.55723,
                places=12,
            )
            with np.load(curve_data_path, allow_pickle=False) as curve:
                np.testing.assert_array_equal(
                    curve["fem_nominal_stress"], fem["nominal_stress"]
                )

    def test_fem_target_response_interpolates_response_grid(self) -> None:
        stress, response_index, provenance = workflow._fem_target_response(
            np.asarray([0.0, 0.005, 0.01], dtype=np.float64),
            np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            0.0067,
        )

        self.assertIsNone(response_index)
        self.assertIsNotNone(stress)
        assert stress is not None
        self.assertAlmostEqual(stress, 13.4, places=12)
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(provenance["method"], "linear_interpolation")
        self.assertEqual(provenance["implementation"], "numpy.interp")
        self.assertEqual(provenance["lower_response_index"], 1)
        self.assertEqual(provenance["upper_response_index"], 2)

    def test_completed_plot_without_exact_fem_snapshot_keeps_response_comparison(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64
        )
        triangles = np.asarray([[0, 1, 2]], dtype=np.int64)
        target_load = 0.0067
        plot_data = {
            "coordinates": coordinates,
            "triangles": triangles,
            "loads": np.asarray([0.0, target_load], dtype=np.float64),
            "q": np.zeros((2, 6), dtype=np.float64),
            "damage": np.zeros((2, 3), dtype=np.float64),
            "reaction": np.asarray([0.0, 12.0], dtype=np.float64),
            "target_index": 1,
            "field_equations_validated": True,
            "kind": "completed",
            "n_material": 3,
        }
        fem = {
            "coords": coordinates,
            "triangles": triangles,
            "response_load": np.asarray([0.0, 0.003, target_load], dtype=np.float64),
            "nominal_strain": np.asarray([0.0, 0.003, target_load], dtype=np.float64),
            "nominal_stress": np.asarray([0.0, 535.537, 5.55723], dtype=np.float64),
            # The response reaches the target, but the saved field snapshots do not.
            "snapshot_load": np.asarray([0.0, 0.003], dtype=np.float64),
            "u": np.zeros((2, 3, 2), dtype=np.float64),
            "d": np.zeros((2, 3), dtype=np.float64),
            "Lx": 1.0,
            "Ly": 1.0,
            "path": "fem.npz",
            "sha256": "fem-sha256",
            "gate_mode": "warning",
            "boundary_mode": "bottom_vertical_point_horizontal",
            "plane_assumption": "plane_strain",
        }
        with tempfile.TemporaryDirectory(prefix="engineering-workflow-plot-no-snapshot-") as directory:
            root = Path(directory)
            archive = root / "archive.npz"
            _save_npz(
                archive,
                {
                    "archive_schema": np.asarray(
                        engineering.ARCHIVE_SCHEMA, dtype=np.str_
                    ),
                    "acceptance_mode": np.asarray("strict", dtype=np.str_),
                    "metadata_json": np.asarray("{}", dtype=np.str_),
                },
            )
            with (
                mock.patch.object(workflow, "_plot_inputs", return_value=plot_data),
                mock.patch.object(workflow, "_load_fem_archive", return_value=fem),
                mock.patch.object(workflow, "_write_figure"),
            ):
                report = workflow.plot_archive(
                    archive,
                    root / "fem.npz",
                    target_load=target_load,
                    fields_path=root / "fields.png",
                    curve_path=root / "curve.png",
                    curve_data_path=root / "curve.npz",
                    report_path=root / "plot.json",
                    overwrite=False,
                )

        self.assertIsNone(report["fem_snapshot_index"])
        self.assertFalse(report["fem_snapshot_available"])
        self.assertIsNone(report["field_ranges"]["fem"])
        self.assertTrue(report["fem_curve_comparison_available"])
        self.assertEqual(report["fem_response_index"], 2)
        self.assertEqual(report["fem_response_provenance"]["method"], "exact_response_load")
        self.assertAlmostEqual(report["fem_nominal_stress_at_target"], 5.55723)


def _tiny_material_cloud():
    n_side = 31
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
    delta = 0.048
    pairs: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
            if 0.0 < distance < delta:
                pairs.extend(((first, second), (second, first)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    return coordinates, triangles_array, volumes, bottom, top, edges, delta


class StabilizedWorkflowGuardTest(unittest.TestCase):
    """The stabilized storage preflight must gate the GPU adapter."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        coordinates, triangles, volumes, bottom, top, edges, delta = (
            _tiny_material_cloud()
        )
        cls.source_directory = tempfile.TemporaryDirectory(
            prefix="engineering-guard-source-"
        )
        cls.source_path = Path(cls.source_directory.name) / "source.npz"
        _save_npz(
            cls.source_path,
            {
                "coordinates": coordinates,
                "triangles": triangles,
                "volumes": volumes,
                "bottom": bottom,
                "top": top,
                "anchor_particle": np.asarray(0, dtype=np.int64),
                "d048": edges,
            },
        )
        _, cls.source_metadata = workflow._load_source_cloud(cls.source_path, delta)
        cls.geometry = build_union_geometry_from_arrays(
            coordinates,
            triangles,
            volumes,
            bottom,
            top,
            0,
            edges,
            delta,
            0.024,
            0.0,
        )
        cls.model = engineering.build_engineering_grip_model(
            cls.geometry, 2.0, 0.3, 1.0e-8, 1.0e-12, 1.7, 0.65
        )
        cls.config = stabilized.StabilizedEngineeringGripConfig.smoke()
        cls.result = stabilized.run_stabilized_engineering_grip(
            cls.model, cls.config
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.source_directory.cleanup()
        jax.config.update("jax_enable_x64", cls.old_x64)

    @property
    def _cloud(self) -> dict[str, np.ndarray]:
        return {
            "coordinates": self.geometry.material_reference_coordinates,
            "triangles": self.geometry.material_reference_triangles,
        }

    def _fem(self) -> dict[str, np.ndarray]:
        return {
            "coords": self.geometry.material_reference_coordinates,
            "triangles": self.geometry.material_reference_triangles,
        }

    def _arguments(self, root: Path) -> Any:
        parser = workflow.build_parser()
        return parser.parse_args(
            [
                "run",
                "--source-cloud", str(self.source_path),
                "--fem-archive", str(root / "fem.npz"),
                "--continuation-strategy", "stabilized",
                "--relaxation-policy", "sufficient_decrease",
                "--output-directory", str(root),
                "--run-id", "guard",
                "--archive-budget-bytes", "500000000",
                "--progress-every-sweeps", "2",
                "--load-final", "1e-4",
                "--max-stagger-iterations", "20",
                "--gpu-device-index", "0",
                "--delta", "0.048",
                "--h-g", "0.024",
                "--du-initial", "1e-4",
                "--du-min", "5e-5",
                "--visibility-tolerance", "1e-12",
                "--youngs-modulus", "2.0",
                "--poisson-ratio", "0.3",
                "--k-res", "1e-8",
                "--rank-rtol", "1e-12",
                "--Gc", "1.7",
                "--l0", "0.65",
                "--omega-initial", "0.5",
                "--omega-min", "0.03125",
                "--omega-reduction", "0.5",
                "--sufficient-decrease", "1e-2",
                "--max-backtracks", "5",
            ]
        )

    def test_storage_preflight_failure_precedes_gpu_adapter_and_continuation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-guard-fail-") as directory:
            root = Path(directory)
            arguments = self._arguments(root)
            order: list[str] = []

            def failing_preflight(*args: Any, **kwargs: Any) -> None:
                order.append("preflight")
                raise OSError("insufficient free storage for stabilized archive preflight")

            with (
                mock.patch.object(workflow, "_load_fem_archive", return_value=self._fem()),
                mock.patch.object(workflow, "_preflight_artifacts"),
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(
                        self.model,
                        self.config.base,
                        self._cloud,
                        self.source_metadata,
                    ),
                ),
                mock.patch.object(
                    workflow, "_preflight_stabilized_storage", side_effect=failing_preflight
                ),
                mock.patch.object(
                    workflow,
                    "_construct_adapter",
                    side_effect=lambda *args, **kwargs: order.append("adapter"),
                ),
                mock.patch.object(
                    stabilized,
                    "run_stabilized_engineering_grip",
                    side_effect=lambda *args, **kwargs: order.append("continuation"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "insufficient free storage"):
                    workflow._run_stabilized_workflow(arguments)
            self.assertEqual(order, ["preflight"])
            self.assertFalse((root / "guard.npz").exists())

    def test_storage_preflight_runs_before_adapter_and_is_stored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-guard-pass-") as directory:
            root = Path(directory)
            arguments = self._arguments(root)
            order: list[str] = []
            real_preflight = workflow._preflight_stabilized_storage

            def recording_preflight(*args: Any, **kwargs: Any) -> Any:
                order.append("preflight")
                return real_preflight(*args, **kwargs)

            def recording_adapter(*args: Any, **kwargs: Any) -> tuple[Any, str]:
                order.append("adapter")
                return object(), "test:gpu"

            def recording_controller(*args: Any, **kwargs: Any) -> Any:
                order.append("continuation")
                self.assertEqual(kwargs["archive_budget_bytes"], 500000000)
                self.assertEqual(kwargs["archive_output_path"], root / "guard.npz")
                self.assertIs(kwargs["progress_callback"], workflow._print_stabilized_progress)
                self.assertEqual(kwargs["progress_every_sweeps"], 2)
                return self.result

            with (
                mock.patch.object(workflow, "_load_fem_archive", return_value=self._fem()),
                mock.patch.object(workflow, "_preflight_artifacts"),
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(
                        self.model,
                        self.config.base,
                        self._cloud,
                        self.source_metadata,
                    ),
                ),
                mock.patch.object(
                    workflow,
                    "_preflight_stabilized_storage",
                    side_effect=recording_preflight,
                ),
                mock.patch.object(
                    workflow, "_construct_adapter", side_effect=recording_adapter
                ),
                mock.patch.object(
                    stabilized,
                    "run_stabilized_engineering_grip",
                    side_effect=recording_controller,
                ),
                mock.patch.object(
                    workflow,
                    "replay_archive",
                    return_value={
                        "experimental": True,
                        "field_equations_validated": True,
                        "path_replayed": False,
                        "rejected_trial_path_available": False,
                        "replay": {"field_equations_validated": True},
                    },
                ),
                mock.patch.object(workflow, "analyze_archive", return_value={}),
                mock.patch.object(workflow, "plot_archive", return_value={}),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    code = workflow._run_stabilized_workflow(arguments)
            self.assertEqual(code, 0)
            self.assertEqual(order, ["preflight", "adapter", "continuation"])
            output = captured.getvalue()
            self.assertIn("STABILIZED_ARCHIVE_STORAGE_PREFLIGHT", output)
            preflight_line = (
                output.split("STABILIZED_ARCHIVE_STORAGE_PREFLIGHT", 1)[1]
                .strip()
                .splitlines()[0]
            )
            printed = json.loads(preflight_line)
            self.assertTrue(printed["fits"])
            self.assertEqual(printed["static_bound_kind"], "rebuilt_from_source")
            estimate = stabilized.estimate_stabilized_archive_bytes(
                self.model.n_material,
                self.model.n_dofs,
                self.config,
            )
            self.assertEqual(
                printed["physical_rows"],
                {
                    "states": estimate["n_states"],
                    "accepted_steps": estimate["accepted_steps"],
                },
            )
            self.assertEqual(
                set(printed["scalar_rows"]),
                {"solver_summaries"},
            )

            summary = json.loads(
                (root / "guard.summary.json").read_text(encoding="utf-8")
            )
            stored = summary["archive_storage_preflight"]
            self.assertEqual(stored["required_free_bytes"], estimate["required_free_bytes"])
            self.assertEqual(stored["archive_bytes"], estimate["archive_bytes"])
            self.assertEqual(stored["static_bound_kind"], "rebuilt_from_source")
            self.assertEqual(stored["available_bytes"], printed["available_bytes"])
            self.assertEqual(stored["archive_budget_bytes"], 500000000)
            self.assertEqual(stored["free_bytes"], printed["free_bytes"])
            self.assertEqual(
                stored["budget_required_free_bytes"],
                500000000 + stabilized.STABILIZED_FILESYSTEM_RESERVE_BYTES,
            )

    def test_anderson_writer_closes_when_completed_archive_write_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-writer-close-") as directory:
            root = Path(directory)
            arguments = self._arguments(root)
            arguments.relaxation_policy = "anderson"
            arguments.omega_initial = 0.5
            arguments.omega_min = 0.5
            arguments.omega_reduction = None
            arguments.sufficient_decrease = 1.0e-2
            arguments.max_backtracks = 0
            arguments.anderson_history_depth = 5
            arguments.anderson_beta = 0.5
            arguments.anderson_scalar_diagnostic = True

            writers: list[Any] = []

            class TrackingWriter:
                def __init__(self, path: Path, *, binding: Mapping[str, Any]) -> None:
                    del binding
                    self.path = path
                    self.close_calls = 0
                    writers.append(self)

                def emit(self, event: Mapping[str, Any]) -> None:
                    del event

                def finalize(self, archive_path: Path, *, status: str) -> str:
                    del archive_path, status
                    return "0" * 64

                def close(self) -> None:
                    self.close_calls += 1

            with (
                mock.patch.object(workflow, "_load_fem_archive", return_value=self._fem()),
                mock.patch.object(workflow, "_preflight_artifacts"),
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(
                        self.model,
                        self.config.base,
                        self._cloud,
                        self.source_metadata,
                    ),
                ),
                mock.patch.object(
                    workflow,
                    "_preflight_stabilized_storage",
                    return_value={"fits": True},
                ),
                mock.patch.object(
                    workflow, "_construct_adapter", return_value=(object(), "test:gpu")
                ),
                mock.patch.object(
                    stabilized,
                    "AndersonScalarDiagnosticWriter",
                    TrackingWriter,
                ),
                mock.patch.object(
                    stabilized,
                    "run_stabilized_engineering_grip",
                    return_value=self.result,
                ),
                mock.patch.object(
                    workflow,
                    "_atomic_stabilized_archive",
                    side_effect=OSError("manufactured completed archive write failure"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "completed archive write"):
                    workflow._run_stabilized_workflow(arguments)

            self.assertEqual(len(writers), 1)
            self.assertEqual(writers[0].close_calls, 1)

    def test_anderson_writer_closes_when_failure_archive_write_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="engineering-failure-writer-close-") as directory:
            root = Path(directory)
            arguments = self._arguments(root)
            arguments.relaxation_policy = "anderson"
            arguments.omega_initial = 0.5
            arguments.omega_min = 0.5
            arguments.omega_reduction = None
            arguments.sufficient_decrease = 1.0e-2
            arguments.max_backtracks = 0
            arguments.anderson_history_depth = 5
            arguments.anderson_beta = 0.5
            arguments.anderson_scalar_diagnostic = True
            writer = mock.Mock()
            failure = stabilized.StabilizedEngineeringGripContinuationFailure(
                "manufactured continuation failure",
                self.result.final_state,
                tuple(self.result.attempts),
                self.result.work_check,
            )

            with (
                mock.patch.object(workflow, "_load_fem_archive", return_value=self._fem()),
                mock.patch.object(workflow, "_preflight_artifacts"),
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(
                        self.model,
                        self.config.base,
                        self._cloud,
                        self.source_metadata,
                    ),
                ),
                mock.patch.object(
                    workflow,
                    "_preflight_stabilized_storage",
                    return_value={"fits": True},
                ),
                mock.patch.object(
                    workflow, "_construct_adapter", return_value=(object(), "test:gpu")
                ),
                mock.patch.object(
                    stabilized,
                    "AndersonScalarDiagnosticWriter",
                    return_value=writer,
                ),
                mock.patch.object(
                    stabilized,
                    "run_stabilized_engineering_grip",
                    side_effect=failure,
                ),
                mock.patch.object(
                    workflow,
                    "_atomic_stabilized_failure_archive",
                    side_effect=OSError("manufactured failure archive write failure"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "failure archive write"):
                    workflow._run_stabilized_workflow(arguments)

            writer.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
