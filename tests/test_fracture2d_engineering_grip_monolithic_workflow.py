"""Manufactured workflow-seam tests for strict monolithic continuation.

These tests exercise parser/config/dispatch/artifact semantics only.  The
actual 5x5 controller and owner-authenticated archive/replay are covered by
the dedicated monolithic continuation/archive suites; no rounded-slot or
physical workflow is launched here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import ExitStack, redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import sys
import unittest
from unittest import mock

import numpy as np

from examples.fracture2d import engineering_grip_workflow as workflow
from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import nonlocal_engineering_grip_monolithic as monolithic
from examples.fracture2d import (
    nonlocal_engineering_grip_monolithic_archive as monolithic_archive,
)


def _arguments(root: Path, *extra: str):
    return workflow.build_parser().parse_args(
        [
            "run",
            "--continuation-strategy",
            "monolithic",
            "--archive-budget-bytes",
            "50000000",
            "--output-directory",
            str(root),
            "--run-id",
            "manufactured-monolithic",
            "--max-stagger-iterations",
            "100",
            "--max-newton-iterations",
            "30",
            "--gmres-rtol",
            "1e-10",
            "--gmres-atol",
            "1e-13",
            "--gmres-restart",
            "50",
            "--gmres-maxiter",
            "200",
            "--armijo-c",
            "1e-4",
            "--line-search-reduction",
            "0.5",
            "--line-search-max-backtracks",
            "12",
            "--line-search-alpha-min",
            "1e-8",
            *extra,
        ]
    )


def _canonical_base_config():
    return replace(
        engineering.EngineeringGripConfig.smoke(),
        max_stagger_iterations=workflow.CANONICAL_SOLVER_BASE_MAX_STAGGER_ITERATIONS,
    )


@dataclass(frozen=True)
class _FakeState:
    u_bar: float


class MonolithicWorkflowTest(unittest.TestCase):
    @staticmethod
    def _bundle(root: Path) -> Path:
        return root / "manufactured-monolithic.monolithic"

    @staticmethod
    def _staging(root: Path) -> list[Path]:
        return sorted(root.glob(".manufactured-monolithic.monolithic-*"))

    def _run_post_write_replay_failure(
        self,
        root: Path,
        replay_side_effect,
        *extra_patches,
    ) -> None:
        arguments = _arguments(root)
        base = _canonical_base_config()
        result = SimpleNamespace(
            completion_status="completed",
            validated=True,
            accepted_steps=(),
            attempts=(),
            final_state=_FakeState(0.0),
            failure_reason=None,
        )

        def write_archive(path, _result, _budget, *, overwrite):
            del _result, _budget, overwrite
            Path(path).write_bytes(b"manufactured archive")

        if replay_side_effect is None:
            replay_side_effect = SimpleNamespace(
                result=result,
                metadata_json=json.dumps(
                    {
                        "attempt_scalar_evidence": [],
                        "failure": {
                            "failed_target": None,
                            "failure_reason": None,
                            "failed_attempt_index": None,
                        },
                    },
                    sort_keys=True,
                ).encode("utf-8"),
                state_u_bar=np.asarray([0.0], dtype=np.float64),
                state_material_damage=np.zeros((1, 1), dtype=np.float64),
                state_material_history=np.zeros((1, 1), dtype=np.float64),
                accepted_reaction=np.zeros(0, dtype=np.float64),
                accepted_mechanics_energy=np.zeros(0, dtype=np.float64),
                state_digests=("1" * 64,),
                attempt_digests=(),
                result_digest="2" * 64,
                schema=monolithic_archive.MONOLITHIC_ARCHIVE_SCHEMA,
            )

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(object(), base, {}, {}),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    monolithic,
                    "run_monolithic_strict_continuation",
                    return_value=result,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    monolithic_archive,
                    "write_monolithic_continuation_archive",
                    side_effect=write_archive,
                )
            )
            replay_patch = (
                mock.patch.object(
                    monolithic_archive,
                    "replay_monolithic_continuation_archive",
                    side_effect=replay_side_effect,
                )
                if isinstance(replay_side_effect, BaseException)
                else mock.patch.object(
                    monolithic_archive,
                    "replay_monolithic_continuation_archive",
                    return_value=replay_side_effect,
                )
            )
            stack.enter_context(replay_patch)
            for patcher in extra_patches:
                stack.enter_context(patcher)
            workflow.run_workflow(arguments)

    def test_parser_requires_all_newton_controls(self) -> None:
        parser = workflow.build_parser()
        arguments = parser.parse_args(
            [
                "run",
                "--continuation-strategy",
                "monolithic",
                "--archive-budget-bytes",
                "50000000",
            ]
        )
        with self.assertRaisesRegex(ValueError, "max-newton-iterations"):
            workflow._monolithic_config_from_arguments(
                arguments,
                engineering.EngineeringGripConfig.smoke(),
                monolithic,
            )

    def test_monolithic_rejects_warning_and_stabilized_controls(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-args-") as directory:
            root = Path(directory)
            for extra, pattern in (
                (("--acceptance-mode", "warning"), "acceptance-mode"),
                (("--relaxation-policy", "anderson"), "relaxation-policy"),
                (("--anderson-scalar-diagnostic",), "anderson-scalar-diagnostic"),
                (("--progress-every-sweeps", "2"), "progress-every-sweeps"),
                (("--omega-initial", "0.25"), "omega-initial"),
                (("--omega-min", "0.25"), "omega-min"),
                (("--max-backtracks", "1"), "max-backtracks"),
                (("--omega-reduction", "0.5"), "omega-reduction"),
                (("--sufficient-decrease", "0.1"), "sufficient-decrease"),
                (("--anderson-history-depth", "2"), "anderson-history-depth"),
                (("--anderson-beta", "0.5"), "anderson-beta"),
                (("--anderson-alpha-reduction", "0.5"), "anderson-alpha-reduction"),
                (("--anderson-regularization-mu", "1.0"), "anderson-regularization-mu"),
                (("--gpu-device-index", "99"), "gpu-device-index is unused"),
                (("--fem-archive", "/tmp/foreign-fem.npz"), "fem-archive is unused"),
            ):
                with self.subTest(extra=extra):
                    arguments = _arguments(root, *extra)
                    with self.assertRaisesRegex(ValueError, pattern):
                        workflow._validate_monolithic_arguments(
                            arguments, monolithic_archive
                        )

    def test_monolithic_rejects_stagger_value_as_non_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-stagger-") as directory:
            arguments = _arguments(
                Path(directory), "--max-stagger-iterations", "999"
            )
            with self.assertRaisesRegex(
                ValueError, "not a monolithic iteration budget"
            ):
                workflow._validate_monolithic_arguments(arguments, monolithic_archive)

    def test_monolithic_accepts_exact_solver_base_stagger_compatibility_value(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-stagger-ok-") as directory:
            arguments = _arguments(Path(directory))
            workflow._validate_monolithic_arguments(arguments, monolithic_archive)
            self.assertEqual(
                arguments.max_stagger_iterations,
                workflow.CANONICAL_SOLVER_BASE_MAX_STAGGER_ITERATIONS,
            )

    def test_monolithic_rejects_overwrite_even_when_requested(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-overwrite-") as directory:
            arguments = _arguments(Path(directory), "--overwrite")
            with self.assertRaisesRegex(ValueError, "overwrite is not supported"):
                workflow._validate_monolithic_arguments(arguments, monolithic_archive)

    def test_dispatch_selects_monolithic_before_legacy_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-dispatch-") as directory:
            arguments = _arguments(Path(directory))
            with mock.patch.object(
                workflow, "_run_monolithic_workflow", return_value=17
            ) as runner:
                self.assertEqual(workflow.run_workflow(arguments), 17)
            runner.assert_called_once_with(arguments)

    def test_cpu_workflow_persists_and_replays_without_fem_or_gpu(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-run-") as directory:
            root = Path(directory)
            arguments = _arguments(root)
            base = _canonical_base_config()
            source_metadata = {"path": "manufactured-source.npz", "sha256": "0" * 64}
            model = object()
            cloud = {"coordinates": np.zeros((1, 2), dtype=np.float64)}
            result = SimpleNamespace(
                completion_status="completed",
                validated=True,
                accepted_steps=(),
                attempts=(),
                final_state=_FakeState(0.0),
                failure_reason=None,
            )
            metadata = {
                "attempt_scalar_evidence": [],
                "failure": {
                    "failed_target": None,
                    "failure_reason": None,
                    "failed_attempt_index": None,
                },
            }
            replay = SimpleNamespace(
                result=result,
                metadata_json=json.dumps(metadata, sort_keys=True).encode("utf-8"),
                state_u_bar=np.asarray([0.0], dtype=np.float64),
                state_material_damage=np.zeros((1, 1), dtype=np.float64),
                state_material_history=np.zeros((1, 1), dtype=np.float64),
                accepted_reaction=np.zeros(0, dtype=np.float64),
                accepted_mechanics_energy=np.zeros(0, dtype=np.float64),
                state_digests=("1" * 64,),
                attempt_digests=(),
                result_digest="2" * 64,
                schema=monolithic_archive.MONOLITHIC_ARCHIVE_SCHEMA,
            )

            staging_modes: list[int] = []

            def write_archive(path, _result, _budget, *, overwrite):
                del _result, _budget, overwrite
                staging_modes.append(Path(path).parent.stat().st_mode & 0o777)
                Path(path).write_bytes(b"manufactured archive")

            with (
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(model, base, cloud, source_metadata),
                ),
                mock.patch.object(
                    monolithic,
                    "run_monolithic_strict_continuation",
                    return_value=result,
                ),
                mock.patch.object(
                    monolithic_archive,
                    "write_monolithic_continuation_archive",
                    side_effect=write_archive,
                ) as writer,
                mock.patch.object(
                    monolithic_archive,
                    "replay_monolithic_continuation_archive",
                    return_value=replay,
                ) as owner_replay,
                mock.patch.object(
                    workflow,
                    "_construct_adapter",
                    side_effect=AssertionError("monolithic path must not build a GPU adapter"),
                ),
                mock.patch.object(
                    workflow,
                    "_load_fem_archive",
                    side_effect=AssertionError("monolithic path must not load FEM"),
                ),
            ):
                self.assertEqual(workflow.run_workflow(arguments), 0)

            writer.assert_called_once()
            archive_staging_path = Path(writer.call_args.args[0])
            self.assertTrue(archive_staging_path.parent.name.startswith("."))
            self.assertEqual(staging_modes, [0o700])
            bundle = self._bundle(root)
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
            self.assertEqual(
                sorted(path.name for path in bundle.iterdir()),
                ["archive.npz", "manifest.json", "replay.json", "summary.json"],
            )
            owner_replay.assert_called_once_with(
                archive_staging_path,
                model,
                50000000,
            )
            summary_path = bundle / "summary.json"
            replay_path = bundle / "replay.json"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(replay_path.is_file())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["continuation_strategy"], "monolithic")
            self.assertTrue(summary["validated"])
            self.assertTrue(summary["path_replayed"])
            self.assertEqual(summary["state_summary"]["attempts"], 0)
            self.assertFalse(summary["field_equations_validated"])
            self.assertEqual(
                summary["field_equations_validation_scope"],
                workflow.MONOLITHIC_FIELD_EQUATIONS_VALIDATION_SCOPE,
            )
            replay_report = json.loads(replay_path.read_text(encoding="utf-8"))
            self.assertEqual(replay_report["archive_path"], str(bundle / "archive.npz"))
            self.assertNotIn(str(archive_staging_path.parent), replay_path.read_text())
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["bundle"]["path"], str(bundle))
            self.assertEqual(
                manifest["contents"]["archive"]["path"],
                str(bundle / "archive.npz"),
            )
            self.assertEqual(
                manifest["contents"]["archive"]["sha256"],
                replay_report["archive_sha256"],
            )
            self.assertEqual(
                manifest["contents"]["manifest"]["fingerprint_scope"],
                "self-reference-excluded",
            )

    def test_replay_structural_failure_cleans_final_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-cleanup-") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "owner replay structural failure"):
                self._run_post_write_replay_failure(
                    root,
                    ValueError("owner replay structural failure"),
                )
            self.assertFalse(self._bundle(root).exists())
            self.assertEqual(self._staging(root), [])

    def test_monkeypatched_replacement_cannot_delete_a_final_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-replacement-") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "owner replay structural failure"):
                self._run_post_write_replay_failure(
                    root,
                    ValueError("owner replay structural failure"),
                    mock.patch.object(
                        workflow.os,
                        "replace",
                        side_effect=AssertionError("final replacement must not be used"),
                    ),
                )
            self.assertFalse(self._bundle(root).exists())
            self.assertEqual(self._staging(root), [])

    def test_two_publishers_use_atomic_no_clobber_and_leave_one_bundle(self) -> None:
        if sys.platform != "linux":
            self.skipTest("renameat2 publication contract is Linux-specific")
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-race-") as directory:
            root = Path(directory)
            base = _canonical_base_config()
            model = object()
            cloud = {"coordinates": np.zeros((1, 2), dtype=np.float64)}
            source_metadata = {"path": "manufactured-source.npz", "sha256": "0" * 64}
            result = SimpleNamespace(
                completion_status="completed",
                validated=True,
                accepted_steps=(),
                attempts=(),
                final_state=_FakeState(0.0),
                failure_reason=None,
            )
            replay = SimpleNamespace(
                result=result,
                metadata_json=json.dumps(
                    {
                        "attempt_scalar_evidence": [],
                        "failure": {
                            "failed_target": None,
                            "failure_reason": None,
                            "failed_attempt_index": None,
                        },
                    },
                    sort_keys=True,
                ).encode("utf-8"),
                state_u_bar=np.asarray([0.0], dtype=np.float64),
                state_material_damage=np.zeros((1, 1), dtype=np.float64),
                state_material_history=np.zeros((1, 1), dtype=np.float64),
                accepted_reaction=np.zeros(0, dtype=np.float64),
                accepted_mechanics_energy=np.zeros(0, dtype=np.float64),
                state_digests=("1" * 64,),
                attempt_digests=(),
                result_digest="2" * 64,
                schema=monolithic_archive.MONOLITHIC_ARCHIVE_SCHEMA,
            )
            barrier = threading.Barrier(2)

            def write_archive(path, _result, _budget, *, overwrite):
                del _result, _budget, overwrite
                Path(path).write_bytes(b"manufactured archive")
                barrier.wait(timeout=10.0)

            outcomes: list[int] = []
            errors: list[BaseException] = []

            def worker() -> None:
                try:
                    outcomes.append(workflow.run_workflow(_arguments(root)))
                except BaseException as error:  # noqa: BLE001 - test boundary
                    errors.append(error)

            with (
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(model, base, cloud, source_metadata),
                ),
                mock.patch.object(
                    monolithic,
                    "run_monolithic_strict_continuation",
                    return_value=result,
                ),
                mock.patch.object(
                    monolithic_archive,
                    "write_monolithic_continuation_archive",
                    side_effect=write_archive,
                ),
                mock.patch.object(
                    monolithic_archive,
                    "replay_monolithic_continuation_archive",
                    return_value=replay,
                ),
            ):
                threads = [threading.Thread(target=worker) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=30.0)
                    self.assertFalse(thread.is_alive(), "publisher thread did not finish")

            self.assertEqual(outcomes, [0])
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], FileExistsError)
            self.assertTrue(self._bundle(root).is_dir())
            self.assertEqual(self._staging(root), [])
            self.assertEqual(
                sorted(path.name for path in self._bundle(root).iterdir()),
                ["archive.npz", "manifest.json", "replay.json", "summary.json"],
            )

    def test_summary_generation_failure_removes_only_private_staging(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-summary-failure-") as directory:
            root = Path(directory)
            real_write_json = workflow._write_json_fsync

            def fail_summary(path, payload):
                if Path(path).name == "summary.json":
                    raise ValueError("manufactured summary generation failure")
                return real_write_json(path, payload)

            with self.assertRaisesRegex(ValueError, "summary generation failure"):
                self._run_post_write_replay_failure(
                    root,
                    None,
                    mock.patch.object(workflow, "_write_json_fsync", side_effect=fail_summary),
                )
            self.assertFalse(self._bundle(root).exists())
            self.assertEqual(self._staging(root), [])

    def test_manifest_generation_failure_removes_only_private_staging(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-manifest-failure-") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "manifest generation failure"):
                self._run_post_write_replay_failure(
                    root,
                    None,
                    mock.patch.object(
                        workflow,
                        "_write_monolithic_manifest",
                        side_effect=ValueError("manufactured manifest generation failure"),
                    ),
                )
            self.assertFalse(self._bundle(root).exists())
            self.assertEqual(self._staging(root), [])

    def test_parent_fsync_failure_reports_uncertainty_and_keeps_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-parent-fsync-") as directory:
            root = Path(directory)
            real_fsync_directory = workflow._fsync_directory

            def fail_parent(path: Path) -> None:
                if Path(path) == root:
                    raise OSError("manufactured parent fsync failure")
                real_fsync_directory(path)

            with self.assertRaises(workflow.MonolithicBundleDurabilityError) as raised:
                self._run_post_write_replay_failure(
                    root,
                    None,
                    mock.patch.object(
                        workflow,
                        "_fsync_directory",
                        side_effect=fail_parent,
                    ),
                )
            error = raised.exception
            self.assertEqual(error.bundle_path, self._bundle(root))
            self.assertIn("durability is uncertain", str(error))
            self.assertTrue(self._bundle(root).is_dir())
            self.assertEqual(self._staging(root), [])
            self.assertEqual(
                sorted(path.name for path in self._bundle(root).iterdir()),
                ["archive.npz", "manifest.json", "replay.json", "summary.json"],
            )

    def test_main_propagates_parent_durability_uncertainty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-main-") as directory:
            arguments = _arguments(Path(directory))
            parser = mock.Mock()
            parser.parse_args.return_value = arguments
            error = workflow.MonolithicBundleDurabilityError(
                self._bundle(Path(directory)), OSError("manufactured fsync failure")
            )
            stderr = io.StringIO()
            with (
                mock.patch.object(workflow, "build_parser", return_value=parser),
                mock.patch.object(workflow, "run_workflow", side_effect=error),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(workflow.MonolithicBundleDurabilityError):
                    workflow.main([])
            self.assertEqual(stderr.getvalue(), "")

    def test_structural_exception_writes_no_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-structural-") as directory:
            root = Path(directory)
            arguments = _arguments(root)
            base = _canonical_base_config()
            with (
                mock.patch.object(
                    workflow,
                    "_construct_model",
                    return_value=(object(), base, {}, {}),
                ),
                mock.patch.object(
                    monolithic,
                    "run_monolithic_strict_continuation",
                    side_effect=TypeError("foreign structural object"),
                ),
                mock.patch.object(
                    monolithic_archive,
                    "write_monolithic_continuation_archive",
                ) as writer,
            ):
                with self.assertRaisesRegex(TypeError, "foreign structural object"):
                    workflow.run_workflow(arguments)
            writer.assert_not_called()
            self.assertFalse(self._bundle(root).exists())
            self.assertEqual(self._staging(root), [])

    def test_standalone_monolithic_replay_requires_explicit_owner_and_budget(self) -> None:
        values = {
            "metadata_json": np.asarray(
                np.frombuffer(
                    json.dumps(
                        {"schema": monolithic_archive.MONOLITHIC_ARCHIVE_SCHEMA}
                    ).encode("utf-8"),
                    dtype=np.uint8,
                ),
                dtype=np.uint8,
            )
        }
        with tempfile.TemporaryDirectory(prefix="monolithic-workflow-replay-") as directory:
            root = Path(directory)
            archive_path = root / "archive.npz"
            report_path = root / "replay.json"
            archive_path.write_bytes(b"manufactured archive")
            with mock.patch.object(workflow, "_load_npz", return_value=values):
                with self.assertRaisesRegex(ValueError, "owner_model"):
                    workflow.replay_archive(
                        archive_path, report_path, overwrite=False
                    )


if __name__ == "__main__":
    unittest.main()
