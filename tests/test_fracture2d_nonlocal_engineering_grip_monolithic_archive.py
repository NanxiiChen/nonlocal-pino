"""Manufactured tests for the durable monolithic continuation archive.

The archive tests use the existing 5x5 manufactured cloud only.  They do not
invoke the workflow, CLI, rounded-slot geometry, or a physical continuation.
"""

from __future__ import annotations

import io
import inspect
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import zipfile
from typing import Any

import jax
import numpy as np

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import nonlocal_engineering_grip_monolithic as monolithic
from examples.fracture2d import (
    nonlocal_engineering_grip_monolithic_archive as archive,
)
from examples.fracture2d.nonlocal_grip_diagnostic import build_union_geometry_from_arrays
from tests.test_fracture2d_nonlocal_engineering_grip_monolithic import _material_cloud


class MonolithicArchiveTest(unittest.TestCase):
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
        cls.newton = monolithic.MonolithicNewtonConfig(
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
        cls.config = monolithic.MonolithicContinuationConfig(
            cls.base.load_start,
            cls.base.load_final,
            cls.base.du_initial,
            cls.base.du_min,
            cls.base.cutback_factor,
            cls.base.max_cutbacks,
            cls.base,
            cls.newton,
        )
        cls.result = monolithic.run_monolithic_strict_continuation(
            cls.model, cls.config, None
        )
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary_directory.name) / "success.npz"
        archive.write_monolithic_continuation_archive(
            cls.path,
            cls.result,
            50_000_000,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()
        jax.config.update("jax_enable_x64", cls.old_x64)

    @staticmethod
    def _read_arrays(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: np.array(loaded[name], copy=True) for name in loaded.files}

    @staticmethod
    def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
        with path.open("wb") as handle:
            np.savez_compressed(file=handle, **arrays)  # type: ignore[arg-type]

    @staticmethod
    def _write_zip32_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
        with path.open("wb") as handle:
            archive._write_zip32_npz(handle, arrays)

    @staticmethod
    def _read_raw_members(path: Path) -> dict[str, bytes]:
        with zipfile.ZipFile(path, mode="r") as loaded:
            return {name: loaded.read(name) for name in loaded.namelist()}

    @staticmethod
    def _write_raw_members(
        path: Path,
        payloads: dict[str, bytes],
        *,
        archive_comment: bytes = b"",
        member_extra: bytes = b"",
        symlink_member: str | None = None,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> None:
        with zipfile.ZipFile(path, mode="w", compression=compression) as output:
            output.comment = archive_comment
            for name in archive.MONOLITHIC_ARCHIVE_MEMBER_NAMES:
                info = zipfile.ZipInfo(name)
                info.compress_type = compression
                info.extra = member_extra
                if name == symlink_member:
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(info, payloads[name])

    @staticmethod
    def _npy_bytes(value: np.ndarray, *, allow_pickle: bool = False) -> bytes:
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=allow_pickle)
        return stream.getvalue()

    def test_api_requires_explicit_budget_and_has_no_workflow_entry(self) -> None:
        write_parameters = inspect.signature(
            archive.write_monolithic_continuation_archive
        ).parameters
        self.assertEqual(
            tuple(write_parameters),
            ("path", "result", "archive_budget_bytes", "overwrite"),
        )
        self.assertIs(
            write_parameters["archive_budget_bytes"].default,
            inspect.Parameter.empty,
        )
        self.assertEqual(
            tuple(inspect.signature(archive.replay_monolithic_continuation_archive).parameters),
            ("path", "model", "archive_budget_bytes"),
        )

    def test_roundtrip_is_owner_authenticated_and_arrays_are_immutable(self) -> None:
        with mock.patch.object(
            monolithic,
            "run_monolithic_strict_continuation",
            side_effect=AssertionError("replaceable public controller was called"),
        ), mock.patch.object(
            monolithic,
            "run_verified_monolithic_attempt",
            side_effect=AssertionError("replaceable public Phase-2B runner was called"),
        ), mock.patch.object(
            engineering,
            "run_zero_damage_work_check",
            side_effect=AssertionError("replaceable public work-check runner was called"),
        ), mock.patch.object(
            monolithic,
            "_AUTHORITATIVE_RUN_MONOLITHIC_STRICT_CONTINUATION",
            side_effect=AssertionError("replaceable continuation alias was called"),
        ), mock.patch.object(
            monolithic,
            "_AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT",
            side_effect=AssertionError("replaceable Phase-2B alias was called"),
        ), mock.patch.object(
            engineering,
            "_AUTHORITATIVE_RUN_ZERO_DAMAGE_WORK_CHECK",
            side_effect=AssertionError("replaceable work-check alias was called"),
        ):
            replay = archive.replay_monolithic_continuation_archive(
                self.path,
                self.model,
                50_000_000,
            )
        self.assertEqual(replay.completion_status, "completed")
        np.testing.assert_array_equal(replay.state_u_bar, [0.0, 1.0e-4])
        self.assertEqual(replay.result.final_state.digest, replay.state_digests[-1])
        for value in (
            replay.state_u_bar,
            replay.state_union_displacement,
            replay.state_material_damage,
            replay.state_material_history,
            replay.accepted_reaction,
            replay.accepted_mechanics_energy,
        ):
            self.assertFalse(value.flags.writeable)
            with self.assertRaises(ValueError):
                value.flat[0] = value.flat[0]

    def test_member_allowlist_and_rejected_vectors_are_not_durable(self) -> None:
        with np.load(self.path, allow_pickle=False) as loaded:
            self.assertEqual(
                tuple(loaded.files),
                tuple(name.removesuffix(".npy") for name in archive.MONOLITHIC_ARCHIVE_MEMBER_NAMES),
            )
            for name in loaded.files:
                self.assertNotEqual(loaded[name].dtype.kind, "O")
            self.assertIn("accepted_mechanics_energy", loaded.files)
            self.assertFalse(any(name.endswith("total_energy") for name in loaded.files))
            metadata = bytes(np.asarray(loaded["metadata_json"], dtype=np.uint8).tolist())
        for forbidden in (b"raw_q", b"raw_damage", b"raw_history", b"current_z", b"initial_z"):
            self.assertNotIn(forbidden, metadata)
        self.assertIn(b"attempt_scalar_evidence", metadata)
        self.assertIn(b"accepted_mechanics_energy_semantics", metadata)

    def test_mutated_state_and_truncated_archive_are_rejected(self) -> None:
        arrays = self._read_arrays(self.path)
        arrays["state_material_damage"] = np.array(
            arrays["state_material_damage"], copy=True
        )
        arrays["state_material_damage"][1, 0] += 1.0e-12
        mutated = Path(self.temporary_directory.name) / "mutated.npz"
        self._write_arrays(mutated, arrays)
        with self.assertRaises(ValueError):
            archive.replay_monolithic_continuation_archive(
                mutated, self.model, 50_000_000
            )

        truncated = Path(self.temporary_directory.name) / "truncated.npz"
        truncated.write_bytes(self.path.read_bytes()[:-17])
        with self.assertRaises(ValueError):
            archive.replay_monolithic_continuation_archive(
                truncated, self.model, 50_000_000
            )

    def test_raw_npy_order_trailing_object_and_schema_type_are_rejected(self) -> None:
        arrays = self._read_arrays(self.path)

        for label, mutated in (
            (
                "fortran",
                {
                    **arrays,
                    "state_union_displacement": np.asfortranarray(
                        arrays["state_union_displacement"]
                    ),
                },
            ),
            (
                "object",
                {
                    **arrays,
                    "state_material_damage": np.asarray(
                        [["forbidden"]], dtype=object
                    ),
                },
            ),
        ):
            with self.subTest(label=label):
                path = Path(self.temporary_directory.name) / f"{label}.npz"
                if label == "object":
                    raw = self._read_raw_members(self.path)
                    raw["state_material_damage.npy"] = self._npy_bytes(
                        mutated["state_material_damage"], allow_pickle=True
                    )
                    self._write_raw_members(path, raw)
                else:
                    self._write_arrays(path, mutated)
                with self.assertRaises(ValueError):
                    archive.replay_monolithic_continuation_archive(
                        path, self.model, 50_000_000
                    )

        trailing = Path(self.temporary_directory.name) / "trailing.npy.npz"
        raw = self._read_raw_members(self.path)
        raw["state_u_bar.npy"] += b"trailing"
        self._write_raw_members(trailing, raw)
        with self.assertRaises(ValueError):
            archive.replay_monolithic_continuation_archive(
                trailing, self.model, 50_000_000
            )

        metadata_type = Path(self.temporary_directory.name) / "schema-type.npz"
        metadata = json.loads(bytes(arrays["metadata_json"].tolist()).decode("utf-8"))
        metadata["schema_version"] = True
        arrays["metadata_json"] = np.frombuffer(
            archive._canonical_json(metadata), dtype=np.uint8
        )
        self._write_arrays(metadata_type, arrays)
        with self.assertRaises(ValueError):
            archive.replay_monolithic_continuation_archive(
                metadata_type, self.model, 50_000_000
            )

    def test_zip_metadata_and_opaque_scalar_evidence_are_rejected(self) -> None:
        self.assertRaises(TypeError, archive._scalar_evidence, object(), "probe")
        raw = self._read_raw_members(self.path)
        cases: tuple[tuple[str, dict[str, Any]], ...] = (
            ("comment", {"archive_comment": b"forbidden"}),
            ("extra", {"member_extra": b"\x05\x00\x01\x00x"}),
            (
                "symlink",
                {"symlink_member": archive.MONOLITHIC_ARCHIVE_MEMBER_NAMES[0]},
            ),
        )
        for label, kwargs in cases:
            with self.subTest(label=label):
                path = Path(self.temporary_directory.name) / f"zip-{label}.npz"
                self._write_raw_members(path, raw, **kwargs)
                with self.assertRaises(ValueError):
                    archive.replay_monolithic_continuation_archive(
                        path, self.model, 50_000_000
                    )

    def test_zip32_only_rejects_local_sentinels_and_practical_budget_overflow(self) -> None:
        local_sentinel = Path(self.temporary_directory.name) / "local-sentinel.npz"
        raw = bytearray(self.path.read_bytes())
        raw[18:22] = (0xFFFFFFFF).to_bytes(4, "little")
        local_sentinel.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "ZIP64 local-header sentinel"):
            archive.replay_monolithic_continuation_archive(
                local_sentinel, self.model, 50_000_000
            )

        central_zip64 = Path(self.temporary_directory.name) / "central-zip64.npz"
        raw_members = self._read_raw_members(self.path)
        self._write_raw_members(
            central_zip64,
            raw_members,
            member_extra=(
                b"\x01\x00\x10\x00"
                + (1).to_bytes(8, "little")
                + (1).to_bytes(8, "little")
            ),
        )
        with self.assertRaisesRegex(ValueError, "ZIP64"):
            archive.replay_monolithic_continuation_archive(
                central_zip64, self.model, 50_000_000
            )

        central_sentinel = Path(self.temporary_directory.name) / "central-sentinel.npz"
        raw = bytearray(self.path.read_bytes())
        central_offset = raw.find(b"PK\x01\x02")
        self.assertGreaterEqual(central_offset, 0)
        raw[central_offset + 24 : central_offset + 28] = (0xFFFFFFFF).to_bytes(
            4, "little"
        )
        central_sentinel.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "ZIP64 central-directory sentinel"):
            archive.replay_monolithic_continuation_archive(
                central_sentinel, self.model, 50_000_000
            )

        with self.assertRaisesRegex(ValueError, "practical ZIP32-only v1 maximum"):
            archive.replay_monolithic_continuation_archive(
                self.path,
                self.model,
                archive.MONOLITHIC_ARCHIVE_MAX_BUDGET_BYTES + 1,
            )

        raw_archive = self.path.read_bytes()
        self.assertNotIn(b"PK\x06\x06", raw_archive)
        self.assertNotIn(b"PK\x06\x07", raw_archive)
        with zipfile.ZipFile(self.path, mode="r") as loaded:
            self.assertTrue(all(not info.extra for info in loaded.infolist()))

    def test_zip32_eocd_fields_and_ambiguity_are_rejected(self) -> None:
        source = self.path.read_bytes()
        eocd_offset = source.rfind(b"PK\x05\x06")
        self.assertGreaterEqual(eocd_offset, 0)
        expected_count = len(archive.MONOLITHIC_ARCHIVE_MEMBER_NAMES)
        field_mutations = (
            ("disk", 4, 2, 1, "disk number"),
            ("central-disk", 6, 2, 1, "central-directory start disk"),
            ("on-disk-sentinel", 8, 2, 0xFFFF, "entry-count sentinel"),
            ("total-sentinel", 10, 2, 0xFFFF, "entry-count sentinel"),
            ("count-mismatch", 10, 2, expected_count - 1, "entry counts"),
            ("central-size-mismatch", 12, 4, None, "central-directory size/offset"),
            ("central-offset-mismatch", 16, 4, None, "central-directory size/offset"),
        )
        for label, field_offset, width, value, message in field_mutations:
            with self.subTest(label=label):
                mutated = bytearray(source)
                if value is None:
                    current = int.from_bytes(
                        mutated[eocd_offset + field_offset : eocd_offset + field_offset + width],
                        "little",
                    )
                    value = current + 1
                mutated[
                    eocd_offset + field_offset : eocd_offset + field_offset + width
                ] = int(value).to_bytes(width, "little")
                path = Path(self.temporary_directory.name) / f"eocd-{label}.npz"
                path.write_bytes(mutated)
                with self.assertRaisesRegex(ValueError, message):
                    archive.replay_monolithic_continuation_archive(
                        path, self.model, 50_000_000
                    )

        duplicate = Path(self.temporary_directory.name) / "eocd-duplicate.npz"
        duplicate.write_bytes(source + source[eocd_offset : eocd_offset + 22])
        with self.assertRaisesRegex(ValueError, "ambiguous|preceding/trailing"):
            archive.replay_monolithic_continuation_archive(
                duplicate, self.model, 50_000_000
            )

        trailing = Path(self.temporary_directory.name) / "eocd-trailing.npz"
        trailing.write_bytes(source + b"trailing-bytes")
        with self.assertRaisesRegex(ValueError, "does not end at archive EOF"):
            archive.replay_monolithic_continuation_archive(
                trailing, self.model, 50_000_000
            )

        zip64_locator = Path(self.temporary_directory.name) / "eocd-zip64-locator.npz"
        locator = b"PK\x06\x07" + b"\x00" * 16
        zip64_locator.write_bytes(source[:eocd_offset] + locator + source[eocd_offset:])
        with self.assertRaisesRegex(ValueError, "ZIP64 locator"):
            archive.replay_monolithic_continuation_archive(
                zip64_locator, self.model, 50_000_000
            )

        payload_signature = Path(self.temporary_directory.name) / "payload-signature.npz"
        raw_members = self._read_raw_members(self.path)
        raw_members[archive.MONOLITHIC_ARCHIVE_MEMBER_NAMES[0]] += b"PK\x06\x06"
        self._write_raw_members(
            payload_signature,
            raw_members,
            compression=zipfile.ZIP_STORED,
        )
        archive._inspect_zip(payload_signature, 50_000_000)

    def test_budget_and_overwrite_fail_without_leaving_temporary_files(self) -> None:
        with self.assertRaises(ValueError):
            archive.replay_monolithic_continuation_archive(
                self.path, self.model, 100
            )
        destination = Path(self.temporary_directory.name) / "budget.npz"
        with self.assertRaises(ValueError):
            archive.write_monolithic_continuation_archive(
                destination,
                self.result,
                100,
            )
        self.assertFalse(destination.exists())
        self.assertEqual(
            tuple(destination.parent.glob(f".{destination.name}.*")),
            (),
        )
        with self.assertRaises(FileExistsError):
            archive.write_monolithic_continuation_archive(
                self.path,
                self.result,
                50_000_000,
            )

    def test_atomic_no_clobber_install_survives_creation_race(self) -> None:
        destination = Path(self.temporary_directory.name) / "race.npz"

        def create_destination_then_fail(_temporary: str, _destination: Path) -> None:
            destination.write_bytes(b"concurrent writer")
            raise FileExistsError("simulated destination race")

        with mock.patch.object(archive.os, "link", side_effect=create_destination_then_fail):
            with self.assertRaises(FileExistsError):
                archive.write_monolithic_continuation_archive(
                    destination,
                    self.result,
                    50_000_000,
                )
        self.assertEqual(destination.read_bytes(), b"concurrent writer")
        self.assertEqual(
            tuple(destination.parent.glob(f".{destination.name}.*")),
            (),
        )

    def test_atomic_link_directory_fsync_failure_cleans_installed_destination(self) -> None:
        destination = Path(self.temporary_directory.name) / "directory-fsync.npz"
        calls = 0

        def fsync_with_directory_failure(file_descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated directory fsync failure")

        with mock.patch.object(
            archive.os, "fsync", side_effect=fsync_with_directory_failure
        ):
            with self.assertRaisesRegex(
                OSError,
                "directory fsync failed.*destination cleanup status: "
                "destination_unlinked; cleanup_directory_fsync=passed",
            ):
                archive.write_monolithic_continuation_archive(
                    destination,
                    self.result,
                    50_000_000,
                )
        self.assertFalse(destination.exists())
        self.assertEqual(tuple(destination.parent.glob(f".{destination.name}.*")), ())

    def test_replay_reads_inspected_and_loaded_bytes_from_the_same_descriptor(self) -> None:
        target = Path(self.temporary_directory.name) / "same-fd-target.npz"
        replacement = Path(self.temporary_directory.name) / "replacement.npz"
        target.write_bytes(self.path.read_bytes())
        arrays = self._read_arrays(self.path)
        arrays["state_material_damage"] = np.array(
            arrays["state_material_damage"], copy=True
        )
        arrays["state_material_damage"][0, 0] += 1.0e-12
        self._write_zip32_arrays(replacement, arrays)
        original_inspect = archive._inspect_zip_handle
        replaced = False

        def inspect_then_replace(handle: Any, budget: int) -> None:
            nonlocal replaced
            original_inspect(handle, budget)
            if not replaced:
                os.replace(replacement, target)
                replaced = True

        with mock.patch.object(
            archive, "_inspect_zip_handle", side_effect=inspect_then_replace
        ):
            replay = archive.replay_monolithic_continuation_archive(
                target, self.model, 50_000_000
            )
        self.assertTrue(replaced)
        np.testing.assert_array_equal(
            replay.state_material_damage,
            self._read_arrays(self.path)["state_material_damage"],
        )

    def test_replay_uses_an_immutable_snapshot_during_in_place_rewrite(self) -> None:
        target = Path(self.temporary_directory.name) / "snapshot-target.npz"
        replacement = Path(self.temporary_directory.name) / "snapshot-replacement.npz"
        target.write_bytes(self.path.read_bytes())
        arrays = self._read_arrays(self.path)
        arrays["state_material_damage"] = np.array(
            arrays["state_material_damage"], copy=True
        )
        arrays["state_material_damage"][0, 0] += 1.0e-12
        self._write_zip32_arrays(replacement, arrays)
        replacement_bytes = replacement.read_bytes()
        original_inspect = archive._inspect_zip_handle
        rewritten = False

        def inspect_then_rewrite(handle: Any, budget: int) -> None:
            nonlocal rewritten
            original_inspect(handle, budget)
            if not rewritten:
                with target.open("r+b") as output:
                    output.seek(0)
                    output.write(replacement_bytes)
                    output.truncate()
                rewritten = True

        with mock.patch.object(
            archive, "_inspect_zip_handle", side_effect=inspect_then_rewrite
        ):
            replay = archive.replay_monolithic_continuation_archive(
                target, self.model, 50_000_000
            )
        self.assertTrue(rewritten)
        np.testing.assert_array_equal(
            replay.state_material_damage,
            self._read_arrays(self.path)["state_material_damage"],
        )

    def test_atomic_cleanup_failure_is_reported_and_does_not_claim_absence(self) -> None:
        destination = Path(self.temporary_directory.name) / "cleanup-failure.npz"
        real_unlink = archive.os.unlink

        def fail_destination_cleanup(path: Any, *args: Any, **kwargs: Any) -> None:
            if Path(path) == destination:
                raise OSError("simulated cleanup unlink failure")
            real_unlink(path, *args, **kwargs)

        calls = 0

        def fail_first_directory_fsync(file_descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated directory fsync failure")

        with mock.patch.object(archive.os, "unlink", side_effect=fail_destination_cleanup), mock.patch.object(
            archive.os, "fsync", side_effect=fail_first_directory_fsync
        ):
            with self.assertRaisesRegex(
                OSError,
                "cleanup_unlink_failed:simulated cleanup unlink failure",
            ):
                archive.write_monolithic_continuation_archive(
                    destination,
                    self.result,
                    50_000_000,
                )
        self.assertTrue(destination.exists())
        self.assertEqual(tuple(destination.parent.glob(f".{destination.name}.*")), ())

    def test_atomic_install_fails_closed_when_hard_links_are_unsupported(self) -> None:
        destination = Path(self.temporary_directory.name) / "no-hard-link.npz"
        with mock.patch.object(
            archive.os,
            "link",
            side_effect=OSError("simulated hard-link unsupported"),
        ):
            with self.assertRaisesRegex(OSError, "requires hard links; refusing a fallback"):
                archive.write_monolithic_continuation_archive(
                    destination,
                    self.result,
                    50_000_000,
                )
        self.assertFalse(destination.exists())
        self.assertEqual(tuple(destination.parent.glob(f".{destination.name}.*")), ())

    def test_failed_result_roundtrips_with_terminal_rejection_provenance(self) -> None:
        failed_newton = monolithic.MonolithicNewtonConfig(
            max_newton_iterations=1,
            gmres_rtol=self.newton.gmres_rtol,
            gmres_atol=self.newton.gmres_atol,
            gmres_restart=self.newton.gmres_restart,
            gmres_maxiter=self.newton.gmres_maxiter,
            armijo_c=self.newton.armijo_c,
            line_search_reduction=self.newton.line_search_reduction,
            line_search_max_backtracks=self.newton.line_search_max_backtracks,
            line_search_alpha_min=self.newton.line_search_alpha_min,
        )
        failed_config = monolithic.MonolithicContinuationConfig(
            self.base.load_start,
            self.base.load_final,
            self.base.du_initial,
            self.base.du_min,
            self.base.cutback_factor,
            self.base.max_cutbacks,
            self.base,
            failed_newton,
        )
        with self.assertRaises(monolithic.MonolithicContinuationFailure) as raised:
            monolithic.run_monolithic_strict_continuation(
                self.model, failed_config, None
            )
        failed_result = raised.exception.result
        path = Path(self.temporary_directory.name) / "failed.npz"
        archive.write_monolithic_continuation_archive(
            path,
            failed_result,
            50_000_000,
        )
        replay = archive.replay_monolithic_continuation_archive(
            path,
            self.model,
            50_000_000,
        )
        self.assertEqual(replay.completion_status, "failed")
        failed_attempt = replay.result.failed_attempt
        self.assertIsNotNone(failed_attempt)
        assert failed_attempt is not None
        self.assertIs(failed_attempt, replay.result.attempts[-1])
        self.assertEqual(failed_attempt.failure_stage, "newton")

    def test_controller_no_progress_archive_roundtrip_and_forged_provenance(self) -> None:
        with mock.patch.object(
            engineering,
            "run_zero_damage_work_check",
            return_value=self.result.work_check,
        ), mock.patch.object(engineering, "_next_target", return_value=0.0):
            with self.assertRaises(monolithic.MonolithicContinuationFailure) as raised:
                monolithic.run_monolithic_strict_continuation(
                    self.model,
                    self.config,
                    None,
                )
            no_progress = raised.exception.result
            path = Path(self.temporary_directory.name) / "no-progress.npz"
            owner_calls: list[monolithic.MonolithicContinuationResult] = []
            original_owner_validation = (
                monolithic.MonolithicContinuationResult.validate_with_owner
            )

            def owner_probe(
                result: monolithic.MonolithicContinuationResult,
                owner: engineering.EngineeringGripModel,
                **kwargs: Any,
            ) -> None:
                owner_calls.append(result)
                original_owner_validation(result, owner, **kwargs)

            with mock.patch.object(
                monolithic.MonolithicContinuationResult,
                "validate_with_owner",
                owner_probe,
            ):
                archive.write_monolithic_continuation_archive(
                    path,
                    no_progress,
                    50_000_000,
                )
                replay = archive.replay_monolithic_continuation_archive(
                    path,
                    self.model,
                    50_000_000,
                )
            self.assertEqual(replay.completion_status, "failed")
            self.assertEqual(replay.result.failure_reason, no_progress.failure_reason)
            self.assertEqual(len(replay.result.accepted_steps), 0)
            self.assertEqual(len(replay.result.attempts), 1)
            self.assertGreaterEqual(len(owner_calls), 2)
            failed_attempt = replay.result.failed_attempt
            self.assertIsNotNone(failed_attempt)
            assert failed_attempt is not None
            self.assertEqual(failed_attempt.failure_stage, "controller")
            self.assertEqual(failed_attempt.increment, 0.0)
            self.assertEqual(
                failed_attempt.failure_reason,
                "continuation made no floating-point load progress",
            )

            forged_arrays = self._read_arrays(path)
            forged_metadata = json.loads(
                bytes(forged_arrays["metadata_json"].tolist()).decode("utf-8")
            )
            forged_metadata["failure"]["failure_reason"] = "no_progress: forged"
            forged_metadata["result_digest"] = archive._archive_digest(
                forged_metadata,
                {name: forged_arrays[name] for name in archive._ARRAY_KEYS},
            )
            forged_arrays["metadata_json"] = np.frombuffer(
                archive._canonical_json(forged_metadata), dtype=np.uint8
            )
            forged_path = Path(self.temporary_directory.name) / "no-progress-forged.npz"
            self._write_zip32_arrays(forged_path, forged_arrays)
            with self.assertRaisesRegex(ValueError, "failure provenance"):
                archive.replay_monolithic_continuation_archive(
                    forged_path,
                    self.model,
                    50_000_000,
                )


if __name__ == "__main__":
    unittest.main()
