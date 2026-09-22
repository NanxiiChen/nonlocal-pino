"""Durable, owner-authenticated archives for the private monolithic path.

This module is deliberately separate from the stabilized engineering-grip
archive and from the workflow layer.  It persists only the accepted
continuation states and scalar audit provenance.  Raw/rejected ``q``, ``d``,
and ``H`` vectors never become archive members.

The archive is an ordinary compressed NPZ (a ZIP of NPY members), but its
member list and JSON metadata are strict.  Loading is not a deserialization of
an in-memory solver graph: the supplied :class:`EngineeringGripModel` is the
owner, the two configs are reconstructed from canonical metadata, and the
private strict controller is rerun on the CPU reference path before any
archive is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np

from . import nonlocal_engineering_grip as engineering
from . import nonlocal_engineering_grip_monolithic as monolithic
from .nonlocal_baseline import GateTolerance
from .nonlocal_engineering_grip import (
    EngineeringGripConfig,
    EngineeringGripModel,
    EngineeringGripState,
)


__all__ = [
    "MONOLITHIC_ARCHIVE_SCHEMA",
    "MONOLITHIC_ARCHIVE_SCHEMA_VERSION",
    "MONOLITHIC_ARCHIVE_MEMBER_NAMES",
    "MONOLITHIC_ARCHIVE_MAX_BUDGET_BYTES",
    "MonolithicArchiveReplay",
    "write_monolithic_continuation_archive",
    "replay_monolithic_continuation_archive",
    "load_monolithic_continuation_archive",
    "read_monolithic_continuation_archive",
]


# Archive replay captures these owner runners when this private module is
# imported.  Passing them explicitly through ``validate_with_owner`` prevents
# a caller from replacing the monolithic module attributes after inspection.
_ARCHIVE_AUTHORITATIVE_WORK_CHECK = engineering._AUTHORITATIVE_RUN_ZERO_DAMAGE_WORK_CHECK
_ARCHIVE_AUTHORITATIVE_ATTEMPT = monolithic._AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT


MONOLITHIC_ARCHIVE_SCHEMA = "fracture2d-monolithic-continuation-v1"
MONOLITHIC_ARCHIVE_SCHEMA_VERSION = 1
_FLOAT64 = np.dtype(np.float64)
_UINT8 = np.dtype(np.uint8)
_ARCHIVE_RESULT_DIGEST_FIELD = "result_digest"
_ARRAY_KEYS = (
    "state_u_bar",
    "state_union_displacement",
    "state_material_damage",
    "state_material_history",
    "accepted_reaction",
    "accepted_mechanics_energy",
)
_MEMBER_KEYS = ("metadata_json", *_ARRAY_KEYS)
MONOLITHIC_ARCHIVE_MEMBER_NAMES = tuple(f"{key}.npy" for key in _MEMBER_KEYS)
_SOURCE_BUNDLE_HASH_KIND = "deterministic-monolithic-source-bundle-sha256-v1"
_SOURCE_BUNDLE_FILES = tuple(
    engineering._SOURCE_BUNDLE_FILES
    + (
        "examples/fracture2d/nonlocal_engineering_grip_monolithic.py",
        "examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py",
    )
)
_ACCEPTANCE_MODE = "strict"
_COMPLETION_STATUSES = ("completed", "failed")
_CANONICAL_MECHANICS_ENERGY_SEMANTICS = (
    "A2MechanicsSolveResult.energy at each accepted canonical mechanics-after state"
)
_OWNER_RECOMPUTED_FIELDS = (
    "psi_after",
    "H_next",
    "crack_region_strain",
    "static_operators",
    "canonical_equations",
    "canonical_gates",
)
_EPHEMERAL_FIELDS = frozenset(
    {
        "static_identity",
        "static_runtime_identity",
        "damage_runtime_identity",
    }
)
_OMITTED = object()
_NPY_MAX_HEADER_SIZE = 10_000
_MAX_COMPRESSION_RATIO = 1_000
_ZIP_OVERHEAD_BUDGET_BYTES = 64 * 1024
_ZIP_LOCAL_HEADER_SIZE = 30
_ZIP_LOCAL_HEADER = struct.Struct("<4s5H3I2H")
_ZIP_CENTRAL_HEADER_SIZE = 46
_ZIP_CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
_ZIP_EOCD_SIZE = 22
_ZIP_EOCD = struct.Struct("<4s4H2IH")
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP32_MAX_UINT16 = (1 << 16) - 1
_ZIP32_MAX_UINT32 = (1 << 32) - 1
_ZIP32_MAX_ARCHIVE_BYTES = _ZIP32_MAX_UINT32
_ZIP32_MAX_MEMBER_BYTES = _ZIP32_MAX_UINT32
# The public workflow's current practical archive setting is 500 MB.  v1 is
# intentionally bounded below the ZIP32 ceiling rather than presenting a
# multi-gigabyte ZIP64-like budget that its writer/reader do not support.
MONOLITHIC_ARCHIVE_MAX_BUDGET_BYTES = 500_000_000
_ZIP64_EXTRA_FIELD_ID = 0x0001


def _run_archive_authoritative_continuation(
    model: EngineeringGripModel,
    config: monolithic.MonolithicContinuationConfig,
    adapter: Any,
    *,
    _runner: Any = monolithic._run_monolithic_strict_continuation_impl,
    _work_check_runner: Any = _ARCHIVE_AUTHORITATIVE_WORK_CHECK,
    _attempt_runner: Any = _ARCHIVE_AUTHORITATIVE_ATTEMPT,
) -> monolithic.MonolithicContinuationResult:
    """Run the captured CPU controller below replaceable module attributes."""

    return _runner(
        model,
        config,
        adapter,
        work_check_runner=_work_check_runner,
        attempt_runner=_attempt_runner,
    )


def _validate_archive_owner(
    result: monolithic.MonolithicContinuationResult,
    model: EngineeringGripModel,
) -> None:
    """Run the public owner boundary backed by captured authoritative runners.

    Calling the public method is intentional: the archive supplies runners
    captured at module import, so replaceable public controller attributes are
    not trusted.  The owner method therefore makes ``psi_after/H_next`` and
    canonical equations/gates part of the same authenticated replay contract
    as the accepted state arrays.
    """

    result.validate_with_owner(
        model,
        _work_check_runner=_ARCHIVE_AUTHORITATIVE_WORK_CHECK,
        _attempt_runner=_ARCHIVE_AUTHORITATIVE_ATTEMPT,
    )


def _immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Return a bytes-backed read-only C-order copy."""

    if isinstance(values, (bytes, bytearray, memoryview)):
        source = np.frombuffer(bytes(values), dtype=np.uint8).copy()
        if dtype is not None and np.dtype(dtype) != _UINT8:
            source = source.astype(dtype, copy=False)
    else:
        source = np.array(values, dtype=dtype, copy=True, order="C")
    result = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result


def _require_budget(value: Any, name: str = "archive_budget_bytes") -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(raw)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    if result > MONOLITHIC_ARCHIVE_MAX_BUDGET_BYTES:
        raise ValueError(
            f"{name} exceeds the practical ZIP32-only v1 maximum "
            f"({result} > {MONOLITHIC_ARCHIVE_MAX_BUDGET_BYTES})"
        )
    if result > _ZIP32_MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"{name} exceeds the ZIP32 archive limit "
            f"({result} > {_ZIP32_MAX_ARCHIVE_BYTES})"
        )
    return result


def _path(value: Any, name: str) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise TypeError(f"{name} must be path-like") from error
    if isinstance(raw, bytes):
        if not raw:
            raise ValueError(f"{name} must be non-empty")
        result = Path(os.fsdecode(raw))
    else:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{name} must be a non-empty path")
        result = Path(raw)
    if result.name in ("", ".", ".."):
        raise ValueError(f"{name} must name an archive file")
    return result


def _valid_digest(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _source_bundle_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    entries = tuple((name, root / name) for name in _SOURCE_BUNDLE_FILES)
    return engineering._source_bundle_hash(entries)


def _canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("archive metadata is not canonical JSON data") from error
    return text.encode("utf-8")


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON metadata key {key!r}")
        result[key] = value
    return result


def _json_reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _decode_metadata(values: np.ndarray) -> tuple[dict[str, Any], bytes]:
    array = np.asarray(values)
    if array.dtype != _UINT8 or array.ndim != 1 or array.size == 0:
        raise ValueError("metadata_json must be a nonempty one-dimensional uint8 array")
    raw = bytes(array.tolist())
    try:
        text = raw.decode("utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=_json_no_duplicate_keys,
            parse_constant=_json_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("metadata_json is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json top level must be an object")
    if _canonical_json(decoded) != raw:
        raise ValueError("metadata_json is not in canonical encoding")
    return decoded, raw


def _exact_keys(value: Any, expected: tuple[str, ...], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unknown = sorted(set(value) - set(expected))
        raise ValueError(f"{name} keys differ; missing={missing}, unknown={unknown}")
    return value


def _json_string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _json_float(value: Any, name: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


def _json_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative JSON integer")
    return value


def _json_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _json_digest(value: Any, name: str) -> str:
    return _valid_digest(_json_string(value, name), name)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_digest(name: str, value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise ValueError(f"archive array {name} has object dtype")
    digest = hashlib.sha256()
    encoded_name = name.encode("utf-8")
    digest.update(b"monolithic-archive-array-v1\0")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _archive_digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    without_result = dict(metadata)
    without_result.pop(_ARCHIVE_RESULT_DIGEST_FIELD, None)
    array_digests = {
        name: _array_digest(name, arrays[name])
        for name in _ARRAY_KEYS
    }
    return _hash_json({"metadata": without_result, "arrays": array_digests})


def _json_scalar(value: Any, name: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) is bool or value is None or type(value) is str:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"{name} contains a non-finite scalar")
        return value
    raise ValueError(f"{name} is not a supported scalar")


def _scalar_evidence(value: Any, name: str = "evidence") -> Any:
    """Project an audit tree to JSON-safe scalar/digest provenance.

    Numerical arrays are intentionally omitted.  A dataclass retains a digest
    of the names of omitted array fields so optional-field presence remains a
    scalar audit fact without persisting rejected solver vectors.
    """

    if isinstance(value, np.ndarray):
        return _OMITTED
    if isinstance(value, EngineeringGripModel):
        return {
            "type": "EngineeringGripModelIdentity",
            "model_digest": value.model_digest,
            "static_digest": value.static_digest,
            "material_hash": value.material_hash,
            "union_hash": value.union_hash,
        }
    if isinstance(value, EngineeringGripConfig):
        return {
            "type": "EngineeringGripConfigIdentity",
            "config_digest": value.config_digest,
        }
    if isinstance(value, EngineeringGripState):
        return {
            "type": "EngineeringGripStateIdentity",
            "u_bar": float(value.u_bar),
            "digest": value.digest,
        }
    if isinstance(value, np.generic):
        return _json_scalar(value, name)
    if value is None or type(value) in (bool, int, float, str):
        return _json_scalar(value, name)
    if is_dataclass(value):
        encoded: dict[str, Any] = {"type": type(value).__qualname__}
        omitted_fields: list[str] = []
        field_values: dict[str, Any] = {}
        for field in fields(value):
            if field.name in _EPHEMERAL_FIELDS:
                continue
            field_value = getattr(value, field.name)
            projected = _scalar_evidence(field_value, f"{name}.{field.name}")
            if projected is _OMITTED:
                omitted_fields.append(field.name)
            else:
                field_values[field.name] = projected
        if omitted_fields:
            field_values["array_presence_digest"] = _hash_json(sorted(omitted_fields))
        encoded["fields"] = field_values
        return encoded
    if isinstance(value, Mapping):
        projected_mapping: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise ValueError(f"{name} mapping keys must be strings")
            projected = _scalar_evidence(value[key], f"{name}[{key!r}]")
            if projected is not _OMITTED:
                projected_mapping[key] = projected
        return projected_mapping
    if isinstance(value, (tuple, list)):
        result: list[Any] = []
        for index, item in enumerate(value):
            projected = _scalar_evidence(item, f"{name}[{index}]")
            result.append(None if projected is _OMITTED else projected)
        return result
    # Runtime/cache identities must be named explicitly in
    # ``_EPHEMERAL_FIELDS``.  An opaque fallback would allow a future durable
    # audit field to disappear from the archive without a schema decision.
    raise TypeError(
        f"{name} contains unsupported non-durable value of type "
        f"{type(value).__qualname__}"
    )


def _gate_payload(value: GateTolerance) -> dict[str, Any]:
    if not isinstance(value, GateTolerance):
        raise TypeError("archive gate value must be GateTolerance")
    return {
        "atol": float(value.atol),
        "rtol": float(value.rtol),
        "scale_floor": float(value.scale_floor),
    }


_ENGINEERING_CONFIG_FIELDS = (
    "load_start",
    "load_final",
    "du_initial",
    "du_min",
    "cutback_factor",
    "max_cutbacks",
    "max_stagger_iterations",
    "cg_rtol",
    "cg_atol",
    "cg_maxiter",
    "displacement",
    "damage",
    "history",
    "reaction",
    "damage_residual",
    "force",
    "moment",
    "healing_atol",
    "damage_range_atol",
    "work_check_load",
    "work_epsilon",
    "work_relative_limit",
)
_NEWTON_CONFIG_FIELDS = (
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


def _engineering_config_payload(config: EngineeringGripConfig) -> dict[str, Any]:
    values: dict[str, Any] = {
        "type": "EngineeringGripConfig",
    }
    for name in _ENGINEERING_CONFIG_FIELDS:
        value = getattr(config, name)
        values[name] = _gate_payload(value) if isinstance(value, GateTolerance) else _json_scalar(value, name)
    values["config_digest"] = config.config_digest
    return values


def _newton_config_payload(config: monolithic.MonolithicNewtonConfig) -> dict[str, Any]:
    values: dict[str, Any] = {"type": "MonolithicNewtonConfig"}
    for name in _NEWTON_CONFIG_FIELDS:
        values[name] = _json_scalar(getattr(config, name), name)
    values["config_digest"] = _hash_json(values)
    return values


def _continuation_config_payload(
    config: monolithic.MonolithicContinuationConfig,
) -> dict[str, Any]:
    return {
        "type": "MonolithicContinuationConfig",
        "load_start": float(config.load_start),
        "load_final": float(config.load_final),
        "du_initial": float(config.du_initial),
        "du_min": float(config.du_min),
        "cutback_factor": float(config.cutback_factor),
        "max_cutbacks": int(config.max_cutbacks),
        "acceptance_mode": config.acceptance_mode,
        "engineering_config_digest": config.engineering_config.config_digest,
        "newton_config_digest": _newton_config_payload(config.newton_config)["config_digest"],
    }


def _work_check_payload(check: Any) -> dict[str, Any]:
    if not isinstance(check, engineering.EngineeringGripWorkCheck):
        raise TypeError("archive work_check must be EngineeringGripWorkCheck")
    return {
        "type": "EngineeringGripWorkCheck",
        "u_bar": float(check.u_bar),
        "epsilon": float(check.epsilon),
        "reaction": float(check.reaction),
        "reaction_fd": float(check.reaction_fd),
        "relative_error": float(check.relative_error),
        "relative_limit": float(check.relative_limit),
        "passed": bool(check.passed),
        "cg_rtol": float(check.cg_rtol),
        "cg_atol": float(check.cg_atol),
        "cg_maxiter": int(check.cg_maxiter),
        "center_result_digest": check.center_result_digest,
        "plus_result_digest": check.plus_result_digest,
        "minus_result_digest": check.minus_result_digest,
    }


def _model_payload(model: EngineeringGripModel) -> dict[str, Any]:
    return {
        "type": "EngineeringGripModelIdentity",
        "model_digest": model.model_digest,
        "static_digest": model.static_digest,
        "material_hash": model.material_hash,
        "union_hash": model.union_hash,
        "n_material": int(model.n_material),
        "n_union": int(model.n_union),
        "n_dofs": int(model.n_dofs),
    }


def _attempt_scalar_evidence_payload(
    attempt: monolithic.MonolithicContinuationAttempt,
) -> dict[str, Any]:
    projected = _scalar_evidence(attempt, "attempt")
    if not isinstance(projected, dict):
        raise TypeError("attempt scalar evidence did not produce an object")
    phase2b = attempt.phase2b
    if phase2b.canonical_scaled_residual is None:
        canonical_norms: list[float] | None = None
    else:
        if phase2b.canonical_scales is None:
            raise ValueError("canonical residual is present without canonical scales")
        owner = monolithic.MonolithicGripResidual(
            phase2b.model,
            phase2b.target_u_bar,
            phase2b.committed_state.material_history,
        )
        canonical_norms = [
            float(value)
            for value in monolithic._scaled_block_norms(
                owner, phase2b.canonical_scaled_residual
            )
        ]
    projected["canonical_scaled_block_norms"] = canonical_norms
    digest = _hash_json(projected)
    return {"scalar_evidence": projected, "scalar_evidence_digest": digest}


def _failure_payload(result: monolithic.MonolithicContinuationResult) -> dict[str, Any]:
    failed_index = None
    if result.failed_attempt is not None:
        failed_index = int(result.failed_attempt.attempt_index)
    return {
        "failed_target": None if result.failed_target is None else float(result.failed_target),
        "failure_reason": result.failure_reason,
        "failed_attempt_index": failed_index,
    }


def _state_arrays(
    result: monolithic.MonolithicContinuationResult,
) -> dict[str, np.ndarray]:
    model = result.model
    states = result.states
    steps = result.accepted_steps
    arrays: dict[str, np.ndarray] = {
        "state_u_bar": np.asarray([state.u_bar for state in states], dtype=np.float64),
        "state_union_displacement": np.stack(
            [np.asarray(state.union_displacement, dtype=np.float64) for state in states], axis=0
        ),
        "state_material_damage": np.stack(
            [np.asarray(state.material_damage, dtype=np.float64) for state in states], axis=0
        ),
        "state_material_history": np.stack(
            [np.asarray(state.material_history, dtype=np.float64) for state in states], axis=0
        ),
    }
    reactions: list[float] = []
    energies: list[float] = []
    for index, step in enumerate(steps):
        auxiliary = step.phase2b.auxiliary
        if auxiliary is None or auxiliary.mechanics_after is None:
            raise ValueError(f"accepted step {index} has no canonical mechanics-after result")
        reactions.append(float(auxiliary.mechanics_after.outer_reaction))
        energies.append(float(auxiliary.mechanics_after.energy))
    arrays["accepted_reaction"] = np.asarray(reactions, dtype=np.float64)
    # This is the A2 mechanics energy evaluated by the canonical
    # mechanics-after solve, not a sum of all fracture or history energies.
    arrays["accepted_mechanics_energy"] = np.asarray(energies, dtype=np.float64)
    for name, value in arrays.items():
        _validate_archive_array(name, value)
    if arrays["state_union_displacement"].shape[1:] != (model.n_dofs,):
        raise ValueError("accepted state displacement shape does not match the owner model")
    if arrays["state_material_damage"].shape[1:] != (model.n_material,):
        raise ValueError("accepted state damage shape does not match the owner model")
    if arrays["state_material_history"].shape[1:] != (model.n_material,):
        raise ValueError("accepted state history shape does not match the owner model")
    return arrays


def _validate_archive_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _FLOAT64:
        raise ValueError(f"archive array {name} must use exact float64 dtype")
    if array.dtype.kind == "O" or not np.all(np.isfinite(array)):
        raise ValueError(f"archive array {name} must be finite and non-object")
    return array


def _metadata_for_result(
    result: monolithic.MonolithicContinuationResult,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    model = result.model
    state_digests = [state.digest for state in result.states]
    attempt_scalar_evidence = [
        _attempt_scalar_evidence_payload(attempt) for attempt in result.attempts
    ]
    metadata: dict[str, Any] = {
        "schema": MONOLITHIC_ARCHIVE_SCHEMA,
        "schema_version": MONOLITHIC_ARCHIVE_SCHEMA_VERSION,
        "acceptance_mode": result.acceptance_mode,
        "completion_status": result.completion_status,
        "model": _model_payload(model),
        "source_bundle": {
            "kind": _SOURCE_BUNDLE_HASH_KIND,
            "files": list(_SOURCE_BUNDLE_FILES),
            "sha256": _source_bundle_hash(),
        },
        "engineering_config": _engineering_config_payload(
            result.config.engineering_config
        ),
        "newton_config": _newton_config_payload(result.config.newton_config),
        "continuation_config": _continuation_config_payload(result.config),
        "work_check": _work_check_payload(result.work_check),
        "counts": {
            "state_count": len(result.states),
            "accepted_count": len(result.accepted_steps),
            "attempt_count": len(result.attempts),
        },
        "state_digests": state_digests,
        "attempt_scalar_evidence": attempt_scalar_evidence,
        "failure": _failure_payload(result),
        "array_contract": {
            "member_names": list(MONOLITHIC_ARCHIVE_MEMBER_NAMES),
            "accepted_mechanics_energy_semantics": _CANONICAL_MECHANICS_ENERGY_SEMANTICS,
            "owner_recomputed_fields": list(_OWNER_RECOMPUTED_FIELDS),
        },
        _ARCHIVE_RESULT_DIGEST_FIELD: "",
    }
    metadata[_ARCHIVE_RESULT_DIGEST_FIELD] = _archive_digest(metadata, arrays)
    return metadata


def _validate_result_for_archive(
    result: monolithic.MonolithicContinuationResult,
) -> None:
    if not isinstance(result, monolithic.MonolithicContinuationResult):
        raise TypeError("result must be MonolithicContinuationResult")
    # This is deliberately the owner boundary required before serialization.
    _validate_archive_owner(result, result.model)
    for index, attempt in enumerate(result.attempts):
        if attempt.phase2b.adapter_kind != "cpu_reference":
            raise ValueError(
                f"attempt {index} is not CPU-authoritative and cannot enter this archive"
            )


def _atomic_write_npz(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    archive_budget_bytes: int,
    *,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing monolithic archive {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    installed_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".npz.tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            _write_zip32_npz(handle, arrays)
            handle.flush()
            os.fsync(handle.fileno())

        _inspect_zip(Path(temporary_name), archive_budget_bytes)
        if overwrite:
            os.replace(temporary_name, destination)
            temporary_name = None
        else:
            # ``exists()`` above is only an early diagnostic.  The hard link is
            # the actual no-clobber operation and remains atomic if another
            # writer creates the destination after that check.
            try:
                os.link(temporary_name, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing monolithic archive {destination}"
                ) from error
            except OSError as error:
                # There is deliberately no os.replace/copy fallback here.  A
                # no-clobber install is fail-closed when hard links are not
                # available on the destination filesystem.
                raise OSError(
                    error.errno,
                    "ZIP32 monolithic archive no-clobber install requires "
                    f"hard links; refusing a fallback for {destination}: {error}",
                ) from error
            identity = os.stat(temporary_name, follow_symlinks=False)
            installed_identity = (int(identity.st_dev), int(identity.st_ino))
            os.unlink(temporary_name)
            temporary_name = None

        directory_fd: int | None = None
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            os.fsync(directory_fd)
        except OSError as error:
            if installed_identity is None:
                raise
            cleanup_status = _cleanup_no_clobber_install(
                destination,
                installed_identity,
                directory_fd,
            )
            raise OSError(
                error.errno,
                "directory fsync failed after no-clobber archive install; "
                f"destination cleanup status: {cleanup_status}",
            ) from error
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _write_zip32_npz(handle: Any, arrays: Mapping[str, np.ndarray]) -> None:
    """Write the canonical v1 NPZ without ever enabling ZIP64."""

    with zipfile.ZipFile(
        handle,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=False,
    ) as output:
        for key in _MEMBER_KEYS:
            array = np.asarray(arrays[key])
            stream = io.BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            payload = stream.getvalue()
            if len(payload) > _ZIP32_MAX_MEMBER_BYTES:
                raise ValueError(
                    f"canonical ZIP32 member {key}.npy exceeds the ZIP32 member limit"
                )
            output.writestr(f"{key}.npy", payload)


def _cleanup_no_clobber_install(
    destination: Path,
    installed_identity: tuple[int, int],
    directory_fd: int | None,
) -> str:
    """Best-effort cleanup after a linked install loses directory durability.

    The identity check avoids unlinking a replacement created by another
    process after the link succeeded.  The returned status is included in the
    raised error; in particular, a missing or replaced destination is never
    silently reported as successfully removed.
    """

    try:
        current = os.stat(destination, follow_symlinks=False)
    except FileNotFoundError:
        status = "destination_already_absent"
    except OSError as error:
        status = f"cleanup_stat_failed:{error}"
    else:
        current_identity = (int(current.st_dev), int(current.st_ino))
        if current_identity != installed_identity:
            status = "cleanup_not_performed:destination_identity_changed"
        else:
            try:
                os.unlink(destination)
            except FileNotFoundError:
                status = "destination_disappeared_before_cleanup"
            except OSError as error:
                status = f"cleanup_unlink_failed:{error}"
            else:
                status = "destination_unlinked"

    if directory_fd is None:
        return f"{status}; cleanup_directory_fsync=not_available"
    try:
        os.fsync(directory_fd)
    except OSError as error:
        return f"{status}; cleanup_directory_fsync_failed:{error}"
    return f"{status}; cleanup_directory_fsync=passed"


def _read_zip32_eocd(
    handle: Any,
    physical_size: int,
    archive_budget_bytes: int,
    expected_member_count: int,
) -> tuple[int, int, int, int, int, int, int]:
    """Read and validate the one canonical ZIP32 EOCD from ``handle``.

    ZIP readers are allowed to locate an EOCD by scanning only the final
    ``65535 + 22`` bytes.  The v1 contract is stricter: exactly one EOCD
    signature may occur in that legal window, its fixed record must end at the
    descriptor's EOF with a zero comment, and all single-disk/count/central
    directory fields must describe this exact allowlist.
    """

    if physical_size < _ZIP_EOCD_SIZE:
        raise ValueError("ZIP32 EOCD record is missing from the archive")
    search_start = max(
        0,
        physical_size - _ZIP_EOCD_SIZE - _ZIP32_MAX_UINT16,
    )
    handle.seek(search_start)
    tail = handle.read(physical_size - search_start)
    if len(tail) != physical_size - search_start:
        raise ValueError("ZIP32 EOCD search window is truncated")

    signatures: list[int] = []
    cursor = 0
    while True:
        relative = tail.find(_ZIP_EOCD_SIGNATURE, cursor)
        if relative < 0:
            break
        signatures.append(relative)
        cursor = relative + 1
    if not signatures:
        raise ValueError("ZIP32 EOCD record is missing from the archive")
    if len(signatures) != 1:
        raise ValueError(
            "ZIP32 EOCD is ambiguous or has a preceding/trailing EOCD record"
        )

    relative = signatures[0]
    eocd_offset = search_start + relative
    if relative + _ZIP_EOCD_SIZE > len(tail):
        raise ValueError("ZIP32 EOCD fixed record is truncated")
    (
        signature,
        disk_number,
        central_start_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = _ZIP_EOCD.unpack(tail[relative : relative + _ZIP_EOCD_SIZE])
    if signature != _ZIP_EOCD_SIGNATURE:
        raise ValueError("ZIP32 EOCD signature is invalid")
    if comment_length != 0:
        raise ValueError("ZIP32 EOCD comments are forbidden")
    if eocd_offset + _ZIP_EOCD_SIZE != physical_size:
        raise ValueError("ZIP32 EOCD record does not end at archive EOF")
    # A ZIP64 locator, when present, occupies the fixed 20 bytes immediately
    # before the classic EOCD. Do not scan member payload bytes for record
    # signatures: compressed data may contain those sequences by chance.
    if eocd_offset >= 20:
        handle.seek(eocd_offset - 20)
        predecessor = handle.read(20)
        if len(predecessor) != 20:
            raise ValueError("ZIP32 EOCD predecessor record is truncated")
        if predecessor.startswith(_ZIP64_LOCATOR_SIGNATURE):
            raise ValueError(
                "ZIP64 locator precedes the ZIP32 EOCD; ZIP64 is not supported"
            )
    if disk_number != 0:
        raise ValueError("ZIP32 EOCD disk number must be zero")
    if central_start_disk != 0:
        raise ValueError("ZIP32 EOCD central-directory start disk must be zero")
    if entries_on_disk == _ZIP32_MAX_UINT16 or total_entries == _ZIP32_MAX_UINT16:
        raise ValueError("ZIP32 EOCD entry-count sentinel is forbidden")
    if entries_on_disk != expected_member_count or total_entries != expected_member_count:
        raise ValueError(
            "ZIP32 EOCD entry counts do not match the exact archive allowlist"
        )
    if entries_on_disk != total_entries:
        raise ValueError("ZIP32 EOCD entries-on-disk and total entries differ")
    if central_size == _ZIP32_MAX_UINT32 or central_offset == _ZIP32_MAX_UINT32:
        raise ValueError("ZIP32 EOCD central-directory sentinel is forbidden")
    if central_offset > physical_size or central_size > physical_size:
        raise ValueError("ZIP32 EOCD central-directory fields exceed file bounds")
    if central_size > archive_budget_bytes + _ZIP_OVERHEAD_BUDGET_BYTES:
        raise ValueError("ZIP32 EOCD central-directory size exceeds the archive budget")
    central_end = central_offset + central_size
    if central_end > physical_size:
        raise ValueError("ZIP32 EOCD central directory exceeds file bounds")
    if central_end != eocd_offset:
        raise ValueError(
            "ZIP32 EOCD central-directory size/offset does not end at the EOCD"
        )
    return (
        eocd_offset,
        disk_number,
        central_start_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
    )


def _inspect_zip_handle(handle: Any, archive_budget_bytes: int) -> None:
    """Inspect one ZIP from an already-open file object.

    The caller may subsequently read members from the same handle.  Keeping
    inspection and member loading on that descriptor avoids accepting one
    inode and loading a replacement path supplied by a concurrent writer.
    """

    budget = _require_budget(archive_budget_bytes)
    try:
        handle.seek(0, os.SEEK_END)
        physical_size = int(handle.tell())
        if physical_size > _ZIP32_MAX_ARCHIVE_BYTES:
            raise ValueError(
                "monolithic archive exceeds the ZIP32 physical-size limit; "
                "ZIP64 archives are not supported"
            )
        if physical_size > budget + _ZIP_OVERHEAD_BUDGET_BYTES:
            raise ValueError(
                "monolithic archive exceeds the physical ZIP budget "
                f"({physical_size} > {budget + _ZIP_OVERHEAD_BUDGET_BYTES})"
            )
        eocd = _read_zip32_eocd(
            handle,
            physical_size,
            budget,
            len(MONOLITHIC_ARCHIVE_MEMBER_NAMES),
        )
        handle.seek(0)
        with zipfile.ZipFile(handle, mode="r", allowZip64=False) as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if names != MONOLITHIC_ARCHIVE_MEMBER_NAMES:
                raise ValueError(
                    "monolithic archive members differ from the exact allowlist/order"
                )
            if len(set(names)) != len(names):
                raise ValueError("monolithic archive contains duplicate members")
            if len(infos) > _ZIP32_MAX_UINT16:
                raise ValueError(
                    "monolithic archive contains more than the ZIP32 member-count limit"
                )
            if archive.comment:
                raise ValueError("monolithic archive comments are forbidden")
            if len(archive.comment) > _ZIP32_MAX_UINT16:
                raise ValueError("ZIP archive comment exceeds the ZIP32 limit")

            # ``zipfile`` expands ZIP64 values from the central extra field
            # before exposing ``ZipInfo``.  Parse the raw central records too,
            # so a ZIP64 sentinel cannot be mistaken for an ordinary v1 size.
            (
                eocd_offset,
                _eocd_disk_number,
                _eocd_central_start_disk,
                eocd_entries_on_disk,
                eocd_total_entries,
                eocd_central_size,
                eocd_central_offset,
            ) = eocd
            if eocd_entries_on_disk != len(infos) or eocd_total_entries != len(infos):
                raise ValueError("ZIP32 EOCD entry counts differ from parsed central records")
            central_offset = int(archive.start_dir)
            if central_offset < 0 or central_offset > _ZIP32_MAX_UINT32:
                raise ValueError(
                    "ZIP central-directory offset exceeds the ZIP32 limit; "
                    "ZIP64 archives are not supported"
                )
            if central_offset > physical_size:
                raise ValueError("ZIP central-directory offset is beyond the archive")
            if central_offset != eocd_central_offset:
                raise ValueError(
                    "ZIP32 EOCD central-directory offset differs from parsed central records"
                )
            handle.seek(central_offset)
            for info in infos:
                raw_header = handle.read(_ZIP_CENTRAL_HEADER_SIZE)
                if len(raw_header) != _ZIP_CENTRAL_HEADER_SIZE:
                    raise ValueError("ZIP central-directory header is truncated")
                (
                    central_signature,
                    _version_made,
                    _version_needed,
                    _central_flags,
                    _central_method,
                    _central_mtime,
                    _central_mdate,
                    _central_crc,
                    central_compressed_size,
                    central_file_size,
                    central_name_length,
                    central_extra_length,
                    central_comment_length,
                    central_disk_start,
                    _central_internal_attr,
                    _central_external_attr,
                    central_header_offset,
                ) = _ZIP_CENTRAL_HEADER.unpack(raw_header)
                if central_signature != b"PK\x01\x02":
                    raise ValueError("ZIP central-directory signature is invalid")
                central_name = handle.read(central_name_length)
                central_extra = handle.read(central_extra_length)
                central_comment = handle.read(central_comment_length)
                if (
                    central_compressed_size == _ZIP32_MAX_UINT32
                    or central_file_size == _ZIP32_MAX_UINT32
                    or central_header_offset == _ZIP32_MAX_UINT32
                    or central_disk_start == _ZIP32_MAX_UINT16
                ):
                    raise ValueError(
                        "ZIP64 central-directory sentinel is forbidden in the "
                        f"ZIP32-only monolithic archive for {info.filename}"
                    )
                if central_extra:
                    if _ZIP64_EXTRA_FIELD_ID.to_bytes(2, "little") in central_extra:
                        raise ValueError(
                            "ZIP64 central-directory extra field is forbidden in the "
                            f"ZIP32-only monolithic archive for {info.filename}"
                        )
                    raise ValueError(
                        "ZIP central-directory extra fields are forbidden in the "
                        f"ZIP32-only monolithic archive for {info.filename}"
                    )
                if central_comment:
                    raise ValueError(
                        "ZIP central-directory comments are forbidden in the "
                        f"ZIP32-only monolithic archive for {info.filename}"
                    )
                try:
                    expected_name = info.filename.encode("ascii")
                except UnicodeEncodeError as error:
                    raise ValueError(
                        f"ZIP member name is not ASCII: {info.filename!r}"
                    ) from error
                if central_name != expected_name:
                    raise ValueError(
                        "ZIP central-directory name differs from decoded metadata "
                        f"for {info.filename}"
                    )
                if central_disk_start != 0:
                    raise ValueError(
                        "ZIP central-directory record disk start must be zero"
                    )
                if (
                    _central_flags != info.flag_bits
                    or _central_method != info.compress_type
                    or _central_crc != info.CRC
                    or central_compressed_size != info.compress_size
                    or central_file_size != info.file_size
                    or central_header_offset != info.header_offset
                ):
                    raise ValueError(
                        "ZIP central-directory values differ from decoded metadata "
                        f"for {info.filename}"
                    )
            central_records_end = int(handle.tell())
            central_size = central_records_end - central_offset
            if central_size < 0 or central_size > _ZIP32_MAX_UINT32:
                raise ValueError(
                    "ZIP central-directory size exceeds the ZIP32 limit; "
                    "ZIP64 archives are not supported"
                )
            if central_size != eocd_central_size or central_records_end != eocd_offset:
                raise ValueError(
                    "ZIP32 EOCD central-directory size/location differs from parsed records"
                )

            total = 0
            compressed_total = 0
            previous_data_end = 0
            for info in infos:
                name = info.filename
                pure = PurePosixPath(name)
                if (
                    "\\" in name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or pure.as_posix() != name
                    or info.is_dir()
                ):
                    raise ValueError(f"monolithic archive contains unsafe member {name!r}")
                if info.flag_bits != 0:
                    raise ValueError(
                        f"monolithic archive member flags are forbidden for {name}: "
                        f"0x{info.flag_bits:04x}"
                    )
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError(f"unsupported archive compression for {name}")
                if info.extra or info.comment:
                    raise ValueError(
                        "ZIP extras/comments are forbidden in the ZIP32-only "
                        f"monolithic archive for {name}"
                    )
                mode = (int(info.external_attr) >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type not in (0, stat.S_IFREG) or stat.S_ISLNK(mode):
                    raise ValueError(f"special-file ZIP metadata is forbidden for {name}")
                if int(info.external_attr) & 0x10:
                    raise ValueError(f"directory ZIP metadata is forbidden for {name}")
                if (
                    info.file_size < 0
                    or info.compress_size < 0
                    or info.file_size > _ZIP32_MAX_MEMBER_BYTES
                    or info.compress_size > _ZIP32_MAX_MEMBER_BYTES
                ):
                    raise ValueError(f"archive member has invalid size: {name}")
                if info.file_size > budget:
                    raise ValueError(
                        f"archive member {name} exceeds uncompressed byte budget"
                    )
                total += info.file_size
                compressed_total += info.compress_size
                if total > budget:
                    raise ValueError(
                        "monolithic archive exceeds required uncompressed byte budget "
                        f"({total} > {budget})"
                    )
                if compressed_total > budget:
                    raise ValueError(
                        "monolithic archive exceeds required compressed byte budget "
                        f"({compressed_total} > {budget})"
                    )
                if info.file_size > _MAX_COMPRESSION_RATIO * max(info.compress_size, 1):
                    raise ValueError(
                        f"archive member {name} exceeds the maximum compression ratio"
                    )

                offset = int(info.header_offset)
                if (
                    offset < previous_data_end
                    or offset + _ZIP_LOCAL_HEADER_SIZE > archive.start_dir
                ):
                    raise ValueError(f"ZIP local header offset is invalid for {name}")
                handle.seek(offset)
                local = handle.read(_ZIP_LOCAL_HEADER_SIZE)
                if len(local) != _ZIP_LOCAL_HEADER_SIZE:
                    raise ValueError(f"ZIP local header is truncated for {name}")
                (
                    signature,
                    _version,
                    local_flags,
                    local_method,
                    _mtime,
                    _mdate,
                    local_crc,
                    local_compressed_size,
                    local_file_size,
                    local_name_length,
                    local_extra_length,
                ) = _ZIP_LOCAL_HEADER.unpack(local)
                if signature != b"PK\x03\x04":
                    raise ValueError(f"ZIP local header signature is invalid for {name}")
                local_name = handle.read(local_name_length)
                local_extra = handle.read(local_extra_length)
                try:
                    expected_name = name.encode("ascii")
                except UnicodeEncodeError as error:
                    raise ValueError(f"ZIP member name is not ASCII: {name!r}") from error
                if local_name != expected_name:
                    raise ValueError(f"ZIP local name differs from central metadata for {name}")
                if local_file_size == _ZIP32_MAX_UINT32 or local_compressed_size == _ZIP32_MAX_UINT32:
                    raise ValueError(
                        "ZIP64 local-header sentinel is forbidden in the "
                        f"ZIP32-only monolithic archive for {name}"
                    )
                if local_extra:
                    if _ZIP64_EXTRA_FIELD_ID.to_bytes(2, "little") in local_extra:
                        raise ValueError(
                            "ZIP64 local-header extra field is forbidden in the "
                            f"ZIP32-only monolithic archive for {name}"
                        )
                    raise ValueError(
                        "ZIP local-header extra fields are forbidden in the "
                        f"ZIP32-only monolithic archive for {name}"
                    )
                if (
                    local_flags != info.flag_bits
                    or local_method != info.compress_type
                    or local_crc != info.CRC
                    or local_compressed_size != info.compress_size
                    or local_file_size != info.file_size
                ):
                    raise ValueError(f"ZIP local header differs from central metadata for {name}")
                data_start = (
                    offset
                    + _ZIP_LOCAL_HEADER_SIZE
                    + local_name_length
                    + local_extra_length
                )
                data_end = data_start + int(info.compress_size)
                if data_end > archive.start_dir or data_end > physical_size:
                    raise ValueError(f"ZIP member data extends beyond its archive for {name}")
                previous_data_end = data_end
        handle.seek(0)
    except zipfile.LargeZipFile as error:
        raise ValueError(
            "ZIP64 central/end records are forbidden in the ZIP32-only "
            "monolithic archive"
        ) from error
    except zipfile.BadZipFile as error:
        raise ValueError("monolithic archive is not a valid ZIP/NPZ") from error
    except (EOFError, OSError, struct.error) as error:
        raise ValueError("monolithic archive ZIP structure cannot be inspected") from error


def _inspect_zip(path: Path, archive_budget_bytes: int) -> None:
    try:
        with path.open("rb") as handle:
            _inspect_zip_handle(handle, archive_budget_bytes)
    except zipfile.LargeZipFile as error:
        raise ValueError(
            "ZIP64 central/end records are forbidden in the ZIP32-only "
            "monolithic archive"
        ) from error
    except zipfile.BadZipFile as error:
        raise ValueError("monolithic archive is not a valid ZIP/NPZ") from error


def write_monolithic_continuation_archive(
    path: Any,
    result: monolithic.MonolithicContinuationResult,
    archive_budget_bytes: int,
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically write one lean monolithic continuation archive.

    ``archive_budget_bytes`` is required and applies to the sum of ZIP member
    uncompressed sizes.  The budget is checked before the temporary file can
    replace ``path``.  Only CPU-reference owner-authenticated results are
    accepted; this private schema has no warning or adapter acceptance mode.
    """

    destination = _path(path, "archive path")
    budget = _require_budget(archive_budget_bytes)
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing monolithic archive {destination}"
        )
    _validate_result_for_archive(result)
    arrays = _state_arrays(result)
    metadata = _metadata_for_result(result, arrays)
    archive_arrays: dict[str, np.ndarray] = {
        "metadata_json": _immutable(_canonical_json(metadata), dtype=np.uint8),
        **{name: _immutable(value, dtype=np.float64) for name, value in arrays.items()},
    }
    for name, value in archive_arrays.items():
        if np.asarray(value).dtype.kind == "O":
            raise ValueError(f"archive field {name} has forbidden object dtype")
    _atomic_write_npz(destination, archive_arrays, budget, overwrite=overwrite)


def _decode_npy_member(name: str, payload: bytes) -> np.ndarray:
    """Decode one exact, C-order, non-object NPY member."""

    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version in ((1, 0), (2, 0)):
            header_length_width = 2 if version == (1, 0) else 4
            header_length_bytes = stream.read(header_length_width)
            if len(header_length_bytes) != header_length_width:
                raise ValueError("NPY header length is truncated")
            header_length = int.from_bytes(header_length_bytes, "little")
            if header_length > _NPY_MAX_HEADER_SIZE:
                raise ValueError("NPY header exceeds the archive header limit")
            stream.seek(8)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"archive NPY member {name} has unsupported version {version}")
    except (ValueError, TypeError, EOFError, struct.error) as error:
        raise ValueError(f"archive NPY member {name} has an invalid header") from error

    if fortran_order:
        raise ValueError(f"archive NPY member {name} must use C order")
    if dtype.hasobject:
        raise ValueError(f"archive NPY member {name} has forbidden object dtype")
    expected_dtype = _UINT8 if name == "metadata_json" else _FLOAT64
    if dtype != expected_dtype:
        raise ValueError(
            f"archive NPY member {name} must use exact {expected_dtype} dtype, got {dtype}"
        )
    if not isinstance(shape, tuple) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError(f"archive NPY member {name} has an invalid shape")
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    payload_start = int(stream.tell())
    expected_payload_bytes = element_count * int(dtype.itemsize)
    if len(payload) - payload_start != expected_payload_bytes:
        raise ValueError(f"archive NPY member {name} has truncated or trailing payload bytes")

    stream.seek(0)
    try:
        array = np.lib.format.read_array(
            stream,
            allow_pickle=False,
        )
    except (ValueError, EOFError, OSError) as error:
        raise ValueError(f"archive NPY member {name} cannot be decoded") from error
    if stream.tell() != len(payload):
        raise ValueError(f"archive NPY member {name} has trailing bytes")
    array = np.asarray(array)
    if array.dtype != expected_dtype or not array.flags.c_contiguous:
        raise ValueError(f"archive NPY member {name} is not an exact C-order array")
    return _immutable(array, dtype=expected_dtype)


def _read_archive_snapshot(handle: Any, archive_budget_bytes: int) -> bytes:
    """Read one bounded immutable snapshot from an already-open archive."""

    budget = _require_budget(archive_budget_bytes)
    handle.seek(0, os.SEEK_END)
    physical_size = int(handle.tell())
    if physical_size > _ZIP32_MAX_ARCHIVE_BYTES:
        raise ValueError(
            "monolithic archive exceeds the ZIP32 physical-size limit; "
            "ZIP64 archives are not supported"
        )
    if physical_size > budget + _ZIP_OVERHEAD_BUDGET_BYTES:
        raise ValueError(
            "monolithic archive exceeds the physical ZIP budget "
            f"({physical_size} > {budget + _ZIP_OVERHEAD_BUDGET_BYTES})"
        )
    handle.seek(0)
    payload = handle.read(physical_size)
    if len(payload) != physical_size or handle.read(1):
        raise ValueError("monolithic archive changed while its snapshot was read")
    return payload


def _load_npz_arrays(path: Path, archive_budget_bytes: int) -> tuple[dict[str, np.ndarray], bytes]:
    try:
        with path.open("rb") as handle:
            # Snapshot bounded bytes from the opened descriptor first. Path
            # replacement cannot switch the inode, and an in-place writer
            # cannot alter these bytes between validation and decompression.
            snapshot = _read_archive_snapshot(handle, archive_budget_bytes)
            snapshot_handle = io.BytesIO(snapshot)
            _inspect_zip_handle(snapshot_handle, archive_budget_bytes)
            snapshot_handle.seek(0)
            with zipfile.ZipFile(snapshot_handle, mode="r", allowZip64=False) as archive:
                names = tuple(info.filename for info in archive.infolist())
                if names != MONOLITHIC_ARCHIVE_MEMBER_NAMES:
                    raise ValueError("loaded NPZ fields differ from the exact allowlist/order")
                payloads = {
                    name: archive.read(name)
                    for name in names
                }
            arrays = {
                name.removesuffix(".npy"): _decode_npy_member(
                    name.removesuffix(".npy"), payloads[name]
                )
                for name in names
            }
    except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot load monolithic archive {path}: {error}") from error
    metadata, raw = _decode_metadata(arrays["metadata_json"])
    for name in _ARRAY_KEYS:
        _validate_archive_array(name, arrays[name])
    return arrays, raw


def _parse_gate(value: Any, name: str) -> GateTolerance:
    payload = _exact_keys(value, ("atol", "rtol", "scale_floor"), name)
    return GateTolerance(
        _json_float(payload["atol"], f"{name}.atol"),
        _json_float(payload["rtol"], f"{name}.rtol"),
        _json_float(payload["scale_floor"], f"{name}.scale_floor"),
    )


def _parse_engineering_config(value: Any) -> EngineeringGripConfig:
    expected = ("type", *_ENGINEERING_CONFIG_FIELDS, "config_digest")
    payload = _exact_keys(value, expected, "engineering_config")
    if payload["type"] != "EngineeringGripConfig":
        raise ValueError("engineering_config type is not canonical")
    gates = {
        name: _parse_gate(payload[name], f"engineering_config.{name}")
        for name in (
            "displacement",
            "damage",
            "history",
            "reaction",
            "damage_residual",
            "force",
            "moment",
        )
    }
    config = EngineeringGripConfig(
        load_start=_json_float(payload["load_start"], "engineering_config.load_start"),
        load_final=_json_float(payload["load_final"], "engineering_config.load_final"),
        du_initial=_json_float(payload["du_initial"], "engineering_config.du_initial"),
        du_min=_json_float(payload["du_min"], "engineering_config.du_min"),
        cutback_factor=_json_float(payload["cutback_factor"], "engineering_config.cutback_factor"),
        max_cutbacks=_json_int(payload["max_cutbacks"], "engineering_config.max_cutbacks"),
        max_stagger_iterations=_json_int(
            payload["max_stagger_iterations"], "engineering_config.max_stagger_iterations"
        ),
        cg_rtol=_json_float(payload["cg_rtol"], "engineering_config.cg_rtol"),
        cg_atol=_json_float(payload["cg_atol"], "engineering_config.cg_atol"),
        cg_maxiter=_json_int(payload["cg_maxiter"], "engineering_config.cg_maxiter"),
        displacement=gates["displacement"],
        damage=gates["damage"],
        history=gates["history"],
        reaction=gates["reaction"],
        damage_residual=gates["damage_residual"],
        force=gates["force"],
        moment=gates["moment"],
        healing_atol=_json_float(payload["healing_atol"], "engineering_config.healing_atol"),
        damage_range_atol=_json_float(
            payload["damage_range_atol"], "engineering_config.damage_range_atol"
        ),
        work_check_load=_json_float(payload["work_check_load"], "engineering_config.work_check_load"),
        work_epsilon=_json_float(payload["work_epsilon"], "engineering_config.work_epsilon"),
        work_relative_limit=_json_float(
            payload["work_relative_limit"], "engineering_config.work_relative_limit"
        ),
    )
    if config.config_digest != _json_digest(payload["config_digest"], "engineering_config.config_digest"):
        raise ValueError("engineering_config digest differs from its canonical fields")
    if _canonical_json(_engineering_config_payload(config)) != _canonical_json(payload):
        raise ValueError("engineering_config is not canonical for its reconstructed values")
    return config


def _parse_newton_config(value: Any) -> monolithic.MonolithicNewtonConfig:
    expected = ("type", *_NEWTON_CONFIG_FIELDS, "config_digest")
    payload = _exact_keys(value, expected, "newton_config")
    if payload["type"] != "MonolithicNewtonConfig":
        raise ValueError("newton_config type is not canonical")
    config = monolithic.MonolithicNewtonConfig(
        max_newton_iterations=_json_int(
            payload["max_newton_iterations"], "newton_config.max_newton_iterations"
        ),
        gmres_rtol=_json_float(payload["gmres_rtol"], "newton_config.gmres_rtol"),
        gmres_atol=_json_float(payload["gmres_atol"], "newton_config.gmres_atol"),
        gmres_restart=_json_int(payload["gmres_restart"], "newton_config.gmres_restart"),
        gmres_maxiter=_json_int(payload["gmres_maxiter"], "newton_config.gmres_maxiter"),
        armijo_c=_json_float(payload["armijo_c"], "newton_config.armijo_c"),
        line_search_reduction=_json_float(
            payload["line_search_reduction"], "newton_config.line_search_reduction"
        ),
        line_search_max_backtracks=_json_int(
            payload["line_search_max_backtracks"], "newton_config.line_search_max_backtracks"
        ),
        line_search_alpha_min=_json_float(
            payload["line_search_alpha_min"], "newton_config.line_search_alpha_min"
        ),
    )
    expected_digest = _hash_json({key: payload[key] for key in ("type", *_NEWTON_CONFIG_FIELDS)})
    if _json_digest(payload["config_digest"], "newton_config.config_digest") != expected_digest:
        raise ValueError("newton_config digest differs from its canonical fields")
    if _canonical_json(_newton_config_payload(config)) != _canonical_json(payload):
        raise ValueError("newton_config is not canonical for its reconstructed values")
    return config


def _parse_continuation_config(
    value: Any,
    engineering_config: EngineeringGripConfig,
    newton_config: monolithic.MonolithicNewtonConfig,
) -> monolithic.MonolithicContinuationConfig:
    expected = (
        "type",
        "load_start",
        "load_final",
        "du_initial",
        "du_min",
        "cutback_factor",
        "max_cutbacks",
        "acceptance_mode",
        "engineering_config_digest",
        "newton_config_digest",
    )
    payload = _exact_keys(value, expected, "continuation_config")
    if payload["type"] != "MonolithicContinuationConfig":
        raise ValueError("continuation_config type is not canonical")
    if payload["acceptance_mode"] != _ACCEPTANCE_MODE:
        raise ValueError("continuation_config acceptance_mode is not strict")
    if _json_digest(payload["engineering_config_digest"], "continuation_config.engineering_config_digest") != engineering_config.config_digest:
        raise ValueError("continuation_config engineering digest is not paired")
    newton_digest = _newton_config_payload(newton_config)["config_digest"]
    if _json_digest(payload["newton_config_digest"], "continuation_config.newton_config_digest") != newton_digest:
        raise ValueError("continuation_config Newton digest is not paired")
    config = monolithic.MonolithicContinuationConfig(
        load_start=_json_float(payload["load_start"], "continuation_config.load_start"),
        load_final=_json_float(payload["load_final"], "continuation_config.load_final"),
        du_initial=_json_float(payload["du_initial"], "continuation_config.du_initial"),
        du_min=_json_float(payload["du_min"], "continuation_config.du_min"),
        cutback_factor=_json_float(payload["cutback_factor"], "continuation_config.cutback_factor"),
        max_cutbacks=_json_int(payload["max_cutbacks"], "continuation_config.max_cutbacks"),
        engineering_config=engineering_config,
        newton_config=newton_config,
    )
    if _canonical_json(_continuation_config_payload(config)) != _canonical_json(payload):
        raise ValueError("continuation_config is not canonical for its reconstructed values")
    return config


def _validate_top_metadata(
    metadata: dict[str, Any],
    model: EngineeringGripModel,
    arrays: Mapping[str, np.ndarray],
) -> tuple[monolithic.MonolithicContinuationConfig, tuple[str, ...], tuple[str, ...]]:
    expected = (
        "schema",
        "schema_version",
        "acceptance_mode",
        "completion_status",
        "model",
        "source_bundle",
        "engineering_config",
        "newton_config",
        "continuation_config",
        "work_check",
        "counts",
        "state_digests",
        "attempt_scalar_evidence",
        "failure",
        "array_contract",
        "result_digest",
    )
    _exact_keys(metadata, expected, "metadata")
    if metadata["schema"] != MONOLITHIC_ARCHIVE_SCHEMA:
        raise ValueError("monolithic archive schema is not supported")
    if _json_int(metadata["schema_version"], "metadata.schema_version") != MONOLITHIC_ARCHIVE_SCHEMA_VERSION:
        raise ValueError("monolithic archive schema version is not supported")
    if metadata["acceptance_mode"] != _ACCEPTANCE_MODE:
        raise ValueError("monolithic archive acceptance mode is not strict")
    if metadata["completion_status"] not in _COMPLETION_STATUSES:
        raise ValueError("monolithic archive completion status is invalid")

    model_payload = _exact_keys(
        metadata["model"],
        (
            "type",
            "model_digest",
            "static_digest",
            "material_hash",
            "union_hash",
            "n_material",
            "n_union",
            "n_dofs",
        ),
        "metadata.model",
    )
    if model_payload["type"] != "EngineeringGripModelIdentity":
        raise ValueError("metadata.model type is not canonical")
    for name, actual in (
        ("model_digest", model.model_digest),
        ("static_digest", model.static_digest),
        ("material_hash", model.material_hash),
        ("union_hash", model.union_hash),
    ):
        if _json_digest(model_payload[name], f"metadata.model.{name}") != actual:
            raise ValueError(f"metadata model {name} differs from supplied owner")
    for name, actual in (
        ("n_material", model.n_material),
        ("n_union", model.n_union),
        ("n_dofs", model.n_dofs),
    ):
        if _json_int(model_payload[name], f"metadata.model.{name}") != actual:
            raise ValueError(f"metadata model {name} differs from supplied owner")

    source = _exact_keys(
        metadata["source_bundle"],
        ("kind", "files", "sha256"),
        "metadata.source_bundle",
    )
    if source["kind"] != _SOURCE_BUNDLE_HASH_KIND:
        raise ValueError("monolithic archive source-bundle kind is not canonical")
    if source["files"] != list(_SOURCE_BUNDLE_FILES):
        raise ValueError("monolithic archive source-bundle file list differs")
    if _json_digest(source["sha256"], "metadata.source_bundle.sha256") != _source_bundle_hash():
        raise ValueError("monolithic archive source bundle differs from current code")

    engineering_config = _parse_engineering_config(metadata["engineering_config"])
    newton_config = _parse_newton_config(metadata["newton_config"])
    continuation_config = _parse_continuation_config(
        metadata["continuation_config"], engineering_config, newton_config
    )
    if metadata["acceptance_mode"] != continuation_config.acceptance_mode:
        raise ValueError("metadata continuation mode is not paired")

    work = _exact_keys(
        metadata["work_check"],
        (
            "type",
            "u_bar",
            "epsilon",
            "reaction",
            "reaction_fd",
            "relative_error",
            "relative_limit",
            "passed",
            "cg_rtol",
            "cg_atol",
            "cg_maxiter",
            "center_result_digest",
            "plus_result_digest",
            "minus_result_digest",
        ),
        "metadata.work_check",
    )
    if work["type"] != "EngineeringGripWorkCheck":
        raise ValueError("metadata.work_check type is not canonical")
    for name in (
        "u_bar",
        "epsilon",
        "reaction",
        "reaction_fd",
        "relative_error",
        "relative_limit",
        "cg_rtol",
        "cg_atol",
    ):
        _json_float(work[name], f"metadata.work_check.{name}")
    _json_bool(work["passed"], "metadata.work_check.passed")
    _json_int(work["cg_maxiter"], "metadata.work_check.cg_maxiter")
    for name in ("center_result_digest", "plus_result_digest", "minus_result_digest"):
        _json_digest(work[name], f"metadata.work_check.{name}")
    if work["passed"] is not True:
        raise ValueError("monolithic archive cannot contain a failed work check")

    counts = _exact_keys(
        metadata["counts"],
        ("state_count", "accepted_count", "attempt_count"),
        "metadata.counts",
    )
    state_count = _json_int(counts["state_count"], "metadata.counts.state_count")
    accepted_count = _json_int(counts["accepted_count"], "metadata.counts.accepted_count")
    attempt_count = _json_int(counts["attempt_count"], "metadata.counts.attempt_count")
    if state_count != accepted_count + 1:
        raise ValueError("metadata counts do not satisfy state = accepted + initial")
    if arrays["state_u_bar"].shape != (state_count,):
        raise ValueError("state_u_bar shape does not match metadata counts")
    if arrays["state_union_displacement"].shape != (state_count, model.n_dofs):
        raise ValueError("state displacement shape does not match owner/counts")
    if arrays["state_material_damage"].shape != (state_count, model.n_material):
        raise ValueError("state damage shape does not match owner/counts")
    if arrays["state_material_history"].shape != (state_count, model.n_material):
        raise ValueError("state history shape does not match owner/counts")
    if arrays["accepted_reaction"].shape != (accepted_count,):
        raise ValueError("accepted reaction shape does not match metadata counts")
    if arrays["accepted_mechanics_energy"].shape != (accepted_count,):
        raise ValueError("accepted mechanics energy shape does not match metadata counts")

    state_digest_values = metadata["state_digests"]
    if not isinstance(state_digest_values, list) or len(state_digest_values) != state_count:
        raise ValueError("metadata.state_digests has the wrong length")
    state_digests = tuple(
        _json_digest(value, f"metadata.state_digests[{index}]")
        for index, value in enumerate(state_digest_values)
    )
    for index in range(state_count):
        expected = engineering._state_digest(
            arrays["state_u_bar"][index],
            arrays["state_union_displacement"][index],
            arrays["state_material_damage"][index],
            arrays["state_material_history"][index],
        )
        if state_digests[index] != expected:
            raise ValueError(f"archive state digest {index} is not paired with its arrays")

    attempt_values = metadata["attempt_scalar_evidence"]
    if not isinstance(attempt_values, list) or len(attempt_values) != attempt_count:
        raise ValueError("metadata.attempt_scalar_evidence has the wrong length")
    attempt_digests: list[str] = []
    for index, item in enumerate(attempt_values):
        record = _exact_keys(
            item,
            ("scalar_evidence", "scalar_evidence_digest"),
            f"metadata.attempt_scalar_evidence[{index}]",
        )
        if not isinstance(record["scalar_evidence"], dict):
            raise ValueError(
                f"metadata.attempt_scalar_evidence[{index}].scalar_evidence "
                "must be an object"
            )
        digest = _json_digest(
            record["scalar_evidence_digest"],
            f"metadata.attempt_scalar_evidence[{index}].scalar_evidence_digest",
        )
        if digest != _hash_json(record["scalar_evidence"]):
            raise ValueError(
                f"metadata.attempt_scalar_evidence[{index}] digest is not paired"
            )
        attempt_digests.append(digest)

    failure = _exact_keys(
        metadata["failure"],
        ("failed_target", "failure_reason", "failed_attempt_index"),
        "metadata.failure",
    )
    if failure["failed_target"] is not None:
        _json_float(failure["failed_target"], "metadata.failure.failed_target")
    if failure["failure_reason"] is not None:
        _json_string(failure["failure_reason"], "metadata.failure.failure_reason")
    if failure["failed_attempt_index"] is not None:
        _json_int(failure["failed_attempt_index"], "metadata.failure.failed_attempt_index")
    if metadata["completion_status"] == "completed" and any(
        value is not None for value in failure.values()
    ):
        raise ValueError("completed archive retains failure provenance")
    if metadata["completion_status"] == "failed" and (
        failure["failed_target"] is None
        or failure["failure_reason"] is None
        or failure["failed_attempt_index"] is None
    ):
        raise ValueError("failed archive is missing failure provenance")
    if failure["failed_attempt_index"] is not None and failure["failed_attempt_index"] != attempt_count - 1:
        raise ValueError("failed attempt index is not the terminal attempt")

    contract = _exact_keys(
        metadata["array_contract"],
        (
            "member_names",
            "accepted_mechanics_energy_semantics",
            "owner_recomputed_fields",
        ),
        "metadata.array_contract",
    )
    if contract["member_names"] != list(MONOLITHIC_ARCHIVE_MEMBER_NAMES):
        raise ValueError("metadata array member contract differs from the allowlist")
    if (
        contract["accepted_mechanics_energy_semantics"]
        != _CANONICAL_MECHANICS_ENERGY_SEMANTICS
    ):
        raise ValueError("metadata accepted mechanics energy semantics are not canonical")
    if contract["owner_recomputed_fields"] != list(_OWNER_RECOMPUTED_FIELDS):
        raise ValueError("metadata owner-recomputed field contract is not canonical")
    _json_digest(metadata["result_digest"], "metadata.result_digest")

    return continuation_config, state_digests, tuple(attempt_digests)


def _expected_arrays_from_result(
    result: monolithic.MonolithicContinuationResult,
) -> dict[str, np.ndarray]:
    return _state_arrays(result)


def _compare_arrays(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
) -> None:
    for name in _ARRAY_KEYS:
        left = np.asarray(actual[name])
        right = np.asarray(expected[name])
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"archive array {name} differs from authoritative replay")


def _assert_owner_recomputed_acceptance(
    result: monolithic.MonolithicContinuationResult,
) -> None:
    """Assert the accepted canonical equations after owner-backed replay.

    ``psi_after`` and the crack-region/static mechanics quantities are not
    archive members.  They are authoritative fields of the fresh owner replay;
    this assertion keeps the accepted history update and canonical residual/
    gate pairing explicit rather than treating the accepted state arrays as a
    standalone certificate.
    """

    for index, step in enumerate(result.accepted_steps):
        phase2b = step.phase2b
        auxiliary = phase2b.auxiliary
        state = phase2b.state
        if auxiliary is None or state is None:
            raise ValueError(f"owner replay accepted step {index} lacks its canonical audit")
        if auxiliary.psi_after is None or auxiliary.H_next is None:
            raise ValueError(f"owner replay accepted step {index} lacks psi_after/H_next")
        expected_history = np.maximum(
            step.committed_state.material_history,
            auxiliary.psi_after,
        )
        if not np.array_equal(auxiliary.H_next, expected_history):
            raise ValueError(
                f"owner replay accepted step {index} history is not recomputed from psi_after"
            )
        if not np.array_equal(state.material_history, auxiliary.H_next):
            raise ValueError(
                f"owner replay accepted step {index} history is not state-paired"
            )
        if phase2b.canonical_z is None or phase2b.canonical_scales is None:
            raise ValueError(f"owner replay accepted step {index} lacks canonical residual data")
        residual_owner = monolithic.MonolithicGripResidual(
            result.model,
            step.load_target,
            step.committed_state.material_history,
        )
        expected_scaled = residual_owner.scaled_residual(
            phase2b.canonical_z,
            phase2b.canonical_scales,
        )
        if phase2b.canonical_scaled_residual is None or not np.array_equal(
            phase2b.canonical_scaled_residual,
            expected_scaled,
        ):
            raise ValueError(
                f"owner replay accepted step {index} canonical equation was not recomputed"
            )
        expected_active = residual_owner.active_mask(phase2b.canonical_z)
        if phase2b.canonical_active_mask is None or not np.array_equal(
            phase2b.canonical_active_mask,
            expected_active,
        ):
            raise ValueError(
                f"owner replay accepted step {index} canonical active set was not recomputed"
            )
        gates = dict(phase2b.canonical_gate_passes)
        if gates.get("coupled") != all(
            value for name, value in gates.items() if name != "coupled"
        ):
            raise ValueError(
                f"owner replay accepted step {index} canonical gates are not recomputed"
            )
        for mechanics_name in ("mechanics_before", "mechanics_after"):
            mechanics = getattr(auxiliary, mechanics_name)
            if mechanics is None or mechanics.static_digest != result.model.static_digest:
                raise ValueError(
                    f"owner replay accepted step {index} lacks digest-matched {mechanics_name}"
                )


def _compare_replayed_result(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    result: monolithic.MonolithicContinuationResult,
    state_digests: tuple[str, ...],
    attempt_digests: tuple[str, ...],
) -> None:
    expected_arrays = _expected_arrays_from_result(result)
    _compare_arrays(arrays, expected_arrays)
    expected_state_digests = tuple(state.digest for state in result.states)
    if expected_state_digests != state_digests:
        raise ValueError("archive state digests differ from authoritative replay")
    expected_attempt_records = tuple(
        _attempt_scalar_evidence_payload(attempt) for attempt in result.attempts
    )
    archived_attempt_records = tuple(metadata["attempt_scalar_evidence"])
    if _canonical_json(expected_attempt_records) != _canonical_json(archived_attempt_records):
        raise ValueError("ordered attempt scalar provenance differs from authoritative replay")
    if tuple(
        record["scalar_evidence_digest"] for record in expected_attempt_records
    ) != attempt_digests:
        raise ValueError("attempt digests differ from authoritative replay")
    if metadata["completion_status"] != result.completion_status:
        raise ValueError("archive completion status differs from authoritative replay")
    archived_failure = metadata["failure"]
    expected_failure = _failure_payload(result)
    if _canonical_json(archived_failure) != _canonical_json(expected_failure):
        raise ValueError("archive failure provenance differs from authoritative replay")
    expected_work = _work_check_payload(result.work_check)
    if _canonical_json(metadata["work_check"]) != _canonical_json(expected_work):
        raise ValueError("archive work-check evidence differs from authoritative replay")


@dataclass(frozen=True, slots=True)
class MonolithicArchiveReplay:
    """Frozen result and immutable arrays returned after owner-authenticated replay."""

    path: Path
    result: monolithic.MonolithicContinuationResult
    metadata_json: bytes
    state_u_bar: np.ndarray
    state_union_displacement: np.ndarray
    state_material_damage: np.ndarray
    state_material_history: np.ndarray
    accepted_reaction: np.ndarray
    accepted_mechanics_energy: np.ndarray
    state_digests: tuple[str, ...]
    attempt_digests: tuple[str, ...]
    result_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("archive replay path must be Path")
        if not isinstance(self.result, monolithic.MonolithicContinuationResult):
            raise TypeError("archive replay result has the wrong type")
        object.__setattr__(self, "metadata_json", bytes(self.metadata_json))
        for name in _ARRAY_KEYS:
            object.__setattr__(
                self,
                name,
                _immutable(getattr(self, name), dtype=np.float64),
            )
        object.__setattr__(self, "state_digests", tuple(self.state_digests))
        object.__setattr__(self, "attempt_digests", tuple(self.attempt_digests))
        object.__setattr__(
            self,
            "result_digest",
            _valid_digest(self.result_digest, "result_digest"),
        )

    @property
    def schema(self) -> str:
        return MONOLITHIC_ARCHIVE_SCHEMA

    @property
    def completion_status(self) -> str:
        return self.result.completion_status

    @property
    def final_state(self) -> EngineeringGripState:
        return self.result.final_state


def replay_monolithic_continuation_archive(
    path: Any,
    model: EngineeringGripModel,
    archive_budget_bytes: int,
) -> MonolithicArchiveReplay:
    """Load, validate, and owner-authenticate a monolithic archive.

    The supplied ``model`` is the sole owner identity.  The reconstructed
    continuation is always rerun with the authoritative CPU reference adapter
    ``None``; a serialized adapter/runtime object is never trusted.
    """

    archive_path = _path(path, "archive path")
    budget = _require_budget(archive_budget_bytes)
    if not isinstance(model, EngineeringGripModel):
        raise TypeError("model must be EngineeringGripModel")
    arrays, metadata_json = _load_npz_arrays(archive_path, budget)
    metadata, canonical_metadata = _decode_metadata(arrays["metadata_json"])
    if metadata_json != canonical_metadata:
        raise ValueError("metadata bytes changed during archive load")
    continuation_config, state_digests, attempt_digests = _validate_top_metadata(
        metadata, model, arrays
    )
    if _archive_digest(metadata, {name: arrays[name] for name in _ARRAY_KEYS}) != metadata["result_digest"]:
        raise ValueError("archive result digest does not match its immutable members")

    try:
        fresh = _run_archive_authoritative_continuation(
            model, continuation_config, None
        )
    except monolithic.MonolithicContinuationFailure as error:
        fresh = error.result
    if not isinstance(fresh, monolithic.MonolithicContinuationResult):
        raise TypeError("authoritative continuation replay returned the wrong type")
    _validate_archive_owner(fresh, model)
    _assert_owner_recomputed_acceptance(fresh)
    _compare_replayed_result(
        metadata,
        arrays,
        fresh,
        state_digests,
        attempt_digests,
    )
    return MonolithicArchiveReplay(
        path=archive_path,
        result=fresh,
        metadata_json=metadata_json,
        state_u_bar=arrays["state_u_bar"],
        state_union_displacement=arrays["state_union_displacement"],
        state_material_damage=arrays["state_material_damage"],
        state_material_history=arrays["state_material_history"],
        accepted_reaction=arrays["accepted_reaction"],
        accepted_mechanics_energy=arrays["accepted_mechanics_energy"],
        state_digests=state_digests,
        attempt_digests=attempt_digests,
        result_digest=metadata["result_digest"],
    )


# These names are intentionally aliases, not separate read paths.
load_monolithic_continuation_archive = replay_monolithic_continuation_archive
read_monolithic_continuation_archive = replay_monolithic_continuation_archive
