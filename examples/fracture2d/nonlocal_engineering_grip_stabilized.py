"""Experimental convergence-first engineering-grip continuation.

This module is deliberately separate from :mod:`nonlocal_engineering_grip`.
It keeps the approved mechanics and material-only AT2/Amor equations unchanged
and changes only the numerical fixed-point iteration:

``d_trial = d_iter + omega * (d_solved - d_iter)``.

The unrelaxed ``d_solved`` is still obtained from the original fixed-history
damage equation.  The ``sufficient_decrease`` policy uses the existing
dimensionless merit backtracking condition.  The explicit ``fixed`` policy
evaluates exactly one configured damped-Picard candidate per sweep and records
merit without requiring a merit decrease.  The explicit ``anderson`` policy
uses the standard damped Type-II residual-difference update with a fixed
history depth and beta, while retaining the exact ``omega=0.5`` Picard
candidate as its only fallback.  Strict mode commits a load step
only when every coupled convergence gate passes.  The separate
``stabilized_warning`` mode may commit only a complete final-budget candidate
whose finite, mechanics, conservation, history-order, damage-equation, and
globalization/selection gates pass; the declared convergence/range/healing
defects remain raw warning provenance.  Non-finite, structural, mechanics,
damage-equation, globalization, and work check failures remain hard failures.

The module owns a distinct numerical-only archive and an independent replay
path.  Existing strict, bounded-warning, and unbounded-warning engineering-grip
controllers and schemas are not modified here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from scipy.linalg import qr as _pivoted_qr
from scipy.linalg import solve_triangular as _solve_triangular

from . import nonlocal_engineering_grip as engineering
from .nonlocal_engineering_grip import (
    A2MechanicsSolveResult,
    EngineeringGripDiagnostics,
    EngineeringGripModel,
    EngineeringGripState,
    EngineeringGripWorkCheck,
    JaxA2CPUReplayAdapter,
)
from .nonlocal_baseline import FixedHistoryDamageResult, GateTolerance


__all__ = [
    "STABILIZED_ACCEPTANCE_MODE",
    "STABILIZED_ACCEPTANCE_STATUS",
    "STABILIZED_ACCEPTANCE_MODES",
    "STABILIZED_ARCHIVE_SCHEMA",
    "STABILIZED_ARCHIVE_LAYOUT",
    "STABILIZED_COMPLETION_STATUS",
    "STABILIZED_FAILURE_ARCHIVE_SCHEMA",
    "STABILIZED_FAILURE_STATUS",
    "STABILIZED_WARNING_FAILURE_CODE",
    "STABILIZED_WARNING_ACCEPTANCE_MODE",
    "STABILIZED_WARNING_ACCEPTANCE_STATUS",
    "STABILIZED_WARNING_ARCHIVE_LAYOUT",
    "STABILIZED_WARNING_ARCHIVE_SCHEMA",
    "STABILIZED_WARNING_COMPLETION_STATUS",
    "STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA",
    "STABILIZED_WARNING_FAILURE_STATUS",
    "STABILIZED_WARNING_GATE_NAMES",
    "STABILIZED_WARNING_HARD_GATE_NAMES",
    "STABILIZED_FAILURE_CODE",
    "STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE",
    "STABILIZED_SCHEDULE_FAILURE_CODE",
    "STABILIZED_REPLAY_CONTRACT",
    "STABILIZED_FILESYSTEM_RESERVE_BYTES",
    "STABILIZED_COUPLED_GATE_NAMES",
    "STABILIZED_ANDERSON_POLICIES",
    "STABILIZED_ANDERSON_RESIDUAL_POLICIES",
    "STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_POLICY",
    "STABILIZED_ANDERSON_REGULARIZED_RESIDUAL_BACKTRACKING_POLICY",
    "STABILIZED_RELAXATION_POLICIES",
    "STABILIZED_ANDERSON_HISTORY_DEPTH",
    "STABILIZED_ANDERSON_BETA",
    "STABILIZED_ANDERSON_ALPHA_REDUCTION",
    "STABILIZED_ANDERSON_MAX_BACKTRACKS",
    "STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY",
    "STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY",
    "STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY",
    "STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY",
    "STABILIZED_ANDERSON_REGULARIZATION_STUDY_VALUES",
    "STABILIZED_SCHEDULE_ARTIFACT_SCHEMA",
    "STABILIZED_SCHEDULE_ARCHIVE_SCHEMA",
    "STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA",
    "STABILIZED_SCHEDULE_REPLAY_CONTRACT",
    "STABILIZED_SCHEDULE_ARCHIVE_LAYOUT",
    "STABILIZED_SCHEDULE_TARGET_SHA256",
    "STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256",
    "ANDERSON_RESIDUAL_HARD_GATE_NAMES",
    "ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA",
    "StabilizedLoadSchedule",
    "load_stabilized_load_schedule",
    "extract_stabilized_load_schedule",
    "ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA",
    "AndersonScalarDiagnosticWriter",
    "load_anderson_scalar_diagnostic",
    "summarize_anderson_scalar_diagnostic",
    "StabilizedEngineeringGripAcceptedStep",
    "StabilizedEngineeringGripAttempt",
    "StabilizedEngineeringGripArchiveBudgetFailure",
    "StabilizedEngineeringGripConfig",
    "StabilizedEngineeringGripContinuationFailure",
    "StabilizedEngineeringGripResult",
    "StabilizedEngineeringGripSweepFailure",
    "StabilizedRelaxationSweepDiagnostics",
    "StabilizedRelaxationTrial",
    "load_stabilized_engineering_grip_archive",
    "estimate_stabilized_archive_bytes",
    "estimate_stabilized_archive_bytes_for_counts",
    "preflight_stabilized_archive_storage",
    "replay_stabilized_engineering_grip_archive",
    "load_stabilized_source_cloud",
    "run_stabilized_engineering_grip",
    "write_stabilized_engineering_grip_archive",
    "write_stabilized_engineering_grip_failure_archive",
]


STABILIZED_ACCEPTANCE_MODE = "stabilized"
STABILIZED_ACCEPTANCE_STATUS = "stabilized_converged"
STABILIZED_COMPLETION_STATUS = "stabilized_completed"
STABILIZED_FAILURE_STATUS = "stabilized_continuation_failure"
STABILIZED_FAILURE_CODE = "stabilized_continuation_failure"
STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE = "archive_budget_exhausted"
STABILIZED_SCHEDULE_FAILURE_CODE = "stabilized_schedule_target_failure"
STABILIZED_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-stabilized-v5"
STABILIZED_FAILURE_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-failure-v5"
)
STABILIZED_REPLAY_CONTRACT = "fracture2d-engineering-grip-stabilized-replay-v5"
STABILIZED_ARCHIVE_LAYOUT = "accepted-state-plus-final-solver-summary-v5"
STABILIZED_RELAXATION_POLICIES = (
    "sufficient_decrease",
    "fixed",
    "anderson",
    "anderson_residual",
    "anderson_residual_backtracking",
    "anderson_residual_regularized_backtracking",
)
STABILIZED_ANDERSON_POLICIES = (
    "anderson",
    "anderson_residual",
    "anderson_residual_backtracking",
    "anderson_residual_regularized_backtracking",
)
STABILIZED_ANDERSON_RESIDUAL_POLICIES = (
    "anderson_residual",
    "anderson_residual_backtracking",
    "anderson_residual_regularized_backtracking",
)
STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_POLICY = (
    "anderson_residual_backtracking"
)
STABILIZED_ANDERSON_REGULARIZED_RESIDUAL_BACKTRACKING_POLICY = (
    "anderson_residual_regularized_backtracking"
)
STABILIZED_ANDERSON_HISTORY_DEPTH = 5
STABILIZED_ANDERSON_BETA = 0.5
STABILIZED_ANDERSON_ALPHA_REDUCTION = 0.5
STABILIZED_ANDERSON_MAX_BACKTRACKS = 4
STABILIZED_ANDERSON_REGULARIZATION_STUDY_VALUES = (0.25, 1.0, 2.0)
STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY = "legacy_anderson_composite_v1"
STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY = (
    "committed_history_map_residual_v1"
)
STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY = (
    "committed_history_map_residual_backtracking_v1"
)
STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY = (
    "committed_history_map_residual_regularized_backtracking_v1"
)
STABILIZED_SCHEDULE_ARTIFACT_SCHEMA = (
    "fracture2d-engineering-grip-load-schedule-v1"
)
STABILIZED_SCHEDULE_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-schedule-v1"
)
STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-schedule-failure-v1"
)
STABILIZED_SCHEDULE_REPLAY_CONTRACT = (
    "fracture2d-engineering-grip-stabilized-schedule-replay-v1"
)
STABILIZED_SCHEDULE_ARCHIVE_LAYOUT = (
    "accepted-state-plus-final-solver-summary-schedule-v1"
)
STABILIZED_SCHEDULE_TARGET_SHA256 = (
    "13d294d9f3817bfbf0da5eec1622f278cc6217f0e4c528275fbdbfe1fbb10cb5"
)
STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256 = (
    "4c9a6f7e68f26afe01ba3fcf15a44a7b588174671128001dab440dc6b08fcbdc"
)
STABILIZED_SCHEDULE_TARGET_COUNT = 66
STABILIZED_SCHEDULE_SOURCE_STATE_FIELD = "state_u_bar"
STABILIZED_SCHEDULE_TARGET_DTYPE = "<f8"
STABILIZED_SCHEDULE_ARCHIVE_FIELDS = {
    "schedule_mode",
    "schedule_path",
    "schedule_file_sha256",
    "schedule_target_sha256",
    "schedule_target_count",
    "schedule_load_start",
    "schedule_load_final",
    "schedule_endpoint",
    "schedule_source_path",
    "schedule_source_sha256",
    "schedule_source_state_count",
}
_ANDERSON_SELECTIONS = ("accelerated", "picard_fallback")

# Warning acceptance is a separate experimental state machine.  Its distinct
# identities are intentional: a warning-accepted state is finite and auditable,
# but it is not a strict stabilized convergence result and must not be loaded as
# one by a historical v5 consumer.
STABILIZED_WARNING_ACCEPTANCE_MODE = "stabilized_warning"
STABILIZED_WARNING_ACCEPTANCE_STATUS = "stabilized_warning_accepted"
STABILIZED_WARNING_COMPLETION_STATUS = "stabilized_warning_completed"
STABILIZED_WARNING_FAILURE_STATUS = "stabilized_warning_continuation_failure"
STABILIZED_WARNING_FAILURE_CODE = "stabilized_warning_continuation_failure"
STABILIZED_WARNING_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-warning-v1"
)
STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-warning-failure-v1"
)
STABILIZED_WARNING_REPLAY_CONTRACT = (
    "fracture2d-engineering-grip-stabilized-warning-replay-v1"
)
STABILIZED_WARNING_ARCHIVE_LAYOUT = (
    "accepted-state-plus-final-solver-summary-stabilized-warning-v1"
)
STABILIZED_ACCEPTANCE_MODES = (
    STABILIZED_ACCEPTANCE_MODE,
    STABILIZED_WARNING_ACCEPTANCE_MODE,
)

# The archive is deliberately bounded independently of NPZ compression.  The
# reserve is a named filesystem policy rather than an implicit disk-space
# margin; the workflow reports it alongside the count/byte estimate before any
# GPU continuation work starts.
STABILIZED_FILESYSTEM_RESERVE_BYTES = 2 * 1024**3

_ARCHIVE_COUNT_NAMES = ("accepted_steps",)
_ARCHIVE_BUDGET_STAGE_KINDS = ("accepted_step",)
_ARCHIVE_BUDGET_EXHAUSTION_KINDS = ("archive_budget", "filesystem")

# A replayed CG solve is mathematically equivalent to the production solve but
# need not reproduce every last bit of its floating-point path.  Trial q-after
# digests therefore fingerprint q after a declared decimal quantisation.  The
# committed state digest remains the exact byte-level digest used for rollback.
# Independent replay uses a different CG operation order from the production
# adapter.  Eleven decimal places retain a useful tamper fingerprint while
# absorbing the observed sub-picometer replay roundoff in q-after.
_TRIAL_Q_DIGEST_DECIMALS = 11

# These are the existing engineering-grip gates that a stabilized candidate
# must satisfy.  The warning path protects finite/mechanics/conservation,
# history ordering, the damage equation, and selection of one complete
# globalization candidate.  The remaining coupled diagnostics are explicitly
# bypassable warnings; their raw values and failed-gate provenance are retained.
STABILIZED_COUPLED_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
    "displacement",
    "damage",
    "history",
    "reaction",
    "damage_residual",
    "history_monotonic",
    "force",
    "moment",
    "damage_fixed_point",
    "damage_equation",
    "globalization",
)

STABILIZED_WARNING_HARD_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
    "history_monotonic",
    "force",
    "moment",
    "damage_equation",
    "globalization",
)
STABILIZED_WARNING_GATE_NAMES = (
    "displacement",
    "damage",
    "history",
    "reaction",
    "damage_residual",
    "damage_fixed_point",
    "damage_range",
    "healing",
    "coupled",
)

# Warning archives add only the provenance needed to distinguish strict and
# warning-accepted physical rows.  Keep these fields separate from the strict
# v5 field set so a strict consumer can never reinterpret a warning row.
_WARNING_ARCHIVE_EXTRA_FIELDS = {
    "warning_gate_names",
    "state_bypassed_gate_mask",
    "accepted_warning_present",
    "accepted_warning_range_excess",
    "accepted_warning_healing_magnitude",
}

_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
_BOOL = np.dtype(bool)
_EPS64 = np.finfo(np.float64).eps
_RELATIVE_FLOOR = np.finfo(np.float64).tiny
_STABILIZED_SOURCE_BUNDLE_FILES = engineering._SOURCE_BUNDLE_FILES + (
    "examples/fracture2d/nonlocal_engineering_grip_stabilized.py",
)
ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA = (
    "fracture2d-engineering-grip-anderson-scalar-v1"
)
ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA = (
    "fracture2d-engineering-grip-anderson-residual-hard-gates-v3"
)
# This is the only order in which residual-policy hard-gate failures are
# serialized or counted.  The two damage-range bounds are deliberately split;
# the production diagnostic keeps their common aggregate pass flag, but its raw
# extrema already distinguish the lower and upper checks without another
# evaluation.
ANDERSON_RESIDUAL_HARD_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
    "force",
    "moment",
    "damage_range_lower",
    "damage_range_upper",
    "healing",
    "history_ordering",
)
_ANDERSON_HARD_GATE_DETAIL_FIELDS = (
    "available",
    "passed",
    "raw",
    "scale",
    "limit",
    "normalized",
    "ratio_available",
    "metric_reason",
    "stage_reason",
    "cg_info",
    "cg_status_pass",
    "residual_limit_pass",
    "stage_metrics",
)
_ANDERSON_HARD_GATE_STAGE_NAMES = ("before", "after")
_ANDERSON_HARD_GATE_STATUSES = (
    "evaluated",
    "not_evaluated",
    "candidate_map_failure",
)
_ANDERSON_HARD_GATE_STAGE_METRIC_FIELDS = (
    "error",
    "scale",
    "limit",
    "normalized",
    "ratio_available",
    "passed",
)
_ANDERSON_HARD_GATE_METRIC_REASONS = {
    "finite": "boolean_gate",
    "mechanics_before": "cg_info_and_free_residual_limit",
    "mechanics_after": "cg_info_and_free_residual_limit",
    "force": "stage_worst_ratio_and_conjunction",
    "moment": "stage_worst_ratio_and_conjunction",
    "damage_range_lower": "damage_range_excess_and_limit",
    "damage_range_upper": "damage_range_excess_and_limit",
    "healing": "healing_excess_and_limit",
    "history_ordering": "history_ordering_excess_and_zero_limit",
}
_ANDERSON_HARD_GATE_UNAVAILABLE_REASONS = {
    "candidate_not_fully_evaluated",
    "mechanics_before_not_evaluated",
    "mechanics_after_not_evaluated",
    "damage_not_evaluated",
    "history_not_evaluated",
}


def _validate_stabilized_acceptance_mode(value: Any) -> str:
    if not isinstance(value, str) or value not in STABILIZED_ACCEPTANCE_MODES:
        raise ValueError(
            "stabilized acceptance mode must be one of "
            f"{STABILIZED_ACCEPTANCE_MODES}, got {value!r}"
        )
    return value


def _is_anderson_policy(policy: str) -> bool:
    return policy in STABILIZED_ANDERSON_POLICIES


def _is_anderson_residual_backtracking_policy(policy: str) -> bool:
    return policy in {
        STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_POLICY,
        STABILIZED_ANDERSON_REGULARIZED_RESIDUAL_BACKTRACKING_POLICY,
    }


def _is_anderson_regularized_policy(policy: str) -> bool:
    return policy == STABILIZED_ANDERSON_REGULARIZED_RESIDUAL_BACKTRACKING_POLICY


def _is_anderson_residual_policy(policy: str) -> bool:
    return policy in STABILIZED_ANDERSON_RESIDUAL_POLICIES


def _anderson_backtracking_alpha_for_index(index: int, name: str) -> float:
    """Return the canonical alpha bound to one new-policy trial index."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"{name} must be a JSON integer")
    if index < 0 or index > STABILIZED_ANDERSON_MAX_BACKTRACKS:
        raise ValueError(f"{name} is outside the configured alpha schedule")
    alpha = float(STABILIZED_ANDERSON_ALPHA_REDUCTION**index)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"{name} produces an invalid alpha")
    return alpha


def _anderson_merit_identity(policy: str) -> str:
    if policy == "anderson":
        return STABILIZED_ANDERSON_LEGACY_MERIT_IDENTITY
    if policy == "anderson_residual":
        return STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY
    if _is_anderson_residual_backtracking_policy(policy):
        if _is_anderson_regularized_policy(policy):
            return STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY
        return STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY
    raise ValueError(f"{policy!r} is not an Anderson relaxation policy")


def _stabilized_contract(
    mode: str, *, failure: bool, schedule: bool = False
) -> dict[str, Any]:
    """Return the immutable archive identity for one stabilized mode."""

    validated_mode = _validate_stabilized_acceptance_mode(mode)
    if schedule:
        if validated_mode != STABILIZED_ACCEPTANCE_MODE:
            raise ValueError(
                "explicit load schedules are valid only for strict stabilized mode"
            )
        return {
            "acceptance_mode": STABILIZED_ACCEPTANCE_MODE,
            "acceptance_status": STABILIZED_ACCEPTANCE_STATUS,
            "completion_status": (
                STABILIZED_FAILURE_STATUS
                if failure
                else STABILIZED_COMPLETION_STATUS
            ),
            "archive_schema": (
                STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA
                if failure
                else STABILIZED_SCHEDULE_ARCHIVE_SCHEMA
            ),
            "archive_layout": STABILIZED_SCHEDULE_ARCHIVE_LAYOUT,
            "replay_contract": STABILIZED_SCHEDULE_REPLAY_CONTRACT,
            "failure_code": (
                STABILIZED_SCHEDULE_FAILURE_CODE
                if failure
                else None
            ),
        }
    if validated_mode == STABILIZED_ACCEPTANCE_MODE:
        return {
            "acceptance_mode": STABILIZED_ACCEPTANCE_MODE,
            "acceptance_status": STABILIZED_ACCEPTANCE_STATUS,
            "completion_status": (
                STABILIZED_FAILURE_STATUS
                if failure
                else STABILIZED_COMPLETION_STATUS
            ),
            "archive_schema": (
                STABILIZED_FAILURE_ARCHIVE_SCHEMA
                if failure
                else STABILIZED_ARCHIVE_SCHEMA
            ),
            "archive_layout": STABILIZED_ARCHIVE_LAYOUT,
            "replay_contract": STABILIZED_REPLAY_CONTRACT,
            "failure_code": (
                STABILIZED_FAILURE_CODE
                if failure
                else None
            ),
        }
    return {
        "acceptance_mode": STABILIZED_WARNING_ACCEPTANCE_MODE,
        "acceptance_status": STABILIZED_WARNING_ACCEPTANCE_STATUS,
        "completion_status": (
            STABILIZED_WARNING_FAILURE_STATUS
            if failure
            else STABILIZED_WARNING_COMPLETION_STATUS
        ),
        "archive_schema": (
            STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA
            if failure
            else STABILIZED_WARNING_ARCHIVE_SCHEMA
        ),
        "archive_layout": STABILIZED_WARNING_ARCHIVE_LAYOUT,
        "replay_contract": STABILIZED_WARNING_REPLAY_CONTRACT,
    }


def _stabilized_failure_code(mode: str, *, schedule: bool = False) -> str:
    """Return the failure identity paired with one stabilized mode."""

    validated_mode = _validate_stabilized_acceptance_mode(mode)
    if schedule:
        if validated_mode != STABILIZED_ACCEPTANCE_MODE:
            raise ValueError("scheduled failure code requires strict acceptance")
        return STABILIZED_SCHEDULE_FAILURE_CODE
    return (
        STABILIZED_WARNING_FAILURE_CODE
        if validated_mode == STABILIZED_WARNING_ACCEPTANCE_MODE
        else STABILIZED_FAILURE_CODE
    )


def _immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    return engineering._immutable(values, dtype)


def _f64(
    value: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
    *,
    nonfinite_error: type[Exception] = ValueError,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise nonfinite_error(f"{name} must contain only finite values")
    return array


def _i64(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _INT64:
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _bool(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _BOOL:
        raise ValueError(f"{name} must use exact boolean dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _scalar(value: Any, name: str) -> float:
    return engineering._scalar(value, name)


def _optional_scalar(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _scalar(value, name)


def _optional_nonnegative_integer(value: Any | None, name: str) -> int | None:
    if value is None:
        return None
    result = engineering._integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _optional_f64(
    value: Any | None, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray | None:
    if value is None:
        return None
    return _immutable(_f64(value, name, shape))


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a hexadecimal SHA256 digest") from error
    return value


def _current_stabilized_source_bundle_hash() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    return engineering._source_bundle_hash(
        tuple(
            (relative_name, repository_root / relative_name)
            for relative_name in _STABILIZED_SOURCE_BUNDLE_FILES
        )
    )


def _archive_sha256(path: Any) -> str:
    """Return the SHA256 of the exact archive bytes replayed from disk."""

    filename = Path(os.fspath(path))
    digest = hashlib.sha256()
    try:
        with filename.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise OSError(f"cannot hash stabilized archive {filename}") from error
    return digest.hexdigest()


def _array_close(
    actual: Any,
    expected: Any,
    name: str,
    relative_limit: float = 2.0e-9,
    absolute_limit: float | None = None,
) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"{name} shape/dtype does not match replay")
    defect = float(np.max(np.abs(left - right))) if left.size else 0.0
    scale = max(
        float(np.max(np.abs(left))) if left.size else 0.0,
        float(np.max(np.abs(right))) if right.size else 0.0,
        1.0,
    )
    limit = relative_limit * scale
    if absolute_limit is not None:
        limit = max(limit, float(absolute_limit))
    if not np.isfinite(defect) or defect > limit:
        raise ValueError(f"{name} does not match replay: defect={defect:.17g}")


def _compare_scalar(
    actual: Any,
    expected: Any,
    name: str,
    relative_limit: float = 2.0e-8,
    absolute_limit: float | None = None,
) -> None:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError(f"{name} availability does not match replay")
        return
    left = _scalar(actual, f"{name} archive")
    right = _scalar(expected, f"{name} replay")
    limit = relative_limit * max(abs(left), abs(right), 1.0)
    if absolute_limit is not None:
        limit = max(limit, float(absolute_limit))
    if abs(left - right) > limit:
        raise ValueError(f"{name} does not match replay: defect={abs(left-right):.17g}")


def _diagnostic_from_payload(
    payload: Any,
    name: str,
    *,
    array_values: Mapping[str, np.ndarray] | None = None,
) -> EngineeringGripDiagnostics:
    """Rebuild one ordinary diagnostic from compact canonical JSON.

    The compact diagnostic JSON carries only scalar/boolean provenance.  The
    optional arrays are an in-memory/dataclass reconstruction seam; the public
    v5 archive does not serialize them and supplies no trial-array rows.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a diagnostic mapping")
    diagnostic_fields = {
        field.name for field in dataclass_fields(EngineeringGripDiagnostics)
    }
    required = diagnostic_fields | {"gate_passes"}
    missing = sorted(required - set(payload))
    unexpected = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"{name} is missing diagnostic fields: {missing}")
    if unexpected:
        raise ValueError(f"{name} contains unknown diagnostic fields: {unexpected}")
    for field_name in (
        "complete_sweep",
        "passed",
        *engineering._DIAGNOSTIC_BOOL_FIELDS,
    ):
        if type(payload[field_name]) is not bool:
            raise ValueError(f"{name}.{field_name} must be a JSON boolean")
    available = payload["available_fields"]
    if not isinstance(available, list) or not all(
        isinstance(item, str) for item in available
    ):
        raise ValueError(f"{name}.available_fields is invalid")
    gates = payload["gate_passes"]
    expected_gates = set(engineering._WARNING_HARD_GATE_NAMES) | set(
        engineering.WARNING_GATE_NAMES
    )
    if not isinstance(gates, Mapping) or set(gates) != expected_gates or any(
        type(item) is not bool for item in gates.values()
    ):
        raise ValueError(f"{name}.gate_passes is invalid")
    diagnostic_payload = {
        field_name: payload[field_name] for field_name in diagnostic_fields
    }
    for field_name in engineering._DIAGNOSTIC_ARRAY_FIELDS:
        if diagnostic_payload[field_name] is not None:
            raise ValueError(
                f"{name}.{field_name} must be null in compact stabilized JSON"
            )
        if array_values is not None and field_name in array_values:
            diagnostic_payload[field_name] = array_values[field_name]
    if diagnostic_payload["damage_before"] is None or diagnostic_payload["d_iter"] is None:
        raise ValueError(f"{name} is missing typed damage provenance arrays")
    diagnostic = EngineeringGripDiagnostics(**diagnostic_payload)
    if list(diagnostic.available_fields) != available:
        raise ValueError(f"{name}.available_fields are not canonical")
    if diagnostic.gate_passes != dict(gates):
        raise ValueError(f"{name}.gate_passes are forged")
    return diagnostic


def _compact_diagnostic_payload(
    diagnostic: EngineeringGripDiagnostics,
) -> dict[str, Any]:
    """Serialize diagnostic scalars/flags without duplicating numerical arrays.

    Runtime sweep diagnostics retain fixed-shape numerical primitives for the
    controller.  Complete relaxation candidates are represented in the archive
    by scalar/Boolean audit fields and explicit ``null`` array markers only.
    """

    payload = engineering._diagnostic_payload(diagnostic)
    for name in engineering._DIAGNOSTIC_ARRAY_FIELDS:
        payload[name] = None
    return payload


def _trial_q_after_digest(
    target: float,
    q_after: np.ndarray,
    d_trial: np.ndarray,
    H_next: np.ndarray,
) -> str:
    """Return the replay-stable fingerprint for one complete trial state."""

    q = _f64(q_after, "trial q-after digest displacement")
    decimals = _TRIAL_Q_DIGEST_DECIMALS
    quantized = np.round(q, decimals=decimals)
    damage = np.round(_f64(d_trial, "trial q-after digest damage"), decimals=decimals)
    history = np.round(_f64(H_next, "trial q-after digest history"), decimals=decimals)
    return engineering._state_digest(target, quantized, damage, history)


def _schedule_target_sha256(targets: Any) -> str:
    """Hash the canonical contiguous little-endian schedule bytes only."""

    array = np.asarray(targets)
    if array.dtype.str != STABILIZED_SCHEDULE_TARGET_DTYPE:
        raise ValueError(
            "load schedule targets must use exact little-endian float64 dtype"
        )
    if array.ndim != 1:
        raise ValueError("load schedule targets must be one-dimensional")
    contiguous = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _schedule_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _schedule_json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON mapping")
    try:
        text = _schedule_json_text(value)
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON-compatible") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON mapping")
    return parsed


def _schedule_source_state(
    path: Path,
    expected_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the historical archive state sequence without pickle."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"matched-load schedule source archive does not exist: {resolved}"
        )
    actual_sha256 = _archive_sha256(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "matched-load schedule source archive SHA256 does not match metadata"
        )
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            values = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            f"cannot load matched-load schedule source archive {resolved}: {error}"
        ) from error
    if any(value.dtype.kind == "O" for value in values.values()):
        raise ValueError("matched-load schedule source archive contains object arrays")
    if STABILIZED_SCHEDULE_SOURCE_STATE_FIELD not in values:
        raise ValueError(
            "matched-load schedule source archive is missing state_u_bar"
        )
    state_u = np.asarray(values[STABILIZED_SCHEDULE_SOURCE_STATE_FIELD])
    if state_u.dtype.str != STABILIZED_SCHEDULE_TARGET_DTYPE:
        raise ValueError(
            "matched-load schedule source state_u_bar must use exact little-endian "
            "float64 dtype"
        )
    if state_u.ndim != 1 or len(state_u) < 2:
        raise ValueError(
            "matched-load schedule source state_u_bar must be a one-dimensional "
            "sequence containing exact zero and at least one target"
        )
    if not np.all(np.isfinite(state_u)) or state_u[0] != 0.0:
        raise ValueError(
            "matched-load schedule source state_u_bar must be finite and begin at zero"
        )
    if np.any(np.diff(state_u) <= 0.0):
        raise ValueError(
            "matched-load schedule source state_u_bar must be strictly increasing"
        )
    if "archive_schema" not in values:
        raise ValueError(
            "matched-load schedule source must identify the completed historical "
            "stabilized archive"
        )
    schema = np.asarray(values["archive_schema"])
    if schema.shape != () or schema.dtype.kind not in "SU":
        raise ValueError("matched-load schedule source archive_schema is invalid")
    if str(schema.item()) != STABILIZED_ARCHIVE_SCHEMA:
        raise ValueError(
            "matched-load schedule source must be the completed historical "
            "stabilized archive"
        )
    if "metadata_json" not in values:
        raise ValueError(
            "matched-load schedule source must carry canonical historical metadata"
        )
    raw_metadata = np.asarray(values["metadata_json"])
    if raw_metadata.shape != () or raw_metadata.dtype.kind not in "SU":
        raise ValueError("matched-load schedule source metadata_json is invalid")
    try:
        parsed = json.loads(str(raw_metadata.item()))
    except json.JSONDecodeError as error:
        raise ValueError(
            "matched-load schedule source metadata_json is not valid JSON"
        ) from error
    if not isinstance(parsed, Mapping):
        raise ValueError(
            "matched-load schedule source metadata_json must be a mapping"
        )
    metadata = dict(parsed)
    if metadata.get("schema") != STABILIZED_ARCHIVE_SCHEMA:
        raise ValueError("matched-load schedule source metadata schema is invalid")
    if metadata.get("kind") != "completed":
        raise ValueError("matched-load schedule source is not a completed archive")
    if metadata.get("completion_status") != STABILIZED_COMPLETION_STATUS:
        raise ValueError("matched-load schedule source completion status is invalid")
    stable = metadata.get("stabilized_config")
    if not isinstance(stable, Mapping) or stable.get("relaxation_policy") != "anderson":
        raise ValueError(
            "matched-load schedule source must use historical anderson policy"
        )
    return np.asarray(state_u, dtype=np.dtype("<f8")), metadata


@dataclass(frozen=True)
class StabilizedLoadSchedule:
    """Validated explicit load targets and their immutable provenance binding."""

    path: str
    file_sha256: str
    targets: np.ndarray
    canonical_target_sha256: str
    source_archive_path: str
    source_archive_sha256: str
    target_count: int
    load_start: float
    load_final: float
    source_state_count: int
    source_state_field: str = STABILIZED_SCHEDULE_SOURCE_STATE_FIELD

    def __post_init__(self) -> None:
        path = Path(self.path)
        source_path = Path(self.source_archive_path)
        if not path.is_absolute() or str(path.resolve()) != self.path:
            raise ValueError("load schedule path must be a resolved absolute path")
        if not source_path.is_absolute() or str(source_path.resolve()) != self.source_archive_path:
            raise ValueError(
                "load schedule source archive path must be a resolved absolute path"
            )
        file_sha256 = _digest(self.file_sha256, "load schedule file_sha256")
        target_sha256 = _digest(
            self.canonical_target_sha256,
            "load schedule canonical_target_sha256",
        )
        source_sha256 = _digest(
            self.source_archive_sha256,
            "load schedule source_archive_sha256",
        )
        targets = np.asarray(self.targets)
        if targets.dtype.str != STABILIZED_SCHEDULE_TARGET_DTYPE:
            raise ValueError(
                "load schedule targets must use exact little-endian float64 dtype"
            )
        if targets.ndim != 1 or len(targets) == 0:
            raise ValueError("load schedule targets must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(targets)):
            raise ValueError("load schedule targets must be finite")
        load_start = _scalar(self.load_start, "load schedule load_start")
        load_final = _scalar(self.load_final, "load schedule load_final")
        if load_final <= load_start:
            raise ValueError("load schedule load_final must exceed load_start")
        if np.any(targets <= load_start) or np.any(np.diff(targets) <= 0.0):
            raise ValueError(
                "load schedule targets must be nonzero, above load_start, and strictly increasing"
            )
        if targets[-1] != load_final:
            raise ValueError(
                "load schedule final target must equal the explicit load_final exactly"
            )
        target_count = engineering._integer(self.target_count, "load schedule target_count", positive=True)
        if target_count != len(targets):
            raise ValueError("load schedule target_count does not match targets")
        source_state_count = engineering._integer(
            self.source_state_count,
            "load schedule source_state_count",
            positive=True,
        )
        if source_state_count != target_count + 1:
            raise ValueError(
                "load schedule source_state_count must equal target_count plus exact zero"
            )
        if self.source_state_field != STABILIZED_SCHEDULE_SOURCE_STATE_FIELD:
            raise ValueError("load schedule source_state_field is invalid")
        provenance = self._validate_provenance()
        object.__setattr__(self, "path", str(path))
        object.__setattr__(self, "file_sha256", provenance["file_sha256"])
        object.__setattr__(
            self,
            "targets",
            _immutable(np.ascontiguousarray(provenance["targets"], dtype=np.dtype("<f8"))),
        )
        object.__setattr__(
            self, "canonical_target_sha256", provenance["canonical_target_sha256"]
        )
        object.__setattr__(
            self, "source_archive_path", provenance["source_archive_path"]
        )
        object.__setattr__(
            self, "source_archive_sha256", provenance["source_archive_sha256"]
        )
        object.__setattr__(self, "target_count", provenance["target_count"])
        object.__setattr__(self, "load_start", provenance["load_start"])
        object.__setattr__(self, "load_final", provenance["load_final"])
        object.__setattr__(
            self, "source_state_count", provenance["source_state_count"]
        )

    def _validate_provenance(self) -> dict[str, Any]:
        """Reload and verify the immutable schedule/source binding once.

        This is intentionally a filesystem operation at the configuration
        boundary.  The controller calls it before numerical work starts, but it
        is never reachable from a stagger sweep or a JIT-compiled function.
        """

        path = Path(self.path).resolve()
        provenance = _read_stabilized_load_schedule_artifact(
            path,
            load_start=self.load_start,
            load_final=self.load_final,
        )
        expected = {
            "path": str(path),
            "file_sha256": _digest(self.file_sha256, "load schedule file_sha256"),
            "canonical_target_sha256": _digest(
                self.canonical_target_sha256,
                "load schedule canonical_target_sha256",
            ),
            "source_archive_path": str(Path(self.source_archive_path).resolve()),
            "source_archive_sha256": _digest(
                self.source_archive_sha256,
                "load schedule source_archive_sha256",
            ),
            "source_state_field": self.source_state_field,
            "target_count": engineering._integer(
                self.target_count, "load schedule target_count", positive=True
            ),
            "load_start": _scalar(self.load_start, "load schedule load_start"),
            "load_final": _scalar(self.load_final, "load schedule load_final"),
            "source_state_count": engineering._integer(
                self.source_state_count,
                "load schedule source_state_count",
                positive=True,
            ),
        }
        for name, value in expected.items():
            if name == "targets":
                continue
            if provenance[name] != value:
                raise ValueError(
                    f"load schedule {name} is not paired with the validated artifact"
                )
        if not np.array_equal(np.asarray(self.targets), provenance["targets"]):
            raise ValueError("load schedule targets are not paired with the artifact")
        return provenance

    def payload(self) -> dict[str, Any]:
        payload = {
            "mode": "explicit_matched_load_schedule_v1",
            "path": self.path,
            "file_sha256": self.file_sha256,
            "canonical_target_sha256": self.canonical_target_sha256,
            "source_archive_path": self.source_archive_path,
            "source_archive_sha256": self.source_archive_sha256,
            "source_state_field": self.source_state_field,
            "source_state_count": self.source_state_count,
            "target_count": self.target_count,
            "target_dtype": STABILIZED_SCHEDULE_TARGET_DTYPE,
            "target_shape": [self.target_count],
            "load_start": self.load_start,
            "load_final": self.load_final,
            "endpoint": self.load_final,
        }
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        load_start: float,
        load_final: float,
    ) -> "StabilizedLoadSchedule":
        if not isinstance(payload, Mapping):
            raise ValueError("load schedule configuration must be a mapping")
        if "path" not in payload:
            raise ValueError("load schedule configuration path is missing")
        path_value = payload["path"]
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("load schedule configuration path is missing")
        schedule = load_stabilized_load_schedule(
            Path(path_value), load_start=load_start, load_final=load_final
        )
        if dict(payload) != schedule.payload():
            raise ValueError("load schedule configuration payload is not canonical")
        return schedule


def _read_stabilized_load_schedule_artifact(
    path: Any,
    *,
    load_start: float,
    load_final: float,
) -> dict[str, Any]:
    """Read and validate the reviewed numeric schedule artifact.

    The helper returns typed provenance rather than constructing the public
    dataclass so its constructor can independently enforce the same on-disk
    contract without a recursion or caller-supplied validation token.
    """

    resolved = Path(os.fspath(path)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"load schedule does not exist: {resolved}")
    file_sha256 = _archive_sha256(resolved)
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            values = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot load load schedule NPZ {resolved}: {error}") from error
    if any(value.dtype.kind == "O" for value in values.values()):
        raise ValueError("load schedule NPZ contains forbidden object arrays")
    required = {"targets", "metadata_json"}
    if set(values) != required:
        raise ValueError(
            "load schedule NPZ fields differ from the explicit contract: "
            f"missing={sorted(required - set(values))}, extra={sorted(set(values) - required)}"
        )
    targets = np.asarray(values["targets"])
    if targets.dtype.str != STABILIZED_SCHEDULE_TARGET_DTYPE:
        raise ValueError("load schedule targets must use exact little-endian float64 dtype")
    if targets.ndim != 1 or len(targets) == 0:
        raise ValueError("load schedule targets must have nonempty one-dimensional shape")
    metadata_array = np.asarray(values["metadata_json"])
    if metadata_array.shape != () or metadata_array.dtype.kind not in "SU":
        raise ValueError("load schedule metadata_json must be a scalar string")
    metadata_text = str(metadata_array.item())
    try:
        metadata = json.loads(metadata_text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("load schedule metadata_json is not valid JSON") from error
    expected_metadata_fields = {
        "schema",
        "source_archive_path",
        "source_archive_sha256",
        "source_state_field",
        "source_state_count",
        "target_count",
        "target_dtype",
        "target_shape",
        "canonical_target_sha256",
        "load_start",
        "load_final",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata_fields:
        raise ValueError("load schedule metadata fields differ from the explicit contract")
    canonical_metadata = _schedule_json_text(metadata)
    if canonical_metadata != metadata_text:
        raise ValueError("load schedule metadata_json is not canonical")
    if metadata["schema"] != STABILIZED_SCHEDULE_ARTIFACT_SCHEMA:
        raise ValueError("load schedule artifact schema is invalid")
    if metadata["target_dtype"] != STABILIZED_SCHEDULE_TARGET_DTYPE:
        raise ValueError("load schedule metadata target dtype is invalid")
    if (
        metadata["target_shape"] != [len(targets)]
        or type(metadata["target_count"]) is not int
        or metadata["target_count"] != len(targets)
    ):
        raise ValueError("load schedule metadata target shape/count is invalid")
    actual_target_sha256 = _schedule_target_sha256(targets)
    if metadata["canonical_target_sha256"] != actual_target_sha256:
        raise ValueError("load schedule target SHA256 does not match raw target bytes")
    if actual_target_sha256 != STABILIZED_SCHEDULE_TARGET_SHA256:
        raise ValueError("load schedule target SHA256 is not the reviewed canonical digest")
    if len(targets) != STABILIZED_SCHEDULE_TARGET_COUNT:
        raise ValueError("load schedule target count is not the reviewed canonical count")
    source_path_value = metadata["source_archive_path"]
    source_sha_value = metadata["source_archive_sha256"]
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ValueError("load schedule source archive path is invalid")
    source_path = Path(source_path_value)
    if not source_path.is_absolute() or str(source_path.resolve()) != source_path_value:
        raise ValueError("load schedule source archive path must be resolved absolute")
    _digest(source_sha_value, "load schedule source archive SHA256")
    if source_sha_value != STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256:
        raise ValueError("load schedule source archive SHA256 is not the reviewed archive")
    source_state, _ = _schedule_source_state(source_path, source_sha_value)
    if metadata["source_state_field"] != STABILIZED_SCHEDULE_SOURCE_STATE_FIELD:
        raise ValueError("load schedule source state field is invalid")
    if (
        type(metadata["source_state_count"]) is not int
        or metadata["source_state_count"] != len(source_state)
    ):
        raise ValueError("load schedule source state count is not paired")
    if not np.array_equal(source_state[1:], targets):
        raise ValueError(
            "load schedule targets do not match source archive state_u_bar[1:] bitwise"
        )
    declared_start = _scalar(metadata["load_start"], "load schedule metadata load_start")
    declared_final = _scalar(metadata["load_final"], "load schedule metadata load_final")
    if declared_start != _scalar(load_start, "load schedule load_start"):
        raise ValueError("load schedule load_start does not match the configuration")
    if declared_final != _scalar(load_final, "load schedule load_final"):
        raise ValueError("load schedule load_final does not match the configuration")
    if targets[0] <= declared_start or targets[-1] != declared_final:
        raise ValueError("load schedule endpoints do not match the explicit load rule")
    return {
        "path": str(resolved),
        "file_sha256": file_sha256,
        "targets": np.ascontiguousarray(targets, dtype=np.dtype("<f8")),
        "canonical_target_sha256": actual_target_sha256,
        "source_archive_path": source_path_value,
        "source_archive_sha256": source_sha_value,
        "target_count": len(targets),
        "load_start": declared_start,
        "load_final": declared_final,
        "source_state_count": len(source_state),
        "source_state_field": STABILIZED_SCHEDULE_SOURCE_STATE_FIELD,
    }


def load_stabilized_load_schedule(
    path: Any,
    *,
    load_start: float,
    load_final: float,
) -> StabilizedLoadSchedule:
    """Load the reviewed numeric schedule artifact with all source bindings."""

    payload = _read_stabilized_load_schedule_artifact(
        path,
        load_start=load_start,
        load_final=load_final,
    )
    return StabilizedLoadSchedule(**payload)


def extract_stabilized_load_schedule(
    source_archive: Any,
    output_path: Any,
) -> StabilizedLoadSchedule:
    """Extract the reviewed source load sequence to a unique numeric NPZ."""

    source = Path(os.fspath(source_archive)).resolve()
    output = Path(os.fspath(output_path)).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"matched-load schedule source archive does not exist: {source}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite load schedule {output}")
    source_sha256 = _archive_sha256(source)
    if source_sha256 != STABILIZED_SCHEDULE_SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            "schedule extraction source archive is not the reviewed historical archive"
        )
    state_u, _ = _schedule_source_state(source, source_sha256)
    targets = np.ascontiguousarray(state_u[1:], dtype=np.dtype("<f8"))
    metadata = {
        "schema": STABILIZED_SCHEDULE_ARTIFACT_SCHEMA,
        "source_archive_path": str(source),
        "source_archive_sha256": source_sha256,
        "source_state_field": STABILIZED_SCHEDULE_SOURCE_STATE_FIELD,
        "source_state_count": len(state_u),
        "target_count": len(targets),
        "target_dtype": STABILIZED_SCHEDULE_TARGET_DTYPE,
        "target_shape": [len(targets)],
        "canonical_target_sha256": _schedule_target_sha256(targets),
        "load_start": float(state_u[0]),
        "load_final": float(state_u[-1]),
    }
    if metadata["canonical_target_sha256"] != STABILIZED_SCHEDULE_TARGET_SHA256:
        raise ValueError("historical source archive does not contain the reviewed schedule")
    arrays = {
        "targets": targets,
        "metadata_json": np.asarray(_schedule_json_text(metadata), dtype=np.str_),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".npz", dir=output.parent, delete=False
        ) as handle:
            temporary_name = handle.name
        np.savez_compressed(temporary_name, **arrays)
        with Path(temporary_name).open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-linking the completed temporary file is an exclusive create:
        # unlike ``os.replace``, a concurrent creator cannot be overwritten.
        try:
            os.link(temporary_name, output)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite load schedule {output}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    try:
        directory_fd = os.open(output.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return load_stabilized_load_schedule(
        output,
        load_start=float(state_u[0]),
        load_final=float(state_u[-1]),
    )


@dataclass(frozen=True)
class StabilizedEngineeringGripConfig:
    """Base engineering settings plus explicit fixed-point globalization."""

    base: engineering.EngineeringGripConfig
    omega_initial: float
    omega_min: float
    omega_reduction: float | None
    sufficient_decrease: float | None
    max_backtracks: int
    # Keep the historical positional constructor order intact while making the
    # policy explicit in the serialized configuration and workflow CLI.
    relaxation_policy: str = "sufficient_decrease"
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE
    # These fields are deliberately absent for the non-Anderson policies.  The
    # workflow requires them explicitly for either Anderson policy; ``None`` is
    # a validation sentinel, not a numerical default.
    anderson_history_depth: int | None = None
    anderson_beta: float | None = None
    anderson_regularization_mu: float | None = None
    # The backtracked residual policy has its own alpha reduction parameter;
    # it must not be overloaded onto the historical omega reduction field.
    anderson_alpha_reduction: float | None = None
    # A schedule is a separate strict numerical contract.  It is intentionally
    # optional so historical adaptive configurations retain their v5 payloads.
    load_schedule: StabilizedLoadSchedule | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base, engineering.EngineeringGripConfig):
            raise TypeError("stabilized config base must be EngineeringGripConfig")
        policy = self.relaxation_policy
        if policy not in STABILIZED_RELAXATION_POLICIES:
            raise ValueError(
                "relaxation_policy must be one of "
                f"{STABILIZED_RELAXATION_POLICIES}, got {policy!r}"
            )
        mode = _validate_stabilized_acceptance_mode(self.acceptance_mode)
        if mode == STABILIZED_WARNING_ACCEPTANCE_MODE and policy != "fixed":
            raise ValueError(
                "stabilized warning acceptance requires the fixed relaxation policy"
            )
        schedule = self.load_schedule
        if schedule is not None:
            if not isinstance(schedule, StabilizedLoadSchedule):
                raise TypeError("stabilized load_schedule has the wrong type")
            # Recheck the external immutable binding at the configuration
            # boundary.  This catches a source/schedule replacement between
            # public schedule construction and config creation without moving
            # any filesystem I/O into the continuation/JIT path.
            schedule._validate_provenance()
            if mode != STABILIZED_ACCEPTANCE_MODE:
                raise ValueError(
                    "explicit load schedules require strict stabilized acceptance"
                )
            if policy != "anderson_residual":
                raise ValueError(
                    "explicit load schedules require the anderson_residual policy"
                )
            if schedule.load_start != self.base.load_start:
                raise ValueError(
                    "load schedule start does not match the base configuration"
                )
            if schedule.load_final != self.base.load_final:
                raise ValueError(
                    "load schedule final target does not match the base configuration"
                )
        omega_initial = _scalar(self.omega_initial, "omega_initial")
        omega_min = _scalar(self.omega_min, "omega_min")
        omega_reduction = _optional_scalar(
            self.omega_reduction, "omega_reduction"
        )
        sufficient = _optional_scalar(
            self.sufficient_decrease, "sufficient_decrease"
        )
        max_backtracks = engineering._integer(
            self.max_backtracks, "max_backtracks"
        )
        if not 0.0 < omega_min <= omega_initial <= 1.0:
            raise ValueError(
                "stabilized relaxation requires 0 < omega_min <= omega_initial <= 1"
            )
        anderson_depth: int | None = None
        anderson_beta: float | None = None
        anderson_regularization_mu: float | None = None
        anderson_alpha_reduction: float | None = None
        if policy == "fixed":
            if omega_min != omega_initial:
                raise ValueError(
                    "fixed relaxation requires omega_min == omega_initial"
                )
            if max_backtracks != 0:
                raise ValueError("fixed relaxation requires max_backtracks == 0")
            if omega_reduction is not None:
                raise ValueError(
                    "omega_reduction is not valid for fixed relaxation"
                )
            if sufficient is not None:
                raise ValueError(
                    "sufficient_decrease is not valid for fixed relaxation"
                )
            if (
                self.anderson_history_depth is not None
                or self.anderson_beta is not None
                or self.anderson_regularization_mu is not None
                or self.anderson_alpha_reduction is not None
            ):
                raise ValueError(
                    "Anderson fields are only valid for an Anderson relaxation policy"
                )
        elif policy == "sufficient_decrease":
            if omega_reduction is None:
                raise ValueError(
                    "sufficient_decrease relaxation requires omega_reduction"
                )
            if sufficient is None:
                raise ValueError(
                    "sufficient_decrease relaxation requires sufficient_decrease"
                )
            if not 0.0 < omega_reduction < 1.0:
                raise ValueError(
                    "omega_reduction must lie strictly between zero and one"
                )
            if not 0.0 < sufficient < 1.0:
                raise ValueError(
                    "sufficient_decrease must lie strictly between zero and one"
                )
            if (
                self.anderson_history_depth is not None
                or self.anderson_beta is not None
                or self.anderson_regularization_mu is not None
                or self.anderson_alpha_reduction is not None
            ):
                raise ValueError(
                    "Anderson fields are only valid for an Anderson relaxation policy"
                )
        elif policy in ("anderson", "anderson_residual"):
            if omega_initial != STABILIZED_ANDERSON_BETA or omega_min != STABILIZED_ANDERSON_BETA:
                raise ValueError(
                    f"{policy} relaxation requires omega_initial == omega_min == 0.5"
                )
            if max_backtracks != 0:
                raise ValueError(f"{policy} relaxation requires max_backtracks == 0")
            if omega_reduction is not None:
                raise ValueError(
                    f"omega_reduction is not valid for {policy} relaxation"
                )
            if sufficient is None:
                raise ValueError(
                    f"{policy} relaxation requires sufficient_decrease for its explicit residual threshold"
                )
            if not 0.0 < sufficient < 1.0:
                raise ValueError(
                    "sufficient_decrease must lie strictly between zero and one"
                )
            if self.anderson_history_depth is None:
                raise ValueError(
                    f"{policy} relaxation requires anderson_history_depth"
                )
            if self.anderson_beta is None:
                raise ValueError(f"{policy} relaxation requires anderson_beta")
            anderson_depth = engineering._integer(
                self.anderson_history_depth,
                "anderson_history_depth",
                positive=True,
            )
            anderson_beta = _scalar(self.anderson_beta, "anderson_beta")
            if anderson_depth != STABILIZED_ANDERSON_HISTORY_DEPTH:
                raise ValueError(
                    f"{policy} history depth must equal "
                    f"{STABILIZED_ANDERSON_HISTORY_DEPTH}"
                )
            if anderson_beta != STABILIZED_ANDERSON_BETA:
                raise ValueError(
                    f"{policy} beta must equal {STABILIZED_ANDERSON_BETA}"
                )
            if self.anderson_alpha_reduction is not None:
                raise ValueError(
                    "anderson_alpha_reduction is only valid for "
                    "a residual Anderson backtracking policy"
                )
            if self.anderson_regularization_mu is not None:
                raise ValueError(
                    "anderson_regularization_mu is only valid for "
                    "anderson_residual_regularized_backtracking"
                )
        else:
            if policy not in (
                STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_POLICY,
                STABILIZED_ANDERSON_REGULARIZED_RESIDUAL_BACKTRACKING_POLICY,
            ):
                raise ValueError(f"unsupported Anderson relaxation policy {policy!r}")
            if omega_initial != STABILIZED_ANDERSON_BETA or omega_min != STABILIZED_ANDERSON_BETA:
                raise ValueError(
                    f"{policy} relaxation requires omega_initial == omega_min == 0.5"
                )
            if max_backtracks != STABILIZED_ANDERSON_MAX_BACKTRACKS:
                raise ValueError(
                    f"{policy} relaxation requires max_backtracks == "
                    f"{STABILIZED_ANDERSON_MAX_BACKTRACKS}"
                )
            if omega_reduction is not None:
                raise ValueError(
                    f"omega_reduction is not valid for {policy} relaxation"
                )
            if sufficient is None:
                raise ValueError(
                    f"{policy} relaxation requires sufficient_decrease for its "
                    "explicit residual threshold"
                )
            if not 0.0 < sufficient < 1.0:
                raise ValueError(
                    "sufficient_decrease must lie strictly between zero and one"
                )
            if self.anderson_history_depth is None:
                raise ValueError(
                    f"{policy} relaxation requires anderson_history_depth"
                )
            if self.anderson_beta is None:
                raise ValueError(f"{policy} relaxation requires anderson_beta")
            if self.anderson_alpha_reduction is None:
                raise ValueError(
                    f"{policy} relaxation requires anderson_alpha_reduction"
                )
            if policy == STABILIZED_ANDERSON_REGULARIZED_RESIDUAL_BACKTRACKING_POLICY:
                if self.anderson_regularization_mu is None:
                    raise ValueError(
                        f"{policy} relaxation requires anderson_regularization_mu"
                    )
                anderson_regularization_mu = _scalar(
                    self.anderson_regularization_mu,
                    "anderson_regularization_mu",
                )
                if anderson_regularization_mu <= 0.0:
                    raise ValueError("anderson_regularization_mu must be positive")
            elif self.anderson_regularization_mu is not None:
                raise ValueError(
                    "anderson_regularization_mu is only valid for "
                    "anderson_residual_regularized_backtracking"
                )
            anderson_depth = engineering._integer(
                self.anderson_history_depth,
                "anderson_history_depth",
                positive=True,
            )
            anderson_beta = _scalar(self.anderson_beta, "anderson_beta")
            anderson_alpha_reduction = _scalar(
                self.anderson_alpha_reduction,
                "anderson_alpha_reduction",
            )
            if anderson_depth != STABILIZED_ANDERSON_HISTORY_DEPTH:
                raise ValueError(
                    f"{policy} history depth must equal "
                    f"{STABILIZED_ANDERSON_HISTORY_DEPTH}"
                )
            if anderson_beta != STABILIZED_ANDERSON_BETA:
                raise ValueError(
                    f"{policy} beta must equal {STABILIZED_ANDERSON_BETA}"
                )
            if anderson_alpha_reduction != STABILIZED_ANDERSON_ALPHA_REDUCTION:
                raise ValueError(
                    f"{policy} alpha reduction must equal "
                    f"{STABILIZED_ANDERSON_ALPHA_REDUCTION}"
                )
        # A zero limit would make a dimensionless ratio undefined.  The base
        # controller permits such a gate, but the stabilized merit requires an
        # explicit positive absolute or relative tolerance instead of an
        # implicit numerical floor.
        for name in STABILIZED_COUPLED_GATE_NAMES:
            if name in {
                "finite",
                "mechanics_before",
                "mechanics_after",
                "damage_fixed_point",
                "damage_equation",
                "globalization",
            }:
                continue
            if not hasattr(self.base, name):
                continue
            gate = getattr(self.base, name)
            if gate.atol == 0.0 and gate.rtol == 0.0:
                raise ValueError(
                    f"stabilized merit gate {name} needs a positive tolerance"
                )
        object.__setattr__(self, "omega_initial", omega_initial)
        object.__setattr__(self, "omega_min", omega_min)
        object.__setattr__(self, "omega_reduction", omega_reduction)
        object.__setattr__(self, "sufficient_decrease", sufficient)
        object.__setattr__(self, "max_backtracks", max_backtracks)
        object.__setattr__(self, "relaxation_policy", policy)
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "anderson_history_depth", anderson_depth)
        object.__setattr__(self, "anderson_beta", anderson_beta)
        object.__setattr__(
            self,
            "anderson_regularization_mu",
            anderson_regularization_mu,
        )
        object.__setattr__(
            self,
            "anderson_alpha_reduction",
            anderson_alpha_reduction,
        )
        object.__setattr__(self, "load_schedule", schedule)

    @classmethod
    def smoke(cls) -> "StabilizedEngineeringGripConfig":
        """Return settings for the short manufactured checkpoint only."""

        return cls(
            base=engineering.EngineeringGripConfig.smoke(),
            omega_initial=0.5,
            omega_min=0.03125,
            omega_reduction=0.5,
            sufficient_decrease=1.0e-2,
            max_backtracks=5,
            relaxation_policy="sufficient_decrease",
        )

    def payload(self) -> dict[str, Any]:
        payload = {
            "relaxation_policy": self.relaxation_policy,
            "omega_initial": self.omega_initial,
            "omega_min": self.omega_min,
            "omega_reduction": self.omega_reduction,
            "sufficient_decrease": self.sufficient_decrease,
            "max_backtracks": self.max_backtracks,
        }
        if _is_anderson_policy(self.relaxation_policy):
            if self.anderson_history_depth is None or self.anderson_beta is None:
                raise ValueError("anderson configuration is missing explicit fields")
            payload.update(
                {
                    "anderson_history_depth": self.anderson_history_depth,
                    "anderson_beta": self.anderson_beta,
                }
            )
            if _is_anderson_residual_policy(self.relaxation_policy):
                payload["anderson_merit_identity"] = (
                    _anderson_merit_identity(self.relaxation_policy)
                )
            if _is_anderson_residual_backtracking_policy(self.relaxation_policy):
                if self.anderson_alpha_reduction is None:
                    raise ValueError(
                        "backtracked Anderson configuration is missing alpha reduction"
                    )
                payload["anderson_alpha_reduction"] = (
                    self.anderson_alpha_reduction
                )
            if _is_anderson_regularized_policy(self.relaxation_policy):
                if self.anderson_regularization_mu is None:
                    raise ValueError(
                        "regularized Anderson configuration is missing mu"
                    )
                payload["anderson_regularization_mu"] = (
                    self.anderson_regularization_mu
                )
        # Historical strict v5 configuration payloads are intentionally left
        # byte/schema compatible.  The experimental warning archive carries
        # its mode explicitly because it is a different contract.
        if self.acceptance_mode != STABILIZED_ACCEPTANCE_MODE:
            payload["acceptance_mode"] = self.acceptance_mode
        if self.load_schedule is not None:
            payload["load_schedule"] = self.load_schedule.payload()
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        base: engineering.EngineeringGripConfig,
    ) -> "StabilizedEngineeringGripConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("stabilized config payload must be a mapping")
        required = {
            "relaxation_policy",
            "omega_initial",
            "omega_min",
            "omega_reduction",
            "sufficient_decrease",
            "max_backtracks",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"stabilized config is missing fields: {missing}")
        policy = payload["relaxation_policy"]
        if _is_anderson_policy(policy):
            anderson_fields = {"anderson_history_depth", "anderson_beta"}
            missing_anderson = sorted(anderson_fields - set(payload))
            if missing_anderson:
                raise ValueError(
                    f"stabilized Anderson config is missing fields: {missing_anderson}"
                )
        else:
            anderson_fields = set()
            unexpected_anderson = sorted(
                {"anderson_history_depth", "anderson_beta"} & set(payload)
            )
            if unexpected_anderson:
                raise ValueError(
                    "stabilized non-Anderson config contains Anderson fields: "
                    f"{unexpected_anderson}"
                )
        alpha_fields = (
            {"anderson_alpha_reduction"}
            if _is_anderson_residual_backtracking_policy(policy)
            else set()
        )
        missing_alpha = sorted(alpha_fields - set(payload))
        if missing_alpha:
            raise ValueError(
                "stabilized backtracked Anderson config is missing fields: "
                f"{missing_alpha}"
            )
        regularization_fields = (
            {"anderson_regularization_mu"}
            if _is_anderson_regularized_policy(policy)
            else set()
        )
        missing_regularization = sorted(regularization_fields - set(payload))
        if missing_regularization:
            raise ValueError(
                "stabilized regularized Anderson config is missing fields: "
                f"{missing_regularization}"
            )
        if not _is_anderson_regularized_policy(policy) and (
            "anderson_regularization_mu" in payload
        ):
            raise ValueError(
                "anderson_regularization_mu is only valid for "
                "anderson_residual_regularized_backtracking"
            )
        identity_fields = (
            {"anderson_merit_identity"}
            if _is_anderson_residual_policy(policy)
            else set()
        )
        if policy == "anderson" and "anderson_merit_identity" in payload:
            raise ValueError(
                "historical anderson configuration cannot contain a residual merit identity"
            )
        if _is_anderson_residual_policy(policy):
            if payload.get("anderson_merit_identity") != _anderson_merit_identity(policy):
                raise ValueError(f"{policy} merit identity is invalid")
        if not _is_anderson_residual_backtracking_policy(policy) and (
            "anderson_alpha_reduction" in payload
        ):
            raise ValueError(
                "anderson_alpha_reduction is only valid for "
                "a residual Anderson backtracking policy"
            )
        schedule_payload = payload.get("load_schedule")
        if schedule_payload is not None and not isinstance(schedule_payload, Mapping):
            raise ValueError("stabilized load_schedule payload must be a mapping")
        allowed = (
            required
            | {"acceptance_mode", "load_schedule"}
            | anderson_fields
            | alpha_fields
            | regularization_fields
            | identity_fields
        )
        unexpected = sorted(set(payload) - allowed)
        if unexpected:
            raise ValueError(f"stabilized config contains unknown fields: {unexpected}")
        acceptance_mode = (
            STABILIZED_ACCEPTANCE_MODE
            if "acceptance_mode" not in payload
            else payload["acceptance_mode"]
        )
        schedule = (
            None
            if schedule_payload is None
            else StabilizedLoadSchedule.from_payload(
                schedule_payload,
                load_start=base.load_start,
                load_final=base.load_final,
            )
        )
        return cls(
            base=base,
            relaxation_policy=policy,
            omega_initial=payload["omega_initial"],
            omega_min=payload["omega_min"],
            omega_reduction=payload["omega_reduction"],
            sufficient_decrease=payload["sufficient_decrease"],
            max_backtracks=payload["max_backtracks"],
            acceptance_mode=acceptance_mode,
            anderson_history_depth=(
                payload["anderson_history_depth"]
                if _is_anderson_policy(policy)
                else None
            ),
            anderson_beta=(payload["anderson_beta"] if _is_anderson_policy(policy) else None),
            anderson_regularization_mu=(
                payload["anderson_regularization_mu"]
                if _is_anderson_regularized_policy(policy)
                else None
            ),
            anderson_alpha_reduction=(
                payload["anderson_alpha_reduction"]
                if _is_anderson_residual_backtracking_policy(policy)
                else None
            ),
            load_schedule=schedule,
        )


@dataclass(frozen=True)
class _AndersonProposal:
    """Scalar audit plus a candidate for one Type-II Anderson sweep."""

    candidate: np.ndarray | None
    history_size: int
    qr_rank: int
    fallback_reason: str | None
    restart_reason: str | None
    anderson_coefficient_l2: float | None = None
    anderson_regularization_mu: float | None = None
    anderson_current_residual_l2: float | None = None
    anderson_regularization_lambda: float | None = None
    anderson_unregularized_ls_residual_l2: float | None = None


def _anderson_rank_tolerance(matrix: np.ndarray) -> float:
    """Return the deterministic scale-aware QR rank threshold."""

    values = _f64(matrix, "Anderson QR matrix")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Anderson QR matrix must be two-dimensional and nonempty")
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    # The matrix scale is intentional: residual differences may be far below
    # one in the late fixed-point path, so an absolute floor of one would turn
    # every small but resolved direction into a false rank deficiency.
    return float(
        max(values.shape)
        * _EPS64
        * max(scale, np.finfo(np.float64).tiny)
    )


def _anderson_scalar_relation_close(actual: float, expected: float) -> bool:
    """Compare derived Anderson scalars with a scale-aware float64 tolerance."""

    actual_value = float(actual)
    expected_value = float(expected)
    if not np.isfinite(actual_value) or not np.isfinite(expected_value):
        return False
    scale = max(
        abs(actual_value),
        abs(expected_value),
        np.finfo(np.float64).tiny,
    )
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    return abs(actual_value - expected_value) <= tolerance


def _anderson_propose(
    history_x: Sequence[np.ndarray],
    history_f: Sequence[np.ndarray],
    x_current: np.ndarray,
    g_current: np.ndarray,
    *,
    history_depth: int,
    beta: float,
    damage_range_atol: float,
    raw_range_safeguard: bool = True,
) -> _AndersonProposal:
    """Build one safeguarded damped Type-II Anderson candidate.

    The residual is ``f=g-x``.  With the most recent ``m`` residual/state
    differences, the pivoted-QR solve is

    ``min ||f_k - DeltaF gamma||_2``

    and the damped Type-II update is

    ``x_k + beta*f_k - (DeltaX + beta*DeltaF) gamma``.

    No candidate is clipped or projected.  A deficient QR or non-finite result
    returns an explicit fallback reason and asks the caller to restart its
    history.  Historical policies additionally apply a raw damage-range
    safeguard to the proposal; the distinct backtracking policy defers that
    existing hard gate to each finite alpha trial.
    """

    x = _f64(x_current, "Anderson current iterate")
    g = _f64(g_current, "Anderson current fixed-point image", x.shape)
    depth = engineering._integer(history_depth, "Anderson history depth", positive=True)
    mixing = _scalar(beta, "Anderson beta")
    range_atol = _scalar(damage_range_atol, "Anderson damage range tolerance")
    if type(raw_range_safeguard) is not bool:
        raise TypeError("Anderson raw range safeguard must be a boolean")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("Anderson beta must lie in (0, 1]")
    if range_atol < 0.0:
        raise ValueError("Anderson damage range tolerance must be non-negative")
    if len(history_x) != len(history_f):
        raise ValueError("Anderson state/residual history lengths differ")
    for index, (past_x, past_f) in enumerate(zip(history_x, history_f)):
        _f64(past_x, f"Anderson history x[{index}]", x.shape)
        _f64(past_f, f"Anderson history f[{index}]", x.shape)
    current_f = g - x
    if not np.all(np.isfinite(current_f)):
        return _AndersonProposal(
            None, 0, 0, "nonfinite_residual", "nonfinite_residual"
        )
    history_size = min(len(history_f), depth)
    if history_size == 0:
        return _AndersonProposal(None, 0, 0, "insufficient_history", None)

    past_x = tuple(np.asarray(value, dtype=np.float64) for value in history_x[-history_size:])
    past_f = tuple(np.asarray(value, dtype=np.float64) for value in history_f[-history_size:])
    all_x = (*past_x, x)
    all_f = (*past_f, current_f)
    delta_x = np.column_stack(
        [all_x[index + 1] - all_x[index] for index in range(history_size)]
    )
    delta_f = np.column_stack(
        [all_f[index + 1] - all_f[index] for index in range(history_size)]
    )
    if not np.all(np.isfinite(delta_x)) or not np.all(np.isfinite(delta_f)):
        return _AndersonProposal(
            None,
            history_size,
            0,
            "nonfinite_history_difference",
            "nonfinite_history_difference",
        )
    tolerance = _anderson_rank_tolerance(delta_f)
    try:
        q_matrix, r_matrix, pivots = _pivoted_qr(
            delta_f,
            mode="economic",
            pivoting=True,
            check_finite=False,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        del error
        return _AndersonProposal(
            None,
            history_size,
            0,
            "qr_failure",
            "qr_failure",
        )
    diagonal = np.abs(np.diag(r_matrix))
    qr_rank = int(np.count_nonzero(diagonal > tolerance))
    if qr_rank < history_size:
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "rank_deficient",
            "rank_deficient",
        )
    projected = q_matrix.T @ current_f
    if not np.all(np.isfinite(projected)):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "nonfinite_qr_projection",
            "nonfinite_qr_projection",
        )
    try:
        gamma_permuted = _solve_triangular(
            r_matrix[:qr_rank, :qr_rank],
            projected[:qr_rank],
            lower=False,
            check_finite=False,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        del error
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "triangular_solve_failure",
            "triangular_solve_failure",
        )
    gamma_permuted = np.asarray(gamma_permuted, dtype=np.float64)
    gamma = np.zeros(history_size, dtype=np.float64)
    gamma[np.asarray(pivots[:qr_rank], dtype=np.int64)] = gamma_permuted
    candidate = x + mixing * current_f - (delta_x + mixing * delta_f) @ gamma
    if not np.all(np.isfinite(candidate)):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "nonfinite_candidate",
            "nonfinite_candidate",
        )
    if raw_range_safeguard and (
        float(np.min(candidate)) < -range_atol
        or float(np.max(candidate)) > 1.0 + range_atol
    ):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "out_of_range_candidate",
            "out_of_range_candidate",
        )
    coefficient_norm = float(np.linalg.norm(gamma))
    if not np.isfinite(coefficient_norm):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "nonfinite_coefficient_norm",
            "nonfinite_coefficient_norm",
        )
    return _AndersonProposal(
        np.asarray(candidate, dtype=np.float64),
        history_size,
        qr_rank,
        None,
        None,
        coefficient_norm,
    )


def _anderson_propose_regularized(
    history_x: Sequence[np.ndarray],
    history_f: Sequence[np.ndarray],
    x_current: np.ndarray,
    g_current: np.ndarray,
    *,
    history_depth: int,
    beta: float,
    regularization_mu: float,
    damage_range_atol: float,
) -> _AndersonProposal:
    """Build the fixed-mu augmented-QR Type-II Anderson proposal.

    The state/history map and damped Type-II candidate are identical to
    :func:`_anderson_propose`.  Only the coefficient solve changes: the
    consecutive residual-difference block is augmented with
    ``sqrt(lambda) * I``, where ``lambda = mu * ||f||_2**2``.  The solve is
    performed by pivoted QR on the augmented matrix; normal equations are not
    used.  A structural QR/rank/finite failure requests the same history
    restart semantics as the historical proposal.
    """

    x = _f64(x_current, "regularized Anderson current iterate")
    g = _f64(g_current, "regularized Anderson current fixed-point image", x.shape)
    depth = engineering._integer(
        history_depth, "regularized Anderson history depth", positive=True
    )
    mixing = _scalar(beta, "Anderson beta")
    mu = _scalar(regularization_mu, "anderson_regularization_mu")
    if mu <= 0.0:
        raise ValueError("anderson_regularization_mu must be positive")
    if not 0.0 < mixing <= 1.0:
        raise ValueError("Anderson beta must lie in (0, 1]")
    range_atol = _scalar(
        damage_range_atol, "regularized Anderson damage range tolerance"
    )
    if range_atol < 0.0:
        raise ValueError(
            "regularized Anderson damage range tolerance must be non-negative"
        )
    if len(history_x) != len(history_f):
        raise ValueError("regularized Anderson state/residual history lengths differ")
    for index, (past_x, past_f) in enumerate(zip(history_x, history_f)):
        _f64(past_x, f"regularized Anderson history x[{index}]", x.shape)
        _f64(past_f, f"regularized Anderson history f[{index}]", x.shape)

    current_f = g - x
    current_residual_l2 = float(np.linalg.norm(current_f))
    regularization_lambda = float(mu * current_residual_l2**2)
    if not np.isfinite(current_residual_l2) or not np.isfinite(regularization_lambda):
        return _AndersonProposal(
            None,
            0,
            0,
            "nonfinite_residual",
            "nonfinite_residual",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )
    history_size = min(len(history_f), depth)
    if history_size == 0:
        return _AndersonProposal(
            None,
            0,
            0,
            "insufficient_history",
            None,
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )

    past_x = tuple(
        np.asarray(value, dtype=np.float64) for value in history_x[-history_size:]
    )
    past_f = tuple(
        np.asarray(value, dtype=np.float64) for value in history_f[-history_size:]
    )
    all_x = (*past_x, x)
    all_f = (*past_f, current_f)
    delta_x = np.column_stack(
        [all_x[index + 1] - all_x[index] for index in range(history_size)]
    )
    delta_f = np.column_stack(
        [all_f[index + 1] - all_f[index] for index in range(history_size)]
    )
    if not np.all(np.isfinite(delta_x)) or not np.all(np.isfinite(delta_f)):
        return _AndersonProposal(
            None,
            history_size,
            0,
            "nonfinite_history_difference",
            "nonfinite_history_difference",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )

    augmented = np.vstack(
        (
            delta_f,
            np.sqrt(regularization_lambda)
            * np.eye(history_size, dtype=np.float64),
        )
    )
    right_hand_side = np.concatenate(
        (current_f, np.zeros(history_size, dtype=np.float64))
    )
    if not np.all(np.isfinite(augmented)) or not np.all(
        np.isfinite(right_hand_side)
    ):
        return _AndersonProposal(
            None,
            history_size,
            0,
            "nonfinite_augmented_system",
            "nonfinite_augmented_system",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )
    tolerance = _anderson_rank_tolerance(augmented)
    try:
        q_matrix, r_matrix, pivots = _pivoted_qr(
            augmented,
            mode="economic",
            pivoting=True,
            check_finite=False,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        del error
        return _AndersonProposal(
            None,
            history_size,
            0,
            "qr_failure",
            "qr_failure",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )
    diagonal = np.abs(np.diag(r_matrix))
    qr_rank = int(np.count_nonzero(diagonal > tolerance))
    if qr_rank < history_size:
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "rank_deficient",
            "rank_deficient",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )
    projected = q_matrix.T @ right_hand_side
    if not np.all(np.isfinite(projected)):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "nonfinite_qr_projection",
            "nonfinite_qr_projection",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )
    try:
        gamma_permuted = _solve_triangular(
            r_matrix[:qr_rank, :qr_rank],
            projected[:qr_rank],
            lower=False,
            check_finite=False,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        del error
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "triangular_solve_failure",
            "triangular_solve_failure",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
        )
    gamma_permuted = np.asarray(gamma_permuted, dtype=np.float64)
    gamma = np.zeros(history_size, dtype=np.float64)
    gamma[np.asarray(pivots[:qr_rank], dtype=np.int64)] = gamma_permuted
    candidate = x + mixing * current_f - (delta_x + mixing * delta_f) @ gamma
    coefficient_l2 = float(np.linalg.norm(gamma))
    unregularized_ls_residual_l2 = float(
        np.linalg.norm(current_f - delta_f @ gamma)
    )
    if not np.all(np.isfinite(candidate)):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "nonfinite_candidate",
            "nonfinite_candidate",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
            anderson_unregularized_ls_residual_l2=unregularized_ls_residual_l2,
        )
    if not np.isfinite(coefficient_l2) or not np.isfinite(
        unregularized_ls_residual_l2
    ):
        return _AndersonProposal(
            None,
            history_size,
            qr_rank,
            "nonfinite_coefficient_norm",
            "nonfinite_coefficient_norm",
            anderson_regularization_mu=mu,
            anderson_current_residual_l2=current_residual_l2,
            anderson_regularization_lambda=regularization_lambda,
            anderson_unregularized_ls_residual_l2=unregularized_ls_residual_l2,
        )
    return _AndersonProposal(
        np.asarray(candidate, dtype=np.float64),
        history_size,
        qr_rank,
        None,
        None,
        coefficient_l2,
        mu,
        current_residual_l2,
        regularization_lambda,
        unregularized_ls_residual_l2,
    )


def _archive_dimension(value: Any, name: str) -> int:
    result = engineering._integer(value, name, positive=True)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _archive_count(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer count")
    result = engineering._integer(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _archive_counts(values: Mapping[str, Any], *, name: str) -> dict[str, int]:
    """Validate one complete observed/archive count mapping."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(values) != set(_ARCHIVE_COUNT_NAMES):
        raise ValueError(
            f"{name} must contain exactly {list(_ARCHIVE_COUNT_NAMES)}"
        )
    return {
        count_name: _archive_count(values[count_name], f"{name}.{count_name}")
        for count_name in _ARCHIVE_COUNT_NAMES
    }


def _archive_count_delta(
    retained: Mapping[str, Any], prospective: Mapping[str, Any]
) -> dict[str, int]:
    """Return the non-negative row delta between two archive count sets."""

    retained_counts = _archive_counts(retained, name="retained archive counts")
    prospective_counts = _archive_counts(
        prospective, name="prospective archive counts"
    )
    delta: dict[str, int] = {}
    for name in _ARCHIVE_COUNT_NAMES:
        if prospective_counts[name] < retained_counts[name]:
            raise ValueError(
                f"prospective archive {name} count is below retained count"
            )
        delta[name] = prospective_counts[name] - retained_counts[name]
    return delta


_ARCHIVE_STAGE_DETAIL_NAMES = ("step_index", "target_u_bar")


def _archive_stage_details(
    values: Mapping[str, Any], stage_kind: str
) -> dict[str, Any]:
    """Validate the context needed to reconstruct a prospective row delta."""

    if stage_kind not in _ARCHIVE_BUDGET_STAGE_KINDS:
        raise ValueError(f"unknown stabilized archive budget stage: {stage_kind}")
    if not isinstance(values, Mapping):
        raise TypeError("stabilized archive budget stage details must be a mapping")
    if set(values) != set(_ARCHIVE_STAGE_DETAIL_NAMES):
        raise ValueError(
            "stabilized archive budget stage details must contain exactly "
            f"{list(_ARCHIVE_STAGE_DETAIL_NAMES)}"
        )
    result: dict[str, Any] = {}
    for name in ("step_index",):
        value = values[name]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"stabilized archive stage detail {name} must be an integer")
        result[name] = engineering._integer(
            value, f"stabilized archive stage detail {name}"
        )
        if result[name] < 0:
            raise ValueError(
                f"stabilized archive stage detail {name} must be non-negative"
            )
    target = values["target_u_bar"]
    if isinstance(target, (bool, np.bool_)):
        raise TypeError("stabilized archive stage detail target_u_bar must be numeric")
    if not isinstance(target, (int, float, np.integer, np.floating)):
        raise TypeError(
            "stabilized archive stage detail target_u_bar must be numeric"
        )
    target_value = float(target)
    if not np.isfinite(target_value):
        raise ValueError(
            "stabilized archive stage detail target_u_bar must be finite"
        )
    result["target_u_bar"] = target_value
    return result


def _archive_stage_delta(
    stage_kind: str, details: Mapping[str, Any]
) -> dict[str, int]:
    """Derive the rows a budget check was about to retain."""

    context = _archive_stage_details(details, stage_kind)
    if stage_kind != "accepted_step":
        raise ValueError(f"unknown stabilized archive budget stage: {stage_kind}")
    if context["step_index"] < 1:
        raise ValueError("stabilized archive accepted-step index must be positive")
    return {"accepted_steps": 1}


def _validate_archive_stage_label(
    stage: Any, stage_kind: str, details: Mapping[str, Any]
) -> str:
    """Require the human-readable stage label to match its typed context."""

    if not isinstance(stage, str) or not stage:
        raise ValueError("stabilized archive budget stage must be non-empty")
    context = _archive_stage_details(details, stage_kind)
    expected = f"step={context['step_index']} accepted_step"
    if stage != expected:
        raise ValueError("stabilized archive budget stage label is forged")
    return stage


def _validated_archive_dimensions(
    n_material: int, n_union_dofs: int
) -> tuple[int, int]:
    material_count = _archive_dimension(n_material, "n_material")
    union_dof_count = _archive_dimension(n_union_dofs, "n_union_dofs")
    if union_dof_count % 2:
        raise ValueError("n_union_dofs must be an even displacement-DOF count")
    if union_dof_count < 2 * material_count:
        raise ValueError("n_union_dofs must cover every material displacement DOF")
    return material_count, union_dof_count


def _planned_archive_counts(
    config: StabilizedEngineeringGripConfig, *, nominal: bool
) -> dict[str, int]:
    base = config.base
    increment_count = max(
        1,
        int(
            math.ceil(
                base.load_final / (base.du_initial if nominal else base.du_min)
            )
        ),
    )
    del nominal
    return {"accepted_steps": increment_count}


def _validate_archive_budget(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("archive_budget_bytes must be a positive integer byte count")
    return engineering._integer(value, "archive_budget_bytes", positive=True)


def _archive_storage_metadata(
    storage: Mapping[str, Any] | None,
    *,
    archive_budget_bytes: int | None,
) -> dict[str, Any] | None:
    """Validate the compact storage provenance retained in archive metadata."""

    budget = (
        None
        if archive_budget_bytes is None
        else _validate_archive_budget(archive_budget_bytes)
    )
    if storage is None:
        return None if budget is None else {"archive_budget_bytes": budget}
    if not isinstance(storage, Mapping):
        raise TypeError("archive_storage_preflight must be a mapping")
    if budget is None:
        raise ValueError(
            "archive_storage_preflight requires archive_budget_bytes"
        )
    required = (
        "nominal_archive_bytes",
        "theoretical_worst_case_archive_bytes",
        "archive_budget_bytes",
        "available_bytes",
        "filesystem_reserve_bytes",
        "budget_required_free_bytes",
    )
    missing = [name for name in required if name not in storage]
    if missing:
        raise ValueError(
            "archive_storage_preflight is missing required fields: "
            f"{missing}"
        )
    result: dict[str, Any] = {}
    for name in required:
        value = storage[name]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"archive_storage_preflight {name} must be an integer byte count"
            )
        result[name] = engineering._integer(value, f"archive_storage_preflight {name}")
        if result[name] < 0:
            raise ValueError(
                f"archive_storage_preflight {name} must be non-negative"
            )
    if result["archive_budget_bytes"] != budget:
        raise ValueError(
            "archive_storage_preflight archive budget does not match "
            "archive_budget_bytes"
        )
    if result["archive_budget_bytes"] <= 0:
        raise ValueError("archive_storage_preflight archive budget must be positive")
    if result["theoretical_worst_case_archive_bytes"] < result[
        "nominal_archive_bytes"
    ]:
        raise ValueError(
            "archive_storage_preflight worst-case estimate is below nominal estimate"
        )
    if result["budget_required_free_bytes"] != (
        result["archive_budget_bytes"] + result["filesystem_reserve_bytes"]
    ):
        raise ValueError(
            "archive_storage_preflight budget free-space requirement is inconsistent"
        )
    # Keep only the stable, scalar fields needed to reproduce the launch
    # decision.  Paths and the detailed size breakdown remain in workflow
    # summaries, not in every numerical archive.
    result.update(
        {
            "worst_case_archive_bytes": result[
                "theoretical_worst_case_archive_bytes"
            ],
            "free_bytes": result["available_bytes"],
        }
    )
    return result


def _npy_header_bytes(value: np.ndarray) -> int:
    """Return the exact NumPy header length without serializing array data."""

    array = np.asarray(value)
    header = {
        "descr": np.lib.format.dtype_to_descr(array.dtype),
        "fortran_order": bool(array.flags.f_contiguous),
        "shape": array.shape,
    }
    stream = io.BytesIO()
    try:
        np.lib.format.write_array_header_1_0(stream, header)
    except ValueError:
        stream = io.BytesIO()
        np.lib.format.write_array_header_2_0(stream, header)
    return int(stream.tell())


def _uncompressed_archive_bytes(arrays: Mapping[str, Any]) -> int:
    """Bound the uncompressed NPZ bytes for an already materialized payload."""

    total = 22  # End-of-central-directory record.
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("stabilized archive array names must be non-empty strings")
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"stabilized archive field {name} has object dtype")
        member_name = f"{name}.npy".encode("utf-8")
        member_overhead = 30 + len(member_name) + 46 + len(member_name)
        total += _npy_header_bytes(array) + int(array.nbytes) + member_overhead
    # The reserve covers ZIP64 central-directory records and implementation
    # metadata without pretending compression changes the byte contract.
    return int(total + _V5_NPZ_RESERVE_BYTES)


def _archive_free_bytes(probe: Path, free_bytes: int | None) -> int:
    if free_bytes is None:
        try:
            return int(shutil.disk_usage(probe).free)
        except OSError as error:
            raise OSError(
                f"cannot determine free storage for stabilized archive path {probe}"
            ) from error
    if isinstance(free_bytes, (bool, np.bool_)):
        raise TypeError("free_bytes must be an integer byte count")
    result = engineering._integer(free_bytes, "free_bytes")
    if result < 0:
        raise ValueError("free_bytes must be non-negative")
    return result


def _precheck_archive_write(
    path: Any,
    arrays: Mapping[str, Any],
    *,
    archive_budget_bytes: int,
    free_bytes: int | None = None,
) -> dict[str, int]:
    """Check the materialized uncompressed archive before writing it."""

    budget = _validate_archive_budget(archive_budget_bytes)
    filename = os.fsdecode(os.fspath(path))
    if not filename:
        raise ValueError("stabilized archive path must be non-empty")
    output = Path(filename)
    probe = output.parent
    if not probe.exists():
        raise FileNotFoundError(
            f"stabilized archive output directory does not exist: {probe}"
        )
    if not probe.is_dir():
        raise NotADirectoryError(
            f"stabilized archive output filesystem probe is not a directory: {probe}"
        )
    actual = _uncompressed_archive_bytes(arrays)
    available = _archive_free_bytes(probe, free_bytes)
    required = actual + STABILIZED_FILESYSTEM_RESERVE_BYTES
    if actual > budget:
        raise OSError(
            "stabilized archive uncompressed size exceeds archive budget: "
            f"actual_uncompressed_bytes={actual}, archive_budget_bytes={budget}, "
            f"path={output}"
        )
    if available < required:
        raise OSError(
            "insufficient free storage for stabilized archive write: "
            f"required={required} bytes "
            f"(actual_uncompressed_bytes={actual}, "
            f"filesystem_reserve={STABILIZED_FILESYSTEM_RESERVE_BYTES}), "
            f"available={available} bytes, path={output}"
        )
    return {
        "actual_uncompressed_bytes": actual,
        "archive_budget_bytes": budget,
        "available_bytes": available,
        "required_free_bytes": required,
    }


class _StabilizedArchiveBudgetExceeded(Exception):
    """Internal stop raised before a prospective archive row is evaluated."""

    def __init__(
        self,
        stage: str,
        stage_kind: str,
        stage_details: Mapping[str, Any],
        exhaustion_kind: str,
        retained_counts: Mapping[str, Any],
        prospective_counts: Mapping[str, Any],
        estimate: Mapping[str, Any],
        retained_estimate: Mapping[str, Any],
        archive_budget_bytes: int,
        available_bytes: int | None,
    ) -> None:
        if stage_kind not in _ARCHIVE_BUDGET_STAGE_KINDS:
            raise ValueError(f"unknown stabilized archive budget stage: {stage_kind}")
        if exhaustion_kind not in _ARCHIVE_BUDGET_EXHAUSTION_KINDS:
            raise ValueError(
                f"unknown stabilized archive budget exhaustion: {exhaustion_kind}"
            )
        self.stage_kind = stage_kind
        if not isinstance(stage_details, Mapping):
            raise TypeError("stabilized archive budget stage details must be a mapping")
        self.stage_details = _archive_stage_details(stage_details, stage_kind)
        self.stage = _validate_archive_stage_label(
            stage, stage_kind, self.stage_details
        )
        self.exhaustion_kind = exhaustion_kind
        self.retained_counts = _archive_counts(
            retained_counts, name="retained archive counts"
        )
        self.prospective_counts = _archive_counts(
            prospective_counts, name="prospective archive counts"
        )
        self.prospective_delta = _archive_count_delta(
            self.retained_counts, self.prospective_counts
        )
        if self.prospective_delta != _archive_stage_delta(
            stage_kind, self.stage_details
        ):
            raise ValueError(
                "stabilized archive budget stage details do not match row delta"
            )
        self.estimate = dict(estimate)
        self.retained_estimate = dict(retained_estimate)
        self.archive_budget_bytes = int(archive_budget_bytes)
        self.available_bytes = available_bytes
        available_text = (
            "unknown" if available_bytes is None else str(available_bytes)
        )
        super().__init__(
            f"stabilized archive {exhaustion_kind} exhausted before "
            f"{stage} ({stage_kind}): "
            f"retained_counts={self.retained_counts}, "
            f"prospective_counts={self.prospective_counts}, "
            f"estimated_uncompressed_bytes={estimate['archive_bytes']}, "
            f"required_free_bytes={estimate['required_free_bytes']}, "
            f"archive_budget_bytes={archive_budget_bytes}, "
            f"available_bytes={available_text}"
        )


class _StabilizedArchiveBudget:
    """Runtime counter/checker for the bounded stabilized archive contract."""

    def __init__(
        self,
        model: EngineeringGripModel,
        config: StabilizedEngineeringGripConfig,
        archive_budget_bytes: int,
        *,
        archive_output_path: Any | None,
        free_bytes: int | None,
    ) -> None:
        self.model = model
        self.config = config
        self.archive_budget_bytes = _validate_archive_budget(archive_budget_bytes)
        self.archive_output_path = (
            None if archive_output_path is None else Path(os.fspath(archive_output_path))
        )
        if self.archive_output_path is not None:
            probe = self.archive_output_path.parent
            if not probe.exists():
                raise FileNotFoundError(
                    f"stabilized archive output directory does not exist: {probe}"
                )
            if not probe.is_dir():
                raise NotADirectoryError(
                    f"stabilized archive output filesystem probe is not a directory: {probe}"
                )
        if free_bytes is not None:
            self.fixed_free_bytes = _archive_free_bytes(
                self.archive_output_path.parent
                if self.archive_output_path is not None
                else Path.cwd(),
                free_bytes,
            )
        else:
            self.fixed_free_bytes = None

    def check(
        self,
        *,
        accepted_steps: int,
        stage: str,
        stage_kind: str,
        stage_details: Mapping[str, Any],
        retained_counts: Mapping[str, Any],
    ) -> dict[str, Any]:
        if stage_kind not in _ARCHIVE_BUDGET_STAGE_KINDS:
            raise ValueError(f"unknown stabilized archive budget stage: {stage_kind}")
        if not isinstance(stage_details, Mapping):
            raise TypeError("stabilized archive budget stage details must be a mapping")
        retained = _archive_counts(
            retained_counts, name="retained archive counts"
        )
        prospective = {
            "accepted_steps": _archive_count(accepted_steps, "accepted_steps")
        }
        prospective_delta = _archive_count_delta(retained, prospective)
        validated_details = _archive_stage_details(stage_details, stage_kind)
        validated_stage = _validate_archive_stage_label(
            stage, stage_kind, validated_details
        )
        if prospective_delta != _archive_stage_delta(stage_kind, validated_details):
            raise ValueError(
                "stabilized archive budget stage details do not match row delta"
            )
        estimate = estimate_stabilized_archive_bytes_for_counts(
            self.model.n_material,
            self.model.n_dofs,
            self.config,
            accepted_steps=prospective["accepted_steps"],
        )
        retained_estimate = estimate_stabilized_archive_bytes_for_counts(
            self.model.n_material,
            self.model.n_dofs,
            self.config,
            accepted_steps=retained["accepted_steps"],
        )
        available = self.fixed_free_bytes
        if available is None and self.archive_output_path is not None:
            available = _archive_free_bytes(
                self.archive_output_path.parent,
                None,
            )
        archive_exhausted = int(estimate["archive_bytes"]) > self.archive_budget_bytes
        filesystem_exhausted = (
            available is not None
            and int(estimate["required_free_bytes"]) > available
        )
        if archive_exhausted or filesystem_exhausted:
            exhaustion_kind = (
                "archive_budget" if archive_exhausted else "filesystem"
            )
            raise _StabilizedArchiveBudgetExceeded(
                stage,
                stage_kind,
                validated_details,
                exhaustion_kind,
                retained,
                prospective,
                estimate,
                retained_estimate,
                self.archive_budget_bytes,
                available,
            )
        return {
            **estimate,
            "archive_budget_bytes": self.archive_budget_bytes,
            "available_bytes": available,
            "stage": validated_stage,
            "stage_kind": stage_kind,
            "stage_details": validated_details,
            "retained_counts": retained,
            "prospective_counts": prospective,
            "prospective_delta": prospective_delta,
        }


@dataclass(frozen=True)
class StabilizedRelaxationTrial:
    """One explicit relaxation/backtracking candidate audit."""

    omega: float | None
    complete: bool
    hard_pass: bool
    globalization_pass: bool
    coupled_pass: bool
    selected: bool
    merit: float | None
    sufficient_decrease_limit: float | None
    failure_stage: str
    failure_reason: str
    candidate_diagnostics: EngineeringGripDiagnostics | None
    d_trial: np.ndarray | None
    q_after: np.ndarray | None
    d_trial_digest: str | None
    q_after_digest: str | None
    candidate_kind: str = "picard"
    # Historical accelerated rows intentionally retain ``None`` because their
    # raw proposal is not a line-search trial.  The distinct backtracking
    # policy records the actual alpha explicitly rather than overloading omega.
    trial_alpha: float | None = None
    candidate_fp_error: float | None = None
    candidate_fp_scale: float | None = None
    candidate_damage_residual_scale: float | None = None
    # The controller may replace an incomplete candidate diagnostic with a
    # stage-specific record after a hard rejection or candidate-map failure.
    # Retain only the already-computed scalar gate snapshot for the optional
    # residual-policy JSONL path; no numerical arrays are duplicated or added to
    # the lean v5 archive payload.
    candidate_hard_gate_attribution: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        omega = _optional_scalar(self.omega, "relaxation trial omega")
        if omega is not None and not 0.0 < omega <= 1.0:
            raise ValueError("relaxation trial omega must lie in (0, 1]")
        if self.candidate_kind not in ("picard", "picard_fallback", "anderson_accelerated"):
            raise ValueError(
                "relaxation trial candidate_kind is invalid: "
                f"{self.candidate_kind!r}"
            )
        if self.candidate_kind == "anderson_accelerated" and omega is not None:
            raise ValueError("Anderson accelerated trial cannot retain an omega")
        if self.candidate_kind != "anderson_accelerated" and omega is None:
            raise ValueError("Picard relaxation trial requires an omega")
        trial_alpha = _optional_scalar(self.trial_alpha, "relaxation trial alpha")
        if trial_alpha is not None and not 0.0 < trial_alpha <= 1.0:
            raise ValueError("relaxation trial alpha must lie in (0, 1]")
        if self.candidate_kind != "anderson_accelerated" and trial_alpha is not None:
            raise ValueError("Picard relaxation trial cannot retain trial_alpha")
        for name in ("failure_stage", "failure_reason"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"relaxation trial {name} must be a string")
        merit = _optional_scalar(self.merit, "relaxation trial merit")
        limit = _optional_scalar(
            self.sufficient_decrease_limit, "relaxation trial sufficient limit"
        )
        candidate_fp_error = _optional_scalar(
            self.candidate_fp_error, "relaxation trial candidate fixed-point error"
        )
        candidate_fp_scale = _optional_scalar(
            self.candidate_fp_scale, "relaxation trial candidate fixed-point scale"
        )
        candidate_residual_scale = _optional_scalar(
            self.candidate_damage_residual_scale,
            "relaxation trial candidate damage residual scale",
        )
        candidate = self.candidate_diagnostics
        if candidate is not None and not isinstance(
            candidate, EngineeringGripDiagnostics
        ):
            raise TypeError("relaxation trial candidate diagnostics have the wrong type")
        hard_gate_attribution = self.candidate_hard_gate_attribution
        if hard_gate_attribution is not None:
            if not isinstance(hard_gate_attribution, Mapping):
                raise TypeError(
                    "relaxation trial hard-gate attribution has the wrong type"
                )
            _validate_anderson_scalar_event(hard_gate_attribution)
        d_trial = _optional_f64(self.d_trial, "relaxation trial d_trial")
        q_after = _optional_f64(self.q_after, "relaxation trial q_after")
        if merit is not None and merit < 0.0:
            raise ValueError("relaxation trial merit must be non-negative")
        if limit is not None and limit < 0.0:
            raise ValueError("relaxation trial sufficient limit must be non-negative")
        for name, value in (
            ("candidate_fp_error", candidate_fp_error),
            ("candidate_fp_scale", candidate_fp_scale),
            ("candidate_damage_residual_scale", candidate_residual_scale),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.candidate_kind != "anderson_accelerated" and any(
            value is not None
            for value in (
                candidate_fp_error,
                candidate_fp_scale,
                candidate_residual_scale,
            )
        ):
            raise ValueError(
                "Picard relaxation trial retains Anderson candidate metrics"
            )
        if self.candidate_kind == "anderson_accelerated" and self.complete and any(
            value is None
            for value in (
                candidate_fp_error,
                candidate_fp_scale,
                candidate_residual_scale,
            )
        ):
            raise ValueError("complete Anderson trial is missing candidate metrics")
        if candidate is None:
            raise ValueError("relaxation trial is missing candidate diagnostics")
        if bool(candidate.complete_sweep) != bool(self.complete):
            raise ValueError("relaxation trial candidate completion is not paired")
        if d_trial is None or candidate.d_next is None:
            if d_trial is not None or candidate.d_next is not None:
                raise ValueError("relaxation trial d_trial availability is not paired")
        elif not np.array_equal(d_trial, candidate.d_next):
            raise ValueError("relaxation trial d_trial is not paired with diagnostics")
        if q_after is None or candidate.q_after is None:
            if q_after is not None or candidate.q_after is not None:
                raise ValueError("relaxation trial q_after availability is not paired")
        elif not np.array_equal(q_after, candidate.q_after):
            _array_close(
                q_after,
                candidate.q_after,
                "relaxation trial q_after",
                relative_limit=2.0e-10,
                absolute_limit=2.0e-12,
            )
        if self.complete:
            if not self.hard_pass:
                raise ValueError("complete relaxation trial must pass hard gates")
            if merit is None and self.candidate_kind != "picard_fallback":
                raise ValueError("complete relaxation trial is missing merit fields")
            if (
                d_trial is None
                or q_after is None
                or self.d_trial_digest is None
                or self.q_after_digest is None
            ):
                raise ValueError("complete relaxation trial is missing state digests")
            if self.d_trial_digest != engineering._damage_digest(d_trial):
                raise ValueError("complete relaxation trial damage digest is not paired")
            assert candidate.q_after is not None
            assert candidate.H_next is not None
            if self.q_after_digest != _trial_q_after_digest(
                candidate.target_u_bar,
                q_after,
                d_trial,
                candidate.H_next,
            ):
                raise ValueError("complete relaxation trial q-after digest is not paired")
        elif self.hard_pass or self.globalization_pass or self.coupled_pass or self.selected:
            raise ValueError("incomplete relaxation trial contains complete-only flags")
        if bool(self.selected) != bool(self.globalization_pass):
            raise ValueError("relaxation trial selection is not paired with globalization")
        if self.selected and not (
            self.complete and self.hard_pass and self.globalization_pass
        ):
            raise ValueError("selected relaxation trial did not pass globalization")
        for name in ("d_trial_digest", "q_after_digest"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, f"relaxation trial {name}")
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(self, "hard_pass", bool(self.hard_pass))
        object.__setattr__(self, "globalization_pass", bool(self.globalization_pass))
        object.__setattr__(self, "coupled_pass", bool(self.coupled_pass))
        object.__setattr__(self, "selected", bool(self.selected))
        object.__setattr__(self, "merit", merit)
        object.__setattr__(self, "sufficient_decrease_limit", limit)
        object.__setattr__(self, "candidate_diagnostics", candidate)
        object.__setattr__(self, "candidate_hard_gate_attribution", hard_gate_attribution)
        object.__setattr__(self, "d_trial", d_trial)
        object.__setattr__(self, "q_after", q_after)
        object.__setattr__(self, "candidate_kind", self.candidate_kind)
        object.__setattr__(self, "trial_alpha", trial_alpha)
        object.__setattr__(self, "candidate_fp_error", candidate_fp_error)
        object.__setattr__(self, "candidate_fp_scale", candidate_fp_scale)
        object.__setattr__(
            self,
            "candidate_damage_residual_scale",
            candidate_residual_scale,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "omega": self.omega,
            "complete": self.complete,
            "hard_pass": self.hard_pass,
            "globalization_pass": self.globalization_pass,
            "coupled_pass": self.coupled_pass,
            "selected": self.selected,
            "merit": self.merit,
            "sufficient_decrease_limit": self.sufficient_decrease_limit,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "candidate_diagnostics": (
                None
                if self.candidate_diagnostics is None
                else _compact_diagnostic_payload(self.candidate_diagnostics)
            ),
            # Keep null markers in JSON for explicit availability without
            # copying numerical trial vectors into the lean archive.
            "d_trial": None,
            "q_after": None,
            "d_trial_digest": self.d_trial_digest,
            "q_after_digest": self.q_after_digest,
            "candidate_kind": self.candidate_kind,
            "candidate_fp_error": self.candidate_fp_error,
            "candidate_fp_scale": self.candidate_fp_scale,
            "candidate_damage_residual_scale": self.candidate_damage_residual_scale,
        }
        # Historical trial payloads do not gain a null field merely because the
        # dataclass now understands the distinct backtracking policy. Every
        # accelerated row emitted by that policy carries a non-null alpha; the
        # compact v5 solver summary records the explicit null fallback.
        if self.trial_alpha is not None:
            payload["trial_alpha"] = self.trial_alpha
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        candidate_arrays: Mapping[str, np.ndarray] | None = None,
        trial_arrays: Mapping[str, np.ndarray] | None = None,
    ) -> "StabilizedRelaxationTrial":
        if not isinstance(payload, Mapping):
            raise ValueError("relaxation trial payload must be a mapping")
        required = {
            "omega",
            "complete",
            "hard_pass",
            "globalization_pass",
            "coupled_pass",
            "selected",
            "merit",
            "sufficient_decrease_limit",
            "failure_stage",
            "failure_reason",
            "candidate_diagnostics",
            "d_trial",
            "q_after",
            "d_trial_digest",
            "q_after_digest",
            "candidate_kind",
            "candidate_fp_error",
            "candidate_fp_scale",
            "candidate_damage_residual_scale",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"relaxation trial is missing fields: {missing}")
        optional = {"trial_alpha"}
        unexpected = sorted(set(payload) - required - optional)
        if unexpected:
            raise ValueError(f"relaxation trial contains unknown fields: {unexpected}")
        candidate_payload = payload["candidate_diagnostics"]
        candidate = (
            None
            if candidate_payload is None
            else _diagnostic_from_payload(
                candidate_payload,
                "relaxation trial candidate diagnostics",
                array_values=candidate_arrays,
            )
        )
        trial_d_trial = (
            None
            if trial_arrays is None
            else trial_arrays.get("d_trial")
        )
        trial_q_after = (
            None
            if trial_arrays is None
            else trial_arrays.get("q_after")
        )
        if payload["d_trial"] is not None or payload["q_after"] is not None:
            raise ValueError(
                "relaxation trial numerical arrays must be null in compact stabilized JSON"
            )
        return cls(
            omega=payload["omega"],
            complete=payload["complete"],
            hard_pass=payload["hard_pass"],
            globalization_pass=payload["globalization_pass"],
            coupled_pass=payload["coupled_pass"],
            selected=payload["selected"],
            merit=payload["merit"],
            sufficient_decrease_limit=payload["sufficient_decrease_limit"],
            failure_stage=payload["failure_stage"],
            failure_reason=payload["failure_reason"],
            candidate_diagnostics=candidate,
            d_trial=trial_d_trial,
            q_after=trial_q_after,
            d_trial_digest=payload["d_trial_digest"],
            q_after_digest=payload["q_after_digest"],
            candidate_kind=payload["candidate_kind"],
            trial_alpha=payload.get("trial_alpha"),
            candidate_fp_error=payload["candidate_fp_error"],
            candidate_fp_scale=payload["candidate_fp_scale"],
            candidate_damage_residual_scale=payload[
                "candidate_damage_residual_scale"
            ],
        )


@dataclass(frozen=True)
class StabilizedRelaxationSweepDiagnostics:
    """A complete or explicitly incomplete stabilized sweep."""

    base_diagnostics: EngineeringGripDiagnostics
    complete_sweep: bool
    passed: bool
    failure_stage: str
    failure_reason: str
    d_solved: np.ndarray | None
    d_trial: np.ndarray | None
    omega: float | None
    backtrack_count: int
    reference_merit: float | None
    merit: float | None
    sufficient_decrease_limit: float | None
    fixed_point_error: float | None
    fixed_point_scale: float | None
    fixed_point_pass: bool
    damage_equation_pass: bool
    globalization_pass: bool
    coupled_pass: bool
    trials: tuple[StabilizedRelaxationTrial, ...]
    anderson_history_size: int | None = None
    anderson_qr_rank: int | None = None
    anderson_coefficient_l2: float | None = None
    anderson_regularization_mu: float | None = None
    anderson_current_residual_l2: float | None = None
    anderson_regularization_lambda: float | None = None
    anderson_unregularized_ls_residual_l2: float | None = None
    anderson_selection: str | None = None
    trial_alpha: float | None = None
    anderson_fallback_reason: str | None = None
    anderson_restart_reason: str | None = None
    candidate_fp_error: float | None = None
    candidate_fp_scale: float | None = None
    candidate_damage_residual_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_diagnostics, EngineeringGripDiagnostics):
            raise TypeError("stabilized sweep base diagnostics have the wrong type")
        d_shape = self.base_diagnostics.damage_before.shape
        for name in ("d_solved", "d_trial"):
            array = _optional_f64(
                getattr(self, name), f"stabilized sweep {name}", d_shape
            )
            object.__setattr__(self, name, array)
        if self.d_trial is not None and self.base_diagnostics.d_next is not None:
            if not np.array_equal(self.d_trial, self.base_diagnostics.d_next):
                raise ValueError("stabilized d_trial is not paired with base d_next")
        omega = _optional_scalar(self.omega, "stabilized sweep omega")
        if omega is not None and not 0.0 < omega <= 1.0:
            raise ValueError("stabilized sweep omega must lie in (0, 1]")
        reference = _optional_scalar(self.reference_merit, "stabilized reference merit")
        merit = _optional_scalar(self.merit, "stabilized merit")
        limit = _optional_scalar(
            self.sufficient_decrease_limit, "stabilized sufficient limit"
        )
        fixed_error = _optional_scalar(
            self.fixed_point_error, "stabilized fixed-point error"
        )
        fixed_scale = _optional_scalar(
            self.fixed_point_scale, "stabilized fixed-point scale"
        )
        candidate_fp_error = _optional_scalar(
            self.candidate_fp_error, "stabilized candidate fixed-point error"
        )
        candidate_fp_scale = _optional_scalar(
            self.candidate_fp_scale, "stabilized candidate fixed-point scale"
        )
        candidate_residual_scale = _optional_scalar(
            self.candidate_damage_residual_scale,
            "stabilized candidate damage residual scale",
        )
        coefficient_norm = _optional_scalar(
            self.anderson_coefficient_l2,
            "stabilized Anderson coefficient l2",
        )
        regularization_mu = _optional_scalar(
            self.anderson_regularization_mu,
            "stabilized Anderson regularization mu",
        )
        current_residual_l2 = _optional_scalar(
            self.anderson_current_residual_l2,
            "stabilized Anderson current residual l2",
        )
        regularization_lambda = _optional_scalar(
            self.anderson_regularization_lambda,
            "stabilized Anderson regularization lambda",
        )
        unregularized_ls_residual_l2 = _optional_scalar(
            self.anderson_unregularized_ls_residual_l2,
            "stabilized Anderson unregularized least-squares residual l2",
        )
        for name, value in (
            ("reference_merit", reference),
            ("merit", merit),
            ("sufficient_decrease_limit", limit),
            ("fixed_point_error", fixed_error),
            ("fixed_point_scale", fixed_scale),
            ("candidate_fp_error", candidate_fp_error),
            ("candidate_fp_scale", candidate_fp_scale),
            ("candidate_damage_residual_scale", candidate_residual_scale),
            ("anderson_coefficient_l2", coefficient_norm),
            ("anderson_current_residual_l2", current_residual_l2),
            ("regularization_lambda", regularization_lambda),
            (
                "anderson_unregularized_ls_residual_l2",
                unregularized_ls_residual_l2,
            ),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if regularization_mu is not None and regularization_mu <= 0.0:
            raise ValueError("stabilized Anderson regularization mu must be positive")
        if regularization_lambda is not None and current_residual_l2 is None:
            raise ValueError(
                "stabilized Anderson lambda lacks its current residual l2"
            )
        if regularization_lambda is not None and regularization_mu is None:
            raise ValueError("stabilized Anderson lambda lacks its regularization mu")
        if (
            regularization_lambda is not None
            and regularization_mu is not None
            and current_residual_l2 is not None
            and not _anderson_scalar_relation_close(
                regularization_lambda,
                regularization_mu * current_residual_l2**2,
            )
        ):
            raise ValueError(
                "stabilized Anderson lambda is not paired with mu and current residual l2"
            )
        backtracks = engineering._integer(
            self.backtrack_count, "stabilized backtrack_count"
        )
        trials = tuple(self.trials)
        if any(not isinstance(trial, StabilizedRelaxationTrial) for trial in trials):
            raise TypeError("stabilized sweep trials contain a wrong type")
        if trials:
            alpha_trials = sum(trial.trial_alpha is not None for trial in trials)
            expected_backtracks = (
                max(0, alpha_trials - 1)
                if alpha_trials
                else len(trials) - 1
            )
            if backtracks != expected_backtracks:
                raise ValueError("stabilized sweep backtrack count is not paired")
            alpha_positions = [
                index
                for index, trial in enumerate(trials)
                if trial.trial_alpha is not None
            ]
            if alpha_trials and alpha_positions != list(range(alpha_trials)):
                raise ValueError(
                    "stabilized sweep alpha trials are not a canonical prefix"
                )
            for index in alpha_positions:
                expected_alpha = _anderson_backtracking_alpha_for_index(
                    index, "stabilized sweep trial alpha index"
                )
                if trials[index].trial_alpha != expected_alpha:
                    raise ValueError(
                        "stabilized sweep trial alpha is not paired with its index"
                    )
        history_size = _optional_nonnegative_integer(
            self.anderson_history_size, "Anderson history size"
        )
        qr_rank = _optional_nonnegative_integer(
            self.anderson_qr_rank, "Anderson QR rank"
        )
        trial_alpha = _optional_scalar(self.trial_alpha, "Anderson trial alpha")
        if trial_alpha is not None and not 0.0 < trial_alpha <= 1.0:
            raise ValueError("Anderson trial alpha must lie in (0, 1]")
        if trial_alpha is not None:
            expected_alpha = _anderson_backtracking_alpha_for_index(
                backtracks, "Anderson sweep backtrack_count"
            )
            if trial_alpha != expected_alpha:
                raise ValueError(
                    "Anderson trial alpha is not paired with backtrack_count"
                )
        selection = self.anderson_selection
        if selection is not None and selection not in _ANDERSON_SELECTIONS:
            raise ValueError("stabilized Anderson selection is invalid")
        for name in ("anderson_fallback_reason", "anderson_restart_reason"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")
        if selection == "accelerated" and self.anderson_fallback_reason is not None:
            raise ValueError("accelerated Anderson sweep retains a fallback reason")
        if selection == "picard_fallback" and self.anderson_fallback_reason is None:
            raise ValueError("Anderson Picard fallback lacks a fallback reason")
        if history_size is not None and qr_rank is not None and qr_rank > history_size:
            raise ValueError("Anderson QR rank exceeds the history size")
        if self.anderson_selection != "accelerated" and any(
            value is not None
            for value in (
                candidate_fp_error,
                candidate_fp_scale,
                candidate_residual_scale,
            )
        ):
            raise ValueError(
                "non-accelerated stabilized sweep retains Anderson candidate metrics"
            )
        selected = [trial for trial in trials if trial.selected]
        if len(selected) > 1:
            raise ValueError("stabilized sweep selects more than one trial")
        selected_trial = selected[0] if selected else None
        if self.complete_sweep:
            if not self.base_diagnostics.complete_sweep:
                raise ValueError("complete stabilized sweep needs complete base diagnostics")
            if self.d_solved is None or self.d_trial is None:
                raise ValueError("complete stabilized sweep is missing relaxation fields")
            if reference is None or fixed_error is None:
                raise ValueError("complete stabilized sweep is missing merit fields")
            if selected_trial is None:
                raise ValueError("complete stabilized sweep has no selected trial")
            if merit is None and selected_trial.candidate_kind != "picard_fallback":
                raise ValueError("complete accelerated sweep is missing selection merit")
            if selected_trial.omega != omega:
                raise ValueError("selected trial omega is not paired with sweep omega")
            if selected_trial.candidate_kind == "anderson_accelerated":
                if self.anderson_selection != "accelerated":
                    raise ValueError("accelerated trial is missing Anderson selection")
                if self.sufficient_decrease_limit is None:
                    raise ValueError("accelerated sweep is missing sufficient limit")
                if any(
                    value is None
                    for value in (
                        candidate_fp_error,
                        candidate_fp_scale,
                        candidate_residual_scale,
                    )
                ):
                    raise ValueError(
                        "accelerated sweep is missing candidate metrics"
                    )
                if (
                    candidate_fp_error != selected_trial.candidate_fp_error
                    or candidate_fp_scale != selected_trial.candidate_fp_scale
                    or candidate_residual_scale
                    != selected_trial.candidate_damage_residual_scale
                ):
                        raise ValueError(
                            "accelerated sweep candidate metrics are not paired"
                        )
                if trial_alpha != selected_trial.trial_alpha:
                    raise ValueError("accelerated sweep trial alpha is not paired")
            elif selected_trial.candidate_kind == "picard_fallback":
                if self.anderson_selection != "picard_fallback":
                    raise ValueError("Picard fallback is missing Anderson selection")
                if self.sufficient_decrease_limit is not None:
                    raise ValueError("Picard fallback retains a sufficient limit")
                if trial_alpha is not None:
                    raise ValueError("Picard fallback retains trial alpha")
        elif trial_alpha is not None:
            if not trials or trials[-1].trial_alpha != trial_alpha:
                raise ValueError("unselected stabilized sweep trial alpha is not paired")
        if self.passed and not (self.complete_sweep and self.coupled_pass and self.globalization_pass):
            raise ValueError("passed stabilized sweep does not pass all stabilized gates")
        if not self.complete_sweep and self.passed:
            raise ValueError("incomplete stabilized sweep cannot pass")
        if not isinstance(self.failure_stage, str) or not isinstance(
            self.failure_reason, str
        ):
            raise TypeError("stabilized sweep failure fields must be strings")
        if self.complete_sweep and not self.passed and not self.failure_reason:
            raise ValueError("failed complete stabilized sweep needs a failure reason")
        if not self.complete_sweep and (not self.failure_stage or not self.failure_reason):
            raise ValueError("incomplete stabilized sweep needs stage and reason")
        object.__setattr__(self, "complete_sweep", bool(self.complete_sweep))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "backtrack_count", backtracks)
        object.__setattr__(self, "reference_merit", reference)
        object.__setattr__(self, "merit", merit)
        object.__setattr__(self, "sufficient_decrease_limit", limit)
        object.__setattr__(self, "fixed_point_error", fixed_error)
        object.__setattr__(self, "fixed_point_scale", fixed_scale)
        object.__setattr__(self, "anderson_history_size", history_size)
        object.__setattr__(self, "anderson_qr_rank", qr_rank)
        object.__setattr__(self, "anderson_coefficient_l2", coefficient_norm)
        object.__setattr__(self, "anderson_regularization_mu", regularization_mu)
        object.__setattr__(self, "anderson_current_residual_l2", current_residual_l2)
        object.__setattr__(self, "anderson_regularization_lambda", regularization_lambda)
        object.__setattr__(
            self,
            "anderson_unregularized_ls_residual_l2",
            unregularized_ls_residual_l2,
        )
        object.__setattr__(self, "anderson_selection", selection)
        object.__setattr__(self, "trial_alpha", trial_alpha)
        object.__setattr__(self, "candidate_fp_error", candidate_fp_error)
        object.__setattr__(self, "candidate_fp_scale", candidate_fp_scale)
        object.__setattr__(
            self,
            "candidate_damage_residual_scale",
            candidate_residual_scale,
        )
        for name in (
            "fixed_point_pass",
            "damage_equation_pass",
            "globalization_pass",
            "coupled_pass",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(self, "trials", trials)

    def __getattr__(self, name: str) -> Any:
        # Keep base diagnostic field names available to workflow/report code,
        # while the explicit stabilized fields above remain authoritative.
        base = object.__getattribute__(self, "base_diagnostics")
        return getattr(base, name)

    @property
    def d_next(self) -> np.ndarray | None:
        return self.d_trial

    @property
    def available_fields(self) -> tuple[str, ...]:
        fields = set(self.base_diagnostics.available_fields)
        if self.d_solved is not None:
            fields.add("d_solved")
        if self.d_trial is not None:
            fields.add("d_trial")
        return tuple(sorted(fields))

    @property
    def gate_passes(self) -> dict[str, bool]:
        return {
            **self.base_diagnostics.gate_passes,
            "damage_fixed_point": self.fixed_point_pass,
            "damage_equation": self.damage_equation_pass,
            "globalization": self.globalization_pass,
            "coupled": self.coupled_pass,
        }

    @property
    def sweep_index(self) -> int:
        return self.base_diagnostics.sweep_index

    @property
    def target_u_bar(self) -> float:
        return self.base_diagnostics.target_u_bar

    def payload(self) -> dict[str, Any]:
        payload = _compact_diagnostic_payload(self.base_diagnostics)
        payload.update(
            {
                "stabilized_diagnostic": True,
                "sweep_index": self.sweep_index,
                "target_u_bar": self.target_u_bar,
                "complete_sweep": self.complete_sweep,
                "passed": self.passed,
                "failure_stage": self.failure_stage,
                "failure_reason": self.failure_reason,
                "base_failure_stage": self.base_diagnostics.failure_stage,
                "base_failure_reason": self.base_diagnostics.failure_reason,
                "base_passed": self.base_diagnostics.passed,
                "base_complete_sweep": self.base_diagnostics.complete_sweep,
                "base_gate_passes": self.base_diagnostics.gate_passes,
                "base_available_fields": list(self.base_diagnostics.available_fields),
                "gate_passes": self.gate_passes,
                "available_fields": list(self.available_fields),
                # These arrays are stored once in the typed sweep arrays.
                "d_solved": None,
                "d_trial": None,
                "omega": self.omega,
                "backtrack_count": self.backtrack_count,
                "reference_merit": self.reference_merit,
                "merit": self.merit,
                "sufficient_decrease_limit": self.sufficient_decrease_limit,
                "fixed_point_error": self.fixed_point_error,
                "fixed_point_scale": self.fixed_point_scale,
                "fixed_point_pass": self.fixed_point_pass,
                "damage_equation_pass": self.damage_equation_pass,
                "globalization_pass": self.globalization_pass,
                "coupled_pass": self.coupled_pass,
                 "anderson_history_size": self.anderson_history_size,
                 "anderson_qr_rank": self.anderson_qr_rank,
                "anderson_selection": self.anderson_selection,
                "anderson_fallback_reason": self.anderson_fallback_reason,
                "anderson_restart_reason": self.anderson_restart_reason,
                "candidate_fp_error": self.candidate_fp_error,
                "candidate_fp_scale": self.candidate_fp_scale,
                 "candidate_damage_residual_scale": self.candidate_damage_residual_scale,
                 "relaxation_trials": [trial.payload() for trial in self.trials],
             }
         )
        if self.anderson_regularization_mu is not None:
            payload.update(
                {
                    "anderson_regularization_mu": self.anderson_regularization_mu,
                    "anderson_current_residual_l2": self.anderson_current_residual_l2,
                    "anderson_regularization_lambda": self.anderson_regularization_lambda,
                    "anderson_unregularized_ls_residual_l2": (
                        self.anderson_unregularized_ls_residual_l2
                    ),
                    "anderson_coefficient_l2": self.anderson_coefficient_l2,
                }
            )
        if self.trial_alpha is not None:
            payload["trial_alpha"] = self.trial_alpha
        return payload


@dataclass(frozen=True)
class StabilizedEngineeringGripAcceptedStep:
    """One committed stabilized load increment.

    Strict steps pass every stabilized gate.  Warning steps are committed only
    by the separate final-complete-sweep policy and retain the exact failed
    non-hard gate list and raw safety diagnostics.
    """

    load_start: float
    load_target: float
    increment: float
    state: EngineeringGripState
    diagnostics: StabilizedRelaxationSweepDiagnostics
    stagger_iterations: int
    cutback_count: int
    acceptance_status: str
    bypassed_gates: tuple[str, ...] = ()
    warning_range_excess: float | None = None
    warning_healing_magnitude: float | None = None
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        start = _scalar(self.load_start, "stabilized accepted load_start")
        target = _scalar(self.load_target, "stabilized accepted load_target")
        increment = _scalar(self.increment, "stabilized accepted increment")
        if target <= start or target - start != increment:
            raise ValueError("stabilized accepted load interval is invalid")
        if not isinstance(self.state, EngineeringGripState):
            raise TypeError("stabilized accepted state has the wrong type")
        if not isinstance(self.diagnostics, StabilizedRelaxationSweepDiagnostics):
            raise TypeError("stabilized accepted diagnostics have the wrong type")
        if not self.diagnostics.passed and self.acceptance_status != STABILIZED_WARNING_ACCEPTANCE_STATUS:
            raise ValueError("stabilized accepted step needs a passing diagnostic")
        mode = _validate_stabilized_acceptance_mode(self.acceptance_mode)
        bypassed = tuple(self.bypassed_gates)
        if any(
            not isinstance(name, str) or name not in STABILIZED_WARNING_GATE_NAMES
            for name in bypassed
        ) or len(set(bypassed)) != len(bypassed):
            raise ValueError("stabilized accepted bypassed gates are invalid")
        canonical_bypassed = tuple(
            name for name in STABILIZED_WARNING_GATE_NAMES if name in bypassed
        )
        if bypassed != canonical_bypassed:
            raise ValueError("stabilized accepted bypassed gates are not canonical")
        if mode == STABILIZED_ACCEPTANCE_MODE:
            if self.acceptance_status != STABILIZED_ACCEPTANCE_STATUS:
                raise ValueError("strict stabilized accepted status is invalid")
            if (
                bypassed
                or self.warning_range_excess is not None
                or self.warning_healing_magnitude is not None
            ):
                raise ValueError("strict stabilized steps cannot retain warning fields")
        elif self.acceptance_status == STABILIZED_ACCEPTANCE_STATUS:
            if not self.diagnostics.passed or bypassed:
                raise ValueError("strict-passed warning-mode step has warning fields")
            if (
                self.warning_range_excess is not None
                or self.warning_healing_magnitude is not None
            ):
                raise ValueError("strict-passed warning-mode step cannot retain warning fields")
        elif self.acceptance_status == STABILIZED_WARNING_ACCEPTANCE_STATUS:
            if self.diagnostics.passed:
                raise ValueError("warning-accepted step must retain passed=False")
            if not bypassed:
                raise ValueError("warning-accepted step needs failed gates")
            expected = _stabilized_warning_failed_gates(self.diagnostics)
            if bypassed != expected:
                raise ValueError("warning-accepted bypassed gates are not exact")
            if self.warning_range_excess is None or self.warning_healing_magnitude is None:
                raise ValueError("warning-accepted step needs raw safety diagnostics")
            expected_range, expected_healing = _stabilized_warning_excesses(
                self.diagnostics
            )
            if (
                self.warning_range_excess != expected_range
                or self.warning_healing_magnitude != expected_healing
            ):
                raise ValueError("warning-accepted raw safety diagnostics are not paired")
        else:
            raise ValueError("stabilized accepted status is invalid")
        if self.state.u_bar != target:
            raise ValueError("stabilized accepted state load is not paired")
        if self.diagnostics.q_after is None or not np.array_equal(
            self.state.q, self.diagnostics.q_after
        ):
            raise ValueError("stabilized accepted state displacement is not paired")
        if self.diagnostics.d_trial is None or self.diagnostics.H_next is None:
            raise ValueError("stabilized accepted state fields are missing")
        if not np.array_equal(self.state.d, self.diagnostics.d_trial) or not np.array_equal(
            self.state.H, self.diagnostics.H_next
        ):
            raise ValueError("stabilized accepted damage/history fields are not paired")
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "increment", increment)
        object.__setattr__(
            self,
            "stagger_iterations",
            engineering._integer(
                self.stagger_iterations, "stabilized stagger_iterations", positive=True
            ),
        )
        object.__setattr__(
            self, "cutback_count", engineering._integer(self.cutback_count, "stabilized cutback_count")
        )
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "bypassed_gates", canonical_bypassed)
        object.__setattr__(
            self,
            "warning_range_excess",
            None
            if self.warning_range_excess is None
            else _scalar(
                self.warning_range_excess,
                "stabilized warning range excess",
            ),
        )
        object.__setattr__(
            self,
            "warning_healing_magnitude",
            None
            if self.warning_healing_magnitude is None
            else _scalar(
                self.warning_healing_magnitude,
                "stabilized warning healing magnitude",
            ),
        )

    @property
    def u_bar(self) -> float:
        return self.load_target


@dataclass(frozen=True)
class StabilizedEngineeringGripAttempt:
    """One load target, including relaxation trials and rollback identity."""

    cutback_count: int
    load_start: float
    load_target: float
    accepted: bool
    acceptance_status: str | None
    failure_stage: str | None
    failure_reason: str | None
    exception_type: str | None
    committed_before_digest: str
    committed_after_digest: str
    sweeps: tuple[StabilizedRelaxationSweepDiagnostics, ...]
    diagnostics: StabilizedRelaxationSweepDiagnostics
    bypassed_gates: tuple[str, ...] = ()
    warning_range_excess: float | None = None
    warning_healing_magnitude: float | None = None
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        start = _scalar(self.load_start, "stabilized attempt load_start")
        target = _scalar(self.load_target, "stabilized attempt load_target")
        if target <= start:
            raise ValueError("stabilized attempt target must exceed load_start")
        before = _digest(self.committed_before_digest, "stabilized attempt before digest")
        after = _digest(self.committed_after_digest, "stabilized attempt after digest")
        sweeps = tuple(self.sweeps)
        if not sweeps or sweeps[-1] is not self.diagnostics:
            raise ValueError("stabilized attempt must retain its final sweep object")
        if any(not isinstance(item, StabilizedRelaxationSweepDiagnostics) for item in sweeps):
            raise TypeError("stabilized attempt sweeps contain a wrong type")
        mode = _validate_stabilized_acceptance_mode(self.acceptance_mode)
        bypassed = tuple(self.bypassed_gates)
        if any(
            not isinstance(name, str) or name not in STABILIZED_WARNING_GATE_NAMES
            for name in bypassed
        ) or len(set(bypassed)) != len(bypassed):
            raise ValueError("stabilized attempt bypassed gates are invalid")
        canonical_bypassed = tuple(
            name for name in STABILIZED_WARNING_GATE_NAMES if name in bypassed
        )
        if bypassed != canonical_bypassed:
            raise ValueError("stabilized attempt bypassed gates are not canonical")
        for index, sweep in enumerate(sweeps):
            if sweep.sweep_index != index or sweep.target_u_bar != target:
                raise ValueError("stabilized attempt sweep pairing is invalid")
            if not sweep.complete_sweep and index != len(sweeps) - 1:
                raise ValueError("incomplete stabilized sweep must be final")
            if sweep.passed and index != len(sweeps) - 1:
                raise ValueError("passing stabilized sweep must be final")
        if self.accepted:
            if self.failure_stage or self.failure_reason or self.exception_type:
                raise ValueError("accepted stabilized attempt retains failure fields")
            if before == after:
                raise ValueError("accepted stabilized attempt did not change state digest")
            if self.acceptance_status == STABILIZED_ACCEPTANCE_STATUS:
                if not self.diagnostics.passed or canonical_bypassed:
                    raise ValueError(
                        "strict-passed stabilized attempt has warning fields"
                    )
                if (
                    self.warning_range_excess is not None
                    or self.warning_healing_magnitude is not None
                ):
                    raise ValueError(
                        "strict-passed stabilized attempt has raw warning fields"
                    )
            elif self.acceptance_status == STABILIZED_WARNING_ACCEPTANCE_STATUS:
                if mode != STABILIZED_WARNING_ACCEPTANCE_MODE:
                    raise ValueError(
                        "warning-accepted stabilized attempt has strict mode"
                    )
                if self.diagnostics.passed or not canonical_bypassed:
                    raise ValueError(
                        "warning-accepted stabilized attempt has invalid gates"
                    )
                if canonical_bypassed != _stabilized_warning_failed_gates(
                    self.diagnostics
                ):
                    raise ValueError(
                        "warning-accepted stabilized attempt gates are not exact"
                    )
                if (
                    self.warning_range_excess is None
                    or self.warning_healing_magnitude is None
                ):
                    raise ValueError(
                        "warning-accepted stabilized attempt lacks raw excesses"
                    )
                expected_range, expected_healing = _stabilized_warning_excesses(
                    self.diagnostics
                )
                if (
                    self.warning_range_excess != expected_range
                    or self.warning_healing_magnitude != expected_healing
                ):
                    raise ValueError(
                        "warning-accepted stabilized attempt excesses are not paired"
                    )
            else:
                raise ValueError("accepted stabilized attempt status is invalid")
        else:
            if self.acceptance_status is not None:
                raise ValueError("rejected stabilized attempt has an acceptance status")
            if not self.failure_stage or not self.failure_reason:
                raise ValueError("rejected stabilized attempt needs failure fields")
            if self.exception_type != StabilizedEngineeringGripSweepFailure.__name__:
                raise ValueError("rejected stabilized attempt exception type is invalid")
            if self.diagnostics.passed:
                raise ValueError("rejected stabilized attempt diagnostic passes")
            if before != after:
                raise ValueError("rejected stabilized attempt changed committed state")
            if canonical_bypassed:
                raise ValueError("rejected stabilized attempt cannot bypass gates")
            if (
                self.warning_range_excess is not None
                or self.warning_healing_magnitude is not None
            ):
                raise ValueError("rejected stabilized attempt cannot retain warning fields")
        object.__setattr__(self, "cutback_count", engineering._integer(self.cutback_count, "stabilized attempt cutback_count"))
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "committed_before_digest", before)
        object.__setattr__(self, "committed_after_digest", after)
        object.__setattr__(self, "sweeps", sweeps)
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "bypassed_gates", canonical_bypassed)
        object.__setattr__(
            self,
            "warning_range_excess",
            None
            if self.warning_range_excess is None
            else _scalar(
                self.warning_range_excess,
                "stabilized attempt warning range excess",
            ),
        )
        object.__setattr__(
            self,
            "warning_healing_magnitude",
            None
            if self.warning_healing_magnitude is None
            else _scalar(
                self.warning_healing_magnitude,
                "stabilized attempt warning healing magnitude",
            ),
        )


class StabilizedEngineeringGripSweepFailure(RuntimeError):
    """Typed numerical failure eligible for load cutback."""

    def __init__(self, stage: str, message: str, diagnostics: StabilizedRelaxationSweepDiagnostics):
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


class StabilizedEngineeringGripContinuationFailure(RuntimeError):
    """Typed minimum-increment/cutback failure with immutable audit records."""

    def __init__(
        self,
        message: str,
        last_committed_state: EngineeringGripState,
        attempts: tuple[StabilizedEngineeringGripAttempt, ...],
        work_check: EngineeringGripWorkCheck,
        *,
        failure_code: str = STABILIZED_FAILURE_CODE,
        acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    ):
        super().__init__(message)
        if not isinstance(failure_code, str) or not failure_code:
            raise ValueError("stabilized continuation failure_code must be non-empty")
        self.last_committed_state = last_committed_state
        self.attempts = attempts
        self.work_check = work_check
        self.failure_code = failure_code
        self.acceptance_mode = _validate_stabilized_acceptance_mode(acceptance_mode)


class StabilizedEngineeringGripArchiveBudgetFailure(
    StabilizedEngineeringGripContinuationFailure
):
    """Hard stop raised before the next archive row would exceed its budget."""

    def __init__(
        self,
        message: str,
        last_committed_state: EngineeringGripState,
        attempts: tuple[StabilizedEngineeringGripAttempt, ...],
        work_check: EngineeringGripWorkCheck,
        *,
        stage: str,
        stage_kind: str,
        stage_details: Mapping[str, Any],
        exhaustion_kind: str,
        retained_counts: Mapping[str, Any],
        prospective_counts: Mapping[str, Any],
        retained_estimate: Mapping[str, Any],
        estimate: Mapping[str, Any],
        archive_budget_bytes: int,
        available_bytes: int | None,
        acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    ) -> None:
        super().__init__(
            message,
            last_committed_state,
            attempts,
            work_check,
            failure_code=STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE,
            acceptance_mode=acceptance_mode,
        )
        if not isinstance(stage, str) or not stage:
            raise ValueError("stabilized archive budget failure stage must be non-empty")
        if stage_kind not in _ARCHIVE_BUDGET_STAGE_KINDS:
            raise ValueError(f"unknown stabilized archive budget stage: {stage_kind}")
        if not isinstance(stage_details, Mapping):
            raise TypeError("stabilized archive budget stage details must be a mapping")
        if exhaustion_kind not in _ARCHIVE_BUDGET_EXHAUSTION_KINDS:
            raise ValueError(
                f"unknown stabilized archive budget exhaustion: {exhaustion_kind}"
            )
        self.stage_kind = stage_kind
        self.stage_details = _archive_stage_details(stage_details, stage_kind)
        self.stage = _validate_archive_stage_label(
            stage, stage_kind, self.stage_details
        )
        self.exhaustion_kind = exhaustion_kind
        self.retained_counts = _archive_counts(
            retained_counts, name="retained archive counts"
        )
        self.prospective_counts = _archive_counts(
            prospective_counts, name="prospective archive counts"
        )
        self.prospective_delta = _archive_count_delta(
            self.retained_counts, self.prospective_counts
        )
        if self.prospective_delta != _archive_stage_delta(
            stage_kind, self.stage_details
        ):
            raise ValueError(
                "stabilized archive budget stage details do not match row delta"
            )
        self.retained_estimate = dict(retained_estimate)
        self.estimate = dict(estimate)
        self.archive_budget_bytes = _validate_archive_budget(archive_budget_bytes)
        self.available_bytes = available_bytes


@dataclass(frozen=True)
class StabilizedEngineeringGripResult:
    """Experimental result; never promoted to validated evidence."""

    model: EngineeringGripModel
    config: StabilizedEngineeringGripConfig
    work_check: EngineeringGripWorkCheck
    initial_state: EngineeringGripState
    states: tuple[EngineeringGripState, ...]
    accepted_steps: tuple[StabilizedEngineeringGripAcceptedStep, ...]
    attempts: tuple[StabilizedEngineeringGripAttempt, ...]
    completion_status: str = STABILIZED_COMPLETION_STATUS
    _acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "accepted_steps", tuple(self.accepted_steps))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def acceptance_mode(self) -> str:
        return _validate_stabilized_acceptance_mode(self._acceptance_mode)

    @property
    def acceptance_status(self) -> str:
        return _stabilized_contract(
            self.acceptance_mode,
            failure=False,
            schedule=self.config.load_schedule is not None,
        )[
            "acceptance_status"
        ]

    @property
    def validated(self) -> bool:
        return False

    @property
    def experimental(self) -> bool:
        return True

    @property
    def final_state(self) -> EngineeringGripState:
        return self.states[-1]

    def validate(self) -> None:
        if not isinstance(self.model, EngineeringGripModel) or not isinstance(
            self.config, StabilizedEngineeringGripConfig
        ):
            raise TypeError("stabilized result model/config types are invalid")
        if not isinstance(self.work_check, EngineeringGripWorkCheck):
            raise TypeError("stabilized result work_check has the wrong type")
        engineering._validate_work_check_against_model_config(
            self.work_check, self.model, self.config.base
        )
        if not self.work_check.passed:
            raise ValueError("stabilized result work check did not pass")
        mode = self.acceptance_mode
        schedule = self.config.load_schedule
        expected_contract = _stabilized_contract(
            mode, failure=False, schedule=schedule is not None
        )
        if self.completion_status != expected_contract["completion_status"]:
            raise ValueError("stabilized result completion status is invalid")
        states = tuple(self.states)
        steps = tuple(self.accepted_steps)
        attempts = tuple(self.attempts)
        if not states or states[0] is not self.initial_state:
            raise ValueError("stabilized result must begin with the exact initial state")
        if len(states) != len(steps) + 1:
            raise ValueError("stabilized state/step counts are inconsistent")
        initial = states[0]
        if (
            initial.u_bar != 0.0
            or not np.array_equal(initial.q, np.zeros_like(initial.q))
            or not np.array_equal(initial.d, np.zeros_like(initial.d))
            or not np.array_equal(initial.H, np.zeros_like(initial.H))
        ):
            raise ValueError("stabilized initial state must be exact zero")
        if initial.q.shape != (self.model.n_dofs,) or initial.d.shape != (self.model.n_material,):
            raise ValueError("stabilized initial state shape is inconsistent")
        if states[-1].u_bar != self.config.base.load_final:
            raise ValueError("stabilized result did not reach exact load_final")
        if schedule is not None:
            expected_loads = np.concatenate(
                (
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray(schedule.targets, dtype=np.float64),
                )
            )
            actual_loads = np.asarray([state.u_bar for state in states], dtype=np.float64)
            if not np.array_equal(actual_loads, expected_loads):
                raise ValueError(
                    "scheduled stabilized result loads do not equal the exact target sequence"
                )
            if len(attempts) != schedule.target_count or len(steps) != schedule.target_count:
                raise ValueError(
                    "scheduled stabilized result must retain exactly one accepted attempt per target"
                )
        for state in states:
            if state.q.shape != (self.model.n_dofs,) or state.d.shape != (self.model.n_material,):
                raise ValueError("stabilized state shape does not match model")
            if not np.all(np.isfinite(state.q)) or not np.all(np.isfinite(state.d)):
                raise ValueError("stabilized committed state contains non-finite fields")
            if np.any(state.H < 0.0) or not np.all(np.isfinite(state.H)):
                raise ValueError("stabilized committed history is invalid")
            engineering._check_exact_constraints(
                state.q, self.model.constraints, state.u_bar, "stabilized committed state"
            )
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, StabilizedEngineeringGripAcceptedStep):
                raise TypeError("stabilized accepted steps contain a wrong type")
            if step.state is not states[index]:
                raise ValueError("stabilized accepted step/state object pairing is invalid")
            if step.load_start != states[index - 1].u_bar or step.load_target != states[index].u_bar:
                raise ValueError("stabilized accepted load/state pairing is invalid")
            if schedule is not None and step.cutback_count != 0:
                raise ValueError("scheduled stabilized accepted step contains a cutback")
            diag = step.diagnostics
            if step.acceptance_mode != mode:
                raise ValueError("stabilized accepted step mode is not paired")
            if (
                (step.acceptance_status == STABILIZED_ACCEPTANCE_STATUS and not diag.passed)
                or diag.d_trial is None
                or diag.d_solved is None
            ):
                raise ValueError("stabilized accepted step lacks a valid fixed-point audit")
            if (
                mode == STABILIZED_ACCEPTANCE_MODE
                and _is_anderson_policy(self.config.relaxation_policy)
                and (not diag.damage_range_pass or not diag.healing_pass)
            ):
                raise ValueError(
                    "Anderson accepted step lacks strict range/healing gates"
                )
            selected_trials = [trial for trial in diag.trials if trial.selected]
            if len(selected_trials) != 1:
                raise ValueError("stabilized accepted step has an invalid selected trial")
            selected_trial = selected_trials[0]
            if selected_trial.candidate_kind == "anderson_accelerated":
                if selected_trial.d_trial is None or diag.d_trial is None:
                    raise ValueError("accelerated stabilized damage field is missing")
                if not np.array_equal(selected_trial.d_trial, diag.d_trial):
                    raise ValueError("accelerated stabilized damage field is forged")
            else:
                if diag.omega is None:
                    raise ValueError("Picard stabilized damage field is missing omega")
                expected_trial = diag.d_iter + diag.omega * (
                    diag.d_solved - diag.d_iter
                )
                if not np.array_equal(expected_trial, diag.d_trial):
                    raise ValueError("stabilized relaxed damage field is forged")
        if not attempts:
            raise ValueError("stabilized result must retain attempt audits")
        current = 0
        accepted_index = 0
        for attempt in attempts:
            if attempt.acceptance_mode != mode:
                raise ValueError("stabilized attempt mode is not paired")
            if schedule is not None and (
                attempt.cutback_count != 0 or not attempt.accepted
            ):
                raise ValueError(
                    "scheduled stabilized result contains a retry or rejected attempt"
                )
            if attempt.load_start != states[current].u_bar:
                raise ValueError("stabilized attempt load start is not paired")
            if attempt.committed_before_digest != states[current].digest:
                raise ValueError("stabilized attempt before digest is not paired")
            if attempt.accepted:
                if accepted_index >= len(steps) or attempt.diagnostics is not steps[accepted_index].diagnostics:
                    raise ValueError("stabilized accepted attempt is not paired")
                current += 1
                if attempt.committed_after_digest != states[current].digest:
                    raise ValueError("stabilized accepted attempt after digest is not paired")
                accepted_index += 1
            elif attempt.committed_after_digest != states[current].digest:
                raise ValueError("stabilized rejected attempt changed committed state")
        if current != len(states) - 1 or accepted_index != len(steps):
            raise ValueError("stabilized attempt audit does not reach final state")
        object.__setattr__(self, "_acceptance_mode", mode)


def _normalized(error: float, gate: GateTolerance, scale: float, name: str) -> float:
    limit = gate.limit(scale)
    if limit <= 0.0:
        raise ValueError(f"{name} has a zero merit limit")
    value = abs(float(error)) / limit
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} merit ratio is non-finite")
    return value


def _reference_merit_components(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    d_iter: np.ndarray,
    d_solved: np.ndarray,
    H_used: np.ndarray,
) -> dict[str, float]:
    base = config.base
    rho_iter, _, _, _, residual_scale = engineering._damage_residuals(
        model, d_iter, H_used, H_used
    )
    fixed_error = float(np.max(np.abs(d_solved - d_iter)))
    fixed_scale = max(float(np.max(np.abs(d_solved))), float(np.max(np.abs(d_iter))))
    components = {
        "reference_fixed_point": _normalized(
            fixed_error, base.damage, fixed_scale, "reference fixed-point"
        ),
        "reference_damage_residual": _normalized(
            float(np.max(np.abs(rho_iter))),
            base.damage_residual,
            residual_scale,
            "reference damage residual",
        ),
    }
    if not all(np.isfinite(value) for value in components.values()):
        raise FloatingPointError("reference merit component is non-finite")
    return components


def _reference_merit(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    d_iter: np.ndarray,
    d_solved: np.ndarray,
    H_used: np.ndarray,
) -> float:
    components = _reference_merit_components(
        model, config, d_iter, d_solved, H_used
    )
    merit = max(components.values())
    if not np.isfinite(merit):
        raise FloatingPointError("reference merit is non-finite")
    return float(merit)


def _candidate_merit_components(
    diagnostics: EngineeringGripDiagnostics,
    fixed_point_error: float,
    fixed_point_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> dict[str, float]:
    base = config.base
    fields = (
        (
            "material_displacement",
            diagnostics.e_u_material,
            base.displacement,
            diagnostics.displacement_scale,
            "material displacement",
        ),
        (
            "union_displacement",
            diagnostics.e_u_union,
            base.displacement,
            diagnostics.displacement_scale,
            "union displacement",
        ),
        ("damage_transition", diagnostics.e_damage, base.damage, diagnostics.damage_scale, "damage"),
        ("history_transition", diagnostics.e_history, base.history, diagnostics.history_scale, "history"),
        ("reaction_transition", diagnostics.e_reaction, base.reaction, diagnostics.reaction_scale, "reaction"),
        (
            "used_damage_residual",
            diagnostics.rho_used_inf,
            base.damage_residual,
            diagnostics.damage_residual_scale,
            "used damage residual",
        ),
        (
            "next_damage_residual",
            diagnostics.rho_next_inf,
            base.damage_residual,
            diagnostics.damage_residual_scale,
            "next damage residual",
        ),
        ("force", diagnostics.force_error, base.force, diagnostics.force_scale, "force"),
        ("moment", diagnostics.moment_error, base.moment, diagnostics.moment_scale, "moment"),
    )
    components = {
        key: _normalized(float(error), gate, float(scale), name)
        for key, error, gate, scale, name in fields
        if error is not None and scale is not None
    }
    components["fixed_point"] = _normalized(
        fixed_point_error, base.damage, fixed_point_scale, "fixed-point"
    )
    if not all(np.isfinite(value) for value in components.values()):
        raise FloatingPointError("candidate merit component is non-finite")
    return components


def _candidate_merit(
    diagnostics: EngineeringGripDiagnostics,
    fixed_point_error: float,
    fixed_point_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> float:
    components = _candidate_merit_components(
        diagnostics, fixed_point_error, fixed_point_scale, config
    )
    merit = max(components.values(), default=0.0)
    if not np.isfinite(merit):
        raise FloatingPointError("candidate merit is non-finite")
    return float(merit)


def _candidate_damage_residual_scale(
    model: EngineeringGripModel,
    damage: np.ndarray,
    history: np.ndarray,
) -> float:
    """Return the constituent scale for one candidate ``H_next`` residual."""

    _, _, _, _, scale = engineering._damage_residuals(
        model, damage, history, history
    )
    if not np.isfinite(scale) or scale < 0.0:
        raise FloatingPointError(
            "Anderson candidate damage residual scale is non-finite"
        )
    return float(scale)


def _validate_anderson_candidate_map(
    model: EngineeringGripModel,
    H_next: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Validate the extra fixed-history solve used by one Anderson candidate.

    The returned damage is the coupled map evaluated at ``H_next``.  It is not
    retained as a trial vector in the lean v5 archive; only the scalar metrics
    derived from it are recorded for an accepted accelerated sweep.
    """

    result = engineering.solve_fixed_history_damage(
        model.material_operator, model.Gc, model.l0, H_next
    )
    if not isinstance(result, FixedHistoryDamageResult):
        raise TypeError(
            "Anderson candidate fixed-history solve returned the wrong type"
        )
    history = _f64(
        result.history,
        "Anderson candidate fixed-history result history",
        H_next.shape,
        nonfinite_error=FloatingPointError,
    )
    if not np.array_equal(history, H_next):
        raise ValueError(
            "Anderson candidate fixed-history result history is not paired with H_next"
        )
    damage = _f64(
        result.damage,
        "Anderson candidate fixed-history result damage",
        H_next.shape,
        nonfinite_error=FloatingPointError,
    )
    strong = _f64(
        result.strong_residual,
        "Anderson candidate fixed-history strong residual",
        damage.shape,
        nonfinite_error=FloatingPointError,
    )
    volume = _f64(
        result.euclidean_residual,
        "Anderson candidate fixed-history volume residual",
        damage.shape,
        nonfinite_error=FloatingPointError,
    )
    try:
        engineering._compare_damage_residual_reconstruction(
            strong,
            volume,
            damage,
            history,
            model.material_operator.to_scipy().tocsr(),
            model.material_operator.volumes,
            model.Gc,
            model.l0,
            "Anderson candidate fixed-history residual",
        )
    except ValueError as error:
        # The result arrays have already passed the structural shape/dtype and
        # finite checks above.  The remaining ValueError from this helper is a
        # numerical residual mismatch; preserve structural ValueErrors from
        # the explicit validation boundary.
        if "residual reconstruction exceeds" not in str(error):
            raise
        raise FloatingPointError(
            "Anderson candidate fixed-history residual reconstruction failed"
        ) from error
    return damage, _candidate_damage_residual_scale(model, damage, history)


def _anderson_candidate_merit_components(
    diagnostics: EngineeringGripDiagnostics,
    candidate_fp_error: float,
    candidate_fp_scale: float,
    candidate_damage_residual_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> dict[str, float]:
    """Evaluate Anderson globalization using only the candidate map metrics."""

    base = config.base
    fields = (
        (
            "material_displacement",
            diagnostics.e_u_material,
            base.displacement,
            diagnostics.displacement_scale,
            "candidate material displacement",
        ),
        (
            "union_displacement",
            diagnostics.e_u_union,
            base.displacement,
            diagnostics.displacement_scale,
            "candidate union displacement",
        ),
        (
            "damage_transition",
            diagnostics.e_damage,
            base.damage,
            diagnostics.damage_scale,
            "candidate damage",
        ),
        (
            "history_transition",
            diagnostics.e_history,
            base.history,
            diagnostics.history_scale,
            "candidate history",
        ),
        (
            "reaction_transition",
            diagnostics.e_reaction,
            base.reaction,
            diagnostics.reaction_scale,
            "candidate reaction",
        ),
        (
            "candidate_next_damage_residual",
            diagnostics.rho_next_inf,
            base.damage_residual,
            candidate_damage_residual_scale,
            "candidate next damage residual",
        ),
        (
            "force",
            diagnostics.force_error,
            base.force,
            diagnostics.force_scale,
            "candidate force",
        ),
        (
            "moment",
            diagnostics.moment_error,
            base.moment,
            diagnostics.moment_scale,
            "candidate moment",
        ),
    )
    components = {
        key: _normalized(float(error), gate, float(scale), name)
        for key, error, gate, scale, name in fields
        if error is not None and scale is not None
    }
    components["candidate_fixed_point"] = _normalized(
        candidate_fp_error,
        base.damage,
        candidate_fp_scale,
        "candidate fixed-point",
    )
    if not all(np.isfinite(value) for value in components.values()):
        raise FloatingPointError("Anderson candidate merit component is non-finite")
    return components


def _anderson_candidate_merit(
    diagnostics: EngineeringGripDiagnostics,
    candidate_fp_error: float,
    candidate_fp_scale: float,
    candidate_damage_residual_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> float:
    components = _anderson_candidate_merit_components(
        diagnostics,
        candidate_fp_error,
        candidate_fp_scale,
        candidate_damage_residual_scale,
        config,
    )
    merit = max(components.values(), default=0.0)
    if not np.isfinite(merit):
        raise FloatingPointError("Anderson candidate merit is non-finite")
    return float(merit)


def _anderson_residual_merit_components(
    residual_error: float,
    residual_scale: float,
    config: StabilizedEngineeringGripConfig,
    name: str,
) -> dict[str, float]:
    """Return the map-only merit for the explicit residual policy.

    This deliberately excludes all transition and post-history equation terms.
    ``residual_scale`` is paired with the same reference or candidate map
    evaluation that produced ``residual_error``; sharing a scale would change
    the declared selection rule.
    """

    value = _normalized(
        residual_error,
        config.base.damage,
        residual_scale,
        f"{name} committed-history map residual",
    )
    if not np.isfinite(value):
        raise FloatingPointError(f"{name} residual merit is non-finite")
    return {"committed_history_map_residual": value}


def _anderson_residual_reference_merit_components(
    d_iter: np.ndarray,
    d_solved: np.ndarray,
    config: StabilizedEngineeringGripConfig,
) -> dict[str, float]:
    error = float(np.max(np.abs(d_solved - d_iter)))
    scale = max(float(np.max(np.abs(d_solved))), float(np.max(np.abs(d_iter))))
    return _anderson_residual_merit_components(
        error,
        scale,
        config,
        "reference",
    )


def _anderson_residual_reference_merit(
    d_iter: np.ndarray,
    d_solved: np.ndarray,
    config: StabilizedEngineeringGripConfig,
) -> float:
    return max(
        _anderson_residual_reference_merit_components(
            d_iter,
            d_solved,
            config,
        ).values()
    )


def _anderson_residual_candidate_merit_components(
    candidate_fp_error: float,
    candidate_fp_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> dict[str, float]:
    return _anderson_residual_merit_components(
        candidate_fp_error,
        candidate_fp_scale,
        config,
        "candidate",
    )


def _anderson_residual_candidate_merit(
    candidate_fp_error: float,
    candidate_fp_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> float:
    return max(
        _anderson_residual_candidate_merit_components(
            candidate_fp_error,
            candidate_fp_scale,
            config,
        ).values()
    )


def _anderson_residual_backtracking_alpha(
    config: StabilizedEngineeringGripConfig,
    index: int,
) -> float:
    """Return one of the bounded secant-direction line-search alphas."""

    if not _is_anderson_residual_backtracking_policy(config.relaxation_policy):
        raise ValueError(
            "backtracking alpha is only defined for "
            "a residual Anderson backtracking policy"
        )
    if index < 0 or index > config.max_backtracks:
        raise ValueError("Anderson backtracking alpha index is outside its budget")
    reduction = config.anderson_alpha_reduction
    if reduction is None:
        raise ValueError("backtracked Anderson alpha reduction is missing")
    alpha = float(reduction**index)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("Anderson backtracking alpha is invalid")
    return alpha


def _anderson_residual_sufficient_decrease_limit(
    reference_merit: float,
    config: StabilizedEngineeringGripConfig,
    *,
    trial_alpha: float,
) -> float:
    """Return the explicit residual-map decrease threshold for one trial."""

    reference = _scalar(reference_merit, "Anderson residual reference merit")
    alpha = _scalar(trial_alpha, "Anderson residual trial alpha")
    if reference < 0.0 or not 0.0 < alpha <= 1.0:
        raise ValueError("Anderson residual threshold inputs are invalid")
    sufficient = config.sufficient_decrease
    beta = config.anderson_beta
    if sufficient is None or beta is None:
        raise ValueError("Anderson residual threshold configuration is incomplete")
    limit = (1.0 - sufficient * beta * alpha) * reference
    if not np.isfinite(limit):
        raise FloatingPointError("Anderson residual sufficient limit is non-finite")
    return float(max(0.0, limit))


def _anderson_residual_hard_pass(
    diagnostics: EngineeringGripDiagnostics,
) -> bool:
    """Return the hard candidate prerequisites for residual Anderson.

    Transition convergence and the fixed-history damage-equation residual are
    intentionally excluded here; they remain original final gates.  Raw range
    and healing, mechanics/CG, conservation, and history ordering are not
    allowed to be traded for a lower map residual.
    """

    return bool(
        diagnostics.finite_pass
        and diagnostics.mechanics_before_pass
        and diagnostics.mechanics_after_pass
        and diagnostics.force_pass
        and diagnostics.moment_pass
        and diagnostics.damage_range_pass
        and diagnostics.healing_pass
        and diagnostics.history_monotonic_pass
    )


def _anderson_force_moment_stage_metric(
    result: A2MechanicsSolveResult,
    gate_result: tuple[bool, bool, float, float, bool, bool],
    gate: GateTolerance,
    *,
    component: str,
    geometry_length: float,
) -> dict[str, Any]:
    """Serialize one already-evaluated force or moment mechanics stage.

    ``gate_result`` is the tuple returned by the production mechanics gate.
    This helper performs only the corresponding scalar scale/limit arithmetic;
    it does not evaluate mechanics or change the production pass flag.
    """

    if component not in {"force", "moment"}:
        raise ValueError(f"unknown force/moment attribution component {component!r}")
    if not np.isfinite(geometry_length) or geometry_length <= 0.0:
        raise ValueError("force/moment attribution geometry length must be positive")
    force_scale = max(
        1.0,
        abs(float(result.outer_reaction)),
        abs(float(result.bottom_reaction)),
        abs(float(result.anchor_reaction)),
    )
    error = (
        float(gate_result[2])
        if component == "force"
        else float(gate_result[3])
    )
    scale = force_scale if component == "force" else force_scale * geometry_length
    limit = float(gate.limit(scale))
    if not np.isfinite(error) or error < 0.0:
        raise FloatingPointError(
            f"residual hard-gate {component} stage error is non-finite"
        )
    if not np.isfinite(scale) or scale < 0.0 or not np.isfinite(limit) or limit < 0.0:
        raise FloatingPointError(
            f"residual hard-gate {component} stage scale/limit is invalid"
        )
    ratio_available = limit > 0.0
    normalized = None if not ratio_available else error / limit
    if normalized is not None and not np.isfinite(normalized):
        raise FloatingPointError(
            f"residual hard-gate {component} stage ratio is non-finite"
        )
    return {
        "error": error,
        "scale": float(scale),
        "limit": limit,
        "normalized": None if normalized is None else float(normalized),
        "ratio_available": ratio_available,
        "passed": bool(
            gate_result[4] if component == "force" else gate_result[5]
        ),
    }


def _anderson_force_moment_stage_severity(
    metric: Mapping[str, Any],
) -> tuple[int, float]:
    """Return the deterministic severity key for one force/moment stage.

    A positive error against a zero limit is an infinite normalized violation,
    even though its ratio is intentionally unavailable.  A zero error against
    a zero limit is an exact pass and therefore has zero severity.  Positive
    limits use their available normalized ratio.  The canonical stage order is
    used by ``max`` for ties.
    """

    error = float(metric["error"])
    limit = float(metric["limit"])
    if limit == 0.0:
        return (2, error) if error > 0.0 else (0, 0.0)
    normalized = metric["normalized"]
    if normalized is None:
        raise ValueError("positive force/moment limit is missing its ratio")
    return (1, float(normalized))


def _anderson_force_moment_worst_stage(
    stage_metrics: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return the canonical worst force/moment stage without new evaluation."""

    if tuple(stage_metrics) != _ANDERSON_HARD_GATE_STAGE_NAMES:
        raise ValueError("force/moment stage metrics are not canonical")
    return max(
        _ANDERSON_HARD_GATE_STAGE_NAMES,
        key=lambda name: _anderson_force_moment_stage_severity(stage_metrics[name]),
    )


def _anderson_force_moment_aggregate_detail(
    stage_metrics: Mapping[str, Mapping[str, Any]],
    *,
    passed: bool,
) -> dict[str, Any]:
    """Build a self-consistent scalar aggregate from two stage metrics.

    The top-level raw/scale/limit are the tuple belonging to the deterministic
    worst stage, with a positive error against a zero limit taking precedence
    over finite-ratio stages.  The canonical ``before``/``after`` order breaks
    ties.  They are a representative metric, not a physical combination of the
    two stage values.
    """

    worst_name = _anderson_force_moment_worst_stage(stage_metrics)
    worst = stage_metrics[worst_name]
    normalized = worst["normalized"]
    return {
        "available": True,
        "passed": bool(passed),
        "raw": float(worst["error"]),
        "scale": float(worst["scale"]),
        "limit": float(worst["limit"]),
        "normalized": None if normalized is None else float(normalized),
        "ratio_available": bool(worst["ratio_available"]),
        "metric_reason": "stage_worst_ratio_and_conjunction",
        "stage_reason": None,
        "cg_info": None,
        "cg_status_pass": None,
        "residual_limit_pass": None,
        "stage_metrics": {
            name: dict(stage_metrics[name])
            for name in _ANDERSON_HARD_GATE_STAGE_NAMES
        },
    }


def _anderson_residual_hard_gate_attribution(
    diagnostics: EngineeringGripDiagnostics,
    config: StabilizedEngineeringGripConfig,
    *,
    mechanics_before: A2MechanicsSolveResult | None = None,
    mechanics_after: A2MechanicsSolveResult | None = None,
    geometry_length: float = 1.0,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Describe already-computed residual-policy hard gates as scalars.

    This helper is observational only.  It consumes the primitive values that
    ``_full_diagnostics`` or ``_incomplete_diagnostics`` already produced and
    never evaluates a mechanics map, damage solve, or additional residual.  A
    gate whose complete input stage is absent is reported as unavailable with
    a false pass flag; that preserves the hard-rejection meaning while making
    the missing stage explicit instead of attributing a fabricated metric.
    """

    if not isinstance(diagnostics, EngineeringGripDiagnostics):
        raise TypeError("hard-gate attribution needs engineering diagnostics")
    base = config.base

    def present(*names: str) -> bool:
        return all(getattr(diagnostics, name, None) is not None for name in names)

    def unavailable_reason(name: str) -> str:
        if name == "finite":
            return "candidate_not_fully_evaluated"
        if name == "mechanics_before":
            return "mechanics_before_not_evaluated"
        if name in {"mechanics_after", "force", "moment"}:
            return (
                "mechanics_after_not_evaluated"
                if mechanics_after is None
                else "mechanics_before_not_evaluated"
            )
        if name in {"damage_range_lower", "damage_range_upper", "healing"}:
            return "damage_not_evaluated"
        if name == "history_ordering":
            return "history_not_evaluated"
        raise ValueError(f"unknown residual hard gate {name}")

    def detail(
        name: str,
        *,
        passed: bool,
        available: bool,
        raw: float | None,
        limit: float | None,
        normalized: float | None,
        ratio_available: bool,
        metric_reason: str,
        scale: float | None = None,
        cg_info: int | None = None,
        cg_status_pass: bool | None = None,
        residual_limit_pass: bool | None = None,
        stage_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not available:
            return {
                "available": False,
                "passed": False,
                "raw": None,
                "scale": None,
                "limit": None,
                "normalized": None,
                "ratio_available": False,
                "metric_reason": "unavailable",
                "stage_reason": unavailable_reason(name),
                "cg_info": None,
                "cg_status_pass": None,
                "residual_limit_pass": None,
                "stage_metrics": None,
            }
        return {
            "available": True,
            "passed": bool(passed),
            "raw": raw,
            "scale": scale,
            "limit": limit,
            "normalized": normalized,
            "ratio_available": bool(ratio_available),
            "metric_reason": metric_reason,
            "stage_reason": None,
            "cg_info": cg_info,
            "cg_status_pass": cg_status_pass,
            "residual_limit_pass": residual_limit_pass,
            "stage_metrics": (
                None
                if stage_metrics is None
                else {
                    stage: dict(metrics)
                    for stage, metrics in stage_metrics.items()
                }
            ),
        }

    # The mechanics result is passed from the already-completed solve.  The
    # diagnostic object intentionally does not retain the CG limit because the
    # v5 archive schema is independent of this opt-in JSONL instrumentation.
    # Falling back to the relative residual here would silently replace the
    # production gate with a different metric, so a missing result is explicit
    # unavailability rather than an inferred limit.
    mechanics_before_available = mechanics_before is not None and present(
        "q_before",
        "force_before",
        "moment_before",
        "free_residual_norm_before",
        "free_residual_relative_before",
        "cg_info_before",
    )
    mechanics_after_available = mechanics_after is not None and present(
        "q_after",
        "force_after",
        "moment_after",
        "free_residual_norm_after",
        "free_residual_relative_after",
        "cg_info_after",
    )
    before_gate = (
        None
        if mechanics_before is None
        else engineering._mechanics_gate(
            mechanics_before,
            base,
            float(geometry_length),
        )
    )
    after_gate = (
        None
        if mechanics_after is None
        else engineering._mechanics_gate(
            mechanics_after,
            base,
            float(geometry_length),
        )
    )
    mechanics_before_gate_pass = bool(
        before_gate is not None and before_gate[0]
    )
    mechanics_after_gate_pass = bool(after_gate is not None and after_gate[0])
    finite_available = present(
        "q_before",
        "q_after",
        "force_before",
        "force_after",
        "moment_before",
        "moment_after",
        "free_residual_norm_before",
        "free_residual_norm_after",
        "free_residual_relative_before",
        "free_residual_relative_after",
        "cg_info_before",
        "cg_info_after",
        "d_iter",
        "d_next",
        "psi_before",
        "psi_after",
        "H_used",
        "H_next",
        "rho_used",
        "rho_next",
        "rho_used_volume",
        "rho_next_volume",
    ) and mechanics_before_available and mechanics_after_available
    both_mechanics_available = (
        mechanics_before_available and mechanics_after_available
    )
    force_available = both_mechanics_available and present(
        "force_error", "force_scale"
    )
    moment_available = both_mechanics_available and present(
        "moment_error", "moment_scale"
    )
    range_available = present("d_next", "damage_min", "damage_max")
    healing_available = present("d_next", "healing_min")
    history_available = present("H_next", "history_min_gap")

    def ratio(raw: float, limit: float, name: str) -> tuple[float | None, bool]:
        raw_value = float(raw)
        limit_value = float(limit)
        if not np.isfinite(raw_value) or not np.isfinite(limit_value):
            raise FloatingPointError(f"{name} metric is non-finite")
        if limit_value <= 0.0:
            return None, False
        value = raw_value / limit_value
        if not np.isfinite(value):
            raise FloatingPointError(f"{name} metric ratio is non-finite")
        return float(value), True

    force_stage_metrics = (
        None
        if not force_available or before_gate is None or after_gate is None
        else {
            "before": _anderson_force_moment_stage_metric(
                mechanics_before,
                before_gate,
                base.force,
                component="force",
                geometry_length=float(geometry_length),
            ),
            "after": _anderson_force_moment_stage_metric(
                mechanics_after,
                after_gate,
                base.force,
                component="force",
                geometry_length=float(geometry_length),
            ),
        }
    )
    moment_stage_metrics = (
        None
        if not moment_available or before_gate is None or after_gate is None
        else {
            "before": _anderson_force_moment_stage_metric(
                mechanics_before,
                before_gate,
                base.moment,
                component="moment",
                geometry_length=float(geometry_length),
            ),
            "after": _anderson_force_moment_stage_metric(
                mechanics_after,
                after_gate,
                base.moment,
                component="moment",
                geometry_length=float(geometry_length),
            ),
        }
    )
    damage_min = (
        float(diagnostics.damage_min) if range_available else None
    )
    damage_max = (
        float(diagnostics.damage_max) if range_available else None
    )
    healing_min = (
        float(diagnostics.healing_min) if healing_available else None
    )
    history_min_gap = (
        float(diagnostics.history_min_gap) if history_available else None
    )
    damage_lower_excess = (
        None if damage_min is None else max(0.0, -damage_min)
    )
    damage_upper_excess = (
        None if damage_max is None else max(0.0, damage_max - 1.0)
    )
    healing_excess = (
        None if healing_min is None else max(0.0, -healing_min)
    )
    history_ordering_excess = (
        None if history_min_gap is None else max(0.0, -history_min_gap)
    )
    force_detail = (
        _anderson_force_moment_aggregate_detail(
            force_stage_metrics,
            passed=all(
                bool(force_stage_metrics[name]["passed"])
                for name in _ANDERSON_HARD_GATE_STAGE_NAMES
            ),
        )
        if force_stage_metrics is not None
        else detail(
            "force",
            passed=False,
            available=False,
            raw=None,
            limit=None,
            normalized=None,
            ratio_available=False,
            metric_reason="gate_error_and_ratio",
        )
    )
    moment_detail = (
        _anderson_force_moment_aggregate_detail(
            moment_stage_metrics,
            passed=all(
                bool(moment_stage_metrics[name]["passed"])
                for name in _ANDERSON_HARD_GATE_STAGE_NAMES
            ),
        )
        if moment_stage_metrics is not None
        else detail(
            "moment",
            passed=False,
            available=False,
            raw=None,
            limit=None,
            normalized=None,
            ratio_available=False,
            metric_reason="gate_error_and_ratio",
        )
    )
    details: dict[str, dict[str, Any]] = {
        "finite": detail(
            "finite",
            passed=diagnostics.finite_pass,
            available=finite_available,
            raw=None,
            limit=None,
            normalized=None,
            ratio_available=False,
            metric_reason="boolean_gate",
        ),
        "mechanics_before": detail(
            "mechanics_before",
            passed=mechanics_before_gate_pass,
            available=mechanics_before_available,
            raw=(
                None
                if not mechanics_before_available
                else float(mechanics_before.free_residual_norm)
            ),
            limit=(
                None
                if not mechanics_before_available
                else float(mechanics_before.cg_convergence_limit)
            ),
            normalized=(
                None
                if not mechanics_before_available
                else ratio(
                    float(mechanics_before.free_residual_norm),
                    float(mechanics_before.cg_convergence_limit),
                    "residual hard-gate mechanics_before",
                )[0]
            ),
            ratio_available=(
                False
                if not mechanics_before_available
                else ratio(
                    float(mechanics_before.free_residual_norm),
                    float(mechanics_before.cg_convergence_limit),
                    "residual hard-gate mechanics_before",
                )[1]
            ),
            metric_reason="cg_info_and_free_residual_limit",
            cg_info=(
                None
                if not mechanics_before_available
                else int(mechanics_before.cg_info)
            ),
            cg_status_pass=(
                None
                if not mechanics_before_available
                else bool(mechanics_before.cg_info == 0)
            ),
            residual_limit_pass=(
                None
                if not mechanics_before_available
                else bool(
                    mechanics_before.cg_convergence_limit > 0.0
                    and mechanics_before.free_residual_norm
                    < mechanics_before.cg_convergence_limit
                )
            ),
        ),
        "mechanics_after": detail(
            "mechanics_after",
            passed=mechanics_after_gate_pass,
            available=mechanics_after_available,
            raw=(
                None
                if not mechanics_after_available
                else float(mechanics_after.free_residual_norm)
            ),
            limit=(
                None
                if not mechanics_after_available
                else float(mechanics_after.cg_convergence_limit)
            ),
            normalized=(
                None
                if not mechanics_after_available
                else ratio(
                    float(mechanics_after.free_residual_norm),
                    float(mechanics_after.cg_convergence_limit),
                    "residual hard-gate mechanics_after",
                )[0]
            ),
            ratio_available=(
                False
                if not mechanics_after_available
                else ratio(
                    float(mechanics_after.free_residual_norm),
                    float(mechanics_after.cg_convergence_limit),
                    "residual hard-gate mechanics_after",
                )[1]
            ),
            metric_reason="cg_info_and_free_residual_limit",
            cg_info=(
                None
                if not mechanics_after_available
                else int(mechanics_after.cg_info)
            ),
            cg_status_pass=(
                None
                if not mechanics_after_available
                else bool(mechanics_after.cg_info == 0)
            ),
            residual_limit_pass=(
                None
                if not mechanics_after_available
                else bool(
                    mechanics_after.cg_convergence_limit > 0.0
                    and mechanics_after.free_residual_norm
                    < mechanics_after.cg_convergence_limit
                )
            ),
        ),
        "force": force_detail,
        "moment": moment_detail,
        "damage_range_lower": detail(
            "damage_range_lower",
            passed=(
                range_available
                and damage_lower_excess <= float(base.damage_range_atol)
            ),
            available=range_available,
            raw=damage_lower_excess,
            limit=float(base.damage_range_atol) if range_available else None,
            normalized=(
                None
                if damage_lower_excess is None
                else ratio(
                    damage_lower_excess,
                    float(base.damage_range_atol),
                    "residual hard-gate damage lower range",
                )[0]
            ),
            ratio_available=(
                False
                if damage_lower_excess is None
                else ratio(
                    damage_lower_excess,
                    float(base.damage_range_atol),
                    "residual hard-gate damage lower range",
                )[1]
            ),
            metric_reason="damage_range_excess_and_limit",
        ),
        "damage_range_upper": detail(
            "damage_range_upper",
            passed=(
                range_available
                and damage_upper_excess <= float(base.damage_range_atol)
            ),
            available=range_available,
            raw=damage_upper_excess,
            limit=float(base.damage_range_atol) if range_available else None,
            normalized=(
                None
                if damage_upper_excess is None
                else ratio(
                    damage_upper_excess,
                    float(base.damage_range_atol),
                    "residual hard-gate damage upper range",
                )[0]
            ),
            ratio_available=(
                False
                if damage_upper_excess is None
                else ratio(
                    damage_upper_excess,
                    float(base.damage_range_atol),
                    "residual hard-gate damage upper range",
                )[1]
            ),
            metric_reason="damage_range_excess_and_limit",
        ),
        "healing": detail(
            "healing",
            passed=(
                healing_available
                and healing_excess <= float(base.healing_atol)
            ),
            available=healing_available,
            raw=healing_excess,
            limit=float(base.healing_atol) if healing_available else None,
            normalized=(
                None
                if healing_excess is None
                else ratio(
                    healing_excess,
                    float(base.healing_atol),
                    "residual hard-gate healing",
                )[0]
            ),
            ratio_available=(
                False
                if healing_excess is None
                else ratio(
                    healing_excess,
                    float(base.healing_atol),
                    "residual hard-gate healing",
                )[1]
            ),
            metric_reason="healing_excess_and_limit",
        ),
        "history_ordering": detail(
            "history_ordering",
            passed=(history_available and history_min_gap >= 0.0),
            available=history_available,
            raw=history_ordering_excess,
            limit=0.0 if history_available else None,
            normalized=None,
            ratio_available=False,
            metric_reason="history_ordering_excess_and_zero_limit",
        ),
    }
    failed = [
        name
        for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES
        if not details[name]["passed"]
    ]
    return details, failed


def _anderson_residual_hard_gate_record(
    diagnostics: EngineeringGripDiagnostics,
    config: StabilizedEngineeringGripConfig,
    *,
    mechanics_before: A2MechanicsSolveResult | None = None,
    mechanics_after: A2MechanicsSolveResult | None = None,
    geometry_length: float = 1.0,
    hard_gate_status: str = "evaluated",
) -> dict[str, Any]:
    """Return the scalar hard-gate fields retained for one callback event."""

    if hard_gate_status not in _ANDERSON_HARD_GATE_STATUSES:
        raise ValueError(
            "hard-gate attribution status must be one of "
            f"{_ANDERSON_HARD_GATE_STATUSES}"
        )

    details, failed = _anderson_residual_hard_gate_attribution(
        diagnostics,
        config,
        mechanics_before=mechanics_before,
        mechanics_after=mechanics_after,
        geometry_length=geometry_length,
    )
    if hard_gate_status == "candidate_map_failure" and failed:
        raise ValueError(
            "candidate-map failure attribution cannot contain hard-gate failures"
        )
    return {
        "record_type": "proposal",
        "hard_pass": hard_gate_status == "evaluated" and not failed,
        "hard_gate_status": hard_gate_status,
        "hard_gate_attribution_schema": (
            ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA
        ),
        "hard_gate_diagnostics": details,
        "failed_hard_gates": failed,
    }


def _anderson_transition_diagnostic_components(
    diagnostics: EngineeringGripDiagnostics,
    config: StabilizedEngineeringGripConfig,
    *,
    damage_residual_scale: float | None,
) -> dict[str, float] | None:
    """Return non-selection transition/final diagnostics for scalar logging."""

    if damage_residual_scale is None:
        damage_residual_scale = diagnostics.damage_residual_scale
    if damage_residual_scale is None:
        return None
    fields = (
        (
            "material_displacement",
            diagnostics.e_u_material,
            config.base.displacement,
            diagnostics.displacement_scale,
            "transition material displacement",
        ),
        (
            "union_displacement",
            diagnostics.e_u_union,
            config.base.displacement,
            diagnostics.displacement_scale,
            "transition union displacement",
        ),
        (
            "damage_transition",
            diagnostics.e_damage,
            config.base.damage,
            diagnostics.damage_scale,
            "transition damage",
        ),
        (
            "history_transition",
            diagnostics.e_history,
            config.base.history,
            diagnostics.history_scale,
            "transition history",
        ),
        (
            "reaction_transition",
            diagnostics.e_reaction,
            config.base.reaction,
            diagnostics.reaction_scale,
            "transition reaction",
        ),
        (
            "next_damage_residual",
            diagnostics.rho_next_inf,
            config.base.damage_residual,
            damage_residual_scale,
            "transition next damage residual",
        ),
        (
            "force",
            diagnostics.force_error,
            config.base.force,
            diagnostics.force_scale,
            "transition force",
        ),
        (
            "moment",
            diagnostics.moment_error,
            config.base.moment,
            diagnostics.moment_scale,
            "transition moment",
        ),
    )
    components = {
        key: _normalized(float(error), gate, float(scale), name)
        for key, error, gate, scale, name in fields
        if error is not None and scale is not None
    }
    if not all(np.isfinite(value) for value in components.values()):
        raise FloatingPointError("transition diagnostic component is non-finite")
    return components


def _fixed_point_pass(
    fixed_point_error: float,
    fixed_point_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> bool:
    return bool(
        fixed_point_error
        <= config.base.damage.limit(fixed_point_scale)
    )


def _damage_equation_pass(
    rho_used: np.ndarray,
    residual_scale: float,
    config: StabilizedEngineeringGripConfig,
) -> bool:
    return bool(
        float(np.max(np.abs(rho_used)))
        <= config.base.damage_residual.limit(residual_scale)
    )


def _coupled_pass(
    base_diagnostics: EngineeringGripDiagnostics,
    fixed_point_pass: bool,
    damage_equation_pass: bool,
) -> bool:
    gates = base_diagnostics.gate_passes
    return bool(
        all(gates[name] for name in STABILIZED_COUPLED_GATE_NAMES if name in gates)
        and fixed_point_pass
        and damage_equation_pass
    )


def _stabilized_warning_failed_gates(
    diagnostic: "StabilizedRelaxationSweepDiagnostics",
) -> tuple[str, ...]:
    """Return the canonical non-hard gates failed by one complete sweep."""

    gates = diagnostic.gate_passes
    expected = set(STABILIZED_WARNING_HARD_GATE_NAMES) | set(
        STABILIZED_WARNING_GATE_NAMES
    )
    if set(gates) != expected:
        raise ValueError("stabilized warning diagnostic gate names are not canonical")
    if any(not gates[name] for name in STABILIZED_WARNING_HARD_GATE_NAMES):
        return ()
    return tuple(
        name for name in STABILIZED_WARNING_GATE_NAMES if not gates[name]
    )


def _stabilized_warning_excesses(
    diagnostic: "StabilizedRelaxationSweepDiagnostics",
) -> tuple[float, float]:
    """Return raw damage-range excess and healing magnitude for a trial."""

    damage = diagnostic.d_trial
    if damage is None:
        raise ValueError("stabilized warning excesses need trial damage")
    previous = diagnostic.base_diagnostics.damage_before
    if not np.all(np.isfinite(damage)) or not np.all(np.isfinite(previous)):
        raise FloatingPointError("stabilized warning excesses are non-finite")
    range_excess = max(
        0.0,
        -float(np.min(damage)),
        float(np.max(damage)) - 1.0,
    )
    healing_magnitude = max(0.0, -float(np.min(damage - previous)))
    if not np.isfinite(range_excess) or not np.isfinite(healing_magnitude):
        raise FloatingPointError("stabilized warning excesses are non-finite")
    return float(range_excess), float(healing_magnitude)


def _stabilized_warning_acceptance_decision(
    diagnostic: "StabilizedRelaxationSweepDiagnostics",
) -> tuple[bool, tuple[str, ...], float, float]:
    """Classify a final complete sweep under the isolated warning policy."""

    if not diagnostic.complete_sweep:
        return False, (), 0.0, 0.0
    if any(
        not diagnostic.gate_passes[name]
        for name in STABILIZED_WARNING_HARD_GATE_NAMES
    ):
        return False, (), 0.0, 0.0
    failed = _stabilized_warning_failed_gates(diagnostic)
    if not failed:
        return False, (), 0.0, 0.0
    range_excess, healing_magnitude = _stabilized_warning_excesses(diagnostic)
    return True, failed, range_excess, healing_magnitude


def _stabilized_failure_reason(
    base_diagnostics: EngineeringGripDiagnostics,
    fixed_point_pass: bool,
    damage_equation_pass: bool,
    coupled_pass: bool,
    *,
    include_raw_gates: bool,
) -> str:
    gates = {
        **base_diagnostics.gate_passes,
        "damage_fixed_point": fixed_point_pass,
        "damage_equation": damage_equation_pass,
        "globalization": True,
        "coupled": coupled_pass,
    }
    if include_raw_gates:
        names = (*STABILIZED_WARNING_HARD_GATE_NAMES, *STABILIZED_WARNING_GATE_NAMES)
    else:
        names = (*STABILIZED_COUPLED_GATE_NAMES, "coupled")
    return ", ".join(name for name in names if not gates[name])


def _wrap_sweep(
    base_diagnostics: EngineeringGripDiagnostics,
    *,
    complete_sweep: bool,
    passed: bool,
    failure_stage: str,
    failure_reason: str,
    d_solved: np.ndarray | None,
    d_trial: np.ndarray | None,
    omega: float | None,
    reference_merit: float | None,
    merit: float | None,
    sufficient_decrease_limit: float | None,
    fixed_point_error: float | None,
    fixed_point_scale: float | None,
    fixed_point_pass: bool,
    damage_equation_pass: bool,
    globalization_pass: bool,
    coupled_pass: bool,
    trials: Sequence[StabilizedRelaxationTrial],
    anderson_history_size: int | None = None,
    anderson_qr_rank: int | None = None,
    anderson_coefficient_l2: float | None = None,
    anderson_regularization_mu: float | None = None,
    anderson_current_residual_l2: float | None = None,
    anderson_regularization_lambda: float | None = None,
    anderson_unregularized_ls_residual_l2: float | None = None,
    anderson_selection: str | None = None,
    trial_alpha: float | None = None,
    anderson_fallback_reason: str | None = None,
    anderson_restart_reason: str | None = None,
    candidate_fp_error: float | None = None,
    candidate_fp_scale: float | None = None,
    candidate_damage_residual_scale: float | None = None,
) -> StabilizedRelaxationSweepDiagnostics:
    trial_values = tuple(trials)
    accelerated_alpha_trials = sum(
        trial.trial_alpha is not None for trial in trial_values
    )
    backtrack_count = (
        max(0, accelerated_alpha_trials - 1)
        if accelerated_alpha_trials
        else max(0, len(trial_values) - 1)
    )
    return StabilizedRelaxationSweepDiagnostics(
        base_diagnostics=base_diagnostics,
        complete_sweep=complete_sweep,
        passed=passed,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        d_solved=d_solved,
        d_trial=d_trial,
        omega=omega,
        backtrack_count=backtrack_count,
        reference_merit=reference_merit,
        merit=merit,
        sufficient_decrease_limit=sufficient_decrease_limit,
        fixed_point_error=fixed_point_error,
        fixed_point_scale=fixed_point_scale,
        fixed_point_pass=fixed_point_pass,
        damage_equation_pass=damage_equation_pass,
        globalization_pass=globalization_pass,
        coupled_pass=coupled_pass,
        trials=trial_values,
        anderson_history_size=anderson_history_size,
        anderson_qr_rank=anderson_qr_rank,
        anderson_coefficient_l2=anderson_coefficient_l2,
        anderson_regularization_mu=anderson_regularization_mu,
        anderson_current_residual_l2=anderson_current_residual_l2,
        anderson_regularization_lambda=anderson_regularization_lambda,
        anderson_unregularized_ls_residual_l2=anderson_unregularized_ls_residual_l2,
        anderson_selection=anderson_selection,
        trial_alpha=trial_alpha,
        anderson_fallback_reason=anderson_fallback_reason,
        anderson_restart_reason=anderson_restart_reason,
        candidate_fp_error=candidate_fp_error,
        candidate_fp_scale=candidate_fp_scale,
        candidate_damage_residual_scale=candidate_damage_residual_scale,
    )


def _progress_json_safe(value: Any) -> Any:
    """Normalize one transient progress value to strict JSON primitives.

    Progress is intentionally a scalar-only interface.  In particular, arrays
    are rejected rather than converted to lists so a callback cannot
    accidentally retain a large trial field.  Non-finite optional values are
    represented by JSON ``null``; the controller's numerical diagnostics still
    retain their normal hard finite checks.
    """

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _progress_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_progress_json_safe(item) for item in value]
    raise TypeError(
        "stabilized progress events may contain only JSON scalars, strings, "
        "lists, and mappings; numerical arrays are not allowed"
    )


def _anderson_diagnostic_json_safe(value: Any) -> Any:
    """Normalize one durable scalar diagnostic value and reject arrays."""

    if isinstance(value, np.ndarray):
        raise TypeError("Anderson scalar diagnostics cannot contain numerical arrays")
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            # A JSON shape descriptor is structural metadata, not a retained
            # numerical vector.  Keep the exception explicit rather than
            # weakening the scalar-only rule for arbitrary lists.
            if str(key) == "target_shape":
                if not isinstance(item, (list, tuple)) or any(
                    isinstance(entry, bool) or not isinstance(entry, (int, np.integer))
                    or int(entry) < 0
                    for entry in item
                ):
                    raise TypeError("Anderson schedule target_shape is invalid")
                normalized[str(key)] = [int(entry) for entry in item]
            elif str(key) == "expected_archive_bindings":
                if not isinstance(item, list) or any(
                    not isinstance(entry, Mapping) for entry in item
                ):
                    raise TypeError(
                        "Anderson archive bindings must be a list of mappings"
                    )
                normalized[str(key)] = [
                    _anderson_diagnostic_json_safe(dict(entry)) for entry in item
                ]
            else:
                normalized[str(key)] = _anderson_diagnostic_json_safe(item)
        return normalized
    if isinstance(value, (list, tuple)):
        if any(not isinstance(item, str) for item in value):
            raise TypeError(
                "Anderson scalar diagnostics may contain only string lists, "
                "not numerical vectors"
            )
        return list(value)
    return _progress_json_safe(value)


def _anderson_json_scalar(value: Any, name: str) -> float | None:
    """Validate one optional finite JSON number used by gate attribution."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite JSON number or null")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _anderson_json_integer(value: Any, name: str) -> int:
    """Validate one non-negative JSON integer used by scalar provenance."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


_ANDERSON_ARCHIVE_BINDING_FIELDS = frozenset(
    {"path", "path_suffix", "kind", "completion_status", "archive_schema"}
)


def _anderson_archive_bindings(
    binding: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Validate the path-to-archive identity table in a diagnostic header.

    Current workflow streams bind exactly one completed and one failure
    archive.  The explicit table is intentionally redundant with the older
    path/schema/status lists: the redundancy keeps old readers compatible while
    making it impossible to pair a completed path with a failure schema or
    status (and vice versa).
    """

    raw = binding.get("expected_archive_bindings")
    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(
            "Anderson diagnostic archive bindings must contain completed and failure entries"
        )
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != _ANDERSON_ARCHIVE_BINDING_FIELDS:
            raise ValueError(
                f"Anderson diagnostic archive binding {index} has non-canonical fields"
            )
        path = item["path"]
        path_suffix = item["path_suffix"]
        kind = item["kind"]
        completion_status = item["completion_status"]
        archive_schema = item["archive_schema"]
        if any(
            not isinstance(value, str) or not value
            for value in (path, path_suffix, kind, completion_status, archive_schema)
        ):
            raise ValueError(
                f"Anderson diagnostic archive binding {index} contains invalid strings"
            )
        if kind not in {"completed", "failure"}:
            raise ValueError(
                f"Anderson diagnostic archive binding {index} has an invalid kind"
            )
        expected_suffix = ".failure.npz" if kind == "failure" else ".npz"
        if (
            path_suffix != expected_suffix
            or not Path(path).name.endswith(path_suffix)
            or (kind == "completed" and Path(path).name.endswith(".failure.npz"))
        ):
            raise ValueError(
                f"Anderson diagnostic {kind} archive path does not have the required "
                f"{expected_suffix} suffix"
            )
        normalized.append(
            {
                "path": path,
                "path_suffix": path_suffix,
                "kind": kind,
                "completion_status": completion_status,
                "archive_schema": archive_schema,
            }
        )
    if tuple(item["kind"] for item in normalized) != ("completed", "failure"):
        raise ValueError(
            "Anderson diagnostic archive bindings must be ordered completed then failure"
        )
    paths = [item["path"] for item in normalized]
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("Anderson diagnostic archive bindings must use distinct paths")

    primary = binding.get("expected_archive_path")
    if primary is not None and primary != paths[0]:
        raise ValueError(
            "Anderson diagnostic primary archive path is not the completed binding"
        )
    legacy_paths = binding.get("expected_archive_paths")
    if legacy_paths is not None and legacy_paths != paths:
        raise ValueError(
            "Anderson diagnostic archive path lists do not match the binding table"
        )
    legacy_schemas = binding.get("expected_archive_schemas")
    schemas = [item["archive_schema"] for item in normalized]
    if legacy_schemas is not None and legacy_schemas != schemas:
        raise ValueError(
            "Anderson diagnostic archive schema lists do not match the binding table"
        )
    legacy_statuses = binding.get("expected_completion_statuses")
    statuses = [item["completion_status"] for item in normalized]
    if legacy_statuses is not None and legacy_statuses != statuses:
        raise ValueError(
            "Anderson diagnostic completion-status lists do not match the binding table"
        )
    return tuple(normalized)


def _anderson_archive_binding_for_path(
    binding: Mapping[str, Any], archive_path: Any
) -> dict[str, str] | None:
    """Return the canonical archive identity paired with ``archive_path``."""

    bindings = _anderson_archive_bindings(binding)
    if not bindings:
        return None
    actual = Path(os.fspath(archive_path)).resolve()
    for item in bindings:
        if Path(item["path"]).resolve() == actual:
            return dict(item)
    raise ValueError("Anderson diagnostic final archive path is not bound")


def _anderson_archive_scalar_string(archive: Any, name: str) -> str:
    if name not in archive:
        raise ValueError(f"Anderson diagnostic archive is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype.kind not in "SU":
        raise ValueError(
            f"Anderson diagnostic archive {name} must be a scalar string"
        )
    return str(value.item())


def _anderson_archive_identity(path: Any) -> dict[str, Any]:
    """Read the scalar identity fields from one stabilized NPZ archive."""

    filename = Path(os.fspath(path))
    try:
        with np.load(filename, allow_pickle=False) as archive:
            archive_schema = _anderson_archive_scalar_string(
                archive, "archive_schema"
            )
            completion_status = _anderson_archive_scalar_string(
                archive, "completion_status"
            )
            metadata_text = _anderson_archive_scalar_string(archive, "metadata_json")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"cannot validate Anderson diagnostic archive identity: {filename}"
        ) from error
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Anderson diagnostic archive metadata is not valid JSON: {filename}"
        ) from error
    if not isinstance(metadata, Mapping):
        raise ValueError("Anderson diagnostic archive metadata must be a mapping")
    kind = metadata.get("kind")
    if kind not in {"completed", "failure"}:
        raise ValueError("Anderson diagnostic archive metadata kind is invalid")
    if metadata.get("schema") != archive_schema:
        raise ValueError("Anderson diagnostic archive metadata schema is forged")
    if metadata.get("completion_status") != completion_status:
        raise ValueError(
            "Anderson diagnostic archive metadata completion status is forged"
        )
    return {
        "archive_schema": archive_schema,
        "completion_status": completion_status,
        "kind": kind,
        "metadata": dict(metadata),
    }


def _validate_anderson_bound_archive(
    path: Any,
    binding: Mapping[str, str],
    *,
    status: str,
    schedule_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate path, NPZ identity, status, schema, and optional schedule."""

    if status != binding["completion_status"]:
        raise ValueError(
            "Anderson diagnostic final status does not match its bound archive"
        )
    identity = _anderson_archive_identity(path)
    for key in ("kind", "completion_status", "archive_schema"):
        if identity[key] != binding[key]:
            raise ValueError(
                "Anderson diagnostic archive path, kind, status, and schema are not paired"
            )
    if schedule_binding is not None:
        metadata = identity["metadata"]
        if metadata.get("load_schedule") != dict(schedule_binding):
            raise ValueError(
                "scheduled Anderson diagnostic archive schedule binding is forged"
            )
    return identity


def _validate_anderson_scalar_event(event: Mapping[str, Any]) -> None:
    """Validate the optional current residual hard-gate fields.

    Historical scalar streams predate these fields, so absence is explicitly
    accepted.  Once present, however, the record is checked as a complete
    canonical attribution: every metric has one declared meaning, unavailable
    gates cannot retain numeric provenance, the ordered failed list must be
    derivable from the per-gate booleans, and numerical candidate-map failure
    is distinct from a hard-gate rejection.  This is scalar contract
    validation, not an additional physical solve.
    """

    if not isinstance(event, Mapping):
        raise TypeError("Anderson scalar diagnostic event must be a mapping")
    trial_alpha_present = "trial_alpha" in event
    trial_alpha_index_present = "trial_alpha_index" in event
    identity = event.get("selection_merit_identity")
    trial_record = event.get("record_type") in ("proposal", "sweep")
    record_type = event.get("record_type")
    backtracking_identity = identity in {
        STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY,
        STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY,
    }
    if trial_alpha_index_present and (
        not backtracking_identity or record_type != "proposal"
    ):
        raise ValueError(
            "trial_alpha_index is valid only on backtracked Anderson trial records"
        )
    if trial_alpha_present or (backtracking_identity and trial_record):
        if not backtracking_identity or not trial_record:
            raise ValueError(
                "trial_alpha is valid only on backtracked Anderson trial records"
            )
        if not trial_alpha_present:
            raise ValueError("backtracked Anderson trial record is missing trial_alpha")
        alpha = _anderson_json_scalar(event.get("trial_alpha"), "trial_alpha")
        if record_type == "proposal":
            candidate_kind = event.get("candidate_kind")
            if candidate_kind not in ("anderson_accelerated", "picard_fallback"):
                raise ValueError("backtracked proposal candidate_kind is invalid")
            if candidate_kind == "anderson_accelerated" and alpha is None:
                raise ValueError("backtracked accelerated trial is missing trial_alpha")
            if candidate_kind == "picard_fallback" and alpha is not None:
                raise ValueError("backtracked Picard fallback retains trial_alpha")
            if candidate_kind == "anderson_accelerated":
                valid_alphas = tuple(
                    STABILIZED_ANDERSON_ALPHA_REDUCTION**index
                    for index in range(STABILIZED_ANDERSON_MAX_BACKTRACKS + 1)
                )
                if alpha not in valid_alphas:
                    raise ValueError("trial_alpha is not on the configured schedule")
                if not trial_alpha_index_present:
                    raise ValueError(
                        "backtracked accelerated trial is missing trial_alpha_index"
                    )
                alpha_index = _anderson_json_integer(
                    event["trial_alpha_index"], "trial_alpha_index"
                )
                expected_alpha = _anderson_backtracking_alpha_for_index(
                    alpha_index, "trial_alpha_index"
                )
                if alpha != expected_alpha:
                    raise ValueError(
                        "trial_alpha is not paired with trial_alpha_index"
                    )
            elif trial_alpha_index_present:
                raise ValueError("backtracked Picard fallback retains trial_alpha_index")
        else:
            if trial_alpha_index_present:
                raise ValueError("backtracked sweep retains trial_alpha_index")
            selection = event.get("anderson_selection")
            if selection not in ("accelerated", "picard_fallback"):
                raise ValueError("backtracked sweep selection is invalid")
            if selection == "accelerated" and alpha is None:
                raise ValueError("backtracked accelerated sweep is missing trial_alpha")
            if selection != "accelerated" and alpha is not None:
                raise ValueError("backtracked non-accelerated sweep retains trial_alpha")
            if "backtrack_count" not in event:
                raise ValueError("backtracked sweep is missing backtrack_count")
            backtrack_count = _anderson_json_integer(
                event["backtrack_count"], "backtrack_count"
            )
            if selection == "accelerated":
                valid_alphas = tuple(
                    STABILIZED_ANDERSON_ALPHA_REDUCTION**index
                    for index in range(STABILIZED_ANDERSON_MAX_BACKTRACKS + 1)
                )
                if alpha not in valid_alphas:
                    raise ValueError("trial_alpha is not on the configured schedule")
                expected_alpha = _anderson_backtracking_alpha_for_index(
                    backtrack_count, "backtrack_count"
                )
                if alpha != expected_alpha:
                    raise ValueError(
                        "trial_alpha is not paired with sweep backtrack_count"
                    )
    regularized_identity = (
        identity == STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY
    )
    regularization_fields = {
        "anderson_regularization_mu",
        "anderson_current_residual_l2",
        "anderson_regularization_lambda",
        "anderson_unregularized_ls_residual_l2",
        "anderson_coefficient_l2",
    }
    ambiguous_regularization_fields = {
        "regularization_mu",
        "regularization_lambda",
        "unregularized_residual_norm",
        "coefficient_norm",
    }
    ambiguous_present = ambiguous_regularization_fields.intersection(event)
    if ambiguous_present:
        raise ValueError(
            "ambiguous legacy regularization diagnostics are forbidden: "
            f"{sorted(ambiguous_present)}"
        )
    present_regularization_fields = regularization_fields.intersection(event)
    if present_regularization_fields and not regularized_identity:
        raise ValueError(
            "regularization diagnostics require the regularized Anderson merit identity"
        )
    if regularized_identity and record_type in (
        "anderson_proposal",
        "proposal",
        "sweep",
    ):
        if "anderson_coefficient_norm" in event:
            raise ValueError(
                "regularized Anderson scalar records must use "
                "anderson_coefficient_l2"
            )
        missing_regularization = sorted(regularization_fields - set(event))
        if missing_regularization:
            raise ValueError(
                "regularized Anderson scalar record is missing fields: "
                f"{missing_regularization}"
            )
        mu = _anderson_json_scalar(
            event["anderson_regularization_mu"], "anderson_regularization_mu"
        )
        if mu is None or mu <= 0.0:
            raise ValueError("anderson_regularization_mu must be finite and positive")
        current_residual_l2 = _anderson_json_scalar(
            event["anderson_current_residual_l2"],
            "anderson_current_residual_l2",
        )
        regularization_lambda = _anderson_json_scalar(
            event["anderson_regularization_lambda"],
            "anderson_regularization_lambda",
        )
        unregularized_ls_residual_l2 = _anderson_json_scalar(
            event["anderson_unregularized_ls_residual_l2"],
            "anderson_unregularized_ls_residual_l2",
        )
        coefficient_l2 = _anderson_json_scalar(
            event["anderson_coefficient_l2"], "anderson_coefficient_l2"
        )
        if regularization_lambda is not None and current_residual_l2 is None:
            raise ValueError(
                "anderson_regularization_lambda lacks current residual l2"
            )
        if regularization_lambda is not None and regularization_lambda < 0.0:
            raise ValueError("anderson_regularization_lambda must be non-negative")
        if current_residual_l2 is not None and current_residual_l2 < 0.0:
            raise ValueError("anderson_current_residual_l2 must be non-negative")
        if (
            unregularized_ls_residual_l2 is not None
            and unregularized_ls_residual_l2 < 0.0
        ):
            raise ValueError(
                "anderson_unregularized_ls_residual_l2 must be non-negative"
            )
        if coefficient_l2 is not None and coefficient_l2 < 0.0:
            raise ValueError("anderson_coefficient_l2 must be non-negative")
        if (
            regularization_lambda is not None
            and current_residual_l2 is not None
            and not _anderson_scalar_relation_close(
                regularization_lambda,
                mu * current_residual_l2**2,
            )
        ):
            raise ValueError(
                "anderson_regularization_lambda is not paired with mu and current residual l2"
            )
    detail_key = "hard_gate_diagnostics"
    failed_key = "failed_hard_gates"
    schema_key = "hard_gate_attribution_schema"
    present_details = detail_key in event
    present_failed = failed_key in event
    present_schema = schema_key in event
    present_status = "hard_gate_status" in event
    if not (present_details or present_failed or present_schema):
        if present_status:
            raise ValueError(
                "hard_gate_status cannot appear without hard-gate attribution"
            )
        return
    if present_details != present_failed:
        raise ValueError(
            "residual hard-gate attribution must contain both diagnostics and failed list"
        )
    if not present_details:
        raise ValueError(
            "hard_gate_attribution_schema cannot appear without hard-gate attribution"
        )
    if not present_schema:
        raise ValueError("hard-gate attribution is missing its schema")
    if event.get("record_type") not in (None, "proposal"):
        raise ValueError(
            "residual hard-gate attribution is valid only on proposal records"
        )
    if not present_status:
        raise ValueError("v3 hard-gate attribution is missing hard_gate_status")
    hard_gate_status = event["hard_gate_status"]
    if hard_gate_status not in _ANDERSON_HARD_GATE_STATUSES:
        raise ValueError("residual hard-gate attribution status is invalid")
    candidate_map_status = event.get("candidate_map_status")
    if "schema" in event and candidate_map_status is None:
        raise ValueError("v3 proposal attribution is missing candidate_map_status")
    if candidate_map_status is not None:
        if candidate_map_status not in {"completed", "failed", "not_requested"}:
            raise ValueError("hard-gate attribution has invalid candidate_map_status")
        candidate_map_failure = hard_gate_status == "candidate_map_failure"
        if candidate_map_failure != (candidate_map_status == "failed"):
            raise ValueError(
                "hard-gate status is inconsistent with candidate_map_status"
            )
        if candidate_map_failure and (
            event.get("candidate_kind") != "anderson_accelerated"
            or event.get("failure_stage") != "anderson_candidate_damage"
            or event.get("complete") is not False
        ):
            raise ValueError(
                "candidate-map failure has inconsistent candidate context"
            )
    if hard_gate_status == "not_evaluated" and (
        "schema" in event or candidate_map_status is not None
    ) and (
        event.get("candidate_kind") != "anderson_accelerated"
        or candidate_map_status != "not_requested"
        or event.get("complete") is not False
        or event.get("failure_stage")
        not in {"anderson_candidate_finite", "mechanics_after"}
    ):
        raise ValueError("not-evaluated hard-gate attribution has inconsistent context")
    if present_schema and event[schema_key] != ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA:
        raise ValueError("hard-gate attribution schema is invalid")
    details = event[detail_key]
    if not isinstance(details, Mapping) or tuple(details) != tuple(
        ANDERSON_RESIDUAL_HARD_GATE_NAMES
    ):
        raise ValueError("hard_gate_diagnostics do not contain canonical gate names")
    if "hard_pass" not in event or type(event["hard_pass"]) is not bool:
        raise ValueError("residual hard-gate attribution requires hard_pass")

    expected_detail_fields = tuple(_ANDERSON_HARD_GATE_DETAIL_FIELDS)

    def check_ratio(
        item: Mapping[str, Any],
        gate_name: str,
        *,
        strict: bool,
    ) -> bool:
        raw_value = _anderson_json_scalar(item["raw"], f"{gate_name}.raw")
        limit_value = _anderson_json_scalar(item["limit"], f"{gate_name}.limit")
        if raw_value is None or limit_value is None:
            raise ValueError(f"{gate_name} requires finite raw and limit metrics")
        if raw_value < 0.0 or limit_value < 0.0:
            raise ValueError(f"{gate_name} raw/limit metrics must be non-negative")
        expected_ratio_available = limit_value > 0.0
        if type(item["ratio_available"]) is not bool or item[
            "ratio_available"
        ] != expected_ratio_available:
            raise ValueError(f"{gate_name}.ratio_available is inconsistent")
        normalized = _anderson_json_scalar(
            item["normalized"], f"{gate_name}.normalized"
        )
        if expected_ratio_available:
            if normalized is None:
                raise ValueError(f"{gate_name}.normalized is missing")
            expected_normalized = raw_value / limit_value
            if not np.isclose(
                normalized,
                expected_normalized,
                rtol=2.0e-12,
                atol=2.0e-15,
            ):
                raise ValueError(f"{gate_name}.normalized is not raw/limit")
        elif normalized is not None:
            raise ValueError(
                f"{gate_name}.normalized must be null when its limit is zero"
            )
        return raw_value < limit_value if strict else raw_value <= limit_value

    def check_stage_metric(
        item: Mapping[str, Any], gate_name: str, stage_name: str
    ) -> dict[str, Any]:
        if not isinstance(item, Mapping) or tuple(item) != tuple(
            _ANDERSON_HARD_GATE_STAGE_METRIC_FIELDS
        ):
            raise ValueError(
                f"{gate_name}.stage_metrics[{stage_name}] is not canonical"
            )
        error = _anderson_json_scalar(
            item["error"], f"{gate_name}.{stage_name}.error"
        )
        scale = _anderson_json_scalar(
            item["scale"], f"{gate_name}.{stage_name}.scale"
        )
        limit = _anderson_json_scalar(
            item["limit"], f"{gate_name}.{stage_name}.limit"
        )
        normalized = _anderson_json_scalar(
            item["normalized"], f"{gate_name}.{stage_name}.normalized"
        )
        if (
            error is None
            or scale is None
            or limit is None
            or error < 0.0
            or scale < 0.0
            or limit < 0.0
        ):
            raise ValueError(f"{gate_name}.{stage_name} metric is invalid")
        expected_ratio_available = limit > 0.0
        if type(item["ratio_available"]) is not bool or item[
            "ratio_available"
        ] != expected_ratio_available:
            raise ValueError(
                f"{gate_name}.{stage_name}.ratio_available is inconsistent"
            )
        if expected_ratio_available:
            if normalized is None:
                raise ValueError(
                    f"{gate_name}.{stage_name}.normalized is missing"
                )
            expected_normalized = error / limit
            if not np.isclose(
                normalized,
                expected_normalized,
                rtol=2.0e-12,
                atol=2.0e-15,
            ):
                raise ValueError(
                    f"{gate_name}.{stage_name}.normalized is not error/limit"
                )
        elif normalized is not None:
            raise ValueError(
                f"{gate_name}.{stage_name}.normalized must be null when its limit is zero"
            )
        if type(item["passed"]) is not bool:
            raise ValueError(f"{gate_name}.{stage_name}.passed must be boolean")
        if item["passed"] != (error <= limit):
            raise ValueError(
                f"{gate_name}.{stage_name}.passed is inconsistent with error/limit"
            )
        return {
            "error": error,
            "scale": scale,
            "limit": limit,
            "normalized": normalized,
            "ratio_available": item["ratio_available"],
            "passed": item["passed"],
        }

    for gate_name in ANDERSON_RESIDUAL_HARD_GATE_NAMES:
        item = details[gate_name]
        if not isinstance(item, Mapping) or tuple(item) != expected_detail_fields:
            raise ValueError(f"hard_gate_diagnostics[{gate_name}] is not canonical")
        if (
            type(item["available"]) is not bool
            or type(item["passed"]) is not bool
            or type(item["ratio_available"]) is not bool
        ):
            raise ValueError(
                f"hard_gate_diagnostics[{gate_name}] availability/pass must be booleans"
            )
        stage_reason = item["stage_reason"]
        if not item["available"]:
            if type(stage_reason) is not str or stage_reason not in (
                _ANDERSON_HARD_GATE_UNAVAILABLE_REASONS
            ):
                raise ValueError(
                    f"hard_gate_diagnostics[{gate_name}] has a non-canonical unavailable reason"
                )
            if (
                item["passed"]
                or item["raw"] is not None
                or item["scale"] is not None
                or item["limit"] is not None
                or item["normalized"] is not None
                or item["ratio_available"]
                or item["cg_info"] is not None
                or item["cg_status_pass"] is not None
                or item["residual_limit_pass"] is not None
                or item["stage_metrics"] is not None
            ):
                raise ValueError(
                    f"unavailable hard gate {gate_name} retains a metric or pass"
                )
            if item["metric_reason"] != "unavailable":
                raise ValueError(
                    f"unavailable hard gate {gate_name} has a metric reason"
                )
            continue

        if stage_reason is not None:
            raise ValueError(
                f"hard_gate_diagnostics[{gate_name}] has an unexpected stage reason"
            )
        expected_reason = _ANDERSON_HARD_GATE_METRIC_REASONS[gate_name]
        if item["metric_reason"] != expected_reason:
            raise ValueError(
                f"hard_gate_diagnostics[{gate_name}] has a non-canonical metric reason"
            )

        if gate_name == "finite":
            if any(
                item[field] is not None
                for field in (
                    "raw",
                    "scale",
                    "limit",
                    "normalized",
                    "cg_info",
                    "cg_status_pass",
                    "residual_limit_pass",
                    "stage_metrics",
                )
            ) or item["ratio_available"]:
                raise ValueError("finite hard gate cannot retain a numeric metric")
            continue

        if gate_name in {"mechanics_before", "mechanics_after"}:
            cg_info = item["cg_info"]
            if type(cg_info) is not int or cg_info < 0:
                raise ValueError(f"{gate_name}.cg_info must be a non-negative integer")
            if (
                type(item["cg_status_pass"]) is not bool
                or type(item["residual_limit_pass"]) is not bool
            ):
                raise ValueError(f"{gate_name} CG status fields must be booleans")
            if item["scale"] is not None or item["stage_metrics"] is not None:
                raise ValueError(f"{gate_name} cannot retain force/moment metrics")
            raw_value = _anderson_json_scalar(item["raw"], f"{gate_name}.raw")
            limit_value = _anderson_json_scalar(item["limit"], f"{gate_name}.limit")
            if raw_value is None or limit_value is None or raw_value < 0.0 or limit_value < 0.0:
                raise ValueError(f"{gate_name} residual metrics are invalid")
            if item["cg_status_pass"] != (cg_info == 0):
                raise ValueError(f"{gate_name}.cg_status_pass is inconsistent")
            expected_residual_pass = limit_value > 0.0 and raw_value < limit_value
            if item["residual_limit_pass"] != expected_residual_pass:
                raise ValueError(f"{gate_name}.residual_limit_pass is inconsistent")
            expected_pass = bool(
                item["cg_status_pass"] and item["residual_limit_pass"]
            )
            if item["passed"] != expected_pass:
                raise ValueError(f"{gate_name}.passed is inconsistent with the CG gate")
            if check_ratio(item, gate_name, strict=True) != expected_residual_pass:
                raise ValueError(f"{gate_name} residual threshold is inconsistent")
            continue

        if gate_name in {"force", "moment"}:
            if any(
                item[field] is not None
                for field in ("cg_info", "cg_status_pass", "residual_limit_pass")
            ):
                raise ValueError(f"{gate_name} cannot retain CG-only fields")
            stage_metrics = item["stage_metrics"]
            if not isinstance(stage_metrics, Mapping) or tuple(stage_metrics) != tuple(
                _ANDERSON_HARD_GATE_STAGE_NAMES
            ):
                raise ValueError(f"{gate_name}.stage_metrics are not canonical")
            checked_stages = {
                stage_name: check_stage_metric(
                    stage_metrics[stage_name], gate_name, stage_name
                )
                for stage_name in _ANDERSON_HARD_GATE_STAGE_NAMES
            }
            raw_value = _anderson_json_scalar(item["raw"], f"{gate_name}.raw")
            scale_value = _anderson_json_scalar(item["scale"], f"{gate_name}.scale")
            limit_value = _anderson_json_scalar(item["limit"], f"{gate_name}.limit")
            normalized = _anderson_json_scalar(
                item["normalized"], f"{gate_name}.normalized"
            )
            if (
                raw_value is None
                or scale_value is None
                or limit_value is None
                or raw_value < 0.0
                or scale_value < 0.0
                or limit_value < 0.0
            ):
                raise ValueError(f"{gate_name} aggregate metric is invalid")
            expected_ratio_available = limit_value > 0.0
            if type(item["ratio_available"]) is not bool or item[
                "ratio_available"
            ] != expected_ratio_available:
                raise ValueError(
                    f"{gate_name}.ratio_available is inconsistent with its limit"
                )
            if expected_ratio_available:
                if normalized is None:
                    raise ValueError(f"{gate_name}.normalized is missing")
            elif normalized is not None:
                raise ValueError(
                    f"{gate_name}.normalized must be null when its limit is zero"
                )
            worst_name = _anderson_force_moment_worst_stage(checked_stages)
            worst = checked_stages[worst_name]
            if expected_ratio_available:
                if not np.isclose(
                    normalized,
                    worst["normalized"],
                    rtol=2.0e-12,
                    atol=2.0e-15,
                ):
                    raise ValueError(
                        f"{gate_name}.normalized is not the worst stage ratio"
                    )
            for field, stage_field, value in (
                ("raw", "error", raw_value),
                ("scale", "scale", scale_value),
                ("limit", "limit", limit_value),
            ):
                if not np.isclose(
                    value,
                    worst[stage_field],
                    rtol=2.0e-12,
                    atol=2.0e-15,
                ):
                    raise ValueError(
                        f"{gate_name}.{field} is not the worst-stage representative"
                    )
            if expected_ratio_available and not np.isclose(
                normalized,
                raw_value / limit_value,
                rtol=2.0e-12,
                atol=2.0e-15,
            ):
                raise ValueError(f"{gate_name}.normalized is not raw/limit")
            expected_pass = all(
                checked_stages[stage_name]["passed"]
                for stage_name in _ANDERSON_HARD_GATE_STAGE_NAMES
            )
            if item["passed"] != expected_pass:
                raise ValueError(
                    f"{gate_name}.passed is inconsistent with stage conjunction"
                )
            continue

        if gate_name in {
            "damage_range_lower",
            "damage_range_upper",
            "healing",
        }:
            if item["scale"] is not None or item["stage_metrics"] is not None:
                raise ValueError(f"{gate_name} cannot retain stage metrics")
            expected_pass = check_ratio(item, gate_name, strict=False)
            if item["passed"] != expected_pass:
                raise ValueError(f"{gate_name}.passed is inconsistent with raw/limit")
            if any(
                item[field] is not None
                for field in ("cg_info", "cg_status_pass", "residual_limit_pass")
            ):
                raise ValueError(f"{gate_name} cannot retain CG-only fields")
            continue

        if gate_name == "history_ordering":
            if any(
                item[field] is not None
                for field in (
                    "normalized",
                    "scale",
                    "cg_info",
                    "cg_status_pass",
                    "residual_limit_pass",
                    "stage_metrics",
                )
            ) or item["ratio_available"]:
                raise ValueError("history ordering cannot claim a ratio")
            raw_value = _anderson_json_scalar(item["raw"], "history_ordering.raw")
            limit_value = _anderson_json_scalar(
                item["limit"], "history_ordering.limit"
            )
            if raw_value is None or limit_value != 0.0 or raw_value < 0.0:
                raise ValueError("history ordering metric is invalid")
            if item["passed"] != (raw_value <= 0.0):
                raise ValueError("history ordering pass is inconsistent")
            continue

        raise ValueError(f"unknown residual hard gate {gate_name}")

    failed = event[failed_key]
    if not isinstance(failed, list) or any(
        type(name) is not str for name in failed
    ):
        raise ValueError("failed_hard_gates must be a string list")
    expected_failed = [
        name for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES if not details[name]["passed"]
    ]
    if failed != expected_failed:
        raise ValueError("failed_hard_gates do not match per-gate pass flags")
    if hard_gate_status == "evaluated":
        if event["hard_pass"] != (not expected_failed):
            raise ValueError("hard_pass does not match failed_hard_gates")
    elif hard_gate_status == "candidate_map_failure" and (
        expected_failed or event["hard_pass"] is not False
    ):
        raise ValueError(
            "candidate-map failure attribution must have no failed hard gates "
            "and hard_pass=false"
        )
    elif hard_gate_status == "not_evaluated":
        if event["hard_pass"] is not False or not any(
            not details[name]["available"] for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES
        ):
            raise ValueError(
                "not-evaluated attribution must retain unavailable hard gates "
                "and hard_pass=false"
            )


def _validate_anderson_durable_event_schema(event: Mapping[str, Any]) -> None:
    """Reject internal attribution fragments at durable stream boundaries."""

    attribution_fields = {
        "hard_gate_attribution_schema",
        "hard_gate_status",
        "hard_gate_diagnostics",
        "failed_hard_gates",
    }
    if attribution_fields.intersection(event) and event.get("schema") != (
        ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA
    ):
        raise ValueError(
            "durable hard-gate attribution requires the current scalar schema"
        )


class AndersonScalarDiagnosticWriter:
    """Write a durable scalar-only Anderson diagnostic JSONL stream.

    The stream is deliberately separate from the lean v5 archive.  Its header
    binds the run to source/configuration metadata and its final record binds
    the completed stream to the exact archive bytes written on disk.  No field
    arrays are accepted by :meth:`emit`.
    """

    def __init__(
        self,
        path: Any,
        *,
        binding: Mapping[str, Any],
    ) -> None:
        output = Path(os.fspath(path))
        if not str(output):
            raise ValueError("Anderson diagnostic path must be non-empty")
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing Anderson diagnostic {output}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(binding, Mapping):
            raise TypeError("Anderson diagnostic binding must be a mapping")
        reserved = {
            "record_type",
            "schema",
            "final_archive_sha256",
            "final_archive_path",
            "final_archive_path_suffix",
            "final_archive_kind",
            "final_archive_schema",
            "final_archive_completion_status",
            "finalized",
        }
        if reserved & set(binding):
            raise ValueError(
                "Anderson diagnostic binding contains reserved record fields"
            )
        self.path = output
        safe_binding = _anderson_diagnostic_json_safe(dict(binding))
        if not isinstance(safe_binding, dict):
            raise TypeError("Anderson diagnostic binding must serialize as a mapping")
        _anderson_archive_bindings(safe_binding)
        self._binding = safe_binding
        try:
            self._handle = output.open(
                "x",
                encoding="utf-8",
                buffering=1,
            )
        except OSError as error:
            raise OSError(f"cannot open Anderson diagnostic {output}") from error
        self._closed = False
        self._finalized = False
        header = {
            "record_type": "header",
            "schema": ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
            "final_archive_sha256": None,
            "final_archive_path": None,
            "finalized": False,
            **self._binding,
        }
        try:
            self._write(header)
        except BaseException:
            self.close()
            raise

    def _write(self, payload: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Anderson diagnostic writer is closed")
        if not isinstance(payload, Mapping):
            raise TypeError("Anderson diagnostic record must be a mapping")
        safe = _anderson_diagnostic_json_safe(dict(payload))
        if not isinstance(safe, dict):
            raise TypeError("Anderson diagnostic record must serialize as a mapping")
        try:
            self._handle.write(
                json.dumps(
                    safe,
                    # Preserve the canonical attribution key order.  The
                    # validator intentionally treats that order as part of the
                    # scalar schema; scalar-only durability does not require
                    # alphabetic JSON keys.
                    sort_keys=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            self._handle.flush()
        except (OSError, TypeError, ValueError) as error:
            raise OSError(f"cannot write Anderson diagnostic {self.path}") from error

    def emit(self, event: Mapping[str, Any]) -> None:
        if self._finalized:
            raise RuntimeError("Anderson diagnostic is already finalized")
        if not isinstance(event, Mapping):
            raise TypeError("Anderson diagnostic event must be a mapping")
        if event.get("record_type") in {"header", "final"}:
            raise ValueError("Anderson diagnostic events cannot forge header/final records")
        if "schema" in event and event["schema"] != ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA:
            raise ValueError("Anderson diagnostic event schema is invalid")
        _validate_anderson_durable_event_schema(event)
        instrumentation = self._binding.get("instrumentation_contract")
        if (
            isinstance(instrumentation, Mapping)
            and instrumentation.get("hard_gate_attribution") is True
            and event.get("record_type") == "proposal"
            and (
                "hard_gate_diagnostics" not in event
                or "failed_hard_gates" not in event
            )
        ):
            raise ValueError(
                "current residual Anderson proposal is missing hard-gate attribution"
            )
        _validate_anderson_scalar_event(event)
        self._write(dict(event))

    def finalize(self, archive_path: Any, *, status: str) -> str:
        if self._finalized:
            raise RuntimeError("Anderson diagnostic is already finalized")
        if not isinstance(status, str) or not status:
            raise ValueError("Anderson diagnostic final status must be non-empty")
        archive = Path(os.fspath(archive_path))
        if not archive.is_file():
            raise FileNotFoundError(
                f"Anderson diagnostic final archive does not exist: {archive}"
            )
        schedule_binding = self._binding.get("load_schedule")
        if schedule_binding is not None and not isinstance(schedule_binding, Mapping):
            raise ValueError("Anderson diagnostic schedule binding is invalid")
        bound_archive = _anderson_archive_binding_for_path(self._binding, archive)
        archive_identity: dict[str, Any] | None = None
        if bound_archive is not None:
            archive_identity = _validate_anderson_bound_archive(
                archive,
                bound_archive,
                status=status,
                schedule_binding=schedule_binding,
            )
        expected_archive_paths = self._binding.get("expected_archive_paths")
        if expected_archive_paths is not None:
            if (
                not isinstance(expected_archive_paths, list)
                or not expected_archive_paths
                or any(
                    type(item) is not str or not item
                    for item in expected_archive_paths
                )
            ):
                raise ValueError(
                    "Anderson diagnostic archive path binding is invalid"
                )
            if archive.resolve() not in {
                Path(item).resolve() for item in expected_archive_paths
            }:
                raise ValueError(
                    "Anderson diagnostic final archive path is not bound"
                )
        else:
            expected_archive_path = self._binding.get("expected_archive_path")
            if expected_archive_path is not None and (
                not isinstance(expected_archive_path, str)
                or not expected_archive_path
                or Path(expected_archive_path).resolve() != archive.resolve()
            ):
                raise ValueError(
                    "Anderson diagnostic final archive path is not bound"
                )
        expected_statuses = self._binding.get("expected_completion_statuses")
        if expected_statuses is not None:
            if not isinstance(expected_statuses, list) or any(
                not isinstance(item, str) for item in expected_statuses
            ) or status not in expected_statuses:
                raise ValueError(
                    "Anderson diagnostic final status is not bound to the archive contract"
                )
        if schedule_binding is not None:
            expected_schemas = self._binding.get("expected_archive_schemas")
            if not isinstance(expected_schemas, list) or any(
                not isinstance(item, str) for item in expected_schemas
            ):
                raise ValueError(
                    "Anderson diagnostic schedule archive schema binding is invalid"
                )
            if archive_identity is None:
                try:
                    with np.load(archive, allow_pickle=False) as archive_values:
                        archive_schema = _anderson_archive_scalar_string(
                            archive_values, "archive_schema"
                        )
                        metadata_text = _anderson_archive_scalar_string(
                            archive_values, "metadata_json"
                        )
                    metadata = json.loads(metadata_text)
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "cannot validate scheduled Anderson diagnostic archive binding"
                    ) from error
            else:
                archive_schema = archive_identity["archive_schema"]
                metadata = archive_identity["metadata"]
            if archive_schema not in expected_schemas:
                raise ValueError(
                    "scheduled Anderson diagnostic archive schema is not bound"
                )
            if not isinstance(metadata, Mapping) or metadata.get(
                "load_schedule"
            ) != dict(schedule_binding):
                raise ValueError(
                    "scheduled Anderson diagnostic final schedule binding is forged"
                )
        archive_sha256 = _archive_sha256(archive)
        final_record: dict[str, Any] = {
            "record_type": "final",
            "schema": ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
            "finalized": True,
            "final_archive_path": str(archive),
            "final_archive_sha256": archive_sha256,
            "final_archive_size_bytes": archive.stat().st_size,
            "status": status,
        }
        if bound_archive is not None:
            final_record.update(
                {
                    "final_archive_path_suffix": bound_archive["path_suffix"],
                    "final_archive_kind": bound_archive["kind"],
                    "final_archive_schema": bound_archive["archive_schema"],
                    "final_archive_completion_status": bound_archive[
                        "completion_status"
                    ],
                }
            )
        if schedule_binding is not None:
            final_record.update(
                {
                    "load_schedule": dict(schedule_binding),
                    "schedule_target_sha256": schedule_binding.get(
                        "canonical_target_sha256"
                    ),
                    "schedule_target_count": schedule_binding.get("target_count"),
                    "schedule_endpoint": schedule_binding.get("endpoint"),
                }
            )
        self._write(final_record)
        try:
            os.fsync(self._handle.fileno())
        except OSError as error:
            raise OSError(f"cannot finalize Anderson diagnostic {self.path}") from error
        self._finalized = True
        return archive_sha256

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._handle.close()
        except OSError as error:
            raise OSError(f"cannot close Anderson diagnostic {self.path}") from error
        finally:
            self._closed = True


def load_anderson_scalar_diagnostic(
    path: Any,
    *,
    archive_path: Any | None = None,
) -> dict[str, Any]:
    """Load and validate one scalar Anderson JSONL stream.

    The v1 stream remains backwards compatible with historical rows that do
    not contain residual hard-gate attribution.  Current attribution rows are
    strict: JSON arrays may only be string provenance lists (apart from the
    declared schedule shape), and every gate list/metric relationship is
    checked before a report consumes it.
    """

    filename = Path(os.fspath(path))
    if not filename.is_file():
        raise FileNotFoundError(f"Anderson scalar diagnostic does not exist: {filename}")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    try:
        lines = filename.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError(f"cannot read Anderson scalar diagnostic {filename}") from error
    if not lines:
        raise ValueError("Anderson scalar diagnostic is empty")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(
                f"Anderson scalar diagnostic contains a blank line at {line_number}"
            )
        try:
            parsed = json.loads(line, parse_constant=reject_constant)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot parse Anderson scalar diagnostic line {line_number}"
            ) from error
        if not isinstance(parsed, Mapping):
            raise ValueError(
                f"Anderson scalar diagnostic line {line_number} is not a mapping"
            )
        try:
            safe = _anderson_diagnostic_json_safe(dict(parsed))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Anderson scalar diagnostic line {line_number} is not scalar-only"
            ) from error
        if not isinstance(safe, dict):
            raise ValueError(
                f"Anderson scalar diagnostic line {line_number} is not a mapping"
            )
        record_type = safe.get("record_type")
        if record_type not in {"header", "final", "anderson_proposal", "proposal", "sweep"}:
            raise ValueError(
                f"Anderson scalar diagnostic line {line_number} has an unknown record_type"
            )
        if "schema" in safe and safe["schema"] != ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA:
            raise ValueError(
                f"Anderson scalar diagnostic line {line_number} has an invalid schema"
            )
        _validate_anderson_durable_event_schema(safe)
        _validate_anderson_scalar_event(safe)
        records.append(safe)

    if records[0].get("record_type") != "header":
        raise ValueError("Anderson scalar diagnostic must begin with a header")
    if records[-1].get("record_type") != "final":
        raise ValueError("Anderson scalar diagnostic must end with a final record")
    if sum(record.get("record_type") == "header" for record in records) != 1:
        raise ValueError("Anderson scalar diagnostic contains duplicate headers")
    if sum(record.get("record_type") == "final" for record in records) != 1:
        raise ValueError("Anderson scalar diagnostic contains duplicate final records")
    header = records[0]
    final = records[-1]
    if type(header.get("finalized")) is not bool or header["finalized"]:
        raise ValueError("Anderson scalar diagnostic header is not open")
    if type(final.get("finalized")) is not bool or not final["finalized"]:
        raise ValueError("Anderson scalar diagnostic final record is not finalized")
    if header.get("final_archive_sha256") is not None:
        raise ValueError("Anderson scalar diagnostic header already has an archive hash")
    final_archive_sha = _digest(
        final.get("final_archive_sha256"),
        "Anderson scalar diagnostic final archive SHA256",
    )
    final_archive_path = final.get("final_archive_path")
    if not isinstance(final_archive_path, str) or not final_archive_path:
        raise ValueError("Anderson scalar diagnostic final archive path is invalid")
    status = final.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("Anderson scalar diagnostic final status is invalid")
    bound_archive = _anderson_archive_binding_for_path(header, final_archive_path)
    if bound_archive is not None:
        if status != bound_archive["completion_status"]:
            raise ValueError(
                "Anderson scalar diagnostic final status is not paired with archive path"
            )
        for field, expected in (
            ("final_archive_path_suffix", bound_archive["path_suffix"]),
            ("final_archive_kind", bound_archive["kind"]),
            ("final_archive_schema", bound_archive["archive_schema"]),
            (
                "final_archive_completion_status",
                bound_archive["completion_status"],
            ),
        ):
            if final.get(field) != expected:
                raise ValueError(
                    f"Anderson scalar diagnostic final {field} is not paired"
                )
    expected_statuses = header.get("expected_completion_statuses")
    if expected_statuses is not None:
        if not isinstance(expected_statuses, list) or any(
            type(item) is not str for item in expected_statuses
        ) or status not in expected_statuses:
            raise ValueError("Anderson scalar diagnostic final status is not bound")
    instrumentation = header.get("instrumentation_contract")
    attribution_required = bool(
        isinstance(instrumentation, Mapping)
        and instrumentation.get("hard_gate_attribution") is True
    )
    if attribution_required:
        if header.get("selection_merit_identity") not in (
            STABILIZED_ANDERSON_RESIDUAL_MERIT_IDENTITY,
            STABILIZED_ANDERSON_RESIDUAL_BACKTRACKING_MERIT_IDENTITY,
            STABILIZED_ANDERSON_REGULARIZED_MERIT_IDENTITY,
        ):
            raise ValueError(
                "hard-gate attribution binding is not paired with residual Anderson"
            )
        required_fields = header.get("required_scalar_fields")
        if not isinstance(required_fields, list) or any(
            type(item) is not str for item in required_fields
        ) or not {
            "hard_gate_attribution_schema",
            "hard_gate_status",
            "hard_gate_diagnostics",
            "failed_hard_gates",
        }.issubset(required_fields):
            raise ValueError("hard-gate attribution binding is incomplete")
        for record in records[1:-1]:
            if record.get("record_type") == "proposal" and (
                "hard_gate_status" not in record
                or "hard_gate_diagnostics" not in record
                or "failed_hard_gates" not in record
            ):
                raise ValueError(
                    "current residual Anderson proposal is missing hard-gate attribution"
                )
    size = final.get("final_archive_size_bytes")
    if size is not None and (
        type(size) is not int or size < 0
    ):
        raise ValueError("Anderson scalar diagnostic final archive size is invalid")
    schedule_binding = header.get("load_schedule")
    if schedule_binding is not None and not isinstance(schedule_binding, Mapping):
        raise ValueError("Anderson diagnostic schedule binding is invalid")
    archive = None if archive_path is None else Path(os.fspath(archive_path))
    if archive is not None:
        actual_archive = archive.resolve()
        if not actual_archive.is_file():
            raise FileNotFoundError(
                f"Anderson scalar diagnostic archive does not exist: {actual_archive}"
            )
        if _archive_sha256(actual_archive) != final_archive_sha:
            raise ValueError(
                "Anderson scalar diagnostic final archive SHA256 does not match archive"
            )
        if Path(final_archive_path).resolve() != actual_archive:
            raise ValueError(
                "Anderson scalar diagnostic final archive path does not match archive"
            )
        archive_identity: dict[str, Any] | None = None
        if bound_archive is not None:
            archive_identity = _validate_anderson_bound_archive(
                actual_archive,
                bound_archive,
                status=status,
                schedule_binding=schedule_binding,
            )
        expected_archive_paths = header.get("expected_archive_paths")
        if expected_archive_paths is not None:
            if (
                not isinstance(expected_archive_paths, list)
                or not expected_archive_paths
                or any(
                    type(item) is not str or not item
                    for item in expected_archive_paths
                )
            ):
                raise ValueError(
                    "Anderson scalar diagnostic expected archive paths are invalid"
                )
            expected_archive_path = header.get("expected_archive_path")
            if expected_archive_path is not None and (
                not isinstance(expected_archive_path, str)
                or expected_archive_path not in expected_archive_paths
            ):
                raise ValueError(
                    "Anderson scalar diagnostic primary archive path is not paired"
                )
            if actual_archive not in {
                Path(item).resolve() for item in expected_archive_paths
            }:
                raise ValueError(
                    "Anderson scalar diagnostic expected archive path is not paired"
                )
        else:
            expected_archive_path = header.get("expected_archive_path")
            if expected_archive_path is not None and (
                not isinstance(expected_archive_path, str)
                or Path(expected_archive_path).resolve() != actual_archive
            ):
                raise ValueError(
                    "Anderson scalar diagnostic expected archive path is not paired"
                )
        if schedule_binding is not None and archive_identity is None:
            expected_schemas = header.get("expected_archive_schemas")
            if not isinstance(expected_schemas, list) or any(
                not isinstance(item, str) for item in expected_schemas
            ):
                raise ValueError(
                    "Anderson diagnostic schedule archive schema binding is invalid"
                )
            try:
                with np.load(actual_archive, allow_pickle=False) as archive_values:
                    archive_schema = _anderson_archive_scalar_string(
                        archive_values, "archive_schema"
                    )
                    metadata_text = _anderson_archive_scalar_string(
                        archive_values, "metadata_json"
                    )
                metadata = json.loads(metadata_text)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "cannot validate scheduled Anderson diagnostic archive binding"
                ) from error
            if archive_schema not in expected_schemas:
                raise ValueError(
                    "scheduled Anderson diagnostic archive schema is not bound"
                )
            if not isinstance(metadata, Mapping) or metadata.get(
                "load_schedule"
            ) != dict(schedule_binding):
                raise ValueError(
                    "scheduled Anderson diagnostic final schedule binding is forged"
                )
    return {
        "path": str(filename),
        "schema": ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
        "header": header,
        "events": records[1:-1],
        "final": final,
        "final_archive_sha256": final_archive_sha,
        "hard_gate_attribution_available": any(
            "hard_gate_diagnostics" in record for record in records[1:-1]
        ),
    }


def summarize_anderson_scalar_diagnostic(
    stream: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Count residual hard-gate failures and pairwise overlaps.

    The result intentionally reports marginal counts and overlaps.  It never
    assigns one rejection to a supposedly causal gate.
    """

    parsed = (
        load_anderson_scalar_diagnostic(stream)
        if not isinstance(stream, Mapping)
        else dict(stream)
    )
    events = parsed.get("events")
    if not isinstance(events, list):
        raise ValueError("Anderson scalar diagnostic stream events are invalid")
    proposal_records = [
        record
        for record in events
        if isinstance(record, Mapping) and record.get("record_type") == "proposal"
    ]
    attributed = [
        record
        for record in proposal_records
        if "hard_gate_diagnostics" in record
    ]
    failure_counts = {name: 0 for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES}
    unavailable_counts = {name: 0 for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES}
    overlap_counts: dict[str, int] = {}
    candidate_kind_counts: dict[str, int] = {}
    for record in attributed:
        candidate_kind = record.get("candidate_kind")
        if isinstance(candidate_kind, str):
            candidate_kind_counts[candidate_kind] = (
                candidate_kind_counts.get(candidate_kind, 0) + 1
            )
        details = record["hard_gate_diagnostics"]
        failed = record["failed_hard_gates"]
        hard_gate_evaluated = record.get("hard_gate_status") != "not_evaluated"
        if hard_gate_evaluated:
            for name in failed:
                failure_counts[name] += 1
        for name in ANDERSON_RESIDUAL_HARD_GATE_NAMES:
            if not details[name]["available"]:
                unavailable_counts[name] += 1
        if hard_gate_evaluated:
            for first_index, first in enumerate(failed):
                for second in failed[first_index + 1 :]:
                    key = f"{first}+{second}"
                    overlap_counts[key] = overlap_counts.get(key, 0) + 1
    header = parsed.get("header")
    final = parsed.get("final")
    if not isinstance(header, Mapping) or not isinstance(final, Mapping):
        raise ValueError("Anderson scalar diagnostic stream binding is invalid")
    return {
        "schema": "fracture2d-engineering-grip-anderson-hard-gate-analysis-v1",
        "stream_schema": parsed.get("schema"),
        "path": parsed.get("path"),
        "run_id": header.get("run_id"),
        "source_sha256": header.get("source_sha256"),
        "configuration_sha256": header.get("configuration_sha256"),
        "final_archive_sha256": parsed.get("final_archive_sha256"),
        "final_archive_path": final.get("final_archive_path"),
        "final_archive_path_suffix": final.get("final_archive_path_suffix"),
        "final_archive_kind": final.get("final_archive_kind"),
        "final_archive_schema": final.get("final_archive_schema"),
        "final_archive_completion_status": final.get(
            "final_archive_completion_status", final.get("status")
        ),
        "expected_archive_bindings": header.get("expected_archive_bindings"),
        "selection_merit_identity": header.get("selection_merit_identity"),
        "hard_gate_order": list(ANDERSON_RESIDUAL_HARD_GATE_NAMES),
        "proposal_records": len(proposal_records),
        "attributed_proposal_records": len(attributed),
        "unattributed_proposal_records": len(proposal_records) - len(attributed),
        "hard_gate_attribution_available": bool(attributed),
        "candidate_kind_counts": candidate_kind_counts,
        "hard_gate_failure_counts": failure_counts,
        "hard_gate_unavailable_counts": unavailable_counts,
        "hard_gate_overlap_counts": dict(sorted(overlap_counts.items())),
    }


def _emit_anderson_diagnostic_event(
    callback: Callable[[Mapping[str, Any]], None] | None,
    event: Mapping[str, Any],
) -> None:
    """Validate and deliver one scalar Anderson diagnostic event."""

    if callback is None:
        return
    if not callable(callback):
        raise TypeError("Anderson diagnostic callback must be callable")
    safe = _anderson_diagnostic_json_safe(dict(event))
    if not isinstance(safe, dict):
        raise TypeError("Anderson diagnostic event must be a mapping")
    _validate_anderson_scalar_event(safe)
    json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    callback(dict(safe))


def _diagnostic_dominant_component(
    components: Mapping[str, float | None],
) -> tuple[str | None, float | None]:
    available = {
        name: float(value)
        for name, value in components.items()
        if value is not None and np.isfinite(float(value))
    }
    if not available:
        return None, None
    name = max(available, key=available.__getitem__)
    return name, available[name]


def _anderson_scalar_trial_record(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    trial: StabilizedRelaxationTrial,
    reference_components: Mapping[str, float] | None,
    *,
    cached_q_before: bool,
    coefficient_l2: float | None,
    trial_alpha_index: int | None,
    mechanics_before_evaluations: int,
    sweep_elapsed_seconds: float | None,
) -> dict[str, Any]:
    candidate = trial.candidate_diagnostics
    candidate_components: dict[str, float | None] | None = None
    transition_components: dict[str, float] | None = None
    if candidate is not None:
        try:
            if trial.candidate_kind == "anderson_accelerated":
                if (
                    trial.candidate_fp_error is not None
                    and trial.candidate_fp_scale is not None
                    and trial.candidate_damage_residual_scale is not None
                ):
                    candidate_components = (
                        _anderson_residual_candidate_merit_components(
                            trial.candidate_fp_error,
                            trial.candidate_fp_scale,
                            config,
                        )
                        if _is_anderson_residual_policy(config.relaxation_policy)
                        else _anderson_candidate_merit_components(
                            candidate,
                            trial.candidate_fp_error,
                            trial.candidate_fp_scale,
                            trial.candidate_damage_residual_scale,
                            config,
                        )
                    )
                    transition_components = _anderson_transition_diagnostic_components(
                        candidate,
                        config,
                        damage_residual_scale=trial.candidate_damage_residual_scale,
                    )
            elif diagnostic.fixed_point_error is not None and diagnostic.fixed_point_scale is not None:
                transition_components = _anderson_transition_diagnostic_components(
                    candidate,
                    config,
                    damage_residual_scale=candidate.damage_residual_scale,
                )
                if not _is_anderson_residual_policy(config.relaxation_policy):
                    candidate_components = _candidate_merit_components(
                        candidate,
                        diagnostic.fixed_point_error,
                        diagnostic.fixed_point_scale,
                        config,
                    )
        except (FloatingPointError, ValueError):
            candidate_components = None
            transition_components = None
    if candidate_components is not None:
        dominant_component, dominant_value = _diagnostic_dominant_component(
            candidate_components
        )
    else:
        dominant_component, dominant_value = None, None
    hard_gate_diagnostics: dict[str, dict[str, Any]] | None = None
    failed_hard_gates: list[str] | None = None
    if _is_anderson_residual_policy(config.relaxation_policy):
        if trial.candidate_hard_gate_attribution is not None:
            hard_gate_diagnostics = trial.candidate_hard_gate_attribution[
                "hard_gate_diagnostics"
            ]
            failed_hard_gates = trial.candidate_hard_gate_attribution[
                "failed_hard_gates"
            ]
        else:
            if candidate is None:
                raise ValueError(
                    "residual Anderson trial is missing hard-gate diagnostics"
                )
            raise ValueError(
                "residual Anderson trial is missing authoritative hard-gate results"
            )
    reference_merit = diagnostic.reference_merit
    candidate_merit = (
        None
        if _is_anderson_residual_policy(config.relaxation_policy)
        and trial.candidate_kind != "anderson_accelerated"
        else trial.merit
    )
    merit_reduction = (
        None
        if reference_merit is None or candidate_merit is None
        else reference_merit - candidate_merit
    )
    merit_ratio = (
        None
        if reference_merit is None
        or candidate_merit is None
        or reference_merit <= 0.0
        else candidate_merit / reference_merit
    )
    candidate_map_evaluated = bool(
        trial.candidate_kind == "anderson_accelerated"
        and (
            trial.candidate_fp_error is not None
            or trial.failure_stage == "anderson_candidate_damage"
        )
    )
    candidate_map_status = (
        "completed"
        if trial.candidate_fp_error is not None
        else "failed"
        if candidate_map_evaluated
        else "not_requested"
    )
    mechanics_after_evaluated = not (
        trial.candidate_kind == "anderson_accelerated"
        and trial.failure_stage == "anderson_candidate_finite"
    )
    record = {
        "record_type": "proposal",
        "schema": ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
        "target_u_bar": diagnostic.target_u_bar,
        "sweep_index": diagnostic.sweep_index,
        "candidate_kind": trial.candidate_kind,
        "selected": trial.selected,
        "complete": trial.complete,
        "hard_pass": trial.hard_pass,
        "globalization_pass": trial.globalization_pass,
        "coupled_pass": trial.coupled_pass,
        "failure_stage": trial.failure_stage or None,
        "failure_reason": trial.failure_reason or None,
        "reference_merit": reference_merit,
        "reference_merit_components": (
            None if reference_components is None else dict(reference_components)
        ),
        "reference_fixed_point_error": diagnostic.fixed_point_error,
        "candidate_merit": candidate_merit,
        "candidate_merit_reduction": merit_reduction,
        "candidate_merit_ratio": merit_ratio,
        "candidate_merit_components": candidate_components,
        "selection_merit_identity": _anderson_merit_identity(
            config.relaxation_policy
        ),
        "selection_merit_components": candidate_components,
        "transition_diagnostic_components": transition_components,
        "candidate_dominant_component": dominant_component,
        "candidate_dominant_value": dominant_value,
        "candidate_actual_map_residual_inf": trial.candidate_fp_error,
        "candidate_fp_scale": trial.candidate_fp_scale,
        "candidate_damage_residual_scale": trial.candidate_damage_residual_scale,
        "candidate_next_damage_residual_inf": (
            None if candidate is None else candidate.rho_next_inf
        ),
        "candidate_damage_transition_inf": (
            None if candidate is None else candidate.e_damage
        ),
        "candidate_history_transition_inf": (
            None if candidate is None else candidate.e_history
        ),
        "sufficient_decrease_limit": trial.sufficient_decrease_limit,
        "anderson_history_size": diagnostic.anderson_history_size,
        "anderson_qr_rank": diagnostic.anderson_qr_rank,
        "anderson_coefficient_norm": coefficient_l2,
        "anderson_selection": diagnostic.anderson_selection,
        "anderson_fallback_reason": diagnostic.anderson_fallback_reason,
        "anderson_restart_reason": diagnostic.anderson_restart_reason,
        "cached_q_before": bool(cached_q_before),
        "evaluation_counts": {
            "mechanics_before": mechanics_before_evaluations,
            "mechanics_after": int(mechanics_after_evaluated),
            "candidate_map": int(candidate_map_evaluated),
        },
        "candidate_map_status": candidate_map_status,
        "picard_comparison_merit": None,
        "picard_comparison_evaluated": False,
        "sweep_elapsed_seconds": sweep_elapsed_seconds,
    }
    if _is_anderson_residual_backtracking_policy(config.relaxation_policy):
        record["trial_alpha"] = trial.trial_alpha
        if trial.candidate_kind == "anderson_accelerated":
            if trial_alpha_index is None:
                raise ValueError(
                    "backtracked accelerated trial is missing its alpha index"
                )
            record["trial_alpha_index"] = trial_alpha_index
    if hard_gate_diagnostics is not None and failed_hard_gates is not None:
        record.update(
            {
                "hard_gate_attribution_schema": (
                    ANDERSON_RESIDUAL_HARD_GATE_ATTRIBUTION_SCHEMA
                ),
                "hard_gate_status": trial.candidate_hard_gate_attribution.get(
                    "hard_gate_status", "evaluated"
                ),
                "hard_gate_diagnostics": hard_gate_diagnostics,
                "failed_hard_gates": failed_hard_gates,
            }
        )
    if _is_anderson_regularized_policy(config.relaxation_policy):
        record.pop("anderson_coefficient_norm", None)
        record.update(
            {
                "anderson_regularization_mu": diagnostic.anderson_regularization_mu,
                "anderson_current_residual_l2": (
                    diagnostic.anderson_current_residual_l2
                ),
                "anderson_regularization_lambda": (
                    diagnostic.anderson_regularization_lambda
                ),
                "anderson_unregularized_ls_residual_l2": (
                    diagnostic.anderson_unregularized_ls_residual_l2
                ),
                "anderson_coefficient_l2": coefficient_l2,
            }
        )
    _validate_anderson_scalar_event(record)
    return record


def _anderson_scalar_sweep_events(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    *,
    cached_q_before: bool,
    coefficient_l2: float | None,
    proposal_candidate_available: bool,
    proposal_fallback_reason: str | None,
    proposal_restart_reason: str | None,
    mechanics_before_evaluations: int,
    sweep_elapsed_seconds: float | None,
) -> tuple[dict[str, Any], ...]:
    reference_components: dict[str, float] | None = None
    if (
        diagnostic.d_solved is not None
        and diagnostic.d_iter is not None
        and diagnostic.H_used is not None
    ):
        reference_components = (
            _anderson_residual_reference_merit_components(
                diagnostic.d_iter,
                diagnostic.d_solved,
                config,
            )
            if _is_anderson_residual_policy(config.relaxation_policy)
            else _reference_merit_components(
                model,
                config,
                diagnostic.d_iter,
                diagnostic.d_solved,
                diagnostic.H_used,
            )
        )
    records: list[dict[str, Any]] = [
        {
            "record_type": "anderson_proposal",
            "schema": ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
            "selection_merit_identity": _anderson_merit_identity(
                config.relaxation_policy
            ),
            "target_u_bar": diagnostic.target_u_bar,
            "sweep_index": diagnostic.sweep_index,
            "candidate_available": proposal_candidate_available,
            "history_size": diagnostic.anderson_history_size,
            "qr_rank": diagnostic.anderson_qr_rank,
            "anderson_coefficient_norm": coefficient_l2,
            "fallback_reason": proposal_fallback_reason,
            "restart_reason": proposal_restart_reason,
            "candidate_evaluated": any(
                trial.candidate_kind == "anderson_accelerated"
                for trial in diagnostic.trials
            ),
            "selection_merit": None,
            "selected": diagnostic.anderson_selection,
            "sweep_elapsed_seconds": sweep_elapsed_seconds,
        }
    ]
    if _is_anderson_regularized_policy(config.relaxation_policy):
        records[0].pop("anderson_coefficient_norm", None)
        records[0].update(
            {
                "anderson_regularization_mu": diagnostic.anderson_regularization_mu,
                "anderson_current_residual_l2": (
                    diagnostic.anderson_current_residual_l2
                ),
                "anderson_regularization_lambda": (
                    diagnostic.anderson_regularization_lambda
                ),
                "anderson_unregularized_ls_residual_l2": (
                    diagnostic.anderson_unregularized_ls_residual_l2
                ),
                "anderson_coefficient_l2": coefficient_l2,
            }
        )
    records.extend(
        _anderson_scalar_trial_record(
            model,
            config,
            diagnostic,
            trial,
            reference_components,
            cached_q_before=cached_q_before,
            coefficient_l2=coefficient_l2,
            trial_alpha_index=(
                trial_index if trial.trial_alpha is not None else None
            ),
            mechanics_before_evaluations=mechanics_before_evaluations,
            sweep_elapsed_seconds=sweep_elapsed_seconds,
        )
        for trial_index, trial in enumerate(diagnostic.trials)
    )
    records.append(
        {
            "record_type": "sweep",
            "schema": ANDERSON_SCALAR_DIAGNOSTIC_SCHEMA,
            "target_u_bar": diagnostic.target_u_bar,
            "sweep_index": diagnostic.sweep_index,
            "complete": diagnostic.complete_sweep,
            "passed": diagnostic.passed,
            "failure_stage": diagnostic.failure_stage or None,
            "failure_reason": diagnostic.failure_reason or None,
            "reference_merit": diagnostic.reference_merit,
            "selection_merit_identity": _anderson_merit_identity(
                config.relaxation_policy
            ),
            "reference_merit_components": reference_components,
            "merit": diagnostic.merit,
            "merit_reduction": (
                None
                if diagnostic.reference_merit is None or diagnostic.merit is None
                else diagnostic.reference_merit - diagnostic.merit
            ),
            "merit_ratio": (
                None
                if diagnostic.reference_merit is None
                or diagnostic.merit is None
                or diagnostic.reference_merit <= 0.0
                else diagnostic.merit / diagnostic.reference_merit
            ),
            "sufficient_decrease_limit": diagnostic.sufficient_decrease_limit,
            "fixed_point_error": diagnostic.fixed_point_error,
            "candidate_actual_map_residual_inf": diagnostic.candidate_fp_error,
            "selection_merit": diagnostic.merit,
            "selection_merit_components": None,
            "transition_diagnostic_components": None,
            "anderson_history_size": diagnostic.anderson_history_size,
            "anderson_qr_rank": diagnostic.anderson_qr_rank,
            "anderson_coefficient_norm": coefficient_l2,
            "anderson_selection": diagnostic.anderson_selection,
            "anderson_fallback_reason": diagnostic.anderson_fallback_reason,
            "anderson_restart_reason": diagnostic.anderson_restart_reason,
            "proposal_count": len(diagnostic.trials),
            "evaluation_counts": {
                "mechanics_before": mechanics_before_evaluations,
                "mechanics_after": sum(
                    not (
                        trial.candidate_kind == "anderson_accelerated"
                        and trial.failure_stage == "anderson_candidate_finite"
                    )
                    for trial in diagnostic.trials
                ),
                "candidate_map": sum(
                    1
                    for trial in diagnostic.trials
                    if trial.candidate_kind == "anderson_accelerated"
                    and (
                        trial.candidate_fp_error is not None
                        or trial.failure_stage == "anderson_candidate_damage"
                    )
                ),
            },
            "cached_q_before": bool(cached_q_before),
            "picard_comparison_merit": None,
            "picard_comparison_evaluated": False,
            "sweep_elapsed_seconds": sweep_elapsed_seconds,
            "backtrack_count": diagnostic.backtrack_count,
        }
    )
    if _is_anderson_residual_backtracking_policy(config.relaxation_policy):
        records[-1]["trial_alpha"] = diagnostic.trial_alpha
    if _is_anderson_regularized_policy(config.relaxation_policy):
        records[-1].pop("anderson_coefficient_norm", None)
        records[-1].update(
            {
                "anderson_regularization_mu": diagnostic.anderson_regularization_mu,
                "anderson_current_residual_l2": (
                    diagnostic.anderson_current_residual_l2
                ),
                "anderson_regularization_lambda": (
                    diagnostic.anderson_regularization_lambda
                ),
                "anderson_unregularized_ls_residual_l2": (
                    diagnostic.anderson_unregularized_ls_residual_l2
                ),
                "anderson_coefficient_l2": coefficient_l2,
            }
        )
    return tuple(records)


def _progress_ratio(
    error: float | None,
    gate: GateTolerance,
    scale: float | None,
) -> float | None:
    """Return one finite gate ratio, or ``None`` when its inputs are absent."""

    if error is None or scale is None:
        return None
    error_value = float(error)
    scale_value = float(scale)
    limit = float(gate.limit(scale_value))
    if not (
        np.isfinite(error_value)
        and np.isfinite(scale_value)
        and np.isfinite(limit)
        and limit > 0.0
    ):
        return None
    ratio = abs(error_value) / limit
    return float(ratio) if np.isfinite(ratio) else None


def _progress_sweep_is_due(
    sweep_index: int,
    progress_every_sweeps: int,
    max_sweeps: int,
    passed: bool,
) -> bool:
    """Return whether one zero-based sweep should cross the progress seam."""

    one_based = sweep_index + 1
    return bool(
        one_based == 1
        or one_based % progress_every_sweeps == 0
        or one_based == max_sweeps
        or passed
    )


def _progress_sweep_event(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    *,
    cached_q_before: bool,
    max_sweeps: int,
) -> dict[str, Any]:
    """Build the scalar-only event for one selected outer stagger sweep."""

    damage = diagnostic.d_next
    if damage is None:
        damage = diagnostic.d_iter
    if damage is None or not np.all(np.isfinite(damage)):
        damage_min: float | None = None
        damage_max: float | None = None
    else:
        damage_min = float(np.min(damage))
        damage_max = float(np.max(damage))

    history = diagnostic.H_next
    if history is None:
        history = diagnostic.H_used
    history_max = (
        None
        if history is None or not np.all(np.isfinite(history))
        else float(np.max(history))
    )

    displacement = diagnostic.q_after
    if displacement is None:
        displacement = diagnostic.q_before
    max_material_displacement: float | None = None
    displacement_amplification: float | None = None
    if displacement is not None:
        values = np.asarray(displacement)
        required = 2 * model.n_material
        if values.ndim == 1 and values.size >= required and np.all(
            np.isfinite(values[:required])
        ):
            material = values[:required].reshape(model.n_material, 2)
            magnitude = np.linalg.norm(material, axis=1)
            if np.all(np.isfinite(magnitude)):
                max_material_displacement = float(np.max(magnitude))
                displacement_amplification = float(
                    max_material_displacement
                    / max(float(diagnostic.target_u_bar), np.finfo(np.float64).tiny)
                )

    reaction = diagnostic.top_reaction_after
    if reaction is None:
        reaction = diagnostic.top_reaction_before
    failing_gates = sorted(
        name for name, passed in diagnostic.gate_passes.items() if not passed
    )
    return {
        "event": "sweep",
        "target_load": diagnostic.target_u_bar,
        "sweep": diagnostic.sweep_index + 1,
        "max_sweeps": max_sweeps,
        "complete_sweep": diagnostic.complete_sweep,
        "passed": diagnostic.passed,
        "cached_q_before": bool(cached_q_before),
        "policy": config.relaxation_policy,
        "omega": diagnostic.omega,
        "anderson_history_size": diagnostic.anderson_history_size,
        "anderson_qr_rank": diagnostic.anderson_qr_rank,
        "anderson_selection": diagnostic.anderson_selection,
        "anderson_fallback_reason": diagnostic.anderson_fallback_reason,
        "anderson_restart_reason": diagnostic.anderson_restart_reason,
        "merit": diagnostic.merit,
        "d_max": damage_max,
        "d_min": damage_min,
        "H_max": history_max,
        "max_material_displacement": max_material_displacement,
        "displacement_amplification": displacement_amplification,
        "reaction": reaction,
        "fixed_point_error": diagnostic.fixed_point_error,
        "fixed_point_ratio": _progress_ratio(
            diagnostic.fixed_point_error,
            config.base.damage,
            diagnostic.fixed_point_scale,
        ),
        "history_ratio": _progress_ratio(
            diagnostic.e_history,
            config.base.history,
            diagnostic.history_scale,
        ),
        "damage_residual_ratio": _progress_ratio(
            diagnostic.rho_next_inf,
            config.base.damage_residual,
            diagnostic.damage_residual_scale,
        ),
        "failing_gates": failing_gates,
        "failure_stage": diagnostic.failure_stage or None,
        "failure_reason": diagnostic.failure_reason or None,
    }


def _progress_accepted_event(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    step: StabilizedEngineeringGripAcceptedStep,
) -> dict[str, Any]:
    """Build the scalar-only event for one committed step."""

    sweep = _progress_sweep_event(
        model,
        config,
        step.diagnostics,
        cached_q_before=False,
        max_sweeps=config.base.max_stagger_iterations,
    )
    return {
        "event": "accepted",
        "load": step.load_target,
        "increment": step.increment,
        "sweeps": step.stagger_iterations,
        "policy": config.relaxation_policy,
        "reaction": sweep["reaction"],
        "d_max": sweep["d_max"],
        "H_max": sweep["H_max"],
        "max_material_displacement": sweep["max_material_displacement"],
        "displacement_amplification": sweep["displacement_amplification"],
    }


def _emit_progress_event(
    callback: Callable[[dict[str, Any]], None] | None,
    event: Mapping[str, Any],
    started: float,
) -> None:
    """Validate and deliver one transient progress event without printing."""

    if callback is None:
        return
    payload = _progress_json_safe(dict(event))
    if not isinstance(payload, dict):
        raise TypeError("stabilized progress event must be a mapping")
    payload["elapsed"] = float(time.perf_counter() - started)
    payload = _progress_json_safe(payload)
    # Validate the exact object delivered to the callback.  This both rejects
    # accidental arrays and guarantees that a workflow printer can use
    # allow_nan=False without changing the numerical path.
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    callback(dict(payload))


def _emit_failure_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    reason: str,
    committed_load: float,
    target_load: float | None,
    accepted_steps: int,
    attempts: int,
    failure_type: str,
    started: float,
) -> None:
    """Emit the scalar-only terminal failure event."""

    _emit_progress_event(
        callback,
        {
            "event": "failure",
            "reason": reason,
            "committed_load": committed_load,
            "target_load": target_load,
            "accepted_steps": accepted_steps,
            "attempts": attempts,
            "failure_type": failure_type,
        },
        started,
    )


def _incomplete_base(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    committed: EngineeringGripState,
    d_iter: np.ndarray,
    target: float,
    stage: str,
    reason: str,
    sweep_index: int,
    *,
    before: A2MechanicsSolveResult | None = None,
    after: A2MechanicsSolveResult | None = None,
    psi_before: np.ndarray | None = None,
    psi_after: np.ndarray | None = None,
    H_used: np.ndarray | None = None,
    d_next: np.ndarray | None = None,
    rho_used: np.ndarray | None = None,
    rho_used_volume: np.ndarray | None = None,
) -> EngineeringGripDiagnostics:
    return engineering._incomplete_diagnostics(
        config.base,
        committed,
        d_iter,
        target,
        stage,
        reason,
        before=before,
        after=after,
        psi_before=psi_before,
        psi_after=psi_after,
        H_used=H_used,
        d_next=d_next,
        rho_used=rho_used,
        rho_used_volume=rho_used_volume,
        model=model,
        sweep_index=sweep_index,
    )


def _relaxation_omega(config: StabilizedEngineeringGripConfig, index: int) -> float:
    if index < 0:
        raise ValueError("relaxation candidate index must be non-negative")
    if config.relaxation_policy == "fixed":
        if index != 0:
            raise ValueError("fixed relaxation has exactly one candidate per sweep")
        return float(config.omega_initial)
    omega_reduction = config.omega_reduction
    assert omega_reduction is not None
    return float(
        max(
            config.omega_min,
            config.omega_initial * omega_reduction**index,
        )
    )


def _validate_reusable_mechanics_result(
    result: A2MechanicsSolveResult,
) -> None:
    """Require that a transient mechanics result owns immutable arrays.

    The stabilized cache retains the CPU-authoritative result returned by the
    selected mechanics solve.  In particular, it must not retain the JAX
    adapter's device result or an array view backed by mutable caller storage.
    ``A2MechanicsSolveResult`` and ``MechanicsEvaluation`` normally enforce
    this in their constructors; checking the flags here makes that ownership
    contract explicit at the cache boundary as well.
    """

    if not isinstance(result, A2MechanicsSolveResult):
        raise TypeError("stabilized mechanics reuse needs an A2 mechanics result")
    arrays = [result.displacement, result.free_residual, result.total_force]
    arrays.extend(
        getattr(result.evaluation, field.name)
        for field in dataclass_fields(result.evaluation)
        if isinstance(getattr(result.evaluation, field.name), np.ndarray)
    )
    if any(np.asarray(array).flags.writeable for array in arrays):
        raise ValueError(
            "stabilized mechanics reuse requires read-only mechanics arrays"
        )


@dataclass(frozen=True)
class _StabilizedMechanicsReuse:
    """One same-target/fixed-damage mechanics result held for one next sweep.

    This object is deliberately private and is never passed to an archive,
    result, or workflow layer.  The damage copy/digest and exact static,
    constraint, and load identities prevent a result from crossing a target or
    boundary-condition boundary.  The full result is retained rather than
    just ``q`` so all mechanics diagnostics used by the next ``q_before`` are
    the exact result of the selected candidate.
    """

    target: float
    static_runtime_identity: object
    static_digest: str
    constraints: object
    damage: np.ndarray
    damage_digest: str
    result: A2MechanicsSolveResult

    def __post_init__(self) -> None:
        target = _scalar(self.target, "stabilized mechanics reuse target")
        damage = _f64(self.damage, "stabilized mechanics reuse damage")
        digest = _digest(
            self.damage_digest, "stabilized mechanics reuse damage digest"
        )
        if digest != engineering._damage_digest(damage):
            raise ValueError("stabilized mechanics reuse damage digest is not paired")
        static_digest = _digest(
            self.static_digest, "stabilized mechanics reuse static digest"
        )
        _validate_reusable_mechanics_result(self.result)
        if self.result.constraints is not self.constraints:
            raise ValueError(
                "stabilized mechanics reuse constraints are not the exact bound object"
            )
        if self.result.static_runtime_identity is not self.static_runtime_identity:
            raise ValueError(
                "stabilized mechanics reuse static identity is not paired"
            )
        if self.result.static_digest != static_digest:
            raise ValueError("stabilized mechanics reuse static digest is not paired")
        if self.result.outer_displacement != target:
            raise ValueError("stabilized mechanics reuse load is not paired")
        if self.result.damage_digest != digest:
            raise ValueError("stabilized mechanics reuse result damage is not paired")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "static_digest", static_digest)
        object.__setattr__(self, "damage", _immutable(damage))
        object.__setattr__(self, "damage_digest", digest)

    @classmethod
    def from_selected(
        cls,
        model: EngineeringGripModel,
        target: float,
        d_trial: np.ndarray,
        result: A2MechanicsSolveResult,
    ) -> "_StabilizedMechanicsReuse":
        """Capture one selected result without retaining mutable aliases."""

        damage = _immutable(_f64(d_trial, "selected stabilized trial damage"))
        digest = engineering._damage_digest(damage)
        if result.constraints is not model.constraints:
            raise ValueError(
                "selected stabilized mechanics result has foreign constraints"
            )
        if result.static_runtime_identity is not model.static.runtime_identity:
            raise ValueError(
                "selected stabilized mechanics result has foreign static identity"
            )
        if result.static_digest != model.static.deterministic_digest:
            raise ValueError(
                "selected stabilized mechanics result has a foreign static digest"
            )
        if result.outer_displacement != _scalar(target, "selected stabilized target"):
            raise ValueError("selected stabilized mechanics result has a foreign load")
        if result.damage_digest != digest:
            raise ValueError(
                "selected stabilized mechanics result has a foreign damage digest"
            )
        _validate_reusable_mechanics_result(result)
        return cls(
            target=target,
            static_runtime_identity=model.static.runtime_identity,
            static_digest=model.static.deterministic_digest,
            constraints=model.constraints,
            damage=damage,
            damage_digest=digest,
            result=result,
        )

    def take(
        self,
        model: EngineeringGripModel,
        view: Any,
        target: float,
    ) -> A2MechanicsSolveResult | None:
        """Return the result only for an exact same-target/fixed-damage view."""

        if self.target != _scalar(target, "stabilized reuse target"):
            return None
        if self.static_runtime_identity is not model.static.runtime_identity:
            return None
        if self.static_digest != model.static.deterministic_digest:
            return None
        if self.constraints is not model.constraints:
            return None
        view_damage = _f64(
            view.material_damage,
            "stabilized reuse view damage",
            self.damage.shape,
        )
        if self.damage_digest != view.damage_digest:
            return None
        if not np.array_equal(self.damage, view_damage):
            return None
        result = self.result
        if result.constraints is not model.constraints:
            return None
        if result.static_runtime_identity is not model.static.runtime_identity:
            return None
        if result.static_digest != model.static.deterministic_digest:
            return None
        if result.outer_displacement != target:
            return None
        if result.damage_digest != view.damage_digest:
            return None
        _validate_reusable_mechanics_result(result)
        return result


def _run_stabilized_attempt(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    committed: EngineeringGripState,
    target: float,
    cutback_count: int,
    step_index: int,
    mechanics_adapter: JaxA2CPUReplayAdapter | None,
    *,
    reuse_selected_mechanics: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    anderson_diagnostic_callback: Callable[[Mapping[str, Any]], None]
    | None = None,
    progress_every_sweeps: int | None = None,
    progress_started: float | None = None,
    acceptance_mode: str | None = None,
) -> tuple[StabilizedEngineeringGripAcceptedStep | None, StabilizedEngineeringGripAttempt]:
    base = config.base
    mode = config.acceptance_mode if acceptance_mode is None else _validate_stabilized_acceptance_mode(acceptance_mode)
    anderson_policy = _is_anderson_policy(config.relaxation_policy)
    residual_anderson_policy = _is_anderson_residual_policy(config.relaxation_policy)
    residual_backtracking_policy = _is_anderson_residual_backtracking_policy(
        config.relaxation_policy
    )
    regularized_residual_backtracking_policy = _is_anderson_regularized_policy(
        config.relaxation_policy
    )
    if mode != config.acceptance_mode:
        raise ValueError("stabilized attempt mode does not match configuration")
    if (
        mode == STABILIZED_WARNING_ACCEPTANCE_MODE
        and config.relaxation_policy != "fixed"
    ):
        raise ValueError(
            "stabilized warning acceptance requires the fixed relaxation policy"
        )
    if type(reuse_selected_mechanics) is not bool:
        raise TypeError("reuse_selected_mechanics must be a boolean")
    d_iter = np.array(committed.d, dtype=np.float64, copy=True)
    q_seed = engineering._direct_seed(
        committed.q, committed.u_bar, target, model.constraints
    )
    sweeps: list[StabilizedRelaxationSweepDiagnostics] = []
    mechanics_reuse: _StabilizedMechanicsReuse | None = None
    anderson_history_x: list[np.ndarray] = []
    anderson_history_f: list[np.ndarray] = []
    sweep_started = 0.0
    anderson_coefficient_l2: float | None = None
    anderson_regularization_mu: float | None = (
        config.anderson_regularization_mu
        if regularized_residual_backtracking_policy
        else None
    )
    anderson_regularization_lambda: float | None = None
    anderson_current_residual_l2: float | None = None
    anderson_unregularized_ls_residual_l2: float | None = None
    anderson_proposal_candidate_available = False
    anderson_proposal_fallback_reason: str | None = None
    anderson_proposal_restart_reason: str | None = None

    def retain_sweep(
        diagnostic: StabilizedRelaxationSweepDiagnostics,
        stage_kind: str,
        *,
        cached_q_before: bool,
    ) -> None:
        """Retain a runtime sweep without charging it to archive storage."""

        del stage_kind
        sweeps.append(diagnostic)
        if anderson_diagnostic_callback is not None:
            for event in _anderson_scalar_sweep_events(
                model,
                config,
                diagnostic,
                cached_q_before=cached_q_before,
                coefficient_l2=anderson_coefficient_l2,
                proposal_candidate_available=anderson_proposal_candidate_available,
                proposal_fallback_reason=anderson_proposal_fallback_reason,
                proposal_restart_reason=anderson_proposal_restart_reason,
                mechanics_before_evaluations=0 if cached_q_before else 1,
                sweep_elapsed_seconds=(
                    None
                    if sweep_started == 0.0
                    else time.perf_counter() - sweep_started
                ),
            ):
                _emit_anderson_diagnostic_event(
                    anderson_diagnostic_callback,
                    event,
                )
        if progress_callback is None:
            return
        if progress_every_sweeps is None or progress_started is None:
            raise RuntimeError("stabilized progress settings were not initialized")
        if _progress_sweep_is_due(
            diagnostic.sweep_index,
            progress_every_sweeps,
            base.max_stagger_iterations,
            diagnostic.passed,
        ):
            _emit_progress_event(
                progress_callback,
                _progress_sweep_event(
                    model,
                    config,
                    diagnostic,
                    cached_q_before=cached_q_before,
                    max_sweeps=base.max_stagger_iterations,
                ),
                progress_started,
            )

    for sweep_index in range(base.max_stagger_iterations):
        sweep_started = time.perf_counter()
        anderson_coefficient_l2 = None
        anderson_regularization_lambda = None
        anderson_current_residual_l2 = None
        anderson_unregularized_ls_residual_l2 = None
        anderson_proposal_candidate_available = False
        anderson_proposal_fallback_reason = None
        anderson_proposal_restart_reason = None
        view_before = model.fixed_damage_view(d_iter)
        before: A2MechanicsSolveResult | None = None
        cached_q_before = False
        if reuse_selected_mechanics and mechanics_reuse is not None:
            candidate_reuse = mechanics_reuse
            # A cache is single-use.  A failure after this point must not leave
            # a result available to any later path, even though this attempt
            # returns immediately on all such failures.
            mechanics_reuse = None
            before = candidate_reuse.take(model, view_before, target)
            cached_q_before = before is not None
        if before is None:
            try:
                before = engineering._solve_direct_mechanics(
                    mechanics_adapter,
                    view_before,
                    model.constraints,
                    target,
                    base.cg_rtol,
                    base.cg_atol,
                    base.cg_maxiter,
                    initial_guess=q_seed,
                    seed_kind="stabilized_previous_direct_solution",
                    source_state="stabilized_committed"
                    if sweep_index == 0
                    else "stabilized_complete_trial_sweep",
                    source_step=step_index,
                    source_sweep=sweep_index,
                    source_damage_digest=engineering._damage_digest(committed.d)
                    if sweep_index == 0
                    else engineering._damage_digest(d_iter),
                )
            except (RuntimeError, FloatingPointError) as error:
                mechanics_reuse = None
                base_diag = _incomplete_base(
                    model,
                    config,
                    committed,
                    d_iter,
                    target,
                    "mechanics_before",
                    str(error),
                    sweep_index,
                )
                diag = _wrap_sweep(
                    base_diag,
                    complete_sweep=False,
                    passed=False,
                    failure_stage="mechanics_before",
                    failure_reason=str(error),
                    d_solved=None,
                    d_trial=None,
                    omega=None,
                    reference_merit=None,
                    merit=None,
                    sufficient_decrease_limit=None,
                    fixed_point_error=None,
                    fixed_point_scale=None,
                    fixed_point_pass=False,
                    damage_equation_pass=False,
                    globalization_pass=False,
                    coupled_pass=False,
                    trials=(),
                )
                retain_sweep(
                    diag,
                    "incomplete_sweep",
                    cached_q_before=cached_q_before,
                )
                return _rejected_attempt(
                    committed,
                    target,
                    cutback_count,
                    sweeps,
                    diag,
                    acceptance_mode=mode,
                )

        mechanics_before, _, _, _, force_before_pass, moment_before_pass = engineering._mechanics_gate(
            before, base, engineering._geometry_length(model)
        )
        if not mechanics_before or not force_before_pass or not moment_before_pass:
            mechanics_reuse = None
            reason_parts = []
            if not mechanics_before:
                reason_parts.append("mechanics")
            if not force_before_pass:
                reason_parts.append("force")
            if not moment_before_pass:
                reason_parts.append("moment")
            reason = "mechanics-before " + "/".join(reason_parts) + " gate failed"
            base_diag = _incomplete_base(
                model,
                config,
                committed,
                d_iter,
                target,
                "mechanics_before_gate",
                reason,
                sweep_index,
                before=before,
            )
            diag = _wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="mechanics_before_gate",
                failure_reason=reason,
                d_solved=None,
                d_trial=None,
                omega=None,
                reference_merit=None,
                merit=None,
                sufficient_decrease_limit=None,
                fixed_point_error=None,
                fixed_point_scale=None,
                fixed_point_pass=False,
                damage_equation_pass=False,
                globalization_pass=False,
                coupled_pass=False,
                trials=(),
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        try:
            amor_before = engineering.compute_material_source_amor(
                view_before, before.displacement, committed.H
            )
            psi_before = _f64(
                amor_before.particle_driver,
                "stabilized psi_before",
                (model.n_material,),
                nonfinite_error=FloatingPointError,
            )
            H_used = _immutable(np.maximum(committed.H, psi_before))
        except (RuntimeError, FloatingPointError) as error:
            mechanics_reuse = None
            base_diag = _incomplete_base(
                model,
                config,
                committed,
                d_iter,
                target,
                "amor_before",
                str(error),
                sweep_index,
                before=before,
            )
            diag = _wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="amor_before",
                failure_reason=str(error),
                d_solved=None,
                d_trial=None,
                omega=None,
                reference_merit=None,
                merit=None,
                sufficient_decrease_limit=None,
                fixed_point_error=None,
                fixed_point_scale=None,
                fixed_point_pass=False,
                damage_equation_pass=False,
                globalization_pass=False,
                coupled_pass=False,
                trials=(),
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        try:
            damage_result = engineering.solve_fixed_history_damage(
                model.material_operator, model.Gc, model.l0, H_used
            )
            if not isinstance(damage_result, FixedHistoryDamageResult):
                raise TypeError("stabilized fixed-history solve returned the wrong type")
            d_solved = _f64(
                damage_result.damage,
                "stabilized fixed-history damage",
                (model.n_material,),
                nonfinite_error=FloatingPointError,
            )
            if not np.array_equal(damage_result.history, H_used):
                raise ValueError("stabilized fixed-history result has a different H_used")
            rho_used = _f64(
                damage_result.strong_residual,
                "stabilized fixed-history strong residual",
                d_solved.shape,
                nonfinite_error=FloatingPointError,
            )
            rho_used_volume = _f64(
                damage_result.euclidean_residual,
                "stabilized fixed-history volume residual",
                d_solved.shape,
                nonfinite_error=FloatingPointError,
            )
            engineering._compare_damage_residual_reconstruction(
                rho_used,
                rho_used_volume,
                d_solved,
                H_used,
                model.material_operator.to_scipy().tocsr(),
                model.material_operator.volumes,
                model.Gc,
                model.l0,
                "stabilized fixed-history residual",
            )
            _, _, _, _, solved_scale = engineering._damage_residuals(
                model, d_solved, H_used, H_used
            )
        except (RuntimeError, FloatingPointError) as error:
            mechanics_reuse = None
            base_diag = _incomplete_base(
                model,
                config,
                committed,
                d_iter,
                target,
                "damage",
                str(error),
                sweep_index,
                before=before,
                psi_before=psi_before,
                H_used=H_used,
            )
            diag = _wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="damage",
                failure_reason=str(error),
                d_solved=None,
                d_trial=None,
                omega=None,
                reference_merit=None,
                merit=None,
                sufficient_decrease_limit=None,
                fixed_point_error=None,
                fixed_point_scale=None,
                fixed_point_pass=False,
                damage_equation_pass=False,
                globalization_pass=False,
                coupled_pass=False,
                trials=(),
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        fixed_point_error = float(np.max(np.abs(d_solved - d_iter)))
        fixed_point_scale = max(float(np.max(np.abs(d_solved))), float(np.max(np.abs(d_iter))))
        fixed_pass = _fixed_point_pass(fixed_point_error, fixed_point_scale, config)
        damage_eq_pass = _damage_equation_pass(rho_used, solved_scale, config)
        reference_merit = (
            _anderson_residual_reference_merit(d_iter, d_solved, config)
            if residual_anderson_policy
            else _reference_merit(model, config, d_iter, d_solved, H_used)
        )
        trials: list[StabilizedRelaxationTrial] = []
        selected_diag: StabilizedRelaxationSweepDiagnostics | None = None
        selected_after: A2MechanicsSolveResult | None = None
        last_base_diag: EngineeringGripDiagnostics | None = None
        last_candidate_after: A2MechanicsSolveResult | None = None
        last_candidate_damage: np.ndarray | None = None

        anderson_history_size: int | None = None
        anderson_qr_rank: int | None = None
        anderson_selection: str | None = None
        anderson_fallback_reason: str | None = None
        anderson_restart_reason: str | None = None
        candidate_fp_error: float | None = None
        candidate_fp_scale: float | None = None
        candidate_damage_residual_scale: float | None = None
        candidate_specs: list[
            tuple[str, np.ndarray, float | None, float | None]
        ] = []
        if anderson_policy:
            assert config.anderson_history_depth is not None
            assert config.anderson_beta is not None
            if regularized_residual_backtracking_policy:
                assert config.anderson_regularization_mu is not None
                proposal = _anderson_propose_regularized(
                    anderson_history_x,
                    anderson_history_f,
                    d_iter,
                    d_solved,
                    history_depth=config.anderson_history_depth,
                    beta=config.anderson_beta,
                    regularization_mu=config.anderson_regularization_mu,
                    damage_range_atol=base.damage_range_atol,
                )
            else:
                proposal = _anderson_propose(
                    anderson_history_x,
                    anderson_history_f,
                    d_iter,
                    d_solved,
                    history_depth=config.anderson_history_depth,
                    beta=config.anderson_beta,
                    damage_range_atol=base.damage_range_atol,
                    raw_range_safeguard=not residual_backtracking_policy,
                )
            anderson_history_size = proposal.history_size
            anderson_qr_rank = proposal.qr_rank
            anderson_coefficient_l2 = proposal.anderson_coefficient_l2
            anderson_regularization_mu = proposal.anderson_regularization_mu
            anderson_current_residual_l2 = proposal.anderson_current_residual_l2
            anderson_regularization_lambda = proposal.anderson_regularization_lambda
            anderson_unregularized_ls_residual_l2 = (
                proposal.anderson_unregularized_ls_residual_l2
            )
            anderson_proposal_candidate_available = proposal.candidate is not None
            anderson_proposal_fallback_reason = proposal.fallback_reason
            anderson_proposal_restart_reason = proposal.restart_reason
            anderson_fallback_reason = proposal.fallback_reason
            anderson_restart_reason = proposal.restart_reason
            if proposal.candidate is not None:
                if residual_backtracking_policy:
                    for alpha_index in range(config.max_backtracks + 1):
                        alpha = _anderson_residual_backtracking_alpha(
                            config, alpha_index
                        )
                        candidate_specs.append(
                            (
                                "anderson_accelerated",
                                d_iter + alpha * (proposal.candidate - d_iter),
                                None,
                                alpha,
                            )
                        )
                else:
                    candidate_specs.append(
                        ("anderson_accelerated", proposal.candidate, None, None)
                    )
            candidate_specs.append(
                (
                    "picard_fallback",
                    d_iter + config.anderson_beta * (d_solved - d_iter),
                    config.anderson_beta,
                    None,
                )
            )
            trial_count = len(candidate_specs)
        else:
            trial_count = (
                1
                if config.relaxation_policy == "fixed"
                else config.max_backtracks + 1
            )
        for backtrack_count in range(trial_count):
            candidate_fp_error = None
            candidate_fp_scale = None
            candidate_damage_residual_scale = None
            if anderson_policy:
                (
                    candidate_kind,
                    candidate_values,
                    omega,
                    trial_alpha,
                ) = candidate_specs[backtrack_count]
                try:
                    d_trial = _f64(
                        candidate_values,
                        "stabilized Anderson candidate damage",
                        (model.n_material,),
                        nonfinite_error=FloatingPointError,
                    )
                except (RuntimeError, FloatingPointError) as error:
                    if not residual_backtracking_policy:
                        raise
                    incomplete_diag = _incomplete_base(
                        model,
                        config,
                        committed,
                        d_iter,
                        target,
                        "anderson_candidate_finite",
                        str(error),
                        sweep_index,
                        before=before,
                        psi_before=psi_before,
                        H_used=H_used,
                        rho_used=rho_used,
                        rho_used_volume=rho_used_volume,
                    )
                    trials.append(
                        StabilizedRelaxationTrial(
                            omega=None,
                            complete=False,
                            hard_pass=False,
                            globalization_pass=False,
                            coupled_pass=False,
                            selected=False,
                            merit=None,
                            sufficient_decrease_limit=None,
                            failure_stage="anderson_candidate_finite",
                            failure_reason=str(error),
                            candidate_diagnostics=incomplete_diag,
                            d_trial=None,
                            q_after=None,
                            d_trial_digest=None,
                            q_after_digest=None,
                            candidate_kind=candidate_kind,
                            trial_alpha=trial_alpha,
                            candidate_hard_gate_attribution=(
                                _anderson_residual_hard_gate_record(
                                    incomplete_diag,
                                    config,
                                    mechanics_before=before,
                                    mechanics_after=None,
                                    geometry_length=engineering._geometry_length(model),
                                    hard_gate_status="not_evaluated",
                                )
                                if anderson_diagnostic_callback is not None
                                and residual_anderson_policy
                                else None
                            ),
                        )
                    )
                    anderson_fallback_reason = "anderson_candidate_finite"
                    continue
            else:
                candidate_kind = "picard"
                omega = _relaxation_omega(config, backtrack_count)
                trial_alpha = None
                d_trial = _f64(
                    d_iter + omega * (d_solved - d_iter),
                    "stabilized relaxed damage",
                    (model.n_material,),
                    nonfinite_error=FloatingPointError,
                )
            if config.relaxation_policy == "fixed" or candidate_kind == "picard_fallback":
                sufficient_limit = None
            else:
                sufficient_decrease = config.sufficient_decrease
                assert sufficient_decrease is not None
                if (
                    residual_backtracking_policy
                    and candidate_kind == "anderson_accelerated"
                ):
                    assert trial_alpha is not None
                    sufficient_limit = _anderson_residual_sufficient_decrease_limit(
                        reference_merit,
                        config,
                        trial_alpha=trial_alpha,
                    )
                else:
                    merit_omega = (
                        config.anderson_beta
                        if candidate_kind == "anderson_accelerated"
                        else omega
                    )
                    assert merit_omega is not None
                    sufficient_limit = max(
                        0.0,
                        (1.0 - sufficient_decrease * merit_omega) * reference_merit,
                    )
            view_after = model.fixed_damage_view(d_trial)
            candidate_after: A2MechanicsSolveResult | None = None
            try:
                after = engineering._solve_direct_mechanics(
                    mechanics_adapter,
                    view_after,
                    model.constraints,
                    target,
                    base.cg_rtol,
                    base.cg_atol,
                    base.cg_maxiter,
                    initial_guess=engineering._direct_seed(
                        before.displacement, target, target, model.constraints
                    ),
                    seed_kind="stabilized_mechanics_before_solution",
                    source_state="stabilized_mechanics_before",
                    source_step=step_index,
                    source_sweep=sweep_index,
                    source_damage_digest=engineering._damage_digest(d_iter),
                )
                candidate_after = after
                after_mechanics, _, _, _, _, _ = engineering._mechanics_gate(
                    after, base, engineering._geometry_length(model)
                )
                if not after_mechanics:
                    raise RuntimeError("stabilized mechanics-after gate failed")
                amor_after = engineering.compute_material_source_amor(
                    view_after, after.displacement, committed.H
                )
                psi_after = _f64(
                    amor_after.particle_driver,
                    "stabilized psi_after",
                    (model.n_material,),
                    nonfinite_error=FloatingPointError,
                )
                H_next = _immutable(np.maximum(committed.H, psi_after))
                _, rho_next, _, rho_next_volume, residual_scale = engineering._damage_residuals(
                    model, d_trial, H_used, H_next
                )
                engineering._compare_damage_residual_reconstruction(
                    rho_used,
                    rho_used_volume,
                    d_solved,
                    H_used,
                    model.material_operator.to_scipy().tocsr(),
                    model.material_operator.volumes,
                    model.Gc,
                    model.l0,
                    "stabilized rho_used reconstruction",
                )
                base_diag = engineering._full_diagnostics(
                    base,
                    model,
                    committed,
                    d_iter,
                    d_trial,
                    H_used,
                    H_next,
                    before,
                    after,
                    psi_before,
                    psi_after,
                    rho_used,
                    rho_next,
                    rho_used_volume,
                    rho_next_volume,
                    residual_scale,
                    sweep_index,
                    target,
                )
                last_base_diag = base_diag
                last_candidate_after = after
                last_candidate_damage = np.array(d_trial, dtype=np.float64, copy=True)
                hard_pass = (
                    _anderson_residual_hard_pass(base_diag)
                    if residual_anderson_policy
                    else bool(
                        base_diag.finite_pass
                        and base_diag.mechanics_before_pass
                        and base_diag.mechanics_after_pass
                    )
                )
                if (
                    anderson_policy
                    and not hard_pass
                    and (
                        candidate_kind == "anderson_accelerated"
                        or residual_anderson_policy
                    )
                ):
                    failure_reason = (
                        "accelerated candidate failed hard gates"
                        if candidate_kind == "anderson_accelerated"
                        else "Picard fallback candidate failed hard gates"
                    )
                    incomplete_diag = _incomplete_base(
                        model,
                        config,
                        committed,
                        d_iter,
                        target,
                        "anderson_candidate_hard_gate",
                        failure_reason,
                        sweep_index,
                        before=before,
                        after=after,
                        psi_before=psi_before,
                        psi_after=psi_after,
                        H_used=H_used,
                        d_next=d_trial,
                        rho_used=rho_used,
                        rho_used_volume=rho_used_volume,
                    )
                    trial = StabilizedRelaxationTrial(
                        omega=(
                            None
                            if candidate_kind == "anderson_accelerated"
                            else omega
                        ),
                        complete=False,
                        hard_pass=False,
                        globalization_pass=False,
                        coupled_pass=False,
                        selected=False,
                        merit=None,
                        sufficient_decrease_limit=sufficient_limit,
                        failure_stage="anderson_candidate_hard_gate",
                        failure_reason=failure_reason,
                        candidate_diagnostics=incomplete_diag,
                        d_trial=d_trial,
                        q_after=after.displacement,
                        d_trial_digest=engineering._damage_digest(d_trial),
                        q_after_digest=None,
                        candidate_kind=candidate_kind,
                        trial_alpha=trial_alpha,
                        candidate_hard_gate_attribution=(
                            _anderson_residual_hard_gate_record(
                                base_diag,
                                config,
                                mechanics_before=before,
                                mechanics_after=after,
                                geometry_length=engineering._geometry_length(model),
                            )
                            if anderson_diagnostic_callback is not None
                            and residual_anderson_policy
                            else None
                        ),
                    )
                    trials.append(trial)
                    anderson_fallback_reason = (
                        "accelerated_candidate_hard_gate"
                        if candidate_kind == "anderson_accelerated"
                        else "picard_fallback_hard_gate"
                    )
                    if not residual_backtracking_policy:
                        anderson_restart_reason = anderson_fallback_reason
                    last_base_diag = incomplete_diag
                    last_candidate_after = after
                    last_candidate_damage = np.array(
                        d_trial, dtype=np.float64, copy=True
                    )
                    if candidate_kind == "anderson_accelerated":
                        continue
                    break
                if anderson_policy and candidate_kind == "anderson_accelerated":
                    try:
                        candidate_map, _ = _validate_anderson_candidate_map(
                            model, H_next
                        )
                        with np.errstate(over="raise", invalid="raise"):
                            candidate_fp_error = engineering._max_abs(
                                candidate_map - d_trial
                            )
                        candidate_fp_scale = max(
                            engineering._max_abs(candidate_map),
                            engineering._max_abs(d_trial),
                        )
                        candidate_damage_residual_scale = (
                            _candidate_damage_residual_scale(model, d_trial, H_next)
                        )
                        if not np.isfinite(candidate_fp_error) or not np.isfinite(
                            candidate_fp_scale
                        ):
                            raise FloatingPointError(
                                "Anderson candidate fixed-point metric is non-finite"
                            )
                        merit = (
                            _anderson_residual_candidate_merit(
                                candidate_fp_error,
                                candidate_fp_scale,
                                config,
                            )
                            if residual_anderson_policy
                            else _anderson_candidate_merit(
                                base_diag,
                                candidate_fp_error,
                                candidate_fp_scale,
                                candidate_damage_residual_scale,
                                config,
                            )
                        )
                    except (RuntimeError, FloatingPointError) as error:
                        failure_reason = str(error)
                        incomplete_diag = _incomplete_base(
                            model,
                            config,
                            committed,
                            d_iter,
                            target,
                            "anderson_candidate_damage",
                            failure_reason,
                            sweep_index,
                            before=before,
                            after=after,
                            psi_before=psi_before,
                            psi_after=psi_after,
                            H_used=H_used,
                            d_next=d_trial,
                            rho_used=rho_used,
                            rho_used_volume=rho_used_volume,
                        )
                        trial = StabilizedRelaxationTrial(
                            omega=None,
                            complete=False,
                            hard_pass=False,
                            globalization_pass=False,
                            coupled_pass=False,
                            selected=False,
                            merit=None,
                            sufficient_decrease_limit=sufficient_limit,
                            failure_stage="anderson_candidate_damage",
                            failure_reason=failure_reason,
                            candidate_diagnostics=incomplete_diag,
                            d_trial=d_trial,
                            q_after=after.displacement,
                            d_trial_digest=engineering._damage_digest(d_trial),
                            q_after_digest=None,
                            candidate_kind=candidate_kind,
                            trial_alpha=trial_alpha,
                            candidate_hard_gate_attribution=(
                                _anderson_residual_hard_gate_record(
                                    base_diag,
                                    config,
                                    mechanics_before=before,
                                    mechanics_after=after,
                                    geometry_length=engineering._geometry_length(model),
                                    hard_gate_status="candidate_map_failure",
                                )
                                if anderson_diagnostic_callback is not None
                                and residual_anderson_policy
                                else None
                            ),
                        )
                        trials.append(trial)
                        anderson_fallback_reason = "anderson_candidate_damage"
                        if not residual_backtracking_policy:
                            anderson_restart_reason = anderson_fallback_reason
                        last_base_diag = incomplete_diag
                        last_candidate_after = after
                        last_candidate_damage = np.array(
                            d_trial, dtype=np.float64, copy=True
                        )
                        continue
                elif residual_anderson_policy and candidate_kind == "picard_fallback":
                    # The exact fallback has no candidate committed-history map
                    # evaluation.  Do not label a composite transition diagnostic
                    # as the residual-policy selection merit.
                    merit = None
                else:
                    merit = _candidate_merit(
                        base_diag, fixed_point_error, fixed_point_scale, config
                    )
                if config.relaxation_policy == "fixed" or candidate_kind == "picard_fallback":
                    # Fixed damping is an explicit Picard policy, not a
                    # globalization test.  The candidate is selected after
                    # the complete hard solve; its merit remains diagnostic.
                    globalization_pass = hard_pass
                else:
                    assert sufficient_limit is not None
                    globalization_pass = bool(
                        hard_pass and merit <= sufficient_limit
                    )
                coupled_pass = _coupled_pass(base_diag, fixed_pass, damage_eq_pass)
                trial = StabilizedRelaxationTrial(
                    omega=omega,
                    complete=True,
                    hard_pass=hard_pass,
                    globalization_pass=globalization_pass,
                    coupled_pass=coupled_pass,
                    selected=globalization_pass,
                    merit=merit,
                    sufficient_decrease_limit=sufficient_limit,
                    failure_stage=""
                    if globalization_pass
                    else "globalization",
                    failure_reason=""
                    if globalization_pass
                    else (
                        "accelerated candidate failed hard gates"
                        if candidate_kind == "anderson_accelerated" and not hard_pass
                        else "sufficient-decrease merit condition failed"
                    ),
                    candidate_diagnostics=base_diag,
                    d_trial=d_trial,
                    q_after=after.displacement,
                    d_trial_digest=engineering._damage_digest(d_trial),
                    q_after_digest=_trial_q_after_digest(
                        target, after.displacement, d_trial, H_next
                    ),
                    candidate_kind=candidate_kind,
                    trial_alpha=trial_alpha,
                    candidate_fp_error=candidate_fp_error,
                    candidate_fp_scale=candidate_fp_scale,
                    candidate_damage_residual_scale=candidate_damage_residual_scale,
                    candidate_hard_gate_attribution=(
                        _anderson_residual_hard_gate_record(
                            base_diag,
                            config,
                            mechanics_before=before,
                            mechanics_after=after,
                            geometry_length=engineering._geometry_length(model),
                        )
                        if anderson_diagnostic_callback is not None
                        and residual_anderson_policy
                        else None
                    ),
                )
                if (
                    anderson_policy
                    and candidate_kind == "anderson_accelerated"
                    and not globalization_pass
                ):
                    trials.append(trial)
                    anderson_fallback_reason = (
                        "accelerated_candidate_hard_gate"
                        if not hard_pass
                        else "insufficient_decrease"
                    )
                    if not residual_backtracking_policy:
                        anderson_restart_reason = anderson_fallback_reason
                    continue
                trials.append(trial)
                if globalization_pass:
                    if anderson_policy:
                        anderson_selection = (
                            "accelerated"
                            if candidate_kind == "anderson_accelerated"
                            else "picard_fallback"
                        )
                        if anderson_selection == "accelerated":
                            # Earlier rejected alpha trials remain in the trial
                            # audit, but a selected accelerated row cannot carry
                            # a fallback reason in the public sweep contract.
                            anderson_fallback_reason = None
                        if (
                            anderson_selection == "picard_fallback"
                            and anderson_fallback_reason is None
                        ):
                            anderson_fallback_reason = "accelerated_candidate_rejected"
                    candidate_pass = coupled_pass
                    anderson_strict = anderson_policy and mode == STABILIZED_ACCEPTANCE_MODE
                    if anderson_strict:
                        candidate_pass = bool(
                            candidate_pass
                            and base_diag.damage_range_pass
                            and base_diag.healing_pass
                        )
                    if mode == STABILIZED_WARNING_ACCEPTANCE_MODE:
                        # Warning mode must not classify a raw range/healing
                        # defect as a strict pass.  The strict stabilized path
                        # intentionally keeps its historical coupled-gate
                        # definition unchanged.
                        candidate_pass = bool(
                            coupled_pass
                            and base_diag.damage_range_pass
                            and base_diag.healing_pass
                        )
                    failure_reason = _stabilized_failure_reason(
                        base_diag,
                        fixed_pass,
                        damage_eq_pass,
                        coupled_pass,
                        include_raw_gates=(
                            mode == STABILIZED_WARNING_ACCEPTANCE_MODE
                            or anderson_strict
                        ),
                    )
                    selected_diag = _wrap_sweep(
                        base_diag,
                        complete_sweep=True,
                        passed=candidate_pass,
                        failure_stage="" if candidate_pass else "gates",
                        failure_reason="" if candidate_pass else failure_reason,
                        d_solved=d_solved,
                        d_trial=d_trial,
                        omega=omega,
                        reference_merit=reference_merit,
                        merit=merit,
                        sufficient_decrease_limit=sufficient_limit,
                        fixed_point_error=fixed_point_error,
                        fixed_point_scale=fixed_point_scale,
                        fixed_point_pass=fixed_pass,
                        damage_equation_pass=damage_eq_pass,
                        globalization_pass=True,
                        coupled_pass=coupled_pass,
                        trials=trials,
                        anderson_history_size=anderson_history_size,
                        anderson_qr_rank=anderson_qr_rank,
                        anderson_coefficient_l2=anderson_coefficient_l2,
                        anderson_regularization_mu=anderson_regularization_mu,
                        anderson_current_residual_l2=anderson_current_residual_l2,
                        anderson_regularization_lambda=anderson_regularization_lambda,
                        anderson_unregularized_ls_residual_l2=(
                            anderson_unregularized_ls_residual_l2
                        ),
                        anderson_selection=anderson_selection,
                        trial_alpha=trial_alpha,
                        anderson_fallback_reason=anderson_fallback_reason,
                        anderson_restart_reason=anderson_restart_reason,
                        candidate_fp_error=candidate_fp_error,
                        candidate_fp_scale=candidate_fp_scale,
                        candidate_damage_residual_scale=(
                            candidate_damage_residual_scale
                        ),
                    )
                    selected_after = after
                    break
            except (RuntimeError, FloatingPointError) as error:
                mechanics_reuse = None
                last_base_diag = _incomplete_base(
                    model,
                    config,
                    committed,
                    d_iter,
                    target,
                    "mechanics_after",
                    str(error),
                    sweep_index,
                    before=before,
                    after=candidate_after,
                    psi_before=psi_before,
                    H_used=H_used,
                    d_next=d_trial,
                    rho_used=rho_used,
                    rho_used_volume=rho_used_volume,
                )
                last_candidate_after = candidate_after
                last_candidate_damage = np.array(d_trial, dtype=np.float64, copy=True)
                trial = StabilizedRelaxationTrial(
                    omega=omega,
                    complete=False,
                    hard_pass=False,
                    globalization_pass=False,
                    coupled_pass=False,
                    selected=False,
                    merit=None,
                    sufficient_decrease_limit=sufficient_limit,
                    failure_stage="mechanics_after",
                    failure_reason=str(error),
                    candidate_diagnostics=last_base_diag,
                    d_trial=d_trial,
                    q_after=None
                    if candidate_after is None
                    else candidate_after.displacement,
                    d_trial_digest=engineering._damage_digest(d_trial),
                    q_after_digest=None,
                    candidate_kind=candidate_kind,
                    trial_alpha=trial_alpha,
                    candidate_hard_gate_attribution=(
                                _anderson_residual_hard_gate_record(
                                    last_base_diag,
                                    config,
                                    mechanics_before=before,
                                    mechanics_after=candidate_after,
                                    geometry_length=engineering._geometry_length(model),
                                    hard_gate_status="not_evaluated",
                                )
                        if anderson_diagnostic_callback is not None
                        and residual_anderson_policy
                        else None
                    ),
                )
                trials.append(trial)
                if (
                    anderson_policy
                    and candidate_kind == "anderson_accelerated"
                ):
                    anderson_fallback_reason = (
                        f"accelerated_candidate_{last_base_diag.failure_stage}"
                    )
                    if not residual_backtracking_policy:
                        anderson_restart_reason = anderson_fallback_reason
                    continue

            if anderson_policy:
                if candidate_kind == "anderson_accelerated":
                    continue
                break
            if backtrack_count >= config.max_backtracks:
                break
            next_omega = _relaxation_omega(config, backtrack_count + 1)
            if next_omega == omega:
                break

        if selected_diag is None:
            # No candidate satisfied the policy selection condition.  Preserve
            # the last finite candidate's base fields when available, but do
            # not update d_iter or the committed state.
            mechanics_reuse = None
            selection_failure_reason = (
                "fixed relaxation candidate failed hard gates"
                if config.relaxation_policy == "fixed"
                else (
                    "Anderson Picard fallback candidate failed hard gates"
                    if anderson_policy
                    else "no relaxation candidate satisfied sufficient decrease"
                )
            )
            if last_base_diag is None:
                base_diag = _incomplete_base(
                    model,
                    config,
                    committed,
                    d_iter,
                    target,
                    "globalization",
                    selection_failure_reason,
                    sweep_index,
                    before=before,
                    psi_before=psi_before,
                    H_used=H_used,
                    rho_used=rho_used,
                    rho_used_volume=rho_used_volume,
                )
                last_candidate_damage = None
            else:
                base_diag = last_base_diag
            diag = _wrap_sweep(
                base_diag,
                complete_sweep=False,
                passed=False,
                failure_stage="globalization",
                failure_reason=selection_failure_reason,
                d_solved=d_solved,
                d_trial=last_candidate_damage,
                omega=trials[-1].omega if trials else None,
                reference_merit=reference_merit,
                merit=trials[-1].merit if trials else None,
                sufficient_decrease_limit=(
                    trials[-1].sufficient_decrease_limit if trials else None
                ),
                fixed_point_error=fixed_point_error,
                fixed_point_scale=fixed_point_scale,
                fixed_point_pass=fixed_pass,
                damage_equation_pass=damage_eq_pass,
                globalization_pass=False,
                coupled_pass=False,
                trials=trials,
                anderson_history_size=anderson_history_size,
                anderson_qr_rank=anderson_qr_rank,
                anderson_coefficient_l2=anderson_coefficient_l2,
                anderson_regularization_mu=anderson_regularization_mu,
                anderson_current_residual_l2=anderson_current_residual_l2,
                anderson_regularization_lambda=anderson_regularization_lambda,
                anderson_unregularized_ls_residual_l2=(
                    anderson_unregularized_ls_residual_l2
                ),
                anderson_selection=(
                    "picard_fallback"
                    if anderson_policy
                    else anderson_selection
                ),
                trial_alpha=(
                    trials[-1].trial_alpha
                    if trials and trials[-1].candidate_kind == "anderson_accelerated"
                    else None
                ),
                anderson_fallback_reason=anderson_fallback_reason,
                anderson_restart_reason=anderson_restart_reason,
            )
            retain_sweep(
                diag,
                "incomplete_sweep",
                cached_q_before=cached_q_before,
            )
            return _rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                diag,
                acceptance_mode=mode,
            )

        warning_acceptance = False
        warning_bypassed: tuple[str, ...] = ()
        warning_range_excess: float | None = None
        warning_healing_magnitude: float | None = None
        if (
            mode == STABILIZED_WARNING_ACCEPTANCE_MODE
            and not selected_diag.passed
            and sweep_index + 1 == base.max_stagger_iterations
        ):
            (
                warning_acceptance,
                warning_bypassed,
                warning_range_excess,
                warning_healing_magnitude,
            ) = _stabilized_warning_acceptance_decision(selected_diag)

        if anderson_policy:
            assert config.anderson_history_depth is not None
            if anderson_restart_reason is not None:
                anderson_history_x.clear()
                anderson_history_f.clear()
            anderson_history_x.append(
                _immutable(np.asarray(d_iter, dtype=np.float64))
            )
            anderson_history_f.append(
                _immutable(np.asarray(d_solved - d_iter, dtype=np.float64))
            )
            if len(anderson_history_x) > config.anderson_history_depth:
                del anderson_history_x[:-config.anderson_history_depth]
                del anderson_history_f[:-config.anderson_history_depth]

        if selected_diag.passed or warning_acceptance:
            mechanics_reuse = None
            retain_sweep(
                selected_diag,
                "complete_sweep",
                cached_q_before=cached_q_before,
            )
            assert selected_after is not None
            assert selected_diag.d_trial is not None
            assert selected_diag.H_next is not None
            state = EngineeringGripState(
                u_bar=target,
                union_displacement=selected_after.displacement,
                material_damage=selected_diag.d_trial,
                material_history=selected_diag.H_next,
            )
            step = StabilizedEngineeringGripAcceptedStep(
                load_start=committed.u_bar,
                load_target=target,
                increment=target - committed.u_bar,
                state=state,
                diagnostics=selected_diag,
                stagger_iterations=sweep_index + 1,
                cutback_count=cutback_count,
                acceptance_status=(
                    STABILIZED_WARNING_ACCEPTANCE_STATUS
                    if warning_acceptance
                    else STABILIZED_ACCEPTANCE_STATUS
                ),
                bypassed_gates=warning_bypassed,
                warning_range_excess=warning_range_excess,
                warning_healing_magnitude=warning_healing_magnitude,
                acceptance_mode=mode,
            )
            attempt = StabilizedEngineeringGripAttempt(
                cutback_count=cutback_count,
                load_start=committed.u_bar,
                load_target=target,
                accepted=True,
                acceptance_status=(
                    STABILIZED_WARNING_ACCEPTANCE_STATUS
                    if warning_acceptance
                    else STABILIZED_ACCEPTANCE_STATUS
                ),
                failure_stage=None,
                failure_reason=None,
                exception_type=None,
                committed_before_digest=committed.digest,
                committed_after_digest=state.digest,
                sweeps=tuple(sweeps),
                diagnostics=selected_diag,
                bypassed_gates=warning_bypassed,
                warning_range_excess=warning_range_excess,
                warning_healing_magnitude=warning_healing_magnitude,
                acceptance_mode=mode,
            )
            return step, attempt

        retain_sweep(
            selected_diag,
            "complete_sweep",
            cached_q_before=cached_q_before,
        )
        if sweep_index + 1 == base.max_stagger_iterations:
            mechanics_reuse = None
            return _rejected_attempt(
                committed,
                target,
                cutback_count,
                sweeps,
                selected_diag,
                acceptance_mode=mode,
            )
        assert selected_after is not None
        d_iter = np.array(selected_diag.d_trial, dtype=np.float64, copy=True)
        q_seed = engineering._direct_seed(
            selected_after.displacement, target, target, model.constraints
        )
        if reuse_selected_mechanics:
            mechanics_reuse = _StabilizedMechanicsReuse.from_selected(
                model,
                target,
                d_iter,
                selected_after,
            )
        else:
            mechanics_reuse = None

    raise RuntimeError("stabilized attempt produced no final diagnostic")


def _rejected_attempt(
    committed: EngineeringGripState,
    target: float,
    cutback_count: int,
    sweeps: Sequence[StabilizedRelaxationSweepDiagnostics],
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    *,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
) -> tuple[None, StabilizedEngineeringGripAttempt]:
    attempt = StabilizedEngineeringGripAttempt(
        cutback_count=cutback_count,
        load_start=committed.u_bar,
        load_target=target,
        accepted=False,
        acceptance_status=None,
        failure_stage=diagnostic.failure_stage or "gates",
        failure_reason=diagnostic.failure_reason or "stabilized sweep failed",
        exception_type=StabilizedEngineeringGripSweepFailure.__name__,
        committed_before_digest=committed.digest,
        committed_after_digest=committed.digest,
        sweeps=tuple(sweeps),
        diagnostics=diagnostic,
        acceptance_mode=_validate_stabilized_acceptance_mode(acceptance_mode),
    )
    return None, attempt


def _validate_progress_settings(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_every_sweeps: int | None,
) -> int | None:
    """Validate the transient progress callback/cadence pair."""

    if progress_callback is None:
        if progress_every_sweeps is not None:
            raise ValueError(
                "progress_every_sweeps requires progress_callback"
            )
        return None
    if not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    if progress_every_sweeps is None:
        raise ValueError(
            "progress_every_sweeps is required when progress_callback is provided"
        )
    return engineering._integer(
        progress_every_sweeps,
        "progress_every_sweeps",
        positive=True,
    )


def run_stabilized_engineering_grip(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    mechanics_adapter: JaxA2CPUReplayAdapter | None = None,
    *,
    archive_budget_bytes: int | None = None,
    archive_output_path: Any | None = None,
    archive_free_bytes: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    anderson_diagnostic_callback: Callable[[Mapping[str, Any]], None]
    | None = None,
    progress_every_sweeps: int | None = None,
    _reuse_selected_mechanics: bool = True,
) -> StabilizedEngineeringGripResult:
    """Run the isolated convergence-first fresh-zero continuation."""

    if not isinstance(model, EngineeringGripModel) or not isinstance(
        config, StabilizedEngineeringGripConfig
    ):
        raise TypeError("run_stabilized_engineering_grip requires model/config types")
    if config.load_schedule is not None:
        # This is a one-time launch-boundary check.  No schedule file or source
        # archive is read from the numerical loop below.
        config.load_schedule._validate_provenance()
    progress_every = _validate_progress_settings(
        progress_callback, progress_every_sweeps
    )
    if anderson_diagnostic_callback is not None:
        if not _is_anderson_policy(config.relaxation_policy):
            raise ValueError(
                "anderson_diagnostic_callback requires an Anderson relaxation policy"
            )
        if not callable(anderson_diagnostic_callback):
            raise TypeError("anderson_diagnostic_callback must be callable")
    started = time.perf_counter()
    if type(_reuse_selected_mechanics) is not bool:
        raise TypeError("_reuse_selected_mechanics must be a boolean")
    if archive_budget_bytes is None and (
        archive_output_path is not None or archive_free_bytes is not None
    ):
        raise ValueError(
            "archive_output_path/archive_free_bytes require archive_budget_bytes"
        )
    adapter = engineering._validated_mechanics_adapter(model, mechanics_adapter)
    try:
        work_check = engineering.run_zero_damage_work_check(
            model,
            config.base.work_check_load,
            config.base.work_epsilon,
            config.base.work_relative_limit,
            config.base.cg_rtol,
            config.base.cg_atol,
            config.base.cg_maxiter,
            mechanics_adapter=adapter,
        )
    except Exception as error:
        _emit_failure_progress(
            progress_callback,
            reason=str(error),
            committed_load=0.0,
            target_load=None,
            accepted_steps=0,
            attempts=0,
            failure_type=type(error).__name__,
            started=started,
        )
        raise
    initial = EngineeringGripState(
        u_bar=0.0,
        union_displacement=np.zeros(model.n_dofs, dtype=np.float64),
        material_damage=np.zeros(model.n_material, dtype=np.float64),
        material_history=np.zeros(model.n_material, dtype=np.float64),
    )
    archive_budget = (
        None
        if archive_budget_bytes is None
        else _StabilizedArchiveBudget(
            model,
            config,
            archive_budget_bytes,
            archive_output_path=archive_output_path,
            free_bytes=archive_free_bytes,
        )
    )
    states = [initial]
    steps: list[StabilizedEngineeringGripAcceptedStep] = []
    attempts: list[StabilizedEngineeringGripAttempt] = []
    committed = initial
    schedule = config.load_schedule
    increment = config.base.du_initial
    cutbacks = 0
    while (
        len(steps) < schedule.target_count
        if schedule is not None
        else committed.u_bar < config.base.load_final
    ):
        if schedule is not None:
            target = float(schedule.targets[len(steps)])
            cutbacks = 0
        else:
            target = engineering._next_target(
                committed.u_bar, increment, config.base.load_final
            )
        if target <= committed.u_bar:
            message = "stabilized continuation made no load progress"
            _emit_failure_progress(
                progress_callback,
                reason=message,
                committed_load=committed.u_bar,
                target_load=target,
                accepted_steps=len(steps),
                attempts=len(attempts),
                failure_type=StabilizedEngineeringGripContinuationFailure.__name__,
                started=started,
            )
            raise StabilizedEngineeringGripContinuationFailure(
                message,
                committed,
                tuple(attempts),
                work_check,
                failure_code=_stabilized_failure_code(
                    config.acceptance_mode, schedule=schedule is not None
                ),
                acceptance_mode=config.acceptance_mode,
            )
        if archive_budget is not None:
            retained_counts = {"accepted_steps": len(steps)}
            try:
                archive_budget.check(
                    accepted_steps=retained_counts["accepted_steps"] + 1,
                    stage=f"step={len(steps) + 1} accepted_step",
                    stage_kind="accepted_step",
                    stage_details={
                        "step_index": len(steps) + 1,
                        "target_u_bar": target,
                    },
                    retained_counts=retained_counts,
                )
            except _StabilizedArchiveBudgetExceeded as exhausted:
                failure = StabilizedEngineeringGripArchiveBudgetFailure(
                    str(exhausted),
                    committed,
                    tuple(attempts),
                    work_check,
                    stage=exhausted.stage,
                    stage_kind=exhausted.stage_kind,
                    stage_details=exhausted.stage_details,
                    exhaustion_kind=exhausted.exhaustion_kind,
                    retained_counts=exhausted.retained_counts,
                    prospective_counts=exhausted.prospective_counts,
                    retained_estimate=exhausted.retained_estimate,
                    estimate=exhausted.estimate,
                    archive_budget_bytes=exhausted.archive_budget_bytes,
                    available_bytes=exhausted.available_bytes,
                    acceptance_mode=config.acceptance_mode,
                )
                _emit_failure_progress(
                    progress_callback,
                    reason=str(failure),
                    committed_load=committed.u_bar,
                    target_load=target,
                    accepted_steps=len(steps),
                    attempts=len(attempts),
                    failure_type=type(failure).__name__,
                    started=started,
                )
                raise failure from exhausted
        _emit_progress_event(
            progress_callback,
            {
                "event": "attempt_start",
                "step_index": len(steps) + 1,
                "committed_load": committed.u_bar,
                "target_load": target,
                "increment": target - committed.u_bar,
                "cutback_count": cutbacks,
                "max_sweeps": config.base.max_stagger_iterations,
                "omega": config.omega_initial,
                "policy": config.relaxation_policy,
            },
            started,
        )
        try:
            step, attempt = _run_stabilized_attempt(
                model,
                config,
                committed,
                target,
                cutbacks,
                len(steps) + 1,
                adapter,
                reuse_selected_mechanics=_reuse_selected_mechanics,
                progress_callback=progress_callback,
                anderson_diagnostic_callback=anderson_diagnostic_callback,
                progress_every_sweeps=progress_every,
                progress_started=started,
                acceptance_mode=config.acceptance_mode,
            )
        except _StabilizedArchiveBudgetExceeded as exhausted:
            failure = StabilizedEngineeringGripArchiveBudgetFailure(
                str(exhausted),
                committed,
                tuple(attempts),
                work_check,
                stage=exhausted.stage,
                stage_kind=exhausted.stage_kind,
                stage_details=exhausted.stage_details,
                exhaustion_kind=exhausted.exhaustion_kind,
                retained_counts=exhausted.retained_counts,
                prospective_counts=exhausted.prospective_counts,
                retained_estimate=exhausted.retained_estimate,
                estimate=exhausted.estimate,
                archive_budget_bytes=exhausted.archive_budget_bytes,
                available_bytes=exhausted.available_bytes,
                acceptance_mode=config.acceptance_mode,
            )
            _emit_failure_progress(
                progress_callback,
                reason=str(failure),
                committed_load=committed.u_bar,
                target_load=target,
                accepted_steps=len(steps),
                attempts=len(attempts),
                failure_type=type(failure).__name__,
                started=started,
            )
            raise failure from exhausted
        attempts.append(attempt)
        if step is not None:
            steps.append(step)
            committed = step.state
            states.append(committed)
            if schedule is not None:
                if step.cutback_count != 0 or step.load_target != target:
                    raise RuntimeError(
                        "scheduled stabilized accepted step is not paired with its target"
                    )
                continue
            increment = step.increment
            cutbacks = 0
            if progress_callback is not None:
                _emit_progress_event(
                    progress_callback,
                    _progress_accepted_event(model, config, step),
                    started,
                )
            continue
        if attempt.committed_before_digest != attempt.committed_after_digest:
            raise RuntimeError("stabilized rejected attempt changed committed state")
        if schedule is not None:
            message = (
                "scheduled stabilized continuation failed at exact target "
                f"u_bar={target:.17g}: {attempt.failure_reason}"
            )
            _emit_failure_progress(
                progress_callback,
                reason=message,
                committed_load=committed.u_bar,
                target_load=target,
                accepted_steps=len(steps),
                attempts=len(attempts),
                failure_type=StabilizedEngineeringGripContinuationFailure.__name__,
                started=started,
            )
            raise StabilizedEngineeringGripContinuationFailure(
                message,
                committed,
                tuple(attempts),
                work_check,
                failure_code=_stabilized_failure_code(
                    config.acceptance_mode, schedule=True
                ),
                acceptance_mode=config.acceptance_mode,
            )
        at_minimum = engineering._is_minimum_increment(
            target - committed.u_bar,
            config.base.du_min,
            committed.u_bar,
            target,
        )
        at_cutback_limit = cutbacks >= config.base.max_cutbacks
        if at_minimum or at_cutback_limit:
            reasons = []
            if at_minimum:
                reasons.append("du_min")
            if at_cutback_limit:
                reasons.append("max_cutbacks")
            message = (
                "stabilized continuation failed at "
                f"u_bar={target:.17g} ({', '.join(reasons)}): {attempt.failure_reason}"
            )
            _emit_failure_progress(
                progress_callback,
                reason=message,
                committed_load=committed.u_bar,
                target_load=target,
                accepted_steps=len(steps),
                attempts=len(attempts),
                failure_type=StabilizedEngineeringGripContinuationFailure.__name__,
                started=started,
            )
            raise StabilizedEngineeringGripContinuationFailure(
                message,
                committed,
                tuple(attempts),
                work_check,
                failure_code=_stabilized_failure_code(config.acceptance_mode),
                acceptance_mode=config.acceptance_mode,
            )
        old_increment = target - committed.u_bar
        cutbacks += 1
        increment = max(
            config.base.du_min,
            old_increment * config.base.cutback_factor,
        )
        _emit_progress_event(
            progress_callback,
            {
                "event": "cutback",
                "rejected_target": target,
                "reason": attempt.failure_reason,
                "old_increment": old_increment,
                "new_increment": increment,
                "cutback_count": cutbacks,
                "committed_load": committed.u_bar,
            },
            started,
        )
    result = StabilizedEngineeringGripResult(
        model=model,
        config=config,
        work_check=work_check,
        initial_state=initial,
        states=tuple(states),
        accepted_steps=tuple(steps),
        attempts=tuple(attempts),
        _acceptance_mode=config.acceptance_mode,
        completion_status=_stabilized_contract(
            config.acceptance_mode,
            failure=False,
            schedule=schedule is not None,
        )["completion_status"],
    )
    _emit_progress_event(
        progress_callback,
        {
            "event": "completed",
            "load": result.final_state.u_bar,
            "accepted_steps": len(result.accepted_steps),
            "attempts": len(result.attempts),
        },
        started,
    )
    return result


def _write_npz(path: Any, arrays: Mapping[str, np.ndarray]) -> None:
    filename = os.fsdecode(os.fspath(path))
    if not filename:
        raise ValueError("stabilized archive path must be non-empty")
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez_compressed(filename, **dict(arrays))  # type: ignore[arg-type]


def _prepare_archive_path(path: Any) -> Path:
    filename = os.fsdecode(os.fspath(path))
    if not filename:
        raise ValueError("stabilized archive path must be non-empty")
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _v5_string_vector(
    values: Mapping[str, np.ndarray], name: str, length: int | None = None
) -> np.ndarray:
    if name not in values:
        raise ValueError(f"v5 archive is missing {name}")
    result = np.asarray(values[name])
    if result.dtype.kind not in "SU" or result.ndim != 1:
        raise ValueError(f"v5 archive {name} must be a one-dimensional string vector")
    if length is not None and result.shape != (length,):
        raise ValueError(f"v5 archive {name} has the wrong length")
    return result


def _v5_json_record(value: Any, name: str) -> dict[str, Any]:
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"v5 archive {name} is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"v5 archive {name} must contain a mapping")
    try:
        canonical = json.dumps(parsed, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"v5 archive {name} contains non-scalar JSON") from error
    if canonical != text:
        raise ValueError(f"v5 archive {name} is not canonical JSON")
    return dict(parsed)


def _v5_json_records(
    values: Mapping[str, np.ndarray], name: str
) -> list[dict[str, Any]]:
    raw = _v5_string_vector(values, name)
    return [_v5_json_record(item, f"{name}[{index}]") for index, item in enumerate(raw)]


def _v5_validate_scalar_tree(value: Any, name: str) -> None:
    """Reject numerical vectors from a scalar log while allowing string lists."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _v5_validate_scalar_tree(item, f"{name}.{key}")
        return
    if isinstance(value, list):
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"v5 scalar log {name} contains a numerical vector")
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"v5 scalar log {name} contains a non-finite value")


def _v5_number(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"v5 {name} must be a scalar number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"v5 {name} must be finite")
    return result


def _v5_integer(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"v5 {name} must be a JSON integer")
    return int(value)


_V5_DIAGNOSTIC_BOOL_FIELDS = tuple(engineering._DIAGNOSTIC_BOOL_FIELDS)
_V5_SOLVER_SUMMARY_SCALAR_FIELDS = (
    "energy_before",
    "energy_after",
    "top_reaction_before",
    "top_reaction_after",
    "bottom_reaction_before",
    "bottom_reaction_after",
    "anchor_reaction_before",
    "anchor_reaction_after",
    "moment_before",
    "moment_after",
    "free_residual_norm_before",
    "free_residual_norm_after",
    "free_residual_relative_before",
    "free_residual_relative_after",
    "cg_info_before",
    "cg_info_after",
    "e_u_material",
    "e_u_union",
    "e_damage",
    "e_history",
    "e_reaction",
    "damage_min",
    "damage_max",
    "healing_min",
    "history_min_gap",
    "rho_used_inf",
    "rho_next_inf",
    "displacement_scale",
    "damage_scale",
    "history_scale",
    "reaction_scale",
    "damage_residual_scale",
    "force_error",
    "force_scale",
    "moment_error",
    "moment_scale",
)
_V5_SOLVER_SUMMARY_BOOL_FIELDS = (
    *_V5_DIAGNOSTIC_BOOL_FIELDS,
    "fixed_point_pass",
    "damage_equation_pass",
    "globalization_pass",
    "coupled_pass",
)
_V5_SOLVER_SUMMARY_GATE_NAMES = (
    *engineering._WARNING_HARD_GATE_NAMES,
    *engineering.WARNING_GATE_NAMES,
    "damage_fixed_point",
    "damage_equation",
    "globalization",
    "coupled",
)
_ANDERSON_SOLVER_SUMMARY_FIELDS = {
    "anderson_history_size",
    "anderson_qr_rank",
    "anderson_selection",
    "anderson_fallback_reason",
    "anderson_restart_reason",
    "candidate_fp_error",
    "candidate_fp_scale",
    "candidate_damage_residual_scale",
}
_ANDERSON_RESIDUAL_SOLVER_SUMMARY_FIELDS = {
    "anderson_merit_identity",
}
_ANDERSON_RESIDUAL_BACKTRACKING_SOLVER_SUMMARY_FIELDS = {
    "trial_alpha",
}
_ANDERSON_REGULARIZED_SOLVER_SUMMARY_FIELDS = {
    "anderson_regularization_mu",
    "anderson_current_residual_l2",
    "anderson_regularization_lambda",
    "anderson_unregularized_ls_residual_l2",
    "anderson_coefficient_l2",
}
_ANDERSON_PROVENANCE_FIELDS = {
    "status",
    "selection_recorded",
    "selection_reconstructed",
    "fallback_recorded",
    "fallback_reconstructed",
    "restart_recorded",
    "restart_reconstructed",
    "history_rank_recorded",
    "history_rank_reconstructed",
}
_V5_SOLVER_SUMMARY_FIELDS = {
    "summary_kind",
    "state_index",
    "load_start",
    "load_target",
    "increment",
    "cutback_count",
    "stagger_iterations",
    "relaxation_policy",
    "acceptance_status",
    "accepted",
    "sweep_index",
    "target_u_bar",
    "complete_sweep",
    "passed",
    "failure_stage",
    "failure_reason",
    "gate_passes",
    "omega",
    "backtrack_count",
    "reference_merit",
    "merit",
    "sufficient_decrease_limit",
    "fixed_point_error",
    "fixed_point_scale",
    "fixed_point_pass",
    "damage_equation_pass",
    "globalization_pass",
    "coupled_pass",
    *_V5_SOLVER_SUMMARY_SCALAR_FIELDS,
    *_V5_DIAGNOSTIC_BOOL_FIELDS,
}


def _v5_solver_summary_fields(relaxation_policy: str) -> set[str]:
    if relaxation_policy not in STABILIZED_RELAXATION_POLICIES:
        raise ValueError(
            "relaxation_policy must be one of "
            f"{STABILIZED_RELAXATION_POLICIES}, got {relaxation_policy!r}"
        )
    return set(_V5_SOLVER_SUMMARY_FIELDS) | (
        set(_ANDERSON_SOLVER_SUMMARY_FIELDS)
        if _is_anderson_policy(relaxation_policy)
        else set()
    ) | (
        set(_ANDERSON_RESIDUAL_SOLVER_SUMMARY_FIELDS)
        if _is_anderson_residual_policy(relaxation_policy)
        else set()
    ) | (
        set(_ANDERSON_RESIDUAL_BACKTRACKING_SOLVER_SUMMARY_FIELDS)
        if _is_anderson_residual_backtracking_policy(relaxation_policy)
        else set()
    ) | (
        set(_ANDERSON_REGULARIZED_SOLVER_SUMMARY_FIELDS)
        if _is_anderson_regularized_policy(relaxation_policy)
        else set()
    )


def _v5_anderson_provenance(relaxation_policy: str) -> dict[str, Any]:
    """Describe which Anderson scalars are recorded versus reconstructed."""

    if relaxation_policy not in STABILIZED_RELAXATION_POLICIES:
        raise ValueError(
            "relaxation_policy must be one of "
            f"{STABILIZED_RELAXATION_POLICIES}, got {relaxation_policy!r}"
        )
    applicable = _is_anderson_policy(relaxation_policy)
    return {
        "status": (
            "recorded_not_reconstructed" if applicable else "not_applicable"
        ),
        "selection_recorded": applicable,
        "selection_reconstructed": False,
        "fallback_recorded": applicable,
        "fallback_reconstructed": False,
        "restart_recorded": applicable,
        "restart_reconstructed": False,
        "history_rank_recorded": applicable,
        "history_rank_reconstructed": False,
    }


def _v5_validate_anderson_provenance(
    value: Any, relaxation_policy: str, name: str
) -> dict[str, Any]:
    """Validate one lean replay provenance report without trusting its claims."""

    expected = _v5_anderson_provenance(relaxation_policy)
    if not isinstance(value, Mapping) or set(value) != _ANDERSON_PROVENANCE_FIELDS:
        raise ValueError(f"{name} has an invalid Anderson provenance structure")
    if any(type(value[field]) is not type(expected[field]) for field in value):
        raise ValueError(f"{name} Anderson provenance booleans/status are mistyped")
    if dict(value) != expected:
        raise ValueError(f"{name} Anderson provenance is inconsistent")
    return dict(expected)


def _v5_validate_solver_summary(
    payload: Mapping[str, Any],
    name: str,
    *,
    relaxation_policy: str | None = None,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    mode = _validate_stabilized_acceptance_mode(acceptance_mode)
    warning = mode == STABILIZED_WARNING_ACCEPTANCE_MODE
    if "relaxation_policy" not in payload:
        raise ValueError(f"v5 {name}.relaxation_policy is missing")
    row_policy = payload["relaxation_policy"]
    if not isinstance(row_policy, str) or row_policy not in STABILIZED_RELAXATION_POLICIES:
        raise ValueError(f"v5 {name}.relaxation_policy is invalid")
    expected_fields = _v5_solver_summary_fields(row_policy)
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        extra = sorted(set(payload) - expected_fields)
        raise ValueError(
            f"v5 {name} summary fields differ: missing={missing}, extra={extra}"
        )
    kind = payload["summary_kind"]
    if kind not in {"initial", "accepted", "failure"}:
        raise ValueError(f"v5 {name}.summary_kind is invalid")
    state_index = _v5_integer(payload["state_index"], f"{name}.state_index")
    assert state_index is not None
    if kind == "initial" and state_index != 0:
        raise ValueError(f"v5 {name} initial state index is invalid")
    if kind != "initial" and state_index < -1:
        raise ValueError(f"v5 {name}.state_index is invalid")
    for field in (
        "load_start",
        "load_target",
        "increment",
        "target_u_bar",
        "omega",
        "reference_merit",
        "merit",
        "sufficient_decrease_limit",
        "fixed_point_error",
        "fixed_point_scale",
    ):
        _v5_number(payload[field], f"{name}.{field}", allow_none=True)
    for field in ("cutback_count", "stagger_iterations", "backtrack_count"):
        value = _v5_integer(payload[field], f"{name}.{field}")
        assert value is not None
        if value < 0:
            raise ValueError(f"v5 {name}.{field} is negative")
        if (
            field == "backtrack_count"
            and _is_anderson_residual_backtracking_policy(row_policy)
            and value > STABILIZED_ANDERSON_MAX_BACKTRACKS
        ):
            raise ValueError(f"v5 {name}.backtrack_count exceeds the alpha schedule")
    sweep_index = _v5_integer(payload["sweep_index"], f"{name}.sweep_index")
    assert sweep_index is not None
    if kind == "initial" and sweep_index != -1:
        raise ValueError(f"v5 {name} initial sweep index is invalid")
    if kind != "initial" and sweep_index < 0:
        raise ValueError(f"v5 {name}.sweep_index is negative")
    for field in ("accepted", "complete_sweep", "passed", *_V5_SOLVER_SUMMARY_BOOL_FIELDS):
        if type(payload[field]) is not bool:
            raise ValueError(f"v5 {name}.{field} must be a JSON boolean")
    gates = payload["gate_passes"]
    if not isinstance(gates, Mapping) or set(gates) != set(_V5_SOLVER_SUMMARY_GATE_NAMES):
        raise ValueError(f"v5 {name}.gate_passes is invalid")
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError(f"v5 {name}.gate_passes values are invalid")
    for field in ("failure_stage", "failure_reason"):
        if not isinstance(payload[field], str):
            raise ValueError(f"v5 {name}.{field} must be a string")
    status = payload["acceptance_status"]
    if status is not None and not isinstance(status, str):
        raise ValueError(f"v5 {name}.acceptance_status must be a string or null")
    if relaxation_policy is not None and row_policy != relaxation_policy:
        raise ValueError(f"v5 {name}.relaxation_policy does not match configuration")
    if _is_anderson_policy(row_policy):
        if _is_anderson_residual_policy(row_policy):
            if payload["anderson_merit_identity"] != (
                _anderson_merit_identity(row_policy)
            ):
                raise ValueError(f"v5 {name} residual merit identity is invalid")
        history_size = _v5_integer(
            payload["anderson_history_size"],
            f"{name}.anderson_history_size",
            allow_none=True,
        )
        qr_rank = _v5_integer(
            payload["anderson_qr_rank"],
            f"{name}.anderson_qr_rank",
            allow_none=True,
        )
        if history_size is not None and history_size < 0:
            raise ValueError(f"v5 {name}.anderson_history_size is negative")
        if qr_rank is not None and qr_rank < 0:
            raise ValueError(f"v5 {name}.anderson_qr_rank is negative")
        if (
            history_size is not None
            and qr_rank is not None
            and qr_rank > history_size
        ):
            raise ValueError(f"v5 {name}.anderson_qr_rank exceeds history size")
        selection = payload["anderson_selection"]
        if kind == "initial":
            if selection != "initial":
                raise ValueError(f"v5 {name}.anderson_selection is invalid")
        elif kind == "accepted":
            if not isinstance(selection, str) or selection not in _ANDERSON_SELECTIONS:
                raise ValueError(f"v5 {name}.anderson_selection is invalid")
        elif selection is not None and (
            not isinstance(selection, str) or selection not in _ANDERSON_SELECTIONS
        ):
            raise ValueError(f"v5 {name}.anderson_selection is invalid")
        for field in ("anderson_fallback_reason", "anderson_restart_reason"):
            value = payload[field]
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"v5 {name}.{field} is invalid")
        candidate_metrics = tuple(
            _v5_number(payload[field], f"{name}.{field}", allow_none=True)
            for field in (
                "candidate_fp_error",
                "candidate_fp_scale",
                "candidate_damage_residual_scale",
            )
        )
        residual_merits: tuple[float | None, ...] = ()
        if any(value is not None and value < 0.0 for value in candidate_metrics):
            raise ValueError(f"v5 {name} Anderson candidate metrics are negative")
        trial_alpha: float | None = None
        if _is_anderson_residual_backtracking_policy(row_policy):
            trial_alpha = _v5_number(
                payload["trial_alpha"], f"{name}.trial_alpha", allow_none=True
            )
            valid_alphas = tuple(
                STABILIZED_ANDERSON_ALPHA_REDUCTION**index
                for index in range(STABILIZED_ANDERSON_MAX_BACKTRACKS + 1)
            )
            if trial_alpha is not None and trial_alpha not in valid_alphas:
                raise ValueError(f"v5 {name}.trial_alpha is not on the configured schedule")
        if _is_anderson_residual_policy(row_policy):
            residual_merits = tuple(
                _v5_number(payload[field], f"{name}.{field}", allow_none=True)
                for field in ("reference_merit", "merit")
            )
            if any(value is not None and value < 0.0 for value in residual_merits):
                raise ValueError(f"v5 {name} residual merits are negative")
        if selection == "accelerated" and payload["anderson_fallback_reason"] is not None:
            raise ValueError(f"v5 {name} accelerated row retains a fallback reason")
        if selection == "picard_fallback" and payload["anderson_fallback_reason"] is None:
            raise ValueError(f"v5 {name} Picard fallback lacks a reason")
        if _is_anderson_residual_backtracking_policy(row_policy):
            if selection == "accelerated" and trial_alpha is None:
                raise ValueError(f"v5 {name} accelerated row lacks trial_alpha")
            if selection != "accelerated" and trial_alpha is not None:
                raise ValueError(f"v5 {name} non-accelerated row retains trial_alpha")
            if selection == "accelerated":
                expected_alpha = _anderson_backtracking_alpha_for_index(
                    payload["backtrack_count"],
                    f"v5 {name}.backtrack_count",
                )
                if trial_alpha != expected_alpha:
                    raise ValueError(
                        f"v5 {name}.trial_alpha is not paired with backtrack_count"
                    )
        if _is_anderson_regularized_policy(row_policy):
            summary_mu = _v5_number(
                payload["anderson_regularization_mu"],
                f"{name}.anderson_regularization_mu",
                allow_none=True,
            )
            if summary_mu is None or summary_mu <= 0.0:
                raise ValueError(
                    f"v5 {name}.anderson_regularization_mu is missing or invalid"
                )
            expected_mu = (
                None
                if anderson_regularization_mu is None
                else _v5_number(
                    anderson_regularization_mu,
                    "configured Anderson regularization mu",
                )
            )
            if expected_mu is not None and summary_mu != expected_mu:
                raise ValueError(
                    f"v5 {name}.anderson_regularization_mu does not match configuration"
                )
            current_residual_l2 = _v5_number(
                payload["anderson_current_residual_l2"],
                f"{name}.anderson_current_residual_l2",
                allow_none=True,
            )
            regularization_lambda = _v5_number(
                payload["anderson_regularization_lambda"],
                f"{name}.anderson_regularization_lambda",
                allow_none=True,
            )
            unregularized_ls_residual_l2 = _v5_number(
                payload["anderson_unregularized_ls_residual_l2"],
                f"{name}.anderson_unregularized_ls_residual_l2",
                allow_none=True,
            )
            coefficient_l2 = _v5_number(
                payload["anderson_coefficient_l2"],
                f"{name}.anderson_coefficient_l2",
                allow_none=True,
            )
            if any(
                value is not None and value < 0.0
                for value in (
                    current_residual_l2,
                    regularization_lambda,
                    unregularized_ls_residual_l2,
                    coefficient_l2,
                )
            ):
                raise ValueError(f"v5 {name} regularization metrics are negative")
            if (regularization_lambda is None) != (current_residual_l2 is None):
                raise ValueError(
                    f"v5 {name} regularization lambda/current residual availability is inconsistent"
                )
            if (
                regularization_lambda is not None
                and current_residual_l2 is not None
                and not _anderson_scalar_relation_close(
                    regularization_lambda,
                    summary_mu * current_residual_l2**2,
                )
            ):
                raise ValueError(
                    f"v5 {name} regularization lambda is not paired with current residual l2"
                )
            if kind == "initial" and any(
                value is not None
                for value in (
                    current_residual_l2,
                    regularization_lambda,
                    unregularized_ls_residual_l2,
                    coefficient_l2,
                )
            ):
                raise ValueError(
                    f"v5 {name} initial regularization metrics are not null"
                )
            if kind != "initial" and selection in (
                "accelerated",
                "picard_fallback",
            ) and (
                current_residual_l2 is None or regularization_lambda is None
            ):
                raise ValueError(
                    f"v5 {name} selected regularized row lacks current residual/lambda"
                )
        if selection == "accelerated":
            if _is_anderson_residual_policy(row_policy) and not payload["complete_sweep"]:
                if any(value is not None for value in candidate_metrics):
                    raise ValueError(
                        f"v5 {name} incomplete residual row retains candidate metrics"
                    )
            else:
                if any(value is None for value in candidate_metrics):
                    raise ValueError(f"v5 {name} accelerated row lacks candidate metrics")
                if _is_anderson_residual_policy(row_policy) and any(
                    value is None for value in residual_merits
                ):
                    raise ValueError(
                        f"v5 {name} accelerated residual row lacks merit metrics"
                    )
            if payload["omega"] is not None:
                raise ValueError(f"v5 {name} accelerated row retains omega")
            if (
                (payload["complete_sweep"] or not _is_anderson_residual_policy(row_policy))
                and payload["sufficient_decrease_limit"] is None
            ):
                raise ValueError(f"v5 {name} accelerated row lacks a sufficient limit")
        elif selection == "picard_fallback":
            if payload["omega"] != STABILIZED_ANDERSON_BETA:
                raise ValueError(f"v5 {name} Picard fallback omega is invalid")
            if payload["sufficient_decrease_limit"] is not None:
                raise ValueError(f"v5 {name} Picard fallback retains a sufficient limit")
            if _is_anderson_residual_policy(row_policy):
                if payload["merit"] is not None or any(
                    value is not None for value in candidate_metrics
                ):
                    raise ValueError(
                        f"v5 {name} residual Picard fallback retains selection metrics"
                    )
        elif kind == "initial" and (
            history_size != 0
            or qr_rank != 0
            or payload["anderson_fallback_reason"] is not None
            or payload["anderson_restart_reason"] is not None
            or any(value is not None for value in candidate_metrics)
        ):
            raise ValueError(f"v5 {name} initial Anderson provenance is inconsistent")
        elif selection != "accelerated" and any(
            value is not None for value in candidate_metrics
        ):
            raise ValueError(
                f"v5 {name} non-accelerated row retains candidate metrics"
            )
        if kind == "accepted" and (
            history_size is None or qr_rank is None
        ):
            raise ValueError(f"v5 {name} accepted Anderson rank fields are missing")
    if row_policy == "fixed" and payload["complete_sweep"]:
        if payload["backtrack_count"] != 0:
            raise ValueError("v5 fixed relaxation summary contains backtracking")
        if payload["sufficient_decrease_limit"] is not None:
            raise ValueError(
                "v5 fixed relaxation summary contains a sufficient-decrease limit"
            )
    elif row_policy == "sufficient_decrease" and payload["complete_sweep"]:
        if payload["sufficient_decrease_limit"] is None:
            raise ValueError(
                "v5 sufficient-decrease summary is missing its sufficient limit"
            )
    if kind == "initial":
        if (
            payload["acceptance_status"] != "initial"
            or not payload["accepted"]
            or payload["state_index"] != 0
            or payload["load_start"] != 0.0
            or payload["load_target"] != 0.0
            or payload["increment"] != 0.0
            or payload["target_u_bar"] != 0.0
            or payload["complete_sweep"]
            or not payload["passed"]
            or payload["failure_stage"]
            or payload["failure_reason"]
            or payload["stagger_iterations"] != 0
            or payload["cutback_count"] != 0
            or payload["backtrack_count"] != 0
            or any(gates for gates in payload["gate_passes"].values())
        ):
            raise ValueError(f"v5 {name} initial summary is inconsistent")
    elif kind == "accepted":
        if (
            payload["acceptance_status"]
            not in (
                (STABILIZED_ACCEPTANCE_STATUS, STABILIZED_WARNING_ACCEPTANCE_STATUS)
                if warning
                else (STABILIZED_ACCEPTANCE_STATUS,)
            )
            or not payload["accepted"]
            or payload["state_index"] <= 0
            or payload["load_target"] <= payload["load_start"]
            or payload["increment"] != payload["load_target"] - payload["load_start"]
            or payload["target_u_bar"] != payload["load_target"]
            or not payload["complete_sweep"]
            or payload["failure_stage"]
            or payload["failure_reason"]
            or payload["stagger_iterations"] <= 0
        ):
            raise ValueError(f"v5 {name} accepted summary is inconsistent")
        if not warning and (
            not payload["passed"]
            or not payload["fixed_point_pass"]
            or not payload["damage_equation_pass"]
            or not payload["globalization_pass"]
            or not payload["coupled_pass"]
        ):
            raise ValueError(f"v5 {name} accepted summary is inconsistent")
        if not warning and _is_anderson_policy(row_policy) and (
            not payload["gate_passes"]["damage_range"]
            or not payload["gate_passes"]["healing"]
        ):
            raise ValueError(
                f"v5 {name} Anderson accepted summary fails range/healing gates"
            )
        if warning and payload["acceptance_status"] == STABILIZED_WARNING_ACCEPTANCE_STATUS:
            if payload["passed"]:
                raise ValueError(f"v5 {name} warning summary must retain passed=False")
            gates = payload["gate_passes"]
            if any(not gates[name] for name in STABILIZED_WARNING_HARD_GATE_NAMES):
                raise ValueError(f"v5 {name} warning summary fails a hard gate")
            if all(gates[name] for name in STABILIZED_WARNING_GATE_NAMES):
                raise ValueError(f"v5 {name} warning summary has no bypassed gate")
        elif warning and (
            payload["acceptance_status"] != STABILIZED_ACCEPTANCE_STATUS
            or not payload["passed"]
            or not payload["fixed_point_pass"]
            or not payload["damage_equation_pass"]
            or not payload["globalization_pass"]
            or not payload["coupled_pass"]
        ):
            raise ValueError(f"v5 {name} strict summary is inconsistent")
    else:
        if (
            kind != "failure"
            or payload["state_index"] != -1
            or payload["accepted"]
            or payload["passed"]
            or payload["acceptance_status"] is not None
            or not payload["failure_stage"]
            or not payload["failure_reason"]
        ):
            raise ValueError(f"v5 {name} failure summary is inconsistent")
        if _is_anderson_policy(row_policy):
            selection = payload["anderson_selection"]
            if selection == "picard_fallback" and payload["anderson_fallback_reason"] is None:
                raise ValueError(f"v5 {name} failure fallback lacks a reason")
            if selection == "accelerated" and payload["anderson_fallback_reason"] is not None:
                raise ValueError(f"v5 {name} failure acceleration retains fallback reason")
    for field in _V5_SOLVER_SUMMARY_SCALAR_FIELDS:
        if field.startswith("cg_info_"):
            _v5_integer(payload[field], f"{name}.{field}", allow_none=True)
        else:
            _v5_number(payload[field], f"{name}.{field}", allow_none=True)
    _v5_validate_scalar_tree(payload, name)
    return dict(payload)


def _v5_validate_header(
    values: Mapping[str, np.ndarray],
    *,
    schedule: bool | None = None,
) -> tuple[
    dict[str, Any],
    engineering.EngineeringGripConfig,
    StabilizedEngineeringGripConfig,
    bool,
    EngineeringGripModel,
    dict[str, np.ndarray],
]:
    schema = _archive_string(values, "archive_schema")
    warning = schema in {
        STABILIZED_WARNING_ARCHIVE_SCHEMA,
        STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
    }
    failure = schema in {
        STABILIZED_FAILURE_ARCHIVE_SCHEMA,
        STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
        STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA,
    }
    schedule_schema = schema in {
        STABILIZED_SCHEDULE_ARCHIVE_SCHEMA,
        STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA,
    }
    if schedule is None:
        schedule = schedule_schema
    if schedule != schedule_schema:
        raise ValueError("schedule archive schema was not paired with its validator")
    mode = (
        STABILIZED_WARNING_ACCEPTANCE_MODE
        if warning
        else STABILIZED_ACCEPTANCE_MODE
    )
    contract = _stabilized_contract(mode, failure=failure, schedule=schedule)
    expected_schema = contract["archive_schema"]
    if schema != expected_schema:
        raise ValueError("v5 stabilized archive schema mismatch")
    if _archive_string(values, "acceptance_mode") != mode:
        raise ValueError("v5 stabilized archive acceptance mode is forged")
    if _archive_string(values, "acceptance_status") != contract["acceptance_status"]:
        raise ValueError("v5 stabilized archive acceptance status is forged")
    expected_completion = contract["completion_status"]
    if _archive_string(values, "completion_status") != expected_completion:
        raise ValueError("v5 stabilized archive completion status is forged")
    if _archive_bool(values, "validated") or not _archive_bool(values, "experimental"):
        raise ValueError("v5 stabilized archive validation flags are forged")
    metadata_text = _archive_string(values, "metadata_json")
    metadata = _archive_json(values, "metadata_json")
    if _v5_json_text(metadata) != metadata_text:
        raise ValueError("v5 stabilized metadata JSON is not canonical")
    if metadata.get("schema") != schema or metadata.get("layout") != contract["archive_layout"]:
        raise ValueError("v5 stabilized metadata identity is forged")
    if metadata.get("kind") != ("failure" if failure else "completed"):
        raise ValueError("v5 stabilized metadata kind is forged")
    if metadata.get("replay_contract") != contract["replay_contract"]:
        raise ValueError("v5 stabilized replay contract is forged")
    if metadata.get("validated") is not False or metadata.get("experimental") is not True:
        raise ValueError("v5 stabilized metadata validation flags are forged")
    if metadata.get("field_equations_validated") is not False:
        raise ValueError("v5 stabilized field-equation metadata is forged")
    if metadata.get("path_replayed") is not False or metadata.get(
        "rejected_trial_path_available"
    ) is not False:
        raise ValueError("v5 stabilized path-replay metadata is forged")
    if metadata.get("acceptance_mode") != mode or metadata.get(
        "acceptance_status"
    ) != contract["acceptance_status"]:
        raise ValueError("v5 stabilized metadata acceptance fields are forged")
    if metadata.get("completion_status") != expected_completion:
        raise ValueError("v5 stabilized metadata completion status is forged")
    if metadata.get("visibility_rule") != engineering._VISIBILITY_RULE or metadata.get(
        "crack_region_rule"
    ) != engineering._CRACK_REGION_RULE:
        raise ValueError("v5 stabilized metadata formulation rules are forged")
    if metadata.get("crack_localization_atol") != 1.0e-10:
        raise ValueError("v5 stabilized localization tolerance is forged")
    if metadata.get("code_hash_files") != list(_STABILIZED_SOURCE_BUNDLE_FILES):
        raise ValueError("v5 stabilized source-bundle file list is forged")
    if metadata.get("code_hash_kind") != engineering._SOURCE_BUNDLE_HASH_KIND:
        raise ValueError("v5 stabilized source-bundle hash meaning is forged")
    if _digest(metadata.get("code_hash"), "v5 metadata code_hash") != _current_stabilized_source_bundle_hash():
        raise ValueError("v5 stabilized source-bundle hash does not match replay")
    for name in (
        "material_hash",
        "union_hash",
        "static_digest",
        "source_sha256",
        "configuration_sha256",
    ):
        _digest(metadata.get(name), f"v5 metadata {name}")
    raw_counts = metadata.get("archive_counts")
    if not isinstance(raw_counts, Mapping):
        raise TypeError("v5 metadata archive_counts must be a mapping")
    _archive_counts(raw_counts, name="v5 metadata archive_counts")
    source_metadata = metadata.get("source")
    if not isinstance(source_metadata, Mapping):
        raise ValueError("v5 stabilized source metadata is missing")
    if source_metadata.get("sha256") != metadata.get("source_sha256"):
        raise ValueError("v5 stabilized source hash alias is forged")
    base_payload = metadata.get("config")
    stable_payload = metadata.get("stabilized_config")
    if not isinstance(base_payload, Mapping) or not isinstance(stable_payload, Mapping):
        raise ValueError("v5 stabilized configuration payload is missing")
    base = engineering._config_from_payload(base_payload)
    config = StabilizedEngineeringGripConfig.from_payload(stable_payload, base)
    if schedule:
        _validate_schedule_archive_binding(values, metadata, config)
    elif config.load_schedule is not None:
        raise ValueError("historical v5 archive contains an explicit load schedule")
    canonical_base = _v5_json_value(engineering._config_payload(base))
    canonical_stable = _v5_json_value(config.payload())
    if base_payload != canonical_base or stable_payload != canonical_stable:
        raise ValueError("v5 stabilized configuration payload is not canonical")
    if metadata.get("configuration_sha256") != _v5_json_sha256(
        {"base": canonical_base, "stabilized": canonical_stable}
    ):
        raise ValueError("v5 stabilized configuration hash is forged")
    if warning:
        if metadata.get("warning_hard_gate_names") != list(
            STABILIZED_WARNING_HARD_GATE_NAMES
        ) or metadata.get("warning_bypassable_gate_names") != list(
            STABILIZED_WARNING_GATE_NAMES
        ):
            raise ValueError("stabilized warning gate partition is forged")
    else:
        forbidden = {
            "warning_hard_gate_names",
            "warning_bypassable_gate_names",
        }
        if any(name in metadata for name in forbidden):
            raise ValueError("strict stabilized metadata contains warning gate fields")
    archive_budget = metadata.get("archive_budget_bytes")
    if archive_budget is not None:
        _validate_archive_budget(archive_budget)
    if "archive_storage_preflight" in metadata:
        storage = metadata["archive_storage_preflight"]
        canonical_storage = _archive_storage_metadata(
            storage, archive_budget_bytes=archive_budget
        )
        if storage != canonical_storage:
            raise ValueError("v5 stabilized storage preflight metadata is not canonical")
    if failure:
        if metadata.get("failure_type") not in {
            StabilizedEngineeringGripContinuationFailure.__name__,
            StabilizedEngineeringGripArchiveBudgetFailure.__name__,
        }:
            raise ValueError("v5 stabilized failure type is forged")
        if not isinstance(metadata.get("failure_message"), str) or not metadata.get(
            "failure_message"
        ):
            raise ValueError("v5 stabilized failure message is missing")
        if metadata.get("failure_code") not in {
            _stabilized_failure_code(mode, schedule=schedule),
            STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE,
        }:
            raise ValueError("v5 stabilized failure code is forged")
    model, source = _v5_rebuild_model(metadata)
    if config.acceptance_mode != mode:
        raise ValueError("stabilized archive configuration mode is forged")
    return metadata, base, config, failure, model, source


def _v5_replay_work_check(
    values: Mapping[str, np.ndarray],
    static: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: engineering.EngineeringGripConfig,
) -> dict[str, Any]:
    load = _scalar(values["work_check_u_bar"], "v5 replay work-check load")
    epsilon = _scalar(values["work_check_epsilon"], "v5 replay work-check epsilon")
    limit = _scalar(
        values["work_check_relative_limit"], "v5 replay work-check limit"
    )
    if (
        load != config.work_check_load
        or epsilon != config.work_epsilon
        or limit != config.work_relative_limit
    ):
        raise ValueError("v5 work-check settings do not match the configuration")
    center = engineering._replay_direct_solve(
        np.zeros(static["n_material"], dtype=np.float64),
        load,
        static["ba"],
        static["material_phase"],
        static["grip_phase"],
        static["volume_bottom"],
        static["volume_top"],
        0,
        _scalar(metadata["youngs_modulus"], "v5 replay youngs_modulus"),
        _scalar(metadata["poisson_ratio"], "v5 replay poisson_ratio"),
        _scalar(metadata["k_res"], "v5 replay k_res"),
        static["n_material"],
        cg_rtol=config.cg_rtol,
        cg_atol=config.cg_atol,
        cg_maxiter=config.cg_maxiter,
    )
    plus = engineering._replay_direct_solve(
        np.zeros(static["n_material"], dtype=np.float64),
        load + epsilon,
        static["ba"],
        static["material_phase"],
        static["grip_phase"],
        static["volume_bottom"],
        static["volume_top"],
        0,
        _scalar(metadata["youngs_modulus"], "v5 replay youngs_modulus"),
        _scalar(metadata["poisson_ratio"], "v5 replay poisson_ratio"),
        _scalar(metadata["k_res"], "v5 replay k_res"),
        static["n_material"],
        cg_rtol=config.cg_rtol,
        cg_atol=config.cg_atol,
        cg_maxiter=config.cg_maxiter,
    )
    minus = engineering._replay_direct_solve(
        np.zeros(static["n_material"], dtype=np.float64),
        load - epsilon,
        static["ba"],
        static["material_phase"],
        static["grip_phase"],
        static["volume_bottom"],
        static["volume_top"],
        0,
        _scalar(metadata["youngs_modulus"], "v5 replay youngs_modulus"),
        _scalar(metadata["poisson_ratio"], "v5 replay poisson_ratio"),
        _scalar(metadata["k_res"], "v5 replay k_res"),
        static["n_material"],
        cg_rtol=config.cg_rtol,
        cg_atol=config.cg_atol,
        cg_maxiter=config.cg_maxiter,
    )
    reaction_fd = float(
        (plus["response"]["energy"] - minus["response"]["energy"])
        / (2.0 * epsilon)
    )
    relative_error = engineering._relative(reaction_fd, center["top"])
    if relative_error > limit:
        raise ValueError("v5 replay work check fails")
    _compare_scalar(
        values["work_check_reaction"],
        center["top"],
        "v5 replay work reaction",
        2.0e-8,
    )
    _compare_scalar(
        values["work_check_reaction_fd"],
        reaction_fd,
        "v5 replay work reaction FD",
        2.0e-8,
    )
    _compare_scalar(
        values["work_check_relative_error"],
        relative_error,
        "v5 replay work relative error",
        2.0e-8,
    )
    if not _archive_bool(values, "work_check_passed"):
        raise ValueError("v5 replay work-check status is forged")
    return {
        "u_bar": load,
        "epsilon": epsilon,
        "reaction": float(center["top"]),
        "reaction_fd": reaction_fd,
        "relative_error": relative_error,
        "relative_limit": limit,
        "passed": True,
    }


def _v5_compare_diagnostic_scalar(
    payload: Mapping[str, Any], name: str, expected: Any
) -> None:
    _compare_scalar(payload.get(name), expected, f"v5 replay {name}", 2.0e-8)


def _v5_replay_states(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    base_config: engineering.EngineeringGripConfig,
    stabilized_config: StabilizedEngineeringGripConfig,
    model: EngineeringGripModel,
    static: Mapping[str, Any],
    failure: bool,
) -> dict[str, Any]:
    n_states = len(_f64(values["state_u_bar"], "v5 replay state loads"))
    if n_states == 0:
        raise ValueError("v5 replay requires at least the initial state")
    state_u = _f64(values["state_u_bar"], "v5 replay state loads", (n_states,))
    state_q = _f64(
        values["state_union_displacement"],
        "v5 replay state displacement",
        (n_states, model.n_dofs),
    )
    state_d = _f64(
        values["state_material_damage"],
        "v5 replay state damage",
        (n_states, model.n_material),
    )
    state_H = _f64(
        values["state_material_history"],
        "v5 replay state history",
        (n_states, model.n_material),
    )
    state_digest = _v5_string_vector(values, "state_digest", n_states)
    state_status = _v5_string_vector(values, "state_acceptance_status", n_states)
    state_work = _f64(values["state_external_work"], "v5 replay external work", (n_states,))
    accepted_count = n_states - 1
    accepted_top = _f64(
        values["accepted_top_reaction_after"],
        "v5 replay accepted top reactions",
        (accepted_count,),
    )
    accepted_bottom = _f64(
        values["accepted_bottom_reaction_after"],
        "v5 replay accepted bottom reactions",
        (accepted_count,),
    )
    accepted_anchor = _f64(
        values["accepted_anchor_reaction_after"],
        "v5 replay accepted anchor reactions",
        (accepted_count,),
    )
    summaries = _v5_json_records(values, "solver_summary_json")
    if len(summaries) != n_states:
        raise ValueError("v5 replay solver-summary count is inconsistent")
    initial_summary = _v5_validate_solver_summary(
        summaries[0],
        "v5 initial solver summary",
        relaxation_policy=stabilized_config.relaxation_policy,
        acceptance_mode=stabilized_config.acceptance_mode,
        anderson_regularization_mu=stabilized_config.anderson_regularization_mu,
    )
    if initial_summary["summary_kind"] != "initial":
        raise ValueError("v5 replay initial solver summary kind is forged")
    if str(state_status[0]) != "initial" or initial_summary["acceptance_status"] != "initial":
        raise ValueError("v5 replay initial acceptance status is forged")
    if (
        state_u[0] != 0.0
        or not np.array_equal(state_q[0], np.zeros(model.n_dofs, dtype=np.float64))
        or not np.array_equal(state_d[0], np.zeros(model.n_material, dtype=np.float64))
        or not np.array_equal(state_H[0], np.zeros(model.n_material, dtype=np.float64))
        or state_work[0] != 0.0
    ):
        raise ValueError("v5 replay initial state is not exact zero")
    if np.any(np.diff(state_u) <= 0.0):
        raise ValueError("v5 replay state loads are not strictly increasing")
    if failure:
        if not state_u[-1] < base_config.load_final:
            raise ValueError("v5 replay failure state reaches load_final")
    elif state_u[-1] != base_config.load_final:
        raise ValueError("v5 replay completed state does not reach load_final")
    if np.any(state_H < 0.0):
        raise ValueError("v5 replay committed history is negative")
    expected_digests = np.asarray(
        [
            engineering._state_digest(
                state_u[index], state_q[index], state_d[index], state_H[index]
            )
            for index in range(n_states)
        ],
        dtype=np.str_,
    )
    if not np.array_equal(state_digest, expected_digests):
        raise ValueError("v5 replay state digests are forged")
    expected_work = _v5_external_work(
        state_u,
        np.concatenate((np.asarray([0.0], dtype=np.float64), accepted_top)),
    )
    _array_close(state_work, expected_work, "v5 replay cumulative external work", 2.0e-8)
    top_dofs = 2 * np.flatnonzero(static["volume_top"]) + 1
    constrained = static["constrained"]
    interface_mask = static["material_bottom"] | static["material_top"]
    E = _scalar(metadata["youngs_modulus"], "v5 replay youngs_modulus")
    nu = _scalar(metadata["poisson_ratio"], "v5 replay poisson_ratio")
    k_res = _scalar(metadata["k_res"], "v5 replay k_res")
    Gc = _scalar(metadata["Gc"], "v5 replay Gc")
    l0 = _scalar(metadata["l0"], "v5 replay l0")
    delta = _scalar(metadata["delta"], "v5 replay delta")
    field_equations_validated = True
    state_ranges: list[dict[str, Any]] = [
        {
            "damage_min": 0.0,
            "damage_max": 0.0,
            "healing_min": 0.0,
            "history_min_gap": 0.0,
        }
    ]
    for index in range(1, n_states):
        load = float(state_u[index])
        q = state_q[index]
        damage = state_d[index]
        history = state_H[index]
        prescribed = np.zeros(model.n_dofs, dtype=np.float64)
        prescribed[top_dofs] = load
        if not np.array_equal(q[constrained], prescribed[constrained]):
            raise ValueError("v5 replay state hard constraints are forged")
        payload = _v5_validate_solver_summary(
            summaries[index],
            f"v5 accepted solver summary {index}",
            relaxation_policy=stabilized_config.relaxation_policy,
            acceptance_mode=stabilized_config.acceptance_mode,
            anderson_regularization_mu=stabilized_config.anderson_regularization_mu,
        )
        warning_row = (
            stabilized_config.acceptance_mode == STABILIZED_WARNING_ACCEPTANCE_MODE
            and payload["acceptance_status"]
            == STABILIZED_WARNING_ACCEPTANCE_STATUS
        )
        if (
            payload["summary_kind"] != "accepted"
            or payload["state_index"] != index
            or payload["acceptance_status"]
            not in (
                (STABILIZED_ACCEPTANCE_STATUS, STABILIZED_WARNING_ACCEPTANCE_STATUS)
                if stabilized_config.acceptance_mode
                == STABILIZED_WARNING_ACCEPTANCE_MODE
                else (STABILIZED_ACCEPTANCE_STATUS,)
            )
            or payload["target_u_bar"] != load
            or payload["load_start"] != state_u[index - 1]
            or payload["load_target"] != load
            or payload["increment"] != load - state_u[index - 1]
            or payload["complete_sweep"] is not True
        ):
            raise ValueError("v5 accepted solver summary/state pairing is forged")
        if str(state_status[index]) != (
            STABILIZED_WARNING_ACCEPTANCE_STATUS
            if warning_row
            else STABILIZED_ACCEPTANCE_STATUS
        ):
            raise ValueError("v5 replay state acceptance status is forged")
        if warning_row and payload["passed"]:
            raise ValueError("v5 warning accepted summary is marked as passed")
        if not warning_row and not payload["passed"]:
            raise ValueError("v5 strict accepted summary is not marked as passed")
        if payload["stagger_iterations"] > stabilized_config.base.max_stagger_iterations:
            raise ValueError("v5 accepted solver summary stagger count is invalid")
        if (
            not _is_anderson_policy(stabilized_config.relaxation_policy)
            and payload["backtrack_count"] > stabilized_config.max_backtracks
        ):
            raise ValueError("v5 accepted solver summary backtrack count is invalid")
        if stabilized_config.relaxation_policy == "fixed":
            if payload["omega"] != stabilized_config.omega_initial:
                raise ValueError("v5 fixed relaxation omega is not configured omega_initial")
            if payload["backtrack_count"] != 0:
                raise ValueError("v5 fixed relaxation accepted summary backtracks")
            if payload["sufficient_decrease_limit"] is not None:
                raise ValueError(
                    "v5 fixed relaxation accepted summary has a sufficient limit"
                )
        elif _is_anderson_policy(stabilized_config.relaxation_policy):
            max_anderson_backtracks = (
                STABILIZED_ANDERSON_MAX_BACKTRACKS
                if _is_anderson_residual_backtracking_policy(
                    stabilized_config.relaxation_policy
                )
                else 1
            )
            if payload["backtrack_count"] > max_anderson_backtracks:
                raise ValueError("v5 Anderson summary contains too many candidates")
            selection = payload["anderson_selection"]
            if selection == "accelerated":
                if payload["omega"] is not None:
                    raise ValueError("v5 Anderson accelerated summary has omega")
                if payload["sufficient_decrease_limit"] is None:
                    raise ValueError(
                        "v5 Anderson accelerated summary lacks a sufficient limit"
                    )
                if _is_anderson_residual_backtracking_policy(
                    stabilized_config.relaxation_policy
                ):
                    alpha = payload["trial_alpha"]
                    if alpha is None:
                        raise ValueError(
                            "v5 backtracked Anderson summary lacks trial_alpha"
                        )
            elif selection == "picard_fallback":
                if payload["omega"] != STABILIZED_ANDERSON_BETA:
                    raise ValueError("v5 Anderson fallback summary has invalid omega")
                if payload["sufficient_decrease_limit"] is not None:
                    raise ValueError(
                        "v5 Anderson fallback summary has a sufficient limit"
                    )
                if _is_anderson_residual_backtracking_policy(
                    stabilized_config.relaxation_policy
                ) and payload["trial_alpha"] is not None:
                    raise ValueError(
                        "v5 backtracked Anderson fallback retains trial_alpha"
                    )
            else:
                raise ValueError("v5 Anderson accepted summary has invalid selection")
        elif payload["sufficient_decrease_limit"] is None:
            raise ValueError(
                "v5 sufficient-decrease accepted summary lacks a sufficient limit"
            )
        required_gates = (
            STABILIZED_WARNING_HARD_GATE_NAMES
            if warning_row
            else (
                (*STABILIZED_COUPLED_GATE_NAMES, "coupled")
                if not _is_anderson_policy(stabilized_config.relaxation_policy)
                else (*STABILIZED_COUPLED_GATE_NAMES, "damage_range", "healing", "coupled")
            )
        )
        for gate_name in required_gates:
            if payload["gate_passes"].get(gate_name) is not True:
                raise ValueError(f"v5 accepted hard gate {gate_name} failed")
        if not warning_row and (
            payload["fixed_point_pass"] is not True
            or payload["damage_equation_pass"] is not True
            or payload["globalization_pass"] is not True
        ):
            raise ValueError("v5 accepted stabilized gates failed")
        if warning_row and (
            payload["damage_equation_pass"] is not True
            or payload["globalization_pass"] is not True
        ):
            raise ValueError("v5 warning accepted hard gates failed")
        if _v5_integer(payload["cg_info_after"], "v5 accepted cg_info_after") != 0:
            raise ValueError("v5 accepted production CG status is not zero")
        mechanics = engineering._replay_mechanics_at_state(
            q,
            damage,
            load,
            static["ba"],
            static["material_phase"],
            static["grip_phase"],
            static["volume_bottom"],
            static["volume_top"],
            static["free"],
            0,
            E,
            nu,
            k_res,
            model.n_material,
            base_config,
        )
        if not mechanics["finite"] or not mechanics["mechanics_pass"]:
            raise ValueError("v5 replay accepted-state mechanics gate fails")
        if not mechanics["force_pass"] or not mechanics["moment_pass"]:
            raise ValueError("v5 replay accepted-state conservation gate fails")
        for name, expected in (
            ("energy_after", mechanics["energy"]),
            ("top_reaction_after", mechanics["top"]),
            ("bottom_reaction_after", mechanics["bottom"]),
            ("anchor_reaction_after", mechanics["anchor"]),
            ("moment_after", mechanics["moment"]),
            ("free_residual_norm_after", mechanics["free_residual_norm"]),
            ("free_residual_relative_after", mechanics["free_residual_relative_norm"]),
            ("force_error", mechanics["force_error"]),
            ("force_scale", mechanics["force_scale"]),
            ("moment_error", mechanics["moment_error"]),
            ("moment_scale", mechanics["moment_scale"]),
        ):
            _v5_compare_diagnostic_scalar(payload, name, expected)
        for actual, expected, name in (
            (accepted_top[index - 1], mechanics["top"], "v5 accepted top reaction"),
            (accepted_bottom[index - 1], mechanics["bottom"], "v5 accepted bottom reaction"),
            (accepted_anchor[index - 1], mechanics["anchor"], "v5 accepted anchor reaction"),
        ):
            _compare_scalar(actual, expected, name, 2.0e-8)
        driver, expected_history, particle_strain = engineering._replay_amor(
            q, state_H[index - 1], static["ba"], E, nu, model.n_material
        )
        _array_close(history, expected_history, "v5 replay accepted history equation", 2.0e-8)
        _, _, _, matrix, _ = engineering._replay_damage(
            history,
            static["material_coordinates"],
            static["material_volumes"],
            static["material_edges"],
            delta,
            Gc,
            l0,
        )
        rho = np.asarray(
            Gc / l0 * damage
            + Gc * l0 * np.asarray(matrix @ damage, dtype=np.float64)
            + 2.0 * history * damage
            - 2.0 * history,
            dtype=np.float64,
        )
        rho_volume = static["material_volumes"] * rho
        residual_scale = engineering._replay_damage_residual_scale(
            damage, history, history, matrix, Gc, l0
        )
        damage_equation_valid = bool(
            float(np.max(np.abs(rho)))
            <= base_config.damage_residual.limit(residual_scale)
        )
        if not warning_row and not damage_equation_valid:
            raise ValueError("v5 replay accepted-state damage equation gate fails")
        field_equations_validated = bool(
            field_equations_validated and damage_equation_valid
        )
        _v5_compare_diagnostic_scalar(payload, "rho_next_inf", np.max(np.abs(rho)))
        _v5_compare_diagnostic_scalar(payload, "damage_residual_scale", residual_scale)
        damage_min = float(np.min(damage))
        damage_max = float(np.max(damage))
        healing_min = float(np.min(damage - state_d[index - 1]))
        history_min_gap = float(np.min(history - state_H[index - 1]))
        for name, expected in (
            ("damage_min", damage_min),
            ("damage_max", damage_max),
            ("healing_min", healing_min),
            ("history_min_gap", history_min_gap),
        ):
            _v5_compare_diagnostic_scalar(payload, name, expected)
        expected_range = bool(
            damage_min >= -base_config.damage_range_atol
            and damage_max <= 1.0 + base_config.damage_range_atol
        )
        expected_healing = bool(healing_min >= -base_config.healing_atol)
        expected_history_monotonic = bool(history_min_gap >= 0.0)
        if payload["gate_passes"]["damage_range"] != expected_range or payload["gate_passes"]["healing"] != expected_healing:
            raise ValueError("v5 accepted range/healing diagnostics are forged")
        if payload["gate_passes"]["history_monotonic"] != expected_history_monotonic:
            raise ValueError("v5 accepted history monotonic diagnostic is forged")
        state_ranges.append(
            {
                "damage_min": damage_min,
                "damage_max": damage_max,
                "healing_min": healing_min,
                "history_min_gap": history_min_gap,
                "amor_driver_max": float(np.max(driver)),
                "strain_max": float(np.max(np.abs(particle_strain))),
                "rho_inf": float(np.max(np.abs(rho))),
                "rho_volume_inf": float(np.max(np.abs(rho_volume))),
                "localization_pass": bool(
                    engineering.compute_crack_region_diagnostics(
                        static["material_coordinates"],
                        delta,
                        damage,
                        history,
                        particle_strain,
                        driver,
                        interface_mask,
                        allow_negative_damage=True,
                    ).localization_pass
                ),
            }
        )
    return {
        "u": state_u,
        "q": state_q,
        "d": state_d,
        "H": state_H,
        "digest": state_digest,
        "field_equations_validated": field_equations_validated,
        "status": state_status,
        "summaries": summaries,
        "state_ranges": state_ranges,
    }


def _archive_string(values: Mapping[str, np.ndarray], name: str) -> str:
    if name not in values:
        raise ValueError(f"stabilized archive is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.kind not in "SU":
        raise ValueError(f"stabilized archive {name} must be a scalar string")
    return str(value)


def _archive_bool(values: Mapping[str, np.ndarray], name: str) -> bool:
    if name not in values:
        raise ValueError(f"stabilized archive is missing {name}")
    value = np.asarray(values[name])
    if value.shape != () or value.dtype != _BOOL:
        raise ValueError(f"stabilized archive {name} must be a scalar boolean")
    return bool(value)


def _archive_json(values: Mapping[str, np.ndarray], name: str) -> dict[str, Any]:
    try:
        payload = json.loads(_archive_string(values, name))
    except json.JSONDecodeError as error:
        raise ValueError(f"stabilized archive {name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"stabilized archive {name} must contain a mapping")
    return payload


# ---------------------------------------------------------------------------
# Version 5 lean archive
# ---------------------------------------------------------------------------
#
# The v5 archive is the only stabilized archive layout.  It serializes the
# accepted physical states, the load/reaction/work scalars, and one compact
# final solver-summary row per state.  It never stores the union geometry,
# padded BA supports, operators, or any intermediate sweep/trial q/d/H/force/
# rho vectors.  Replay rebuilds the exact model from the explicitly hashed
# external source cloud and independently re-evaluates every accepted state.

_V5_SOURCE_FIELDS = (
    "coordinates",
    "triangles",
    "volumes",
    "bottom",
    "top",
    "anchor_particle",
    "d048",
)
_V5_FLOAT_SOURCE_FIELDS = {"coordinates", "volumes"}
_V5_INDEX_SOURCE_FIELDS = {"triangles", "anchor_particle", "d048"}
_V5_BOOL_SOURCE_FIELDS = {"bottom", "top"}
_V5_ARCHIVE_BASE_FIELDS = {
    "archive_schema",
    "acceptance_mode",
    "acceptance_status",
    "completion_status",
    "validated",
    "experimental",
    "metadata_json",
    "state_u_bar",
    "state_union_displacement",
    "state_material_damage",
    "state_material_history",
    "state_digest",
    "state_acceptance_status",
    "accepted_top_reaction_after",
    "accepted_bottom_reaction_after",
    "accepted_anchor_reaction_after",
    "state_external_work",
    "solver_summary_json",
    "work_check_u_bar",
    "work_check_epsilon",
    "work_check_reaction",
    "work_check_reaction_fd",
    "work_check_relative_error",
    "work_check_relative_limit",
    "work_check_passed",
}
_V5_FAILURE_FIELDS = {
    "failure_type",
    "failure_message",
    "failure_summary_json",
    "last_committed_u_bar",
    "last_committed_damage_max",
    "last_committed_history_max",
}
_WARNING_ARCHIVE_BASE_FIELDS = set(_V5_ARCHIVE_BASE_FIELDS) | set(
    _WARNING_ARCHIVE_EXTRA_FIELDS
)
_WARNING_FAILURE_FIELDS = set(_V5_FAILURE_FIELDS)
_WARNING_ARCHIVE_ARRAY_COUNT = (
    len(_WARNING_ARCHIVE_BASE_FIELDS | _WARNING_FAILURE_FIELDS)
)
_WARNING_ARCHIVE_SCHEMA_FIELDS = {
    "warning_gate_names",
    "state_bypassed_gate_mask",
    "accepted_warning_present",
    "accepted_warning_range_excess",
    "accepted_warning_healing_magnitude",
}
_V5_SOLVER_SUMMARY_RESERVE_BYTES = 16 * 1024
_V5_FAILURE_SUMMARY_RESERVE_BYTES = 16 * 1024
_V5_METADATA_RESERVE_BYTES = 256 * 1024
_V5_NPY_HEADER_RESERVE_BYTES = 512
_V5_NPZ_RESERVE_BYTES = 64 * 1024
# One uniform envelope count protects the byte contract from NumPy header
# growth without making the estimate depend on the archive kind.
_V5_ARCHIVE_ARRAY_COUNT = len(_V5_ARCHIVE_BASE_FIELDS | _V5_FAILURE_FIELDS)


def _v5_json_value(value: Any) -> Any:
    """Convert scalar provenance to JSON while rejecting numerical vectors."""

    if isinstance(value, np.ndarray):
        raise TypeError("v5 archive JSON cannot contain NumPy arrays")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _v5_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_v5_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("v5 archive JSON cannot contain non-finite values")
        return result
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("v5 archive JSON cannot contain non-finite values")
        return value
    raise TypeError(f"unsupported v5 archive JSON value type: {type(value).__name__}")


def _v5_json_text(value: Any) -> str:
    return json.dumps(_v5_json_value(value), sort_keys=True, allow_nan=False)


def _v5_json_sha256(value: Any) -> str:
    return hashlib.sha256(_v5_json_text(value).encode("utf-8")).hexdigest()


def _schedule_archive_header_arrays(
    schedule: StabilizedLoadSchedule,
) -> dict[str, np.ndarray]:
    """Return the scalar schedule identity retained in schedule archives."""

    payload = schedule.payload()
    return {
        "schedule_mode": np.asarray(payload["mode"], dtype=np.str_),
        "schedule_path": np.asarray(payload["path"], dtype=np.str_),
        "schedule_file_sha256": np.asarray(payload["file_sha256"], dtype=np.str_),
        "schedule_target_sha256": np.asarray(
            payload["canonical_target_sha256"], dtype=np.str_
        ),
        "schedule_target_count": np.asarray(payload["target_count"], dtype=np.int64),
        "schedule_load_start": np.asarray(payload["load_start"], dtype=np.float64),
        "schedule_load_final": np.asarray(payload["load_final"], dtype=np.float64),
        "schedule_endpoint": np.asarray(payload["endpoint"], dtype=np.float64),
        "schedule_source_path": np.asarray(
            payload["source_archive_path"], dtype=np.str_
        ),
        "schedule_source_sha256": np.asarray(
            payload["source_archive_sha256"], dtype=np.str_
        ),
        "schedule_source_state_count": np.asarray(
            payload["source_state_count"], dtype=np.int64
        ),
    }


def _validate_schedule_archive_binding(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    config: StabilizedEngineeringGripConfig,
) -> StabilizedLoadSchedule:
    """Validate schedule archive scalars, metadata, and the external NPZ."""

    schedule = config.load_schedule
    if schedule is None:
        raise ValueError("schedule archive configuration does not contain a schedule")
    raw_payload = metadata.get("load_schedule")
    if not isinstance(raw_payload, Mapping) or dict(raw_payload) != schedule.payload():
        raise ValueError("schedule archive metadata binding is forged")
    expected = _schedule_archive_header_arrays(schedule)
    for name, expected_value in expected.items():
        if name not in values:
            raise ValueError(f"schedule archive is missing {name}")
        actual = np.asarray(values[name])
        if actual.shape != expected_value.shape or actual.dtype != expected_value.dtype:
            raise ValueError(f"schedule archive {name} has the wrong scalar dtype/shape")
        if not np.array_equal(actual, expected_value):
            raise ValueError(f"schedule archive {name} is not paired with metadata")
    loaded = load_stabilized_load_schedule(
        schedule.path,
        load_start=config.base.load_start,
        load_final=config.base.load_final,
    )
    if loaded.payload() != schedule.payload():
        raise ValueError("schedule archive configuration is not paired with the artifact")
    return loaded


def load_stabilized_source_cloud(
    source_metadata: Mapping[str, Any], delta: float | None = None
) -> dict[str, np.ndarray]:
    """Load and verify the external source cloud named by v5 provenance.

    This is deliberately a source-path operation rather than an archive
    geometry loader.  A replay therefore fails if the original source binary is
    absent, changed, or has a different dtype/shape contract.
    """

    if not isinstance(source_metadata, Mapping):
        raise TypeError("v5 source metadata must be a mapping")
    path_value = source_metadata.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("v5 source metadata path is missing")
    path = Path(path_value)
    if not path.is_absolute() or str(path.resolve()) != path_value:
        raise ValueError("v5 source metadata path must be the resolved absolute path")
    if not path.is_file():
        raise FileNotFoundError(f"v5 source cloud does not exist: {path}")
    expected_sha = source_metadata.get("sha256")
    _digest(expected_sha, "v5 source metadata sha256")
    actual_sha = _archive_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError("v5 source cloud SHA256 does not match metadata")
    size = source_metadata.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("v5 source metadata size_bytes is invalid")
    if path.stat().st_size != size:
        raise ValueError("v5 source cloud size does not match metadata")
    required = source_metadata.get("required_fields")
    if required != list(_V5_SOURCE_FIELDS):
        raise ValueError("v5 source metadata required_fields are forged")
    descriptors = source_metadata.get("fields")
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(_V5_SOURCE_FIELDS):
        raise ValueError("v5 source metadata field descriptors are forged")

    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot load v5 source cloud {path}: {error}") from error
    for name, value in raw.items():
        if value.dtype.kind == "O":
            raise ValueError(f"v5 source cloud field {name} has forbidden object dtype")
    missing = sorted(set(_V5_SOURCE_FIELDS) - set(raw))
    if missing:
        raise ValueError(f"v5 source cloud is missing fields: {missing}")

    result: dict[str, np.ndarray] = {}
    for name in _V5_SOURCE_FIELDS:
        value = raw[name]
        descriptor = descriptors[name]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"dtype", "shape"}:
            raise ValueError(f"v5 source descriptor for {name} is invalid")
        if descriptor["dtype"] != str(value.dtype) or descriptor["shape"] != list(value.shape):
            raise ValueError(f"v5 source descriptor for {name} does not match the file")
        if name in _V5_FLOAT_SOURCE_FIELDS:
            result[name] = _f64(value, f"v5 source {name}")
        elif name in _V5_INDEX_SOURCE_FIELDS:
            result[name] = _i64(value, f"v5 source {name}")
        else:
            result[name] = _bool(value, f"v5 source {name}")

    coordinates = result["coordinates"]
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("v5 source coordinates must have shape (N, 2)")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("v5 source coordinates must be unique")
    n_material = len(coordinates)
    triangles = result["triangles"]
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("v5 source triangles must have shape (T, 3)")
    if np.any(triangles < 0) or np.any(triangles >= n_material):
        raise ValueError("v5 source triangles contain an invalid index")
    volumes = result["volumes"]
    if volumes.shape != (n_material,) or np.any(volumes <= 0.0):
        raise ValueError("v5 source volumes must be positive per-particle volumes")
    bottom = result["bottom"]
    top = result["top"]
    if bottom.shape != (n_material,) or top.shape != (n_material,):
        raise ValueError("v5 source boundary masks have the wrong shape")
    if not np.any(bottom) or not np.any(top) or np.any(bottom & top):
        raise ValueError("v5 source boundary masks are invalid")
    anchor = result["anchor_particle"]
    if anchor.shape != () or int(anchor) != 0:
        raise ValueError("v5 source anchor_particle must be exact int64 zero")
    if not np.array_equal(coordinates[0], np.asarray([0.0, 0.0], dtype=np.float64)):
        raise ValueError("v5 source particle zero must be the material anchor")
    edges = result["d048"]
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("v5 source d048 edges must have shape (E, 2)")
    if np.any(edges < 0) or np.any(edges >= n_material) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("v5 source d048 edges contain an invalid index")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("v5 source d048 edges contain duplicates")
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("v5 source d048 edges are not reciprocal")
    if delta is not None:
        delta_value = _scalar(delta, "v5 source delta")
        if delta_value <= 0.0:
            raise ValueError("v5 source delta must be positive")
        # The serialized reference graph field is named ``d048``; it is only
        # the construction input for the exact delta=0.048 engineering-grip
        # model, mirroring the workflow source-cloud contract.
        if delta_value != 0.048:
            raise ValueError(
                "the explicit source field 'd048' is only compatible with delta=0.048"
            )
    return result


def _v5_validate_source_matches_model(
    source: Mapping[str, np.ndarray], model: EngineeringGripModel
) -> None:
    geometry = model.geometry
    exact_pairs = (
        ("coordinates", source["coordinates"], geometry.material_reference_coordinates),
        ("triangles", source["triangles"], geometry.material_reference_triangles),
        ("volumes", source["volumes"], geometry.material_reference_volumes),
        ("bottom", source["bottom"], geometry.material_bottom_mask[: model.n_material]),
        ("top", source["top"], geometry.material_top_mask[: model.n_material]),
        ("d048", source["d048"], geometry.material_reference_edges),
    )
    for name, actual, expected in exact_pairs:
        if not np.array_equal(actual, expected):
            raise ValueError(f"v5 source {name} is not the model construction input")
    if int(source["anchor_particle"]) != 0:
        raise ValueError("v5 source anchor is not the model anchor")


def _v5_rebuild_model(
    metadata: Mapping[str, Any],
) -> tuple[EngineeringGripModel, dict[str, np.ndarray]]:
    """Rebuild the union geometry/model from the external source cloud."""

    source = load_stabilized_source_cloud(metadata["source"], metadata["delta"])
    from .nonlocal_grip_diagnostic import build_union_geometry_from_arrays

    geometry = build_union_geometry_from_arrays(
        source["coordinates"],
        source["triangles"],
        source["volumes"],
        source["bottom"],
        source["top"],
        int(source["anchor_particle"]),
        source["d048"],
        _scalar(metadata["delta"], "v5 replay delta"),
        _scalar(metadata["h_g"], "v5 replay h_g"),
        _scalar(metadata["visibility_tolerance"], "v5 replay visibility tolerance"),
    )
    model = engineering.build_engineering_grip_model(
        geometry,
        _scalar(metadata["youngs_modulus"], "v5 replay youngs_modulus"),
        _scalar(metadata["poisson_ratio"], "v5 replay poisson_ratio"),
        _scalar(metadata["k_res"], "v5 replay k_res"),
        _scalar(metadata["rank_rtol"], "v5 replay rank_rtol"),
        _scalar(metadata["Gc"], "v5 replay Gc"),
        _scalar(metadata["l0"], "v5 replay l0"),
    )
    _v5_validate_source_matches_model(source, model)
    for name, actual in (
        ("material_hash", model.material_hash),
        ("union_hash", model.union_hash),
        ("static_digest", model.static_digest),
    ):
        if metadata.get(name) != actual:
            raise ValueError(f"v5 replay {name} does not match rebuilt model")
    for name, actual in (
        ("delta", model.geometry.delta),
        ("h_g", model.geometry.h_g),
        ("visibility_tolerance", model.geometry.visibility_tolerance),
        ("youngs_modulus", model.static.youngs_modulus),
        ("poisson_ratio", model.static.poisson_ratio),
        ("k_res", model.static.k_res),
        ("rank_rtol", model.static.rank_rtol),
        ("Gc", model.Gc),
        ("l0", model.l0),
    ):
        if _scalar(metadata[name], f"v5 metadata {name}") != float(actual):
            raise ValueError(f"v5 replay model scalar {name} is forged")
    return model, source


def _v5_replay_static(model: EngineeringGripModel) -> dict[str, Any]:
    """Create replay-only static data in memory; none is read from the archive."""

    geometry = model.geometry
    ba = engineering._replay_ba(
        geometry.coordinates,
        geometry.union_volumes,
        geometry.directed_edges,
        geometry.delta,
        model.static.rank_rtol,
    )
    n_material = model.n_material
    return {
        "ba": ba,
        "material_coordinates": np.asarray(
            geometry.material_reference_coordinates, dtype=np.float64
        ),
        "material_volumes": np.asarray(
            geometry.material_reference_volumes, dtype=np.float64
        ),
        "material_edges": np.asarray(geometry.material_reference_edges, dtype=np.int64),
        "material_bottom": np.asarray(
            geometry.material_bottom_mask[:n_material], dtype=bool
        ),
        "material_top": np.asarray(geometry.material_top_mask[:n_material], dtype=bool),
        "material_phase": np.asarray(geometry.material_volumes, dtype=np.float64),
        "grip_phase": np.asarray(geometry.grip_volumes, dtype=np.float64),
        "union_volumes": np.asarray(geometry.union_volumes, dtype=np.float64),
        "volume_bottom": np.asarray(model.constraints.outer_bottom_mask, dtype=bool),
        "volume_top": np.asarray(model.constraints.outer_top_mask, dtype=bool),
        "constrained": np.asarray(model.constraints.constrained_dofs, dtype=np.int64),
        "free": np.asarray(model.constraints.free_dofs, dtype=np.int64),
        "crack": np.asarray(model.crack_region_mask, dtype=bool),
        "n_material": n_material,
        "n_union": model.n_union,
    }


def _v5_solver_summary_for_diagnostic(
    diagnostic: StabilizedRelaxationSweepDiagnostics,
    *,
    state_index: int,
    load_start: float,
    load_target: float,
    increment: float,
    cutback_count: int,
    acceptance_status: str | None,
    accepted: bool,
    summary_kind: str,
    relaxation_policy: str,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    """Return one scalar-only row for a final sweep or failure diagnostic."""

    if relaxation_policy not in STABILIZED_RELAXATION_POLICIES:
        raise ValueError(
            "relaxation_policy must be one of "
            f"{STABILIZED_RELAXATION_POLICIES}, got {relaxation_policy!r}"
        )
    mode = _validate_stabilized_acceptance_mode(acceptance_mode)
    if mode == STABILIZED_WARNING_ACCEPTANCE_MODE:
        if acceptance_status not in (
            None,
            STABILIZED_ACCEPTANCE_STATUS,
            STABILIZED_WARNING_ACCEPTANCE_STATUS,
        ):
            raise ValueError("stabilized warning solver summary status is invalid")
    elif acceptance_status not in (None, STABILIZED_ACCEPTANCE_STATUS, "initial"):
        raise ValueError("stabilized solver summary status is invalid")
    failure_stage = diagnostic.failure_stage
    failure_reason = diagnostic.failure_reason
    if (
        summary_kind == "accepted"
        and accepted
        and acceptance_status == STABILIZED_WARNING_ACCEPTANCE_STATUS
    ):
        # A warning-accepted row is an accepted physical state, not a failure
        # row.  Its failed convergence gates remain truthful in ``gate_passes``
        # and the warning archive bypass mask; accepted-summary failure fields
        # must stay empty so the row cannot be confused with the final failure
        # summary contract.
        failure_stage = ""
        failure_reason = ""
    payload: dict[str, Any] = {
        "summary_kind": summary_kind,
        "state_index": state_index,
        "load_start": load_start,
        "load_target": load_target,
        "increment": increment,
        "cutback_count": cutback_count,
        "stagger_iterations": diagnostic.sweep_index + 1,
        "relaxation_policy": relaxation_policy,
        "acceptance_status": acceptance_status,
        "accepted": accepted,
        "sweep_index": diagnostic.sweep_index,
        "target_u_bar": diagnostic.target_u_bar,
        "complete_sweep": diagnostic.complete_sweep,
        "passed": diagnostic.passed,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "gate_passes": diagnostic.gate_passes,
        "omega": diagnostic.omega,
        "backtrack_count": diagnostic.backtrack_count,
        "reference_merit": diagnostic.reference_merit,
        "merit": diagnostic.merit,
        "sufficient_decrease_limit": diagnostic.sufficient_decrease_limit,
        "fixed_point_error": diagnostic.fixed_point_error,
        "fixed_point_scale": diagnostic.fixed_point_scale,
        "fixed_point_pass": diagnostic.fixed_point_pass,
        "damage_equation_pass": diagnostic.damage_equation_pass,
        "globalization_pass": diagnostic.globalization_pass,
        "coupled_pass": diagnostic.coupled_pass,
    }
    payload.update(
        {
            name: getattr(diagnostic, name)
            for name in _V5_SOLVER_SUMMARY_SCALAR_FIELDS
        }
    )
    if _is_anderson_policy(relaxation_policy):
        payload.update(
            {
                "anderson_history_size": diagnostic.anderson_history_size,
                "anderson_qr_rank": diagnostic.anderson_qr_rank,
                "anderson_selection": diagnostic.anderson_selection,
                "anderson_fallback_reason": diagnostic.anderson_fallback_reason,
                "anderson_restart_reason": diagnostic.anderson_restart_reason,
                "candidate_fp_error": diagnostic.candidate_fp_error,
                "candidate_fp_scale": diagnostic.candidate_fp_scale,
                "candidate_damage_residual_scale": diagnostic.candidate_damage_residual_scale,
            }
        )
        if _is_anderson_residual_policy(relaxation_policy):
            payload["anderson_merit_identity"] = (
                _anderson_merit_identity(relaxation_policy)
            )
        if _is_anderson_residual_backtracking_policy(relaxation_policy):
            payload["trial_alpha"] = diagnostic.trial_alpha
        if _is_anderson_regularized_policy(relaxation_policy):
            regularization_mu = (
                diagnostic.anderson_regularization_mu
                if diagnostic.anderson_regularization_mu is not None
                else anderson_regularization_mu
            )
            if regularization_mu is None:
                raise ValueError(
                    "regularized Anderson summary is missing regularization mu"
                )
            payload.update(
                {
                    "anderson_regularization_mu": regularization_mu,
                    "anderson_current_residual_l2": (
                        diagnostic.anderson_current_residual_l2
                    ),
                    "anderson_regularization_lambda": (
                        diagnostic.anderson_regularization_lambda
                    ),
                    "anderson_unregularized_ls_residual_l2": (
                        diagnostic.anderson_unregularized_ls_residual_l2
                    ),
                    "anderson_coefficient_l2": diagnostic.anderson_coefficient_l2,
                }
            )
    payload.update(
        {
            name: bool(getattr(diagnostic, name))
            for name in _V5_DIAGNOSTIC_BOOL_FIELDS
        }
    )
    return _v5_json_value(payload)


def _v5_initial_solver_summary(
    relaxation_policy: str = "sufficient_decrease",
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    """Return the explicit zero-state summary row without a solver solve."""

    if relaxation_policy not in STABILIZED_RELAXATION_POLICIES:
        raise ValueError(
            "relaxation_policy must be one of "
            f"{STABILIZED_RELAXATION_POLICIES}, got {relaxation_policy!r}"
        )
    payload: dict[str, Any] = {
        "summary_kind": "initial",
        "state_index": 0,
        "load_start": 0.0,
        "load_target": 0.0,
        "increment": 0.0,
        "cutback_count": 0,
        "stagger_iterations": 0,
        "relaxation_policy": relaxation_policy,
        "acceptance_status": "initial",
        "accepted": True,
        "sweep_index": -1,
        "target_u_bar": 0.0,
        "complete_sweep": False,
        "passed": True,
        "failure_stage": "",
        "failure_reason": "",
        "gate_passes": {name: False for name in _V5_SOLVER_SUMMARY_GATE_NAMES},
        "omega": None,
        "backtrack_count": 0,
        "reference_merit": None,
        "merit": None,
        "sufficient_decrease_limit": None,
        "fixed_point_error": None,
        "fixed_point_scale": None,
        "fixed_point_pass": False,
        "damage_equation_pass": False,
        "globalization_pass": False,
        "coupled_pass": False,
    }
    if _is_anderson_policy(relaxation_policy):
        payload.update(
            {
                "anderson_history_size": 0,
                "anderson_qr_rank": 0,
                "anderson_selection": "initial",
                "anderson_fallback_reason": None,
                "anderson_restart_reason": None,
                "candidate_fp_error": None,
                "candidate_fp_scale": None,
                "candidate_damage_residual_scale": None,
            }
        )
        if _is_anderson_residual_policy(relaxation_policy):
            payload["anderson_merit_identity"] = (
                _anderson_merit_identity(relaxation_policy)
            )
        if _is_anderson_residual_backtracking_policy(relaxation_policy):
            payload["trial_alpha"] = None
        if _is_anderson_regularized_policy(relaxation_policy):
            mu = _optional_scalar(
                anderson_regularization_mu,
                "initial regularized Anderson mu",
            )
            if mu is None or mu <= 0.0:
                raise ValueError(
                    "initial regularized Anderson summary requires positive mu"
                )
            payload.update(
                {
                    "anderson_regularization_mu": mu,
                    "anderson_current_residual_l2": None,
                    "anderson_regularization_lambda": None,
                    "anderson_unregularized_ls_residual_l2": None,
                    "anderson_coefficient_l2": None,
                }
            )
    payload.update({name: None for name in _V5_SOLVER_SUMMARY_SCALAR_FIELDS})
    payload.update({name: False for name in _V5_DIAGNOSTIC_BOOL_FIELDS})
    return _v5_json_value(payload)


def _v5_solver_summary_payload(
    state_index: int,
    step: StabilizedEngineeringGripAcceptedStep,
    *,
    relaxation_policy: str = "sufficient_decrease",
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    return _v5_solver_summary_for_diagnostic(
        step.diagnostics,
        state_index=state_index,
        load_start=step.load_start,
        load_target=step.load_target,
        increment=step.increment,
        cutback_count=step.cutback_count,
        acceptance_status=step.acceptance_status,
        accepted=True,
        summary_kind="accepted",
        relaxation_policy=relaxation_policy,
        acceptance_mode=acceptance_mode,
        anderson_regularization_mu=anderson_regularization_mu,
    )


def _v5_failure_summary(
    failure: StabilizedEngineeringGripContinuationFailure,
    *,
    relaxation_policy: str = "sufficient_decrease",
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    """Return one scalar-only final failure summary; no rejected vectors."""

    available = bool(failure.attempts and not failure.attempts[-1].accepted)
    payload: dict[str, Any] = {
        "available": available,
        "failure_code": failure.failure_code,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure),
        "last_committed_u_bar": failure.last_committed_state.u_bar,
        "failure_stage": None,
        "failure_reason": None,
        "load_start": None,
        "load_target": None,
        "increment": None,
        "cutback_count": None,
        "solver_summary": None,
    }
    if available:
        attempt = failure.attempts[-1]
        payload.update(
            {
                "failure_stage": attempt.failure_stage,
                "failure_reason": attempt.failure_reason,
                "load_start": attempt.load_start,
                "load_target": attempt.load_target,
                "increment": attempt.load_target - attempt.load_start,
                "cutback_count": attempt.cutback_count,
                "solver_summary": _v5_solver_summary_for_diagnostic(
                    attempt.diagnostics,
                    state_index=-1,
                    load_start=attempt.load_start,
                    load_target=attempt.load_target,
                    increment=attempt.load_target - attempt.load_start,
                    cutback_count=attempt.cutback_count,
                    acceptance_status=None,
                    accepted=False,
                    summary_kind="failure",
                    relaxation_policy=relaxation_policy,
                    acceptance_mode=acceptance_mode,
                    anderson_regularization_mu=anderson_regularization_mu,
                ),
            }
        )
    else:
        payload["failure_reason"] = (
            "failure stopped without a retained rejected candidate"
            if failure.attempts
            else "archive budget exhausted before an attempt was retained"
        )
    return _v5_json_value(payload)


def _v5_string_array(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([_v5_json_text(record) for record in records], dtype=np.str_)


def _v5_external_work(
    loads: np.ndarray, top_reactions: np.ndarray
) -> np.ndarray:
    """Return cumulative trapezoidal work along the retained load path."""

    load_values = _f64(loads, "v5 external-work loads")
    reaction_values = _f64(
        top_reactions, "v5 external-work top reactions", load_values.shape
    )
    if load_values.ndim != 1 or len(load_values) == 0:
        raise ValueError("v5 external-work loads must be a nonempty vector")
    if np.any(np.diff(load_values) <= 0.0):
        raise ValueError("v5 external-work loads must be strictly increasing")
    result = np.zeros(len(load_values), dtype=np.float64)
    if len(load_values) > 1:
        result[1:] = np.cumsum(
            0.5
            * (reaction_values[1:] + reaction_values[:-1])
            * (load_values[1:] - load_values[:-1])
        )
    return result


def _v5_size_for_counts(
    n_material: int,
    n_union_dofs: int,
    config: StabilizedEngineeringGripConfig,
    *,
    accepted_steps: int,
    count_kind: str,
) -> dict[str, Any]:
    material_count, union_dof_count = _validated_archive_dimensions(
        n_material, n_union_dofs
    )
    counts = {"accepted_steps": _archive_count(accepted_steps, "accepted_steps")}
    states = counts["accepted_steps"] + 1
    # State rows are the only vector-valued numerical records in v5.  String
    # reserves cover exact state digests/statuses without relying on NPZ
    # compression or object arrays.  One bounded final solver-summary row is
    # retained for every state, including the explicit initial zero state.
    state_bytes = states * (
        (union_dof_count + 2 * material_count) * _FLOAT64.itemsize
        + 2 * _FLOAT64.itemsize
        + 2 * 128
    )
    accepted_bytes = counts["accepted_steps"] * (3 * _FLOAT64.itemsize)
    summary_bytes = states * _V5_SOLVER_SUMMARY_RESERVE_BYTES
    work_bytes = 6 * _FLOAT64.itemsize + 1
    failure_bytes = (
        _V5_FAILURE_SUMMARY_RESERVE_BYTES
        + 2 * 256
        + 3 * _FLOAT64.itemsize
    )
    warning_bytes = 0
    if config.acceptance_mode == STABILIZED_WARNING_ACCEPTANCE_MODE:
        warning_gate_count = len(STABILIZED_WARNING_GATE_NAMES)
        warning_bytes = (
            warning_gate_count * 128
            + counts["accepted_steps"]
            * (warning_gate_count + 2 * _FLOAT64.itemsize + 128)
        )
    array_count = (
        _WARNING_ARCHIVE_ARRAY_COUNT
        if config.acceptance_mode == STABILIZED_WARNING_ACCEPTANCE_MODE
        else _V5_ARCHIVE_ARRAY_COUNT
    )
    if config.load_schedule is not None:
        array_count += len(STABILIZED_SCHEDULE_ARCHIVE_FIELDS)
    typed_bytes = (
        state_bytes
        + accepted_bytes
        + summary_bytes
        + work_bytes
        + failure_bytes
        + warning_bytes
    )
    archive_bytes = int(
        typed_bytes
        + _V5_METADATA_RESERVE_BYTES
        + array_count * _V5_NPY_HEADER_RESERVE_BYTES
        + _V5_NPZ_RESERVE_BYTES
    )
    required_free = archive_bytes + STABILIZED_FILESYSTEM_RESERVE_BYTES
    count_payload = {
        "count_kind": count_kind,
        "n_material": material_count,
        "n_union_dofs": union_dof_count,
        "n_states": states,
        "accepted_steps": counts["accepted_steps"],
        "solver_summary_rows": states,
        "archive_array_count": array_count,
        "static_bound_kind": "rebuilt_from_source",
    }
    breakdown = {
        "state_bytes": state_bytes,
        "accepted_scalar_bytes": accepted_bytes,
        "solver_summary_bytes": summary_bytes,
        "work_check_bytes": work_bytes,
        "failure_bytes": failure_bytes,
        "warning_provenance_bytes": warning_bytes,
        "metadata_bytes": _V5_METADATA_RESERVE_BYTES,
        "npy_header_bytes": array_count * _V5_NPY_HEADER_RESERVE_BYTES,
        "npz_container_bytes": _V5_NPZ_RESERVE_BYTES,
        "typed_member_bytes": typed_bytes,
        "archive_bytes": archive_bytes,
        "filesystem_reserve_bytes": STABILIZED_FILESYSTEM_RESERVE_BYTES,
        "required_free_bytes": required_free,
    }
    return {
        **count_payload,
        **breakdown,
        "n_increments": counts["accepted_steps"],
        "max_accepted_steps": counts["accepted_steps"],
        "required_free_bytes": required_free,
        "breakdown": {"counts": count_payload, "bytes": breakdown},
    }


def estimate_stabilized_archive_bytes_for_counts(
    n_material: int,
    n_union_dofs: int,
    config: StabilizedEngineeringGripConfig,
    *,
    accepted_steps: int,
) -> dict[str, Any]:
    """Return a conservative v5 state-plus-summary byte bound.

    Static operators and geometry are never serialized: replay rebuilds them
    from the hashed source cloud, so the estimate counts only accepted
    physical state rows, load/reaction/work scalars, and one compact final
    solver-summary row per state.
    """

    if not isinstance(config, StabilizedEngineeringGripConfig):
        raise TypeError("config must be a StabilizedEngineeringGripConfig")
    return _v5_size_for_counts(
        n_material,
        n_union_dofs,
        config,
        accepted_steps=accepted_steps,
        count_kind="observed",
    )


def estimate_stabilized_archive_bytes(
    n_material: int,
    n_union_dofs: int,
    config: StabilizedEngineeringGripConfig,
    *,
    accepted_steps: int | None = None,
) -> dict[str, Any]:
    """Return v5 nominal/worst-case or observed-count archive estimates."""

    if accepted_steps is not None:
        assert accepted_steps is not None
        return estimate_stabilized_archive_bytes_for_counts(
            n_material,
            n_union_dofs,
            config,
            accepted_steps=accepted_steps,
        )
    if not isinstance(config, StabilizedEngineeringGripConfig):
        raise TypeError("config must be a StabilizedEngineeringGripConfig")
    worst_counts = _planned_archive_counts(config, nominal=False)
    nominal_counts = _planned_archive_counts(config, nominal=True)
    worst = _v5_size_for_counts(
        n_material,
        n_union_dofs,
        config,
        **worst_counts,
        count_kind="theoretical_worst_case",
    )
    nominal = _v5_size_for_counts(
        n_material,
        n_union_dofs,
        config,
        **nominal_counts,
        count_kind="nominal_baseline",
    )
    return {
        **worst,
        "nominal_baseline": nominal,
        "theoretical_worst_case": dict(worst),
        "nominal_archive_bytes": nominal["archive_bytes"],
        "nominal_required_free_bytes": nominal["required_free_bytes"],
        "theoretical_worst_case_archive_bytes": worst["archive_bytes"],
        "theoretical_worst_case_required_free_bytes": worst["required_free_bytes"],
    }


def preflight_stabilized_archive_storage(
    output_path: Any,
    n_material: int,
    n_union_dofs: int,
    config: StabilizedEngineeringGripConfig,
    *,
    archive_budget_bytes: int | None = None,
    free_bytes: int | None = None,
) -> dict[str, Any]:
    """Preflight the v5 lean archive before adapter/continuation work."""

    filename = os.fsdecode(os.fspath(output_path))
    if not filename:
        raise ValueError("stabilized archive output path must be non-empty")
    path = Path(filename)
    probe = path if path.is_dir() else path.parent
    if not probe.exists():
        raise FileNotFoundError(f"stabilized archive output directory does not exist: {probe}")
    if not probe.is_dir():
        raise NotADirectoryError(f"stabilized archive output path is not a directory: {probe}")
    available = _archive_free_bytes(probe, free_bytes)
    budget = None if archive_budget_bytes is None else _validate_archive_budget(archive_budget_bytes)
    estimate = estimate_stabilized_archive_bytes(n_material, n_union_dofs, config)
    nominal_archive = int(estimate["nominal_baseline"]["archive_bytes"])
    worst_archive = int(estimate["archive_bytes"])
    if budget is not None and nominal_archive > budget:
        raise OSError(
            "nominal v5 stabilized archive estimate exceeds archive budget: "
            f"nominal_archive_bytes={nominal_archive}, archive_budget_bytes={budget}, "
            f"theoretical_worst_case_bytes={worst_archive}, path={path}"
        )
    required = (
        budget + STABILIZED_FILESYSTEM_RESERVE_BYTES
        if budget is not None
        else int(estimate["required_free_bytes"])
    )
    if available < required:
        raise OSError(
            "insufficient free storage for v5 stabilized archive preflight: "
            f"required={required}, available={available}, path={path}"
        )
    return {
        **estimate,
        "archive_budget_bytes": budget,
        "nominal_fits_budget": budget is None or nominal_archive <= budget,
        "theoretical_worst_case_fits_budget": (
            None if budget is None else worst_archive <= budget
        ),
        "budget_required_free_bytes": (
            None if budget is None else budget + STABILIZED_FILESYSTEM_RESERVE_BYTES
        ),
        "output_path": str(path),
        "filesystem_probe": str(probe),
        "available_bytes": available,
        "free_bytes": available,
        "worst_case_archive_bytes": worst_archive,
        "fits": True,
        "archive_layout": _stabilized_contract(
            config.acceptance_mode,
            failure=False,
            schedule=config.load_schedule is not None,
        )["archive_layout"],
    }


def _v5_archive_metadata(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    *,
    kind: str,
    completion_status: str,
    source_metadata: Mapping[str, Any],
    archive_budget_bytes: int | None,
    archive_storage_preflight: Mapping[str, Any] | None,
    counts: Mapping[str, int],
    failure: StabilizedEngineeringGripContinuationFailure | None,
) -> dict[str, Any]:
    mode = config.acceptance_mode
    schedule_mode = config.load_schedule is not None
    contract = _stabilized_contract(
        mode, failure=failure is not None, schedule=schedule_mode
    )
    if completion_status != contract["completion_status"]:
        raise ValueError("stabilized archive completion status does not match mode")
    source = _v5_json_value(source_metadata)
    if not isinstance(source, Mapping):
        raise TypeError("v5 source metadata must be a mapping")
    base_payload = _v5_json_value(engineering._config_payload(config.base))
    stabilized_payload = _v5_json_value(config.payload())
    metadata: dict[str, Any] = {
        "schema": contract["archive_schema"],
        "layout": contract["archive_layout"],
        "replay_contract": contract["replay_contract"],
        "kind": kind,
        "validated": False,
        "experimental": True,
        "field_equations_validated": False,
        "path_replayed": False,
        "rejected_trial_path_available": False,
        "acceptance_mode": mode,
        "acceptance_status": contract["acceptance_status"],
        "completion_status": completion_status,
        "material_hash": model.material_hash,
        "union_hash": model.union_hash,
        "static_digest": model.static_digest,
        "code_hash": _current_stabilized_source_bundle_hash(),
        "code_hash_kind": engineering._SOURCE_BUNDLE_HASH_KIND,
        "code_hash_files": list(_STABILIZED_SOURCE_BUNDLE_FILES),
        "youngs_modulus": model.static.youngs_modulus,
        "poisson_ratio": model.static.poisson_ratio,
        "k_res": model.static.k_res,
        "rank_rtol": model.static.rank_rtol,
        "delta": model.geometry.delta,
        "h_g": model.geometry.h_g,
        "Gc": model.Gc,
        "l0": model.l0,
        "visibility_tolerance": model.geometry.visibility_tolerance,
        "visibility_rule": engineering._VISIBILITY_RULE,
        "crack_region_rule": engineering._CRACK_REGION_RULE,
        "crack_localization_atol": 1.0e-10,
        "config": base_payload,
        "stabilized_config": stabilized_payload,
        "source": source,
        "source_sha256": source["sha256"],
        "configuration_sha256": _v5_json_sha256(
            {"base": base_payload, "stabilized": stabilized_payload}
        ),
        "archive_counts": dict(counts),
    }
    if schedule_mode:
        assert config.load_schedule is not None
        metadata["load_schedule"] = _v5_json_value(config.load_schedule.payload())
    if mode == STABILIZED_WARNING_ACCEPTANCE_MODE:
        metadata.update(
            {
                "warning_hard_gate_names": list(STABILIZED_WARNING_HARD_GATE_NAMES),
                "warning_bypassable_gate_names": list(STABILIZED_WARNING_GATE_NAMES),
            }
        )
    if archive_budget_bytes is not None:
        metadata["archive_budget_bytes"] = _validate_archive_budget(
            archive_budget_bytes
        )
    if archive_storage_preflight is not None:
        metadata["archive_storage_preflight"] = _archive_storage_metadata(
            archive_storage_preflight,
            archive_budget_bytes=archive_budget_bytes,
        )
    if failure is not None:
        metadata.update(
            {
                "failure_type": type(failure).__name__,
                "failure_message": str(failure),
                "failure_code": getattr(
                    failure,
                    "failure_code",
                    _stabilized_failure_code(mode, schedule=schedule_mode),
                ),
            }
        )
        if isinstance(failure, StabilizedEngineeringGripArchiveBudgetFailure):
            available = failure.available_bytes
            metadata["archive_budget_failure"] = {
                "stage": failure.stage,
                "stage_kind": failure.stage_kind,
                "stage_details": _v5_json_value(failure.stage_details),
                "exhaustion_kind": failure.exhaustion_kind,
                "actual_counts": _v5_json_value(failure.retained_counts),
                "retained_counts": _v5_json_value(failure.retained_counts),
                "prospective_counts": _v5_json_value(failure.prospective_counts),
                "prospective_delta": _v5_json_value(failure.prospective_delta),
                "retained_archive_bytes": int(
                    failure.retained_estimate["archive_bytes"]
                ),
                "retained_required_free_bytes": int(
                    failure.retained_estimate["required_free_bytes"]
                ),
                "estimated_archive_bytes": int(failure.estimate["archive_bytes"]),
                "required_free_bytes": int(failure.estimate["required_free_bytes"]),
                "archive_budget_bytes": int(failure.archive_budget_bytes),
                "available_bytes": None if available is None else int(available),
            }
    return metadata


def _v5_archive_arrays(
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    work_check: EngineeringGripWorkCheck,
    states: Sequence[EngineeringGripState],
    steps: Sequence[StabilizedEngineeringGripAcceptedStep],
    attempts: Sequence[StabilizedEngineeringGripAttempt],
    *,
    completion_status: str,
    source_metadata: Mapping[str, Any],
    failure: StabilizedEngineeringGripContinuationFailure | None = None,
    archive_budget_bytes: int | None = None,
    archive_storage_preflight: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    mode = config.acceptance_mode
    schedule_mode = config.load_schedule is not None
    contract = _stabilized_contract(
        mode, failure=failure is not None, schedule=schedule_mode
    )
    if not states or states[0].u_bar != 0.0:
        raise ValueError("v5 archive needs the exact initial state")
    if len(states) != len(steps) + 1:
        raise ValueError("v5 archive state/accepted-step counts are inconsistent")
    if completion_status != contract["completion_status"]:
        raise ValueError("stabilized archive completion status is invalid")
    for step in steps:
        if step.acceptance_mode != mode:
            raise ValueError("stabilized archive step mode is not paired")
    for attempt in attempts:
        if attempt.acceptance_mode != mode:
            raise ValueError("stabilized archive attempt mode is not paired")

    source = load_stabilized_source_cloud(source_metadata, model.geometry.delta)
    _v5_validate_source_matches_model(source, model)
    if len(attempts) == 0 and failure is None:
        raise ValueError("v5 completed archive needs attempt provenance")
    counts = {"accepted_steps": len(steps)}
    metadata = _v5_archive_metadata(
        model,
        config,
        kind="failure" if failure is not None else "completed",
        completion_status=completion_status,
        source_metadata=source_metadata,
        archive_budget_bytes=archive_budget_bytes,
        archive_storage_preflight=archive_storage_preflight,
        counts=counts,
        failure=failure,
    )
    state_u = np.asarray([state.u_bar for state in states], dtype=np.float64)
    state_q = np.stack([_f64(state.q, "v5 state q", (model.n_dofs,)) for state in states])
    state_d = np.stack(
        [_f64(state.d, "v5 state damage", (model.n_material,)) for state in states]
    )
    state_H = np.stack(
        [_f64(state.H, "v5 state history", (model.n_material,)) for state in states]
    )
    accepted_top = np.asarray(
        [step.diagnostics.top_reaction_after for step in steps], dtype=np.float64
    )
    state_top = np.concatenate((np.asarray([0.0], dtype=np.float64), accepted_top))
    state_external_work = _v5_external_work(state_u, state_top)
    solver_summaries = [
        _v5_initial_solver_summary(
            config.relaxation_policy,
            config.anderson_regularization_mu,
        )
    ]
    solver_summaries.extend(
        _v5_solver_summary_payload(
            index,
            step,
            relaxation_policy=config.relaxation_policy,
            acceptance_mode=mode,
            anderson_regularization_mu=config.anderson_regularization_mu,
        )
        for index, step in enumerate(steps, start=1)
    )
    arrays: dict[str, np.ndarray] = {
        "archive_schema": np.asarray(
            contract["archive_schema"],
            dtype=np.str_,
        ),
        "acceptance_mode": np.asarray(mode, dtype=np.str_),
        "acceptance_status": np.asarray(
            contract["acceptance_status"], dtype=np.str_
        ),
        "completion_status": np.asarray(completion_status, dtype=np.str_),
        "validated": np.asarray(False, dtype=np.bool_),
        "experimental": np.asarray(True, dtype=np.bool_),
        "metadata_json": np.asarray(_v5_json_text(metadata), dtype=np.str_),
        "state_u_bar": state_u,
        "state_union_displacement": state_q,
        "state_material_damage": state_d,
        "state_material_history": state_H,
        "state_digest": np.asarray([state.digest for state in states], dtype=np.str_),
        "state_acceptance_status": np.asarray(
            ["initial"] + [step.acceptance_status for step in steps], dtype=np.str_
        ),
        "accepted_top_reaction_after": np.asarray(
            [step.diagnostics.top_reaction_after for step in steps], dtype=np.float64
        ),
        "accepted_bottom_reaction_after": np.asarray(
            [step.diagnostics.bottom_reaction_after for step in steps], dtype=np.float64
        ),
        "accepted_anchor_reaction_after": np.asarray(
            [step.diagnostics.anchor_reaction_after for step in steps], dtype=np.float64
        ),
        "state_external_work": state_external_work,
        "solver_summary_json": _v5_string_array(solver_summaries),
        "work_check_u_bar": np.asarray(work_check.u_bar, dtype=np.float64),
        "work_check_epsilon": np.asarray(work_check.epsilon, dtype=np.float64),
        "work_check_reaction": np.asarray(work_check.reaction, dtype=np.float64),
        "work_check_reaction_fd": np.asarray(work_check.reaction_fd, dtype=np.float64),
        "work_check_relative_error": np.asarray(work_check.relative_error, dtype=np.float64),
        "work_check_relative_limit": np.asarray(work_check.relative_limit, dtype=np.float64),
        "work_check_passed": np.asarray(work_check.passed, dtype=np.bool_),
    }
    if schedule_mode:
        assert config.load_schedule is not None
        arrays.update(_schedule_archive_header_arrays(config.load_schedule))
    if mode == STABILIZED_WARNING_ACCEPTANCE_MODE:
        warning_names = tuple(STABILIZED_WARNING_GATE_NAMES)
        arrays.update(
            {
                "warning_gate_names": np.asarray(warning_names, dtype=np.str_),
                "state_bypassed_gate_mask": np.asarray(
                    [
                        [name in set(step.bypassed_gates) for name in warning_names]
                        for step in steps
                    ],
                    dtype=np.bool_,
                ).reshape(len(steps), len(warning_names)),
                "accepted_warning_present": np.asarray(
                    [
                        step.acceptance_status
                        == STABILIZED_WARNING_ACCEPTANCE_STATUS
                        for step in steps
                    ],
                    dtype=np.bool_,
                ),
                "accepted_warning_range_excess": np.asarray(
                    [
                        0.0
                        if step.warning_range_excess is None
                        else step.warning_range_excess
                        for step in steps
                    ],
                    dtype=np.float64,
                ),
                "accepted_warning_healing_magnitude": np.asarray(
                    [
                        0.0
                        if step.warning_healing_magnitude is None
                        else step.warning_healing_magnitude
                        for step in steps
                    ],
                    dtype=np.float64,
                ),
            }
        )
    if failure is not None:
        last = failure.last_committed_state
        arrays.update(
            {
                "failure_type": np.asarray(type(failure).__name__, dtype=np.str_),
                "failure_message": np.asarray(str(failure), dtype=np.str_),
                "failure_summary_json": np.asarray(
                    _v5_json_text(
                        _v5_failure_summary(
                            failure,
                            relaxation_policy=config.relaxation_policy,
                            acceptance_mode=mode,
                            anderson_regularization_mu=(
                                config.anderson_regularization_mu
                            ),
                        )
                    ),
                    dtype=np.str_,
                ),
                "last_committed_u_bar": np.asarray(last.u_bar, dtype=np.float64),
                "last_committed_damage_max": np.asarray(
                    np.max(last.d), dtype=np.float64
                ),
                "last_committed_history_max": np.asarray(
                    np.max(last.H), dtype=np.float64
                ),
            }
        )
    for name, value in arrays.items():
        if np.asarray(value).dtype.kind == "O":
            raise ValueError(f"v5 archive field {name} has forbidden object dtype")
    return arrays


def _v5_write_arrays(
    path: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    archive_budget_bytes: int | None,
    free_bytes: int | None,
) -> None:
    if archive_budget_bytes is not None:
        _precheck_archive_write(
            path,
            arrays,
            archive_budget_bytes=archive_budget_bytes,
            free_bytes=free_bytes,
        )
    _write_npz(path, arrays)


def write_stabilized_engineering_grip_archive(
    path: Any,
    result: StabilizedEngineeringGripResult,
    *,
    source_metadata: Mapping[str, Any],
    archive_budget_bytes: int | None = None,
    free_bytes: int | None = None,
    archive_storage_preflight: Mapping[str, Any] | None = None,
) -> None:
    """Write the v5 lean completed archive.

    Source provenance is mandatory: replay cannot reconstruct geometry from
    the lean archive itself.
    """

    if not isinstance(result, StabilizedEngineeringGripResult):
        raise TypeError("result must be a StabilizedEngineeringGripResult")
    if archive_budget_bytes is None and free_bytes is not None:
        raise ValueError("free_bytes requires archive_budget_bytes")
    result.validate()
    output = _prepare_archive_path(path)
    arrays = _v5_archive_arrays(
        result.model,
        result.config,
        result.work_check,
        result.states,
        result.accepted_steps,
        result.attempts,
        completion_status=result.completion_status,
        source_metadata=source_metadata,
        archive_budget_bytes=archive_budget_bytes,
        archive_storage_preflight=archive_storage_preflight,
    )
    _v5_write_arrays(
        output,
        arrays,
        archive_budget_bytes=archive_budget_bytes,
        free_bytes=free_bytes,
    )


def write_stabilized_engineering_grip_failure_archive(
    path: Any,
    failure: StabilizedEngineeringGripContinuationFailure,
    model: EngineeringGripModel,
    config: StabilizedEngineeringGripConfig,
    *,
    source_metadata: Mapping[str, Any],
    archive_budget_bytes: int | None = None,
    free_bytes: int | None = None,
    archive_storage_preflight: Mapping[str, Any] | None = None,
) -> None:
    """Write a v5 failure audit with accepted states and compact candidates."""

    if not isinstance(failure, StabilizedEngineeringGripContinuationFailure):
        raise TypeError("failure must be a StabilizedEngineeringGripContinuationFailure")
    if not isinstance(model, EngineeringGripModel) or not isinstance(
        config, StabilizedEngineeringGripConfig
    ):
        raise TypeError("failure archive model/config types are invalid")
    if failure.acceptance_mode != config.acceptance_mode:
        raise ValueError("failure archive mode does not match configuration")
    if isinstance(failure, StabilizedEngineeringGripArchiveBudgetFailure):
        failure_budget = failure.archive_budget_bytes
        if archive_budget_bytes is None:
            archive_budget_bytes = failure_budget
        elif _validate_archive_budget(archive_budget_bytes) != failure_budget:
            raise ValueError("failure archive budget does not match budget failure")
    if archive_budget_bytes is None and free_bytes is not None:
        raise ValueError("free_bytes requires archive_budget_bytes")
    attempts = tuple(failure.attempts)
    initial = EngineeringGripState(
        u_bar=0.0,
        union_displacement=np.zeros(model.n_dofs, dtype=np.float64),
        material_damage=np.zeros(model.n_material, dtype=np.float64),
        material_history=np.zeros(model.n_material, dtype=np.float64),
    )
    states: list[EngineeringGripState] = [initial]
    steps: list[StabilizedEngineeringGripAcceptedStep] = []
    for attempt in attempts:
        if attempt.accepted:
            if attempt.acceptance_status is None:
                raise ValueError("accepted failure-audit attempt lacks acceptance status")
            diagnostic = attempt.diagnostics
            if diagnostic.q_after is None or diagnostic.d_trial is None or diagnostic.H_next is None:
                raise ValueError("accepted failure-audit attempt lacks state fields")
            state = EngineeringGripState(
                u_bar=attempt.load_target,
                union_displacement=diagnostic.q_after,
                material_damage=diagnostic.d_trial,
                material_history=diagnostic.H_next,
            )
            step = StabilizedEngineeringGripAcceptedStep(
                load_start=attempt.load_start,
                load_target=attempt.load_target,
                increment=attempt.load_target - attempt.load_start,
                state=state,
                diagnostics=diagnostic,
                stagger_iterations=len(attempt.sweeps),
                cutback_count=attempt.cutback_count,
                acceptance_status=attempt.acceptance_status,
                bypassed_gates=attempt.bypassed_gates,
                warning_range_excess=attempt.warning_range_excess,
                warning_healing_magnitude=attempt.warning_healing_magnitude,
                acceptance_mode=attempt.acceptance_mode,
            )
            steps.append(step)
            states.append(state)
    last = failure.last_committed_state
    if (
        states[-1].u_bar != last.u_bar
        or not np.array_equal(states[-1].q, last.q)
        or not np.array_equal(states[-1].d, last.d)
        or not np.array_equal(states[-1].H, last.H)
    ):
        raise ValueError("v5 failure last committed state is not paired with audit")
    output = _prepare_archive_path(path)
    arrays = _v5_archive_arrays(
        model,
        config,
        failure.work_check,
        states,
        steps,
        attempts,
        completion_status=_stabilized_contract(
            config.acceptance_mode,
            failure=True,
            schedule=config.load_schedule is not None,
        )["completion_status"],
        source_metadata=source_metadata,
        failure=failure,
        archive_budget_bytes=archive_budget_bytes,
        archive_storage_preflight=archive_storage_preflight,
    )
    _v5_write_arrays(
        output,
        arrays,
        archive_budget_bytes=archive_budget_bytes,
        free_bytes=free_bytes,
    )


def load_stabilized_engineering_grip_archive(path: Any) -> dict[str, np.ndarray]:
    """Load a v5 archive with pickle explicitly disabled."""

    filename = os.fspath(path)
    with np.load(filename, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name, value in values.items():
        if value.dtype.kind == "O":
            raise ValueError(f"stabilized archive field {name} has forbidden object dtype")
    schema = _archive_string(values, "archive_schema")
    if schema not in {
        STABILIZED_ARCHIVE_SCHEMA,
        STABILIZED_FAILURE_ARCHIVE_SCHEMA,
        STABILIZED_WARNING_ARCHIVE_SCHEMA,
        STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
        STABILIZED_SCHEDULE_ARCHIVE_SCHEMA,
        STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA,
    }:
        raise ValueError(
            "stabilized archive loader only accepts v5, warning-v1, or "
            "matched-load schedule stabilized archives"
        )
    return values


def _v5_validate_field_set(
    values: Mapping[str, np.ndarray],
    *,
    failure: bool,
    warning: bool = False,
    schedule: bool = False,
) -> None:
    """Require exactly the lean v5 members; no operators or trial vectors."""

    expected = set(
        _WARNING_ARCHIVE_BASE_FIELDS if warning else _V5_ARCHIVE_BASE_FIELDS
    )
    if schedule:
        if warning:
            raise ValueError("scheduled stabilized archives cannot use warning mode")
        expected |= STABILIZED_SCHEDULE_ARCHIVE_FIELDS
    if failure:
        expected |= set(_WARNING_FAILURE_FIELDS if warning else _V5_FAILURE_FIELDS)
    missing = sorted(expected - set(values))
    extra = sorted(set(values) - expected)
    if missing or extra:
        raise ValueError(
            f"v5 archive layout is forged: missing={missing}, extra={extra}"
        )


def _validate_warning_archive_fields(
    values: Mapping[str, np.ndarray],
    states: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    failure: bool,
) -> dict[str, Any]:
    """Validate the additional state-level provenance of warning archives."""

    names = _v5_string_vector(
        values, "warning_gate_names", len(STABILIZED_WARNING_GATE_NAMES)
    )
    if tuple(str(name) for name in names) != STABILIZED_WARNING_GATE_NAMES:
        raise ValueError("stabilized warning gate names are not canonical")
    state_u = _f64(states["u"], "warning archive state loads")
    accepted_count = len(state_u) - 1
    mask = _bool(
        values["state_bypassed_gate_mask"],
        "warning archive bypassed-gate mask",
        (accepted_count, len(STABILIZED_WARNING_GATE_NAMES)),
    )
    present = _bool(
        values["accepted_warning_present"],
        "warning archive warning-present flags",
        (accepted_count,),
    )
    range_excess = _f64(
        values["accepted_warning_range_excess"],
        "warning archive range excess",
        (accepted_count,),
    )
    healing = _f64(
        values["accepted_warning_healing_magnitude"],
        "warning archive healing magnitude",
        (accepted_count,),
    )
    if np.any(range_excess < 0.0) or np.any(healing < 0.0):
        raise ValueError("stabilized warning excess fields must be non-negative")
    state_status = _v5_string_vector(values, "state_acceptance_status", len(state_u))
    if str(state_status[0]) != "initial":
        raise ValueError("stabilized warning initial status is forged")
    if len(summaries) != len(state_u):
        raise ValueError("stabilized warning summary/state counts are inconsistent")
    for index in range(accepted_count):
        status = str(state_status[index + 1])
        summary = summaries[index + 1]
        warning_row = status == STABILIZED_WARNING_ACCEPTANCE_STATUS
        if status not in (STABILIZED_ACCEPTANCE_STATUS, STABILIZED_WARNING_ACCEPTANCE_STATUS):
            raise ValueError("stabilized warning state acceptance status is invalid")
        if summary["acceptance_status"] != status:
            raise ValueError("stabilized warning state/summary status is not paired")
        expected_mask = np.asarray(
            [not summary["gate_passes"][name] for name in STABILIZED_WARNING_GATE_NAMES],
            dtype=bool,
        )
        if not np.array_equal(mask[index], expected_mask):
            raise ValueError("stabilized warning bypassed-gate provenance is forged")
        if bool(present[index]) != warning_row:
            raise ValueError("stabilized warning-present provenance is forged")
        if not warning_row:
            if bool(np.any(mask[index])) or range_excess[index] != 0.0 or healing[index] != 0.0:
                raise ValueError("strict stabilized row contains warning provenance")
            continue
        previous = np.asarray(states["d"][index], dtype=np.float64)
        current = np.asarray(states["d"][index + 1], dtype=np.float64)
        expected_range = max(0.0, -float(np.min(current)), float(np.max(current)) - 1.0)
        expected_healing = max(0.0, -float(np.min(current - previous)))
        if range_excess[index] != expected_range or healing[index] != expected_healing:
            raise ValueError("stabilized warning raw excess is not paired with state fields")
        if not bool(np.any(mask[index])):
            raise ValueError("stabilized warning row has no bypassed gate")
    if failure:
        # A failed target is not a committed warning state.  Its final solver
        # summary is validated separately, so no extra state row is expected.
        return {
            "gate_names": tuple(str(name) for name in names),
            "warning_accepted_steps": int(np.count_nonzero(present)),
        }
    return {
        "gate_names": tuple(str(name) for name in names),
        "warning_accepted_steps": int(np.count_nonzero(present)),
    }


def _v5_validate_solver_summaries(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    states: Mapping[str, Any],
    *,
    failure: bool,
    relaxation_policy: str | None = None,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    """Validate the one scalar solver-summary row retained per state."""

    raw_counts = metadata.get("archive_counts")
    if not isinstance(raw_counts, Mapping):
        raise TypeError("v5 metadata archive_counts must be a mapping")
    counts = _archive_counts(raw_counts, name="v5 metadata archive_counts")
    state_u = _f64(states["u"], "v5 summary state loads")
    summaries = _v5_json_records(values, "solver_summary_json")
    expected_states = len(state_u)
    if counts["accepted_steps"] != expected_states - 1:
        raise ValueError("v5 archive count does not match retained state rows")
    if len(summaries) != expected_states:
        raise ValueError("v5 solver-summary rows do not match retained states")
    for index, summary in enumerate(summaries):
        payload = _v5_validate_solver_summary(
            summary,
            f"v5 solver summary {index}",
            relaxation_policy=relaxation_policy,
            acceptance_mode=acceptance_mode,
            anderson_regularization_mu=anderson_regularization_mu,
        )
        if index == 0:
            if payload["summary_kind"] != "initial":
                raise ValueError("v5 first solver summary is not the initial row")
        elif payload["summary_kind"] != "accepted" or payload["state_index"] != index:
            raise ValueError("v5 accepted solver summary index is forged")
    if failure:
        if "failure_summary_json" not in values:
            raise ValueError("v5 failure archive is missing failure summary")
    elif "failure_summary_json" in values:
        raise ValueError("v5 completed archive contains a failure summary")
    return {"counts": counts, "summaries": summaries}


def _v5_validate_budget_failure(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    config: StabilizedEngineeringGripConfig,
    model: EngineeringGripModel,
    observed_counts: Mapping[str, int],
) -> dict[str, Any] | None:
    """Validate the compact archive-budget failure provenance in metadata."""

    if metadata.get("failure_code") != STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE:
        return None
    details = metadata.get("archive_budget_failure")
    if not isinstance(details, Mapping):
        raise ValueError("v5 archive budget failure details are missing")
    expected_fields = {
        "stage",
        "stage_kind",
        "stage_details",
        "exhaustion_kind",
        "actual_counts",
        "retained_counts",
        "prospective_counts",
        "prospective_delta",
        "retained_archive_bytes",
        "retained_required_free_bytes",
        "estimated_archive_bytes",
        "required_free_bytes",
        "archive_budget_bytes",
        "available_bytes",
    }
    if set(details) != expected_fields:
        raise ValueError("v5 archive budget failure fields are forged")
    stage_kind = details["stage_kind"]
    stage_details = _archive_stage_details(details["stage_details"], stage_kind)
    _validate_archive_stage_label(details["stage"], stage_kind, stage_details)
    if details["exhaustion_kind"] not in _ARCHIVE_BUDGET_EXHAUSTION_KINDS:
        raise ValueError("v5 archive budget exhaustion kind is forged")
    retained = _archive_counts(details["retained_counts"], name="retained counts")
    if details["actual_counts"] != dict(retained) or retained != dict(
        observed_counts
    ):
        raise ValueError(
            "v5 archive budget retained counts do not match the retained rows"
        )
    prospective = _archive_counts(
        details["prospective_counts"], name="prospective counts"
    )
    delta = _archive_count_delta(retained, prospective)
    if details["prospective_delta"] != dict(delta) or delta != _archive_stage_delta(
        stage_kind, stage_details
    ):
        raise ValueError(
            "v5 archive budget prospective counts do not match the stop stage"
        )
    budget = _validate_archive_budget(details["archive_budget_bytes"])
    if metadata.get("archive_budget_bytes") != budget:
        raise ValueError("v5 archive budget failure budget is forged")
    prospective_estimate = estimate_stabilized_archive_bytes_for_counts(
        model.n_material,
        model.n_dofs,
        config,
        accepted_steps=prospective["accepted_steps"],
    )
    retained_estimate = estimate_stabilized_archive_bytes_for_counts(
        model.n_material,
        model.n_dofs,
        config,
        accepted_steps=retained["accepted_steps"],
    )
    for name, expected in (
        ("estimated_archive_bytes", prospective_estimate["archive_bytes"]),
        ("required_free_bytes", prospective_estimate["required_free_bytes"]),
        ("retained_archive_bytes", retained_estimate["archive_bytes"]),
        (
            "retained_required_free_bytes",
            retained_estimate["required_free_bytes"],
        ),
    ):
        if details[name] != int(expected):
            raise ValueError(f"v5 archive budget failure {name} is forged")
    available = details["available_bytes"]
    if available is not None and (
        isinstance(available, bool) or not isinstance(available, int)
    ):
        raise ValueError("v5 archive budget available bytes are forged")
    if details["exhaustion_kind"] == "archive_budget":
        if int(prospective_estimate["archive_bytes"]) <= budget:
            raise ValueError(
                "v5 archive budget failure did not actually exceed its budget"
            )
    elif available is not None and int(prospective_estimate["required_free_bytes"]) <= available:
        raise ValueError(
            "v5 archive budget failure did not actually exceed free storage"
        )
    return dict(details)


def _v5_validate_failure_summary(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    states: Mapping[str, Any],
    *,
    relaxation_policy: str | None = None,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    """Validate the compact final failure summary without rejected vectors."""

    raw = _archive_string(values, "failure_summary_json")
    payload = _v5_json_record(raw, "failure_summary_json")
    expected_fields = {
        "available",
        "failure_code",
        "failure_type",
        "failure_message",
        "last_committed_u_bar",
        "failure_stage",
        "failure_reason",
        "load_start",
        "load_target",
        "increment",
        "cutback_count",
        "solver_summary",
    }
    if set(payload) != expected_fields:
        raise ValueError("v5 failure summary fields differ from the writer layout")
    _v5_validate_scalar_tree(payload, "v5 failure summary")
    if type(payload["available"]) is not bool:
        raise ValueError("v5 failure summary availability is invalid")
    for field in ("failure_code", "failure_type", "failure_message"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"v5 failure summary {field} is invalid")
    for field in ("failure_stage", "failure_reason"):
        if payload[field] is not None and (
            not isinstance(payload[field], str) or not payload[field]
        ):
            raise ValueError(f"v5 failure summary {field} is invalid")
    for field in ("last_committed_u_bar", "load_start", "load_target", "increment"):
        _v5_number(payload[field], f"v5 failure summary {field}", allow_none=True)
    cutback = _v5_integer(
        payload["cutback_count"], "v5 failure summary cutback_count", allow_none=True
    )
    if cutback is not None and cutback < 0:
        raise ValueError("v5 failure summary cutback_count is negative")
    if payload["failure_code"] != metadata.get("failure_code"):
        raise ValueError("v5 failure summary code differs from metadata")
    if payload["failure_type"] != metadata.get("failure_type"):
        raise ValueError("v5 failure summary type differs from metadata")
    if payload["failure_message"] != metadata.get("failure_message"):
        raise ValueError("v5 failure summary message differs from metadata")
    state_u = _f64(states["u"], "v5 failure summary state loads")
    last_committed = _scalar(
        payload["last_committed_u_bar"], "v5 failure summary last committed load"
    )
    if last_committed != float(state_u[-1]):
        raise ValueError("v5 failure summary last committed load is not the final state")
    solver_summary = payload["solver_summary"]
    if payload["available"]:
        if not isinstance(solver_summary, Mapping):
            raise ValueError("v5 available failure summary lacks solver summary")
        if not isinstance(payload["failure_stage"], str) or not isinstance(
            payload["failure_reason"], str
        ):
            raise ValueError("v5 available failure summary lacks failure provenance")
        load_start = _v5_number(
            payload["load_start"], "v5 failure summary load_start"
        )
        load_target = _v5_number(
            payload["load_target"], "v5 failure summary load_target"
        )
        increment = _v5_number(
            payload["increment"], "v5 failure summary increment"
        )
        assert load_start is not None and load_target is not None and increment is not None
        if load_target <= load_start or increment != load_target - load_start:
            raise ValueError("v5 failure summary load interval is invalid")
        if load_start != float(state_u[-1]):
            raise ValueError("v5 failure summary does not start at the final state")
        summary = _v5_validate_solver_summary(
            solver_summary,
            "v5 failure solver summary",
            relaxation_policy=relaxation_policy,
            acceptance_mode=acceptance_mode,
            anderson_regularization_mu=anderson_regularization_mu,
        )
        if (
            summary["summary_kind"] != "failure"
            or summary["load_start"] != load_start
            or summary["load_target"] != load_target
            or summary["increment"] != increment
            or summary["cutback_count"] != cutback
            or summary["failure_stage"] != payload["failure_stage"]
            or summary["failure_reason"] != payload["failure_reason"]
        ):
            raise ValueError("v5 failure summary is not paired with its solver row")
    else:
        if solver_summary is not None:
            raise ValueError("v5 unavailable failure summary retains a solver row")
        if payload["failure_stage"] is not None or not payload["failure_reason"]:
            raise ValueError("v5 unavailable failure summary provenance is invalid")
        if any(
            payload[field] is not None
            for field in ("load_start", "load_target", "increment", "cutback_count")
        ):
            raise ValueError("v5 unavailable failure summary retains candidate fields")
    return payload


def _v5_validate_archive_summary(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    states: Mapping[str, Any],
    *,
    failure: bool,
    relaxation_policy: str | None = None,
    acceptance_mode: str = STABILIZED_ACCEPTANCE_MODE,
    anderson_regularization_mu: float | None = None,
) -> dict[str, Any]:
    """Validate retained summary rows and the optional failure summary."""

    summaries = _v5_validate_solver_summaries(
        values,
        metadata,
        states,
        failure=failure,
        relaxation_policy=relaxation_policy,
        acceptance_mode=acceptance_mode,
        anderson_regularization_mu=anderson_regularization_mu,
    )
    failure_summary = (
        _v5_validate_failure_summary(
            values,
            metadata,
            states,
            relaxation_policy=relaxation_policy,
            acceptance_mode=acceptance_mode,
            anderson_regularization_mu=anderson_regularization_mu,
        )
        if failure
        else None
    )
    return {
        "counts": summaries["counts"],
        "summaries": summaries["summaries"],
        "failure_summary": failure_summary,
    }


def _v5_validate_schedule_sequence(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    config: StabilizedEngineeringGripConfig,
    states: Mapping[str, Any],
    summaries: Mapping[str, Any],
    *,
    failure: bool,
) -> dict[str, Any]:
    """Validate the exact accepted prefix and next-target schedule identity."""

    schedule = config.load_schedule
    if schedule is None:
        raise ValueError("scheduled replay configuration is missing its schedule")
    loads = _f64(states["u"], "scheduled replay state loads")
    accepted_steps = len(loads) - 1
    if accepted_steps > schedule.target_count:
        raise ValueError("scheduled archive contains more accepted states than its schedule")
    expected_prefix = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.asarray(schedule.targets[:accepted_steps], dtype=np.float64),
        )
    )
    if not np.array_equal(loads, expected_prefix):
        raise ValueError(
            "scheduled archive state_u_bar is not the exact zero-plus-schedule prefix"
        )
    failure_summary = summaries.get("failure_summary")
    if not failure:
        if accepted_steps != schedule.target_count:
            raise ValueError(
                "scheduled completed archive does not contain every reviewed target"
            )
        if loads[-1] != schedule.load_final:
            raise ValueError("scheduled completed archive endpoint is not load_final")
        if failure_summary is not None:
            raise ValueError("scheduled completed archive contains a failure summary")
        return {
            "target_count": schedule.target_count,
            "target_index": None,
            "next_target": None,
            "completed": True,
        }

    if accepted_steps >= schedule.target_count:
        raise ValueError("scheduled failure archive has no exact next target")
    next_target = float(schedule.targets[accepted_steps])
    if failure_summary is not None and failure_summary["available"]:
        for name, expected in (
            ("load_start", float(loads[-1])),
            ("load_target", next_target),
            ("increment", next_target - float(loads[-1])),
            ("cutback_count", 0),
        ):
            if failure_summary[name] != expected:
                raise ValueError(
                    f"scheduled failure summary {name} is not the exact next target binding"
                )
    elif metadata.get("failure_code") == STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE:
        budget_failure = metadata.get("archive_budget_failure")
        if not isinstance(budget_failure, Mapping):
            raise ValueError("scheduled budget failure lacks archive details")
        stage_details = budget_failure.get("stage_details")
        if not isinstance(stage_details, Mapping):
            raise ValueError("scheduled budget failure lacks target stage details")
        if stage_details.get("target_u_bar") != next_target:
            raise ValueError("scheduled budget failure next target is forged")
    else:
        raise ValueError("scheduled failure archive lacks an exact next-target summary")
    return {
        "target_count": schedule.target_count,
        "target_index": accepted_steps,
        "next_target": next_target,
        "completed": False,
    }


def replay_stabilized_engineering_grip_archive(path: Any) -> dict[str, Any]:
    """Independently replay one v5 lean stabilized archive.

    This function never constructs the production controller, mechanics
    solver, or damage solver.  It verifies the v5 header/schema/layout,
    metadata canonicality, retained-state and solver-summary counts; reloads
    the external source cloud by exact
    path/SHA256/size; rebuilds the exact delta=0.048 union model; and then
    independently evaluates every committed state's hard displacement
    constraints, mechanics residual and CG status, force/moment conservation,
    reactions, Amor history consistency and monotonicity, the AT2 damage
    residual, and the centered work check.

    The lean archive retains only accepted physical states and final solver
    summaries.
    The exact rejected-trial numerical path cannot be reconstructed from it,
    so a successful replay reports ``path_replayed=False`` and
    ``rejected_trial_path_available=False`` while reporting truthfully that
    the accepted-state field equations were validated.  The archive itself
    remains ``validated=False`` and ``experimental=True``.
    """

    values = load_stabilized_engineering_grip_archive(path)
    schema = _archive_string(values, "archive_schema")
    warning = schema in {
        STABILIZED_WARNING_ARCHIVE_SCHEMA,
        STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
    }
    failure = schema in {
        STABILIZED_FAILURE_ARCHIVE_SCHEMA,
        STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
        STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA,
    }
    schedule = schema in {
        STABILIZED_SCHEDULE_ARCHIVE_SCHEMA,
        STABILIZED_SCHEDULE_FAILURE_ARCHIVE_SCHEMA,
    }
    _v5_validate_field_set(
        values, failure=failure, warning=warning, schedule=schedule
    )
    metadata, base_config, config, failure, model, source = _v5_validate_header(
        values, schedule=schedule
    )
    del source
    static = _v5_replay_static(model)
    work_check = _v5_replay_work_check(values, static, metadata, base_config)
    states = _v5_replay_states(
        values,
        metadata,
        base_config,
        config,
        model,
        static,
        failure,
    )
    summaries = _v5_validate_archive_summary(
        values,
        metadata,
        states,
        failure=failure,
        relaxation_policy=config.relaxation_policy,
        acceptance_mode=config.acceptance_mode,
        anderson_regularization_mu=config.anderson_regularization_mu,
    )
    schedule_sequence = (
        _v5_validate_schedule_sequence(
            values,
            metadata,
            config,
            states,
            summaries,
            failure=failure,
        )
        if schedule
        else None
    )
    if schedule:
        for index, summary in enumerate(summaries["summaries"]):
            if index > 0 and summary["cutback_count"] != 0:
                raise ValueError("scheduled archive accepted summary contains a cutback")
        failure_summary = summaries["failure_summary"]
        if failure_summary is not None and failure_summary["cutback_count"] != 0:
            raise ValueError("scheduled archive failure summary contains a cutback")
    warning_provenance = (
        _validate_warning_archive_fields(
            values,
            states,
            summaries["summaries"],
            failure=failure,
        )
        if warning
        else None
    )
    warning_state_present = bool(
        warning_provenance is not None
        and warning_provenance["warning_accepted_steps"] > 0
    )
    budget_failure = _v5_validate_budget_failure(
        values, metadata, config, model, summaries["counts"]
    )
    completion_status = _archive_string(values, "completion_status")
    replay: dict[str, Any] = {
        # ``schema`` is retained as a short alias; ``archive_schema`` is the
        # authoritative numerical archive identity used by report bindings.
        "schema": schema,
        "archive_schema": schema,
        "archive_layout": _stabilized_contract(
            config.acceptance_mode, failure=failure, schedule=schedule
        )["archive_layout"],
        "archive_sha256": _archive_sha256(path),
        "replay_contract": _stabilized_contract(
            config.acceptance_mode, failure=failure, schedule=schedule
        )["replay_contract"],
        "acceptance_mode": config.acceptance_mode,
        "acceptance_status": _stabilized_contract(
            config.acceptance_mode, failure=failure, schedule=schedule
        )["acceptance_status"],
        "completion_status": completion_status,
        "source_bundle_hash": metadata["code_hash"],
        "source_path": metadata["source"]["path"],
        "source_sha256": metadata["source_sha256"],
        "validated": False,
        "experimental": True,
        "field_equations_validated": bool(
            summaries["counts"]["accepted_steps"] > 0
            and states["field_equations_validated"]
            and not warning_state_present
        ),
        "path_replayed": False,
        "rejected_trial_path_available": False,
        "accepted_steps": summaries["counts"]["accepted_steps"],
        "solver_summary_rows": len(summaries["summaries"]),
        "states": len(np.asarray(states["u"])),
        "anderson_provenance": _v5_anderson_provenance(
            config.relaxation_policy
        ),
        "work_check": work_check,
        "replayed": True,
    }
    if schedule:
        assert config.load_schedule is not None
        replay["load_schedule"] = config.load_schedule.payload()
        assert schedule_sequence is not None
        replay["schedule_target_count"] = schedule_sequence["target_count"]
        replay["schedule_target_index"] = schedule_sequence["target_index"]
        replay["schedule_next_target"] = schedule_sequence["next_target"]
        replay["schedule_completed"] = schedule_sequence["completed"]
    if warning_provenance is not None:
        replay["warning_acceptance"] = warning_provenance
    if _is_anderson_residual_policy(config.relaxation_policy):
        replay["relaxation_policy"] = config.relaxation_policy
        replay["selection_merit_identity"] = (
            _anderson_merit_identity(config.relaxation_policy)
        )
    if _is_anderson_regularized_policy(config.relaxation_policy):
        if config.anderson_regularization_mu is None:
            raise ValueError("regularized Anderson replay is missing configured mu")
        replay["anderson_regularization_mu"] = config.anderson_regularization_mu
    if (archive_budget := metadata.get("archive_budget_bytes")) is not None:
        replay["archive_budget_bytes"] = int(archive_budget)
    if "archive_storage_preflight" in metadata:
        replay["archive_storage_preflight"] = dict(
            metadata["archive_storage_preflight"]
        )
    if budget_failure is not None:
        replay["archive_budget_failure"] = budget_failure
    if summaries["failure_summary"] is not None:
        replay["failure_summary"] = summaries["failure_summary"]
    return replay
