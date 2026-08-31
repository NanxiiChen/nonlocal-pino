"""Lean direct fracture2d reference with engineering volume grips.

This module is intentionally separate from the matched-interface Candidate A2
controller.  It uses the same approved phase-measured mechanics energy and the
same material-only fracture equations, but the continuation coordinate is the
prescribed vertical displacement of every added collar particle.  The top
reaction is consequently the residual sum over the complete top volume grip.

The controller has two explicit acceptance modes.  ``strict`` is unchanged:
every post-sweep acceptance gate and the final crack-localization gate must
pass and completed results are validated.  ``warning`` is the separately
archived engineering diagnostic mode.  Its explicit safety policy is either
``bounded`` (the existing finite raw damage range/healing limit) or
``unbounded`` (finite raw range/healing violations are warning-only regardless
of magnitude).  In both policies, warning acceptance is considered only on a
final complete sweep and every warning-mode artifact stays
``validated=False``.

The source-backed basis is the target-inclusive bond-associated kinematics,
the plane-strain linear elastic map, and the Amor positive-energy split already
audited for the direct fracture2d baseline.  The union phase measure, collar
construction, mixed component hard constraints, material-source history, and
engineering reaction are project-owned discrete choices.  No matched physical
opening, normalized unit solve, opening scaling, effective reaction, or
per-sweep work solve belongs to this path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.sparse import coo_matrix, diags, issparse
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, cg, spsolve

from pinn.operators import ExplicitVolumeSparseD

from .nonlocal_a2_coupled import (
    A2FixedDamageView,
    A2MechanicsSolveResult,
    A2StaticMechanics,
    compute_a2_amor_history,
    solve_a2_fixed_outer_displacement,
)
from .nonlocal_baseline import (
    FixedHistoryDamageResult,
    GateTolerance,
    solve_fixed_history_damage,
)
from .nonlocal_engineering_grip_jax import JaxA2CPUReplayAdapter
from .nonlocal_grip_diagnostic import (
    AffineConstraints,
    LineQuadrature,
    UnionGripGeometry,
    _build_union_visibility_graph,
    _triangle_quadrature,
    build_volume_grip_constraints,
    physical_interface_line_weights,
)


__all__ = [
    "ACCEPTANCE_MODES",
    "WARNING_SAFETY_POLICIES",
    "ARCHIVE_SCHEMA",
    "WARNING_ARCHIVE_SCHEMA",
    "UNBOUNDED_WARNING_ARCHIVE_SCHEMA",
    "WARNING_ACCEPTANCE_STATUS",
    "WARNING_COMPLETION_STATUS",
    "WARNING_CONTINUATION_FAILURE_STATUS",
    "WARNING_GATE_NAMES",
    "EngineeringGripAcceptedStep",
    "EngineeringGripAttempt",
    "EngineeringGripConfig",
    "EngineeringGripContinuationFailure",
    "EngineeringGripDiagnostics",
    "EngineeringGripModel",
    "EngineeringGripResult",
    "EngineeringGripSweepFailure",
    "EngineeringGripState",
    "EngineeringGripWorkCheck",
    "EngineeringGripWorkCheckFailure",
    "CrackRegionDiagnostics",
    "GateTolerance",
    "A2FixedDamageView",
    "A2MechanicsSolveResult",
    "A2StaticMechanics",
    "JaxA2CPUReplayAdapter",
    "build_crack_region_mask",
    "build_engineering_grip_model",
    "build_volume_grip_constraints",
    "compute_crack_region_diagnostics",
    "compute_interface_opening_diagnostic",
    "compute_material_source_amor",
    "load_engineering_grip_archive",
    "replay_engineering_grip_archive",
    "run_strict_engineering_grip",
    "run_strict_continuation",
    "run_warning_engineering_grip",
    "run_zero_damage_work_check",
    "solve_engineering_grip_fixed_outer_displacement",
    "solve_fixed_outer_displacement",
    "write_engineering_grip_archive",
    "build_volume_grip_model",
    "compute_material_amor_history",
]


_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
_BOOL = np.dtype(bool)
_EPS64 = np.finfo(np.float64).eps
_RELATIVE_FLOOR = float(np.finfo(np.float64).tiny)
_REPLAY_BOUND = 512.0 * _EPS64
_REPLAY_COMPARISON_BOUND = 3.0 * _REPLAY_BOUND
_IDENTITY_2 = np.eye(2, dtype=np.float64)
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_KERNEL = "cubic"
_VISIBILITY_RULE = "independent union-polygon segment containment"
_STALE_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-v1"
ARCHIVE_SCHEMA = "fracture2d-engineering-grip-v2"
WARNING_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-warning-v1"
UNBOUNDED_WARNING_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-warning-unbounded-v1"
_ACCEPTANCE_MODE = "strict"
_ACCEPTANCE_STATUS = "strict_passed"
_COMPLETION_STATUS = "completed"
_WARNING_ACCEPTANCE_MODE = "warning"
_WARNING_SAFETY_POLICY_BOUNDED = "bounded"
_WARNING_SAFETY_POLICY_UNBOUNDED = "unbounded"
_WARNING_ACCEPTANCE_STATUS = "warning_accepted"
_WARNING_COMPLETION_STATUS = "warning_completed"
_WARNING_CONTINUATION_FAILURE_STATUS = "warning_continuation_failure"
# Public spellings are kept alongside the private constants so orchestration
# and audit consumers do not need to duplicate status literals.
WARNING_ACCEPTANCE_STATUS = _WARNING_ACCEPTANCE_STATUS
WARNING_COMPLETION_STATUS = _WARNING_COMPLETION_STATUS
WARNING_CONTINUATION_FAILURE_STATUS = _WARNING_CONTINUATION_FAILURE_STATUS
# Acceptance modes shared by the strict and warning engineering controllers.
# ``WARNING_GATE_NAMES`` is the canonical fixed ordering of every sweep gate a
# warning-mode commit may bypass at the final complete sweep.  The ``finite``
# and ``mechanics_before``/``mechanics_after`` gates are deliberately absent:
# non-finite values and failed mechanics/CG solves never return a candidate
# and stay on the audited typed-failure/cutback path in both modes.
ACCEPTANCE_MODES = ("strict", "warning")
WARNING_SAFETY_POLICIES = (
    _WARNING_SAFETY_POLICY_BOUNDED,
    _WARNING_SAFETY_POLICY_UNBOUNDED,
)
WARNING_GATE_NAMES = (
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
)
_WARNING_GATE_INDEX = {name: index for index, name in enumerate(WARNING_GATE_NAMES)}
_WARNING_HARD_GATE_NAMES = ("finite", "mechanics_before", "mechanics_after")
_WARNING_INCOMPLETE_STAGES = (
    "mechanics_before",
    "mechanics_before_gate",
    "amor_before",
    "damage",
    "mechanics_after",
    "amor_after",
    "mechanics_after_gate",
)
_SOURCE_BUNDLE_HASH_KIND = "deterministic-source-bundle-sha256"
_SOURCE_BUNDLE_FILES = (
    # The engineering controller itself and every local formulation module it
    # imports directly or transitively for geometry, mechanics, and damage.
    "examples/fracture2d/nonlocal_engineering_grip.py",
    "examples/fracture2d/nonlocal_engineering_grip_jax.py",
    "examples/fracture2d/nonlocal_a2_coupled.py",
    "examples/fracture2d/nonlocal_grip_diagnostic.py",
    "examples/fracture2d/nonlocal_geometry.py",
    "examples/fracture2d/nonlocal_baseline.py",
    "examples/fracture2d/nonlocal_mechanics.py",
    "src/pinn/operators.py",
)
_CRACK_TIP = np.asarray([0.5, 0.5], dtype=np.float64)
_CRACK_REGION_RULE = "material distance <= delta from exact tip particle (0.5, 0.5)"
_CRACK_LOCALIZATION_ATOL = 1.0e-10
_CRACK_INTERFACE_RULE = (
    "positive damage/history maxima must not occur on material y=0 or y=1 interfaces; "
    "exact-zero fields are exempt"
)


def _immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Make a bytes-backed immutable numerical array."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    result = np.frombuffer(source.tobytes(order="C"), dtype=source.dtype)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar, got shape {array.shape}")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _top_collar_coordinate_roundoff_bound(
    coordinate: Any, interface_coordinate: Any, offset: Any
) -> np.ndarray:
    """Bound top-row coordinate subtraction/addition roundoff in float64.

    The serialized top ordinate is formed by adding the interface ordinate and
    a collar offset, while replay derives the offset by subtracting the
    interface ordinate again.  The expected offset also contains the rounded
    integer-spacing multiplication.  With unit roundoff ``u``, four ``u``
    budgets applied to the magnitudes of those operands safely cover these
    rounded operations without treating a meaningful coordinate change as
    equivalent.
    """

    coordinate_array = np.asarray(coordinate)
    interface_array = np.asarray(interface_coordinate)
    offset_array = np.asarray(offset)
    if (
        coordinate_array.dtype != _FLOAT64
        or interface_array.dtype != _FLOAT64
        or interface_array.shape != ()
        or offset_array.dtype != _FLOAT64
    ):
        raise ValueError("top-collar roundoff inputs must use exact float64")
    scale = (
        np.abs(coordinate_array)
        + abs(float(interface_array))
        + np.abs(offset_array)
    )
    bound = 4.0 * (_EPS64 / 2.0) * scale
    if not np.all(np.isfinite(bound)) or np.any(bound < 0.0):
        raise ValueError("top-collar roundoff bound is invalid")
    return np.asarray(bound, dtype=np.float64)


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(array)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _f64(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
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


def _max_abs(value: Any) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _relative(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), _RELATIVE_FLOOR
    )


def _validate_acceptance_mode(value: Any) -> str:
    """Return the exact acceptance mode string or fail loudly."""

    if isinstance(value, str) and value in ACCEPTANCE_MODES:
        return value
    raise ValueError(
        "acceptance mode must be one of "
        + "/".join(ACCEPTANCE_MODES)
        + f", got {value!r}"
    )


def _validate_warning_safety_policy(value: Any) -> str:
    """Return the exact warning safety policy string or fail loudly."""

    if isinstance(value, str) and value in WARNING_SAFETY_POLICIES:
        return value
    raise ValueError(
        "warning safety policy must be one of "
        + "/".join(WARNING_SAFETY_POLICIES)
        + f", got {value!r}"
    )


def _validate_warning_safety_limit(value: Any) -> float:
    """Return the explicit warning range/healing safety limit."""

    if value is None:
        raise ValueError(
            "warning_safety_limit is required for warning acceptance mode"
        )
    limit = _scalar(value, "warning_safety_limit")
    if limit <= 0.0:
        raise ValueError("warning_safety_limit must be positive")
    return limit


def _validate_warning_safety_configuration(
    policy: Any, warning_safety_limit: Any
) -> tuple[str, float | None]:
    """Validate the policy/limit pair used by warning acceptance.

    The bounded policy retains the original positive finite limit contract.
    Unbounded warning mode has no numerical sentinel for its absent limit;
    callers must pass ``None`` and archives record a separate presence flag.
    """

    validated_policy = _validate_warning_safety_policy(policy)
    if validated_policy == _WARNING_SAFETY_POLICY_BOUNDED:
        return validated_policy, _validate_warning_safety_limit(warning_safety_limit)
    if warning_safety_limit is not None:
        raise ValueError(
            "warning_safety_limit is forbidden for unbounded warning safety policy"
        )
    return validated_policy, None


def _validate_bypassed_gates(values: Any) -> tuple[str, ...]:
    """Validate a canonical ordered bypassed-gate name tuple."""

    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        raise TypeError("bypassed_gates must be a tuple of canonical gate names")
    names = tuple(values)
    previous_index = -1
    for name in names:
        if not isinstance(name, str) or name not in _WARNING_GATE_INDEX:
            raise ValueError(
                f"bypassed gate {name!r} is not a canonical warning gate name"
            )
        index = _WARNING_GATE_INDEX[name]
        if index <= previous_index:
            raise ValueError(
                "bypassed_gates must list canonical gate names in order and "
                "without duplicates"
            )
        previous_index = index
    return names


def _failed_warning_gates(diagnostics: Any) -> tuple[str, ...]:
    """Return the canonical ordered list of failed bypassable gates."""

    passes = getattr(diagnostics, "gate_passes", None)
    expected = set(_WARNING_HARD_GATE_NAMES) | set(WARNING_GATE_NAMES)
    if not isinstance(passes, Mapping) or set(passes) != expected:
        raise ValueError("warning diagnostic gate_passes must contain the canonical gates")
    return tuple(name for name in WARNING_GATE_NAMES if not bool(passes[name]))


def _warning_excesses(
    d_next: np.ndarray, committed_damage: np.ndarray
) -> tuple[float, float]:
    """Raw final-sweep range/healing magnitudes without any clipping."""

    candidate = np.asarray(d_next, dtype=np.float64)
    committed = np.asarray(committed_damage, dtype=np.float64)
    if candidate.shape != committed.shape:
        raise ValueError("warning excess damage shapes are incompatible")
    if candidate.ndim != 1 or candidate.size == 0:
        raise ValueError("warning excess damage arrays must be nonempty vectors")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(committed)):
        raise FloatingPointError("warning excess damage arrays must be finite")
    range_excess = max(0.0, -float(np.min(candidate)), float(np.max(candidate)) - 1.0)
    healing_magnitude = max(0.0, -float(np.min(candidate - committed)))
    return range_excess, healing_magnitude


def _require_warning_excess_records(
    range_excess: Any, healing_magnitude: Any
) -> tuple[float, float]:
    """Validate one recorded raw warning excess pair."""

    first = _scalar(range_excess, "warning_range_excess")
    second = _scalar(healing_magnitude, "warning_healing_magnitude")
    if first < 0.0 or second < 0.0:
        raise ValueError("warning excess records must be non-negative")
    return first, second


def _warning_acceptance_decision(
    diagnostics: Any,
    committed_damage: Any,
    warning_safety_limit: Any = None,
    *,
    warning_safety_policy: Any = _WARNING_SAFETY_POLICY_BOUNDED,
) -> tuple[bool, tuple[str, ...], float, float]:
    """Return the final-sweep warning decision and its raw safety values.

    This is deliberately shared by the production controller, in-memory
    result validation, and independent archive replay.  It does not inspect or
    alter any numerical field; it only classifies a complete diagnostic.
    """

    # Keep the original third positional argument as the bounded-limit API,
    # while also accepting the explicit policy as the third positional value
    # for the new warning entry point.
    if (
        isinstance(warning_safety_limit, str)
        and warning_safety_policy == _WARNING_SAFETY_POLICY_BOUNDED
    ):
        warning_safety_policy = warning_safety_limit
        warning_safety_limit = None
    policy, limit = _validate_warning_safety_configuration(
        warning_safety_policy, warning_safety_limit
    )
    d_next = getattr(diagnostics, "d_next", None)
    if d_next is None:
        return False, (), math.inf, math.inf
    range_excess, healing_magnitude = _warning_excesses(
        np.asarray(d_next, dtype=np.float64),
        np.asarray(committed_damage, dtype=np.float64),
    )
    if not bool(getattr(diagnostics, "complete_sweep", False)):
        return False, (), range_excess, healing_magnitude
    if bool(getattr(diagnostics, "passed", False)):
        return False, (), range_excess, healing_magnitude
    gate_passes = getattr(diagnostics, "gate_passes", None)
    if not isinstance(gate_passes, Mapping):
        raise ValueError("warning diagnostic gate_passes must be a mapping")
    expected_gate_names = set(_WARNING_HARD_GATE_NAMES) | set(WARNING_GATE_NAMES)
    if set(gate_passes) != expected_gate_names:
        raise ValueError("warning diagnostic gate_passes do not contain the canonical gates")
    hard_failed = tuple(
        name for name in _WARNING_HARD_GATE_NAMES if not bool(gate_passes[name])
    )
    bypassed = _failed_warning_gates(diagnostics)
    if policy == _WARNING_SAFETY_POLICY_BOUNDED:
        assert limit is not None
        safety_pass = range_excess <= limit and healing_magnitude <= limit
    else:
        safety_pass = True
    accepted = bool(bypassed and not hard_failed and safety_pass)
    return accepted, bypassed if accepted else (), range_excess, healing_magnitude


def _array_close(
    first: Any, second: Any, name: str, limit: float = 2.0e-9
) -> None:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"{name} shape/dtype does not match replay")
    defect = _max_abs(left - right)
    scale = max(_max_abs(left), _max_abs(right), 1.0)
    if not np.isfinite(defect) or defect > limit * scale:
        raise ValueError(f"{name} does not match replay: defect={defect:.17g}")


def _compare_damage_residual_reconstruction(
    strong_actual: Any,
    volume_actual: Any,
    damage: Any,
    history: Any,
    matrix: Any,
    volumes: Any,
    Gc: Any,
    l0: Any,
    name: str,
) -> None:
    """Compare damage residuals using the scale of their signed constituents.

    The AT2 strong residual is the cancellation-prone sum

    ``Gc/l0*d + Gc*l0*(D @ d) + 2*H*d - 2*H``.

    A reconstruction error is therefore bounded componentwise by 512 float64
    epsilons times the sum of the absolute constituent terms.  The Euclidean
    residual is the same strong residual multiplied by the strictly positive
    particle volume, so its bound is multiplied by that volume as well.  No
    unit scale floor is appropriate for either comparison.
    """

    d = _f64(damage, f"{name} damage")
    if d.ndim != 1 or d.size == 0:
        raise ValueError(f"{name} damage must be a nonempty one-dimensional array")
    H = _f64(history, f"{name} history", d.shape)
    V = _f64(volumes, f"{name} volumes", d.shape)
    if np.any(V <= 0.0):
        raise ValueError(f"{name} volumes must be strictly positive")

    if issparse(matrix):
        M = matrix.tocsr()
        if M.dtype != _FLOAT64:
            raise ValueError(
                f"{name} damage matrix must use exact float64 dtype, got {M.dtype}"
            )
        if M.shape != (d.size, d.size):
            raise ValueError(
                f"{name} damage matrix must have shape {(d.size, d.size)}, "
                f"got {M.shape}"
            )
        if M.nnz and not np.all(np.isfinite(M.data)):
            raise ValueError(f"{name} damage matrix must contain only finite values")
    else:
        M = _f64(matrix, f"{name} damage matrix", (d.size, d.size))

    gc_value = _scalar(Gc, f"{name} Gc")
    l0_value = _scalar(l0, f"{name} l0")
    if gc_value <= 0.0 or l0_value <= 0.0:
        raise ValueError(f"{name} Gc/l0 must be positive")

    graph = np.asarray(M @ d, dtype=np.float64)
    if graph.shape != d.shape or not np.all(np.isfinite(graph)):
        raise ValueError(f"{name} damage graph product must be finite with damage shape")
    local = np.asarray(gc_value / l0_value * d, dtype=np.float64)
    graph_term = np.asarray(gc_value * l0_value * graph, dtype=np.float64)
    history_damage = np.asarray(2.0 * H * d, dtype=np.float64)
    history_constant = np.asarray(2.0 * H, dtype=np.float64)
    terms = (local, graph_term, history_damage, history_constant)
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise ValueError(f"{name} damage residual constituents must be finite")
    term_scale = np.zeros(d.shape, dtype=np.float64)
    for term in terms:
        term_scale += np.abs(term)
    bound = np.asarray(_REPLAY_BOUND * term_scale, dtype=np.float64)
    expected = np.asarray(
        local + graph_term + history_damage - history_constant,
        dtype=np.float64,
    )
    expected_volume = np.asarray(V * expected, dtype=np.float64)
    volume_bound = np.asarray(V * bound, dtype=np.float64)
    if not all(
        np.all(np.isfinite(value))
        for value in (term_scale, bound, expected, expected_volume, volume_bound)
    ):
        raise ValueError(f"{name} damage residual reconstruction bound is non-finite")

    actual = _f64(strong_actual, f"{name} strong residual", d.shape)
    actual_volume = _f64(volume_actual, f"{name} volume residual", d.shape)
    strong_defect = np.abs(actual - expected)
    volume_defect = np.abs(actual_volume - expected_volume)
    if not np.all(np.isfinite(strong_defect)) or not np.all(
        np.isfinite(volume_defect)
    ):
        raise ValueError(f"{name} residual reconstruction defect is non-finite")
    if np.any(strong_defect > bound) or np.any(volume_defect > volume_bound):
        raise ValueError(
            f"{name} residual reconstruction exceeds 512eps constituent bound: "
            f"strong_defect={_max_abs(strong_defect):.17g}, "
            f"strong_tolerance={_max_abs(bound):.17g}, "
            f"volume_defect={_max_abs(volume_defect):.17g}, "
            f"volume_tolerance={_max_abs(volume_bound):.17g}"
        )
def _replay_reduction_atol(constituent_scale: Any, name: str) -> float:
    """Return the two-evaluation float64 reduction bound for one scalar.

    ``constituent_scale`` is the sum (or norm, when the scalar is a vector
    norm) of the magnitudes that were reduced to form the compared quantity.
    Each side of the recorded/independent comparison may have its own
    reduction order, and the derived scalar may perform one additional
    reduction, hence the factor of three over the repository's 512-eps
    reconstruction bound.  This is deliberately an absolute tolerance: a
    cancellation-prone result may be much smaller than its constituents.
    """

    scale = _scalar(constituent_scale, f"{name} constituent scale")
    if scale < 0.0:
        raise ValueError(f"{name} constituent scale must be non-negative")
    tolerance = _REPLAY_COMPARISON_BOUND * scale
    if not np.isfinite(tolerance):
        raise ValueError(f"{name} reduction tolerance is non-finite")
    return float(tolerance)


def _compare_replay_vector(
    actual: Any,
    expected: Any,
    name: str,
    constituent_scale: Any,
) -> None:
    """Compare a reduction-produced vector componentwise by its constituents."""

    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"{name} shape/dtype does not match replay")
    if left.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {left.dtype}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"{name} must contain only finite values")
    scale = np.asarray(constituent_scale)
    if scale.dtype != _FLOAT64 or scale.shape != left.shape:
        raise ValueError(
            f"{name} constituent scale must have dtype/shape {_FLOAT64}/{left.shape}, "
            f"got {scale.dtype}/{scale.shape}"
        )
    if np.any(scale < 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError(f"{name} constituent scale is invalid")
    defect = np.abs(left - right)
    tolerance = _REPLAY_COMPARISON_BOUND * scale
    if not np.all(defect <= tolerance):
        raise ValueError(
            f"{name} does not match replay: defect={_max_abs(defect):.17g}, "
            f"tolerance={_max_abs(tolerance):.17g}"
        )


def _replay_ratio_atol(
    numerator: Any,
    denominator: Any,
    numerator_constituent_scale: Any,
    denominator_constituent_scale: Any,
    name: str,
) -> float:
    """Bound reduction-order differences in a quotient of two norms."""

    num = _scalar(numerator, f"{name} numerator")
    den = _scalar(denominator, f"{name} denominator")
    if num < 0.0 or den < 0.0:
        raise ValueError(f"{name} quotient inputs must be non-negative")
    num_atol = _replay_reduction_atol(numerator_constituent_scale, name)
    den_atol = _replay_reduction_atol(denominator_constituent_scale, name)
    den_lower = max(den - den_atol, _RELATIVE_FLOOR)
    den_upper = den + den_atol
    if not np.isfinite(den_lower) or not np.isfinite(den_upper):
        raise ValueError(f"{name} quotient denominator bounds are non-finite")
    expected = num / max(den, _RELATIVE_FLOOR)
    lower = max(0.0, num - num_atol) / den_upper
    upper = (num + num_atol) / den_lower
    tolerance = max(abs(expected - lower), abs(upper - expected))
    # Include the final quotient operation on both independently reduced
    # norms.  The term-based interval above covers the norm inputs.
    tolerance += _REPLAY_COMPARISON_BOUND * max(abs(expected), _RELATIVE_FLOOR)
    if not np.isfinite(tolerance):
        raise ValueError(f"{name} quotient tolerance is non-finite")
    return float(tolerance)


def _hash_named(values: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        if not isinstance(name, str) or not name:
            raise ValueError("identity field names must be nonempty strings")
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(b"|str|")
            digest.update(np.asarray([len(encoded)], dtype=_INT64).tobytes())
            digest.update(encoded)
            continue
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"identity field {name!r} has object dtype")
        digest.update(name.encode("utf-8"))
        digest.update(b"|array|")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=_INT64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _source_bundle_hash(entries: Iterable[tuple[str, Any]]) -> str:
    """Hash a deterministically framed bundle of relative filenames and bytes.

    ``entries`` accepts either ``(relative_name, bytes)`` pairs, which keeps
    the helper directly testable with temporary in-memory fixtures, or
    ``(relative_name, path)`` pairs for the live source bundle.  Length
    framing for both names and payloads prevents concatenation ambiguity; the
    sorted relative names make the digest independent of import traversal
    order.
    """

    normalized: list[tuple[str, bytes]] = []
    names: set[str] = set()
    for relative_name, source in entries:
        if type(relative_name) is not str or not relative_name:
            raise ValueError("source-bundle relative filenames must be nonempty strings")
        relative_path = PurePosixPath(relative_name)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative_name
            or ".." in relative_path.parts
        ):
            raise ValueError(
                f"source-bundle filename must be canonical relative POSIX text: {relative_name!r}"
            )
        if relative_name in names:
            raise ValueError(f"source-bundle filename is duplicated: {relative_name!r}")
        names.add(relative_name)
        if isinstance(source, (bytes, bytearray, memoryview)):
            payload = bytes(source)
        else:
            source_path = Path(source)
            try:
                payload = source_path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"source-bundle file cannot be read: {relative_name!r} ({source_path})"
                ) from error
        normalized.append((relative_name, payload))
    if not normalized:
        raise ValueError("source-bundle hash requires at least one file")

    digest = hashlib.sha256()
    digest.update(b"nonlocal-KKS-source-bundle-v1\0")
    for relative_name, payload in sorted(normalized):
        name_bytes = relative_name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, byteorder="big", signed=False))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _current_source_bundle_hash() -> str:
    """Return the hash of the exact local source bundle used by this path."""

    repository_root = Path(__file__).resolve().parents[2]
    entries = tuple(
        (relative_name, repository_root / relative_name)
        for relative_name in _SOURCE_BUNDLE_FILES
    )
    return _source_bundle_hash(entries)


def _state_digest(
    u_bar: float,
    union_displacement: np.ndarray,
    material_damage: np.ndarray,
    material_history: np.ndarray,
) -> str:
    return _hash_named(
        (
            ("u_bar", np.asarray(u_bar, dtype=np.float64)),
            ("union_displacement", union_displacement),
            ("material_damage", material_damage),
            ("material_history", material_history),
        )
    )


def _damage_digest(damage: np.ndarray) -> str:
    return _hash_named((("material_damage", damage),))


def _json_scalar(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "SU":
        raise ValueError(f"{name} must be a scalar string array")
    return str(array.item())


def _string_vector(value: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "SU" or array.shape != (length,):
        raise ValueError(f"{name} must be a string vector of length {length}")
    return array


def _sha256(value: Any, name: str) -> str:
    """Validate one canonical hexadecimal SHA256 identity string."""

    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must contain only hexadecimal characters") from error
    return value


def _optional_integer(value: Any | None, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _source_digest(model: "EngineeringGripModel") -> tuple[str, str]:
    geometry = model.geometry
    material_hash = _hash_named(
        (
            ("coordinates", geometry.material_reference_coordinates),
            ("volumes", geometry.material_reference_volumes),
            ("triangles", geometry.material_reference_triangles),
            ("edges", geometry.material_reference_edges),
            ("bottom", geometry.material_bottom_mask[: geometry.n_material]),
            ("top", geometry.material_top_mask[: geometry.n_material]),
            ("anchor", 0),
        )
    )
    union_hash = _hash_named(
        (
            ("coordinates", geometry.coordinates),
            ("material_phase_volumes", geometry.material_volumes),
            ("grip_phase_volumes", geometry.grip_volumes),
            ("union_volumes", geometry.union_volumes),
            ("union_triangles", geometry.union_triangles),
            ("candidate_edges", geometry.candidate_edges),
            ("visible_edges", geometry.directed_edges),
            ("removed_edges", geometry.removed_edges),
            ("volume_bottom_mask", model.constraints.outer_bottom_mask),
            ("volume_top_mask", model.constraints.outer_top_mask),
            ("visibility_tolerance", np.asarray(geometry.visibility_tolerance, dtype=np.float64)),
        )
    )
    return material_hash, union_hash


def _replay_static_digest(
    ba: Mapping[str, Any],
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    material_volumes: np.ndarray,
    constrained_dofs: np.ndarray,
    free_dofs: np.ndarray,
    volume_bottom: np.ndarray,
    volume_top: np.ndarray,
    anchor_particle: int,
    top_line_indices: np.ndarray,
    top_line_weights: np.ndarray,
    top_line_length: float,
    bottom_line_indices: np.ndarray,
    bottom_line_weights: np.ndarray,
    bottom_line_length: float,
    youngs_modulus: float,
    poisson_ratio: float,
    k_res: float,
    rank_rtol: float,
    delta: float,
    h_g: float,
    n_material: int,
) -> str:
    """Rebuild the A2 static identity from replayed canonical primitives.

    This mirrors ``A2StaticMechanics.from_geometry``'s digest fields without
    constructing a production mechanics object.  In particular, no archived
    static digest or archived derived static array is used as an input to the
    identity calculation.
    """

    coordinates = _f64(ba["coordinates"], "replay static coordinates")
    union_volumes = _f64(ba["volumes"], "replay static union volumes", (len(coordinates),))
    source = _i64(ba["source"], "replay static source", (len(ba["source"]),))
    target = _i64(ba["target"], "replay static target", source.shape)
    distance = _f64(ba["distance"], "replay static edge distance", source.shape)
    omega = _f64(ba["omega"], "replay static edge omega", source.shape)
    family = _f64(ba["family"], "replay static family mass", (len(coordinates),))
    eta = _f64(ba["eta"], "replay static eta", source.shape)
    support_indices, support_mask, beta, xi = _padded_replay_supports(ba)
    inverse = _f64(
        np.stack(ba["inverse"], axis=0),
        "replay static shape inverse",
        (len(source), 2, 2),
    )
    shape_rank = _i64(ba["shape_rank"], "replay static shape rank", source.shape)
    shape_condition = _f64(
        ba["shape_condition"], "replay static shape condition", source.shape
    )
    support_sizes = np.asarray(
        [len(support) for support in ba["supports"]], dtype=np.int64
    )
    gradient_coefficient = np.asarray(
        np.einsum("es,esb,ebc->esc", beta, xi, inverse, optimize=True),
        dtype=np.float64,
    )
    target_beta = np.asarray(beta[:, 0], dtype=np.float64)
    E = _scalar(youngs_modulus, "replay static youngs_modulus")
    nu = _scalar(poisson_ratio, "replay static poisson_ratio")
    kres = _scalar(k_res, "replay static k_res")
    rank = _scalar(rank_rtol, "replay static rank_rtol")
    horizon = _scalar(delta, "replay static delta")
    grip_spacing = _scalar(h_g, "replay static h_g")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    prefix_map = np.arange(n_material, dtype=np.int64)
    q_zero = np.zeros(2 * len(coordinates), dtype=np.float64)
    load_direction = np.zeros(2 * len(coordinates), dtype=np.float64)
    load_direction[2 * np.flatnonzero(volume_top) + 1] = 1.0
    outer_top_selector = load_direction.copy()
    outer_bottom_selector = np.zeros(2 * len(coordinates), dtype=np.float64)
    outer_bottom_selector[2 * np.flatnonzero(volume_bottom) + 1] = 1.0
    anchor_selector = np.zeros(2 * len(coordinates), dtype=np.float64)
    anchor_selector[2 * anchor_particle] = 1.0
    return _hash_named(
        (
            ("coordinates", coordinates),
            ("union_volumes", union_volumes),
            ("directed_edges", ba["edges"]),
            ("source", source),
            ("target", target),
            ("edge_distance", distance),
            ("edge_omega", omega),
            ("family_mass", family),
            ("eta", eta),
            ("support_indices", support_indices),
            ("support_mask", support_mask),
            ("support_sizes", support_sizes),
            ("beta", beta),
            ("xi", xi),
            ("shape_inverse", inverse),
            ("shape_rank", shape_rank),
            ("shape_condition", shape_condition),
            ("target_beta", target_beta),
            ("gradient_coefficient", gradient_coefficient),
            ("youngs_modulus", E),
            ("poisson_ratio", nu),
            ("lame_lambda", lam),
            ("shear_modulus", mu),
            ("k_res", kres),
            ("rank_rtol", rank),
            ("delta", horizon),
            ("h_g", grip_spacing),
            ("material_count", n_material),
            ("material_prefix_map", prefix_map),
            ("material_damage_volumes", material_volumes),
            ("union_material_phase_volumes", material_phase),
            ("grip_phase_volumes", grip_phase),
            ("constrained_dofs", constrained_dofs),
            ("free_dofs", free_dofs),
            ("q_zero", q_zero),
            ("load_direction", load_direction),
            ("outer_top_selector", outer_top_selector),
            ("outer_bottom_selector", outer_bottom_selector),
            ("anchor_selector", anchor_selector),
            ("outer_bottom_mask", volume_bottom),
            ("outer_top_mask", volume_top),
            ("anchor_particle", anchor_particle),
            ("top_line_indices", top_line_indices),
            ("top_line_weights", top_line_weights),
            ("top_line_length", top_line_length),
            ("bottom_line_indices", bottom_line_indices),
            ("bottom_line_weights", bottom_line_weights),
            ("bottom_line_length", bottom_line_length),
        )
    )


def build_crack_region_mask(coordinates: Any, delta: Any) -> np.ndarray:
    """Return material particles within one horizon of the unique notch tip."""

    values = _f64(coordinates, "crack-region coordinates")
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("crack-region coordinates must have shape (N, 2)")
    horizon = _scalar(delta, "crack-region delta")
    if horizon <= 0.0:
        raise ValueError("crack-region delta must be positive")
    tip_matches = np.flatnonzero(np.all(values == _CRACK_TIP[None, :], axis=1))
    if len(tip_matches) != 1:
        raise ValueError(
            "crack-region coordinates must contain exactly one tip particle (0.5, 0.5)"
        )
    mask = np.linalg.norm(values - _CRACK_TIP[None, :], axis=1) <= horizon
    if not bool(mask[int(tip_matches[0])]):
        raise ValueError("the exact crack tip is outside its own crack region")
    return _immutable(np.asarray(mask, dtype=bool))


@dataclass(frozen=True)
class CrackRegionDiagnostics:
    """Material crack-region extrema retained for localization diagnostics.

    The public diagnostic constructor keeps the strict non-negative damage
    contract.  The separately explicit warning-mode path may retain signed
    damage extrema so that its raw range excess is never clipped.
    """

    mask: np.ndarray
    tip_index: int
    damage_inside_max: float
    damage_outside_max: float
    history_inside_max: float
    history_outside_max: float
    strain_inside_max: float
    strain_outside_max: float
    amor_inside_max: float
    amor_outside_max: float
    localization_pass: bool

    def __post_init__(self) -> None:
        mask = _bool(self.mask, "crack-region mask")
        if mask.ndim != 1 or not np.any(mask):
            raise ValueError("crack-region mask must contain at least one particle")
        tip = _integer(self.tip_index, "crack-region tip_index")
        if tip >= len(mask) or not mask[tip]:
            raise ValueError("crack-region tip_index must belong to the mask")
        object.__setattr__(self, "mask", _immutable(mask))
        object.__setattr__(self, "tip_index", tip)
        for name in (
            "damage_inside_max",
            "damage_outside_max",
            "history_inside_max",
            "history_outside_max",
            "strain_inside_max",
            "strain_outside_max",
            "amor_inside_max",
            "amor_outside_max",
        ):
            value = _scalar(getattr(self, name), f"crack-region {name}")
            if not name.startswith("damage_") and value < 0.0:
                raise ValueError(f"crack-region {name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "localization_pass", bool(self.localization_pass))

    @property
    def damage_max_inside(self) -> float:
        return self.damage_inside_max

    @property
    def damage_max_outside(self) -> float:
        return self.damage_outside_max

    @property
    def history_max_inside(self) -> float:
        return self.history_inside_max

    @property
    def history_max_outside(self) -> float:
        return self.history_outside_max

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "damage_max_inside": "damage_inside_max",
            "damage_max_outside": "damage_outside_max",
            "history_max_inside": "history_inside_max",
            "history_max_outside": "history_outside_max",
        }
        return getattr(self, aliases.get(key, key))


def _field_extrema(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    inside = np.asarray(values)[mask]
    outside = np.asarray(values)[~mask]
    inside_max = float(np.max(inside)) if len(inside) else 0.0
    outside_max = float(np.max(outside)) if len(outside) else 0.0
    return inside_max, outside_max


def _material_interface_mask_from_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Infer the declared material interfaces from the unit-square cloud."""

    tolerance = 1.0e-12 * max(1.0, _max_abs(coordinates))
    return np.asarray(
        (np.abs(coordinates[:, 1]) <= tolerance)
        | (np.abs(coordinates[:, 1] - 1.0) <= tolerance),
        dtype=bool,
    )


def _localization_pass(
    damage: np.ndarray,
    history: np.ndarray,
    crack_mask: np.ndarray,
    interface_mask: np.ndarray,
) -> bool:
    """Apply the final-only notch localization and interface exclusion gate.

    An exactly zero field has no physically meaningful maximizing particle, so
    its interface check is intentionally vacuous.  This keeps clean/zero
    precrack fields out of the stagger acceptance path while requiring every
    positive final maximum to be away from a material interface.
    """

    for values in (damage, history):
        inside_max, outside_max = _field_extrema(values, crack_mask)
        if outside_max > inside_max + _CRACK_LOCALIZATION_ATOL:
            return False
        global_max = float(np.max(values))
        if global_max > 0.0 and np.any(interface_mask & (values == global_max)):
            return False
    return True


def compute_crack_region_diagnostics(
    coordinates: Any,
    delta: Any,
    material_damage: Any,
    material_history: Any,
    material_particle_strain: Any,
    material_amor_driver: Any,
    material_interface_mask: Any | None = None,
    *,
    allow_negative_damage: bool = False,
) -> CrackRegionDiagnostics:
    """Compute crack-region maxima without changing any solution field.

    ``material_interface_mask`` is optional for compatibility with standalone
    manufactured-cloud diagnostics.  When omitted, the declared unit-square
    ``y=0``/``y=1`` material interfaces are inferred from the coordinates.
    ``allow_negative_damage`` is an explicit warning-mode diagnostic choice;
    it preserves signed damage extrema while history, strain, and Amor fields
    remain non-negative.
    """

    if type(allow_negative_damage) is not bool:
        raise TypeError("allow_negative_damage must be a bool")
    coords = _f64(coordinates, "crack-region coordinates")
    mask = build_crack_region_mask(coords, delta)
    damage = _f64(material_damage, "crack-region damage", (len(coords),))
    history = _f64(material_history, "crack-region history", (len(coords),))
    strain = _f64(
        material_particle_strain,
        "crack-region particle strain",
        (len(coords), 2, 2),
    )
    amor = _f64(material_amor_driver, "crack-region Amor driver", (len(coords),))
    if (
        (not allow_negative_damage and np.any(damage < 0.0))
        or np.any(history < 0.0)
        or np.any(amor < 0.0)
    ):
        raise ValueError("crack-region fields must be non-negative")
    if material_interface_mask is None:
        interface = _material_interface_mask_from_coordinates(coords)
    else:
        interface = _bool(
            material_interface_mask,
            "crack-region material interface mask",
            (len(coords),),
        )
    strain_norm = np.linalg.norm(strain, axis=(1, 2))
    damage_inside, damage_outside = _field_extrema(damage, mask)
    history_inside, history_outside = _field_extrema(history, mask)
    strain_inside, strain_outside = _field_extrema(strain_norm, mask)
    amor_inside, amor_outside = _field_extrema(amor, mask)
    localization_pass = _localization_pass(damage, history, mask, interface)
    tip_index = int(np.flatnonzero(np.all(coords == _CRACK_TIP[None, :], axis=1))[0])
    return CrackRegionDiagnostics(
        mask=mask,
        tip_index=tip_index,
        damage_inside_max=damage_inside,
        damage_outside_max=damage_outside,
        history_inside_max=history_inside,
        history_outside_max=history_outside,
        strain_inside_max=strain_inside,
        strain_outside_max=strain_outside,
        amor_inside_max=amor_inside,
        amor_outside_max=amor_outside,
        localization_pass=localization_pass,
    )


def compute_interface_opening_diagnostic(
    displacement: Any,
    geometry: UnionGripGeometry,
    top_line: LineQuadrature,
    bottom_line: LineQuadrature,
) -> float:
    """Measure the material-interface opening without controlling the solve.

    This is an archived diagnostic only.  It is deliberately evaluated after
    a direct volume-grip solve and is never used as a continuation coordinate,
    load scale, reaction scale, or acceptance gate.
    """

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    if not isinstance(top_line, LineQuadrature) or not isinstance(
        bottom_line, LineQuadrature
    ):
        raise TypeError("interface diagnostic line data are invalid")
    if top_line.side != "top" or bottom_line.side != "bottom":
        raise ValueError("interface diagnostic lines must be top then bottom")
    values = _f64(
        displacement, "interface diagnostic displacement", (geometry.n_dofs,)
    ).reshape(-1, 2)
    top_mean = float(
        np.sum(top_line.weights * values[top_line.indices, 1]) / top_line.length
    )
    bottom_mean = float(
        np.sum(bottom_line.weights * values[bottom_line.indices, 1])
        / bottom_line.length
    )
    opening = top_mean - bottom_mean
    if not np.isfinite(opening):
        raise FloatingPointError("interface diagnostic opening is non-finite")
    return opening


def _interface_opening_from_arrays(
    displacement: np.ndarray,
    top_indices: np.ndarray,
    top_weights: np.ndarray,
    top_length: float,
    bottom_indices: np.ndarray,
    bottom_weights: np.ndarray,
    bottom_length: float,
) -> float:
    values = _f64(displacement, "interface diagnostic displacement").reshape(-1, 2)
    top_mean = float(np.sum(top_weights * values[top_indices, 1]) / top_length)
    bottom_mean = float(
        np.sum(bottom_weights * values[bottom_indices, 1]) / bottom_length
    )
    result = top_mean - bottom_mean
    if not np.isfinite(result):
        raise FloatingPointError("interface diagnostic opening is non-finite")
    return result


@dataclass(frozen=True)
class EngineeringGripModel:
    """Unequal union/material model used by the direct engineering controller."""

    geometry: UnionGripGeometry
    static: A2StaticMechanics
    constraints: AffineConstraints
    top_line: LineQuadrature
    bottom_line: LineQuadrature
    material_operator: ExplicitVolumeSparseD
    Gc: float
    l0: float
    crack_region_mask: np.ndarray
    material_hash: str
    union_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, UnionGripGeometry):
            raise TypeError("engineering-grip geometry must be UnionGripGeometry")
        if not isinstance(self.static, A2StaticMechanics):
            raise TypeError("engineering-grip static must be A2StaticMechanics")
        if not isinstance(self.constraints, AffineConstraints):
            raise TypeError("engineering-grip constraints must be AffineConstraints")
        if self.constraints.n_particles != self.geometry.n_particles:
            raise ValueError("engineering-grip constraint/model particle counts differ")
        expected = build_volume_grip_constraints(self.geometry, 0)
        for name in (
            "constrained_dofs",
            "free_dofs",
            "outer_bottom_mask",
            "outer_top_mask",
            "load_direction",
        ):
            if not np.array_equal(getattr(self.constraints, name), getattr(expected, name)):
                raise ValueError(f"engineering-grip constraint field {name} is not canonical")
        if not isinstance(self.top_line, LineQuadrature) or not isinstance(
            self.bottom_line, LineQuadrature
        ):
            raise TypeError("engineering-grip interface line data are invalid")
        if not isinstance(self.material_operator, ExplicitVolumeSparseD):
            raise TypeError("material_operator must be ExplicitVolumeSparseD")
        n_material = self.geometry.n_material
        if self.material_operator.shape != (n_material, n_material):
            raise ValueError("material operator must remain material-only")
        if self.material_operator.kernel != _KERNEL:
            raise ValueError("material operator must use the frozen cubic kernel")
        if not np.array_equal(
            np.asarray(self.material_operator.coordinates),
            self.geometry.coordinates[:n_material],
        ) or not np.array_equal(
            np.asarray(self.material_operator.volumes),
            self.geometry.material_reference_volumes,
        ) or not np.array_equal(
            np.asarray(self.material_operator.directed_edges, dtype=np.int64),
            self.geometry.material_reference_edges,
        ):
            raise ValueError("material operator is not the exact original material operator")
        Gc = _scalar(self.Gc, "engineering-grip Gc")
        l0 = _scalar(self.l0, "engineering-grip l0")
        if Gc <= 0.0 or l0 <= 0.0:
            raise ValueError("engineering-grip Gc/l0 must be positive")
        object.__setattr__(self, "Gc", Gc)
        object.__setattr__(self, "l0", l0)
        mask = _bool(self.crack_region_mask, "engineering-grip crack_region_mask", (n_material,))
        object.__setattr__(self, "crack_region_mask", _immutable(mask))
        if not isinstance(self.material_hash, str) or len(self.material_hash) != 64:
            raise ValueError("engineering-grip material_hash must be SHA256")
        if not isinstance(self.union_hash, str) or len(self.union_hash) != 64:
            raise ValueError("engineering-grip union_hash must be SHA256")
        expected_material, expected_union = _source_digest(self)
        if self.material_hash != expected_material or self.union_hash != expected_union:
            raise ValueError("engineering-grip cloud identity hash mismatch")

    @property
    def n_material(self) -> int:
        return int(self.geometry.n_material)

    @property
    def n_union(self) -> int:
        return int(self.geometry.n_particles)

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_union

    @property
    def static_digest(self) -> str:
        return self.static.deterministic_digest

    @property
    def material_phase_volumes(self) -> np.ndarray:
        return self.static.union_material_phase_volumes

    @property
    def grip_phase_volumes(self) -> np.ndarray:
        return self.static.grip_phase_volumes

    def fixed_damage_view(self, material_damage: Any) -> A2FixedDamageView:
        return A2FixedDamageView(self.static, material_damage)


def build_engineering_grip_model(
    geometry: UnionGripGeometry,
    youngs_modulus: Any,
    poisson_ratio: Any,
    k_res: Any,
    rank_rtol: Any,
    Gc: Any,
    l0: Any,
) -> EngineeringGripModel:
    """Build the fixed union mechanics/material-only damage model."""

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    if not np.array_equal(
        geometry.material_reference_coordinates[0], np.asarray([0.0, 0.0], dtype=np.float64)
    ):
        raise ValueError("the original material particle zero must be the anchor (0, 0)")
    constraints = build_volume_grip_constraints(geometry, 0)
    top_line = physical_interface_line_weights(geometry, "top")
    bottom_line = physical_interface_line_weights(geometry, "bottom")
    static = A2StaticMechanics.from_geometry(
        geometry,
        constraints,
        top_line,
        bottom_line,
        youngs_modulus,
        poisson_ratio,
        k_res,
        rank_rtol,
    )
    material_operator = ExplicitVolumeSparseD(
        geometry.coordinates[: geometry.n_material],
        geometry.material_reference_volumes,
        geometry.material_reference_edges,
        geometry.delta,
        _KERNEL,
    )
    mask = build_crack_region_mask(
        geometry.material_reference_coordinates, geometry.delta
    )
    # Compute the two hashes from the fully declared constructor inputs,
    # keeping the final dataclass validation authoritative.
    material_hash = _hash_named(
        (
            ("coordinates", geometry.material_reference_coordinates),
            ("volumes", geometry.material_reference_volumes),
            ("triangles", geometry.material_reference_triangles),
            ("edges", geometry.material_reference_edges),
            ("bottom", geometry.material_bottom_mask[: geometry.n_material]),
            ("top", geometry.material_top_mask[: geometry.n_material]),
            ("anchor", 0),
        )
    )
    union_hash = _hash_named(
        (
            ("coordinates", geometry.coordinates),
            ("material_phase_volumes", geometry.material_volumes),
            ("grip_phase_volumes", geometry.grip_volumes),
            ("union_volumes", geometry.union_volumes),
            ("union_triangles", geometry.union_triangles),
            ("candidate_edges", geometry.candidate_edges),
            ("visible_edges", geometry.directed_edges),
            ("removed_edges", geometry.removed_edges),
            ("volume_bottom_mask", constraints.outer_bottom_mask),
            ("volume_top_mask", constraints.outer_top_mask),
            (
                "visibility_tolerance",
                np.asarray(geometry.visibility_tolerance, dtype=np.float64),
            ),
        )
    )
    return EngineeringGripModel(
        geometry=geometry,
        static=static,
        constraints=constraints,
        top_line=top_line,
        bottom_line=bottom_line,
        material_operator=material_operator,
        Gc=Gc,
        l0=l0,
        crack_region_mask=mask,
        material_hash=material_hash,
        union_hash=union_hash,
    )


build_volume_grip_model = build_engineering_grip_model


@dataclass(frozen=True)
class EngineeringGripConfig:
    """Explicit strict continuation and one-time work-check settings."""

    load_start: float
    load_final: float
    du_initial: float
    du_min: float
    cutback_factor: float
    max_cutbacks: int
    max_stagger_iterations: int
    cg_rtol: float
    cg_atol: float
    cg_maxiter: int
    displacement: GateTolerance
    damage: GateTolerance
    history: GateTolerance
    reaction: GateTolerance
    damage_residual: GateTolerance
    force: GateTolerance
    moment: GateTolerance
    healing_atol: float
    damage_range_atol: float
    work_check_load: float
    work_epsilon: float
    work_relative_limit: float

    def __post_init__(self) -> None:
        start = _scalar(self.load_start, "load_start")
        final = _scalar(self.load_final, "load_final")
        du = _scalar(self.du_initial, "du_initial")
        du_min = _scalar(self.du_min, "du_min")
        factor = _scalar(self.cutback_factor, "cutback_factor")
        if start != 0.0 or final <= start:
            raise ValueError("engineering-grip load interval must start at exact zero and increase")
        if du <= 0.0 or du_min <= 0.0 or du_min > du or du > final:
            raise ValueError("engineering-grip increments are invalid")
        if not 0.0 < factor < 1.0:
            raise ValueError("cutback_factor must lie strictly between zero and one")
        max_cutbacks = _integer(self.max_cutbacks, "max_cutbacks", positive=True)
        max_stagger = _integer(
            self.max_stagger_iterations, "max_stagger_iterations", positive=True
        )
        rtol = _scalar(self.cg_rtol, "cg_rtol")
        atol = _scalar(self.cg_atol, "cg_atol")
        maxiter = _integer(self.cg_maxiter, "cg_maxiter", positive=True)
        if rtol < 0.0 or atol < 0.0 or (rtol == 0.0 and atol == 0.0):
            raise ValueError("CG tolerances must be non-negative and not both zero")
        for name in (
            "displacement",
            "damage",
            "history",
            "reaction",
            "damage_residual",
            "force",
            "moment",
        ):
            if not isinstance(getattr(self, name), GateTolerance):
                raise TypeError(f"{name} must be GateTolerance")
        healing = _scalar(self.healing_atol, "healing_atol")
        range_atol = _scalar(self.damage_range_atol, "damage_range_atol")
        check_load = _scalar(self.work_check_load, "work_check_load")
        epsilon = _scalar(self.work_epsilon, "work_epsilon")
        relative_limit = _scalar(self.work_relative_limit, "work_relative_limit")
        if healing < 0.0 or range_atol < 0.0:
            raise ValueError("healing/range tolerances must be non-negative")
        if check_load <= 0.0 or check_load > final:
            raise ValueError("work_check_load must lie in (0, load_final]")
        if epsilon <= 0.0 or epsilon >= check_load:
            raise ValueError("work_epsilon must lie strictly between zero and work_check_load")
        if relative_limit < 0.0:
            raise ValueError("work_relative_limit must be non-negative")
        for name, value in (
            ("load_start", start),
            ("load_final", final),
            ("du_initial", du),
            ("du_min", du_min),
            ("cutback_factor", factor),
            ("cg_rtol", rtol),
            ("cg_atol", atol),
            ("healing_atol", healing),
            ("damage_range_atol", range_atol),
            ("work_check_load", check_load),
            ("work_epsilon", epsilon),
            ("work_relative_limit", relative_limit),
        ):
            object.__setattr__(self, name, value)
        for name, value in (
            ("max_cutbacks", max_cutbacks),
            ("max_stagger_iterations", max_stagger),
            ("cg_maxiter", maxiter),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def smoke(cls) -> "EngineeringGripConfig":
        """Return the frozen short checkpoint configuration.

        This constructor is only a settings helper; it does not run the
        rounded cloud or select a production horizon.
        """

        gate = GateTolerance(1.0e-9, 1.0e-8, 1.0)
        residual = GateTolerance(1.0e-10, 1.0e-8, 1.0)
        return cls(
            load_start=0.0,
            load_final=1.0e-4,
            du_initial=1.0e-4,
            du_min=5.0e-5,
            cutback_factor=0.5,
            max_cutbacks=8,
            max_stagger_iterations=20,
            cg_rtol=1.0e-10,
            cg_atol=1.0e-13,
            cg_maxiter=3000,
            displacement=gate,
            damage=gate,
            history=gate,
            reaction=gate,
            damage_residual=residual,
            force=gate,
            moment=gate,
            healing_atol=1.0e-10,
            damage_range_atol=1.0e-10,
            work_check_load=1.0e-4,
            work_epsilon=1.0e-5,
            work_relative_limit=1.0e-7,
        )

    @property
    def u_final(self) -> float:
        return self.load_final

    @property
    def du(self) -> float:
        return self.du_initial

    @property
    def acceptance_mode(self) -> str:
        return _ACCEPTANCE_MODE


@dataclass(frozen=True)
class EngineeringGripState:
    """Immutable committed direct-load state."""

    u_bar: float
    union_displacement: np.ndarray
    material_damage: np.ndarray
    material_history: np.ndarray

    def __post_init__(self) -> None:
        load = _scalar(self.u_bar, "state.u_bar")
        if load < 0.0:
            raise ValueError("state.u_bar must be non-negative")
        q = _f64(self.union_displacement, "state.union_displacement")
        d = _f64(self.material_damage, "state.material_damage")
        history = _f64(self.material_history, "state.material_history")
        if q.ndim != 1 or q.size == 0 or q.size % 2:
            raise ValueError("state.union_displacement must be a nonempty flat vector")
        if d.ndim != 1 or d.size == 0 or history.shape != d.shape:
            raise ValueError("state damage/history shapes are incompatible")
        if np.any(history < 0.0):
            raise ValueError("state history must be non-negative")
        object.__setattr__(self, "u_bar", load)
        object.__setattr__(self, "union_displacement", _immutable(q))
        object.__setattr__(self, "material_damage", _immutable(d))
        object.__setattr__(self, "material_history", _immutable(history))

    @property
    def q(self) -> np.ndarray:
        return self.union_displacement

    @property
    def d(self) -> np.ndarray:
        return self.material_damage

    @property
    def H(self) -> np.ndarray:
        return self.material_history

    @property
    def digest(self) -> str:
        return _state_digest(
            self.u_bar,
            self.union_displacement,
            self.material_damage,
            self.material_history,
        )


def _check_exact_constraints(
    displacement: np.ndarray,
    constraints: AffineConstraints,
    u_bar: float,
    name: str,
) -> None:
    prescribed = constraints.prescribed_vector(u_bar)
    if not np.array_equal(
        displacement[constraints.constrained_dofs],
        prescribed[constraints.constrained_dofs],
    ):
        raise ValueError(f"{name} does not satisfy exact volume-grip constraints")


def solve_engineering_grip_fixed_outer_displacement(
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    u_bar: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: np.ndarray | None = None,
    seed_kind: str = "direct_load",
    source_state: str = "direct",
    source_step: int = 0,
    source_sweep: int = 0,
    source_damage_digest: str | None = None,
) -> A2MechanicsSolveResult:
    """Solve one fixed-damage direct top-volume-grip displacement problem."""

    if not isinstance(view, A2FixedDamageView):
        raise TypeError("view must be an A2FixedDamageView")
    if not isinstance(constraints, AffineConstraints):
        raise TypeError("constraints must be AffineConstraints")
    load = _scalar(u_bar, "u_bar")
    if source_damage_digest is None:
        source_damage_digest = view.damage_digest
    result = solve_a2_fixed_outer_displacement(
        view,
        constraints,
        load,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=initial_guess,
        seed_kind=seed_kind,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
    )
    _check_exact_constraints(result.displacement, constraints, load, "direct solve")
    if result.U_ext != load:
        raise ValueError("direct mechanics solve changed u_bar")
    return result


solve_fixed_outer_displacement = solve_engineering_grip_fixed_outer_displacement


def _validated_mechanics_adapter(
    model: EngineeringGripModel,
    mechanics_adapter: Any,
) -> JaxA2CPUReplayAdapter | None:
    """Validate the restricted mechanics adapter against the model identity.

    The default ``None`` keeps the CPU NumPy/SciPy reference path.  Only the
    exact ``JaxA2CPUReplayAdapter`` type is accepted; generic callables are
    rejected because an unverified solve cannot back the strict gates.
    """

    if mechanics_adapter is None:
        return None
    if not isinstance(mechanics_adapter, JaxA2CPUReplayAdapter):
        raise TypeError(
            "mechanics_adapter must be a JaxA2CPUReplayAdapter instance; "
            "generic callables are not accepted"
        )
    if model.static is not mechanics_adapter.cpu_static:
        raise ValueError(
            "mechanics_adapter static identity does not match the engineering-grip model"
        )
    if model.constraints is not mechanics_adapter.constraints:
        raise ValueError(
            "mechanics_adapter constraints are not the exact model constraint object"
        )
    return mechanics_adapter


def _solve_direct_mechanics(
    mechanics_adapter: JaxA2CPUReplayAdapter | None,
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    u_bar: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: np.ndarray | None,
    seed_kind: str,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str | None,
) -> A2MechanicsSolveResult:
    """Run one direct fixed-damage solve through the selected backend.

    The default path is the module-level CPU reference so the existing
    manufactured-test contract (including solver patching) is unchanged.  The
    restricted adapter path returns a CPU-authoritative result from the same
    device backend for every work-check and stagger solve.
    """

    if mechanics_adapter is None:
        return solve_fixed_outer_displacement(
            view,
            constraints,
            u_bar,
            cg_rtol,
            cg_atol,
            cg_maxiter,
            initial_guess=initial_guess,
            seed_kind=seed_kind,
            source_state=source_state,
            source_step=source_step,
            source_sweep=source_sweep,
            source_damage_digest=source_damage_digest,
        )
    return mechanics_adapter.solve_fixed_outer_displacement(
        view,
        constraints,
        u_bar,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=initial_guess,
        seed_kind=seed_kind,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
    )


def compute_material_source_amor(
    view: A2FixedDamageView,
    displacement: Any,
    previous_history: Any,
) -> Any:
    """Return the material-only Amor driver from all union targets.

    Every union edge with a material source is retained, including edges whose
    target is a collar particle.  Grip-source centers are not assigned a
    history field.
    """

    if not isinstance(view, A2FixedDamageView):
        raise TypeError("view must be an A2FixedDamageView")
    previous = _f64(
        previous_history, "material history", (view.static.material_count,)
    )
    if np.any(previous < 0.0):
        raise ValueError("material history must be non-negative")
    source = view.static.source
    retained = source < view.static.material_count
    if not np.any(retained):
        raise ValueError("union mechanics graph has no material source centers")
    eta_sum = np.bincount(
        source[retained],
        weights=view.static.eta[retained],
        minlength=view.static.material_count,
    )
    if np.max(np.abs(eta_sum - 1.0)) > _REPLAY_BOUND:
        raise ValueError("material-source eta does not sum to one")
    result = compute_a2_amor_history(view, displacement, previous)
    if result.particle_driver.shape != (view.static.material_count,):
        raise ValueError("material-source Amor driver has a nonmaterial shape")
    if result.trial_history.shape != result.particle_driver.shape:
        raise ValueError("material-source trial history has an incompatible shape")
    return result


compute_material_amor_history = compute_material_source_amor


@dataclass(frozen=True)
class EngineeringGripWorkCheck:
    """One exact-zero-damage centered mechanics energy/reaction check."""

    u_bar: float
    epsilon: float
    reaction: float
    reaction_fd: float
    relative_error: float
    relative_limit: float
    passed: bool
    center_result: A2MechanicsSolveResult
    plus_result: A2MechanicsSolveResult
    minus_result: A2MechanicsSolveResult

    def __post_init__(self) -> None:
        load = _scalar(self.u_bar, "work_check.u_bar")
        epsilon = _scalar(self.epsilon, "work_check.epsilon")
        reaction = _scalar(self.reaction, "work_check.reaction")
        fd = _scalar(self.reaction_fd, "work_check.reaction_fd")
        error = _scalar(self.relative_error, "work_check.relative_error")
        limit = _scalar(self.relative_limit, "work_check.relative_limit")
        if load <= 0.0 or epsilon <= 0.0 or epsilon >= load or limit < 0.0:
            raise ValueError("work-check scalar settings are invalid")
        if not isinstance(self.center_result, A2MechanicsSolveResult):
            raise TypeError("work_check.center_result has the wrong type")
        if not isinstance(self.plus_result, A2MechanicsSolveResult) or not isinstance(
            self.minus_result, A2MechanicsSolveResult
        ):
            raise TypeError("work_check side results have the wrong type")
        object.__setattr__(self, "u_bar", load)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "reaction", reaction)
        object.__setattr__(self, "reaction_fd", fd)
        object.__setattr__(self, "relative_error", error)
        object.__setattr__(self, "relative_limit", limit)
        object.__setattr__(self, "passed", bool(self.passed))

    @property
    def top_reaction(self) -> float:
        return self.reaction


class EngineeringGripWorkCheckFailure(RuntimeError):
    """Raised before continuation when the one-time work identity fails."""

    def __init__(self, check: EngineeringGripWorkCheck):
        super().__init__(
            "exact-zero-damage engineering-grip work check failed: "
            f"error={check.relative_error:.17g}, limit={check.relative_limit:.17g}"
        )
        self.check = check


def run_zero_damage_work_check(
    model: EngineeringGripModel,
    u_bar: Any,
    epsilon: Any,
    relative_limit: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    mechanics_adapter: JaxA2CPUReplayAdapter | None = None,
) -> EngineeringGripWorkCheck:
    """Run the single centered zero-damage reduced-energy/top-reaction check.

    ``mechanics_adapter`` is the explicit restricted JAX adapter.  When given,
    the center/plus/minus solves all run on that one device backend and the
    archived acceptance values remain CPU-authoritative.  The default ``None``
    keeps the CPU reference path.
    """

    if not isinstance(model, EngineeringGripModel):
        raise TypeError("model must be an EngineeringGripModel")
    adapter = _validated_mechanics_adapter(model, mechanics_adapter)
    load = _scalar(u_bar, "work_check.u_bar")
    perturbation = _scalar(epsilon, "work_check.epsilon")
    limit = _scalar(relative_limit, "work_check.relative_limit")
    if load <= 0.0 or perturbation <= 0.0 or perturbation >= load:
        raise ValueError("work-check load/epsilon must satisfy 0 < epsilon < u_bar")
    if limit < 0.0:
        raise ValueError("work-check relative limit must be non-negative")
    damage = np.zeros(model.n_material, dtype=np.float64)
    view = model.fixed_damage_view(damage)
    digest = view.damage_digest
    center = _solve_direct_mechanics(
        adapter,
        view,
        model.constraints,
        load,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="zero_work_check",
        source_state="zero_work_check",
        source_step=0,
        source_sweep=0,
        source_damage_digest=digest,
    )
    plus = _solve_direct_mechanics(
        adapter,
        view,
        model.constraints,
        load + perturbation,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="zero_work_check",
        source_state="zero_work_check",
        source_step=0,
        source_sweep=0,
        source_damage_digest=digest,
    )
    minus = _solve_direct_mechanics(
        adapter,
        view,
        model.constraints,
        load - perturbation,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="zero_work_check",
        source_state="zero_work_check",
        source_step=0,
        source_sweep=0,
        source_damage_digest=digest,
    )
    reaction_fd = (plus.energy - minus.energy) / (2.0 * perturbation)
    relative_error = _relative(reaction_fd, center.outer_reaction)
    check = EngineeringGripWorkCheck(
        u_bar=load,
        epsilon=perturbation,
        reaction=center.outer_reaction,
        reaction_fd=reaction_fd,
        relative_error=relative_error,
        relative_limit=limit,
        passed=bool(relative_error <= limit),
        center_result=center,
        plus_result=plus,
        minus_result=minus,
    )
    if not check.passed:
        raise EngineeringGripWorkCheckFailure(check)
    return check


zero_damage_work_check = run_zero_damage_work_check


def _optional_array(
    value: Any | None,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    array = _f64(value, name, shape)
    return _immutable(array)


def _optional_scalar(value: Any | None, name: str) -> float | None:
    if value is None:
        return None
    return _scalar(value, name)


def _required_diagnostic_array(
    diagnostic: "EngineeringGripDiagnostics", name: str
) -> np.ndarray:
    value = getattr(diagnostic, name)
    if value is None:
        raise ValueError(f"complete diagnostic is missing required field {name}")
    return np.asarray(value)


@dataclass(frozen=True)
class EngineeringGripDiagnostics:
    """Primitive strict-sweep diagnostics, complete or explicitly incomplete."""

    sweep_index: int
    target_u_bar: float
    complete_sweep: bool
    passed: bool
    failure_stage: str
    failure_reason: str
    damage_before: np.ndarray
    d_iter: np.ndarray
    d_next: np.ndarray | None
    q_before: np.ndarray | None
    q_after: np.ndarray | None
    psi_before: np.ndarray | None
    psi_after: np.ndarray | None
    H_used: np.ndarray | None
    H_next: np.ndarray | None
    rho_used: np.ndarray | None
    rho_next: np.ndarray | None
    rho_used_volume: np.ndarray | None
    rho_next_volume: np.ndarray | None
    energy_before: float | None
    energy_after: float | None
    top_reaction_before: float | None
    top_reaction_after: float | None
    bottom_reaction_before: float | None
    bottom_reaction_after: float | None
    anchor_reaction_before: float | None
    anchor_reaction_after: float | None
    force_before: np.ndarray | None
    force_after: np.ndarray | None
    moment_before: float | None
    moment_after: float | None
    free_residual_norm_before: float | None
    free_residual_norm_after: float | None
    free_residual_relative_before: float | None
    free_residual_relative_after: float | None
    cg_info_before: int | None
    cg_info_after: int | None
    e_u_material: float | None
    e_u_union: float | None
    e_damage: float | None
    e_history: float | None
    e_reaction: float | None
    damage_min: float | None
    damage_max: float | None
    healing_min: float | None
    history_min_gap: float | None
    rho_used_inf: float | None
    rho_next_inf: float | None
    displacement_scale: float | None
    damage_scale: float | None
    history_scale: float | None
    reaction_scale: float | None
    damage_residual_scale: float | None
    force_error: float | None
    force_scale: float | None
    moment_error: float | None
    moment_scale: float | None
    finite_pass: bool
    mechanics_before_pass: bool
    mechanics_after_pass: bool
    displacement_pass: bool
    damage_pass: bool
    history_pass: bool
    reaction_pass: bool
    damage_residual_pass: bool
    damage_range_pass: bool
    healing_pass: bool
    history_monotonic_pass: bool
    force_pass: bool
    moment_pass: bool
    available_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sweep_index", _integer(self.sweep_index, "sweep_index"))
        object.__setattr__(self, "target_u_bar", _scalar(self.target_u_bar, "target_u_bar"))
        if self.target_u_bar <= 0.0:
            raise ValueError("diagnostic target_u_bar must be positive")
        object.__setattr__(self, "complete_sweep", bool(self.complete_sweep))
        object.__setattr__(self, "passed", bool(self.passed))
        if not isinstance(self.failure_stage, str) or not isinstance(
            self.failure_reason, str
        ):
            raise TypeError("diagnostic failure fields must be strings")
        damage_before = _f64(self.damage_before, "diagnostic.damage_before")
        d_iter = _f64(self.d_iter, "diagnostic.d_iter", damage_before.shape)
        object.__setattr__(self, "damage_before", _immutable(damage_before))
        object.__setattr__(self, "d_iter", _immutable(d_iter))
        array_shapes = {
            "d_next": damage_before.shape,
            "psi_before": damage_before.shape,
            "psi_after": damage_before.shape,
            "H_used": damage_before.shape,
            "H_next": damage_before.shape,
            "rho_used": damage_before.shape,
            "rho_next": damage_before.shape,
            "rho_used_volume": damage_before.shape,
            "rho_next_volume": damage_before.shape,
            "q_before": (2 * len(damage_before),),
            "q_after": (2 * len(damage_before),),
            "force_before": (2,),
            "force_after": (2,),
        }
        # The displacement arrays contain the union, not the material prefix;
        # their exact length is checked by the controller/result pairing.
        for name, shape in array_shapes.items():
            value = getattr(self, name)
            if value is None:
                continue
            if name.startswith("q_"):
                array = _f64(value, f"diagnostic.{name}")
                if array.ndim != 1 or array.size == 0 or array.size % 2:
                    raise ValueError(f"diagnostic.{name} must be flat union displacement")
                object.__setattr__(self, name, _immutable(array))
            else:
                object.__setattr__(self, name, _immutable(_f64(value, f"diagnostic.{name}", shape)))
        for name in (
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
        ):
            object.__setattr__(
                self,
                name,
                _optional_scalar(getattr(self, name), f"diagnostic.{name}"),
            )
        for name in ("cg_info_before", "cg_info_after"):
            object.__setattr__(
                self,
                name,
                _optional_integer(getattr(self, name), f"diagnostic.{name}"),
            )
        for name in (
            "finite_pass",
            "mechanics_before_pass",
            "mechanics_after_pass",
            "displacement_pass",
            "damage_pass",
            "history_pass",
            "reaction_pass",
            "damage_residual_pass",
            "damage_range_pass",
            "healing_pass",
            "history_monotonic_pass",
            "force_pass",
            "moment_pass",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.complete_sweep and not self.failure_reason and not self.passed:
            raise ValueError("complete failed sweep must retain a failure reason")
        if not self.complete_sweep and (
            not self.failure_stage or not self.failure_reason
        ):
            raise ValueError("incomplete sweep must retain failure stage and reason")
        if self.passed and not self.complete_sweep:
            raise ValueError("incomplete sweep cannot pass")
        object.__setattr__(self, "available_fields", tuple(sorted(set(self.available_fields))))

    @property
    def gate_passes(self) -> dict[str, bool]:
        return {
            "finite": self.finite_pass,
            "mechanics_before": self.mechanics_before_pass,
            "mechanics_after": self.mechanics_after_pass,
            "displacement": self.displacement_pass,
            "damage": self.damage_pass,
            "history": self.history_pass,
            "reaction": self.reaction_pass,
            "damage_residual": self.damage_residual_pass,
            "damage_range": self.damage_range_pass,
            "healing": self.healing_pass,
            "history_monotonic": self.history_monotonic_pass,
            "force": self.force_pass,
            "moment": self.moment_pass,
        }

    @property
    def u_bar(self) -> float:
        return self.target_u_bar

    @property
    def damage_residual_inf(self) -> float | None:
        return self.rho_next_inf


def _mechanics_fields(
    result: A2MechanicsSolveResult | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "energy": None,
            "top": None,
            "bottom": None,
            "anchor": None,
            "force": None,
            "moment": None,
            "free_norm": None,
            "free_relative": None,
        }
    return {
        "energy": result.energy,
        "top": result.outer_reaction,
        "bottom": result.bottom_reaction,
        "anchor": result.anchor_reaction,
        "force": result.total_force,
        "moment": result.total_moment,
        "free_norm": result.free_residual_norm,
        "free_relative": result.free_residual_relative_norm,
    }


def _geometry_length(model: EngineeringGripModel) -> float:
    return max(_max_abs(model.geometry.coordinates), _max_abs(np.ptp(model.geometry.coordinates, axis=0)), 1.0)


def _mechanics_gate(
    result: A2MechanicsSolveResult | None,
    config: EngineeringGripConfig,
    geometry_length: float,
) -> tuple[bool, bool, float, float, bool, bool]:
    if result is None:
        return False, False, math.inf, math.inf, False, False
    finite = bool(
        np.all(np.isfinite(result.displacement))
        and np.all(np.isfinite(result.residual))
        and np.all(np.isfinite(result.total_force))
        and np.isfinite(result.total_moment)
    )
    mechanics = bool(
        finite
        and result.cg_info == 0
        and result.free_residual_norm < result.cg_convergence_limit
    )
    force_error = _max_abs(result.total_force)
    force_scale = max(
        abs(result.outer_reaction),
        abs(result.bottom_reaction),
        abs(result.anchor_reaction),
        1.0,
    )
    force_pass = bool(force_error <= config.force.limit(force_scale))
    moment_error = abs(result.total_moment)
    moment_scale = force_scale * geometry_length
    moment_pass = bool(moment_error <= config.moment.limit(moment_scale))
    return (
        mechanics,
        finite,
        force_error,
        moment_error,
        force_pass,
        moment_pass,
    )


def _damage_residuals(
    model: EngineeringGripModel,
    damage: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    d = _f64(damage, "damage residual damage", (model.n_material,))
    used = _f64(H_used, "damage residual H_used", d.shape)
    next_history = _f64(H_next, "damage residual H_next", d.shape)
    D = model.material_operator.to_scipy().tocsr()
    graph = np.asarray(D @ d, dtype=np.float64)
    local = model.Gc / model.l0 * d
    graph_term = model.Gc * model.l0 * graph
    rho_used = local + graph_term + 2.0 * used * d - 2.0 * used
    rho_next = local + graph_term + 2.0 * next_history * d - 2.0 * next_history
    volume = np.asarray(model.material_operator.volumes, dtype=np.float64)
    terms = (
        local,
        graph_term,
        2.0 * used * d,
        2.0 * used,
        2.0 * next_history * d,
        2.0 * next_history,
    )
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise FloatingPointError("engineering-grip damage residual terms are non-finite")
    scale = max((_max_abs(term) for term in terms), default=0.0)
    return (
        np.asarray(rho_used, dtype=np.float64),
        np.asarray(rho_next, dtype=np.float64),
        np.asarray(volume * rho_used, dtype=np.float64),
        np.asarray(volume * rho_next, dtype=np.float64),
        scale,
    )


def _incomplete_diagnostics(
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    d_iter: np.ndarray,
    target: float,
    stage: str,
    reason: str,
    *,
    before: A2MechanicsSolveResult | None = None,
    after: A2MechanicsSolveResult | None = None,
    psi_before: np.ndarray | None = None,
    psi_after: np.ndarray | None = None,
    H_used: np.ndarray | None = None,
    d_next: np.ndarray | None = None,
    rho_used: np.ndarray | None = None,
    rho_used_volume: np.ndarray | None = None,
    model: EngineeringGripModel,
    sweep_index: int,
) -> EngineeringGripDiagnostics:
    before_fields = _mechanics_fields(before)
    after_fields = _mechanics_fields(after)
    available: list[str] = ["damage_before", "d_iter"]
    for name, value in (
        ("q_before", None if before is None else before.displacement),
        ("q_after", None if after is None else after.displacement),
        ("psi_before", psi_before),
        ("psi_after", psi_after),
        ("H_used", H_used),
        ("d_next", d_next),
        ("rho_used", rho_used),
        ("rho_used_volume", rho_used_volume),
    ):
        if value is not None:
            available.append(name)
    (
        before_mechanics,
        before_finite,
        before_force,
        before_moment,
        before_force_pass,
        before_moment_pass,
    ) = _mechanics_gate(before, config, _geometry_length(model))
    (
        after_mechanics,
        after_finite,
        after_force,
        after_moment,
        after_force_pass,
        after_moment_pass,
    ) = _mechanics_gate(after, config, _geometry_length(model))
    force_error = max(
        before_force if before is not None else 0.0,
        after_force if after is not None else 0.0,
    )
    moment_error = max(
        before_moment if before is not None else 0.0,
        after_moment if after is not None else 0.0,
    )
    force_scale = max(
        value
        for value in (
            1.0,
            abs(before_fields["top"] or 0.0),
            abs(after_fields["top"] or 0.0),
            abs(before_fields["bottom"] or 0.0),
            abs(after_fields["bottom"] or 0.0),
            abs(before_fields["anchor"] or 0.0),
            abs(after_fields["anchor"] or 0.0),
        )
    )
    moment_scale = force_scale * _geometry_length(model)
    current = d_next if d_next is not None else d_iter
    return EngineeringGripDiagnostics(
        sweep_index=sweep_index,
        target_u_bar=target,
        complete_sweep=False,
        passed=False,
        failure_stage=stage,
        failure_reason=reason,
        damage_before=committed.material_damage,
        d_iter=d_iter,
        d_next=d_next,
        q_before=None if before is None else before.displacement,
        q_after=None if after is None else after.displacement,
        psi_before=psi_before,
        psi_after=psi_after,
        H_used=H_used,
        H_next=None,
        rho_used=rho_used,
        rho_next=None,
        rho_used_volume=rho_used_volume,
        rho_next_volume=None,
        energy_before=before_fields["energy"],
        energy_after=after_fields["energy"],
        top_reaction_before=before_fields["top"],
        top_reaction_after=after_fields["top"],
        bottom_reaction_before=before_fields["bottom"],
        bottom_reaction_after=after_fields["bottom"],
        anchor_reaction_before=before_fields["anchor"],
        anchor_reaction_after=after_fields["anchor"],
        force_before=before_fields["force"],
        force_after=after_fields["force"],
        moment_before=before_fields["moment"],
        moment_after=after_fields["moment"],
        free_residual_norm_before=before_fields["free_norm"],
        free_residual_norm_after=after_fields["free_norm"],
        free_residual_relative_before=before_fields["free_relative"],
        free_residual_relative_after=after_fields["free_relative"],
        cg_info_before=None if before is None else before.cg_info,
        cg_info_after=None if after is None else after.cg_info,
        e_u_material=None,
        e_u_union=None,
        e_damage=None if d_next is None else _max_abs(d_next - d_iter),
        e_history=None,
        e_reaction=None,
        damage_min=float(np.min(current)),
        damage_max=float(np.max(current)),
        healing_min=None if d_next is None else float(np.min(d_next - committed.material_damage)),
        history_min_gap=None,
        rho_used_inf=None if rho_used is None else _max_abs(rho_used),
        rho_next_inf=None,
        displacement_scale=None,
        damage_scale=None if d_next is None else max(_max_abs(d_iter), _max_abs(d_next)),
        history_scale=None,
        reaction_scale=None,
        damage_residual_scale=None,
        force_error=force_error if before is not None or after is not None else None,
        force_scale=force_scale if before is not None or after is not None else None,
        moment_error=moment_error if before is not None or after is not None else None,
        moment_scale=moment_scale if before is not None or after is not None else None,
        finite_pass=bool(before_finite and after_finite),
        mechanics_before_pass=before_mechanics,
        mechanics_after_pass=after_mechanics,
        displacement_pass=False,
        damage_pass=False,
        history_pass=False,
        reaction_pass=False,
        damage_residual_pass=False,
        damage_range_pass=(
            d_next is not None
            and float(np.min(d_next)) >= -config.damage_range_atol
            and float(np.max(d_next)) <= 1.0 + config.damage_range_atol
        ),
        healing_pass=(
            d_next is not None
            and float(np.min(d_next - committed.material_damage)) >= -config.healing_atol
        ),
        history_monotonic_pass=False,
        # Preserve the independent status of every mechanics conservation
        # gate that was actually evaluated.  An incomplete before-stage
        # failure has no ``after`` solve; treating that absent result as a
        # failed force *and* moment gate forges the unavailable moment
        # provenance and changes the warning bypass audit.  With neither
        # solve available, both aggregate gates remain false.
        force_pass=bool(
            (before is not None or after is not None)
            and (before is None or before_force_pass)
            and (after is None or after_force_pass)
        ),
        moment_pass=bool(
            (before is not None or after is not None)
            and (before is None or before_moment_pass)
            and (after is None or after_moment_pass)
        ),
        available_fields=tuple(available),
    )


def _full_diagnostics(
    config: EngineeringGripConfig,
    model: EngineeringGripModel,
    committed: EngineeringGripState,
    d_iter: np.ndarray,
    d_next: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
    before: A2MechanicsSolveResult,
    after: A2MechanicsSolveResult,
    psi_before: np.ndarray,
    psi_after: np.ndarray,
    rho_used: np.ndarray,
    rho_next: np.ndarray,
    rho_used_volume: np.ndarray,
    rho_next_volume: np.ndarray,
    residual_scale: float,
    sweep_index: int,
    target: float,
    warning_safety_limit: float | None = None,
) -> EngineeringGripDiagnostics:
    geometry_length = _geometry_length(model)
    before_fields = _mechanics_fields(before)
    after_fields = _mechanics_fields(after)
    (
        mechanics_before,
        finite_before,
        force_before,
        moment_before,
        force_before_pass,
        moment_before_pass,
    ) = _mechanics_gate(before, config, geometry_length)
    (
        mechanics_after,
        finite_after,
        force_after,
        moment_after,
        force_after_pass,
        moment_after_pass,
    ) = _mechanics_gate(after, config, geometry_length)
    q_before = before.displacement
    q_after = after.displacement
    e_u_material = _max_abs(
        q_after[: 2 * model.n_material] - q_before[: 2 * model.n_material]
    )
    displacement_scale = max(
        _max_abs(q_before[: 2 * model.n_material]),
        _max_abs(q_after[: 2 * model.n_material]),
        abs(target),
    )
    e_u_union = _max_abs(q_after - q_before)
    union_scale = max(
        _max_abs(q_before),
        _max_abs(q_after),
        abs(target),
    )
    e_damage = _max_abs(d_next - d_iter)
    damage_scale = max(_max_abs(d_iter), _max_abs(d_next))
    e_history = _max_abs(H_next - H_used)
    history_scale = max(
        _max_abs(committed.material_history), _max_abs(H_used), _max_abs(H_next)
    )
    e_reaction = abs(after.outer_reaction - before.outer_reaction)
    reaction_scale = max(abs(before.outer_reaction), abs(after.outer_reaction))
    damage_min = float(np.min(d_next))
    damage_max = float(np.max(d_next))
    healing_min = float(np.min(d_next - committed.material_damage))
    # The warning limit is a final acceptance-safety value, not a gate
    # tolerance.  Keep validating it at this boundary for callers that pass
    # the warning-mode setting, but never use it to change the truthful strict
    # range/healing flags below.
    if warning_safety_limit is not None:
        _validate_warning_safety_limit(warning_safety_limit)
    history_min_gap = float(np.min(H_next - committed.material_history))
    force_error = max(force_before, force_after)
    force_scale = max(
        1.0,
        abs(before.outer_reaction),
        abs(after.outer_reaction),
        abs(before.bottom_reaction),
        abs(after.bottom_reaction),
        abs(before.anchor_reaction),
        abs(after.anchor_reaction),
    )
    moment_error = max(moment_before, moment_after)
    moment_scale = force_scale * geometry_length
    finite_pass = bool(
        finite_before
        and finite_after
        and all(
            np.all(np.isfinite(array))
            for array in (
                d_iter,
                d_next,
                psi_before,
                psi_after,
                H_used,
                H_next,
                rho_used,
                rho_next,
                rho_used_volume,
                rho_next_volume,
            )
        )
    )
    passes = {
        "finite": finite_pass,
        "mechanics_before": mechanics_before,
        "mechanics_after": mechanics_after,
        "displacement": bool(
            e_u_material <= config.displacement.limit(displacement_scale)
            and e_u_union <= config.displacement.limit(union_scale)
        ),
        "damage": e_damage <= config.damage.limit(damage_scale),
        "history": e_history <= config.history.limit(history_scale),
        "reaction": e_reaction <= config.reaction.limit(reaction_scale),
        "damage_residual": _max_abs(rho_next)
        <= config.damage_residual.limit(residual_scale),
        "damage_range": bool(
            damage_min >= -config.damage_range_atol
            and damage_max <= 1.0 + config.damage_range_atol
        ),
        "healing": bool(healing_min >= -config.healing_atol),
        "history_monotonic": history_min_gap >= 0.0,
        "force": bool(force_before_pass and force_after_pass),
        "moment": bool(moment_before_pass and moment_after_pass),
    }
    passed = bool(all(passes.values()))
    reason = "" if passed else ", ".join(name for name, value in passes.items() if not value)
    return EngineeringGripDiagnostics(
        sweep_index=sweep_index,
        target_u_bar=target,
        complete_sweep=True,
        passed=passed,
        failure_stage="" if passed else "gates",
        failure_reason=reason,
        damage_before=committed.material_damage,
        d_iter=d_iter,
        d_next=d_next,
        q_before=q_before,
        q_after=q_after,
        psi_before=psi_before,
        psi_after=psi_after,
        H_used=H_used,
        H_next=H_next,
        rho_used=rho_used,
        rho_next=rho_next,
        rho_used_volume=rho_used_volume,
        rho_next_volume=rho_next_volume,
        energy_before=before.energy,
        energy_after=after.energy,
        top_reaction_before=before.outer_reaction,
        top_reaction_after=after.outer_reaction,
        bottom_reaction_before=before.bottom_reaction,
        bottom_reaction_after=after.bottom_reaction,
        anchor_reaction_before=before.anchor_reaction,
        anchor_reaction_after=after.anchor_reaction,
        force_before=before.total_force,
        force_after=after.total_force,
        moment_before=before.total_moment,
        moment_after=after.total_moment,
        free_residual_norm_before=before.free_residual_norm,
        free_residual_norm_after=after.free_residual_norm,
        free_residual_relative_before=before.free_residual_relative_norm,
        free_residual_relative_after=after.free_residual_relative_norm,
        cg_info_before=before.cg_info,
        cg_info_after=after.cg_info,
        e_u_material=e_u_material,
        e_u_union=e_u_union,
        e_damage=e_damage,
        e_history=e_history,
        e_reaction=e_reaction,
        damage_min=damage_min,
        damage_max=damage_max,
        healing_min=healing_min,
        history_min_gap=history_min_gap,
        rho_used_inf=_max_abs(rho_used),
        rho_next_inf=_max_abs(rho_next),
        displacement_scale=union_scale,
        damage_scale=damage_scale,
        history_scale=history_scale,
        reaction_scale=reaction_scale,
        damage_residual_scale=residual_scale,
        force_error=force_error,
        force_scale=force_scale,
        moment_error=moment_error,
        moment_scale=moment_scale,
        finite_pass=finite_pass,
        mechanics_before_pass=mechanics_before,
        mechanics_after_pass=mechanics_after,
        displacement_pass=bool(passes["displacement"]),
        damage_pass=bool(passes["damage"]),
        history_pass=bool(passes["history"]),
        reaction_pass=bool(passes["reaction"]),
        damage_residual_pass=bool(passes["damage_residual"]),
        damage_range_pass=bool(passes["damage_range"]),
        healing_pass=bool(passes["healing"]),
        history_monotonic_pass=bool(passes["history_monotonic"]),
        force_pass=bool(passes["force"]),
        moment_pass=bool(passes["moment"]),
        available_fields=(
            "damage_before",
            "d_iter",
            "d_next",
            "q_before",
            "q_after",
            "psi_before",
            "psi_after",
            "H_used",
            "H_next",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        ),
    )


@dataclass(frozen=True)
class EngineeringGripAcceptedStep:
    """One committed complete strict sweep."""

    load_start: float
    load_target: float
    increment: float
    state: EngineeringGripState
    damage_before: np.ndarray
    d_iter: np.ndarray
    d_next: np.ndarray
    q_before: np.ndarray
    q_after: np.ndarray
    psi_before: np.ndarray
    psi_after: np.ndarray
    H_used: np.ndarray
    H_next: np.ndarray
    rho_used: np.ndarray
    rho_next: np.ndarray
    rho_used_volume: np.ndarray
    rho_next_volume: np.ndarray
    mechanics_before: A2MechanicsSolveResult
    mechanics_after: A2MechanicsSolveResult
    diagnostics: EngineeringGripDiagnostics
    stagger_iterations: int
    cutback_count: int
    acceptance_mode: str
    acceptance_status: str
    bypassed_gates: tuple[str, ...]
    warning_range_excess: float | None
    warning_healing_magnitude: float | None

    def __post_init__(self) -> None:
        start = _scalar(self.load_start, "accepted.load_start")
        target = _scalar(self.load_target, "accepted.load_target")
        increment = _scalar(self.increment, "accepted.increment")
        if target <= start or increment <= 0.0 or target - start != increment:
            raise ValueError("accepted engineering-grip load interval is invalid")
        if not isinstance(self.state, EngineeringGripState):
            raise TypeError("accepted state has the wrong type")
        if not isinstance(self.diagnostics, EngineeringGripDiagnostics):
            raise TypeError("accepted diagnostics have the wrong type")
        if not isinstance(self.mechanics_before, A2MechanicsSolveResult) or not isinstance(
            self.mechanics_after, A2MechanicsSolveResult
        ):
            raise TypeError("accepted mechanics results have the wrong type")
        mode = _validate_acceptance_mode(self.acceptance_mode)
        bypassed_gates = _validate_bypassed_gates(self.bypassed_gates)
        expected_bypassed = _failed_warning_gates(self.diagnostics)
        if mode == _ACCEPTANCE_MODE:
            if self.acceptance_status != _ACCEPTANCE_STATUS:
                raise ValueError("accepted engineering-grip status must be strict_passed")
            if bypassed_gates or self.warning_range_excess is not None or self.warning_healing_magnitude is not None:
                raise ValueError(
                    "strict_passed engineering-grip steps cannot record warning fields"
                )
        elif self.acceptance_status == _ACCEPTANCE_STATUS:
            if self.warning_range_excess is None or self.warning_healing_magnitude is None:
                raise ValueError(
                    "warning-mode strict_passed steps must record raw excesses"
                )
            _require_warning_excess_records(
                self.warning_range_excess, self.warning_healing_magnitude
            )
        elif self.acceptance_status == _WARNING_ACCEPTANCE_STATUS:
            if self.warning_range_excess is None or self.warning_healing_magnitude is None:
                raise ValueError(
                    "warning_accepted steps must record raw excesses"
                )
            _require_warning_excess_records(
                self.warning_range_excess, self.warning_healing_magnitude
            )
        else:
            raise ValueError(
                "accepted engineering-grip status must be strict_passed or "
                "warning_accepted"
            )
        if not self.diagnostics.complete_sweep:
            raise ValueError("accepted step requires a complete sweep diagnostic")
        if self.acceptance_status != _WARNING_ACCEPTANCE_STATUS:
            if not self.diagnostics.passed:
                raise ValueError("accepted step requires a complete passing diagnostic")
            if bypassed_gates:
                raise ValueError("strict_passed steps cannot bypass acceptance gates")
        else:
            if self.diagnostics.passed:
                raise ValueError(
                    "warning_accepted steps must retain passed=False diagnostics"
                )
            if not expected_bypassed:
                raise ValueError(
                    "warning_accepted steps require at least one failed bypassable gate"
                )
            if bypassed_gates != expected_bypassed:
                raise ValueError(
                    "accepted bypassed_gates must exactly match the canonical "
                    "failed-gate list"
                )
        n = self.state.material_damage.size
        for name in (
            "damage_before",
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
        ):
            object.__setattr__(
                self,
                name,
                _immutable(_f64(getattr(self, name), f"accepted.{name}", (n,))),
            )
        for name in ("q_before", "q_after"):
            q = _f64(getattr(self, name), f"accepted.{name}")
            if q.ndim != 1 or q.size == 0 or q.size % 2:
                raise ValueError(f"accepted.{name} must be a flat union displacement")
            object.__setattr__(self, name, _immutable(q))
        if self.state.u_bar != target or not np.array_equal(self.state.q, self.q_after):
            raise ValueError("accepted state is not paired with q_after/load_target")
        if not np.array_equal(self.state.d, self.d_next) or not np.array_equal(
            self.state.H, self.H_next
        ):
            raise ValueError("accepted state is not paired with damage/history fields")
        for name in (
            "damage_before",
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
        ):
            diagnostic_value = getattr(self.diagnostics, name)
            if diagnostic_value is None:
                raise ValueError(f"accepted diagnostics are missing {name}")
            if not np.array_equal(getattr(self, name), diagnostic_value):
                raise ValueError(f"accepted {name} is not paired with diagnostics")
        diagnostic_q_before = self.diagnostics.q_before
        diagnostic_q_after = self.diagnostics.q_after
        if diagnostic_q_before is None or diagnostic_q_after is None:
            raise ValueError("accepted diagnostics are missing displacement fields")
        if not np.array_equal(self.q_before, diagnostic_q_before) or not np.array_equal(
            self.q_after, diagnostic_q_after
        ):
            raise ValueError("accepted displacement fields are not paired with diagnostics")
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "increment", increment)
        object.__setattr__(self, "stagger_iterations", _integer(self.stagger_iterations, "stagger_iterations", positive=True))
        object.__setattr__(self, "cutback_count", _integer(self.cutback_count, "cutback_count"))
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "acceptance_status", self.acceptance_status)
        object.__setattr__(self, "bypassed_gates", bypassed_gates)
        object.__setattr__(
            self,
            "warning_range_excess",
            None if self.warning_range_excess is None else float(self.warning_range_excess),
        )
        object.__setattr__(
            self,
            "warning_healing_magnitude",
            None
            if self.warning_healing_magnitude is None
            else float(self.warning_healing_magnitude),
        )

    @property
    def u_bar(self) -> float:
        return self.load_target

    @property
    def top_reaction(self) -> float:
        return self.mechanics_after.outer_reaction


@dataclass(frozen=True)
class EngineeringGripAttempt:
    """One attempted load target, including every strict trial sweep."""

    cutback_count: int
    load_start: float
    load_target: float
    accepted: bool
    acceptance_mode: str
    acceptance_status: str | None
    failure_stage: str | None
    failure_reason: str | None
    exception_type: str | None
    committed_before_digest: str
    committed_after_digest: str
    sweeps: tuple[EngineeringGripDiagnostics, ...]
    diagnostics: EngineeringGripDiagnostics
    bypassed_gates: tuple[str, ...]
    warning_range_excess: float | None
    warning_healing_magnitude: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutback_count", _integer(self.cutback_count, "attempt.cutback_count"))
        start = _scalar(self.load_start, "attempt.load_start")
        target = _scalar(self.load_target, "attempt.load_target")
        if target <= start:
            raise ValueError("attempt load target must exceed load start")
        mode = _validate_acceptance_mode(self.acceptance_mode)
        bypassed_gates = _validate_bypassed_gates(self.bypassed_gates)
        if mode == _ACCEPTANCE_MODE:
            if self.warning_range_excess is not None or self.warning_healing_magnitude is not None:
                raise ValueError("strict attempts cannot record warning excesses")
        if self.accepted:
            if mode == _ACCEPTANCE_MODE:
                if self.acceptance_status != _ACCEPTANCE_STATUS:
                    raise ValueError("accepted attempt status must be strict_passed")
            elif self.acceptance_status not in (_ACCEPTANCE_STATUS, _WARNING_ACCEPTANCE_STATUS):
                raise ValueError(
                    "accepted warning attempt status must be strict_passed or "
                    "warning_accepted"
                )
        elif self.acceptance_status is not None:
            raise ValueError("rejected attempt cannot have an acceptance status")
        for name in ("committed_before_digest", "committed_after_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"attempt {name} must be a SHA256 digest")
        if not self.sweeps or self.sweeps[-1] is not self.diagnostics:
            raise ValueError("attempt must retain its final sweep diagnostic")
        for index, sweep in enumerate(self.sweeps):
            if sweep.sweep_index != index:
                raise ValueError("attempt sweep indices must be contiguous")
            if sweep.target_u_bar != target:
                raise ValueError("attempt sweep target is not paired")
            if not sweep.complete_sweep and index != len(self.sweeps) - 1:
                raise ValueError("an incomplete sweep must be the final attempt sweep")
            if sweep.complete_sweep:
                if any(
                    not sweep.gate_passes[name] for name in _WARNING_HARD_GATE_NAMES
                ):
                    raise ValueError(
                        "complete sweep cannot retain a failed warning hard gate"
                    )
                if sweep.passed and index != len(self.sweeps) - 1:
                    raise ValueError(
                        "a passing complete sweep must be the final attempt sweep"
                    )
            elif (
                mode == _WARNING_ACCEPTANCE_MODE
                and sweep.failure_stage == "damage_acceptance"
            ):
                raise ValueError(
                    "warning attempts cannot retain a damage_acceptance incomplete sweep"
                )
        if mode != _ACCEPTANCE_MODE and self.accepted:
            if self.warning_range_excess is None or self.warning_healing_magnitude is None:
                raise ValueError(
                    "accepted warning attempts must record final-sweep excesses"
                )
            _require_warning_excess_records(
                self.warning_range_excess, self.warning_healing_magnitude
            )
        if self.accepted:
            if self.failure_stage is not None or self.failure_reason is not None or self.exception_type is not None:
                raise ValueError("accepted attempt cannot retain failure fields")
            if self.acceptance_status == _WARNING_ACCEPTANCE_STATUS:
                if self.diagnostics.passed:
                    raise ValueError(
                        "warning_accepted attempt diagnostic must retain passed=False"
                    )
                if bypassed_gates != _failed_warning_gates(self.diagnostics):
                    raise ValueError(
                        "warning_accepted attempt bypassed gates must match the "
                        "canonical failed-gate list"
                    )
            elif not self.diagnostics.passed or bypassed_gates:
                raise ValueError("accepted attempt diagnostic must pass without bypass")
        else:
            if not self.failure_stage or not self.failure_reason:
                raise ValueError("rejected attempt must retain failure classification")
            if self.diagnostics.passed:
                raise ValueError("rejected attempt cannot pass")
            if bypassed_gates:
                raise ValueError("rejected attempt cannot record bypassed gates")
            if self.diagnostics.complete_sweep:
                if self.exception_type is not None:
                    raise ValueError(
                        "complete failed attempt cannot retain an exception type"
                    )
                if mode != _ACCEPTANCE_MODE:
                    if self.warning_range_excess is None or self.warning_healing_magnitude is None:
                        raise ValueError(
                            "complete rejected warning attempt must record its "
                            "final-sweep excesses"
                        )
                    _require_warning_excess_records(
                        self.warning_range_excess, self.warning_healing_magnitude
                    )
            else:
                if self.exception_type != EngineeringGripSweepFailure.__name__:
                    raise ValueError(
                        "incomplete attempt must retain an EngineeringGripSweepFailure type"
                    )
                if mode != _ACCEPTANCE_MODE and (
                    self.warning_range_excess is not None
                    or self.warning_healing_magnitude is not None
                ):
                    raise ValueError(
                        "incomplete rejected warning attempt cannot record excesses"
                    )
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "acceptance_mode", mode)
        object.__setattr__(self, "sweeps", tuple(self.sweeps))
        object.__setattr__(self, "bypassed_gates", bypassed_gates)
        object.__setattr__(
            self,
            "warning_range_excess",
            None if self.warning_range_excess is None else float(self.warning_range_excess),
        )
        object.__setattr__(
            self,
            "warning_healing_magnitude",
            None
            if self.warning_healing_magnitude is None
            else float(self.warning_healing_magnitude),
        )

    @property
    def complete_sweep(self) -> bool:
        return self.diagnostics.complete_sweep


class EngineeringGripSweepFailure(RuntimeError):
    """Typed numerical failure eligible for strict cutback."""

    def __init__(self, stage: str, message: str, diagnostics: EngineeringGripDiagnostics):
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


class EngineeringGripContinuationFailure(RuntimeError):
    """Typed minimum-increment/cutback failure with immutable audit records."""

    def __init__(
        self,
        message: str,
        last_committed_state: EngineeringGripState,
        attempts: tuple[EngineeringGripAttempt, ...],
        work_check: EngineeringGripWorkCheck,
    ):
        super().__init__(message)
        self.last_committed_state = last_committed_state
        self.attempts = attempts
        self.work_check = work_check


def _recompute_model_crack_region_diagnostics(
    model: EngineeringGripModel,
    state: EngineeringGripState,
    *,
    allow_negative_damage: bool = False,
) -> CrackRegionDiagnostics:
    """Recompute final localization primitives from the committed state."""

    view = model.fixed_damage_view(state.material_damage)
    evaluation = view.evaluate(state.union_displacement)
    amor = compute_material_source_amor(
        view, state.union_displacement, state.material_history
    )
    return compute_crack_region_diagnostics(
        model.geometry.material_reference_coordinates,
        model.geometry.delta,
        state.material_damage,
        state.material_history,
        evaluation.particle_strain[: model.n_material],
        amor.particle_driver,
        model.geometry.interface_mask[: model.n_material],
        allow_negative_damage=allow_negative_damage,
    )


@dataclass(frozen=True)
class EngineeringGripResult:
    """Complete strict/warning result with zero state, accepted states, and attempts."""

    model: EngineeringGripModel
    config: EngineeringGripConfig
    work_check: EngineeringGripWorkCheck
    initial_state: EngineeringGripState
    states: tuple[EngineeringGripState, ...]
    accepted_steps: tuple[EngineeringGripAcceptedStep, ...]
    attempts: tuple[EngineeringGripAttempt, ...]
    acceptance_mode: str
    acceptance_status: str
    completion_status: str
    warning_safety_policy: str | None
    warning_safety_limit: float | None

    def __post_init__(self) -> None:
        mode = _validate_acceptance_mode(self.acceptance_mode)
        if mode == _ACCEPTANCE_MODE:
            warning_policy = None
            warning_limit = None
        else:
            warning_policy, warning_limit = _validate_warning_safety_configuration(
                self.warning_safety_policy, self.warning_safety_limit
            )
        self.validate()
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "accepted_steps", tuple(self.accepted_steps))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "warning_safety_policy", warning_policy)
        object.__setattr__(
            self,
            "warning_safety_limit",
            None
            if self.warning_safety_limit is None
            else float(self.warning_safety_limit),
        )

    @property
    def final_state(self) -> EngineeringGripState:
        return self.states[-1]

    @property
    def validated(self) -> bool:
        """Only strict-mode results can ever be validated."""

        return self.acceptance_mode == _ACCEPTANCE_MODE

    def validate(self) -> None:
        if not isinstance(self.model, EngineeringGripModel) or not isinstance(
            self.config, EngineeringGripConfig
        ):
            raise TypeError("engineering-grip result model/config types are invalid")
        mode = _validate_acceptance_mode(self.acceptance_mode)
        damage_range_bound: float
        healing_bound: float
        warning_policy: str | None
        warning_limit: float | None
        if mode == _ACCEPTANCE_MODE:
            if self.warning_safety_policy is not None:
                raise ValueError(
                    "strict engineering-grip results cannot record a warning safety policy"
                )
            if self.warning_safety_limit is not None:
                raise ValueError(
                    "strict engineering-grip results cannot record a warning limit"
                )
            if self.acceptance_status != _ACCEPTANCE_STATUS:
                raise ValueError("engineering-grip result status must be strict_passed")
            if self.completion_status != _COMPLETION_STATUS:
                raise ValueError("engineering-grip result completion status is not completed")
            damage_range_bound = self.config.damage_range_atol
            healing_bound = self.config.healing_atol
            warning_policy = None
            warning_limit = None
        else:
            warning_policy, warning_limit = _validate_warning_safety_configuration(
                self.warning_safety_policy, self.warning_safety_limit
            )
            expected_status = (
                _WARNING_ACCEPTANCE_STATUS
                if any(
                    step.acceptance_status == _WARNING_ACCEPTANCE_STATUS
                    for step in self.accepted_steps
                )
                else _ACCEPTANCE_STATUS
            )
            if self.acceptance_status != expected_status:
                raise ValueError(
                    "warning engineering-grip result status must match its steps"
                )
            if self.completion_status != _WARNING_COMPLETION_STATUS:
                raise ValueError(
                    "warning engineering-grip result completion status is not "
                    "warning_completed"
                )
            # Strict-passed steps retain the strict committed-state bounds;
            # only warning-accepted steps use the explicit warning safety
            # limit.  This keeps a smaller diagnostic limit from changing the
            # strict path when a warning run converges before its budget.
            damage_range_bound = self.config.damage_range_atol
            healing_bound = self.config.healing_atol
        if not isinstance(self.work_check, EngineeringGripWorkCheck):
            raise TypeError("engineering-grip result work_check type is invalid")
        if (
            not self.work_check.passed
            or self.work_check.relative_error > self.work_check.relative_limit
        ):
            raise ValueError("engineering-grip result zero-damage work check did not pass")
        if not isinstance(self.initial_state, EngineeringGripState) or not self.states:
            raise ValueError("engineering-grip result must retain a nonempty state sequence")
        if self.states[0] is not self.initial_state:
            raise ValueError("state sequence must begin with the exact initial state object")
        if len(self.states) != len(self.accepted_steps) + 1:
            raise ValueError("state sequence must contain initial plus accepted states")
        initial = self.initial_state
        if (
            initial.u_bar != 0.0
            or not np.array_equal(initial.q, np.zeros_like(initial.q))
            or not np.array_equal(initial.d, np.zeros_like(initial.d))
            or not np.array_equal(initial.H, np.zeros_like(initial.H))
        ):
            raise ValueError("engineering-grip initial state must be exact zero")
        if initial.q.shape != (self.model.n_dofs,) or initial.d.shape != (self.model.n_material,):
            raise ValueError("engineering-grip initial state shape is inconsistent")
        if self.final_state.u_bar != self.config.load_final:
            raise ValueError("engineering-grip result did not reach exact load_final")
        for index, state in enumerate(self.states):
            if state.q.shape != (self.model.n_dofs,) or state.d.shape != (self.model.n_material,):
                raise ValueError("engineering-grip state shape does not match model")
            _check_exact_constraints(state.q, self.model.constraints, state.u_bar, "committed state")
            state_range_bound: float | None = damage_range_bound
            if (
                mode == _WARNING_ACCEPTANCE_MODE
                and index > 0
                and self.accepted_steps[index - 1].acceptance_status
                == _WARNING_ACCEPTANCE_STATUS
            ):
                state_range_bound = warning_limit
            if state_range_bound is not None and (
                np.any(state.d < -state_range_bound)
                or np.any(state.d > 1.0 + state_range_bound)
            ):
                raise ValueError("engineering-grip committed damage is outside its hard range")
            if np.any(state.H < 0.0):
                raise ValueError("engineering-grip committed history is negative")
        for index, (previous, state) in enumerate(
            zip(self.states[:-1], self.states[1:], strict=True), start=1
        ):
            if state.u_bar <= previous.u_bar:
                raise ValueError("engineering-grip committed loads are not strictly increasing")
            transition_healing_bound: float | None = healing_bound
            if (
                mode == _WARNING_ACCEPTANCE_MODE
                and self.accepted_steps[index - 1].acceptance_status
                == _WARNING_ACCEPTANCE_STATUS
            ):
                transition_healing_bound = warning_limit
            if (
                transition_healing_bound is not None
                and np.min(state.d - previous.d) < -transition_healing_bound
            ):
                raise ValueError("engineering-grip committed damage is healing")
            if np.min(state.H - previous.H) < 0.0:
                raise ValueError("engineering-grip committed history is not monotone")
        for index, step in enumerate(self.accepted_steps, start=1):
            if not isinstance(step, EngineeringGripAcceptedStep) or self.states[index] is not step.state:
                raise ValueError("accepted steps and states are not paired")
            if step.acceptance_mode != mode:
                raise ValueError("accepted step acceptance mode does not match the result")
            if step.load_start != self.states[index - 1].u_bar or step.load_target != self.states[index].u_bar:
                raise ValueError("accepted load/state pairing is invalid")
            if mode == _WARNING_ACCEPTANCE_MODE:
                range_excess, healing_magnitude = _warning_excesses(
                    step.d_next, self.states[index - 1].d
                )
                if (
                    step.warning_range_excess != range_excess
                    or step.warning_healing_magnitude != healing_magnitude
                ):
                    raise ValueError(
                        "warning accepted step does not retain its raw excess values"
                    )
                if step.acceptance_status == _WARNING_ACCEPTANCE_STATUS:
                    if (
                        warning_policy == _WARNING_SAFETY_POLICY_BOUNDED
                        and warning_limit is not None
                        and (
                            range_excess > warning_limit
                            or healing_magnitude > warning_limit
                        )
                    ):
                        raise ValueError(
                            "warning accepted state exceeds its raw damage safety limit"
                        )
                    decision, expected_bypassed, _, _ = _warning_acceptance_decision(
                        step.diagnostics,
                        self.states[index - 1].d,
                        warning_limit,
                        warning_safety_policy=warning_policy,
                    )
                    if not decision:
                        raise ValueError(
                            "warning accepted step fails the final-sweep warning decision"
                        )
                    if step.stagger_iterations != self.config.max_stagger_iterations:
                        raise ValueError(
                            "warning accepted step must use the full stagger budget"
                        )
                    if (
                        step.diagnostics.sweep_index + 1
                        != self.config.max_stagger_iterations
                    ):
                        raise ValueError(
                            "warning accepted step diagnostic is not at the full "
                            "stagger budget"
                        )
                    if step.bypassed_gates != expected_bypassed:
                        raise ValueError(
                            "warning accepted step bypassed gates are not canonical"
                        )
        if not self.attempts:
            raise ValueError("engineering-grip result must retain attempt audit records")
        current = 0
        accepted_step_index = 0
        for attempt in self.attempts:
            if not isinstance(attempt, EngineeringGripAttempt):
                raise TypeError("attempt sequence contains a wrong type")
            if attempt.acceptance_mode != mode:
                raise ValueError("attempt acceptance mode does not match the result")
            if attempt.load_start != self.states[current].u_bar:
                raise ValueError("attempt load start is not paired with committed state")
            if attempt.committed_before_digest != self.states[current].digest:
                raise ValueError("attempt committed-before digest is not paired")
            if attempt.accepted:
                if accepted_step_index >= len(self.accepted_steps):
                    raise ValueError("accepted attempt sequence exceeds accepted steps")
                step = self.accepted_steps[accepted_step_index]
                if attempt.diagnostics is not step.diagnostics:
                    raise ValueError("accepted attempt diagnostic is not paired with its step")
                if len(attempt.sweeps) != step.stagger_iterations:
                    raise ValueError(
                        "accepted attempt sweep count is not paired with its step"
                    )
                if attempt.acceptance_status != step.acceptance_status:
                    raise ValueError("accepted attempt status is not paired with its step")
                if attempt.bypassed_gates != step.bypassed_gates:
                    raise ValueError("accepted attempt bypassed gates are not paired")
                if mode == _WARNING_ACCEPTANCE_MODE and (
                    attempt.warning_range_excess != step.warning_range_excess
                    or attempt.warning_healing_magnitude
                    != step.warning_healing_magnitude
                ):
                    raise ValueError(
                        "accepted attempt warning excesses are not paired with its step"
                    )
                accepted_step_index += 1
                current += 1
                if current >= len(self.states) or attempt.committed_after_digest != self.states[current].digest:
                    raise ValueError("accepted attempt committed-after digest is not paired")
            elif attempt.committed_after_digest != self.states[current].digest:
                raise ValueError("rejected attempt changed committed state")
            elif mode == _WARNING_ACCEPTANCE_MODE and attempt.diagnostics.complete_sweep:
                d_next = attempt.diagnostics.d_next
                if d_next is None:
                    raise ValueError(
                        "complete warning attempt diagnostic is missing d_next"
                    )
                range_excess, healing_magnitude = _warning_excesses(
                    d_next, self.states[current].d
                )
                if (
                    attempt.warning_range_excess != range_excess
                    or attempt.warning_healing_magnitude != healing_magnitude
                ):
                    raise ValueError(
                        "rejected warning attempt does not retain its raw excess values"
                    )
            if (
                not attempt.accepted
                and attempt.diagnostics.complete_sweep
                and len(attempt.sweeps) != self.config.max_stagger_iterations
            ):
                raise ValueError(
                    "rejected complete attempt must retain the full stagger budget"
                )
        if current != len(self.states) - 1:
            raise ValueError("attempt audit does not reach the final state")
        if accepted_step_index != len(self.accepted_steps):
            raise ValueError("attempt audit does not cover every accepted step")
        final_localization = _recompute_model_crack_region_diagnostics(
            self.model,
            self.final_state,
            allow_negative_damage=mode == _WARNING_ACCEPTANCE_MODE,
        )
        if mode == _ACCEPTANCE_MODE and not final_localization.localization_pass:
            raise ValueError(
                "engineering-grip final crack localization/interface gate failed"
            )


def _direct_seed(
    previous_q: np.ndarray,
    previous_load: float,
    target_load: float,
    constraints: AffineConstraints,
) -> np.ndarray:
    """Transfer a prior direct-load state without normalized loading algebra."""

    q = _f64(previous_q, "direct mechanics seed", (constraints.n_dofs,)).copy()
    _check_exact_constraints(q, constraints, previous_load, "source direct seed")
    prescribed = constraints.prescribed_vector(target_load)
    q[constraints.constrained_dofs] = prescribed[constraints.constrained_dofs]
    _check_exact_constraints(q, constraints, target_load, "target direct seed")
    if not np.all(np.isfinite(q)):
        raise FloatingPointError("direct mechanics seed is non-finite")
    return _immutable(q)


def _run_engineering_attempt(
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    target: float,
    cutback_count: int,
    step_index: int,
    mechanics_adapter: JaxA2CPUReplayAdapter | None = None,
    *,
    acceptance_mode: str,
    warning_safety_limit: float | None = None,
    warning_safety_policy: str | None = None,
) -> tuple[EngineeringGripAcceptedStep | None, EngineeringGripAttempt]:
    mode = _validate_acceptance_mode(acceptance_mode)
    warning_mode = mode == _WARNING_ACCEPTANCE_MODE
    warning_policy: str | None = None
    warning_limit: float | None = None
    if warning_mode:
        warning_policy, warning_limit = _validate_warning_safety_configuration(
            _WARNING_SAFETY_POLICY_BOUNDED
            if warning_safety_policy is None
            else warning_safety_policy,
            warning_safety_limit,
        )
    elif warning_safety_limit is not None or warning_safety_policy is not None:
        raise ValueError(
            "warning safety policy/limit are only valid in warning mode"
        )
    d_iter = np.array(committed.material_damage, dtype=np.float64, copy=True)
    q_seed = _direct_seed(
        committed.union_displacement,
        committed.u_bar,
        target,
        model.constraints,
    )
    sweeps: list[EngineeringGripDiagnostics] = []
    final: EngineeringGripDiagnostics | None = None
    exception_type: str | None = None
    for sweep_index in range(config.max_stagger_iterations):
        view_before = model.fixed_damage_view(d_iter)
        try:
            before = _solve_direct_mechanics(
                mechanics_adapter,
                view_before,
                model.constraints,
                target,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
                initial_guess=q_seed,
                seed_kind="previous_direct_solution",
                source_state="committed" if sweep_index == 0 else "complete_trial_sweep",
                source_step=step_index,
                source_sweep=sweep_index,
                source_damage_digest=_damage_digest(committed.material_damage)
                if sweep_index == 0
                else _damage_digest(d_iter),
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_before",
                str(error),
                model=model,
                sweep_index=sweep_index,
            )
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        (
            mechanics_before,
            _,
            _,
            _,
            force_before_pass,
            moment_before_pass,
        ) = _mechanics_gate(before, config, _geometry_length(model))
        if not mechanics_before or (
            not warning_mode and (not force_before_pass or not moment_before_pass)
        ):
            reason_parts: list[str] = []
            if not mechanics_before:
                reason_parts.append("mechanics")
            if not force_before_pass:
                reason_parts.append("force")
            if not moment_before_pass:
                reason_parts.append("moment")
            reason = "mechanics-before " + "/".join(reason_parts) + " gate failed"
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_before_gate",
                reason,
                before=before,
                model=model,
                sweep_index=sweep_index,
            )
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        try:
            amor_before = compute_material_source_amor(
                view_before, before.displacement, committed.material_history
            )
            psi_before = np.asarray(amor_before.particle_driver, dtype=np.float64)
            H_used = _immutable(
                np.maximum(committed.material_history, psi_before).astype(
                    np.float64, copy=False
                )
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "amor_before",
                str(error),
                before=before,
                model=model,
                sweep_index=sweep_index,
            )
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        try:
            damage_result = solve_fixed_history_damage(
                model.material_operator, model.Gc, model.l0, H_used
            )
            if not isinstance(damage_result, FixedHistoryDamageResult):
                raise TypeError("fixed-history solve returned the wrong result type")
            d_next = _f64(
                damage_result.damage,
                "fixed-history damage",
                (model.n_material,),
            )
            if not np.array_equal(damage_result.history, H_used):
                raise ValueError("fixed-history result is paired with a different H_used")
            rho_used = _f64(
                damage_result.strong_residual,
                "fixed-history strong residual",
                d_next.shape,
            )
            rho_used_volume = _f64(
                damage_result.euclidean_residual,
                "fixed-history volume residual",
                d_next.shape,
            )
            _compare_damage_residual_reconstruction(
                rho_used,
                rho_used_volume,
                d_next,
                H_used,
                model.material_operator.to_scipy().tocsr(),
                model.material_operator.volumes,
                model.Gc,
                model.l0,
                "fixed-history residual",
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "damage",
                str(error),
                before=before,
                psi_before=psi_before,
                H_used=H_used,
                model=model,
                sweep_index=sweep_index,
            )
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        range_pass = bool(
            np.min(d_next) >= -config.damage_range_atol
            and np.max(d_next) <= 1.0 + config.damage_range_atol
        )
        healing_pass = bool(
            np.min(d_next - committed.material_damage) >= -config.healing_atol
        )
        if not range_pass or not healing_pass:
            if not warning_mode:
                names = []
                if not range_pass:
                    names.append("damage_range")
                if not healing_pass:
                    names.append("healing")
                diagnostic = _incomplete_diagnostics(
                    config,
                    committed,
                    d_iter,
                    target,
                    "damage_acceptance",
                    ", ".join(names),
                    before=before,
                    psi_before=psi_before,
                    H_used=H_used,
                    d_next=d_next,
                    rho_used=rho_used,
                    rho_used_volume=rho_used_volume,
                    model=model,
                    sweep_index=sweep_index,
                )
                sweeps.append(diagnostic)
                final = diagnostic
                exception_type = EngineeringGripSweepFailure.__name__
                break
            # Warning mode never short-circuits a finite out-of-range/healing
            # candidate: the sweep completes through mechanics-after/history so
            # later sweeps may recover, and the raw magnitudes are only
            # compared with the warning safety limit on the final complete
            # sweep below.

        view_after = model.fixed_damage_view(d_next)
        try:
            after = _solve_direct_mechanics(
                mechanics_adapter,
                view_after,
                model.constraints,
                target,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
                initial_guess=_direct_seed(
                    before.displacement,
                    target,
                    target,
                    model.constraints,
                ),
                seed_kind="mechanics_before_solution",
                source_state="mechanics_before",
                source_step=step_index,
                source_sweep=sweep_index,
                source_damage_digest=_damage_digest(d_iter),
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_after",
                str(error),
                before=before,
                psi_before=psi_before,
                H_used=H_used,
                d_next=d_next,
                rho_used=rho_used,
                rho_used_volume=rho_used_volume,
                model=model,
                sweep_index=sweep_index,
            )
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        try:
            amor_after = compute_material_source_amor(
                view_after, after.displacement, committed.material_history
            )
            psi_after = np.asarray(amor_after.particle_driver, dtype=np.float64)
        except (RuntimeError, FloatingPointError) as error:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "amor_after",
                str(error),
                before=before,
                after=after,
                psi_before=psi_before,
                H_used=H_used,
                d_next=d_next,
                rho_used=rho_used,
                rho_used_volume=rho_used_volume,
                model=model,
                sweep_index=sweep_index,
            )
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        (
            mechanics_after,
            _,
            _,
            _,
            force_after_pass,
            moment_after_pass,
        ) = _mechanics_gate(after, config, _geometry_length(model))
        if not mechanics_after or (
            not warning_mode and (not force_after_pass or not moment_after_pass)
        ):
            reason_parts = []
            if not mechanics_after:
                reason_parts.append("mechanics")
            if not force_after_pass:
                reason_parts.append("force")
            if not moment_after_pass:
                reason_parts.append("moment")
            reason = "mechanics-after " + "/".join(reason_parts) + " gate failed"
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_after_gate",
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
            sweeps.append(diagnostic)
            final = diagnostic
            exception_type = EngineeringGripSweepFailure.__name__
            break

        # The committed history, not H_used, is the lower envelope for the
        # final history update.  This ordering is the central rollback rule.
        H_next = _immutable(
            np.maximum(committed.material_history, psi_after).astype(
                np.float64, copy=False
            )
        )
        _, rho_next, _, rho_next_volume, residual_scale = _damage_residuals(
            model, d_next, H_used, H_next
        )
        _compare_damage_residual_reconstruction(
            rho_used,
            rho_used_volume,
            d_next,
            H_used,
            model.material_operator.to_scipy().tocsr(),
            model.material_operator.volumes,
            model.Gc,
            model.l0,
            "rho_used reconstruction",
        )
        at_stagger_budget = sweep_index + 1 == config.max_stagger_iterations
        diagnostics = _full_diagnostics(
            config,
            model,
            committed,
            d_iter,
            d_next,
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
            warning_safety_limit=(
                warning_limit
                if warning_mode and at_stagger_budget
                else None
            ),
        )
        sweeps.append(diagnostics)
        final = diagnostics
        range_excess, healing_magnitude = _warning_excesses(
            d_next, committed.material_damage
        )
        warning_acceptance = False
        bypassed_gates: tuple[str, ...] = ()
        if not diagnostics.passed and warning_mode and at_stagger_budget:
            # Warning acceptance is considered only on the final complete
            # sweep.  Finite bypassable-gate failures are recorded exactly;
            # non-finite values and any hard-gate failure stay rejected.
            (
                warning_acceptance,
                bypassed_gates,
                range_excess,
                healing_magnitude,
            ) = _warning_acceptance_decision(
                diagnostics,
                committed.material_damage,
                warning_limit,
                warning_safety_policy=warning_policy,
            )
        if diagnostics.passed or warning_acceptance:
            state = EngineeringGripState(
                u_bar=target,
                union_displacement=after.displacement,
                material_damage=d_next,
                material_history=H_next,
            )
            step = EngineeringGripAcceptedStep(
                load_start=committed.u_bar,
                load_target=target,
                increment=target - committed.u_bar,
                state=state,
                damage_before=committed.material_damage,
                d_iter=d_iter,
                d_next=d_next,
                q_before=before.displacement,
                q_after=after.displacement,
                psi_before=psi_before,
                psi_after=psi_after,
                H_used=H_used,
                H_next=H_next,
                rho_used=rho_used,
                rho_next=rho_next,
                rho_used_volume=rho_used_volume,
                rho_next_volume=rho_next_volume,
                mechanics_before=before,
                mechanics_after=after,
                diagnostics=diagnostics,
                stagger_iterations=sweep_index + 1,
                cutback_count=cutback_count,
                acceptance_mode=mode,
                acceptance_status=(
                    _WARNING_ACCEPTANCE_STATUS
                    if warning_acceptance
                    else _ACCEPTANCE_STATUS
                ),
                bypassed_gates=bypassed_gates if warning_acceptance else (),
                warning_range_excess=(
                    range_excess if warning_mode else None
                ),
                warning_healing_magnitude=(
                    healing_magnitude if warning_mode else None
                ),
            )
            attempt = EngineeringGripAttempt(
                cutback_count=cutback_count,
                load_start=committed.u_bar,
                load_target=target,
                accepted=True,
                acceptance_mode=mode,
                acceptance_status=(
                    _WARNING_ACCEPTANCE_STATUS
                    if warning_acceptance
                    else _ACCEPTANCE_STATUS
                ),
                failure_stage=None,
                failure_reason=None,
                exception_type=None,
                committed_before_digest=committed.digest,
                committed_after_digest=state.digest,
                sweeps=tuple(sweeps),
                diagnostics=diagnostics,
                bypassed_gates=bypassed_gates if warning_acceptance else (),
                warning_range_excess=(
                    range_excess if warning_mode else None
                ),
                warning_healing_magnitude=(
                    healing_magnitude if warning_mode else None
                ),
            )
            return step, attempt
        if not at_stagger_budget:
            # A complete failed sweep may update only local trial variables;
            # committed damage/history and the cutback state are untouched.
            d_iter = np.array(d_next, dtype=np.float64, copy=True)
            q_seed = _direct_seed(
                after.displacement, target, target, model.constraints
            )
            continue
        break

    if final is None:
        raise RuntimeError("engineering-grip attempt produced no diagnostics")
    attempt = EngineeringGripAttempt(
        cutback_count=cutback_count,
        load_start=committed.u_bar,
        load_target=target,
        accepted=False,
        acceptance_mode=mode,
        acceptance_status=None,
        failure_stage=final.failure_stage or "gates",
        failure_reason=final.failure_reason or "strict engineering-grip sweep failed",
        exception_type=exception_type,
        committed_before_digest=committed.digest,
        committed_after_digest=committed.digest,
        sweeps=tuple(sweeps),
        diagnostics=final,
        bypassed_gates=(),
        warning_range_excess=(
            _warning_excesses(final.d_next, committed.material_damage)[0]
            if warning_mode and final.complete_sweep and final.d_next is not None
            else None
        ),
        warning_healing_magnitude=(
            _warning_excesses(final.d_next, committed.material_damage)[1]
            if warning_mode and final.complete_sweep and final.d_next is not None
            else None
        ),
    )
    return None, attempt


def _next_target(current: float, increment: float, final: float) -> float:
    remaining = final - current
    target = current + min(increment, remaining)
    if final - target <= 8.0 * abs(np.spacing(np.float64(final))):
        return final
    return target


def _is_minimum_increment(
    trial_increment: float,
    du_min: float,
    load_start: float,
    load_target: float,
) -> bool:
    """Treat a float64 representational excess over ``du_min`` as minimum.

    The subtraction is classified with eight ULPs at the largest participating
    load/increment scale, matching the endpoint-snap roundoff budget.
    """

    trial = _scalar(trial_increment, "trial_increment")
    minimum = _scalar(du_min, "du_min")
    start = _scalar(load_start, "load_start")
    target = _scalar(load_target, "load_target")
    scale = max(abs(trial), abs(minimum), abs(start), abs(target))
    ulp = abs(float(np.spacing(np.float64(scale))))
    if not np.isfinite(ulp):
        # Keep the bound finite at the largest representable endpoint.
        ulp = _EPS64 * scale
    roundoff_bound = 8.0 * ulp
    return bool(trial <= minimum + roundoff_bound)


def _validate_replay_sweep_termination(
    diagnostic: Any,
    sweep_index: int,
    sweep_count: int,
) -> None:
    """Reject sweep sequences that production would have terminated earlier."""

    if bool(getattr(diagnostic, "passed", False)) and sweep_index != sweep_count - 1:
        raise ValueError(
            "engineering-grip archive continues after a strict-passing sweep"
        )


def run_strict_engineering_grip(
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    mechanics_adapter: JaxA2CPUReplayAdapter | None = None,
) -> EngineeringGripResult:
    """Run strict direct-load staggered continuation with rollback/cutback.

    ``mechanics_adapter`` is the explicit restricted JAX adapter for the
    one-time work check and every stagger mechanics solve.  It must match the
    model static/constraint identity exactly; the default ``None`` keeps the
    CPU reference path used by the manufactured tests.
    """

    if not isinstance(model, EngineeringGripModel) or not isinstance(
        config, EngineeringGripConfig
    ):
        raise TypeError("run_strict_engineering_grip requires model/config types")
    return _run_engineering_continuation(
        model,
        config,
        mechanics_adapter,
        _ACCEPTANCE_MODE,
        None,
        None,
    )


def run_warning_engineering_grip(
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    warning_safety_limit: float | str | None = None,
    mechanics_adapter: JaxA2CPUReplayAdapter | None = None,
    *,
    warning_safety_policy: str | None = _WARNING_SAFETY_POLICY_BOUNDED,
) -> EngineeringGripResult:
    """Run the separately archived engineering-grip warning diagnostic.

    This mode shares the exact strict equations, sweep ordering, tolerances,
    rollback, cutback rules, one-time work check, and typed hard failures.
    It differs only at acceptance: a trial runs the full configured stagger
    budget unless every strict gate passes earlier, finite raw damage
    range/healing violations are never short-circuited at intermediate sweeps,
    and only on the final complete sweep may finite failed bypassable gates
    (:data:`WARNING_GATE_NAMES`) be warning-accepted.  Under the bounded
    policy, the raw ``range_excess=max(0,-min(d_next),max(d_next)-1)`` and
    ``healing_magnitude=max(0,-min(d_next-d_committed))`` must be at most
    ``warning_safety_limit``.  Under the unbounded policy, both values are
    retained and reported but have no magnitude cutoff.  The committed
    candidate is the exact unmodified ``(q_after, d_next, H_next)`` with
    ``diagnostics.passed=False`` and the exact canonical bypassed-gate list.
    The result is always ``validated=False``; final crack-localization
    failure is archived as a warning, never as validation.
    """

    if not isinstance(model, EngineeringGripModel) or not isinstance(
        config, EngineeringGripConfig
    ):
        raise TypeError("run_warning_engineering_grip requires model/config types")
    # The original API accepted the bounded limit as the third positional
    # argument.  A string in that position is the explicit policy spelling for
    # the new API; an optional fourth positional value is retained as the
    # bounded limit for callers using ``(model, config, policy, limit)``.
    if isinstance(warning_safety_limit, str):
        if warning_safety_policy != _WARNING_SAFETY_POLICY_BOUNDED:
            raise ValueError(
                "warning safety policy was supplied both positionally and by keyword"
            )
        warning_safety_policy = warning_safety_limit
        if mechanics_adapter is not None and not isinstance(
            mechanics_adapter, JaxA2CPUReplayAdapter
        ):
            warning_safety_limit = mechanics_adapter
            mechanics_adapter = None
        else:
            warning_safety_limit = None
    policy, limit = _validate_warning_safety_configuration(
        _WARNING_SAFETY_POLICY_BOUNDED
        if warning_safety_policy is None
        else warning_safety_policy,
        warning_safety_limit,
    )
    return _run_engineering_continuation(
        model,
        config,
        mechanics_adapter,
        _WARNING_ACCEPTANCE_MODE,
        limit,
        policy,
    )


def _run_engineering_continuation(
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    mechanics_adapter: JaxA2CPUReplayAdapter | None,
    acceptance_mode: str,
    warning_safety_limit: float | None,
    warning_safety_policy: str | None,
) -> EngineeringGripResult:
    """Shared strict/warning staggered continuation with rollback/cutback."""

    mode = _validate_acceptance_mode(acceptance_mode)
    if mode == _WARNING_ACCEPTANCE_MODE:
        policy, warning_limit = _validate_warning_safety_configuration(
            _WARNING_SAFETY_POLICY_BOUNDED
            if warning_safety_policy is None
            else warning_safety_policy,
            warning_safety_limit,
        )
    else:
        if warning_safety_limit is not None or warning_safety_policy is not None:
            raise ValueError(
                "warning safety policy/limit are only valid in warning mode"
            )
        policy = None
        warning_limit = None
    adapter = _validated_mechanics_adapter(model, mechanics_adapter)

    # This is deliberately outside the stagger loop and is called exactly
    # once.  No fixed-damage work solve is hidden in any sweep gate.
    work_check = run_zero_damage_work_check(
        model,
        config.work_check_load,
        config.work_epsilon,
        config.work_relative_limit,
        config.cg_rtol,
        config.cg_atol,
        config.cg_maxiter,
        mechanics_adapter=adapter,
    )
    initial = EngineeringGripState(
        u_bar=0.0,
        union_displacement=np.zeros(model.n_dofs, dtype=np.float64),
        material_damage=np.zeros(model.n_material, dtype=np.float64),
        material_history=np.zeros(model.n_material, dtype=np.float64),
    )
    states: list[EngineeringGripState] = [initial]
    steps: list[EngineeringGripAcceptedStep] = []
    attempts: list[EngineeringGripAttempt] = []
    committed = initial
    increment = config.du_initial
    cutbacks = 0
    while committed.u_bar < config.load_final:
        target = _next_target(committed.u_bar, increment, config.load_final)
        if target <= committed.u_bar:
            raise EngineeringGripContinuationFailure(
                f"{mode} engineering-grip continuation made no load progress",
                committed,
                tuple(attempts),
                work_check,
            )
        step, attempt = _run_engineering_attempt(
            model,
            config,
            committed,
            target,
            cutbacks,
            len(steps) + 1,
            mechanics_adapter=adapter,
            acceptance_mode=mode,
            warning_safety_limit=warning_limit,
            warning_safety_policy=policy,
        )
        attempts.append(attempt)
        if step is not None:
            steps.append(step)
            committed = step.state
            states.append(committed)
            increment = step.increment
            cutbacks = 0
            continue
        if attempt.committed_before_digest != attempt.committed_after_digest:
            raise RuntimeError("rejected engineering-grip attempt changed committed state")
        at_minimum = _is_minimum_increment(
            target - committed.u_bar,
            config.du_min,
            committed.u_bar,
            target,
        )
        at_cutback_limit = cutbacks >= config.max_cutbacks
        if at_minimum or at_cutback_limit:
            reasons = []
            if at_minimum:
                reasons.append("du_min")
            if at_cutback_limit:
                reasons.append("max_cutbacks")
            raise EngineeringGripContinuationFailure(
                f"{mode} engineering-grip continuation failed at "
                f"u_bar={target:.17g} ({', '.join(reasons)}): {attempt.failure_reason}",
                committed,
                tuple(attempts),
                work_check,
            )
        cutbacks += 1
        increment = max(config.du_min, (target - committed.u_bar) * config.cutback_factor)
    if mode == _ACCEPTANCE_MODE:
        acceptance_status = _ACCEPTANCE_STATUS
        completion_status = _COMPLETION_STATUS
    else:
        acceptance_status = (
            _WARNING_ACCEPTANCE_STATUS
            if any(
                step.acceptance_status == _WARNING_ACCEPTANCE_STATUS
                for step in steps
            )
            else _ACCEPTANCE_STATUS
        )
        completion_status = _WARNING_COMPLETION_STATUS
    return EngineeringGripResult(
        model=model,
        config=config,
        work_check=work_check,
        initial_state=initial,
        states=tuple(states),
        accepted_steps=tuple(steps),
        attempts=tuple(attempts),
        acceptance_mode=mode,
        acceptance_status=acceptance_status,
        completion_status=completion_status,
        warning_safety_policy=policy,
        warning_safety_limit=warning_limit,
    )


run_strict_continuation = run_strict_engineering_grip


def _gate_payload(config: EngineeringGripConfig) -> dict[str, Any]:
    return {
        name: {
            "atol": float(getattr(config, name).atol),
            "rtol": float(getattr(config, name).rtol),
            "scale_floor": float(getattr(config, name).scale_floor),
        }
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


def _config_payload(config: EngineeringGripConfig) -> dict[str, Any]:
    return {
        "load_start": config.load_start,
        "load_final": config.load_final,
        "du_initial": config.du_initial,
        "du_min": config.du_min,
        "cutback_factor": config.cutback_factor,
        "max_cutbacks": config.max_cutbacks,
        "max_stagger_iterations": config.max_stagger_iterations,
        "cg_rtol": config.cg_rtol,
        "cg_atol": config.cg_atol,
        "cg_maxiter": config.cg_maxiter,
        "gates": _gate_payload(config),
        "healing_atol": config.healing_atol,
        "damage_range_atol": config.damage_range_atol,
        "work_check_load": config.work_check_load,
        "work_epsilon": config.work_epsilon,
        "work_relative_limit": config.work_relative_limit,
    }


def _config_from_payload(payload: Mapping[str, Any]) -> EngineeringGripConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("engineering-grip config payload must be a mapping")
    required = {
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
        "gates",
        "healing_atol",
        "damage_range_atol",
        "work_check_load",
        "work_epsilon",
        "work_relative_limit",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"engineering-grip config is missing fields: {missing}")
    gates = payload["gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("engineering-grip config gates must be a mapping")
    gate_values: dict[str, GateTolerance] = {}
    for name in (
        "displacement",
        "damage",
        "history",
        "reaction",
        "damage_residual",
        "force",
        "moment",
    ):
        raw = gates.get(name)
        if not isinstance(raw, Mapping) or not {"atol", "rtol", "scale_floor"}.issubset(raw):
            raise ValueError(f"engineering-grip config gate {name} is incomplete")
        gate_values[name] = GateTolerance(
            raw["atol"], raw["rtol"], raw["scale_floor"]
        )
    return EngineeringGripConfig(
        load_start=payload["load_start"],
        load_final=payload["load_final"],
        du_initial=payload["du_initial"],
        du_min=payload["du_min"],
        cutback_factor=payload["cutback_factor"],
        max_cutbacks=payload["max_cutbacks"],
        max_stagger_iterations=payload["max_stagger_iterations"],
        cg_rtol=payload["cg_rtol"],
        cg_atol=payload["cg_atol"],
        cg_maxiter=payload["cg_maxiter"],
        displacement=gate_values["displacement"],
        damage=gate_values["damage"],
        history=gate_values["history"],
        reaction=gate_values["reaction"],
        damage_residual=gate_values["damage_residual"],
        force=gate_values["force"],
        moment=gate_values["moment"],
        healing_atol=payload["healing_atol"],
        damage_range_atol=payload["damage_range_atol"],
        work_check_load=payload["work_check_load"],
        work_epsilon=payload["work_epsilon"],
        work_relative_limit=payload["work_relative_limit"],
    )


def _diagnostic_payload(diagnostic: EngineeringGripDiagnostics) -> dict[str, Any]:
    array_names = (
        "damage_before",
        "d_iter",
        "d_next",
        "q_before",
        "q_after",
        "psi_before",
        "psi_after",
        "H_used",
        "H_next",
        "rho_used",
        "rho_next",
        "rho_used_volume",
        "rho_next_volume",
        "force_before",
        "force_after",
    )
    scalars = {
        name: getattr(diagnostic, name)
        for name in (
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
    }
    return {
        "sweep_index": diagnostic.sweep_index,
        "target_u_bar": diagnostic.target_u_bar,
        "complete_sweep": diagnostic.complete_sweep,
        "passed": diagnostic.passed,
        "failure_stage": diagnostic.failure_stage,
        "failure_reason": diagnostic.failure_reason,
        "gate_passes": diagnostic.gate_passes,
        **{
            name: bool(getattr(diagnostic, name))
            for name in _DIAGNOSTIC_BOOL_FIELDS
        },
        "available_fields": list(diagnostic.available_fields),
        **{
            name: (
                None
                if getattr(diagnostic, name) is None
                else np.asarray(getattr(diagnostic, name)).tolist()
            )
            for name in array_names
        },
        **scalars,
    }


_DIAGNOSTIC_ARRAY_FIELDS = (
    "damage_before",
    "d_iter",
    "d_next",
    "q_before",
    "q_after",
    "psi_before",
    "psi_after",
    "H_used",
    "H_next",
    "rho_used",
    "rho_next",
    "rho_used_volume",
    "rho_next_volume",
    "force_before",
    "force_after",
)
_DIAGNOSTIC_SCALAR_FIELDS = (
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
_DIAGNOSTIC_BOOL_FIELDS = (
    "finite_pass",
    "mechanics_before_pass",
    "mechanics_after_pass",
    "displacement_pass",
    "damage_pass",
    "history_pass",
    "reaction_pass",
    "damage_residual_pass",
    "damage_range_pass",
    "healing_pass",
    "history_monotonic_pass",
    "force_pass",
    "moment_pass",
)


def _diag_namespace(payload: Mapping[str, Any]) -> Any:
    """Convert one numerical JSON diagnostic into replay attributes."""

    if not isinstance(payload, Mapping):
        raise ValueError("engineering-grip diagnostic payload must be a mapping")
    converted: dict[str, Any] = {}
    for name, value in payload.items():
        if name in _DIAGNOSTIC_ARRAY_FIELDS:
            converted[name] = (
                None
                if value is None
                else np.asarray(value, dtype=np.float64)
            )
        elif name == "available_fields":
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError("engineering-grip diagnostic available_fields is invalid")
            converted[name] = tuple(value)
        elif name == "gate_passes":
            if not isinstance(value, Mapping):
                raise ValueError("engineering-grip diagnostic gate_passes is invalid")
            expected = set(_WARNING_HARD_GATE_NAMES) | set(WARNING_GATE_NAMES)
            if set(value) != expected or any(type(item) is not bool for item in value.values()):
                raise ValueError(
                    "engineering-grip diagnostic gate_passes must contain canonical booleans"
                )
            converted[name] = dict(value)
        elif name in ("cg_info_before", "cg_info_after"):
            converted[name] = _optional_integer(value, f"diagnostic.{name}")
        else:
            converted[name] = value
    return SimpleNamespace(**converted)


def _compare_replay_scalar(
    actual: Any,
    expected: Any,
    name: str,
    limit: float = 2.0e-8,
    absolute_atol: float | None = None,
    scale_floor: float = 1.0,
) -> None:
    """Compare one finite scalar against an independently rebuilt value."""

    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{name} availability does not match replay")
        return
    left = _scalar(actual, f"replay {name}")
    right = _scalar(expected, f"expected replay {name}")
    tolerance = limit * max(
        abs(left), abs(right), _scalar(scale_floor, f"{name} scale floor")
    )
    if absolute_atol is not None:
        tolerance = max(tolerance, _scalar(absolute_atol, f"{name} absolute tolerance"))
    if abs(left - right) > tolerance:
        raise ValueError(f"{name} does not match replay: defect={abs(left - right):.17g}")


def _compare_diagnostic_records(
    first: Any,
    second: Any,
    name: str,
) -> None:
    """Compare an archived diagnostic duplicate without object identity tests."""

    for field in (
        "sweep_index",
        "target_u_bar",
        "complete_sweep",
        "passed",
        "failure_stage",
        "failure_reason",
        "available_fields",
    ):
        if getattr(first, field, None) != getattr(second, field, None):
            raise ValueError(f"{name}.{field} differs")
    for field in _DIAGNOSTIC_ARRAY_FIELDS:
        left = getattr(first, field, None)
        right = getattr(second, field, None)
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"{name}.{field} availability differs")
        else:
            _array_close(left, right, f"{name}.{field}", 2.0e-12)
    for field in _DIAGNOSTIC_SCALAR_FIELDS:
        _compare_replay_scalar(
            getattr(first, field, None), getattr(second, field, None), f"{name}.{field}", 2.0e-12
        )
    for field in _DIAGNOSTIC_BOOL_FIELDS:
        if bool(getattr(first, field, False)) != bool(getattr(second, field, False)):
            raise ValueError(f"{name}.{field} differs")
    first_gates = getattr(first, "gate_passes", None)
    second_gates = getattr(second, "gate_passes", None)
    if first_gates != second_gates:
        raise ValueError(f"{name}.gate_passes differs")


def _state_field_arrays(
    result: EngineeringGripResult,
) -> dict[str, np.ndarray]:
    model = result.model
    states = result.states
    material_strain = [
        np.zeros((model.n_material, 2, 2), dtype=np.float64)
    ]
    amor_driver = [np.zeros(model.n_material, dtype=np.float64)]
    interface_opening = [0.0]
    interface_damage_bottom: list[float] = [0.0]
    interface_damage_top: list[float] = [0.0]
    interface_history_bottom: list[float] = [0.0]
    interface_history_top: list[float] = [0.0]
    material_bottom = model.geometry.interface_bottom_mask[: model.n_material]
    material_top = model.geometry.interface_top_mask[: model.n_material]
    for step in result.accepted_steps:
        material_strain.append(
            np.asarray(
                step.mechanics_after.evaluation.particle_strain[: model.n_material],
                dtype=np.float64,
            )
        )
        amor_driver.append(np.asarray(step.psi_after, dtype=np.float64))
        interface_opening.append(
            compute_interface_opening_diagnostic(
                step.state.q, model.geometry, model.top_line, model.bottom_line
            )
        )
        interface_damage_bottom.append(float(np.max(step.state.d[material_bottom])))
        interface_damage_top.append(float(np.max(step.state.d[material_top])))
        interface_history_bottom.append(float(np.max(step.state.H[material_bottom])))
        interface_history_top.append(float(np.max(step.state.H[material_top])))
    strain = np.stack(material_strain)
    amor = np.stack(amor_driver)
    crack = model.crack_region_mask
    crack_strain = strain[:, crack]
    crack_amor = amor[:, crack]
    crack_damage_inside: list[float] = []
    crack_damage_outside: list[float] = []
    crack_history_inside: list[float] = []
    crack_history_outside: list[float] = []
    crack_strain_inside: list[float] = []
    crack_strain_outside: list[float] = []
    crack_amor_inside: list[float] = []
    crack_amor_outside: list[float] = []
    crack_localization_pass: list[bool] = []
    interface_mask = model.geometry.interface_mask[: model.n_material]
    for state, state_strain, state_amor in zip(states, strain, amor):
        diagnostics = compute_crack_region_diagnostics(
            model.geometry.material_reference_coordinates,
            model.geometry.delta,
            state.material_damage,
            state.material_history,
            state_strain,
            state_amor,
            interface_mask,
            allow_negative_damage=result.acceptance_mode == _WARNING_ACCEPTANCE_MODE,
        )
        crack_damage_inside.append(diagnostics.damage_inside_max)
        crack_damage_outside.append(diagnostics.damage_outside_max)
        crack_history_inside.append(diagnostics.history_inside_max)
        crack_history_outside.append(diagnostics.history_outside_max)
        crack_strain_inside.append(diagnostics.strain_inside_max)
        crack_strain_outside.append(diagnostics.strain_outside_max)
        crack_amor_inside.append(diagnostics.amor_inside_max)
        crack_amor_outside.append(diagnostics.amor_outside_max)
        crack_localization_pass.append(diagnostics.localization_pass)
    return {
        "state_material_displacement": np.stack(
            [
                state.q[: 2 * model.n_material].reshape(model.n_material, 2)
                for state in states
            ]
        ),
        "state_material_particle_strain": strain,
        "state_material_amor_driver": amor,
        "state_physical_interface_opening": np.asarray(
            interface_opening, dtype=np.float64
        ),
        "state_interface_damage_bottom_max": np.asarray(
            interface_damage_bottom, dtype=np.float64
        ),
        "state_interface_damage_top_max": np.asarray(
            interface_damage_top, dtype=np.float64
        ),
        "state_interface_history_bottom_max": np.asarray(
            interface_history_bottom, dtype=np.float64
        ),
        "state_interface_history_top_max": np.asarray(
            interface_history_top, dtype=np.float64
        ),
        "state_crack_particle_strain": crack_strain,
        "state_crack_amor_driver": crack_amor,
        "state_crack_damage_inside_max": np.asarray(crack_damage_inside, dtype=np.float64),
        "state_crack_damage_outside_max": np.asarray(crack_damage_outside, dtype=np.float64),
        "state_crack_history_inside_max": np.asarray(crack_history_inside, dtype=np.float64),
        "state_crack_history_outside_max": np.asarray(crack_history_outside, dtype=np.float64),
        "state_crack_strain_inside_max": np.asarray(crack_strain_inside, dtype=np.float64),
        "state_crack_strain_outside_max": np.asarray(crack_strain_outside, dtype=np.float64),
        "state_crack_amor_inside_max": np.asarray(crack_amor_inside, dtype=np.float64),
        "state_crack_amor_outside_max": np.asarray(crack_amor_outside, dtype=np.float64),
        "state_crack_localization_pass": np.asarray(
            crack_localization_pass, dtype=np.bool_
        ),
    }


def write_engineering_grip_archive(path: Any, result: EngineeringGripResult) -> None:
    """Write a compact numerical-only archive suitable for independent replay."""

    if not isinstance(result, EngineeringGripResult):
        raise TypeError("result must be an EngineeringGripResult")
    result.validate()
    filename = os.fsdecode(os.fspath(path))
    if not filename:
        raise ValueError("archive path must be non-empty")
    parent = os.path.dirname(os.fsdecode(filename))
    if parent:
        os.makedirs(parent, exist_ok=True)
    model = result.model
    geometry = model.geometry
    static = model.static
    states = result.states
    steps = result.accepted_steps
    attempts = result.attempts
    warning_mode = result.acceptance_mode == _WARNING_ACCEPTANCE_MODE
    warning_policy: str | None = None
    warning_limit: float | None = None
    warning_archive_schema = ARCHIVE_SCHEMA
    if warning_mode:
        warning_policy, warning_limit = _validate_warning_safety_configuration(
            result.warning_safety_policy, result.warning_safety_limit
        )
        warning_archive_schema = (
            WARNING_ARCHIVE_SCHEMA
            if warning_policy == _WARNING_SAFETY_POLICY_BOUNDED
            else UNBOUNDED_WARNING_ARCHIVE_SCHEMA
        )
    state_fields = _state_field_arrays(result)
    crack = model.crack_region_mask
    metadata = {
        "schema": warning_archive_schema,
        "raw_source_used": False,
        "material_hash": model.material_hash,
        "union_hash": model.union_hash,
        "static_digest": model.static_digest,
        "code_hash": _current_source_bundle_hash(),
        "code_hash_kind": _SOURCE_BUNDLE_HASH_KIND,
        "code_hash_files": list(_SOURCE_BUNDLE_FILES),
        "youngs_modulus": static.youngs_modulus,
        "poisson_ratio": static.poisson_ratio,
        "k_res": static.k_res,
        "rank_rtol": static.rank_rtol,
        "delta": geometry.delta,
        "h_g": geometry.h_g,
        "Gc": model.Gc,
        "l0": model.l0,
        "visibility_tolerance": geometry.visibility_tolerance,
        "visibility_rule": _VISIBILITY_RULE,
        "crack_region_rule": _CRACK_REGION_RULE,
        "crack_localization_atol": _CRACK_LOCALIZATION_ATOL,
        "crack_interface_rule": _CRACK_INTERFACE_RULE,
        "config": _config_payload(result.config),
    }
    if warning_mode:
        metadata["acceptance_mode"] = _WARNING_ACCEPTANCE_MODE
        if warning_policy == _WARNING_SAFETY_POLICY_BOUNDED:
            assert warning_limit is not None
            metadata["warning_safety_limit"] = warning_limit
        else:
            metadata["warning_safety_policy"] = warning_policy
            metadata["warning_safety_limit_present"] = False
    arrays: dict[str, np.ndarray] = {
        "archive_schema": np.asarray(
            warning_archive_schema
        ),
        "acceptance_mode": np.asarray(result.acceptance_mode),
        "acceptance_status": np.asarray(result.acceptance_status),
        "completion_status": np.asarray(result.completion_status),
        "raw_source_used": np.asarray(False, dtype=np.bool_),
        "material_coordinates": geometry.material_reference_coordinates,
        "material_volumes": geometry.material_reference_volumes,
        "material_triangles": geometry.material_reference_triangles,
        "material_edges": geometry.material_reference_edges,
        "material_bottom_mask": geometry.material_bottom_mask[: model.n_material],
        "material_top_mask": geometry.material_top_mask[: model.n_material],
        "material_anchor": np.asarray(0, dtype=np.int64),
        "union_coordinates": geometry.coordinates,
        "union_material_phase_volumes": geometry.material_volumes,
        "union_grip_phase_volumes": geometry.grip_volumes,
        "union_volumes": geometry.union_volumes,
        "grip_triangles": geometry.grip_triangles,
        "union_triangles": geometry.union_triangles,
        "candidate_edges": geometry.candidate_edges,
        "visible_edges": geometry.directed_edges,
        "removed_edges": geometry.removed_edges,
        "interface_bottom_mask": geometry.interface_bottom_mask,
        "interface_top_mask": geometry.interface_top_mask,
        "volume_bottom_mask": model.constraints.outer_bottom_mask,
        "volume_top_mask": model.constraints.outer_top_mask,
        "constrained_dofs": model.constraints.constrained_dofs,
        "free_dofs": model.constraints.free_dofs,
        "top_line_indices": model.top_line.indices,
        "top_line_weights": model.top_line.weights,
        "top_line_length": np.asarray(model.top_line.length, dtype=np.float64),
        "bottom_line_indices": model.bottom_line.indices,
        "bottom_line_weights": model.bottom_line.weights,
        "bottom_line_length": np.asarray(model.bottom_line.length, dtype=np.float64),
        "crack_region_mask": crack,
        "crack_localization_atol": np.asarray(
            _CRACK_LOCALIZATION_ATOL, dtype=np.float64
        ),
        "state_u_bar": np.asarray([state.u_bar for state in states], dtype=np.float64),
        "state_union_displacement": np.stack([state.q for state in states]),
        "state_material_damage": np.stack([state.d for state in states]),
        "state_material_history": np.stack([state.H for state in states]),
        "state_digest": np.asarray([state.digest for state in states], dtype=np.str_),
        "state_acceptance_status": np.asarray(
            ["initial"] + [step.acceptance_status for step in steps], dtype=np.str_
        ),
        "static_edge_distance": static.edge_distance,
        "static_edge_omega": static.edge_omega,
        "static_family_mass": static.family_mass,
        "static_eta": static.eta,
        "static_support_indices": static.support_indices,
        "static_support_mask": static.support_mask,
        "static_support_sizes": static.support_sizes,
        "static_beta": static.beta,
        "static_xi": static.xi,
        "static_shape_inverse": static.shape_inverse,
        "static_shape_rank": static.shape_rank,
        "static_shape_condition": static.shape_condition,
        "static_gradient_coefficient": static.gradient_coefficient,
        "accepted_load_target": np.asarray([step.load_target for step in steps], dtype=np.float64),
        "accepted_load_start": np.asarray([step.load_start for step in steps], dtype=np.float64),
        "accepted_cutback_count": np.asarray([step.cutback_count for step in steps], dtype=np.int64),
        "accepted_stagger_iterations": np.asarray([step.stagger_iterations for step in steps], dtype=np.int64),
        "accepted_damage_before": np.stack([step.damage_before for step in steps]),
        "accepted_d_iter": np.stack([step.d_iter for step in steps]),
        "accepted_d_next": np.stack([step.d_next for step in steps]),
        "accepted_q_before": np.stack([step.q_before for step in steps]),
        "accepted_q_after": np.stack([step.q_after for step in steps]),
        "accepted_psi_before": np.stack([step.psi_before for step in steps]),
        "accepted_psi_after": np.stack([step.psi_after for step in steps]),
        "accepted_H_used": np.stack([step.H_used for step in steps]),
        "accepted_H_next": np.stack([step.H_next for step in steps]),
        "accepted_rho_used": np.stack([step.rho_used for step in steps]),
        "accepted_rho_next": np.stack([step.rho_next for step in steps]),
        "accepted_rho_used_volume": np.stack([step.rho_used_volume for step in steps]),
        "accepted_rho_next_volume": np.stack([step.rho_next_volume for step in steps]),
        "accepted_energy_before": np.asarray([step.diagnostics.energy_before for step in steps], dtype=np.float64),
        "accepted_energy_after": np.asarray([step.diagnostics.energy_after for step in steps], dtype=np.float64),
        "accepted_top_reaction_before": np.asarray([step.diagnostics.top_reaction_before for step in steps], dtype=np.float64),
        "accepted_top_reaction_after": np.asarray([step.diagnostics.top_reaction_after for step in steps], dtype=np.float64),
        "accepted_bottom_reaction_before": np.asarray([step.diagnostics.bottom_reaction_before for step in steps], dtype=np.float64),
        "accepted_bottom_reaction_after": np.asarray([step.diagnostics.bottom_reaction_after for step in steps], dtype=np.float64),
        "accepted_anchor_reaction_before": np.asarray([step.diagnostics.anchor_reaction_before for step in steps], dtype=np.float64),
        "accepted_anchor_reaction_after": np.asarray([step.diagnostics.anchor_reaction_after for step in steps], dtype=np.float64),
        "accepted_cg_info_before": np.asarray(
            [step.mechanics_before.cg_info for step in steps], dtype=np.int64
        ),
        "accepted_cg_info_after": np.asarray(
            [step.mechanics_after.cg_info for step in steps], dtype=np.int64
        ),
        "accepted_force_before": np.stack(
            [_required_diagnostic_array(step.diagnostics, "force_before") for step in steps]
        ),
        "accepted_force_after": np.stack(
            [_required_diagnostic_array(step.diagnostics, "force_after") for step in steps]
        ),
        "accepted_moment_before": np.asarray([step.diagnostics.moment_before for step in steps], dtype=np.float64),
        "accepted_moment_after": np.asarray([step.diagnostics.moment_after for step in steps], dtype=np.float64),
        "accepted_interface_opening_before": np.asarray(
            [
                compute_interface_opening_diagnostic(
                    step.q_before, geometry, model.top_line, model.bottom_line
                )
                for step in steps
            ],
            dtype=np.float64,
        ),
        "accepted_interface_opening_after": np.asarray(
            [
                compute_interface_opening_diagnostic(
                    step.q_after, geometry, model.top_line, model.bottom_line
                )
                for step in steps
            ],
            dtype=np.float64,
        ),
        "accepted_diagnostics_json": np.asarray(
            [json.dumps(_diagnostic_payload(step.diagnostics), sort_keys=True) for step in steps],
            dtype=np.str_,
        ),
        "attempt_cutback_count": np.asarray([attempt.cutback_count for attempt in attempts], dtype=np.int64),
        "attempt_load_start": np.asarray([attempt.load_start for attempt in attempts], dtype=np.float64),
        "attempt_load_target": np.asarray([attempt.load_target for attempt in attempts], dtype=np.float64),
        "attempt_accepted": np.asarray([attempt.accepted for attempt in attempts], dtype=np.bool_),
        "attempt_before_digest": np.asarray([attempt.committed_before_digest for attempt in attempts], dtype=np.str_),
        "attempt_after_digest": np.asarray([attempt.committed_after_digest for attempt in attempts], dtype=np.str_),
        "attempt_failure_stage": np.asarray([attempt.failure_stage or "" for attempt in attempts], dtype=np.str_),
        "attempt_failure_reason": np.asarray([attempt.failure_reason or "" for attempt in attempts], dtype=np.str_),
        "attempt_exception_type": np.asarray([attempt.exception_type or "" for attempt in attempts], dtype=np.str_),
        "attempt_sweep_count": np.asarray([len(attempt.sweeps) for attempt in attempts], dtype=np.int64),
        "attempt_sweeps_json": np.asarray(
            [
                json.dumps(
                    [_diagnostic_payload(sweep) for sweep in attempt.sweeps],
                    sort_keys=True,
                )
                for attempt in attempts
            ],
            dtype=np.str_,
        ),
        "attempt_diagnostics_json": np.asarray(
            [json.dumps(_diagnostic_payload(attempt.diagnostics), sort_keys=True) for attempt in attempts],
            dtype=np.str_,
        ),
        "work_check_u_bar": np.asarray(result.work_check.u_bar, dtype=np.float64),
        "work_check_epsilon": np.asarray(result.work_check.epsilon, dtype=np.float64),
        "work_check_reaction": np.asarray(result.work_check.reaction, dtype=np.float64),
        "work_check_reaction_fd": np.asarray(result.work_check.reaction_fd, dtype=np.float64),
        "work_check_relative_error": np.asarray(result.work_check.relative_error, dtype=np.float64),
        "work_check_relative_limit": np.asarray(result.work_check.relative_limit, dtype=np.float64),
        "work_check_passed": np.asarray(result.work_check.passed, dtype=np.bool_),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **state_fields,
    }
    if warning_mode:
        # Independent warning provenance: per-step and per-attempt acceptance
        # statuses, canonical bypass masks/lists, raw final-sweep excesses,
        # and the explicit safety limit.  Every warning archive is durable
        # evidence only and can never report validated=True.
        gate_count = len(WARNING_GATE_NAMES)
        arrays.update(
            {
                "validated": np.asarray(False, dtype=np.bool_),
                "warning_gate_names": np.asarray(WARNING_GATE_NAMES, dtype=np.str_),
                "accepted_acceptance_status": np.asarray(
                    [step.acceptance_status for step in steps], dtype=np.str_
                ),
                "accepted_warning": np.asarray(
                    [
                        step.acceptance_status == _WARNING_ACCEPTANCE_STATUS
                        for step in steps
                    ],
                    dtype=np.bool_,
                ),
                "accepted_bypassed_gate_mask": np.asarray(
                    [
                        [name in set(step.bypassed_gates) for name in WARNING_GATE_NAMES]
                        for step in steps
                    ],
                    dtype=np.bool_,
                ).reshape(len(steps), gate_count),
                "accepted_warning_range_excess": np.asarray(
                    [
                        0.0 if step.warning_range_excess is None else step.warning_range_excess
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
                "attempt_acceptance_status": np.asarray(
                    [attempt.acceptance_status or "" for attempt in attempts],
                    dtype=np.str_,
                ),
                "attempt_bypassed_gate_mask": np.asarray(
                    [
                        [name in set(attempt.bypassed_gates) for name in WARNING_GATE_NAMES]
                        for attempt in attempts
                    ],
                    dtype=np.bool_,
                ).reshape(len(attempts), gate_count),
                "attempt_warning_excess_present": np.asarray(
                    [
                        attempt.warning_range_excess is not None
                        and attempt.warning_healing_magnitude is not None
                        for attempt in attempts
                    ],
                    dtype=np.bool_,
                ),
                "attempt_warning_range_excess": np.asarray(
                    [
                        0.0 if attempt.warning_range_excess is None else attempt.warning_range_excess
                        for attempt in attempts
                    ],
                    dtype=np.float64,
                ),
                "attempt_warning_healing_magnitude": np.asarray(
                    [
                        0.0
                        if attempt.warning_healing_magnitude is None
                        else attempt.warning_healing_magnitude
                        for attempt in attempts
                    ],
                    dtype=np.float64,
                ),
            }
        )
        if warning_policy == _WARNING_SAFETY_POLICY_BOUNDED:
            assert warning_limit is not None
            arrays["warning_safety_limit"] = np.asarray(
                warning_limit, dtype=np.float64
            )
        else:
            arrays["warning_safety_policy"] = np.asarray(
                warning_policy, dtype=np.str_
            )
            arrays["warning_safety_limit_present"] = np.asarray(
                False, dtype=np.bool_
            )
    for name, value in arrays.items():
        if np.asarray(value).dtype.kind == "O":
            raise ValueError(f"engineering-grip archive field {name} has object dtype")
    archive_fields: dict[str, Any] = dict(arrays)
    np.savez_compressed(file=filename, **archive_fields)


def load_engineering_grip_archive(path: Any) -> dict[str, np.ndarray]:
    """Load the numerical-only archive with pickle explicitly disabled."""

    filename = os.fspath(path)
    with np.load(filename, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name, value in result.items():
        if value.dtype.kind == "O":
            raise ValueError(f"engineering-grip archive field {name} has forbidden object dtype")
    return result


def _replay_ba(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    edges: np.ndarray,
    delta: float,
    rank_rtol: float,
) -> dict[str, Any]:
    """Rebuild union BA supports independently of all production objects."""

    x = _f64(coordinates, "replay union coordinates")
    v = _f64(volumes, "replay union volumes", (len(x),))
    e = _i64(edges, "replay union edges")
    if e.ndim != 2 or e.shape[1] != 2 or len(e) == 0:
        raise ValueError("replay union edges must have nonempty shape (E, 2)")
    if np.any(e < 0) or np.any(e >= len(x)) or np.any(e[:, 0] == e[:, 1]):
        raise ValueError("replay union edges contain invalid indices")
    if len(np.unique(e, axis=0)) != len(e):
        raise ValueError("replay union edges contain duplicate directed edges")
    if not np.array_equal(e, e[np.lexsort((e[:, 1], e[:, 0]))]):
        raise ValueError("replay union edges are not lexicographically sorted")
    edge_set = {tuple(edge) for edge in e.tolist()}
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("replay union edges are not reciprocal")
    horizon = _scalar(delta, "replay delta")
    rank_limit = _scalar(rank_rtol, "replay rank_rtol")
    distance = np.linalg.norm(x[e[:, 1]] - x[e[:, 0]], axis=1)
    if np.any(distance <= 0.0) or np.any(distance >= horizon):
        raise ValueError("replay union edges violate the strict horizon")
    omega = 120.0 / (np.pi * horizon**4) * (1.0 - distance / horizon) ** 2
    source = e[:, 0]
    target = e[:, 1]
    family = np.bincount(
        source, weights=omega * v[target], minlength=len(x)
    ).astype(np.float64, copy=False)
    if np.any(family <= 0.0) or not np.all(np.isfinite(family)):
        raise ValueError("replay union family masses are invalid")
    adjacency: list[set[int]] = [set() for _ in range(len(x))]
    omega_map: dict[tuple[int, int], float] = {}
    for (i_raw, j_raw), value in zip(e, omega):
        i, j = int(i_raw), int(j_raw)
        adjacency[i].add(j)
        omega_map[i, j] = float(value)
    supports: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    xis: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    identity_endpoint_geometry_scales: list[np.ndarray] = []
    ranks: list[int] = []
    conditions: list[float] = []
    for i_raw, j_raw in e:
        i, j = int(i_raw), int(j_raw)
        common = sorted(adjacency[i].intersection(adjacency[j]) - {i, j})
        support = np.asarray([j, *common], dtype=np.int64)
        beta = np.asarray(
            [omega_map[i, j] * v[j]]
            + [omega_map[i, k] * v[k] for k in common],
            dtype=np.float64,
        )
        xi = x[support] - x[i]
        shape = np.einsum("s,sa,sb->ab", beta, xi, xi, optimize=True)
        singular = np.linalg.svd(shape, compute_uv=False)
        rank = int(np.count_nonzero(singular > rank_limit * singular[0]))
        if rank < 2:
            raise ValueError(f"replay BA support is rank deficient at ({i}, {j})")
        condition = float(singular[0] / singular[-1])
        if not np.isfinite(condition):
            raise ValueError("replay BA condition number is non-finite")
        inverse = np.linalg.inv(shape)
        # The replay deformation gradient subtracts the identity before
        # constitutive evaluation.  Retain a compact geometric scale for that
        # cancellation so repeated response evaluations need not rebuild the
        # same matrix bounds.
        shape_constituent_scale = np.einsum(
            "s,sa,sb->ab", np.abs(beta), np.abs(xi), np.abs(xi), optimize=True
        )
        deformation_scale = shape_constituent_scale @ np.abs(inverse)
        deformation_scale += np.abs(_IDENTITY_2) + _IDENTITY_2
        deformation_scale = float(np.max(np.abs(deformation_scale)))
        identity_endpoint_geometry_scales.append(
            np.asarray(
                [
                    beta[local]
                    * deformation_scale
                    * (np.abs(inverse.T) @ np.abs(xi[local]))
                    for local in range(len(support))
                ],
                dtype=np.float64,
            )
        )
        supports.append(support)
        betas.append(beta)
        xis.append(xi)
        inverses.append(inverse)
        ranks.append(rank)
        conditions.append(condition)
    eta = omega * v[target] / family[source]
    if np.any(eta <= 0.0) or not np.all(np.isfinite(eta)):
        raise ValueError("replay union eta is invalid")
    return {
        "coordinates": x,
        "volumes": v,
        "edges": e,
        "source": source,
        "target": target,
        "distance": distance,
        "omega": omega,
        "family": family,
        "supports": tuple(supports),
        "beta": tuple(betas),
        "xi": tuple(xis),
        "inverse": tuple(inverses),
        "identity_endpoint_geometry_scale": tuple(identity_endpoint_geometry_scales),
        "eta": eta,
        "shape_rank": np.asarray(ranks, dtype=np.int64),
        "shape_condition": np.asarray(conditions, dtype=np.float64),
    }


def _padded_replay_supports(
    ba: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = ba.get("_padded_supports")
    if cached is not None:
        if not isinstance(cached, tuple) or len(cached) != 4:
            raise ValueError("replay padded support cache is malformed")
        return cached
    count = len(ba["supports"])
    width = max(len(value) for value in ba["supports"])
    indices = np.full((count, width), -1, dtype=np.int64)
    mask = np.zeros((count, width), dtype=bool)
    beta = np.zeros((count, width), dtype=np.float64)
    xi = np.zeros((count, width, 2), dtype=np.float64)
    for row, (support, values, offsets) in enumerate(
        zip(ba["supports"], ba["beta"], ba["xi"])
    ):
        size = len(support)
        indices[row, :size] = support
        mask[row, :size] = True
        beta[row, :size] = values
        xi[row, :size] = offsets
    result = indices, mask, beta, xi
    if isinstance(ba, dict):
        ba["_padded_supports"] = result
    return result


def _replay_vectorized_geometry(
    ba: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed padded support data for matrix-free replay actions.

    The replay archive retains variable-length supports for auditability.  CG
    actions use one padded, masked representation so the edge/support algebra
    is evaluated by NumPy reductions rather than Python loops.  The cache is
    derived only from the independently rebuilt ``ba`` mapping and contains no
    global mechanics matrix.
    """

    cached = ba.get("_vectorized_geometry")
    if cached is not None:
        if not isinstance(cached, tuple) or len(cached) != 5:
            raise ValueError("replay vectorized geometry cache is malformed")
        return cached
    support_indices, support_mask, beta, xi = _padded_replay_supports(ba)
    safe_support = np.where(support_mask, support_indices, 0).astype(
        np.int64, copy=False
    )
    inverse = _f64(
        np.asarray(ba["inverse"], dtype=np.float64),
        "replay shape inverse",
        (len(ba["source"]), 2, 2),
    )
    gradient_coefficient = np.asarray(
        np.einsum(
            "es,esb,ebc->esc", beta, xi, inverse, optimize=True
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(gradient_coefficient)):
        raise FloatingPointError("replay gradient coefficient is non-finite")
    result = safe_support, beta, xi, inverse, gradient_coefficient
    if isinstance(ba, dict):
        ba["_vectorized_geometry"] = result
    return result


def _replay_padded_identity_endpoint_scale(
    ba: Mapping[str, Any], support_shape: tuple[int, int, int]
) -> np.ndarray:
    """Pad the one-time identity-roundoff bounds to the support shape."""

    cached = ba.get("_padded_identity_endpoint_scale")
    if cached is not None:
        return _f64(
            cached,
            "replay padded identity endpoint scale",
            support_shape,
        )
    raw = ba["identity_endpoint_geometry_scale"]
    if len(raw) != support_shape[0]:
        raise ValueError("replay identity endpoint scale has the wrong edge count")
    padded = np.zeros(support_shape, dtype=np.float64)
    for edge_index, values in enumerate(raw):
        values_array = _f64(
            values,
            "replay identity endpoint geometry scale",
        )
        if values_array.ndim != 2 or values_array.shape[1:] != (2,) or len(values_array) > support_shape[1]:
            raise ValueError("replay identity endpoint geometry scale has invalid support size")
        padded[edge_index, : len(values_array), :] = values_array
    if not np.all(np.isfinite(padded)) or np.any(padded < 0.0):
        raise ValueError("replay identity endpoint geometry scale is invalid")
    if isinstance(ba, dict):
        ba["_padded_identity_endpoint_scale"] = padded
    return padded


def _replay_center_weights(
    damage: np.ndarray,
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    k_res: float,
    n_material: int,
) -> np.ndarray:
    d = _f64(damage, "replay damage", (n_material,))
    g = np.pad(
        (1.0 - d) ** 2 + _scalar(k_res, "replay k_res"),
        (0, len(material_phase) - n_material),
    )
    weights = material_phase * g + grip_phase * (1.0 + _scalar(k_res, "replay k_res"))
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("replay center weights are invalid")
    return np.asarray(weights, dtype=np.float64)


def _replay_response(
    q: np.ndarray,
    damage: np.ndarray,
    ba: Mapping[str, Any],
    center_weights: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    n_material: int,
) -> dict[str, Any]:
    values = _f64(q, "replay displacement", (2 * len(ba["coordinates"]),)).reshape(-1, 2)
    weights = _f64(center_weights, "replay center weights", (len(values),))
    d = _f64(damage, "replay response damage", (n_material,))
    if not np.all(np.isfinite(values)):
        raise ValueError("replay displacement is non-finite")
    E = _scalar(youngs_modulus, "replay youngs_modulus")
    nu = _scalar(poisson_ratio, "replay poisson_ratio")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    safe_support, beta, xi, inverse, _ = _replay_vectorized_geometry(ba)
    source = _i64(
        np.asarray(ba["source"]),
        "replay source",
        (len(ba["source"]),),
    )
    eta = _f64(np.asarray(ba["eta"]), "replay eta", (len(source),))
    if np.any(source < 0) or np.any(source >= len(values)):
        raise ValueError("replay source contains an invalid particle index")

    # The zero state is an analytic state of the declared energy.  In
    # particular, do not manufacture a nonzero strain by evaluating the
    # rounded identity ``sum(beta*xi*xi) @ inverse``.
    if np.array_equal(values, np.zeros_like(values)):
        edge_count = len(source)
        zero_edges = np.zeros((edge_count, 2, 2), dtype=np.float64)
        return {
            "energy": 0.0,
            "energy_constituent_scale": 0.0,
            "residual": np.zeros(values.size, dtype=np.float64),
            "residual_reduction_scale": np.zeros(values.size, dtype=np.float64),
            "residual_constituent_scale": np.zeros(values.size, dtype=np.float64),
            "edge_f": np.broadcast_to(_IDENTITY_2, (edge_count, 2, 2)).copy(),
            "edge_strain": zero_edges,
            "particle_strain": np.zeros((len(values), 2, 2), dtype=np.float64),
        }

    identity_stress_scale = 2.0 * (abs(lam) + abs(mu))
    source_values = values[source]
    current = xi + values[safe_support] - source_values[:, None, :]
    moment = np.einsum("es,esa,esb->eab", beta, current, xi, optimize=True)
    edge_f = np.einsum("eab,ebc->eac", moment, inverse, optimize=True)
    increment = edge_f - _IDENTITY_2[None, :, :]
    edge_strain = 0.5 * (increment + np.swapaxes(increment, 1, 2))
    trace = np.trace(edge_strain, axis1=1, axis2=2)
    stress = (
        lam * trace[:, None, None] * _IDENTITY_2[None, :, :]
        + 2.0 * mu * edge_strain
    )
    density = 0.5 * lam * trace**2 + mu * np.sum(
        edge_strain * edge_strain, axis=(1, 2)
    )
    factor = weights[source] * eta
    energy_terms = factor * density
    identity_endpoint_scale = _replay_padded_identity_endpoint_scale(
        ba, (beta.shape[0], beta.shape[1], 2)
    )
    endpoint = np.einsum(
        "eab,ebc,esc->esa",
        stress,
        np.swapaxes(inverse, 1, 2),
        xi,
        optimize=True,
    )
    endpoint *= beta[:, :, None] * factor[:, None, None]
    contribution_magnitude = np.abs(endpoint)
    endpoint_scale = (
        factor[:, None, None]
        * identity_stress_scale
        * identity_endpoint_scale
    )

    def accumulate_support(edge_values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                np.bincount(
                    safe_support.reshape(-1),
                    weights=edge_values[:, :, 0].reshape(-1),
                    minlength=len(values),
                ),
                np.bincount(
                    safe_support.reshape(-1),
                    weights=edge_values[:, :, 1].reshape(-1),
                    minlength=len(values),
                ),
            )
        )

    def accumulate_source(edge_values: np.ndarray, sign: float) -> np.ndarray:
        return np.column_stack(
            (
                np.bincount(
                    source,
                    weights=sign * np.sum(edge_values[:, :, 0], axis=1),
                    minlength=len(values),
                ),
                np.bincount(
                    source,
                    weights=sign * np.sum(edge_values[:, :, 1], axis=1),
                    minlength=len(values),
                ),
            )
        )

    residual = (accumulate_support(endpoint) + accumulate_source(endpoint, -1.0)).reshape(-1)
    residual_reduction_scale = (
        accumulate_support(contribution_magnitude)
        + accumulate_source(contribution_magnitude, 1.0)
    ).reshape(-1)
    residual_constituent_scale = (
        accumulate_support(contribution_magnitude + endpoint_scale)
        + accumulate_source(contribution_magnitude + endpoint_scale, 1.0)
    ).reshape(-1)
    particle_strain = np.zeros((len(values), 2, 2), dtype=np.float64)
    np.add.at(
        particle_strain,
        source,
        eta[:, None, None] * edge_strain,
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (
            edge_f,
            edge_strain,
            stress,
            density,
            residual,
            residual_reduction_scale,
            residual_constituent_scale,
            particle_strain,
        )
    ) or not np.isfinite(energy_terms).all():
        raise FloatingPointError("replay mechanics response is non-finite")
    return {
        "energy": float(np.sum(energy_terms)),
        "energy_constituent_scale": float(np.sum(np.abs(energy_terms))),
        "residual": residual,
        "residual_reduction_scale": residual_reduction_scale,
        "residual_constituent_scale": residual_constituent_scale,
        "edge_f": np.asarray(edge_f, dtype=np.float64),
        "edge_strain": np.asarray(edge_strain, dtype=np.float64),
        "particle_strain": particle_strain,
    }


def _replay_mechanics_at_state(
    q: np.ndarray,
    damage: np.ndarray,
    load: float,
    ba: Mapping[str, Any],
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    volume_bottom_mask: np.ndarray,
    volume_top_mask: np.ndarray,
    free_dofs: np.ndarray,
    anchor_particle: int,
    youngs_modulus: float,
    poisson_ratio: float,
    k_res: float,
    n_material: int,
    config: EngineeringGripConfig,
) -> dict[str, Any]:
    """Evaluate stored mechanics primitives without replacing the stored q."""

    coordinates = ba["coordinates"]
    n_union = len(coordinates)
    values = _f64(q, "replay stored displacement", (2 * n_union,))
    d = _f64(damage, "replay stored damage", (n_material,))
    target = _scalar(load, "replay stored load")
    center = _replay_center_weights(
        d, material_phase, grip_phase, k_res, n_material
    )
    top_dofs = 2 * np.flatnonzero(volume_top_mask) + 1
    bottom_dofs = 2 * np.flatnonzero(volume_bottom_mask) + 1
    constrained = np.sort(
        np.concatenate(
            (
                bottom_dofs,
                top_dofs,
                np.asarray([2 * anchor_particle], dtype=np.int64),
            )
        )
    )
    prescribed = np.zeros(2 * n_union, dtype=np.float64)
    prescribed[top_dofs] = target
    if not np.array_equal(values[constrained], prescribed[constrained]):
        raise ValueError("replay stored displacement violates hard constraints")

    response = _replay_response(
        values,
        d,
        ba,
        center,
        youngs_modulus,
        poisson_ratio,
        n_material,
    )
    prescribed_response = _replay_response(
        prescribed,
        d,
        ba,
        center,
        youngs_modulus,
        poisson_ratio,
        n_material,
    )
    residual = np.asarray(response["residual"], dtype=np.float64)
    rhs = -np.asarray(prescribed_response["residual"], dtype=np.float64)[free_dofs]
    free_residual = residual[free_dofs]
    free_norm = float(np.linalg.norm(free_residual))
    rhs_norm = float(np.linalg.norm(rhs))
    free_relative = free_norm / max(rhs_norm, 1.0e-300)
    cg_limit = max(config.cg_atol, config.cg_rtol * rhs_norm)
    residual_matrix = residual.reshape(-1, 2)
    residual_constituent_scale = _f64(
        response["residual_constituent_scale"],
        "replay residual constituent scale",
        residual.shape,
    )
    residual_reduction_scale = _f64(
        response["residual_reduction_scale"],
        "replay residual reduction scale",
        residual.shape,
    )
    prescribed_constituent_scale = _f64(
        prescribed_response["residual_constituent_scale"],
        "replay prescribed residual constituent scale",
        residual.shape,
    )
    prescribed_reduction_scale = _f64(
        prescribed_response["residual_reduction_scale"],
        "replay prescribed residual reduction scale",
        residual.shape,
    )
    free_norm_constituent_scale = float(
        np.linalg.norm(residual_constituent_scale[free_dofs])
    )
    rhs_norm_constituent_scale = float(
        np.linalg.norm(prescribed_constituent_scale[free_dofs])
    )
    relative_roundoff_atol = _replay_ratio_atol(
        free_norm,
        rhs_norm,
        free_norm_constituent_scale,
        rhs_norm_constituent_scale,
        "replay free residual relative norm",
    )
    force = np.asarray(np.sum(residual_matrix, axis=0), dtype=np.float64)
    moment = float(
        np.sum(
            coordinates[:, 0] * residual_matrix[:, 1]
            - coordinates[:, 1] * residual_matrix[:, 0]
        )
    )
    top = float(np.sum(residual[top_dofs]))
    bottom = float(np.sum(residual[bottom_dofs]))
    anchor = float(residual[2 * anchor_particle])
    force_constituent_scale = np.asarray(
        np.sum(residual_constituent_scale.reshape(-1, 2), axis=0),
        dtype=np.float64,
    )
    moment_constituent_scale = float(
        np.sum(
            np.abs(coordinates[:, 0]) * residual_constituent_scale[1::2]
            + np.abs(coordinates[:, 1]) * residual_constituent_scale[0::2]
        )
    )
    top_constituent_scale = float(np.sum(residual_constituent_scale[top_dofs]))
    bottom_constituent_scale = float(np.sum(residual_constituent_scale[bottom_dofs]))
    anchor_constituent_scale = float(residual_constituent_scale[2 * anchor_particle])
    finite = bool(
        np.all(np.isfinite(values))
        and np.all(np.isfinite(residual))
        and np.all(np.isfinite(force))
        and np.isfinite(moment)
        and np.isfinite(free_norm)
        and np.isfinite(free_relative)
        and np.isfinite(rhs_norm)
        and np.isfinite(cg_limit)
    )
    mechanics_pass = bool(finite and free_norm < cg_limit)
    force_error = _max_abs(force)
    force_scale = max(abs(top), abs(bottom), abs(anchor), 1.0)
    force_limit = config.force.limit(force_scale)
    force_pass = bool(finite and force_error <= force_limit)
    moment_error = abs(moment)
    geometry_length = max(
        _max_abs(coordinates), _max_abs(np.ptp(coordinates, axis=0)), 1.0
    )
    moment_scale = force_scale * geometry_length
    moment_limit = config.moment.limit(moment_scale)
    moment_pass = bool(finite and moment_error <= moment_limit)
    return {
        "center": center,
        "response": response,
        "prescribed_response": prescribed_response,
        "energy_constituent_scale": response["energy_constituent_scale"],
        "top_constituent_scale": top_constituent_scale,
        "bottom_constituent_scale": bottom_constituent_scale,
        "anchor_constituent_scale": anchor_constituent_scale,
        "free_residual_norm_constituent_scale": free_norm_constituent_scale,
        "free_rhs_norm_constituent_scale": rhs_norm_constituent_scale,
        "free_residual_relative_roundoff_atol": relative_roundoff_atol,
        "force_constituent_scale": force_constituent_scale,
        "moment_constituent_scale": moment_constituent_scale,
        "residual": residual,
        "free_residual": np.asarray(free_residual, dtype=np.float64),
        "free_rhs_norm": rhs_norm,
        "free_residual_norm": free_norm,
        "free_residual_relative_norm": free_relative,
        "cg_convergence_limit": cg_limit,
        "energy": float(response["energy"]),
        "top": top,
        "bottom": bottom,
        "anchor": anchor,
        "force": force,
        "moment": moment,
        "finite": finite,
        "mechanics_pass": mechanics_pass,
        "force_error": force_error,
        "force_scale": force_scale,
        "force_limit": force_limit,
        "force_pass": force_pass,
        "moment_error": moment_error,
        "moment_scale": moment_scale,
        "moment_limit": moment_limit,
        "moment_pass": moment_pass,
    }


def _replay_amor(
    q: np.ndarray,
    previous_history: np.ndarray,
    ba: Mapping[str, Any],
    youngs_modulus: float,
    poisson_ratio: float,
    n_material: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous = _f64(previous_history, "replay previous history", (n_material,))
    if np.any(previous < 0.0):
        raise ValueError("replay previous history is negative")
    values = _f64(
        q,
        "replay Amor displacement",
        (2 * len(ba["coordinates"]),),
    )
    if np.array_equal(values, np.zeros_like(values)):
        return (
            np.zeros(n_material, dtype=np.float64),
            np.array(previous, dtype=np.float64, copy=True),
            np.zeros((n_material, 2, 2), dtype=np.float64),
        )
    response = _replay_response(
        values,
        np.zeros(n_material, dtype=np.float64),
        ba,
        np.ones(len(ba["coordinates"]), dtype=np.float64),
        youngs_modulus,
        poisson_ratio,
        n_material,
    )
    strain = np.asarray(response["edge_strain"], dtype=np.float64)
    E = _scalar(youngs_modulus, "replay youngs_modulus")
    nu = _scalar(poisson_ratio, "replay poisson_ratio")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    embedded = np.zeros((len(strain), 3, 3), dtype=np.float64)
    embedded[:, :2, :2] = strain
    trace = np.trace(embedded, axis1=1, axis2=2)
    deviatoric = embedded - (trace / 3.0)[:, None, None] * _IDENTITY_3
    edge_positive = 0.5 * (lam + 2.0 * mu / 3.0) * np.maximum(trace, 0.0) ** 2 + mu * np.sum(
        deviatoric * deviatoric, axis=(1, 2)
    )
    retained = ba["source"] < n_material
    if not np.any(retained):
        raise ValueError("replay Amor has no material source centers")
    eta_sum = np.bincount(
        ba["source"][retained],
        weights=ba["eta"][retained],
        minlength=n_material,
    )
    if np.max(np.abs(eta_sum - 1.0)) > _REPLAY_BOUND:
        raise ValueError("replay Amor material-source eta is not normalized")
    driver = np.zeros(n_material, dtype=np.float64)
    np.add.at(
        driver,
        ba["source"][retained],
        ba["eta"][retained] * edge_positive[retained],
    )
    return (
        np.asarray(driver, dtype=np.float64),
        np.asarray(np.maximum(previous, driver), dtype=np.float64),
        np.asarray(response["particle_strain"][:n_material], dtype=np.float64),
    )


def _replay_damage(
    H_used: np.ndarray,
    material_coordinates: np.ndarray,
    material_volumes: np.ndarray,
    material_edges: np.ndarray,
    delta: float,
    Gc: float,
    l0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, float]:
    coordinates = _f64(material_coordinates, "replay material coordinates")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("replay material coordinates must have shape (N, 2)")
    volumes = _f64(
        material_volumes,
        "replay material volumes",
        (len(coordinates),),
    )
    if np.any(volumes <= 0.0):
        raise ValueError("replay material volumes must be strictly positive")
    edges = _i64(material_edges, "replay material edges")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("replay material edges must have nonempty shape (E, 2)")
    if np.any(edges < 0) or np.any(edges >= len(coordinates)):
        raise ValueError("replay material edges contain invalid indices")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("replay material edges contain self edges")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("replay material edges contain duplicate directed edges")
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("replay material edges are not reciprocal")
    horizon = _scalar(delta, "replay material delta")
    if horizon <= 0.0:
        raise ValueError("replay material delta must be positive")
    gc_value = _scalar(Gc, "replay Gc")
    l0_value = _scalar(l0, "replay l0")
    if gc_value <= 0.0 or l0_value <= 0.0:
        raise ValueError("replay Gc/l0 must be positive")
    n = len(coordinates)
    H = _f64(H_used, "replay H_used", (n,))
    if np.any(H < 0.0):
        raise ValueError("replay H_used must be non-negative")
    row = edges[:, 0]
    col = edges[:, 1]
    outgoing_count = np.bincount(row, minlength=n)
    if np.any(outgoing_count <= 0):
        missing = np.flatnonzero(outgoing_count <= 0)
        raise ValueError(
            "replay material graph has no positive outgoing family for particles "
            f"{missing[:10].tolist()}"
        )
    distance = np.linalg.norm(coordinates[col] - coordinates[row], axis=1)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0) or np.any(
        distance >= horizon
    ):
        raise ValueError("replay material edges violate the strict horizon")
    kernel = 120.0 / (np.pi * horizon**4) * (1.0 - distance / horizon) ** 2
    coefficient = np.asarray(kernel * volumes[col], dtype=np.float64)
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient <= 0.0):
        raise ValueError("replay material graph coefficients are invalid")

    # This is the same directed graph Laplacian as the material SparseD, but
    # assembled independently as CSR.  No global dense operator is formed.
    diagonal = np.bincount(row, weights=coefficient, minlength=n)
    matrix = (
        diags(diagonal, offsets=0, shape=(n, n), format="csr")
        + coo_matrix(
            (-coefficient, (row, col)), shape=(n, n), dtype=np.float64
        ).tocsr()
    ).tocsr()
    matrix.sort_indices()
    if matrix.nnz and not np.all(np.isfinite(matrix.data)):
        raise FloatingPointError("replay material graph operator is non-finite")
    strong = (
        diags(
            gc_value / l0_value + 2.0 * H,
            offsets=0,
            shape=(n, n),
            format="csr",
        )
        + (gc_value * l0_value) * matrix
    ).tocsr()
    system = (
        diags(volumes, offsets=0, shape=(n, n), format="csr") @ strong
    ).tocsr()
    rhs = np.asarray(2.0 * volumes * H, dtype=np.float64)
    if not np.all(np.isfinite(rhs)) or (
        system.nnz and not np.all(np.isfinite(system.data))
    ):
        raise FloatingPointError("replay material damage system is non-finite")
    skew = system - system.T
    symmetry_defect = float(np.max(np.abs(skew.data))) if skew.nnz else 0.0
    system_scale = float(np.max(np.abs(system.data))) if system.nnz else 0.0
    if symmetry_defect > 2.0e-12 * max(system_scale, 1.0):
        raise ValueError("replay material damage system is not symmetric")
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            damage = np.asarray(spsolve(system, rhs), dtype=np.float64)
        except (MatrixRankWarning, RuntimeError, np.linalg.LinAlgError) as error:
            raise RuntimeError(
                "replay material damage sparse direct solve failed or is singular"
            ) from error
    if damage.shape != (n,) or not np.all(np.isfinite(damage)):
        raise FloatingPointError("replay material damage solve returned non-finite values")
    rho = np.asarray(strong @ damage - 2.0 * H, dtype=np.float64)
    volume_rho = np.asarray(volumes * rho, dtype=np.float64)
    scale = max(
        _max_abs(gc_value / l0_value * damage),
        _max_abs(gc_value * l0_value * (matrix @ damage)),
        _max_abs(2.0 * H * damage),
        _max_abs(2.0 * H),
    )
    if not all(np.all(np.isfinite(value)) for value in (rho, volume_rho)):
        raise FloatingPointError("replay material damage residual is non-finite")
    return damage, rho, volume_rho, matrix, scale


def _replay_damage_residual_scale(
    damage: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
    matrix: Any,
    Gc: float,
    l0: float,
) -> float:
    """Rebuild the six-term scale used by the strict damage-residual gate."""

    d = _f64(damage, "replay residual-scale damage")
    used = _f64(H_used, "replay residual-scale H_used", d.shape)
    next_history = _f64(H_next, "replay residual-scale H_next", d.shape)
    if not issparse(matrix):
        raise TypeError("replay residual-scale matrix must be sparse")
    M = matrix.tocsr()
    if M.shape != (len(d), len(d)) or (
        M.nnz and not np.all(np.isfinite(M.data))
    ):
        raise ValueError("replay residual-scale matrix has an invalid shape/data")
    terms = (
        _scalar(Gc, "replay residual-scale Gc") / _scalar(l0, "replay residual-scale l0") * d,
        _scalar(Gc, "replay residual-scale Gc")
        * _scalar(l0, "replay residual-scale l0")
        * (M @ d),
        2.0 * used * d,
        2.0 * used,
        2.0 * next_history * d,
        2.0 * next_history,
    )
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise ValueError("replay damage residual scale is non-finite")
    return max((_max_abs(term) for term in terms), default=0.0)


def _replay_tangent_action(
    direction: np.ndarray,
    ba: Mapping[str, Any],
    center_weights: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
) -> np.ndarray:
    values = _f64(
        direction,
        "replay tangent direction",
        (2 * len(ba["coordinates"]),),
    ).reshape(-1, 2)
    E = _scalar(youngs_modulus, "replay youngs_modulus")
    nu = _scalar(poisson_ratio, "replay poisson_ratio")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    weights = _f64(center_weights, "replay tangent center weights", (len(values),))
    safe_support, beta, xi, inverse, gradient_coefficient = _replay_vectorized_geometry(
        ba
    )
    source = _i64(
        np.asarray(ba["source"]),
        "replay tangent source",
        (len(ba["source"]),),
    )
    eta = _f64(np.asarray(ba["eta"]), "replay tangent eta", (len(source),))
    if np.any(source < 0) or np.any(source >= len(values)):
        raise ValueError("replay tangent source contains an invalid index")
    differences = values[safe_support] - values[source, None, :]
    delta_f = np.asarray(
        np.einsum(
            "esa,esc->eac", differences, gradient_coefficient, optimize=True
        ),
        dtype=np.float64,
    )
    delta_strain = 0.5 * (delta_f + np.swapaxes(delta_f, 1, 2))
    trace = np.trace(delta_strain, axis1=1, axis2=2)
    delta_stress = (
        lam * trace[:, None, None] * _IDENTITY_2[None, :, :]
        + 2.0 * mu * delta_strain
    )
    factor = weights[source] * eta
    endpoint = np.einsum(
        "eab,ebc,esc->esa",
        delta_stress,
        np.swapaxes(inverse, 1, 2),
        xi,
        optimize=True,
    )
    endpoint *= beta[:, :, None] * factor[:, None, None]
    support_result = np.column_stack(
        (
            np.bincount(
                safe_support.reshape(-1),
                weights=endpoint[:, :, 0].reshape(-1),
                minlength=len(values),
            ),
            np.bincount(
                safe_support.reshape(-1),
                weights=endpoint[:, :, 1].reshape(-1),
                minlength=len(values),
            ),
        )
    )
    source_result = np.column_stack(
        (
            np.bincount(
                source,
                weights=-np.sum(endpoint[:, :, 0], axis=1),
                minlength=len(values),
            ),
            np.bincount(
                source,
                weights=-np.sum(endpoint[:, :, 1], axis=1),
                minlength=len(values),
            ),
        )
    )
    result = (support_result + source_result).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("replay tangent action is non-finite")
    return np.asarray(result, dtype=np.float64)


def _replay_tangent_block_diagonal(
    ba: Mapping[str, Any],
    center_weights: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
) -> np.ndarray:
    """Build only the particle-local blocks of the replay Hessian.

    These blocks are a preconditioner, not a replacement for the full
    matrix-free tangent.  They are the diagonal particle blocks of the same
    ``B^* C B`` action used by ``_replay_tangent_action`` and therefore scale
    with particles/supports rather than with a dense global DOF square.
    """

    weights = _f64(center_weights, "replay preconditioner center weights", (len(ba["coordinates"]),))
    E = _scalar(youngs_modulus, "replay youngs_modulus")
    nu = _scalar(poisson_ratio, "replay poisson_ratio")
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    safe_support, _, _, _, gradient_coefficient = _replay_vectorized_geometry(ba)
    source = _i64(
        np.asarray(ba["source"]),
        "replay preconditioner source",
        (len(ba["source"]),),
    )
    eta = _f64(
        np.asarray(ba["eta"]),
        "replay preconditioner eta",
        (len(source),),
    )
    factor = weights[source] * eta
    cx = gradient_coefficient[:, :, 0]
    cy = gradient_coefficient[:, :, 1]
    hxx = factor[:, None] * (
        (lam + 2.0 * mu) * cx * cx + mu * cy * cy
    )
    hxy = factor[:, None] * (lam + mu) * cx * cy
    hyy = factor[:, None] * (
        mu * cx * cx + (lam + 2.0 * mu) * cy * cy
    )
    support_flat = safe_support.reshape(-1)
    n_particles = len(weights)
    blocks = np.zeros((n_particles, 2, 2), dtype=np.float64)
    blocks[:, 0, 0] = np.bincount(
        support_flat, weights=hxx.reshape(-1), minlength=n_particles
    )
    blocks[:, 0, 1] = np.bincount(
        support_flat, weights=hxy.reshape(-1), minlength=n_particles
    )
    blocks[:, 1, 0] = blocks[:, 0, 1]
    blocks[:, 1, 1] = np.bincount(
        support_flat, weights=hyy.reshape(-1), minlength=n_particles
    )
    source_coefficient = np.sum(gradient_coefficient, axis=1)
    source_cx = source_coefficient[:, 0]
    source_cy = source_coefficient[:, 1]
    source_hxx = factor * (
        (lam + 2.0 * mu) * source_cx * source_cx + mu * source_cy * source_cy
    )
    source_hxy = factor * (lam + mu) * source_cx * source_cy
    source_hyy = factor * (
        mu * source_cx * source_cx + (lam + 2.0 * mu) * source_cy * source_cy
    )
    blocks[:, 0, 0] += np.bincount(
        source, weights=source_hxx, minlength=n_particles
    )
    source_xy = np.bincount(source, weights=source_hxy, minlength=n_particles)
    blocks[:, 0, 1] += source_xy
    blocks[:, 1, 0] += source_xy
    blocks[:, 1, 1] += np.bincount(
        source, weights=source_hyy, minlength=n_particles
    )
    if not np.all(np.isfinite(blocks)):
        raise FloatingPointError("replay tangent preconditioner is non-finite")
    eigenvalues = np.linalg.eigvalsh(blocks)
    if np.any(eigenvalues <= 0.0):
        bad = np.flatnonzero(np.any(eigenvalues <= 0.0, axis=1))
        raise ValueError(
            "replay tangent preconditioner is not positive definite at particles "
            f"{bad[:10].tolist()}"
        )
    return blocks


def _replay_direct_solve(
    damage: np.ndarray,
    load: float,
    ba: Mapping[str, Any],
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    volume_bottom_mask: np.ndarray,
    volume_top_mask: np.ndarray,
    anchor_particle: int,
    youngs_modulus: float,
    poisson_ratio: float,
    k_res: float,
    n_material: int,
    *,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
) -> dict[str, Any]:
    """Solve one replay fixed-damage load with a matrix-free tangent.

    The replay deliberately rebuilds its own vectorized BA action and sparse
    damage path.  It does not call a production mechanics solve, form a dense
    identity-column tangent, or silently fall back after a failed CG solve.
    """

    center_weights = _replay_center_weights(
        damage, material_phase, grip_phase, k_res, n_material
    )
    n = len(ba["coordinates"])
    n_dofs = 2 * n
    target_load = _scalar(load, "replay direct load")
    rtol = _scalar(cg_rtol, "replay cg_rtol")
    atol = _scalar(cg_atol, "replay cg_atol")
    maxiter = _integer(cg_maxiter, "replay cg_maxiter", positive=True)
    if rtol < 0.0 or atol < 0.0 or (rtol == 0.0 and atol == 0.0):
        raise ValueError("replay CG tolerances must be non-negative and not both zero")
    volume_bottom = _bool(volume_bottom_mask, "replay volume bottom mask", (n,))
    volume_top = _bool(volume_top_mask, "replay volume top mask", (n,))
    if np.any(volume_bottom & volume_top):
        raise ValueError("replay volume grip masks overlap")
    anchor = _integer(anchor_particle, "replay anchor particle")
    if anchor >= n:
        raise ValueError("replay anchor particle is outside the union cloud")
    top_dofs = 2 * np.flatnonzero(volume_top) + 1
    bottom_dofs = 2 * np.flatnonzero(volume_bottom) + 1
    constrained = np.sort(
        np.concatenate((bottom_dofs, top_dofs, np.asarray([2 * anchor], dtype=np.int64)))
    )
    free = np.setdiff1d(np.arange(n_dofs, dtype=np.int64), constrained)
    if len(free) == 0:
        raise ValueError("replay direct constraints leave no free mechanics DOFs")
    prescribed = np.zeros(n_dofs, dtype=np.float64)
    prescribed[top_dofs] = target_load

    def tangent_action(values: np.ndarray) -> np.ndarray:
        return _replay_tangent_action(
            values, ba, center_weights, youngs_modulus, poisson_ratio
        )

    tangent = LinearOperator(
        shape=(n_dofs, n_dofs), matvec=tangent_action, dtype=np.float64
    )

    def inject(values: np.ndarray) -> np.ndarray:
        free_values = _f64(values, "replay CG free values", (len(free),))
        result = np.zeros(n_dofs, dtype=np.float64)
        result[free] = free_values
        return result

    rhs = -tangent.matvec(prescribed)[free]
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("replay direct mechanics right-hand side is non-finite")
    rhs_norm = float(np.linalg.norm(rhs))
    convergence_limit = max(atol, rtol * rhs_norm)
    callback_count: list[int] = []
    if rhs_norm == 0.0:
        free_values = np.zeros(len(free), dtype=np.float64)
        info = 0
    else:
        def free_action(values: np.ndarray) -> np.ndarray:
            return tangent_action(inject(values))[free]

        blocks = _replay_tangent_block_diagonal(
            ba, center_weights, youngs_modulus, poisson_ratio
        )
        free_mask = np.zeros((n, 2), dtype=bool)
        free_mask.reshape(-1)[free] = True
        inverse_blocks = np.zeros_like(blocks)
        both_free = np.all(free_mask, axis=1)
        x_only = free_mask[:, 0] & ~free_mask[:, 1]
        y_only = free_mask[:, 1] & ~free_mask[:, 0]
        if np.any(both_free):
            inverse_blocks[both_free] = np.linalg.inv(blocks[both_free])
        if np.any(x_only):
            inverse_blocks[x_only, 0, 0] = 1.0 / blocks[x_only, 0, 0]
        if np.any(y_only):
            inverse_blocks[y_only, 1, 1] = 1.0 / blocks[y_only, 1, 1]
        if not np.all(np.isfinite(inverse_blocks)):
            raise FloatingPointError("replay tangent preconditioner inverse is non-finite")

        def precondition(values: np.ndarray) -> np.ndarray:
            free_values = _f64(values, "replay CG preconditioner values", (len(free),))
            embedded = np.zeros((n, 2), dtype=np.float64)
            embedded.reshape(-1)[free] = free_values
            result = np.einsum(
                "pab,pb->pa", inverse_blocks, embedded, optimize=True
            )
            return np.asarray(result.reshape(-1)[free], dtype=np.float64)

        def callback(_: np.ndarray) -> None:
            callback_count.append(1)

        operator = LinearOperator(
            shape=(len(free), len(free)), matvec=free_action, dtype=np.float64
        )
        preconditioner = LinearOperator(
            shape=(len(free), len(free)), matvec=precondition, dtype=np.float64
        )
        free_values, info = cg(
            operator,
            rhs,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            M=preconditioner,
            callback=callback,
        )
        info = int(info)
        if info != 0:
            raise RuntimeError(
                "replay direct mechanics CG did not converge: "
                f"info={info}, iterations={len(callback_count)}, "
                f"rtol={rtol}, atol={atol}, maxiter={maxiter}"
            )
    free_values = np.asarray(free_values, dtype=np.float64)
    if free_values.shape != (len(free),) or not np.all(np.isfinite(free_values)):
        raise FloatingPointError("replay direct mechanics CG returned non-finite values")
    q = np.asarray(prescribed + inject(free_values), dtype=np.float64)
    if not np.array_equal(q[constrained], prescribed[constrained]):
        raise ValueError("replay direct mechanics solution violates constraints")
    response = _replay_response(
        q,
        damage,
        ba,
        center_weights,
        youngs_modulus,
        poisson_ratio,
        n_material,
    )
    residual = np.asarray(response["residual"], dtype=np.float64)
    free_residual = residual[free]
    free_residual_norm = float(np.linalg.norm(free_residual))
    residual_gate_failed = (
        not np.isfinite(free_residual_norm)
        or (
            convergence_limit == 0.0
            and free_residual_norm != 0.0
        )
        or (
            convergence_limit > 0.0
            and free_residual_norm >= convergence_limit
        )
    )
    if residual_gate_failed:
        raise RuntimeError(
            "replay direct mechanics explicit residual is outside its CG gate: "
            f"norm={free_residual_norm}, allowed={convergence_limit}"
        )
    residual_matrix = residual.reshape(-1, 2)
    force = np.asarray(np.sum(residual_matrix, axis=0), dtype=np.float64)
    moment = float(
        np.sum(
            ba["coordinates"][:, 0] * residual_matrix[:, 1]
            - ba["coordinates"][:, 1] * residual_matrix[:, 0]
        )
    )
    if not np.all(np.isfinite(force)) or not np.isfinite(moment):
        raise FloatingPointError("replay direct mechanics conservation data are non-finite")
    return {
        "q": q,
        "response": response,
        "residual": residual,
        "top": float(np.sum(residual[top_dofs])),
        "bottom": float(np.sum(residual[bottom_dofs])),
        "anchor": float(residual[2 * anchor]),
        "force": force,
        "moment": moment,
        "free_residual_norm": free_residual_norm,
        "free_rhs_norm": rhs_norm,
        "free_residual_relative_norm": free_residual_norm / max(rhs_norm, 1.0e-300),
        "cg_convergence_limit": convergence_limit,
        "cg_info": int(info),
        "cg_iterations": len(callback_count),
        "tangent": tangent,
    }


def _is_warning_archive_schema(schema: Any) -> bool:
    return schema in (WARNING_ARCHIVE_SCHEMA, UNBOUNDED_WARNING_ARCHIVE_SCHEMA)


def _archive_required(
    archive: Mapping[str, np.ndarray],
    warning: bool = False,
    unbounded: bool = False,
) -> None:
    if "archive_schema" in archive:
        archive_schema = _json_scalar(archive["archive_schema"], "archive_schema")
        if archive_schema == _STALE_ARCHIVE_SCHEMA:
            raise ValueError(
                "engineering-grip archive schema v1 is an unsupported stale schema; "
                "no migration or defaults are provided"
            )
        if archive_schema == UNBOUNDED_WARNING_ARCHIVE_SCHEMA:
            warning = True
            unbounded = True
    required = {
        "archive_schema",
        "acceptance_mode",
        "acceptance_status",
        "completion_status",
        "raw_source_used",
        "metadata_json",
        "material_coordinates",
        "material_volumes",
        "material_triangles",
        "material_edges",
        "material_bottom_mask",
        "material_top_mask",
        "material_anchor",
        "union_coordinates",
        "union_material_phase_volumes",
        "union_grip_phase_volumes",
        "union_volumes",
        "grip_triangles",
        "union_triangles",
        "candidate_edges",
        "visible_edges",
        "removed_edges",
        "interface_bottom_mask",
        "interface_top_mask",
        "volume_bottom_mask",
        "volume_top_mask",
        "constrained_dofs",
        "free_dofs",
        "top_line_indices",
        "top_line_weights",
        "top_line_length",
        "bottom_line_indices",
        "bottom_line_weights",
        "bottom_line_length",
        "crack_region_mask",
        "crack_localization_atol",
        "state_u_bar",
        "state_union_displacement",
        "state_material_damage",
        "state_material_history",
        "state_digest",
        "state_acceptance_status",
        "static_edge_distance",
        "static_edge_omega",
        "static_family_mass",
        "static_eta",
        "static_support_indices",
        "static_support_mask",
        "static_support_sizes",
        "static_beta",
        "static_xi",
        "static_shape_inverse",
        "static_shape_rank",
        "static_shape_condition",
        "static_gradient_coefficient",
        "accepted_load_target",
        "accepted_load_start",
        "accepted_cutback_count",
        "accepted_stagger_iterations",
        "accepted_damage_before",
        "accepted_d_iter",
        "accepted_d_next",
        "accepted_q_before",
        "accepted_q_after",
        "accepted_psi_before",
        "accepted_psi_after",
        "accepted_H_used",
        "accepted_H_next",
        "accepted_rho_used",
        "accepted_rho_next",
        "accepted_rho_used_volume",
        "accepted_rho_next_volume",
        "accepted_energy_before",
        "accepted_energy_after",
        "accepted_top_reaction_before",
        "accepted_top_reaction_after",
        "accepted_bottom_reaction_before",
        "accepted_bottom_reaction_after",
        "accepted_anchor_reaction_before",
        "accepted_anchor_reaction_after",
        "accepted_cg_info_before",
        "accepted_cg_info_after",
        "accepted_force_before",
        "accepted_force_after",
        "accepted_moment_before",
        "accepted_moment_after",
        "accepted_diagnostics_json",
        "attempt_cutback_count",
        "attempt_load_start",
        "attempt_load_target",
        "attempt_accepted",
        "attempt_before_digest",
        "attempt_after_digest",
        "attempt_failure_stage",
        "attempt_failure_reason",
        "attempt_exception_type",
        "attempt_sweep_count",
        "attempt_sweeps_json",
        "attempt_diagnostics_json",
         "work_check_u_bar",
        "work_check_epsilon",
        "work_check_reaction",
        "work_check_reaction_fd",
        "work_check_relative_error",
        "work_check_relative_limit",
         "work_check_passed",
         "state_material_displacement",
         "state_material_particle_strain",
         "state_material_amor_driver",
         "state_physical_interface_opening",
         "state_interface_damage_bottom_max",
         "state_interface_damage_top_max",
         "state_interface_history_bottom_max",
         "state_interface_history_top_max",
         "state_crack_particle_strain",
        "state_crack_amor_driver",
        "state_crack_damage_inside_max",
        "state_crack_damage_outside_max",
        "state_crack_history_inside_max",
        "state_crack_history_outside_max",
        "state_crack_strain_inside_max",
        "state_crack_strain_outside_max",
        "state_crack_amor_inside_max",
        "state_crack_amor_outside_max",
         "state_crack_localization_pass",
          "accepted_interface_opening_before",
          "accepted_interface_opening_after",
      }
    if warning:
        warning_fields = {
            "validated",
                "warning_gate_names",
                "accepted_acceptance_status",
                "accepted_warning",
                "accepted_bypassed_gate_mask",
                "accepted_warning_range_excess",
                "accepted_warning_healing_magnitude",
                "attempt_acceptance_status",
                "attempt_bypassed_gate_mask",
                "attempt_warning_excess_present",
                "attempt_warning_range_excess",
                "attempt_warning_healing_magnitude",
        }
        if unbounded:
            warning_fields.update(
                {"warning_safety_policy", "warning_safety_limit_present"}
            )
        else:
            warning_fields.add("warning_safety_limit")
        required.update(warning_fields)
    missing = sorted(required - set(archive))
    if missing:
        raise ValueError(f"engineering-grip archive is missing fields: {missing}")


def _replay_scalar_fields(
    archive: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], EngineeringGripConfig, bool, str | None, float | None]:
    archive_schema = _json_scalar(archive["archive_schema"], "archive_schema")
    if archive_schema == _STALE_ARCHIVE_SCHEMA:
        raise ValueError(
            "engineering-grip archive schema v1 is an unsupported stale schema; "
            "no migration or defaults are provided"
        )
    warning = _is_warning_archive_schema(archive_schema)
    unbounded = archive_schema == UNBOUNDED_WARNING_ARCHIVE_SCHEMA
    if not warning and archive_schema != ARCHIVE_SCHEMA:
        raise ValueError("engineering-grip archive schema mismatch")
    if warning:
        if _json_scalar(archive["acceptance_mode"], "acceptance_mode") != _WARNING_ACCEPTANCE_MODE:
            raise ValueError("engineering-grip warning archive acceptance mode is forged")
        if _json_scalar(archive["acceptance_status"], "acceptance_status") not in (
            _ACCEPTANCE_STATUS,
            _WARNING_ACCEPTANCE_STATUS,
        ):
            raise ValueError("engineering-grip warning archive acceptance status is forged")
        if (
            _json_scalar(archive["completion_status"], "completion_status")
            != _WARNING_COMPLETION_STATUS
        ):
            raise ValueError("engineering-grip warning archive completion status is forged")
        validated_flag = np.asarray(archive["validated"])
        if (
            validated_flag.shape != ()
            or validated_flag.dtype != _BOOL
            or bool(validated_flag)
        ):
            raise ValueError("engineering-grip warning archive validated flag is forged")
        if unbounded:
            warning_policy = _validate_warning_safety_policy(
                _json_scalar(
                    archive["warning_safety_policy"], "warning_safety_policy"
                )
            )
            if warning_policy != _WARNING_SAFETY_POLICY_UNBOUNDED:
                raise ValueError(
                    "engineering-grip unbounded warning archive safety policy is forged"
                )
            limit_present = _bool(
                archive["warning_safety_limit_present"],
                "archive warning safety limit presence",
                (),
            )
            if bool(limit_present):
                raise ValueError(
                    "engineering-grip unbounded warning archive limit presence is forged"
                )
            if "warning_safety_limit" in archive:
                raise ValueError(
                    "engineering-grip unbounded warning archive contains a safety limit"
                )
            warning_limit = None
        else:
            warning_policy = _WARNING_SAFETY_POLICY_BOUNDED
            warning_limit = _scalar(
                _f64(
                    archive["warning_safety_limit"],
                    "archive warning safety limit",
                    (),
                ),
                "archive warning safety limit",
            )
            _validate_warning_safety_limit(warning_limit)
    else:
        if _json_scalar(archive["acceptance_mode"], "acceptance_mode") != _ACCEPTANCE_MODE:
            raise ValueError("engineering-grip archive acceptance mode is not strict")
        if _json_scalar(archive["acceptance_status"], "acceptance_status") != _ACCEPTANCE_STATUS:
            raise ValueError("engineering-grip archive acceptance status is forged")
        if _json_scalar(archive["completion_status"], "completion_status") != _COMPLETION_STATUS:
            raise ValueError("engineering-grip archive completion status is forged")
        warning_policy = None
        warning_limit = None
    raw = np.asarray(archive["raw_source_used"])
    if raw.shape != () or raw.dtype != _BOOL or bool(raw):
        raise ValueError("engineering-grip archive raw_source_used must be exact False")
    try:
        parsed = json.loads(_json_scalar(archive["metadata_json"], "metadata_json"))
    except json.JSONDecodeError as error:
        raise ValueError("engineering-grip metadata_json is not valid JSON") from error
    if not isinstance(parsed, Mapping):
        raise ValueError("engineering-grip metadata_json must be a mapping")
    metadata_schema = parsed.get("schema")
    if metadata_schema == _STALE_ARCHIVE_SCHEMA:
        raise ValueError(
            "engineering-grip metadata schema v1 is an unsupported stale schema; "
            "no migration or defaults are provided"
        )
    expected_schema = (
        UNBOUNDED_WARNING_ARCHIVE_SCHEMA
        if unbounded
        else WARNING_ARCHIVE_SCHEMA
        if warning
        else ARCHIVE_SCHEMA
    )
    if metadata_schema != expected_schema or parsed.get("raw_source_used") is not False:
        raise ValueError("engineering-grip metadata schema/raw provenance is forged")
    if warning:
        if parsed.get("acceptance_mode") != _WARNING_ACCEPTANCE_MODE:
            raise ValueError("engineering-grip warning metadata acceptance mode is forged")
        if unbounded:
            if parsed.get("warning_safety_policy") != warning_policy:
                raise ValueError(
                    "engineering-grip unbounded warning metadata safety policy is forged"
                )
            if parsed.get("warning_safety_limit_present") is not False:
                raise ValueError(
                    "engineering-grip unbounded warning metadata limit presence is forged"
                )
            if "warning_safety_limit" in parsed:
                raise ValueError(
                    "engineering-grip unbounded warning metadata contains a safety limit"
                )
        else:
            if "warning_safety_limit" not in parsed:
                raise ValueError(
                    "engineering-grip warning metadata safety limit is missing"
                )
            metadata_limit = parsed["warning_safety_limit"]
            if (
                not isinstance(metadata_limit, (int, float))
                or not np.isfinite(float(metadata_limit))
                or float(metadata_limit) != warning_limit
            ):
                raise ValueError(
                    "engineering-grip warning metadata safety limit is forged"
                )
    if parsed.get("crack_region_rule") != _CRACK_REGION_RULE:
        raise ValueError("engineering-grip metadata crack-region rule is forged")
    if parsed.get("crack_localization_atol") != _CRACK_LOCALIZATION_ATOL:
        raise ValueError("engineering-grip metadata localization tolerance is forged")
    if parsed.get("crack_interface_rule") != _CRACK_INTERFACE_RULE:
        raise ValueError("engineering-grip metadata interface rule is forged")
    config_payload = parsed.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("engineering-grip metadata config must be a mapping")
    config = _config_from_payload(config_payload)
    if json.dumps(_config_payload(config), sort_keys=True) != json.dumps(config_payload, sort_keys=True):
        raise ValueError("engineering-grip config payload is not canonical")
    code_hash_kind = parsed.get("code_hash_kind")
    if code_hash_kind != _SOURCE_BUNDLE_HASH_KIND:
        raise ValueError("engineering-grip metadata code hash meaning is forged")
    code_hash_files = parsed.get("code_hash_files")
    if code_hash_files != list(_SOURCE_BUNDLE_FILES):
        raise ValueError("engineering-grip metadata code hash file bundle is forged")
    code_hash = parsed.get("code_hash")
    _sha256(parsed.get("static_digest"), "metadata.static_digest")
    _sha256(parsed.get("material_hash"), "metadata.material_hash")
    _sha256(parsed.get("union_hash"), "metadata.union_hash")
    code_hash = _sha256(code_hash, "metadata.code_hash")
    current_hash = _current_source_bundle_hash()
    if code_hash != current_hash:
        raise ValueError(
            "engineering-grip archive source-bundle code hash does not match replay"
        )
    if parsed.get("visibility_rule") != _VISIBILITY_RULE:
        raise ValueError("engineering-grip metadata visibility rule is forged")
    values: dict[str, float] = {}
    for name in (
        "youngs_modulus",
        "poisson_ratio",
        "k_res",
        "rank_rtol",
        "delta",
        "h_g",
        "Gc",
        "l0",
        "visibility_tolerance",
    ):
        if name not in parsed:
            raise ValueError(f"engineering-grip metadata is missing {name}")
        values[name] = _scalar(parsed[name], f"metadata.{name}")
    if values["youngs_modulus"] <= 0.0:
        raise ValueError("engineering-grip metadata youngs_modulus must be positive")
    if not -1.0 < values["poisson_ratio"] < 0.5:
        raise ValueError("engineering-grip metadata poisson_ratio is outside its range")
    if values["k_res"] <= 0.0:
        raise ValueError("engineering-grip metadata k_res must be positive")
    if not 0.0 < values["rank_rtol"] < 1.0:
        raise ValueError("engineering-grip metadata rank_rtol must lie in (0, 1)")
    if values["delta"] <= 0.0:
        raise ValueError("engineering-grip metadata delta must be positive")
    if not 0.0 < values["h_g"] < values["delta"]:
        raise ValueError("engineering-grip metadata h_g must lie in (0, delta)")
    if values["Gc"] <= 0.0 or values["l0"] <= 0.0:
        raise ValueError("engineering-grip metadata Gc/l0 must be positive")
    if values["visibility_tolerance"] < 0.0:
        raise ValueError(
            "engineering-grip metadata visibility_tolerance must be non-negative"
        )
    return dict(parsed), config, warning, warning_policy, warning_limit


def _replay_expected_available_fields(diag: Any) -> tuple[str, ...]:
    """Rebuild the array-availability contract of one sweep diagnostic."""

    names = ["damage_before", "d_iter"]
    names.extend(
        name
        for name in _DIAGNOSTIC_ARRAY_FIELDS
        if name not in ("damage_before", "d_iter", "force_before", "force_after")
        and getattr(diag, name, None) is not None
    )
    return tuple(sorted(set(names)))


def _replay_mechanics_recorded_fields(
    diag: Any,
    prefix: str,
    mechanics: Mapping[str, Any] | None,
    *,
    require_cg_success: bool = True,
) -> None:
    """Compare all archived mechanics scalars with one stored-q evaluation."""

    cg_info = getattr(diag, f"cg_info_{prefix}", None)
    if mechanics is None:
        if cg_info is not None:
            raise ValueError(
                f"replay cg_info_{prefix} is fabricated without mechanics"
            )
        expected = {
            "energy": None,
            "top": None,
            "bottom": None,
            "anchor": None,
            "moment": None,
            "free_residual_norm": None,
            "free_residual_relative_norm": None,
        }
        force = None
        tolerances: dict[str, float] | None = None
    else:
        if cg_info is None:
            raise ValueError(f"replay cg_info_{prefix} is unavailable")
        cg_status = _integer(cg_info, f"replay cg_info_{prefix}")
        if require_cg_success and cg_status != 0:
            raise ValueError(
                f"replay production cg_info_{prefix} is not zero"
            )
        expected = {
            "energy": mechanics["energy"],
            "top": mechanics["top"],
            "bottom": mechanics["bottom"],
            "anchor": mechanics["anchor"],
            "moment": mechanics["moment"],
            "free_residual_norm": mechanics["free_residual_norm"],
            "free_residual_relative_norm": mechanics["free_residual_relative_norm"],
        }
        force = mechanics["force"]
        tolerances = {
            "energy": _replay_reduction_atol(
                mechanics["energy_constituent_scale"],
                f"replay energy_{prefix}",
            ),
            "top": _replay_reduction_atol(
                mechanics["top_constituent_scale"],
                f"replay top_reaction_{prefix}",
            ),
            "bottom": _replay_reduction_atol(
                mechanics["bottom_constituent_scale"],
                f"replay bottom_reaction_{prefix}",
            ),
            "anchor": _replay_reduction_atol(
                mechanics["anchor_constituent_scale"],
                f"replay anchor_reaction_{prefix}",
            ),
            "moment": _replay_reduction_atol(
                mechanics["moment_constituent_scale"],
                f"replay moment_{prefix}",
            ),
            "free_residual_norm": _replay_reduction_atol(
                mechanics["free_residual_norm_constituent_scale"],
                f"replay free_residual_norm_{prefix}",
            ),
            "free_residual_relative_norm": mechanics[
                "free_residual_relative_roundoff_atol"
            ],
        }
    diagnostic_names = {
        "energy": f"energy_{prefix}",
        "top": f"top_reaction_{prefix}",
        "bottom": f"bottom_reaction_{prefix}",
        "anchor": f"anchor_reaction_{prefix}",
        "moment": f"moment_{prefix}",
        "free_residual_norm": f"free_residual_norm_{prefix}",
        "free_residual_relative_norm": f"free_residual_relative_{prefix}",
    }
    for field, value in expected.items():
        if tolerances is None:
            _compare_replay_scalar(
                getattr(diag, diagnostic_names[field], None),
                value,
                f"replay {field}_{prefix}",
            )
        else:
            _compare_replay_scalar(
                getattr(diag, diagnostic_names[field], None),
                value,
                f"replay {field}_{prefix}",
                limit=0.0,
                absolute_atol=tolerances[field],
                scale_floor=0.0,
            )
    actual_force = getattr(diag, f"force_{prefix}", None)
    if force is None:
        if actual_force is not None:
            raise ValueError(f"replay force_{prefix} is fabricated without mechanics")
    elif actual_force is None:
        raise ValueError(f"replay force_{prefix} is unavailable")
    else:
        if tolerances is None:
            _array_close(actual_force, force, f"replay force_{prefix}", 2.0e-8)
        else:
            assert mechanics is not None
            _compare_replay_vector(
                actual_force,
                force,
                f"replay force_{prefix}",
                mechanics["force_constituent_scale"],
            )


def _replay_complete_sweep(
    diag: Any,
    committed_damage: np.ndarray,
    committed_history: np.ndarray,
    local_damage: np.ndarray,
    target: float,
    ba: Mapping[str, Any],
    material_coordinates: np.ndarray,
    material_volumes: np.ndarray,
    material_edges: np.ndarray,
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    volume_bottom: np.ndarray,
    volume_top: np.ndarray,
    free_dofs: np.ndarray,
    E: float,
    nu: float,
    k_res: float,
    Gc: float,
    l0: float,
    delta: float,
    n_material: int,
    config: EngineeringGripConfig,
    warning_safety_limit: float | None = None,
    final_sweep: bool = False,
    warning_safety_policy: str | None = None,
) -> np.ndarray:
    """Independently replay every primitive of one complete sweep."""

    if not bool(getattr(diag, "complete_sweep", False)):
        raise ValueError("complete replay requires complete_sweep=True")
    # ``warning_safety_limit`` is deliberately not part of complete-sweep
    # gate reconstruction.  The replayed flags must remain the strict
    # config-atol results; the caller applies the raw safety limit separately
    # when classifying a final warning candidate.
    if final_sweep and warning_safety_policy is not None:
        _validate_warning_safety_configuration(
            warning_safety_policy, warning_safety_limit
        )
    elif final_sweep and warning_safety_limit is not None:
        _validate_warning_safety_limit(warning_safety_limit)
    committed_d = _f64(committed_damage, "replay committed damage", (n_material,))
    committed_H = _f64(committed_history, "replay committed history", (n_material,))
    d_iter = _f64(getattr(diag, "d_iter", None), "replay complete d_iter", (n_material,))
    if not np.array_equal(d_iter, local_damage):
        raise ValueError("replay complete sweep d_iter is not the local trial iterate")
    damage_before = _f64(
        getattr(diag, "damage_before", None),
        "replay complete damage_before",
        (n_material,),
    )
    if not np.array_equal(damage_before, committed_d):
        raise ValueError("replay complete sweep damage_before is not committed damage")
    d_next = _f64(getattr(diag, "d_next", None), "replay complete d_next", (n_material,))
    q_before = _f64(
        getattr(diag, "q_before", None),
        "replay complete q_before",
        (2 * len(ba["coordinates"]),),
    )
    q_after = _f64(
        getattr(diag, "q_after", None),
        "replay complete q_after",
        (2 * len(ba["coordinates"]),),
    )
    psi_before = _f64(
        getattr(diag, "psi_before", None),
        "replay complete psi_before",
        (n_material,),
    )
    psi_after = _f64(
        getattr(diag, "psi_after", None),
        "replay complete psi_after",
        (n_material,),
    )
    H_used = _f64(
        getattr(diag, "H_used", None),
        "replay complete H_used",
        (n_material,),
    )
    H_next = _f64(
        getattr(diag, "H_next", None),
        "replay complete H_next",
        (n_material,),
    )
    rho_used = _f64(
        getattr(diag, "rho_used", None),
        "replay complete rho_used",
        (n_material,),
    )
    rho_next = _f64(
        getattr(diag, "rho_next", None),
        "replay complete rho_next",
        (n_material,),
    )
    rho_used_volume = _f64(
        getattr(diag, "rho_used_volume", None),
        "replay complete rho_used_volume",
        (n_material,),
    )
    rho_next_volume = _f64(
        getattr(diag, "rho_next_volume", None),
        "replay complete rho_next_volume",
        (n_material,),
    )
    before = _replay_mechanics_at_state(
        q_before,
        d_iter,
        target,
        ba,
        material_phase,
        grip_phase,
        volume_bottom,
        volume_top,
        free_dofs,
        0,
        E,
        nu,
        k_res,
        n_material,
        config,
    )
    after = _replay_mechanics_at_state(
        q_after,
        d_next,
        target,
        ba,
        material_phase,
        grip_phase,
        volume_bottom,
        volume_top,
        free_dofs,
        0,
        E,
        nu,
        k_res,
        n_material,
        config,
    )
    replay_psi_before, expected_H_used, _ = _replay_amor(
        q_before, committed_H, ba, E, nu, n_material
    )
    replay_psi_after, expected_H_next, _ = _replay_amor(
        q_after, committed_H, ba, E, nu, n_material
    )
    expected_d_next, _, _, matrix, _ = _replay_damage(
        H_used,
        material_coordinates,
        material_volumes,
        material_edges,
        delta,
        Gc,
        l0,
    )
    for name, actual, expected in (
        ("psi_before", psi_before, replay_psi_before),
        ("psi_after", psi_after, replay_psi_after),
        ("H_used", H_used, expected_H_used),
        ("H_next", H_next, expected_H_next),
        ("d_next", d_next, expected_d_next),
    ):
        _array_close(actual, expected, f"replay complete {name}", 2.0e-8)
    _compare_damage_residual_reconstruction(
        rho_used,
        rho_used_volume,
        d_next,
        H_used,
        matrix,
        material_volumes,
        Gc,
        l0,
        "replay complete rho_used",
    )
    _compare_damage_residual_reconstruction(
        rho_next,
        rho_next_volume,
        d_next,
        H_next,
        matrix,
        material_volumes,
        Gc,
        l0,
        "replay complete rho_next",
    )
    _replay_mechanics_recorded_fields(diag, "before", before)
    _replay_mechanics_recorded_fields(diag, "after", after)

    e_u_material = _max_abs(
        q_after[: 2 * n_material] - q_before[: 2 * n_material]
    )
    material_scale = max(
        _max_abs(q_before[: 2 * n_material]),
        _max_abs(q_after[: 2 * n_material]),
        abs(target),
    )
    e_u_union = _max_abs(q_after - q_before)
    union_scale = max(_max_abs(q_before), _max_abs(q_after), abs(target))
    e_damage = _max_abs(d_next - d_iter)
    damage_scale = max(_max_abs(d_iter), _max_abs(d_next))
    e_history = _max_abs(H_next - H_used)
    history_scale = max(_max_abs(committed_H), _max_abs(H_used), _max_abs(H_next))
    e_reaction = abs(after["top"] - before["top"])
    reaction_scale = max(abs(before["top"]), abs(after["top"]))
    damage_min = float(np.min(d_next))
    damage_max = float(np.max(d_next))
    healing_min = float(np.min(d_next - committed_d))
    history_min_gap = float(np.min(H_next - committed_H))
    force_error = max(float(before["force_error"]), float(after["force_error"]))
    force_scale = max(
        1.0,
        abs(before["top"]),
        abs(after["top"]),
        abs(before["bottom"]),
        abs(after["bottom"]),
        abs(before["anchor"]),
        abs(after["anchor"]),
    )
    moment_error = max(float(before["moment_error"]), float(after["moment_error"]))
    geometry_length = max(
        _max_abs(ba["coordinates"]), _max_abs(np.ptp(ba["coordinates"], axis=0)), 1.0
    )
    moment_scale = force_scale * geometry_length
    residual_scale = _replay_damage_residual_scale(
        d_next, H_used, H_next, matrix, Gc, l0
    )
    finite_pass = bool(
        before["finite"]
        and after["finite"]
        and all(
            np.all(np.isfinite(values))
            for values in (
                d_iter,
                d_next,
                psi_before,
                psi_after,
                H_used,
                H_next,
                rho_used,
                rho_next,
                rho_used_volume,
                rho_next_volume,
            )
        )
    )
    passes = {
        "finite": finite_pass,
        "mechanics_before": bool(before["mechanics_pass"]),
        "mechanics_after": bool(after["mechanics_pass"]),
        "displacement": bool(
            e_u_material <= config.displacement.limit(material_scale)
            and e_u_union <= config.displacement.limit(union_scale)
        ),
        "damage": bool(e_damage <= config.damage.limit(damage_scale)),
        "history": bool(e_history <= config.history.limit(history_scale)),
        "reaction": bool(e_reaction <= config.reaction.limit(reaction_scale)),
        "damage_residual": bool(
            _max_abs(rho_next) <= config.damage_residual.limit(residual_scale)
        ),
        "damage_range": bool(
            damage_min >= -config.damage_range_atol
            and damage_max <= 1.0 + config.damage_range_atol
        ),
        "healing": bool(healing_min >= -config.healing_atol),
        "history_monotonic": bool(history_min_gap >= 0.0),
        "force": bool(before["force_pass"] and after["force_pass"]),
        "moment": bool(before["moment_pass"] and after["moment_pass"]),
    }
    if any(not passes[name] for name in _WARNING_HARD_GATE_NAMES):
        raise ValueError(
            "replay complete sweep contains a failed warning hard gate"
        )
    expected_passed = bool(all(passes.values()))
    if bool(getattr(diag, "passed", False)) != expected_passed:
        raise ValueError("replay complete passed flag does not match primitive gates")
    expected_stage = "" if expected_passed else "gates"
    expected_reason = "" if expected_passed else ", ".join(
        name for name, value in passes.items() if not value
    )
    if getattr(diag, "failure_stage", None) != expected_stage:
        raise ValueError("replay complete failure stage does not match primitive gates")
    if getattr(diag, "failure_reason", None) != expected_reason:
        raise ValueError("replay complete failure reason does not match primitive gates")
    expected_scalars = {
        "energy_before": before["energy"],
        "energy_after": after["energy"],
        "top_reaction_before": before["top"],
        "top_reaction_after": after["top"],
        "bottom_reaction_before": before["bottom"],
        "bottom_reaction_after": after["bottom"],
        "anchor_reaction_before": before["anchor"],
        "anchor_reaction_after": after["anchor"],
        "moment_before": before["moment"],
        "moment_after": after["moment"],
        "free_residual_norm_before": before["free_residual_norm"],
        "free_residual_norm_after": after["free_residual_norm"],
        "free_residual_relative_before": before["free_residual_relative_norm"],
        "free_residual_relative_after": after["free_residual_relative_norm"],
        "e_u_material": e_u_material,
        "e_u_union": e_u_union,
        "e_damage": e_damage,
        "e_history": e_history,
        "e_reaction": e_reaction,
        "damage_min": damage_min,
        "damage_max": damage_max,
        "healing_min": healing_min,
        "history_min_gap": history_min_gap,
        "rho_used_inf": _max_abs(rho_used),
        "rho_next_inf": _max_abs(rho_next),
        "displacement_scale": union_scale,
        "damage_scale": damage_scale,
        "history_scale": history_scale,
        "reaction_scale": reaction_scale,
        "damage_residual_scale": residual_scale,
        "force_error": force_error,
        "force_scale": force_scale,
        "moment_error": moment_error,
        "moment_scale": moment_scale,
    }
    for name, expected in expected_scalars.items():
        _compare_replay_scalar(
            getattr(diag, name, None), expected, f"replay complete {name}"
        )
    expected_flags = {
        "finite_pass": passes["finite"],
        "mechanics_before_pass": passes["mechanics_before"],
        "mechanics_after_pass": passes["mechanics_after"],
        "displacement_pass": passes["displacement"],
        "damage_pass": passes["damage"],
        "history_pass": passes["history"],
        "reaction_pass": passes["reaction"],
        "damage_residual_pass": passes["damage_residual"],
        "damage_range_pass": passes["damage_range"],
        "healing_pass": passes["healing"],
        "history_monotonic_pass": passes["history_monotonic"],
        "force_pass": passes["force"],
        "moment_pass": passes["moment"],
    }
    for name, expected in expected_flags.items():
        if bool(getattr(diag, name, False)) != bool(expected):
            raise ValueError(f"replay complete {name} does not match primitive gates")
    gate_payload = getattr(diag, "gate_passes", None)
    if not isinstance(gate_payload, Mapping) or set(gate_payload) != set(passes):
        raise ValueError("replay complete gate payload is forged")
    for name, expected in passes.items():
        if bool(gate_payload[name]) != bool(expected):
            raise ValueError(f"replay complete gate payload {name} is forged")
    if tuple(getattr(diag, "available_fields", ())) != _replay_expected_available_fields(diag):
        raise ValueError("replay complete diagnostic availability is forged")
    return np.asarray(d_next, dtype=np.float64)


def _replay_incomplete_sweep(
    diag: Any,
    committed_damage: np.ndarray,
    committed_history: np.ndarray,
    local_damage: np.ndarray,
    target: float,
    ba: Mapping[str, Any],
    material_coordinates: np.ndarray,
    material_volumes: np.ndarray,
    material_edges: np.ndarray,
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    volume_bottom: np.ndarray,
    volume_top: np.ndarray,
    free_dofs: np.ndarray,
    E: float,
    nu: float,
    k_res: float,
    Gc: float,
    l0: float,
    delta: float,
    n_material: int,
    config: EngineeringGripConfig,
    *,
    warning_mode: bool = False,
) -> None:
    """Validate all primitives retained before an incomplete sweep failure."""

    if bool(getattr(diag, "complete_sweep", False)) or bool(
        getattr(diag, "passed", False)
    ):
        raise ValueError("replay incomplete sweep status is forged")
    committed_d = _f64(committed_damage, "replay incomplete committed damage", (n_material,))
    committed_H = _f64(committed_history, "replay incomplete committed history", (n_material,))
    d_iter = _f64(getattr(diag, "d_iter", None), "replay incomplete d_iter", (n_material,))
    if not np.array_equal(d_iter, local_damage):
        raise ValueError("replay incomplete sweep d_iter is not the local trial iterate")
    damage_before = _f64(
        getattr(diag, "damage_before", None),
        "replay incomplete damage_before",
        (n_material,),
    )
    if not np.array_equal(damage_before, committed_d):
        raise ValueError("replay incomplete damage_before is not committed damage")
    q_before_raw = getattr(diag, "q_before", None)
    q_after_raw = getattr(diag, "q_after", None)
    psi_before_raw = getattr(diag, "psi_before", None)
    psi_after_raw = getattr(diag, "psi_after", None)
    H_used_raw = getattr(diag, "H_used", None)
    d_next_raw = getattr(diag, "d_next", None)
    rho_used_raw = getattr(diag, "rho_used", None)
    rho_used_volume_raw = getattr(diag, "rho_used_volume", None)
    H_next_raw = getattr(diag, "H_next", None)
    rho_next_raw = getattr(diag, "rho_next", None)
    rho_next_volume_raw = getattr(diag, "rho_next_volume", None)
    if H_next_raw is not None or rho_next_raw is not None or rho_next_volume_raw is not None:
        raise ValueError("replay incomplete sweep fabricated final history/residual fields")

    q_before = (
        None
        if q_before_raw is None
        else _f64(q_before_raw, "replay incomplete q_before", (2 * len(ba["coordinates"]),))
    )
    q_after = (
        None
        if q_after_raw is None
        else _f64(q_after_raw, "replay incomplete q_after", (2 * len(ba["coordinates"]),))
    )
    psi_before = (
        None
        if psi_before_raw is None
        else _f64(psi_before_raw, "replay incomplete psi_before", (n_material,))
    )
    psi_after = (
        None
        if psi_after_raw is None
        else _f64(psi_after_raw, "replay incomplete psi_after", (n_material,))
    )
    H_used = (
        None
        if H_used_raw is None
        else _f64(H_used_raw, "replay incomplete H_used", (n_material,))
    )
    d_next = (
        None
        if d_next_raw is None
        else _f64(d_next_raw, "replay incomplete d_next", (n_material,))
    )
    rho_used = (
        None
        if rho_used_raw is None
        else _f64(rho_used_raw, "replay incomplete rho_used", (n_material,))
    )
    rho_used_volume = (
        None
        if rho_used_volume_raw is None
        else _f64(
            rho_used_volume_raw,
            "replay incomplete rho_used_volume",
            (n_material,),
        )
    )
    before = None
    after = None
    if q_before is not None:
        before = _replay_mechanics_at_state(
            q_before,
            d_iter,
            target,
            ba,
            material_phase,
            grip_phase,
            volume_bottom,
            volume_top,
            free_dofs,
            0,
            E,
            nu,
            k_res,
            n_material,
            config,
        )
    if q_after is not None:
        if d_next is None:
            raise ValueError("replay incomplete q_after lacks d_next")
        after = _replay_mechanics_at_state(
            q_after,
            d_next,
            target,
            ba,
            material_phase,
            grip_phase,
            volume_bottom,
            volume_top,
            free_dofs,
            0,
            E,
            nu,
            k_res,
            n_material,
            config,
        )
    if psi_before is not None:
        if before is None:
            raise ValueError("replay incomplete psi_before lacks q_before")
        assert q_before is not None
        replay_psi_before, expected_H_used, _ = _replay_amor(
            q_before, committed_H, ba, E, nu, n_material
        )
        _array_close(psi_before, replay_psi_before, "replay incomplete psi_before", 2.0e-8)
        if H_used is None:
            raise ValueError("replay incomplete psi_before lacks H_used")
        _array_close(H_used, expected_H_used, "replay incomplete H_used", 2.0e-8)
    elif H_used is not None:
        raise ValueError("replay incomplete H_used lacks psi_before")
    if d_next is not None:
        if H_used is None or rho_used is None or rho_used_volume is None:
            raise ValueError("replay incomplete d_next lacks damage-solve primitives")
        expected_d_next, _, _, matrix, _ = _replay_damage(
            H_used,
            material_coordinates,
            material_volumes,
            material_edges,
            delta,
            Gc,
            l0,
        )
        _array_close(d_next, expected_d_next, "replay incomplete d_next", 2.0e-8)
        _compare_damage_residual_reconstruction(
            rho_used,
            rho_used_volume,
            d_next,
            H_used,
            matrix,
            material_volumes,
            Gc,
            l0,
            "replay incomplete rho_used",
        )
    else:
        matrix = None
        if rho_used is not None or rho_used_volume is not None:
            raise ValueError("replay incomplete residual fields lack d_next")
    if psi_after is not None:
        if after is None:
            raise ValueError("replay incomplete psi_after lacks q_after")
        assert q_after is not None
        replay_psi_after, _, _ = _replay_amor(
            q_after, committed_H, ba, E, nu, n_material
        )
        _array_close(psi_after, replay_psi_after, "replay incomplete psi_after", 2.0e-8)

    _replay_mechanics_recorded_fields(
        diag, "before", before, require_cg_success=False
    )
    _replay_mechanics_recorded_fields(
        diag, "after", after, require_cg_success=False
    )
    before_mechanics_pass = bool(
        before is not None
        and before["mechanics_pass"]
        and getattr(diag, "cg_info_before", None) == 0
    )
    after_mechanics_pass = bool(
        after is not None
        and after["mechanics_pass"]
        and getattr(diag, "cg_info_after", None) == 0
    )
    before_force_pass = bool(before is not None and before["force_pass"])
    before_moment_pass = bool(before is not None and before["moment_pass"])
    after_force_pass = bool(after is not None and after["force_pass"])
    after_moment_pass = bool(after is not None and after["moment_pass"])
    stage = getattr(diag, "failure_stage", None)
    reason = getattr(diag, "failure_reason", None)
    allowed_stages: set[str] = set(_WARNING_INCOMPLETE_STAGES)
    if not warning_mode:
        allowed_stages.add("damage_acceptance")
    if not isinstance(stage, str) or stage not in allowed_stages:
        raise ValueError("replay incomplete failure stage is unknown")
    if not isinstance(reason, str) or not reason:
        raise ValueError("replay incomplete failure reason is unavailable")
    expected_presence = {
        "mechanics_before": {
            "q_before": False,
            "q_after": False,
            "psi_before": False,
            "psi_after": False,
            "H_used": False,
            "d_next": False,
            "rho_used": False,
            "rho_used_volume": False,
        },
        "mechanics_before_gate": {
            "q_before": True,
            "q_after": False,
            "psi_before": False,
            "psi_after": False,
            "H_used": False,
            "d_next": False,
            "rho_used": False,
            "rho_used_volume": False,
        },
        "amor_before": {
            "q_before": True,
            "q_after": False,
            "psi_before": False,
            "psi_after": False,
            "H_used": False,
            "d_next": False,
            "rho_used": False,
            "rho_used_volume": False,
        },
        "damage": {
            "q_before": True,
            "q_after": False,
            "psi_before": True,
            "psi_after": False,
            "H_used": True,
            "d_next": False,
            "rho_used": False,
            "rho_used_volume": False,
        },
        "damage_acceptance": {
            "q_before": True,
            "q_after": False,
            "psi_before": True,
            "psi_after": False,
            "H_used": True,
            "d_next": True,
            "rho_used": True,
            "rho_used_volume": True,
        },
        "mechanics_after": {
            "q_before": True,
            "q_after": False,
            "psi_before": True,
            "psi_after": False,
            "H_used": True,
            "d_next": True,
            "rho_used": True,
            "rho_used_volume": True,
        },
        "amor_after": {
            "q_before": True,
            "q_after": True,
            "psi_before": True,
            "psi_after": False,
            "H_used": True,
            "d_next": True,
            "rho_used": True,
            "rho_used_volume": True,
        },
        "mechanics_after_gate": {
            "q_before": True,
            "q_after": True,
            "psi_before": True,
            "psi_after": True,
            "H_used": True,
            "d_next": True,
            "rho_used": True,
            "rho_used_volume": True,
        },
    }[stage]
    actual_presence = {
        name: value is not None
        for name, value in (
            ("q_before", q_before),
            ("q_after", q_after),
            ("psi_before", psi_before),
            ("psi_after", psi_after),
            ("H_used", H_used),
            ("d_next", d_next),
            ("rho_used", rho_used),
            ("rho_used_volume", rho_used_volume),
        )
    }
    if actual_presence != expected_presence:
        raise ValueError("replay incomplete failure stage has inconsistent primitives")
    if stage in {
        "amor_before",
        "damage",
        "damage_acceptance",
        "mechanics_after",
        "amor_after",
        "mechanics_after_gate",
    } and (
        not before_mechanics_pass
        or (
            not warning_mode
            and (not before_force_pass or not before_moment_pass)
        )
    ):
        raise ValueError("replay incomplete failure lacks the prior mechanics gate")
    if d_next is not None:
        prior_range_pass = bool(
            np.min(d_next) >= -config.damage_range_atol
            and np.max(d_next) <= 1.0 + config.damage_range_atol
        )
        prior_healing_pass = bool(
            np.min(d_next - committed_d) >= -config.healing_atol
        )
    else:
        prior_range_pass = False
        prior_healing_pass = False
    if stage in {"mechanics_after", "amor_after", "mechanics_after_gate"} and (
        not warning_mode
        and (not prior_range_pass or not prior_healing_pass)
    ):
        raise ValueError("replay incomplete failure bypasses prior damage gates")
    if q_before is None:
        expected_stages = {"mechanics_before"}
    elif psi_before is None:
        expected_stages = {"mechanics_before_gate", "amor_before"}
    elif d_next is None:
        expected_stages = {"damage"}
    elif q_after is None:
        expected_stages = {"damage_acceptance", "mechanics_after"}
    elif psi_after is None:
        expected_stages = {"amor_after"}
    else:
        expected_stages = {"mechanics_after_gate"}
    if stage not in expected_stages:
        raise ValueError(
            "replay incomplete failure stage is inconsistent with available primitives"
        )
    if stage == "mechanics_before_gate":
        if before is None:
            raise ValueError("replay mechanics-before gate lacks q_before")
        failed = []
        if not before_mechanics_pass:
            failed.append("mechanics")
        if not before_force_pass:
            failed.append("force")
        if not before_moment_pass:
            failed.append("moment")
        if not failed or getattr(diag, "failure_reason", None) != (
            "mechanics-before " + "/".join(failed) + " gate failed"
        ):
            raise ValueError("replay mechanics-before failure gate is forged")
    elif stage == "mechanics_after_gate":
        if after is None:
            raise ValueError("replay mechanics-after gate lacks q_after")
        failed = []
        if not after_mechanics_pass:
            failed.append("mechanics")
        if not after_force_pass:
            failed.append("force")
        if not after_moment_pass:
            failed.append("moment")
        if not failed or getattr(diag, "failure_reason", None) != (
            "mechanics-after " + "/".join(failed) + " gate failed"
        ):
            raise ValueError("replay mechanics-after failure gate is forged")
    elif stage == "damage_acceptance":
        if d_next is None:
            raise ValueError("replay damage-acceptance failure lacks d_next")
        range_pass = bool(
            np.min(d_next) >= -config.damage_range_atol
            and np.max(d_next) <= 1.0 + config.damage_range_atol
        )
        healing_pass = bool(
            np.min(d_next - committed_d) >= -config.healing_atol
        )
        failed = []
        if not range_pass:
            failed.append("damage_range")
        if not healing_pass:
            failed.append("healing")
        if not failed or getattr(diag, "failure_reason", None) != ", ".join(failed):
            raise ValueError("replay damage-acceptance failure gate is forged")

    force_error = max(
        float(before["force_error"]) if before is not None else 0.0,
        float(after["force_error"]) if after is not None else 0.0,
    )
    force_scale = max(
        1.0,
        abs(float(before["top"])) if before is not None else 0.0,
        abs(float(after["top"])) if after is not None else 0.0,
        abs(float(before["bottom"])) if before is not None else 0.0,
        abs(float(after["bottom"])) if after is not None else 0.0,
        abs(float(before["anchor"])) if before is not None else 0.0,
        abs(float(after["anchor"])) if after is not None else 0.0,
    )
    moment_error = max(
        float(before["moment_error"]) if before is not None else 0.0,
        float(after["moment_error"]) if after is not None else 0.0,
    )
    geometry_length = max(
        _max_abs(ba["coordinates"]), _max_abs(np.ptp(ba["coordinates"], axis=0)), 1.0
    )
    moment_scale = force_scale * geometry_length
    if d_next is None:
        damage_min = float(np.min(d_iter))
        damage_max = float(np.max(d_iter))
        damage_scale = None
        healing_min = None
        damage_range = False
        healing = False
        rho_used_inf = None
    else:
        assert matrix is not None
        damage_min = float(np.min(d_next))
        damage_max = float(np.max(d_next))
        damage_scale = max(_max_abs(d_iter), _max_abs(d_next))
        healing_min = float(np.min(d_next - committed_d))
        damage_range = bool(
            damage_min >= -config.damage_range_atol
            and damage_max <= 1.0 + config.damage_range_atol
        )
        healing = bool(healing_min >= -config.healing_atol)
        rho_used_inf = _max_abs(rho_used) if rho_used is not None else None
    finite_pass = bool(
        before is not None
        and after is not None
        and before["finite"]
        and after["finite"]
        and all(
            np.all(np.isfinite(values))
            for values in (
                d_iter,
                *(value for value in (psi_before, psi_after, H_used, d_next, rho_used, rho_used_volume) if value is not None),
            )
        )
    )
    expected_flags = {
        "finite_pass": finite_pass,
        "mechanics_before_pass": before_mechanics_pass,
        "mechanics_after_pass": after_mechanics_pass,
        "displacement_pass": False,
        "damage_pass": False,
        "history_pass": False,
        "reaction_pass": False,
        "damage_residual_pass": False,
        "damage_range_pass": damage_range,
        "healing_pass": healing,
        "history_monotonic_pass": False,
        "force_pass": bool(
            (before is not None or after is not None)
            and (before is None or before_force_pass)
            and (after is None or after_force_pass)
        ),
        "moment_pass": bool(
            (before is not None or after is not None)
            and (before is None or before_moment_pass)
            and (after is None or after_moment_pass)
        ),
    }
    for name, expected in expected_flags.items():
        if bool(getattr(diag, name, False)) != bool(expected):
            raise ValueError(f"replay incomplete {name} does not match primitive gates")
    gate_payload = getattr(diag, "gate_passes", None)
    expected_gates = {
        "finite": expected_flags["finite_pass"],
        "mechanics_before": expected_flags["mechanics_before_pass"],
        "mechanics_after": expected_flags["mechanics_after_pass"],
        "displacement": expected_flags["displacement_pass"],
        "damage": expected_flags["damage_pass"],
        "history": expected_flags["history_pass"],
        "reaction": expected_flags["reaction_pass"],
        "damage_residual": expected_flags["damage_residual_pass"],
        "damage_range": expected_flags["damage_range_pass"],
        "healing": expected_flags["healing_pass"],
        "history_monotonic": expected_flags["history_monotonic_pass"],
        "force": expected_flags["force_pass"],
        "moment": expected_flags["moment_pass"],
    }
    if not isinstance(gate_payload, Mapping) or set(gate_payload) != set(expected_gates):
        raise ValueError("replay incomplete gate payload is forged")
    for name, expected in expected_gates.items():
        if bool(gate_payload[name]) != bool(expected):
            raise ValueError(f"replay incomplete gate payload {name} is forged")
    if tuple(getattr(diag, "available_fields", ())) != _replay_expected_available_fields(diag):
        raise ValueError("replay incomplete diagnostic availability is forged")

    expected_scalars: dict[str, Any] = {
        "energy_before": None if before is None else before["energy"],
        "energy_after": None if after is None else after["energy"],
        "top_reaction_before": None if before is None else before["top"],
        "top_reaction_after": None if after is None else after["top"],
        "bottom_reaction_before": None if before is None else before["bottom"],
        "bottom_reaction_after": None if after is None else after["bottom"],
        "anchor_reaction_before": None if before is None else before["anchor"],
        "anchor_reaction_after": None if after is None else after["anchor"],
        "moment_before": None if before is None else before["moment"],
        "moment_after": None if after is None else after["moment"],
        "free_residual_norm_before": None if before is None else before["free_residual_norm"],
        "free_residual_norm_after": None if after is None else after["free_residual_norm"],
        "free_residual_relative_before": None if before is None else before["free_residual_relative_norm"],
        "free_residual_relative_after": None if after is None else after["free_residual_relative_norm"],
        "e_u_material": None,
        "e_u_union": None,
        "e_damage": None if d_next is None else _max_abs(d_next - d_iter),
        "e_history": None,
        "e_reaction": None,
        "damage_min": damage_min,
        "damage_max": damage_max,
        "healing_min": healing_min,
        "history_min_gap": None,
        "rho_used_inf": rho_used_inf,
        "rho_next_inf": None,
        "displacement_scale": None,
        "damage_scale": damage_scale,
        "history_scale": None,
        "reaction_scale": None,
        "damage_residual_scale": None,
        "force_error": None if before is None and after is None else force_error,
        "force_scale": None if before is None and after is None else force_scale,
        "moment_error": None if before is None and after is None else moment_error,
        "moment_scale": None if before is None and after is None else moment_scale,
    }
    for name, expected in expected_scalars.items():
        _compare_replay_scalar(
            getattr(diag, name, None), expected, f"replay incomplete {name}"
        )


def replay_engineering_grip_archive(path: Any) -> dict[str, Any]:
    """Replay crack-region and accepted-step primitives without production APIs.

    Strict v2 archives retain their exact existing contract.  Warning-v1
    archives additionally have every warning decision independently
    reconstructed; ``replayed=True`` reports integrity only and never
    validation, so warning replays always return ``validated=False``.
    """

    archive = load_engineering_grip_archive(path)
    archive_schema = _json_scalar(archive["archive_schema"], "archive_schema")
    warning = _is_warning_archive_schema(archive_schema)
    unbounded = archive_schema == UNBOUNDED_WARNING_ARCHIVE_SCHEMA
    _archive_required(archive, warning=warning, unbounded=unbounded)
    metadata, config, warning, warning_policy, warning_limit = _replay_scalar_fields(
        archive, archive
    )
    material_coordinates = _f64(archive["material_coordinates"], "archive material coordinates")
    union_coordinates = _f64(archive["union_coordinates"], "archive union coordinates")
    if material_coordinates.ndim != 2 or material_coordinates.shape[1] != 2 or len(material_coordinates) == 0:
        raise ValueError("archive material coordinates must have nonempty shape (N, 2)")
    if union_coordinates.ndim != 2 or union_coordinates.shape[1] != 2 or len(union_coordinates) == 0:
        raise ValueError("archive union coordinates must have nonempty shape (N, 2)")
    material_count = len(material_coordinates)
    union_count = len(union_coordinates)
    delta = _scalar(metadata["delta"], "metadata.delta")
    h_g = _scalar(metadata["h_g"], "metadata.h_g")
    visibility_tolerance = _scalar(
        metadata["visibility_tolerance"], "metadata.visibility_tolerance"
    )
    Gc = _scalar(metadata["Gc"], "metadata.Gc")
    l0 = _scalar(metadata["l0"], "metadata.l0")
    E = _scalar(metadata["youngs_modulus"], "metadata.youngs_modulus")
    nu = _scalar(metadata["poisson_ratio"], "metadata.poisson_ratio")
    k_res = _scalar(metadata["k_res"], "metadata.k_res")
    rank_rtol = _scalar(metadata["rank_rtol"], "metadata.rank_rtol")
    if union_count <= material_count or not np.array_equal(
        union_coordinates[:material_count], material_coordinates
    ):
        raise ValueError("engineering-grip archive does not preserve the material prefix")
    material_volumes = _f64(archive["material_volumes"], "archive material volumes", (material_count,))
    material_edges = _i64(archive["material_edges"], "archive material edges")
    union_volumes = _f64(archive["union_volumes"], "archive union volumes", (union_count,))
    material_phase = _f64(
        archive["union_material_phase_volumes"],
        "archive union material phase volumes",
        (union_count,),
    )
    grip_phase = _f64(
        archive["union_grip_phase_volumes"],
        "archive union grip phase volumes",
        (union_count,),
    )
    if not np.array_equal(material_phase[:material_count], material_volumes) or not np.array_equal(
        material_phase[material_count:],
        np.zeros(union_count - material_count, dtype=np.float64),
    ):
        raise ValueError("engineering-grip archive material phase prefix is forged")
    if not np.allclose(union_volumes, material_phase + grip_phase, rtol=0.0, atol=2.0e-12):
        raise ValueError("engineering-grip archive union volume phase sum is forged")
    material_triangles = _i64(archive["material_triangles"], "archive material triangles")
    grip_triangles = _i64(archive["grip_triangles"], "archive grip triangles")
    union_triangles = _i64(archive["union_triangles"], "archive union triangles")
    for name, triangles in (
        ("material", material_triangles),
        ("grip", grip_triangles),
        ("union", union_triangles),
    ):
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise ValueError(f"archive {name} triangles must have nonempty shape (T, 3)")
        if np.any(triangles < 0) or np.any(triangles >= union_count):
            raise ValueError(f"archive {name} triangles contain invalid indices")
        if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
            raise ValueError(f"archive {name} triangles contain duplicates")
    if np.any(material_triangles >= material_count):
        raise ValueError("archive material triangles are not material-only")
    if not np.array_equal(
        union_triangles,
        np.concatenate((material_triangles, grip_triangles), axis=0),
    ):
        raise ValueError("engineering-grip archive union triangles are not phase-concatenated")
    _, expected_material_phase, _ = _triangle_quadrature(
        union_coordinates, material_triangles, "archive material triangles"
    )
    _, expected_grip_phase, _ = _triangle_quadrature(
        union_coordinates, grip_triangles, "archive grip triangles"
    )
    _, expected_union_volumes, _ = _triangle_quadrature(
        union_coordinates, union_triangles, "archive union triangles"
    )
    _array_close(
        material_phase,
        expected_material_phase,
        "archive material phase quadrature",
        2.0e-12,
    )
    _array_close(
        grip_phase,
        expected_grip_phase,
        "archive grip phase quadrature",
        2.0e-12,
    )
    _array_close(
        union_volumes,
        expected_union_volumes,
        "archive union volume quadrature",
        2.0e-12,
    )
    material_anchor = np.asarray(archive["material_anchor"])
    if material_anchor.shape != () or material_anchor.dtype != _INT64 or int(material_anchor) != 0:
        raise ValueError("engineering-grip archive anchor is forged")
    volume_bottom = _bool(archive["volume_bottom_mask"], "archive volume bottom mask", (union_count,))
    volume_top = _bool(archive["volume_top_mask"], "archive volume top mask", (union_count,))
    collar = np.arange(union_count, dtype=np.int64) >= material_count
    expected_bottom = collar & (union_coordinates[:, 1] < 0.0)
    expected_top = collar & (union_coordinates[:, 1] > 1.0)
    if not np.array_equal(volume_bottom, expected_bottom) or not np.array_equal(volume_top, expected_top):
        raise ValueError("engineering-grip archive volume masks are forged")
    if np.any(collar & ~(expected_bottom | expected_top)):
        raise ValueError("engineering-grip archive contains an unclassified collar particle")
    material_bottom = _bool(
        archive["material_bottom_mask"],
        "archive material_bottom_mask (material bottom mask)",
        (material_count,),
    )
    material_top = _bool(
        archive["material_top_mask"],
        "archive material_top_mask (material top mask)",
        (material_count,),
    )
    interface_bottom = _bool(
        archive["interface_bottom_mask"], "archive interface bottom mask", (union_count,)
    )
    interface_top = _bool(
        archive["interface_top_mask"], "archive interface top mask", (union_count,)
    )
    material_prefix = np.arange(union_count, dtype=np.int64) < material_count
    coordinate_tolerance = 1.0e-12 * max(
        1.0, float(np.max(np.abs(union_coordinates)))
    )
    expected_interface_bottom = material_prefix & (
        np.abs(union_coordinates[:, 1]) <= coordinate_tolerance
    )
    expected_interface_top = material_prefix & (
        np.abs(union_coordinates[:, 1] - 1.0) <= coordinate_tolerance
    )
    if not np.array_equal(interface_bottom, expected_interface_bottom) or not np.array_equal(
        interface_top, expected_interface_top
    ):
        raise ValueError("engineering-grip archive material interface masks are forged")
    expected_material_bottom = expected_interface_bottom[:material_count]
    expected_material_top = expected_interface_top[:material_count]
    if not np.array_equal(material_bottom, expected_material_bottom):
        raise ValueError(
            "engineering-grip archive material_bottom_mask does not match coordinates"
        )
    if not np.array_equal(material_top, expected_material_top):
        raise ValueError(
            "engineering-grip archive material_top_mask does not match coordinates"
        )
    if not np.array_equal(
        np.flatnonzero(material_bottom), np.flatnonzero(interface_bottom)
    ) or not np.array_equal(
        np.flatnonzero(material_top), np.flatnonzero(interface_top)
    ):
        raise ValueError(
            "engineering-grip archive material masks do not match interface indices"
        )
    offsets = [h_g]
    offset_index = 2
    while offset_index * h_g < delta:
        offsets.append(float(offset_index * h_g))
        offset_index += 1
    offsets.append(delta)
    expected_offsets = np.asarray(offsets, dtype=np.float64)
    interface_x = {
        "bottom": np.sort(union_coordinates[np.flatnonzero(expected_interface_bottom), 0]),
        "top": np.sort(union_coordinates[np.flatnonzero(expected_interface_top), 0]),
    }
    for side, sign in (("bottom", -1.0), ("top", 1.0)):
        side_mask = expected_bottom if side == "bottom" else expected_top
        interface_y = 0.0 if side == "bottom" else 1.0
        side_coordinates = np.unique(union_coordinates[side_mask, 1])
        actual_offsets = np.sort(sign * (side_coordinates - interface_y))
        if side == "top":
            if actual_offsets.shape != expected_offsets.shape:
                raise ValueError(
                    f"engineering-grip archive {side} collar spacing is forged"
                )
            offset_bound = _top_collar_coordinate_roundoff_bound(
                side_coordinates, np.asarray(interface_y, dtype=np.float64), expected_offsets
            )
            if not np.all(np.abs(actual_offsets - expected_offsets) <= offset_bound):
                raise ValueError(
                    f"engineering-grip archive {side} collar spacing is forged"
                )
        elif not np.array_equal(actual_offsets, expected_offsets):
            raise ValueError(f"engineering-grip archive {side} collar spacing is forged")
        for offset in expected_offsets:
            expected_coordinate = np.asarray(
                interface_y + sign * offset, dtype=np.float64
            )
            if side == "top":
                coordinate_bound = _top_collar_coordinate_roundoff_bound(
                    union_coordinates[:, 1],
                    np.asarray(interface_y, dtype=np.float64),
                    np.asarray(offset, dtype=np.float64),
                )
                row = np.flatnonzero(
                    side_mask
                    & (
                        np.abs(union_coordinates[:, 1] - expected_coordinate)
                        <= coordinate_bound
                    )
                )
            else:
                row = np.flatnonzero(
                    side_mask & (union_coordinates[:, 1] == expected_coordinate)
                )
            if not np.array_equal(
                np.sort(union_coordinates[row, 0]), interface_x[side]
            ):
                raise ValueError(f"engineering-grip archive {side} collar row is forged")
    top_line_indices = _i64(archive["top_line_indices"], "archive top line indices")
    top_line_weights = _f64(archive["top_line_weights"], "archive top line weights")
    bottom_line_indices = _i64(
        archive["bottom_line_indices"], "archive bottom line indices"
    )
    bottom_line_weights = _f64(
        archive["bottom_line_weights"], "archive bottom line weights"
    )

    def _replay_line(mask: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, float]:
        indices = np.flatnonzero(mask).astype(np.int64)
        if len(indices) < 2:
            raise ValueError(f"archive {side} line has too few interface particles")
        order = np.lexsort((union_coordinates[indices, 1], union_coordinates[indices, 0]))
        indices = indices[order]
        spacing = np.diff(union_coordinates[indices, 0])
        if np.any(spacing <= 0.0):
            raise ValueError(f"archive {side} line is not strictly ordered")
        median = float(np.median(spacing))
        if np.any(np.abs(spacing - median) > coordinate_tolerance):
            raise ValueError(f"archive {side} line spacing is not uniform")
        weights = np.empty(len(indices), dtype=np.float64)
        weights[0] = 0.5 * spacing[0]
        weights[-1] = 0.5 * spacing[-1]
        if len(indices) > 2:
            weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
        length = float(np.sum(spacing))
        if not np.isclose(length, 1.0, rtol=0.0, atol=coordinate_tolerance):
            raise ValueError(f"archive {side} line length is not one")
        return indices, weights, length

    expected_top_line = _replay_line(expected_interface_top, "top")
    expected_bottom_line = _replay_line(expected_interface_bottom, "bottom")
    if not np.array_equal(top_line_indices, expected_top_line[0]) or not np.array_equal(
        bottom_line_indices, expected_bottom_line[0]
    ):
        raise ValueError("engineering-grip archive interface line indices are forged")
    _array_close(top_line_weights, expected_top_line[1], "archive top line weights", 2.0e-12)
    _array_close(
        bottom_line_weights, expected_bottom_line[1], "archive bottom line weights", 2.0e-12
    )
    _array_close(
        archive["top_line_length"], np.asarray(expected_top_line[2]), "archive top line length", 2.0e-12
    )
    _array_close(
        archive["bottom_line_length"], np.asarray(expected_bottom_line[2]), "archive bottom line length", 2.0e-12
    )
    constrained = _i64(archive["constrained_dofs"], "archive constrained dofs")
    free = _i64(archive["free_dofs"], "archive free dofs")
    expected_constrained = np.sort(
        np.concatenate(
            (
                2 * np.flatnonzero(expected_bottom) + 1,
                2 * np.flatnonzero(expected_top) + 1,
                np.asarray([0], dtype=np.int64),
            )
        )
    )
    expected_free = np.setdiff1d(
        np.arange(2 * union_count, dtype=np.int64), expected_constrained
    )
    if not np.array_equal(constrained, expected_constrained) or not np.array_equal(free, expected_free):
        raise ValueError("engineering-grip archive DOF constraints are forged")
    visible_edges = _i64(archive["visible_edges"], "archive visible edges")
    candidate_edges = _i64(archive["candidate_edges"], "archive candidate edges")
    removed_edges = _i64(archive["removed_edges"], "archive removed edges")
    expected_polygon, expected_candidate, expected_visible, expected_removed, _ = (
        _build_union_visibility_graph(
            union_coordinates,
            union_triangles,
            delta,
            visibility_tolerance,
            material_count,
            material_edges,
        )
    )
    del expected_polygon
    for name, actual, expected in (
        ("candidate", candidate_edges, expected_candidate),
        ("visible", visible_edges, expected_visible),
        ("removed", removed_edges, expected_removed),
    ):
        if actual.ndim != 2 or actual.shape[1] != 2:
            raise ValueError(f"archive {name} edges must have shape (E, 2)")
        if not np.array_equal(actual, expected):
            raise ValueError(f"engineering-grip archive {name} edge partition is forged")
    ba = _replay_ba(union_coordinates, union_volumes, visible_edges, delta, rank_rtol)
    if not np.array_equal(ba["edges"], visible_edges):
        raise ValueError("engineering-grip replay edge identity changed")
    expected_supports = _padded_replay_supports(ba)
    static_support_indices = _i64(
        archive["static_support_indices"],
        "archive static support indices",
        expected_supports[0].shape,
    )
    static_support_mask = _bool(
        archive["static_support_mask"],
        "archive static_support_mask (static support mask)",
        expected_supports[1].shape,
    )
    static_support_sizes = _i64(
        archive["static_support_sizes"],
        "archive static_support_sizes (static support sizes)",
        (len(ba["supports"]),),
    )
    static_shape_rank = _i64(
        archive["static_shape_rank"],
        "archive static_shape_rank (static shape rank)",
        ba["shape_rank"].shape,
    )
    static_edge_distance = _f64(
        archive["static_edge_distance"],
        "archive static edge distance",
        ba["distance"].shape,
    )
    static_edge_omega = _f64(
        archive["static_edge_omega"],
        "archive static edge omega",
        ba["omega"].shape,
    )
    static_family_mass = _f64(
        archive["static_family_mass"],
        "archive static family mass",
        ba["family"].shape,
    )
    static_eta = _f64(
        archive["static_eta"],
        "archive static eta",
        ba["eta"].shape,
    )
    static_beta = _f64(
        archive["static_beta"],
        "archive static beta",
        expected_supports[2].shape,
    )
    static_xi = _f64(
        archive["static_xi"],
        "archive static xi",
        expected_supports[3].shape,
    )
    static_shape_condition = _f64(
        archive["static_shape_condition"],
        "archive static shape condition",
        ba["shape_condition"].shape,
    )
    inverse = np.asarray(ba["inverse"], dtype=np.float64)
    static_shape_inverse = _f64(
        archive["static_shape_inverse"],
        "archive static shape inverse",
        (len(ba["source"]), 2, 2),
    )
    expected_gradient = np.einsum(
        "es,esb,ebc->esc", expected_supports[2], expected_supports[3], inverse, optimize=True
    )
    static_gradient = _f64(
        archive["static_gradient_coefficient"],
        "archive static gradient coefficient",
        expected_gradient.shape,
    )
    if not np.array_equal(static_support_indices, expected_supports[0]):
        raise ValueError("engineering-grip archive static support indices are forged")
    if not np.array_equal(static_support_mask, expected_supports[1]):
        raise ValueError("engineering-grip archive support mask is forged")
    _array_close(static_beta, expected_supports[2], "archive static beta", 2.0e-12)
    _array_close(static_xi, expected_supports[3], "archive static xi", 2.0e-12)
    _array_close(static_edge_distance, ba["distance"], "archive edge distance", 2.0e-12)
    _array_close(static_edge_omega, ba["omega"], "archive edge omega", 2.0e-12)
    _array_close(static_family_mass, ba["family"], "archive family mass", 2.0e-12)
    _array_close(static_eta, ba["eta"], "archive eta", 2.0e-12)
    if not np.array_equal(
        static_support_sizes,
        np.asarray([len(s) for s in ba["supports"]], dtype=np.int64),
    ):
        raise ValueError("engineering-grip archive support sizes are forged")
    if not np.array_equal(static_shape_rank, ba["shape_rank"]):
        raise ValueError("engineering-grip archive support ranks are forged")
    _array_close(static_shape_condition, ba["shape_condition"], "archive shape condition", 2.0e-12)
    _array_close(static_shape_inverse, inverse, "archive shape inverse", 2.0e-12)
    _array_close(
        static_gradient,
        expected_gradient,
        "archive gradient coefficient",
        2.0e-12,
    )
    expected_static_digest = _replay_static_digest(
        ba,
        material_phase,
        grip_phase,
        material_volumes,
        constrained,
        free,
        volume_bottom,
        volume_top,
        int(material_anchor),
        top_line_indices,
        top_line_weights,
        float(expected_top_line[2]),
        bottom_line_indices,
        bottom_line_weights,
        float(expected_bottom_line[2]),
        E,
        nu,
        k_res,
        rank_rtol,
        delta,
        h_g,
        material_count,
    )
    if metadata.get("static_digest") != expected_static_digest:
        raise ValueError("engineering-grip archive static digest is forged")
    expected_material_hash = _hash_named(
        (
            ("coordinates", material_coordinates),
            ("volumes", material_volumes),
            ("triangles", _i64(archive["material_triangles"], "archive material triangles")),
            ("edges", material_edges),
            ("bottom", material_bottom),
            ("top", material_top),
            ("anchor", 0),
        )
    )
    expected_union_hash = _hash_named(
        (
            ("coordinates", union_coordinates),
            ("material_phase_volumes", material_phase),
            ("grip_phase_volumes", grip_phase),
            ("union_volumes", union_volumes),
            ("union_triangles", _i64(archive["union_triangles"], "archive union triangles")),
            ("candidate_edges", candidate_edges),
            ("visible_edges", visible_edges),
            ("removed_edges", removed_edges),
            ("volume_bottom_mask", volume_bottom),
            ("volume_top_mask", volume_top),
            (
                "visibility_tolerance",
                np.asarray(visibility_tolerance, dtype=np.float64),
            ),
        )
    )
    if metadata.get("material_hash") != expected_material_hash or metadata.get("union_hash") != expected_union_hash:
        raise ValueError("engineering-grip archive cloud hash is forged")
    crack_mask = _bool(archive["crack_region_mask"], "archive crack region mask", (material_count,))
    expected_crack = build_crack_region_mask(material_coordinates, delta)
    if not np.array_equal(crack_mask, expected_crack):
        raise ValueError("engineering-grip archive crack-region mask is forged")
    archive_localization_atol = _f64(
        archive["crack_localization_atol"],
        "archive crack localization tolerance",
        (),
    )
    if float(archive_localization_atol) != _CRACK_LOCALIZATION_ATOL:
        raise ValueError("engineering-grip archive localization tolerance is forged")

    state_u = _f64(archive["state_u_bar"], "archive state u_bar")
    state_q = _f64(archive["state_union_displacement"], "archive state q")
    state_d = _f64(archive["state_material_damage"], "archive state damage")
    state_H = _f64(archive["state_material_history"], "archive state history")
    state_digest = _string_vector(archive["state_digest"], "archive state digest", len(state_u))
    state_status = _string_vector(archive["state_acceptance_status"], "archive state status", len(state_u))
    if state_u.ndim != 1 or len(state_u) == 0 or state_q.shape != (len(state_u), 2 * union_count) or state_d.shape != (len(state_u), material_count) or state_H.shape != state_d.shape:
        raise ValueError("engineering-grip archive state shapes are inconsistent")
    allowed_state_status = (
        {_ACCEPTANCE_STATUS}
        if not warning
        else {_ACCEPTANCE_STATUS, _WARNING_ACCEPTANCE_STATUS}
    )
    if (
        state_status[0] != "initial"
        or not set(str(value) for value in state_status[1:]) <= allowed_state_status
    ):
        raise ValueError("engineering-grip archive state status is forged")
    if (
        state_u[0] != 0.0
        or not np.array_equal(state_q[0], np.zeros(2 * union_count, dtype=np.float64))
        or not np.array_equal(state_d[0], np.zeros(material_count, dtype=np.float64))
        or not np.array_equal(state_H[0], np.zeros(material_count, dtype=np.float64))
    ):
        raise ValueError("engineering-grip archive initial state is not exact zero")
    if state_u[-1] != config.load_final:
        raise ValueError("engineering-grip archive final load is not exact load_final")
    expected_digests = np.asarray(
        [_state_digest(state_u[i], state_q[i], state_d[i], state_H[i]) for i in range(len(state_u))],
        dtype=np.str_,
    )
    if not np.array_equal(state_digest, expected_digests):
        raise ValueError("engineering-grip archive state digest is forged")
    if np.any(np.diff(state_u) <= 0.0):
        raise ValueError("engineering-grip archive state loads are not strictly increasing")
    state_range_bounds = np.full(
        len(state_u), config.damage_range_atol, dtype=np.float64
    )
    state_healing_bounds = np.full(
        len(state_u), config.healing_atol, dtype=np.float64
    )
    warning_state = np.zeros(len(state_u), dtype=bool)
    if warning:
        warning_state[1:] = state_status[1:] == _WARNING_ACCEPTANCE_STATUS
        if warning_policy == _WARNING_SAFETY_POLICY_BOUNDED:
            assert warning_limit is not None
            state_range_bounds[warning_state] = warning_limit
            state_healing_bounds[warning_state] = warning_limit
    for index in range(1, len(state_u)):
        if warning_state[index] and warning_policy == _WARNING_SAFETY_POLICY_UNBOUNDED:
            continue
        if np.any(state_d[index] < -state_range_bounds[index]) or np.any(
            state_d[index] > 1.0 + state_range_bounds[index]
        ):
            raise ValueError(
                "engineering-grip archive state damage is outside its hard range"
            )
    if np.any(state_H < 0.0):
        raise ValueError("engineering-grip archive state history is negative")
    for index in range(1, len(state_u)):
        if not (
            warning_state[index]
            and warning_policy == _WARNING_SAFETY_POLICY_UNBOUNDED
        ) and np.any(
            state_d[index] - state_d[index - 1] < -state_healing_bounds[index]
        ):
            raise ValueError("engineering-grip archive state irreversibility is forged")
        if np.any(state_H[index] - state_H[index - 1] < 0.0):
            raise ValueError("engineering-grip archive state irreversibility is forged")

    state_material_strain = _f64(
        archive["state_material_particle_strain"],
        "archive state material strain",
        (len(state_u), material_count, 2, 2),
    )
    state_amor = _f64(
        archive["state_material_amor_driver"],
        "archive state material Amor",
        (len(state_u), material_count),
    )
    state_material_displacement = _f64(
        archive["state_material_displacement"],
        "archive state material displacement",
        (len(state_u), material_count, 2),
    )
    state_interface_opening = _f64(
        archive["state_physical_interface_opening"],
        "archive state physical interface opening",
        (len(state_u),),
    )
    state_interface_damage_bottom = _f64(
        archive["state_interface_damage_bottom_max"],
        "archive state interface bottom damage",
        (len(state_u),),
    )
    state_interface_damage_top = _f64(
        archive["state_interface_damage_top_max"],
        "archive state interface top damage",
        (len(state_u),),
    )
    state_interface_history_bottom = _f64(
        archive["state_interface_history_bottom_max"],
        "archive state interface bottom history",
        (len(state_u),),
    )
    state_interface_history_top = _f64(
        archive["state_interface_history_top_max"],
        "archive state interface top history",
        (len(state_u),),
    )
    material_interface_bottom = expected_interface_bottom[:material_count]
    material_interface_top = expected_interface_top[:material_count]
    for index in range(len(state_u)):
        q = state_q[index]
        d = state_d[index]
        prescribed = np.zeros(2 * union_count, dtype=np.float64)
        prescribed[2 * np.flatnonzero(volume_top) + 1] = state_u[index]
        if not np.array_equal(q[constrained], prescribed[constrained]):
            raise ValueError("engineering-grip archive state constraints are forged")
        expected_material_displacement = q[: 2 * material_count].reshape(
            material_count, 2
        )
        expected_strain = np.zeros((material_count, 2, 2), dtype=np.float64)
        expected_amor = np.zeros(material_count, dtype=np.float64)
        if index == 0:
            # There is no production mechanics result for the committed zero
            # state.  Validate its derived fields analytically instead of
            # forming F=K K^-1 and turning identity roundoff into a fake
            # equilibrium failure.
            expected_material_displacement = np.zeros((material_count, 2), dtype=np.float64)
            expected_strain = np.zeros(
                (material_count, 2, 2), dtype=np.float64
            )
            expected_amor = np.zeros(material_count, dtype=np.float64)
        else:
            center = _replay_center_weights(
                d, material_phase, grip_phase, k_res, material_count
            )
            response = _replay_response(q, d, ba, center, E, nu, material_count)
            expected_amor, _, expected_strain = _replay_amor(
                q, state_H[index], ba, E, nu, material_count
            )
        if index == 0:
            if not np.array_equal(
                state_material_displacement[index], expected_material_displacement
            ) or not np.array_equal(state_material_strain[index], expected_strain) or not np.array_equal(
                state_amor[index], expected_amor
            ):
                raise ValueError(
                    "engineering-grip archive initial derived fields are not exact zero"
                )
        else:
            _array_close(
                state_material_displacement[index],
                q[: 2 * material_count].reshape(material_count, 2),
                "state material displacement",
                2.0e-9,
            )
            _array_close(
                state_material_strain[index], expected_strain, "state material strain", 2.0e-9
            )
            _array_close(
                state_amor[index], expected_amor, "state material Amor", 2.0e-9
            )
        expected_opening = _interface_opening_from_arrays(
            q,
            top_line_indices,
            top_line_weights,
            expected_top_line[2],
            bottom_line_indices,
            bottom_line_weights,
            expected_bottom_line[2],
        )
        _array_close(
            state_interface_opening[index],
            np.asarray(expected_opening),
            "state interface opening",
            2.0e-9,
        )
        for name, actual, expected in (
            (
                "state interface bottom damage",
                state_interface_damage_bottom[index],
                np.max(d[material_interface_bottom]),
            ),
            (
                "state interface top damage",
                state_interface_damage_top[index],
                np.max(d[material_interface_top]),
            ),
            (
                "state interface bottom history",
                state_interface_history_bottom[index],
                np.max(state_H[index][material_interface_bottom]),
            ),
            (
                "state interface top history",
                state_interface_history_top[index],
                np.max(state_H[index][material_interface_top]),
            ),
        ):
            if index == 0 and float(actual) != 0.0:
                raise ValueError(
                    f"engineering-grip archive initial {name} is not exact zero"
                )
            _array_close(actual, np.asarray(expected), name, 2.0e-9)
    state_fields = {
        name: _f64(archive[name], name)
        for name in (
            "state_crack_damage_inside_max",
            "state_crack_damage_outside_max",
            "state_crack_history_inside_max",
            "state_crack_history_outside_max",
            "state_crack_strain_inside_max",
            "state_crack_strain_outside_max",
            "state_crack_amor_inside_max",
            "state_crack_amor_outside_max",
        )
    }
    state_localization = _bool(
        archive["state_crack_localization_pass"],
        "archive state crack localization status",
        (len(state_u),),
    )
    state_crack_strain = _f64(
        archive["state_crack_particle_strain"],
        "archive state crack strain",
        (len(state_u), int(np.count_nonzero(crack_mask)), 2, 2),
    )
    state_crack_amor = _f64(
        archive["state_crack_amor_driver"],
        "archive state crack Amor",
        (len(state_u), int(np.count_nonzero(crack_mask))),
    )
    for index in range(len(state_u)):
        q = state_q[index]
        d = state_d[index]
        diag = compute_crack_region_diagnostics(
            material_coordinates,
            delta,
            state_d[index],
            state_H[index],
            state_material_strain[index],
            state_amor[index],
            (expected_interface_bottom | expected_interface_top)[:material_count],
            allow_negative_damage=warning,
        )
        if not np.array_equal(state_crack_strain[index], state_material_strain[index][crack_mask]) or not np.array_equal(state_crack_amor[index], state_amor[index][crack_mask]):
            raise ValueError("engineering-grip archive crack fields are forged")
        expected = (
            diag.damage_inside_max,
            diag.damage_outside_max,
            diag.history_inside_max,
            diag.history_outside_max,
            diag.strain_inside_max,
            diag.strain_outside_max,
            diag.amor_inside_max,
            diag.amor_outside_max,
        )
        actual = tuple(float(state_fields[name][index]) for name in state_fields)
        if any(abs(left - right) > 2.0e-9 * max(abs(left), abs(right), 1.0) for left, right in zip(actual, expected)):
            raise ValueError("engineering-grip archive crack diagnostics are forged")
        if bool(state_localization[index]) != diag.localization_pass:
            raise ValueError("engineering-grip archive crack localization status is forged")
        if index == 0:
            # The exact initial committed state is an analytic zero solution;
            # it has no solved mechanics result whose loop-order residual can
            # be used as an equilibrium gate.
            continue
        mechanics_state = _replay_mechanics_at_state(
            q,
            d,
            float(state_u[index]),
            ba,
            material_phase,
            grip_phase,
            volume_bottom,
            volume_top,
            free,
            0,
            E,
            nu,
            k_res,
            material_count,
            config,
        )
        if not mechanics_state["mechanics_pass"]:
            raise ValueError(
                "engineering-grip archive state free residual/equilibrium gate failed"
            )
        # Force and moment are warning-bypassable final-sweep gates.  A
        # warning-accepted state may therefore retain either independent
        # failure, while strict-passed states and all strict archives still
        # require both conservation gates at the committed state.
        warning_accepted_state = (
            warning and state_status[index] == _WARNING_ACCEPTANCE_STATUS
        )
        if not warning_accepted_state and not mechanics_state["force_pass"]:
            raise ValueError("engineering-grip archive state force gate failed")
        if not warning_accepted_state and not mechanics_state["moment_pass"]:
            raise ValueError("engineering-grip archive state moment gate failed")
    if not warning and not bool(state_localization[-1]):
        raise ValueError("engineering-grip archive final crack localization gate failed")

    accepted_target = _f64(archive["accepted_load_target"], "archive accepted target")
    if accepted_target.ndim != 1:
        raise ValueError("engineering-grip archive accepted target must be one-dimensional")
    accepted_count = len(accepted_target)
    if accepted_count != len(state_u) - 1:
        raise ValueError("engineering-grip archive accepted/state counts are inconsistent")
    accepted_cutback = _i64(
        archive["accepted_cutback_count"],
        "archive accepted cutback count",
        (accepted_count,),
    )
    accepted_stagger = _i64(
        archive["accepted_stagger_iterations"],
        "archive accepted stagger iterations",
        (accepted_count,),
    )
    if np.any(accepted_cutback < 0) or np.any(accepted_stagger <= 0):
        raise ValueError("engineering-grip archive accepted iteration counts are invalid")
    accepted_specs = {
        "accepted_load_start": (accepted_count,),
        "accepted_damage_before": (accepted_count, material_count),
        "accepted_d_iter": (accepted_count, material_count),
        "accepted_d_next": (accepted_count, material_count),
        "accepted_q_before": (accepted_count, 2 * union_count),
        "accepted_q_after": (accepted_count, 2 * union_count),
        "accepted_psi_before": (accepted_count, material_count),
        "accepted_psi_after": (accepted_count, material_count),
        "accepted_H_used": (accepted_count, material_count),
        "accepted_H_next": (accepted_count, material_count),
        "accepted_rho_used": (accepted_count, material_count),
        "accepted_rho_next": (accepted_count, material_count),
        "accepted_rho_used_volume": (accepted_count, material_count),
        "accepted_rho_next_volume": (accepted_count, material_count),
        "accepted_energy_before": (accepted_count,),
        "accepted_energy_after": (accepted_count,),
        "accepted_top_reaction_before": (accepted_count,),
        "accepted_top_reaction_after": (accepted_count,),
        "accepted_bottom_reaction_before": (accepted_count,),
        "accepted_bottom_reaction_after": (accepted_count,),
        "accepted_anchor_reaction_before": (accepted_count,),
        "accepted_anchor_reaction_after": (accepted_count,),
        "accepted_force_before": (accepted_count, 2),
        "accepted_force_after": (accepted_count, 2),
        "accepted_moment_before": (accepted_count,),
        "accepted_moment_after": (accepted_count,),
        "accepted_interface_opening_before": (accepted_count,),
        "accepted_interface_opening_after": (accepted_count,),
    }
    accepted_arrays = {
        name: _f64(archive[name], name, shape)
        for name, shape in accepted_specs.items()
    }
    accepted_integer_specs = {
        "accepted_cg_info_before": (accepted_count,),
        "accepted_cg_info_after": (accepted_count,),
    }
    accepted_arrays.update(
        {
            name: _i64(archive[name], name, shape)
            for name, shape in accepted_integer_specs.items()
        }
    )
    accepted_diag_json = _string_vector(
        archive["accepted_diagnostics_json"],
        "accepted diagnostics",
        accepted_count,
    )
    attempt_count = len(np.asarray(archive["attempt_load_target"]))
    if warning:
        warning_gate_names = _string_vector(
            archive["warning_gate_names"],
            "warning gate names",
            len(WARNING_GATE_NAMES),
        )
        if tuple(str(name) for name in warning_gate_names) != WARNING_GATE_NAMES:
            raise ValueError("engineering-grip warning gate names are forged")
        accepted_acceptance_status = _string_vector(
            archive["accepted_acceptance_status"],
            "accepted warning acceptance status",
            accepted_count,
        )
        accepted_warning = _bool(
            archive["accepted_warning"],
            "accepted warning status",
            (accepted_count,),
        )
        accepted_bypassed_gate_mask = _bool(
            archive["accepted_bypassed_gate_mask"],
            "accepted warning bypassed-gate mask",
            (accepted_count, len(WARNING_GATE_NAMES)),
        )
        accepted_warning_range_excess = _f64(
            archive["accepted_warning_range_excess"],
            "accepted warning range excess",
            (accepted_count,),
        )
        accepted_warning_healing_magnitude = _f64(
            archive["accepted_warning_healing_magnitude"],
            "accepted warning healing magnitude",
            (accepted_count,),
        )
        attempt_acceptance_status = _string_vector(
            archive["attempt_acceptance_status"],
            "attempt warning acceptance status",
            attempt_count,
        )
        attempt_bypassed_gate_mask = _bool(
            archive["attempt_bypassed_gate_mask"],
            "attempt warning bypassed-gate mask",
            (attempt_count, len(WARNING_GATE_NAMES)),
        )
        attempt_warning_excess_present = _bool(
            archive["attempt_warning_excess_present"],
            "attempt warning excess presence",
            (attempt_count,),
        )
        attempt_warning_range_excess = _f64(
            archive["attempt_warning_range_excess"],
            "attempt warning range excess",
            (attempt_count,),
        )
        attempt_warning_healing_magnitude = _f64(
            archive["attempt_warning_healing_magnitude"],
            "attempt warning healing magnitude",
            (attempt_count,),
        )
    else:
        accepted_acceptance_status = None
        accepted_warning = None
        accepted_bypassed_gate_mask = None
        accepted_warning_range_excess = None
        accepted_warning_healing_magnitude = None
        attempt_acceptance_status = None
        attempt_bypassed_gate_mask = None
        attempt_warning_excess_present = None
        attempt_warning_range_excess = None
        attempt_warning_healing_magnitude = None

    # Replay the one-time zero-damage work identity independently with fixed-load
    # solves.  It is intentionally not part of the stagger loop.
    work_load = _scalar(archive["work_check_u_bar"], "archive work check load")
    work_epsilon = _scalar(archive["work_check_epsilon"], "archive work check epsilon")
    work_limit = _scalar(
        archive["work_check_relative_limit"], "archive work check relative limit"
    )
    if (
        work_load != config.work_check_load
        or work_epsilon != config.work_epsilon
        or work_limit != config.work_relative_limit
    ):
        raise ValueError(
            "engineering-grip archive work-check settings do not match canonical config"
        )
    work_center = _replay_direct_solve(
        np.zeros(material_count, dtype=np.float64),
        work_load,
        ba,
        material_phase,
        grip_phase,
        volume_bottom,
        volume_top,
        0,
        E,
        nu,
        k_res,
        material_count,
        cg_rtol=config.cg_rtol,
        cg_atol=config.cg_atol,
        cg_maxiter=config.cg_maxiter,
    )
    work_plus = _replay_direct_solve(
        np.zeros(material_count, dtype=np.float64),
        work_load + work_epsilon,
        ba,
        material_phase,
        grip_phase,
        volume_bottom,
        volume_top,
        0,
        E,
        nu,
        k_res,
        material_count,
        cg_rtol=config.cg_rtol,
        cg_atol=config.cg_atol,
        cg_maxiter=config.cg_maxiter,
    )
    work_minus = _replay_direct_solve(
        np.zeros(material_count, dtype=np.float64),
        work_load - work_epsilon,
        ba,
        material_phase,
        grip_phase,
        volume_bottom,
        volume_top,
        0,
        E,
        nu,
        k_res,
        material_count,
        cg_rtol=config.cg_rtol,
        cg_atol=config.cg_atol,
        cg_maxiter=config.cg_maxiter,
    )
    work_fd = (
        work_plus["response"]["energy"] - work_minus["response"]["energy"]
    ) / (2.0 * work_epsilon)
    work_error = _relative(work_fd, work_center["top"])
    _array_close(
        archive["work_check_reaction"],
        np.asarray(work_center["top"]),
        "archive work reaction",
        2.0e-8,
    )
    _array_close(
        archive["work_check_reaction_fd"],
        np.asarray(work_fd),
        "archive work reaction FD",
        2.0e-8,
    )
    _array_close(
        archive["work_check_relative_error"],
        np.asarray(work_error),
        "archive work error",
        2.0e-8,
    )
    work_passed = _bool(archive["work_check_passed"], "archive work status", ())
    expected_work_passed = bool(work_error <= config.work_relative_limit)
    if not expected_work_passed:
        raise ValueError(
            "engineering-grip archive recomputed work-check failed: "
            f"error={work_error:.17g}, limit={config.work_relative_limit:.17g}"
        )
    if bool(work_passed) != expected_work_passed:
        raise ValueError("engineering-grip archive work-check status is forged")

    attempts_before = _string_vector(
        archive["attempt_before_digest"],
        "attempt before digest",
        len(archive["attempt_before_digest"]),
    )
    attempt_count = len(attempts_before)
    attempts_after = _string_vector(
        archive["attempt_after_digest"], "attempt after digest", attempt_count
    )
    attempt_cutback = _i64(
        archive["attempt_cutback_count"],
        "archive attempt cutback count",
        (attempt_count,),
    )
    attempt_target = _f64(
        archive["attempt_load_target"], "attempt target", (attempt_count,)
    )
    attempt_start = _f64(
        archive["attempt_load_start"], "attempt start", (attempt_count,)
    )
    attempt_accepted = _bool(
        archive["attempt_accepted"], "attempt accepted", (attempt_count,)
    )
    attempt_diag_json = _string_vector(
        archive["attempt_diagnostics_json"],
        "attempt diagnostics",
        attempt_count,
    )
    attempt_sweeps_json = _string_vector(
        archive["attempt_sweeps_json"],
        "attempt sweep diagnostics",
        attempt_count,
    )
    attempt_sweeps = _i64(
        archive["attempt_sweep_count"],
        "archive attempt sweep count",
        (attempt_count,),
    )
    attempt_failure_stage = _string_vector(
        archive["attempt_failure_stage"],
        "attempt failure stage",
        attempt_count,
    )
    attempt_failure_reason = _string_vector(
        archive["attempt_failure_reason"],
        "attempt failure reason",
        attempt_count,
    )
    attempt_exception = _string_vector(
        archive["attempt_exception_type"],
        "attempt exception",
        attempt_count,
    )
    if (
        np.any(attempt_cutback < 0)
        or np.any(attempt_sweeps <= 0)
        or np.any(attempt_cutback > config.max_cutbacks)
        or np.any(attempt_sweeps > config.max_stagger_iterations)
    ):
        raise ValueError("engineering-grip archive attempt arrays are inconsistent")
    if attempt_count == 0:
        raise ValueError("engineering-grip archive has no attempt audit")

    accepted_array_fields = {
        "damage_before": "accepted_damage_before",
        "d_iter": "accepted_d_iter",
        "d_next": "accepted_d_next",
        "q_before": "accepted_q_before",
        "q_after": "accepted_q_after",
        "psi_before": "accepted_psi_before",
        "psi_after": "accepted_psi_after",
        "H_used": "accepted_H_used",
        "H_next": "accepted_H_next",
        "rho_used": "accepted_rho_used",
        "rho_next": "accepted_rho_next",
        "rho_used_volume": "accepted_rho_used_volume",
        "rho_next_volume": "accepted_rho_next_volume",
    }
    accepted_scalar_fields = {
        "energy_before": "accepted_energy_before",
        "energy_after": "accepted_energy_after",
        "top_reaction_before": "accepted_top_reaction_before",
        "top_reaction_after": "accepted_top_reaction_after",
        "bottom_reaction_before": "accepted_bottom_reaction_before",
        "bottom_reaction_after": "accepted_bottom_reaction_after",
        "anchor_reaction_before": "accepted_anchor_reaction_before",
        "anchor_reaction_after": "accepted_anchor_reaction_after",
        "cg_info_before": "accepted_cg_info_before",
        "cg_info_after": "accepted_cg_info_after",
        "moment_before": "accepted_moment_before",
        "moment_after": "accepted_moment_after",
    }
    current = 0
    accepted_index = 0
    expected_cutback_count = 0
    expected_increment = config.du_initial
    for attempt_index in range(attempt_count):
        if current >= len(state_u):
            raise ValueError("engineering-grip archive attempt state index overflow")
        if (
            str(attempts_before[attempt_index]) != str(state_digest[current])
            or attempt_start[attempt_index] != state_u[current]
        ):
            raise ValueError("engineering-grip archive attempt before pairing is forged")
        if attempt_cutback[attempt_index] != expected_cutback_count:
            raise ValueError(
                "engineering-grip archive attempt cutback count is not paired"
            )
        target = float(attempt_target[attempt_index])
        expected_target = _next_target(
            float(state_u[current]), expected_increment, config.load_final
        )
        if target != expected_target:
            raise ValueError("engineering-grip archive attempt target schedule is forged")
        if target <= attempt_start[attempt_index] or target > config.load_final:
            raise ValueError("engineering-grip archive attempt interval is invalid")
        sweep_payloads_raw = json.loads(str(attempt_sweeps_json[attempt_index]))
        if not isinstance(sweep_payloads_raw, list) or not sweep_payloads_raw:
            raise ValueError("engineering-grip archive attempt has no sweep audit")
        if len(sweep_payloads_raw) != int(attempt_sweeps[attempt_index]):
            raise ValueError("engineering-grip archive attempt sweep count is forged")
        local_damage = np.array(state_d[current], dtype=np.float64, copy=True)
        sweep_diagnostics = []
        for sweep_index, sweep_payload in enumerate(sweep_payloads_raw):
            diag = _diag_namespace(sweep_payload)
            sweep_diagnostics.append(diag)
            if getattr(diag, "sweep_index", None) != sweep_index:
                raise ValueError("engineering-grip archive sweep indices are not contiguous")
            if getattr(diag, "target_u_bar", None) != target:
                raise ValueError("engineering-grip archive sweep target is not paired")
            if bool(getattr(diag, "complete_sweep", False)):
                local_damage = _replay_complete_sweep(
                    diag,
                    state_d[current],
                    state_H[current],
                    local_damage,
                    target,
                    ba,
                    material_coordinates,
                    material_volumes,
                    material_edges,
                    material_phase,
                    grip_phase,
                    volume_bottom,
                    volume_top,
                    free,
                    E,
                    nu,
                    k_res,
                    Gc,
                    l0,
                    delta,
                    material_count,
                    config,
                    warning_safety_limit=warning_limit if warning else None,
                    final_sweep=sweep_index == len(sweep_payloads_raw) - 1,
                    warning_safety_policy=warning_policy if warning else None,
                )
                _validate_replay_sweep_termination(
                    diag, sweep_index, len(sweep_payloads_raw)
                )
            else:
                if sweep_index != len(sweep_payloads_raw) - 1:
                    raise ValueError("incomplete sweep is not the final sweep")
                _replay_incomplete_sweep(
                    diag,
                    state_d[current],
                    state_H[current],
                    local_damage,
                    target,
                    ba,
                    material_coordinates,
                    material_volumes,
                    material_edges,
                    material_phase,
                    grip_phase,
                    volume_bottom,
                    volume_top,
                    free,
                    E,
                    nu,
                    k_res,
                    Gc,
                    l0,
                    delta,
                    material_count,
                    config,
                    warning_mode=warning,
                )
        final_diag = sweep_diagnostics[-1]
        attempt_diag = _diag_namespace(
            json.loads(str(attempt_diag_json[attempt_index]))
        )
        _compare_diagnostic_records(
            final_diag,
            attempt_diag,
            "engineering-grip archive attempt/final diagnostic",
        )
        if (
            final_diag.complete_sweep
            and not final_diag.passed
            and len(sweep_payloads_raw) != config.max_stagger_iterations
        ):
            raise ValueError(
                "engineering-grip rejected complete attempt must retain the full "
                "stagger budget"
            )
        expected_attempt_accepted = bool(final_diag.passed)
        if warning:
            if final_diag.complete_sweep:
                final_d_next = getattr(final_diag, "d_next", None)
                if final_d_next is None:
                    raise ValueError(
                        "engineering-grip warning complete final sweep is missing d_next"
                    )
                (
                    warning_decision,
                    expected_bypassed_gates,
                    expected_range_excess,
                    expected_healing_magnitude,
                ) = _warning_acceptance_decision(
                    final_diag,
                    state_d[current],
                    warning_limit,
                    warning_safety_policy=warning_policy,
                )
                expected_attempt_accepted = bool(
                    final_diag.passed or warning_decision
                )
                expected_attempt_status = (
                    _ACCEPTANCE_STATUS
                    if final_diag.passed
                    else (
                        _WARNING_ACCEPTANCE_STATUS
                        if warning_decision
                        else ""
                    )
                )
                expected_warning_excess_present = True
            else:
                expected_bypassed_gates = ()
                expected_range_excess = None
                expected_healing_magnitude = None
                expected_attempt_status = ""
                expected_warning_excess_present = False
            expected_attempt_mask = np.asarray(
                [
                    name in set(expected_bypassed_gates)
                    for name in WARNING_GATE_NAMES
                ],
                dtype=bool,
            )
            assert attempt_acceptance_status is not None
            assert attempt_bypassed_gate_mask is not None
            assert attempt_warning_excess_present is not None
            assert attempt_warning_range_excess is not None
            assert attempt_warning_healing_magnitude is not None
            if (
                str(attempt_acceptance_status[attempt_index])
                != expected_attempt_status
                or not np.array_equal(
                    attempt_bypassed_gate_mask[attempt_index], expected_attempt_mask
                )
                or bool(attempt_warning_excess_present[attempt_index])
                != expected_warning_excess_present
            ):
                raise ValueError(
                    "engineering-grip warning attempt acceptance audit is forged"
                )
            if expected_warning_excess_present:
                assert expected_range_excess is not None
                assert expected_healing_magnitude is not None
                _compare_replay_scalar(
                    attempt_warning_range_excess[attempt_index],
                    expected_range_excess,
                    "engineering-grip attempt warning range excess",
                    0.0,
                    absolute_atol=2.0e-12,
                    scale_floor=0.0,
                )
                _compare_replay_scalar(
                    attempt_warning_healing_magnitude[attempt_index],
                    expected_healing_magnitude,
                    "engineering-grip attempt warning healing magnitude",
                    0.0,
                    absolute_atol=2.0e-12,
                    scale_floor=0.0,
                )
            elif (
                attempt_warning_range_excess[attempt_index] != 0.0
                or attempt_warning_healing_magnitude[attempt_index] != 0.0
            ):
                raise ValueError(
                    "engineering-grip incomplete warning attempt excesses are forged"
                )
        else:
            expected_attempt_status = None
            expected_attempt_mask = None
        if bool(attempt_accepted[attempt_index]) != expected_attempt_accepted:
            raise ValueError("engineering-grip archive attempt status is forged")
        if bool(attempt_accepted[attempt_index]):
            if (
                str(attempt_failure_stage[attempt_index])
                or str(attempt_failure_reason[attempt_index])
                or str(attempt_exception[attempt_index])
            ):
                raise ValueError(
                    "engineering-grip archive accepted attempt classification is forged"
                )
            if accepted_index >= accepted_count:
                raise ValueError("engineering-grip archive accepted count exceeds states")
            if str(attempts_after[attempt_index]) != str(state_digest[current + 1]):
                raise ValueError(
                    "engineering-grip archive accepted attempt after pairing is forged"
                )
            if (
                accepted_cutback[accepted_index] != attempt_cutback[attempt_index]
                or accepted_stagger[accepted_index] != attempt_sweeps[attempt_index]
            ):
                raise ValueError("engineering-grip archive accepted iteration audit is forged")
            if warning:
                assert accepted_acceptance_status is not None
                assert accepted_warning is not None
                assert accepted_bypassed_gate_mask is not None
                assert accepted_warning_range_excess is not None
                assert accepted_warning_healing_magnitude is not None
                assert attempt_warning_range_excess is not None
                assert attempt_warning_healing_magnitude is not None
                assert expected_attempt_mask is not None
                if (
                    str(accepted_acceptance_status[accepted_index])
                    != expected_attempt_status
                    or bool(accepted_warning[accepted_index])
                    != (expected_attempt_status == _WARNING_ACCEPTANCE_STATUS)
                    or not np.array_equal(
                        accepted_bypassed_gate_mask[accepted_index],
                        expected_attempt_mask,
                    )
                    or accepted_warning_range_excess[accepted_index]
                    != attempt_warning_range_excess[attempt_index]
                    or accepted_warning_healing_magnitude[accepted_index]
                    != attempt_warning_healing_magnitude[attempt_index]
                    or state_status[current + 1] != expected_attempt_status
                ):
                    raise ValueError(
                        "engineering-grip warning accepted-state audit is forged"
                    )
                if expected_attempt_status == _WARNING_ACCEPTANCE_STATUS:
                    if accepted_stagger[accepted_index] != config.max_stagger_iterations:
                        raise ValueError(
                            "engineering-grip warning acceptance did not use the full "
                            "stagger budget"
                        )
            accepted_diag = _diag_namespace(
                json.loads(str(accepted_diag_json[accepted_index]))
            )
            _compare_diagnostic_records(
                final_diag,
                accepted_diag,
                "engineering-grip archive accepted diagnostic",
            )
            if (
                accepted_arrays["accepted_load_start"][accepted_index]
                != state_u[current]
                or accepted_target[accepted_index] != state_u[current + 1]
            ):
                raise ValueError("engineering-grip archive accepted load pairing is forged")
            for field, archive_name in accepted_array_fields.items():
                actual = accepted_arrays[archive_name][accepted_index]
                expected = getattr(final_diag, field, None)
                if expected is None or not np.array_equal(actual, expected):
                    raise ValueError(
                        f"engineering-grip archive accepted {field} is not paired with final sweep"
                    )
            for field, archive_name in accepted_scalar_fields.items():
                _compare_replay_scalar(
                    accepted_arrays[archive_name][accepted_index],
                    getattr(final_diag, field, None),
                    f"engineering-grip archive accepted {field}",
                    2.0e-12,
                )
            if not np.array_equal(
                accepted_arrays["accepted_q_after"][accepted_index], state_q[current + 1]
            ):
                raise ValueError("engineering-grip archive accepted q_after/state pairing is forged")
            if not np.array_equal(
                accepted_arrays["accepted_d_next"][accepted_index], state_d[current + 1]
            ) or not np.array_equal(
                accepted_arrays["accepted_H_next"][accepted_index], state_H[current + 1]
            ):
                raise ValueError(
                    "engineering-grip archive accepted damage/history state pairing is forged"
                )
            if not np.array_equal(
                accepted_arrays["accepted_damage_before"][accepted_index], state_d[current]
            ):
                raise ValueError(
                    "engineering-grip archive accepted damage_before preceding-state pairing is forged"
                )
            for phase, q_name in (
                ("before", "accepted_q_before"),
                ("after", "accepted_q_after"),
            ):
                q_value = accepted_arrays[q_name][accepted_index]
                opening = _interface_opening_from_arrays(
                    q_value,
                    top_line_indices,
                    top_line_weights,
                    expected_top_line[2],
                    bottom_line_indices,
                    bottom_line_weights,
                    expected_bottom_line[2],
                )
                _array_close(
                    accepted_arrays[f"accepted_interface_opening_{phase}"][accepted_index],
                    np.asarray(opening),
                    f"accepted interface opening {phase}",
                    2.0e-9,
                )
            accepted_index += 1
            current += 1
            expected_cutback_count = 0
            expected_increment = target - float(attempt_start[attempt_index])
        else:
            if str(attempts_after[attempt_index]) != str(state_digest[current]):
                raise ValueError("engineering-grip archive rejected attempt rollback is forged")
            if not str(attempt_failure_stage[attempt_index]) or not str(
                attempt_failure_reason[attempt_index]
            ):
                raise ValueError(
                    "engineering-grip archive rejected failure classification is missing"
                )
            if final_diag.failure_stage != str(attempt_failure_stage[attempt_index]) or (
                final_diag.failure_reason != str(attempt_failure_reason[attempt_index])
            ):
                raise ValueError(
                    "engineering-grip archive rejected failure classification is forged"
                )
            exception = str(attempt_exception[attempt_index])
            if final_diag.complete_sweep:
                if exception:
                    raise ValueError(
                        "engineering-grip complete failed attempt has an exception type"
                    )
            elif exception != EngineeringGripSweepFailure.__name__:
                raise ValueError(
                    "engineering-grip incomplete attempt exception type is forged"
                )
            trial_increment = target - float(attempt_start[attempt_index])
            if (
                _is_minimum_increment(
                    trial_increment,
                    config.du_min,
                    float(attempt_start[attempt_index]),
                    target,
                )
                or attempt_cutback[attempt_index] >= config.max_cutbacks
            ):
                raise ValueError(
                    "engineering-grip archive rejected attempt exceeds the cutback schedule"
                )
            expected_increment = max(
                config.du_min, trial_increment * config.cutback_factor
            )
            expected_cutback_count += 1
    if accepted_index != len(state_u) - 1 or current != len(state_u) - 1:
        raise ValueError("engineering-grip archive attempts do not reach final state")
    if warning:
        reported_status = _json_scalar(
            archive["acceptance_status"], "acceptance_status"
        )
        expected_status = (
            _WARNING_ACCEPTANCE_STATUS
            if any(
                str(value) == _WARNING_ACCEPTANCE_STATUS for value in state_status[1:]
            )
            else _ACCEPTANCE_STATUS
        )
        if reported_status != expected_status:
            raise ValueError(
                "engineering-grip warning archive acceptance status does not "
                "match its accepted-state statuses"
            )
    return {
        "schema": (
            UNBOUNDED_WARNING_ARCHIVE_SCHEMA
            if unbounded
            else WARNING_ARCHIVE_SCHEMA
            if warning
            else ARCHIVE_SCHEMA
        ),
        "acceptance_mode": _WARNING_ACCEPTANCE_MODE if warning else _ACCEPTANCE_MODE,
        "warning_safety_policy": warning_policy,
        "warning_safety_limit": warning_limit,
        "warning_safety_limit_present": (
            warning_limit is not None if warning else False
        ),
        "acceptance_status": _json_scalar(
            archive["acceptance_status"], "acceptance_status"
        ),
        "completion_status": _WARNING_COMPLETION_STATUS
        if warning
        else _COMPLETION_STATUS,
        "validated": False if warning else True,
        "accepted_steps": accepted_index,
        "attempts": attempt_count,
        "crack_region_particles": int(np.count_nonzero(crack_mask)),
        "replayed": True,
    }
