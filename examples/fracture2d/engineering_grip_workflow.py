"""Durable run, replay, analysis, and plotting workflow for engineering grips.

The numerical formulation lives in :mod:`nonlocal_engineering_grip`.  This
module is deliberately a thin orchestration layer: it validates an explicit
source-cloud contract, constructs the already-reviewed model, writes durable
numerical evidence atomically, and never changes a solver equation or gate.

The production ``run`` command requires an explicit GPU index and uses the
restricted JAX/CPU-replay adapter.  ``replay`` is independent of that
production path and delegates to the audited archive replay implementation.
Failure archives use separate numerical-only schemas because completed and
rejected continuation audits have different state contracts.  The unbounded
warning and stabilized policies each have separate completed/failure schema
identifiers so historical artifacts remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


__all__ = [
    "FAILURE_ARCHIVE_SCHEMA",
    "UNBOUNDED_FAILURE_ARCHIVE_SCHEMA",
    "STABILIZED_REPLAY_REPORT_SCHEMA",
    "DEFAULT_OUTPUT_ROOT",
    "artifact_paths",
    "analyze_archive",
    "build_parser",
    "main",
    "plot_archive",
    "replay_archive",
    "run_workflow",
    "write_failure_archive",
]


DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-tmp/data/fracture2d/engineering_grip"
)
DEFAULT_SOURCE_CLOUD = DEFAULT_OUTPUT_ROOT / "source/material_cloud_d048.npz"
DEFAULT_FEM_ARCHIVE = Path(
    "/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz"
)
FAILURE_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-failure-v1"
UNBOUNDED_FAILURE_ARCHIVE_SCHEMA = "fracture2d-engineering-grip-failure-unbounded-v1"
ANALYSIS_SCHEMA = "fracture2d-engineering-grip-analysis-v1"
PLOT_SCHEMA = "fracture2d-engineering-grip-plot-v1"
MANIFEST_SCHEMA = "fracture2d-engineering-grip-manifest-v1"
STABILIZED_REPLAY_REPORT_SCHEMA = (
    "fracture2d-engineering-grip-stabilized-replay-report-v5"
)

# These reproduce the latest engineering-grip experiment unless explicitly
# overridden. Acceptance mode, warning policy, storage budget, progress cadence,
# and run identity stay separate so the unvalidated warning path is never the
# implicit execution mode.
DEFAULT_LOAD_FINAL = 7.5e-3
DEFAULT_MAX_STAGGER_ITERATIONS = 144
DEFAULT_GPU_DEVICE_INDEX = 0
DEFAULT_DELTA = 4.8e-2
DEFAULT_GRIP_SPACING = 2.0e-2
DEFAULT_DU_INITIAL = 1.0e-4
DEFAULT_DU_MIN = 6.25e-6
DEFAULT_VISIBILITY_TOLERANCE = 1.0e-12
DEFAULT_YOUNGS_MODULUS = 210000.0
DEFAULT_POISSON_RATIO = 0.3
DEFAULT_K_RES = 1.0e-8
DEFAULT_RANK_RTOL = 1.0e-12
DEFAULT_GC = 2.7
DEFAULT_L0 = 2.4e-2
DEFAULT_CUTBACK_FACTOR = 0.5
DEFAULT_MAX_CUTBACKS = 8
DEFAULT_CG_RTOL = 1.0e-10
DEFAULT_CG_ATOL = 1.0e-13
DEFAULT_CG_MAXITER = 3000
DEFAULT_GATE_ATOL = 1.0e-9
DEFAULT_GATE_RTOL = 1.0e-8
DEFAULT_GATE_SCALE_FLOOR = 1.0
DEFAULT_DAMAGE_RESIDUAL_ATOL = 1.0e-10
DEFAULT_DAMAGE_RESIDUAL_RTOL = 1.0e-8
DEFAULT_DAMAGE_RESIDUAL_SCALE_FLOOR = 1.0
DEFAULT_HEALING_ATOL = 1.0e-10
DEFAULT_DAMAGE_RANGE_ATOL = 1.0e-10
DEFAULT_WORK_CHECK_LOAD = 1.0e-4
DEFAULT_WORK_EPSILON = 1.0e-5
DEFAULT_WORK_RELATIVE_LIMIT = 1.0e-7
DEFAULT_RELAXATION_POLICY = "fixed"
DEFAULT_OMEGA = 0.5
DEFAULT_MAX_BACKTRACKS = 0
CONTINUATION_STRATEGIES = ("legacy", "stabilized")
RELAXATION_POLICIES = ("sufficient_decrease", "fixed")


def _json_safe(value: Any) -> Any:
    """Convert NumPy/path values to strict JSON without NaN/Infinity."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    return value


def _print_stabilized_progress(event: Mapping[str, Any]) -> None:
    """Print one compact, flushed stabilized-controller progress event."""

    if not isinstance(event, Mapping):
        raise TypeError("stabilized progress event must be a mapping")
    payload = _json_safe(dict(event))
    print(
        "STABILIZED_PROGRESS "
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise OSError(f"cannot hash file {path}") from error
    return digest.hexdigest()


def _require_file(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def _prepare_output_directory(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _assert_no_object_arrays(arrays: Mapping[str, Any], context: str) -> None:
    for name, value in arrays.items():
        if np.asarray(value).dtype.kind == "O":
            raise ValueError(f"{context} field {name} has forbidden object dtype")


def _atomic_write(
    path: Path,
    writer: Any,
    *,
    overwrite: bool,
    suffix: str,
) -> None:
    """Write one file in the destination directory and replace atomically."""

    _prepare_output_directory(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact {path}; pass --overwrite"
        )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=suffix,
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    data = _json_bytes(payload)

    def writer(handle: Any) -> None:
        handle.write(data)

    _atomic_write(path, writer, overwrite=overwrite, suffix=".json.tmp")


def _atomic_npz(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    overwrite: bool,
    context: str,
) -> None:
    _assert_no_object_arrays(arrays, context)

    def writer(handle: Any) -> None:
        np.savez_compressed(handle, **arrays)

    _atomic_write(path, writer, overwrite=overwrite, suffix=".npz.tmp")


def _atomic_engineering_archive(
    path: Path,
    result: Any,
    *,
    overwrite: bool,
) -> None:
    """Use the audited writer while keeping its destination atomic."""

    _prepare_output_directory(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact {path}; pass --overwrite"
        )
    temporary_name: str | None = None
    try:
        # The audited writer accepts a path, not a file object, and NumPy adds
        # ``.npz`` to string paths without that suffix.  A real temporary .npz
        # pathname therefore keeps its behavior unchanged.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".npz",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
        from . import nonlocal_engineering_grip as engineering

        engineering.write_engineering_grip_archive(temporary_name, result)
        with Path(temporary_name).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_path_writer(
    path: Path,
    writer: Any,
    *,
    overwrite: bool,
    suffix: str = ".npz",
) -> None:
    """Atomically run a path-based archive writer in the destination directory."""

    _prepare_output_directory(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact {path}; pass --overwrite"
        )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=suffix,
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
        writer(temporary_name)
        with Path(temporary_name).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_stabilized_archive(
    path: Path,
    result: Any,
    *,
    source_metadata: Mapping[str, Any],
    archive_budget_bytes: int,
    archive_storage_preflight: Mapping[str, Any],
    overwrite: bool,
) -> None:
    """Write a stabilized result through its path-based writer atomically."""

    from . import nonlocal_engineering_grip_stabilized as stabilized

    _atomic_path_writer(
        path,
        lambda temporary: stabilized.write_stabilized_engineering_grip_archive(
            temporary,
            result,
            source_metadata=source_metadata,
            archive_budget_bytes=archive_budget_bytes,
            archive_storage_preflight=archive_storage_preflight,
        ),
        overwrite=overwrite,
    )


def _atomic_stabilized_failure_archive(
    path: Path,
    failure: Any,
    model: Any,
    config: Any,
    *,
    source_metadata: Mapping[str, Any],
    archive_budget_bytes: int,
    archive_storage_preflight: Mapping[str, Any],
    overwrite: bool,
) -> None:
    """Write a stabilized failure audit through its path-based writer atomically."""

    from . import nonlocal_engineering_grip_stabilized as stabilized

    _atomic_path_writer(
        path,
        lambda temporary: stabilized.write_stabilized_engineering_grip_failure_archive(
            temporary,
            failure,
            model,
            config,
            source_metadata=source_metadata,
            archive_budget_bytes=archive_budget_bytes,
            archive_storage_preflight=archive_storage_preflight,
        ),
        overwrite=overwrite,
    )


def _load_npz(path: Path, context: str) -> dict[str, np.ndarray]:
    _require_file(path, context)
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            f"cannot load {context} as a numerical NPZ: {path}: {error}"
        ) from error
    _assert_no_object_arrays(values, context)
    return values


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "SU":
        raise ValueError(f"{name} must be a scalar string array")
    return str(array.item())


def _require_float64(
    value: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_int64(
    value: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _require_integer_indices(
    value: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Accept an integer-index array and normalize it to exact int64."""

    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.size and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains an index above int64 range")
    if array.size and np.any(array < 0):
        raise ValueError(f"{name} contains a negative index")
    return np.asarray(array, dtype=np.int64)


def _require_bool(
    value: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must use exact boolean dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _load_source_cloud(path: Path, delta: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the explicit stage-3 cloud contract without casting its arrays."""

    values = _load_npz(path, "source cloud")
    required = {
        "coordinates",
        "triangles",
        "volumes",
        "bottom",
        "top",
        "anchor_particle",
        "d048",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"source cloud is missing required fields: {missing}")
    coordinates = _require_float64(values["coordinates"], "source coordinates")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("source coordinates must have nonempty shape (N, 2)")
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("source coordinates must be unique")
    n_material = len(coordinates)
    triangles = _require_int64(values["triangles"], "source triangles")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("source triangles must have nonempty shape (T, 3)")
    if np.any(triangles < 0) or np.any(triangles >= n_material):
        raise ValueError("source triangles contain an out-of-range node")
    volumes = _require_float64(values["volumes"], "source volumes", (n_material,))
    if np.any(volumes <= 0.0):
        raise ValueError("source volumes must be strictly positive")
    bottom = _require_bool(values["bottom"], "source bottom mask", (n_material,))
    top = _require_bool(values["top"], "source top mask", (n_material,))
    if not np.any(bottom) or not np.any(top) or np.any(bottom & top):
        raise ValueError("source bottom/top masks must be nonempty and disjoint")
    anchor = _require_int64(values["anchor_particle"], "source anchor particle")
    if anchor.shape != () or int(anchor) != 0:
        raise ValueError("engineering-grip source anchor_particle must be exact int64 zero")
    if not np.array_equal(coordinates[0], np.asarray([0.0, 0.0], dtype=np.float64)):
        raise ValueError("engineering-grip source particle zero must be the (0, 0) anchor")
    edges = _require_int64(values["d048"], "source d048 edges")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("source d048 edges must have nonempty shape (E, 2)")
    if np.any(edges < 0) or np.any(edges >= n_material):
        raise ValueError("source d048 edges contain an out-of-range index")
    if float(delta) != 0.048:
        raise ValueError(
            "the explicit source field 'd048' is only compatible with delta=0.048"
        )
    normalized = {
        "coordinates": coordinates,
        "triangles": triangles,
        "volumes": volumes,
        "bottom": bottom,
        "top": top,
        "anchor_particle": anchor,
        "d048": edges,
    }
    source_metadata = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "copied_into_output": False,
        "required_fields": [
            "coordinates",
            "triangles",
            "volumes",
            "bottom",
            "top",
            "anchor_particle",
            "d048",
        ],
        "fields": {
            name: {
                "dtype": str(np.asarray(values[name]).dtype),
                "shape": list(np.asarray(values[name]).shape),
            }
            for name in (
                "coordinates",
                "triangles",
                "volumes",
                "bottom",
                "top",
                "anchor_particle",
                "d048",
            )
        },
    }
    return normalized, source_metadata


def artifact_paths(output_directory: Path, run_id: str) -> dict[str, Path]:
    """Return stable success/failure and derived artifact names."""

    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a nonempty simple filename stem")
    root = Path(output_directory)
    stem = root / run_id
    return {
        "archive": stem.with_suffix(".npz"),
        "summary": stem.with_suffix(".summary.json"),
        "replay": stem.with_suffix(".replay.json"),
        "analysis": stem.with_suffix(".analysis.json"),
        "fields": stem.with_suffix(".fields.png"),
        "stress_strain": stem.with_suffix(".stress_strain.png"),
        "curve_data": stem.with_suffix(".curve.npz"),
        "plot_report": stem.with_suffix(".plot.json"),
        "failure_archive": root / f"{run_id}.failure.npz",
        "failure_summary": root / f"{run_id}.failure.summary.json",
        "failure_replay": root / f"{run_id}.failure.replay.json",
        "failure_analysis": root / f"{run_id}.failure.analysis.json",
        "failure_fields": root / f"{run_id}.failure.fields.png",
        "failure_stress_strain": root / f"{run_id}.failure.stress_strain.png",
        "failure_curve_data": root / f"{run_id}.failure.curve.npz",
        "failure_plot_report": root / f"{run_id}.failure.plot.json",
        "manifest": root / f"{run_id}.manifest.json",
    }


def _preflight_artifacts(paths: Mapping[str, Path], *, overwrite: bool) -> None:
    _prepare_output_directory(next(iter(paths.values())).parent)
    if overwrite:
        return
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing workflow artifacts; pass --overwrite: "
            + ", ".join(existing)
        )


def _diagnostic_json(diagnostic: Any, engineering: Any) -> str:
    payload = engineering._diagnostic_payload(diagnostic)
    return json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False)


def _stack_optional_diagnostic_field(
    diagnostics: Sequence[Any],
    name: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(diagnostics), *shape), dtype=np.float64)
    present = np.zeros(len(diagnostics), dtype=np.bool_)
    for index, diagnostic in enumerate(diagnostics):
        value = getattr(diagnostic, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float64) or array.shape != shape:
            raise ValueError(
                f"failure diagnostic {name} must use float64 shape {shape}, "
                f"got {array.dtype}/{array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"failure diagnostic {name} contains non-finite values")
        values[index] = array
        present[index] = True
    return values, present


def write_failure_archive(
    path: Path,
    failure: Any,
    model: Any,
    config: Any,
    source_metadata: Mapping[str, Any],
    *,
    device: str,
    acceptance_mode: str,
    warning_safety_limit: float | None,
    warning_safety_policy: str | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    """Write a complete numerical audit for a typed continuation failure."""

    from . import nonlocal_engineering_grip as engineering

    if not isinstance(failure, engineering.EngineeringGripContinuationFailure):
        raise TypeError("failure must be EngineeringGripContinuationFailure")
    if not isinstance(model, engineering.EngineeringGripModel):
        raise TypeError("model must be EngineeringGripModel")
    if not isinstance(config, engineering.EngineeringGripConfig):
        raise TypeError("config must be EngineeringGripConfig")
    mode = engineering._validate_acceptance_mode(acceptance_mode)
    if mode == engineering._WARNING_ACCEPTANCE_MODE:
        policy, warning_limit = engineering._validate_warning_safety_configuration(
            engineering._WARNING_SAFETY_POLICY_BOUNDED
            if warning_safety_policy is None
            else warning_safety_policy,
            warning_safety_limit,
        )
    elif warning_safety_limit is not None or warning_safety_policy is not None:
        raise ValueError(
            "warning safety policy/limit are only valid in warning mode"
        )
    else:
        policy = None
        warning_limit = None
    failure_schema = (
        UNBOUNDED_FAILURE_ARCHIVE_SCHEMA
        if policy == engineering._WARNING_SAFETY_POLICY_UNBOUNDED
        else FAILURE_ARCHIVE_SCHEMA
    )
    failure_status = (
        engineering.WARNING_CONTINUATION_FAILURE_STATUS
        if mode == engineering._WARNING_ACCEPTANCE_MODE
        else "strict_continuation_failure"
    )
    attempts = tuple(failure.attempts)
    if not attempts:
        raise ValueError("continuation failure must retain at least one attempt")
    diagnostics = tuple(
        diagnostic for attempt in attempts for diagnostic in attempt.sweeps
    )
    if not diagnostics:
        raise ValueError("continuation failure must retain at least one sweep")
    n_material = model.n_material
    n_union = model.n_union
    q_shape = (2 * n_union,)
    d_shape = (n_material,)

    q_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("q_before", "q_after"):
        q_fields[name] = _stack_optional_diagnostic_field(diagnostics, name, q_shape)
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
        q_fields[name] = _stack_optional_diagnostic_field(diagnostics, name, d_shape)
    force_fields = {
        name: _stack_optional_diagnostic_field(diagnostics, name, (2,))
        for name in ("force_before", "force_after")
    }

    accepted_attempts = tuple(attempt for attempt in attempts if attempt.accepted)
    accepted_states_q = [
        np.zeros(q_shape, dtype=np.float64)
    ]
    accepted_states_d = [
        np.zeros(d_shape, dtype=np.float64)
    ]
    accepted_states_H = [
        np.zeros(d_shape, dtype=np.float64)
    ]
    accepted_states_u = [0.0]
    accepted_top = [0.0]
    accepted_bottom = [0.0]
    accepted_anchor = [0.0]
    for attempt in accepted_attempts:
        diagnostic = attempt.diagnostics
        q_after = diagnostic.q_after
        d_next = diagnostic.d_next
        H_next = diagnostic.H_next
        top_reaction = diagnostic.top_reaction_after
        bottom_reaction = diagnostic.bottom_reaction_after
        anchor_reaction = diagnostic.anchor_reaction_after
        if any(
            value is None
            for value in (
                q_after,
                d_next,
                H_next,
                top_reaction,
                bottom_reaction,
                anchor_reaction,
            )
        ):
            raise ValueError("accepted failure audit is missing committed state fields")
        assert q_after is not None
        assert d_next is not None
        assert H_next is not None
        assert top_reaction is not None
        assert bottom_reaction is not None
        assert anchor_reaction is not None
        accepted_states_q.append(np.asarray(q_after, dtype=np.float64))
        accepted_states_d.append(np.asarray(d_next, dtype=np.float64))
        accepted_states_H.append(np.asarray(H_next, dtype=np.float64))
        accepted_states_u.append(float(attempt.load_target))
        accepted_top.append(float(top_reaction))
        accepted_bottom.append(float(bottom_reaction))
        accepted_anchor.append(float(anchor_reaction))
    last = failure.last_committed_state
    if not np.isclose(
        accepted_states_u[-1], last.u_bar, rtol=0.0, atol=0.0
    ):
        raise ValueError("failure last committed load is not paired with accepted audit")
    if not np.array_equal(accepted_states_q[-1], last.q):
        raise ValueError("failure last committed q is not paired with accepted audit")
    if not np.array_equal(accepted_states_d[-1], last.d):
        raise ValueError("failure last committed damage is not paired with accepted audit")
    if not np.array_equal(accepted_states_H[-1], last.H):
        raise ValueError("failure last committed history is not paired with accepted audit")

    metadata = {
        "schema": failure_schema,
        "completed_archive_schema": (
            (
                engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA
                if policy == engineering._WARNING_SAFETY_POLICY_UNBOUNDED
                else engineering.WARNING_ARCHIVE_SCHEMA
            )
            if mode == engineering._WARNING_ACCEPTANCE_MODE
            else engineering.ARCHIVE_SCHEMA
        ),
        "validated": False,
        "completion_status": failure_status,
        "acceptance_mode": mode,
        "failure_type": type(failure).__name__,
        "failure_message": str(failure),
        "device": str(device),
        "material_hash": model.material_hash,
        "union_hash": model.union_hash,
        "static_digest": model.static_digest,
        "code_hash": engineering._current_source_bundle_hash(),
        "code_hash_kind": engineering._SOURCE_BUNDLE_HASH_KIND,
        "code_hash_files": list(engineering._SOURCE_BUNDLE_FILES),
        "youngs_modulus": model.static.youngs_modulus,
        "poisson_ratio": model.static.poisson_ratio,
        "k_res": model.static.k_res,
        "rank_rtol": model.static.rank_rtol,
        "delta": model.geometry.delta,
        "h_g": model.geometry.h_g,
        "Gc": model.Gc,
        "l0": model.l0,
        "visibility_tolerance": model.geometry.visibility_tolerance,
        "config": engineering._config_payload(config),
        "source": dict(source_metadata),
    }
    if mode == engineering._WARNING_ACCEPTANCE_MODE:
        if policy == engineering._WARNING_SAFETY_POLICY_BOUNDED:
            metadata["warning_safety_limit"] = warning_limit
        else:
            metadata["warning_safety_policy"] = policy
            metadata["warning_safety_limit_present"] = False
    else:
        metadata["warning_safety_limit"] = None
    attempt_sweep_offsets = np.zeros(len(attempts) + 1, dtype=np.int64)
    attempt_sweep_offsets[1:] = np.cumsum(
        np.asarray([len(attempt.sweeps) for attempt in attempts], dtype=np.int64)
    )
    arrays: dict[str, Any] = {
        "archive_schema": np.asarray(failure_schema, dtype=np.str_),
        "validated": np.asarray(False, dtype=np.bool_),
        "completion_status": np.asarray(failure_status, dtype=np.str_),
        "acceptance_mode": np.asarray(mode, dtype=np.str_),
        "failure_type": np.asarray(type(failure).__name__, dtype=np.str_),
        "failure_message": np.asarray(str(failure), dtype=np.str_),
        "device": np.asarray(str(device), dtype=np.str_),
        "metadata_json": np.asarray(
            json.dumps(_json_safe(metadata), sort_keys=True, allow_nan=False),
            dtype=np.str_,
        ),
        "material_coordinates": model.geometry.material_reference_coordinates,
        "material_volumes": model.geometry.material_reference_volumes,
        "material_triangles": model.geometry.material_reference_triangles,
        "material_edges": model.geometry.material_reference_edges,
        "material_bottom_mask": model.geometry.material_bottom_mask[:n_material],
        "material_top_mask": model.geometry.material_top_mask[:n_material],
        "material_anchor": np.asarray(0, dtype=np.int64),
        "delta": np.asarray(model.geometry.delta, dtype=np.float64),
        "h_g": np.asarray(model.geometry.h_g, dtype=np.float64),
        "last_committed_u_bar": np.asarray(last.u_bar, dtype=np.float64),
        "last_committed_q": np.asarray(last.q, dtype=np.float64),
        "last_committed_d": np.asarray(last.d, dtype=np.float64),
        "last_committed_H": np.asarray(last.H, dtype=np.float64),
        "attempt_sweep_offsets": attempt_sweep_offsets,
        "attempt_load_start": np.asarray(
            [attempt.load_start for attempt in attempts], dtype=np.float64
        ),
        "attempt_load_target": np.asarray(
            [attempt.load_target for attempt in attempts], dtype=np.float64
        ),
        "attempt_cutback_count": np.asarray(
            [attempt.cutback_count for attempt in attempts], dtype=np.int64
        ),
        "attempt_accepted": np.asarray(
            [attempt.accepted for attempt in attempts], dtype=np.bool_
        ),
        "attempt_before_digest": np.asarray(
            [attempt.committed_before_digest for attempt in attempts], dtype=np.str_
        ),
        "attempt_after_digest": np.asarray(
            [attempt.committed_after_digest for attempt in attempts], dtype=np.str_
        ),
        "attempt_failure_stage": np.asarray(
            [attempt.failure_stage or "" for attempt in attempts], dtype=np.str_
        ),
        "attempt_failure_reason": np.asarray(
            [attempt.failure_reason or "" for attempt in attempts], dtype=np.str_
        ),
        "attempt_exception_type": np.asarray(
            [attempt.exception_type or "" for attempt in attempts], dtype=np.str_
        ),
        "attempt_sweep_count": np.asarray(
            [len(attempt.sweeps) for attempt in attempts], dtype=np.int64
        ),
        "attempt_diagnostics_json": np.asarray(
            [_diagnostic_json(attempt.diagnostics, engineering) for attempt in attempts],
            dtype=np.str_,
        ),
        "attempt_sweeps_json": np.asarray(
            [
                json.dumps(
                    [
                        json.loads(_diagnostic_json(diagnostic, engineering))
                        for diagnostic in attempt.sweeps
                    ],
                    sort_keys=True,
                    allow_nan=False,
                )
                for attempt in attempts
            ],
            dtype=np.str_,
        ),
        "sweep_attempt_index": np.asarray(
            [
                attempt_index
                for attempt_index, attempt in enumerate(attempts)
                for _ in attempt.sweeps
            ],
            dtype=np.int64,
        ),
        "sweep_index": np.asarray(
            [diagnostic.sweep_index for diagnostic in diagnostics], dtype=np.int64
        ),
        "sweep_target_u_bar": np.asarray(
            [diagnostic.target_u_bar for diagnostic in diagnostics], dtype=np.float64
        ),
        "sweep_complete": np.asarray(
            [diagnostic.complete_sweep for diagnostic in diagnostics], dtype=np.bool_
        ),
        "sweep_passed": np.asarray(
            [diagnostic.passed for diagnostic in diagnostics], dtype=np.bool_
        ),
        "sweep_failure_stage": np.asarray(
            [diagnostic.failure_stage for diagnostic in diagnostics], dtype=np.str_
        ),
        "sweep_failure_reason": np.asarray(
            [diagnostic.failure_reason for diagnostic in diagnostics], dtype=np.str_
        ),
        "sweep_diagnostics_json": np.asarray(
            [_diagnostic_json(diagnostic, engineering) for diagnostic in diagnostics],
            dtype=np.str_,
        ),
        "accepted_state_u_bar": np.asarray(accepted_states_u, dtype=np.float64),
        "accepted_state_q": np.stack(accepted_states_q),
        "accepted_state_d": np.stack(accepted_states_d),
        "accepted_state_H": np.stack(accepted_states_H),
        "accepted_top_reaction": np.asarray(accepted_top, dtype=np.float64),
        "accepted_bottom_reaction": np.asarray(accepted_bottom, dtype=np.float64),
        "accepted_anchor_reaction": np.asarray(accepted_anchor, dtype=np.float64),
        "work_check_u_bar": np.asarray(failure.work_check.u_bar, dtype=np.float64),
        "work_check_epsilon": np.asarray(failure.work_check.epsilon, dtype=np.float64),
        "work_check_reaction": np.asarray(failure.work_check.reaction, dtype=np.float64),
        "work_check_reaction_fd": np.asarray(
            failure.work_check.reaction_fd, dtype=np.float64
        ),
        "work_check_relative_error": np.asarray(
            failure.work_check.relative_error, dtype=np.float64
        ),
        "work_check_relative_limit": np.asarray(
            failure.work_check.relative_limit, dtype=np.float64
        ),
        "work_check_passed": np.asarray(failure.work_check.passed, dtype=np.bool_),
    }
    if policy != engineering._WARNING_SAFETY_POLICY_UNBOUNDED:
        arrays["warning_safety_limit"] = np.asarray(
            0.0 if warning_limit is None else warning_limit, dtype=np.float64
        )
    if mode == engineering._WARNING_ACCEPTANCE_MODE:
        gate_count = len(engineering.WARNING_GATE_NAMES)
        arrays.update(
            {
                "warning_gate_names": np.asarray(
                    engineering.WARNING_GATE_NAMES, dtype=np.str_
                ),
                "accepted_state_acceptance_status": np.asarray(
                    ["initial"]
                    + [attempt.acceptance_status or "" for attempt in accepted_attempts],
                    dtype=np.str_,
                ),
                "attempt_acceptance_status": np.asarray(
                    [attempt.acceptance_status or "" for attempt in attempts],
                    dtype=np.str_,
                ),
                "attempt_bypassed_gate_mask": np.asarray(
                    [
                        [
                            name in set(attempt.bypassed_gates)
                            for name in engineering.WARNING_GATE_NAMES
                        ]
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
                        0.0
                        if attempt.warning_range_excess is None
                        else attempt.warning_range_excess
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
        if policy == engineering._WARNING_SAFETY_POLICY_UNBOUNDED:
            arrays["warning_safety_policy"] = np.asarray(
                policy, dtype=np.str_
            )
            arrays["warning_safety_limit_present"] = np.asarray(
                False, dtype=np.bool_
            )
    for name, (values, present) in q_fields.items():
        arrays[f"sweep_{name}"] = values
        arrays[f"sweep_{name}_present"] = present
    for name, (values, present) in force_fields.items():
        arrays[f"sweep_{name}"] = values
        arrays[f"sweep_{name}_present"] = present
    _atomic_npz(
        path,
        arrays,
        overwrite=overwrite,
        context="engineering-grip failure archive",
    )
    return {
        "schema": failure_schema,
        "validated": False,
        "completion_status": failure_status,
        "acceptance_mode": mode,
        "warning_safety_policy": policy,
        "warning_safety_limit": warning_limit,
        "failure_message": str(failure),
        "attempts": len(attempts),
        "sweeps": len(diagnostics),
        "accepted_states": len(accepted_states_u),
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _config_from_arguments(arguments: argparse.Namespace, engineering: Any) -> Any:
    gate = engineering.GateTolerance(
        arguments.gate_atol,
        arguments.gate_rtol,
        arguments.gate_scale_floor,
    )
    damage_residual = engineering.GateTolerance(
        arguments.damage_residual_atol,
        arguments.damage_residual_rtol,
        arguments.damage_residual_scale_floor,
    )
    return engineering.EngineeringGripConfig(
        load_start=0.0,
        load_final=arguments.load_final,
        du_initial=arguments.du_initial,
        du_min=arguments.du_min,
        cutback_factor=arguments.cutback_factor,
        max_cutbacks=arguments.max_cutbacks,
        max_stagger_iterations=arguments.max_stagger_iterations,
        cg_rtol=arguments.cg_rtol,
        cg_atol=arguments.cg_atol,
        cg_maxiter=arguments.cg_maxiter,
        displacement=gate,
        damage=gate,
        history=gate,
        reaction=gate,
        damage_residual=damage_residual,
        force=gate,
        moment=gate,
        healing_atol=arguments.healing_atol,
        damage_range_atol=arguments.damage_range_atol,
        work_check_load=arguments.work_check_load,
        work_epsilon=arguments.work_epsilon,
        work_relative_limit=arguments.work_relative_limit,
    )


def _stabilized_config_from_arguments(
    arguments: argparse.Namespace,
    base_config: Any,
    stabilized: Any,
) -> Any:
    """Build the experimental stabilized configuration from explicit flags."""

    policy = arguments.relaxation_policy
    if policy not in RELAXATION_POLICIES:
        raise ValueError(
            "--relaxation-policy must be one of "
            f"{RELAXATION_POLICIES}, got {policy!r}"
        )
    names = ("omega_initial", "omega_min", "max_backtracks")
    if policy == "sufficient_decrease":
        names += ("omega_reduction", "sufficient_decrease")
    missing = [name for name in names if getattr(arguments, name) is None]
    if arguments.archive_budget_bytes is None:
        missing.append("archive_budget_bytes")
    if missing:
        raise ValueError(
            "--continuation-strategy stabilized requires explicit flags: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if arguments.acceptance_mode == "strict":
        acceptance_mode = stabilized.STABILIZED_ACCEPTANCE_MODE
        if (
            arguments.warning_safety_limit is not None
            or arguments.warning_safety_policy is not None
        ):
            raise ValueError(
                "warning safety policy/limit are incompatible with strict "
                "stabilized continuation"
            )
    elif arguments.acceptance_mode == "warning":
        acceptance_mode = stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE
        if arguments.warning_safety_policy != "unbounded":
            raise ValueError(
                "stabilized warning acceptance requires explicit "
                "--warning-safety-policy unbounded"
            )
        if arguments.warning_safety_limit is not None:
            raise ValueError(
                "--warning-safety-limit is incompatible with unbounded "
                "stabilized warning continuation"
            )
        if policy != "fixed":
            raise ValueError(
                "stabilized warning continuation requires "
                "--relaxation-policy fixed"
            )
    else:
        raise ValueError(
            "stabilized continuation requires --acceptance-mode strict or warning"
        )
    if policy == "fixed":
        if arguments.omega_reduction is not None:
            raise ValueError(
                "--omega-reduction is forbidden with --relaxation-policy fixed"
            )
        if arguments.sufficient_decrease is not None:
            raise ValueError(
                "--sufficient-decrease is forbidden with --relaxation-policy fixed"
            )
        omega_reduction = None
        sufficient_decrease = None
    else:
        omega_reduction = arguments.omega_reduction
        sufficient_decrease = arguments.sufficient_decrease
    return stabilized.StabilizedEngineeringGripConfig(
        base=base_config,
        omega_initial=arguments.omega_initial,
        omega_min=arguments.omega_min,
        omega_reduction=omega_reduction,
        sufficient_decrease=sufficient_decrease,
        max_backtracks=arguments.max_backtracks,
        relaxation_policy=policy,
        acceptance_mode=acceptance_mode,
    )


def _workflow_config_payload(config: Any, strategy: str, engineering: Any) -> dict[str, Any]:
    """Serialize the base and optional stabilized configuration explicitly."""

    if strategy == "stabilized":
        return {
            "continuation_strategy": strategy,
            "base": _json_safe(engineering._config_payload(config.base)),
            "stabilized": _json_safe(config.payload()),
        }
    return {
        "continuation_strategy": strategy,
        **_json_safe(engineering._config_payload(config)),
    }


def _acceptance_from_arguments(
    arguments: argparse.Namespace, engineering: Any
) -> tuple[str, float | None]:
    """Validate the workflow acceptance mode and return its legacy pair.

    The two-value return is retained for bounded-warning callers.  Use
    :func:`_acceptance_details_from_arguments` when the explicit safety policy
    also needs to be propagated.
    """

    mode, _, warning_limit = _acceptance_details_from_arguments(arguments, engineering)
    return mode, warning_limit


def _acceptance_details_from_arguments(
    arguments: argparse.Namespace, engineering: Any
) -> tuple[str, str | None, float | None]:
    """Validate mode, warning policy, and the optional policy-specific limit."""

    mode = engineering._validate_acceptance_mode(arguments.acceptance_mode)
    warning_limit = arguments.warning_safety_limit
    raw_policy = getattr(arguments, "warning_safety_policy", None)
    if mode == engineering._WARNING_ACCEPTANCE_MODE:
        policy = engineering._validate_warning_safety_policy(
            engineering._WARNING_SAFETY_POLICY_BOUNDED
            if raw_policy is None
            else raw_policy
        )
        policy, warning_limit = engineering._validate_warning_safety_configuration(
            policy, warning_limit
        )
        return mode, policy, warning_limit
    if warning_limit is not None or raw_policy is not None:
        raise ValueError(
            "--warning-safety-policy/--warning-safety-limit are only valid "
            "with --acceptance-mode warning"
        )
    return mode, None, None


def _construct_model(
    arguments: argparse.Namespace,
) -> tuple[Any, Any, dict[str, np.ndarray], dict[str, Any]]:
    """Construct the CPU model and base config after input validation.

    This stage performs no GPU work.  Keeping it separate lets the stabilized
    workflow preflight archive storage from the exact model dimensions and
    static arrays before any GPU adapter or continuation is created.
    """

    import jax

    # The direct fracture2d path is frozen to float64.  Enabling x64 here is
    # explicit orchestration, not a numerical fallback or a downcast.
    jax.config.update("jax_enable_x64", True)
    from . import nonlocal_engineering_grip as engineering
    from .nonlocal_grip_diagnostic import build_union_geometry_from_arrays

    cloud, source_metadata = _load_source_cloud(
        arguments.source_cloud, arguments.delta
    )
    geometry = build_union_geometry_from_arrays(
        cloud["coordinates"],
        cloud["triangles"],
        cloud["volumes"],
        cloud["bottom"],
        cloud["top"],
        int(cloud["anchor_particle"]),
        cloud["d048"],
        arguments.delta,
        arguments.h_g,
        arguments.visibility_tolerance,
    )
    model = engineering.build_engineering_grip_model(
        geometry,
        arguments.youngs_modulus,
        arguments.poisson_ratio,
        arguments.k_res,
        arguments.rank_rtol,
        arguments.Gc,
        arguments.l0,
    )
    config = _config_from_arguments(arguments, engineering)
    return model, config, cloud, source_metadata


def _select_gpu_device(arguments: argparse.Namespace) -> Any:
    import jax

    gpu_devices = tuple(device for device in jax.devices() if device.platform == "gpu")
    if not gpu_devices:
        raise RuntimeError(
            "the engineering-grip production workflow requires a visible JAX GPU; "
            "no CPU fallback is provided"
        )
    if arguments.gpu_device_index >= len(gpu_devices):
        raise ValueError(
            f"gpu-device-index {arguments.gpu_device_index} is out of range; "
            f"visible GPU count is {len(gpu_devices)}"
        )
    return gpu_devices[arguments.gpu_device_index]


def _construct_adapter(
    arguments: argparse.Namespace, model: Any
) -> tuple[Any, str]:
    """Create the restricted JAX GPU adapter for an already-built model."""

    from . import nonlocal_engineering_grip_jax as engineering_jax

    device = _select_gpu_device(arguments)
    adapter = engineering_jax.JaxA2CPUReplayAdapter.from_static(
        model.static, model.constraints, device
    )
    return adapter, str(device)


def _construct_run(
    arguments: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, np.ndarray], dict[str, Any], str]:
    """Construct model/config/adapter after all explicit input validation."""

    model, config, cloud, source_metadata = _construct_model(arguments)
    adapter, device = _construct_adapter(arguments, model)
    return model, config, adapter, cloud, source_metadata, device


def _preflight_stabilized_storage(
    archive_path: Path,
    model: Any,
    config: Any,
    stabilized: Any,
    *,
    archive_budget_bytes: int | None,
) -> dict[str, Any]:
    """Guard stabilized continuation storage before any GPU work.

    The lean v5 estimate uses the exact model dimensions: accepted physical
    state rows plus one scalar solver-summary row per state, with every static
    operator and the geometry itself rebuilt from the hashed source cloud
    rather than serialized.  Insufficient free disk raises :class:`OSError`
    before a GPU adapter or continuation exists.  The returned estimate is
    printed and stored in every run summary.
    """

    if archive_budget_bytes is None:
        raise ValueError(
            "--continuation-strategy stabilized requires "
            "--archive-budget-bytes"
        )
    payload = stabilized.preflight_stabilized_archive_storage(
        archive_path,
        model.n_material,
        model.n_dofs,
        config,
        archive_budget_bytes=archive_budget_bytes,
    )
    print(
        "STABILIZED_ARCHIVE_STORAGE_PREFLIGHT "
        + json.dumps(
            _json_safe(
                {
                    "output_path": payload["output_path"],
                    "filesystem_probe": payload["filesystem_probe"],
                    "available_bytes": payload["available_bytes"],
                    "free_bytes": payload["free_bytes"],
                    "nominal_archive_bytes": payload["nominal_archive_bytes"],
                    "worst_case_archive_bytes": payload[
                        "worst_case_archive_bytes"
                    ],
                    "archive_budget_bytes": payload["archive_budget_bytes"],
                    "archive_bytes": payload["archive_bytes"],
                    "filesystem_reserve_bytes": payload[
                        "filesystem_reserve_bytes"
                    ],
                    "required_free_bytes": payload["required_free_bytes"],
                    "static_bound_kind": payload["static_bound_kind"],
                    "n_material": payload["n_material"],
                    "n_union_dofs": payload["n_union_dofs"],
                    "n_increments": payload["n_increments"],
                    "solver_summary_rows": payload["solver_summary_rows"],
                    "archive_array_count": payload["archive_array_count"],
                    "archive_layout": payload["archive_layout"],
                    "physical_rows": {
                        "states": payload["n_states"],
                        "accepted_steps": payload["accepted_steps"],
                    },
                    "scalar_rows": {
                        "solver_summaries": payload["solver_summary_rows"],
                    },
                    "fits": payload["fits"],
                }
            ),
            sort_keys=True,
        )
    )
    return payload


def _source_bundle_manifest(engineering: Any, solver_module: Any | None = None) -> dict[str, Any]:
    bundle_module = engineering if solver_module is None else solver_module
    repository_root = Path(bundle_module.__file__).resolve().parents[2]
    bundle_files = tuple(
        getattr(bundle_module, "_STABILIZED_SOURCE_BUNDLE_FILES", engineering._SOURCE_BUNDLE_FILES)
    )
    bundle_hash_kind = getattr(
        bundle_module, "_SOURCE_BUNDLE_HASH_KIND", engineering._SOURCE_BUNDLE_HASH_KIND
    )
    bundle_hash_function = getattr(
        bundle_module, "_current_stabilized_source_bundle_hash", engineering._current_source_bundle_hash
    )
    files = []
    for relative_name in bundle_files:
        path = repository_root / relative_name
        _require_file(path, "source-bundle file")
        files.append(
            {
                "path": relative_name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    workflow_path = Path(__file__).resolve()
    return {
        "solver_bundle_hash_kind": bundle_hash_kind,
        "solver_bundle_hash": bundle_hash_function(),
        "solver_bundle_files": files,
        "workflow_file": {
            "path": str(workflow_path),
            "sha256": _sha256_file(workflow_path),
            "size_bytes": workflow_path.stat().st_size,
        },
    }


def _artifact_record(path: Path, kind: str, root: Path) -> dict[str, Any]:
    _require_file(path, "artifact")
    return {
        "path": str(path),
        "relative_path": os.path.relpath(path, root),
        "kind": kind,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _acceptance_payload(
    acceptance_mode: str,
    warning_safety_limit: float | None,
    warning_safety_policy: str | None,
) -> dict[str, Any]:
    """Build the stable acceptance report without an absent-value sentinel."""

    payload: dict[str, Any] = {"mode": acceptance_mode}
    if acceptance_mode == "stabilized":
        if warning_safety_limit is not None or warning_safety_policy is not None:
            raise ValueError(
                "warning safety policy/limit are incompatible with stabilized acceptance"
            )
        return payload
    if acceptance_mode == "stabilized_warning":
        if warning_safety_limit is not None or warning_safety_policy != "unbounded":
            raise ValueError(
                "stabilized warning acceptance requires an explicit unbounded "
                "warning safety policy without a numeric limit"
            )
        payload["warning_safety_policy"] = "unbounded"
        payload["warning_safety_limit_present"] = False
        return payload
    if acceptance_mode != "warning":
        payload["warning_safety_limit"] = None
        return payload
    policy = "bounded" if warning_safety_policy is None else warning_safety_policy
    payload["warning_safety_policy"] = policy
    if policy == "unbounded":
        payload["warning_safety_limit_present"] = False
    else:
        payload["warning_safety_limit"] = warning_safety_limit
    return payload


def _write_manifest(
    path: Path,
    paths: Mapping[str, Path],
    source_metadata: Mapping[str, Any],
    engineering: Any,
    config: Any,
    *,
    device: str,
    command: Sequence[str],
    validated: bool,
    field_equations_validated: bool | None = None,
    path_replayed: bool | None = None,
    acceptance_mode: str,
    warning_safety_limit: float | None,
    warning_safety_policy: str | None = None,
    strategy: str = "legacy",
    solver_module: Any | None = None,
    archive_storage_preflight: Mapping[str, Any] | None = None,
    overwrite: bool,
) -> dict[str, Any]:
    root = path.parent
    completed_keys = (
        "archive",
        "summary",
        "replay",
    )
    failure_keys = (
        "failure_archive",
        "failure_summary",
        "failure_replay",
    )
    derived_keys = (
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
    stabilized = strategy == "stabilized"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "experimental": stabilized,
        "source": dict(source_metadata),
        "completed_archive": [
            _artifact_record(paths[key], "completed_archive", root)
            for key in completed_keys
            if paths[key].is_file()
        ],
        "failure_audit": [
            _artifact_record(paths[key], "failure_audit", root)
            for key in failure_keys
            if paths[key].is_file()
        ],
        "derived": [
            _artifact_record(paths[key], "derived", root)
            for key in derived_keys
            if paths[key].is_file()
        ],
        "code": _source_bundle_manifest(engineering, solver_module),
        "configuration": {
            **_workflow_config_payload(config, strategy, engineering),
            "acceptance": _acceptance_payload(
                acceptance_mode, warning_safety_limit, warning_safety_policy
            ),
        },
        "execution": {
            "command": list(command),
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
            "jax_x64_enabled": True,
        },
        "provenance_policy": {
            "source_binary_copied": False,
            "temporary_paths_are_not_sole_provenance": True,
            "archive_pickle_loading": False,
        },
    }
    if stabilized:
        # Stabilized v5 stores accepted physical fields and compact solver
        # summaries only.
        # It can therefore report the core field-equation check, but it cannot
        # claim replay of the omitted rejected-trial numerical path.
        if solver_module is None:
            raise ValueError("stabilized manifest requires its solver module")
        contract = solver_module._stabilized_contract(
            acceptance_mode, failure=False
        )
        manifest.update(
            {
                "archive_layout": contract["archive_layout"],
                "replay_contract": contract["replay_contract"],
                "acceptance_mode": contract["acceptance_mode"],
                "acceptance_status": contract["acceptance_status"],
                "field_equations_validated": bool(
                    field_equations_validated
                ),
                "path_replayed": False,
                "rejected_trial_path_available": False,
            }
        )
        if path_replayed not in (None, False):
            raise ValueError("stabilized v5 manifest cannot claim path replay")
    else:
        manifest["validated"] = bool(validated)
    if archive_storage_preflight is not None:
        if not isinstance(archive_storage_preflight, Mapping):
            raise TypeError("archive_storage_preflight must be a mapping")
        manifest["archive_storage_preflight"] = _json_safe(
            archive_storage_preflight
        )
    _atomic_json(path, manifest, overwrite=overwrite)
    return manifest


def _state_summary_from_completed(
    values: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    loads = np.asarray(values["state_u_bar"], dtype=np.float64)
    reaction = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.asarray(values["accepted_top_reaction_after"], dtype=np.float64),
        )
    )
    damage = np.asarray(values["state_material_damage"], dtype=np.float64)
    history = np.asarray(values["state_material_history"], dtype=np.float64)
    summary = {
        "state_u_bar": loads.tolist(),
        "state_top_reaction": reaction.tolist(),
        "state_damage_max": np.max(damage, axis=1).tolist(),
        "state_history_max": np.max(history, axis=1).tolist(),
        "accepted_steps": int(max(0, len(loads) - 1))
        if "solver_summary_json" in values
        else int(len(values["accepted_load_target"])),
    }
    if "solver_summary_json" in values:
        summary["solver_summary_rows"] = int(len(values["solver_summary_json"]))
        summary["state_external_work"] = np.asarray(
            values["state_external_work"], dtype=np.float64
        ).tolist()
    else:
        summary["attempts"] = int(
            len(values["attempt_log_json"])
            if "attempt_log_json" in values
            else len(values["attempt_load_target"])
        )
    return summary


def _failure_state_summary(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if "accepted_state_u_bar" in values:
        loads = np.asarray(values["accepted_state_u_bar"], dtype=np.float64)
        damage = np.asarray(values["accepted_state_d"], dtype=np.float64)
        history = np.asarray(values["accepted_state_H"], dtype=np.float64)
        reaction_name = (
            "accepted_top_reaction_after"
            if "accepted_top_reaction_after" in values
            else "accepted_top_reaction"
        )
        reaction = np.asarray(values[reaction_name], dtype=np.float64)
    else:
        # Stabilized v5 failure archives retain the same full state arrays as
        # completed archives; only the final rejected trial is omitted.
        loads = np.asarray(values["state_u_bar"], dtype=np.float64)
        damage = np.asarray(values["state_material_damage"], dtype=np.float64)
        history = np.asarray(values["state_material_history"], dtype=np.float64)
        reaction = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.asarray(values["accepted_top_reaction_after"], dtype=np.float64),
            )
        )
    summary = {
        "state_u_bar": loads.tolist(),
        "state_top_reaction": reaction.tolist(),
        "state_damage_max": np.max(damage, axis=1).tolist(),
        "state_history_max": np.max(history, axis=1).tolist(),
        "accepted_steps": int(max(0, len(loads) - 1))
        if "solver_summary_json" in values
        else int(max(0, len(loads) - 1)),
    }
    if "solver_summary_json" in values:
        summary["solver_summary_rows"] = int(len(values["solver_summary_json"]))
        summary["state_external_work"] = np.asarray(
            values["state_external_work"], dtype=np.float64
        ).tolist()
    else:
        summary["attempts"] = int(
            len(values["attempt_log_json"])
            if "attempt_log_json" in values
            else len(values["attempt_load_target"])
        )
    return summary


def _run_summary(
    *,
    validated: bool,
    status: str,
    elapsed_seconds: float,
    archive_path: Path,
    replay_path: Path | None,
    analysis_path: Path | None,
    plot_report_path: Path | None,
    state_summary: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    device: str,
    config_payload: Mapping[str, Any],
    acceptance_mode: str,
    warning_safety_limit: float | None,
    warning_safety_policy: str | None = None,
    field_equations_validated: bool | None = None,
    path_replayed: bool | None = None,
    archive_storage_preflight: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    strategy = config_payload.get("continuation_strategy", "legacy")
    stabilized = strategy == "stabilized"
    payload: dict[str, Any] = {
        "schema": "fracture2d-engineering-grip-run-summary-v1",
        "nonproduction": True,
        "completion_status": status,
        "continuation_strategy": strategy,
        "experimental": stabilized,
        "elapsed_seconds": float(elapsed_seconds),
        "device": str(device),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(archive_path)
        if archive_path.is_file()
        else None,
        "replay_report": str(replay_path) if replay_path is not None else None,
        "analysis_report": str(analysis_path) if analysis_path is not None else None,
        "plot_report": str(plot_report_path) if plot_report_path is not None else None,
        "source": dict(source_metadata),
        "configuration": dict(config_payload),
        "acceptance": _acceptance_payload(
            acceptance_mode, warning_safety_limit, warning_safety_policy
        ),
        "state_summary": dict(state_summary),
    }
    if stabilized:
        # v5 validates accepted physical fields independently, but its lean
        # summary layout cannot replay the omitted rejected-trial path.
        payload.update(
            {
                "field_equations_validated": bool(field_equations_validated),
                "path_replayed": False,
                "rejected_trial_path_available": False,
            }
        )
        if path_replayed not in (None, False):
            raise ValueError("stabilized v5 summary cannot claim path replay")
    else:
        payload["validated"] = bool(validated)
    if error is not None:
        payload["error"] = str(error)
    if archive_storage_preflight is not None:
        if not isinstance(archive_storage_preflight, Mapping):
            raise TypeError("archive_storage_preflight must be a mapping")
        payload["archive_storage_preflight"] = _json_safe(
            archive_storage_preflight
        )
    return payload


def _run_stabilized_workflow(arguments: argparse.Namespace) -> int:
    """Execute the experimental stabilized path without changing legacy flow."""

    from . import nonlocal_engineering_grip as engineering
    from . import nonlocal_engineering_grip_stabilized as stabilized

    if arguments.archive_budget_bytes is None:
        raise ValueError(
            "--continuation-strategy stabilized requires "
            "--archive-budget-bytes"
        )
    if arguments.progress_every_sweeps is None:
        raise ValueError(
            "--continuation-strategy stabilized requires explicit "
            "--progress-every-sweeps"
        )
    fem = _load_fem_archive(arguments.fem_archive)
    paths = artifact_paths(arguments.output_directory, arguments.run_id)
    _preflight_artifacts(paths, overwrite=arguments.overwrite)
    started = time.perf_counter()
    model, base_config, cloud, source_metadata = _construct_model(arguments)
    config = _stabilized_config_from_arguments(arguments, base_config, stabilized)
    if cloud["coordinates"].shape != fem["coords"].shape or not np.array_equal(
        cloud["coordinates"], fem["coords"]
    ):
        raise ValueError(
            "source cloud and FEM comparison coordinates are not bitwise aligned"
        )
    if not _triangles_match(cloud["triangles"], fem["triangles"]):
        raise ValueError("source cloud and FEM comparison triangle sets are not aligned")
    # Guard durable archive storage after the CPU model dimensions are known
    # but strictly before the GPU adapter or any continuation work is created.
    # Insufficient free storage raises OSError here and stops the workflow.
    storage_preflight = _preflight_stabilized_storage(
        paths["archive"],
        model,
        config,
        stabilized,
        archive_budget_bytes=arguments.archive_budget_bytes,
    )
    adapter, device = _construct_adapter(arguments, model)
    config_payload = _workflow_config_payload(config, "stabilized", engineering)
    acceptance_mode = config.acceptance_mode
    failure_contract = stabilized._stabilized_contract(acceptance_mode, failure=True)

    try:
        result = stabilized.run_stabilized_engineering_grip(
            model,
            config,
            mechanics_adapter=adapter,
            archive_budget_bytes=arguments.archive_budget_bytes,
            archive_output_path=paths["archive"],
            progress_callback=_print_stabilized_progress,
            progress_every_sweeps=arguments.progress_every_sweeps,
        )
    except stabilized.StabilizedEngineeringGripContinuationFailure as failure:
        _atomic_stabilized_failure_archive(
            paths["failure_archive"],
            failure,
            model,
            config,
            source_metadata=source_metadata,
            archive_budget_bytes=arguments.archive_budget_bytes,
            archive_storage_preflight=storage_preflight,
            overwrite=arguments.overwrite,
        )
        failure_values = _load_npz(paths["failure_archive"], "stabilized failure archive")
        failure_replay_payload = replay_archive(
            paths["failure_archive"],
            paths["failure_replay"],
            overwrite=arguments.overwrite,
        )
        failure_fields_validated = bool(
            failure_replay_payload.get("field_equations_validated", False)
        )
        failure_summary = _run_summary(
            validated=False,
            status=failure_contract["completion_status"],
            elapsed_seconds=time.perf_counter() - started,
            archive_path=paths["failure_archive"],
            replay_path=paths["failure_replay"],
            analysis_path=paths["failure_analysis"],
            plot_report_path=paths["failure_plot_report"],
            state_summary=_failure_state_summary(failure_values),
            source_metadata=source_metadata,
            device=device,
            config_payload=config_payload,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=None,
            warning_safety_policy=arguments.warning_safety_policy,
            field_equations_validated=failure_fields_validated,
            archive_storage_preflight=storage_preflight,
            error=str(failure),
        )
        failure_summary["replay"] = failure_replay_payload
        failure_summary["archive_storage_preflight"] = _json_safe(
            storage_preflight
        )
        if isinstance(
            failure, stabilized.StabilizedEngineeringGripArchiveBudgetFailure
        ):
            failure_summary["archive_budget_failure"] = {
                "stage": failure.stage,
                "estimated_archive_bytes": int(failure.estimate["archive_bytes"]),
                "archive_budget_bytes": int(failure.archive_budget_bytes),
                "free_bytes": (
                    None
                    if failure.available_bytes is None
                    else int(failure.available_bytes)
                ),
            }
        failure_summary["failure_archive"] = {
            "schema": failure_contract["archive_schema"],
            "experimental": True,
            "field_equations_validated": failure_fields_validated,
            "path_replayed": False,
            "completion_status": failure_contract["completion_status"],
            "acceptance_mode": acceptance_mode,
            "failure_message": str(failure),
            "states": len(failure_values["state_u_bar"]),
            "accepted_steps": max(0, len(failure_values["state_u_bar"]) - 1),
            "solver_summary_rows": len(failure_values["solver_summary_json"]),
            "path": str(paths["failure_archive"]),
            "sha256": _sha256_file(paths["failure_archive"]),
        }
        _atomic_json(
            paths["failure_summary"], failure_summary, overwrite=arguments.overwrite
        )
        failure_analysis = analyze_archive(
            paths["failure_archive"],
            paths["failure_analysis"],
            overwrite=arguments.overwrite,
            replay_result=failure_replay_payload["replay"],
        )
        failure_plot = plot_archive(
            paths["failure_archive"],
            arguments.fem_archive,
            target_load=float(failure.last_committed_state.u_bar),
            fields_path=paths["failure_fields"],
            curve_path=paths["failure_stress_strain"],
            curve_data_path=paths["failure_curve_data"],
            report_path=paths["failure_plot_report"],
            overwrite=arguments.overwrite,
            replay_result=failure_replay_payload["replay"],
        )
        failure_summary["analysis"] = failure_analysis
        failure_summary["plots"] = failure_plot
        failure_summary["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(paths["failure_summary"], failure_summary, overwrite=True)
        _write_manifest(
            paths["manifest"],
            paths,
            source_metadata,
            engineering,
            config,
            device=device,
            command=sys.argv,
            validated=False,
            field_equations_validated=failure_fields_validated,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=arguments.warning_safety_limit,
            warning_safety_policy=arguments.warning_safety_policy,
            strategy="stabilized",
            solver_module=stabilized,
            archive_storage_preflight=storage_preflight,
            overwrite=arguments.overwrite,
        )
        raise

    _atomic_stabilized_archive(
        paths["archive"],
        result,
        source_metadata=source_metadata,
        archive_budget_bytes=arguments.archive_budget_bytes,
        archive_storage_preflight=storage_preflight,
        overwrite=arguments.overwrite,
    )
    try:
        replay_payload = replay_archive(
            paths["archive"],
            paths["replay"],
            overwrite=arguments.overwrite,
        )
    except Exception as error:
        failed_summary = _run_summary(
            validated=False,
            status="replay_failed",
            elapsed_seconds=time.perf_counter() - started,
            archive_path=paths["archive"],
            replay_path=paths["replay"],
            analysis_path=None,
            plot_report_path=None,
            state_summary=_state_summary_from_completed(
                _load_npz(paths["archive"], "stabilized completed archive")
            ),
            source_metadata=source_metadata,
            device=device,
            config_payload=config_payload,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=None,
            warning_safety_policy=arguments.warning_safety_policy,
            field_equations_validated=False,
            archive_storage_preflight=storage_preflight,
            error=str(error),
        )
        failed_summary["archive_storage_preflight"] = _json_safe(
            storage_preflight
        )
        _atomic_json(paths["summary"], failed_summary, overwrite=arguments.overwrite)
        _write_manifest(
            paths["manifest"],
            paths,
            source_metadata,
            engineering,
            config,
            device=device,
            command=sys.argv,
            validated=False,
            field_equations_validated=False,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=None,
            warning_safety_policy=arguments.warning_safety_policy,
            strategy="stabilized",
            solver_module=stabilized,
            archive_storage_preflight=storage_preflight,
            overwrite=arguments.overwrite,
        )
        raise

    analysis = analyze_archive(
        paths["archive"],
        paths["analysis"],
        overwrite=arguments.overwrite,
        replay_result=replay_payload["replay"],
    )
    plot = plot_archive(
        paths["archive"],
        arguments.fem_archive,
        target_load=config.base.load_final,
        fields_path=paths["fields"],
        curve_path=paths["stress_strain"],
        curve_data_path=paths["curve_data"],
        report_path=paths["plot_report"],
        overwrite=arguments.overwrite,
        replay_result=replay_payload["replay"],
    )
    values = _load_npz(paths["archive"], "stabilized completed archive")
    summary = _run_summary(
        validated=False,
        status=result.completion_status,
        elapsed_seconds=time.perf_counter() - started,
        archive_path=paths["archive"],
        replay_path=paths["replay"],
        analysis_path=paths["analysis"],
        plot_report_path=paths["plot_report"],
        state_summary=_state_summary_from_completed(values),
        source_metadata=source_metadata,
        device=device,
        config_payload=config_payload,
        acceptance_mode=acceptance_mode,
        warning_safety_limit=arguments.warning_safety_limit,
        warning_safety_policy=arguments.warning_safety_policy,
        field_equations_validated=bool(
            replay_payload["field_equations_validated"]
        ),
        archive_storage_preflight=storage_preflight,
    )
    summary["replay"] = replay_payload
    summary["analysis"] = analysis
    summary["plots"] = plot
    summary["archive_storage_preflight"] = _json_safe(storage_preflight)
    _atomic_json(paths["summary"], summary, overwrite=arguments.overwrite)
    _write_manifest(
        paths["manifest"],
        paths,
        source_metadata,
        engineering,
        config,
        device=device,
        command=sys.argv,
        validated=False,
        field_equations_validated=bool(
            replay_payload["field_equations_validated"]
        ),
        acceptance_mode=acceptance_mode,
        warning_safety_limit=arguments.warning_safety_limit,
        warning_safety_policy=arguments.warning_safety_policy,
        strategy="stabilized",
        solver_module=stabilized,
        archive_storage_preflight=storage_preflight,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def run_workflow(arguments: argparse.Namespace) -> int:
    """Execute one fresh-zero engineering-grip workflow path."""

    if arguments.continuation_strategy == "stabilized":
        return _run_stabilized_workflow(arguments)

    if arguments.archive_budget_bytes is not None:
        raise ValueError(
            "--archive-budget-bytes is only valid with "
            "--continuation-strategy stabilized"
        )
    if arguments.progress_every_sweeps is not None:
        raise ValueError(
            "--progress-every-sweeps is only valid with "
            "--continuation-strategy stabilized"
        )

    from . import nonlocal_engineering_grip as engineering

    acceptance_mode, warning_safety_policy, warning_safety_limit = (
        _acceptance_details_from_arguments(
            arguments, engineering
        )
    )

    # Validate the independent comparison file before constructing the model or
    # launching any long GPU work.  Its triangle connectivity may be int32 in
    # the FEM archive, so _load_fem_archive normalizes only that independent
    # index field to int64.
    fem = _load_fem_archive(arguments.fem_archive)
    paths = artifact_paths(arguments.output_directory, arguments.run_id)
    _preflight_artifacts(paths, overwrite=arguments.overwrite)
    started = time.perf_counter()
    model, config, adapter, cloud, source_metadata, device = _construct_run(arguments)
    if cloud["coordinates"].shape != fem["coords"].shape or not np.array_equal(
        cloud["coordinates"], fem["coords"]
    ):
        raise ValueError(
            "source cloud and FEM comparison coordinates are not bitwise aligned"
        )
    if not _triangles_match(cloud["triangles"], fem["triangles"]):
        raise ValueError("source cloud and FEM comparison triangle sets are not aligned")
    config_payload = engineering._config_payload(config)
    try:
        if acceptance_mode == engineering._WARNING_ACCEPTANCE_MODE:
            result = engineering.run_warning_engineering_grip(
                model,
                config,
                warning_safety_limit,
                mechanics_adapter=adapter,
                warning_safety_policy=warning_safety_policy,
            )
        else:
            result = engineering.run_strict_engineering_grip(
                model,
                config,
                mechanics_adapter=adapter,
            )
    except engineering.EngineeringGripContinuationFailure as failure:
        failure_info = write_failure_archive(
            paths["failure_archive"],
            failure,
            model,
            config,
            source_metadata,
            device=device,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=warning_safety_limit,
            warning_safety_policy=warning_safety_policy,
            overwrite=arguments.overwrite,
        )
        failure_values = _load_npz(paths["failure_archive"], "failure archive")
        failure_summary = _run_summary(
            validated=False,
            status=(
                engineering.WARNING_CONTINUATION_FAILURE_STATUS
                if acceptance_mode == engineering._WARNING_ACCEPTANCE_MODE
                else "strict_continuation_failure"
            ),
            elapsed_seconds=time.perf_counter() - started,
            archive_path=paths["failure_archive"],
            replay_path=None,
            analysis_path=paths["failure_analysis"],
            plot_report_path=paths["failure_plot_report"],
            state_summary=_failure_state_summary(failure_values),
            source_metadata=source_metadata,
            device=device,
            config_payload=config_payload,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=warning_safety_limit,
            warning_safety_policy=warning_safety_policy,
            error=str(failure),
        )
        failure_summary["failure_archive"] = failure_info
        _atomic_json(
            paths["failure_summary"], failure_summary, overwrite=arguments.overwrite
        )
        failure_analysis = analyze_archive(
            paths["failure_archive"],
            paths["failure_analysis"],
            overwrite=arguments.overwrite,
            replay_result=None,
        )
        failure_plot = plot_archive(
            paths["failure_archive"],
            arguments.fem_archive,
            target_load=float(failure.last_committed_state.u_bar),
            fields_path=paths["failure_fields"],
            curve_path=paths["failure_stress_strain"],
            curve_data_path=paths["failure_curve_data"],
            report_path=paths["failure_plot_report"],
            overwrite=arguments.overwrite,
        )
        failure_summary["analysis"] = failure_analysis
        failure_summary["plots"] = failure_plot
        failure_summary["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(
            paths["failure_summary"], failure_summary, overwrite=True
        )
        _write_manifest(
            paths["manifest"],
            paths,
            source_metadata,
            engineering,
            config,
            device=device,
            command=sys.argv,
            validated=False,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=warning_safety_limit,
            warning_safety_policy=warning_safety_policy,
            overwrite=arguments.overwrite,
        )
        raise

    _atomic_engineering_archive(
        paths["archive"], result, overwrite=arguments.overwrite
    )
    try:
        replay = engineering.replay_engineering_grip_archive(paths["archive"])
    except Exception as error:
        failed_summary = _run_summary(
            validated=False,
            status="replay_failed",
            elapsed_seconds=time.perf_counter() - started,
            archive_path=paths["archive"],
            replay_path=paths["replay"],
            analysis_path=None,
            plot_report_path=None,
            state_summary=_state_summary_from_completed(
                _load_npz(paths["archive"], "completed archive")
            ),
            source_metadata=source_metadata,
            device=device,
            config_payload=config_payload,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=warning_safety_limit,
            warning_safety_policy=warning_safety_policy,
            error=str(error),
        )
        _atomic_json(paths["summary"], failed_summary, overwrite=arguments.overwrite)
        _write_manifest(
            paths["manifest"],
            paths,
            source_metadata,
            engineering,
            config,
            device=device,
            command=sys.argv,
            validated=False,
            acceptance_mode=acceptance_mode,
            warning_safety_limit=warning_safety_limit,
            warning_safety_policy=warning_safety_policy,
            overwrite=arguments.overwrite,
        )
        raise

    replay_payload = {
        "schema": "fracture2d-engineering-grip-replay-report-v1",
        "validated": bool(
            replay.get("replayed", False)
            and acceptance_mode != engineering._WARNING_ACCEPTANCE_MODE
        ),
        "archive_path": str(paths["archive"]),
        "archive_sha256": _sha256_file(paths["archive"]),
        "replay": replay,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(paths["replay"], replay_payload, overwrite=arguments.overwrite)
    analysis = analyze_archive(
        paths["archive"],
        paths["analysis"],
        overwrite=arguments.overwrite,
        replay_result=replay,
    )
    plot = plot_archive(
        paths["archive"],
        arguments.fem_archive,
        target_load=config.load_final,
        fields_path=paths["fields"],
        curve_path=paths["stress_strain"],
        curve_data_path=paths["curve_data"],
        report_path=paths["plot_report"],
        overwrite=arguments.overwrite,
    )
    values = _load_npz(paths["archive"], "completed archive")
    summary = _run_summary(
        validated=bool(result.validated),
        status=result.completion_status,
        elapsed_seconds=time.perf_counter() - started,
        archive_path=paths["archive"],
        replay_path=paths["replay"],
        analysis_path=paths["analysis"],
        plot_report_path=paths["plot_report"],
        state_summary=_state_summary_from_completed(values),
        source_metadata=source_metadata,
        device=device,
        config_payload=config_payload,
        acceptance_mode=acceptance_mode,
        warning_safety_limit=warning_safety_limit,
        warning_safety_policy=warning_safety_policy,
    )
    summary["replay"] = replay_payload
    summary["analysis"] = analysis
    summary["plots"] = plot
    _atomic_json(paths["summary"], summary, overwrite=arguments.overwrite)
    _write_manifest(
        paths["manifest"],
        paths,
        source_metadata,
        engineering,
        config,
        device=device,
        command=sys.argv,
        validated=bool(result.validated),
        acceptance_mode=acceptance_mode,
        warning_safety_limit=warning_safety_limit,
        warning_safety_policy=warning_safety_policy,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


def _archive_kind(values: Mapping[str, np.ndarray], engineering: Any) -> str:
    schema = _scalar_string(values["archive_schema"], "archive_schema")
    if schema in (
        engineering.ARCHIVE_SCHEMA,
        engineering.WARNING_ARCHIVE_SCHEMA,
        engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA,
    ):
        return "completed"
    if schema in (
        FAILURE_ARCHIVE_SCHEMA,
        UNBOUNDED_FAILURE_ARCHIVE_SCHEMA,
    ):
        return "failure"
    from . import nonlocal_engineering_grip_stabilized as stabilized

    if schema in (
        stabilized.STABILIZED_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_WARNING_ARCHIVE_SCHEMA,
    ):
        return "completed"
    if schema in (
        stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
    ):
        return "failure"
    raise ValueError(f"unsupported engineering-grip workflow archive schema: {schema}")


def _archive_strategy(values: Mapping[str, np.ndarray], engineering: Any) -> str:
    """Return the numerical controller family declared by an archive schema."""

    schema = _scalar_string(values["archive_schema"], "archive_schema")
    from . import nonlocal_engineering_grip_stabilized as stabilized

    if schema in (
        stabilized.STABILIZED_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_WARNING_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
    ):
        return "stabilized"
    if schema in (
        engineering.ARCHIVE_SCHEMA,
        engineering.WARNING_ARCHIVE_SCHEMA,
        engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA,
        FAILURE_ARCHIVE_SCHEMA,
        UNBOUNDED_FAILURE_ARCHIVE_SCHEMA,
    ):
        return "legacy"
    raise ValueError(f"unsupported engineering-grip archive schema: {schema}")


def _stabilized_replay_binding(
    archive_path: Path | None,
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    replay: Mapping[str, Any] | None,
    stabilized: Any,
) -> tuple[bool, dict[str, Any], str | None]:
    """Build and verify the cryptographic identity of one stabilized replay.

    The numerical replay is useful evidence only when it describes the exact
    archive currently on disk.  Counts and provenance are derived from typed
    archive arrays here rather than from replay self-reporting.  Source binding
    is checked through the core v5 loader so reports never silently fall back
    to archive-embedded geometry.
    """

    archive_schema = None
    completion_status = None
    expected: dict[str, Any] = {}
    try:
        archive_schema = _scalar_string(values["archive_schema"], "archive_schema")
        completion_status = _scalar_string(
            values["completion_status"], "completion_status"
        )
        acceptance_mode = _scalar_string(
            values["acceptance_mode"], "acceptance_mode"
        )
        failure = archive_schema in {
            stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA,
            stabilized.STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
        }
        contract = stabilized._stabilized_contract(
            acceptance_mode, failure=failure
        )
        if archive_schema != contract["archive_schema"]:
            raise ValueError("stabilized replay acceptance mode does not match archive")
        if completion_status != contract["completion_status"]:
            raise ValueError("stabilized replay completion status does not match archive")
        if archive_path is None:
            raise ValueError("stabilized replay binding needs an archive path")
        # The lean v5 layout retains one physical row per accepted state and
        # one compact solver-summary row per physical row.  Attempt/sweep/trial
        # rows are intentionally absent and must not be inferred.
        for field_name in ("state_u_bar", "solver_summary_json"):
            if field_name not in values:
                raise ValueError(f"archive is missing {field_name} for replay binding")
            array = np.asarray(values[field_name])
            if array.ndim != 1 or len(array) == 0:
                raise ValueError(
                    f"archive {field_name} must be a nonempty one-dimensional field"
                )
        state_rows = int(len(values["state_u_bar"]))
        summary_rows = int(len(values["solver_summary_json"]))
        counts = {
            "accepted_steps": state_rows - 1,
            "solver_summary_rows": summary_rows,
            "states": state_rows,
        }
        source_metadata = metadata.get("source")
        if not isinstance(source_metadata, Mapping):
            raise ValueError("stabilized archive source metadata is missing")
        source = stabilized.load_stabilized_source_cloud(
            source_metadata, metadata.get("delta")
        )
        del source
        if metadata.get("source_sha256") != source_metadata.get("sha256"):
            raise ValueError("stabilized archive source hash alias is forged")
        current_source_hash = stabilized._current_stabilized_source_bundle_hash()
        expected = {
            "archive_sha256": _sha256_file(archive_path),
            "archive_schema": archive_schema,
            "archive_layout": contract["archive_layout"],
            "replay_contract": contract["replay_contract"],
            "completion_status": completion_status,
            "source_bundle_hash": current_source_hash,
            "source_path": source_metadata.get("path"),
            "source_sha256": metadata.get("source_sha256"),
            **counts,
        }
        if acceptance_mode == stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE:
            expected["acceptance_mode"] = acceptance_mode
            expected["acceptance_status"] = contract["acceptance_status"]
        if metadata.get("acceptance_mode") != acceptance_mode:
            raise ValueError("stabilized archive metadata acceptance mode is forged")
        if metadata.get("acceptance_status") != contract["acceptance_status"]:
            raise ValueError("stabilized archive metadata acceptance status is forged")
        if metadata.get("layout") != contract["archive_layout"]:
            raise ValueError("stabilized archive metadata layout is forged")
        if metadata.get("replay_contract") != contract["replay_contract"]:
            raise ValueError("stabilized archive metadata replay contract is forged")
        metadata_source_hash = metadata.get("code_hash")
        if metadata_source_hash != current_source_hash:
            raise ValueError("archive source-bundle hash does not match current source")
    except (KeyError, OSError, TypeError, ValueError) as error:
        return False, expected, str(error)

    if not isinstance(replay, Mapping):
        return False, expected, "stabilized replay result is not a mapping"
    if type(replay.get("experimental")) is not bool or not replay["experimental"]:
        return False, expected, "stabilized replay result experimental flag is forged"
    if type(replay.get("field_equations_validated")) is not bool:
        return False, expected, (
            "stabilized replay field-equation validation flag is missing or mistyped"
        )
    if (
        expected.get("accepted_steps", 0) == 0
        and replay["field_equations_validated"] is not False
    ):
        return False, expected, (
            "stabilized replay cannot validate field equations without an "
            "accepted state"
        )
    if replay.get("path_replayed") is not False:
        return False, expected, "stabilized replay path-replayed flag is forged"
    if replay.get("rejected_trial_path_available") is not False:
        return False, expected, (
            "stabilized replay rejected-trial path flag is forged"
        )
    for name, value in expected.items():
        if name not in replay or type(replay[name]) is not type(value):
            return False, expected, f"stabilized replay binding field {name} is missing or mistyped"
        if replay[name] != value:
            return False, expected, f"stabilized replay binding field {name} does not match archive"
    # The core result retains ``schema`` as a legacy alias.  If present, bind
    # it too; the workflow report itself uses ``schema`` for its report schema.
    if "schema" in replay and replay["schema"] != expected["archive_schema"]:
        return False, expected, "stabilized replay schema alias does not match archive"
    return True, expected, None


def _stabilized_replay_semantics(
    replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expose the core v5 replay result without generic validation claims."""

    field_equations_validated = bool(
        isinstance(replay, Mapping)
        and type(replay.get("field_equations_validated")) is bool
        and replay["field_equations_validated"]
    )
    if isinstance(replay, Mapping) and replay.get("path_replayed") not in (
        None,
        False,
    ):
        raise ValueError("stabilized v5 replay cannot claim path replay")
    return {
        "experimental": True,
        "field_equations_validated": field_equations_validated,
        "path_replayed": False,
        "rejected_trial_path_available": False,
    }


def _metadata_from_archive(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    try:
        payload = json.loads(_scalar_string(values["metadata_json"], "metadata_json"))
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("archive metadata_json is missing or invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError("archive metadata_json must contain a mapping")
    return dict(payload)


def _stabilized_source_geometry(
    metadata: Mapping[str, Any],
    stabilized: Any,
) -> dict[str, np.ndarray]:
    """Load the stabilized geometry from its hashed source cloud.

    The lean v5 archive deliberately stores no geometry or static operators.
    Every derived report therefore reloads the exact source binary by path,
    SHA256, size, and field descriptors instead of trusting archive-embedded
    coordinates; there is no generic path fallback.
    """

    source = stabilized.load_stabilized_source_cloud(
        metadata["source"], metadata["delta"]
    )
    return {
        "coordinates": source["coordinates"],
        "triangles": source["triangles"],
    }


def _archive_storage_report(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return validated storage-budget provenance for derived reports."""

    storage = metadata.get("archive_storage_preflight")
    if storage is None:
        budget = metadata.get("archive_budget_bytes")
        if budget is None:
            return None
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("archive budget metadata is invalid")
        return {"archive_budget_bytes": budget}
    if not isinstance(storage, Mapping):
        raise ValueError("archive storage preflight metadata must be a mapping")
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
            "archive storage preflight metadata is missing fields: "
            f"{missing}"
        )
    values: dict[str, int] = {}
    for name in required:
        value = storage[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"archive storage preflight metadata {name} is invalid"
            )
        values[name] = value
    if values["archive_budget_bytes"] <= 0:
        raise ValueError("archive storage preflight metadata budget is invalid")
    if values["theoretical_worst_case_archive_bytes"] < values[
        "nominal_archive_bytes"
    ]:
        raise ValueError("archive storage preflight worst-case estimate is invalid")
    if values["budget_required_free_bytes"] != (
        values["archive_budget_bytes"] + values["filesystem_reserve_bytes"]
    ):
        raise ValueError("archive storage preflight free-space requirement is invalid")
    report = {
        "nominal_archive_bytes": values["nominal_archive_bytes"],
        "worst_case_archive_bytes": values[
            "theoretical_worst_case_archive_bytes"
        ],
        "theoretical_worst_case_archive_bytes": values[
            "theoretical_worst_case_archive_bytes"
        ],
        "archive_budget_bytes": values["archive_budget_bytes"],
        "free_bytes": values["available_bytes"],
        "available_bytes": values["available_bytes"],
        "filesystem_reserve_bytes": values["filesystem_reserve_bytes"],
        "budget_required_free_bytes": values["budget_required_free_bytes"],
        "nominal_fits_budget": values["nominal_archive_bytes"]
        <= values["archive_budget_bytes"],
        "theoretical_worst_case_fits_budget": values[
            "theoretical_worst_case_archive_bytes"
        ]
        <= values["archive_budget_bytes"],
    }
    if storage.get("archive_layout") in {
        "accepted-state-plus-final-solver-summary-v5",
        "accepted-state-plus-final-solver-summary-stabilized-warning-v1",
    }:
        row_fields = ("n_states", "accepted_steps", "solver_summary_rows")
        missing_rows = [name for name in row_fields if name not in storage]
        if missing_rows:
            raise ValueError(
                "v5 archive storage preflight is missing row counts: "
                f"{missing_rows}"
            )
        row_counts: dict[str, int] = {}
        for name in row_fields:
            value = storage[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"v5 archive storage row count {name} is invalid")
            row_counts[name] = value
        if row_counts["n_states"] != row_counts["accepted_steps"] + 1:
            raise ValueError("v5 archive storage state/accepted row counts are invalid")
        if row_counts["solver_summary_rows"] != row_counts["n_states"]:
            raise ValueError(
                "v5 archive storage state/solver-summary row counts are invalid"
            )
        report.update(
            {
                "archive_layout": storage["archive_layout"],
                "static_bound_kind": storage.get("static_bound_kind"),
                "physical_rows": {
                    "states": row_counts["n_states"],
                    "accepted_steps": row_counts["accepted_steps"],
                },
                "scalar_rows": {
                    "solver_summaries": row_counts["solver_summary_rows"],
                },
            }
        )
    return report


def _archive_acceptance_details(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    kind: str,
    engineering: Any,
) -> tuple[str, bool, str | None, float | None]:
    """Validate archive acceptance provenance for completed and failure audits."""

    archive_schema = _scalar_string(values["archive_schema"], "archive_schema")
    from . import nonlocal_engineering_grip_stabilized as stabilized

    stabilized_completed_schemas = {
        stabilized.STABILIZED_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_WARNING_ARCHIVE_SCHEMA,
    }
    stabilized_failure_schemas = {
        stabilized.STABILIZED_FAILURE_ARCHIVE_SCHEMA,
        stabilized.STABILIZED_WARNING_FAILURE_ARCHIVE_SCHEMA,
    }
    if archive_schema in stabilized_completed_schemas | stabilized_failure_schemas:
        expected_kind = (
            "completed"
            if archive_schema in stabilized_completed_schemas
            else "failure"
        )
        if kind != expected_kind:
            raise ValueError("stabilized archive kind does not match its schema")
        acceptance_mode = _scalar_string(values["acceptance_mode"], "acceptance_mode")
        contract = stabilized._stabilized_contract(
            acceptance_mode, failure=expected_kind == "failure"
        )
        if archive_schema != contract["archive_schema"]:
            raise ValueError("stabilized archive acceptance mode is forged")
        if "acceptance_status" not in values or _scalar_string(
            values["acceptance_status"], "acceptance_status"
        ) != contract["acceptance_status"]:
            raise ValueError("stabilized archive acceptance status is forged")
        if metadata.get("acceptance_mode") != contract["acceptance_mode"]:
            raise ValueError("stabilized archive metadata acceptance mode is forged")
        if metadata.get("acceptance_status") != contract["acceptance_status"]:
            raise ValueError("stabilized archive metadata acceptance status is forged")
        if metadata.get("replay_contract") != contract["replay_contract"]:
            raise ValueError("stabilized archive replay contract is forged")
        if metadata.get("layout") != contract["archive_layout"]:
            raise ValueError("stabilized archive layout is forged")
        expected_completion_status = contract["completion_status"]
        completion_status = _scalar_string(
            values["completion_status"], "completion_status"
        )
        if completion_status != expected_completion_status:
            raise ValueError("stabilized archive completion status is forged")
        if (
            metadata.get("schema") != archive_schema
            or metadata.get("kind") != kind
            or metadata.get("completion_status") != expected_completion_status
        ):
            raise ValueError("stabilized archive metadata identity is forged")
        validated = _require_bool(values["validated"], "stabilized archive validated")
        experimental = _require_bool(
            values["experimental"], "stabilized archive experimental"
        )
        if bool(validated) or not bool(experimental):
            raise ValueError("stabilized archive validation flags are forged")
        if metadata.get("experimental") is not True or metadata.get("validated") is not False:
            raise ValueError("stabilized archive validation flags are forged")
        if metadata.get("field_equations_validated") is not False or metadata.get(
            "path_replayed"
        ) is not False or metadata.get("rejected_trial_path_available") is not False:
            raise ValueError("stabilized archive replay semantics are forged")
        if acceptance_mode == stabilized.STABILIZED_ACCEPTANCE_MODE:
            forbidden = {
                "warning_safety_limit",
                "warning_safety_policy",
                "warning_safety_limit_present",
            }
            present = sorted(forbidden & set(values))
            if present:
                raise ValueError(
                    f"stabilized archive contains warning acceptance fields: {present}"
                )
            if any(name in metadata for name in forbidden):
                raise ValueError(
                    "stabilized archive metadata contains warning acceptance fields"
                )
        else:
            if metadata.get("warning_hard_gate_names") != list(
                stabilized.STABILIZED_WARNING_HARD_GATE_NAMES
            ) or metadata.get("warning_bypassable_gate_names") != list(
                stabilized.STABILIZED_WARNING_GATE_NAMES
            ):
                raise ValueError("stabilized warning gate metadata is forged")
        return (
            acceptance_mode,
            acceptance_mode == stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE,
            "unbounded"
            if acceptance_mode == stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE
            else None,
            None,
        )

    acceptance_mode = _scalar_string(values["acceptance_mode"], "acceptance_mode")
    acceptance_mode = engineering._validate_acceptance_mode(acceptance_mode)
    expected_policy: str | None = None
    if kind == "completed":
        if archive_schema == engineering.ARCHIVE_SCHEMA:
            warning = False
            expected_mode = "strict"
        elif archive_schema in (
            engineering.WARNING_ARCHIVE_SCHEMA,
            engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA,
        ):
            warning = True
            expected_mode = "warning"
        else:
            raise ValueError("completed archive schema is invalid")
        if acceptance_mode != expected_mode:
            raise ValueError("archive acceptance mode does not match its schema")
    elif kind == "failure":
        if archive_schema not in (FAILURE_ARCHIVE_SCHEMA, UNBOUNDED_FAILURE_ARCHIVE_SCHEMA):
            raise ValueError("failure archive schema is invalid")
        warning = acceptance_mode == engineering._WARNING_ACCEPTANCE_MODE
        if archive_schema == UNBOUNDED_FAILURE_ARCHIVE_SCHEMA and not warning:
            raise ValueError("unbounded failure archive must use warning acceptance mode")
        if archive_schema == FAILURE_ARCHIVE_SCHEMA and warning:
            expected_policy = engineering._WARNING_SAFETY_POLICY_BOUNDED
        elif archive_schema == UNBOUNDED_FAILURE_ARCHIVE_SCHEMA:
            expected_policy = engineering._WARNING_SAFETY_POLICY_UNBOUNDED
        else:
            expected_policy = None
    else:
        raise ValueError(f"unsupported engineering-grip archive kind: {kind}")

    if warning:
        if kind == "completed":
            expected_policy = (
                engineering._WARNING_SAFETY_POLICY_UNBOUNDED
                if archive_schema == engineering.UNBOUNDED_WARNING_ARCHIVE_SCHEMA
                else engineering._WARNING_SAFETY_POLICY_BOUNDED
            )
        policy = engineering._validate_warning_safety_policy(expected_policy)
        if policy == engineering._WARNING_SAFETY_POLICY_UNBOUNDED:
            if metadata.get("warning_safety_policy") != policy:
                raise ValueError("unbounded archive warning safety policy is forged")
            if metadata.get("warning_safety_limit_present") is not False:
                raise ValueError("unbounded archive warning limit presence is forged")
            if "warning_safety_limit" in metadata:
                raise ValueError("unbounded archive metadata contains a safety limit")
            if "warning_safety_policy" not in values or "warning_safety_limit_present" not in values:
                raise ValueError("unbounded archive warning policy fields are missing")
            if _scalar_string(values["warning_safety_policy"], "warning_safety_policy") != policy:
                raise ValueError("unbounded archive warning safety policy is forged")
            limit_present = _require_bool(
                values["warning_safety_limit_present"],
                "warning_safety_limit_present",
            )
            if bool(limit_present):
                raise ValueError("unbounded archive warning limit presence is forged")
            warning_safety_limit = None
        else:
            warning_safety_limit = engineering._validate_warning_safety_limit(
                metadata.get("warning_safety_limit")
            )
            if "warning_safety_limit" in values:
                archive_limit = _require_float64(
                    values["warning_safety_limit"],
                    "warning_safety_limit",
                )
                if archive_limit.shape != () or float(archive_limit) != warning_safety_limit:
                    raise ValueError("archive warning safety limit is forged")
    else:
        policy = None
        warning_safety_limit = None
    return acceptance_mode, warning, policy, warning_safety_limit


def _archive_acceptance(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    kind: str,
    engineering: Any,
) -> tuple[str, bool, float | None]:
    """Return the historical acceptance tuple for workflow callers/tests."""

    mode, warning, _, warning_safety_limit = _archive_acceptance_details(
        values, metadata, kind, engineering
    )
    return mode, warning, warning_safety_limit


class _MissingExactLoadError(ValueError):
    """Raised when a valid load grid has no exact target entry."""


def _exact_load_index(loads: np.ndarray, target: float, name: str) -> int:
    values = np.asarray(loads, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite nonempty one-dimensional load array")
    tolerance = 16.0 * np.finfo(np.float64).eps * max(
        1.0, abs(float(target)), float(np.max(np.abs(values)))
    )
    matches = np.flatnonzero(np.abs(values - float(target)) <= tolerance)
    if len(matches) == 0:
        raise _MissingExactLoadError(
            f"{name} must contain exactly one load {target:.17g}; found 0"
        )
    if len(matches) != 1:
        raise ValueError(
            f"{name} must contain exactly one load {target:.17g}; found {len(matches)}"
        )
    return int(matches[0])


def _fem_target_response(
    response_load: np.ndarray,
    response_stress: np.ndarray,
    target_load: float,
) -> tuple[float | None, int | None, dict[str, Any] | None]:
    """Look up or interpolate FEM stress on the response-load grid.

    FEM field snapshots and response histories are saved on independent load
    grids.  Keep this response lookup separate from the snapshot index used to
    select plotted fields.
    """

    loads = np.asarray(response_load)
    stress = np.asarray(response_stress)
    target = float(target_load)
    if not np.isfinite(target):
        raise ValueError("FEM target load must be finite")
    if (
        loads.ndim != 1
        or stress.ndim != 1
        or len(loads) != len(stress)
        or len(loads) == 0
        or not np.all(np.isfinite(loads))
        or not np.all(np.isfinite(stress))
        or np.any(np.diff(loads) < 0.0)
    ):
        raise ValueError(
            "FEM response loads and stresses must be finite, nonempty, "
            "one-dimensional arrays with non-decreasing loads"
        )

    try:
        response_index = _exact_load_index(
            loads, target, "FEM response loads"
        )
    except ValueError:
        response_index = None
    if response_index is not None:
        return (
            float(stress[response_index]),
            response_index,
            {
                "method": "exact_response_load",
                "response_index": response_index,
                "response_load": float(loads[response_index]),
            },
        )

    if target < float(loads[0]) or target > float(loads[-1]):
        return None, None, None

    target_stress = float(np.interp(target, loads, stress))
    upper = int(np.searchsorted(loads, target, side="right"))
    if upper <= 0:
        lower = upper = 0
    elif upper >= len(loads):
        lower = upper = len(loads) - 1
    else:
        lower = upper - 1
    lower_load = float(loads[lower])
    upper_load = float(loads[upper])
    if upper == lower or upper_load == lower_load:
        upper_weight: float | None = None
    else:
        upper_weight = float((target - lower_load) / (upper_load - lower_load))
    return (
        target_stress,
        None,
        {
            "method": "linear_interpolation",
            "implementation": "numpy.interp",
            "coordinate": "fem_response_load",
            "target_load": target,
            "lower_response_index": lower,
            "upper_response_index": upper,
            "lower_response_load": lower_load,
            "upper_response_load": upper_load,
            "upper_weight": upper_weight,
        },
    )


def _failure_diagnostic_payloads(values: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    if "solver_summary_json" in values:
        payloads: list[dict[str, Any]] = []
        for raw in values["solver_summary_json"]:
            payload = json.loads(str(raw))
            if not isinstance(payload, Mapping):
                raise ValueError("v5 solver summary JSON must contain mappings")
            payloads.append(dict(payload))
        if "failure_summary_json" in values:
            failure_summary = json.loads(str(values["failure_summary_json"]))
            if not isinstance(failure_summary, Mapping):
                raise ValueError("v5 failure summary JSON must contain a mapping")
            if failure_summary.get("available"):
                candidate = failure_summary.get("solver_summary")
                if not isinstance(candidate, Mapping):
                    raise ValueError("v5 available failure summary lacks solver summary")
                payloads.append(dict(candidate))
        return payloads
    field = (
        "sweep_log_json" if "sweep_log_json" in values else "sweep_diagnostics_json"
    )
    payloads: list[dict[str, Any]] = []
    for raw in values[field]:
        payload = json.loads(str(raw))
        if not isinstance(payload, Mapping):
            raise ValueError("failure sweep diagnostic JSON must contain mappings")
        payloads.append(dict(payload))
    return payloads


def _analysis_gate_ratios(
    diagnostics: Sequence[Mapping[str, Any]],
    config: Any,
) -> dict[str, Any]:
    specs = (
        ("displacement_material", "e_u_material", "displacement_scale", "displacement"),
        ("displacement_union", "e_u_union", "displacement_scale", "displacement"),
        ("damage", "e_damage", "damage_scale", "damage"),
        ("history", "e_history", "history_scale", "history"),
        ("reaction", "e_reaction", "reaction_scale", "reaction"),
        ("rho_used", "rho_used_inf", "damage_residual_scale", "damage_residual"),
        ("rho_next", "rho_next_inf", "damage_residual_scale", "damage_residual"),
        ("force", "force_error", "force_scale", "force"),
        ("moment", "moment_error", "moment_scale", "moment"),
    )
    result: dict[str, Any] = {}
    for name, error_name, scale_name, gate_name in specs:
        candidates: list[tuple[int, float, float, float]] = []
        gate = getattr(config, gate_name)
        for index, diagnostic in enumerate(diagnostics):
            error = diagnostic.get(error_name)
            scale = diagnostic.get(scale_name)
            if error is None or scale is None:
                continue
            error_value = float(error)
            limit = float(gate.limit(float(scale)))
            if not np.isfinite(error_value) or not np.isfinite(limit) or limit <= 0.0:
                continue
            candidates.append((index, error_value / limit, error_value, limit))
        if not candidates:
            result[name] = {"available": False}
            continue
        index, ratio, error, limit = max(candidates, key=lambda item: item[1])
        result[name] = {
            "available": True,
            "max_ratio": ratio,
            "sweep_index": index,
            "target_u_bar": float(diagnostics[index]["target_u_bar"]),
            "error": error,
            "limit": limit,
        }
    for prefix in ("before", "after"):
        values = []
        for index, diagnostic in enumerate(diagnostics):
            relative = diagnostic.get(f"free_residual_relative_{prefix}")
            if relative is None:
                continue
            values.append(
                (
                    index,
                    float(relative) / max(float(config.cg_rtol), np.finfo(float).tiny),
                    float(relative),
                )
            )
        key = f"mechanics_{prefix}_relative_to_cg_rtol_proxy"
        if values:
            index, ratio, relative = max(values, key=lambda item: item[1])
            result[key] = {
                "available": True,
                "max_ratio": ratio,
                "sweep_index": index,
                "target_u_bar": float(diagnostics[index]["target_u_bar"]),
                "relative_residual": relative,
                "reference": float(config.cg_rtol),
                "interpretation": (
                    "diagnostic proxy; the authoritative mechanics gate also uses "
                    "the absolute CG tolerance and reconstructed RHS norm"
                ),
            }
        else:
            result[key] = {"available": False}
    return result


def _field_maxima(
    coordinates: np.ndarray,
    delta: float,
    damage: np.ndarray,
    history: np.ndarray,
) -> dict[str, Any]:
    from . import nonlocal_engineering_grip as engineering

    mask = engineering.build_crack_region_mask(coordinates, delta)
    interface = (np.abs(coordinates[:, 1]) <= 1.0e-12) | (
        np.abs(coordinates[:, 1] - 1.0) <= 1.0e-12
    )

    def extrema(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inside = np.max(values[:, mask], axis=1)
        outside = (
            np.max(values[:, ~mask], axis=1)
            if np.any(~mask)
            else np.zeros(len(values), dtype=np.float64)
        )
        return inside, outside

    damage_inside, damage_outside = extrema(damage)
    history_inside, history_outside = extrema(history)
    interface_damage = np.max(damage[:, interface], axis=1)
    interface_history = np.max(history[:, interface], axis=1)
    damage_final_index = int(np.argmax(damage[-1]))
    history_final_index = int(np.argmax(history[-1]))
    return {
        "crack_region_particles": int(np.count_nonzero(mask)),
        "state_crack_damage_inside_max": damage_inside.tolist(),
        "state_crack_damage_outside_max": damage_outside.tolist(),
        "state_crack_history_inside_max": history_inside.tolist(),
        "state_crack_history_outside_max": history_outside.tolist(),
        "state_interface_damage_max": np.max(interface_damage, axis=0).item()
        if interface_damage.ndim > 1
        else interface_damage.tolist(),
        "state_interface_history_max": np.max(interface_history, axis=0).item()
        if interface_history.ndim > 1
        else interface_history.tolist(),
        "final_damage_max_coordinate": coordinates[damage_final_index].tolist(),
        "final_history_max_coordinate": coordinates[history_final_index].tolist(),
        "final_damage_max": float(np.max(damage[-1])),
        "final_history_max": float(np.max(history[-1])),
        "final_damage_inside_max": float(damage_inside[-1]),
        "final_damage_outside_max": float(damage_outside[-1]),
        "final_history_inside_max": float(history_inside[-1]),
        "final_history_outside_max": float(history_outside[-1]),
        "final_interface_damage_max": float(interface_damage[-1]),
        "final_interface_history_max": float(interface_history[-1]),
        "minimum_damage_localization_margin": float(
            np.min(damage_inside - damage_outside)
        ),
        "minimum_history_localization_margin": float(
            np.min(history_inside - history_outside)
        ),
        "damage_monotone": bool(np.all(np.diff(damage, axis=0) >= -1.0e-10)),
        "history_monotone": bool(np.all(np.diff(history, axis=0) >= 0.0)),
    }


def analyze_archive(
    archive_path: Path,
    report_path: Path,
    *,
    overwrite: bool,
    replay_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Analyze either a completed archive or a failure audit."""

    from . import nonlocal_engineering_grip as engineering

    values = _load_npz(archive_path, "engineering-grip archive")
    kind = _archive_kind(values, engineering)
    strategy = _archive_strategy(values, engineering)
    metadata = _metadata_from_archive(values)
    stabilized = None
    if strategy == "stabilized":
        from . import nonlocal_engineering_grip_stabilized as stabilized_module

        stabilized = stabilized_module
        config = engineering._config_from_payload(metadata["config"])
        stabilized_config = stabilized.StabilizedEngineeringGripConfig.from_payload(
            metadata["stabilized_config"], config
        )
        analysis_config = config
    else:
        config = engineering._config_from_payload(metadata["config"])
        stabilized_config = None
        analysis_config = config
    (
        acceptance_mode,
        warning,
        warning_safety_policy,
        warning_safety_limit,
    ) = _archive_acceptance_details(
        values, metadata, kind, engineering
    )
    independent_replay: Mapping[str, Any] | None = replay_result
    stabilized_geometry: dict[str, np.ndarray] | None = None
    if strategy == "stabilized":
        assert stabilized is not None
        stabilized_geometry = _stabilized_source_geometry(metadata, stabilized)
    if kind == "completed":
        if replay_result is None:
            if strategy == "stabilized":
                assert stabilized is not None
                replay_result = stabilized.replay_stabilized_engineering_grip_archive(
                    archive_path
                )
            else:
                replay_result = engineering.replay_engineering_grip_archive(archive_path)
        independent_replay = replay_result
        if stabilized_geometry is not None:
            coordinates = _require_float64(
                stabilized_geometry["coordinates"], "analysis coordinates"
            )
        else:
            coordinates = _require_float64(
                values["material_coordinates"], "analysis coordinates"
            )
        loads = _require_float64(values["state_u_bar"], "analysis state loads")
        damage = _require_float64(values["state_material_damage"], "analysis state damage")
        history = _require_float64(values["state_material_history"], "analysis state history")
        reactions = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                _require_float64(
                    values["accepted_top_reaction_after"],
                    "analysis accepted top reaction",
                ),
            )
        )
        if strategy == "stabilized":
            diagnostics = _failure_diagnostic_payloads(values)
            attempt_count: int | None = None
        else:
            diagnostics = [
                json.loads(str(raw)) for raw in values["accepted_diagnostics_json"]
            ]
            attempt_count = int(len(values["attempt_load_target"]))
        accepted_count = (
            int(max(0, len(loads) - 1))
            if strategy == "stabilized"
            else int(len(values["accepted_load_target"]))
        )
        failure_message = None
    else:
        if strategy == "stabilized":
            assert stabilized_geometry is not None
            coordinates = _require_float64(
                stabilized_geometry["coordinates"], "analysis coordinates"
            )
            loads = _require_float64(
                values["state_u_bar"], "analysis state loads"
            )
            damage = _require_float64(
                values["state_material_damage"], "analysis state damage"
            )
            history = _require_float64(
                values["state_material_history"], "analysis state history"
            )
            reactions = np.concatenate(
                (
                    np.asarray([0.0], dtype=np.float64),
                    _require_float64(
                        values["accepted_top_reaction_after"],
                        "analysis accepted top reaction",
                    ),
                )
            )
        else:
            coordinates = _require_float64(
                values["material_coordinates"], "analysis coordinates"
            )
            loads = _require_float64(
                values["accepted_state_u_bar"], "analysis state loads"
            )
            damage = _require_float64(
                values["accepted_state_d"], "analysis state damage"
            )
            history = _require_float64(
                values["accepted_state_H"], "analysis state history"
            )
            reactions = _require_float64(
                values["accepted_top_reaction"], "analysis accepted top reaction"
            )
        if strategy == "stabilized":
            diagnostics = _failure_diagnostic_payloads(values)
            attempt_count = None
        else:
            diagnostics = _failure_diagnostic_payloads(values)
            attempt_count = int(len(values["attempt_load_target"]))
        accepted_count = int(max(0, len(loads) - 1))
        failure_message = _scalar_string(values["failure_message"], "failure_message")
        if strategy == "stabilized" and replay_result is None:
            assert stabilized is not None
            replay_result = stabilized.replay_stabilized_engineering_grip_archive(
                archive_path
            )
            independent_replay = replay_result
    stabilized_binding: dict[str, Any] = {}
    stabilized_binding_valid = False
    stabilized_binding_error: str | None = None
    stabilized_semantics: dict[str, Any] = {}
    if strategy == "stabilized":
        assert stabilized is not None
        (
            stabilized_binding_valid,
            stabilized_binding,
            stabilized_binding_error,
        ) = _stabilized_replay_binding(
            archive_path,
            values,
            metadata,
            independent_replay,
            stabilized,
        )
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or damage.ndim != 2
        or history.shape != damage.shape
        or len(loads) != len(damage)
        or len(reactions) != len(loads)
    ):
        raise ValueError("archive state arrays have incompatible analysis shapes")
    if np.any(np.diff(loads) < 0.0):
        raise ValueError("archive state loads are not non-decreasing")
    field_metrics = _field_maxima(
        coordinates,
        float(metadata["delta"]),
        damage,
        history,
    )
    reaction_difference = np.diff(reactions)
    ratios = _analysis_gate_ratios(diagnostics, analysis_config)
    validated = False
    if strategy == "stabilized":
        stabilized_semantics = _stabilized_replay_semantics(independent_replay)
    else:
        validated = bool(kind == "completed" and not warning)
    report: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "archive_kind": kind,
        "continuation_strategy": strategy,
        "experimental": strategy == "stabilized",
        "nonproduction": True,
        "completion_status": _scalar_string(
            values["completion_status"], "completion_status"
        ),
        "acceptance": {
            **_acceptance_payload(
                acceptance_mode, warning_safety_limit, warning_safety_policy
            ),
        },
        "accepted_steps": accepted_count,
        "final_u_bar": float(loads[-1]),
        "reaction_increment_min": float(np.min(reaction_difference))
        if len(reaction_difference)
        else 0.0,
        "reaction_increment_max": float(np.max(reaction_difference))
        if len(reaction_difference)
        else 0.0,
        "reaction_monotone": bool(np.all(reaction_difference > 0.0))
        if len(reaction_difference)
        else True,
        "reaction_balance_max_abs": None,
        "gate_ratios": ratios,
        "field_metrics": field_metrics,
        "configuration": (
            _workflow_config_payload(stabilized_config, "stabilized", engineering)
            if strategy == "stabilized"
            else engineering._config_payload(config)
        ),
    }
    if attempt_count is None:
        report["solver_summary_rows"] = int(len(values["solver_summary_json"]))
        report["state_external_work"] = _require_float64(
            values["state_external_work"], "analysis external work"
        ).tolist()
    else:
        report["attempts"] = attempt_count
    if strategy == "stabilized":
        report.update(stabilized_semantics)
    else:
        report["validated"] = validated
    archive_storage = _archive_storage_report(metadata)
    if archive_storage is not None:
        report["archive_storage_preflight"] = archive_storage
    if strategy == "stabilized":
        report["replay_binding"] = dict(stabilized_binding)
        report["replay_binding_valid"] = stabilized_binding_valid
        if stabilized_binding_error is not None:
            report["replay_binding_error"] = stabilized_binding_error
    if warning:
        gate_names = tuple(str(name) for name in np.asarray(values["warning_gate_names"]))
        if strategy == "stabilized":
            from . import nonlocal_engineering_grip_stabilized as stabilized

            if gate_names != stabilized.STABILIZED_WARNING_GATE_NAMES:
                raise ValueError("archive warning gate names are not canonical")
            state_status = np.asarray(values["state_acceptance_status"])
            if state_status.dtype.kind not in "SU" or len(state_status) != len(loads):
                raise ValueError("archive warning state statuses are invalid")
            accepted_status = state_status[1:]
            allowed = {
                stabilized.STABILIZED_ACCEPTANCE_STATUS,
                stabilized.STABILIZED_WARNING_ACCEPTANCE_STATUS,
            }
            if not set(str(value) for value in accepted_status) <= allowed:
                raise ValueError("archive warning accepted statuses are invalid")
            warning_present = np.asarray(values["accepted_warning_present"])
            bypassed_mask = np.asarray(values["state_bypassed_gate_mask"])
            report["warning_acceptance"] = {
                "safety_policy": warning_safety_policy,
                "safety_limit": None,
                "safety_limit_present": False,
                "accepted_statuses": [str(value) for value in accepted_status],
                "warning_accepted_steps": int(np.count_nonzero(warning_present)),
                "bypassed_gate_counts": {
                    name: int(np.count_nonzero(bypassed_mask[:, index]))
                    for index, name in enumerate(gate_names)
                },
                "range_excess_max": float(
                    np.max(values["accepted_warning_range_excess"], initial=0.0)
                ),
                "healing_magnitude_max": float(
                    np.max(values["accepted_warning_healing_magnitude"], initial=0.0)
                ),
            }
        elif gate_names != engineering.WARNING_GATE_NAMES:
            raise ValueError("archive warning gate names are not canonical")
        else:
            accepted_status_name = (
            "accepted_acceptance_status"
            if kind == "completed"
            else "accepted_state_acceptance_status"
            )
            accepted_status = np.asarray(values[accepted_status_name])
            if accepted_status.dtype.kind not in "SU":
                raise ValueError("archive warning accepted statuses must be strings")
            if kind == "completed":
                allowed = {"strict_passed", "warning_accepted"}
            else:
                allowed = {"initial", "strict_passed", "warning_accepted"}
            if not set(str(value) for value in accepted_status) <= allowed:
                raise ValueError("archive warning accepted statuses are invalid")
            report["warning_acceptance"] = {
                "safety_policy": warning_safety_policy,
                "safety_limit": warning_safety_limit,
                "safety_limit_present": warning_safety_limit is not None,
                "accepted_statuses": [str(value) for value in accepted_status],
                "warning_accepted_steps": int(
                    np.count_nonzero(accepted_status == "warning_accepted")
                ),
                "attempt_statuses": [
                    str(value)
                    for value in np.asarray(values["attempt_acceptance_status"])
                ],
            }
    if kind == "completed":
        bottom_field = (
            "accepted_bottom_reaction_after"
            if strategy == "stabilized"
            else "accepted_bottom_reaction"
        )
        anchor_field = (
            "accepted_anchor_reaction_after"
            if strategy == "stabilized"
            else "accepted_anchor_reaction"
        )
        bottom = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                _require_float64(values[bottom_field], "analysis bottom reaction"),
            )
        )
        anchor = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                _require_float64(values[anchor_field], "analysis anchor reaction"),
            )
        )
        report["reaction_balance_max_abs"] = float(
            np.max(np.abs(reactions + bottom + anchor))
        )
        if strategy == "stabilized":
            accepted_summaries = [
                json.loads(str(raw)) for raw in values["solver_summary_json"][1:]
            ]
            report["accepted_stagger_iterations_max"] = int(
                max(int(item["stagger_iterations"]) for item in accepted_summaries)
            )
            report["accepted_cutbacks_total"] = int(
                sum(int(item["cutback_count"]) for item in accepted_summaries)
            )
        else:
            report["accepted_stagger_iterations_max"] = int(
                np.max(values["accepted_stagger_iterations"])
            )
            report["accepted_cutbacks_total"] = int(
                np.sum(values["accepted_cutback_count"])
            )
        report["independent_replay"] = (
            dict(independent_replay)
            if isinstance(independent_replay, Mapping)
            else {}
        )
    else:
        report["failure_message"] = failure_message
        if strategy == "stabilized":
            assert stabilized is not None
            failure_summary_raw = _scalar_string(
                values["failure_summary_json"], "failure_summary_json"
            )
            failure_summary_payload = json.loads(failure_summary_raw)
            if not isinstance(failure_summary_payload, Mapping):
                raise ValueError("stabilized failure summary must contain a mapping")
            if failure_summary_payload.get("available"):
                report["failure_stage"] = str(
                    failure_summary_payload["failure_stage"]
                )
                report["failure_reason"] = str(
                    failure_summary_payload["failure_reason"]
                )
                solver_summary = failure_summary_payload.get("solver_summary")
                if not isinstance(solver_summary, Mapping):
                    raise ValueError("stabilized failure summary lacks solver summary")
                failed_sweeps = int(solver_summary["stagger_iterations"])
                report["failed_attempt_sweep_count"] = failed_sweeps
                report["failed_sweep_count"] = failed_sweeps
            else:
                report["failure_stage"] = "archive_budget" if metadata.get(
                    "failure_code"
                ) == stabilized.STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE else "continuation"
                report["failure_reason"] = str(
                    failure_summary_payload["failure_reason"]
                )
                report["failed_attempt_sweep_count"] = 0
                report["failed_sweep_count"] = 0
        elif attempt_count == 0:
            if not (
                strategy == "stabilized"
                and stabilized is not None
                and metadata.get("failure_code")
                == stabilized.STABILIZED_ARCHIVE_BUDGET_FAILURE_CODE
            ):
                raise ValueError("failure archive has no attempt audit")
            report["failure_stage"] = "archive_budget"
            report["failure_reason"] = failure_message
            report["failed_attempt_sweep_count"] = 0
            report["failed_sweep_count"] = 0
        else:
            report["failure_stage"] = _scalar_string(
                values["attempt_failure_stage"][-1], "failure stage"
            )
            report["failure_reason"] = _scalar_string(
                values["attempt_failure_reason"][-1], "failure reason"
            )
            report["failed_attempt_sweep_count"] = int(
                values["attempt_sweep_count"][-1]
            )
            report["failed_sweep_count"] = int(
                np.count_nonzero(
                    values["sweep_attempt_index"]
                    == len(values["attempt_load_target"]) - 1
                )
            )
        report["last_committed_u_bar"] = float(values["last_committed_u_bar"])
        if strategy == "stabilized":
            report["last_committed_damage_max"] = float(
                values["last_committed_damage_max"]
            )
            report["last_committed_history_max"] = float(
                values["last_committed_history_max"]
            )
        else:
            report["last_committed_damage_max"] = float(
                np.max(values["last_committed_d"])
            )
            report["last_committed_history_max"] = float(
                np.max(values["last_committed_H"])
            )
        if strategy == "stabilized":
            report["independent_replay"] = (
                dict(independent_replay)
                if isinstance(independent_replay, Mapping)
                else {}
            )
            budget_failure = metadata.get("archive_budget_failure")
            if budget_failure is not None:
                if not isinstance(budget_failure, Mapping):
                    raise ValueError("archive budget failure details are invalid")
                report["archive_budget_failure"] = dict(budget_failure)
    _atomic_json(report_path, report, overwrite=overwrite)
    return report


def _load_fem_archive(path: Path) -> dict[str, Any]:
    values = _load_npz(path, "FEM archive")
    required = {
        "coords",
        "triangles",
        "u_bar",
        "reaction_y",
        "nominal_strain",
        "nominal_stress",
        "snapshot_u_bar",
        "u",
        "d",
        "gate_mode",
        "boundary_mode",
        "plane_assumption",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"FEM archive is missing required fields: {missing}")
    coords = _require_float64(values["coords"], "FEM coordinates")
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError("FEM coordinates must have nonempty shape (N, 2)")
    # The FEM writer stores connectivity as int32.  Unlike the material source
    # contract, this independent comparison input is allowed to use any safe
    # integer dtype and is normalized after range validation.
    triangles_raw = _require_integer_indices(values["triangles"], "FEM triangles")
    triangles = triangles_raw
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise ValueError("FEM triangles must have nonempty shape (T, 3)")
    if np.any(triangles < 0) or np.any(triangles >= len(coords)):
        raise ValueError("FEM triangles contain an out-of-range index")
    gate_mode = _scalar_string(values["gate_mode"], "FEM gate_mode")
    boundary_mode = _scalar_string(values["boundary_mode"], "FEM boundary_mode")
    plane_assumption = _scalar_string(
        values["plane_assumption"], "FEM plane_assumption"
    )
    if gate_mode != "warning":
        raise ValueError(
            "the comparison FEM archive must explicitly declare gate_mode='warning'"
        )
    if boundary_mode != "bottom_vertical_point_horizontal":
        raise ValueError(
            "the comparison FEM archive must be the accepted point-anchor mode"
        )
    if plane_assumption != "plane_strain":
        raise ValueError("the comparison FEM archive must declare plane_strain")
    response_load = _require_float64(values["u_bar"], "FEM response load")
    reaction = _require_float64(values["reaction_y"], "FEM reaction")
    nominal_strain = _require_float64(values["nominal_strain"], "FEM nominal strain")
    nominal_stress = _require_float64(values["nominal_stress"], "FEM nominal stress")
    if not (
        response_load.ndim == reaction.ndim == nominal_strain.ndim == nominal_stress.ndim == 1
        and len(response_load) == len(reaction) == len(nominal_strain) == len(nominal_stress)
    ):
        raise ValueError("FEM response arrays have incompatible shapes")
    if len(response_load) == 0 or np.any(np.diff(response_load) < 0.0):
        raise ValueError("FEM response loads must be non-decreasing and nonempty")
    snapshot_load = _require_float64(values["snapshot_u_bar"], "FEM snapshot loads")
    displacement = _require_float64(values["u"], "FEM displacement")
    damage = _require_float64(values["d"], "FEM damage")
    if (
        snapshot_load.ndim != 1
        or displacement.ndim != 3
        or displacement.shape[2] != 2
        or damage.ndim != 2
        or displacement.shape[0] != len(snapshot_load)
        or damage.shape != displacement.shape[:2]
        or displacement.shape[1] != len(coords)
    ):
        raise ValueError("FEM snapshot arrays have incompatible shapes")
    lx = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    ly = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    if lx <= 0.0 or ly <= 0.0 or not np.isfinite(lx + ly):
        raise ValueError("FEM coordinate extents must be positive and finite")
    expected_strain = response_load / ly
    expected_stress = reaction / lx
    if not np.allclose(nominal_strain, expected_strain, rtol=0.0, atol=2.0e-14):
        raise ValueError("FEM nominal_strain does not equal u_bar/L_y")
    if not np.allclose(nominal_stress, expected_stress, rtol=0.0, atol=2.0e-12):
        raise ValueError("FEM nominal_stress does not equal reaction_y/L_x")
    return {
        "coords": coords,
        "triangles": triangles,
        "response_load": response_load,
        "reaction": reaction,
        "nominal_strain": expected_strain,
        "nominal_stress": expected_stress,
        "snapshot_load": snapshot_load,
        "u": displacement,
        "d": damage,
        "Lx": lx,
        "Ly": ly,
        "gate_mode": gate_mode,
        "boundary_mode": boundary_mode,
        "plane_assumption": plane_assumption,
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def _triangles_match(first: np.ndarray, second: np.ndarray) -> bool:
    def canonical(values: np.ndarray) -> np.ndarray:
        result = np.sort(np.asarray(values, dtype=np.int64), axis=1)
        order = np.lexsort((result[:, 2], result[:, 1], result[:, 0]))
        return result[order]

    return first.shape == second.shape and np.array_equal(canonical(first), canonical(second))


def _plot_inputs(
    values: Mapping[str, np.ndarray],
    kind: str,
    target_load: float,
    engineering: Any,
    *,
    strategy: str = "legacy",
) -> dict[str, Any]:
    archive_schema = _scalar_string(values["archive_schema"], "archive_schema")
    if strategy == "stabilized":
        # The lean v5 archive stores no geometry; plotting triangles and
        # coordinates come from the hashed source cloud, never from a generic
        # path guess.
        from . import nonlocal_engineering_grip_stabilized as stabilized

        metadata = _metadata_from_archive(values)
        source = _stabilized_source_geometry(metadata, stabilized)
        coordinates = _require_float64(
            source["coordinates"], "plot coordinates"
        )
        triangles = _require_int64(source["triangles"], "plot triangles")
    else:
        coordinates = _require_float64(values["material_coordinates"], "plot coordinates")
        triangles = _require_int64(values["material_triangles"], "plot triangles")
    if kind == "completed":
        loads = _require_float64(values["state_u_bar"], "plot state loads")
        q = _require_float64(values["state_union_displacement"], "plot state q")
        damage = _require_float64(values["state_material_damage"], "plot state damage")
        reaction = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                _require_float64(values["accepted_top_reaction_after"], "plot reaction"),
            )
        )
        target_index = _exact_load_index(loads, target_load, "nonlocal state loads")
        if target_index != len(loads) - 1:
            raise ValueError(
                "completed workflow plotting requires target_load to equal the final state"
        )
        if archive_schema == engineering.ARCHIVE_SCHEMA:
            field_equations_validated = True
        else:
            field_equations_validated = False
    else:
        if strategy == "stabilized":
            loads = _require_float64(values["state_u_bar"], "plot state loads")
            q = _require_float64(
                values["state_union_displacement"], "plot state q"
            )
            damage = _require_float64(
                values["state_material_damage"], "plot state damage"
            )
            reaction = np.concatenate(
                (
                    np.asarray([0.0], dtype=np.float64),
                    _require_float64(
                        values["accepted_top_reaction_after"], "plot reaction"
                    ),
                )
            )
        else:
            loads = _require_float64(
                values["accepted_state_u_bar"], "plot accepted loads"
            )
            q = _require_float64(values["accepted_state_q"], "plot accepted q")
            damage = _require_float64(
                values["accepted_state_d"], "plot accepted damage"
            )
            reaction = _require_float64(values["accepted_top_reaction"], "plot reaction")
        target_index = _exact_load_index(loads, target_load, "accepted state loads")
        field_equations_validated = False
    material_dofs = 2 * len(coordinates)
    if (
        q.ndim != 2
        or q.shape[0] != len(loads)
        or q.shape[1] < material_dofs
        or q.shape[1] % 2 != 0
    ):
        raise ValueError(
            "plot displacement states must have shape "
            f"({len(loads)}, 2*n_union) with n_union >= {len(coordinates)}, "
            f"got {q.shape}"
        )
    # The archive stores the full union displacement, whereas field plots use
    # the source-rebuilt material coordinates/triangles only.  The material
    # prefix is canonical because material particles precede the collars.
    q = q[:, :material_dofs]
    if damage.shape != (len(loads), len(coordinates)):
        raise ValueError("plot damage states have incompatible shape")
    if reaction.shape != loads.shape:
        raise ValueError("plot reaction states have incompatible shape")
    return {
        "coordinates": coordinates,
        "triangles": triangles,
        "loads": loads,
        "q": q,
        "damage": damage,
        "reaction": reaction,
        "target_index": target_index,
        "field_equations_validated": field_equations_validated,
        "kind": kind,
        "n_material": len(coordinates),
    }


def _field_norm(first: np.ndarray, second: np.ndarray, field: str) -> Any:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    combined = np.concatenate((np.asarray(first).reshape(-1), np.asarray(second).reshape(-1)))
    if field == "ux":
        bound = float(np.max(np.abs(combined)))
        if bound == 0.0:
            bound = 1.0
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    lower = min(0.0, float(np.min(combined)))
    upper = float(np.max(combined))
    if upper <= lower:
        upper = lower + 1.0
    return Normalize(vmin=lower, vmax=upper)


def _write_figure(path: Path, figure: Any, *, overwrite: bool) -> None:
    def writer(handle: Any) -> None:
        figure.savefig(handle, format="png", dpi=220, bbox_inches="tight")

    _atomic_write(path, writer, overwrite=overwrite, suffix=".png.tmp")


def plot_archive(
    archive_path: Path,
    fem_archive_path: Path,
    *,
    target_load: float,
    fields_path: Path,
    curve_path: Path,
    curve_data_path: Path,
    report_path: Path,
    overwrite: bool,
    replay_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot accepted material fields and nominal response against explicit FEM."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    from . import nonlocal_engineering_grip as engineering

    for path in (fields_path, curve_path, curve_data_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite plotting artifact {path}; pass --overwrite"
            )
    values = _load_npz(archive_path, "engineering-grip archive")
    kind = _archive_kind(values, engineering)
    strategy = _archive_strategy(values, engineering)
    plot_data = _plot_inputs(
        values, kind, target_load, engineering, strategy=strategy
    )
    metadata = _metadata_from_archive(values)
    (
        acceptance_mode,
        _,
        warning_safety_policy,
        warning_safety_limit,
    ) = _archive_acceptance_details(values, metadata, kind, engineering)
    stabilized_binding: dict[str, Any] = {}
    stabilized_binding_valid = False
    stabilized_binding_error: str | None = None
    stabilized_semantics: dict[str, Any] = {}
    archive_validated = False
    if strategy == "stabilized":
        from . import nonlocal_engineering_grip_stabilized as stabilized

        if replay_result is None:
            replay_result = stabilized.replay_stabilized_engineering_grip_archive(
                archive_path
            )
        (
            stabilized_binding_valid,
            stabilized_binding,
            stabilized_binding_error,
        ) = _stabilized_replay_binding(
            archive_path,
            values,
            metadata,
            replay_result,
            stabilized,
        )
        stabilized_semantics = _stabilized_replay_semantics(replay_result)
        plot_data["field_equations_validated"] = stabilized_semantics[
            "field_equations_validated"
        ]
    else:
        archive_validated = bool(plot_data["field_equations_validated"])
    fem = _load_fem_archive(fem_archive_path)
    coordinates = plot_data["coordinates"]
    triangles = plot_data["triangles"]
    if coordinates.shape != fem["coords"].shape or not np.allclose(
        coordinates, fem["coords"], rtol=0.0, atol=1.0e-14
    ):
        raise ValueError("nonlocal and FEM material coordinates are not aligned")
    if not _triangles_match(triangles, fem["triangles"]):
        raise ValueError("nonlocal and FEM triangle sets are not aligned")
    fem_snapshot_index: int | None
    try:
        fem_snapshot_index = _exact_load_index(
            fem["snapshot_load"], target_load, "FEM snapshot loads"
        )
    except _MissingExactLoadError:
        fem_snapshot_index = None
    nl_q = plot_data["q"][plot_data["target_index"], : 2 * len(coordinates)].reshape(
        len(coordinates), 2
    )
    nl_d = plot_data["damage"][plot_data["target_index"]]
    fem_available = fem_snapshot_index is not None

    nl_tri = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
    figure, axes = plt.subplots(
        3,
        2 if fem_available else 1,
        figsize=(10.2, 12.4) if fem_available else (5.8, 12.4),
        squeeze=False,
        constrained_layout=True,
    )
    specifications = (
        ("d", r"Damage $d$", "inferno"),
        ("ux", r"Horizontal displacement $u_x$", "coolwarm"),
        ("uy", r"Vertical displacement $u_y$", "viridis"),
    )
    nl_fields = {"d": nl_d, "ux": nl_q[:, 0], "uy": nl_q[:, 1]}
    fem_fields: dict[str, np.ndarray] = {}
    fem_tri = None
    if fem_available:
        assert fem_snapshot_index is not None
        fem_q = fem["u"][fem_snapshot_index]
        fem_d = fem["d"][fem_snapshot_index]
        fem_fields = {"d": fem_d, "ux": fem_q[:, 0], "uy": fem_q[:, 1]}
        fem_tri = mtri.Triangulation(
            fem["coords"][:, 0], fem["coords"][:, 1], fem["triangles"]
        )
    for row, (field, label, cmap) in enumerate(specifications):
        norm = _field_norm(
            nl_fields[field],
            fem_fields[field] if fem_available else nl_fields[field],
            field,
        )
        images = []
        entries = [(nl_tri, nl_fields[field], "Nonlocal engineering grips")]
        if fem_available:
            assert fem_tri is not None
            entries.append(
                (
                    fem_tri,
                    fem_fields[field],
                    "Local FEM point anchor (warning-mode reference)",
                )
            )
        for column, (triangulation, field_values, title) in enumerate(entries):
            axis = axes[row, column]
            image = axis.tripcolor(
                triangulation,
                field_values,
                shading="gouraud",
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            images.append(image)
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_title(f"{title}\n{label}", fontsize=11)
        figure.colorbar(
            images[-1],
            ax=axes[row, : len(entries)],
            location="right",
            shrink=0.91,
            pad=0.02,
        ).set_label(label)
    figure.suptitle(
        f"Material fields at prescribed $\\bar{{u}}={target_load:.6g}$",
        fontsize=15,
    )
    _write_figure(fields_path, figure, overwrite=overwrite)
    plt.close(figure)

    loads = plot_data["loads"]
    lx = float(np.max(coordinates[:, 0]) - np.min(coordinates[:, 0]))
    ly = float(np.max(coordinates[:, 1]) - np.min(coordinates[:, 1]))
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("nonlocal coordinate extents must be positive")
    nl_strain = loads / ly
    nl_stress = plot_data["reaction"] / lx
    fem_load = fem["response_load"]
    fem_strain = fem["nominal_strain"]
    fem_stress = fem["nominal_stress"]
    (
        fem_stress_at_target,
        fem_response_index,
        fem_response_provenance,
    ) = _fem_target_response(fem_load, fem_stress, target_load)
    fem_curve_mask = fem_load <= target_load + (
        16.0 * np.finfo(np.float64).eps * max(1.0, abs(target_load))
    )
    if not np.any(fem_curve_mask):
        fem_curve_mask = np.zeros(len(fem_load), dtype=bool)
    figure, axis = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    if np.any(fem_curve_mask):
        axis.plot(
            fem_strain[fem_curve_mask],
            fem_stress[fem_curve_mask],
            color="#222222",
            linewidth=2.2,
            label="Local FEM point anchor (warning-mode reference)",
        )
    axis.plot(
        nl_strain,
        nl_stress,
        color="#d1495b",
        linewidth=2.3,
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label=(
                "Nonlocal engineering grips (strict)"
                if plot_data["field_equations_validated"] and strategy == "legacy"
                else (
                    "Nonlocal engineering grips (stabilized experimental)"
                    if strategy == "stabilized"
                    and acceptance_mode == "stabilized"
                    and kind == "completed"
                    else (
                        "Nonlocal engineering grips (stabilized warning accepted states)"
                        if strategy == "stabilized" and kind == "completed"
                        else (
                            "Nonlocal engineering grips (warning accepted states)"
                            if kind == "completed"
                            else "Nonlocal engineering grips (accepted states before failure)"
                        )
                    )
                )
            )
    )
    axis.scatter([nl_strain[-1]], [nl_stress[-1]], s=48, color="#d1495b", zorder=4)
    axis.set_xlim(0.0, max(float(target_load / ly) * 1.04, 1.0e-12))
    upper = float(np.max(nl_stress))
    if np.any(fem_curve_mask):
        upper = max(upper, float(np.max(fem_stress[fem_curve_mask])))
    axis.set_ylim(0.0, max(upper * 1.08, 1.0e-12))
    axis.set_xlabel(r"Nominal strain $\varepsilon_{nom}=\bar{u}/L_y$")
    axis.set_ylabel(r"Nominal stress $\sigma_{nom}=R_y/L_x$")
    axis.set_title("Nominal stress-strain response at the requested endpoint")
    axis.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.8)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.99,
        0.03,
        "$L_x=L_y=1$ normalization; interface opening is diagnostic",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    _write_figure(curve_path, figure, overwrite=overwrite)
    plt.close(figure)

    fem_stress_at_nl: np.ndarray
    comparison_available = bool(
        len(fem_load) > 0 and target_load <= float(fem_load[-1])
    )
    if comparison_available:
        fem_stress_at_nl = np.interp(nl_strain, fem_strain, fem_stress)
        stress_difference = nl_stress - fem_stress_at_nl
        curve_rmse = float(np.sqrt(np.mean(stress_difference**2)))
    else:
        fem_stress_at_nl = np.full_like(nl_strain, np.nan)
        stress_difference = np.full_like(nl_strain, np.nan)
        curve_rmse = None
    curve_arrays = {
        "nonlocal_nominal_strain": np.asarray(nl_strain, dtype=np.float64),
        "nonlocal_nominal_stress": np.asarray(nl_stress, dtype=np.float64),
        "fem_nominal_strain": np.asarray(fem_strain, dtype=np.float64),
        "fem_nominal_stress": np.asarray(fem_stress, dtype=np.float64),
        "fem_stress_at_nonlocal_strain": np.asarray(
            fem_stress_at_nl, dtype=np.float64
        ),
        "nonlocal_minus_fem_stress": np.asarray(stress_difference, dtype=np.float64),
    }
    _atomic_npz(
        curve_data_path,
        curve_arrays,
        overwrite=overwrite,
        context="engineering-grip curve data",
    )
    report = {
        "schema": PLOT_SCHEMA,
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "fem_archive": {
            "path": fem["path"],
            "sha256": fem["sha256"],
            "gate_mode": fem["gate_mode"],
            "boundary_mode": fem["boundary_mode"],
            "plane_assumption": fem["plane_assumption"],
        },
        "archive_kind": kind,
        "continuation_strategy": strategy,
        "experimental": strategy == "stabilized",
        "acceptance": _acceptance_payload(
            acceptance_mode, warning_safety_limit, warning_safety_policy
        ),
        "target_load": float(target_load),
        "nonlocal_state_index": int(plot_data["target_index"]),
        "fem_snapshot_index": fem_snapshot_index,
        "fem_response_index": fem_response_index,
        "fem_response_provenance": fem_response_provenance,
        "fem_snapshot_available": fem_available,
        "fem_curve_comparison_available": comparison_available,
        "nonlocal_final_nominal_stress": float(nl_stress[-1]),
        "fem_nominal_stress_at_target": (
            float(fem_stress_at_target)
            if fem_stress_at_target is not None
            else None
        ),
        "final_stress_relative_difference": (
            float(
                abs(nl_stress[-1] - fem_stress_at_target)
                / max(abs(fem_stress_at_target), np.finfo(float).tiny)
            )
            if fem_stress_at_target is not None
            else None
        ),
        "curve_rmse_through_target": curve_rmse,
        "normalization": {
            "nonlocal": "epsilon_nom=u_bar/L_y; sigma_nom=top_reaction/L_x",
            "fem": "epsilon_nom=u_bar/L_y; sigma_nom=reaction_y/L_x",
            "nonlocal_Lx": lx,
            "nonlocal_Ly": ly,
            "fem_Lx": fem["Lx"],
            "fem_Ly": fem["Ly"],
            "interface_opening_used_for_curve": False,
        },
        "fields_figure": str(fields_path),
        "curve_figure": str(curve_path),
        "curve_data": str(curve_data_path),
        "independent_replay": (
            dict(replay_result)
            if strategy == "stabilized" and isinstance(replay_result, Mapping)
            else {}
            if strategy == "stabilized"
            else None
        ),
        "field_ranges": {
            "nonlocal": {
                "d": [float(np.min(nl_d)), float(np.max(nl_d))],
                "ux": [float(np.min(nl_q[:, 0])), float(np.max(nl_q[:, 0]))],
                "uy": [float(np.min(nl_q[:, 1])), float(np.max(nl_q[:, 1]))],
            },
            "fem": (
                {
                    "d": [
                        float(np.min(fem_fields["d"])),
                        float(np.max(fem_fields["d"])),
                    ],
                    "ux": [
                        float(np.min(fem_fields["ux"])),
                        float(np.max(fem_fields["ux"])),
                    ],
                    "uy": [
                        float(np.min(fem_fields["uy"])),
                        float(np.max(fem_fields["uy"])),
                    ],
                }
                if fem_available
                else None
            ),
        },
    }
    archive_storage = _archive_storage_report(metadata)
    if archive_storage is not None:
        report["archive_storage_preflight"] = archive_storage
    if strategy == "stabilized":
        report.update(stabilized_semantics)
        report["replay_binding"] = dict(stabilized_binding)
        report["replay_binding_valid"] = stabilized_binding_valid
        if stabilized_binding_error is not None:
            report["replay_binding_error"] = stabilized_binding_error
    else:
        report["validated"] = archive_validated
    _atomic_json(report_path, report, overwrite=overwrite)
    return report


def replay_archive(
    archive_path: Path,
    report_path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Independently replay one supported engineering-grip archive."""

    from . import nonlocal_engineering_grip as engineering

    values = _load_npz(archive_path, "engineering-grip archive")
    kind = _archive_kind(values, engineering)
    strategy = _archive_strategy(values, engineering)
    if kind != "completed" and strategy != "stabilized":
        raise ValueError("independent replay is defined for completed legacy archives")
    metadata = _metadata_from_archive(values)
    (
        acceptance_mode,
        warning,
        warning_safety_policy,
        warning_safety_limit,
    ) = _archive_acceptance_details(
        values, metadata, kind, engineering
    )
    archive_storage = _archive_storage_report(metadata)
    started = time.perf_counter()
    if strategy == "stabilized":
        from . import nonlocal_engineering_grip_stabilized as stabilized

        report_schema = STABILIZED_REPLAY_REPORT_SCHEMA
        binding_valid, binding, binding_error = _stabilized_replay_binding(
            archive_path, values, metadata, None, stabilized
        )

        def binding_fields() -> dict[str, Any]:
            return {
                "archive_schema": binding.get("archive_schema"),
                "archive_layout": binding.get("archive_layout"),
                "replay_contract": binding.get("replay_contract"),
                "completion_status": binding.get("completion_status"),
                "source_bundle_hash": binding.get("source_bundle_hash"),
                "source_path": binding.get("source_path"),
                "source_sha256": binding.get("source_sha256"),
                "accepted_steps": binding.get("accepted_steps"),
                "states": binding.get("states"),
                "solver_summary_rows": binding.get("solver_summary_rows"),
            }

        try:
            if acceptance_mode == stabilized.STABILIZED_WARNING_ACCEPTANCE_MODE:
                # A warning row may commit before the pre/post candidate pair
                # converges. Its force/moment scales combine both mechanics
                # states, but warning-v1 intentionally stores only the accepted
                # physical state. Recompute every accepted-state equation and
                # conservation gate while leaving those two transition-only
                # summary scales outside the replay claim.
                compare_scalar = stabilized._v5_compare_diagnostic_scalar

                def compare_warning_scalar(
                    payload: Mapping[str, Any], name: str, expected: Any
                ) -> None:
                    if name in {"force_scale", "moment_scale"}:
                        return
                    compare_scalar(payload, name, expected)

                stabilized._v5_compare_diagnostic_scalar = compare_warning_scalar
                try:
                    replay = stabilized.replay_stabilized_engineering_grip_archive(
                        archive_path
                    )
                finally:
                    stabilized._v5_compare_diagnostic_scalar = compare_scalar
                replay["non_replayed_summary_scalars"] = [
                    "force_scale",
                    "moment_scale",
                ]
            else:
                replay = stabilized.replay_stabilized_engineering_grip_archive(
                    archive_path
                )
        except Exception as error:
            payload = {
                "schema": report_schema,
                "experimental": True,
                "field_equations_validated": False,
                "path_replayed": False,
                "rejected_trial_path_available": False,
                "continuation_strategy": strategy,
                "archive_path": str(archive_path),
                "archive_sha256": _sha256_file(archive_path),
                "elapsed_seconds": time.perf_counter() - started,
                **binding_fields(),
                "replay_binding": dict(binding),
                "replay_binding_valid": False,
                "replay_binding_error": binding_error,
                "acceptance": _acceptance_payload(
                    acceptance_mode, warning_safety_limit, warning_safety_policy
                ),
                "error": str(error),
            }
            if archive_storage is not None:
                payload["archive_storage_preflight"] = archive_storage
            _atomic_json(report_path, payload, overwrite=overwrite)
            raise
        stabilized_semantics = _stabilized_replay_semantics(replay)
        binding_valid, binding, binding_error = _stabilized_replay_binding(
            archive_path, values, metadata, replay, stabilized
        )
        payload = {
            "schema": report_schema,
            "experimental": True,
            **stabilized_semantics,
            "continuation_strategy": strategy,
            "archive_path": str(archive_path),
            "archive_sha256": _sha256_file(archive_path),
            "elapsed_seconds": time.perf_counter() - started,
            **binding_fields(),
            "replay_binding": dict(binding),
            "replay_binding_valid": binding_valid,
            "acceptance": _acceptance_payload(
                acceptance_mode, warning_safety_limit, warning_safety_policy
            ),
            "replay": replay,
        }
        if archive_storage is not None:
            payload["archive_storage_preflight"] = archive_storage
        if binding_error is not None:
            payload["replay_binding_error"] = binding_error
        _atomic_json(report_path, payload, overwrite=overwrite)
        return payload

    try:
        replay = engineering.replay_engineering_grip_archive(archive_path)
    except Exception as error:
        payload = {
            "schema": "fracture2d-engineering-grip-replay-report-v1",
            "validated": False,
            "archive_path": str(archive_path),
            "archive_sha256": _sha256_file(archive_path),
            "elapsed_seconds": time.perf_counter() - started,
            "acceptance": _acceptance_payload(
                acceptance_mode, warning_safety_limit, warning_safety_policy
            ),
            "error": str(error),
        }
        _atomic_json(report_path, payload, overwrite=overwrite)
        raise
    payload = {
        "schema": "fracture2d-engineering-grip-replay-report-v1",
        "validated": bool(replay.get("replayed", False) and not warning),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "elapsed_seconds": time.perf_counter() - started,
        "acceptance": _acceptance_payload(
            acceptance_mode, warning_safety_limit, warning_safety_policy
        ),
        "replay": replay,
    }
    _atomic_json(report_path, payload, overwrite=overwrite)
    return payload


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _fraction(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise argparse.ArgumentTypeError("value must be strictly between zero and one")
    return result


def _poisson_ratio(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Poisson ratio must be a number") from error
    if not np.isfinite(result) or not -1.0 < result < 0.5:
        raise argparse.ArgumentTypeError(
            "Poisson ratio must be finite and lie strictly between -1 and 0.5"
        )
    return result


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-cloud", type=Path, default=DEFAULT_SOURCE_CLOUD)
    parser.add_argument("--fem-archive", type=Path, default=DEFAULT_FEM_ARCHIVE)
    parser.add_argument(
        "--continuation-strategy",
        choices=CONTINUATION_STRATEGIES,
        default="legacy",
        help=(
            "legacy strict/warning controller, or the experimental stabilized "
            "under-relaxed controller"
        ),
    )
    parser.add_argument(
        "--archive-budget-bytes",
        type=_positive_int,
        default=None,
        help=(
            "positive uncompressed stabilized-archive byte budget; required "
            "for stabilized and forbidden for legacy"
        ),
    )
    parser.add_argument(
        "--progress-every-sweeps",
        type=_positive_int,
        default=None,
        help=(
            "positive stabilized progress cadence in outer sweeps; required "
            "for stabilized and forbidden for legacy"
        ),
    )
    parser.add_argument(
        "--relaxation-policy",
        choices=RELAXATION_POLICIES,
        default=DEFAULT_RELAXATION_POLICY,
        help=(
            "stabilized relaxation policy: sufficient_decrease preserves merit "
            "backtracking; fixed evaluates exactly one fixed-omega candidate"
        ),
    )
    parser.add_argument(
        "--acceptance-mode",
        choices=("strict", "warning"),
        default="strict",
        help="strict validated endpoint or separately archived warning evidence",
    )
    parser.add_argument(
        "--warning-safety-limit",
        type=_positive_float,
        default=None,
        help="required raw damage range/healing bound in warning mode",
    )
    parser.add_argument(
        "--warning-safety-policy",
        choices=("bounded", "unbounded"),
        default=None,
        help=(
            "warning evidence policy: bounded retains the explicit limit; "
            "unbounded records no damage range/healing limit"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"durable output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--run-id",
        default="engineering_grip",
        help="simple output stem; existing artifacts require --overwrite",
    )
    parser.add_argument("--load-final", type=_positive_float, default=DEFAULT_LOAD_FINAL)
    parser.add_argument(
        "--max-stagger-iterations",
        type=_positive_int,
        default=DEFAULT_MAX_STAGGER_ITERATIONS,
    )
    parser.add_argument(
        "--gpu-device-index", type=_nonnegative_int, default=DEFAULT_GPU_DEVICE_INDEX
    )
    parser.add_argument("--delta", type=_positive_float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--h-g", dest="h_g", type=_positive_float, default=DEFAULT_GRIP_SPACING
    )
    parser.add_argument("--du-initial", type=_positive_float, default=DEFAULT_DU_INITIAL)
    parser.add_argument("--du-min", type=_positive_float, default=DEFAULT_DU_MIN)
    parser.add_argument(
        "--visibility-tolerance",
        type=_nonnegative_float,
        default=DEFAULT_VISIBILITY_TOLERANCE,
    )
    parser.add_argument(
        "--youngs-modulus", type=_positive_float, default=DEFAULT_YOUNGS_MODULUS
    )
    parser.add_argument(
        "--poisson-ratio", type=_poisson_ratio, default=DEFAULT_POISSON_RATIO
    )
    parser.add_argument("--k-res", type=_positive_float, default=DEFAULT_K_RES)
    parser.add_argument("--rank-rtol", type=_positive_float, default=DEFAULT_RANK_RTOL)
    parser.add_argument("--Gc", type=_positive_float, default=DEFAULT_GC)
    parser.add_argument("--l0", type=_positive_float, default=DEFAULT_L0)
    parser.add_argument(
        "--cutback-factor",
        type=_fraction,
        default=DEFAULT_CUTBACK_FACTOR,
        help=f"strict cutback factor (default: {DEFAULT_CUTBACK_FACTOR})",
    )
    parser.add_argument(
        "--max-cutbacks",
        type=_positive_int,
        default=DEFAULT_MAX_CUTBACKS,
        help=f"strict cutback budget (default: {DEFAULT_MAX_CUTBACKS})",
    )
    parser.add_argument("--cg-rtol", type=_nonnegative_float, default=DEFAULT_CG_RTOL)
    parser.add_argument("--cg-atol", type=_nonnegative_float, default=DEFAULT_CG_ATOL)
    parser.add_argument("--cg-maxiter", type=_positive_int, default=DEFAULT_CG_MAXITER)
    parser.add_argument("--gate-atol", type=_nonnegative_float, default=DEFAULT_GATE_ATOL)
    parser.add_argument("--gate-rtol", type=_nonnegative_float, default=DEFAULT_GATE_RTOL)
    parser.add_argument(
        "--gate-scale-floor",
        type=_nonnegative_float,
        default=DEFAULT_GATE_SCALE_FLOOR,
    )
    parser.add_argument(
        "--damage-residual-atol",
        type=_nonnegative_float,
        default=DEFAULT_DAMAGE_RESIDUAL_ATOL,
    )
    parser.add_argument(
        "--damage-residual-rtol",
        type=_nonnegative_float,
        default=DEFAULT_DAMAGE_RESIDUAL_RTOL,
    )
    parser.add_argument(
        "--damage-residual-scale-floor",
        type=_nonnegative_float,
        default=DEFAULT_DAMAGE_RESIDUAL_SCALE_FLOOR,
    )
    parser.add_argument(
        "--healing-atol", type=_nonnegative_float, default=DEFAULT_HEALING_ATOL
    )
    parser.add_argument(
        "--damage-range-atol",
        type=_nonnegative_float,
        default=DEFAULT_DAMAGE_RANGE_ATOL,
    )
    parser.add_argument(
        "--work-check-load",
        type=_positive_float,
        default=DEFAULT_WORK_CHECK_LOAD,
    )
    parser.add_argument(
        "--work-epsilon",
        type=_positive_float,
        default=DEFAULT_WORK_EPSILON,
    )
    parser.add_argument(
        "--work-relative-limit",
        type=_nonnegative_float,
        default=DEFAULT_WORK_RELATIVE_LIMIT,
    )
    parser.add_argument(
        "--omega-initial",
        type=_fraction,
        default=DEFAULT_OMEGA,
        help=f"stabilized initial damage relaxation (default: {DEFAULT_OMEGA})",
    )
    parser.add_argument(
        "--omega-min",
        type=_fraction,
        default=DEFAULT_OMEGA,
        help=f"stabilized minimum damage relaxation (default: {DEFAULT_OMEGA})",
    )
    parser.add_argument(
        "--omega-reduction",
        type=_fraction,
        default=None,
        help="stabilized relaxation reduction factor (required for stabilized)",
    )
    parser.add_argument(
        "--sufficient-decrease",
        type=_fraction,
        default=None,
        help="stabilized merit sufficient-decrease coefficient (required for stabilized)",
    )
    parser.add_argument(
        "--max-backtracks",
        type=_nonnegative_int,
        default=DEFAULT_MAX_BACKTRACKS,
        help=(
            "stabilized relaxation backtracking budget "
            f"(default: {DEFAULT_MAX_BACKTRACKS})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing all named output artifacts",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Durable fracture2d engineering-grip workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="fresh-zero strict GPU run, replay, analysis, and plots",
    )
    _add_run_arguments(run)

    replay = subparsers.add_parser(
        "replay", help="independently replay one supported numerical archive"
    )
    replay.add_argument("--archive", type=Path, required=True)
    replay.add_argument("--report", type=Path, required=True)
    replay.add_argument("--overwrite", action="store_true")

    analyze = subparsers.add_parser(
        "analyze", help="independently analyze a completed or failure archive"
    )
    analyze.add_argument("--archive", type=Path, required=True)
    analyze.add_argument("--report", type=Path, required=True)
    analyze.add_argument("--overwrite", action="store_true")

    plot = subparsers.add_parser(
        "plot", help="plot accepted fields and nominal response against FEM"
    )
    plot.add_argument("--archive", type=Path, required=True)
    plot.add_argument("--fem-archive", type=Path, required=True)
    plot.add_argument("--target-load", type=_nonnegative_float, required=True)
    plot.add_argument("--fields", type=Path, required=True)
    plot.add_argument("--curve", type=Path, required=True)
    plot.add_argument("--curve-data", type=Path, required=True)
    plot.add_argument("--report", type=Path, required=True)
    plot.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        try:
            return run_workflow(arguments)
        except Exception as error:
            # Typed continuation failures have already written their audit and
            # manifest.  Preserve the required nonzero status without turning
            # an unknown exception into a numerical fallback.
            from . import nonlocal_engineering_grip as engineering
            from . import nonlocal_engineering_grip_stabilized as stabilized

            if isinstance(
                error,
                (
                    engineering.EngineeringGripContinuationFailure,
                    stabilized.StabilizedEngineeringGripContinuationFailure,
                ),
            ):
                print(str(error), file=sys.stderr)
                return 2
            raise
    if arguments.command == "replay":
        payload = replay_archive(
            arguments.archive,
            arguments.report,
            overwrite=arguments.overwrite,
        )
        print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
        return 0
    if arguments.command == "analyze":
        payload = analyze_archive(
            arguments.archive,
            arguments.report,
            overwrite=arguments.overwrite,
            replay_result=None,
        )
        print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
        return 0
    if arguments.command == "plot":
        payload = plot_archive(
            arguments.archive,
            arguments.fem_archive,
            target_load=arguments.target_load,
            fields_path=arguments.fields,
            curve_path=arguments.curve,
            curve_data_path=arguments.curve_data,
            report_path=arguments.report,
            overwrite=arguments.overwrite,
        )
        print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"unsupported workflow command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
