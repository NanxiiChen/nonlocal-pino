"""Stage 4 fixed subproblems and Stage 5 controller for fracture2d.

The fixed-history, fixed-damage, and Amor-history functions remain pure direct
subproblem APIs.  The Stage 5 continuation controller wraps only those
approved operations; it does not change the equations or select a production
horizon. Mechanics defaults to the NumPy/SciPy bond-associated path in
:mod:`nonlocal_mechanics`; an explicit accelerator adapter may return a fresh
CPU replay result. Damage uses the explicit-volume operator exported by
:mod:`pinn.operators`.

The controller has two explicit acceptance modes.  ``strict`` is unchanged:
every post-sweep acceptance gate must pass.  ``warning`` is the separately
archived all-gate diagnostic mode: after the complete stagger budget, any
failed finite bypassable gate may be bypassed, the unmodified candidate is
committed with ``passed=False`` diagnostics, and the exact canonical bypassed
gate list is retained in the step, audit, and archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from pinn.operators import ExplicitVolumeSparseD

from .nonlocal_mechanics import (
    BondAssociatedMechanics,
    FixedDamageBondAssociatedMechanics,
    HardBoundaryConstraints,
    MechanicsEvaluation,
    MechanicsSolveResult,
    _mechanics_backend_token,
)


__all__ = [
    "ACCEPTANCE_MODES",
    "ACCEPTANCE_STATUSES",
    "ARCHIVE_SCHEMA_VERSION",
    "WARNING_GATE_NAMES",
    "AmorHistoryResult",
    "AcceptedStep",
    "AttemptRecord",
    "CommittedState",
    "ContinuationFailure",
    "ContinuationMetadata",
    "ContinuationResult",
    "FixedHistoryDamageResult",
    "GateTolerance",
    "ArchiveMetadata",
    "SupplementalArchiveMetadata",
    "StaggerDiagnostics",
    "StrictContinuationConfig",
    "amor_positive_energy_from_strain",
    "compute_amor_history",
    "compute_amor_history_from_edge_strain",
    "run_strict_continuation",
    "run_warning_continuation",
    "solve_fixed_damage_mechanics",
    "solve_fixed_history_damage",
    "update_history_max",
    "write_continuation_archive",
]


_FLOAT64 = np.dtype(np.float64)
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_OPERATOR_RTOL = 5.0e-13
_ETA_SUM_ATOL = 5.0e-13

# Acceptance-mode contract shared by the strict and warning controllers.
# ``WARNING_GATE_NAMES`` is the canonical fixed ordering of every gate that a
# warning-mode commit may bypass at the stagger budget; the ``mechanics`` gate
# is deliberately absent because a failed mechanics/CG solve never returns a
# valid displacement candidate and stays on the audited cutback path.
ARCHIVE_SCHEMA_VERSION = 2
ACCEPTANCE_MODES = ("strict", "warning")
ACCEPTANCE_STATUSES = ("strict_passed", "warning_accepted")
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
_WARNING_GATE_INDEX = {
    name: index for index, name in enumerate(WARNING_GATE_NAMES)
}


def _bytes_backed_copy(
    values: Any, dtype: np.dtype[Any] | None = None
) -> np.ndarray:
    """Return a read-only array whose ultimate storage object is ``bytes``."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result


def _readonly_copy(values: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """Copy fixed numerical data into storage that cannot be made writable."""

    return _bytes_backed_copy(values, dtype)


def _require_float64_array(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    try:
        result = float(array)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_material_parameter(value: Any, name: str) -> float:
    result = _scalar_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result


def _validate_operator(
    operator: ExplicitVolumeSparseD,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(operator, ExplicitVolumeSparseD):
        raise TypeError("operator must be an ExplicitVolumeSparseD instance")

    coordinates = _require_float64_array(
        np.asarray(operator.coordinates), "operator coordinates"
    )
    if coordinates.shape != (operator.shape[0], 2):
        raise ValueError(
            "fracture2d operator coordinates must have exact float64 shape "
            f"({operator.shape[0]}, 2), got {coordinates.shape}"
        )
    if operator.kernel != "cubic":
        raise ValueError(
            "fracture2d fixed-history damage requires the frozen cubic kernel, "
            f"got {operator.kernel!r}"
        )
    volumes = _require_float64_array(
        np.asarray(operator.volumes), "operator volumes"
    )
    if volumes.ndim != 1 or volumes.shape != (operator.shape[0],):
        raise ValueError(
            "operator volumes must have exact float64 shape "
            f"({operator.shape[0]},), got {volumes.shape}"
        )
    if np.any(volumes <= 0.0):
        raise ValueError("operator volumes must be strictly positive")

    edges = np.asarray(operator.directed_edges)
    if edges.dtype != np.dtype(np.int32) and edges.dtype != np.dtype(np.int64):
        raise ValueError(
            "operator directed_edges must contain integer indices, "
            f"got {edges.dtype}"
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("operator directed_edges must have shape (E, 2)")
    edges = np.asarray(edges, dtype=np.int64)
    if np.any(edges < 0) or np.any(edges >= operator.shape[0]):
        raise ValueError("operator directed_edges contain an out-of-range index")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("operator directed_edges must not contain self edges")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("operator directed_edges contain duplicate edges")
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("operator directed_edges must be reciprocal")

    scipy_operator = operator.to_scipy().tocsr(copy=True)
    scipy_operator.sort_indices()
    if scipy_operator.shape != operator.shape:
        raise ValueError(
            "operator.to_scipy() shape does not match the operator shape: "
            f"{scipy_operator.shape} versus {operator.shape}"
        )
    if scipy_operator.dtype != _FLOAT64:
        raise ValueError(
            "operator.to_scipy() must preserve exact float64 coefficients, "
            f"got {scipy_operator.dtype}"
        )
    if scipy_operator.nnz and not np.all(np.isfinite(scipy_operator.data)):
        raise ValueError("operator.to_scipy() must contain only finite values")

    row = edges[:, 0]
    col = edges[:, 1]
    delta = _validate_material_parameter(operator.delta, "operator delta")
    distances = np.linalg.norm(coordinates[col] - coordinates[row], axis=1)
    if np.any(distances <= 0.0) or np.any(distances >= delta):
        raise ValueError(
            "operator directed_edges must satisfy the strict "
            "0 < distance < delta contract"
        )
    kernel = 120.0 / (np.pi * delta**4) * (1.0 - distances / delta) ** 2
    positive_weights = kernel * volumes[col]
    diagonal = np.bincount(
        row, weights=positive_weights, minlength=operator.shape[0]
    )
    expected = (
        coo_matrix(
            (-positive_weights, (row, col)),
            shape=operator.shape,
            dtype=np.float64,
        ).tocsr()
        + diags(diagonal, format="csr")
    )
    expected.sort_indices()
    if not np.array_equal(
        scipy_operator.indptr, expected.indptr
    ) or not np.array_equal(scipy_operator.indices, expected.indices):
        raise ValueError(
            "operator.to_scipy() sparsity does not match the frozen directed graph"
        )
    coefficient_scale = max(
        float(np.max(np.abs(expected.data))), np.finfo(float).tiny
    )
    coefficient_defect = float(
        np.max(np.abs(scipy_operator.data - expected.data))
    )
    if coefficient_defect > _OPERATOR_RTOL * coefficient_scale:
        raise ValueError(
            "operator.to_scipy() coefficients do not match the frozen 2D cubic "
            f"operator: defect={coefficient_defect}, scale={coefficient_scale}"
        )

    weighted = (diags(volumes, format="csr") @ scipy_operator).tocsr()
    weighted_skew = weighted - weighted.T
    weighted_scale = max(
        float(np.max(np.abs(weighted.data))) if weighted.nnz else 0.0,
        np.finfo(float).tiny,
    )
    weighted_defect = (
        float(np.max(np.abs(weighted_skew.data))) if weighted_skew.nnz else 0.0
    )
    if weighted_defect > _OPERATOR_RTOL * weighted_scale:
        raise ValueError(
            "operator.to_scipy() violates volume self-adjointness: "
            f"defect={weighted_defect}, scale={weighted_scale}"
        )
    constant_defect = np.asarray(
        scipy_operator @ np.ones(operator.shape[0], dtype=np.float64)
    )
    row_scale = max(
        float(np.max(np.asarray(np.abs(scipy_operator).sum(axis=1)))),
        np.finfo(float).tiny,
    )
    if float(np.max(np.abs(constant_defect))) > _OPERATOR_RTOL * row_scale:
        raise ValueError("operator.to_scipy() does not annihilate constants")
    return scipy_operator, volumes.copy(), edges, coordinates.copy()


def _validate_history(history: Any, n_particles: int) -> np.ndarray:
    values = _require_float64_array(
        history, "history", shape=(n_particles,)
    )
    if np.any(values < 0.0):
        raise ValueError("history must be non-negative")
    return values


@dataclass(frozen=True)
class FixedHistoryDamageResult:
    """Immutable direct solution and diagnostics for the fixed-history AT2 solve."""

    damage: np.ndarray
    history: np.ndarray
    crack_local_energy: float
    history_energy: float
    graph_energy: float
    total_energy: float
    strong_residual: np.ndarray
    euclidean_residual: np.ndarray
    strong_residual_norm: float
    strong_residual_inf: float
    euclidean_residual_norm: float
    euclidean_residual_inf: float
    damage_min: float
    damage_max: float
    below_zero_count: int
    above_one_count: int
    symmetry_defect: float
    symmetry_relative_defect: float

    def __post_init__(self) -> None:
        damage = np.asarray(self.damage)
        history = np.asarray(self.history)
        if damage.dtype != _FLOAT64 or damage.ndim != 1:
            raise ValueError("FixedHistoryDamageResult.damage must be float64 (N,)")
        if history.dtype != _FLOAT64 or history.shape != damage.shape:
            raise ValueError(
                "FixedHistoryDamageResult.history must be float64 with damage shape"
            )
        residual_specs = (
            ("strong_residual", self.strong_residual),
            ("euclidean_residual", self.euclidean_residual),
        )
        for name, values in residual_specs:
            array = np.asarray(values)
            if array.dtype != _FLOAT64 or array.shape != damage.shape:
                raise ValueError(
                    f"FixedHistoryDamageResult.{name} must be float64 with damage shape"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be finite")
            object.__setattr__(self, name, _readonly_copy(array))
        for name, values in (("damage", damage), ("history", history)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be finite")
            object.__setattr__(self, name, _readonly_copy(values))

        scalar_names = (
            "crack_local_energy",
            "history_energy",
            "graph_energy",
            "total_energy",
            "strong_residual_norm",
            "strong_residual_inf",
            "euclidean_residual_norm",
            "euclidean_residual_inf",
            "damage_min",
            "damage_max",
            "symmetry_defect",
            "symmetry_relative_defect",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("below_zero_count", "above_one_count"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"FixedHistoryDamageResult.{name} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(
                    f"FixedHistoryDamageResult.{name} must be non-negative"
                )
            object.__setattr__(self, name, value)

    @property
    def crack_energy(self) -> float:
        """Alias for the local crack-density energy component."""

        return self.crack_local_energy

    @property
    def local_energy(self) -> float:
        """Alias for the local crack-density energy component."""

        return self.crack_local_energy

    @property
    def crack_local(self) -> float:
        return self.crack_local_energy

    @property
    def energy(self) -> float:
        """Alias for the total fixed-history damage energy."""

        return self.total_energy

    @property
    def min_damage(self) -> float:
        return self.damage_min

    @property
    def max_damage(self) -> float:
        return self.damage_max

    @property
    def symmetry_error(self) -> float:
        return self.symmetry_defect

    @property
    def residual(self) -> np.ndarray:
        """Alias for the strong-form residual."""

        return self.strong_residual

    @property
    def system_residual(self) -> np.ndarray:
        """Alias for the Euclidean volume-multiplied residual."""

        return self.euclidean_residual


def solve_fixed_history_damage(
    operator: ExplicitVolumeSparseD,
    Gc: Any,
    l0: Any,
    history: Any,
) -> FixedHistoryDamageResult:
    """Solve the dimensional fixed-history nonlocal AT2 system directly.

    The returned system is the Euclidean volume-multiplied system

    ``S = M @ (diag(Gc/l0 + 2*H) + Gc*l0*D)``, ``b = 2*M@H``.

    The directed graph energy is evaluated as
    ``Gc*l0/4 * sum_(i,j) Vi*Vj*Jij*(dj-di)**2``; no damage clipping or hidden
    bound projection is performed.
    """

    gc_value = _validate_material_parameter(Gc, "Gc")
    l0_value = _validate_material_parameter(l0, "l0")
    D, volumes, edges, _ = _validate_operator(operator)
    n_particles = int(operator.shape[0])
    history_values = _validate_history(history, n_particles)

    mass = diags(volumes, format="csr")
    strong_matrix = diags(
        gc_value / l0_value + 2.0 * history_values,
        format="csr",
    ) + (gc_value * l0_value) * D
    system = (mass @ strong_matrix).tocsr()
    rhs = np.asarray(2.0 * volumes * history_values, dtype=np.float64)
    if system.shape != (n_particles, n_particles) or rhs.shape != (n_particles,):
        raise ValueError("fixed-history system has an incompatible shape")
    if system.nnz and not np.all(np.isfinite(system.data)):
        raise FloatingPointError("fixed-history system contains non-finite values")
    if not np.all(np.isfinite(rhs)):
        raise FloatingPointError("fixed-history right-hand side is non-finite")

    skew = system - system.T
    symmetry_defect = float(np.linalg.norm(skew.data, ord=np.inf)) if skew.nnz else 0.0
    system_norm = float(np.linalg.norm(system.data, ord=np.inf)) if system.nnz else 0.0
    symmetry_relative_defect = symmetry_defect / max(system_norm, 1.0e-300)
    if not np.isfinite(symmetry_defect) or not np.isfinite(symmetry_relative_defect):
        raise FloatingPointError("fixed-history system symmetry diagnostic is non-finite")
    if symmetry_relative_defect > _OPERATOR_RTOL:
        raise ValueError(
            "fixed-history Euclidean system is not symmetric within the frozen "
            f"float64 contract: relative defect={symmetry_relative_defect}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            damage = np.asarray(spsolve(system, rhs), dtype=np.float64)
        except (MatrixRankWarning, RuntimeError, np.linalg.LinAlgError) as error:
            raise RuntimeError("fixed-history sparse direct solve failed or is singular") from error
    if damage.shape != (n_particles,) or not np.all(np.isfinite(damage)):
        raise FloatingPointError("fixed-history sparse direct solve returned non-finite damage")

    # Reconstruct both residual representations from the same signed
    # constituents used by the strong-form audit.  ``system @ damage - rhs``
    # is mathematically identical, but its separate sparse mass/strong-matrix
    # matvec can introduce an avoidable cancellation defect when the solved
    # residual is near machine precision.  Keeping the operation order here
    # makes the returned strong and volume residuals satisfy the declared
    # constituent-scale contract without changing the fixed-history equation.
    graph_damage = np.asarray(D @ damage, dtype=np.float64)
    strong_residual = np.asarray(
        gc_value / l0_value * damage
        + gc_value * l0_value * graph_damage
        + 2.0 * history_values * damage
        - 2.0 * history_values,
        dtype=np.float64,
    )
    euclidean_residual = np.asarray(volumes * strong_residual, dtype=np.float64)
    if not np.all(np.isfinite(strong_residual)) or not np.all(
        np.isfinite(euclidean_residual)
    ):
        raise FloatingPointError("fixed-history solve residual is non-finite")

    crack_local_energy = float(
        np.sum(volumes * (gc_value / (2.0 * l0_value)) * damage**2)
    )
    history_energy = float(np.sum(volumes * history_values * (1.0 - damage) ** 2))
    row = edges[:, 0]
    col = edges[:, 1]
    directed_coefficient = np.asarray(D[row, col]).reshape(-1)
    directed_weight = -directed_coefficient
    if not np.all(np.isfinite(directed_weight)) or np.any(directed_weight <= 0.0):
        raise ValueError("fixed-history graph coefficients must be finite and positive")
    graph_energy = float(
        gc_value
        * l0_value
        / 4.0
        * np.sum(
            volumes[row]
            * directed_weight
            * (damage[col] - damage[row]) ** 2
        )
    )
    total_energy = crack_local_energy + history_energy + graph_energy
    energy_values = (
        crack_local_energy,
        history_energy,
        graph_energy,
        total_energy,
    )
    if not all(np.isfinite(value) for value in energy_values):
        raise FloatingPointError("fixed-history damage energy is non-finite")

    return FixedHistoryDamageResult(
        damage=damage,
        history=history_values,
        crack_local_energy=crack_local_energy,
        history_energy=history_energy,
        graph_energy=graph_energy,
        total_energy=total_energy,
        strong_residual=strong_residual,
        euclidean_residual=euclidean_residual,
        strong_residual_norm=float(np.linalg.norm(strong_residual)),
        strong_residual_inf=float(np.max(np.abs(strong_residual))),
        euclidean_residual_norm=float(np.linalg.norm(euclidean_residual)),
        euclidean_residual_inf=float(np.max(np.abs(euclidean_residual))),
        damage_min=float(np.min(damage)),
        damage_max=float(np.max(damage)),
        below_zero_count=int(np.count_nonzero(damage < 0.0)),
        above_one_count=int(np.count_nonzero(damage > 1.0)),
        symmetry_defect=symmetry_defect,
        symmetry_relative_defect=symmetry_relative_defect,
    )


def update_history_max(previous_history: Any, particle_driver: Any) -> np.ndarray:
    """Return a trial max-history value without mutating committed history."""

    previous = _require_float64_array(previous_history, "previous_history")
    driver = _require_float64_array(particle_driver, "particle_driver")
    if previous.ndim != 1:
        raise ValueError(
            f"previous_history must have one-dimensional shape (N,), got {previous.shape}"
        )
    if driver.shape != previous.shape:
        raise ValueError(
            "particle_driver must have the same exact float64 shape as previous_history"
        )
    if np.any(previous < 0.0):
        raise ValueError("previous_history must be non-negative")
    if np.any(driver < 0.0):
        raise ValueError("particle_driver must be non-negative")
    trial = np.maximum(previous, driver)
    if not np.all(np.isfinite(trial)):
        raise FloatingPointError("trial history maximum is non-finite")
    return _readonly_copy(trial)


def amor_positive_energy_from_strain(
    edge_strain: Any,
    lame_lambda: Any,
    shear_modulus: Any,
) -> np.ndarray:
    """Evaluate the plane-strain Amor positive energy on every directed edge."""

    strain = _require_float64_array(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError(
            "edge_strain must have exact float64 shape (E, 2, 2), "
            f"got {strain.shape}"
        )
    strain_scale = max(float(np.max(np.abs(strain))) if len(strain) else 0.0, 1.0)
    strain_skew = strain - np.swapaxes(strain, 1, 2)
    if float(np.max(np.abs(strain_skew))) > 32.0 * np.finfo(float).eps * strain_scale:
        raise ValueError("edge_strain must be symmetric for the Amor energy")
    lam = _scalar_float(lame_lambda, "lame_lambda")
    mu = _validate_material_parameter(shear_modulus, "shear_modulus")
    bulk = lam + 2.0 * mu / 3.0
    if not np.isfinite(bulk) or bulk <= 0.0:
        raise ValueError("Amor bulk modulus K must be positive and finite")

    embedded = np.zeros((len(strain), 3, 3), dtype=np.float64)
    embedded[:, :2, :2] = strain
    trace = np.trace(embedded, axis1=1, axis2=2)
    deviatoric = embedded - (trace / 3.0)[:, None, None] * _IDENTITY_3
    positive_trace = np.maximum(trace, 0.0)
    energy = 0.5 * bulk * positive_trace**2 + mu * np.sum(
        deviatoric * deviatoric, axis=(1, 2)
    )
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise FloatingPointError("Amor positive edge energy is invalid")
    return _readonly_copy(energy)


@dataclass(frozen=True)
class AmorHistoryResult:
    """Immutable edge driver and committed/trial particle histories."""

    edge_positive_energy: np.ndarray
    particle_driver: np.ndarray
    previous_history: np.ndarray
    trial_history: np.ndarray

    def __post_init__(self) -> None:
        edge = np.asarray(self.edge_positive_energy)
        driver = np.asarray(self.particle_driver)
        previous = np.asarray(self.previous_history)
        trial = np.asarray(self.trial_history)
        for name, array in (
            ("edge_positive_energy", edge),
            ("particle_driver", driver),
            ("previous_history", previous),
            ("trial_history", trial),
        ):
            if array.dtype != _FLOAT64:
                raise ValueError(f"AmorHistoryResult.{name} must use exact float64")
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"AmorHistoryResult.{name} must be finite and one-dimensional")
            object.__setattr__(self, name, _readonly_copy(array))
        if previous.shape != driver.shape or trial.shape != previous.shape:
            raise ValueError("AmorHistoryResult particle histories have incompatible shapes")
        if np.any(previous < 0.0):
            raise ValueError("AmorHistoryResult.previous_history must be non-negative")
        if np.any(driver < 0.0) or np.any(trial < previous):
            raise ValueError(
                "AmorHistoryResult driver/trial history violates the maximum-history contract"
            )

    @property
    def history_previous(self) -> np.ndarray:
        return self.previous_history

    @property
    def history_trial(self) -> np.ndarray:
        return self.trial_history


def compute_amor_history_from_edge_strain(
    edge_strain: Any,
    source: Any,
    eta: Any,
    lame_lambda: Any,
    shear_modulus: Any,
    previous_history: Any,
) -> AmorHistoryResult:
    """Build the Amor particle driver using source-only ``np.add.at`` averaging."""

    strain = _require_float64_array(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError(
            "edge_strain must have exact float64 shape (E, 2, 2), "
            f"got {strain.shape}"
        )
    source_array = np.asarray(source)
    if not np.issubdtype(source_array.dtype, np.integer) or source_array.shape != (
        len(strain),
    ):
        raise ValueError(
            f"source must have integer shape ({len(strain)},), got {source_array.shape}"
        )
    source_array = np.asarray(source_array, dtype=np.int64)
    eta_array = _require_float64_array(eta, "eta", shape=(len(strain),))
    if np.any(eta_array <= 0.0):
        raise ValueError("eta must contain strictly positive normalized weights")
    if np.any(source_array < 0):
        raise ValueError("source contains a negative particle index")
    previous = _require_float64_array(previous_history, "previous_history")
    if previous.ndim != 1:
        raise ValueError(
            "previous_history must have one-dimensional shape (N,), "
            f"got {previous.shape}"
        )
    if np.any(previous < 0.0):
        raise ValueError("previous_history must be non-negative")
    n_particles = len(previous)
    if len(source_array) and np.max(source_array) >= n_particles:
        raise ValueError(
            "source contains a particle index outside previous_history shape "
            f"({n_particles},)"
        )
    eta_sum = np.bincount(
        source_array, weights=eta_array, minlength=n_particles
    ).astype(np.float64, copy=False)
    if not np.allclose(eta_sum, 1.0, rtol=0.0, atol=_ETA_SUM_ATOL):
        bad = np.flatnonzero(np.abs(eta_sum - 1.0) > _ETA_SUM_ATOL)
        raise ValueError(
            "eta must sum to one for every source particle; invalid sources "
            f"{bad[:10].tolist()}"
        )

    edge_positive = amor_positive_energy_from_strain(
        strain, lame_lambda, shear_modulus
    )
    particle_driver = np.zeros(n_particles, dtype=np.float64)
    # This is deliberately source-only.  There is no endpoint average or
    # damage factor in the history driver.
    np.add.at(particle_driver, source_array, eta_array * edge_positive)
    if not np.all(np.isfinite(particle_driver)) or np.any(particle_driver < 0.0):
        raise FloatingPointError("Amor particle history driver is invalid")
    trial_history = update_history_max(previous, particle_driver)
    return AmorHistoryResult(
        edge_positive_energy=edge_positive,
        particle_driver=particle_driver,
        previous_history=previous,
        trial_history=trial_history,
    )


def compute_amor_history(
    mechanics: BondAssociatedMechanics,
    displacement: Any,
    previous_history: Any,
) -> AmorHistoryResult:
    """Evaluate Amor history from a BondAssociatedMechanics edge strain."""

    if not isinstance(mechanics, BondAssociatedMechanics):
        raise TypeError("mechanics must be a BondAssociatedMechanics instance")
    evaluation = mechanics.evaluate(displacement)
    return compute_amor_history_from_edge_strain(
        evaluation.edge_strain,
        mechanics.source,
        mechanics.eta,
        mechanics.lame_lambda,
        mechanics.shear_modulus,
        previous_history,
    )


def solve_fixed_damage_mechanics(
    mechanics: FixedDamageBondAssociatedMechanics,
    constraints: Any,
    u_bar: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: Any | None = None,
    allow_initial_early_exit: bool = False,
) -> Any:
    """Run the inherited fixed-damage hard-constraint mechanics solve."""

    if not isinstance(mechanics, FixedDamageBondAssociatedMechanics):
        raise TypeError(
            "mechanics must be a FixedDamageBondAssociatedMechanics instance"
        )
    return mechanics.solve(
        constraints,
        u_bar,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=initial_guess,
        allow_initial_early_exit=allow_initial_early_exit,
    )


# Descriptive aliases retained for callers that name the two pure operations
# directly rather than using the longer public function names.
solve_fixed_history = solve_fixed_history_damage
update_trial_history = update_history_max


def _max_abs(values: Any) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _validate_nonnegative_scalar(value: Any, name: str) -> float:
    result = _scalar_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    return result


def _validate_acceptance_mode(value: Any) -> str:
    """Return the exact acceptance mode string or fail loudly."""

    if isinstance(value, str) and value in ACCEPTANCE_MODES:
        return value
    raise ValueError(
        "acceptance mode must be one of "
        + "/".join(ACCEPTANCE_MODES)
        + f", got {value!r}"
    )


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


def _failed_warning_gates(diagnostics: "StaggerDiagnostics") -> tuple[str, ...]:
    """Return the canonical ordered list of failed bypassable gates."""

    passes = diagnostics.gate_passes
    return tuple(name for name in WARNING_GATE_NAMES if not passes[name])


@dataclass(frozen=True)
class GateTolerance:
    """Absolute/relative gate with an explicit scale floor.

    The gate limit is
    ``max(atol, rtol * max(error_scale, scale_floor))``.  All three
    constructor arguments are deliberately required; continuation tolerances
    must not be inherited from an unrelated solver configuration.
    """

    atol: float
    rtol: float
    scale_floor: float

    def __post_init__(self) -> None:
        atol = _validate_nonnegative_scalar(self.atol, "GateTolerance.atol")
        rtol = _validate_nonnegative_scalar(self.rtol, "GateTolerance.rtol")
        scale_floor = _scalar_float(self.scale_floor, "GateTolerance.scale_floor")
        if scale_floor <= 0.0:
            raise ValueError("GateTolerance.scale_floor must be positive and finite")
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "scale_floor", scale_floor)

    def limit(self, error_scale: Any) -> float:
        """Return the explicit acceptance limit for one non-negative scale."""

        scale = _scalar_float(error_scale, "error_scale")
        if scale < 0.0:
            raise ValueError("error_scale must be non-negative and finite")
        return max(self.atol, self.rtol * max(scale, self.scale_floor))


@dataclass(frozen=True)
class StrictContinuationConfig:
    """All settings for the strict fracture2d staggered controller.

    The controller uses a fixed accepted-step increment after a cutback; it
    does not invent an unrequested increment-growth rule.  ``load_start`` is
    required to be exactly zero because the initial zero state is part of the
    continuation archive.
    """

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

    def __post_init__(self) -> None:
        load_start = _scalar_float(self.load_start, "load_start")
        load_final = _scalar_float(self.load_final, "load_final")
        du_initial = _scalar_float(self.du_initial, "du_initial")
        du_min = _scalar_float(self.du_min, "du_min")
        cutback_factor = _scalar_float(self.cutback_factor, "cutback_factor")
        if load_start != 0.0:
            raise ValueError("load_start must be exactly zero for the archived zero state")
        if load_final <= load_start:
            raise ValueError("load_final must be strictly greater than load_start")
        if du_min <= 0.0 or du_initial <= 0.0:
            raise ValueError("du_initial and du_min must be positive and finite")
        if du_min > du_initial:
            raise ValueError("du_min must not exceed du_initial")
        if du_initial > load_final - load_start:
            raise ValueError("du_initial must not exceed the configured load span")
        if not 0.0 < cutback_factor < 1.0:
            raise ValueError("cutback_factor must lie strictly between zero and one")

        raw_max_cutbacks = np.asarray(self.max_cutbacks)
        if (
            raw_max_cutbacks.ndim != 0
            or not np.issubdtype(raw_max_cutbacks.dtype, np.integer)
        ):
            raise ValueError("max_cutbacks must be a positive integer")
        max_cutbacks = int(raw_max_cutbacks)
        if max_cutbacks < 1:
            raise ValueError("max_cutbacks must be a positive integer")

        raw_max_stagger = np.asarray(self.max_stagger_iterations)
        if (
            raw_max_stagger.ndim != 0
            or not np.issubdtype(raw_max_stagger.dtype, np.integer)
        ):
            raise ValueError("max_stagger_iterations must be a positive integer")
        max_stagger_iterations = int(raw_max_stagger)
        if max_stagger_iterations < 1:
            raise ValueError("max_stagger_iterations must be a positive integer")

        cg_rtol = _validate_nonnegative_scalar(self.cg_rtol, "cg_rtol")
        cg_atol = _validate_nonnegative_scalar(self.cg_atol, "cg_atol")
        if cg_rtol == 0.0 and cg_atol == 0.0:
            raise ValueError("cg_rtol and cg_atol cannot both be zero")
        raw_cg_maxiter = np.asarray(self.cg_maxiter)
        if (
            raw_cg_maxiter.ndim != 0
            or not np.issubdtype(raw_cg_maxiter.dtype, np.integer)
        ):
            raise ValueError("cg_maxiter must be a positive integer")
        cg_maxiter = int(raw_cg_maxiter)
        if cg_maxiter <= 0:
            raise ValueError("cg_maxiter must be a positive integer")

        tolerance_names = (
            "displacement",
            "damage",
            "history",
            "reaction",
            "damage_residual",
            "force",
            "moment",
        )
        for name in tolerance_names:
            if not isinstance(getattr(self, name), GateTolerance):
                raise TypeError(f"{name} must be a GateTolerance instance")

        healing_atol = _validate_nonnegative_scalar(self.healing_atol, "healing_atol")
        damage_range_atol = _validate_nonnegative_scalar(
            self.damage_range_atol, "damage_range_atol"
        )
        object.__setattr__(self, "load_start", load_start)
        object.__setattr__(self, "load_final", load_final)
        object.__setattr__(self, "du_initial", du_initial)
        object.__setattr__(self, "du_min", du_min)
        object.__setattr__(self, "cutback_factor", cutback_factor)
        object.__setattr__(self, "max_cutbacks", max_cutbacks)
        object.__setattr__(self, "max_stagger_iterations", max_stagger_iterations)
        object.__setattr__(self, "cg_rtol", cg_rtol)
        object.__setattr__(self, "cg_atol", cg_atol)
        object.__setattr__(self, "cg_maxiter", cg_maxiter)
        object.__setattr__(self, "healing_atol", healing_atol)
        object.__setattr__(self, "damage_range_atol", damage_range_atol)

    # These aliases retain the vocabulary used by the existing fracture1d
    # controllers without introducing optional constructor arguments.
    @property
    def u_final(self) -> float:
        return self.load_final

    @property
    def du(self) -> float:
        return self.du_initial

    @property
    def max_stagger_iters(self) -> int:
        return self.max_stagger_iterations

    @property
    def displacement_tolerance(self) -> GateTolerance:
        return self.displacement

    @property
    def damage_tolerance(self) -> GateTolerance:
        return self.damage

    @property
    def history_tolerance(self) -> GateTolerance:
        return self.history

    @property
    def reaction_tolerance(self) -> GateTolerance:
        return self.reaction

    @property
    def damage_residual_tolerance(self) -> GateTolerance:
        return self.damage_residual

    @property
    def force_tolerance(self) -> GateTolerance:
        return self.force

    @property
    def moment_tolerance(self) -> GateTolerance:
        return self.moment

    @property
    def atol_u(self) -> GateTolerance:
        return self.displacement

    @property
    def atol_d(self) -> GateTolerance:
        return self.damage

    @property
    def atol_H(self) -> GateTolerance:
        return self.history

    @property
    def atol_R(self) -> GateTolerance:
        return self.reaction


@dataclass(frozen=True)
class ContinuationMetadata:
    """Immutable static model data captured by a continuation result."""

    coordinates: np.ndarray
    volumes: np.ndarray
    directed_edges: np.ndarray
    bottom_mask: np.ndarray
    top_mask: np.ndarray
    anchor_particle: int
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    youngs_modulus: float
    poisson_ratio: float
    k_res: float
    rank_rtol: float
    Gc: float
    l0: float
    delta: float
    kernel: str
    plane_strain: bool
    stabilization: str
    mechanics_backend: str
    acceptance_mode: str
    dtype: np.dtype[Any]
    config: StrictContinuationConfig

    def __post_init__(self) -> None:
        coordinates = _require_float64_array(self.coordinates, "metadata.coordinates")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
            raise ValueError("metadata.coordinates must have non-empty shape (N, 2)")
        if len(np.unique(coordinates, axis=0)) != len(coordinates):
            raise ValueError("metadata.coordinates must contain unique particles")
        volumes = _require_float64_array(
            self.volumes, "metadata.volumes", shape=(len(coordinates),)
        )
        if np.any(volumes <= 0.0):
            raise ValueError("metadata.volumes must be strictly positive")

        edges = np.asarray(self.directed_edges)
        if edges.dtype != np.dtype(np.int64) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("metadata.directed_edges must have exact int64 shape (E, 2)")
        if len(edges) == 0:
            raise ValueError("metadata.directed_edges must be non-empty")
        if np.any(edges < 0) or np.any(edges >= len(coordinates)):
            raise ValueError("metadata.directed_edges contain an out-of-range index")
        if np.any(edges[:, 0] == edges[:, 1]) or len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("metadata.directed_edges must be unique and self-edge-free")
        edge_set = {tuple(edge) for edge in edges.tolist()}
        if any((j, i) not in edge_set for i, j in edge_set):
            raise ValueError("metadata.directed_edges must be reciprocal")
        sorted_edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
        if not np.array_equal(edges, sorted_edges):
            raise ValueError("metadata.directed_edges must be lexicographically sorted")

        bottom = np.asarray(self.bottom_mask)
        top = np.asarray(self.top_mask)
        if (
            bottom.dtype != np.dtype(bool)
            or top.dtype != np.dtype(bool)
            or bottom.shape != (len(coordinates),)
            or top.shape != (len(coordinates),)
        ):
            raise ValueError("metadata boundary masks must be boolean shape (N,)")
        if not np.any(bottom) or not np.any(top) or np.any(bottom & top):
            raise ValueError("metadata bottom/top masks must be nonempty and disjoint")
        raw_anchor = np.asarray(self.anchor_particle)
        if (
            raw_anchor.ndim != 0
            or not np.issubdtype(raw_anchor.dtype, np.integer)
            or isinstance(self.anchor_particle, (bool, np.bool_))
        ):
            raise ValueError("metadata.anchor_particle must be an integer")
        anchor = int(raw_anchor)
        if anchor < 0 or anchor >= len(coordinates) or not bottom[anchor]:
            raise ValueError("metadata.anchor_particle must be a bottom particle")

        constrained = np.asarray(self.constrained_dofs)
        free = np.asarray(self.free_dofs)
        for name, values in (("constrained_dofs", constrained), ("free_dofs", free)):
            if values.dtype != np.dtype(np.int64) or values.ndim != 1:
                raise ValueError(f"metadata.{name} must be one-dimensional int64")
            if len(values) and (np.min(values) < 0 or np.max(values) >= 2 * len(coordinates)):
                raise ValueError(f"metadata.{name} contains an out-of-range DOF")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"metadata.{name} contains duplicate DOFs")
        expected_constrained = np.concatenate(
            (
                2 * np.flatnonzero(bottom) + 1,
                2 * np.flatnonzero(top) + 1,
                np.asarray([2 * anchor], dtype=np.int64),
            )
        )
        expected_free = np.setdiff1d(
            np.arange(2 * len(coordinates), dtype=np.int64),
            expected_constrained,
        )
        if not np.array_equal(constrained, expected_constrained) or not np.array_equal(
            free, expected_free
        ):
            raise ValueError("metadata constrained/free DOFs are inconsistent with masks")

        youngs_modulus = _validate_material_parameter(
            self.youngs_modulus, "metadata.youngs_modulus"
        )
        poisson_ratio = _scalar_float(self.poisson_ratio, "metadata.poisson_ratio")
        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError("metadata.poisson_ratio must satisfy -1 < nu < 0.5")
        k_res = _validate_material_parameter(self.k_res, "metadata.k_res")
        rank_rtol = _scalar_float(self.rank_rtol, "metadata.rank_rtol")
        if rank_rtol <= 0.0 or rank_rtol >= 1.0:
            raise ValueError(
                "metadata.rank_rtol must be positive, finite, and less than one"
            )
        gc_value = _validate_material_parameter(self.Gc, "metadata.Gc")
        l0_value = _validate_material_parameter(self.l0, "metadata.l0")
        delta = _validate_material_parameter(self.delta, "metadata.delta")

        row = edges[:, 0]
        col = edges[:, 1]
        distances = np.linalg.norm(
            coordinates[col] - coordinates[row], axis=1
        )
        if (
            not np.all(np.isfinite(distances))
            or np.any(distances <= 0.0)
            or np.any(distances >= delta)
        ):
            raise ValueError(
                "metadata.directed_edges must satisfy strict 0 < distance < delta"
            )
        weights = 120.0 / (np.pi * delta**4) * (1.0 - distances / delta) ** 2
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError(
                "metadata directed edges must have positive finite cubic weights"
            )
        family_mass = np.bincount(
            row,
            weights=weights * volumes[col],
            minlength=len(coordinates),
        ).astype(np.float64, copy=False)
        if not np.all(np.isfinite(family_mass)) or np.any(family_mass <= 0.0):
            bad = np.flatnonzero(
                (~np.isfinite(family_mass)) | (family_mass <= 0.0)
            )
            raise ValueError(
                "every metadata particle must have positive finite outgoing "
                f"family mass; invalid particles {bad[:10].tolist()}"
            )
        if self.kernel != "cubic":
            raise ValueError("metadata.kernel must be 'cubic'")
        if self.plane_strain is not True:
            raise ValueError("metadata.plane_strain must be exactly True")
        if self.stabilization != "none":
            raise ValueError("metadata.stabilization must be exactly 'none'")
        if not isinstance(self.mechanics_backend, str) or not self.mechanics_backend:
            raise ValueError("metadata.mechanics_backend must be a non-empty string")
        if self.mechanics_backend != "numpy_scipy" and not (
            self.mechanics_backend.startswith("jax_")
            and self.mechanics_backend.endswith("_cpu_replay")
        ):
            raise ValueError("metadata.mechanics_backend is not recognized")
        acceptance_mode = _validate_acceptance_mode(self.acceptance_mode)
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError("metadata.dtype must be exactly float64") from error
        if dtype != _FLOAT64:
            raise ValueError("metadata.dtype must be exactly float64")
        if not isinstance(self.config, StrictContinuationConfig):
            raise TypeError("metadata.config must be StrictContinuationConfig")

        for name, values in (
            ("coordinates", coordinates),
            ("volumes", volumes),
            ("directed_edges", edges),
            ("bottom_mask", bottom),
            ("top_mask", top),
            ("constrained_dofs", constrained),
            ("free_dofs", free),
        ):
            object.__setattr__(self, name, _bytes_backed_copy(values))
        object.__setattr__(self, "anchor_particle", anchor)
        object.__setattr__(self, "youngs_modulus", youngs_modulus)
        object.__setattr__(self, "poisson_ratio", poisson_ratio)
        object.__setattr__(self, "k_res", k_res)
        object.__setattr__(self, "rank_rtol", rank_rtol)
        object.__setattr__(self, "Gc", gc_value)
        object.__setattr__(self, "l0", l0_value)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "plane_strain", True)
        object.__setattr__(self, "acceptance_mode", acceptance_mode)
        object.__setattr__(self, "dtype", dtype)

    def validate(self) -> None:
        """Re-run the complete metadata contract before serialization."""

        ContinuationMetadata(
            coordinates=self.coordinates,
            volumes=self.volumes,
            directed_edges=self.directed_edges,
            bottom_mask=self.bottom_mask,
            top_mask=self.top_mask,
            anchor_particle=self.anchor_particle,
            constrained_dofs=self.constrained_dofs,
            free_dofs=self.free_dofs,
            youngs_modulus=self.youngs_modulus,
            poisson_ratio=self.poisson_ratio,
            k_res=self.k_res,
            rank_rtol=self.rank_rtol,
            Gc=self.Gc,
            l0=self.l0,
            delta=self.delta,
            kernel=self.kernel,
            plane_strain=self.plane_strain,
            stabilization=self.stabilization,
            mechanics_backend=self.mechanics_backend,
            acceptance_mode=self.acceptance_mode,
            dtype=self.dtype,
            config=self.config,
        )

    @property
    def E(self) -> float:
        return self.youngs_modulus

    @property
    def nu(self) -> float:
        return self.poisson_ratio

    @property
    def n_particles(self) -> int:
        return int(len(self.coordinates))

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_particles

    def validate_state(self, state: "CommittedState") -> None:
        if not isinstance(state, CommittedState):
            raise TypeError("continuation state must be CommittedState")
        if state.displacement.shape != (self.n_dofs,):
            raise ValueError("continuation displacement has the wrong metadata size")
        if state.damage.shape != (self.n_particles,) or state.history.shape != (
            self.n_particles,
        ):
            raise ValueError("continuation damage/history have the wrong metadata size")
        if state.u_bar < self.config.load_start or state.u_bar > self.config.load_final:
            raise ValueError("continuation state load lies outside configured metadata range")

    def damage_operator(self) -> Any:
        """Reconstruct the frozen cubic CSR operator for result validation."""

        row = self.directed_edges[:, 0]
        col = self.directed_edges[:, 1]
        distance = np.linalg.norm(self.coordinates[col] - self.coordinates[row], axis=1)
        kernel = 120.0 / (np.pi * self.delta**4) * (1.0 - distance / self.delta) ** 2
        weights = kernel * self.volumes[col]
        diagonal = np.bincount(row, weights=weights, minlength=self.n_particles)
        return (
            coo_matrix(
                (-weights, (row, col)),
                shape=(self.n_particles, self.n_particles),
                dtype=np.float64,
            ).tocsr()
            + diags(diagonal, format="csr")
        ).tocsr()


@dataclass(frozen=True)
class SupplementalArchiveMetadata:
    """Required mesh/provenance data not available to the controller."""

    triangles: np.ndarray
    first_moments: np.ndarray
    visibility_tolerance: float
    visibility_rule: str
    quadrature_rule: str
    study_label: str
    model_provenance: str

    def __post_init__(self) -> None:
        triangles = np.asarray(self.triangles)
        if triangles.dtype != np.dtype(np.int64) or triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("archive triangles must have exact int64 shape (T, 3)")
        if len(triangles) == 0:
            raise ValueError("archive triangles must be non-empty")
        first_moments = _require_float64_array(
            self.first_moments, "archive first_moments", shape=(2,)
        )
        visibility_tolerance = _validate_nonnegative_scalar(
            self.visibility_tolerance, "visibility_tolerance"
        )
        string_values = (
            "visibility_rule",
            "quadrature_rule",
            "study_label",
            "model_provenance",
        )
        for name in string_values:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"archive {name} must be a non-empty string")
        object.__setattr__(self, "triangles", _bytes_backed_copy(triangles))
        object.__setattr__(self, "first_moments", _bytes_backed_copy(first_moments))
        object.__setattr__(self, "visibility_tolerance", visibility_tolerance)

    def validate(self) -> None:
        """Re-run the immutable supplemental metadata contract."""

        SupplementalArchiveMetadata(
            triangles=self.triangles,
            first_moments=self.first_moments,
            visibility_tolerance=self.visibility_tolerance,
            visibility_rule=self.visibility_rule,
            quadrature_rule=self.quadrature_rule,
            study_label=self.study_label,
            model_provenance=self.model_provenance,
        )

    def validate_against(self, metadata: ContinuationMetadata) -> None:
        if not isinstance(metadata, ContinuationMetadata):
            raise TypeError("metadata must be ContinuationMetadata")
        self.validate()
        if np.any(self.triangles < 0) or np.any(self.triangles >= metadata.n_particles):
            raise ValueError("archive triangles contain an invalid metadata particle index")
        if np.any(
            (self.triangles[:, 0] == self.triangles[:, 1])
            | (self.triangles[:, 1] == self.triangles[:, 2])
            | (self.triangles[:, 0] == self.triangles[:, 2])
        ):
            raise ValueError("archive triangles must not repeat a particle")
        points = metadata.coordinates[self.triangles]
        cross = (
            (points[:, 1, 0] - points[:, 0, 0])
            * (points[:, 2, 1] - points[:, 0, 1])
            - (points[:, 1, 1] - points[:, 0, 1])
            * (points[:, 2, 0] - points[:, 0, 0])
        )
        areas = 0.5 * np.abs(cross)
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
            raise ValueError("archive triangles must have positive finite area")
        volume_moment = np.sum(metadata.volumes[:, None] * metadata.coordinates, axis=0)
        if not np.allclose(
            self.first_moments,
            volume_moment,
            rtol=0.0,
            atol=5.0e-12 * max(_max_abs(volume_moment), 1.0),
        ):
            raise ValueError("archive first_moments do not match metadata volumes/coordinates")
        if "barycentric" in self.quadrature_rule.lower():
            area_scale = max(float(np.sum(areas)), float(np.sum(metadata.volumes)), 1.0)
            if abs(float(np.sum(areas)) - float(np.sum(metadata.volumes))) > 5.0e-12 * area_scale:
                raise ValueError("archive triangle area does not match metadata volumes")
            triangle_moment = np.sum(
                areas[:, None] * np.mean(points, axis=1), axis=0
            )
            if not np.allclose(
                self.first_moments,
                triangle_moment,
                rtol=0.0,
                atol=5.0e-12 * max(_max_abs(triangle_moment), 1.0),
            ):
                raise ValueError("archive first_moments do not match archive triangles")


ArchiveMetadata = SupplementalArchiveMetadata


@dataclass(frozen=True)
class CommittedState:
    """A committed load state with storage that cannot be made writable."""

    u_bar: float
    displacement: np.ndarray
    damage: np.ndarray
    history: np.ndarray

    def __post_init__(self) -> None:
        u_bar = _scalar_float(self.u_bar, "CommittedState.u_bar")
        if u_bar < 0.0:
            raise ValueError("CommittedState.u_bar must be non-negative")
        displacement = _require_float64_array(self.displacement, "displacement")
        damage = _require_float64_array(self.damage, "damage")
        history = _require_float64_array(self.history, "history")
        if displacement.ndim != 1 or displacement.size == 0 or displacement.size % 2:
            raise ValueError("CommittedState.displacement must be a non-empty flat (2*N,) array")
        if damage.ndim != 1 or damage.size == 0:
            raise ValueError("CommittedState.damage must be a non-empty (N,) array")
        if history.shape != damage.shape:
            raise ValueError("CommittedState.history must have the same shape as damage")
        if displacement.size != 2 * damage.size:
            raise ValueError("CommittedState.displacement must contain two DOFs per damage particle")
        if np.any(history < 0.0):
            raise ValueError("CommittedState.history must be non-negative")
        object.__setattr__(self, "u_bar", u_bar)
        object.__setattr__(self, "displacement", _bytes_backed_copy(displacement))
        object.__setattr__(self, "damage", _bytes_backed_copy(damage))
        object.__setattr__(self, "history", _bytes_backed_copy(history))

    @property
    def load(self) -> float:
        return self.u_bar


@dataclass(frozen=True)
class StaggerDiagnostics:
    """All acceptance-gate values from one complete stagger sweep."""

    e_u: float
    u_scale: float
    u_limit: float
    u_pass: bool
    e_d: float
    d_scale: float
    d_limit: float
    d_pass: bool
    e_H: float
    H_scale: float
    H_limit: float
    H_pass: bool
    e_R: float
    R_scale: float
    R_limit: float
    R_pass: bool
    rho_used: np.ndarray
    rho_next: np.ndarray
    rho_used_volume: np.ndarray
    rho_next_volume: np.ndarray
    e_rho: float
    rho_scale: float
    rho_limit: float
    rho_pass: bool
    damage_min: float
    damage_max: float
    damage_range_pass: bool
    healing_min: float
    healing_pass: bool
    history_min_gap: float
    history_monotonic_pass: bool
    reaction_before: float
    reaction_after: float
    mechanics_before_free_residual_norm: float
    mechanics_after_free_residual_norm: float
    mechanics_before_free_residual_relative_norm: float
    mechanics_after_free_residual_relative_norm: float
    mechanics_before_cg_info: int
    mechanics_after_cg_info: int
    force_before: np.ndarray
    force_after: np.ndarray
    moment_before: float
    moment_after: float
    force_error: float
    force_scale: float
    force_limit: float
    force_pass: bool
    moment_error: float
    moment_scale: float
    moment_limit: float
    moment_pass: bool
    mechanics_pass: bool
    passed: bool
    complete_sweep: bool
    rejection_reason: str

    def __post_init__(self) -> None:
        # Complete-sweep diagnostics are acceptance evidence and must be fully
        # finite.  Incomplete failure diagnostics may retain the ``inf``
        # sentinels used to mark unavailable quantities, but never NaN.
        complete_sweep = bool(self.complete_sweep)
        array_specs = (
            ("rho_used", self.rho_used),
            ("rho_next", self.rho_next),
            ("rho_used_volume", self.rho_used_volume),
            ("rho_next_volume", self.rho_next_volume),
            ("force_before", self.force_before),
            ("force_after", self.force_after),
        )
        for name, values in array_specs:
            array = np.asarray(values)
            if array.dtype != _FLOAT64:
                raise ValueError(f"StaggerDiagnostics.{name} must use exact float64")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"StaggerDiagnostics.{name} must contain only finite values")
            if name.startswith("rho_") and array.ndim != 1:
                raise ValueError(f"StaggerDiagnostics.{name} must be one-dimensional")
            if name.startswith("force_") and array.shape != (2,):
                raise ValueError(f"StaggerDiagnostics.{name} must have shape (2,)")
            object.__setattr__(self, name, _bytes_backed_copy(array))

        scalar_names = (
            "e_u", "u_scale", "u_limit", "e_d", "d_scale", "d_limit",
            "e_H", "H_scale", "H_limit", "e_R", "R_scale", "R_limit",
            "e_rho", "rho_scale", "rho_limit", "damage_min", "damage_max",
            "healing_min", "history_min_gap", "reaction_before", "reaction_after",
            "mechanics_before_free_residual_norm", "mechanics_after_free_residual_norm",
            "mechanics_before_free_residual_relative_norm",
            "mechanics_after_free_residual_relative_norm", "moment_before",
            "moment_after", "force_error", "force_scale", "force_limit",
            "moment_error", "moment_scale", "moment_limit",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if np.isnan(value):
                raise ValueError(f"StaggerDiagnostics.{name} must not be NaN")
            if complete_sweep and not np.isfinite(value):
                raise ValueError(
                    f"StaggerDiagnostics.{name} must be finite for a complete sweep"
                )
            object.__setattr__(self, name, value)
        for name in (
            "u_pass", "d_pass", "H_pass", "R_pass", "rho_pass",
            "damage_range_pass", "healing_pass", "history_monotonic_pass",
            "force_pass", "moment_pass", "mechanics_pass", "passed",
            "complete_sweep",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("mechanics_before_cg_info", "mechanics_after_cg_info"):
            object.__setattr__(self, name, int(getattr(self, name)))
        if not isinstance(self.rejection_reason, str):
            raise TypeError("StaggerDiagnostics.rejection_reason must be a string")

    @property
    def displacement_error(self) -> float:
        return self.e_u

    @property
    def displacement_scale(self) -> float:
        return self.u_scale

    @property
    def displacement_limit(self) -> float:
        return self.u_limit

    @property
    def displacement_pass(self) -> bool:
        return self.u_pass

    @property
    def damage_error(self) -> float:
        return self.e_d

    @property
    def damage_scale(self) -> float:
        return self.d_scale

    @property
    def damage_limit(self) -> float:
        return self.d_limit

    @property
    def damage_pass(self) -> bool:
        return self.d_pass

    @property
    def history_error(self) -> float:
        return self.e_H

    @property
    def history_scale(self) -> float:
        return self.H_scale

    @property
    def history_limit(self) -> float:
        return self.H_limit

    @property
    def history_pass(self) -> bool:
        return self.H_pass

    @property
    def reaction_error(self) -> float:
        return self.e_R

    @property
    def reaction_scale(self) -> float:
        return self.R_scale

    @property
    def reaction_limit(self) -> float:
        return self.R_limit

    @property
    def reaction_pass(self) -> bool:
        return self.R_pass

    @property
    def damage_residual_error(self) -> float:
        return self.e_rho

    @property
    def damage_residual_scale(self) -> float:
        return self.rho_scale

    @property
    def damage_residual_limit(self) -> float:
        return self.rho_limit

    @property
    def damage_residual_pass(self) -> bool:
        return self.rho_pass

    @property
    def damage_residual_used(self) -> np.ndarray:
        return self.rho_used

    @property
    def damage_residual_committed(self) -> np.ndarray:
        return self.rho_next

    @property
    def gate_passes(self) -> dict[str, bool]:
        return {
            "displacement": self.u_pass,
            "damage": self.d_pass,
            "history": self.H_pass,
            "reaction": self.R_pass,
            "damage_residual": self.rho_pass,
            "damage_range": self.damage_range_pass,
            "healing": self.healing_pass,
            "history_monotonic": self.history_monotonic_pass,
            "force": self.force_pass,
            "moment": self.moment_pass,
            "mechanics": self.mechanics_pass,
        }

    @property
    def errors(self) -> dict[str, float]:
        return {
            "displacement": self.e_u,
            "damage": self.e_d,
            "history": self.e_H,
            "reaction": self.e_R,
            "damage_residual": self.e_rho,
            "force": self.force_error,
            "moment": self.moment_error,
        }

    @property
    def scales(self) -> dict[str, float]:
        return {
            "displacement": self.u_scale,
            "damage": self.d_scale,
            "history": self.H_scale,
            "reaction": self.R_scale,
            "damage_residual": self.rho_scale,
            "force": self.force_scale,
            "moment": self.moment_scale,
        }

    @property
    def limits(self) -> dict[str, float]:
        return {
            "displacement": self.u_limit,
            "damage": self.d_limit,
            "history": self.H_limit,
            "reaction": self.R_limit,
            "damage_residual": self.rho_limit,
            "force": self.force_limit,
            "moment": self.moment_limit,
        }

    @property
    def passes(self) -> dict[str, bool]:
        return self.gate_passes


@dataclass(frozen=True)
class AttemptRecord:
    """Audit record for one trial target, including rollback digests."""

    cutback: int
    load_start: float
    load_target: float
    accepted: bool
    failure_stage: str | None
    failure_message: str | None
    committed_before_digest: str
    committed_after_digest: str
    diagnostics: StaggerDiagnostics | None
    stagger_iterations: int
    exception_type: str | None
    acceptance_status: str | None
    sweep_complete: bool
    bypassed_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        raw_cutback = np.asarray(self.cutback)
        raw_iterations = np.asarray(self.stagger_iterations)
        for value, name in ((raw_cutback, "cutback"), (raw_iterations, "stagger_iterations")):
            if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"AttemptRecord.{name} must be an integer")
        cutback = int(raw_cutback)
        stagger_iterations = int(raw_iterations)
        if cutback < 0 or stagger_iterations < 0:
            raise ValueError("AttemptRecord integer counters must be non-negative")
        load_start = _scalar_float(self.load_start, "AttemptRecord.load_start")
        load_target = _scalar_float(self.load_target, "AttemptRecord.load_target")
        if load_start < 0.0 or load_target <= load_start:
            raise ValueError("AttemptRecord load range is invalid")
        for name in ("committed_before_digest", "committed_after_digest"):
            digest = getattr(self, name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"AttemptRecord.{name} must be a SHA256 hex digest")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(f"AttemptRecord.{name} must be a SHA256 hex digest") from error
        for name in ("failure_stage", "failure_message", "exception_type"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"AttemptRecord.{name} must be a string or None")
        bypassed_gates = _validate_bypassed_gates(self.bypassed_gates)
        if self.accepted and self.failure_stage is not None:
            raise ValueError("an accepted attempt cannot have a failure stage")
        if self.accepted and self.failure_message is not None:
            raise ValueError("an accepted attempt cannot have a failure message")
        if self.accepted and self.exception_type is not None:
            raise ValueError("an accepted attempt cannot have an exception type")
        if self.accepted and self.diagnostics is None:
            raise ValueError("an accepted attempt must retain stagger diagnostics")
        if self.accepted and stagger_iterations == 0:
            raise ValueError("an accepted attempt must complete a stagger sweep")
        if self.diagnostics is not None and not isinstance(self.diagnostics, StaggerDiagnostics):
            raise TypeError("AttemptRecord.diagnostics must be StaggerDiagnostics or None")
        sweep_complete = bool(self.sweep_complete)
        diagnostics = self.diagnostics if self.accepted else None
        if self.accepted:
            if diagnostics is None:
                raise ValueError("an accepted attempt must retain stagger diagnostics")
            if self.acceptance_status not in ACCEPTANCE_STATUSES:
                raise ValueError(
                    "an accepted attempt must record "
                    + "/".join(ACCEPTANCE_STATUSES)
                    + f" acceptance, got {self.acceptance_status!r}"
                )
            if not sweep_complete or not diagnostics.complete_sweep:
                raise ValueError(
                    "an accepted attempt must record a complete stagger sweep"
                )
            if self.acceptance_status == "strict_passed":
                if not diagnostics.passed:
                    raise ValueError(
                        "a strict_passed attempt must carry passing diagnostics"
                    )
                if bypassed_gates:
                    raise ValueError(
                        "a strict_passed attempt cannot bypass acceptance gates"
                    )
            else:
                if diagnostics.passed:
                    raise ValueError(
                        "a warning_accepted attempt must retain passed=False "
                        "diagnostics"
                    )
                if not bypassed_gates:
                    raise ValueError(
                        "a warning_accepted attempt must record its bypassed gates"
                    )
        else:
            if self.acceptance_status is not None:
                raise ValueError(
                    "a rejected attempt cannot record an acceptance status"
                )
            if bypassed_gates:
                raise ValueError(
                    "a rejected attempt cannot record bypassed acceptance gates"
                )
            if sweep_complete != (
                self.diagnostics is not None and self.diagnostics.complete_sweep
            ):
                raise ValueError(
                    "AttemptRecord.sweep_complete must match its diagnostics"
                )
        object.__setattr__(self, "cutback", cutback)
        object.__setattr__(self, "load_start", load_start)
        object.__setattr__(self, "load_target", load_target)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "stagger_iterations", stagger_iterations)
        object.__setattr__(self, "acceptance_status", self.acceptance_status)
        object.__setattr__(self, "sweep_complete", sweep_complete)
        object.__setattr__(self, "bypassed_gates", bypassed_gates)

    @property
    def cutback_count(self) -> int:
        return self.cutback

    @property
    def success(self) -> bool:
        return self.accepted

    @property
    def stage(self) -> str:
        return "accepted" if self.accepted else str(self.failure_stage)

    @property
    def message(self) -> str | None:
        return self.failure_message

    @property
    def committed_before_sha256(self) -> str:
        return self.committed_before_digest

    @property
    def committed_after_sha256(self) -> str:
        return self.committed_after_digest


@dataclass(frozen=True)
class AcceptedStep:
    """One accepted load increment and its final stagger-sweep objects."""

    load_start: float
    load_target: float
    increment: float
    state: CommittedState
    history_used: np.ndarray
    amor_driver: AmorHistoryResult
    fixed_history_result: FixedHistoryDamageResult
    mechanics_result: Any
    diagnostics: StaggerDiagnostics
    stagger_iterations: int
    cutback_count: int
    acceptance_status: str
    bypassed_gates: tuple[str, ...]
    damage_before: np.ndarray
    mechanics_before_result: Any

    def __post_init__(self) -> None:
        if not isinstance(self.state, CommittedState):
            raise TypeError("AcceptedStep.state must be a CommittedState")
        if not isinstance(self.amor_driver, AmorHistoryResult):
            raise TypeError("AcceptedStep.amor_driver must be an AmorHistoryResult")
        if not isinstance(self.fixed_history_result, FixedHistoryDamageResult):
            raise TypeError(
                "AcceptedStep.fixed_history_result must be a FixedHistoryDamageResult"
            )
        if not isinstance(self.mechanics_result, MechanicsSolveResult):
            raise TypeError("AcceptedStep.mechanics_result must be a MechanicsSolveResult")
        if not isinstance(
            self.mechanics_before_result, MechanicsSolveResult
        ):
            raise TypeError(
                "AcceptedStep.mechanics_before_result must be a MechanicsSolveResult"
            )
        if not isinstance(self.diagnostics, StaggerDiagnostics):
            raise TypeError("AcceptedStep.diagnostics must be StaggerDiagnostics")
        acceptance_status = self.acceptance_status
        if acceptance_status not in ACCEPTANCE_STATUSES:
            raise ValueError(
                "AcceptedStep.acceptance_status must be one of "
                + "/".join(ACCEPTANCE_STATUSES)
                + f", got {acceptance_status!r}"
            )
        bypassed_gates = _validate_bypassed_gates(self.bypassed_gates)
        if not self.diagnostics.complete_sweep:
            raise ValueError(
                "AcceptedStep diagnostics must come from a complete stagger sweep"
            )
        if not self.diagnostics.mechanics_pass:
            raise ValueError(
                "AcceptedStep diagnostics must record passing mechanics solves"
            )
        expected_bypassed = _failed_warning_gates(self.diagnostics)
        if acceptance_status == "strict_passed":
            if not self.diagnostics.passed:
                raise ValueError(
                    "AcceptedStep strict_passed diagnostics must indicate a passed sweep"
                )
            if bypassed_gates:
                raise ValueError(
                    "AcceptedStep strict_passed cannot bypass acceptance gates"
                )
        else:
            if self.diagnostics.passed:
                raise ValueError(
                    "AcceptedStep warning_accepted diagnostics must retain passed=False"
                )
            if not expected_bypassed:
                raise ValueError(
                    "AcceptedStep warning_accepted requires at least one failed "
                    "bypassable gate"
                )
            if bypassed_gates != expected_bypassed:
                raise ValueError(
                    "AcceptedStep.bypassed_gates must exactly match the canonical "
                    "failed-gate list"
                )
        load_start = _scalar_float(self.load_start, "AcceptedStep.load_start")
        load_target = _scalar_float(self.load_target, "AcceptedStep.load_target")
        increment = _scalar_float(self.increment, "AcceptedStep.increment")
        if load_target <= load_start or increment <= 0.0:
            raise ValueError("AcceptedStep load range/increment is invalid")
        if not np.isclose(
            load_target - load_start, increment, rtol=0.0, atol=8.0 * np.finfo(float).eps
        ):
            raise ValueError("AcceptedStep.increment does not match its load range")
        if self.state.u_bar != load_target:
            raise ValueError("AcceptedStep state.u_bar must equal load_target exactly")
        if self.mechanics_result.u_bar != load_target:
            raise ValueError("AcceptedStep mechanics_result.u_bar must equal load_target exactly")
        if self.mechanics_before_result.u_bar != load_target:
            raise ValueError(
                "AcceptedStep mechanics_before_result.u_bar must equal load_target exactly"
            )
        damage_before = _require_float64_array(
            self.damage_before,
            "AcceptedStep.damage_before",
            shape=self.state.damage.shape,
        )
        history_used = _require_float64_array(
            self.history_used, "AcceptedStep.history_used", shape=self.state.history.shape
        )
        if np.any(history_used < 0.0):
            raise ValueError("AcceptedStep.history_used must be non-negative")
        if not np.array_equal(self.fixed_history_result.damage, self.state.damage):
            raise ValueError("AcceptedStep state damage is not the fixed-history result damage")
        if not np.array_equal(self.fixed_history_result.history, history_used):
            raise ValueError("AcceptedStep history_used is not the fixed-history input history")
        fixed_history = self.fixed_history_result
        _require_scale_aware_close(
            fixed_history.strong_residual_norm,
            np.linalg.norm(fixed_history.strong_residual),
            "AcceptedStep fixed-history strong residual norm",
        )
        _require_scale_aware_close(
            fixed_history.strong_residual_inf,
            _max_abs(fixed_history.strong_residual),
            "AcceptedStep fixed-history strong residual infinity norm",
        )
        _require_scale_aware_close(
            fixed_history.euclidean_residual_norm,
            np.linalg.norm(fixed_history.euclidean_residual),
            "AcceptedStep fixed-history Euclidean residual norm",
        )
        _require_scale_aware_close(
            fixed_history.euclidean_residual_inf,
            _max_abs(fixed_history.euclidean_residual),
            "AcceptedStep fixed-history Euclidean residual infinity norm",
        )
        _require_scale_aware_close(
            self.mechanics_result.free_residual_norm,
            np.linalg.norm(self.mechanics_result.free_residual),
            "AcceptedStep mechanics free residual norm",
        )
        _require_scale_aware_close(
            self.mechanics_result.free_residual_relative_norm,
            self.mechanics_result.free_residual_norm
            / max(self.mechanics_result.free_rhs_norm, 1.0e-300),
            "AcceptedStep mechanics free residual relative norm",
        )
        if (
            self.mechanics_result.free_residual_norm
            >= self.mechanics_result.cg_convergence_limit
        ):
            raise ValueError(
                "AcceptedStep mechanics residual does not satisfy its CG limit"
            )
        _require_scale_aware_close(
            fixed_history.total_energy,
            fixed_history.crack_local_energy
            + fixed_history.history_energy
            + fixed_history.graph_energy,
            "AcceptedStep fixed-history total energy",
        )
        if fixed_history.damage_min != float(np.min(fixed_history.damage)):
            raise ValueError("AcceptedStep fixed-history damage minimum is inconsistent")
        if fixed_history.damage_max != float(np.max(fixed_history.damage)):
            raise ValueError("AcceptedStep fixed-history damage maximum is inconsistent")
        if fixed_history.below_zero_count != int(
            np.count_nonzero(fixed_history.damage < 0.0)
        ):
            raise ValueError(
                "AcceptedStep fixed-history below-zero count is inconsistent"
            )
        if fixed_history.above_one_count != int(
            np.count_nonzero(fixed_history.damage > 1.0)
        ):
            raise ValueError(
                "AcceptedStep fixed-history above-one count is inconsistent"
            )
        _require_scale_aware_close(
            self.fixed_history_result.strong_residual,
            self.diagnostics.rho_used,
            "AcceptedStep fixed-history strong residual versus diagnostics.rho_used",
        )
        if not np.array_equal(self.amor_driver.trial_history, self.state.history):
            raise ValueError("AcceptedStep state history is not the final Amor history")
        if not np.array_equal(self.mechanics_result.displacement, self.state.displacement):
            raise ValueError("AcceptedStep state displacement is not the final mechanics result")
        if self.diagnostics.rho_next.shape != self.state.damage.shape:
            raise ValueError("AcceptedStep committed residual has the wrong shape")
        if self.diagnostics.rho_next_volume.shape != self.state.damage.shape:
            raise ValueError("AcceptedStep volume residual has the wrong shape")
        if self.diagnostics.damage_min != float(np.min(self.state.damage)):
            raise ValueError("AcceptedStep damage minimum does not match diagnostics")
        if self.diagnostics.damage_max != float(np.max(self.state.damage)):
            raise ValueError("AcceptedStep damage maximum does not match diagnostics")
        if self.diagnostics.reaction_after != self.mechanics_result.top_reaction:
            raise ValueError("AcceptedStep reaction diagnostic does not match mechanics")
        if not np.array_equal(self.diagnostics.force_after, self.mechanics_result.total_force):
            raise ValueError("AcceptedStep force diagnostic does not match mechanics")
        if self.diagnostics.moment_after != self.mechanics_result.total_moment:
            raise ValueError("AcceptedStep moment diagnostic does not match mechanics")
        if (
            self.diagnostics.mechanics_after_cg_info != self.mechanics_result.cg_info
            or self.diagnostics.mechanics_after_free_residual_norm
            != self.mechanics_result.free_residual_norm
            or self.diagnostics.mechanics_after_free_residual_relative_norm
            != self.mechanics_result.free_residual_relative_norm
        ):
            raise ValueError("AcceptedStep mechanics diagnostics do not match mechanics")
        # Final-sweep before-side relationships.  These tie the stored gate
        # summaries to the authoritative before-sweep execution objects.
        if self.diagnostics.reaction_before != self.mechanics_before_result.top_reaction:
            raise ValueError(
                "AcceptedStep before-reaction diagnostic does not match mechanics-before"
            )
        if not np.array_equal(
            self.diagnostics.force_before, self.mechanics_before_result.total_force
        ):
            raise ValueError(
                "AcceptedStep before-force diagnostic does not match mechanics-before"
            )
        if self.diagnostics.moment_before != self.mechanics_before_result.total_moment:
            raise ValueError(
                "AcceptedStep before-moment diagnostic does not match mechanics-before"
            )
        if (
            self.diagnostics.mechanics_before_cg_info
            != self.mechanics_before_result.cg_info
            or self.diagnostics.mechanics_before_free_residual_norm
            != self.mechanics_before_result.free_residual_norm
            or self.diagnostics.mechanics_before_free_residual_relative_norm
            != self.mechanics_before_result.free_residual_relative_norm
        ):
            raise ValueError(
                "AcceptedStep before mechanics diagnostics do not match mechanics-before"
            )
        if self.diagnostics.e_d != _max_abs(self.state.damage - damage_before):
            raise ValueError(
                "AcceptedStep damage error does not match the final-sweep damage iterate"
            )
        raw_iterations = np.asarray(self.stagger_iterations)
        if (
            raw_iterations.ndim != 0
            or not np.issubdtype(raw_iterations.dtype, np.integer)
            or isinstance(self.stagger_iterations, (bool, np.bool_))
        ):
            raise ValueError("AcceptedStep.stagger_iterations must be an integer")
        stagger_iterations = int(raw_iterations)
        if stagger_iterations <= 0:
            raise ValueError(
                "AcceptedStep.stagger_iterations must be positive"
            )
        object.__setattr__(self, "load_start", load_start)
        object.__setattr__(self, "load_target", load_target)
        object.__setattr__(self, "increment", increment)
        object.__setattr__(self, "history_used", _bytes_backed_copy(history_used))
        object.__setattr__(self, "damage_before", _bytes_backed_copy(damage_before))
        raw_cutback = np.asarray(self.cutback_count)
        if raw_cutback.ndim != 0 or not np.issubdtype(raw_cutback.dtype, np.integer):
            raise ValueError("AcceptedStep.cutback_count must be an integer")
        cutback_count = int(raw_cutback)
        if cutback_count < 0:
            raise ValueError("AcceptedStep.cutback_count must be non-negative")
        object.__setattr__(self, "stagger_iterations", stagger_iterations)
        object.__setattr__(self, "cutback_count", cutback_count)
        object.__setattr__(self, "acceptance_status", acceptance_status)
        object.__setattr__(self, "bypassed_gates", bypassed_gates)

    @property
    def cutback(self) -> int:
        return self.cutback_count

    @property
    def H_used(self) -> np.ndarray:
        return self.history_used

    @property
    def amor_history(self) -> AmorHistoryResult:
        return self.amor_driver

    @property
    def fixed_history(self) -> FixedHistoryDamageResult:
        return self.fixed_history_result

    @property
    def mechanics(self) -> Any:
        return self.mechanics_result


def _diagnostics_match(
    left: StaggerDiagnostics | None,
    right: StaggerDiagnostics | None,
) -> bool:
    """Compare diagnostics exactly, including every numerical array."""

    if left is None or right is None:
        return left is right
    if not isinstance(left, StaggerDiagnostics) or not isinstance(
        right, StaggerDiagnostics
    ):
        return False
    for name in StaggerDiagnostics.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            if not isinstance(left_value, np.ndarray) or not isinstance(
                right_value, np.ndarray
            ):
                return False
            if not np.array_equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def _require_diagnostics_replay_match(
    stored: StaggerDiagnostics,
    replayed: StaggerDiagnostics,
    context: str,
) -> None:
    """Compare stored diagnostics against an authoritative replay field by field.

    Booleans, integers, and strings (including every pass flag, the rejection
    reason, and ``complete_sweep``) must match exactly.  Floats and float
    arrays use the shared scale-aware comparison so a coherently forged
    summary, scale, vector, or pass flag cannot validate or archive.
    """

    for name in StaggerDiagnostics.__dataclass_fields__:
        actual = getattr(stored, name)
        expected = getattr(replayed, name)
        if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            _require_scale_aware_close(
                actual,
                expected,
                f"{context} diagnostics {name}",
            )
            continue
        if isinstance(actual, (bool, np.bool_)) or isinstance(
            expected, (bool, np.bool_)
        ):
            if bool(actual) != bool(expected):
                raise ValueError(
                    f"{context} diagnostics {name} does not match the "
                    "authoritative final-sweep replay"
                )
            continue
        if isinstance(actual, str) or isinstance(expected, str):
            if actual != expected:
                raise ValueError(
                    f"{context} diagnostics {name} does not match the "
                    "authoritative final-sweep replay"
                )
            continue
        if isinstance(actual, float) or isinstance(expected, float):
            _require_scale_aware_close(
                actual,
                expected,
                f"{context} diagnostics {name}",
            )
            continue
        if actual != expected:
            raise ValueError(
                f"{context} diagnostics {name} does not match the "
                "authoritative final-sweep replay"
            )


@dataclass(frozen=True)
class ContinuationResult:
    """In-memory strict continuation archive."""

    initial_state: CommittedState
    metadata: ContinuationMetadata
    states: tuple[CommittedState, ...]
    accepted_steps: tuple[AcceptedStep, ...]
    attempts: tuple[AttemptRecord, ...]

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "accepted_steps", tuple(self.accepted_steps))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    def validate(self) -> None:
        """Revalidate all state/step/audit pairings before use or serialization."""

        if not isinstance(self.initial_state, CommittedState):
            raise TypeError("ContinuationResult.initial_state must be a CommittedState")
        if not isinstance(self.metadata, ContinuationMetadata):
            raise TypeError("ContinuationResult.metadata must be ContinuationMetadata")
        states = tuple(self.states)
        accepted_steps = tuple(self.accepted_steps)
        attempts = tuple(self.attempts)
        if not states or states[0] is not self.initial_state:
            raise ValueError("ContinuationResult.states must begin with initial_state")
        if len(states) != len(accepted_steps) + 1:
            raise ValueError("ContinuationResult.states must include initial plus accepted states")
        if accepted_steps and not attempts:
            raise ValueError("ContinuationResult.attempts cannot be empty for accepted steps")
        self.metadata.validate()
        for state in states:
            if not isinstance(state, CommittedState):
                raise TypeError("ContinuationResult.states must contain CommittedState values")
            # Reconstructing the value catches post-construction field rebinding
            # while preserving the original state object pairing below.
            CommittedState(
                state.u_bar,
                state.displacement,
                state.damage,
                state.history,
            )
        self.metadata.validate_state(self.initial_state)
        if self.initial_state.u_bar != self.metadata.config.load_start:
            raise ValueError("ContinuationResult initial load does not match metadata config")
        if not np.all(self.initial_state.displacement == 0.0):
            raise ValueError("ContinuationResult initial displacement must be exactly zero")
        if not np.all(self.initial_state.damage == 0.0) or not np.all(
            self.initial_state.history == 0.0
        ):
            raise ValueError("ContinuationResult initial damage/history must be exactly zero")
        damage_matrix = self.metadata.damage_operator()
        authoritative_intact = BondAssociatedMechanics(
            self.metadata.coordinates,
            self.metadata.volumes,
            self.metadata.directed_edges,
            self.metadata.delta,
            self.metadata.E,
            self.metadata.nu,
            self.metadata.k_res,
            self.metadata.rank_rtol,
        )
        authoritative_constraints = authoritative_intact.build_constraints(
            self.metadata.bottom_mask,
            self.metadata.top_mask,
            self.metadata.anchor_particle,
        )
        edge_row = self.metadata.directed_edges[:, 0]
        edge_col = self.metadata.directed_edges[:, 1]
        edge_distance = np.linalg.norm(
            self.metadata.coordinates[edge_col]
            - self.metadata.coordinates[edge_row],
            axis=1,
        )
        edge_weight = (
            120.0
            / (np.pi * self.metadata.delta**4)
            * (1.0 - edge_distance / self.metadata.delta) ** 2
        )
        center_mass = np.bincount(
            edge_row,
            weights=edge_weight * self.metadata.volumes[edge_col],
            minlength=self.metadata.n_particles,
        )
        eta = edge_weight * self.metadata.volumes[edge_col] / center_mass[edge_row]
        lame_lambda = (
            self.metadata.E
            * self.metadata.nu
            / ((1.0 + self.metadata.nu) * (1.0 - 2.0 * self.metadata.nu))
        )
        shear_modulus = self.metadata.E / (2.0 * (1.0 + self.metadata.nu))
        for index, step in enumerate(accepted_steps, start=1):
            if not isinstance(step, AcceptedStep) or states[index] is not step.state:
                raise ValueError("ContinuationResult states and accepted steps are not paired")
            if (
                self.metadata.acceptance_mode == "strict"
                and step.acceptance_status != "strict_passed"
            ):
                raise ValueError(
                    "ContinuationResult strict metadata cannot pair with a "
                    "warning-accepted step"
                )
            # Re-run the accepted-step constructor so writer-time validation
            # also catches post-construction rebinding of a frozen dataclass.
            AcceptedStep(
                load_start=step.load_start,
                load_target=step.load_target,
                increment=step.increment,
                state=step.state,
                history_used=step.history_used,
                amor_driver=step.amor_driver,
                fixed_history_result=step.fixed_history_result,
                mechanics_result=step.mechanics_result,
                diagnostics=step.diagnostics,
                stagger_iterations=step.stagger_iterations,
                cutback_count=step.cutback_count,
                acceptance_status=step.acceptance_status,
                bypassed_gates=step.bypassed_gates,
                damage_before=step.damage_before,
                mechanics_before_result=step.mechanics_before_result,
            )
            mechanics_result = step.mechanics_result
            if (
                mechanics_result.mechanics_backend
                != self.metadata.mechanics_backend
                or step.mechanics_before_result.mechanics_backend
                != self.metadata.mechanics_backend
                or mechanics_result._backend_provenance
                is not _mechanics_backend_token(self.metadata.mechanics_backend)
                or step.mechanics_before_result._backend_provenance
                is not _mechanics_backend_token(self.metadata.mechanics_backend)
            ):
                raise ValueError(
                    "ContinuationResult mechanics backend does not match metadata"
                )
            authoritative_fixed = FixedDamageBondAssociatedMechanics(
                authoritative_intact, step.state.damage
            )
            _validate_mechanics_result(
                mechanics_result,
                authoritative_constraints,
                step.load_target,
                self.metadata.n_dofs,
                authoritative_fixed,
                self.metadata.config.cg_rtol,
                self.metadata.config.cg_atol,
                self.metadata.config.cg_maxiter,
            )
            evaluation_residual = _require_float64_array(
                mechanics_result.evaluation.residual,
                "ContinuationResult mechanics evaluation residual",
                shape=(self.metadata.n_dofs,),
            )
            _require_scale_aware_close(
                mechanics_result.evaluation.internal_force,
                -evaluation_residual,
                "ContinuationResult mechanics internal force",
            )
            _require_scale_aware_close(
                mechanics_result.free_residual,
                evaluation_residual[self.metadata.free_dofs],
                "ContinuationResult mechanics free residual extraction",
            )
            _require_scale_aware_close(
                mechanics_result.free_residual_norm,
                np.linalg.norm(evaluation_residual[self.metadata.free_dofs]),
                "ContinuationResult mechanics free residual norm",
            )
            residual_matrix = evaluation_residual.reshape(self.metadata.n_particles, 2)
            _require_scale_aware_close(
                mechanics_result.top_reaction,
                np.sum(residual_matrix[self.metadata.top_mask, 1]),
                "ContinuationResult mechanics top reaction",
            )
            _require_scale_aware_close(
                mechanics_result.bottom_reaction,
                np.sum(residual_matrix[self.metadata.bottom_mask, 1]),
                "ContinuationResult mechanics bottom reaction",
            )
            _require_scale_aware_close(
                mechanics_result.anchor_reaction,
                evaluation_residual[2 * self.metadata.anchor_particle],
                "ContinuationResult mechanics anchor reaction",
            )
            _require_scale_aware_close(
                mechanics_result.total_force,
                np.sum(residual_matrix, axis=0),
                "ContinuationResult mechanics total force",
            )
            expected_moment = np.sum(
                self.metadata.coordinates[:, 0] * residual_matrix[:, 1]
                - self.metadata.coordinates[:, 1] * residual_matrix[:, 0]
            )
            _require_scale_aware_close(
                mechanics_result.total_moment,
                expected_moment,
                "ContinuationResult mechanics total moment",
            )
            self.metadata.validate_state(states[index])
            if states[index].u_bar <= states[index - 1].u_bar:
                raise ValueError("ContinuationResult state loads must increase strictly")
            if step.load_start != states[index - 1].u_bar:
                raise ValueError("ContinuationResult accepted load start is not previous state load")
            if step.load_target != states[index].u_bar:
                raise ValueError("ContinuationResult accepted load target is not state load")
            previous_state = states[index - 1]
            expected_amor = compute_amor_history_from_edge_strain(
                step.mechanics_result.evaluation.edge_strain,
                edge_row,
                eta,
                lame_lambda,
                shear_modulus,
                previous_state.history,
            )
            for name in (
                "edge_positive_energy",
                "particle_driver",
                "previous_history",
                "trial_history",
            ):
                if not np.array_equal(
                    getattr(step.amor_driver, name), getattr(expected_amor, name)
                ):
                    raise ValueError(
                        f"ContinuationResult Amor {name.replace('_', ' ')} does not "
                        "match the accepted mechanics/preceding state"
                    )
            # Authoritative final-sweep replay.  Rebuild the before-sweep
            # fixed-damage view from ``damage_before``, validate the stored
            # mechanics-before result against it, replay ``H_used`` from the
            # mechanics-before displacement and the preceding committed
            # history, and recompute the complete final-sweep gate summaries.
            authoritative_before_fixed = FixedDamageBondAssociatedMechanics(
                authoritative_intact, step.damage_before
            )
            _validate_mechanics_result(
                step.mechanics_before_result,
                authoritative_constraints,
                step.load_target,
                self.metadata.n_dofs,
                authoritative_before_fixed,
                self.metadata.config.cg_rtol,
                self.metadata.config.cg_atol,
                self.metadata.config.cg_maxiter,
            )
            if step.stagger_iterations == 1 and not np.array_equal(
                step.damage_before, previous_state.damage
            ):
                raise ValueError(
                    "ContinuationResult single-sweep damage_before must equal "
                    "the preceding committed damage"
                )
            expected_amor_before = compute_amor_history_from_edge_strain(
                step.mechanics_before_result.evaluation.edge_strain,
                edge_row,
                eta,
                lame_lambda,
                shear_modulus,
                previous_state.history,
            )
            if not np.array_equal(
                step.history_used, expected_amor_before.trial_history
            ):
                raise ValueError(
                    "ContinuationResult H_used does not match the mechanics-before "
                    "displacement and preceding committed history"
                )
            geometry_length = max(
                _max_abs(self.metadata.coordinates),
                _max_abs(np.ptp(self.metadata.coordinates, axis=0)),
            )
            replayed_diagnostics = _full_stagger_diagnostics(
                self.metadata.config,
                previous_state,
                step.damage_before,
                states[index].damage,
                step.history_used,
                states[index].history,
                step.mechanics_before_result,
                step.mechanics_result,
                damage_matrix,
                self.metadata.volumes,
                self.metadata.Gc,
                self.metadata.l0,
                geometry_length,
                step.fixed_history_result,
            )
            _require_diagnostics_replay_match(
                step.diagnostics,
                replayed_diagnostics,
                "ContinuationResult",
            )
            if step.acceptance_status == "warning_accepted" and (
                _failed_warning_gates(replayed_diagnostics) != step.bypassed_gates
            ):
                raise ValueError(
                    "ContinuationResult bypassed gates do not match the "
                    "authoritative final-sweep replay"
                )
            damage = states[index].damage
            history = states[index].history
            committed_terms = (
                np.asarray(
                    (self.metadata.Gc / self.metadata.l0) * damage,
                    dtype=np.float64,
                ),
                np.asarray(
                    self.metadata.Gc
                    * self.metadata.l0
                    * (damage_matrix @ damage),
                    dtype=np.float64,
                ),
                np.asarray(2.0 * history * damage, dtype=np.float64),
                np.asarray(-2.0 * history, dtype=np.float64),
            )
            committed_residual = (
                committed_terms[0]
                + committed_terms[1]
                + committed_terms[2]
                + committed_terms[3]
            )
            _require_residual_reconstruction_close(
                step.diagnostics.rho_next,
                committed_residual,
                "ContinuationResult committed residual",
                committed_terms,
            )
            _require_residual_reconstruction_close(
                step.diagnostics.rho_next_volume,
                self.metadata.volumes * committed_residual,
                "ContinuationResult volume residual",
                tuple(self.metadata.volumes * term for term in committed_terms),
            )
            fixed = step.fixed_history_result
            fixed_terms = (
                np.asarray(
                    (self.metadata.Gc / self.metadata.l0) * fixed.damage,
                    dtype=np.float64,
                ),
                np.asarray(
                    self.metadata.Gc
                    * self.metadata.l0
                    * (damage_matrix @ fixed.damage),
                    dtype=np.float64,
                ),
                np.asarray(2.0 * fixed.history * fixed.damage, dtype=np.float64),
                np.asarray(-2.0 * fixed.history, dtype=np.float64),
            )
            fixed_residual = (
                fixed_terms[0]
                + fixed_terms[1]
                + fixed_terms[2]
                + fixed_terms[3]
            )
            _require_residual_reconstruction_close(
                fixed.strong_residual,
                fixed_residual,
                "ContinuationResult fixed-history strong residual",
                fixed_terms,
            )
            _require_residual_reconstruction_close(
                fixed.euclidean_residual,
                self.metadata.volumes * fixed_residual,
                "ContinuationResult fixed-history Euclidean residual",
                tuple(self.metadata.volumes * term for term in fixed_terms),
            )
            crack_local_energy = float(
                np.sum(
                    self.metadata.volumes
                    * (self.metadata.Gc / (2.0 * self.metadata.l0))
                    * fixed.damage**2
                )
            )
            history_energy = float(
                np.sum(
                    self.metadata.volumes
                    * fixed.history
                    * (1.0 - fixed.damage) ** 2
                )
            )
            directed_weight = -np.asarray(
                damage_matrix[edge_row, edge_col]
            ).reshape(-1)
            graph_energy = float(
                self.metadata.Gc
                * self.metadata.l0
                / 4.0
                * np.sum(
                    self.metadata.volumes[edge_row]
                    * directed_weight
                    * (fixed.damage[edge_col] - fixed.damage[edge_row]) ** 2
                )
            )
            for actual, expected, name in (
                (fixed.crack_local_energy, crack_local_energy, "crack energy"),
                (fixed.history_energy, history_energy, "history energy"),
                (fixed.graph_energy, graph_energy, "graph energy"),
                (
                    fixed.total_energy,
                    crack_local_energy + history_energy + graph_energy,
                    "total energy",
                ),
            ):
                _require_scale_aware_close(
                    actual,
                    expected,
                    f"ContinuationResult fixed-history {name}",
                )
            strong_matrix = diags(
                self.metadata.Gc / self.metadata.l0 + 2.0 * fixed.history,
                format="csr",
            ) + self.metadata.Gc * self.metadata.l0 * damage_matrix
            system = (diags(self.metadata.volumes, format="csr") @ strong_matrix).tocsr()
            skew = system - system.T
            symmetry_defect = (
                float(np.linalg.norm(skew.data, ord=np.inf)) if skew.nnz else 0.0
            )
            system_norm = (
                float(np.linalg.norm(system.data, ord=np.inf)) if system.nnz else 0.0
            )
            _require_scale_aware_close(
                fixed.symmetry_defect,
                symmetry_defect,
                "ContinuationResult fixed-history symmetry defect",
            )
            _require_scale_aware_close(
                fixed.symmetry_relative_defect,
                symmetry_defect / max(system_norm, 1.0e-300),
                "ContinuationResult fixed-history relative symmetry defect",
            )

        if any(not isinstance(attempt, AttemptRecord) for attempt in attempts):
            raise TypeError("ContinuationResult.attempts must contain AttemptRecord values")
        accepted_attempts = [attempt for attempt in attempts if attempt.accepted]
        if len(accepted_attempts) != len(accepted_steps):
            raise ValueError(
                "ContinuationResult accepted-attempt count does not match accepted steps"
            )
        if states[-1].u_bar != self.metadata.config.load_final:
            raise ValueError("ContinuationResult final state does not reach load_final")

        for attempt in attempts:
            if not isinstance(attempt, AttemptRecord):
                raise TypeError("ContinuationResult.attempts must contain AttemptRecord values")
            # AttemptRecord is frozen but its fields can still be rebound with
            # object.__setattr__; reconstructing it makes writer validation
            # reject such corruption.
            AttemptRecord(
                cutback=attempt.cutback,
                load_start=attempt.load_start,
                load_target=attempt.load_target,
                accepted=attempt.accepted,
                failure_stage=attempt.failure_stage,
                failure_message=attempt.failure_message,
                committed_before_digest=attempt.committed_before_digest,
                committed_after_digest=attempt.committed_after_digest,
                diagnostics=attempt.diagnostics,
                stagger_iterations=attempt.stagger_iterations,
                exception_type=attempt.exception_type,
                acceptance_status=attempt.acceptance_status,
                sweep_complete=attempt.sweep_complete,
                bypassed_gates=attempt.bypassed_gates,
            )

        current_state = states[0]
        expected_cutback = 0
        accepted_index = 0
        for attempt in attempts:
            if attempt.load_start != current_state.u_bar:
                raise ValueError(
                    "ContinuationResult attempt load_start does not match the last "
                    "committed state"
                )
            if (
                attempt.load_start < self.metadata.config.load_start
                or attempt.load_target > self.metadata.config.load_final
                or attempt.load_target <= attempt.load_start
            ):
                raise ValueError("ContinuationResult attempt load lies outside its valid bounds")
            if attempt.cutback != expected_cutback:
                raise ValueError(
                    "ContinuationResult attempt cutback count is not the exact "
                    "rollback sequence"
                )
            if attempt.cutback > self.metadata.config.max_cutbacks:
                raise ValueError("ContinuationResult attempt exceeds max_cutbacks")
            if (
                attempt.stagger_iterations <= 0
                or attempt.stagger_iterations
                > self.metadata.config.max_stagger_iterations
            ):
                raise ValueError(
                    "ContinuationResult attempt stagger_iterations is outside "
                    "the configured bound"
                )
            expected_digest = _state_digest(current_state)
            if attempt.committed_before_digest != expected_digest:
                raise ValueError(
                    "ContinuationResult attempt committed-before digest does not "
                    "match the current state"
                )
            if attempt.accepted:
                if accepted_index >= len(accepted_steps):
                    raise ValueError("ContinuationResult has an extra accepted attempt")
                step = accepted_steps[accepted_index]
                next_state = states[accepted_index + 1]
                if attempt.load_start != step.load_start or attempt.load_target != step.load_target:
                    raise ValueError(
                        "ContinuationResult accepted attempt load does not match its step"
                    )
                if attempt.cutback != step.cutback_count:
                    raise ValueError(
                        "ContinuationResult accepted attempt cutback does not match its step"
                    )
                if attempt.stagger_iterations != step.stagger_iterations:
                    raise ValueError(
                        "ContinuationResult accepted attempt stagger iterations do not "
                        "match its step"
                    )
                if attempt.acceptance_status != step.acceptance_status:
                    raise ValueError(
                        "ContinuationResult accepted attempt acceptance status does "
                        "not match its step"
                    )
                if attempt.bypassed_gates != step.bypassed_gates:
                    raise ValueError(
                        "ContinuationResult accepted attempt bypassed gates do not "
                        "match its step"
                    )
                if not attempt.sweep_complete or not step.diagnostics.complete_sweep:
                    raise ValueError(
                        "ContinuationResult accepted attempt must record a complete "
                        "stagger sweep"
                    )
                if not _diagnostics_match(attempt.diagnostics, step.diagnostics):
                    raise ValueError(
                        "ContinuationResult accepted attempt diagnostics do not match its step"
                    )
                if attempt.committed_after_digest != _state_digest(next_state):
                    raise ValueError(
                        "ContinuationResult accepted attempt committed-after digest "
                        "does not match its accepted state"
                    )
                current_state = next_state
                accepted_index += 1
                expected_cutback = 0
            else:
                if attempt.committed_after_digest != expected_digest:
                    raise ValueError(
                        "ContinuationResult rejected attempt changed committed state"
                    )
                expected_cutback += 1

        if accepted_index != len(accepted_steps) or current_state is not states[-1]:
            raise ValueError(
                "ContinuationResult attempts do not reach the final accepted state"
            )

    @property
    def accepted_states(self) -> tuple[CommittedState, ...]:
        return self.states[1:]

    @property
    def committed_states(self) -> tuple[CommittedState, ...]:
        return self.states

    @property
    def final_state(self) -> CommittedState:
        return self.states[-1]

    @property
    def cutbacks(self) -> tuple[int, ...]:
        return tuple(step.cutback_count for step in self.accepted_steps)


@dataclass(frozen=True)
class ContinuationFailure(RuntimeError):
    """Typed strict-continuation failure with exact rollback audit."""

    message: str
    last_committed_state: CommittedState
    attempts: tuple[AttemptRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("ContinuationFailure message must be a non-empty string")
        if not isinstance(self.last_committed_state, CommittedState):
            raise TypeError("ContinuationFailure.last_committed_state must be CommittedState")
        attempt_tuple = tuple(self.attempts)
        if any(not isinstance(attempt, AttemptRecord) for attempt in attempt_tuple):
            raise TypeError("ContinuationFailure.attempts must contain AttemptRecord values")
        RuntimeError.__init__(self, self.message)
        object.__setattr__(self, "message", self.message)
        object.__setattr__(self, "last_committed_state", self.last_committed_state)
        object.__setattr__(self, "attempts", attempt_tuple)

    @property
    def state(self) -> CommittedState:
        return self.last_committed_state

    @property
    def last_committed(self) -> CommittedState:
        return self.last_committed_state

    @property
    def attempt_audit(self) -> tuple[AttemptRecord, ...]:
        return self.attempts


def _state_digest(state: CommittedState) -> str:
    """Hash dtype, shape, and C-order bytes of every committed field."""

    digest = hashlib.sha256()
    for name, values in (
        ("u_bar", np.asarray(state.u_bar, dtype=np.float64)),
        ("displacement", state.displacement),
        ("damage", state.damage),
        ("history", state.history),
    ):
        array = np.asarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _constraints_match(
    actual: HardBoundaryConstraints,
    expected: HardBoundaryConstraints,
) -> bool:
    if not isinstance(actual, HardBoundaryConstraints):
        return False
    if actual.anchor_particle != expected.anchor_particle:
        return False
    for name in (
        "bottom_mask",
        "top_mask",
        "constrained_dofs",
        "free_dofs",
        "q_zero",
        "a_top",
        "a_bottom",
        "a_anchor",
    ):
        if not np.array_equal(getattr(actual, name), getattr(expected, name)):
            return False
    return True


def _validate_continuation_setup(
    operator: ExplicitVolumeSparseD,
    intact_mechanics: BondAssociatedMechanics,
    constraints: HardBoundaryConstraints,
    Gc: Any,
    l0: Any,
    config: StrictContinuationConfig,
) -> tuple[Any, np.ndarray, np.ndarray, float, float]:
    """Validate the complete shared-data contract before archiving state."""

    if not isinstance(config, StrictContinuationConfig):
        raise TypeError("config must be a StrictContinuationConfig instance")
    gc_value = _validate_material_parameter(Gc, "Gc")
    l0_value = _validate_material_parameter(l0, "l0")
    if not isinstance(operator, ExplicitVolumeSparseD):
        raise TypeError("operator must be an ExplicitVolumeSparseD instance")
    if type(intact_mechanics) is not BondAssociatedMechanics:
        raise TypeError("intact_mechanics must be a BondAssociatedMechanics instance")
    if not isinstance(constraints, HardBoundaryConstraints):
        raise TypeError("constraints must be a HardBoundaryConstraints instance")

    scipy_operator, operator_volumes, operator_edges, operator_coordinates = _validate_operator(
        operator
    )
    if operator_coordinates.shape[1] != 2:
        raise ValueError("strict fracture2d continuation requires a two-dimensional operator")
    if operator.kernel != "cubic":
        raise ValueError("strict fracture2d continuation requires the cubic operator kernel")
    if not np.array_equal(
        operator_edges,
        operator_edges[np.lexsort((operator_edges[:, 1], operator_edges[:, 0]))],
    ):
        raise ValueError("operator directed_edges must be lexicographically sorted")

    mechanics_coordinates = _require_float64_array(
        intact_mechanics.coordinates, "intact_mechanics.coordinates"
    )
    mechanics_volumes = _require_float64_array(
        intact_mechanics.volumes, "intact_mechanics.volumes"
    )
    mechanics_edges = np.asarray(intact_mechanics.directed_edges)
    if mechanics_coordinates.shape != operator_coordinates.shape:
        raise ValueError("operator and mechanics coordinates have different shapes")
    if not np.array_equal(mechanics_coordinates, operator_coordinates):
        raise ValueError("operator and mechanics coordinates must be exactly identical")
    if not np.array_equal(mechanics_volumes, operator_volumes):
        raise ValueError("operator and mechanics volumes must be exactly identical")
    if mechanics_edges.dtype != np.dtype(np.int64) or mechanics_edges.ndim != 2:
        raise ValueError("intact_mechanics.directed_edges must be sorted int64 pairs")
    if not np.array_equal(mechanics_edges, operator_edges):
        raise ValueError("operator and mechanics directed_edges must be exactly identical")
    if operator.shape != (intact_mechanics.n_particles, intact_mechanics.n_particles):
        raise ValueError("operator and mechanics particle counts are incompatible")
    if operator.delta != intact_mechanics.delta:
        raise ValueError("operator and mechanics delta must be exactly identical")
    if intact_mechanics.n_particles != len(operator_coordinates):
        raise ValueError("operator and mechanics particle counts are incompatible")

    if constraints.n_particles != intact_mechanics.n_particles:
        raise ValueError("constraints have a different particle count from mechanics")
    expected_constraints = intact_mechanics.build_constraints(
        constraints.bottom_mask,
        constraints.top_mask,
        constraints.anchor_particle,
    )
    if not _constraints_match(constraints, expected_constraints):
        raise ValueError(
            "constraints are not identity-compatible with the supplied mechanics "
            "boundary masks and anchor"
        )

    geometry_length = max(
        _max_abs(operator_coordinates),
        _max_abs(np.ptp(operator_coordinates, axis=0)),
    )
    if not np.isfinite(geometry_length) or geometry_length <= 0.0:
        raise ValueError("mechanics coordinates must define a positive geometry length")
    return scipy_operator, operator_volumes, operator_edges, gc_value, l0_value


def _build_continuation_metadata(
    operator: ExplicitVolumeSparseD,
    intact_mechanics: BondAssociatedMechanics,
    constraints: HardBoundaryConstraints,
    Gc: float,
    l0: float,
    config: StrictContinuationConfig,
    mechanics_backend: str,
    acceptance_mode: str,
) -> ContinuationMetadata:
    """Capture the exact static contract after setup validation succeeds."""

    return ContinuationMetadata(
        coordinates=np.asarray(operator.coordinates, dtype=np.float64),
        volumes=np.asarray(operator.volumes, dtype=np.float64),
        directed_edges=np.asarray(operator.directed_edges, dtype=np.int64),
        bottom_mask=np.asarray(constraints.bottom_mask, dtype=bool),
        top_mask=np.asarray(constraints.top_mask, dtype=bool),
        anchor_particle=constraints.anchor_particle,
        constrained_dofs=np.asarray(constraints.constrained_dofs, dtype=np.int64),
        free_dofs=np.asarray(constraints.free_dofs, dtype=np.int64),
        youngs_modulus=intact_mechanics.youngs_modulus,
        poisson_ratio=intact_mechanics.poisson_ratio,
        k_res=intact_mechanics.k_res,
        rank_rtol=intact_mechanics.rank_rtol,
        Gc=Gc,
        l0=l0,
        delta=operator.delta,
        kernel=operator.kernel,
        plane_strain=True,
        stabilization="none",
        mechanics_backend=mechanics_backend,
        acceptance_mode=acceptance_mode,
        dtype=_FLOAT64,
        config=config,
    )


_MECHANICS_VALIDATION_EPS = 512.0 * np.finfo(np.float64).eps
_RESIDUAL_RECONSTRUCTION_EPS = 512.0 * np.finfo(np.float64).eps


def _require_scale_aware_close(
    actual: Any,
    expected: Any,
    name: str,
) -> None:
    """Compare float64 mechanics quantities with an explicit scale-aware floor."""

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"{name} has shape {actual_array.shape}, expected {expected_array.shape}"
        )
    if actual_array.dtype != expected_array.dtype:
        raise ValueError(
            f"{name} has dtype {actual_array.dtype}, expected {expected_array.dtype}"
        )
    if not np.all(np.isfinite(actual_array)) or not np.all(
        np.isfinite(expected_array)
    ):
        raise ValueError(f"{name} must contain only finite values")
    scale = max(_max_abs(actual_array), _max_abs(expected_array), 1.0)
    tolerance = _MECHANICS_VALIDATION_EPS * scale
    if not np.allclose(actual_array, expected_array, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"{name} is inconsistent with the authoritative mechanics value: "
            f"defect={_max_abs(actual_array - expected_array)}, "
            f"tolerance={tolerance}"
        )


def _require_residual_reconstruction_close(
    actual: Any,
    expected: Any,
    name: str,
    terms: tuple[Any, ...],
) -> None:
    """Compare a cancellation-prone residual using its constituent-term scale.

    The residual is formed as a sum of signed terms, so its magnitude can be
    much smaller than the quantities rounded while evaluating that sum.  Bound
    the reconstruction defect componentwise by 512 float64 epsilons times the
    sum of the absolute constituent terms, without the unit scale floor used by
    the generic mechanics comparator.
    """

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"{name} has shape {actual_array.shape}, expected {expected_array.shape}"
        )
    for label, array in (("actual", actual_array), ("expected", expected_array)):
        if array.dtype != _FLOAT64:
            raise ValueError(f"{name} {label} must use exact float64 dtype, got {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} {label} must contain only finite values")

    if not terms:
        raise ValueError(f"{name} requires at least one finite constituent term")
    term_arrays: list[np.ndarray] = []
    for index, term in enumerate(terms):
        term_array = np.asarray(term)
        if term_array.shape != expected_array.shape:
            raise ValueError(
                f"{name} term {index} has shape {term_array.shape}, "
                f"expected {expected_array.shape}"
            )
        if term_array.dtype != _FLOAT64:
            raise ValueError(
                f"{name} term {index} must use exact float64 dtype, "
                f"got {term_array.dtype}"
            )
        if not np.all(np.isfinite(term_array)):
            raise ValueError(f"{name} term {index} must contain only finite values")
        term_arrays.append(term_array)

    term_scale = np.zeros(expected_array.shape, dtype=np.float64)
    for term_array in term_arrays:
        term_scale += np.abs(term_array)
    if not np.all(np.isfinite(term_scale)):
        raise ValueError(f"{name} constituent-term scale must be finite")
    defect = np.abs(actual_array - expected_array)
    if not np.all(np.isfinite(defect)):
        raise ValueError(f"{name} reconstruction defect must be finite")
    tolerance = _RESIDUAL_RECONSTRUCTION_EPS * term_scale
    if not np.all(defect <= tolerance):
        raise ValueError(
            f"{name} exceeds the term-scaled float64 reconstruction bound: "
            f"defect={_max_abs(defect)}, tolerance={_max_abs(tolerance)}"
        )


def _validate_mechanics_result(
    result: Any,
    constraints: HardBoundaryConstraints,
    target_load: float,
    n_dofs: int,
    authoritative_mechanics: BondAssociatedMechanics,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
) -> Any:
    """Validate a mechanics result against a fresh authoritative evaluation.

    The solver result contains useful diagnostics, but those diagnostics are
    not an independent acceptance source.  Recompute the residual through the
    supplied fixed-damage mechanics API, extract the free DOFs exactly, and
    derive all norms, reactions, force, and moment from that residual.
    """

    if not isinstance(authoritative_mechanics, BondAssociatedMechanics):
        raise TypeError("authoritative_mechanics must be a BondAssociatedMechanics instance")
    if authoritative_mechanics.n_dofs != n_dofs:
        raise ValueError("authoritative mechanics has the wrong DOF count")

    required = (
        "displacement",
        "evaluation",
        "constraints",
        "u_bar",
        "free_residual",
        "free_residual_norm",
        "free_rhs_norm",
        "free_residual_relative_norm",
        "initial_free_residual_norm",
        "cg_convergence_limit",
        "initial_guess_supplied",
        "cg_initial_guess_used",
        "early_exit",
        "cg_iterations",
        "cg_info",
        "top_reaction",
        "bottom_reaction",
        "anchor_reaction",
        "total_force",
        "total_moment",
    )
    for name in required:
        if not hasattr(result, name):
            raise ValueError(f"fixed mechanics result is missing required field {name!r}")
    displacement = _require_float64_array(
        result.displacement, "fixed mechanics displacement", shape=(n_dofs,)
    )
    free_residual = _require_float64_array(
        result.free_residual,
        "fixed mechanics free residual",
        shape=(len(constraints.free_dofs),),
    )
    total_force = _require_float64_array(
        result.total_force, "fixed mechanics total force", shape=(2,)
    )
    result_load = _scalar_float(result.u_bar, "fixed mechanics u_bar")
    if result_load != target_load:
        raise ValueError(
            "fixed mechanics result load does not equal the requested target: "
            f"{result_load!r} versus {target_load!r}"
        )
    if not _constraints_match(result.constraints, constraints):
        raise ValueError("fixed mechanics result constraints are incompatible with the trial")
    if not isinstance(result.evaluation, MechanicsEvaluation):
        raise TypeError("fixed mechanics result evaluation must be MechanicsEvaluation")
    for name in (
        "free_residual_norm",
        "free_residual_relative_norm",
        "initial_free_residual_norm",
        "cg_convergence_limit",
        "top_reaction",
        "bottom_reaction",
        "anchor_reaction",
        "total_moment",
    ):
        value = _scalar_float(getattr(result, name), f"fixed mechanics {name}")
        if not np.isfinite(value):
            raise FloatingPointError(f"fixed mechanics {name} is non-finite")
    raw_cg_iterations = np.asarray(result.cg_iterations)
    if (
        raw_cg_iterations.ndim != 0
        or not np.issubdtype(raw_cg_iterations.dtype, np.integer)
        or isinstance(result.cg_iterations, (bool, np.bool_))
        or int(raw_cg_iterations) < 0
    ):
        raise ValueError("fixed mechanics cg_iterations must be a non-negative integer")
    for name in (
        "initial_guess_supplied",
        "cg_initial_guess_used",
        "early_exit",
    ):
        if not isinstance(getattr(result, name), (bool, np.bool_)):
            raise ValueError(f"fixed mechanics {name} must be boolean")
    if result.early_exit and (
        not result.initial_guess_supplied or int(raw_cg_iterations) != 0
    ):
        raise ValueError(
            "fixed mechanics early_exit requires an initial guess and zero iterations"
        )
    raw_cg_info = np.asarray(result.cg_info)
    if raw_cg_info.ndim != 0 or not np.issubdtype(raw_cg_info.dtype, np.integer):
        raise ValueError("fixed mechanics cg_info must be an integer")
    if int(raw_cg_info) != 0:
        raise RuntimeError(f"fixed mechanics solve returned cg_info={int(raw_cg_info)}")
    if int(raw_cg_iterations) > cg_maxiter:
        raise ValueError("fixed mechanics cg_iterations exceeds cg_maxiter")
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(free_residual)):
        raise FloatingPointError("fixed mechanics result contains non-finite arrays")

    prescribed = constraints.prescribed_vector(target_load)
    if not np.array_equal(
        displacement[constraints.constrained_dofs],
        prescribed[constraints.constrained_dofs],
    ):
        raise ValueError("fixed mechanics displacement does not satisfy constraints")

    authoritative_evaluation = authoritative_mechanics.evaluate(displacement)
    authoritative_residual = _require_float64_array(
        authoritative_mechanics.residual(displacement),
        "authoritative mechanics residual",
        shape=(n_dofs,),
    )
    _require_scale_aware_close(
        authoritative_evaluation.residual,
        authoritative_residual,
        "authoritative mechanics evaluation residual",
    )

    evaluation_array_names = (
        "residual",
        "internal_force",
        "edge_deformation_gradient",
        "edge_strain",
        "edge_stress",
        "particle_strain",
        "particle_stress",
        "edge_effective_stress",
        "particle_effective_stress",
        "edge_energy_density",
    )
    for name in evaluation_array_names:
        actual = _require_float64_array(
            getattr(result.evaluation, name),
            f"fixed mechanics evaluation {name}",
        )
        expected = _require_float64_array(
            getattr(authoritative_evaluation, name),
            f"authoritative mechanics evaluation {name}",
        )
        _require_scale_aware_close(
            actual,
            expected,
            f"fixed mechanics evaluation {name}",
        )
    _require_scale_aware_close(
        result.evaluation.energy,
        authoritative_evaluation.energy,
        "fixed mechanics evaluation energy",
    )

    expected_free_residual = authoritative_residual[constraints.free_dofs]
    _require_scale_aware_close(
        free_residual,
        expected_free_residual,
        "fixed mechanics free residual extraction",
    )
    expected_free_norm = float(np.linalg.norm(expected_free_residual))
    prescribed_residual = _require_float64_array(
        authoritative_mechanics.residual(prescribed),
        "authoritative prescribed mechanics residual",
        shape=(n_dofs,),
    )
    expected_rhs_norm = float(
        np.linalg.norm(-prescribed_residual[constraints.free_dofs])
    )
    expected_relative_norm = expected_free_norm / max(expected_rhs_norm, 1.0e-300)
    expected_convergence_limit = max(
        _scalar_float(cg_atol, "fixed mechanics cg_atol"),
        _scalar_float(cg_rtol, "fixed mechanics cg_rtol") * expected_rhs_norm,
    )
    _require_scale_aware_close(
        result.free_residual_norm,
        expected_free_norm,
        "fixed mechanics free residual norm",
    )
    _require_scale_aware_close(
        result.free_rhs_norm,
        expected_rhs_norm,
        "fixed mechanics free right-hand-side norm",
    )
    _require_scale_aware_close(
        result.cg_convergence_limit,
        expected_convergence_limit,
        "fixed mechanics CG convergence limit",
    )
    if result.initial_free_residual_norm < 0.0:
        raise ValueError("fixed mechanics initial free residual norm must be non-negative")
    if expected_free_norm >= expected_convergence_limit:
        raise ValueError(
            "fixed mechanics free residual does not satisfy the configured CG tolerance"
        )
    _require_scale_aware_close(
        result.free_residual_relative_norm,
        expected_relative_norm,
        "fixed mechanics free residual relative norm",
    )

    top_y_dofs = 2 * np.flatnonzero(constraints.top_mask) + 1
    bottom_y_dofs = 2 * np.flatnonzero(constraints.bottom_mask) + 1
    expected_top_reaction = float(np.sum(authoritative_residual[top_y_dofs]))
    expected_bottom_reaction = float(np.sum(authoritative_residual[bottom_y_dofs]))
    expected_anchor_reaction = float(authoritative_residual[constraints.anchor_dof])
    residual_matrix = authoritative_residual.reshape(
        authoritative_mechanics.n_particles, 2
    )
    expected_total_force = np.sum(residual_matrix, axis=0)
    coordinates = _require_float64_array(
        authoritative_mechanics.coordinates,
        "authoritative mechanics coordinates",
        shape=(authoritative_mechanics.n_particles, 2),
    )
    expected_total_moment = float(
        np.sum(
            coordinates[:, 0] * residual_matrix[:, 1]
            - coordinates[:, 1] * residual_matrix[:, 0]
        )
    )
    _require_scale_aware_close(
        result.top_reaction,
        expected_top_reaction,
        "fixed mechanics top reaction",
    )
    _require_scale_aware_close(
        result.bottom_reaction,
        expected_bottom_reaction,
        "fixed mechanics bottom reaction",
    )
    _require_scale_aware_close(
        result.anchor_reaction,
        expected_anchor_reaction,
        "fixed mechanics anchor reaction",
    )
    _require_scale_aware_close(
        total_force,
        expected_total_force,
        "fixed mechanics total force",
    )
    _require_scale_aware_close(
        result.total_moment,
        expected_total_moment,
        "fixed mechanics total moment",
    )
    return result


def _damage_residuals(
    damage_matrix: Any,
    volumes: np.ndarray,
    Gc: float,
    l0: float,
    damage: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    Dd = np.asarray(damage_matrix @ damage, dtype=np.float64)
    if Dd.shape != damage.shape or not np.all(np.isfinite(Dd)):
        raise FloatingPointError("direct damage operator evaluation is non-finite")
    local_term = (Gc / l0) * damage
    graph_term = Gc * l0 * Dd
    rho_used = np.asarray(
        local_term + graph_term + 2.0 * H_used * damage - 2.0 * H_used,
        dtype=np.float64,
    )
    rho_next = np.asarray(
        local_term + graph_term + 2.0 * H_next * damage - 2.0 * H_next,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(rho_used)) or not np.all(np.isfinite(rho_next)):
        raise FloatingPointError("direct damage residual is non-finite")
    rho_used_volume = np.asarray(volumes * rho_used, dtype=np.float64)
    rho_next_volume = np.asarray(volumes * rho_next, dtype=np.float64)
    terms = [
        local_term,
        graph_term,
        2.0 * H_used * damage,
        2.0 * H_used,
        2.0 * H_next * damage,
        2.0 * H_next,
    ]
    term_scale = max((_max_abs(term) for term in terms), default=0.0)
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise FloatingPointError("damage residual term scale is non-finite")
    return rho_used, rho_next, rho_used_volume, rho_next_volume, term_scale


def _failure_diagnostics(
    config: StrictContinuationConfig,
    committed: CommittedState,
    d_iter: np.ndarray,
    target_load: float,
    damage_matrix: Any,
    volumes: np.ndarray,
    Gc: float,
    l0: float,
    geometry_length: float,
    reason: str,
    d_next: np.ndarray | None = None,
    H_used: np.ndarray | None = None,
    mechanics_before: Any | None = None,
    mechanics_after: Any | None = None,
) -> StaggerDiagnostics:
    n_particles = committed.damage.size
    zeros = np.zeros(n_particles, dtype=np.float64)
    zero_force = np.zeros(2, dtype=np.float64)
    d_values = committed.damage if d_next is None else np.asarray(d_next, dtype=np.float64)
    h_used = zeros if H_used is None else np.asarray(H_used, dtype=np.float64)
    rho_used = zeros.copy()
    if d_next is not None and H_used is not None:
        try:
            rho_used, _, rho_used_volume, _, rho_scale = _damage_residuals(
                damage_matrix,
                volumes,
                Gc,
                l0,
                d_values,
                h_used,
                h_used,
            )
        except FloatingPointError:
            rho_used = zeros.copy()
            rho_used_volume = zeros.copy()
            rho_scale = 0.0
    else:
        rho_used_volume = zeros.copy()
        rho_scale = 0.0

    before_u = (
        np.asarray(mechanics_before.displacement, dtype=np.float64)
        if mechanics_before is not None
        else None
    )
    after_u = (
        np.asarray(mechanics_after.displacement, dtype=np.float64)
        if mechanics_after is not None
        else None
    )
    if before_u is not None and after_u is not None:
        e_u = _max_abs(after_u - before_u)
        u_scale = max(_max_abs(before_u), _max_abs(after_u), abs(target_load))
    else:
        e_u = np.inf
        u_scale = 0.0
    e_d = _max_abs(d_values - d_iter) if d_next is not None else np.inf
    d_scale = max(_max_abs(d_values), _max_abs(d_iter))
    if H_used is not None:
        e_H = np.inf
        H_scale = max(_max_abs(committed.history), _max_abs(h_used))
    else:
        e_H = np.inf
        H_scale = _max_abs(committed.history)
    if mechanics_before is not None and mechanics_after is not None:
        reaction_before = float(mechanics_before.top_reaction)
        reaction_after = float(mechanics_after.top_reaction)
        e_R = abs(reaction_after - reaction_before)
        R_scale = max(
            abs(float(mechanics_before.top_reaction)),
            abs(float(mechanics_after.top_reaction)),
            abs(float(mechanics_before.bottom_reaction)),
            abs(float(mechanics_after.bottom_reaction)),
            abs(float(mechanics_before.anchor_reaction)),
            abs(float(mechanics_after.anchor_reaction)),
        )
        force_before = np.asarray(mechanics_before.total_force, dtype=np.float64)
        force_after = np.asarray(mechanics_after.total_force, dtype=np.float64)
        force_error = max(_max_abs(force_before), _max_abs(force_after))
        moment_before = float(mechanics_before.total_moment)
        moment_after = float(mechanics_after.total_moment)
        moment_error = max(abs(moment_before), abs(moment_after))
        force_scale = R_scale
        moment_scale = force_scale * geometry_length
        before_free_norm = float(mechanics_before.free_residual_norm)
        after_free_norm = float(mechanics_after.free_residual_norm)
        before_free_relative = float(mechanics_before.free_residual_relative_norm)
        after_free_relative = float(mechanics_after.free_residual_relative_norm)
        before_cg = int(mechanics_before.cg_info)
        after_cg = int(mechanics_after.cg_info)
    else:
        reaction_before = 0.0
        reaction_after = 0.0
        e_R = np.inf
        R_scale = 0.0
        force_before = zero_force
        force_after = zero_force
        force_error = np.inf
        force_scale = 0.0
        moment_before = 0.0
        moment_after = 0.0
        moment_error = np.inf
        moment_scale = 0.0
        before_free_norm = 0.0
        after_free_norm = 0.0
        before_free_relative = 0.0
        after_free_relative = 0.0
        before_cg = -1
        after_cg = -1

    damage_min = float(np.min(d_values))
    damage_max = float(np.max(d_values))
    healing_min = float(np.min(d_values - committed.damage))
    return StaggerDiagnostics(
        e_u=e_u,
        u_scale=u_scale,
        u_limit=config.displacement.limit(u_scale),
        u_pass=False,
        e_d=e_d,
        d_scale=d_scale,
        d_limit=config.damage.limit(d_scale),
        d_pass=False,
        e_H=e_H,
        H_scale=H_scale,
        H_limit=config.history.limit(H_scale),
        H_pass=False,
        e_R=e_R,
        R_scale=R_scale,
        R_limit=config.reaction.limit(R_scale),
        R_pass=False,
        rho_used=rho_used,
        rho_next=zeros.copy(),
        rho_used_volume=rho_used_volume,
        rho_next_volume=zeros.copy(),
        e_rho=np.inf,
        rho_scale=rho_scale,
        rho_limit=config.damage_residual.limit(rho_scale),
        rho_pass=False,
        damage_min=damage_min,
        damage_max=damage_max,
        damage_range_pass=(
            damage_min >= -config.damage_range_atol
            and damage_max <= 1.0 + config.damage_range_atol
        ),
        healing_min=healing_min,
        healing_pass=healing_min >= -config.healing_atol,
        history_min_gap=0.0,
        history_monotonic_pass=False,
        reaction_before=reaction_before,
        reaction_after=reaction_after,
        mechanics_before_free_residual_norm=before_free_norm,
        mechanics_after_free_residual_norm=after_free_norm,
        mechanics_before_free_residual_relative_norm=before_free_relative,
        mechanics_after_free_residual_relative_norm=after_free_relative,
        mechanics_before_cg_info=before_cg,
        mechanics_after_cg_info=after_cg,
        force_before=force_before,
        force_after=force_after,
        moment_before=moment_before,
        moment_after=moment_after,
        force_error=force_error,
        force_scale=force_scale,
        force_limit=config.force.limit(force_scale),
        force_pass=False,
        moment_error=moment_error,
        moment_scale=moment_scale,
        moment_limit=config.moment.limit(moment_scale),
        moment_pass=False,
        mechanics_pass=False,
        passed=False,
        complete_sweep=False,
        rejection_reason=reason,
    )


def _full_stagger_diagnostics(
    config: StrictContinuationConfig,
    committed: CommittedState,
    d_iter: np.ndarray,
    d_next: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
    mechanics_before: Any,
    mechanics_after: Any,
    damage_matrix: Any,
    volumes: np.ndarray,
    Gc: float,
    l0: float,
    geometry_length: float,
    fixed_history_result: FixedHistoryDamageResult,
) -> StaggerDiagnostics:
    _, rho_next, _, rho_next_volume, rho_scale = _damage_residuals(
        damage_matrix, volumes, Gc, l0, d_next, H_used, H_next
    )
    rho_used = np.asarray(fixed_history_result.strong_residual, dtype=np.float64)
    rho_used_volume = np.asarray(volumes * rho_used, dtype=np.float64)
    e_u = _max_abs(mechanics_after.displacement - mechanics_before.displacement)
    u_scale = max(
        _max_abs(mechanics_before.displacement),
        _max_abs(mechanics_after.displacement),
        abs(float(mechanics_after.u_bar)),
    )
    e_d = _max_abs(d_next - d_iter)
    d_scale = max(_max_abs(d_iter), _max_abs(d_next))
    e_H = _max_abs(H_next - H_used)
    H_scale = max(
        _max_abs(committed.history), _max_abs(H_used), _max_abs(H_next)
    )
    reaction_before = float(mechanics_before.top_reaction)
    reaction_after = float(mechanics_after.top_reaction)
    e_R = abs(reaction_after - reaction_before)
    R_scale = max(
        abs(float(mechanics_before.top_reaction)),
        abs(float(mechanics_after.top_reaction)),
        abs(float(mechanics_before.bottom_reaction)),
        abs(float(mechanics_after.bottom_reaction)),
        abs(float(mechanics_before.anchor_reaction)),
        abs(float(mechanics_after.anchor_reaction)),
    )
    force_before = np.asarray(mechanics_before.total_force, dtype=np.float64)
    force_after = np.asarray(mechanics_after.total_force, dtype=np.float64)
    force_error = max(_max_abs(force_before), _max_abs(force_after))
    force_scale = R_scale
    moment_before = float(mechanics_before.total_moment)
    moment_after = float(mechanics_after.total_moment)
    moment_error = max(abs(moment_before), abs(moment_after))
    moment_scale = force_scale * geometry_length
    history_gap = np.asarray(H_next - committed.history, dtype=np.float64)
    history_min_gap = float(np.min(history_gap))
    damage_min = float(np.min(d_next))
    damage_max = float(np.max(d_next))
    healing_min = float(np.min(d_next - committed.damage))
    mechanics_pass = (
        int(mechanics_before.cg_info) == 0 and int(mechanics_after.cg_info) == 0
    )
    passes = {
        "displacement": e_u <= config.displacement.limit(u_scale),
        "damage": e_d <= config.damage.limit(d_scale),
        "history": e_H <= config.history.limit(H_scale),
        "reaction": e_R <= config.reaction.limit(R_scale),
        "damage_residual": _max_abs(rho_next)
        <= config.damage_residual.limit(rho_scale),
        "damage_range": (
            damage_min >= -config.damage_range_atol
            and damage_max <= 1.0 + config.damage_range_atol
        ),
        "healing": healing_min >= -config.healing_atol,
        "history_monotonic": history_min_gap >= 0.0,
        "force": force_error <= config.force.limit(force_scale),
        "moment": moment_error <= config.moment.limit(moment_scale),
        "mechanics": mechanics_pass,
    }
    failure_names = [name for name, passed in passes.items() if not passed]
    return StaggerDiagnostics(
        e_u=e_u,
        u_scale=u_scale,
        u_limit=config.displacement.limit(u_scale),
        u_pass=passes["displacement"],
        e_d=e_d,
        d_scale=d_scale,
        d_limit=config.damage.limit(d_scale),
        d_pass=passes["damage"],
        e_H=e_H,
        H_scale=H_scale,
        H_limit=config.history.limit(H_scale),
        H_pass=passes["history"],
        e_R=e_R,
        R_scale=R_scale,
        R_limit=config.reaction.limit(R_scale),
        R_pass=passes["reaction"],
        rho_used=rho_used,
        rho_next=rho_next,
        rho_used_volume=rho_used_volume,
        rho_next_volume=rho_next_volume,
        e_rho=_max_abs(rho_next),
        rho_scale=rho_scale,
        rho_limit=config.damage_residual.limit(rho_scale),
        rho_pass=passes["damage_residual"],
        damage_min=damage_min,
        damage_max=damage_max,
        damage_range_pass=passes["damage_range"],
        healing_min=healing_min,
        healing_pass=passes["healing"],
        history_min_gap=history_min_gap,
        history_monotonic_pass=passes["history_monotonic"],
        reaction_before=reaction_before,
        reaction_after=reaction_after,
        mechanics_before_free_residual_norm=float(mechanics_before.free_residual_norm),
        mechanics_after_free_residual_norm=float(mechanics_after.free_residual_norm),
        mechanics_before_free_residual_relative_norm=float(
            mechanics_before.free_residual_relative_norm
        ),
        mechanics_after_free_residual_relative_norm=float(
            mechanics_after.free_residual_relative_norm
        ),
        mechanics_before_cg_info=int(mechanics_before.cg_info),
        mechanics_after_cg_info=int(mechanics_after.cg_info),
        force_before=force_before,
        force_after=force_after,
        moment_before=moment_before,
        moment_after=moment_after,
        force_error=force_error,
        force_scale=force_scale,
        force_limit=config.force.limit(force_scale),
        force_pass=passes["force"],
        moment_error=moment_error,
        moment_scale=moment_scale,
        moment_limit=config.moment.limit(moment_scale),
        moment_pass=passes["moment"],
        mechanics_pass=passes["mechanics"],
        passed=not failure_names,
        complete_sweep=True,
        rejection_reason="" if not failure_names else ", ".join(failure_names),
    )


@dataclass(frozen=True)
class _TrialOutcome:
    accepted_step: AcceptedStep | None
    diagnostics: StaggerDiagnostics
    failure_stage: str | None
    failure_message: str | None
    stagger_iterations: int
    exception_type: str | None


def _committed_mechanics_seed(
    committed: CommittedState,
    constraints: HardBoundaryConstraints,
    target_load: float,
) -> np.ndarray:
    """Build a target-compatible trial seed from immutable committed data."""

    target_prescribed = constraints.prescribed_vector(target_load)
    if committed.u_bar == 0.0:
        return target_prescribed
    committed_prescribed = constraints.prescribed_vector(committed.u_bar)
    ratio = target_load / committed.u_bar
    seed = np.array(target_prescribed, dtype=np.float64, copy=True)
    free = constraints.free_dofs
    seed[free] += ratio * (
        committed.displacement[free] - committed_prescribed[free]
    )
    if not np.all(np.isfinite(seed)):
        raise FloatingPointError("scaled committed mechanics seed is non-finite")
    if not np.array_equal(
        seed[constraints.constrained_dofs],
        target_prescribed[constraints.constrained_dofs],
    ):
        raise RuntimeError("scaled committed mechanics seed changed prescribed values")
    return seed


def _run_trial_attempt(
    operator: ExplicitVolumeSparseD,
    damage_matrix: Any,
    volumes: np.ndarray,
    intact_mechanics: BondAssociatedMechanics,
    constraints: HardBoundaryConstraints,
    Gc: float,
    l0: float,
    config: StrictContinuationConfig,
    committed: CommittedState,
    target_load: float,
    cutback: int,
    geometry_length: float,
    mechanics_solver: Any | None,
    acceptance_mode: str,
) -> _TrialOutcome:
    warning_mode = _validate_acceptance_mode(acceptance_mode) == "warning"
    d_iter = np.array(committed.damage, dtype=np.float64, copy=True)
    empty_diagnostics = _failure_diagnostics(
        config,
        committed,
        d_iter,
        target_load,
        damage_matrix,
        volumes,
        Gc,
        l0,
        geometry_length,
        "no complete stagger sweep",
    )
    last_diagnostics = empty_diagnostics
    mechanics_before_seed = _committed_mechanics_seed(
        committed, constraints, target_load
    )
    mechanics_before_allow_early_exit = False

    for iteration in range(config.max_stagger_iterations):
        stage = "mechanics_before"
        try:
            mechanics_before_view = FixedDamageBondAssociatedMechanics(
                intact_mechanics, d_iter
            )
            if mechanics_solver is None:
                mechanics_before = solve_fixed_damage_mechanics(
                    mechanics_before_view,
                    constraints,
                    target_load,
                    config.cg_rtol,
                    config.cg_atol,
                    config.cg_maxiter,
                    initial_guess=mechanics_before_seed,
                    allow_initial_early_exit=mechanics_before_allow_early_exit,
                )
            else:
                solver_damage = _bytes_backed_copy(d_iter)
                solver_seed = _bytes_backed_copy(mechanics_before_seed)
                mechanics_before = mechanics_solver.solve(
                    solver_damage,
                    target_load,
                    config.cg_rtol,
                    config.cg_atol,
                    config.cg_maxiter,
                    initial_guess=solver_seed,
                    allow_initial_early_exit=mechanics_before_allow_early_exit,
                )
            _validate_mechanics_result(
                mechanics_before,
                constraints,
                target_load,
                intact_mechanics.n_dofs,
                mechanics_before_view,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostics = _failure_diagnostics(
                config,
                committed,
                d_iter,
                target_load,
                damage_matrix,
                volumes,
                Gc,
                l0,
                geometry_length,
                f"{stage}: {error}",
            )
            return _TrialOutcome(
                accepted_step=None,
                diagnostics=diagnostics,
                failure_stage=stage,
                failure_message=str(error),
                stagger_iterations=iteration + 1,
                exception_type=type(error).__name__,
            )

        stage = "fixed_history_damage"
        H_used: np.ndarray | None = None
        try:
            amor_before = compute_amor_history(
                intact_mechanics,
                mechanics_before.displacement,
                committed.history,
            )
            H_used = _require_float64_array(
                amor_before.trial_history,
                "trial H_used",
                shape=committed.history.shape,
            )
            damage_result = solve_fixed_history_damage(operator, Gc, l0, H_used)
            d_next = _require_float64_array(
                damage_result.damage,
                "fixed-history trial damage",
                shape=committed.damage.shape,
            )
            if not np.array_equal(damage_result.history, H_used):
                raise ValueError(
                    "fixed-history result history does not equal the H_used input"
                )
        except (RuntimeError, FloatingPointError) as error:
            diagnostics = _failure_diagnostics(
                config,
                committed,
                d_iter,
                target_load,
                damage_matrix,
                volumes,
                Gc,
                l0,
                geometry_length,
                f"{stage}: {error}",
                H_used=H_used,
                mechanics_before=mechanics_before,
            )
            return _TrialOutcome(
                accepted_step=None,
                diagnostics=diagnostics,
                failure_stage=stage,
                failure_message=str(error),
                stagger_iterations=iteration + 1,
                exception_type=type(error).__name__,
            )

        d_next = np.array(d_next, dtype=np.float64, copy=True)
        damage_min = float(np.min(d_next))
        damage_max = float(np.max(d_next))
        healing_min = float(np.min(d_next - committed.damage))
        if not warning_mode:
            # Strict mode rejects raw out-of-range/healing candidates before
            # the mechanics-after solve.  Warning mode never short-circuits
            # these finite failures; the sweep completes and the gates are
            # bypassed only at the stagger budget.
            range_pass = (
                damage_min >= -config.damage_range_atol
                and damage_max <= 1.0 + config.damage_range_atol
            )
            healing_pass = healing_min >= -config.healing_atol
            if not range_pass or not healing_pass:
                reasons = []
                if not range_pass:
                    reasons.append("damage_range")
                if not healing_pass:
                    reasons.append("healing")
                reason = ", ".join(reasons)
                diagnostics = _failure_diagnostics(
                    config,
                    committed,
                    d_iter,
                    target_load,
                    damage_matrix,
                    volumes,
                    Gc,
                    l0,
                    geometry_length,
                    reason,
                    d_next=d_next,
                    H_used=H_used,
                    mechanics_before=mechanics_before,
                )
                return _TrialOutcome(
                    accepted_step=None,
                    diagnostics=diagnostics,
                    failure_stage="damage_acceptance",
                    failure_message=reason,
                    stagger_iterations=iteration + 1,
                    exception_type=None,
                )

        stage = "mechanics_after"
        try:
            # This is deliberately a new fixed-damage object.  The trial view
            # used for mechanics_before is never retained in committed output.
            mechanics_after_view = FixedDamageBondAssociatedMechanics(
                intact_mechanics, d_next
            )
            if mechanics_solver is None:
                mechanics_after = solve_fixed_damage_mechanics(
                    mechanics_after_view,
                    constraints,
                    target_load,
                    config.cg_rtol,
                    config.cg_atol,
                    config.cg_maxiter,
                    initial_guess=mechanics_before.displacement,
                    allow_initial_early_exit=False,
                )
            else:
                solver_damage = _bytes_backed_copy(d_next)
                solver_seed = _bytes_backed_copy(mechanics_before.displacement)
                mechanics_after = mechanics_solver.solve(
                    solver_damage,
                    target_load,
                    config.cg_rtol,
                    config.cg_atol,
                    config.cg_maxiter,
                    initial_guess=solver_seed,
                    allow_initial_early_exit=False,
                )
            _validate_mechanics_result(
                mechanics_after,
                constraints,
                target_load,
                intact_mechanics.n_dofs,
                mechanics_after_view,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostics = _failure_diagnostics(
                config,
                committed,
                d_iter,
                target_load,
                damage_matrix,
                volumes,
                Gc,
                l0,
                geometry_length,
                f"{stage}: {error}",
                d_next=d_next,
                H_used=H_used,
                mechanics_before=mechanics_before,
            )
            return _TrialOutcome(
                accepted_step=None,
                diagnostics=diagnostics,
                failure_stage=stage,
                failure_message=str(error),
                stagger_iterations=iteration + 1,
                exception_type=type(error).__name__,
            )

        stage = "after_history"
        try:
            amor_after = compute_amor_history(
                intact_mechanics,
                mechanics_after.displacement,
                committed.history,
            )
            H_next = _require_float64_array(
                amor_after.trial_history,
                "trial H_next",
                shape=committed.history.shape,
            )
        except (RuntimeError, FloatingPointError) as error:
            diagnostics = _failure_diagnostics(
                config,
                committed,
                d_iter,
                target_load,
                damage_matrix,
                volumes,
                Gc,
                l0,
                geometry_length,
                f"{stage}: {error}",
                d_next=d_next,
                H_used=H_used,
                mechanics_before=mechanics_before,
                mechanics_after=mechanics_after,
            )
            return _TrialOutcome(
                accepted_step=None,
                diagnostics=diagnostics,
                failure_stage=stage,
                failure_message=str(error),
                stagger_iterations=iteration + 1,
                exception_type=type(error).__name__,
            )

        H_next = np.array(H_next, dtype=np.float64, copy=True)

        diagnostics = _full_stagger_diagnostics(
            config,
            committed,
            d_iter,
            d_next,
            H_used,
            H_next,
            mechanics_before,
            mechanics_after,
            damage_matrix,
            volumes,
            Gc,
            l0,
            geometry_length,
            damage_result,
        )
        last_diagnostics = diagnostics
        if diagnostics.passed:
            accepted_state = CommittedState(
                target_load,
                mechanics_after.displacement,
                d_next,
                H_next,
            )
            accepted_step = AcceptedStep(
                load_start=committed.u_bar,
                load_target=target_load,
                increment=target_load - committed.u_bar,
                state=accepted_state,
                history_used=H_used,
                amor_driver=amor_after,
                fixed_history_result=damage_result,
                mechanics_result=mechanics_after,
                diagnostics=diagnostics,
                stagger_iterations=iteration + 1,
                cutback_count=cutback,
                acceptance_status="strict_passed",
                bypassed_gates=(),
                damage_before=d_iter,
                mechanics_before_result=mechanics_before,
            )
            return _TrialOutcome(
                accepted_step=accepted_step,
                diagnostics=diagnostics,
                failure_stage=None,
                failure_message=None,
                stagger_iterations=iteration + 1,
                exception_type=None,
            )

        if iteration + 1 == config.max_stagger_iterations:
            if warning_mode:
                # Warning commit only at the full stagger budget, only after a
                # complete finite before/damage/after/history sweep, and only
                # for the canonical bypassable gates.  The committed candidate
                # is the exact unmodified (u_after, d_next, H_next).
                bypassed = _failed_warning_gates(diagnostics)
                if (
                    bypassed
                    and diagnostics.complete_sweep
                    and diagnostics.mechanics_pass
                ):
                    accepted_state = CommittedState(
                        target_load,
                        mechanics_after.displacement,
                        d_next,
                        H_next,
                    )
                    accepted_step = AcceptedStep(
                        load_start=committed.u_bar,
                        load_target=target_load,
                        increment=target_load - committed.u_bar,
                        state=accepted_state,
                        history_used=H_used,
                        amor_driver=amor_after,
                        fixed_history_result=damage_result,
                        mechanics_result=mechanics_after,
                        diagnostics=diagnostics,
                        stagger_iterations=iteration + 1,
                        cutback_count=cutback,
                        acceptance_status="warning_accepted",
                        bypassed_gates=bypassed,
                        damage_before=d_iter,
                        mechanics_before_result=mechanics_before,
                    )
                    return _TrialOutcome(
                        accepted_step=accepted_step,
                        diagnostics=diagnostics,
                        failure_stage=None,
                        failure_message=None,
                        stagger_iterations=iteration + 1,
                        exception_type=None,
                    )
            return _TrialOutcome(
                accepted_step=None,
                diagnostics=diagnostics,
                failure_stage="gates",
                failure_message=diagnostics.rejection_reason,
                stagger_iterations=iteration + 1,
                exception_type=None,
            )
        # Only a complete sweep updates the local damage iterate.  The
        # committed history is intentionally never replaced inside this loop.
        d_iter = d_next
        mechanics_before_seed = np.array(
            mechanics_after.displacement, dtype=np.float64, copy=True
        )
        mechanics_before_allow_early_exit = True

    return _TrialOutcome(
        accepted_step=None,
        diagnostics=last_diagnostics,
        failure_stage="gates",
        failure_message=last_diagnostics.rejection_reason,
        stagger_iterations=config.max_stagger_iterations,
        exception_type=None,
    )


def _run_continuation(
    operator: ExplicitVolumeSparseD,
    intact_mechanics: BondAssociatedMechanics,
    constraints: HardBoundaryConstraints,
    Gc: Any,
    l0: Any,
    config: StrictContinuationConfig,
    mechanics_solver: Any | None,
    acceptance_mode: str,
) -> ContinuationResult:
    """Shared strict/warning staggered continuation with rollback/cutback."""

    mode = _validate_acceptance_mode(acceptance_mode)
    damage_matrix, volumes, _, gc_value, l0_value = _validate_continuation_setup(
        operator, intact_mechanics, constraints, Gc, l0, config
    )
    mechanics_backend = "numpy_scipy"
    if mechanics_solver is not None:
        from .nonlocal_mechanics_jax import JaxCPUReplayMechanicsAdapter

        if not isinstance(mechanics_solver, JaxCPUReplayMechanicsAdapter):
            raise TypeError(
                "mechanics_solver must be a JaxCPUReplayMechanicsAdapter instance"
            )
        if mechanics_solver.cpu_mechanics is not intact_mechanics:
            raise ValueError(
                "mechanics_solver must use the supplied intact_mechanics instance"
            )
        if mechanics_solver.constraints is not constraints:
            raise ValueError(
                "mechanics_solver must use the supplied constraints instance"
            )
        mechanics_backend = mechanics_solver.mechanics_backend
    metadata = _build_continuation_metadata(
        operator,
        intact_mechanics,
        constraints,
        gc_value,
        l0_value,
        config,
        mechanics_backend,
        mode,
    )
    n_particles = intact_mechanics.n_particles
    initial_state = CommittedState(
        config.load_start,
        np.zeros(intact_mechanics.n_dofs, dtype=np.float64),
        np.zeros(n_particles, dtype=np.float64),
        np.zeros(n_particles, dtype=np.float64),
    )
    committed = initial_state
    states: list[CommittedState] = [initial_state]
    accepted_steps: list[AcceptedStep] = []
    attempts: list[AttemptRecord] = []
    current_increment = config.du_initial
    cutback = 0

    geometry_length = max(
        _max_abs(np.asarray(intact_mechanics.coordinates)),
        _max_abs(np.ptp(np.asarray(intact_mechanics.coordinates), axis=0)),
    )
    while committed.u_bar < config.load_final:
        target_load = _next_load_target(
            committed.u_bar, current_increment, config.load_final
        )
        trial_increment = target_load - committed.u_bar
        if not target_load > committed.u_bar:
            raise ContinuationFailure(
                f"{mode} continuation made no floating-point load progress: "
                f"committed={committed.u_bar:.17g}, target_load={target_load:.17g}, "
                f"increment={trial_increment:.17g}",
                committed,
                tuple(attempts),
            )
        before_digest = _state_digest(committed)
        outcome = _run_trial_attempt(
            operator,
            damage_matrix,
            volumes,
            intact_mechanics,
            constraints,
            gc_value,
            l0_value,
            config,
            committed,
            target_load,
            cutback,
            geometry_length,
            mechanics_solver,
            mode,
        )

        if outcome.accepted_step is not None:
            next_state = outcome.accepted_step.state
            after_digest = _state_digest(next_state)
            attempts.append(
                AttemptRecord(
                    cutback=cutback,
                    load_start=committed.u_bar,
                    load_target=target_load,
                    accepted=True,
                    failure_stage=None,
                    failure_message=None,
                    committed_before_digest=before_digest,
                    committed_after_digest=after_digest,
                    diagnostics=outcome.diagnostics,
                    stagger_iterations=outcome.stagger_iterations,
                    exception_type=None,
                    acceptance_status=outcome.accepted_step.acceptance_status,
                    sweep_complete=True,
                    bypassed_gates=outcome.accepted_step.bypassed_gates,
                )
            )
            accepted_steps.append(outcome.accepted_step)
            committed = next_state
            states.append(committed)
            # Persist a cutback increment after acceptance; no growth rule is
            # silently introduced between physically validated load states.
            current_increment = trial_increment
            cutback = 0
            continue

        after_digest = _state_digest(committed)
        if after_digest != before_digest:
            raise RuntimeError(
                f"{mode} continuation rollback digest changed during a failed attempt"
            )
        failure_message = outcome.failure_message or outcome.diagnostics.rejection_reason
        attempts.append(
            AttemptRecord(
                cutback=cutback,
                load_start=committed.u_bar,
                load_target=target_load,
                accepted=False,
                failure_stage=outcome.failure_stage,
                failure_message=failure_message,
                committed_before_digest=before_digest,
                committed_after_digest=after_digest,
                diagnostics=outcome.diagnostics,
                stagger_iterations=outcome.stagger_iterations,
                exception_type=outcome.exception_type,
                acceptance_status=None,
                sweep_complete=bool(outcome.diagnostics.complete_sweep),
                bypassed_gates=(),
            )
        )
        at_minimum = trial_increment <= config.du_min
        at_cutback_limit = cutback >= config.max_cutbacks
        if at_minimum or at_cutback_limit:
            reasons = []
            if at_minimum:
                reasons.append("du_min")
            if at_cutback_limit:
                reasons.append("max_cutbacks")
            raise ContinuationFailure(
                f"{mode} continuation failed at load target "
                f"{target_load:.17g} ({', '.join(reasons)}): {failure_message}",
                committed,
                tuple(attempts),
            )
        cutback += 1
        current_increment = max(config.du_min, trial_increment * config.cutback_factor)

    return ContinuationResult(
        initial_state=initial_state,
        metadata=metadata,
        states=tuple(states),
        accepted_steps=tuple(accepted_steps),
        attempts=tuple(attempts),
    )


def _next_load_target(
    committed_load: float,
    current_increment: float,
    load_final: float,
) -> float:
    """Return the next target, snapping an ulp-scale remainder to the endpoint."""

    remaining = load_final - committed_load
    target_load = committed_load + min(current_increment, remaining)
    endpoint_tolerance = 8.0 * abs(np.spacing(np.float64(load_final)))
    if load_final - target_load <= endpoint_tolerance:
        return load_final
    return target_load


def run_strict_continuation(
    operator: ExplicitVolumeSparseD,
    intact_mechanics: BondAssociatedMechanics,
    constraints: HardBoundaryConstraints,
    Gc: Any,
    l0: Any,
    config: StrictContinuationConfig,
    *,
    mechanics_solver: Any | None = None,
) -> ContinuationResult:
    """Run strict staggered continuation with rollback/cutback.

    Every trial starts from the exact last :class:`CommittedState`.  A sweep
    performs mechanics-before, committed-history Amor update, fixed-history
    damage, mechanics-after, and a second committed-history Amor update.  No
    trial history is accumulated and no standalone damage solve is performed
    after the final mechanics solve.  Every post-sweep acceptance gate must
    pass; accepted steps are recorded as ``strict_passed`` and the result
    metadata carries ``acceptance_mode='strict'``.
    """

    return _run_continuation(
        operator,
        intact_mechanics,
        constraints,
        Gc,
        l0,
        config,
        mechanics_solver,
        "strict",
    )


def run_warning_continuation(
    operator: ExplicitVolumeSparseD,
    intact_mechanics: BondAssociatedMechanics,
    constraints: HardBoundaryConstraints,
    Gc: Any,
    l0: Any,
    config: StrictContinuationConfig,
    *,
    mechanics_solver: Any | None = None,
) -> ContinuationResult:
    """Run the separately archived all-gate warning diagnostic continuation.

    This mode shares the exact strict sweep structure, tolerances, rollback,
    and cutback rules.  It differs only at acceptance: a trial runs the full
    configured stagger budget unless every gate passes earlier, finite raw
    damage-range/healing failures are never short-circuited, and at the stagger
    limit any failed finite bypassable gate
    (:data:`WARNING_GATE_NAMES`) may be bypassed.  A warning commit is made
    only after a complete finite mechanics-before, ``H_used``, raw damage,
    mechanics-after, ``H_next`` sweep with complete diagnostics; it commits the
    exact unmodified ``(u_after, d_next, H_next)`` with
    ``diagnostics.passed=False`` and the exact canonical bypassed-gate list.
    Missing/non-finite candidates, structural errors, and mechanics/CG
    failures remain the existing audited hard/cutback failures.  No clipping,
    projection, tolerance change, standalone final damage solve, history
    accumulation, or increment growth is introduced.
    """

    return _run_continuation(
        operator,
        intact_mechanics,
        constraints,
        Gc,
        l0,
        config,
        mechanics_solver,
        "warning",
    )


def _stack_or_empty(
    arrays: list[np.ndarray],
    shape_tail: tuple[int, ...],
    dtype: np.dtype[Any] = _FLOAT64,
) -> np.ndarray:
    if arrays:
        return np.asarray(arrays, dtype=dtype)
    return np.empty((0, *shape_tail), dtype=dtype)


def _json_safe(value: Any) -> Any:
    """Convert NumPy values, including failed inf sentinels, to valid JSON."""

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, Integral)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _diagnostic_payload(diagnostics: StaggerDiagnostics) -> dict[str, Any]:
    """Return complete scalar and array diagnostics for JSON audit storage."""

    payload = {
        "errors": diagnostics.errors,
        "scales": diagnostics.scales,
        "limits": diagnostics.limits,
        "passes": diagnostics.gate_passes,
        "complete_sweep": diagnostics.complete_sweep,
        "rejection_reason": diagnostics.rejection_reason,
        "damage_min": diagnostics.damage_min,
        "damage_max": diagnostics.damage_max,
        "healing_min": diagnostics.healing_min,
        "history_min_gap": diagnostics.history_min_gap,
        "reaction_before": diagnostics.reaction_before,
        "reaction_after": diagnostics.reaction_after,
        "force_before": diagnostics.force_before,
        "force_after": diagnostics.force_after,
        "moment_before": diagnostics.moment_before,
        "moment_after": diagnostics.moment_after,
        "mechanics_before_free_residual_norm": diagnostics.mechanics_before_free_residual_norm,
        "mechanics_after_free_residual_norm": diagnostics.mechanics_after_free_residual_norm,
        "mechanics_before_free_residual_relative_norm": diagnostics.mechanics_before_free_residual_relative_norm,
        "mechanics_after_free_residual_relative_norm": diagnostics.mechanics_after_free_residual_relative_norm,
        "mechanics_before_cg_info": diagnostics.mechanics_before_cg_info,
        "mechanics_after_cg_info": diagnostics.mechanics_after_cg_info,
        "rho_used": diagnostics.rho_used,
        "rho_next": diagnostics.rho_next,
        "rho_used_volume": diagnostics.rho_used_volume,
        "rho_next_volume": diagnostics.rho_next_volume,
    }
    return _json_safe(payload)


def _config_payload(config: StrictContinuationConfig) -> dict[str, Any]:
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
        "displacement": {
            "atol": config.displacement.atol,
            "rtol": config.displacement.rtol,
            "scale_floor": config.displacement.scale_floor,
        },
        "damage": {
            "atol": config.damage.atol,
            "rtol": config.damage.rtol,
            "scale_floor": config.damage.scale_floor,
        },
        "history": {
            "atol": config.history.atol,
            "rtol": config.history.rtol,
            "scale_floor": config.history.scale_floor,
        },
        "reaction": {
            "atol": config.reaction.atol,
            "rtol": config.reaction.rtol,
            "scale_floor": config.reaction.scale_floor,
        },
        "damage_residual": {
            "atol": config.damage_residual.atol,
            "rtol": config.damage_residual.rtol,
            "scale_floor": config.damage_residual.scale_floor,
        },
        "force": {
            "atol": config.force.atol,
            "rtol": config.force.rtol,
            "scale_floor": config.force.scale_floor,
        },
        "moment": {
            "atol": config.moment.atol,
            "rtol": config.moment.rtol,
            "scale_floor": config.moment.scale_floor,
        },
        "healing_atol": config.healing_atol,
        "damage_range_atol": config.damage_range_atol,
    }


def write_continuation_archive(
    path: Any,
    result: ContinuationResult,
    archive_metadata: SupplementalArchiveMetadata,
) -> None:
    """Persist every controller state, accepted solve, diagnostic, and audit.

    This is intentionally a result-only writer rather than a production
    configuration or a solver entry point.  All three arguments are required
    so a caller cannot accidentally write to an implicit output location or
    silently omit mesh/provenance data unavailable to the controller.
    """

    if not isinstance(result, ContinuationResult):
        raise TypeError("result must be a ContinuationResult instance")
    if not isinstance(archive_metadata, SupplementalArchiveMetadata):
        raise TypeError(
            "archive_metadata must be a SupplementalArchiveMetadata instance"
        )
    result.validate()
    archive_metadata.validate_against(result.metadata)
    try:
        archive_path = os.fspath(path)
    except TypeError as error:
        raise TypeError("path must be a filesystem path") from error
    if not isinstance(archive_path, (str, bytes)) or not archive_path:
        raise ValueError("path must be a non-empty filesystem path")
    archive_filename = os.fsdecode(archive_path)
    parent = os.path.dirname(archive_filename)
    if parent:
        os.makedirs(parent, exist_ok=True)

    states = result.states
    accepted = result.accepted_steps
    attempts = result.attempts
    n_dofs = result.initial_state.displacement.size
    n_particles = result.initial_state.damage.size
    state_u_bar = np.asarray([state.u_bar for state in states], dtype=np.float64)
    state_displacement = np.stack(
        [state.displacement for state in states], axis=0
    ).astype(np.float64, copy=False)
    state_damage = np.stack([state.damage for state in states], axis=0).astype(
        np.float64, copy=False
    )
    state_history = np.stack([state.history for state in states], axis=0).astype(
        np.float64, copy=False
    )

    metadata = result.metadata
    config_json = json.dumps(
        _config_payload(metadata.config), sort_keys=True, separators=(",", ":")
    )
    metadata_json = json.dumps(
        _json_safe(
            {
                "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
                "acceptance_mode": metadata.acceptance_mode,
                "E": metadata.E,
                "nu": metadata.nu,
                "k_res": metadata.k_res,
                "rank_rtol": metadata.rank_rtol,
                "Gc": metadata.Gc,
                "l0": metadata.l0,
                "delta": metadata.delta,
                "kernel": metadata.kernel,
                "plane_strain": metadata.plane_strain,
                "stabilization": metadata.stabilization,
                "mechanics_backend": metadata.mechanics_backend,
                "dtype": str(metadata.dtype),
                "anchor_particle": metadata.anchor_particle,
                "config": _config_payload(metadata.config),
                "gate_names": list(WARNING_GATE_NAMES),
                "visibility_tolerance": archive_metadata.visibility_tolerance,
                "visibility_rule": archive_metadata.visibility_rule,
                "quadrature_rule": archive_metadata.quadrature_rule,
                "study_label": archive_metadata.study_label,
                "model_provenance": archive_metadata.model_provenance,
            }
        ),
        sort_keys=True,
        separators=(",", ":"),
    )

    accepted_u_bar = np.asarray(
        [step.load_target for step in accepted], dtype=np.float64
    )
    accepted_increment = np.asarray(
        [step.increment for step in accepted], dtype=np.float64
    )
    accepted_cutbacks = np.asarray(
        [step.cutback_count for step in accepted], dtype=np.int64
    )
    accepted_history_used = _stack_or_empty(
        [step.history_used for step in accepted], (n_particles,)
    )
    accepted_amor_driver = _stack_or_empty(
        [step.amor_driver.particle_driver for step in accepted], (n_particles,)
    )
    accepted_damage_residual = _stack_or_empty(
        [step.fixed_history_result.strong_residual for step in accepted],
        (n_particles,),
    )
    accepted_mechanics_displacement = _stack_or_empty(
        [step.mechanics_result.displacement for step in accepted], (n_dofs,)
    )
    accepted_damage_before = _stack_or_empty(
        [step.damage_before for step in accepted], (n_particles,)
    )
    accepted_mechanics_before_displacement = _stack_or_empty(
        [step.mechanics_before_result.displacement for step in accepted], (n_dofs,)
    )
    accepted_mechanics_before_free_residual = _stack_or_empty(
        [step.mechanics_before_result.free_residual for step in accepted],
        (len(metadata.free_dofs),),
    )
    accepted_reaction = np.asarray(
        [step.mechanics_result.top_reaction for step in accepted], dtype=np.float64
    )
    accepted_cg_info = np.asarray(
        [step.mechanics_result.cg_info for step in accepted], dtype=np.int64
    )
    accepted_cg_iterations = np.asarray(
        [step.mechanics_result.cg_iterations for step in accepted], dtype=np.int64
    )
    accepted_force = _stack_or_empty(
        [step.mechanics_result.total_force for step in accepted], (2,)
    )
    accepted_moment = np.asarray(
        [step.mechanics_result.total_moment for step in accepted], dtype=np.float64
    )
    accepted_free_residual_norm = np.asarray(
        [step.mechanics_result.free_residual_norm for step in accepted],
        dtype=np.float64,
    )
    accepted_free_rhs_norm = np.asarray(
        [step.mechanics_result.free_rhs_norm for step in accepted],
        dtype=np.float64,
    )
    accepted_initial_free_residual_norm = np.asarray(
        [step.mechanics_result.initial_free_residual_norm for step in accepted],
        dtype=np.float64,
    )
    accepted_cg_convergence_limit = np.asarray(
        [step.mechanics_result.cg_convergence_limit for step in accepted],
        dtype=np.float64,
    )
    accepted_initial_guess_supplied = np.asarray(
        [step.mechanics_result.initial_guess_supplied for step in accepted], dtype=bool
    )
    accepted_cg_initial_guess_used = np.asarray(
        [step.mechanics_result.cg_initial_guess_used for step in accepted], dtype=bool
    )
    accepted_early_exit = np.asarray(
        [step.mechanics_result.early_exit for step in accepted], dtype=bool
    )
    accepted_free_residual_relative_norm = np.asarray(
        [step.mechanics_result.free_residual_relative_norm for step in accepted],
        dtype=np.float64,
    )
    accepted_free_residual = _stack_or_empty(
        [step.mechanics_result.free_residual for step in accepted],
        (len(metadata.free_dofs),),
    )
    accepted_load_start = np.asarray(
        [step.load_start for step in accepted], dtype=np.float64
    )
    accepted_attempts = [attempt for attempt in attempts if attempt.accepted]
    if len(accepted_attempts) != len(accepted):
        raise ValueError("continuation archive accepted attempts do not match accepted steps")
    accepted_stagger_iterations = np.asarray(
        [step.stagger_iterations for step in accepted],
        dtype=np.int64,
    )
    accepted_bottom_reaction = np.asarray(
        [step.mechanics_result.bottom_reaction for step in accepted], dtype=np.float64
    )
    accepted_anchor_reaction = np.asarray(
        [step.mechanics_result.anchor_reaction for step in accepted], dtype=np.float64
    )
    accepted_mechanics_energy = np.asarray(
        [step.mechanics_result.evaluation.energy for step in accepted], dtype=np.float64
    )
    accepted_mechanics_residual = _stack_or_empty(
        [step.mechanics_result.evaluation.residual for step in accepted], (n_dofs,)
    )
    accepted_mechanics_internal_force = _stack_or_empty(
        [step.mechanics_result.evaluation.internal_force for step in accepted],
        (n_dofs,),
    )
    n_edges = metadata.directed_edges.shape[0]
    accepted_edge_F = _stack_or_empty(
        [step.mechanics_result.evaluation.edge_deformation_gradient for step in accepted],
        (n_edges, 2, 2),
    )
    accepted_edge_strain = _stack_or_empty(
        [step.mechanics_result.evaluation.edge_strain for step in accepted],
        (n_edges, 2, 2),
    )
    accepted_edge_stress = _stack_or_empty(
        [step.mechanics_result.evaluation.edge_stress for step in accepted],
        (n_edges, 2, 2),
    )
    accepted_edge_effective_stress = _stack_or_empty(
        [step.mechanics_result.evaluation.edge_effective_stress for step in accepted],
        (n_edges, 2, 2),
    )
    accepted_edge_energy_density = _stack_or_empty(
        [step.mechanics_result.evaluation.edge_energy_density for step in accepted],
        (n_edges,),
    )
    accepted_particle_strain = _stack_or_empty(
        [step.mechanics_result.evaluation.particle_strain for step in accepted],
        (n_particles, 2, 2),
    )
    accepted_particle_stress = _stack_or_empty(
        [step.mechanics_result.evaluation.particle_stress for step in accepted],
        (n_particles, 2, 2),
    )
    accepted_particle_effective_stress = _stack_or_empty(
        [step.mechanics_result.evaluation.particle_effective_stress for step in accepted],
        (n_particles, 2, 2),
    )
    accepted_fixed_history_damage = _stack_or_empty(
        [step.fixed_history_result.damage for step in accepted], (n_particles,)
    )
    accepted_fixed_history_history = _stack_or_empty(
        [step.fixed_history_result.history for step in accepted], (n_particles,)
    )
    accepted_fixed_history_strong_residual = _stack_or_empty(
        [step.fixed_history_result.strong_residual for step in accepted],
        (n_particles,),
    )
    accepted_fixed_history_euclidean_residual = _stack_or_empty(
        [step.fixed_history_result.euclidean_residual for step in accepted],
        (n_particles,),
    )
    accepted_fixed_history_crack_energy = np.asarray(
        [step.fixed_history_result.crack_local_energy for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_history_energy = np.asarray(
        [step.fixed_history_result.history_energy for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_graph_energy = np.asarray(
        [step.fixed_history_result.graph_energy for step in accepted], dtype=np.float64
    )
    accepted_fixed_history_total_energy = np.asarray(
        [step.fixed_history_result.total_energy for step in accepted], dtype=np.float64
    )
    accepted_fixed_history_strong_residual_norm = np.asarray(
        [step.fixed_history_result.strong_residual_norm for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_strong_residual_inf = np.asarray(
        [step.fixed_history_result.strong_residual_inf for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_euclidean_residual_norm = np.asarray(
        [step.fixed_history_result.euclidean_residual_norm for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_euclidean_residual_inf = np.asarray(
        [step.fixed_history_result.euclidean_residual_inf for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_damage_min = np.asarray(
        [step.fixed_history_result.damage_min for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_damage_max = np.asarray(
        [step.fixed_history_result.damage_max for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_below_zero_count = np.asarray(
        [step.fixed_history_result.below_zero_count for step in accepted],
        dtype=np.int64,
    )
    accepted_fixed_history_above_one_count = np.asarray(
        [step.fixed_history_result.above_one_count for step in accepted],
        dtype=np.int64,
    )
    accepted_fixed_history_symmetry_defect = np.asarray(
        [step.fixed_history_result.symmetry_defect for step in accepted],
        dtype=np.float64,
    )
    accepted_fixed_history_symmetry_relative_defect = np.asarray(
        [step.fixed_history_result.symmetry_relative_defect for step in accepted],
        dtype=np.float64,
    )
    accepted_H_used = _stack_or_empty(
        [step.history_used for step in accepted], (n_particles,)
    )
    accepted_H_next = _stack_or_empty(
        [step.state.history for step in accepted], (n_particles,)
    )
    accepted_committed_history_energy = np.asarray(
        [
            float(
                np.sum(
                    metadata.volumes
                    * step.state.history
                    * (1.0 - step.state.damage) ** 2
                )
            )
            for step in accepted
        ],
        dtype=np.float64,
    )
    accepted_committed_damage_total_energy = (
        accepted_fixed_history_crack_energy
        + accepted_fixed_history_graph_energy
        + accepted_committed_history_energy
    )
    accepted_amor_edge_positive_energy = _stack_or_empty(
        [step.amor_driver.edge_positive_energy for step in accepted], (n_edges,)
    )
    accepted_amor_particle_driver = _stack_or_empty(
        [step.amor_driver.particle_driver for step in accepted], (n_particles,)
    )
    accepted_amor_previous_history = _stack_or_empty(
        [step.amor_driver.previous_history for step in accepted], (n_particles,)
    )
    accepted_amor_trial_history = _stack_or_empty(
        [step.amor_driver.trial_history for step in accepted], (n_particles,)
    )

    diagnostics = [step.diagnostics for step in accepted]
    accepted_e_u = np.asarray([diag.e_u for diag in diagnostics], dtype=np.float64)
    accepted_e_d = np.asarray([diag.e_d for diag in diagnostics], dtype=np.float64)
    accepted_e_H = np.asarray([diag.e_H for diag in diagnostics], dtype=np.float64)
    accepted_e_R = np.asarray([diag.e_R for diag in diagnostics], dtype=np.float64)
    accepted_e_rho = np.asarray([diag.e_rho for diag in diagnostics], dtype=np.float64)
    accepted_u_scale = np.asarray(
        [diag.u_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_u_limit = np.asarray(
        [diag.u_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_d_scale = np.asarray(
        [diag.d_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_d_limit = np.asarray(
        [diag.d_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_H_scale = np.asarray(
        [diag.H_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_H_limit = np.asarray(
        [diag.H_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_R_scale = np.asarray(
        [diag.R_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_R_limit = np.asarray(
        [diag.R_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_rho_scale = np.asarray(
        [diag.rho_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_rho_limit = np.asarray(
        [diag.rho_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_force_error = np.asarray(
        [diag.force_error for diag in diagnostics], dtype=np.float64
    )
    accepted_force_scale = np.asarray(
        [diag.force_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_force_limit = np.asarray(
        [diag.force_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_moment_error = np.asarray(
        [diag.moment_error for diag in diagnostics], dtype=np.float64
    )
    accepted_moment_scale = np.asarray(
        [diag.moment_scale for diag in diagnostics], dtype=np.float64
    )
    accepted_moment_limit = np.asarray(
        [diag.moment_limit for diag in diagnostics], dtype=np.float64
    )
    accepted_u_pass = np.asarray([diag.u_pass for diag in diagnostics], dtype=bool)
    accepted_d_pass = np.asarray([diag.d_pass for diag in diagnostics], dtype=bool)
    accepted_H_pass = np.asarray([diag.H_pass for diag in diagnostics], dtype=bool)
    accepted_R_pass = np.asarray([diag.R_pass for diag in diagnostics], dtype=bool)
    accepted_rho_pass = np.asarray([diag.rho_pass for diag in diagnostics], dtype=bool)
    accepted_force_pass = np.asarray(
        [diag.force_pass for diag in diagnostics], dtype=bool
    )
    accepted_moment_pass = np.asarray(
        [diag.moment_pass for diag in diagnostics], dtype=bool
    )
    accepted_rho_used = _stack_or_empty(
        [diag.rho_used for diag in diagnostics], (n_particles,)
    )
    accepted_rho_used_volume = _stack_or_empty(
        [diag.rho_used_volume for diag in diagnostics], (n_particles,)
    )
    accepted_rho_next = _stack_or_empty(
        [diag.rho_next for diag in diagnostics], (n_particles,)
    )
    accepted_rho_next_volume = _stack_or_empty(
        [diag.rho_next_volume for diag in diagnostics], (n_particles,)
    )
    accepted_damage_min = np.asarray(
        [diag.damage_min for diag in diagnostics], dtype=np.float64
    )
    accepted_damage_max = np.asarray(
        [diag.damage_max for diag in diagnostics], dtype=np.float64
    )
    accepted_healing_min = np.asarray(
        [diag.healing_min for diag in diagnostics], dtype=np.float64
    )
    accepted_history_min_gap = np.asarray(
        [diag.history_min_gap for diag in diagnostics], dtype=np.float64
    )
    accepted_reaction_before = np.asarray(
        [diag.reaction_before for diag in diagnostics], dtype=np.float64
    )
    accepted_reaction_after = np.asarray(
        [diag.reaction_after for diag in diagnostics], dtype=np.float64
    )
    accepted_mechanics_before_free_residual_norm = np.asarray(
        [diag.mechanics_before_free_residual_norm for diag in diagnostics],
        dtype=np.float64,
    )
    accepted_mechanics_after_free_residual_norm = np.asarray(
        [diag.mechanics_after_free_residual_norm for diag in diagnostics],
        dtype=np.float64,
    )
    accepted_mechanics_before_free_residual_relative_norm = np.asarray(
        [diag.mechanics_before_free_residual_relative_norm for diag in diagnostics],
        dtype=np.float64,
    )
    accepted_mechanics_after_free_residual_relative_norm = np.asarray(
        [diag.mechanics_after_free_residual_relative_norm for diag in diagnostics],
        dtype=np.float64,
    )
    accepted_mechanics_before_cg_info = np.asarray(
        [diag.mechanics_before_cg_info for diag in diagnostics], dtype=np.int64
    )
    accepted_mechanics_after_cg_info = np.asarray(
        [diag.mechanics_after_cg_info for diag in diagnostics], dtype=np.int64
    )
    accepted_force_before = _stack_or_empty(
        [diag.force_before for diag in diagnostics], (2,)
    )
    accepted_force_after = _stack_or_empty(
        [diag.force_after for diag in diagnostics], (2,)
    )
    accepted_moment_before = np.asarray(
        [diag.moment_before for diag in diagnostics], dtype=np.float64
    )
    accepted_moment_after = np.asarray(
        [diag.moment_after for diag in diagnostics], dtype=np.float64
    )
    accepted_history_monotonic_pass = np.asarray(
        [diag.history_monotonic_pass for diag in diagnostics], dtype=bool
    )
    accepted_damage_range_pass = np.asarray(
        [diag.damage_range_pass for diag in diagnostics], dtype=bool
    )
    accepted_healing_pass = np.asarray(
        [diag.healing_pass for diag in diagnostics], dtype=bool
    )
    accepted_mechanics_pass = np.asarray(
        [diag.mechanics_pass for diag in diagnostics], dtype=bool
    )
    accepted_gate_pass = np.asarray(
        [diag.passed for diag in diagnostics], dtype=bool
    )

    state_top_reaction = np.concatenate(
        (np.asarray([0.0], dtype=np.float64), accepted_reaction)
    )
    state_loads = state_u_bar
    cumulative_top_reaction_work = np.zeros(len(state_loads), dtype=np.float64)
    if len(state_loads) > 1:
        cumulative_top_reaction_work[1:] = np.cumsum(
            0.5
            * np.diff(state_loads)
            * (state_top_reaction[1:] + state_top_reaction[:-1])
        )
    accepted_top_reaction_work_increment = np.diff(cumulative_top_reaction_work)

    attempt_audit = []
    for attempt in attempts:
        attempt_audit.append(
            {
                "cutback": attempt.cutback,
                "load_start": attempt.load_start,
                "load_target": attempt.load_target,
                "accepted": attempt.accepted,
                "failure_stage": attempt.failure_stage,
                "failure_message": attempt.failure_message,
                "exception_type": attempt.exception_type,
                "stagger_iterations": attempt.stagger_iterations,
                "acceptance_status": attempt.acceptance_status,
                "sweep_complete": attempt.sweep_complete,
                "bypassed_gates": list(attempt.bypassed_gates),
                "committed_before_digest": attempt.committed_before_digest,
                "committed_after_digest": attempt.committed_after_digest,
                "diagnostics": (
                    None
                    if attempt.diagnostics is None
                    else _diagnostic_payload(attempt.diagnostics)
                ),
            }
        )
    accepted_audit = [
        {
            "load_start": step.load_start,
            "load_target": step.load_target,
            "increment": step.increment,
            "cutback_count": step.cutback_count,
            "stagger_iterations": step.stagger_iterations,
            "acceptance_status": step.acceptance_status,
            "bypassed_gates": list(step.bypassed_gates),
            "diagnostics": _diagnostic_payload(step.diagnostics),
        }
        for step in accepted
    ]
    state_acceptance_status = np.asarray(
        ["initial"] + [step.acceptance_status for step in accepted],
        dtype=np.str_,
    )
    accepted_acceptance_status = np.asarray(
        [step.acceptance_status for step in accepted], dtype=np.str_
    )
    accepted_warning = np.asarray(
        [step.acceptance_status == "warning_accepted" for step in accepted],
        dtype=bool,
    )
    accepted_bypassed_gate_mask = np.asarray(
        [
            [name in set(step.bypassed_gates) for name in WARNING_GATE_NAMES]
            for step in accepted
        ],
        dtype=bool,
    ).reshape(len(accepted), len(WARNING_GATE_NAMES))
    np.savez_compressed(
        archive_filename,
        archive_schema_version=np.asarray(
            ARCHIVE_SCHEMA_VERSION, dtype=np.int64
        ),
        acceptance_mode=np.asarray(metadata.acceptance_mode, dtype=np.str_),
        gate_names=np.asarray(WARNING_GATE_NAMES, dtype=np.str_),
        state_acceptance_status=state_acceptance_status,
        accepted_acceptance_status=accepted_acceptance_status,
        accepted_warning=accepted_warning,
        accepted_bypassed_gate_mask=accepted_bypassed_gate_mask,
        coordinates=metadata.coordinates,
        volumes=metadata.volumes,
        directed_edges=metadata.directed_edges,
        delta=np.asarray(metadata.delta, dtype=np.float64),
        kernel=np.asarray(metadata.kernel, dtype=np.str_),
        bottom_mask=metadata.bottom_mask,
        top_mask=metadata.top_mask,
        anchor_particle=np.asarray(metadata.anchor_particle, dtype=np.int64),
        constrained_dofs=metadata.constrained_dofs,
        free_dofs=metadata.free_dofs,
        E=np.asarray(metadata.E, dtype=np.float64),
        nu=np.asarray(metadata.nu, dtype=np.float64),
        k_res=np.asarray(metadata.k_res, dtype=np.float64),
        rank_rtol=np.asarray(metadata.rank_rtol, dtype=np.float64),
        Gc=np.asarray(metadata.Gc, dtype=np.float64),
        l0=np.asarray(metadata.l0, dtype=np.float64),
        plane_strain=np.asarray(metadata.plane_strain, dtype=bool),
        stabilization=np.asarray(metadata.stabilization, dtype=np.str_),
        mechanics_backend=np.asarray(metadata.mechanics_backend, dtype=np.str_),
        dtype=np.asarray(str(metadata.dtype), dtype=np.str_),
        triangles=archive_metadata.triangles,
        first_moments=archive_metadata.first_moments,
        visibility_tolerance=np.asarray(
            archive_metadata.visibility_tolerance, dtype=np.float64
        ),
        visibility_rule=np.asarray(archive_metadata.visibility_rule, dtype=np.str_),
        quadrature_rule=np.asarray(archive_metadata.quadrature_rule, dtype=np.str_),
        study_label=np.asarray(archive_metadata.study_label, dtype=np.str_),
        model_provenance=np.asarray(archive_metadata.model_provenance, dtype=np.str_),
        config_json=np.asarray(config_json, dtype=np.str_),
        metadata_json=np.asarray(metadata_json, dtype=np.str_),
        state_u_bar=state_u_bar,
        state_displacement=state_displacement,
        state_damage=state_damage,
        state_history=state_history,
        state_top_reaction=state_top_reaction,
        cumulative_top_reaction_work=cumulative_top_reaction_work,
        accepted_u_bar=accepted_u_bar,
        accepted_load_start=accepted_load_start,
        accepted_increment=accepted_increment,
        accepted_cutbacks=accepted_cutbacks,
        accepted_stagger_iterations=accepted_stagger_iterations,
        accepted_history_used=accepted_history_used,
        accepted_H_used=accepted_H_used,
        accepted_H_next=accepted_H_next,
        accepted_amor_driver=accepted_amor_driver,
        accepted_amor_edge_positive_energy=accepted_amor_edge_positive_energy,
        accepted_amor_particle_driver=accepted_amor_particle_driver,
        accepted_amor_previous_history=accepted_amor_previous_history,
        accepted_amor_trial_history=accepted_amor_trial_history,
        accepted_fixed_history_damage=accepted_fixed_history_damage,
        accepted_fixed_history_history=accepted_fixed_history_history,
        accepted_fixed_history_crack_energy=accepted_fixed_history_crack_energy,
        accepted_fixed_history_history_energy=accepted_fixed_history_history_energy,
        accepted_fixed_history_graph_energy=accepted_fixed_history_graph_energy,
        accepted_fixed_history_total_energy=accepted_fixed_history_total_energy,
        accepted_fixed_history_strong_residual=accepted_fixed_history_strong_residual,
        accepted_fixed_history_euclidean_residual=accepted_fixed_history_euclidean_residual,
        accepted_fixed_history_strong_residual_norm=accepted_fixed_history_strong_residual_norm,
        accepted_fixed_history_strong_residual_inf=accepted_fixed_history_strong_residual_inf,
        accepted_fixed_history_euclidean_residual_norm=accepted_fixed_history_euclidean_residual_norm,
        accepted_fixed_history_euclidean_residual_inf=accepted_fixed_history_euclidean_residual_inf,
        accepted_fixed_history_damage_min=accepted_fixed_history_damage_min,
        accepted_fixed_history_damage_max=accepted_fixed_history_damage_max,
        accepted_fixed_history_below_zero_count=accepted_fixed_history_below_zero_count,
        accepted_fixed_history_above_one_count=accepted_fixed_history_above_one_count,
        accepted_fixed_history_symmetry_defect=accepted_fixed_history_symmetry_defect,
        accepted_fixed_history_symmetry_relative_defect=accepted_fixed_history_symmetry_relative_defect,
        accepted_committed_history_energy=accepted_committed_history_energy,
        accepted_committed_damage_total_energy=accepted_committed_damage_total_energy,
        accepted_damage_residual=accepted_damage_residual,
        accepted_damage_before=accepted_damage_before,
        accepted_mechanics_displacement=accepted_mechanics_displacement,
        accepted_mechanics_before_displacement=accepted_mechanics_before_displacement,
        accepted_mechanics_before_free_residual=accepted_mechanics_before_free_residual,
        accepted_mechanics_energy=accepted_mechanics_energy,
        accepted_mechanics_residual=accepted_mechanics_residual,
        accepted_mechanics_internal_force=accepted_mechanics_internal_force,
        accepted_edge_F=accepted_edge_F,
        accepted_edge_strain=accepted_edge_strain,
        accepted_edge_stress=accepted_edge_stress,
        accepted_edge_effective_stress=accepted_edge_effective_stress,
        accepted_edge_energy_density=accepted_edge_energy_density,
        accepted_particle_strain=accepted_particle_strain,
        accepted_particle_stress=accepted_particle_stress,
        accepted_particle_effective_stress=accepted_particle_effective_stress,
        accepted_reaction=accepted_reaction,
        accepted_top_reaction=accepted_reaction,
        accepted_bottom_reaction=accepted_bottom_reaction,
        accepted_anchor_reaction=accepted_anchor_reaction,
        accepted_cg_info=accepted_cg_info,
        accepted_cg_iterations=accepted_cg_iterations,
        accepted_force=accepted_force,
        accepted_moment=accepted_moment,
        accepted_free_residual_norm=accepted_free_residual_norm,
        accepted_free_rhs_norm=accepted_free_rhs_norm,
        accepted_initial_free_residual_norm=accepted_initial_free_residual_norm,
        accepted_cg_convergence_limit=accepted_cg_convergence_limit,
        accepted_initial_guess_supplied=accepted_initial_guess_supplied,
        accepted_cg_initial_guess_used=accepted_cg_initial_guess_used,
        accepted_early_exit=accepted_early_exit,
        accepted_free_residual_relative_norm=accepted_free_residual_relative_norm,
        accepted_free_residual=accepted_free_residual,
        accepted_top_reaction_work_increment=accepted_top_reaction_work_increment,
        accepted_e_u=accepted_e_u,
        accepted_e_d=accepted_e_d,
        accepted_e_H=accepted_e_H,
        accepted_e_R=accepted_e_R,
        accepted_e_rho=accepted_e_rho,
        accepted_u_scale=accepted_u_scale,
        accepted_u_limit=accepted_u_limit,
        accepted_d_scale=accepted_d_scale,
        accepted_d_limit=accepted_d_limit,
        accepted_H_scale=accepted_H_scale,
        accepted_H_limit=accepted_H_limit,
        accepted_R_scale=accepted_R_scale,
        accepted_R_limit=accepted_R_limit,
        accepted_rho_scale=accepted_rho_scale,
        accepted_rho_limit=accepted_rho_limit,
        accepted_force_error=accepted_force_error,
        accepted_force_scale=accepted_force_scale,
        accepted_force_limit=accepted_force_limit,
        accepted_moment_error=accepted_moment_error,
        accepted_moment_scale=accepted_moment_scale,
        accepted_moment_limit=accepted_moment_limit,
        accepted_u_pass=accepted_u_pass,
        accepted_d_pass=accepted_d_pass,
        accepted_H_pass=accepted_H_pass,
        accepted_R_pass=accepted_R_pass,
        accepted_rho_pass=accepted_rho_pass,
        accepted_force_pass=accepted_force_pass,
        accepted_moment_pass=accepted_moment_pass,
        accepted_rho_used=accepted_rho_used,
        accepted_rho_used_volume=accepted_rho_used_volume,
        accepted_rho_next=accepted_rho_next,
        accepted_rho_next_volume=accepted_rho_next_volume,
        accepted_damage_min=accepted_damage_min,
        accepted_damage_max=accepted_damage_max,
        accepted_healing_min=accepted_healing_min,
        accepted_history_min_gap=accepted_history_min_gap,
        accepted_reaction_before=accepted_reaction_before,
        accepted_reaction_after=accepted_reaction_after,
        accepted_mechanics_before_free_residual_norm=accepted_mechanics_before_free_residual_norm,
        accepted_mechanics_after_free_residual_norm=accepted_mechanics_after_free_residual_norm,
        accepted_mechanics_before_free_residual_relative_norm=accepted_mechanics_before_free_residual_relative_norm,
        accepted_mechanics_after_free_residual_relative_norm=accepted_mechanics_after_free_residual_relative_norm,
        accepted_mechanics_before_cg_info=accepted_mechanics_before_cg_info,
        accepted_mechanics_after_cg_info=accepted_mechanics_after_cg_info,
        accepted_force_before=accepted_force_before,
        accepted_force_after=accepted_force_after,
        accepted_moment_before=accepted_moment_before,
        accepted_moment_after=accepted_moment_after,
        accepted_history_monotonic_pass=accepted_history_monotonic_pass,
        accepted_damage_range_pass=accepted_damage_range_pass,
        accepted_healing_pass=accepted_healing_pass,
        accepted_mechanics_pass=accepted_mechanics_pass,
        accepted_gate_pass=accepted_gate_pass,
        attempt_audit_json=np.asarray(json.dumps(attempt_audit), dtype=np.str_),
        accepted_audit_json=np.asarray(json.dumps(accepted_audit), dtype=np.str_),
    )
