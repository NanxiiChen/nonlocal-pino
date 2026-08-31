"""Strict clean-start Candidate A2/material-only fracture coupling.

This is a deliberately separate NumPy/SciPy diagnostic path.  Mechanics uses
the project-owned phase-measured A2 energy on the unequal union cloud; damage
and history remain on the original material cloud and its frozen cubic
``ExplicitVolumeSparseD`` operator.  The module has no warning/permissive
acceptance mode and does not modify the material-only continuation controller.

The bond-associated support, plane-strain constitutive map, and Amor positive
energy are the source-backed basis already audited for the fracture2d direct
baseline.  The phase-measured center energy, outer-face constraints,
matched-interface opening, strict stagger, and archive/replay contracts are
project-owned choices declared in
``docs/tasks/fracture2d-a2-coupled-formulation.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from pinn.operators import ExplicitVolumeSparseD

from . import nonlocal_baseline as _material
from .nonlocal_grip_diagnostic import (
    A2StaticMechanics,
    AffineConstraints,
    LineQuadrature,
    UnionGripGeometry,
    _a2_hash_primitives,
    _immutable as _diagnostic_immutable,
    build_affine_constraints,
    independent_union_ba_reference,
    physical_interface_line_weights,
)
from .nonlocal_mechanics import MechanicsEvaluation


__all__ = [
    "A2AcceptedStep",
    "A2AmorHistoryResult",
    "A2AttemptRecord",
    "A2CommittedState",
    "A2ContinuationFailure",
    "A2ContinuationMetadata",
    "A2ContinuationResult",
    "A2CoupledConfig",
    "A2CoupledModel",
    "A2Diagnostics",
    "A2FixedDamageView",
    "A2MechanicsSolveResult",
    "A2OpeningMechanicsResult",
    "A2SeedProvenance",
    "A2StaticMechanics",
    "A2StrictConfig",
    "A2SweepFailure",
    "A2_ACCEPTANCE_MODE",
    "A2_ACCEPTANCE_STATUS",
    "A2_COMPLETION_STATUS",
    "ARCHIVE_SCHEMA",
    "GateTolerance",
    "build_a2_coupled_model",
    "build_a2_unit_seed",
    "compute_a2_amor_history",
    "compute_a2_amor_history_from_edge_strain",
    "independent_replay",
    "load_a2_archive",
    "physical_interface_opening_a2",
    "replay_a2_archive",
    "run_a2_strict_continuation",
    "run_strict_a2_continuation",
    "scale_a2_unit_solution",
    "solve_a2_fixed_outer_displacement",
    "solve_a2_opening_controlled",
    "solve_a2_unit_load",
    "write_a2_archive",
]


_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)
_BOOL = np.dtype(bool)
_IDENTITY_2 = np.eye(2, dtype=np.float64)
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_EPS64 = np.finfo(np.float64).eps
_REPLAY_BOUND = 512.0 * _EPS64
ARCHIVE_SCHEMA = "fracture2d-a2-coupled-v1"
A2_ACCEPTANCE_MODE = "strict"
A2_ACCEPTANCE_STATUS = "strict_passed"
A2_COMPLETION_STATUS = "completed"
_A2_SWEEP_FAILURE_STAGES = frozenset(
    {
        "mechanics_before",
        "mechanics_before_gate",
        "damage_acceptance",
        "mechanics_after",
        "mechanics_after_gate",
    }
)
_KERNEL = "cubic"
_VISIBILITY_RULE = "independent union-polygon segment containment"

# The public GateTolerance is intentionally the already audited arithmetic
# type.  It is not imported by or used to alter the material-only controller.
GateTolerance = _material.GateTolerance
FixedHistoryDamageResult = _material.FixedHistoryDamageResult


def _immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Make a bytes-backed immutable copy."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result


def _scalar(value: Any, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be scalar, got shape {raw.shape}")
    try:
        result = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _f64(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _i64(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _INT64:
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _bool(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _BOOL:
        raise ValueError(f"{name} must use exact bool dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _relative(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), 1.0
    )


def _max_abs(values: Any) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _array_close(first: Any, second: Any, name: str, limit: float = 1.0e-10) -> None:
    left = np.asarray(first)
    right = np.asarray(second)
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f"{name} shape/dtype does not match the authoritative replay")
    defect = _max_abs(left - right)
    scale = max(_max_abs(left), _max_abs(right), 1.0)
    if defect > limit * scale:
        raise ValueError(f"{name} does not match the authoritative replay: defect={defect}")


def _digest_primitives(values: Iterable[tuple[str, Any]]) -> str:
    return _a2_hash_primitives(tuple(values))


def _digest_state(
    physical_opening: float,
    outer_displacement: float,
    union_displacement: np.ndarray,
    material_damage: np.ndarray,
    material_history: np.ndarray,
) -> str:
    return _digest_primitives(
        (
            ("physical_opening", np.asarray(physical_opening, dtype=np.float64)),
            ("outer_displacement", np.asarray(outer_displacement, dtype=np.float64)),
            ("union_displacement", union_displacement),
            ("material_damage", material_damage),
            ("material_history", material_history),
        )
    )


def _damage_digest(damage: np.ndarray) -> str:
    return _digest_primitives((("material_damage", damage),))


def _unit_system_digest(static_digest: str, damage_digest: str) -> str:
    return _digest_primitives(
        (("static_digest", static_digest), ("damage_digest", damage_digest), ("unit", 1.0))
    )


def _static_digest_from_object(static: Any) -> str:
    """Recompute the frozen static digest from its declared primitive fields."""

    return _digest_primitives(
        (
            ("coordinates", static.coordinates),
            ("union_volumes", static.union_volumes),
            ("directed_edges", static.directed_edges),
            ("source", static.source),
            ("target", static.target),
            ("edge_distance", static.edge_distance),
            ("edge_omega", static.edge_omega),
            ("family_mass", static.family_mass),
            ("eta", static.eta),
            ("support_indices", static.support_indices),
            ("support_mask", static.support_mask),
            ("support_sizes", static.support_sizes),
            ("beta", static.beta),
            ("xi", static.xi),
            ("shape_inverse", static.shape_inverse),
            ("shape_rank", static.shape_rank),
            ("shape_condition", static.shape_condition),
            ("target_beta", static.target_beta),
            ("gradient_coefficient", static.gradient_coefficient),
            ("youngs_modulus", static.youngs_modulus),
            ("poisson_ratio", static.poisson_ratio),
            ("lame_lambda", static.lame_lambda),
            ("shear_modulus", static.shear_modulus),
            ("k_res", static.k_res),
            ("rank_rtol", static.rank_rtol),
            ("delta", static.delta),
            ("h_g", static.h_g),
            ("material_count", static.material_count),
            ("material_prefix_map", static.material_prefix_map),
            ("material_damage_volumes", static.material_damage_volumes),
            ("union_material_phase_volumes", static.union_material_phase_volumes),
            ("grip_phase_volumes", static.grip_phase_volumes),
            ("constrained_dofs", static.constrained_dofs),
            ("free_dofs", static.free_dofs),
            ("q_zero", static.q_zero),
            ("load_direction", static.load_direction),
            ("outer_top_selector", static.outer_top_selector),
            ("outer_bottom_selector", static.outer_bottom_selector),
            ("anchor_selector", static.anchor_selector),
            ("outer_bottom_mask", static.outer_bottom_mask),
            ("outer_top_mask", static.outer_top_mask),
            ("anchor_particle", static.anchor_particle),
            ("top_line_indices", static.top_line_indices),
            ("top_line_weights", static.top_line_weights),
            ("top_line_length", static.top_line_length),
            ("bottom_line_indices", static.bottom_line_indices),
            ("bottom_line_weights", static.bottom_line_weights),
            ("bottom_line_length", static.bottom_line_length),
        )
    )


def _check_exact_constraints(
    displacement: np.ndarray, constraints: AffineConstraints, U_ext: float, name: str
) -> None:
    prescribed = constraints.prescribed_vector(U_ext)
    if not np.array_equal(
        displacement[constraints.constrained_dofs], prescribed[constraints.constrained_dofs]
    ):
        raise ValueError(f"{name} does not satisfy exact constrained values")


class A2FixedDamageView:
    """Fresh phase-measured fixed-damage view over one shared static object."""

    _CACHE_FIELDS = frozenset({"_tangent_block_diagonal_cache", "_tangent_diagonal_cache"})

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_locked", False) and name not in self._CACHE_FIELDS:
            raise AttributeError("A2FixedDamageView is immutable; construct a fresh view")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_locked", False) and name not in self._CACHE_FIELDS:
            raise AttributeError("A2FixedDamageView is immutable; construct a fresh view")
        object.__delattr__(self, name)

    def __init__(self, static: A2StaticMechanics, material_damage: Any) -> None:
        if not isinstance(static, A2StaticMechanics):
            raise TypeError("static must be an A2StaticMechanics instance")
        if _static_digest_from_object(static) != static.deterministic_digest:
            raise ValueError("A2 static primitive digest does not match deterministic_digest")
        damage = _f64(material_damage, "material_damage", (static.material_count,))
        self.static = static
        self.material_damage = _immutable(damage)
        self.damage = self.material_damage
        self.damage_digest = _damage_digest(self.material_damage)
        self.material_damage_runtime_identity = object()
        self.runtime_identity = (static.runtime_identity, self.material_damage_runtime_identity)
        self.static_digest = static.deterministic_digest

        material_degradation = (1.0 - damage) ** 2 + static.k_res
        grip_degradation = 1.0 + static.k_res
        material_full = np.pad(
            material_degradation,
            (0, static.n_particles - static.material_count),
        )
        center_weights = np.asarray(
            static.union_material_phase_volumes * material_full
            + static.grip_phase_volumes * grip_degradation,
            dtype=np.float64,
        )
        effective = center_weights / static.union_volumes
        if not np.all(np.isfinite(center_weights)) or np.any(center_weights <= 0.0):
            raise FloatingPointError("A2 phase center weights are invalid")
        if not np.all(np.isfinite(effective)) or np.any(effective <= 0.0):
            raise FloatingPointError("A2 effective degradation is invalid")
        self.damage_union = _immutable(
            np.concatenate((damage, np.zeros(static.n_particles - static.material_count)))
        )
        self.material_degradation = _immutable(material_degradation)
        self.grip_degradation = float(grip_degradation)
        self.center_weights = _immutable(center_weights)
        self.effective_center_degradation = _immutable(effective)
        self.degradation = self.effective_center_degradation
        self.edge_degradation = _immutable(effective[static.source])
        self.center_factor = _immutable(static.source * 0.0 + center_weights[static.source] * static.eta)
        self.center_factor_vstar_reference = _immutable(
            static.union_volumes[static.source] * effective[static.source] * static.eta
        )
        self.damage_min = float(np.min(damage))
        self.damage_max = float(np.max(damage))
        self.damage_below_zero_count = int(np.count_nonzero(damage < 0.0))
        self.damage_above_one_count = int(np.count_nonzero(damage > 1.0))
        self._locked = True

    @property
    def n_particles(self) -> int:
        return self.static.n_particles

    @property
    def n_material(self) -> int:
        return self.static.material_count

    @property
    def n_edges(self) -> int:
        return self.static.n_edges

    @property
    def n_dofs(self) -> int:
        return self.static.n_dofs

    @property
    def coordinates(self) -> np.ndarray:
        return self.static.coordinates

    @property
    def volumes(self) -> np.ndarray:
        return self.static.union_volumes

    @property
    def directed_edges(self) -> np.ndarray:
        return self.static.directed_edges

    @property
    def edges(self) -> np.ndarray:
        return self.static.directed_edges

    @property
    def source(self) -> np.ndarray:
        return self.static.source

    @property
    def target(self) -> np.ndarray:
        return self.static.target

    @property
    def eta(self) -> np.ndarray:
        return self.static.eta

    @property
    def lame_lambda(self) -> float:
        return self.static.lame_lambda

    @property
    def shear_modulus(self) -> float:
        return self.static.shear_modulus

    @property
    def youngs_modulus(self) -> float:
        return self.static.youngs_modulus

    @property
    def poisson_ratio(self) -> float:
        return self.static.poisson_ratio

    @property
    def k_res(self) -> float:
        return self.static.k_res

    @property
    def delta(self) -> float:
        return self.static.delta

    @property
    def rank_rtol(self) -> float:
        return self.static.rank_rtol

    @property
    def support_indices(self) -> np.ndarray:
        return self.static.support_indices

    @property
    def support_mask(self) -> np.ndarray:
        return self.static.support_mask

    @property
    def support_sizes(self) -> np.ndarray:
        return self.static.support_sizes

    @property
    def beta(self) -> np.ndarray:
        return self.static.beta

    @property
    def xi(self) -> np.ndarray:
        return self.static.xi

    @property
    def shape_inverse(self) -> np.ndarray:
        return self.static.shape_inverse

    @property
    def tangent_block_diagonal(self) -> np.ndarray:
        return self._get_tangent_block_diagonal()

    @property
    def tangent_diagonal(self) -> np.ndarray:
        return self._get_tangent_diagonal()

    def _safe_support(self) -> np.ndarray:
        return np.where(self.static.support_mask, self.static.support_indices, 0)

    def _deformation_gradient(self, displacement: np.ndarray) -> np.ndarray:
        support = self._safe_support()
        values = displacement.reshape(self.n_particles, 2)
        source_values = values[self.static.source[:, None]]
        current = self.static.xi + values[support] - source_values
        moment = np.einsum(
            "es,esa,esb->eab", self.static.beta, current, self.static.xi, optimize=True
        )
        return np.einsum("eab,ebc->eac", moment, self.static.shape_inverse, optimize=True)

    def _strain_stress(self, deformation_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        increment = deformation_gradient - _IDENTITY_2
        strain = 0.5 * (increment + np.swapaxes(increment, 1, 2))
        trace = np.trace(strain, axis1=1, axis2=2)
        stress = (
            self.lame_lambda * trace[:, None, None] * _IDENTITY_2[None, :, :]
            + 2.0 * self.shear_modulus * strain
        )
        return strain, stress

    def _density(self, strain: np.ndarray) -> np.ndarray:
        trace = np.trace(strain, axis1=1, axis2=2)
        return 0.5 * self.lame_lambda * trace**2 + self.shear_modulus * np.sum(
            strain * strain, axis=(1, 2)
        )

    def _endpoint_residual(self, stress: np.ndarray) -> np.ndarray:
        endpoint = np.einsum(
            "eab,ebc,esc->esa",
            stress,
            np.swapaxes(self.static.shape_inverse, 1, 2),
            self.static.xi,
            optimize=True,
        )
        endpoint *= self.static.beta[:, :, None] * self.center_factor[:, None, None]
        support = self._safe_support().reshape(-1)
        support_result = np.column_stack(
            (
                np.bincount(
                    support, weights=endpoint[:, :, 0].reshape(-1), minlength=self.n_particles
                ),
                np.bincount(
                    support, weights=endpoint[:, :, 1].reshape(-1), minlength=self.n_particles
                ),
            )
        )
        source_result = np.column_stack(
            (
                np.bincount(
                    self.static.source,
                    weights=-np.sum(endpoint[:, :, 0], axis=1),
                    minlength=self.n_particles,
                ),
                np.bincount(
                    self.static.source,
                    weights=-np.sum(endpoint[:, :, 1], axis=1),
                    minlength=self.n_particles,
                ),
            )
        )
        return (support_result + source_result).reshape(-1)

    def _delta_gradient(self, direction: np.ndarray) -> np.ndarray:
        support = self._safe_support()
        values = direction.reshape(self.n_particles, 2)
        differences = values[support] - values[self.static.source[:, None]]
        return np.einsum(
            "esa,esc->eac", differences, self.static.gradient_coefficient, optimize=True
        )

    def _constitutive_tangent(self, delta_gradient: np.ndarray) -> np.ndarray:
        delta_strain = 0.5 * (delta_gradient + np.swapaxes(delta_gradient, 1, 2))
        trace = np.trace(delta_strain, axis1=1, axis2=2)
        return (
            self.lame_lambda * trace[:, None, None] * _IDENTITY_2[None, :, :]
            + 2.0 * self.shear_modulus * delta_strain
        )

    def _get_tangent_block_diagonal(self) -> np.ndarray:
        cached = getattr(self, "_tangent_block_diagonal_cache", None)
        if cached is not None:
            return cached
        coefficient = self.static.gradient_coefficient
        cx = coefficient[:, :, 0]
        cy = coefficient[:, :, 1]
        factor = self.center_factor[:, None]
        hxx = factor * (
            (self.lame_lambda + 2.0 * self.shear_modulus) * cx * cx
            + self.shear_modulus * cy * cy
        )
        hxy = factor * (self.lame_lambda + self.shear_modulus) * cx * cy
        hyy = factor * (
            self.shear_modulus * cx * cx
            + (self.lame_lambda + 2.0 * self.shear_modulus) * cy * cy
        )
        support = self._safe_support().reshape(-1)
        blocks = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        blocks[:, 0, 0] = np.bincount(support, weights=hxx.reshape(-1), minlength=self.n_particles)
        blocks[:, 0, 1] = np.bincount(support, weights=hxy.reshape(-1), minlength=self.n_particles)
        blocks[:, 1, 0] = blocks[:, 0, 1]
        blocks[:, 1, 1] = np.bincount(support, weights=hyy.reshape(-1), minlength=self.n_particles)
        source_coefficient = np.sum(coefficient, axis=1)
        scx, scy = source_coefficient[:, 0], source_coefficient[:, 1]
        source_hxx = self.center_factor * (
            (self.lame_lambda + 2.0 * self.shear_modulus) * scx**2
            + self.shear_modulus * scy**2
        )
        source_hxy = self.center_factor * (self.lame_lambda + self.shear_modulus) * scx * scy
        source_hyy = self.center_factor * (
            self.shear_modulus * scx**2
            + (self.lame_lambda + 2.0 * self.shear_modulus) * scy**2
        )
        blocks[:, 0, 0] += np.bincount(self.static.source, weights=source_hxx, minlength=self.n_particles)
        source_xy = np.bincount(self.static.source, weights=source_hxy, minlength=self.n_particles)
        blocks[:, 0, 1] += source_xy
        blocks[:, 1, 0] += source_xy
        blocks[:, 1, 1] += np.bincount(self.static.source, weights=source_hyy, minlength=self.n_particles)
        if not np.all(np.isfinite(blocks)):
            raise FloatingPointError("A2 tangent block cache is non-finite")
        for block in blocks:
            if np.any(np.linalg.eigvalsh(block) <= 0.0):
                raise ValueError("A2 tangent block cache is not positive definite")
        result = _immutable(blocks)
        self._tangent_block_diagonal_cache = result
        return result

    def _get_tangent_diagonal(self) -> np.ndarray:
        cached = getattr(self, "_tangent_diagonal_cache", None)
        if cached is not None:
            return cached
        blocks = self._get_tangent_block_diagonal()
        diagonal = np.empty(self.n_dofs, dtype=np.float64)
        diagonal[0::2] = blocks[:, 0, 0]
        diagonal[1::2] = blocks[:, 1, 1]
        result = _immutable(diagonal)
        self._tangent_diagonal_cache = result
        return result

    def energy(self, displacement: Any) -> float:
        q = _f64(displacement, "A2 displacement", (self.n_dofs,))
        strain, _ = self._strain_stress(self._deformation_gradient(q))
        value = float(np.sum(self.center_factor * self._density(strain)))
        if not np.isfinite(value):
            raise FloatingPointError("A2 mechanics energy is non-finite")
        return value

    def residual(self, displacement: Any) -> np.ndarray:
        q = _f64(displacement, "A2 displacement", (self.n_dofs,))
        _, stress = self._strain_stress(self._deformation_gradient(q))
        result = self._endpoint_residual(stress)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("A2 mechanics residual is non-finite")
        return result

    def internal_force(self, displacement: Any) -> np.ndarray:
        return -self.residual(displacement)

    def tangent_action(self, direction: Any) -> np.ndarray:
        values = _f64(direction, "A2 direction", (self.n_dofs,))
        result = self._endpoint_residual(self._constitutive_tangent(self._delta_gradient(values)))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("A2 tangent action is non-finite")
        return result

    def evaluate(self, displacement: Any) -> MechanicsEvaluation:
        q = _f64(displacement, "A2 displacement", (self.n_dofs,))
        deformation_gradient = self._deformation_gradient(q)
        strain, stress = self._strain_stress(deformation_gradient)
        density = self._density(strain)
        residual = self._endpoint_residual(stress)
        particle_strain = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        particle_stress = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        np.add.at(particle_strain, self.static.source, self.static.eta[:, None, None] * strain)
        np.add.at(particle_stress, self.static.source, self.static.eta[:, None, None] * stress)
        edge_effective = self.edge_degradation[:, None, None] * stress
        particle_effective = self.effective_center_degradation[:, None, None] * particle_stress
        energy = float(np.sum(self.center_factor * density))
        if not np.isfinite(energy) or not np.all(np.isfinite(residual)):
            raise FloatingPointError("A2 mechanics evaluation is non-finite")
        return MechanicsEvaluation(
            energy=energy,
            residual=residual,
            internal_force=-residual,
            edge_deformation_gradient=deformation_gradient,
            edge_strain=strain,
            edge_stress=stress,
            particle_strain=particle_strain,
            particle_stress=particle_stress,
            edge_effective_stress=edge_effective,
            particle_effective_stress=particle_effective,
            edge_energy_density=density,
        )


@dataclass(frozen=True)
class A2SeedProvenance:
    """Auditable normalized mechanics seed provenance."""

    kind: str
    source_state: str
    source_step: int
    source_sweep: int
    source_damage_digest: str
    target_damage_digest: str
    unit_system_digest: str
    seed_digest: str
    initial_residual_norm: float
    final_residual_norm: float
    early_exit_eligible: bool
    early_exit_used: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("A2SeedProvenance.kind must be non-empty")
        if not isinstance(self.source_state, str) or not self.source_state:
            raise ValueError("A2SeedProvenance.source_state must be non-empty")
        for name in (
            "source_damage_digest", "target_damage_digest", "unit_system_digest", "seed_digest"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"A2SeedProvenance.{name} must be a SHA256 digest")
        for name in ("source_step", "source_sweep"):
            object.__setattr__(self, name, _integer(getattr(self, name), f"A2SeedProvenance.{name}"))
        for name in ("initial_residual_norm", "final_residual_norm"):
            value = _scalar(getattr(self, name), f"A2SeedProvenance.{name}")
            if value < 0.0:
                raise ValueError(f"A2SeedProvenance.{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "early_exit_eligible", bool(self.early_exit_eligible))
        object.__setattr__(self, "early_exit_used", bool(self.early_exit_used))
        if self.early_exit_used and not self.early_exit_eligible:
            raise ValueError("A2 seed early exit cannot be used when it is ineligible")


@dataclass(frozen=True)
class A2MechanicsSolveResult:
    """One exact-constraint fixed-damage mechanics solve."""

    displacement: np.ndarray
    evaluation: MechanicsEvaluation
    constraints: AffineConstraints
    outer_displacement: float
    free_residual: np.ndarray
    free_residual_norm: float
    free_rhs_norm: float
    free_residual_relative_norm: float
    cg_convergence_limit: float
    cg_iterations: int
    cg_info: int
    outer_reaction: float
    bottom_reaction: float
    anchor_reaction: float
    total_force: np.ndarray
    total_moment: float
    static_runtime_identity: object
    static_digest: str
    damage_runtime_identity: object
    damage_digest: str
    unit_system_digest: str
    seed_provenance: A2SeedProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, MechanicsEvaluation):
            raise TypeError("A2MechanicsSolveResult.evaluation must be MechanicsEvaluation")
        if not isinstance(self.constraints, AffineConstraints):
            raise TypeError("A2MechanicsSolveResult.constraints must be AffineConstraints")
        n = self.constraints.n_dofs
        for name, shape in (
            ("displacement", (n,)),
            ("free_residual", (len(self.constraints.free_dofs),)),
            ("total_force", (2,)),
        ):
            object.__setattr__(self, name, _immutable(_f64(getattr(self, name), name, shape)))
        for name in (
            "outer_displacement", "free_residual_norm", "free_rhs_norm",
            "free_residual_relative_norm", "cg_convergence_limit", "outer_reaction",
            "bottom_reaction", "anchor_reaction", "total_moment",
        ):
            value = _scalar(getattr(self, name), name)
            if name.endswith("norm") and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.free_rhs_norm < 0.0 or self.cg_convergence_limit <= 0.0:
            raise ValueError("A2 mechanics residual scales are invalid")
        if abs(self.free_residual_norm - np.linalg.norm(self.free_residual)) > _REPLAY_BOUND * max(1.0, self.free_residual_norm):
            raise ValueError("A2 free residual norm is inconsistent")
        expected_relative = self.free_residual_norm / max(self.free_rhs_norm, 1.0e-300)
        if abs(self.free_residual_relative_norm - expected_relative) > _REPLAY_BOUND * max(1.0, abs(expected_relative)):
            raise ValueError("A2 free residual relative norm is inconsistent")
        if self.free_residual_norm >= self.cg_convergence_limit:
            raise ValueError("A2 mechanics solve failed its authoritative free residual gate")
        for name in ("cg_iterations", "cg_info"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.static_digest, str) or len(self.static_digest) != 64:
            raise ValueError("A2 mechanics static digest is invalid")
        if not isinstance(self.damage_digest, str) or len(self.damage_digest) != 64:
            raise ValueError("A2 mechanics damage digest is invalid")
        if not isinstance(self.unit_system_digest, str) or len(self.unit_system_digest) != 64:
            raise ValueError("A2 mechanics unit-system digest is invalid")
        expected_unit_digest = _unit_system_digest(self.static_digest, self.damage_digest)
        if self.unit_system_digest != expected_unit_digest:
            raise ValueError("A2 mechanics unit-system digest does not match static/damage digests")
        if not isinstance(self.seed_provenance, A2SeedProvenance):
            raise TypeError("A2 mechanics seed provenance is invalid")
        if self.seed_provenance.target_damage_digest != self.damage_digest:
            raise ValueError("A2 mechanics seed target damage digest is not paired")
        if self.seed_provenance.unit_system_digest != self.unit_system_digest:
            raise ValueError("A2 mechanics seed unit-system digest is not paired")

    @property
    def U_ext(self) -> float:
        return self.outer_displacement

    @property
    def energy(self) -> float:
        return self.evaluation.energy

    @property
    def residual(self) -> np.ndarray:
        return self.evaluation.residual

    @property
    def R_ext(self) -> float:
        return self.outer_reaction


def build_a2_unit_seed(
    constraints: AffineConstraints,
    q_previous: np.ndarray | None,
    U_previous: Any,
) -> np.ndarray:
    """Build a target-compatible normalized unit seed without constrained division."""

    if not isinstance(constraints, AffineConstraints):
        raise TypeError("constraints must be an AffineConstraints instance")
    previous_outer = _scalar(U_previous, "U_previous")
    prescribed = constraints.prescribed_vector(1.0)
    if q_previous is None:
        if previous_outer != 0.0:
            raise ValueError("q_previous is required for a nonzero previous outer displacement")
        seed = prescribed
    else:
        q = _f64(q_previous, "q_previous", (constraints.n_dofs,))
        _check_exact_constraints(q, constraints, previous_outer, "q_previous")
        if previous_outer == 0.0:
            if not np.array_equal(q, np.zeros_like(q)):
                raise ValueError("zero previous outer displacement requires exact zero q_previous")
            seed = prescribed
        else:
            seed = np.array(prescribed, copy=True)
            seed[constraints.free_dofs] = q[constraints.free_dofs] / previous_outer
    _check_exact_constraints(seed, constraints, 1.0, "normalized A2 mechanics seed")
    if not np.all(np.isfinite(seed)):
        raise FloatingPointError("normalized A2 mechanics seed is non-finite")
    return _immutable(seed)


def _seed_record(
    kind: str,
    seed: np.ndarray,
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    initial_residual_norm: float,
    final_residual_norm: float,
    *,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
) -> A2SeedProvenance:
    return A2SeedProvenance(
        kind=kind,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
        target_damage_digest=view.damage_digest,
        unit_system_digest=_unit_system_digest(view.static.deterministic_digest, view.damage_digest),
        seed_digest=_digest_primitives((("seed", seed),)),
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=final_residual_norm,
        early_exit_eligible=False,
        early_exit_used=False,
    )


def _validate_view_constraints(view: A2FixedDamageView, constraints: AffineConstraints) -> None:
    if not isinstance(constraints, AffineConstraints):
        raise TypeError("constraints must be an AffineConstraints instance")
    static = view.static
    if constraints.n_particles != static.n_particles:
        raise ValueError("A2 view and constraints have different particle counts")
    for name in (
        "constrained_dofs", "free_dofs", "q_zero", "load_direction",
        "outer_top_selector", "outer_bottom_selector", "anchor_selector",
        "outer_bottom_mask", "outer_top_mask",
    ):
        if not np.array_equal(getattr(constraints, name), getattr(static, name)):
            raise ValueError(f"A2 constraints do not match the static identity field {name}")
    if constraints.anchor_particle != static.anchor_particle:
        raise ValueError("A2 constraints anchor does not match the static identity")


def _solve_outer(
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    U_ext: float,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int,
    initial_guess: np.ndarray | None,
    seed_kind: str,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
) -> A2MechanicsSolveResult:
    _validate_view_constraints(view, constraints)
    load = _scalar(U_ext, "outer_displacement")
    rtol = _scalar(cg_rtol, "cg_rtol")
    atol = _scalar(cg_atol, "cg_atol")
    maxiter = _integer(cg_maxiter, "cg_maxiter", positive=True)
    if rtol < 0.0 or atol < 0.0 or (rtol == 0.0 and atol == 0.0):
        raise ValueError("cg tolerances must be non-negative and not both zero")
    prescribed = constraints.prescribed_vector(load)
    if initial_guess is None:
        seed = _immutable(prescribed)
        supplied = False
    else:
        seed = _f64(initial_guess, "A2 initial_guess", (view.n_dofs,))
        _check_exact_constraints(seed, constraints, load, "A2 initial_guess")
        seed = _immutable(seed)
        supplied = True
    free = constraints.free_dofs
    prescribed_residual = view.residual(prescribed)
    rhs = -prescribed_residual[free]
    rhs_norm = float(np.linalg.norm(rhs))
    convergence_limit = max(atol, rtol * rhs_norm)
    initial_residual_norm = float(np.linalg.norm(view.residual(seed)[free]))
    callback_count: list[int] = []

    def inject(values: np.ndarray) -> np.ndarray:
        result = np.zeros(view.n_dofs, dtype=np.float64)
        result[free] = np.asarray(values, dtype=np.float64)
        return result

    def action(values: np.ndarray) -> np.ndarray:
        return view.tangent_action(inject(values))[free]

    if rhs_norm == 0.0:
        free_values = np.zeros(len(free), dtype=np.float64)
        info = 0
    else:
        operator = LinearOperator((len(free), len(free)), matvec=action, dtype=np.float64)
        positions = np.full(view.n_dofs, -1, dtype=np.int64)
        positions[free] = np.arange(len(free), dtype=np.int64)
        inverse_blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for particle, block in enumerate(view.tangent_block_diagonal):
            components = np.flatnonzero(positions[2 * particle : 2 * particle + 2] >= 0)
            if len(components):
                local_positions = positions[2 * particle + components]
                local = block[np.ix_(components, components)]
                if np.any(np.linalg.eigvalsh(local) <= 0.0):
                    raise ValueError("A2 free tangent preconditioner is not positive definite")
                inverse_blocks.append((local_positions, np.linalg.inv(local)))

        def precondition(values: np.ndarray) -> np.ndarray:
            result = np.zeros_like(values, dtype=np.float64)
            for local_positions, inverse in inverse_blocks:
                result[local_positions] = inverse @ values[local_positions]
            return result

        preconditioner = LinearOperator((len(free), len(free)), matvec=precondition, dtype=np.float64)

        def callback(_: np.ndarray) -> None:
            callback_count.append(1)

        free_values, info = cg(
            operator,
            rhs,
            x0=seed[free],
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            M=preconditioner,
            callback=callback,
        )
        info = int(info)
        if info != 0:
            raise RuntimeError(
                f"A2 fixed-damage CG did not converge: info={info}, iterations={len(callback_count)}"
            )
    displacement = np.asarray(prescribed, dtype=np.float64)
    displacement[free] += np.asarray(free_values, dtype=np.float64)
    _check_exact_constraints(displacement, constraints, load, "A2 solved displacement")
    evaluation = view.evaluate(displacement)
    free_residual = np.asarray(evaluation.residual[free], dtype=np.float64)
    free_norm = float(np.linalg.norm(free_residual))
    if not np.isfinite(free_norm) or free_norm >= convergence_limit:
        raise RuntimeError(
            f"A2 free residual is outside its authoritative gate: {free_norm} >= {convergence_limit}"
        )
    residual_matrix = evaluation.residual.reshape(view.n_particles, 2)
    total_force = np.sum(residual_matrix, axis=0)
    total_moment = float(
        np.sum(view.coordinates[:, 0] * residual_matrix[:, 1] - view.coordinates[:, 1] * residual_matrix[:, 0])
    )
    if not np.all(np.isfinite(total_force)) or not np.isfinite(total_moment):
        raise FloatingPointError("A2 force/moment diagnostics are non-finite")
    seed_provenance = _seed_record(
        seed_kind,
        seed,
        view,
        constraints,
        initial_residual_norm,
        free_norm,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
    )
    return A2MechanicsSolveResult(
        displacement=displacement,
        evaluation=evaluation,
        constraints=constraints,
        outer_displacement=load,
        free_residual=free_residual,
        free_residual_norm=free_norm,
        free_rhs_norm=rhs_norm,
        free_residual_relative_norm=free_norm / max(rhs_norm, 1.0e-300),
        cg_convergence_limit=convergence_limit,
        cg_iterations=len(callback_count),
        cg_info=info,
        outer_reaction=float(constraints.outer_top_selector @ evaluation.residual),
        bottom_reaction=float(constraints.outer_bottom_selector @ evaluation.residual),
        anchor_reaction=float(constraints.anchor_selector @ evaluation.residual),
        total_force=total_force,
        total_moment=total_moment,
        static_runtime_identity=view.static.runtime_identity,
        static_digest=view.static.deterministic_digest,
        damage_runtime_identity=view.material_damage_runtime_identity,
        damage_digest=view.damage_digest,
        unit_system_digest=_unit_system_digest(view.static.deterministic_digest, view.damage_digest),
        seed_provenance=seed_provenance,
    )


def solve_a2_fixed_outer_displacement(
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    outer_displacement: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: np.ndarray | None,
    seed_kind: str,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
) -> A2MechanicsSolveResult:
    """Solve one fixed-damage outer-displacement problem without scaling."""

    return _solve_outer(
        view,
        constraints,
        _scalar(outer_displacement, "outer_displacement"),
        _scalar(cg_rtol, "cg_rtol"),
        _scalar(cg_atol, "cg_atol"),
        _integer(cg_maxiter, "cg_maxiter", positive=True),
        initial_guess,
        seed_kind,
        source_state,
        _integer(source_step, "source_step"),
        _integer(source_sweep, "source_sweep"),
        source_damage_digest,
    )


def solve_a2_unit_load(
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: np.ndarray | None,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
) -> A2MechanicsSolveResult:
    """Fresh unit outer-displacement solve used by opening control."""

    if initial_guess is None:
        seed = constraints.prescribed_vector(1.0)
        kind = "unit_load_direction"
    else:
        seed = _f64(initial_guess, "unit initial_guess", (view.n_dofs,))
        _check_exact_constraints(seed, constraints, 1.0, "unit initial_guess")
        kind = "normalized_previous_solution"
    result = _solve_outer(
        view,
        constraints,
        1.0,
        _scalar(cg_rtol, "cg_rtol"),
        _scalar(cg_atol, "cg_atol"),
        _integer(cg_maxiter, "cg_maxiter", positive=True),
        seed,
        kind,
        source_state,
        _integer(source_step, "source_step"),
        _integer(source_sweep, "source_sweep"),
        source_damage_digest,
    )
    if result.outer_displacement != 1.0:
        raise ValueError("A2 unit solve did not retain exact outer_displacement=1")
    return result


def physical_interface_opening_a2(
    displacement: Any,
    static: A2StaticMechanics,
) -> float:
    """Evaluate the matched physical material-interface opening."""

    if not isinstance(static, A2StaticMechanics):
        raise TypeError("static must be an A2StaticMechanics instance")
    q = _f64(displacement, "A2 displacement", (static.n_dofs,)).reshape(-1, 2)
    top = float(np.sum(static.top_line_weights * q[static.top_line_indices, 1]) / static.top_line_length)
    bottom = float(np.sum(static.bottom_line_weights * q[static.bottom_line_indices, 1]) / static.bottom_line_length)
    result = top - bottom
    if not np.isfinite(result):
        raise FloatingPointError("A2 physical interface opening is non-finite")
    return result


@dataclass(frozen=True)
class A2OpeningMechanicsResult:
    """Opening-controlled fixed-damage response and both work checks."""

    unit_result: A2MechanicsSolveResult
    mechanics_result: A2MechanicsSolveResult
    target_opening: float
    opening_slope: float
    achieved_opening: float
    outer_displacement: float
    effective_reaction: float
    R_ext_work_fd: float
    R_eff_work_fd: float
    R_ext_work_error: float
    R_eff_work_error: float
    work_pass: bool
    opening_pass: bool
    slope_pass: bool
    force_pass: bool
    moment_pass: bool
    static_runtime_identity: object
    static_digest: str
    damage_runtime_identity: object
    damage_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.unit_result, A2MechanicsSolveResult) or not isinstance(
            self.mechanics_result, A2MechanicsSolveResult
        ):
            raise TypeError("A2 opening result must contain mechanics solve results")
        target = _scalar(self.target_opening, "target_opening")
        if target < 0.0:
            raise ValueError("target_opening must be non-negative")
        for name in (
            "opening_slope", "achieved_opening", "outer_displacement", "effective_reaction",
            "R_ext_work_fd", "R_eff_work_fd", "R_ext_work_error", "R_eff_work_error",
        ):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        object.__setattr__(self, "target_opening", target)
        for name in ("work_pass", "opening_pass", "slope_pass", "force_pass", "moment_pass"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.mechanics_result.outer_displacement != self.outer_displacement:
            raise ValueError("A2 opening result outer displacement is not paired")
        if self.unit_result.static_runtime_identity is not self.static_runtime_identity:
            raise ValueError("A2 opening result static runtime identity is not paired")
        if self.mechanics_result.static_runtime_identity is not self.static_runtime_identity:
            raise ValueError("A2 opening result scaled static runtime identity is not paired")
        if self.unit_result.damage_runtime_identity is not self.damage_runtime_identity:
            raise ValueError("A2 opening result damage runtime identity is not paired")
        if self.mechanics_result.damage_runtime_identity is not self.damage_runtime_identity:
            raise ValueError("A2 opening result scaled damage runtime identity is not paired")
        if self.unit_result.static_digest != self.static_digest or self.mechanics_result.static_digest != self.static_digest:
            raise ValueError("A2 opening result static digest is not paired")
        if self.unit_result.damage_digest != self.damage_digest or self.mechanics_result.damage_digest != self.damage_digest:
            raise ValueError("A2 opening result damage digest is not paired")

    @property
    def displacement(self) -> np.ndarray:
        return self.mechanics_result.displacement

    @property
    def evaluation(self) -> MechanicsEvaluation:
        return self.mechanics_result.evaluation

    @property
    def R_ext(self) -> float:
        return self.mechanics_result.outer_reaction

    @property
    def R_eff(self) -> float:
        return self.effective_reaction

    @property
    def U_ext(self) -> float:
        return self.outer_displacement


def scale_a2_unit_solution(
    unit_result: A2MechanicsSolveResult,
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    target_opening: Any,
    opening_slope: Any,
    cg_rtol: Any,
    cg_atol: Any,
) -> A2MechanicsSolveResult:
    """Scale only a unit result from the same fixed-damage view."""

    if not isinstance(unit_result, A2MechanicsSolveResult):
        raise TypeError("unit_result must be an A2MechanicsSolveResult")
    if unit_result.outer_displacement != 1.0:
        raise ValueError("only an exact unit outer-displacement result may be scaled")
    _validate_view_constraints(view, constraints)
    if unit_result.static_runtime_identity is not view.static.runtime_identity:
        raise ValueError("unit result belongs to a different A2 static runtime identity")
    if unit_result.damage_runtime_identity is not view.material_damage_runtime_identity:
        raise ValueError("unit result belongs to a different fixed-damage view")
    if unit_result.static_digest != view.static.deterministic_digest:
        raise ValueError("unit result static digest does not match the view")
    if unit_result.damage_digest != view.damage_digest:
        raise ValueError("unit result material-damage digest does not match the view")
    expected_unit_digest = _unit_system_digest(
        view.static.deterministic_digest, view.damage_digest
    )
    if unit_result.unit_system_digest != expected_unit_digest:
        raise ValueError("unit result unit-system digest does not match the view")
    if unit_result.seed_provenance.target_damage_digest != view.damage_digest:
        raise ValueError("unit result seed target damage digest does not match the view")
    if unit_result.seed_provenance.unit_system_digest != expected_unit_digest:
        raise ValueError("unit result seed unit-system digest does not match the view")
    target = _scalar(target_opening, "target_opening")
    slope = _scalar(opening_slope, "opening_slope")
    if target < 0.0 or not np.isfinite(slope) or slope <= 0.0:
        raise ValueError("target opening/slope are invalid")
    U_ext = target / slope
    displacement = np.asarray(unit_result.displacement * U_ext, dtype=np.float64)
    _check_exact_constraints(displacement, constraints, U_ext, "scaled A2 displacement")
    evaluation = view.evaluate(displacement)
    free = constraints.free_dofs
    free_residual = evaluation.residual[free]
    rhs_norm = abs(U_ext) * unit_result.free_rhs_norm
    limit = max(_scalar(cg_atol, "cg_atol"), _scalar(cg_rtol, "cg_rtol") * rhs_norm)
    free_norm = float(np.linalg.norm(free_residual))
    if not np.isfinite(free_norm) or free_norm >= limit:
        raise RuntimeError("scaled A2 unit solution failed the authoritative free-residual gate")
    residual_matrix = evaluation.residual.reshape(view.n_particles, 2)
    force = np.sum(residual_matrix, axis=0)
    moment = float(np.sum(view.coordinates[:, 0] * residual_matrix[:, 1] - view.coordinates[:, 1] * residual_matrix[:, 0]))
    seed = _immutable(displacement)
    seed_provenance = _seed_record(
        "scaled_unit_solution",
        seed,
        view,
        constraints,
        unit_result.free_residual_norm,
        free_norm,
        source_state="unit_result",
        source_step=0,
        source_sweep=0,
        source_damage_digest=unit_result.seed_provenance.source_damage_digest,
    )
    return A2MechanicsSolveResult(
        displacement=displacement,
        evaluation=evaluation,
        constraints=constraints,
        outer_displacement=U_ext,
        free_residual=free_residual,
        free_residual_norm=free_norm,
        free_rhs_norm=rhs_norm,
        free_residual_relative_norm=free_norm / max(rhs_norm, 1.0e-300),
        cg_convergence_limit=limit,
        cg_iterations=unit_result.cg_iterations,
        cg_info=unit_result.cg_info,
        outer_reaction=float(constraints.outer_top_selector @ evaluation.residual),
        bottom_reaction=float(constraints.outer_bottom_selector @ evaluation.residual),
        anchor_reaction=float(constraints.anchor_selector @ evaluation.residual),
        total_force=force,
        total_moment=moment,
        static_runtime_identity=view.static.runtime_identity,
        static_digest=view.static.deterministic_digest,
        damage_runtime_identity=view.material_damage_runtime_identity,
        damage_digest=view.damage_digest,
        unit_system_digest=unit_result.unit_system_digest,
        seed_provenance=seed_provenance,
    )


def solve_a2_opening_controlled(
    view: A2FixedDamageView,
    constraints: AffineConstraints,
    target_opening: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    opening_slope_min: Any,
    opening_atol: Any,
    work_epsilon: Any,
    work_relative_limit: Any,
    work_cg_rtol: Any,
    work_cg_atol: Any,
    *,
    initial_guess: np.ndarray | None,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
) -> A2OpeningMechanicsResult:
    """Solve A2 mechanics at an exact physical interface opening."""

    target = _scalar(target_opening, "target_opening")
    if target < 0.0:
        raise ValueError("target_opening must be non-negative")
    unit = solve_a2_unit_load(
        view,
        constraints,
        cg_rtol,
        cg_atol,
        cg_maxiter,
        initial_guess=initial_guess,
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=source_damage_digest,
    )
    slope = physical_interface_opening_a2(unit.displacement, view.static)
    slope_min = _scalar(opening_slope_min, "opening_slope_min")
    if slope_min <= 0.0:
        raise ValueError("opening_slope_min must be positive")
    slope_pass = bool(np.isfinite(slope) and slope >= slope_min)
    if not slope_pass:
        raise RuntimeError(f"A2 physical opening slope failed: {slope}")
    scaled = scale_a2_unit_solution(
        unit, view, constraints, target, slope, _scalar(cg_rtol, "cg_rtol"), _scalar(cg_atol, "cg_atol")
    )
    achieved = physical_interface_opening_a2(scaled.displacement, view.static)
    opening_error = abs(achieved - target)
    opening_pass = opening_error <= _scalar(opening_atol, "opening_atol")

    epsilon = _scalar(work_epsilon, "work_epsilon")
    work_limit = _scalar(work_relative_limit, "work_relative_limit")
    if epsilon <= 0.0 or work_limit < 0.0:
        raise ValueError("work epsilon/relative limit are invalid")
    plus = solve_a2_fixed_outer_displacement(
        view,
        constraints,
        scaled.outer_displacement + epsilon,
        work_cg_rtol,
        work_cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="work_plus_zero_seed",
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=view.damage_digest,
    )
    minus = solve_a2_fixed_outer_displacement(
        view,
        constraints,
        scaled.outer_displacement - epsilon,
        work_cg_rtol,
        work_cg_atol,
        cg_maxiter,
        initial_guess=None,
        seed_kind="work_minus_zero_seed",
        source_state=source_state,
        source_step=source_step,
        source_sweep=source_sweep,
        source_damage_digest=view.damage_digest,
    )
    R_ext_fd = (plus.energy - minus.energy) / (2.0 * epsilon)
    D_plus = physical_interface_opening_a2(plus.displacement, view.static)
    D_minus = physical_interface_opening_a2(minus.displacement, view.static)
    if D_plus == D_minus:
        raise RuntimeError("A2 centered opening work denominator is zero")
    R_eff_fd = (plus.energy - minus.energy) / (D_plus - D_minus)
    R_ext_error = _relative(R_ext_fd, scaled.outer_reaction)
    R_eff = scaled.outer_reaction / slope
    R_eff_error = _relative(R_eff_fd, R_eff)
    work_pass = bool(R_ext_error <= work_limit and R_eff_error <= work_limit)
    # Conservation rejection belongs to the configured continuation gates.  The
    # opening solve only records finite force/moment diagnostics and must not
    # impose an undocumented tolerance of its own.
    force_pass = bool(np.all(np.isfinite(scaled.total_force)))
    moment_pass = bool(np.isfinite(scaled.total_moment))
    return A2OpeningMechanicsResult(
        unit_result=unit,
        mechanics_result=scaled,
        target_opening=target,
        opening_slope=slope,
        achieved_opening=achieved,
        outer_displacement=scaled.outer_displacement,
        effective_reaction=R_eff,
        R_ext_work_fd=R_ext_fd,
        R_eff_work_fd=R_eff_fd,
        R_ext_work_error=R_ext_error,
        R_eff_work_error=R_eff_error,
        work_pass=work_pass,
        opening_pass=opening_pass,
        slope_pass=slope_pass,
        force_pass=force_pass,
        moment_pass=moment_pass,
        static_runtime_identity=view.static.runtime_identity,
        static_digest=view.static.deterministic_digest,
        damage_runtime_identity=view.material_damage_runtime_identity,
        damage_digest=view.damage_digest,
    )


@dataclass(frozen=True)
class A2AmorHistoryResult:
    edge_positive_energy: np.ndarray
    particle_driver: np.ndarray
    previous_history: np.ndarray
    trial_history: np.ndarray

    def __post_init__(self) -> None:
        edge = _f64(self.edge_positive_energy, "edge_positive_energy")
        driver = _f64(self.particle_driver, "particle_driver")
        previous = _f64(self.previous_history, "previous_history")
        trial = _f64(self.trial_history, "trial_history")
        if edge.ndim != 1 or driver.ndim != 1 or previous.shape != driver.shape or trial.shape != driver.shape:
            raise ValueError("A2 Amor arrays have incompatible shapes")
        if np.any(driver < 0.0) or np.any(previous < 0.0) or np.any(trial < previous):
            raise ValueError("A2 Amor history maximum contract is violated")
        for name, value in (("edge_positive_energy", edge), ("particle_driver", driver), ("previous_history", previous), ("trial_history", trial)):
            object.__setattr__(self, name, _immutable(value))

    @property
    def history_previous(self) -> np.ndarray:
        return self.previous_history

    @property
    def history_trial(self) -> np.ndarray:
        return self.trial_history


def _amor_edge_energy(edge_strain: np.ndarray, lame_lambda: float, shear_modulus: float) -> np.ndarray:
    strain = _f64(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError("edge_strain must have shape (E, 2, 2)")
    if _max_abs(strain - np.swapaxes(strain, 1, 2)) > 32.0 * _EPS64 * max(1.0, _max_abs(strain)):
        raise ValueError("edge_strain must be symmetric")
    lam = _scalar(lame_lambda, "lame_lambda")
    mu = _scalar(shear_modulus, "shear_modulus")
    if mu <= 0.0 or lam + 2.0 * mu / 3.0 <= 0.0:
        raise ValueError("Amor constitutive values are invalid")
    embedded = np.zeros((len(strain), 3, 3), dtype=np.float64)
    embedded[:, :2, :2] = strain
    trace = np.trace(embedded, axis1=1, axis2=2)
    deviatoric = embedded - (trace / 3.0)[:, None, None] * _IDENTITY_3
    result = 0.5 * (lam + 2.0 * mu / 3.0) * np.maximum(trace, 0.0) ** 2 + mu * np.sum(
        deviatoric * deviatoric, axis=(1, 2)
    )
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("A2 Amor edge energy is invalid")
    return _immutable(result)


def compute_a2_amor_history_from_edge_strain(
    edge_strain: Any,
    source: Any,
    target: Any,
    eta: Any,
    n_material: Any,
    lame_lambda: Any,
    shear_modulus: Any,
    previous_history: Any,
) -> A2AmorHistoryResult:
    """Accumulate Amor only from material source centers, including all targets."""

    strain = _f64(edge_strain, "edge_strain")
    if strain.ndim != 3 or strain.shape[1:] != (2, 2):
        raise ValueError("edge_strain must have shape (E, 2, 2)")
    E = len(strain)
    source_array = np.asarray(source)
    target_array = np.asarray(target)
    if source_array.dtype != _INT64 or target_array.dtype != _INT64 or source_array.shape != (E,) or target_array.shape != (E,):
        raise ValueError("A2 Amor source/target must be int64 shape (E,)")
    eta_array = _f64(eta, "eta", (E,))
    material_count = _integer(n_material, "n_material", positive=True)
    previous = _f64(previous_history, "previous_history", (material_count,))
    if np.any(source_array < 0) or np.any(target_array < 0):
        raise ValueError("A2 Amor source/target indices must be non-negative")
    retained = source_array < material_count
    if not np.any(retained):
        raise ValueError("A2 Amor union graph has no material source centers")
    source_material = source_array[retained]
    eta_material = eta_array[retained]
    if np.any(eta_material <= 0.0):
        raise ValueError("A2 Amor eta must be strictly positive")
    eta_sum = np.bincount(source_material, weights=eta_material, minlength=material_count)
    if not np.all(np.isfinite(eta_sum)) or np.max(np.abs(eta_sum - 1.0)) > _REPLAY_BOUND:
        raise ValueError("A2 Amor material-source eta must sum to one without renormalization")
    edge_positive_all = _amor_edge_energy(strain, _scalar(lame_lambda, "lame_lambda"), _scalar(shear_modulus, "shear_modulus"))
    edge_positive = edge_positive_all[retained]
    driver = np.zeros(material_count, dtype=np.float64)
    np.add.at(driver, source_material, eta_material * edge_positive)
    trial = np.maximum(previous, driver)
    return A2AmorHistoryResult(edge_positive_all, driver, previous, trial)


def compute_a2_amor_history(
    view: A2FixedDamageView,
    displacement: Any,
    previous_history: Any,
) -> A2AmorHistoryResult:
    if not isinstance(view, A2FixedDamageView):
        raise TypeError("view must be an A2FixedDamageView")
    evaluation = view.evaluate(displacement)
    return compute_a2_amor_history_from_edge_strain(
        evaluation.edge_strain,
        view.static.source,
        view.static.target,
        view.static.eta,
        view.static.material_count,
        view.lame_lambda,
        view.shear_modulus,
        previous_history,
    )


@dataclass(frozen=True)
class A2CoupledModel:
    geometry: UnionGripGeometry
    static: A2StaticMechanics
    constraints: AffineConstraints
    top_line: LineQuadrature
    bottom_line: LineQuadrature
    material_operator: ExplicitVolumeSparseD
    Gc: float
    l0: float

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, UnionGripGeometry) or not isinstance(self.static, A2StaticMechanics):
            raise TypeError("A2 model geometry/static types are invalid")
        _validate_view_constraints(
            A2FixedDamageView(self.static, np.zeros(self.static.material_count, dtype=np.float64)),
            self.constraints,
        )
        if not isinstance(self.top_line, LineQuadrature) or not isinstance(self.bottom_line, LineQuadrature):
            raise TypeError("A2 model line quadratures are invalid")
        if not isinstance(self.material_operator, ExplicitVolumeSparseD):
            raise TypeError("A2 model material_operator must be ExplicitVolumeSparseD")
        if self.material_operator.shape != (self.static.material_count, self.static.material_count):
            raise ValueError("A2 material damage operator must have material-cloud shape")
        if self.material_operator.kernel != _KERNEL:
            raise ValueError("A2 material damage operator must use the cubic kernel")
        if not np.array_equal(np.asarray(self.material_operator.coordinates), self.geometry.coordinates[: self.static.material_count]):
            raise ValueError("A2 material operator coordinates are not the material prefix")
        if not np.array_equal(np.asarray(self.material_operator.volumes), self.static.material_damage_volumes):
            raise ValueError("A2 material operator volumes are not the original damage volumes")
        if not np.array_equal(np.asarray(self.material_operator.directed_edges, dtype=np.int64), self.geometry.material_reference_edges):
            raise ValueError("A2 material operator graph is not the unchanged material graph")
        object.__setattr__(self, "Gc", _scalar(self.Gc, "A2 Gc"))
        object.__setattr__(self, "l0", _scalar(self.l0, "A2 l0"))
        if self.Gc <= 0.0 or self.l0 <= 0.0:
            raise ValueError("A2 Gc/l0 must be positive")


def build_a2_coupled_model(
    geometry: UnionGripGeometry,
    youngs_modulus: Any,
    poisson_ratio: Any,
    k_res: Any,
    rank_rtol: Any,
    Gc: Any,
    l0: Any,
) -> A2CoupledModel:
    """Construct the unequal-cloud A2 mechanics/material-damage model."""

    if not isinstance(geometry, UnionGripGeometry):
        raise TypeError("geometry must be a UnionGripGeometry")
    # The geometry contract already retains the original lower-left anchor at
    # material particle zero.  Use that exact ID rather than a coordinate search
    # for the production/material constructor.
    if geometry.material_reference_coordinates[0].tolist() != [0.0, 0.0]:
        raise ValueError("A2 material prefix particle zero must be the lower-left anchor")
    constraints = build_affine_constraints(
        geometry.coordinates,
        geometry.outer_bottom_mask,
        geometry.outer_top_mask,
        0,
        geometry.delta,
    )
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
    # Construct the damage operator only from the original material prefix and
    # induced graph.  No union particle is passed to the phase-field equation.
    material_operator = ExplicitVolumeSparseD(
        geometry.coordinates[: geometry.n_material],
        geometry.material_reference_volumes,
        geometry.material_reference_edges,
        geometry.delta,
        _KERNEL,
    )
    return A2CoupledModel(
        geometry=geometry,
        static=static,
        constraints=constraints,
        top_line=top_line,
        bottom_line=bottom_line,
        material_operator=material_operator,
        Gc=Gc,
        l0=l0,
    )


@dataclass(frozen=True)
class A2StrictConfig:
    opening_start: float
    opening_final: float
    d_opening_initial: float
    d_opening_min: float
    cutback_factor: float
    max_cutbacks: int
    max_stagger_iterations: int
    cg_rtol: float
    cg_atol: float
    cg_maxiter: int
    displacement_material: GateTolerance
    displacement_union: GateTolerance
    damage: GateTolerance
    history: GateTolerance
    reaction_effective: GateTolerance
    damage_residual: GateTolerance
    force: GateTolerance
    moment: GateTolerance
    opening_atol: float
    opening_slope_min: float
    healing_atol: float
    damage_range_atol: float
    work_epsilon: float
    work_relative_limit: float
    work_cg_rtol: float
    work_cg_atol: float

    def __post_init__(self) -> None:
        opening_start = _scalar(self.opening_start, "opening_start")
        opening_final = _scalar(self.opening_final, "opening_final")
        initial = _scalar(self.d_opening_initial, "d_opening_initial")
        minimum = _scalar(self.d_opening_min, "d_opening_min")
        factor = _scalar(self.cutback_factor, "cutback_factor")
        if opening_start != 0.0 or opening_final <= opening_start:
            raise ValueError("A2 opening interval must start at exact zero and increase")
        if initial <= 0.0 or minimum <= 0.0 or minimum > initial or initial > opening_final:
            raise ValueError("A2 opening increments are invalid")
        if not 0.0 < factor < 1.0:
            raise ValueError("A2 cutback_factor must lie strictly between zero and one")
        max_cutbacks = _integer(self.max_cutbacks, "max_cutbacks", positive=True)
        max_stagger = _integer(self.max_stagger_iterations, "max_stagger_iterations", positive=True)
        cg_rtol = _scalar(self.cg_rtol, "cg_rtol")
        cg_atol = _scalar(self.cg_atol, "cg_atol")
        cg_maxiter = _integer(self.cg_maxiter, "cg_maxiter", positive=True)
        if cg_rtol < 0.0 or cg_atol < 0.0 or (cg_rtol == 0.0 and cg_atol == 0.0):
            raise ValueError("A2 CG tolerances are invalid")
        for name in (
            "displacement_material", "displacement_union", "damage", "history",
            "reaction_effective", "damage_residual", "force", "moment",
        ):
            if not isinstance(getattr(self, name), GateTolerance):
                raise TypeError(f"A2 {name} must be a GateTolerance")
        for name in (
            "opening_atol", "healing_atol", "damage_range_atol", "work_relative_limit",
        ):
            value = _scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"A2 {name} must be non-negative")
            object.__setattr__(self, name, value)
        slope_min = _scalar(self.opening_slope_min, "opening_slope_min")
        epsilon = _scalar(self.work_epsilon, "work_epsilon")
        work_rtol = _scalar(self.work_cg_rtol, "work_cg_rtol")
        work_atol = _scalar(self.work_cg_atol, "work_cg_atol")
        if slope_min <= 0.0 or epsilon <= 0.0 or work_rtol < 0.0 or work_atol < 0.0 or (work_rtol == 0.0 and work_atol == 0.0):
            raise ValueError("A2 opening/work solve settings are invalid")
        for name, value in (
            ("opening_start", opening_start), ("opening_final", opening_final),
            ("d_opening_initial", initial), ("d_opening_min", minimum),
            ("cutback_factor", factor), ("opening_slope_min", slope_min),
            ("work_epsilon", epsilon), ("work_cg_rtol", work_rtol), ("work_cg_atol", work_atol),
        ):
            object.__setattr__(self, name, value)
        for name, value in (
            ("max_cutbacks", max_cutbacks), ("max_stagger_iterations", max_stagger),
            ("cg_maxiter", cg_maxiter), ("cg_rtol", cg_rtol), ("cg_atol", cg_atol),
        ):
            object.__setattr__(self, name, value)

    @property
    def acceptance_mode(self) -> str:
        return A2_ACCEPTANCE_MODE


A2CoupledConfig = A2StrictConfig


@dataclass(frozen=True)
class A2CommittedState:
    physical_opening: float
    outer_displacement: float
    union_displacement: np.ndarray
    material_damage: np.ndarray
    material_history: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical_opening", _scalar(self.physical_opening, "physical_opening"))
        object.__setattr__(self, "outer_displacement", _scalar(self.outer_displacement, "outer_displacement"))
        q = _f64(self.union_displacement, "union_displacement")
        d = _f64(self.material_damage, "material_damage")
        H = _f64(self.material_history, "material_history")
        if q.ndim != 1 or q.size == 0 or q.size % 2:
            raise ValueError("A2 union_displacement must be a nonempty flat (2*N,) array")
        if d.ndim != 1 or d.size == 0 or H.shape != d.shape:
            raise ValueError("A2 material damage/history shapes are invalid")
        if np.any(H < 0.0):
            raise ValueError("A2 material history must be non-negative")
        for name, value in (("union_displacement", q), ("material_damage", d), ("material_history", H)):
            object.__setattr__(self, name, _immutable(value))

    @property
    def Delta(self) -> float:
        return self.physical_opening

    @property
    def U_ext(self) -> float:
        return self.outer_displacement

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
        return _digest_state(
            self.physical_opening,
            self.outer_displacement,
            self.union_displacement,
            self.material_damage,
            self.material_history,
        )


@dataclass(frozen=True)
class A2Diagnostics:
    """Complete or explicitly incomplete primitive sweep diagnostics."""

    sweep_index: int
    target_opening: float
    opening_atol: float
    complete_sweep: bool
    passed: bool
    failure_stage: str
    failure_reason: str
    damage_before: np.ndarray
    d_iter: np.ndarray
    mechanics_target_opening_before: float | None
    mechanics_target_opening_after: float | None
    d_next: np.ndarray | None
    q_before: np.ndarray | None
    q_after: np.ndarray | None
    Psi_before: np.ndarray | None
    Psi_after: np.ndarray | None
    H_used: np.ndarray | None
    H_next: np.ndarray | None
    U_before: float | None
    U_after: float | None
    slope_before: float | None
    slope_after: float | None
    achieved_opening_before: float | None
    achieved_opening_after: float | None
    R_ext_before: float | None
    R_ext_after: float | None
    R_eff_before: float | None
    R_eff_after: float | None
    rho_used: np.ndarray | None
    rho_next: np.ndarray | None
    rho_used_volume: np.ndarray | None
    rho_next_volume: np.ndarray | None
    rho_used_inf: float
    rho_next_inf: float
    e_opening_before: float
    e_opening_after: float
    e_u_material: float
    u_material_scale: float
    u_material_limit: float
    u_material_pass: bool
    e_u_union: float
    u_union_scale: float
    u_union_limit: float
    u_union_pass: bool
    e_damage: float
    damage_scale: float
    damage_limit: float
    damage_pass: bool
    e_history: float
    history_scale: float
    history_limit: float
    history_pass: bool
    e_reaction_effective: float
    reaction_effective_scale: float
    reaction_effective_limit: float
    reaction_effective_pass: bool
    damage_residual_scale: float
    damage_residual_limit: float
    damage_residual_pass: bool
    damage_min: float
    damage_max: float
    damage_range_pass: bool
    healing_min: float
    healing_pass: bool
    history_min_gap: float
    history_monotonic_pass: bool
    force_before: np.ndarray | None
    force_after: np.ndarray | None
    moment_before: float | None
    moment_after: float | None
    force_before_error: float
    force_before_scale: float
    force_before_limit: float
    force_before_pass: bool
    force_after_error: float
    force_after_scale: float
    force_after_limit: float
    force_after_pass: bool
    force_error: float
    force_scale: float
    force_limit: float
    force_pass: bool
    moment_before_error: float
    moment_before_scale: float
    moment_before_limit: float
    moment_before_pass: bool
    moment_after_error: float
    moment_after_scale: float
    moment_after_limit: float
    moment_after_pass: bool
    moment_error: float
    moment_scale: float
    moment_limit: float
    moment_pass: bool
    work_R_ext_before_error: float
    work_R_eff_before_error: float
    work_R_ext_after_error: float
    work_R_eff_after_error: float
    work_limit: float
    work_pass: bool
    opening_before_pass: bool
    opening_after_pass: bool
    slope_before_pass: bool
    slope_after_pass: bool
    work_before_pass: bool
    work_after_pass: bool
    cg_before_pass: bool
    cg_after_pass: bool
    mechanics_before_free_residual_norm: float
    mechanics_after_free_residual_norm: float
    mechanics_before_free_residual_relative_norm: float
    mechanics_after_free_residual_relative_norm: float
    mechanics_pass: bool
    seed_before: A2SeedProvenance | None
    seed_after: A2SeedProvenance | None
    available_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sweep_index", _integer(self.sweep_index, "sweep_index"))
        object.__setattr__(self, "target_opening", _scalar(self.target_opening, "target_opening"))
        opening_atol = _scalar(self.opening_atol, "diagnostics.opening_atol")
        if opening_atol < 0.0:
            raise ValueError("diagnostics.opening_atol must be non-negative")
        object.__setattr__(self, "opening_atol", opening_atol)
        object.__setattr__(self, "complete_sweep", bool(self.complete_sweep))
        object.__setattr__(self, "passed", bool(self.passed))
        if not isinstance(self.failure_stage, str) or not isinstance(self.failure_reason, str):
            raise TypeError("A2 diagnostic failure fields must be strings")
        damage_before = _f64(self.damage_before, "diagnostics.damage_before")
        d_iter = _f64(self.d_iter, "diagnostics.d_iter", damage_before.shape)
        object.__setattr__(self, "damage_before", _immutable(damage_before))
        object.__setattr__(self, "d_iter", _immutable(d_iter))
        optional_specs = {
            "d_next": damage_before.shape,
            "Psi_before": damage_before.shape,
            "Psi_after": damage_before.shape,
            "H_used": damage_before.shape,
            "H_next": damage_before.shape,
            "rho_used": damage_before.shape,
            "rho_next": damage_before.shape,
            "rho_used_volume": damage_before.shape,
            "rho_next_volume": damage_before.shape,
        }
        for name, shape in optional_specs.items():
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _immutable(_f64(value, f"diagnostics.{name}", shape)))
        for name, shape in (("q_before", ()), ("q_after", ())):
            value = getattr(self, name)
            if value is not None:
                array = _f64(value, f"diagnostics.{name}")
                if array.ndim != 1 or array.size == 0 or array.size % 2:
                    raise ValueError(f"diagnostics.{name} must be a flat displacement")
                object.__setattr__(self, name, _immutable(array))
        for name in ("mechanics_target_opening_before", "mechanics_target_opening_after"):
            value = getattr(self, name)
            if value is not None:
                target = _scalar(value, f"diagnostics.{name}")
                if target < 0.0:
                    raise ValueError(f"diagnostics.{name} must be non-negative")
                object.__setattr__(self, name, target)
        for name in ("force_before", "force_after"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _immutable(_f64(value, f"diagnostics.{name}", (2,))))
        for name in (
            "rho_used_inf", "rho_next_inf", "e_opening_before", "e_opening_after", "e_u_material",
            "u_material_scale", "u_material_limit", "e_u_union", "u_union_scale", "u_union_limit",
            "e_damage", "damage_scale", "damage_limit", "e_history", "history_scale", "history_limit",
            "e_reaction_effective", "reaction_effective_scale", "reaction_effective_limit",
            "damage_residual_scale", "damage_residual_limit", "damage_min", "damage_max", "healing_min",
            "history_min_gap", "force_before_error", "force_before_scale", "force_before_limit",
            "force_after_error", "force_after_scale", "force_after_limit", "force_error", "force_scale", "force_limit",
            "moment_before_error", "moment_before_scale", "moment_before_limit",
            "moment_after_error", "moment_after_scale", "moment_after_limit", "moment_error", "moment_scale",
            "moment_limit", "work_R_ext_before_error", "work_R_eff_before_error", "work_R_ext_after_error",
            "work_R_eff_after_error", "work_limit", "mechanics_before_free_residual_norm",
            "mechanics_after_free_residual_norm", "mechanics_before_free_residual_relative_norm",
            "mechanics_after_free_residual_relative_norm",
        ):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 0:
                raise ValueError(f"diagnostics.{name} must be scalar")
            value = float(raw)
            if np.isnan(value):
                raise ValueError(f"diagnostics.{name} must not be NaN")
            if self.complete_sweep and not np.isfinite(value):
                raise ValueError(f"diagnostics.{name} must be finite for a complete sweep")
            object.__setattr__(self, name, value)
        for name in (
            "u_material_pass", "u_union_pass", "damage_pass", "history_pass", "reaction_effective_pass",
            "damage_residual_pass", "damage_range_pass", "healing_pass", "history_monotonic_pass",
            "force_before_pass", "force_after_pass", "force_pass", "moment_before_pass", "moment_after_pass",
            "moment_pass", "work_pass", "opening_before_pass", "opening_after_pass",
            "slope_before_pass", "slope_after_pass", "work_before_pass", "work_after_pass",
            "cg_before_pass", "cg_after_pass", "mechanics_pass",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in ("seed_before", "seed_after"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, A2SeedProvenance):
                raise TypeError(f"diagnostics.{name} must be A2SeedProvenance or None")
        for name in ("U_before", "U_after", "slope_before", "slope_after", "achieved_opening_before", "achieved_opening_after", "R_ext_before", "R_ext_after", "R_eff_before", "R_eff_after", "moment_before", "moment_after"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _scalar(value, f"diagnostics.{name}"))
        if self.complete_sweep and not self.passed and not self.failure_reason:
            raise ValueError("a complete failed A2 sweep must retain a failure reason")
        if not self.complete_sweep and (not self.failure_stage or not self.failure_reason):
            raise ValueError("an incomplete A2 sweep must retain failure stage and reason")
        if self.passed and not self.complete_sweep:
            raise ValueError("an incomplete A2 sweep cannot pass")
        object.__setattr__(self, "available_fields", tuple(sorted(set(self.available_fields))))

    @property
    def rho_used_volume_weighted(self) -> np.ndarray | None:
        return self.rho_used_volume

    @property
    def rho_next_volume_weighted(self) -> np.ndarray | None:
        return self.rho_next_volume

    @property
    def gate_passes(self) -> dict[str, bool]:
        return {
            "opening_before": self.e_opening_before <= self.opening_atol,
            "opening_after": self.e_opening_after <= self.opening_atol,
            "slope_before": self.slope_before_pass,
            "slope_after": self.slope_after_pass,
            "work_before": self.work_before_pass,
            "work_after": self.work_after_pass,
            "cg_before": self.cg_before_pass,
            "cg_after": self.cg_after_pass,
            "displacement_material": self.u_material_pass,
            "displacement_union": self.u_union_pass,
            "damage": self.damage_pass,
            "history": self.history_pass,
            "reaction_effective": self.reaction_effective_pass,
            "damage_residual": self.damage_residual_pass,
            "damage_range": self.damage_range_pass,
            "healing": self.healing_pass,
            "history_monotonic": self.history_monotonic_pass,
            "work": self.work_pass,
            "force_before": self.force_before_pass,
            "force_after": self.force_after_pass,
            "force": self.force_pass,
            "moment_before": self.moment_before_pass,
            "moment_after": self.moment_after_pass,
            "moment": self.moment_pass,
            "mechanics": self.mechanics_pass,
        }


def _none_or_array(value: Any | None) -> Any | None:
    return None if value is None else np.asarray(value)


def _conservation_metrics_from_values(
    config: A2StrictConfig,
    outer_reaction: float | None,
    bottom_reaction: float | None,
    anchor_reaction: float | None,
    total_force: np.ndarray | None,
    total_moment: float | None,
    geometry_length: float,
) -> dict[str, float | bool]:
    """Return one independent force/moment gate record for one mechanics state."""

    length = _scalar(geometry_length, "geometry_length")
    if length <= 0.0:
        raise ValueError("geometry_length must be positive")
    if any(value is None for value in (outer_reaction, bottom_reaction, anchor_reaction, total_force, total_moment)):
        return {
            "force_error": np.inf,
            "force_scale": 0.0,
            "force_limit": config.force.limit(0.0),
            "force_pass": False,
            "moment_error": np.inf,
            "moment_scale": 0.0,
            "moment_limit": config.moment.limit(0.0),
            "moment_pass": False,
        }
    assert outer_reaction is not None
    assert bottom_reaction is not None
    assert anchor_reaction is not None
    assert total_force is not None
    assert total_moment is not None
    force_error = _max_abs(np.asarray(total_force))
    force_scale = max(
        abs(float(outer_reaction)),
        abs(float(bottom_reaction)),
        abs(float(anchor_reaction)),
        1.0,
    )
    force_limit = config.force.limit(force_scale)
    moment_error = abs(float(total_moment))
    moment_scale = force_scale * length
    moment_limit = config.moment.limit(moment_scale)
    return {
        "force_error": force_error,
        "force_scale": force_scale,
        "force_limit": force_limit,
        "force_pass": bool(force_error <= force_limit),
        "moment_error": moment_error,
        "moment_scale": moment_scale,
        "moment_limit": moment_limit,
        "moment_pass": bool(moment_error <= moment_limit),
    }


def _conservation_metrics(
    config: A2StrictConfig,
    mechanics: A2OpeningMechanicsResult | None,
    geometry_length: float,
) -> dict[str, float | bool]:
    if mechanics is None:
        return _conservation_metrics_from_values(config, None, None, None, None, None, geometry_length)
    result = mechanics.mechanics_result
    return _conservation_metrics_from_values(
        config,
        result.outer_reaction,
        result.bottom_reaction,
        result.anchor_reaction,
        result.total_force,
        result.total_moment,
        geometry_length,
    )


def _incomplete_diagnostics(
    config: A2StrictConfig,
    committed: A2CommittedState,
    d_iter: np.ndarray,
    target: float,
    stage: str,
    reason: str,
    *,
    q_before: np.ndarray | None = None,
    q_after: np.ndarray | None = None,
    psi_before: np.ndarray | None = None,
    psi_after: np.ndarray | None = None,
    H_used: np.ndarray | None = None,
    d_next: np.ndarray | None = None,
    rho_used: np.ndarray | None = None,
    rho_used_volume: np.ndarray | None = None,
    mechanics_before: A2OpeningMechanicsResult | None = None,
    mechanics_after: A2OpeningMechanicsResult | None = None,
    seed_before: A2SeedProvenance | None = None,
    seed_after: A2SeedProvenance | None = None,
    sweep_index: int = 0,
    geometry_length: float = 1.0,
) -> A2Diagnostics:
    n = len(committed.material_damage)
    zero_d = np.zeros(n, dtype=np.float64)
    q_b = None if mechanics_before is None else mechanics_before.displacement
    q_a = None if mechanics_after is None else mechanics_after.displacement
    q_b = q_before if q_before is not None else q_b
    q_a = q_after if q_after is not None else q_a
    U_b = None if mechanics_before is None else mechanics_before.outer_displacement
    U_a = None if mechanics_after is None else mechanics_after.outer_displacement
    slope_b = None if mechanics_before is None else mechanics_before.opening_slope
    slope_a = None if mechanics_after is None else mechanics_after.opening_slope
    D_b = None if mechanics_before is None else mechanics_before.achieved_opening
    D_a = None if mechanics_after is None else mechanics_after.achieved_opening
    Rxb = None if mechanics_before is None else mechanics_before.R_ext
    Rxa = None if mechanics_after is None else mechanics_after.R_ext
    Reb = None if mechanics_before is None else mechanics_before.R_eff
    Rea = None if mechanics_after is None else mechanics_after.R_eff
    force_b = None if mechanics_before is None else mechanics_before.mechanics_result.total_force
    force_a = None if mechanics_after is None else mechanics_after.mechanics_result.total_force
    moment_b = None if mechanics_before is None else mechanics_before.mechanics_result.total_moment
    moment_a = None if mechanics_after is None else mechanics_after.mechanics_result.total_moment
    conservation_before = _conservation_metrics(config, mechanics_before, geometry_length)
    conservation_after = _conservation_metrics(config, mechanics_after, geometry_length)
    mechanics_before_flags = _a2_mechanics_gate_flags(
        config, mechanics_before, target, geometry_length
    )
    mechanics_after_flags = _a2_mechanics_gate_flags(
        config, mechanics_after, target, geometry_length
    )
    force_scale = max(float(conservation_before["force_scale"]), float(conservation_after["force_scale"]))
    force_error = max(float(conservation_before["force_error"]), float(conservation_after["force_error"]))
    moment_scale = max(float(conservation_before["moment_scale"]), float(conservation_after["moment_scale"]))
    moment_error = max(float(conservation_before["moment_error"]), float(conservation_after["moment_error"]))
    available = ["damage_before", "d_iter"]
    for name, value in (
        ("q_before", q_b), ("Psi_before", psi_before), ("H_used", H_used),
        ("d_next", d_next), ("rho_used", rho_used), ("rho_used_volume", rho_used_volume),
        ("q_after", q_a), ("Psi_after", psi_after), ("force_before", force_b),
        ("force_after", force_a),
    ):
        if value is not None:
            available.append(name)
    for mechanics, prefix in ((mechanics_before, "before"), (mechanics_after, "after")):
        if mechanics is not None:
            available.extend(
                (
                    f"mechanics_target_opening_{prefix}",
                    f"U_{prefix}", f"slope_{prefix}", f"achieved_opening_{prefix}",
                    f"R_ext_{prefix}", f"R_eff_{prefix}", f"moment_{prefix}",
                    f"work_R_ext_{prefix}_error", f"work_R_eff_{prefix}_error",
                    f"mechanics_{prefix}_free_residual_norm",
                    f"mechanics_{prefix}_free_residual_relative_norm",
                    f"force_{prefix}_error", f"force_{prefix}_scale", f"force_{prefix}_limit", f"force_{prefix}_pass",
                    f"moment_{prefix}_error", f"moment_{prefix}_scale", f"moment_{prefix}_limit", f"moment_{prefix}_pass",
                )
            )
    if mechanics_before is not None:
        available.append("e_opening_before")
    if mechanics_after is not None:
        available.append("e_opening_after")
    if seed_before is not None:
        available.append("seed_before")
    if seed_after is not None:
        available.append("seed_after")
    return A2Diagnostics(
        sweep_index=sweep_index,
        target_opening=target,
        opening_atol=config.opening_atol,
        complete_sweep=False,
        passed=False,
        failure_stage=stage,
        failure_reason=reason,
        damage_before=committed.material_damage,
        d_iter=d_iter,
        mechanics_target_opening_before=(
            None if mechanics_before is None else mechanics_before.target_opening
        ),
        mechanics_target_opening_after=(
            None if mechanics_after is None else mechanics_after.target_opening
        ),
        d_next=d_next,
        q_before=q_b,
        q_after=q_a,
        Psi_before=psi_before,
        Psi_after=psi_after,
        H_used=H_used,
        H_next=None,
        U_before=U_b,
        U_after=U_a,
        slope_before=slope_b,
        slope_after=slope_a,
        achieved_opening_before=D_b,
        achieved_opening_after=D_a,
        R_ext_before=Rxb,
        R_ext_after=Rxa,
        R_eff_before=Reb,
        R_eff_after=Rea,
        rho_used=rho_used,
        rho_next=None,
        rho_used_volume=rho_used_volume,
        rho_next_volume=None,
        rho_used_inf=0.0 if rho_used is None else _max_abs(rho_used),
        rho_next_inf=0.0,
        e_opening_before=np.inf if D_b is None else abs(D_b - target),
        e_opening_after=np.inf if D_a is None else abs(D_a - target),
        e_u_material=np.inf,
        u_material_scale=0.0,
        u_material_limit=config.displacement_material.limit(0.0),
        u_material_pass=False,
        e_u_union=np.inf,
        u_union_scale=0.0,
        u_union_limit=config.displacement_union.limit(0.0),
        u_union_pass=False,
        e_damage=np.inf if d_next is None else _max_abs(d_next - d_iter),
        damage_scale=max(_max_abs(d_iter), _max_abs(d_next) if d_next is not None else 0.0),
        damage_limit=config.damage.limit(max(_max_abs(d_iter), _max_abs(d_next) if d_next is not None else 0.0)),
        damage_pass=False,
        e_history=np.inf,
        history_scale=max(_max_abs(committed.material_history), _max_abs(H_used) if H_used is not None else 0.0),
        history_limit=config.history.limit(max(_max_abs(committed.material_history), _max_abs(H_used) if H_used is not None else 0.0)),
        history_pass=False,
        e_reaction_effective=np.inf,
        reaction_effective_scale=0.0,
        reaction_effective_limit=config.reaction_effective.limit(0.0),
        reaction_effective_pass=False,
        damage_residual_scale=0.0,
        damage_residual_limit=config.damage_residual.limit(0.0),
        damage_residual_pass=False,
        damage_min=float(np.min(d_next if d_next is not None else d_iter)),
        damage_max=float(np.max(d_next if d_next is not None else d_iter)),
        damage_range_pass=(
            d_next is not None
            and float(np.min(d_next)) >= -config.damage_range_atol
            and float(np.max(d_next)) <= 1.0 + config.damage_range_atol
        ),
        healing_min=float(np.min((d_next if d_next is not None else d_iter) - committed.material_damage)),
        healing_pass=(
            d_next is not None
            and float(np.min(d_next - committed.material_damage)) >= -config.healing_atol
        ),
        history_min_gap=0.0,
        history_monotonic_pass=False,
        force_before=force_b,
        force_after=force_a,
        moment_before=moment_b,
        moment_after=moment_a,
         force_before_error=float(conservation_before["force_error"]),
         force_before_scale=float(conservation_before["force_scale"]),
         force_before_limit=float(conservation_before["force_limit"]),
         force_before_pass=bool(conservation_before["force_pass"]),
         force_after_error=float(conservation_after["force_error"]),
         force_after_scale=float(conservation_after["force_scale"]),
         force_after_limit=float(conservation_after["force_limit"]),
         force_after_pass=bool(conservation_after["force_pass"]),
         force_error=force_error,
         force_scale=force_scale,
         force_limit=config.force.limit(force_scale),
         force_pass=bool(conservation_before["force_pass"] and conservation_after["force_pass"]),
         moment_before_error=float(conservation_before["moment_error"]),
         moment_before_scale=float(conservation_before["moment_scale"]),
         moment_before_limit=float(conservation_before["moment_limit"]),
         moment_before_pass=bool(conservation_before["moment_pass"]),
         moment_after_error=float(conservation_after["moment_error"]),
         moment_after_scale=float(conservation_after["moment_scale"]),
         moment_after_limit=float(conservation_after["moment_limit"]),
         moment_after_pass=bool(conservation_after["moment_pass"]),
         moment_error=moment_error,
         moment_scale=moment_scale,
         moment_limit=config.moment.limit(moment_scale),
         moment_pass=bool(conservation_before["moment_pass"] and conservation_after["moment_pass"]),
         work_R_ext_before_error=np.inf if mechanics_before is None else mechanics_before.R_ext_work_error,
         work_R_eff_before_error=np.inf if mechanics_before is None else mechanics_before.R_eff_work_error,
         work_R_ext_after_error=np.inf if mechanics_after is None else mechanics_after.R_ext_work_error,
         work_R_eff_after_error=np.inf if mechanics_after is None else mechanics_after.R_eff_work_error,
        work_limit=config.work_relative_limit,
         work_pass=bool(
             mechanics_before_flags["work"] and mechanics_after_flags["work"]
         ),
         opening_before_pass=mechanics_before_flags["opening"],
         opening_after_pass=mechanics_after_flags["opening"],
         slope_before_pass=mechanics_before_flags["slope"],
         slope_after_pass=mechanics_after_flags["slope"],
         work_before_pass=mechanics_before_flags["work"],
         work_after_pass=mechanics_after_flags["work"],
         cg_before_pass=mechanics_before_flags["cg"],
         cg_after_pass=mechanics_after_flags["cg"],
        mechanics_before_free_residual_norm=np.inf if mechanics_before is None else mechanics_before.mechanics_result.free_residual_norm,
        mechanics_after_free_residual_norm=np.inf if mechanics_after is None else mechanics_after.mechanics_result.free_residual_norm,
        mechanics_before_free_residual_relative_norm=np.inf if mechanics_before is None else mechanics_before.mechanics_result.free_residual_relative_norm,
        mechanics_after_free_residual_relative_norm=np.inf if mechanics_after is None else mechanics_after.mechanics_result.free_residual_relative_norm,
         mechanics_pass=bool(
             mechanics_before_flags["mechanics"]
             and mechanics_after_flags["mechanics"]
         ),
        seed_before=seed_before,
        seed_after=seed_after,
        available_fields=tuple(available),
    )


@dataclass(frozen=True)
class A2AcceptedStep:
    load_start: float
    load_target: float
    increment: float
    state: A2CommittedState
    damage_before: np.ndarray
    d_iter: np.ndarray
    d_next: np.ndarray
    q_before: np.ndarray
    q_after: np.ndarray
    Psi_before: np.ndarray
    Psi_after: np.ndarray
    H_used: np.ndarray
    H_next: np.ndarray
    mechanics_before: A2OpeningMechanicsResult
    mechanics_after: A2OpeningMechanicsResult
    diagnostics: A2Diagnostics
    stagger_iterations: int
    cutback_count: int
    acceptance_mode: str
    acceptance_status: str

    def __post_init__(self) -> None:
        for name in ("load_start", "load_target", "increment"):
            object.__setattr__(self, name, _scalar(getattr(self, name), f"A2AcceptedStep.{name}"))
        if self.load_target <= self.load_start or self.increment <= 0.0:
            raise ValueError("A2 accepted step load interval is invalid")
        if self.load_target - self.load_start != self.increment:
            raise ValueError("A2 accepted increment does not equal its load interval")
        if not isinstance(self.state, A2CommittedState) or not isinstance(self.diagnostics, A2Diagnostics):
            raise TypeError("A2 accepted step state/diagnostics are invalid")
        if not isinstance(self.mechanics_before, A2OpeningMechanicsResult) or not isinstance(self.mechanics_after, A2OpeningMechanicsResult):
            raise TypeError("A2 accepted step mechanics results are invalid")
        if self.acceptance_mode != A2_ACCEPTANCE_MODE:
            raise ValueError("A2 accepted steps have literal acceptance_mode='strict'")
        if self.acceptance_status != A2_ACCEPTANCE_STATUS:
            raise ValueError("A2 accepted steps have literal acceptance_status='strict_passed'")
        if not self.diagnostics.complete_sweep or not self.diagnostics.passed:
            raise ValueError("A2 accepted step requires one complete strict-passed sweep")
        n = self.state.material_damage.size
        for name in ("damage_before", "d_iter", "d_next", "Psi_before", "Psi_after", "H_used", "H_next"):
            object.__setattr__(self, name, _immutable(_f64(getattr(self, name), f"A2AcceptedStep.{name}", (n,))))
        for name in ("q_before", "q_after"):
            q = _f64(getattr(self, name), f"A2AcceptedStep.{name}")
            if q.ndim != 1 or q.size == 0 or q.size % 2:
                raise ValueError(f"A2AcceptedStep.{name} must be a flat displacement")
            object.__setattr__(self, name, _immutable(q))
        if self.state.physical_opening != self.load_target:
            raise ValueError("A2 accepted state opening is not paired with load_target")
        if self.state.outer_displacement != self.mechanics_after.outer_displacement:
            raise ValueError("A2 accepted state outer displacement is not paired")
        if not np.array_equal(self.state.union_displacement, self.q_after) or not np.array_equal(self.state.material_damage, self.d_next) or not np.array_equal(self.state.material_history, self.H_next):
            raise ValueError("A2 accepted state is not paired with final sweep fields")
        if not np.array_equal(self.q_before, self.mechanics_before.displacement) or not np.array_equal(self.q_after, self.mechanics_after.displacement):
            raise ValueError("A2 accepted displacement fields are not paired with mechanics results")
        for name, expected in (
            ("damage_before", self.diagnostics.damage_before),
            ("d_iter", self.diagnostics.d_iter),
            ("d_next", self.diagnostics.d_next),
            ("Psi_before", self.diagnostics.Psi_before),
            ("Psi_after", self.diagnostics.Psi_after),
            ("H_used", self.diagnostics.H_used),
            ("H_next", self.diagnostics.H_next),
            ("rho_used", self.diagnostics.rho_used),
            ("rho_next", self.diagnostics.rho_next),
            ("rho_used_volume", self.diagnostics.rho_used_volume),
            ("rho_next_volume", self.diagnostics.rho_next_volume),
        ):
            actual = getattr(self, name) if hasattr(self, name) else getattr(self.diagnostics, name)
            if expected is None or not np.array_equal(actual, expected):
                raise ValueError(f"A2 accepted field {name} is not paired with diagnostics")
        object.__setattr__(self, "stagger_iterations", _integer(self.stagger_iterations, "stagger_iterations", positive=True))
        object.__setattr__(self, "cutback_count", _integer(self.cutback_count, "cutback_count"))

    @property
    def rho_used(self) -> np.ndarray | None:
        return self.diagnostics.rho_used

    @property
    def rho_next(self) -> np.ndarray | None:
        return self.diagnostics.rho_next

    @property
    def mechanics_before_result(self) -> A2OpeningMechanicsResult:
        return self.mechanics_before

    @property
    def mechanics_after_result(self) -> A2OpeningMechanicsResult:
        return self.mechanics_after


@dataclass(frozen=True)
class A2AttemptRecord:
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
    sweeps: tuple[A2Diagnostics, ...]
    diagnostics: A2Diagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutback_count", _integer(self.cutback_count, "cutback_count"))
        object.__setattr__(self, "load_start", _scalar(self.load_start, "attempt.load_start"))
        object.__setattr__(self, "load_target", _scalar(self.load_target, "attempt.load_target"))
        if self.load_target <= self.load_start:
            raise ValueError("A2 attempt load interval is invalid")
        if self.acceptance_mode != A2_ACCEPTANCE_MODE:
            raise ValueError("A2 attempt acceptance_mode must be strict")
        if self.accepted and self.acceptance_status != A2_ACCEPTANCE_STATUS:
            raise ValueError("accepted A2 attempt must have acceptance_status='strict_passed'")
        if not self.accepted and self.acceptance_status is not None:
            raise ValueError("rejected A2 attempt cannot have an acceptance status")
        for name in ("committed_before_digest", "committed_after_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"A2 attempt {name} is not a SHA256 digest")
        if not isinstance(self.diagnostics, A2Diagnostics) or not self.sweeps:
            raise ValueError("A2 attempt must retain at least one sweep diagnostic")
        if self.sweeps[-1] is not self.diagnostics:
            raise ValueError("A2 attempt diagnostics must be its final sweep")
        if self.accepted:
            if self.failure_stage is not None or self.failure_reason is not None or self.exception_type is not None:
                raise ValueError("accepted A2 attempt cannot retain failure fields")
            if not self.diagnostics.complete_sweep or not self.diagnostics.passed:
                raise ValueError("accepted A2 attempt requires complete strict-passed diagnostics")
        else:
            if self.failure_stage is None or self.failure_reason is None:
                raise ValueError("rejected A2 attempt must retain typed failure fields")
            if self.diagnostics.passed:
                raise ValueError("rejected A2 attempt cannot pass")
            if not self.failure_stage or not self.failure_reason:
                raise ValueError("rejected A2 attempt failure fields must be non-empty")
            _validate_a2_attempt_exception_type(
                self.failure_stage,
                self.diagnostics.complete_sweep,
                self.exception_type,
            )
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "sweeps", tuple(self.sweeps))

    @property
    def complete_sweep(self) -> bool:
        return self.diagnostics.complete_sweep

    @property
    def before_digest(self) -> str:
        return self.committed_before_digest

    @property
    def after_digest(self) -> str:
        return self.committed_after_digest


class A2SweepFailure(RuntimeError):
    """Numerical incomplete/complete sweep failure eligible for cutback."""

    def __init__(self, stage: str, message: str, diagnostics: A2Diagnostics):
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


def _a2_mechanics_gate_flags(
    config: A2StrictConfig,
    mechanics: A2OpeningMechanicsResult | None,
    target: float,
    geometry_length: float,
) -> dict[str, bool]:
    """Return the configured mechanics-gate flags for one opening solve."""

    if mechanics is None:
        return {
            "opening": False,
            "slope": False,
            "work": False,
            "force": False,
            "moment": False,
            "cg": False,
            "mechanics": False,
        }
    if not isinstance(mechanics, A2OpeningMechanicsResult):
        raise TypeError("mechanics must be an A2OpeningMechanicsResult")
    target_value = _scalar(target, "mechanics gate target opening")
    if target_value < 0.0:
        raise ValueError("mechanics gate target opening must be non-negative")
    conservation = _conservation_metrics(config, mechanics, geometry_length)
    force_pass = bool(mechanics.force_pass and conservation["force_pass"])
    moment_pass = bool(mechanics.moment_pass and conservation["moment_pass"])
    opening_pass = bool(
        abs(mechanics.achieved_opening - target_value) <= config.opening_atol
    )
    cg_pass = bool(
        mechanics.mechanics_result.cg_info == 0
        and mechanics.mechanics_result.free_residual_norm
        < mechanics.mechanics_result.cg_convergence_limit
    )
    return {
        "opening": opening_pass,
        "slope": bool(mechanics.slope_pass),
        "work": bool(mechanics.work_pass),
        "force": force_pass,
        "moment": moment_pass,
        "cg": cg_pass,
        "mechanics": bool(
            opening_pass
            and mechanics.slope_pass
            and mechanics.work_pass
            and force_pass
            and moment_pass
            and cg_pass
        ),
    }


def _a2_mechanics_gate_reason_from_flags(
    phase: str, flags: Mapping[str, bool]
) -> str | None:
    """Return the deterministic reason for an early mechanics-gate failure."""

    failed: list[str] = []
    if not flags["opening"]:
        failed.append("opening")
    if not flags["slope"]:
        failed.append("slope")
    if not flags["work"]:
        failed.append("work")
    if not flags["force"] or not flags["moment"]:
        failed.append("conservation")
    if not flags["cg"]:
        failed.append("mechanics")
    if not failed:
        return None
    return f"{phase} {'/'.join(failed)} gate failed"


def _a2_mechanics_gate_reason(
    phase: str,
    mechanics: A2OpeningMechanicsResult,
    config: A2StrictConfig,
    target: float,
    geometry_length: float,
) -> str | None:
    """Return the production mechanics-gate reason from frozen settings."""

    return _a2_mechanics_gate_reason_from_flags(
        phase, _a2_mechanics_gate_flags(config, mechanics, target, geometry_length)
    )


def _validate_a2_attempt_exception_type(
    failure_stage: str | None,
    complete_sweep: bool,
    exception_type: str | None,
) -> None:
    """Validate the typed exception contract for numerical sweep failures."""

    if complete_sweep:
        if exception_type is not None:
            raise ValueError("complete failed A2 attempt cannot retain an exception type")
        return
    if not isinstance(exception_type, str) or not exception_type:
        raise ValueError("incomplete A2 attempt must retain a non-empty exception type")
    if failure_stage in _A2_SWEEP_FAILURE_STAGES and exception_type != A2SweepFailure.__name__:
        raise ValueError(
            "gate-triggered incomplete A2 attempt must retain an A2SweepFailure type"
        )


class A2ContinuationFailure(RuntimeError):
    """Typed strict cutback/minimum failure retaining immutable audit data."""

    def __init__(self, message: str, last_committed_state: A2CommittedState, attempts: tuple[A2AttemptRecord, ...]):
        super().__init__(message)
        self.last_committed_state = last_committed_state
        self.attempts = attempts
        self.message = message


def _damage_residuals(
    operator: ExplicitVolumeSparseD,
    Gc: float,
    l0: float,
    damage: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    D = operator.to_scipy()
    graph = np.asarray(D @ damage, dtype=np.float64)
    local = Gc / l0 * damage
    graph_term = Gc * l0 * graph
    rho_used = local + graph_term + 2.0 * H_used * damage - 2.0 * H_used
    rho_next = local + graph_term + 2.0 * H_next * damage - 2.0 * H_next
    volume = np.asarray(operator.volumes, dtype=np.float64)
    terms = (local, graph_term, 2.0 * H_used * damage, 2.0 * H_used, 2.0 * H_next * damage, 2.0 * H_next)
    scale = max((_max_abs(term) for term in terms), default=0.0)
    if not all(np.all(np.isfinite(term)) for term in terms):
        raise FloatingPointError("A2 damage residual terms are non-finite")
    return rho_used, rho_next, volume * rho_used, volume * rho_next, scale


def _compare_residual_reconstruction(
    strong_actual: np.ndarray,
    volume_actual: np.ndarray,
    damage: np.ndarray,
    history: np.ndarray,
    matrix: np.ndarray,
    volumes: np.ndarray,
    Gc: float,
    l0: float,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare strong/volume residuals with the declared componentwise bound."""

    d = _f64(damage, f"{name} damage")
    H = _f64(history, f"{name} history", d.shape)
    M = _f64(matrix, f"{name} damage matrix", (len(d), len(d)))
    V = _f64(volumes, f"{name} volumes", d.shape)
    local = _scalar(Gc, f"{name} Gc") / _scalar(l0, f"{name} l0") * d
    graph = _scalar(Gc, f"{name} Gc") * _scalar(l0, f"{name} l0") * (M @ d)
    history_d = 2.0 * H * d
    history_constant = 2.0 * H
    expected = local + graph + history_d - history_constant
    bound = 512.0 * _EPS64 * (
        np.abs(local) + np.abs(graph) + np.abs(history_d) + np.abs(history_constant)
    )
    actual = _f64(strong_actual, f"{name} strong residual", d.shape)
    actual_volume = _f64(volume_actual, f"{name} volume residual", d.shape)
    expected_volume = V * expected
    volume_bound = V * bound
    if np.any(np.abs(actual - expected) > bound) or np.any(
        np.abs(actual_volume - expected_volume) > volume_bound
    ):
        strong_defect = _max_abs(actual - expected)
        volume_defect = _max_abs(actual_volume - expected_volume)
        raise ValueError(
            f"{name} residual reconstruction exceeds 512eps constituent bound: "
            f"strong_defect={strong_defect}, volume_defect={volume_defect}"
        )
    return expected, expected_volume


def _full_diagnostics(
    config: A2StrictConfig,
    target: float,
    committed: A2CommittedState,
    d_iter: np.ndarray,
    d_next: np.ndarray,
    H_used: np.ndarray,
    H_next: np.ndarray,
    before: A2OpeningMechanicsResult,
    after: A2OpeningMechanicsResult,
    rho_used: np.ndarray,
    rho_next: np.ndarray,
    rho_used_volume: np.ndarray,
    rho_next_volume: np.ndarray,
    residual_scale: float,
    psi_before: np.ndarray,
    psi_after: np.ndarray,
    geometry_length: float,
    seed_before: A2SeedProvenance,
    seed_after: A2SeedProvenance,
    sweep_index: int,
) -> A2Diagnostics:
    q_before = before.displacement
    q_after = after.displacement
    e_um = _max_abs(q_after[: 2 * len(committed.material_damage)] - q_before[: 2 * len(committed.material_damage)])
    s_um = max(_max_abs(q_before[: 2 * len(committed.material_damage)]), _max_abs(q_after[: 2 * len(committed.material_damage)]), target)
    e_uu = _max_abs(q_after - q_before)
    s_uu = max(_max_abs(q_before), _max_abs(q_after), abs(before.outer_displacement), abs(after.outer_displacement), target)
    e_d = _max_abs(d_next - d_iter)
    s_d = max(_max_abs(d_iter), _max_abs(d_next))
    e_h = _max_abs(H_next - H_used)
    s_h = max(_max_abs(committed.material_history), _max_abs(H_used), _max_abs(H_next))
    e_r = abs(after.R_eff - before.R_eff)
    s_r = max(abs(before.R_eff), abs(after.R_eff))
    damage_min = float(np.min(d_next))
    damage_max = float(np.max(d_next))
    healing_min = float(np.min(d_next - committed.material_damage))
    history_gap = float(np.min(H_next - committed.material_history))
    conservation_before = _conservation_metrics(config, before, geometry_length)
    conservation_after = _conservation_metrics(config, after, geometry_length)
    mechanics_before_flags = _a2_mechanics_gate_flags(
        config, before, target, geometry_length
    )
    mechanics_after_flags = _a2_mechanics_gate_flags(
        config, after, target, geometry_length
    )
    force_error = max(float(conservation_before["force_error"]), float(conservation_after["force_error"]))
    force_scale = max(float(conservation_before["force_scale"]), float(conservation_after["force_scale"]))
    moment_error = max(float(conservation_before["moment_error"]), float(conservation_after["moment_error"]))
    moment_scale = max(float(conservation_before["moment_scale"]), float(conservation_after["moment_scale"]))
    opening_before_pass = mechanics_before_flags["opening"]
    opening_after_pass = mechanics_after_flags["opening"]
    mechanics_pass = bool(
        mechanics_before_flags["mechanics"]
        and mechanics_after_flags["mechanics"]
    )
    passes = {
        "opening_before": opening_before_pass,
        "opening_after": opening_after_pass,
        "slope_before": mechanics_before_flags["slope"],
        "slope_after": mechanics_after_flags["slope"],
        "work_before": mechanics_before_flags["work"],
        "work_after": mechanics_after_flags["work"],
        "cg_before": mechanics_before_flags["cg"],
        "cg_after": mechanics_after_flags["cg"],
        "displacement_material": e_um <= config.displacement_material.limit(s_um),
        "displacement_union": e_uu <= config.displacement_union.limit(s_uu),
        "damage": e_d <= config.damage.limit(s_d),
        "history": e_h <= config.history.limit(s_h),
        "reaction_effective": e_r <= config.reaction_effective.limit(s_r),
        "damage_residual": _max_abs(rho_next) <= config.damage_residual.limit(residual_scale),
        "damage_range": damage_min >= -config.damage_range_atol and damage_max <= 1.0 + config.damage_range_atol,
        "healing": healing_min >= -config.healing_atol,
        "history_monotonic": history_gap >= 0.0,
        "force_before": bool(conservation_before["force_pass"]),
        "force_after": bool(conservation_after["force_pass"]),
        "force": bool(conservation_before["force_pass"] and conservation_after["force_pass"]),
        "moment_before": bool(conservation_before["moment_pass"]),
        "moment_after": bool(conservation_after["moment_pass"]),
        "moment": bool(conservation_before["moment_pass"] and conservation_after["moment_pass"]),
        "work": mechanics_before_flags["work"] and mechanics_after_flags["work"],
        "mechanics": mechanics_pass,
    }
    passed = all(passes.values())
    reasons = ", ".join(name for name, value in passes.items() if not value)
    return A2Diagnostics(
        sweep_index=sweep_index,
        target_opening=target,
        opening_atol=config.opening_atol,
        complete_sweep=True,
        passed=passed,
        failure_stage="" if passed else "gates",
        failure_reason="" if passed else reasons,
        damage_before=committed.material_damage,
        d_iter=d_iter,
        mechanics_target_opening_before=before.target_opening,
        mechanics_target_opening_after=after.target_opening,
        d_next=d_next,
        q_before=q_before,
        q_after=q_after,
        Psi_before=psi_before,
        Psi_after=psi_after,
        H_used=H_used,
        H_next=H_next,
        U_before=before.outer_displacement,
        U_after=after.outer_displacement,
        slope_before=before.opening_slope,
        slope_after=after.opening_slope,
        achieved_opening_before=before.achieved_opening,
        achieved_opening_after=after.achieved_opening,
        R_ext_before=before.R_ext,
        R_ext_after=after.R_ext,
        R_eff_before=before.R_eff,
        R_eff_after=after.R_eff,
        rho_used=rho_used,
        rho_next=rho_next,
        rho_used_volume=rho_used_volume,
        rho_next_volume=rho_next_volume,
        rho_used_inf=_max_abs(rho_used),
        rho_next_inf=_max_abs(rho_next),
        e_opening_before=abs(before.achieved_opening - before.target_opening),
        e_opening_after=abs(after.achieved_opening - after.target_opening),
        e_u_material=e_um,
        u_material_scale=s_um,
        u_material_limit=config.displacement_material.limit(s_um),
        u_material_pass=passes["displacement_material"],
        e_u_union=e_uu,
        u_union_scale=s_uu,
        u_union_limit=config.displacement_union.limit(s_uu),
        u_union_pass=passes["displacement_union"],
        e_damage=e_d,
        damage_scale=s_d,
        damage_limit=config.damage.limit(s_d),
        damage_pass=passes["damage"],
        e_history=e_h,
        history_scale=s_h,
        history_limit=config.history.limit(s_h),
        history_pass=passes["history"],
        e_reaction_effective=e_r,
        reaction_effective_scale=s_r,
        reaction_effective_limit=config.reaction_effective.limit(s_r),
        reaction_effective_pass=passes["reaction_effective"],
        damage_residual_scale=residual_scale,
        damage_residual_limit=config.damage_residual.limit(residual_scale),
        damage_residual_pass=passes["damage_residual"],
        damage_min=damage_min,
        damage_max=damage_max,
        damage_range_pass=passes["damage_range"],
        healing_min=healing_min,
        healing_pass=passes["healing"],
        history_min_gap=history_gap,
        history_monotonic_pass=passes["history_monotonic"],
        force_before=before.mechanics_result.total_force,
        force_after=after.mechanics_result.total_force,
         moment_before=before.mechanics_result.total_moment,
         moment_after=after.mechanics_result.total_moment,
         force_before_error=float(conservation_before["force_error"]),
         force_before_scale=float(conservation_before["force_scale"]),
         force_before_limit=float(conservation_before["force_limit"]),
         force_before_pass=bool(conservation_before["force_pass"]),
         force_after_error=float(conservation_after["force_error"]),
         force_after_scale=float(conservation_after["force_scale"]),
         force_after_limit=float(conservation_after["force_limit"]),
         force_after_pass=bool(conservation_after["force_pass"]),
         force_error=force_error,
         force_scale=force_scale,
         force_limit=config.force.limit(force_scale),
         force_pass=passes["force"],
         moment_before_error=float(conservation_before["moment_error"]),
         moment_before_scale=float(conservation_before["moment_scale"]),
         moment_before_limit=float(conservation_before["moment_limit"]),
         moment_before_pass=bool(conservation_before["moment_pass"]),
         moment_after_error=float(conservation_after["moment_error"]),
         moment_after_scale=float(conservation_after["moment_scale"]),
         moment_after_limit=float(conservation_after["moment_limit"]),
         moment_after_pass=bool(conservation_after["moment_pass"]),
         moment_error=moment_error,
        moment_scale=moment_scale,
        moment_limit=config.moment.limit(moment_scale),
        moment_pass=passes["moment"],
        work_R_ext_before_error=before.R_ext_work_error,
        work_R_eff_before_error=before.R_eff_work_error,
        work_R_ext_after_error=after.R_ext_work_error,
        work_R_eff_after_error=after.R_eff_work_error,
         work_limit=config.work_relative_limit,
         work_pass=passes["work"],
         opening_before_pass=mechanics_before_flags["opening"],
         opening_after_pass=mechanics_after_flags["opening"],
         slope_before_pass=mechanics_before_flags["slope"],
         slope_after_pass=mechanics_after_flags["slope"],
         work_before_pass=mechanics_before_flags["work"],
         work_after_pass=mechanics_after_flags["work"],
         cg_before_pass=mechanics_before_flags["cg"],
         cg_after_pass=mechanics_after_flags["cg"],
         mechanics_before_free_residual_norm=before.mechanics_result.free_residual_norm,
        mechanics_after_free_residual_norm=after.mechanics_result.free_residual_norm,
        mechanics_before_free_residual_relative_norm=before.mechanics_result.free_residual_relative_norm,
        mechanics_after_free_residual_relative_norm=after.mechanics_result.free_residual_relative_norm,
        mechanics_pass=passes["mechanics"],
        seed_before=before.unit_result.seed_provenance,
        seed_after=after.unit_result.seed_provenance,
         available_fields=("damage_before", "d_iter", "d_next", "q_before", "q_after", "Psi_before", "Psi_after", "H_used", "H_next", "rho_used", "rho_next", "rho_used_volume", "rho_next_volume", "seed_before", "seed_after", "mechanics_target_opening_before", "mechanics_target_opening_after"),
    )


@dataclass(frozen=True)
class A2ContinuationMetadata:
    """Unequal material/union primitive metadata used by replay and archives."""

    schema: str
    raw_source_used: bool
    material_coordinates: np.ndarray
    material_volumes: np.ndarray
    material_triangles: np.ndarray
    material_edges: np.ndarray
    material_bottom_mask: np.ndarray
    material_top_mask: np.ndarray
    material_anchor: int
    union_coordinates: np.ndarray
    union_material_phase_volumes: np.ndarray
    union_grip_phase_volumes: np.ndarray
    union_volumes: np.ndarray
    grip_triangles: np.ndarray
    union_triangles: np.ndarray
    candidate_edges: np.ndarray
    visible_edges: np.ndarray
    removed_edges: np.ndarray
    interface_bottom_mask: np.ndarray
    interface_top_mask: np.ndarray
    outer_bottom_mask: np.ndarray
    outer_top_mask: np.ndarray
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    top_line_indices: np.ndarray
    top_line_weights: np.ndarray
    top_line_length: float
    bottom_line_indices: np.ndarray
    bottom_line_weights: np.ndarray
    bottom_line_length: float
    delta: float
    h_g: float
    visibility_tolerance: float
    visibility_rule: str
    youngs_modulus: float
    poisson_ratio: float
    k_res: float
    rank_rtol: float
    Gc: float
    l0: float
    static_digest: str
    material_hash: str
    union_hash: str
    config: A2StrictConfig
    code_hash: str

    @classmethod
    def from_model(cls, model: A2CoupledModel, config: A2StrictConfig) -> "A2ContinuationMetadata":
        geometry = model.geometry
        material_hash = _digest_primitives(
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
        union_hash = _digest_primitives(
            (
                ("coordinates", geometry.coordinates),
                ("material_phase_volumes", geometry.material_volumes),
                ("grip_phase_volumes", geometry.grip_volumes),
                ("union_volumes", geometry.union_volumes),
                ("union_triangles", geometry.union_triangles),
                ("candidate_edges", geometry.candidate_edges),
                ("visible_edges", geometry.directed_edges),
                ("removed_edges", geometry.removed_edges),
            )
        )
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return cls(
            schema=ARCHIVE_SCHEMA,
            raw_source_used=False,
            material_coordinates=geometry.material_reference_coordinates,
            material_volumes=geometry.material_reference_volumes,
            material_triangles=geometry.material_reference_triangles,
            material_edges=geometry.material_reference_edges,
            material_bottom_mask=geometry.material_bottom_mask[: geometry.n_material],
            material_top_mask=geometry.material_top_mask[: geometry.n_material],
            material_anchor=0,
            union_coordinates=geometry.coordinates,
            union_material_phase_volumes=geometry.material_volumes,
            union_grip_phase_volumes=geometry.grip_volumes,
            union_volumes=geometry.union_volumes,
            grip_triangles=geometry.grip_triangles,
            union_triangles=geometry.union_triangles,
            candidate_edges=geometry.candidate_edges,
            visible_edges=geometry.directed_edges,
            removed_edges=geometry.removed_edges,
            interface_bottom_mask=geometry.interface_bottom_mask,
            interface_top_mask=geometry.interface_top_mask,
            outer_bottom_mask=geometry.outer_bottom_mask,
            outer_top_mask=geometry.outer_top_mask,
            constrained_dofs=model.constraints.constrained_dofs,
            free_dofs=model.constraints.free_dofs,
            top_line_indices=model.top_line.indices,
            top_line_weights=model.top_line.weights,
            top_line_length=model.top_line.length,
            bottom_line_indices=model.bottom_line.indices,
            bottom_line_weights=model.bottom_line.weights,
            bottom_line_length=model.bottom_line.length,
            delta=geometry.delta,
            h_g=geometry.h_g,
            visibility_tolerance=geometry.visibility_tolerance,
            visibility_rule=_VISIBILITY_RULE,
            youngs_modulus=model.static.youngs_modulus,
            poisson_ratio=model.static.poisson_ratio,
            k_res=model.static.k_res,
            rank_rtol=model.static.rank_rtol,
            Gc=model.Gc,
            l0=model.l0,
            static_digest=model.static.deterministic_digest,
            material_hash=material_hash,
            union_hash=union_hash,
            config=config,
            code_hash=code_hash,
        )

    def __post_init__(self) -> None:
        if self.schema != ARCHIVE_SCHEMA:
            raise ValueError(f"A2 metadata schema must be {ARCHIVE_SCHEMA!r}")
        if bool(self.raw_source_used):
            raise ValueError("A2 coupled continuation requires raw_source_used=False")
        material = _f64(self.material_coordinates, "metadata.material_coordinates")
        union = _f64(self.union_coordinates, "metadata.union_coordinates")
        if material.ndim != 2 or material.shape[1] != 2 or union.ndim != 2 or union.shape[1] != 2:
            raise ValueError("A2 metadata material/union coordinates must be (N,2)")
        nm, nu = len(material), len(union)
        if nu <= nm or not np.array_equal(union[:nm], material):
            raise ValueError("A2 metadata union coordinates must preserve the material prefix")
        array_specs = (
            ("material_volumes", _FLOAT64, (nm,)),
            ("material_triangles", _INT64, None),
            ("material_edges", _INT64, (None, 2)),
            ("material_bottom_mask", _BOOL, (nm,)),
            ("material_top_mask", _BOOL, (nm,)),
            ("union_material_phase_volumes", _FLOAT64, (nu,)),
            ("union_grip_phase_volumes", _FLOAT64, (nu,)),
            ("union_volumes", _FLOAT64, (nu,)),
            ("grip_triangles", _INT64, None),
            ("union_triangles", _INT64, None),
            ("candidate_edges", _INT64, (None, 2)),
            ("visible_edges", _INT64, (None, 2)),
            ("removed_edges", _INT64, (None, 2)),
            ("interface_bottom_mask", _BOOL, (nu,)),
            ("interface_top_mask", _BOOL, (nu,)),
            ("outer_bottom_mask", _BOOL, (nu,)),
            ("outer_top_mask", _BOOL, (nu,)),
            ("constrained_dofs", _INT64, None),
            ("free_dofs", _INT64, None),
            ("top_line_indices", _INT64, None),
            ("top_line_weights", _FLOAT64, None),
            ("bottom_line_indices", _INT64, None),
            ("bottom_line_weights", _FLOAT64, None),
        )
        for name, dtype, shape in array_specs:
            array = np.asarray(getattr(self, name))
            if array.dtype != dtype:
                raise ValueError(f"A2 metadata {name} has dtype {array.dtype}, expected {dtype}")
            if shape is not None:
                if shape[0] is not None and array.shape[0] != shape[0]:
                    raise ValueError(f"A2 metadata {name} has an invalid shape")
                if len(shape) == 2 and shape[1] is not None and array.shape[1] != shape[1]:
                    raise ValueError(f"A2 metadata {name} has an invalid shape")
            if dtype.kind == "f" and not np.all(np.isfinite(array)):
                raise ValueError(f"A2 metadata {name} is non-finite")
            object.__setattr__(self, name, _immutable(array))
        object.__setattr__(self, "material_coordinates", _immutable(material))
        object.__setattr__(self, "union_coordinates", _immutable(union))
        if not np.array_equal(self.union_material_phase_volumes[:nm], self.material_volumes):
            raise ValueError("A2 metadata material phase prefix differs from damage volumes")
        if not np.array_equal(self.union_material_phase_volumes[nm:], np.zeros(nu - nm, dtype=np.float64)):
            raise ValueError("A2 metadata material phase is nonzero on grips")
        if not np.allclose(
            self.union_volumes,
            self.union_material_phase_volumes + self.union_grip_phase_volumes,
            rtol=0.0,
            atol=_REPLAY_BOUND * max(1.0, _max_abs(self.union_volumes)),
        ):
            raise ValueError("A2 metadata union volume is not the phase sum")
        if not np.array_equal(self.visible_edges[(self.visible_edges[:, 0] < nm) & (self.visible_edges[:, 1] < nm)], self.material_edges):
            raise ValueError("A2 metadata union graph does not induce the material graph")
        if not np.array_equal(np.sort(np.concatenate((self.constrained_dofs, self.free_dofs))), np.arange(2 * nu, dtype=np.int64)):
            raise ValueError("A2 metadata constrained/free DOFs do not partition the union")
        for name in ("top_line_indices", "bottom_line_indices"):
            if len(np.unique(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f"A2 metadata {name} contains duplicates")
        for name in ("top_line_length", "bottom_line_length", "delta", "h_g", "visibility_tolerance", "youngs_modulus", "poisson_ratio", "k_res", "rank_rtol", "Gc", "l0"):
            object.__setattr__(self, name, _scalar(getattr(self, name), f"metadata.{name}"))
        if self.delta <= 0.0 or self.h_g <= 0.0 or self.h_g >= self.delta or self.Gc <= 0.0 or self.l0 <= 0.0:
            raise ValueError("A2 metadata geometry/fracture parameters are invalid")
        if self.visibility_rule != _VISIBILITY_RULE:
            raise ValueError("A2 metadata visibility rule is not the frozen union rule")
        if self.material_anchor != 0:
            raise ValueError("A2 metadata material anchor must be the lower-left prefix particle")
        if not isinstance(self.config, A2StrictConfig):
            raise TypeError("A2 metadata config must be A2StrictConfig")
        for name in ("static_digest", "material_hash", "union_hash", "code_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"A2 metadata {name} must be a SHA256 digest")
        expected_material = _digest_primitives(
            (
                ("coordinates", self.material_coordinates), ("volumes", self.material_volumes),
                ("triangles", self.material_triangles), ("edges", self.material_edges),
                ("bottom", self.material_bottom_mask), ("top", self.material_top_mask), ("anchor", self.material_anchor),
            )
        )
        expected_union = _digest_primitives(
            (
                ("coordinates", self.union_coordinates), ("material_phase_volumes", self.union_material_phase_volumes),
                ("grip_phase_volumes", self.union_grip_phase_volumes), ("union_volumes", self.union_volumes),
                ("union_triangles", self.union_triangles), ("candidate_edges", self.candidate_edges),
                ("visible_edges", self.visible_edges), ("removed_edges", self.removed_edges),
            )
        )
        if self.material_hash != expected_material or self.union_hash != expected_union:
            raise ValueError("A2 metadata cloud hash mismatch")
        object.__setattr__(self, "material_anchor", _integer(self.material_anchor, "material_anchor"))

    @property
    def n_material(self) -> int:
        return len(self.material_coordinates)

    @property
    def n_union(self) -> int:
        return len(self.union_coordinates)

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_union


@dataclass(frozen=True)
class A2ContinuationResult:
    initial_state: A2CommittedState
    metadata: A2ContinuationMetadata
    states: tuple[A2CommittedState, ...]
    accepted_steps: tuple[A2AcceptedStep, ...]
    attempts: tuple[A2AttemptRecord, ...]
    model: A2CoupledModel
    acceptance_mode: str
    acceptance_status: str
    completion_status: str

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "accepted_steps", tuple(self.accepted_steps))
        object.__setattr__(self, "attempts", tuple(self.attempts))

    @property
    def final_state(self) -> A2CommittedState:
        return self.states[-1]

    def validate(self) -> None:
        if not isinstance(self.initial_state, A2CommittedState) or not isinstance(self.metadata, A2ContinuationMetadata) or not isinstance(self.model, A2CoupledModel):
            raise TypeError("A2 continuation result types are invalid")
        if not self.states or self.states[0] is not self.initial_state:
            raise ValueError("A2 states must begin with the exact initial_state object")
        if len(self.states) != len(self.accepted_steps) + 1:
            raise ValueError("A2 states must contain initial plus accepted states")
        if self.acceptance_mode != A2_ACCEPTANCE_MODE:
            raise ValueError("A2 result acceptance mode must be strict")
        if self.acceptance_status != A2_ACCEPTANCE_STATUS:
            raise ValueError("A2 result acceptance status must be strict_passed")
        if self.completion_status != A2_COMPLETION_STATUS:
            raise ValueError("A2 result completion status is not completed")
        if self.metadata.static_digest != self.model.static.deterministic_digest:
            raise ValueError("A2 result metadata/static digest mismatch")
        if self.metadata.config.acceptance_mode != A2_ACCEPTANCE_MODE:
            raise ValueError("A2 result acceptance mode must be strict")
        if self.final_state.physical_opening != self.metadata.config.opening_final:
            raise ValueError("A2 completed result final opening is not exactly opening_final")
        if self.initial_state.physical_opening != self.metadata.config.opening_start or self.initial_state.outer_displacement != 0.0:
            raise ValueError("A2 initial state scalar values are not exact zero-start values")
        if not np.array_equal(self.initial_state.union_displacement, np.zeros(self.model.static.n_dofs, dtype=np.float64)) or not np.array_equal(self.initial_state.material_damage, np.zeros(self.model.static.material_count, dtype=np.float64)) or not np.array_equal(self.initial_state.material_history, np.zeros(self.model.static.material_count, dtype=np.float64)):
            raise ValueError("A2 initial state must be exact zero displacement/damage/history")
        for state in self.states:
            if state.union_displacement.shape != (self.model.static.n_dofs,) or state.material_damage.shape != (self.model.static.material_count,):
                raise ValueError("A2 state shapes do not match unequal-cloud model")
        for index, step in enumerate(self.accepted_steps, start=1):
            if not isinstance(step, A2AcceptedStep) or self.states[index] is not step.state:
                raise ValueError("A2 accepted steps and states are not paired")
            if step.load_start != self.states[index - 1].physical_opening or step.load_target != self.states[index].physical_opening:
                raise ValueError("A2 accepted load pairing is invalid")
        if not self.attempts:
            raise ValueError("A2 continuation must retain attempt audit records")
        for attempt in self.attempts:
            if not isinstance(attempt, A2AttemptRecord):
                raise TypeError("A2 attempts must contain A2AttemptRecord values")
            if attempt.acceptance_mode != self.acceptance_mode:
                raise ValueError("A2 attempt acceptance mode is not paired with result")
            if attempt.accepted and attempt.acceptance_status != self.acceptance_status:
                raise ValueError("A2 accepted attempt status is not paired with result")
        for step in self.accepted_steps:
            if step.acceptance_mode != self.acceptance_mode or step.acceptance_status != self.acceptance_status:
                raise ValueError("A2 accepted step status is not paired with result")
        independent_replay(self)


def _next_opening_target(committed: float, increment: float, final: float) -> float:
    remaining = final - committed
    target = committed + min(increment, remaining)
    if final - target <= 8.0 * abs(np.spacing(np.float64(final))):
        return final
    return target


def _model_geometry_length(model: A2CoupledModel) -> float:
    coordinates = model.geometry.coordinates
    return max(_max_abs(coordinates), _max_abs(np.ptp(coordinates, axis=0)), 1.0)


def _run_a2_attempt(
    model: A2CoupledModel,
    config: A2StrictConfig,
    committed: A2CommittedState,
    target: float,
    cutback_count: int,
    step_index: int,
) -> tuple[A2AcceptedStep | None, A2AttemptRecord]:
    d_iter = np.array(committed.material_damage, dtype=np.float64, copy=True)
    seed_q: np.ndarray | None = None
    seed_source_damage_digest: str | None = None
    sweeps: list[A2Diagnostics] = []
    final_diagnostic: A2Diagnostics | None = None
    exception_type: str | None = None
    for sweep_index in range(config.max_stagger_iterations):
        view_before = A2FixedDamageView(model.static, d_iter)
        if seed_q is None:
            seed_previous_q = committed.union_displacement
            seed_previous_outer = committed.outer_displacement
            seed_source_damage_digest = _damage_digest(committed.material_damage)
            source_state = "committed"
        else:
            seed_previous_q = seed_q
            seed_previous_outer = 1.0
            if seed_source_damage_digest is None:
                raise RuntimeError("A2 normalized seed lost its source damage digest")
            source_state = "complete_trial_sweep"
        try:
            seed_q = build_a2_unit_seed(
                model.constraints,
                seed_previous_q,
                seed_previous_outer,
            )
            before = solve_a2_opening_controlled(
                view_before,
                model.constraints,
                target,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
                config.opening_slope_min,
                config.opening_atol,
                config.work_epsilon,
                config.work_relative_limit,
                config.work_cg_rtol,
                config.work_cg_atol,
                initial_guess=seed_q,
                source_state=source_state,
                source_step=step_index,
                source_sweep=sweep_index,
                source_damage_digest=seed_source_damage_digest,
            )
        except (RuntimeError, FloatingPointError) as error:
            exception_type = type(error).__name__
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_before",
                str(error),
                seed_before=None,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweep_failure = A2SweepFailure("mechanics_before", str(error), diagnostic)
            exception_type = type(sweep_failure).__name__
            sweeps.append(sweep_failure.diagnostics)
            final_diagnostic = sweep_failure.diagnostics
            break

        try:
            psi_before = compute_a2_amor_history(
                view_before, before.displacement, committed.material_history
            ).particle_driver
        except (RuntimeError, FloatingPointError) as error:
            exception_type = type(error).__name__
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "amor_before",
                str(error),
                q_before=before.displacement,
                mechanics_before=before,
                seed_before=before.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweeps.append(diagnostic)
            final_diagnostic = diagnostic
            break

        geometry_length = _model_geometry_length(model)
        mechanics_before_reason = _a2_mechanics_gate_reason(
            "mechanics-before", before, config, target, geometry_length
        )
        if mechanics_before_reason is not None:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_before_gate",
                mechanics_before_reason,
                q_before=before.displacement,
                psi_before=psi_before,
                mechanics_before=before,
                seed_before=before.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweep_failure = A2SweepFailure(
                "mechanics_before_gate", mechanics_before_reason, diagnostic
            )
            exception_type = type(sweep_failure).__name__
            sweeps.append(sweep_failure.diagnostics)
            final_diagnostic = sweep_failure.diagnostics
            break

        H_used = np.maximum(committed.material_history, psi_before).astype(np.float64, copy=False)
        H_used = _immutable(H_used)
        try:
            damage_result = _material.solve_fixed_history_damage(
                model.material_operator, model.Gc, model.l0, H_used
            )
            if not isinstance(damage_result, FixedHistoryDamageResult):
                raise TypeError("fixed-history solve returned the wrong result type")
            d_next = _f64(damage_result.damage, "A2 fixed-history damage", d_iter.shape)
            if not np.array_equal(damage_result.history, H_used):
                raise ValueError("fixed-history damage result is bound to a different H_used")
            rho_used = _f64(damage_result.strong_residual, "A2 rho_used", d_iter.shape)
            rho_used_volume = np.asarray(model.material_operator.volumes * rho_used, dtype=np.float64)
        except (RuntimeError, FloatingPointError) as error:
            exception_type = type(error).__name__
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "damage",
                str(error),
                q_before=before.displacement,
                psi_before=psi_before,
                H_used=H_used,
                mechanics_before=before,
                seed_before=before.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweeps.append(diagnostic)
            final_diagnostic = diagnostic
            break

        d_next = np.asarray(d_next, dtype=np.float64)
        range_pass = float(np.min(d_next)) >= -config.damage_range_atol and float(np.max(d_next)) <= 1.0 + config.damage_range_atol
        healing_pass = float(np.min(d_next - committed.material_damage)) >= -config.healing_atol
        if not range_pass or not healing_pass:
            failure_names = []
            if not range_pass:
                failure_names.append("damage_range")
            if not healing_pass:
                failure_names.append("healing")
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "damage_acceptance",
                ", ".join(failure_names),
                q_before=before.displacement,
                psi_before=psi_before,
                H_used=H_used,
                d_next=d_next,
                rho_used=rho_used,
                rho_used_volume=rho_used_volume,
                mechanics_before=before,
                seed_before=before.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweep_failure = A2SweepFailure(
                "damage_acceptance", ", ".join(failure_names), diagnostic
            )
            exception_type = type(sweep_failure).__name__
            sweeps.append(sweep_failure.diagnostics)
            final_diagnostic = sweep_failure.diagnostics
            break

        view_after = A2FixedDamageView(model.static, d_next)
        try:
            after_unit_seed = build_a2_unit_seed(
                model.constraints,
                before.displacement,
                before.outer_displacement,
            )
            after = solve_a2_opening_controlled(
                view_after,
                model.constraints,
                target,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
                config.opening_slope_min,
                config.opening_atol,
                config.work_epsilon,
                config.work_relative_limit,
                config.work_cg_rtol,
                config.work_cg_atol,
                initial_guess=after_unit_seed,
                source_state="mechanics_before",
                source_step=step_index,
                source_sweep=sweep_index,
                source_damage_digest=view_before.damage_digest,
            )
        except (RuntimeError, FloatingPointError) as error:
            exception_type = type(error).__name__
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_after",
                str(error),
                q_before=before.displacement,
                psi_before=psi_before,
                H_used=H_used,
                d_next=d_next,
                rho_used=rho_used,
                rho_used_volume=rho_used_volume,
                mechanics_before=before,
                seed_before=before.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweep_failure = A2SweepFailure("mechanics_after", str(error), diagnostic)
            exception_type = type(sweep_failure).__name__
            sweeps.append(sweep_failure.diagnostics)
            final_diagnostic = sweep_failure.diagnostics
            break

        try:
            psi_after = compute_a2_amor_history(
                view_after, after.displacement, committed.material_history
            ).particle_driver
        except (RuntimeError, FloatingPointError) as error:
            exception_type = type(error).__name__
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "amor_after",
                str(error),
                q_before=before.displacement,
                q_after=after.displacement,
                psi_before=psi_before,
                H_used=H_used,
                d_next=d_next,
                rho_used=rho_used,
                rho_used_volume=rho_used_volume,
                mechanics_before=before,
                mechanics_after=after,
                seed_before=before.unit_result.seed_provenance,
                seed_after=after.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweeps.append(diagnostic)
            final_diagnostic = diagnostic
            break

        geometry_length = _model_geometry_length(model)
        mechanics_after_reason = _a2_mechanics_gate_reason(
            "mechanics-after", after, config, target, geometry_length
        )
        if mechanics_after_reason is not None:
            diagnostic = _incomplete_diagnostics(
                config,
                committed,
                d_iter,
                target,
                "mechanics_after_gate",
                mechanics_after_reason,
                q_before=before.displacement,
                q_after=after.displacement,
                psi_before=psi_before,
                psi_after=psi_after,
                H_used=H_used,
                d_next=d_next,
                rho_used=rho_used,
                rho_used_volume=rho_used_volume,
                mechanics_before=before,
                mechanics_after=after,
                seed_before=before.unit_result.seed_provenance,
                seed_after=after.unit_result.seed_provenance,
                sweep_index=sweep_index,
                geometry_length=_model_geometry_length(model),
            )
            sweep_failure = A2SweepFailure(
                "mechanics_after_gate", mechanics_after_reason, diagnostic
            )
            exception_type = type(sweep_failure).__name__
            sweeps.append(sweep_failure.diagnostics)
            final_diagnostic = sweep_failure.diagnostics
            break

        H_next = _immutable(np.maximum(committed.material_history, psi_after).astype(np.float64, copy=False))
        rho_used_check, rho_next, rho_used_volume_check, rho_next_volume, residual_scale = _damage_residuals(
            model.material_operator, model.Gc, model.l0, d_next, H_used, H_next
        )
        _array_close(rho_used, rho_used_check, "A2 rho_used reconstruction", 1.0e-10)
        _array_close(rho_used_volume, rho_used_volume_check, "A2 rho_used volume reconstruction", 1.0e-10)
        diagnostics = _full_diagnostics(
            config,
            target,
            committed,
            d_iter,
            d_next,
            H_used,
            H_next,
            before,
            after,
            rho_used,
            rho_next,
            rho_used_volume,
            rho_next_volume,
            residual_scale,
            psi_before,
            psi_after,
            _model_geometry_length(model),
            before.unit_result.seed_provenance,
            after.unit_result.seed_provenance,
            sweep_index,
        )
        sweeps.append(diagnostics)
        final_diagnostic = diagnostics
        if diagnostics.passed:
            state = A2CommittedState(
                physical_opening=target,
                outer_displacement=after.outer_displacement,
                union_displacement=after.displacement,
                material_damage=d_next,
                material_history=H_next,
            )
            step = A2AcceptedStep(
                load_start=committed.physical_opening,
                load_target=target,
                increment=target - committed.physical_opening,
                state=state,
                damage_before=committed.material_damage,
                d_iter=d_iter,
                d_next=d_next,
                q_before=before.displacement,
                q_after=after.displacement,
                Psi_before=psi_before,
                Psi_after=psi_after,
                H_used=H_used,
                H_next=H_next,
                mechanics_before=before,
                mechanics_after=after,
                diagnostics=diagnostics,
                stagger_iterations=sweep_index + 1,
                cutback_count=cutback_count,
                acceptance_mode=A2_ACCEPTANCE_MODE,
                acceptance_status=A2_ACCEPTANCE_STATUS,
            )
            attempt = A2AttemptRecord(
                cutback_count=cutback_count,
                load_start=committed.physical_opening,
                load_target=target,
                accepted=True,
                acceptance_mode=A2_ACCEPTANCE_MODE,
                acceptance_status=A2_ACCEPTANCE_STATUS,
                failure_stage=None,
                failure_reason=None,
                exception_type=None,
                committed_before_digest=committed.digest,
                committed_after_digest=state.digest,
                sweeps=tuple(sweeps),
                diagnostics=diagnostics,
            )
            return step, attempt
        if sweep_index + 1 < config.max_stagger_iterations:
            # Only a complete finite failed sweep updates local d_iter.  The
            # committed state/history and the cutback seed remain unchanged.
            d_iter = np.array(d_next, dtype=np.float64, copy=True)
            seed_q = build_a2_unit_seed(model.constraints, after.displacement, after.outer_displacement)
            seed_source_damage_digest = view_after.damage_digest
            continue
        break

    if final_diagnostic is None:
        raise RuntimeError("A2 attempt produced no diagnostic")
    reason = final_diagnostic.failure_reason or "strict A2 sweep failed"
    attempt = A2AttemptRecord(
        cutback_count=cutback_count,
        load_start=committed.physical_opening,
        load_target=target,
        accepted=False,
        acceptance_mode=A2_ACCEPTANCE_MODE,
        acceptance_status=None,
        failure_stage=final_diagnostic.failure_stage or "gates",
        failure_reason=reason,
        exception_type=exception_type,
        committed_before_digest=committed.digest,
        committed_after_digest=committed.digest,
        sweeps=tuple(sweeps),
        diagnostics=final_diagnostic,
    )
    return None, attempt


def run_a2_strict_continuation(model: A2CoupledModel, config: A2StrictConfig) -> A2ContinuationResult:
    """Run strict clean-zero-start A2/material continuation with rollback."""

    if not isinstance(model, A2CoupledModel) or not isinstance(config, A2StrictConfig):
        raise TypeError("run_a2_strict_continuation requires A2CoupledModel and A2StrictConfig")
    metadata = A2ContinuationMetadata.from_model(model, config)
    initial = A2CommittedState(
        physical_opening=0.0,
        outer_displacement=0.0,
        union_displacement=np.zeros(model.static.n_dofs, dtype=np.float64),
        material_damage=np.zeros(model.static.material_count, dtype=np.float64),
        material_history=np.zeros(model.static.material_count, dtype=np.float64),
    )
    states: list[A2CommittedState] = [initial]
    steps: list[A2AcceptedStep] = []
    attempts: list[A2AttemptRecord] = []
    committed = initial
    increment = config.d_opening_initial
    cutbacks = 0
    while committed.physical_opening < config.opening_final:
        target = _next_opening_target(committed.physical_opening, increment, config.opening_final)
        if target <= committed.physical_opening:
            raise A2ContinuationFailure(
                "strict A2 continuation made no floating-point opening progress",
                committed,
                tuple(attempts),
            )
        step, attempt = _run_a2_attempt(
            model,
            config,
            committed,
            target,
            cutbacks,
            len(steps) + 1,
        )
        attempts.append(attempt)
        if step is not None:
            steps.append(step)
            committed = step.state
            states.append(committed)
            increment = target - (step.load_start)
            cutbacks = 0
            continue
        if attempt.committed_before_digest != attempt.committed_after_digest:
            raise RuntimeError("A2 rejected attempt changed the committed-state digest")
        at_minimum = target - committed.physical_opening <= config.d_opening_min
        at_cutback_limit = cutbacks >= config.max_cutbacks
        if at_minimum or at_cutback_limit:
            reasons = []
            if at_minimum:
                reasons.append("d_opening_min")
            if at_cutback_limit:
                reasons.append("max_cutbacks")
            raise A2ContinuationFailure(
                f"strict A2 continuation failed at opening {target:.17g} ({', '.join(reasons)}): {attempt.failure_reason}",
                committed,
                tuple(attempts),
            )
        cutbacks += 1
        increment = max(config.d_opening_min, (target - committed.physical_opening) * config.cutback_factor)
    return A2ContinuationResult(
        initial_state=initial,
        metadata=metadata,
        states=tuple(states),
        accepted_steps=tuple(steps),
        attempts=tuple(attempts),
        model=model,
        acceptance_mode=A2_ACCEPTANCE_MODE,
        acceptance_status=A2_ACCEPTANCE_STATUS,
        completion_status=A2_COMPLETION_STATUS,
    )


run_strict_a2_continuation = run_a2_strict_continuation


# ---------------------------------------------------------------------------
# Independent primitive replay
# ---------------------------------------------------------------------------


def _replay_ba(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    directed_edges: np.ndarray,
    delta: float,
    rank_rtol: float,
) -> dict[str, Any]:
    """Rebuild all union BA primitives without importing production mechanics."""

    x = _f64(coordinates, "replay coordinates")
    v = _f64(volumes, "replay volumes", (len(x),))
    edges = _i64(directed_edges, "replay directed_edges")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("replay directed_edges must be nonempty (E,2)")
    if np.any(edges[:, 0] == edges[:, 1]) or np.any(edges < 0) or np.any(edges >= len(x)):
        raise ValueError("replay directed_edges are invalid")
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("replay directed_edges are not reciprocal")
    distance = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)
    delta_value = _scalar(delta, "replay delta")
    rank_value = _scalar(rank_rtol, "replay rank_rtol")
    if np.any(distance <= 0.0) or np.any(distance >= delta_value):
        raise ValueError("replay directed_edges violate strict horizon")
    omega = 120.0 / (np.pi * delta_value**4) * (1.0 - distance / delta_value) ** 2
    source = edges[:, 0]
    target = edges[:, 1]
    family = np.bincount(source, weights=omega * v[target], minlength=len(x)).astype(np.float64)
    if np.any(family <= 0.0) or not np.all(np.isfinite(family)):
        raise ValueError("replay family mass is invalid")
    adjacency = [set() for _ in range(len(x))]
    omega_map: dict[tuple[int, int], float] = {}
    for (i_raw, j_raw), value in zip(edges, omega):
        i, j = int(i_raw), int(j_raw)
        adjacency[i].add(j)
        omega_map[i, j] = float(value)
    supports: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    xis: list[np.ndarray] = []
    inverses: list[np.ndarray] = []
    ranks: list[int] = []
    conditions: list[float] = []
    for i_raw, j_raw in edges:
        i, j = int(i_raw), int(j_raw)
        common = sorted(adjacency[i].intersection(adjacency[j]) - {i, j})
        support = np.asarray([j, *common], dtype=np.int64)
        beta = np.asarray(
            [omega_map[i, j] * v[j]] + [omega_map[i, k] * v[k] for k in common],
            dtype=np.float64,
        )
        xi = x[support] - x[i]
        shape = np.einsum("s,sa,sb->ab", beta, xi, xi)
        singular = np.linalg.svd(shape, compute_uv=False)
        rank = int(np.count_nonzero(singular > rank_value * singular[0]))
        if rank < 2:
            raise ValueError("replay BA support is rank deficient")
        ranks.append(rank)
        conditions.append(float(singular[0] / singular[-1]))
        supports.append(support)
        betas.append(beta)
        xis.append(xi)
        inverses.append(np.linalg.inv(shape))
    eta = omega * v[target] / family[source]
    if np.any(eta <= 0.0) or not np.all(np.isfinite(eta)):
        raise ValueError("replay eta is invalid")
    return {
        "coordinates": x,
        "volumes": v,
        "edges": edges,
        "source": source,
        "target": target,
        "distance": distance,
        "omega": omega,
        "family_mass": family,
        "eta": eta,
        "supports": tuple(supports),
        "beta": tuple(betas),
        "xi": tuple(xis),
        "inverse": tuple(inverses),
        "shape_rank": np.asarray(ranks, dtype=np.int64),
        "shape_condition": np.asarray(conditions, dtype=np.float64),
    }


def _replay_static_digest(metadata: A2ContinuationMetadata, ba: Mapping[str, Any]) -> str:
    """Rebuild the exact static identity from archived primitives and BA algebra."""

    n_edges = len(ba["source"])
    max_support = max(len(support) for support in ba["supports"])
    support_indices = np.full((n_edges, max_support), -1, dtype=np.int64)
    support_mask = np.zeros((n_edges, max_support), dtype=bool)
    beta = np.zeros((n_edges, max_support), dtype=np.float64)
    xi = np.zeros((n_edges, max_support, 2), dtype=np.float64)
    for edge_index, support in enumerate(ba["supports"]):
        count = len(support)
        support_indices[edge_index, :count] = support
        support_mask[edge_index, :count] = True
        beta[edge_index, :count] = ba["beta"][edge_index]
        xi[edge_index, :count] = ba["xi"][edge_index]
    inverse = np.asarray(ba["inverse"], dtype=np.float64)
    gradient_coefficient = np.einsum(
        "es,esb,ebc->esc", beta, xi, inverse, optimize=True
    )
    n_dofs = metadata.n_dofs
    top_selector = np.zeros(n_dofs, dtype=np.float64)
    top_selector[2 * np.flatnonzero(metadata.outer_top_mask) + 1] = 1.0
    bottom_selector = np.zeros(n_dofs, dtype=np.float64)
    bottom_selector[2 * np.flatnonzero(metadata.outer_bottom_mask) + 1] = 1.0
    anchor_selector = np.zeros(n_dofs, dtype=np.float64)
    anchor_selector[2 * metadata.material_anchor] = 1.0
    q_zero = np.zeros(n_dofs, dtype=np.float64)
    digest_fields = (
        ("coordinates", metadata.union_coordinates),
        ("union_volumes", metadata.union_volumes),
        ("directed_edges", ba["edges"]),
        ("source", ba["source"]),
        ("target", ba["target"]),
        ("edge_distance", ba["distance"]),
        ("edge_omega", ba["omega"]),
        ("family_mass", ba["family_mass"]),
        ("eta", ba["eta"]),
        ("support_indices", support_indices),
        ("support_mask", support_mask),
        ("support_sizes", np.asarray([len(s) for s in ba["supports"]], dtype=np.int64)),
        ("beta", beta),
        ("xi", xi),
        ("shape_inverse", inverse),
        ("shape_rank", ba["shape_rank"]),
        ("shape_condition", ba["shape_condition"]),
        ("target_beta", beta[:, 0]),
        ("gradient_coefficient", gradient_coefficient),
        ("youngs_modulus", metadata.youngs_modulus),
        ("poisson_ratio", metadata.poisson_ratio),
        ("lame_lambda", metadata.youngs_modulus * metadata.poisson_ratio / ((1.0 + metadata.poisson_ratio) * (1.0 - 2.0 * metadata.poisson_ratio))),
        ("shear_modulus", metadata.youngs_modulus / (2.0 * (1.0 + metadata.poisson_ratio))),
        ("k_res", metadata.k_res),
        ("rank_rtol", metadata.rank_rtol),
        ("delta", metadata.delta),
        ("h_g", metadata.h_g),
        ("material_count", metadata.n_material),
        ("material_prefix_map", np.arange(metadata.n_material, dtype=np.int64)),
        ("material_damage_volumes", metadata.material_volumes),
        ("union_material_phase_volumes", metadata.union_material_phase_volumes),
        ("grip_phase_volumes", metadata.union_grip_phase_volumes),
        ("constrained_dofs", metadata.constrained_dofs),
        ("free_dofs", metadata.free_dofs),
        ("q_zero", q_zero),
        ("load_direction", top_selector),
        ("outer_top_selector", top_selector),
        ("outer_bottom_selector", bottom_selector),
        ("anchor_selector", anchor_selector),
        ("outer_bottom_mask", metadata.outer_bottom_mask),
        ("outer_top_mask", metadata.outer_top_mask),
        ("anchor_particle", metadata.material_anchor),
        ("top_line_indices", metadata.top_line_indices),
        ("top_line_weights", metadata.top_line_weights),
        ("top_line_length", metadata.top_line_length),
        ("bottom_line_indices", metadata.bottom_line_indices),
        ("bottom_line_weights", metadata.bottom_line_weights),
        ("bottom_line_length", metadata.bottom_line_length),
    )
    return _digest_primitives(digest_fields)


def _archive_static_digest(
    metadata: A2ContinuationMetadata,
    ba: Mapping[str, Any],
    archive: Mapping[str, np.ndarray],
) -> str:
    """Validate serialized static supports and hash the serialized primitives."""

    pairs = (
        ("static_edge_distance", ba["distance"]),
        ("static_edge_omega", ba["omega"]),
        ("static_family_mass", ba["family_mass"]),
        ("static_eta", ba["eta"]),
        ("static_support_indices", np.asarray(_replay_static_supports(ba)[0])),
        ("static_support_mask", np.asarray(_replay_static_supports(ba)[1])),
        ("static_support_sizes", np.asarray([len(s) for s in ba["supports"]], dtype=np.int64)),
        ("static_beta", np.asarray(_replay_static_supports(ba)[2])),
        ("static_xi", np.asarray(_replay_static_supports(ba)[3])),
        ("static_shape_inverse", np.asarray(ba["inverse"])),
        ("static_shape_rank", ba["shape_rank"]),
        ("static_shape_condition", ba["shape_condition"]),
        ("static_target_beta", np.asarray(_replay_static_supports(ba)[2])[:, 0]),
        ("static_gradient_coefficient", _replay_static_gradient(ba)),
    )
    for name, expected in pairs:
        actual = archive[name]
        if np.asarray(expected).dtype.kind in "ib" or np.asarray(actual).dtype.kind in "ib":
            if not np.array_equal(actual, expected):
                raise ValueError(f"A2 archive {name} does not match replayed static algebra")
        else:
            _array_close(actual, expected, f"A2 archive {name}", 2.0e-12)
    supports, support_mask, beta, xi = _replay_static_supports(ba)
    inverse = archive["static_shape_inverse"]
    n_dofs = metadata.n_dofs
    top_selector = np.zeros(n_dofs, dtype=np.float64)
    top_selector[2 * np.flatnonzero(metadata.outer_top_mask) + 1] = 1.0
    bottom_selector = np.zeros(n_dofs, dtype=np.float64)
    bottom_selector[2 * np.flatnonzero(metadata.outer_bottom_mask) + 1] = 1.0
    anchor_selector = np.zeros(n_dofs, dtype=np.float64)
    anchor_selector[2 * metadata.material_anchor] = 1.0
    static = SimpleNamespace(
        coordinates=metadata.union_coordinates,
        union_volumes=metadata.union_volumes,
        directed_edges=ba["edges"],
        source=ba["source"],
        target=ba["target"],
        edge_distance=archive["static_edge_distance"],
        edge_omega=archive["static_edge_omega"],
        family_mass=archive["static_family_mass"],
        eta=archive["static_eta"],
        support_indices=archive["static_support_indices"],
        support_mask=archive["static_support_mask"],
        support_sizes=archive["static_support_sizes"],
        beta=archive["static_beta"],
        xi=archive["static_xi"],
        shape_inverse=inverse,
        shape_rank=archive["static_shape_rank"],
        shape_condition=archive["static_shape_condition"],
        target_beta=archive["static_target_beta"],
        gradient_coefficient=archive["static_gradient_coefficient"],
        youngs_modulus=metadata.youngs_modulus,
        poisson_ratio=metadata.poisson_ratio,
        lame_lambda=metadata.youngs_modulus * metadata.poisson_ratio / ((1.0 + metadata.poisson_ratio) * (1.0 - 2.0 * metadata.poisson_ratio)),
        shear_modulus=metadata.youngs_modulus / (2.0 * (1.0 + metadata.poisson_ratio)),
        k_res=metadata.k_res,
        rank_rtol=metadata.rank_rtol,
        delta=metadata.delta,
        h_g=metadata.h_g,
        material_count=metadata.n_material,
        material_prefix_map=archive["material_prefix_map"],
        material_damage_volumes=metadata.material_volumes,
        union_material_phase_volumes=metadata.union_material_phase_volumes,
        grip_phase_volumes=metadata.union_grip_phase_volumes,
        constrained_dofs=metadata.constrained_dofs,
        free_dofs=metadata.free_dofs,
        q_zero=np.zeros(n_dofs, dtype=np.float64),
        load_direction=top_selector,
        outer_top_selector=top_selector,
        outer_bottom_selector=bottom_selector,
        anchor_selector=anchor_selector,
        outer_bottom_mask=metadata.outer_bottom_mask,
        outer_top_mask=metadata.outer_top_mask,
        anchor_particle=metadata.material_anchor,
        top_line_indices=metadata.top_line_indices,
        top_line_weights=metadata.top_line_weights,
        top_line_length=metadata.top_line_length,
        bottom_line_indices=metadata.bottom_line_indices,
        bottom_line_weights=metadata.bottom_line_weights,
        bottom_line_length=metadata.bottom_line_length,
    )
    return _static_digest_from_object(static)


def _replay_static_supports(
    ba: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_edges = len(ba["supports"])
    max_support = max(len(support) for support in ba["supports"])
    support_indices = np.full((n_edges, max_support), -1, dtype=np.int64)
    support_mask = np.zeros((n_edges, max_support), dtype=bool)
    beta = np.zeros((n_edges, max_support), dtype=np.float64)
    xi = np.zeros((n_edges, max_support, 2), dtype=np.float64)
    for edge_index, support in enumerate(ba["supports"]):
        count = len(support)
        support_indices[edge_index, :count] = support
        support_mask[edge_index, :count] = True
        beta[edge_index, :count] = ba["beta"][edge_index]
        xi[edge_index, :count] = ba["xi"][edge_index]
    return support_indices, support_mask, beta, xi


def _replay_static_gradient(ba: Mapping[str, Any]) -> np.ndarray:
    _, _, beta, xi = _replay_static_supports(ba)
    return np.einsum("es,esb,ebc->esc", beta, xi, np.asarray(ba["inverse"]), optimize=True)


def _replay_response(
    q: np.ndarray,
    ba: Mapping[str, Any],
    center_weights: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
) -> dict[str, np.ndarray | float]:
    values = _f64(q, "replay displacement", (2 * len(ba["coordinates"]),)).reshape(-1, 2)
    if np.array_equal(q, np.zeros_like(q)):
        edge_count = len(ba["source"])
        zero_edges = np.zeros((edge_count, 2, 2), dtype=np.float64)
        return {
            "energy": 0.0,
            "residual": np.zeros_like(q),
            "edge_f": np.broadcast_to(_IDENTITY_2, (edge_count, 2, 2)).copy(),
            "edge_strain": zero_edges,
            "edge_stress": zero_edges.copy(),
            "edge_density": np.zeros(edge_count, dtype=np.float64),
        }
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    residual = np.zeros(q.shape, dtype=np.float64)
    edge_f: list[np.ndarray] = []
    edge_strain: list[np.ndarray] = []
    edge_stress: list[np.ndarray] = []
    edge_density: list[float] = []
    energy = 0.0
    for edge_index, i_raw in enumerate(ba["source"]):
        i = int(i_raw)
        support = ba["supports"][edge_index]
        beta = ba["beta"][edge_index]
        xi = ba["xi"][edge_index]
        inverse = ba["inverse"][edge_index]
        moment = np.zeros((2, 2), dtype=np.float64)
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            moment += beta[local] * np.outer(xi[local] + values[k] - values[i], xi[local])
        deformation_gradient = moment @ inverse
        increment = deformation_gradient - _IDENTITY_2
        strain = 0.5 * (increment + increment.T)
        trace = float(np.trace(strain))
        stress = lam * trace * _IDENTITY_2 + 2.0 * mu * strain
        density = 0.5 * lam * trace**2 + mu * float(np.sum(strain * strain))
        factor = center_weights[i] * ba["eta"][edge_index]
        energy += factor * density
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            endpoint = beta[local] * (stress @ inverse.T @ xi[local])
            residual[2 * k : 2 * k + 2] += factor * endpoint
            residual[2 * i : 2 * i + 2] -= factor * endpoint
        edge_f.append(deformation_gradient)
        edge_strain.append(strain)
        edge_stress.append(stress)
        edge_density.append(density)
    return {
        "energy": float(energy),
        "residual": residual,
        "edge_f": np.asarray(edge_f, dtype=np.float64),
        "edge_strain": np.asarray(edge_strain, dtype=np.float64),
        "edge_stress": np.asarray(edge_stress, dtype=np.float64),
        "edge_density": np.asarray(edge_density, dtype=np.float64),
    }


def _replay_tangent_action(
    direction: np.ndarray,
    ba: Mapping[str, Any],
    center_weights: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
) -> np.ndarray:
    values = _f64(direction, "replay tangent direction", (2 * len(ba["coordinates"]),)).reshape(-1, 2)
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    result = np.zeros(direction.shape, dtype=np.float64)
    for edge_index, i_raw in enumerate(ba["source"]):
        i = int(i_raw)
        support = ba["supports"][edge_index]
        beta = ba["beta"][edge_index]
        xi = ba["xi"][edge_index]
        inverse = ba["inverse"][edge_index]
        delta_f = np.zeros((2, 2), dtype=np.float64)
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            delta_f += np.outer(values[k] - values[i], beta[local] * xi[local]) @ inverse
        delta_strain = 0.5 * (delta_f + delta_f.T)
        trace = float(np.trace(delta_strain))
        delta_stress = lam * trace * _IDENTITY_2 + 2.0 * mu * delta_strain
        factor = center_weights[i] * ba["eta"][edge_index]
        for local, k_raw in enumerate(support):
            k = int(k_raw)
            endpoint = beta[local] * (delta_stress @ inverse.T @ xi[local])
            result[2 * k : 2 * k + 2] += factor * endpoint
            result[2 * i : 2 * i + 2] -= factor * endpoint
    return result


def _replay_center_weights(
    material_damage: np.ndarray,
    material_phase: np.ndarray,
    grip_phase: np.ndarray,
    k_res: float,
    n_material: int,
) -> np.ndarray:
    damage = _f64(material_damage, "replay material damage", (n_material,))
    g = np.pad((1.0 - damage) ** 2 + k_res, (0, len(material_phase) - n_material))
    weights = material_phase * g + grip_phase * (1.0 + k_res)
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("replay phase center weights are invalid")
    return weights


def _replay_opening(
    material_damage: np.ndarray,
    target: float,
    metadata: A2ContinuationMetadata,
    ba: Mapping[str, Any],
) -> dict[str, Any]:
    center = _replay_center_weights(
        material_damage,
        metadata.union_material_phase_volumes,
        metadata.union_grip_phase_volumes,
        metadata.k_res,
        metadata.n_material,
    )
    n_dofs = 2 * metadata.n_union
    top_dofs = 2 * np.flatnonzero(metadata.outer_top_mask) + 1
    bottom_dofs = 2 * np.flatnonzero(metadata.outer_bottom_mask) + 1
    anchor_dof = 2 * metadata.material_anchor
    constrained = np.asarray(
        np.sort(np.concatenate((bottom_dofs, top_dofs, np.asarray([anchor_dof], dtype=np.int64)))),
        dtype=np.int64,
    )
    free = np.setdiff1d(np.arange(n_dofs, dtype=np.int64), constrained)
    top_selector = np.zeros(n_dofs, dtype=np.float64)
    top_selector[top_dofs] = 1.0
    bottom_selector = np.zeros(n_dofs, dtype=np.float64)
    bottom_selector[bottom_dofs] = 1.0
    anchor_selector = np.zeros(n_dofs, dtype=np.float64)
    anchor_selector[anchor_dof] = 1.0

    def solve_outer(U: float) -> dict[str, Any]:
        prescribed = np.zeros(n_dofs, dtype=np.float64)
        prescribed[top_dofs] = U
        tangent = np.column_stack(
            [
                _replay_tangent_action(
                    np.eye(n_dofs, dtype=np.float64)[:, column],
                    ba,
                    center,
                    metadata.youngs_modulus,
                    metadata.poisson_ratio,
                )
                for column in range(n_dofs)
            ]
        )
        q = prescribed.copy()
        q[free] = np.linalg.solve(tangent[np.ix_(free, free)], -tangent[np.ix_(free, np.arange(n_dofs))] @ prescribed)
        response = _replay_response(q, ba, center, metadata.youngs_modulus, metadata.poisson_ratio)
        residual = np.asarray(response["residual"], dtype=np.float64)
        rhs = -tangent[np.ix_(free, np.arange(n_dofs))] @ prescribed
        free_residual = residual[free]
        free_residual_norm = float(np.linalg.norm(free_residual))
        free_rhs_norm = float(np.linalg.norm(rhs))
        force = residual.reshape(-1, 2).sum(axis=0)
        moment = float(np.sum(metadata.union_coordinates[:, 0] * residual.reshape(-1, 2)[:, 1] - metadata.union_coordinates[:, 1] * residual.reshape(-1, 2)[:, 0]))
        return {
            "q": q,
            "response": response,
            "residual": residual,
            "free_residual": residual[free],
            "free_residual_norm": free_residual_norm,
            "free_rhs_norm": free_rhs_norm,
            "free_residual_relative_norm": free_residual_norm / max(free_rhs_norm, 1.0e-300),
            "outer_reaction": float(top_selector @ residual),
            "bottom_reaction": float(bottom_selector @ residual),
            "anchor_reaction": float(anchor_selector @ residual),
            "force": force,
            "moment": moment,
            "tangent": tangent,
        }

    unit = solve_outer(1.0)
    top_opening = float(np.sum(metadata.top_line_weights * unit["q"].reshape(-1, 2)[metadata.top_line_indices, 1]) / metadata.top_line_length)
    bottom_opening = float(np.sum(metadata.bottom_line_weights * unit["q"].reshape(-1, 2)[metadata.bottom_line_indices, 1]) / metadata.bottom_line_length)
    slope = top_opening - bottom_opening
    target_value = _scalar(target, "replay target")
    if slope <= 0.0:
        raise ValueError("replay opening slope is not positive")
    U = target_value / slope
    scaled = solve_outer(U)
    q_scaled = scaled["q"]
    q_values = q_scaled.reshape(-1, 2)
    achieved = float(np.sum(metadata.top_line_weights * q_values[metadata.top_line_indices, 1]) / metadata.top_line_length - np.sum(metadata.bottom_line_weights * q_values[metadata.bottom_line_indices, 1]) / metadata.bottom_line_length)
    epsilon = metadata.config.work_epsilon
    plus = solve_outer(U + epsilon)
    minus = solve_outer(U - epsilon)
    R_ext_fd = (float(plus["response"]["energy"]) - float(minus["response"]["energy"])) / (2.0 * epsilon)
    D_plus = float(np.sum(metadata.top_line_weights * plus["q"].reshape(-1, 2)[metadata.top_line_indices, 1]) / metadata.top_line_length - np.sum(metadata.bottom_line_weights * plus["q"].reshape(-1, 2)[metadata.bottom_line_indices, 1]) / metadata.bottom_line_length)
    D_minus = float(np.sum(metadata.top_line_weights * minus["q"].reshape(-1, 2)[metadata.top_line_indices, 1]) / metadata.top_line_length - np.sum(metadata.bottom_line_weights * minus["q"].reshape(-1, 2)[metadata.bottom_line_indices, 1]) / metadata.bottom_line_length)
    R_eff_fd = (float(plus["response"]["energy"]) - float(minus["response"]["energy"])) / (D_plus - D_minus)
    return {
        "unit": unit,
        "scaled": scaled,
        "slope": slope,
        "U": U,
        "achieved": achieved,
        "R_eff": scaled["outer_reaction"] / slope,
        "R_ext_fd": R_ext_fd,
        "R_eff_fd": R_eff_fd,
        "R_ext_error": _relative(R_ext_fd, scaled["outer_reaction"]),
        "R_eff_error": _relative(R_eff_fd, scaled["outer_reaction"] / slope),
    }


def _replay_mechanics_gate_flags(
    response: Mapping[str, Any],
    target: float,
    metadata: A2ContinuationMetadata,
    geometry_length: float,
) -> dict[str, bool]:
    """Reconstruct one mechanics gate set from independent replay primitives."""

    scaled = response["scaled"]
    opening_pass = bool(
        abs(float(response["achieved"]) - _scalar(target, "replay gate target"))
        <= metadata.config.opening_atol
    )
    slope_pass = bool(
        np.isfinite(float(response["slope"]))
        and float(response["slope"]) >= metadata.config.opening_slope_min
    )
    work_pass = bool(
        float(response["R_ext_error"]) <= metadata.config.work_relative_limit
        and float(response["R_eff_error"]) <= metadata.config.work_relative_limit
    )
    conservation = _conservation_metrics_from_values(
        metadata.config,
        scaled["outer_reaction"],
        scaled["bottom_reaction"],
        scaled["anchor_reaction"],
        scaled["force"],
        scaled["moment"],
        geometry_length,
    )
    cg_limit = max(
        metadata.config.cg_atol,
        metadata.config.cg_rtol * float(scaled["free_rhs_norm"]),
    )
    cg_pass = bool(
        np.isfinite(float(scaled["free_residual_norm"]))
        and float(scaled["free_residual_norm"]) < cg_limit
    )
    force_pass = bool(conservation["force_pass"])
    moment_pass = bool(conservation["moment_pass"])
    return {
        "opening": opening_pass,
        "slope": slope_pass,
        "work": work_pass,
        "force": force_pass,
        "moment": moment_pass,
        "cg": cg_pass,
        "mechanics": bool(
            opening_pass
            and slope_pass
            and work_pass
            and force_pass
            and moment_pass
            and cg_pass
        ),
    }


def _replay_mechanics_target(diag: Any, phase: str, attempt_target: float) -> float:
    """Return the target used by one archived fixed-damage mechanics solve."""

    if phase not in ("before", "after"):
        raise ValueError(f"unknown replay mechanics phase {phase!r}")
    value = getattr(diag, f"mechanics_target_opening_{phase}", None)
    if value is None:
        raise ValueError(f"replay mechanics target opening {phase} is unavailable")
    target = _scalar(value, f"replay mechanics target opening {phase}")
    if target < 0.0:
        raise ValueError(f"replay mechanics target opening {phase} must be non-negative")
    return target


def _replay_normalized_seed(
    q_previous: np.ndarray,
    outer_previous: float,
    metadata: A2ContinuationMetadata,
) -> np.ndarray:
    """Rebuild the normalized unit seed without using the production helper."""

    q = _f64(q_previous, "replay seed source displacement", (metadata.n_dofs,))
    U = _scalar(outer_previous, "replay seed source outer displacement")
    prescribed_unit = np.zeros(metadata.n_dofs, dtype=np.float64)
    prescribed_unit[2 * np.flatnonzero(metadata.outer_top_mask) + 1] = 1.0
    constrained = metadata.constrained_dofs
    prescribed = prescribed_unit * U
    if not np.array_equal(q[constrained], prescribed[constrained]):
        raise ValueError("replay seed source does not satisfy exact constraints")
    if U == 0.0:
        if not np.array_equal(q, np.zeros_like(q)):
            raise ValueError("replay zero seed source is not exact zero")
        return prescribed_unit
    seed = prescribed_unit.copy()
    free = metadata.free_dofs
    seed[free] = q[free] / U
    if not np.all(np.isfinite(seed)):
        raise ValueError("replay normalized seed is non-finite")
    return seed


def _validate_replay_seed(
    seed: Any,
    source_q: np.ndarray,
    source_outer: float,
    source_state: str,
    source_step: int,
    source_sweep: int,
    source_damage_digest: str,
    target_damage_digest: str,
    metadata: A2ContinuationMetadata,
) -> None:
    if seed is None:
        raise ValueError("replay seed provenance is unavailable")
    expected_seed = _replay_normalized_seed(source_q, source_outer, metadata)
    expected_seed_digest = _digest_primitives((("seed", expected_seed),))
    expected_unit_digest = _unit_system_digest(metadata.static_digest, target_damage_digest)
    # The residual norm is reconstructed by the complete-sweep validator with
    # the target view.  This helper validates all provenance fields that do not
    # require the repeated response evaluation.
    expected = {
        "source_state": source_state,
        "source_step": int(source_step),
        "source_sweep": int(source_sweep),
        "source_damage_digest": source_damage_digest,
        "target_damage_digest": target_damage_digest,
        "unit_system_digest": expected_unit_digest,
        "seed_digest": expected_seed_digest,
        "kind": "normalized_previous_solution",
        "early_exit_eligible": False,
        "early_exit_used": False,
    }
    for name, value in expected.items():
        actual = getattr(seed, name, None)
        if actual != value:
            raise ValueError(f"replay seed field {name} does not match its source identity")
    for name in ("initial_residual_norm", "final_residual_norm"):
        value = getattr(seed, name, None)
        if value is None or not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"replay seed field {name} is invalid")


def _replay_amor(
    q: np.ndarray,
    previous: np.ndarray,
    metadata: A2ContinuationMetadata,
    ba: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if np.array_equal(q, np.zeros_like(q)):
        return (
            np.zeros(metadata.n_material, dtype=np.float64),
            np.zeros(metadata.n_material, dtype=np.float64),
        )
    response = _replay_response(
        q,
        ba,
        np.ones(metadata.n_union, dtype=np.float64),
        metadata.youngs_modulus,
        metadata.poisson_ratio,
    )
    strain = np.asarray(response["edge_strain"], dtype=np.float64)
    lam = metadata.youngs_modulus * metadata.poisson_ratio / ((1.0 + metadata.poisson_ratio) * (1.0 - 2.0 * metadata.poisson_ratio))
    mu = metadata.youngs_modulus / (2.0 * (1.0 + metadata.poisson_ratio))
    edge_positive = np.empty(len(strain), dtype=np.float64)
    for index, value in enumerate(strain):
        embedded = np.zeros((3, 3), dtype=np.float64)
        embedded[:2, :2] = value
        trace = float(np.trace(embedded))
        deviatoric = embedded - trace / 3.0 * _IDENTITY_3
        edge_positive[index] = 0.5 * (lam + 2.0 * mu / 3.0) * max(trace, 0.0) ** 2 + mu * float(np.sum(deviatoric * deviatoric))
    source = ba["source"]
    eta = ba["eta"]
    driver = np.zeros(metadata.n_material, dtype=np.float64)
    material_edges = source < metadata.n_material
    if not np.any(material_edges):
        raise ValueError("replay Amor found no material source centers")
    source_material = source[material_edges]
    eta_material = eta[material_edges]
    eta_sum = np.bincount(source_material, weights=eta_material, minlength=metadata.n_material)
    if np.max(np.abs(eta_sum - 1.0)) > _REPLAY_BOUND:
        raise ValueError("replay Amor eta does not preserve material-source normalization")
    np.add.at(driver, source_material, eta_material * edge_positive[material_edges])
    return driver, np.maximum(previous, driver)


def _replay_damage(
    H_used: np.ndarray,
    metadata: A2ContinuationMetadata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = metadata.n_material
    matrix = np.zeros((n, n), dtype=np.float64)
    row = metadata.material_edges[:, 0]
    col = metadata.material_edges[:, 1]
    distance = np.linalg.norm(metadata.material_coordinates[col] - metadata.material_coordinates[row], axis=1)
    weight = 120.0 / (np.pi * metadata.delta**4) * (1.0 - distance / metadata.delta) ** 2
    coefficient = weight * metadata.material_volumes[col]
    for i, j, value in zip(row, col, coefficient):
        matrix[int(i), int(i)] += value
        matrix[int(i), int(j)] -= value
    H = _f64(H_used, "replay H_used", (n,))
    strong = np.diag(metadata.Gc / metadata.l0 + 2.0 * H) + metadata.Gc * metadata.l0 * matrix
    mass = np.diag(metadata.material_volumes)
    system = mass @ strong
    rhs = 2.0 * metadata.material_volumes * H
    damage = np.linalg.solve(system, rhs)
    rho = strong @ damage - 2.0 * H
    volume_rho = metadata.material_volumes * rho
    terms = (metadata.Gc / metadata.l0 * damage, metadata.Gc * metadata.l0 * (matrix @ damage), 2.0 * H * damage, 2.0 * H)
    scale = max((_max_abs(term) for term in terms), default=0.0)
    return damage, rho, volume_rho, matrix, scale


def _compare_replay_scalar(actual: Any, expected: Any, name: str, limit: float = 1.0e-8) -> None:
    left = float(actual)
    right = float(expected)
    if not np.isfinite(left) or not np.isfinite(right) or abs(left - right) > limit * max(abs(left), abs(right), 1.0):
        raise ValueError(f"{name} does not match the authoritative primitive replay")


def _replay_complete_diagnostics(
    diag: Any,
    committed_damage: np.ndarray,
    committed_history: np.ndarray,
    metadata: A2ContinuationMetadata,
    ba: Mapping[str, Any],
    seed_before_source_q: np.ndarray,
    seed_before_source_outer: float,
    seed_before_source_state: str,
    seed_before_source_damage_digest: str,
    source_step: int,
) -> None:
    """Replay one complete sweep from only primitive fields."""

    if not bool(getattr(diag, "complete_sweep")):
        raise ValueError("complete replay requires complete_sweep=True")
    d_iter = _f64(getattr(diag, "d_iter"), "replay d_iter", (metadata.n_material,))
    committed_d = _f64(committed_damage, "replay committed damage", d_iter.shape)
    Hc = _f64(committed_history, "replay committed history", (metadata.n_material,))
    target = _scalar(getattr(diag, "target_opening"), "replay target_opening")
    mechanics_target_before = _replay_mechanics_target(diag, "before", target)
    mechanics_target_after = _replay_mechanics_target(diag, "after", target)
    before = _replay_opening(d_iter, mechanics_target_before, metadata, ba)
    psi_before, H_used_expected = _replay_amor(before["scaled"]["q"], Hc, metadata, ba)
    d_next_expected, rho_used_expected, rho_used_volume_expected, matrix, residual_scale = _replay_damage(H_used_expected, metadata)
    after = _replay_opening(d_next_expected, mechanics_target_after, metadata, ba)
    psi_after, H_next_expected = _replay_amor(after["scaled"]["q"], Hc, metadata, ba)
    rho_next_expected = (
        metadata.Gc / metadata.l0 * d_next_expected
        + metadata.Gc * metadata.l0 * (matrix @ d_next_expected)
        + 2.0 * H_next_expected * d_next_expected
        - 2.0 * H_next_expected
    )
    rho_used_component, rho_used_volume_component = _compare_residual_reconstruction(
        getattr(diag, "rho_used"),
        getattr(diag, "rho_used_volume"),
        getattr(diag, "d_next"),
        getattr(diag, "H_used"),
        matrix,
        metadata.material_volumes,
        metadata.Gc,
        metadata.l0,
        "A2 replay rho_used",
    )
    rho_next_component, rho_next_volume_component = _compare_residual_reconstruction(
        getattr(diag, "rho_next"),
        getattr(diag, "rho_next_volume"),
        getattr(diag, "d_next"),
        getattr(diag, "H_next"),
        matrix,
        metadata.material_volumes,
        metadata.Gc,
        metadata.l0,
        "A2 replay rho_next",
    )
    _validate_replay_seed(
        getattr(diag, "seed_before", None),
        seed_before_source_q,
        seed_before_source_outer,
        seed_before_source_state,
        source_step,
        int(getattr(diag, "sweep_index")),
        seed_before_source_damage_digest,
        _damage_digest(d_iter),
        metadata,
    )
    _validate_replay_seed(
        getattr(diag, "seed_after", None),
        getattr(diag, "q_before"),
        getattr(diag, "U_before"),
        "mechanics_before",
        source_step,
        int(getattr(diag, "sweep_index")),
        _damage_digest(d_iter),
        _damage_digest(getattr(diag, "d_next")),
        metadata,
    )
    for seed_name, seed_value, response, damage_for_seed in (
        (
            "seed_before",
            _replay_normalized_seed(seed_before_source_q, seed_before_source_outer, metadata),
            before["unit"],
            d_iter,
        ),
        (
            "seed_after",
            _replay_normalized_seed(getattr(diag, "q_before"), getattr(diag, "U_before"), metadata),
            after["unit"],
            d_next_expected,
        ),
    ):
        center_for_seed = _replay_center_weights(
            damage_for_seed,
            metadata.union_material_phase_volumes,
            metadata.union_grip_phase_volumes,
            metadata.k_res,
            metadata.n_material,
        )
        seed_response = _replay_response(
            seed_value,
            ba,
            center_for_seed,
            metadata.youngs_modulus,
            metadata.poisson_ratio,
        )
        seed_record = getattr(diag, seed_name)
        _compare_replay_scalar(
            seed_record.initial_residual_norm,
            np.linalg.norm(np.asarray(seed_response["residual"])[metadata.free_dofs]),
            f"A2 replay {seed_name}.initial_residual_norm",
        )
        _compare_replay_scalar(
            seed_record.final_residual_norm,
            response["free_residual_norm"],
            f"A2 replay {seed_name}.final_residual_norm",
        )
    for name, expected in (
        ("q_before", before["scaled"]["q"]), ("q_after", after["scaled"]["q"]),
        ("Psi_before", psi_before), ("Psi_after", psi_after), ("H_used", H_used_expected),
        ("H_next", H_next_expected), ("rho_used", rho_used_component), ("rho_next", rho_next_component),
        ("rho_used_volume", rho_used_volume_component), ("rho_next_volume", rho_next_volume_component),
        ("d_next", d_next_expected),
    ):
        actual = getattr(diag, name, None)
        if actual is None:
            raise ValueError(f"complete replay field {name} is unavailable")
        _array_close(actual, expected, f"A2 replay {name}", 2.0e-8 if name.startswith("q_") else 2.0e-9)
    for name, expected in (
        ("damage_before", committed_d), ("d_iter", d_iter),
        ("force_before", before["scaled"]["force"]),
        ("force_after", after["scaled"]["force"]),
    ):
        actual = getattr(diag, name, None)
        if actual is None:
            raise ValueError(f"complete replay field {name} is unavailable")
        _array_close(actual, expected, f"A2 replay {name}", 2.0e-9)
    for name, expected in (
        ("U_before", before["U"]), ("U_after", after["U"]),
        ("mechanics_target_opening_before", mechanics_target_before),
        ("mechanics_target_opening_after", mechanics_target_after),
        ("slope_before", before["slope"]), ("slope_after", after["slope"]),
        ("achieved_opening_before", before["achieved"]), ("achieved_opening_after", after["achieved"]),
        ("R_ext_before", before["scaled"]["outer_reaction"]), ("R_ext_after", after["scaled"]["outer_reaction"]),
        ("R_eff_before", before["R_eff"]), ("R_eff_after", after["R_eff"]),
        ("work_R_ext_before_error", before["R_ext_error"]), ("work_R_eff_before_error", before["R_eff_error"]),
        ("work_R_ext_after_error", after["R_ext_error"]), ("work_R_eff_after_error", after["R_eff_error"]),
        ("rho_used_inf", _max_abs(rho_used_component)), ("rho_next_inf", _max_abs(rho_next_component)),
    ):
        _compare_replay_scalar(getattr(diag, name), expected, f"A2 replay {name}")
    stored_q_before = getattr(diag, "q_before")
    stored_q_after = getattr(diag, "q_after")
    stored_d_next = getattr(diag, "d_next")
    stored_H_used = getattr(diag, "H_used")
    stored_H_next = getattr(diag, "H_next")
    e_um = _max_abs(stored_q_before[: 2 * metadata.n_material] - stored_q_after[: 2 * metadata.n_material])
    s_um = max(_max_abs(stored_q_before[: 2 * metadata.n_material]), _max_abs(stored_q_after[: 2 * metadata.n_material]), target)
    e_uu = _max_abs(stored_q_before - stored_q_after)
    s_uu = max(_max_abs(stored_q_before), _max_abs(stored_q_after), abs(getattr(diag, "U_before")), abs(getattr(diag, "U_after")), target)
    e_d = _max_abs(stored_d_next - d_iter)
    s_d = max(_max_abs(d_iter), _max_abs(stored_d_next))
    e_h = _max_abs(stored_H_next - stored_H_used)
    s_h = max(_max_abs(Hc), _max_abs(stored_H_used), _max_abs(stored_H_next))
    e_r = abs(getattr(diag, "R_eff_after") - getattr(diag, "R_eff_before"))
    s_r = max(abs(getattr(diag, "R_eff_before")), abs(getattr(diag, "R_eff_after")))
    geometry_length = max(_max_abs(metadata.union_coordinates), _max_abs(np.ptp(metadata.union_coordinates, axis=0)), 1.0)
    conservation_before = _conservation_metrics_from_values(
        metadata.config,
        before["scaled"]["outer_reaction"],
        before["scaled"]["bottom_reaction"],
        before["scaled"]["anchor_reaction"],
        before["scaled"]["force"],
        before["scaled"]["moment"],
        geometry_length,
    )
    conservation_after = _conservation_metrics_from_values(
        metadata.config,
        after["scaled"]["outer_reaction"],
        after["scaled"]["bottom_reaction"],
        after["scaled"]["anchor_reaction"],
        after["scaled"]["force"],
        after["scaled"]["moment"],
        geometry_length,
    )
    force_error = max(float(conservation_before["force_error"]), float(conservation_after["force_error"]))
    force_scale = max(float(conservation_before["force_scale"]), float(conservation_after["force_scale"]))
    moment_error = max(float(conservation_before["moment_error"]), float(conservation_after["moment_error"]))
    moment_scale = max(float(conservation_before["moment_scale"]), float(conservation_after["moment_scale"]))
    mechanics_before_flags = _replay_mechanics_gate_flags(
        before, target, metadata, geometry_length
    )
    mechanics_after_flags = _replay_mechanics_gate_flags(
        after, target, metadata, geometry_length
    )
    residual_scale = max(
        _max_abs(metadata.Gc / metadata.l0 * stored_d_next),
        _max_abs(metadata.Gc * metadata.l0 * (matrix @ stored_d_next)),
        _max_abs(2.0 * stored_H_used * stored_d_next),
        _max_abs(2.0 * stored_H_used),
        _max_abs(2.0 * stored_H_next * stored_d_next),
        _max_abs(2.0 * stored_H_next),
    )
    passes = {
        "opening_before": mechanics_before_flags["opening"],
        "opening_after": mechanics_after_flags["opening"],
        "slope_before": mechanics_before_flags["slope"],
        "slope_after": mechanics_after_flags["slope"],
        "work_before": mechanics_before_flags["work"],
        "work_after": mechanics_after_flags["work"],
        "cg_before": mechanics_before_flags["cg"],
        "cg_after": mechanics_after_flags["cg"],
        "displacement_material": e_um <= metadata.config.displacement_material.limit(s_um),
        "displacement_union": e_uu <= metadata.config.displacement_union.limit(s_uu),
        "damage": e_d <= metadata.config.damage.limit(s_d),
        "history": e_h <= metadata.config.history.limit(s_h),
        "reaction_effective": e_r <= metadata.config.reaction_effective.limit(s_r),
        "damage_residual": _max_abs(getattr(diag, "rho_next")) <= metadata.config.damage_residual.limit(residual_scale),
        "damage_range": float(np.min(stored_d_next)) >= -metadata.config.damage_range_atol and float(np.max(stored_d_next)) <= 1.0 + metadata.config.damage_range_atol,
        "healing": float(np.min(stored_d_next - committed_damage)) >= -metadata.config.healing_atol,
        "history_monotonic": float(np.min(stored_H_next - Hc)) >= 0.0,
        "force_before": bool(conservation_before["force_pass"]),
        "force_after": bool(conservation_after["force_pass"]),
        "force": bool(conservation_before["force_pass"] and conservation_after["force_pass"]),
        "moment_before": bool(conservation_before["moment_pass"]),
        "moment_after": bool(conservation_after["moment_pass"]),
        "moment": bool(conservation_before["moment_pass"] and conservation_after["moment_pass"]),
        "work": mechanics_before_flags["work"] and mechanics_after_flags["work"],
        "mechanics": mechanics_before_flags["mechanics"] and mechanics_after_flags["mechanics"],
    }
    expected_passed = all(passes.values())
    if bool(getattr(diag, "passed")) != expected_passed:
        raise ValueError("A2 replay passed flag does not match primitive gates")
    expected_reason = "" if expected_passed else ", ".join(name for name, value in passes.items() if not value)
    if getattr(diag, "failure_stage", None) != ("" if expected_passed else "gates"):
        raise ValueError("A2 replay failure stage does not match primitive gates")
    if getattr(diag, "failure_reason", None) != expected_reason:
        raise ValueError("A2 replay failure reason does not match primitive gates")
    for name, expected in (
        ("opening_atol", metadata.config.opening_atol),
        ("e_opening_before", abs(before["achieved"] - target)),
        ("e_opening_after", abs(after["achieved"] - target)),
        ("e_u_material", e_um), ("u_material_scale", s_um), ("e_u_union", e_uu), ("u_union_scale", s_uu),
        ("u_material_limit", metadata.config.displacement_material.limit(s_um)),
        ("u_union_limit", metadata.config.displacement_union.limit(s_uu)),
        ("e_damage", e_d), ("damage_scale", s_d), ("e_history", e_h), ("history_scale", s_h),
        ("damage_limit", metadata.config.damage.limit(s_d)),
        ("history_limit", metadata.config.history.limit(s_h)),
        ("e_reaction_effective", e_r), ("reaction_effective_scale", s_r),
        ("reaction_effective_limit", metadata.config.reaction_effective.limit(s_r)),
         ("damage_residual_scale", residual_scale), ("damage_min", np.min(d_next_expected)),
         ("damage_max", np.max(d_next_expected)),
         ("healing_min", np.min(d_next_expected - committed_damage)),
         ("history_min_gap", np.min(H_next_expected - Hc)), ("force_error", force_error),
         ("force_scale", force_scale), ("moment_error", moment_error), ("moment_scale", moment_scale),
         ("force_before_error", conservation_before["force_error"]),
         ("force_before_scale", conservation_before["force_scale"]),
         ("force_before_limit", conservation_before["force_limit"]),
         ("force_after_error", conservation_after["force_error"]),
         ("force_after_scale", conservation_after["force_scale"]),
         ("force_after_limit", conservation_after["force_limit"]),
         ("moment_before_error", conservation_before["moment_error"]),
         ("moment_before_scale", conservation_before["moment_scale"]),
         ("moment_before_limit", conservation_before["moment_limit"]),
         ("moment_after_error", conservation_after["moment_error"]),
         ("moment_after_scale", conservation_after["moment_scale"]),
         ("moment_after_limit", conservation_after["moment_limit"]),
        ("damage_residual_limit", metadata.config.damage_residual.limit(residual_scale)),
        ("force_limit", metadata.config.force.limit(force_scale)),
         ("moment_limit", metadata.config.moment.limit(moment_scale)),
        ("work_limit", metadata.config.work_relative_limit),
        ("mechanics_before_free_residual_norm", before["scaled"]["free_residual_norm"]),
        ("mechanics_after_free_residual_norm", after["scaled"]["free_residual_norm"]),
        ("mechanics_before_free_residual_relative_norm", before["scaled"]["free_residual_relative_norm"]),
        ("mechanics_after_free_residual_relative_norm", after["scaled"]["free_residual_relative_norm"]),
    ):
        _compare_replay_scalar(getattr(diag, name), expected, f"A2 replay {name}")
    for name, expected in (
        ("u_material_pass", passes["displacement_material"]), ("u_union_pass", passes["displacement_union"]),
        ("damage_pass", passes["damage"]), ("history_pass", passes["history"]),
        ("reaction_effective_pass", passes["reaction_effective"]), ("damage_residual_pass", passes["damage_residual"]),
        ("damage_range_pass", passes["damage_range"]), ("healing_pass", passes["healing"]),
         ("history_monotonic_pass", passes["history_monotonic"]),
         ("force_before_pass", passes["force_before"]), ("force_after_pass", passes["force_after"]),
         ("force_pass", passes["force"]),
          ("moment_before_pass", passes["moment_before"]), ("moment_after_pass", passes["moment_after"]),
          ("moment_pass", passes["moment"]), ("work_pass", passes["work"]),
          ("opening_before_pass", passes["opening_before"]),
          ("opening_after_pass", passes["opening_after"]),
          ("slope_before_pass", passes["slope_before"]),
          ("slope_after_pass", passes["slope_after"]),
          ("work_before_pass", passes["work_before"]),
          ("work_after_pass", passes["work_after"]),
          ("cg_before_pass", passes["cg_before"]),
          ("cg_after_pass", passes["cg_after"]),
          ("mechanics_pass", passes["mechanics"]),
    ):
        if bool(getattr(diag, name)) != bool(expected):
            raise ValueError(f"A2 replay {name} does not match primitive gates")


def _validate_incomplete_diagnostics(diag: Any) -> None:
    if bool(getattr(diag, "complete_sweep")):
        return
    if getattr(diag, "H_next", None) is not None or getattr(diag, "rho_next", None) is not None or getattr(diag, "rho_next_volume", None) is not None:
        raise ValueError("incomplete A2 diagnostics fabricated later history/residual fields")
    available = set(getattr(diag, "available_fields", ()))
    known_available = {
        "damage_before",
        "d_iter",
        "q_before",
        "Psi_before",
        "H_used",
        "d_next",
        "rho_used",
        "rho_used_volume",
        "q_after",
        "Psi_after",
        "force_before",
        "force_after",
        "mechanics_target_opening_before",
        "mechanics_target_opening_after",
        "seed_before",
        "seed_after",
        "U_before",
        "U_after",
        "slope_before",
        "slope_after",
        "achieved_opening_before",
        "achieved_opening_after",
        "R_ext_before",
        "R_ext_after",
        "R_eff_before",
        "R_eff_after",
        "moment_before",
        "moment_after",
        "e_opening_before",
        "e_opening_after",
        "work_R_ext_before_error",
        "work_R_eff_before_error",
        "work_R_ext_after_error",
        "work_R_eff_after_error",
        "mechanics_before_free_residual_norm",
        "mechanics_after_free_residual_norm",
        "mechanics_before_free_residual_relative_norm",
        "mechanics_after_free_residual_relative_norm",
        "force_before_error",
        "force_before_scale",
        "force_before_limit",
        "force_before_pass",
        "force_after_error",
        "force_after_scale",
        "force_after_limit",
        "force_after_pass",
        "moment_before_error",
        "moment_before_scale",
        "moment_before_limit",
        "moment_before_pass",
        "moment_after_error",
        "moment_after_scale",
        "moment_after_limit",
        "moment_after_pass",
    }
    unknown = available - known_available
    if unknown:
        raise ValueError(
            "incomplete A2 diagnostics availability contains unknown fields: "
            f"{sorted(unknown)}"
        )
    stage = getattr(diag, "failure_stage", None)
    if not isinstance(stage, str) or not stage:
        raise ValueError("incomplete A2 diagnostics failure stage is unavailable")
    if "q_before" not in available:
        expected_stages = {"mechanics_before"}
    elif "Psi_before" not in available:
        expected_stages = {"amor_before"}
    elif "H_used" not in available:
        expected_stages = {"mechanics_before_gate"}
    elif "d_next" not in available:
        expected_stages = {"damage"}
    elif "q_after" not in available:
        # Damage-range/healing rejection occurs before the after solve, while
        # a mechanics-after exception occurs after the same primitive prefix.
        expected_stages = {"damage_acceptance", "mechanics_after"}
    elif "Psi_after" not in available:
        expected_stages = {"amor_after"}
    else:
        expected_stages = {"mechanics_after_gate"}
    if stage not in expected_stages:
        raise ValueError(
            "incomplete A2 diagnostics failure stage is inconsistent with "
            f"available primitives: stage={stage!r}, expected={sorted(expected_stages)}"
        )
    for required_name in ("damage_before", "d_iter"):
        if required_name not in available or getattr(diag, required_name, None) is None:
            raise ValueError(
                f"incomplete A2 diagnostics availability omits required field {required_name}"
            )
    for name in (
        "q_before", "Psi_before", "H_used", "d_next", "rho_used", "rho_used_volume",
        "q_after", "Psi_after", "force_before", "force_after", "seed_before", "seed_after",
    ):
        value = getattr(diag, name, None)
        if (value is None) != (name not in available):
            raise ValueError(f"incomplete A2 diagnostics availability is inconsistent for {name}")
    for name in (
        "mechanics_target_opening_before", "mechanics_target_opening_after",
        "U_before", "U_after", "slope_before", "slope_after", "achieved_opening_before",
        "achieved_opening_after", "R_ext_before", "R_ext_after", "R_eff_before", "R_eff_after",
        "moment_before", "moment_after", "e_opening_before", "e_opening_after",
        "work_R_ext_before_error", "work_R_eff_before_error", "work_R_ext_after_error",
        "work_R_eff_after_error", "mechanics_before_free_residual_norm",
        "mechanics_after_free_residual_norm", "mechanics_before_free_residual_relative_norm",
        "mechanics_after_free_residual_relative_norm",
        "force_before_error", "force_before_scale", "force_before_limit",
        "force_after_error", "force_after_scale", "force_after_limit",
        "moment_before_error", "moment_before_scale", "moment_before_limit",
        "moment_after_error", "moment_after_scale", "moment_after_limit",
    ):
        value = getattr(diag, name, None)
        if name in available:
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"incomplete A2 available scalar {name} is invalid")
        elif value is not None:
            number = float(value)
            if name.endswith("_error") and not np.isinf(number):
                raise ValueError(f"incomplete A2 unavailable scalar {name} was fabricated")
            if name.endswith("_scale") and number != 0.0:
                raise ValueError(f"incomplete A2 unavailable scalar {name} was fabricated")
            if name.endswith("_limit") and (not np.isfinite(number) or number < 0.0):
                raise ValueError(f"incomplete A2 unavailable scalar {name} is invalid")
    for phase in ("before", "after"):
        mechanics_field = f"q_{phase}"
        target_field = f"mechanics_target_opening_{phase}"
        if mechanics_field in available and target_field not in available:
            raise ValueError(
                "incomplete A2 diagnostics availability omits mechanics target "
                f"provenance for {phase}"
            )
    for name in (
        "force_before_pass", "force_after_pass", "moment_before_pass", "moment_after_pass",
    ):
        value = bool(getattr(diag, name))
        if name in available:
            continue
        if value:
            raise ValueError(f"incomplete A2 unavailable gate {name} was fabricated")
def _replay_incomplete_diagnostics(
    diag: Any,
    committed_damage: np.ndarray,
    committed_history: np.ndarray,
    metadata: A2ContinuationMetadata,
    ba: Mapping[str, Any],
    seed_source_q: np.ndarray,
    seed_source_outer: float,
    seed_source_state: str,
    seed_source_damage_digest: str,
    source_step: int,
) -> None:
    """Validate every primitive available before an incomplete failure."""

    _validate_incomplete_diagnostics(diag)
    allowed_stages = {
        "mechanics_before", "amor_before", "mechanics_before_gate", "damage",
        "damage_acceptance", "mechanics_after", "amor_after", "mechanics_after_gate",
    }
    if getattr(diag, "failure_stage", None) not in allowed_stages:
        raise ValueError("incomplete A2 failure stage is unknown")
    d_iter = _f64(diag.d_iter, "replay incomplete d_iter", (metadata.n_material,))
    Hc = _f64(committed_history, "replay incomplete committed history", d_iter.shape)
    target = _scalar(diag.target_opening, "replay incomplete target")
    mechanics_target_before = (
        _replay_mechanics_target(diag, "before", target)
        if diag.q_before is not None
        else None
    )
    mechanics_target_after = (
        _replay_mechanics_target(diag, "after", target)
        if diag.q_after is not None
        else None
    )
    geometry_length = max(_max_abs(metadata.union_coordinates), _max_abs(np.ptp(metadata.union_coordinates, axis=0)), 1.0)
    if diag.q_before is not None:
        if mechanics_target_before is None:
            raise ValueError("incomplete A2 q_before lacks mechanics target provenance")
        before = _replay_opening(d_iter, mechanics_target_before, metadata, ba)
    else:
        before = None
    before_flags: dict[str, bool] | None = None
    after_flags: dict[str, bool] | None = None
    if before is not None:
        before_flags = _replay_mechanics_gate_flags(
            before, target, metadata, geometry_length
        )
        conservation_before = _conservation_metrics_from_values(
            metadata.config,
            before["scaled"]["outer_reaction"],
            before["scaled"]["bottom_reaction"],
            before["scaled"]["anchor_reaction"],
            before["scaled"]["force"],
            before["scaled"]["moment"],
            geometry_length,
        )
        _array_close(diag.q_before, before["scaled"]["q"], "A2 incomplete q_before", 2.0e-8)
        for name, expected in (
            ("mechanics_target_opening_before", mechanics_target_before),
            ("U_before", before["U"]), ("slope_before", before["slope"]),
            ("achieved_opening_before", before["achieved"]),
            ("R_ext_before", before["scaled"]["outer_reaction"]),
            ("R_eff_before", before["R_eff"]),
            ("work_R_ext_before_error", before["R_ext_error"]),
            ("work_R_eff_before_error", before["R_eff_error"]),
            ("mechanics_before_free_residual_norm", before["scaled"]["free_residual_norm"]),
            ("mechanics_before_free_residual_relative_norm", before["scaled"]["free_residual_relative_norm"]),
        ):
            if name in set(diag.available_fields):
                _compare_replay_scalar(getattr(diag, name), expected, f"A2 incomplete {name}")
        if diag.force_before is not None:
            _array_close(diag.force_before, before["scaled"]["force"], "A2 incomplete force_before", 2.0e-9)
        if diag.moment_before is not None:
            _compare_replay_scalar(diag.moment_before, before["scaled"]["moment"], "A2 incomplete moment_before")
        for name in (
            "force_before_error", "force_before_scale", "force_before_limit",
            "moment_before_error", "moment_before_scale", "moment_before_limit",
        ):
            metric_name = (
                "force_" + name.removeprefix("force_before_")
                if name.startswith("force_before_")
                else "moment_" + name.removeprefix("moment_before_")
            )
            _compare_replay_scalar(
                getattr(diag, name),
                conservation_before[metric_name],
                f"A2 incomplete {name}",
            )
        for name, expected in (
            ("force_before_pass", conservation_before["force_pass"]),
            ("moment_before_pass", conservation_before["moment_pass"]),
        ):
            if bool(getattr(diag, name)) != bool(expected):
                raise ValueError(f"A2 incomplete {name} does not match primitive gates")
        for name, expected in (
            ("opening_before_pass", before_flags["opening"]),
            ("slope_before_pass", before_flags["slope"]),
            ("work_before_pass", before_flags["work"]),
            ("cg_before_pass", before_flags["cg"]),
        ):
            if bool(getattr(diag, name)) != bool(expected):
                raise ValueError(f"A2 incomplete {name} does not match primitive gates")
        _validate_replay_seed(
            diag.seed_before,
            seed_source_q,
            seed_source_outer,
            seed_source_state,
            source_step,
            int(diag.sweep_index),
            seed_source_damage_digest,
            _damage_digest(d_iter),
            metadata,
        )
        center = _replay_center_weights(
            d_iter, metadata.union_material_phase_volumes, metadata.union_grip_phase_volumes,
            metadata.k_res, metadata.n_material,
        )
        seed = _replay_normalized_seed(seed_source_q, seed_source_outer, metadata)
        seed_response = _replay_response(seed, ba, center, metadata.youngs_modulus, metadata.poisson_ratio)
        _compare_replay_scalar(diag.seed_before.initial_residual_norm, np.linalg.norm(np.asarray(seed_response["residual"])[metadata.free_dofs]), "A2 incomplete seed_before.initial_residual_norm")
        _compare_replay_scalar(diag.seed_before.final_residual_norm, before["unit"]["free_residual_norm"], "A2 incomplete seed_before.final_residual_norm")
        if diag.Psi_before is not None:
            psi_before, H_used = _replay_amor(before["scaled"]["q"], Hc, metadata, ba)
            _array_close(diag.Psi_before, psi_before, "A2 incomplete Psi_before", 2.0e-9)
            if diag.H_used is not None:
                _array_close(diag.H_used, H_used, "A2 incomplete H_used", 2.0e-9)
    if diag.d_next is not None:
        if diag.H_used is None:
            raise ValueError("incomplete A2 damage field lacks H_used")
        d_next_expected, _, _, matrix, _ = _replay_damage(diag.H_used, metadata)
        _array_close(diag.d_next, d_next_expected, "A2 incomplete d_next", 2.0e-9)
        if diag.failure_stage == "damage_acceptance":
            range_pass = bool(
                np.min(d_next_expected) >= -metadata.config.damage_range_atol
                and np.max(d_next_expected) <= 1.0 + metadata.config.damage_range_atol
            )
            healing_pass = bool(
                np.min(d_next_expected - _f64(committed_damage, "replay committed damage", d_iter.shape))
                >= -metadata.config.healing_atol
            )
            failure_names = []
            if not range_pass:
                failure_names.append("damage_range")
            if not healing_pass:
                failure_names.append("healing")
            expected_reason = ", ".join(failure_names)
            if not expected_reason or diag.failure_reason != expected_reason:
                raise ValueError(
                    "incomplete A2 damage-gate failure reason does not match replay"
                )
        if diag.rho_used is not None and diag.rho_used_volume is not None:
            _compare_residual_reconstruction(
                diag.rho_used, diag.rho_used_volume, diag.d_next, diag.H_used,
                matrix, metadata.material_volumes, metadata.Gc, metadata.l0,
                "A2 incomplete rho_used",
            )
    after = None
    if diag.q_after is not None:
        if diag.d_next is None:
            raise ValueError("incomplete A2 q_after lacks d_next")
        if mechanics_target_after is None:
            raise ValueError("incomplete A2 q_after lacks mechanics target provenance")
        after = _replay_opening(diag.d_next, mechanics_target_after, metadata, ba)
        after_flags = _replay_mechanics_gate_flags(
            after, target, metadata, geometry_length
        )
        conservation_after = _conservation_metrics_from_values(
            metadata.config,
            after["scaled"]["outer_reaction"],
            after["scaled"]["bottom_reaction"],
            after["scaled"]["anchor_reaction"],
            after["scaled"]["force"],
            after["scaled"]["moment"],
            geometry_length,
        )
        _array_close(diag.q_after, after["scaled"]["q"], "A2 incomplete q_after", 2.0e-8)
        for name, expected in (
            ("mechanics_target_opening_after", mechanics_target_after),
            ("U_after", after["U"]), ("slope_after", after["slope"]),
            ("achieved_opening_after", after["achieved"]),
            ("R_ext_after", after["scaled"]["outer_reaction"]),
            ("R_eff_after", after["R_eff"]),
            ("work_R_ext_after_error", after["R_ext_error"]),
            ("work_R_eff_after_error", after["R_eff_error"]),
            ("mechanics_after_free_residual_norm", after["scaled"]["free_residual_norm"]),
            ("mechanics_after_free_residual_relative_norm", after["scaled"]["free_residual_relative_norm"]),
        ):
            if name in set(diag.available_fields):
                _compare_replay_scalar(getattr(diag, name), expected, f"A2 incomplete {name}")
        if diag.force_after is not None:
            _array_close(diag.force_after, after["scaled"]["force"], "A2 incomplete force_after", 2.0e-9)
        if diag.moment_after is not None:
            _compare_replay_scalar(diag.moment_after, after["scaled"]["moment"], "A2 incomplete moment_after")
        for name in (
            "force_after_error", "force_after_scale", "force_after_limit",
            "moment_after_error", "moment_after_scale", "moment_after_limit",
        ):
            metric_name = (
                "force_" + name.removeprefix("force_after_")
                if name.startswith("force_after_")
                else "moment_" + name.removeprefix("moment_after_")
            )
            _compare_replay_scalar(
                getattr(diag, name),
                conservation_after[metric_name],
                f"A2 incomplete {name}",
            )
        for name, expected in (
            ("force_after_pass", conservation_after["force_pass"]),
            ("moment_after_pass", conservation_after["moment_pass"]),
        ):
            if bool(getattr(diag, name)) != bool(expected):
                raise ValueError(f"A2 incomplete {name} does not match primitive gates")
        for name, expected in (
            ("opening_after_pass", after_flags["opening"]),
            ("slope_after_pass", after_flags["slope"]),
            ("work_after_pass", after_flags["work"]),
            ("cg_after_pass", after_flags["cg"]),
        ):
            if bool(getattr(diag, name)) != bool(expected):
                raise ValueError(f"A2 incomplete {name} does not match primitive gates")
        if diag.q_before is None or diag.U_before is None:
            raise ValueError("incomplete A2 q_after lacks mechanics-before source fields")
        _validate_replay_seed(
            diag.seed_after,
            diag.q_before,
            diag.U_before,
            "mechanics_before",
            source_step,
            int(diag.sweep_index),
            _damage_digest(d_iter),
            _damage_digest(diag.d_next),
            metadata,
        )
        if diag.Psi_after is not None:
            _, H_after = _replay_amor(after["scaled"]["q"], Hc, metadata, ba)
            psi_after, _ = _replay_amor(after["scaled"]["q"], Hc, metadata, ba)
            _array_close(diag.Psi_after, psi_after, "A2 incomplete Psi_after", 2.0e-9)

    expected_work = bool(
        before_flags is not None
        and after_flags is not None
        and before_flags["work"]
        and after_flags["work"]
    )
    expected_force = bool(
        before_flags is not None
        and after_flags is not None
        and before_flags["force"]
        and after_flags["force"]
    )
    expected_moment = bool(
        before_flags is not None
        and after_flags is not None
        and before_flags["moment"]
        and after_flags["moment"]
    )
    expected_mechanics = bool(
        before_flags is not None
        and after_flags is not None
        and before_flags["mechanics"]
        and after_flags["mechanics"]
    )
    for name, expected in (
        ("work_pass", expected_work),
        ("force_pass", expected_force),
        ("moment_pass", expected_moment),
        ("mechanics_pass", expected_mechanics),
    ):
        if bool(getattr(diag, name)) != expected:
            raise ValueError(f"A2 incomplete {name} does not match primitive gates")
    if diag.d_next is None:
        expected_range = False
        expected_healing = False
    else:
        expected_range = bool(
            np.min(diag.d_next) >= -metadata.config.damage_range_atol
            and np.max(diag.d_next) <= 1.0 + metadata.config.damage_range_atol
        )
        expected_healing = bool(
            np.min(diag.d_next - _f64(committed_damage, "replay committed damage", d_iter.shape))
            >= -metadata.config.healing_atol
        )
    for name, expected in (
        ("damage_range_pass", expected_range),
        ("healing_pass", expected_healing),
    ):
        if bool(getattr(diag, name)) != expected:
            raise ValueError(f"A2 incomplete {name} does not match primitive gates")
    for prefix, flags in (("before", before_flags), ("after", after_flags)):
        if flags is not None:
            continue
        for name in ("opening", "slope", "work", "cg"):
            if bool(getattr(diag, f"{name}_{prefix}_pass")):
                raise ValueError(
                    f"A2 incomplete {name}_{prefix}_pass was fabricated"
                )
    if diag.failure_stage == "mechanics_before_gate":
        if before_flags is None:
            raise ValueError("A2 incomplete mechanics-before gate lacks replay primitives")
        expected_reason = _a2_mechanics_gate_reason_from_flags(
            "mechanics-before", before_flags
        )
        if expected_reason is None or diag.failure_reason != expected_reason:
            raise ValueError(
                "incomplete A2 mechanics-before gate reason does not match replay"
            )
    elif diag.failure_stage == "mechanics_after_gate":
        if after_flags is None:
            raise ValueError("A2 incomplete mechanics-after gate lacks replay primitives")
        expected_reason = _a2_mechanics_gate_reason_from_flags(
            "mechanics-after", after_flags
        )
        if expected_reason is None or diag.failure_reason != expected_reason:
            raise ValueError(
                "incomplete A2 mechanics-after gate reason does not match replay"
            )


def independent_replay(result: A2ContinuationResult) -> dict[str, Any]:
    """Replay a result using only primitive NumPy algebra, never production helpers."""

    if not isinstance(result, A2ContinuationResult):
        raise TypeError("independent_replay requires an A2ContinuationResult")
    if result.acceptance_mode != A2_ACCEPTANCE_MODE:
        raise ValueError("A2 replay requires literal acceptance_mode='strict'")
    if result.acceptance_status != A2_ACCEPTANCE_STATUS:
        raise ValueError("A2 replay requires literal acceptance_status='strict_passed'")
    if result.completion_status != A2_COMPLETION_STATUS:
        raise ValueError("A2 replay requires a completed result")
    metadata = result.metadata
    static = result.model.static
    if metadata.raw_source_used is not False or metadata.schema != ARCHIVE_SCHEMA:
        raise ValueError("A2 replay requires the clean-start v1 schema and raw_source_used=False")
    if result.final_state.physical_opening != metadata.config.opening_final:
        raise ValueError("A2 replay requires the exact configured final opening")
    ba = _replay_ba(
        metadata.union_coordinates,
        metadata.union_volumes,
        metadata.visible_edges,
        metadata.delta,
        metadata.rank_rtol,
    )
    if _static_digest_from_object(static) != metadata.static_digest:
        raise ValueError("A2 independent replay static digest does not match metadata")
    zero_center = _replay_center_weights(
        np.zeros(metadata.n_material, dtype=np.float64),
        metadata.union_material_phase_volumes,
        metadata.union_grip_phase_volumes,
        metadata.k_res,
        metadata.n_material,
    )
    zero_response = _replay_response(
        np.zeros(metadata.n_dofs, dtype=np.float64),
        ba,
        zero_center,
        metadata.youngs_modulus,
        metadata.poisson_ratio,
    )
    if abs(float(zero_response["energy"])) > _REPLAY_BOUND or _max_abs(zero_response["residual"]) > _REPLAY_BOUND:
        raise ValueError("A2 exact zero state does not replay to zero energy/residual")
    _, zero_history = _replay_amor(
        np.zeros(metadata.n_dofs, dtype=np.float64),
        np.zeros(metadata.n_material, dtype=np.float64),
        metadata,
        ba,
    )
    if not np.array_equal(zero_history, np.zeros(metadata.n_material, dtype=np.float64)):
        raise ValueError("A2 exact zero state does not replay to zero Amor driver")
    current = result.initial_state
    accepted_index = 0
    expected_cutback_count = 0
    for attempt in result.attempts:
        if attempt.acceptance_mode != A2_ACCEPTANCE_MODE:
            raise ValueError("A2 attempt acceptance mode is forged")
        if attempt.accepted and attempt.acceptance_status != A2_ACCEPTANCE_STATUS:
            raise ValueError("A2 accepted attempt status is forged")
        if not attempt.accepted and attempt.acceptance_status is not None:
            raise ValueError("A2 rejected attempt has an acceptance status")
        if attempt.committed_before_digest != current.digest:
            raise ValueError("A2 attempt committed-before digest is not paired")
        if attempt.load_start != current.physical_opening:
            raise ValueError("A2 attempt load_start is not paired with the committed state")
        if attempt.cutback_count < 0:
            raise ValueError("A2 attempt cutback count is negative")
        if attempt.cutback_count != expected_cutback_count:
            raise ValueError("A2 attempt cutback count is not paired with continuation state")
        local_d_iter = np.array(current.material_damage, copy=True)
        seed_source_q = current.union_displacement
        seed_source_outer = current.outer_displacement
        seed_source_state = "committed"
        seed_source_damage_digest = _damage_digest(current.material_damage)
        for expected_sweep_index, sweep in enumerate(attempt.sweeps):
            if sweep.sweep_index != expected_sweep_index:
                raise ValueError("A2 sweep indices are not contiguous")
            if sweep.target_opening != attempt.load_target:
                raise ValueError("A2 sweep target is not paired with its attempt")
            if sweep.complete_sweep:
                if not np.array_equal(sweep.damage_before, current.material_damage):
                    raise ValueError("A2 complete sweep damage_before is not committed damage")
                if not np.array_equal(sweep.d_iter, local_d_iter):
                    raise ValueError("A2 complete sweep d_iter is not the local trial iterate")
                if sweep.q_after is None or sweep.U_after is None or sweep.d_next is None:
                    raise ValueError("A2 complete sweep is missing seed source fields")
                _replay_complete_diagnostics(
                    sweep,
                    current.material_damage,
                    current.material_history,
                    metadata,
                    ba,
                    seed_source_q,
                    seed_source_outer,
                    seed_source_state,
                    seed_source_damage_digest,
                    accepted_index + 1,
                )
                local_d_iter = np.array(sweep.d_next, copy=True)
                seed_source_q = sweep.q_after
                seed_source_outer = sweep.U_after
                seed_source_state = "complete_trial_sweep"
                seed_source_damage_digest = _damage_digest(sweep.d_next)
            else:
                _replay_incomplete_diagnostics(
                    sweep,
                    current.material_damage,
                    current.material_history,
                    metadata,
                    ba,
                    seed_source_q,
                    seed_source_outer,
                    seed_source_state,
                    seed_source_damage_digest,
                    accepted_index + 1,
                )
        final_diagnostic = attempt.diagnostics
        if attempt.accepted != final_diagnostic.passed:
            raise ValueError("A2 attempt status does not match final diagnostics")
        if attempt.accepted:
            if attempt.failure_stage is not None or attempt.failure_reason is not None or attempt.exception_type is not None:
                raise ValueError("accepted A2 attempt retains failure fields")
        elif attempt.failure_stage != final_diagnostic.failure_stage or attempt.failure_reason != final_diagnostic.failure_reason:
            raise ValueError("rejected A2 attempt failure classification is forged")
        elif not final_diagnostic.complete_sweep:
            _validate_a2_attempt_exception_type(
                final_diagnostic.failure_stage,
                False,
                attempt.exception_type,
            )
        else:
            _validate_a2_attempt_exception_type(
                final_diagnostic.failure_stage,
                True,
                attempt.exception_type,
            )
        if attempt.accepted:
            if accepted_index >= len(result.accepted_steps):
                raise ValueError("A2 accepted attempt count exceeds accepted steps")
            step = result.accepted_steps[accepted_index]
            if step.acceptance_mode != A2_ACCEPTANCE_MODE or step.acceptance_status != A2_ACCEPTANCE_STATUS:
                raise ValueError("A2 accepted step status is forged")
            if step.diagnostics is not attempt.diagnostics and step.diagnostics != attempt.diagnostics:
                raise ValueError("A2 accepted attempt/final-step diagnostics are not paired")
            if attempt.committed_after_digest != step.state.digest:
                raise ValueError("A2 accepted attempt committed-after digest is not paired")
            if step.load_start != current.physical_opening or step.load_target != attempt.load_target:
                raise ValueError("A2 accepted load interval is not paired")
            if step.stagger_iterations != final_diagnostic.sweep_index + 1:
                raise ValueError("A2 accepted stagger count is forged")
            if step.cutback_count != attempt.cutback_count:
                raise ValueError("A2 accepted cutback count is forged")
            for name in (
                "damage_before", "d_iter", "d_next", "Psi_before", "Psi_after", "H_used", "H_next",
            ):
                _array_close(getattr(step, name), getattr(final_diagnostic, name), f"A2 accepted {name}", 2.0e-12)
            for name in ("q_before", "q_after"):
                _array_close(getattr(step, name), getattr(final_diagnostic, name), f"A2 accepted {name}", 2.0e-12)
            if not np.array_equal(step.state.union_displacement, step.q_after) or not np.array_equal(step.state.material_damage, step.d_next) or not np.array_equal(step.state.material_history, step.H_next):
                raise ValueError("A2 accepted state primitive pairing is forged")
            current = step.state
            accepted_index += 1
            expected_cutback_count = 0
        else:
            if attempt.committed_after_digest != current.digest:
                raise ValueError("A2 rejected attempt did not roll back committed state")
            _validate_a2_attempt_exception_type(
                final_diagnostic.failure_stage,
                final_diagnostic.complete_sweep,
                attempt.exception_type,
            )
            expected_cutback_count += 1
    if accepted_index != len(result.accepted_steps) or current is not result.final_state:
        raise ValueError("A2 result accepted-state sequence is incomplete")
    return {
        "schema": ARCHIVE_SCHEMA,
        "raw_source_used": False,
        "acceptance_mode": A2_ACCEPTANCE_MODE,
        "acceptance_status": A2_ACCEPTANCE_STATUS,
        "completion_status": A2_COMPLETION_STATUS,
        "accepted_steps": accepted_index,
        "attempts": len(result.attempts),
        "replayed": True,
    }


# ---------------------------------------------------------------------------
# Compact numerical archive
# ---------------------------------------------------------------------------


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _config_payload(config: A2StrictConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, GateTolerance):
            payload[field.name] = {
                "atol": value.atol,
                "rtol": value.rtol,
                "scale_floor": value.scale_floor,
            }
        else:
            payload[field.name] = _json_ready(value)
    payload["acceptance_mode"] = A2_ACCEPTANCE_MODE
    return payload


def _gate_from_payload(value: Mapping[str, Any]) -> GateTolerance:
    return GateTolerance(value["atol"], value["rtol"], value["scale_floor"])


def _config_from_payload(payload: Mapping[str, Any]) -> A2StrictConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("A2 archive config payload must be a mapping")
    expected_names = {field.name for field in fields(A2StrictConfig)} | {"acceptance_mode"}
    unexpected = set(payload) - expected_names
    if unexpected:
        raise ValueError(f"A2 archive config contains unexpected fields: {sorted(unexpected)}")
    mode = payload.get("acceptance_mode")
    if type(mode) is not str or mode != A2_ACCEPTANCE_MODE:
        raise ValueError("A2 archive config requires literal acceptance_mode='strict'")
    values: dict[str, Any] = {}
    for field in fields(A2StrictConfig):
        if field.name not in payload:
            raise ValueError(f"A2 archive config is missing {field.name}")
        value = payload[field.name]
        values[field.name] = _gate_from_payload(value) if field.name in {
            "displacement_material", "displacement_union", "damage", "history",
            "reaction_effective", "damage_residual", "force", "moment",
        } else value
    return A2StrictConfig(**values)


def _metadata_payload(metadata: A2ContinuationMetadata) -> dict[str, Any]:
    return {
        "schema": metadata.schema,
        "acceptance_mode": metadata.config.acceptance_mode,
        "raw_source_used": metadata.raw_source_used,
        "visibility_rule": metadata.visibility_rule,
        "static_digest": metadata.static_digest,
        "material_hash": metadata.material_hash,
        "union_hash": metadata.union_hash,
        "code_hash": metadata.code_hash,
        "material_anchor": metadata.material_anchor,
        "delta": metadata.delta,
        "h_g": metadata.h_g,
        "visibility_tolerance": metadata.visibility_tolerance,
        "youngs_modulus": metadata.youngs_modulus,
        "poisson_ratio": metadata.poisson_ratio,
        "k_res": metadata.k_res,
        "rank_rtol": metadata.rank_rtol,
        "Gc": metadata.Gc,
        "l0": metadata.l0,
        "config": _config_payload(metadata.config),
    }


def _diag_namespace(payload: Mapping[str, Any]) -> SimpleNamespace:
    converted: dict[str, Any] = {}
    for name, value in payload.items():
        if name in {
            "damage_before", "d_iter", "d_next", "Psi_before", "Psi_after", "H_used", "H_next",
            "rho_used", "rho_next", "rho_used_volume", "rho_next_volume", "q_before", "q_after",
            "force_before", "force_after",
        }:
            converted[name] = None if value is None else np.asarray(value, dtype=np.float64)
        elif name in {"seed_before", "seed_after"}:
            converted[name] = None if value is None else SimpleNamespace(**value)
        elif name == "available_fields":
            converted[name] = tuple(value)
        else:
            converted[name] = value
    return SimpleNamespace(**converted)


def _compare_diagnostic_records(first: Any, second: Any, name: str) -> None:
    """Compare an in-memory diagnostic with its JSON/archive duplicate."""

    for field in fields(A2Diagnostics):
        field_name = field.name
        left = getattr(first, field_name, None)
        right = getattr(second, field_name, None)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if left is None or right is None:
                raise ValueError(f"{name}.{field_name} availability differs")
            _array_close(left, right, f"{name}.{field_name}", 2.0e-12)
        elif field_name in {"seed_before", "seed_after"}:
            if left is None or right is None:
                if left is not right:
                    raise ValueError(f"{name}.{field_name} availability differs")
            else:
                for seed_field in fields(A2SeedProvenance):
                    if getattr(left, seed_field.name) != getattr(right, seed_field.name):
                        raise ValueError(f"{name}.{field_name}.{seed_field.name} differs")
        elif isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
            _compare_replay_scalar(left, right, f"{name}.{field_name}", 2.0e-12)
        else:
            if left != right:
                raise ValueError(f"{name}.{field_name} differs")


def _archive_string_scalar(archive: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(archive[name])
    if value.ndim != 0 or value.dtype.kind != "U":
        raise ValueError(f"A2 archive {name} must be a scalar Unicode string")
    return str(value.reshape(()))


def _archive_string_vector(
    archive: Mapping[str, np.ndarray], name: str, length: int,
) -> np.ndarray:
    value = np.asarray(archive[name])
    if value.shape != (length,) or value.dtype.kind != "U":
        raise ValueError(f"A2 archive {name} must be a Unicode string vector of length {length}")
    return value


def _archive_metadata_from_arrays(archive: Mapping[str, np.ndarray]) -> A2ContinuationMetadata:
    metadata_json = json.loads(str(np.asarray(archive["metadata_json"]).reshape(())))
    config = _config_from_payload(metadata_json["config"])
    if metadata_json.get("acceptance_mode") != A2_ACCEPTANCE_MODE:
        raise ValueError("A2 archive metadata requires literal acceptance_mode='strict'")
    return A2ContinuationMetadata(
        schema=str(np.asarray(archive["archive_schema"]).reshape(())),
        raw_source_used=bool(np.asarray(archive["raw_source_used"]).reshape(())),
        material_coordinates=archive["material_coordinates"],
        material_volumes=archive["material_volumes"],
        material_triangles=archive["material_triangles"],
        material_edges=archive["material_edges"],
        material_bottom_mask=archive["material_bottom_mask"],
        material_top_mask=archive["material_top_mask"],
        material_anchor=int(np.asarray(archive["material_anchor"]).reshape(())),
        union_coordinates=archive["union_coordinates"],
        union_material_phase_volumes=archive["union_material_phase_volumes"],
        union_grip_phase_volumes=archive["union_grip_phase_volumes"],
        union_volumes=archive["union_volumes"],
        grip_triangles=archive["grip_triangles"],
        union_triangles=archive["union_triangles"],
        candidate_edges=archive["candidate_edges"],
        visible_edges=archive["visible_edges"],
        removed_edges=archive["removed_edges"],
        interface_bottom_mask=archive["interface_bottom_mask"],
        interface_top_mask=archive["interface_top_mask"],
        outer_bottom_mask=archive["outer_bottom_mask"],
        outer_top_mask=archive["outer_top_mask"],
        constrained_dofs=archive["constrained_dofs"],
        free_dofs=archive["free_dofs"],
        top_line_indices=archive["top_line_indices"],
        top_line_weights=archive["top_line_weights"],
        top_line_length=float(np.asarray(archive["top_line_length"]).reshape(())),
        bottom_line_indices=archive["bottom_line_indices"],
        bottom_line_weights=archive["bottom_line_weights"],
        bottom_line_length=float(np.asarray(archive["bottom_line_length"]).reshape(())),
        delta=float(metadata_json["delta"]),
        h_g=float(metadata_json["h_g"]),
        visibility_tolerance=float(metadata_json["visibility_tolerance"]),
        visibility_rule=str(metadata_json["visibility_rule"]),
        youngs_modulus=float(metadata_json["youngs_modulus"]),
        poisson_ratio=float(metadata_json["poisson_ratio"]),
        k_res=float(metadata_json["k_res"]),
        rank_rtol=float(metadata_json["rank_rtol"]),
        Gc=float(metadata_json["Gc"]),
        l0=float(metadata_json["l0"]),
        static_digest=str(metadata_json["static_digest"]),
        material_hash=str(metadata_json["material_hash"]),
        union_hash=str(metadata_json["union_hash"]),
        config=config,
        code_hash=str(metadata_json["code_hash"]),
    )


def write_a2_archive(path: Any, result: A2ContinuationResult) -> None:
    """Write one compact numerical-only v1 archive."""

    if not isinstance(result, A2ContinuationResult):
        raise TypeError("result must be an A2ContinuationResult")
    result.validate()
    filename = os.fspath(path)
    if not isinstance(filename, (str, bytes)) or not filename:
        raise ValueError("archive path must be non-empty")
    parent = os.path.dirname(os.fsdecode(filename))
    if parent:
        os.makedirs(parent, exist_ok=True)
    metadata = result.metadata
    static = result.model.static
    states = result.states
    steps = result.accepted_steps
    attempts = result.attempts
    arrays: dict[str, np.ndarray] = {
        "archive_schema": np.asarray(ARCHIVE_SCHEMA),
        "acceptance_mode": np.asarray(result.acceptance_mode),
        "acceptance_status": np.asarray(result.acceptance_status),
        "completion_status": np.asarray(result.completion_status),
        "raw_source_used": np.asarray(False, dtype=np.bool_),
        "material_coordinates": metadata.material_coordinates,
        "material_volumes": metadata.material_volumes,
        "material_triangles": metadata.material_triangles,
        "material_edges": metadata.material_edges,
        "material_prefix_map": np.arange(metadata.n_material, dtype=np.int64),
        "material_bottom_mask": metadata.material_bottom_mask,
        "material_top_mask": metadata.material_top_mask,
        "material_anchor": np.asarray(metadata.material_anchor, dtype=np.int64),
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
        "static_target_beta": static.target_beta,
        "static_gradient_coefficient": static.gradient_coefficient,
        "union_coordinates": metadata.union_coordinates,
        "union_material_phase_volumes": metadata.union_material_phase_volumes,
        "union_grip_phase_volumes": metadata.union_grip_phase_volumes,
        "union_volumes": metadata.union_volumes,
        "grip_triangles": metadata.grip_triangles,
        "union_triangles": metadata.union_triangles,
        "candidate_edges": metadata.candidate_edges,
        "visible_edges": metadata.visible_edges,
        "removed_edges": metadata.removed_edges,
        "interface_bottom_mask": metadata.interface_bottom_mask,
        "interface_top_mask": metadata.interface_top_mask,
        "outer_bottom_mask": metadata.outer_bottom_mask,
        "outer_top_mask": metadata.outer_top_mask,
        "constrained_dofs": metadata.constrained_dofs,
        "free_dofs": metadata.free_dofs,
        "top_line_indices": metadata.top_line_indices,
        "top_line_weights": metadata.top_line_weights,
        "top_line_length": np.asarray(metadata.top_line_length, dtype=np.float64),
        "bottom_line_indices": metadata.bottom_line_indices,
        "bottom_line_weights": metadata.bottom_line_weights,
        "bottom_line_length": np.asarray(metadata.bottom_line_length, dtype=np.float64),
        "state_physical_opening": np.asarray([state.physical_opening for state in states], dtype=np.float64),
        "state_outer_displacement": np.asarray([state.outer_displacement for state in states], dtype=np.float64),
        "state_union_displacement": np.stack([state.union_displacement for state in states]),
        "state_material_damage": np.stack([state.material_damage for state in states]),
        "state_material_history": np.stack([state.material_history for state in states]),
        "state_digest": np.asarray([state.digest for state in states], dtype=np.str_),
        "state_acceptance_mode": np.asarray([result.acceptance_mode] * len(states), dtype=np.str_),
        "state_acceptance_status": np.asarray(
            ["initial"] + [step.acceptance_status for step in steps], dtype=np.str_
        ),
        "accepted_target_opening": np.asarray([step.load_target for step in steps], dtype=np.float64),
        "accepted_outer_displacement": np.asarray([step.state.outer_displacement for step in steps], dtype=np.float64),
        "accepted_damage_before": np.stack([step.damage_before for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_d_iter": np.stack([step.d_iter for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_d_next": np.stack([step.d_next for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_q_before": np.stack([step.q_before for step in steps]) if steps else np.empty((0, metadata.n_dofs), dtype=np.float64),
        "accepted_q_after": np.stack([step.q_after for step in steps]) if steps else np.empty((0, metadata.n_dofs), dtype=np.float64),
        "accepted_Psi_before": np.stack([step.Psi_before for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_Psi_after": np.stack([step.Psi_after for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_H_used": np.stack([step.H_used for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_H_next": np.stack([step.H_next for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_rho_used": np.stack([step.diagnostics.rho_used for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_rho_next": np.stack([step.diagnostics.rho_next for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_rho_used_volume": np.stack([step.diagnostics.rho_used_volume for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_rho_next_volume": np.stack([step.diagnostics.rho_next_volume for step in steps]) if steps else np.empty((0, metadata.n_material), dtype=np.float64),
        "accepted_stagger_iterations": np.asarray([step.stagger_iterations for step in steps], dtype=np.int64),
        "accepted_cutback_count": np.asarray([step.cutback_count for step in steps], dtype=np.int64),
        "accepted_acceptance_mode": np.asarray([step.acceptance_mode for step in steps], dtype=np.str_),
        "accepted_acceptance_status": np.asarray([step.acceptance_status for step in steps], dtype=np.str_),
        "accepted_diagnostics_json": np.asarray([json.dumps(_json_ready(step.diagnostics), sort_keys=True) for step in steps], dtype=np.str_),
        "attempt_cutback_count": np.asarray([attempt.cutback_count for attempt in attempts], dtype=np.int64),
        "attempt_load_start": np.asarray([attempt.load_start for attempt in attempts], dtype=np.float64),
        "attempt_load_target": np.asarray([attempt.load_target for attempt in attempts], dtype=np.float64),
        "attempt_accepted": np.asarray([attempt.accepted for attempt in attempts], dtype=np.bool_),
        "attempt_acceptance_mode": np.asarray([attempt.acceptance_mode for attempt in attempts], dtype=np.str_),
        "attempt_acceptance_status": np.asarray(
            [attempt.acceptance_status or "" for attempt in attempts], dtype=np.str_
        ),
        "attempt_before_digest": np.asarray([attempt.committed_before_digest for attempt in attempts], dtype=np.str_),
        "attempt_after_digest": np.asarray([attempt.committed_after_digest for attempt in attempts], dtype=np.str_),
        "attempt_failure_stage": np.asarray([attempt.failure_stage or "" for attempt in attempts], dtype=np.str_),
        "attempt_failure_reason": np.asarray([attempt.failure_reason or "" for attempt in attempts], dtype=np.str_),
        "attempt_exception_type": np.asarray([attempt.exception_type or "" for attempt in attempts], dtype=np.str_),
        "attempt_sweeps_json": np.asarray([json.dumps(_json_ready(attempt.sweeps), sort_keys=True) for attempt in attempts], dtype=np.str_),
        "metadata_json": np.asarray(json.dumps(_metadata_payload(metadata), sort_keys=True)),
        "config_json": np.asarray(json.dumps(_config_payload(metadata.config), sort_keys=True)),
    }
    for name, value in arrays.items():
        if np.asarray(value).dtype.kind == "O":
            raise ValueError(f"A2 archive field {name} has object dtype")
    np.savez_compressed(filename, **arrays)


def load_a2_archive(path: Any) -> dict[str, np.ndarray]:
    """Load a numerical-only archive with pickle disabled."""

    filename = os.fspath(path)
    with np.load(filename, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    for name, value in result.items():
        if value.dtype.kind == "O":
            raise ValueError(f"A2 archive field {name} has forbidden object dtype")
    return result


def replay_a2_archive(path: Any) -> dict[str, Any]:
    """Independently replay an on-disk v1 archive from primitive fields."""

    archive = load_a2_archive(path)
    required = {
        "archive_schema", "acceptance_mode", "acceptance_status", "completion_status",
        "raw_source_used", "metadata_json", "config_json", "state_physical_opening",
        "state_outer_displacement", "state_union_displacement", "state_material_damage",
        "state_material_history", "state_digest", "state_acceptance_mode", "state_acceptance_status",
        "attempt_before_digest", "attempt_after_digest", "attempt_acceptance_mode",
        "attempt_acceptance_status", "attempt_accepted", "attempt_sweeps_json", "material_prefix_map",
        "static_edge_distance", "static_edge_omega", "static_family_mass", "static_eta",
        "static_support_indices", "static_support_mask", "static_support_sizes", "static_beta",
        "static_xi", "static_shape_inverse", "static_shape_rank", "static_shape_condition",
        "static_target_beta", "static_gradient_coefficient",
        "attempt_failure_stage", "attempt_failure_reason", "attempt_exception_type",
        "accepted_target_opening", "accepted_outer_displacement", "accepted_damage_before",
        "accepted_d_iter", "accepted_q_before", "accepted_Psi_before", "accepted_Psi_after",
        "accepted_H_used", "accepted_rho_used", "accepted_rho_used_volume",
        "accepted_d_next", "accepted_q_after", "accepted_H_next", "accepted_rho_next",
        "accepted_rho_next_volume",
        "accepted_stagger_iterations", "accepted_cutback_count", "accepted_acceptance_mode",
        "accepted_acceptance_status",
        "accepted_diagnostics_json",
    }
    missing = sorted(required - set(archive))
    if missing:
        raise ValueError(f"A2 archive is missing required fields: {missing}")
    if _archive_string_scalar(archive, "archive_schema") != ARCHIVE_SCHEMA:
        raise ValueError("A2 archive schema mismatch")
    if _archive_string_scalar(archive, "acceptance_mode") != A2_ACCEPTANCE_MODE:
        raise ValueError("A2 archive requires literal acceptance_mode='strict'")
    if _archive_string_scalar(archive, "acceptance_status") != A2_ACCEPTANCE_STATUS:
        raise ValueError("A2 archive requires literal acceptance_status='strict_passed'")
    if _archive_string_scalar(archive, "completion_status") != A2_COMPLETION_STATUS:
        raise ValueError("A2 archive completion status is not completed")
    if bool(np.asarray(archive["raw_source_used"]).reshape(())):
        raise ValueError("A2 archive raw_source_used must be False")
    metadata = _archive_metadata_from_arrays(archive)
    config_payload = json.loads(_archive_string_scalar(archive, "config_json"))
    config_from_archive = _config_from_payload(config_payload)
    if _config_payload(config_from_archive) != config_payload:
        raise ValueError("A2 archive config_json is not canonical")
    if _config_payload(config_from_archive) != _config_payload(metadata.config):
        raise ValueError("A2 archive config_json does not match metadata config")
    current_code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if metadata.code_hash != current_code_hash:
        raise ValueError("A2 archive code hash does not match the replay implementation")
    if not np.array_equal(archive["material_prefix_map"], np.arange(metadata.n_material, dtype=np.int64)):
        raise ValueError("A2 archive material prefix map is forged")
    ba = _replay_ba(metadata.union_coordinates, metadata.union_volumes, metadata.visible_edges, metadata.delta, metadata.rank_rtol)
    if _archive_static_digest(metadata, ba, archive) != metadata.static_digest:
        raise ValueError("A2 archive static digest does not match serialized primitives")
    zero_center = _replay_center_weights(np.zeros(metadata.n_material, dtype=np.float64), metadata.union_material_phase_volumes, metadata.union_grip_phase_volumes, metadata.k_res, metadata.n_material)
    zero = _replay_response(np.zeros(metadata.n_dofs, dtype=np.float64), ba, zero_center, metadata.youngs_modulus, metadata.poisson_ratio)
    if abs(float(zero["energy"])) > _REPLAY_BOUND or _max_abs(zero["residual"]) > _REPLAY_BOUND:
        raise ValueError("A2 archive zero state failed independent replay")
    state_opening = _f64(archive["state_physical_opening"], "archive state opening")
    state_outer = _f64(archive["state_outer_displacement"], "archive state outer")
    state_q = _f64(archive["state_union_displacement"], "archive state q")
    state_d = _f64(archive["state_material_damage"], "archive state damage")
    state_H = _f64(archive["state_material_history"], "archive state history")
    state_digest = np.asarray(archive["state_digest"])
    state_mode = _archive_string_vector(archive, "state_acceptance_mode", len(state_opening))
    state_status = _archive_string_vector(archive, "state_acceptance_status", len(state_opening))
    if (
        len(state_opening) == 0
        or state_outer.shape != (len(state_opening),)
        or state_q.shape != (len(state_opening), metadata.n_dofs)
        or state_d.shape != (len(state_opening), metadata.n_material)
        or state_H.shape != (len(state_opening), metadata.n_material)
        or state_digest.shape != (len(state_opening),)
    ):
        raise ValueError("A2 archive state shapes are inconsistent")
    if not np.all(state_mode == A2_ACCEPTANCE_MODE):
        raise ValueError("A2 archive state acceptance mode is forged")
    if state_status[0] != "initial" or not np.all(state_status[1:] == A2_ACCEPTANCE_STATUS):
        raise ValueError("A2 archive state acceptance status is forged")
    if state_opening[-1] != metadata.config.opening_final:
        raise ValueError("A2 archive final opening is not exactly opening_final")
    if (
        state_opening[0] != 0.0
        or state_outer[0] != 0.0
        or not np.array_equal(state_q[0], np.zeros(metadata.n_dofs, dtype=np.float64))
        or not np.array_equal(state_d[0], np.zeros(metadata.n_material, dtype=np.float64))
        or not np.array_equal(state_H[0], np.zeros(metadata.n_material, dtype=np.float64))
    ):
        raise ValueError("A2 archive initial state is not exact clean zero")
    expected_digests = np.asarray([_digest_state(state_opening[i], state_outer[i], state_q[i], state_d[i], state_H[i]) for i in range(len(state_opening))], dtype=np.str_)
    if not np.array_equal(state_digest, expected_digests):
        raise ValueError("A2 archive state digest mismatch")
    attempts_before = np.asarray(archive["attempt_before_digest"])
    attempts_after = np.asarray(archive["attempt_after_digest"])
    attempts_accepted = np.asarray(archive["attempt_accepted"], dtype=bool)
    attempts_cutback = _i64(archive["attempt_cutback_count"], "archive attempt cutback count")
    attempts_load_start = _f64(archive["attempt_load_start"], "archive attempt load start")
    attempts_load_target = _f64(archive["attempt_load_target"], "archive attempt load target")
    sweeps_json = np.asarray(archive["attempt_sweeps_json"])
    attempts_stage = np.asarray(archive["attempt_failure_stage"])
    attempts_reason = np.asarray(archive["attempt_failure_reason"])
    attempts_exception = np.asarray(archive["attempt_exception_type"])
    attempts_mode = _archive_string_vector(archive, "attempt_acceptance_mode", len(attempts_before))
    attempts_status = _archive_string_vector(archive, "attempt_acceptance_status", len(attempts_before))
    accepted_diag_json = np.asarray(archive["accepted_diagnostics_json"])
    accepted_target = _f64(archive["accepted_target_opening"], "archive accepted target opening")
    accepted_outer = _f64(archive["accepted_outer_displacement"], "archive accepted outer displacement")
    accepted_damage_before = _f64(archive["accepted_damage_before"], "archive accepted damage_before")
    accepted_d_iter = _f64(archive["accepted_d_iter"], "archive accepted d_iter")
    accepted_q_before = _f64(archive["accepted_q_before"], "archive accepted q_before")
    accepted_psi_before = _f64(archive["accepted_Psi_before"], "archive accepted Psi_before")
    accepted_psi_after = _f64(archive["accepted_Psi_after"], "archive accepted Psi_after")
    accepted_H_used = _f64(archive["accepted_H_used"], "archive accepted H_used")
    accepted_rho_used = _f64(archive["accepted_rho_used"], "archive accepted rho_used")
    accepted_rho_used_volume = _f64(archive["accepted_rho_used_volume"], "archive accepted rho_used_volume")
    accepted_d_next = _f64(archive["accepted_d_next"], "archive accepted d_next")
    accepted_q_after = _f64(archive["accepted_q_after"], "archive accepted q_after")
    accepted_H_next = _f64(archive["accepted_H_next"], "archive accepted H_next")
    accepted_rho_next = _f64(archive["accepted_rho_next"], "archive accepted rho_next")
    accepted_rho_next_volume = _f64(archive["accepted_rho_next_volume"], "archive accepted rho_next_volume")
    accepted_stagger = _i64(archive["accepted_stagger_iterations"], "archive accepted stagger iterations")
    accepted_cutback = _i64(archive["accepted_cutback_count"], "archive accepted cutback count")
    accepted_mode = _archive_string_vector(archive, "accepted_acceptance_mode", len(accepted_target))
    accepted_status = _archive_string_vector(archive, "accepted_acceptance_status", len(accepted_target))
    if (
        len(attempts_before) != len(sweeps_json)
        or len(attempts_before) != len(attempts_accepted)
        or len(attempts_before) != len(attempts_after)
        or len(attempts_before) != len(attempts_stage)
        or len(attempts_before) != len(attempts_reason)
        or len(attempts_before) != len(attempts_exception)
        or len(attempts_before) != len(attempts_cutback)
        or len(attempts_before) != len(attempts_load_start)
        or len(attempts_before) != len(attempts_load_target)
        or len(attempts_before) != len(attempts_mode)
        or len(attempts_before) != len(attempts_status)
        or accepted_diag_json.shape != (len(accepted_target),)
        or any(
            np.asarray(value).shape[0] != len(accepted_target)
            for value in (
                accepted_outer, accepted_damage_before, accepted_d_iter, accepted_q_before,
                accepted_psi_before, accepted_psi_after, accepted_H_used, accepted_rho_used,
                accepted_rho_used_volume, accepted_d_next, accepted_q_after, accepted_H_next,
                accepted_rho_next, accepted_rho_next_volume,
            )
        )
        or len(accepted_target) != len(state_opening) - 1
        or accepted_stagger.shape != (len(accepted_target),)
        or accepted_cutback.shape != (len(accepted_target),)
        or accepted_mode.shape != (len(accepted_target),)
        or accepted_status.shape != (len(accepted_target),)
    ):
        raise ValueError("A2 archive attempt arrays have inconsistent lengths")
    if not np.all(attempts_mode == A2_ACCEPTANCE_MODE):
        raise ValueError("A2 archive attempt acceptance mode is forged")
    if not np.all(accepted_mode == A2_ACCEPTANCE_MODE) or not np.all(accepted_status == A2_ACCEPTANCE_STATUS):
        raise ValueError("A2 archive accepted-step status is forged")
    current_index = 0
    accepted_index = 0
    expected_cutback_count = 0
    for attempt_index in range(len(attempts_before)):
        expected_attempt_status = A2_ACCEPTANCE_STATUS if attempts_accepted[attempt_index] else ""
        if attempts_status[attempt_index] != expected_attempt_status:
            raise ValueError("A2 archive attempt acceptance status is forged")
        if str(attempts_before[attempt_index]) != str(state_digest[current_index]):
            raise ValueError("A2 archive attempt before digest is not paired")
        if attempts_load_start[attempt_index] != state_opening[current_index]:
            raise ValueError("A2 archive attempt load_start is not paired")
        if attempts_load_target[attempt_index] <= attempts_load_start[attempt_index]:
            raise ValueError("A2 archive attempt target is not increasing")
        if attempts_cutback[attempt_index] < 0:
            raise ValueError("A2 archive attempt cutback count is negative")
        if attempts_cutback[attempt_index] != expected_cutback_count:
            raise ValueError("A2 archive attempt cutback count is not paired with continuation state")
        sweep_payloads = json.loads(str(sweeps_json[attempt_index]))
        if not isinstance(sweep_payloads, list) or not sweep_payloads:
            raise ValueError("A2 archive attempt has no sweep audit")
        local_d = np.array(state_d[current_index], copy=True)
        seed_source_q = state_q[current_index]
        seed_source_outer = state_outer[current_index]
        seed_source_state = "committed"
        seed_source_damage_digest = _damage_digest(state_d[current_index])
        for payload in sweep_payloads:
            diag = _diag_namespace(payload)
            if diag.target_opening != attempts_load_target[attempt_index]:
                raise ValueError("A2 archive sweep target is not paired with its attempt")
            if bool(getattr(diag, "complete_sweep")):
                if not np.array_equal(diag.damage_before, state_d[current_index]) or not np.array_equal(diag.d_iter, local_d):
                    raise ValueError("A2 archive complete sweep damage pairing is forged")
                if diag.q_after is None or diag.U_after is None or diag.d_next is None:
                    raise ValueError("A2 archive complete sweep is missing seed source fields")
                _replay_complete_diagnostics(
                    diag,
                    state_d[current_index],
                    state_H[current_index],
                    metadata,
                    ba,
                    seed_source_q,
                    seed_source_outer,
                    seed_source_state,
                    seed_source_damage_digest,
                    accepted_index + 1,
                )
                local_d = np.array(diag.d_next, copy=True)
                seed_source_q = diag.q_after
                seed_source_outer = diag.U_after
                seed_source_state = "complete_trial_sweep"
                seed_source_damage_digest = _damage_digest(diag.d_next)
            else:
                _replay_incomplete_diagnostics(
                    diag,
                    state_d[current_index],
                    state_H[current_index],
                    metadata,
                    ba,
                    seed_source_q,
                    seed_source_outer,
                    seed_source_state,
                    seed_source_damage_digest,
                    accepted_index + 1,
                )
        final_diag = _diag_namespace(sweep_payloads[-1])
        if bool(attempts_accepted[attempt_index]) != bool(final_diag.passed):
            raise ValueError("A2 archive attempt status does not match final diagnostics")
        expected_stage = "" if attempts_accepted[attempt_index] else str(final_diag.failure_stage)
        expected_reason = "" if attempts_accepted[attempt_index] else str(final_diag.failure_reason)
        if str(attempts_stage[attempt_index]) != expected_stage or str(attempts_reason[attempt_index]) != expected_reason:
            raise ValueError("A2 archive attempt failure classification is forged")
        _validate_a2_attempt_exception_type(
            None if attempts_accepted[attempt_index] else str(final_diag.failure_stage),
            bool(final_diag.complete_sweep),
            None if not str(attempts_exception[attempt_index]) else str(attempts_exception[attempt_index]),
        )
        if bool(attempts_accepted[attempt_index]):
            accepted_index += 1
            if accepted_index >= len(state_opening):
                raise ValueError("A2 archive accepted attempt count exceeds states")
            if str(attempts_after[attempt_index]) != str(state_digest[accepted_index]):
                raise ValueError("A2 archive accepted after digest is not paired")
            if accepted_index > len(accepted_diag_json):
                raise ValueError("A2 archive accepted diagnostics are incomplete")
            accepted_diag = _diag_namespace(json.loads(str(accepted_diag_json[accepted_index - 1])))
            _compare_diagnostic_records(final_diag, accepted_diag, "A2 archive accepted diagnostics")
            for name, actual, expected in (
                ("target_opening", accepted_target[accepted_index - 1], final_diag.target_opening),
                ("outer_displacement", accepted_outer[accepted_index - 1], final_diag.U_after),
                ("damage_before", accepted_damage_before[accepted_index - 1], final_diag.damage_before),
                ("d_iter", accepted_d_iter[accepted_index - 1], final_diag.d_iter),
                ("q_before", accepted_q_before[accepted_index - 1], final_diag.q_before),
                ("Psi_before", accepted_psi_before[accepted_index - 1], final_diag.Psi_before),
                ("Psi_after", accepted_psi_after[accepted_index - 1], final_diag.Psi_after),
                ("H_used", accepted_H_used[accepted_index - 1], final_diag.H_used),
                ("rho_used", accepted_rho_used[accepted_index - 1], final_diag.rho_used),
                ("rho_used_volume", accepted_rho_used_volume[accepted_index - 1], final_diag.rho_used_volume),
                ("d_next", accepted_d_next[accepted_index - 1], final_diag.d_next),
                ("q_after", accepted_q_after[accepted_index - 1], final_diag.q_after),
                ("H_next", accepted_H_next[accepted_index - 1], final_diag.H_next),
                ("rho_next", accepted_rho_next[accepted_index - 1], final_diag.rho_next),
                ("rho_next_volume", accepted_rho_next_volume[accepted_index - 1], final_diag.rho_next_volume),
            ):
                if name == "target_opening" or name == "outer_displacement":
                    _compare_replay_scalar(actual, expected, f"A2 archive accepted {name}")
                else:
                    _array_close(actual, expected, f"A2 archive accepted {name}", 2.0e-9)
            if accepted_stagger[accepted_index - 1] != final_diag.sweep_index + 1:
                raise ValueError("A2 archive accepted stagger count is forged")
            if accepted_cutback[accepted_index - 1] != attempts_cutback[attempt_index]:
                raise ValueError("A2 archive accepted cutback count is forged")
            if not np.array_equal(state_q[accepted_index], final_diag.q_after) or not np.array_equal(state_d[accepted_index], final_diag.d_next) or not np.array_equal(state_H[accepted_index], final_diag.H_next):
                raise ValueError("A2 archive accepted state fields are forged")
            _compare_replay_scalar(state_outer[accepted_index], final_diag.U_after, "A2 archive accepted outer displacement")
            current_index = accepted_index
            expected_cutback_count = 0
        elif str(attempts_after[attempt_index]) != str(state_digest[current_index]):
            raise ValueError("A2 archive rejected attempt did not roll back")
        else:
            expected_cutback_count += 1
    if accepted_index != len(state_opening) - 1:
        raise ValueError("A2 archive accepted states are incomplete")
    return {
        "schema": ARCHIVE_SCHEMA,
        "raw_source_used": False,
        "acceptance_mode": A2_ACCEPTANCE_MODE,
        "acceptance_status": A2_ACCEPTANCE_STATUS,
        "completion_status": A2_COMPLETION_STATUS,
        "accepted_steps": accepted_index,
        "attempts": len(attempts_before),
        "replayed": True,
    }
