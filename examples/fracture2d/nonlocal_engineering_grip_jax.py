"""Float64 JAX backend for A2 phase-measured engineering-grip mechanics.

This module is deliberately separate from :mod:`nonlocal_mechanics_jax`.
The material-only backend consumes a ``BondAssociatedMechanics`` geometry and
``HardBoundaryConstraints``; it must not accept A2 phase-measured union views.
Here the accepted production object is an ``A2StaticMechanics`` union model
with its exact ``AffineConstraints``, and the dynamic solve input is the
material damage field only.

The static graph, target-inclusive BA supports, and constraints are flattened
once onto an explicit device.  The phase center weights are computed exactly
as ``Vm*((1-d)^2+k_res)+Vg*(1+k_res)`` with the padded material prefix, the
kinematics/tangent follow the audited target-inclusive bond-associated
construction of the CPU ``A2FixedDamageView``, and the solver is exact
particle block-Jacobi PCG.  No automatic differentiation or finite difference
enters the tangent.

The CPU reference remains authoritative.  ``replay_cpu`` transfers the device
displacement, constructs a fresh ``A2FixedDamageView``, and independently
evaluates CPU energy/residual/reactions/force/moment with an explicit CPU
residual gate, without rerunning CPU CG.  The returned production object is
the existing :class:`A2MechanicsSolveResult`; GPU iteration counts are
non-authoritative execution diagnostics retained on the paired result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .nonlocal_a2_coupled import (
    A2FixedDamageView,
    A2MechanicsSolveResult,
    A2StaticMechanics,
    _check_exact_constraints,
    _immutable,
    _scalar,
    _seed_record,
    _unit_system_digest,
    _validate_view_constraints,
)
from .nonlocal_grip_diagnostic import AffineConstraints


__all__ = [
    "JaxA2CPUReplayAdapter",
    "JaxA2CPUReplayResult",
    "JaxA2SolveResult",
    "JaxA2UnionMechanics",
    "replay_jax_a2_solution_on_cpu",
]


_FLOAT64 = np.dtype(np.float64)
_INT64 = np.dtype(np.int64)

_STATUS_RUNNING = np.int32(-1)
_STATUS_CONVERGED = np.int32(0)
_STATUS_MAXITER = np.int32(1)
_STATUS_BREAKDOWN = np.int32(2)
_STATUS_NONFINITE = np.int32(3)
_STATUS_EXPLICIT_RESIDUAL = np.int32(4)
_STATUS_NAMES = {
    int(_STATUS_CONVERGED): "converged",
    int(_STATUS_MAXITER): "maxiter",
    int(_STATUS_BREAKDOWN): "breakdown",
    int(_STATUS_NONFINITE): "nonfinite",
    int(_STATUS_EXPLICIT_RESIDUAL): "explicit_residual_failure",
}


def _require_x64_enabled() -> None:
    """Require the caller to have enabled JAX x64 before construction/use."""

    if not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "fracture2d JAX A2 mechanics requires jax_enable_x64=True before "
            "constructing or using the backend; it will not enable or downcast"
        )


def _require_static_float64(
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


def _require_static_int64(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _INT64:
        raise ValueError(f"{name} must use exact int64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _require_dynamic_float64(
    values: Any,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray | jax.Array:
    """Validate a dynamic float input without silently downcasting it."""

    array = values if isinstance(values, jax.Array) else np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    finite = jnp.all(jnp.isfinite(array)) if isinstance(array, jax.Array) else np.all(np.isfinite(array))
    if not bool(np.asarray(jax.device_get(finite))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_finite_device_result(values: Any, name: str) -> None:
    leaves = jax.tree_util.tree_leaves(values)
    finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves])
    )
    if not bool(np.asarray(jax.device_get(finite))):
        raise FloatingPointError(f"{name} contains non-finite values")


def _dynamic_float_scalar(
    value: Any,
    name: str,
    *,
    allow_zero: bool,
    nonnegative: bool = True,
) -> np.float64:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    if array.dtype.kind == "f" and array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a scalar number")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and allow_zero and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if nonnegative and not allow_zero and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return np.float64(result)


def _dynamic_maxiter(value: Any) -> np.int32:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.integer)
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError("cg_maxiter must be a positive integer scalar")
    result = int(array)
    if result <= 0 or result > np.iinfo(np.int32).max:
        raise ValueError("cg_maxiter must be a positive int32-range integer")
    return np.int32(result)


def _segment_sum(values: jax.Array, segment_ids: jax.Array, n_segments: int) -> jax.Array:
    """Fixed-size segment reduction used for all edge/endpoint assembly."""

    return jax.ops.segment_sum(values, segment_ids, num_segments=n_segments)


def _closed_form_h_components(
    coefficient: np.ndarray,
    lame_lambda: float,
    shear_modulus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact ``H(c)`` entries for flattened support occurrences."""

    cx = coefficient[:, 0]
    cy = coefficient[:, 1]
    hxx = (lame_lambda + 2.0 * shear_modulus) * cx * cx + shear_modulus * cy * cy
    hxy = (lame_lambda + shear_modulus) * cx * cy
    hyy = shear_modulus * cx * cx + (lame_lambda + 2.0 * shear_modulus) * cy * cy
    return (
        np.asarray(hxx, dtype=np.float64),
        np.asarray(hxy, dtype=np.float64),
        np.asarray(hyy, dtype=np.float64),
    )


@dataclass(frozen=True)
class JaxA2SolveResult:
    """Immutable device solve result with explicit numerical diagnostics.

    The numerical fields remain JAX arrays on the backend device.  Accessing
    ``status`` or ``converged`` intentionally performs the corresponding small
    host read; the PCG loop itself performs no host synchronization.  These
    values are execution diagnostics only; production acceptance is decided by
    the CPU-authoritative replay.
    """

    displacement: jax.Array
    damage: jax.Array
    u_bar: jax.Array
    cg_rtol: jax.Array
    cg_atol: jax.Array
    cg_maxiter: jax.Array
    free_rhs: jax.Array
    free_residual: jax.Array
    recursive_free_residual: jax.Array
    initial_displacement: jax.Array
    initial_free_residual: jax.Array
    free_rhs_norm: jax.Array
    free_residual_norm: jax.Array
    recursive_residual_norm: jax.Array
    free_residual_relative_norm: jax.Array
    initial_free_residual_norm: jax.Array
    cg_convergence_limit: jax.Array
    initial_guess_supplied: jax.Array
    cg_iterations: jax.Array
    cg_info: jax.Array
    status_code: jax.Array
    explicit_converged: jax.Array
    converged_flag: jax.Array
    energy: jax.Array
    top_reaction: jax.Array
    bottom_reaction: jax.Array
    anchor_reaction: jax.Array
    total_force: jax.Array
    total_moment: jax.Array

    @property
    def status(self) -> str:
        code = int(np.asarray(self.status_code))
        return _STATUS_NAMES.get(code, f"unknown({code})")

    @property
    def converged(self) -> bool:
        return bool(np.asarray(self.converged_flag))

    def require_converged(self) -> "JaxA2SolveResult":
        """Raise a strict numerical error unless the solve passed every gate."""

        if self.converged:
            return self
        raise RuntimeError(
            "JAX A2 hard-constraint PCG did not converge: "
            f"status={self.status}, iterations={int(np.asarray(self.cg_iterations))}, "
            f"recursive_norm={float(np.asarray(self.recursive_residual_norm)):.17g}, "
            f"explicit_norm={float(np.asarray(self.free_residual_norm)):.17g}, "
            f"limit={float(np.asarray(self.cg_convergence_limit)):.17g}"
        )


@dataclass(frozen=True)
class JaxA2CPUReplayResult:
    """Pair GPU execution diagnostics with the CPU-authoritative A2 result."""

    jax_result: JaxA2SolveResult
    cpu_result: A2MechanicsSolveResult

    @property
    def gpu_result(self) -> JaxA2SolveResult:
        return self.jax_result

    @property
    def result(self) -> A2MechanicsSolveResult:
        return self.cpu_result


class JaxA2UnionMechanics:
    """Device-resident float64 backend for A2 phase-measured union mechanics.

    ``static`` must be an accepted ``A2StaticMechanics`` union model and
    ``constraints`` its exact ``AffineConstraints`` identity.  All graph,
    support, and constraint arrays are validated then copied once to the
    explicit device; only the material damage field is dynamic per solve.
    """

    def __init__(
        self,
        static: A2StaticMechanics,
        constraints: AffineConstraints,
        device: jax.Device,
    ) -> None:
        _require_x64_enabled()
        if not isinstance(static, A2StaticMechanics):
            raise TypeError("static must be an accepted A2StaticMechanics instance")
        if not isinstance(constraints, AffineConstraints):
            raise TypeError("constraints must be an AffineConstraints instance")
        if constraints.n_particles != static.n_particles:
            raise ValueError("constraints have a different particle count")
        for name in (
            "constrained_dofs",
            "free_dofs",
            "q_zero",
            "load_direction",
            "outer_top_selector",
            "outer_bottom_selector",
            "anchor_selector",
            "outer_bottom_mask",
            "outer_top_mask",
        ):
            if not np.array_equal(getattr(constraints, name), getattr(static, name)):
                raise ValueError(
                    f"constraints do not match the static identity field {name}"
                )
        if constraints.anchor_particle != static.anchor_particle:
            raise ValueError("constraints anchor does not match the static identity")
        if not isinstance(device, jax.Device):
            raise TypeError("device must be an explicit jax.Device instance")
        if device not in tuple(jax.devices(device.platform)):
            raise ValueError(f"explicit JAX device is not available: {device!r}")

        self.static = static
        self.constraints = constraints
        self.device = device
        self.n_particles = int(static.n_particles)
        self.n_material = int(static.material_count)
        self.n_edges = int(static.n_edges)
        self.n_dofs = int(static.n_dofs)
        self.n_free = int(len(constraints.free_dofs))
        if self.n_free == 0:
            raise ValueError("affine constraints leave no free DOFs")
        if max(self.n_particles, self.n_edges, self.n_dofs) > np.iinfo(np.int32).max:
            raise ValueError("the validated union graph is too large for the int32 JAX index contract")

        coordinates = _require_static_float64(
            static.coordinates, "static.coordinates", (self.n_particles, 2)
        )
        material_phase = _require_static_float64(
            static.union_material_phase_volumes,
            "static.union_material_phase_volumes",
            (self.n_particles,),
        )
        grip_phase = _require_static_float64(
            static.grip_phase_volumes, "static.grip_phase_volumes", (self.n_particles,)
        )
        union_volumes = _require_static_float64(
            static.union_volumes, "static.union_volumes", (self.n_particles,)
        )
        if (
            np.any(material_phase < 0.0)
            or np.any(grip_phase < 0.0)
            or np.any(union_volumes <= 0.0)
        ):
            raise ValueError("static phase/union volumes are invalid")

        source = _require_static_int64(static.source, "static.source", (self.n_edges,))
        target = _require_static_int64(static.target, "static.target", (self.n_edges,))
        for name, values in (("source", source), ("target", target)):
            if np.any(values < 0) or np.any(values >= self.n_particles):
                raise ValueError(f"static.{name} contains an out-of-range index")

        support_indices = _require_static_int64(
            static.support_indices, "static.support_indices"
        )
        support_mask = np.asarray(static.support_mask)
        beta = _require_static_float64(static.beta, "static.beta")
        xi = _require_static_float64(static.xi, "static.xi")
        shape_inverse = _require_static_float64(
            static.shape_inverse, "static.shape_inverse", (self.n_edges, 2, 2)
        )
        if support_indices.ndim != 2 or support_indices.shape[0] != self.n_edges:
            raise ValueError(
                "static.support_indices must have shape (E, S) with the "
                f"accepted edge count E={self.n_edges}"
            )
        if support_mask.dtype != np.dtype(bool) or support_mask.shape != support_indices.shape:
            raise ValueError(
                "static.support_mask must be boolean with the same shape as support_indices"
            )
        if beta.shape != support_indices.shape:
            raise ValueError("static.beta must have the padded support shape")
        if xi.shape != support_indices.shape + (2,):
            raise ValueError("static.xi must have the padded support shape (E, S, 2)")
        if not np.any(support_mask):
            raise ValueError("static.support_mask must contain valid occurrences")

        valid_flat = np.flatnonzero(support_mask.reshape(-1))
        support_flat = support_indices.reshape(-1)[valid_flat]
        if np.any(support_flat < 0) or np.any(support_flat >= self.n_particles):
            raise ValueError(
                "valid static support indices contain an out-of-range particle index"
            )
        occurrence_edge64 = valid_flat // support_indices.shape[1]
        occurrence_particle64 = support_flat
        occurrence_source64 = source[occurrence_edge64]
        occurrence_beta = beta.reshape(-1)[valid_flat]
        occurrence_xi = xi.reshape(-1, 2)[valid_flat]
        if not np.all(np.isfinite(occurrence_beta)) or np.any(occurrence_beta <= 0.0):
            raise ValueError("valid support beta coefficients must be finite and positive")
        if not np.all(np.isfinite(occurrence_xi)):
            raise ValueError("valid support xi coefficients must be finite")

        # The first valid occurrence is the target endpoint in the accepted
        # target-inclusive construction.  Check this here so flattening cannot
        # silently change endpoint ownership.
        if not np.all(support_mask[:, 0]):
            raise ValueError("accepted support_mask must include the target slot")
        if not np.array_equal(support_indices[:, 0], target):
            raise ValueError("accepted support_indices first slot must equal target")

        occurrence_inverse = shape_inverse[occurrence_edge64]
        gradient_coefficient = np.einsum(
            "o,oa,oab->ob",
            occurrence_beta,
            occurrence_xi,
            occurrence_inverse,
            optimize=True,
        )

        lame_lambda = float(static.lame_lambda)
        shear_modulus = float(static.shear_modulus)
        hxx_occ, hxy_occ, hyy_occ = _closed_form_h_components(
            gradient_coefficient, lame_lambda, shear_modulus
        )
        source_coefficient = np.zeros((self.n_edges, 2), dtype=np.float64)
        np.add.at(source_coefficient, occurrence_edge64, gradient_coefficient)
        hxx_source, hxy_source, hyy_source = _closed_form_h_components(
            source_coefficient, lame_lambda, shear_modulus
        )
        eta = _require_static_float64(static.eta, "static.eta", (self.n_edges,))
        if np.any(eta <= 0.0):
            raise ValueError("static.eta must contain strictly positive weights")

        free_dofs = _require_static_int64(constraints.free_dofs, "constraints.free_dofs")
        constrained_dofs = _require_static_int64(
            constraints.constrained_dofs, "constraints.constrained_dofs"
        )
        free_mask = np.zeros((self.n_particles, 2), dtype=bool)
        free_mask.reshape(-1)[free_dofs] = True

        # Every index used by a device gather is range checked as int64 first,
        # then converted to int32.  The valid-occurrence arrays are the only
        # support arrays copied to the backend; padded invalid slots are absent.
        self.coordinates = self._put(coordinates)
        self.material_phase_volumes = self._put(material_phase)
        self.grip_phase_volumes = self._put(grip_phase)
        self.source = self._put(source.astype(np.int32, copy=False))
        self.target = self._put(target.astype(np.int32, copy=False))
        self.eta = self._put(eta)
        self.shape_inverse = self._put(shape_inverse)
        self.occurrence_edge = self._put(occurrence_edge64.astype(np.int32, copy=False))
        self.occurrence_particle = self._put(
            occurrence_particle64.astype(np.int32, copy=False)
        )
        self.occurrence_source = self._put(
            occurrence_source64.astype(np.int32, copy=False)
        )
        self.occurrence_beta = self._put(occurrence_beta)
        self.occurrence_xi = self._put(occurrence_xi)
        self.gradient_coefficient = self._put(gradient_coefficient)
        self.occurrence_hxx = self._put(hxx_occ)
        self.occurrence_hxy = self._put(hxy_occ)
        self.occurrence_hyy = self._put(hyy_occ)
        self.source_hxx = self._put(hxx_source)
        self.source_hxy = self._put(hxy_source)
        self.source_hyy = self._put(hyy_source)
        self.free_dofs = self._put(free_dofs.astype(np.int32, copy=False))
        self.constrained_dofs = self._put(constrained_dofs.astype(np.int32, copy=False))
        self.free_mask = self._put(free_mask)
        self.q_zero = self._put(
            _require_static_float64(constraints.q_zero, "constraints.q_zero", (self.n_dofs,))
        )
        self.load_direction = self._put(
            _require_static_float64(
                constraints.load_direction, "constraints.load_direction", (self.n_dofs,)
            )
        )
        self.outer_top_selector = self._put(
            _require_static_float64(
                constraints.outer_top_selector,
                "constraints.outer_top_selector",
                (self.n_dofs,),
            )
        )
        self.outer_bottom_selector = self._put(
            _require_static_float64(
                constraints.outer_bottom_selector,
                "constraints.outer_bottom_selector",
                (self.n_dofs,),
            )
        )
        self.anchor_selector = self._put(
            _require_static_float64(
                constraints.anchor_selector, "constraints.anchor_selector", (self.n_dofs,)
            )
        )
        self._identity_2 = self._put(np.eye(2, dtype=np.float64))
        self._lame_lambda = self._put(np.asarray(lame_lambda, dtype=np.float64))
        self._shear_modulus = self._put(np.asarray(shear_modulus, dtype=np.float64))
        self._k_res = self._put(np.asarray(float(static.k_res), dtype=np.float64))
        self._grip_degradation = self._put(
            np.asarray(1.0 + float(static.k_res), dtype=np.float64)
        )
        self._material_zeros = self._put(
            np.zeros(self.n_particles - self.n_material, dtype=np.float64)
        )

        self._trace_counts = {"solve": 0}
        self._compiled_solve = jax.jit(self._traced_solve)
        self._compiled_residual = jax.jit(self._residual_public)
        self._compiled_energy = jax.jit(self._energy_public)
        self._compiled_tangent = jax.jit(self._tangent_public)
        self._compiled_blocks = jax.jit(self._block_diagonal_public)
        self._compiled_inverse_blocks = jax.jit(self._block_inverse_public)
        self._compiled_preconditioner = jax.jit(self._preconditioner_public)
        self._compiled_center_weights = jax.jit(self._center_weights)

    def _put(self, values: Any) -> jax.Array:
        return jax.device_put(values, self.device)

    @property
    def trace_count(self) -> int:
        """Number of times the solve function has been traced (cache diagnostic)."""

        return int(self._trace_counts["solve"])

    @property
    def flattened_support_count(self) -> int:
        return int(self.occurrence_edge.shape[0])

    def _prepare_damage(self, damage: Any) -> jax.Array:
        """Validate material damage and its device center weights loudly."""

        values = _require_dynamic_float64(damage, "material damage", (self.n_material,))
        device_values = self._put(values)
        center_weights = self._compiled_center_weights(device_values)
        valid = jnp.all(jnp.isfinite(center_weights)) & jnp.all(center_weights > 0.0)
        if not bool(np.asarray(jax.device_get(valid))):
            raise FloatingPointError(
                "A2 phase center weights are non-finite or non-positive"
            )
        return device_values

    def _center_weights(self, damage: jax.Array) -> jax.Array:
        """Exactly ``Vm*((1-d)^2+k_res)+Vg*(1+k_res)`` on the padded prefix."""

        damage_full = jnp.concatenate((damage, self._material_zeros))
        material_degradation = (1.0 - damage_full) ** 2 + self._k_res
        return (
            self.material_phase_volumes * material_degradation
            + self.grip_phase_volumes * self._grip_degradation
        )

    def _edge_factor(self, damage: jax.Array) -> jax.Array:
        center_weights = self._center_weights(damage)
        return center_weights[self.source] * self.eta

    def _deformation_gradient(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> jax.Array:
        del edge_factor
        values = displacement.reshape(self.n_particles, 2)
        displacement_difference = (
            values[self.occurrence_particle] - values[self.occurrence_source]
        )
        current_bond = self.occurrence_xi + displacement_difference
        moment_occurrence = (
            current_bond[:, :, None]
            * self.occurrence_xi[:, None, :]
            * self.occurrence_beta[:, None, None]
        )
        moment = _segment_sum(
            moment_occurrence, self.occurrence_edge, self.n_edges
        )
        return jnp.einsum("eab,ebc->eac", moment, self.shape_inverse)

    def _strain_stress(
        self, deformation_gradient: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        increment = deformation_gradient - self._identity_2
        strain = 0.5 * (increment + jnp.swapaxes(increment, 1, 2))
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        stress = (
            self._lame_lambda * trace[:, None, None] * self._identity_2[None, :, :]
            + 2.0 * self._shear_modulus * strain
        )
        return strain, stress

    def _assemble_residual(
        self, stress: jax.Array, edge_factor: jax.Array
    ) -> jax.Array:
        stress_occurrence = stress[self.occurrence_edge]
        inverse_transpose = jnp.swapaxes(
            self.shape_inverse[self.occurrence_edge], 1, 2
        )
        endpoint = jnp.einsum(
            "oab,obc,oc->oa",
            stress_occurrence,
            inverse_transpose,
            self.occurrence_xi,
        )
        endpoint = (
            endpoint
            * self.occurrence_beta[:, None]
            * edge_factor[self.occurrence_edge, None]
        )
        support_result = _segment_sum(
            endpoint, self.occurrence_particle, self.n_particles
        )
        source_result = _segment_sum(
            -endpoint, self.occurrence_source, self.n_particles
        )
        return (support_result + source_result).reshape(self.n_dofs)

    def _residual_from_factor(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> jax.Array:
        deformation_gradient = self._deformation_gradient(edge_factor, displacement)
        _, stress = self._strain_stress(deformation_gradient)
        return self._assemble_residual(stress, edge_factor)

    def _energy_from_factor(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> jax.Array:
        deformation_gradient = self._deformation_gradient(edge_factor, displacement)
        strain, _ = self._strain_stress(deformation_gradient)
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        density = (
            0.5 * self._lame_lambda * trace**2
            + self._shear_modulus * jnp.sum(strain * strain, axis=(1, 2))
        )
        return jnp.sum(edge_factor * density)

    def _tangent_from_factor(
        self, edge_factor: jax.Array, direction: jax.Array
    ) -> jax.Array:
        values = direction.reshape(self.n_particles, 2)
        displacement_difference = (
            values[self.occurrence_particle] - values[self.occurrence_source]
        )
        delta_f_occurrence = (
            displacement_difference[:, :, None]
            * self.gradient_coefficient[:, None, :]
        )
        delta_f = _segment_sum(
            delta_f_occurrence, self.occurrence_edge, self.n_edges
        )
        delta_strain = 0.5 * (delta_f + jnp.swapaxes(delta_f, 1, 2))
        trace = delta_strain[:, 0, 0] + delta_strain[:, 1, 1]
        delta_stress = (
            self._lame_lambda * trace[:, None, None] * self._identity_2[None, :, :]
            + 2.0 * self._shear_modulus * delta_strain
        )
        return self._assemble_residual(delta_stress, edge_factor)

    def _evaluate_from_factor(
        self, edge_factor: jax.Array, displacement: jax.Array
    ) -> dict[str, jax.Array]:
        energy = self._energy_from_factor(edge_factor, displacement)
        residual = self._residual_from_factor(edge_factor, displacement)
        residual_matrix = residual.reshape(self.n_particles, 2)
        return {
            "energy": energy,
            "residual": residual,
            "internal_force": -residual,
            "top_reaction": jnp.dot(self.outer_top_selector, residual),
            "bottom_reaction": jnp.dot(self.outer_bottom_selector, residual),
            "anchor_reaction": jnp.dot(self.anchor_selector, residual),
            "total_force": jnp.sum(residual_matrix, axis=0),
            "total_moment": jnp.sum(
                self.coordinates[:, 0] * residual_matrix[:, 1]
                - self.coordinates[:, 1] * residual_matrix[:, 0]
            ),
        }

    def _block_diagonal_from_factor(self, edge_factor: jax.Array) -> jax.Array:
        support_hxx = _segment_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hxx,
            self.occurrence_particle,
            self.n_particles,
        )
        support_hxy = _segment_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hxy,
            self.occurrence_particle,
            self.n_particles,
        )
        support_hyy = _segment_sum(
            edge_factor[self.occurrence_edge] * self.occurrence_hyy,
            self.occurrence_particle,
            self.n_particles,
        )
        source_hxx = _segment_sum(
            edge_factor * self.source_hxx, self.source, self.n_particles
        )
        source_hxy = _segment_sum(
            edge_factor * self.source_hxy, self.source, self.n_particles
        )
        source_hyy = _segment_sum(
            edge_factor * self.source_hyy, self.source, self.n_particles
        )
        hxx = support_hxx + source_hxx
        hxy = support_hxy + source_hxy
        hyy = support_hyy + source_hyy
        return jnp.stack(
            (
                jnp.stack((hxx, hxy), axis=-1),
                jnp.stack((hxy, hyy), axis=-1),
            ),
            axis=-2,
        )

    def _block_inverse_from_factor(self, edge_factor: jax.Array) -> jax.Array:
        blocks = self._block_diagonal_from_factor(edge_factor)
        hxx = blocks[:, 0, 0]
        hxy = blocks[:, 0, 1]
        hyy = blocks[:, 1, 1]
        determinant = hxx * hyy - hxy * hxy
        free_x = self.free_mask[:, 0]
        free_y = self.free_mask[:, 1]
        both_free = free_x & free_y
        only_x_free = free_x & ~free_y
        only_y_free = ~free_x & free_y

        # The full two-free inverse is the closed form adjugate/determinant.
        # One-free particles use the exact scalar diagonal inverse of the
        # eliminated sub-block, and no-free particles contribute zero.  The
        # masks are structural and never introduce a shift or clipping.
        inverse_xx = jnp.where(
            both_free,
            hyy / determinant,
            jnp.where(only_x_free, 1.0 / hxx, 0.0),
        )
        inverse_xy = jnp.where(both_free, -hxy / determinant, 0.0)
        inverse_yy = jnp.where(
            both_free,
            hxx / determinant,
            jnp.where(only_y_free, 1.0 / hyy, 0.0),
        )
        return jnp.stack(
            (
                jnp.stack((inverse_xx, inverse_xy), axis=-1),
                jnp.stack((inverse_xy, inverse_yy), axis=-1),
            ),
            axis=-2,
        )

    def _preconditioner_from_inverse(
        self, vector: jax.Array, inverse_blocks: jax.Array
    ) -> jax.Array:
        full = jnp.zeros((self.n_dofs,), dtype=jnp.float64).at[self.free_dofs].set(
            vector
        )
        particle = full.reshape(self.n_particles, 2)
        result = jnp.einsum("nij,nj->ni", inverse_blocks, particle)
        return result.reshape(self.n_dofs)[self.free_dofs]

    def _free_tangent_action(
        self, edge_factor: jax.Array, vector: jax.Array
    ) -> jax.Array:
        full = jnp.zeros((self.n_dofs,), dtype=jnp.float64).at[self.free_dofs].set(
            vector
        )
        return self._tangent_from_factor(edge_factor, full)[self.free_dofs]

    def _pcg(
        self,
        edge_factor: jax.Array,
        rhs: jax.Array,
        initial_values: jax.Array,
        initial_residual: jax.Array,
        convergence_limit: jax.Array,
        maxiter: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        inverse_blocks = self._block_inverse_from_factor(edge_factor)

        def precondition(vector: jax.Array) -> jax.Array:
            return self._preconditioner_from_inverse(vector, inverse_blocks)

        residual = initial_residual
        z = precondition(residual)
        direction = z
        rho = jnp.dot(residual, z)
        residual_norm = jnp.linalg.norm(residual)
        finite_initial = (
            jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(z))
            & jnp.isfinite(rho)
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(convergence_limit)
            & jnp.all(jnp.isfinite(inverse_blocks))
        )
        initially_converged = finite_initial & (residual_norm < convergence_limit)
        initially_breakdown = finite_initial & (~initially_converged) & (rho <= 0.0)
        initial_status = jnp.where(
            ~finite_initial,
            _STATUS_NONFINITE,
            jnp.where(
                initially_converged,
                _STATUS_CONVERGED,
                jnp.where(initially_breakdown, _STATUS_BREAKDOWN, _STATUS_RUNNING),
            ),
        ).astype(jnp.int32)

        def condition(state: tuple[jax.Array, ...]) -> jax.Array:
            _, _, _, _, _, iteration, status = state
            return (status == _STATUS_RUNNING) & (iteration < maxiter)

        def body(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
            values, residual, z, direction, rho, iteration, status = state
            action = self._free_tangent_action(edge_factor, direction)
            denominator = jnp.dot(direction, action)
            finite_step = (
                jnp.all(jnp.isfinite(action))
                & jnp.isfinite(denominator)
                & jnp.isfinite(rho)
                & (denominator > 0.0)
                & (rho > 0.0)
            )

            def invalid_step(_: None) -> tuple[jax.Array, ...]:
                finite_values = (
                    jnp.all(jnp.isfinite(action))
                    & jnp.isfinite(denominator)
                    & jnp.isfinite(rho)
                )
                invalid_status = jnp.where(
                    finite_values, _STATUS_BREAKDOWN, _STATUS_NONFINITE
                ).astype(jnp.int32)
                return values, residual, z, direction, rho, iteration, invalid_status

            def valid_step(_: None) -> tuple[jax.Array, ...]:
                alpha = rho / denominator
                new_values = values + alpha * direction
                new_residual = residual - alpha * action
                new_z = precondition(new_residual)
                new_rho = jnp.dot(new_residual, new_z)
                new_norm = jnp.linalg.norm(new_residual)
                finite_update = (
                    jnp.all(jnp.isfinite(new_values))
                    & jnp.all(jnp.isfinite(new_residual))
                    & jnp.all(jnp.isfinite(new_z))
                    & jnp.isfinite(new_rho)
                    & jnp.isfinite(new_norm)
                )
                converged = finite_update & (new_norm < convergence_limit)
                positive_next = finite_update & (new_rho > 0.0)
                beta = new_rho / rho
                candidate_direction = new_z + beta * direction
                continue_update = finite_update & (~converged) & positive_next
                next_direction = jnp.where(
                    continue_update, candidate_direction, direction
                )
                next_z = jnp.where(continue_update, new_z, z)
                next_rho = jnp.where(continue_update, new_rho, rho)
                next_status = jnp.where(
                    ~finite_update,
                    _STATUS_NONFINITE,
                    jnp.where(
                        converged,
                        _STATUS_CONVERGED,
                        jnp.where(positive_next, _STATUS_RUNNING, _STATUS_BREAKDOWN),
                    ),
                ).astype(jnp.int32)
                return (
                    new_values,
                    new_residual,
                    next_z,
                    next_direction,
                    next_rho,
                    iteration + jnp.int32(1),
                    next_status,
                )

            return jax.lax.cond(finite_step, valid_step, invalid_step, operand=None)

        final_state = jax.lax.while_loop(
            condition,
            body,
            (
                initial_values,
                residual,
                z,
                direction,
                rho,
                jnp.int32(0),
                initial_status,
            ),
        )
        values, residual, _, _, _, iterations, status = final_state
        status = jnp.where(
            status == _STATUS_RUNNING, _STATUS_MAXITER, status
        ).astype(jnp.int32)
        return values, residual, iterations, status

    def _traced_solve(
        self,
        damage: jax.Array,
        load: jax.Array,
        relative_tolerance: jax.Array,
        absolute_tolerance: jax.Array,
        maxiter: jax.Array,
        supplied: jax.Array,
        guess: jax.Array,
    ) -> dict[str, jax.Array]:
        self._trace_counts["solve"] += 1
        edge_factor = self._edge_factor(damage)
        prescribed = self.q_zero + load * self.load_direction
        prescribed_residual = self._residual_from_factor(edge_factor, prescribed)
        free_rhs = -prescribed_residual[self.free_dofs]
        free_rhs_norm = jnp.linalg.norm(free_rhs)
        convergence_limit = jnp.maximum(
            absolute_tolerance, relative_tolerance * free_rhs_norm
        )

        initial_full = jnp.where(supplied, guess, prescribed)
        initial_free_residual = self._residual_from_factor(edge_factor, initial_full)[
            self.free_dofs
        ]
        # The public residual follows the CPU energy-gradient sign.  PCG's
        # Krylov residual is ``rhs - A @ x`` and therefore has the opposite
        # sign for a supplied nonzero initial free field.
        initial_krylov_residual = jnp.where(
            supplied, -initial_free_residual, free_rhs
        )
        initial_free_residual_norm = jnp.linalg.norm(initial_free_residual)
        initial_values = jnp.where(
            supplied,
            initial_full[self.free_dofs],
            jnp.zeros((self.n_free,), dtype=jnp.float64),
        )

        free_values, recursive_residual, iterations, solver_status = self._pcg(
            edge_factor,
            free_rhs,
            initial_values,
            initial_krylov_residual,
            convergence_limit,
            maxiter,
        )
        displacement = prescribed.at[self.free_dofs].set(free_values)
        evaluation = self._evaluate_from_factor(edge_factor, displacement)
        explicit_free_residual = evaluation["residual"][self.free_dofs]
        explicit_norm = jnp.linalg.norm(explicit_free_residual)
        recursive_norm = jnp.linalg.norm(recursive_residual)
        finite_final = (
            jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(recursive_residual))
            & jnp.all(jnp.isfinite(explicit_free_residual))
            & jnp.isfinite(explicit_norm)
            & jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(value))
                        for value in evaluation.values()
                    ]
                )
            )
            & jnp.isfinite(convergence_limit)
        )
        explicit_converged = finite_final & (explicit_norm < convergence_limit)
        final_status = jnp.where(
            ~finite_final,
            _STATUS_NONFINITE,
            jnp.where(
                (solver_status == _STATUS_CONVERGED) & (~explicit_converged),
                _STATUS_EXPLICIT_RESIDUAL,
                solver_status,
            ),
        ).astype(jnp.int32)
        converged = final_status == _STATUS_CONVERGED
        relative_norm = explicit_norm / jnp.maximum(free_rhs_norm, 1.0e-300)
        cg_info = jnp.where(
            final_status == _STATUS_CONVERGED,
            jnp.int32(0),
            jnp.where(final_status == _STATUS_MAXITER, iterations, jnp.int32(-1)),
        ).astype(jnp.int32)
        return {
            "displacement": displacement,
            "free_rhs": free_rhs,
            "free_residual": explicit_free_residual,
            "recursive_free_residual": recursive_residual,
            "initial_displacement": initial_full,
            "initial_free_residual": initial_free_residual,
            "free_rhs_norm": free_rhs_norm,
            "free_residual_norm": explicit_norm,
            "recursive_residual_norm": recursive_norm,
            "free_residual_relative_norm": relative_norm,
            "initial_free_residual_norm": initial_free_residual_norm,
            "cg_convergence_limit": convergence_limit,
            "initial_guess_supplied": supplied,
            "cg_iterations": iterations,
            "cg_info": cg_info,
            "status_code": final_status,
            "explicit_converged": explicit_converged,
            "converged": converged,
            "u_bar": load,
            "cg_rtol": relative_tolerance,
            "cg_atol": absolute_tolerance,
            "cg_maxiter": maxiter,
            **evaluation,
        }

    def _residual_public(self, damage: jax.Array, displacement: jax.Array) -> jax.Array:
        return self._residual_from_factor(self._edge_factor(damage), displacement)

    def _energy_public(self, damage: jax.Array, displacement: jax.Array) -> jax.Array:
        return self._energy_from_factor(self._edge_factor(damage), displacement)

    def _tangent_public(self, damage: jax.Array, direction: jax.Array) -> jax.Array:
        return self._tangent_from_factor(self._edge_factor(damage), direction)

    def _block_diagonal_public(self, damage: jax.Array) -> jax.Array:
        return self._block_diagonal_from_factor(self._edge_factor(damage))

    def _block_inverse_public(self, damage: jax.Array) -> jax.Array:
        return self._block_inverse_from_factor(self._edge_factor(damage))

    def _preconditioner_public(self, damage: jax.Array, vector: jax.Array) -> jax.Array:
        inverse_blocks = self._block_inverse_from_factor(self._edge_factor(damage))
        return self._preconditioner_from_inverse(vector, inverse_blocks)

    def center_weights(self, damage: Any) -> jax.Array:
        """Return the exact phase center weights for a material damage field."""

        _require_x64_enabled()
        result = self._compiled_center_weights(
            self._put(
                _require_dynamic_float64(damage, "material damage", (self.n_material,))
            )
        )
        _require_finite_device_result(result, "JAX A2 center weights")
        return result

    def residual(self, damage: Any, displacement: Any) -> jax.Array:
        _require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        displacement_array = _require_dynamic_float64(
            displacement, "displacement", (self.n_dofs,)
        )
        result = self._compiled_residual(damage_device, self._put(displacement_array))
        _require_finite_device_result(result, "JAX A2 residual")
        return result

    def energy(self, damage: Any, displacement: Any) -> jax.Array:
        _require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        displacement_array = _require_dynamic_float64(
            displacement, "displacement", (self.n_dofs,)
        )
        result = self._compiled_energy(damage_device, self._put(displacement_array))
        _require_finite_device_result(result, "JAX A2 energy")
        return result

    def tangent_action(self, damage: Any, direction: Any) -> jax.Array:
        """Apply the analytic flattened plane-strain Hessian action."""

        _require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        direction_array = _require_dynamic_float64(direction, "direction", (self.n_dofs,))
        result = self._compiled_tangent(damage_device, self._put(direction_array))
        _require_finite_device_result(result, "JAX A2 tangent action")
        return result

    def tangent_block_diagonal(self, damage: Any) -> jax.Array:
        """Return exact particle-local tangent blocks for fixed damage."""

        _require_x64_enabled()
        result = self._compiled_blocks(self._prepare_damage(damage))
        _require_finite_device_result(result, "JAX A2 tangent block diagonal")
        return result

    def block_jacobi_inverse(self, damage: Any) -> jax.Array:
        """Return the exact masked closed-form block-Jacobi inverse."""

        _require_x64_enabled()
        result = self._compiled_inverse_blocks(self._prepare_damage(damage))
        _require_finite_device_result(result, "JAX A2 block-Jacobi inverse")
        return result

    def apply_block_jacobi(self, damage: Any, vector: Any) -> jax.Array:
        """Apply the exact masked block-Jacobi preconditioner to a free vector."""

        _require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        vector_array = _require_dynamic_float64(vector, "free vector", (self.n_free,))
        result = self._compiled_preconditioner(damage_device, self._put(vector_array))
        _require_finite_device_result(result, "JAX A2 block-Jacobi application")
        return result

    def solve(
        self,
        damage: Any,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
    ) -> JaxA2SolveResult:
        """Run device-resident fixed-damage affine-constraint PCG.

        A non-converged numerical solve is returned with an explicit status so
        callers can inspect all device diagnostics.  ``require_converged`` or
        ``solve_strict`` supplies the raising policy when that is desired.
        """

        _require_x64_enabled()
        damage_device = self._prepare_damage(damage)
        load = _dynamic_float_scalar(u_bar, "u_bar", allow_zero=True, nonnegative=False)
        relative_tolerance = _dynamic_float_scalar(
            cg_rtol, "cg_rtol", allow_zero=True
        )
        absolute_tolerance = _dynamic_float_scalar(
            cg_atol, "cg_atol", allow_zero=True
        )
        if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
            raise ValueError("cg_rtol and cg_atol cannot both be zero")
        maxiter = _dynamic_maxiter(cg_maxiter)

        prescribed = np.asarray(
            self.constraints.prescribed_vector(load), dtype=np.float64
        )
        if initial_guess is None:
            supplied = False
            guess_array = prescribed
        else:
            supplied = True
            guess_array = np.asarray(
                _require_dynamic_float64(initial_guess, "initial_guess", (self.n_dofs,)),
                dtype=np.float64,
            )
            if not np.array_equal(
                guess_array[self.constraints.constrained_dofs],
                prescribed[self.constraints.constrained_dofs],
            ):
                raise ValueError(
                    "initial_guess constrained values must exactly match the "
                    "prescribed displacement at u_bar"
                )

        values = self._compiled_solve(
            damage_device,
            self._put(np.asarray(load, dtype=np.float64)),
            self._put(np.asarray(relative_tolerance, dtype=np.float64)),
            self._put(np.asarray(absolute_tolerance, dtype=np.float64)),
            self._put(np.asarray(maxiter, dtype=np.int32)),
            self._put(np.asarray(supplied, dtype=bool)),
            self._put(guess_array),
        )
        return JaxA2SolveResult(
            displacement=values["displacement"],
            damage=damage_device,
            u_bar=values["u_bar"],
            cg_rtol=values["cg_rtol"],
            cg_atol=values["cg_atol"],
            cg_maxiter=values["cg_maxiter"],
            free_rhs=values["free_rhs"],
            free_residual=values["free_residual"],
            recursive_free_residual=values["recursive_free_residual"],
            initial_displacement=values["initial_displacement"],
            initial_free_residual=values["initial_free_residual"],
            free_rhs_norm=values["free_rhs_norm"],
            free_residual_norm=values["free_residual_norm"],
            recursive_residual_norm=values["recursive_residual_norm"],
            free_residual_relative_norm=values["free_residual_relative_norm"],
            initial_free_residual_norm=values["initial_free_residual_norm"],
            cg_convergence_limit=values["cg_convergence_limit"],
            initial_guess_supplied=values["initial_guess_supplied"],
            cg_iterations=values["cg_iterations"],
            cg_info=values["cg_info"],
            status_code=values["status_code"],
            explicit_converged=values["explicit_converged"],
            converged_flag=values["converged"],
            energy=values["energy"],
            top_reaction=values["top_reaction"],
            bottom_reaction=values["bottom_reaction"],
            anchor_reaction=values["anchor_reaction"],
            total_force=values["total_force"],
            total_moment=values["total_moment"],
        )

    def solve_strict(self, *args: Any, **kwargs: Any) -> JaxA2SolveResult:
        result = self.solve(*args, **kwargs)
        return result.require_converged()

    def replay_cpu(
        self,
        jax_result: JaxA2SolveResult,
        damage: Any,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        seed_kind: str = "direct_load",
        source_state: str = "direct",
        source_step: int = 0,
        source_sweep: int = 0,
        source_damage_digest: str | None = None,
    ) -> JaxA2CPUReplayResult:
        """Replay one device displacement through a fresh CPU A2 view.

        The transferred displacement is the only GPU-produced field.  Energy,
        residual, reactions, force, moment, the explicit free-residual gate,
        and the seed provenance are recomputed independently on the CPU
        without rerunning CPU CG.  Only ``cpu_result`` is authoritative.
        """

        if not isinstance(jax_result, JaxA2SolveResult):
            raise TypeError("jax_result must be a JaxA2SolveResult instance")
        _require_x64_enabled()
        damage_array = np.asarray(
            _require_dynamic_float64(damage, "material damage", (self.n_material,)),
            dtype=np.float64,
        )
        load = float(
            _dynamic_float_scalar(u_bar, "u_bar", allow_zero=True, nonnegative=False)
        )
        relative_tolerance = float(
            _dynamic_float_scalar(cg_rtol, "cg_rtol", allow_zero=True)
        )
        absolute_tolerance = float(
            _dynamic_float_scalar(cg_atol, "cg_atol", allow_zero=True)
        )
        if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
            raise ValueError("cg_rtol and cg_atol cannot both be zero")
        maxiter = int(_dynamic_maxiter(cg_maxiter))

        (
            status_code,
            displacement,
            result_damage,
            result_load,
            result_rtol,
            result_atol,
            result_maxiter,
            result_initial_displacement,
            gpu_iterations,
            result_initial_guess_supplied,
            result_cg_info,
        ) = jax.device_get(
            (
                jax_result.status_code,
                jax_result.displacement,
                jax_result.damage,
                jax_result.u_bar,
                jax_result.cg_rtol,
                jax_result.cg_atol,
                jax_result.cg_maxiter,
                jax_result.initial_displacement,
                jax_result.cg_iterations,
                jax_result.initial_guess_supplied,
                jax_result.cg_info,
            )
        )
        result_damage = np.asarray(
            _require_dynamic_float64(result_damage, "jax_result.damage", (self.n_material,)),
            dtype=np.float64,
        )
        displacement = np.asarray(
            _require_dynamic_float64(
                displacement, "jax_result.displacement", (self.n_dofs,)
            ),
            dtype=np.float64,
        )
        result_initial_displacement = np.asarray(
            _require_dynamic_float64(
                result_initial_displacement,
                "jax_result.initial_displacement",
                (self.n_dofs,),
            ),
            dtype=np.float64,
        )
        for name, value in (
            ("u_bar", result_load),
            ("cg_rtol", result_rtol),
            ("cg_atol", result_atol),
        ):
            _require_dynamic_float64(value, f"jax_result.{name}", ())
        if not np.array_equal(result_damage, damage_array):
            raise ValueError("jax_result damage does not match the CPU replay damage")
        for name, actual, expected in (
            ("u_bar", float(result_load), load),
            ("cg_rtol", float(result_rtol), relative_tolerance),
            ("cg_atol", float(result_atol), absolute_tolerance),
        ):
            if actual != expected:
                raise ValueError(f"jax_result {name} does not match the CPU replay value")
        if int(result_maxiter) != maxiter:
            raise ValueError(
                "jax_result cg_maxiter does not match the CPU replay value"
            )
        if int(status_code) != int(_STATUS_CONVERGED):
            raise RuntimeError(
                "cannot replay a non-converged JAX A2 mechanics result: "
                f"status={_STATUS_NAMES.get(int(status_code), str(int(status_code)))}"
            )
        if int(gpu_iterations) > maxiter:
            raise RuntimeError(
                "GPU iteration count exceeds the configured CG limit: "
                f"iterations={int(gpu_iterations)}, maxiter={maxiter}"
            )
        if int(result_cg_info) != 0:
            raise RuntimeError("converged jax_result must have cg_info=0")

        # This is intentionally a fresh CPU view.  The immutable fixed-damage
        # reference owns its own phase weights and tangent caches; nothing is
        # reused from the device execution.
        view = A2FixedDamageView(self.static, damage_array)
        constraints = self.constraints
        prescribed = constraints.prescribed_vector(load)
        if not np.array_equal(
            displacement[constraints.constrained_dofs],
            prescribed[constraints.constrained_dofs],
        ):
            raise RuntimeError(
                "GPU displacement violates the exact CPU affine-constraint values"
            )
        free = constraints.free_dofs
        evaluation = view.evaluate(displacement)
        prescribed_residual = view.residual(prescribed)
        rhs = -prescribed_residual[free]
        rhs_norm = float(np.linalg.norm(rhs))
        convergence_limit = max(absolute_tolerance, relative_tolerance * rhs_norm)
        free_residual = np.asarray(evaluation.residual[free], dtype=np.float64)
        free_residual_norm = float(np.linalg.norm(free_residual))
        if not np.isfinite(free_residual_norm) or free_residual_norm >= convergence_limit:
            raise RuntimeError(
                "GPU displacement fails the strict CPU CG gate: "
                f"norm={free_residual_norm:.17g}, allowed={convergence_limit:.17g}"
            )

        if initial_guess is None:
            if bool(result_initial_guess_supplied):
                raise ValueError(
                    "jax_result initial_guess_supplied does not match the CPU replay seed policy"
                )
            initial_guess_supplied = False
            seed = _immutable(prescribed)
        else:
            if not bool(result_initial_guess_supplied):
                raise ValueError(
                    "jax_result initial_guess_supplied does not match the CPU replay seed policy"
                )
            initial_guess_supplied = True
            seed = np.asarray(
                _require_dynamic_float64(initial_guess, "initial_guess", (self.n_dofs,)),
                dtype=np.float64,
            )
            _check_exact_constraints(seed, constraints, load, "A2 initial_guess")
            seed = _immutable(seed)
        if not np.array_equal(result_initial_displacement, np.asarray(seed)):
            raise ValueError(
                "jax_result initial_displacement does not match the CPU replay seed"
            )
        initial_residual_norm = float(np.linalg.norm(view.residual(np.asarray(seed))[free]))
        residual_matrix = evaluation.residual.reshape(view.n_particles, 2)
        total_force = np.sum(residual_matrix, axis=0)
        total_moment = float(
            np.sum(
                view.coordinates[:, 0] * residual_matrix[:, 1]
                - view.coordinates[:, 1] * residual_matrix[:, 0]
            )
        )
        if not np.all(np.isfinite(total_force)) or not np.isfinite(total_moment):
            raise FloatingPointError("A2 force/moment diagnostics are non-finite")
        if source_damage_digest is None:
            source_damage_digest = view.damage_digest
        seed_provenance = _seed_record(
            seed_kind,
            np.asarray(seed),
            view,
            constraints,
            initial_residual_norm,
            free_residual_norm,
            source_state=source_state,
            source_step=source_step,
            source_sweep=source_sweep,
            source_damage_digest=source_damage_digest,
        )
        cpu_result = A2MechanicsSolveResult(
            displacement=displacement,
            evaluation=evaluation,
            constraints=constraints,
            outer_displacement=load,
            free_residual=free_residual,
            free_residual_norm=free_residual_norm,
            free_rhs_norm=rhs_norm,
            free_residual_relative_norm=free_residual_norm / max(rhs_norm, 1.0e-300),
            cg_convergence_limit=convergence_limit,
            cg_iterations=int(gpu_iterations),
            cg_info=0,
            outer_reaction=float(constraints.outer_top_selector @ evaluation.residual),
            bottom_reaction=float(constraints.outer_bottom_selector @ evaluation.residual),
            anchor_reaction=float(constraints.anchor_selector @ evaluation.residual),
            total_force=total_force,
            total_moment=total_moment,
            static_runtime_identity=view.static.runtime_identity,
            static_digest=view.static.deterministic_digest,
            damage_runtime_identity=view.material_damage_runtime_identity,
            damage_digest=view.damage_digest,
            unit_system_digest=_unit_system_digest(
                view.static.deterministic_digest, view.damage_digest
            ),
            seed_provenance=seed_provenance,
        )
        return JaxA2CPUReplayResult(jax_result=jax_result, cpu_result=cpu_result)


def replay_jax_a2_solution_on_cpu(
    backend: JaxA2UnionMechanics,
    jax_result: JaxA2SolveResult,
    damage: Any,
    u_bar: Any,
    cg_rtol: Any,
    cg_atol: Any,
    cg_maxiter: Any,
    *,
    initial_guess: Any | None = None,
    seed_kind: str = "direct_load",
    source_state: str = "direct",
    source_step: int = 0,
    source_sweep: int = 0,
    source_damage_digest: str | None = None,
) -> JaxA2CPUReplayResult:
    """Explicit functional spelling of :meth:`JaxA2UnionMechanics.replay_cpu`."""

    if not isinstance(backend, JaxA2UnionMechanics):
        raise TypeError("backend must be a JaxA2UnionMechanics instance")
    return backend.replay_cpu(
        jax_result,
        damage,
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


@dataclass(frozen=True)
class JaxA2CPUReplayAdapter:
    """Return CPU-authoritative A2 mechanics results from the JAX backend.

    This is the only restricted adapter accepted by the engineering-grip
    controller.  It serves the work-check center/plus/minus solves and every
    stagger solve from one device backend, so a single fixed-shape compiled
    solve is reused for all changing damage/load/seed/tolerance inputs.
    """

    backend: JaxA2UnionMechanics

    def __post_init__(self) -> None:
        if not isinstance(self.backend, JaxA2UnionMechanics):
            raise TypeError("backend must be a JaxA2UnionMechanics instance")

    @classmethod
    def from_static(
        cls,
        static: A2StaticMechanics,
        constraints: AffineConstraints,
        device: jax.Device,
    ) -> "JaxA2CPUReplayAdapter":
        """Build the adapter from an accepted static model and constraints."""

        return cls(JaxA2UnionMechanics(static, constraints, device))

    @property
    def cpu_static(self) -> A2StaticMechanics:
        return self.backend.static

    @property
    def constraints(self) -> AffineConstraints:
        return self.backend.constraints

    @property
    def mechanics_backend(self) -> str:
        return f"jax_{self.backend.device.platform}_cpu_replay"

    @property
    def device(self) -> jax.Device:
        return self.backend.device

    def ensure_view(self, view: A2FixedDamageView, constraints: AffineConstraints) -> None:
        """Require exact static/constraint identity before any adapter solve."""

        if not isinstance(view, A2FixedDamageView):
            raise TypeError("view must be an A2FixedDamageView")
        if not isinstance(constraints, AffineConstraints):
            raise TypeError("constraints must be an AffineConstraints")
        if view.static is not self.backend.static:
            raise ValueError(
                "mechanics adapter static identity does not match the solve view"
            )
        _validate_view_constraints(view, self.backend.constraints)
        if constraints is not self.backend.constraints:
            raise ValueError(
                "mechanics adapter constraints are not the exact bound object"
            )

    def solve_pair(
        self,
        view: A2FixedDamageView,
        constraints: AffineConstraints,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        seed_kind: str = "direct_load",
        source_state: str = "direct",
        source_step: int = 0,
        source_sweep: int = 0,
        source_damage_digest: str | None = None,
    ) -> JaxA2CPUReplayResult:
        """Run the device solve and return the CPU-authoritative replay pair."""

        self.ensure_view(view, constraints)
        load = _scalar(u_bar, "u_bar")
        damage = np.asarray(view.material_damage, dtype=np.float64)
        jax_result = self.backend.solve_strict(
            damage,
            load,
            cg_rtol,
            cg_atol,
            cg_maxiter,
            initial_guess=initial_guess,
        )
        replay = self.backend.replay_cpu(
            jax_result,
            damage,
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
        _check_exact_constraints(
            replay.cpu_result.displacement, constraints, load, "direct solve"
        )
        if replay.cpu_result.U_ext != load:
            raise ValueError("direct mechanics solve changed u_bar")
        return replay

    def solve_fixed_outer_displacement(
        self,
        view: A2FixedDamageView,
        constraints: AffineConstraints,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        seed_kind: str = "direct_load",
        source_state: str = "direct",
        source_step: int = 0,
        source_sweep: int = 0,
        source_damage_digest: str | None = None,
    ) -> A2MechanicsSolveResult:
        """CPU-authoritative drop-in for the direct engineering-grip solve."""

        return self.solve_pair(
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
        ).cpu_result
