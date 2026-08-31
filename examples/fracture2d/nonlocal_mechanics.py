"""Independent intact bond-associated mechanics for fracture2d.

This module is the Stage 3 direct baseline.  It intentionally does not import
the conventional :class:`pinn.operators.ElasticD`, the local FEniCS solver, or
any JAX code.  Geometry, quadrature, and the reciprocal graph are supplied by
the caller and are kept in NumPy ``float64`` arrays.

The implementation follows Sections 0.3--0.5 and 0.7 of
``docs/tasks/fracture2d-direct-nonlocal.md`` exactly:

* every directed edge uses the target-inclusive support ``{j}`` plus sorted
  common neighbours;
* the target beta is ``omega_ij * V_j`` and common-neighbour betas are
  ``omega_ik * V_k``;
* the intact degradation is the constant ``g = 1 + k_res``;
* the residual is assembled explicitly from
  ``P @ K^{-T} @ xi`` and endpoint source/target signs;
* the tangent action is evaluated independently through ``delta F`` and the
  plane-strain constitutive map; and
* hard-boundary reactions use the residual-gradient sign convention.

The graph input is normalized to lexicographic ``(source, target)`` order
after reciprocal, self-edge, range, and strict-horizon validation.  The
normalization is deterministic and is part of this module's public contract.
No dense global tangent is assembled by production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


__all__ = [
    "BondAssociatedMechanics",
    "FixedDamageBondAssociatedMechanics",
    "HardBoundaryConstraints",
    "IntactBondAssociatedMechanics",
    "MechanicsEvaluation",
    "MechanicsSolveResult",
]


_FLOAT64 = np.dtype(np.float64)
_MECHANICS_BACKEND_TOKENS = {
    "numpy_scipy": object(),
    "jax_cpu_cpu_replay": object(),
    "jax_gpu_cpu_replay": object(),
    "jax_tpu_cpu_replay": object(),
}


def _mechanics_backend_token(name: str) -> object:
    try:
        return _MECHANICS_BACKEND_TOKENS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unrecognized mechanics backend {name!r}") from error
_INT64 = np.dtype(np.int64)
_IDENTITY_2 = np.eye(2, dtype=np.float64)


def _readonly_copy(values: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    """Copy an array into immutable bytes-backed storage.

    A normal read-only ndarray can be made writable again with
    ``setflags(write=True)`` when it owns a writable allocation.  Mechanics
    data can be retained by accepted continuation records, so the immutable
    copy must also make that NumPy operation fail at the buffer boundary.
    """

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    result = result.reshape(source.shape)
    result.setflags(write=False)
    return result


def _scalar_float(value: Any, name: str) -> float:
    """Coerce one scalar numeric value without accepting array-shaped input."""

    raw = np.asarray(value)
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {raw.shape}")
    try:
        result = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a scalar number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _require_float64_array(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Validate an exact float64 array before making the fixed copy."""

    array = np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _require_integer_array(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Validate an integer connectivity array without silently truncating it."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _validate_displacement(values: Any, n_dofs: int, name: str) -> np.ndarray:
    """Enforce the public flat float64 displacement/direction contract."""

    array = np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if array.shape != (n_dofs,):
        raise ValueError(f"{name} must have flat shape ({n_dofs},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_tolerance(value: Any, name: str, *, allow_zero: bool) -> float:
    result = _scalar_float(value, name)
    if allow_zero:
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative and finite, got {value!r}")
    elif result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result


@dataclass(frozen=True)
class MechanicsEvaluation:
    """Response fields at one mechanics displacement state.

    ``edge_stress`` and ``particle_stress`` are the undegraded constitutive
    plane-strain stress ``P`` from Section 0.4.  The corresponding effective
    stress fields are exposed separately as ``edge_effective_stress`` and
    ``particle_effective_stress``; they equal the view's immutable degradation
    factor times ``P``.
    """

    energy: float
    residual: np.ndarray
    internal_force: np.ndarray
    edge_deformation_gradient: np.ndarray
    edge_strain: np.ndarray
    edge_stress: np.ndarray
    particle_strain: np.ndarray
    particle_stress: np.ndarray
    edge_effective_stress: np.ndarray
    particle_effective_stress: np.ndarray
    edge_energy_density: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", float(self.energy))
        array_names = (
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
        for name in array_names:
            array = np.asarray(getattr(self, name))
            if array.dtype != _FLOAT64:
                raise ValueError(f"MechanicsEvaluation.{name} must use float64")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"MechanicsEvaluation.{name} must be finite")
            object.__setattr__(self, name, _readonly_copy(array))
        if not np.isfinite(self.energy):
            raise ValueError("MechanicsEvaluation.energy must be finite")

    @property
    def edge_F(self) -> np.ndarray:
        """Alias for the edge deformation gradients."""

        return self.edge_deformation_gradient

    @property
    def strain(self) -> np.ndarray:
        """Alias for source-centered particle strain."""

        return self.particle_strain

    @property
    def stress(self) -> np.ndarray:
        """Alias for source-centered undegraded constitutive stress."""

        return self.particle_stress

    def __getitem__(self, key: str) -> Any:
        """Provide lightweight mapping-style access for diagnostics."""

        aliases = {
            "edge_F": "edge_deformation_gradient",
            "F": "edge_deformation_gradient",
            "strain": "particle_strain",
            "stress": "particle_stress",
        }
        return getattr(self, aliases.get(key, key))


@dataclass(frozen=True)
class HardBoundaryConstraints:
    """Immutable material-particle hard constraints for the Stage 3 BVP.

    The prescribed values are ``uy=0`` on ``bottom_mask``, ``uy=u_bar`` on
    ``top_mask``, and ``ux=0`` at ``anchor_particle``.  The anchor is required
    to be a bottom-boundary particle, matching the frozen lower-left gauge.
    All other horizontal boundary degrees of freedom remain free.
    """

    bottom_mask: np.ndarray
    top_mask: np.ndarray
    anchor_particle: int
    constrained_dofs: np.ndarray
    free_dofs: np.ndarray
    q_zero: np.ndarray
    a_top: np.ndarray
    a_bottom: np.ndarray
    a_anchor: np.ndarray

    def __post_init__(self) -> None:
        bottom = np.asarray(self.bottom_mask)
        top = np.asarray(self.top_mask)
        if bottom.dtype != np.dtype(bool) or top.dtype != np.dtype(bool):
            raise ValueError("bottom_mask and top_mask must use boolean dtype")
        if bottom.ndim != 1 or top.ndim != 1 or bottom.shape != top.shape:
            raise ValueError("bottom_mask and top_mask must have the same shape (N,)")
        if len(bottom) == 0:
            raise ValueError("Boundary masks must contain at least one particle")
        if not np.any(bottom) or not np.any(top):
            raise ValueError("bottom_mask and top_mask must each contain particles")
        if np.any(bottom & top):
            raise ValueError("bottom_mask and top_mask must not overlap")
        if not isinstance(self.anchor_particle, Integral) or isinstance(
            self.anchor_particle, (bool, np.bool_)
        ):
            raise ValueError("anchor_particle must be an integer particle index")
        anchor = int(self.anchor_particle)
        if anchor < 0 or anchor >= len(bottom):
            raise ValueError("anchor_particle is outside the particle range")
        if not bottom[anchor]:
            raise ValueError("anchor_particle must belong to bottom_mask")

        n_dofs = 2 * len(bottom)
        constrained = np.asarray(self.constrained_dofs)
        free = np.asarray(self.free_dofs)
        for name, array in (("constrained_dofs", constrained), ("free_dofs", free)):
            if array.dtype != _INT64 or array.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional int64 array")
            if len(array) and (np.min(array) < 0 or np.max(array) >= n_dofs):
                raise ValueError(f"{name} contains a DOF outside [0, {n_dofs})")
            if len(np.unique(array)) != len(array):
                raise ValueError(f"{name} contains duplicate DOFs")
        if np.intersect1d(constrained, free).size:
            raise ValueError("constrained_dofs and free_dofs overlap")
        if not np.array_equal(
            np.sort(np.concatenate((constrained, free))), np.arange(n_dofs)
        ):
            raise ValueError("constrained_dofs and free_dofs must partition all DOFs")

        vector_specs = (
            ("q_zero", self.q_zero),
            ("a_top", self.a_top),
            ("a_bottom", self.a_bottom),
            ("a_anchor", self.a_anchor),
        )
        for name, values in vector_specs:
            array = np.asarray(values)
            if array.dtype != _FLOAT64 or array.shape != (n_dofs,):
                raise ValueError(f"{name} must have shape ({n_dofs},) and dtype float64")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")

        expected_constrained = np.concatenate(
            (2 * np.flatnonzero(bottom) + 1,
             2 * np.flatnonzero(top) + 1,
             np.asarray([2 * anchor], dtype=np.int64))
        )
        if not np.array_equal(constrained, expected_constrained):
            raise ValueError("constrained_dofs do not match the explicit boundary masks")

        expected_q_zero = np.zeros(n_dofs, dtype=np.float64)
        expected_top = np.zeros(n_dofs, dtype=np.float64)
        expected_top[2 * np.flatnonzero(top) + 1] = 1.0
        expected_bottom = np.zeros(n_dofs, dtype=np.float64)
        expected_bottom[2 * np.flatnonzero(bottom) + 1] = 1.0
        expected_anchor = np.zeros(n_dofs, dtype=np.float64)
        expected_anchor[2 * anchor] = 1.0
        for name, actual, expected in (
            ("q_zero", np.asarray(self.q_zero), expected_q_zero),
            ("a_top", np.asarray(self.a_top), expected_top),
            ("a_bottom", np.asarray(self.a_bottom), expected_bottom),
            ("a_anchor", np.asarray(self.a_anchor), expected_anchor),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"{name} is inconsistent with the boundary masks")

        for name, value in (
            ("bottom_mask", bottom),
            ("top_mask", top),
            ("constrained_dofs", constrained),
            ("free_dofs", free),
            ("q_zero", self.q_zero),
            ("a_top", self.a_top),
            ("a_bottom", self.a_bottom),
            ("a_anchor", self.a_anchor),
        ):
            object.__setattr__(self, name, _readonly_copy(value))
        object.__setattr__(self, "anchor_particle", anchor)

    @property
    def n_particles(self) -> int:
        return int(len(self.bottom_mask))

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_particles

    @property
    def bottom_particles(self) -> np.ndarray:
        return np.flatnonzero(self.bottom_mask)

    @property
    def top_particles(self) -> np.ndarray:
        return np.flatnonzero(self.top_mask)

    @property
    def anchor_dof(self) -> int:
        return 2 * self.anchor_particle

    def prescribed_vector(self, u_bar: Any) -> np.ndarray:
        """Return ``q_zero + a_top*u_bar`` for one explicit load value."""

        value = _scalar_float(u_bar, "u_bar")
        return np.asarray(self.q_zero + self.a_top * value, dtype=np.float64)


@dataclass(frozen=True)
class MechanicsSolveResult:
    """Full response and diagnostics returned by a hard-constraint solve."""

    displacement: np.ndarray
    evaluation: MechanicsEvaluation
    constraints: HardBoundaryConstraints
    u_bar: float
    free_residual: np.ndarray
    free_residual_norm: float
    free_rhs_norm: float
    free_residual_relative_norm: float
    initial_free_residual_norm: float
    cg_convergence_limit: float
    initial_guess_supplied: bool
    cg_initial_guess_used: bool
    early_exit: bool
    cg_iterations: int
    cg_info: int
    top_reaction: float
    bottom_reaction: float
    anchor_reaction: float
    total_force: np.ndarray
    total_moment: float
    mechanics_backend: str
    _backend_provenance: object

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, MechanicsEvaluation):
            raise TypeError("evaluation in solve result must be MechanicsEvaluation")
        if not isinstance(self.constraints, HardBoundaryConstraints):
            raise TypeError(
                "constraints in solve result must be HardBoundaryConstraints"
            )
        displacement = np.asarray(self.displacement)
        free_residual = np.asarray(self.free_residual)
        total_force = np.asarray(self.total_force)
        for name, array, shape in (
            ("displacement", displacement, (self.constraints.n_dofs,)),
            ("free_residual", free_residual, (len(self.constraints.free_dofs),)),
            ("total_force", total_force, (2,)),
        ):
            if array.dtype != _FLOAT64 or array.shape != shape:
                raise ValueError(f"{name} has invalid shape/dtype in solve result")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} in solve result must be finite")
            object.__setattr__(self, name, _readonly_copy(array))
        scalar_names = (
            "u_bar",
            "free_residual_norm",
            "free_rhs_norm",
            "free_residual_relative_norm",
            "initial_free_residual_norm",
            "cg_convergence_limit",
            "top_reaction",
            "bottom_reaction",
            "anchor_reaction",
            "total_moment",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} in solve result must be finite")
            object.__setattr__(self, name, value)
        if self.free_rhs_norm < 0.0:
            raise ValueError("free_rhs_norm in solve result must be non-negative")
        if self.initial_free_residual_norm < 0.0:
            raise ValueError(
                "initial_free_residual_norm in solve result must be non-negative"
            )
        if self.cg_convergence_limit <= 0.0:
            raise ValueError("cg_convergence_limit in solve result must be positive")
        for name in (
            "initial_guess_supplied",
            "cg_initial_guess_used",
            "early_exit",
        ):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} in solve result must be boolean")
            object.__setattr__(self, name, bool(value))
        if self.early_exit and not self.initial_guess_supplied:
            raise ValueError("early_exit requires an explicit initial guess")
        if self.cg_initial_guess_used and (
            not self.initial_guess_supplied or self.early_exit
        ):
            raise ValueError(
                "cg_initial_guess_used requires a supplied non-early-exit guess"
            )
        residual_norm = float(np.linalg.norm(free_residual))
        scale = max(residual_norm, self.free_residual_norm, 1.0)
        tolerance = 512.0 * np.finfo(np.float64).eps * scale
        if abs(self.free_residual_norm - residual_norm) > tolerance:
            raise ValueError("free_residual_norm is inconsistent with free_residual")
        expected_relative = self.free_residual_norm / max(self.free_rhs_norm, 1.0e-300)
        relative_scale = max(
            abs(expected_relative), abs(self.free_residual_relative_norm), 1.0
        )
        if (
            abs(self.free_residual_relative_norm - expected_relative)
            > 512.0 * np.finfo(np.float64).eps * relative_scale
        ):
            raise ValueError(
                "free_residual_relative_norm is inconsistent with free_rhs_norm"
            )
        for name in ("cg_iterations", "cg_info"):
            raw = np.asarray(getattr(self, name))
            if (
                raw.ndim != 0
                or not np.issubdtype(raw.dtype, np.integer)
                or isinstance(getattr(self, name), (bool, np.bool_))
            ):
                raise ValueError(f"{name} in solve result must be an integer")
            value = int(raw)
            if value < 0:
                raise ValueError(f"{name} in solve result must be non-negative")
            object.__setattr__(self, name, value)
        expected_backend_provenance = _mechanics_backend_token(
            self.mechanics_backend
        )
        if self._backend_provenance is not expected_backend_provenance:
            raise ValueError(
                "mechanics_backend provenance in solve result is invalid"
            )
        if self.early_exit and self.cg_iterations != 0:
            raise ValueError("early_exit requires zero CG iterations")
        if self.free_residual_norm >= self.cg_convergence_limit:
            raise ValueError(
                "free_residual_norm does not satisfy cg_convergence_limit"
            )

    @property
    def energy(self) -> float:
        return self.evaluation.energy

    @property
    def residual(self) -> np.ndarray:
        return self.evaluation.residual

    @property
    def internal_force(self) -> np.ndarray:
        return self.evaluation.internal_force

    @property
    def edge_deformation_gradient(self) -> np.ndarray:
        return self.evaluation.edge_deformation_gradient

    @property
    def edge_F(self) -> np.ndarray:
        return self.evaluation.edge_deformation_gradient

    @property
    def edge_strain(self) -> np.ndarray:
        return self.evaluation.edge_strain

    @property
    def edge_stress(self) -> np.ndarray:
        return self.evaluation.edge_stress

    @property
    def particle_strain(self) -> np.ndarray:
        return self.evaluation.particle_strain

    @property
    def particle_stress(self) -> np.ndarray:
        return self.evaluation.particle_stress

    @property
    def edge_effective_stress(self) -> np.ndarray:
        return self.evaluation.edge_effective_stress

    @property
    def particle_effective_stress(self) -> np.ndarray:
        return self.evaluation.particle_effective_stress


class BondAssociatedMechanics:
    """Float64 intact 2D bond-associated correspondence mechanics.

    Parameters are all required.  ``directed_edges`` may arrive in any order,
    but it must already be an integer, reciprocal, self-edge-free graph whose
    distances satisfy the strict ``0 < r_ij < delta`` horizon.  It is copied
    and normalized to lexicographic source/target order.  Every particle must
    have a positive outgoing center mass.

    ``rank_rtol`` is the singular-value threshold relative to the largest
    shape-tensor singular value.  It must be finite, positive, and below one;
    rank is checked before any shape inverse is formed.  There is no pseudo-
    inverse, clipping, or numerical shape fallback.
    """

    def __init__(
        self,
        coordinates: Any,
        volumes: Any,
        directed_edges: Any,
        delta: Any,
        youngs_modulus: Any,
        poisson_ratio: Any,
        k_res: Any,
        rank_rtol: Any,
    ) -> None:
        coordinates_array = _require_float64_array(coordinates, "coordinates")
        if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 2:
            raise ValueError(
                f"coordinates must have shape (N, 2), got {coordinates_array.shape}"
            )
        if len(coordinates_array) == 0:
            raise ValueError("coordinates must be non-empty")
        if len(np.unique(coordinates_array, axis=0)) != len(coordinates_array):
            raise ValueError("coordinates must contain unique particles")

        volumes_array = _require_float64_array(volumes, "volumes")
        if volumes_array.shape != (len(coordinates_array),):
            raise ValueError(
                "volumes must have one float64 value per coordinate, "
                f"got {volumes_array.shape} for N={len(coordinates_array)}"
            )
        if np.any(volumes_array <= 0.0):
            raise ValueError("volumes must be strictly positive")

        edges_array = _require_integer_array(directed_edges, "directed_edges")
        if edges_array.ndim != 2 or edges_array.shape[1] != 2:
            raise ValueError(
                f"directed_edges must have shape (E, 2), got {edges_array.shape}"
            )
        if len(edges_array) == 0:
            raise ValueError("directed_edges must be non-empty")
        if np.any(edges_array < 0) or np.any(edges_array >= len(coordinates_array)):
            raise ValueError("directed_edges contains an index outside the particle range")
        if np.any(edges_array[:, 0] == edges_array[:, 1]):
            raise ValueError("directed_edges must not contain self edges")

        # Normalize only after validating the input's integer graph semantics.
        order = np.lexsort((edges_array[:, 1], edges_array[:, 0]))
        normalized_edges = np.asarray(edges_array[order], dtype=np.int64)
        if len(np.unique(normalized_edges, axis=0)) != len(normalized_edges):
            raise ValueError("directed_edges contains duplicate directed edges")
        edge_set = {tuple(edge) for edge in normalized_edges.tolist()}
        missing_reciprocal = [
            (i, j) for i, j in edge_set if (j, i) not in edge_set
        ]
        if missing_reciprocal:
            raise ValueError(
                "directed_edges must be reciprocal; missing reverse of "
                f"{missing_reciprocal[0]}"
            )

        delta_value = _scalar_float(delta, "delta")
        if delta_value <= 0.0:
            raise ValueError(f"delta must be positive and finite, got {delta!r}")
        distance = np.linalg.norm(
            coordinates_array[normalized_edges[:, 1]]
            - coordinates_array[normalized_edges[:, 0]],
            axis=1,
        )
        if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
            raise ValueError("directed_edges must connect distinct finite coordinates")
        if np.any(distance >= delta_value):
            bad_index = int(np.flatnonzero(distance >= delta_value)[0])
            raise ValueError(
                "directed_edges must satisfy strict 0 < distance < delta; "
                f"edge {normalized_edges[bad_index].tolist()} has distance "
                f"{distance[bad_index]:.17g} and delta {delta_value:.17g}"
            )

        youngs_value = _scalar_float(youngs_modulus, "youngs_modulus")
        if youngs_value <= 0.0:
            raise ValueError("youngs_modulus must be positive and finite")
        poisson_value = _scalar_float(poisson_ratio, "poisson_ratio")
        if not -1.0 < poisson_value < 0.5:
            raise ValueError("poisson_ratio must satisfy -1 < poisson_ratio < 0.5")
        plane_strain_denominator = 1.0 - 2.0 * poisson_value
        if not np.isfinite(plane_strain_denominator) or plane_strain_denominator <= 0.0:
            raise ValueError("plane-strain denominator 1 - 2*poisson_ratio must be positive")
        k_res_value = _scalar_float(k_res, "k_res")
        if k_res_value <= 0.0:
            raise ValueError("k_res must be positive and finite")
        rank_rtol_value = _scalar_float(rank_rtol, "rank_rtol")
        if rank_rtol_value <= 0.0 or rank_rtol_value >= 1.0:
            raise ValueError("rank_rtol must be positive, finite, and reasonably less than 1")

        self.coordinates = _readonly_copy(coordinates_array)
        self.volumes = _readonly_copy(volumes_array)
        self.directed_edges = _readonly_copy(normalized_edges)
        self.edges = self.directed_edges
        self.delta = delta_value
        self.youngs_modulus = youngs_value
        self.poisson_ratio = poisson_value
        self.k_res = k_res_value
        self.rank_rtol = rank_rtol_value
        self.n_particles = int(len(coordinates_array))
        self.n_edges = int(len(normalized_edges))
        self.n_dofs = 2 * self.n_particles

        self.lame_lambda = youngs_value * poisson_value / (
            (1.0 + poisson_value) * plane_strain_denominator
        )
        self.shear_modulus = youngs_value / (2.0 * (1.0 + poisson_value))
        # These aliases make the constitutive convention explicit to callers.
        self.lame_mu = self.shear_modulus
        self.degradation = 1.0 + k_res_value

        source = normalized_edges[:, 0]
        target = normalized_edges[:, 1]
        omega = 120.0 / (np.pi * delta_value**4) * (1.0 - distance / delta_value) ** 2
        if np.any(~np.isfinite(omega)) or np.any(omega <= 0.0):
            raise ValueError("strict graph edges produced non-positive/non-finite kernel weights")
        center_mass = np.bincount(
            source,
            weights=omega * volumes_array[target],
            minlength=self.n_particles,
        ).astype(np.float64, copy=False)
        if np.any(~np.isfinite(center_mass)) or np.any(center_mass <= 0.0):
            bad = np.flatnonzero((~np.isfinite(center_mass)) | (center_mass <= 0.0))
            raise ValueError(
                "every particle must have positive finite center mass; "
                f"invalid particle indices {bad[:10].tolist()}"
            )

        adjacency: list[set[int]] = [set() for _ in range(self.n_particles)]
        omega_by_edge: dict[tuple[int, int], float] = {}
        for (i_raw, j_raw), weight in zip(normalized_edges, omega):
            i = int(i_raw)
            j = int(j_raw)
            adjacency[i].add(j)
            omega_by_edge[(i, j)] = float(weight)

        support_lists: list[list[int]] = []
        beta_lists: list[np.ndarray] = []
        xi_lists: list[np.ndarray] = []
        shape_inverses: list[np.ndarray] = []
        shape_ranks: list[int] = []
        shape_conditions: list[float] = []
        target_sensitivities: list[float] = []

        for edge_index, (i_raw, j_raw) in enumerate(normalized_edges):
            i = int(i_raw)
            j = int(j_raw)
            common = sorted(adjacency[i].intersection(adjacency[j]) - {i, j})
            support = [j, *common]
            beta = np.asarray(
                [omega_by_edge[(i, j)] * volumes_array[j]]
                + [omega_by_edge[(i, k)] * volumes_array[k] for k in common],
                dtype=np.float64,
            )
            support_array = np.asarray(support, dtype=np.int64)
            xi = coordinates_array[support_array] - coordinates_array[i]
            shape = np.einsum("s,sa,sb->ab", beta, xi, xi)
            singular_values = np.linalg.svd(shape, compute_uv=False)
            largest = float(singular_values[0])
            threshold = rank_rtol_value * largest
            rank = int(np.count_nonzero(singular_values > threshold))
            shape_ranks.append(rank)
            if rank < 2:
                raise ValueError(
                    "rank-deficient bond-associated shape tensor before inversion "
                    f"at directed edge index {edge_index} "
                    f"({i}, {j}); singular values={singular_values.tolist()}, "
                    f"rank_rtol={rank_rtol_value}"
                )
            smallest = float(singular_values[-1])
            condition = largest / smallest
            if not np.isfinite(condition):
                raise ValueError(
                    "bond-associated shape tensor has non-finite condition number "
                    f"at directed edge index {edge_index} ({i}, {j})"
                )
            shape_inverse = np.linalg.inv(shape)
            shape_inverses.append(shape_inverse)
            support_lists.append(support)
            beta_lists.append(beta)
            xi_lists.append(xi)
            shape_conditions.append(condition)

            # Spectral norm of the target-only map B_{i|j,j}; this must be
            # nonzero because the target endpoint is the first valid support.
            target_map = np.empty((4, 2), dtype=np.float64)
            for component in range(2):
                unit = np.zeros(2, dtype=np.float64)
                unit[component] = 1.0
                target_map[:, component] = (
                    beta[0] * np.outer(unit, xi[0]) @ shape_inverse
                ).ravel()
            target_sensitivity = float(np.linalg.svd(target_map, compute_uv=False)[0])
            if not np.isfinite(target_sensitivity) or target_sensitivity <= 0.0:
                raise ValueError(
                    "target-inclusive support has non-positive target sensitivity "
                    f"at directed edge index {edge_index} ({i}, {j})"
                )
            target_sensitivities.append(target_sensitivity)

        maximum_support = max(len(support) for support in support_lists)
        support_indices = np.full(
            (self.n_edges, maximum_support), -1, dtype=np.int64
        )
        support_mask = np.zeros((self.n_edges, maximum_support), dtype=bool)
        beta_array = np.zeros((self.n_edges, maximum_support), dtype=np.float64)
        xi_array = np.zeros((self.n_edges, maximum_support, 2), dtype=np.float64)
        for row, (support, beta, xi) in enumerate(
            zip(support_lists, beta_lists, xi_lists)
        ):
            count = len(support)
            support_indices[row, :count] = support
            support_mask[row, :count] = True
            beta_array[row, :count] = beta
            xi_array[row, :count] = xi

        eta = beta_array[:, 0] / center_mass[source]
        if np.any(~np.isfinite(eta)) or np.any(eta <= 0.0):
            raise ValueError("target eta weights must be finite and strictly positive")

        self.source = _readonly_copy(source, _INT64)
        self.target = _readonly_copy(target, _INT64)
        self.edge_distance = _readonly_copy(distance)
        self.edge_omega = _readonly_copy(omega)
        self.omega = self.edge_omega
        self.center_mass = _readonly_copy(center_mass)
        self.m = self.center_mass
        self.m_i = self.center_mass
        self.support_indices = _readonly_copy(support_indices)
        self.support = self.support_indices
        self.support_mask = _readonly_copy(support_mask)
        self.beta = _readonly_copy(beta_array)
        self.xi = _readonly_copy(xi_array)
        self.shape_inverse = _readonly_copy(np.asarray(shape_inverses, dtype=np.float64))
        self.support_sizes = _readonly_copy(
            np.asarray([len(support) for support in support_lists], dtype=np.int64)
        )
        self.shape_rank = _readonly_copy(np.asarray(shape_ranks, dtype=np.int64))
        self.shape_ranks = self.shape_rank
        self.rank = self.shape_rank
        self.shape_condition = _readonly_copy(
            np.asarray(shape_conditions, dtype=np.float64)
        )
        self.shape_conditions = self.shape_condition
        self.condition_numbers = self.shape_condition
        self.target_sensitivity = _readonly_copy(
            np.asarray(target_sensitivities, dtype=np.float64)
        )
        self.target_beta = _readonly_copy(beta_array[:, 0])
        self.eta = _readonly_copy(eta)
        self.eta_ij = self.eta
        self.center_factor = _readonly_copy(
            self.volumes[source] * self.degradation * eta
        )

        self._safe_support_indices = _readonly_copy(
            np.where(support_mask, support_indices, 0).astype(np.int64, copy=False)
        )
        # Fixed coefficient for the linear displacement-to-gradient map:
        # delta-F[e,a,c] = sum_s difference[e,s,a] * coefficient[e,s,c].
        # Keeping the beta/xi/K^{-1} contraction outside each action avoids
        # rebuilding the same fixed geometry algebra during CG iterations.
        self._gradient_coefficient = _readonly_copy(
            np.einsum(
                "es,esb,ebc->esc",
                beta_array,
                xi_array,
                np.asarray(shape_inverses, dtype=np.float64),
                optimize=True,
            )
        )

    def _deformation_gradient(self, displacement: np.ndarray) -> np.ndarray:
        """Evaluate ``F=A K^{-1}`` from flat interleaved displacement data."""

        support_x = displacement[2 * self._safe_support_indices]
        support_y = displacement[2 * self._safe_support_indices + 1]
        source_x = displacement[2 * self.source[:, None]]
        source_y = displacement[2 * self.source[:, None] + 1]
        difference_x = support_x - source_x
        difference_y = support_y - source_y
        current_bond = np.empty(
            (self.n_edges, self.support_indices.shape[1], 2), dtype=np.float64
        )
        current_bond[:, :, 0] = self.xi[:, :, 0] + difference_x
        current_bond[:, :, 1] = self.xi[:, :, 1] + difference_y
        moment = np.einsum(
            "es,esa,esb->eab", self.beta, current_bond, self.xi, optimize=True
        )
        return np.einsum(
            "eab,ebc->eac", moment, self.shape_inverse, optimize=True
        )

    def _strain_stress(
        self, deformation_gradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        gradient_increment = deformation_gradient - _IDENTITY_2
        strain = 0.5 * (
            gradient_increment + np.swapaxes(gradient_increment, 1, 2)
        )
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        stress = (
            self.lame_lambda * trace[:, None, None] * _IDENTITY_2[None, :, :]
            + 2.0 * self.shear_modulus * strain
        )
        return strain, stress

    def _edge_energy_density(self, strain: np.ndarray) -> np.ndarray:
        trace = strain[:, 0, 0] + strain[:, 1, 1]
        return 0.5 * self.lame_lambda * trace**2 + self.shear_modulus * np.sum(
            strain * strain, axis=(1, 2)
        )

    def _effective_stress_fields(
        self, edge_stress: np.ndarray, particle_stress: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return effective edge and source-centered particle stresses."""

        return self.degradation * edge_stress, self.degradation * particle_stress

    def _assemble_explicit_endpoint_residual(
        self, stress: np.ndarray
    ) -> np.ndarray:
        """Assemble Section 0.5 directly from endpoint force contributions."""

        endpoint = np.einsum(
            "eab,ebc,esc->esa",
            stress,
            np.swapaxes(self.shape_inverse, 1, 2),
            self.xi,
            optimize=True,
        )
        endpoint *= self.beta[:, :, None] * self.center_factor[:, None, None]
        # Invalid padded support entries have beta=0 and therefore contribute
        # exactly zero.  The first endpoint is always the target; every valid
        # support endpoint receives +q and its source receives -q.
        support_flat = self._safe_support_indices.reshape(-1)
        support_result = np.column_stack(
            (
                np.bincount(
                    support_flat,
                    weights=endpoint[:, :, 0].reshape(-1),
                    minlength=self.n_particles,
                ),
                np.bincount(
                    support_flat,
                    weights=endpoint[:, :, 1].reshape(-1),
                    minlength=self.n_particles,
                ),
            )
        )
        source_endpoint = np.sum(endpoint, axis=1)
        source_result = np.column_stack(
            (
                np.bincount(
                    self.source,
                    weights=-source_endpoint[:, 0],
                    minlength=self.n_particles,
                ),
                np.bincount(
                    self.source,
                    weights=-source_endpoint[:, 1],
                    minlength=self.n_particles,
                ),
            )
        )
        residual = support_result + source_result
        return residual.reshape(-1)

    def _delta_deformation_gradient(self, direction: np.ndarray) -> np.ndarray:
        support_x = direction[2 * self._safe_support_indices]
        support_y = direction[2 * self._safe_support_indices + 1]
        source_x = direction[2 * self.source[:, None]]
        source_y = direction[2 * self.source[:, None] + 1]
        difference_x = support_x - source_x
        difference_y = support_y - source_y
        delta_f = np.empty((self.n_edges, 2, 2), dtype=np.float64)
        delta_f[:, 0, :] = np.einsum(
            "es,esc->ec",
            difference_x,
            self._gradient_coefficient,
            optimize=True,
        )
        delta_f[:, 1, :] = np.einsum(
            "es,esc->ec",
            difference_y,
            self._gradient_coefficient,
            optimize=True,
        )
        return delta_f

    def _constitutive_tangent(self, delta_f: np.ndarray) -> np.ndarray:
        delta_strain = 0.5 * (delta_f + np.swapaxes(delta_f, 1, 2))
        trace = delta_strain[:, 0, 0] + delta_strain[:, 1, 1]
        return (
            self.lame_lambda * trace[:, None, None] * _IDENTITY_2[None, :, :]
            + 2.0 * self.shear_modulus * delta_strain
        )

    def _get_tangent_block_diagonal(self) -> np.ndarray:
        """Return exact particle blocks for a matrix-free Jacobi preconditioner.

        This is only a linear-solver preconditioner.  It is assembled from the
        same ``B^* C B`` tangent blocks as ``tangent_action`` and does not add a
        force, energy, stabilization, or correction to the declared model.
        No global matrix is formed.
        """

        cached = getattr(self, "_tangent_block_diagonal_cache", None)
        if cached is not None:
            return cached

        coefficient = self._gradient_coefficient
        cx = coefficient[:, :, 0]
        cy = coefficient[:, :, 1]
        factor = self.center_factor[:, None]
        hxx = factor * (
            (self.lame_lambda + 2.0 * self.shear_modulus) * cx * cx
            + self.shear_modulus * cy * cy
        )
        hxy = factor * (
            (self.lame_lambda + self.shear_modulus) * cx * cy
        )
        hyy = factor * (
            self.shear_modulus * cx * cx
            + (self.lame_lambda + 2.0 * self.shear_modulus) * cy * cy
        )
        support_flat = self._safe_support_indices.reshape(-1)
        blocks = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        blocks[:, 0, 0] = np.bincount(
            support_flat, weights=hxx.reshape(-1), minlength=self.n_particles
        )
        blocks[:, 0, 1] = np.bincount(
            support_flat, weights=hxy.reshape(-1), minlength=self.n_particles
        )
        blocks[:, 1, 0] = blocks[:, 0, 1]
        blocks[:, 1, 1] = np.bincount(
            support_flat, weights=hyy.reshape(-1), minlength=self.n_particles
        )

        source_coefficient = np.sum(coefficient, axis=1)
        source_cx = source_coefficient[:, 0]
        source_cy = source_coefficient[:, 1]
        source_hxx = self.center_factor * (
            (self.lame_lambda + 2.0 * self.shear_modulus) * source_cx**2
            + self.shear_modulus * source_cy**2
        )
        source_hxy = self.center_factor * (
            (self.lame_lambda + self.shear_modulus) * source_cx * source_cy
        )
        source_hyy = self.center_factor * (
            self.shear_modulus * source_cx**2
            + (self.lame_lambda + 2.0 * self.shear_modulus) * source_cy**2
        )
        blocks[:, 0, 0] += np.bincount(
            self.source, weights=source_hxx, minlength=self.n_particles
        )
        source_xy = np.bincount(
            self.source, weights=source_hxy, minlength=self.n_particles
        )
        blocks[:, 0, 1] += source_xy
        blocks[:, 1, 0] += source_xy
        blocks[:, 1, 1] += np.bincount(
            self.source, weights=source_hyy, minlength=self.n_particles
        )

        if not np.all(np.isfinite(blocks)):
            bad = np.flatnonzero(~np.isfinite(blocks))
            raise ValueError(
                "the exact tangent block diagonal is non-finite at entries "
                f"{bad[:10].tolist()}"
            )
        for particle in range(self.n_particles):
            eigenvalues = np.linalg.eigvalsh(blocks[particle])
            if not np.all(eigenvalues > 0.0):
                raise ValueError(
                    "the exact tangent block diagonal is not positive definite "
                    f"at particle {particle}: eigenvalues={eigenvalues.tolist()}"
                )
        blocks = _readonly_copy(blocks)
        self._tangent_block_diagonal_cache = blocks
        return blocks

    def _get_tangent_diagonal(self) -> np.ndarray:
        cached = getattr(self, "_tangent_diagonal_cache", None)
        if cached is not None:
            return cached
        blocks = self._get_tangent_block_diagonal()
        diagonal = np.empty(self.n_dofs, dtype=np.float64)
        diagonal[0::2] = blocks[:, 0, 0]
        diagonal[1::2] = blocks[:, 1, 1]
        diagonal = _readonly_copy(diagonal)
        self._tangent_diagonal_cache = diagonal
        return diagonal

    @property
    def tangent_block_diagonal(self) -> np.ndarray:
        """Exact particle-local tangent blocks used by the CG preconditioner."""

        return self._get_tangent_block_diagonal()

    @property
    def tangent_diagonal(self) -> np.ndarray:
        """Exact diagonal used by the matrix-free CG Jacobi preconditioner."""

        return self._get_tangent_diagonal()

    def energy(self, displacement: Any) -> float:
        """Return the intact center-partition mechanics energy."""

        displacement_flat = _validate_displacement(
            displacement, self.n_dofs, "displacement"
        )
        deformation_gradient = self._deformation_gradient(displacement_flat)
        strain, _ = self._strain_stress(deformation_gradient)
        density = self._edge_energy_density(strain)
        energy = float(np.sum(self.center_factor * density))
        if not np.isfinite(energy):
            raise FloatingPointError("non-finite intact mechanics energy")
        return energy

    def residual(self, displacement: Any) -> np.ndarray:
        """Return the explicit energy-gradient residual in flat interleaved order."""

        displacement_flat = _validate_displacement(
            displacement, self.n_dofs, "displacement"
        )
        deformation_gradient = self._deformation_gradient(displacement_flat)
        _, stress = self._strain_stress(deformation_gradient)
        residual = self._assemble_explicit_endpoint_residual(stress)
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("non-finite intact mechanics residual")
        return residual

    def internal_force(self, displacement: Any) -> np.ndarray:
        """Return the usual internal-force sign, ``-residual``."""

        return -self.residual(displacement)

    def tangent_action(self, direction: Any) -> np.ndarray:
        """Apply the intact Hessian independently through ``delta F`` and ``C``."""

        direction_flat = _validate_displacement(direction, self.n_dofs, "direction")
        delta_f = self._delta_deformation_gradient(direction_flat)
        delta_stress = self._constitutive_tangent(delta_f)
        action = self._assemble_explicit_endpoint_residual(delta_stress)
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("non-finite intact tangent action")
        return action

    def evaluate(self, displacement: Any) -> MechanicsEvaluation:
        """Evaluate energy, explicit residual, edge fields, and eta averages."""

        displacement_flat = _validate_displacement(
            displacement, self.n_dofs, "displacement"
        )
        deformation_gradient = self._deformation_gradient(displacement_flat)
        strain, stress = self._strain_stress(deformation_gradient)
        density = self._edge_energy_density(strain)
        energy = float(np.sum(self.center_factor * density))
        residual = self._assemble_explicit_endpoint_residual(stress)
        if not np.isfinite(energy) or not np.all(np.isfinite(residual)):
            raise FloatingPointError("non-finite intact mechanics evaluation")

        particle_strain = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        particle_stress = np.zeros((self.n_particles, 2, 2), dtype=np.float64)
        np.add.at(
            particle_strain,
            self.source,
            self.eta[:, None, None] * strain,
        )
        np.add.at(
            particle_stress,
            self.source,
            self.eta[:, None, None] * stress,
        )
        edge_effective_stress, particle_effective_stress = self._effective_stress_fields(
            stress, particle_stress
        )
        return MechanicsEvaluation(
            energy=energy,
            residual=residual,
            internal_force=-residual,
            edge_deformation_gradient=deformation_gradient,
            edge_strain=strain,
            edge_stress=stress,
            particle_strain=particle_strain,
            particle_stress=particle_stress,
            edge_effective_stress=edge_effective_stress,
            particle_effective_stress=particle_effective_stress,
            edge_energy_density=density,
        )

    def build_constraints(
        self,
        bottom_mask: Any,
        top_mask: Any,
        anchor_particle: Any,
    ) -> HardBoundaryConstraints:
        """Build immutable explicit bottom/top/anchor hard constraints."""

        bottom = np.asarray(bottom_mask)
        top = np.asarray(top_mask)
        expected_shape = (self.n_particles,)
        if bottom.dtype != np.dtype(bool) or top.dtype != np.dtype(bool):
            raise ValueError("bottom_mask and top_mask must use boolean dtype")
        if bottom.shape != expected_shape or top.shape != expected_shape:
            raise ValueError(
                "bottom_mask and top_mask must each have shape "
                f"{expected_shape}"
            )
        if not isinstance(anchor_particle, Integral) or isinstance(
            anchor_particle, (bool, np.bool_)
        ):
            raise ValueError("anchor_particle must be an integer particle index")
        anchor = int(anchor_particle)
        constrained = np.concatenate(
            (
                2 * np.flatnonzero(bottom) + 1,
                2 * np.flatnonzero(top) + 1,
                np.asarray([2 * anchor], dtype=np.int64),
            )
        ).astype(np.int64, copy=False)
        all_dofs = np.arange(self.n_dofs, dtype=np.int64)
        free = np.setdiff1d(all_dofs, constrained, assume_unique=False)
        q_zero = np.zeros(self.n_dofs, dtype=np.float64)
        a_top = np.zeros(self.n_dofs, dtype=np.float64)
        a_top[2 * np.flatnonzero(top) + 1] = 1.0
        a_bottom = np.zeros(self.n_dofs, dtype=np.float64)
        a_bottom[2 * np.flatnonzero(bottom) + 1] = 1.0
        a_anchor = np.zeros(self.n_dofs, dtype=np.float64)
        if 0 <= anchor < self.n_particles:
            a_anchor[2 * anchor] = 1.0
        return HardBoundaryConstraints(
            bottom_mask=bottom,
            top_mask=top,
            anchor_particle=anchor,
            constrained_dofs=constrained,
            free_dofs=free,
            q_zero=q_zero,
            a_top=a_top,
            a_bottom=a_bottom,
            a_anchor=a_anchor,
        )

    build_hard_constraints = build_constraints

    def _validate_cg_inputs(
        self,
        constraints: HardBoundaryConstraints,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
    ) -> tuple[float, float, float, int]:
        if not isinstance(constraints, HardBoundaryConstraints):
            raise TypeError("constraints must be a HardBoundaryConstraints instance")
        if constraints.n_particles != self.n_particles:
            raise ValueError("constraints have a different particle count")
        load = _scalar_float(u_bar, "u_bar")
        relative_tolerance = _validate_tolerance(cg_rtol, "cg_rtol", allow_zero=True)
        absolute_tolerance = _validate_tolerance(cg_atol, "cg_atol", allow_zero=True)
        if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
            raise ValueError("cg_rtol and cg_atol cannot both be zero")
        raw_maxiter = np.asarray(cg_maxiter)
        if raw_maxiter.ndim != 0 or not np.issubdtype(raw_maxiter.dtype, np.integer):
            raise ValueError("cg_maxiter must be a positive integer")
        maxiter = int(raw_maxiter)
        if maxiter <= 0:
            raise ValueError("cg_maxiter must be a positive integer")
        if len(constraints.free_dofs) == 0:
            raise ValueError("hard-boundary constraints leave no free DOFs for CG")
        return load, relative_tolerance, absolute_tolerance, maxiter

    def solve(
        self,
        constraints: HardBoundaryConstraints,
        u_bar: Any,
        cg_rtol: Any,
        cg_atol: Any,
        cg_maxiter: Any,
        *,
        initial_guess: Any | None = None,
        allow_initial_early_exit: bool = False,
    ) -> MechanicsSolveResult:
        """Solve the intact hard-constraint problem with matrix-free CG.

        The free tangent is the injected operator
        ``v_f -> [K @ inject(v_f)]_free``.  Only a ``LinearOperator`` is passed
        to SciPy; no dense global tangent is formed.  ``cg_info`` and the
        callback iteration count are retained in the result.  Any nonzero CG
        status, nonfinite iterate, or failed free-residual tolerance raises a
        ``RuntimeError`` rather than returning an apparently valid state.
        """

        load, relative_tolerance, absolute_tolerance, maxiter = self._validate_cg_inputs(
            constraints, u_bar, cg_rtol, cg_atol, cg_maxiter
        )
        prescribed = constraints.prescribed_vector(load)
        free = constraints.free_dofs

        def inject(free_values: np.ndarray) -> np.ndarray:
            values = np.asarray(free_values, dtype=np.float64)
            if values.shape != (len(free),):
                raise ValueError(
                    f"CG free vector must have shape ({len(free)},), got {values.shape}"
                )
            full = np.zeros(self.n_dofs, dtype=np.float64)
            full[free] = values
            return full

        def free_action(free_values: np.ndarray) -> np.ndarray:
            return self.tangent_action(inject(free_values))[free]

        prescribed_residual = self.residual(prescribed)
        rhs = -prescribed_residual[free]
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError("non-finite hard-constraint CG right-hand side")
        rhs_norm = float(np.linalg.norm(rhs))
        convergence_scale = max(absolute_tolerance, relative_tolerance * rhs_norm)
        if not isinstance(allow_initial_early_exit, (bool, np.bool_)):
            raise ValueError("allow_initial_early_exit must be boolean")
        allow_early_exit = bool(allow_initial_early_exit)

        if initial_guess is None:
            initial_free_values = np.zeros(len(free), dtype=np.float64)
            initial_free_residual_norm = rhs_norm
        else:
            initial = _validate_displacement(
                initial_guess, self.n_dofs, "initial_guess"
            )
            if not np.array_equal(
                initial[constraints.constrained_dofs],
                prescribed[constraints.constrained_dofs],
            ):
                raise ValueError(
                    "initial_guess constrained values must exactly match the "
                    "prescribed displacement at u_bar"
                )
            initial_free_values = np.array(initial[free], dtype=np.float64, copy=True)
            if np.array_equal(initial, prescribed):
                initial_free_residual_norm = rhs_norm
            else:
                initial_residual = self.residual(initial)
                initial_free_residual_norm = float(
                    np.linalg.norm(initial_residual[free])
                )
            if not np.isfinite(initial_free_residual_norm):
                raise FloatingPointError(
                    "initial_guess free residual norm is non-finite"
                )

        callback_iterations: list[int] = []

        def callback(_: np.ndarray) -> None:
            callback_iterations.append(1)

        early_exit = bool(
            initial_guess is not None
            and allow_early_exit
            and initial_free_residual_norm < convergence_scale
        )
        cg_initial_guess_used = False
        if early_exit:
            free_values = initial_free_values
            cg_info = 0
            cg_iterations = 0
        else:
            free_operator = LinearOperator(
                shape=(len(free), len(free)),
                matvec=free_action,
                dtype=np.float64,
            )
            block_diagonal = self._get_tangent_block_diagonal()
            free_positions = np.full(self.n_dofs, -1, dtype=np.int64)
            free_positions[free] = np.arange(len(free), dtype=np.int64)
            preconditioner_blocks: list[tuple[np.ndarray, np.ndarray]] = []
            for particle in range(self.n_particles):
                components = np.flatnonzero(
                    free_positions[2 * particle : 2 * particle + 2] >= 0
                )
                if len(components) == 0:
                    continue
                positions = free_positions[2 * particle + components]
                local_block = block_diagonal[particle][np.ix_(components, components)]
                if not np.all(np.isfinite(local_block)):
                    raise ValueError(
                        "free tangent preconditioner block is non-finite at "
                        f"particle {particle}"
                    )
                preconditioner_blocks.append(
                    (
                        positions.astype(np.int64, copy=False),
                        np.linalg.inv(local_block),
                    )
                )

            def free_preconditioner(free_values: np.ndarray) -> np.ndarray:
                values = np.asarray(free_values, dtype=np.float64)
                if values.shape != (len(free),):
                    raise ValueError(
                        "CG preconditioner vector has the wrong free-DOF shape"
                    )
                result = np.zeros_like(values)
                for positions, inverse in preconditioner_blocks:
                    result[positions] = inverse @ values[positions]
                return result

            preconditioner = LinearOperator(
                shape=(len(free), len(free)),
                matvec=free_preconditioner,
                dtype=np.float64,
            )
            cg_initial = (
                initial_free_values
                if initial_guess is not None
                and initial_free_residual_norm >= convergence_scale
                else None
            )
            cg_initial_guess_used = cg_initial is not None
            free_values, info = cg(
                free_operator,
                rhs,
                x0=cg_initial,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                maxiter=maxiter,
                callback=callback,
                M=preconditioner,
            )
            cg_info = int(info)
            cg_iterations = len(callback_iterations)
            if cg_info != 0:
                raise RuntimeError(
                    "hard-constraint CG did not converge: "
                    f"info={cg_info}, iterations={cg_iterations}, "
                    f"rtol={relative_tolerance}, atol={absolute_tolerance}, "
                    f"maxiter={maxiter}"
                )
        free_values = np.asarray(free_values, dtype=np.float64)
        if free_values.shape != (len(free),) or not np.all(np.isfinite(free_values)):
            raise FloatingPointError("hard-constraint CG returned a non-finite iterate")

        displacement = prescribed + inject(free_values)
        if not np.all(np.isfinite(displacement)):
            raise FloatingPointError("hard-constraint displacement is non-finite")
        evaluation = self.evaluate(displacement)
        free_residual = evaluation.residual[free]
        free_residual_norm = float(np.linalg.norm(free_residual))
        free_residual_relative_norm = free_residual_norm / max(rhs_norm, 1.0e-300)
        if not np.isfinite(free_residual_norm) or not np.isfinite(
            free_residual_relative_norm
        ):
            raise FloatingPointError("hard-constraint free residual is non-finite")
        if free_residual_norm >= convergence_scale:
            raise RuntimeError(
                "hard-constraint CG reported convergence but the explicit free "
                f"residual is too large: norm={free_residual_norm}, "
                f"allowed={convergence_scale}, "
                f"iterations={cg_iterations}"
            )

        top_y_dofs = 2 * np.flatnonzero(constraints.top_mask) + 1
        bottom_y_dofs = 2 * np.flatnonzero(constraints.bottom_mask) + 1
        anchor_dof = constraints.anchor_dof
        top_reaction = float(np.sum(evaluation.residual[top_y_dofs]))
        bottom_reaction = float(np.sum(evaluation.residual[bottom_y_dofs]))
        anchor_reaction = float(evaluation.residual[anchor_dof])
        residual_matrix = evaluation.residual.reshape(self.n_particles, 2)
        total_force = np.sum(residual_matrix, axis=0)
        total_moment = float(
            np.sum(
                self.coordinates[:, 0] * residual_matrix[:, 1]
                - self.coordinates[:, 1] * residual_matrix[:, 0]
            )
        )
        if not np.all(np.isfinite(total_force)) or not np.isfinite(total_moment):
            raise FloatingPointError("hard-constraint force/moment diagnostics are non-finite")

        return MechanicsSolveResult(
            displacement=displacement,
            evaluation=evaluation,
            constraints=constraints,
            u_bar=load,
            free_residual=free_residual,
            free_residual_norm=free_residual_norm,
            free_rhs_norm=rhs_norm,
            free_residual_relative_norm=free_residual_relative_norm,
            initial_free_residual_norm=initial_free_residual_norm,
            cg_convergence_limit=convergence_scale,
            initial_guess_supplied=initial_guess is not None,
            cg_initial_guess_used=(
                False if early_exit else cg_initial_guess_used
            ),
            early_exit=early_exit,
            cg_iterations=cg_iterations,
            cg_info=cg_info,
            top_reaction=top_reaction,
            bottom_reaction=bottom_reaction,
            anchor_reaction=anchor_reaction,
            total_force=total_force,
            total_moment=total_moment,
            mechanics_backend="numpy_scipy",
            _backend_provenance=_mechanics_backend_token("numpy_scipy"),
        )

    build_hard_boundary_constraints = build_constraints
    solve_hard_constraints = solve


class FixedDamageBondAssociatedMechanics(BondAssociatedMechanics):
    """Immutable fixed-damage view of the Stage 3 BA mechanics path.

    The BA deformation gradients, energy residual, tangent action, source-
    centered field averages, constrained CG solve, and reaction diagnostics are
    inherited unchanged from :class:`BondAssociatedMechanics`.  Only the
    immutable source degradation factors are different:

    ``g_i = (1 - damage[i])**2 + k_res`` and
    ``center_factor[e] = V_i * g_i * eta[e]``.

    Damage is intentionally not clipped or range-rejected.  Its exact
    float64, shape, and finite-value contract is enforced, while the range is
    exposed through ``damage_min``, ``damage_max``,
    ``damage_below_zero_count``, and ``damage_above_one_count``.
    """

    _CACHE_FIELDS = frozenset(
        {"_tangent_block_diagonal_cache", "_tangent_diagonal_cache"}
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            self.__dict__.get("_fixed_damage_locked", False)
            and name not in self._CACHE_FIELDS
        ):
            raise AttributeError(
                "FixedDamageBondAssociatedMechanics state is immutable; "
                "construct a new fixed-damage view"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if (
            self.__dict__.get("_fixed_damage_locked", False)
            and name not in self._CACHE_FIELDS
        ):
            raise AttributeError(
                "FixedDamageBondAssociatedMechanics state is immutable; "
                "construct a new fixed-damage view"
            )
        object.__delattr__(self, name)

    def __init__(self, intact_mechanics: Any, damage: Any) -> None:
        if type(intact_mechanics) is not BondAssociatedMechanics:
            raise TypeError(
                "intact_mechanics must be a BondAssociatedMechanics instance"
            )
        damage_array = np.asarray(damage)
        if damage_array.dtype != _FLOAT64:
            raise ValueError(
                f"damage must use exact float64 dtype, got {damage_array.dtype}"
            )
        if damage_array.ndim != 1:
            raise ValueError(
                f"damage must have one-dimensional shape (N,), got {damage_array.shape}"
            )
        if not np.all(np.isfinite(damage_array)):
            raise ValueError("damage must contain only finite values")
        if damage_array.shape != (intact_mechanics.n_particles,):
            raise ValueError(
                "damage must have one value per particle with shape "
                f"({intact_mechanics.n_particles},), got {damage_array.shape}"
            )

        # Reuse the accepted immutable BA geometry and constitutive data.  Solver
        # caches are deliberately excluded because they depend on center_factor.
        for name, value in intact_mechanics.__dict__.items():
            if name not in {
                "_tangent_block_diagonal_cache",
                "_tangent_diagonal_cache",
            }:
                setattr(self, name, value)
        self.intact_mechanics = intact_mechanics
        self._damage = _readonly_copy(damage_array)
        self._configure_damage_center_factors()
        object.__setattr__(self, "_fixed_damage_locked", True)

    @property
    def damage(self) -> np.ndarray:
        """Return the read-only fixed field; rebinding is not permitted."""

        return self._damage

    def _configure_damage_center_factors(self) -> None:
        particle_degradation = (1.0 - self.damage) ** 2 + self.k_res
        if not np.all(np.isfinite(particle_degradation)) or np.any(
            particle_degradation <= 0.0
        ):
            raise FloatingPointError(
                "fixed-damage degradation is non-finite or non-positive"
            )
        self.particle_degradation = _readonly_copy(particle_degradation)
        # The intact class exposes a scalar ``degradation``.  This separate
        # fixed-damage view exposes the corresponding source-particle g_i field
        # while all inherited mechanics operations consume the immutable edge
        # factors below.
        self.degradation = self.particle_degradation
        self.edge_degradation = _readonly_copy(
            particle_degradation[self.source]
        )
        self.center_factor = _readonly_copy(
            self.volumes[self.source] * self.edge_degradation * self.eta
        )
        self.damage_min = float(np.min(self.damage))
        self.damage_max = float(np.max(self.damage))
        self.damage_below_zero_count = int(np.count_nonzero(self.damage < 0.0))
        self.damage_above_one_count = int(np.count_nonzero(self.damage > 1.0))
        self.damage_range = (self.damage_min, self.damage_max)

    def _effective_stress_fields(
        self, edge_stress: np.ndarray, particle_stress: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.edge_degradation[:, None, None] * edge_stress,
            self.particle_degradation[:, None, None] * particle_stress,
        )


# Descriptive alias for callers that want to emphasize the frozen intact-only
# scope without duplicating a second implementation.
IntactBondAssociatedMechanics = BondAssociatedMechanics
