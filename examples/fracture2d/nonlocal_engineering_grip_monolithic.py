"""Private experimental monolithic q-d-H residual for the engineering grip.

This module defines the exact code-compatible monolithic residual over the
packed unknown

$$
z = (q, d, H) \\in \\mathbb{R}^{n_f + 2 n_m},
$$

where ``q`` are the free union mechanics DOFs, ``d`` is the material damage,
and ``H`` is the material history, together with its analytic generalized
Jacobian action (JVP) and the private Phase-2A scaled semismooth Newton/GMRES
    core.  The workflow entry is kept in the separate orchestration module; this
    numerical module remains free of CLI, artifact, and physical-run policy.

Every equation is the existing accepted engineering-grip formulation, reused
without modification:

- the full displacement is reconstructed by the existing affine hard
  constraints at the target grip displacement ``u_bar``:
  ``Q(q; u_bar) = prescribed(u_bar) + inject(q)``;
- ``Rq(z)`` is the existing free-row ``A2FixedDamageView`` residual of the
  fixed-damage mechanics view evaluated at ``Q``;
- ``Rd(z) = M rho`` with the existing material AT2 strong residual
  ``rho = Gc/l0 d + Gc l0 D d + 2 H (d - 1)`` and material particle volumes
  ``M = diag(V)``; it contains no mechanics-energy derivative;
- ``RH(z) = H - max(H_committed, psi_plus(Q))`` elementwise, where
  ``psi_plus`` is the existing material-source embedded plane-strain Amor
  driver.  The active set is the exact strict comparison
  ``psi_plus > H_committed``; equality is committed (inactive), with no
  smoothing or tolerance.

The analytic generalized JVP blocks at ``z`` are:

- ``Jqq``: the existing tangent action lifted to the free DOFs;
- ``Jqd``: the source center-weight derivative
  ``dW_i/dd_i = -2 V^m_i (1 - d_i)`` of the phase-measured center weight
  ``W_i = V^m_i g(d_i) + V^g_i g(0)``, ``g(d) = (1 - d)^2 + k_res``, applied
  through the exact existing endpoint edge-force assembly (stress held at
  ``Q``);
- ``Jdd delta d = V * ((Gc/l0 + 2 H) * delta d + Gc l0 D delta d)`` and
  ``JdH delta H = V * (2 (d - 1) * delta H)``;
- ``JHq delta q = -active * dpsi_plus/dq`` on active rows, where the Amor
  first variation of one edge is

  $$
  d\\psi_e = \\bigl[K \\max(\\mathrm{tr}\\,\\varepsilon_e, 0) I_2
  + 2\\mu\\,\\mathrm{dev}^{2D}_e\\bigr] : d\\varepsilon_e,
  \\quad K = \\lambda + 2\\mu/3,
  $$

  with the same embedded 3x3 deviatoric block as the existing driver and
  ``d\\varepsilon_e = sym(dF_e)`` from the existing BA kinematic
  linearization; the volumetric factor is single-valued (zero) at
  ``tr = 0``;

- identity in ``H``.

``Jdq = 0`` and ``JHd = 0`` exactly, so the packed Jacobian is nonsymmetric
whenever ``Jqd`` or ``JHq`` is nonzero.  Exact adjoint (transpose) actions are
exposed because each is the exact transpose of an existing assembly or scatter:
``Jqq``, ``Jdd``, and ``JdH`` are symmetric, and the ``Jqd``/``JHq`` adjoints
gather through the same endpoint/support edge data.  Their correctness evidence
is the independent-oracle bilinear identity on the manufactured cloud, not a
dense transpose assembled from production JVP columns.

The residual never enforces a damage obstacle: ``d`` and ``H`` iterates are
only required to be finite float64 arrays of the packed shape.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
from typing import Any, NoReturn

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .nonlocal_a2_coupled import (
    A2FixedDamageView,
    A2MechanicsSolveResult,
    _digest_primitives as _a2_digest_primitives,
)
from .nonlocal_baseline import GateTolerance
from . import nonlocal_engineering_grip as engineering
from .nonlocal_engineering_grip import (
    EngineeringGripConfig,
    EngineeringGripDiagnostics,
    EngineeringGripModel,
    EngineeringGripState,
    compute_material_source_amor,
)


__all__ = [
    "MonolithicGripResidual",
    "MonolithicNewtonConfig",
    "MonolithicNewtonGateConfig",
    "MonolithicNewtonScales",
    "MonolithicNewtonDiagnostics",
    "MonolithicNewtonFailureAudit",
    "MonolithicNewtonResult",
    "MonolithicNewtonFailure",
    "MonolithicBlockDiagonalPreconditioner",
    "solve_monolithic_newton",
    "solve_scaled_semismooth_newton",
    "ScaledSemismoothNewtonConfig",
    "ScaledSemismoothNewtonGateConfig",
    "ScaledSemismoothNewtonResult",
    "MONOLITHIC_AUXILIARY_GATE_NAMES",
    "MonolithicAuxiliarySweep",
    "MonolithicAuxiliaryAudit",
    "MonolithicAttemptFailureAudit",
    "MonolithicVerifiedAttempt",
    "MonolithicAttempt",
    "MonolithicAttemptResult",
    "run_verified_monolithic_attempt",
    "attempt_monolithic_target",
    "solve_verified_monolithic_attempt",
    "run_monolithic_attempt",
    "MonolithicContinuationConfig",
    "MonolithicContinuationAttempt",
    "MonolithicContinuationResult",
    "MonolithicContinuationFailure",
    "run_monolithic_strict_continuation",
    "run_strict_monolithic_continuation",
    "run_monolithic_continuation",
]


_FLOAT64 = np.dtype(np.float64)
_IDENTITY_2 = np.eye(2, dtype=np.float64)
_IDENTITY_3 = np.eye(3, dtype=np.float64)
_RESIDUAL_ROUNDTRIP_BOUND = 512.0 * np.finfo(np.float64).eps
_NEWTON_CONVERGENCE_STATUS = "converged"
_NEWTON_REASON_INITIAL_RESIDUAL = "exact_initial_residual"
_NEWTON_REASON_ACCEPTED_TRIAL = "accepted_trial_tolerance"


def _immutable(values: Any, dtype: Any | None = None) -> np.ndarray:
    """Make a bytes-backed immutable copy."""

    source = np.array(values, dtype=dtype, copy=True, order="C")
    raw = source.tobytes(order="C")
    result = np.frombuffer(raw, dtype=source.dtype, count=source.size)
    if source.shape != result.shape:
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


def _stable_l2_norm(values: np.ndarray, name: str) -> float:
    """Return a scaled Euclidean norm without squaring extreme magnitudes."""

    array = _f64(values, name)
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    normalized = array / scale
    result = scale * float(np.sqrt(np.dot(normalized, normalized)))
    if not np.isfinite(result):
        raise FloatingPointError(f"{name} norm is non-finite")
    return result


def _require_roundtrip_equal(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"{name} has the wrong round-trip shape")
    scale = max(
        1.0,
        float(np.max(np.abs(actual_array))) if actual_array.size else 0.0,
        float(np.max(np.abs(expected_array))) if expected_array.size else 0.0,
    )
    if not np.allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=_RESIDUAL_ROUNDTRIP_BOUND * scale,
    ):
        raise ValueError(f"{name} is not algebraically paired")


def _f64(values: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != _FLOAT64:
        raise ValueError(f"{name} must use exact float64 dtype, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


class _CenterFactorAssembly:
    """Reuse the exact ``A2FixedDamageView`` endpoint assembly verbatim.

    ``A2FixedDamageView._endpoint_residual`` is linear in its per-edge center
    factor for a fixed stress state, so the damage-derivative block calls the
    authoritative implementation through this shim with only the center factor
    replaced.  No assembly equation is duplicated or changed.
    """

    __slots__ = ("_view", "factor")

    def __init__(self, view: A2FixedDamageView, factor: np.ndarray) -> None:
        self._view = view
        self.factor = factor

    @property
    def static(self) -> Any:
        return self._view.static

    @property
    def center_factor(self) -> np.ndarray:
        return self.factor

    @property
    def n_particles(self) -> int:
        return self._view.n_particles

    @property
    def n_edges(self) -> int:
        return self._view.n_edges

    def _safe_support(self) -> np.ndarray:
        return self._view._safe_support()

    def _get_gradient_operator(self) -> scipy.sparse.csr_matrix:
        return self._view._get_gradient_operator()


def _edge_force_assembly(view: A2FixedDamageView, factor: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Run the exact existing endpoint assembly with a replaced center factor."""

    return A2FixedDamageView._endpoint_residual(
        _CenterFactorAssembly(view, factor),  # type: ignore[arg-type]
        stress,
    )


def _amor_first_variation_factor(
    strain: np.ndarray, lame_lambda: float, shear_modulus: float
) -> np.ndarray:
    """Return the per-edge Amor first-variation factor ``G_e``.

    For the existing embedded plane-strain Amor positive energy
    ``psi_e = 0.5 K max(tr, 0)^2 + mu |dev_3|^2`` with
    ``K = lambda + 2 mu / 3``, the first variation with respect to the 2x2
    symmetric edge strain is
    ``d psi_e = G_e : d eps_e`` with
    ``G_e = K max(tr, 0) I2 + 2 mu dev^{2D}_e``.  The trace factor is
    single-valued and zero at ``tr = 0``; this is the exact derivative, not a
    smoothed branch choice.
    """

    values = _f64(strain, "Amor strain")
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("Amor strain must have shape (E, 2, 2)")
    lam = _scalar(lame_lambda, "lame_lambda")
    mu = _scalar(shear_modulus, "shear_modulus")
    if mu <= 0.0 or lam + 2.0 * mu / 3.0 <= 0.0:
        raise ValueError("Amor constitutive values are invalid")
    embedded = np.zeros((len(values), 3, 3), dtype=np.float64)
    embedded[:, :2, :2] = values
    trace = np.trace(embedded, axis1=1, axis2=2)
    deviatoric = embedded - (trace / 3.0)[:, None, None] * _IDENTITY_3
    bulk = lam + 2.0 * mu / 3.0
    factor = (
        bulk * np.maximum(trace, 0.0)[:, None, None] * _IDENTITY_2[None, :, :]
        + 2.0 * mu * deviatoric[:, :2, :2]
    )
    if not np.all(np.isfinite(factor)):
        raise FloatingPointError("Amor first-variation factor is non-finite")
    return factor


class MonolithicGripResidual:
    """Experimental monolithic q-d-H residual and analytic generalized JVP.

    The object binds one :class:`EngineeringGripModel`, one target grip
    displacement ``u_bar``, and one committed material history
    ``H_committed >= 0``.  It precomputes only immutable static data (the
    material AT2 damage matrix with frozen read-only ``data``/``indices``/
    ``indptr`` internals, particle volumes, phase volumes, graph arrays, and
    the prescribed constrained displacement); it never mutates caller arrays
    and never enforces a damage obstacle.  Every cached array is read-only,
    so any in-place mutation attempt fails loudly.
    """

    def __init__(
        self,
        model: EngineeringGripModel,
        u_bar: Any,
        committed_history: Any,
    ) -> None:
        if not isinstance(model, EngineeringGripModel):
            raise TypeError("model must be an EngineeringGripModel instance")
        load = _scalar(u_bar, "monolithic u_bar")
        if load < 0.0:
            raise ValueError("monolithic u_bar must be non-negative")
        history = _f64(
            committed_history, "monolithic committed history", (model.n_material,)
        )
        if np.any(history < 0.0):
            raise ValueError("monolithic committed history must be non-negative")
        static = model.static
        self._model = model
        self._static = static
        self._constraints = model.constraints
        self._free = static.free_dofs
        self._u_bar = load
        self._committed_history = _immutable(history)
        self._n_free = int(len(self._free))
        self._n_material = int(model.n_material)
        self._n_dofs = int(static.n_dofs)
        self._n_packed = self._n_free + 2 * self._n_material
        # ``setflags(write=False)`` alone is reversible for an owning NumPy
        # array.  Rebuild the local CSR from bytes-backed copies so neither a
        # direct write nor a later ``setflags(write=True)`` can mutate this
        # cached graph.  The authoritative model operator is only exported and
        # copied; its storage is never modified here.
        exported_matrix = model.material_operator.to_scipy().tocsr(copy=True)
        csr_data = _immutable(exported_matrix.data)
        csr_indices = _immutable(exported_matrix.indices)
        csr_indptr = _immutable(exported_matrix.indptr)
        damage_matrix = scipy.sparse.csr_matrix(
            (csr_data, csr_indices, csr_indptr),
            shape=exported_matrix.shape,
            copy=False,
        )
        for internal in ("data", "indices", "indptr"):
            cached = getattr(damage_matrix, internal)
            if cached.flags.writeable:
                raise RuntimeError(
                    f"cached damage CSR {internal} unexpectedly became writeable"
                )
            try:
                cached.setflags(write=True)
            except ValueError:
                pass
            else:
                raise RuntimeError(
                    f"cached damage CSR {internal} is not irrevocably immutable"
                )
        self._damage_matrix = damage_matrix
        self._damage_volumes = _immutable(
            np.asarray(model.material_operator.volumes, dtype=np.float64)
        )
        self._source = static.source
        self._eta = static.eta
        self._material_source = _immutable(static.source < static.material_count)
        self._material_phase_volumes = static.union_material_phase_volumes
        self._prescribed = _immutable(self._constraints.prescribed_vector(load))
        geometry_view = model.fixed_damage_view(
            np.zeros(self._n_material, dtype=np.float64)
        )
        self._gradient_operator = geometry_view._get_gradient_operator()
        self._cached_view = geometry_view
        self._jvp_linearization_cache: tuple[
            np.ndarray,
            A2FixedDamageView,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ] | None = None

    # ------------------------------------------------------------------
    # Sizes and packing
    # ------------------------------------------------------------------

    @property
    def n_free(self) -> int:
        return self._n_free

    @property
    def n_material(self) -> int:
        return self._n_material

    @property
    def n_packed(self) -> int:
        return self._n_packed

    @property
    def u_bar(self) -> float:
        return self._u_bar

    @property
    def committed_history(self) -> np.ndarray:
        return self._committed_history

    def pack(self, q: Any, d: Any, H: Any) -> np.ndarray:
        """Pack ``(q, d, H)`` into the monolithic unknown vector."""

        q_values = _f64(q, "monolithic q", (self._n_free,))
        d_values = _f64(d, "monolithic d", (self._n_material,))
        h_values = _f64(H, "monolithic H", (self._n_material,))
        return np.concatenate((q_values, d_values, h_values))

    def unpack(self, z: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the ``(q, d, H)`` views of a validated packed vector."""

        values = _f64(z, "monolithic z", (self._n_packed,))
        return (
            values[: self._n_free],
            values[self._n_free : self._n_free + self._n_material],
            values[self._n_free + self._n_material :],
        )

    def split_packed(self, values: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split any packed-length vector into its ``q/d/H`` components."""

        array = _f64(values, "packed vector", (self._n_packed,))
        return self.unpack(array)

    def reconstruct_displacement(self, z: Any) -> np.ndarray:
        """Reconstruct the full union displacement at the fixed target load."""

        q, _, _ = self.unpack(z)
        displacement = np.array(self._prescribed, dtype=np.float64, copy=True)
        displacement[self._free] += q
        return displacement

    # ------------------------------------------------------------------
    # Private shared evaluation helpers
    # ------------------------------------------------------------------

    def _view(self, d: np.ndarray) -> A2FixedDamageView:
        if np.array_equal(d, self._cached_view.material_damage):
            return self._cached_view
        view = self._model.fixed_damage_view(d)
        view._gradient_operator_cache = self._gradient_operator
        self._cached_view = view
        return view

    def _jvp_linearization(
        self, z: np.ndarray, d: np.ndarray
    ) -> tuple[A2FixedDamageView, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._jvp_linearization_cache
        if cached is not None and np.array_equal(z, cached[0]):
            return cached[1], cached[2], cached[3], cached[4]
        view = self._view(d)
        displacement = self.reconstruct_displacement(z)
        strain, stress = view._strain_stress(view._deformation_gradient(displacement))
        factor = _amor_first_variation_factor(
            strain, view.lame_lambda, view.shear_modulus
        )
        self._jvp_linearization_cache = (
            _immutable(z),
            view,
            _immutable(displacement),
            _immutable(stress),
            _immutable(factor),
        )
        return view, displacement, stress, factor

    def _inject(self, direction: np.ndarray) -> np.ndarray:
        result = np.zeros(self._n_dofs, dtype=np.float64)
        result[self._free] = direction
        return result

    def _padded(self, material_values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (material_values, np.zeros(self._static.n_particles - self._n_material))
        )

    def _amor_driver(self, view: A2FixedDamageView, displacement: np.ndarray) -> np.ndarray:
        result = compute_material_source_amor(
            view, displacement, self._committed_history
        )
        driver = np.asarray(result.particle_driver, dtype=np.float64)
        if driver.shape != (self._n_material,) or not np.all(np.isfinite(driver)):
            raise FloatingPointError("monolithic Amor driver is invalid")
        return driver

    def _amor_driver_direction(
        self, view: A2FixedDamageView, displacement: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Exact action of ``d psi_plus / dQ`` on a full-union direction."""

        gradient = view._deformation_gradient(displacement)
        strain, _ = view._strain_stress(gradient)
        factor = _amor_first_variation_factor(
            strain, view.lame_lambda, view.shear_modulus
        )
        return self._amor_driver_direction_from_factor(view, factor, direction)

    def _amor_driver_direction_from_factor(
        self, view: A2FixedDamageView, factor: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        delta_gradient = view._delta_gradient(direction)
        delta_strain = 0.5 * (delta_gradient + np.swapaxes(delta_gradient, 1, 2))
        edge = np.sum(factor * delta_strain, axis=(1, 2))
        driver = np.zeros(self._n_material, dtype=np.float64)
        np.add.at(
            driver,
            self._source[self._material_source],
            self._eta[self._material_source] * edge[self._material_source],
        )
        if not np.all(np.isfinite(driver)):
            raise FloatingPointError("monolithic Amor driver direction is non-finite")
        return driver

    def _center_weight_direction(self, d: np.ndarray, delta_d: np.ndarray) -> np.ndarray:
        """Per-edge center-factor derivative for a material damage direction.

        ``dW_i/dd_i = -2 V^m_i (1 - d_i)`` on material particles and exactly
        zero on grip-only particles (their material phase volume is zero), so
        the padded damage value never contributes.
        """

        weight = -2.0 * self._material_phase_volumes * (1.0 - self._padded(d))
        return self._eta * weight[self._source] * self._padded(delta_d)[self._source]

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def psi_plus(self, z: Any) -> np.ndarray:
        """Return the material-source Amor driver ``psi_plus(Q)``."""

        _, d, _ = self.unpack(z)
        displacement = self.reconstruct_displacement(z)
        return self._amor_driver(self._view(d), displacement)

    def active_mask(self, z: Any) -> np.ndarray:
        """Exact strict active set ``psi_plus > H_committed`` (tie committed)."""

        driver = self.psi_plus(z)
        return np.asarray(driver > self._committed_history, dtype=bool)

    def residual(self, z: Any) -> np.ndarray:
        """Return the packed monolithic residual ``(Rq, Rd, RH)``."""

        _, d, history = self.unpack(z)
        displacement = self.reconstruct_displacement(z)
        view = self._view(d)
        residual_q = np.asarray(view.residual(displacement)[self._free], dtype=np.float64)
        local = self._model.Gc / self._model.l0 * d
        graph = np.asarray(self._damage_matrix @ d, dtype=np.float64)
        graph_term = self._model.Gc * self._model.l0 * graph
        rho = local + graph_term + 2.0 * history * d - 2.0 * history
        residual_d = np.asarray(self._damage_volumes * rho, dtype=np.float64)
        driver = self._amor_driver(view, displacement)
        residual_h = history - np.maximum(self._committed_history, driver)
        blocks = (residual_q, residual_d, residual_h)
        if not all(np.all(np.isfinite(block)) for block in blocks):
            raise FloatingPointError("monolithic residual block is non-finite")
        return np.concatenate(blocks)

    # ------------------------------------------------------------------
    # Analytic generalized JVP blocks
    # ------------------------------------------------------------------

    def jqq_action(self, z: Any, direction: Any) -> np.ndarray:
        """Free-row tangent action ``Jqq delta q`` (existing tangent path)."""

        _, d, _ = self.unpack(z)
        dq = _f64(direction, "Jqq direction", (self._n_free,))
        view = self._view(d)
        result = np.asarray(
            view.tangent_action(self._inject(dq))[self._free], dtype=np.float64
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Jqq action is non-finite")
        return result

    def jqd_action(self, z: Any, direction: Any) -> np.ndarray:
        """Damage-derivative action ``Jqd delta d`` of the free mechanics row.

        The stress state is held at ``Q`` and the existing endpoint edge-force
        assembly is reused verbatim with the exact center-weight derivative
        ``-2 V^m (1 - d) delta d`` as its per-edge factor.
        """

        _, d, _ = self.unpack(z)
        dd = _f64(direction, "Jqd direction", (self._n_material,))
        view = self._view(d)
        displacement = self.reconstruct_displacement(z)
        stress = view._strain_stress(view._deformation_gradient(displacement))[1]
        factor = self._center_weight_direction(d, dd)
        result = np.asarray(
            _edge_force_assembly(view, factor, stress)[self._free], dtype=np.float64
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Jqd action is non-finite")
        return result

    def jdd_action(self, z: Any, direction: Any) -> np.ndarray:
        """Damage derivative ``Jdd delta d`` of the volume-weighted AT2 row."""

        _, d, history = self.unpack(z)
        dd = _f64(direction, "Jdd direction", (self._n_material,))
        graph = np.asarray(self._damage_matrix @ dd, dtype=np.float64)
        result = np.asarray(
            self._damage_volumes
            * (
                self._model.Gc / self._model.l0 * dd
                + self._model.Gc * self._model.l0 * graph
                + 2.0 * history * dd
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Jdd action is non-finite")
        return result

    def jdh_action(self, z: Any, direction: Any) -> np.ndarray:
        """History derivative ``JdH delta H`` of the volume-weighted AT2 row."""

        _, d, _ = self.unpack(z)
        dh = _f64(direction, "JdH direction", (self._n_material,))
        result = np.asarray(
            self._damage_volumes * (2.0 * (d - 1.0) * dh), dtype=np.float64
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("JdH action is non-finite")
        return result

    def jhq_action(self, z: Any, direction: Any) -> np.ndarray:
        """History-row mechanics action ``JHq delta q``.

        Exactly ``-d psi_plus/dq delta q`` on active rows and exactly zero on
        inactive rows (including the committed tie), with no smoothing.
        """

        _, d, _ = self.unpack(z)
        dq = _f64(direction, "JHq direction", (self._n_free,))
        active = self.active_mask(z)
        if not np.any(active):
            return np.zeros(self._n_material, dtype=np.float64)
        displacement = self.reconstruct_displacement(z)
        driver_direction = self._amor_driver_direction(
            self._view(d), displacement, self._inject(dq)
        )
        result = np.where(active, -driver_direction, 0.0)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("JHq action is non-finite")
        return result

    def jvp_with_active_set(
        self, z: Any, direction: Any, active_mask: Any
    ) -> np.ndarray:
        """Apply the analytic JVP with an explicitly frozen active mask.

        A Newton linearization must not change its generalized derivative while
        GMRES is applying different directions.  The required ``active_mask``
        argument makes that invariant visible at the API boundary.
        """

        _, d, history = self.unpack(z)
        dq, dd, dh = self.unpack(
            _f64(direction, "monolithic direction", (self._n_packed,))
        )
        active = np.asarray(active_mask)
        if active.dtype != np.dtype(bool) or active.shape != (self._n_material,):
            raise ValueError(
                "frozen monolithic active_mask must have boolean dtype and shape "
                f"({self._n_material},)"
            )
        view, displacement, stress, amor_factor = self._jvp_linearization(z, d)

        residual_q = np.asarray(
            view.tangent_action(self._inject(dq))[self._free], dtype=np.float64
        )
        residual_q = residual_q + np.asarray(
            _edge_force_assembly(
                view, self._center_weight_direction(d, dd), stress
            )[self._free],
            dtype=np.float64,
        )

        graph = np.asarray(self._damage_matrix @ dd, dtype=np.float64)
        residual_d = np.asarray(
            self._damage_volumes
            * (
                self._model.Gc / self._model.l0 * dd
                + self._model.Gc * self._model.l0 * graph
                + 2.0 * history * dd
                + 2.0 * (d - 1.0) * dh
            ),
            dtype=np.float64,
        )

        if np.any(active):
            driver_direction = self._amor_driver_direction_from_factor(
                view, amor_factor, self._inject(dq)
            )
            residual_h = dh - np.where(active, driver_direction, 0.0)
        else:
            residual_h = np.array(dh, dtype=np.float64, copy=True)

        blocks = (residual_q, residual_d, residual_h)
        if not all(np.all(np.isfinite(block)) for block in blocks):
            raise FloatingPointError("monolithic JVP block is non-finite")
        return np.concatenate(blocks)

    def jvp(self, z: Any, direction: Any) -> np.ndarray:
        """Return the packed analytic generalized Jacobian action ``J dz``."""

        return self.jvp_with_active_set(z, direction, self.active_mask(z))

    def scaled_residual(
        self, z: Any, scales: "MonolithicNewtonScales"
    ) -> np.ndarray:
        """Return ``[Rq/Lq, (M^-1 Rd)/Lrho, RH/LH]``."""

        _validate_model_scale_shapes(self, scales, "scaled residual")
        residual = self.residual(z)
        q, damage_volume, history = self.unpack(residual)
        damage_strong = damage_volume / self._damage_volumes
        result = np.concatenate(
            (
                q / scales.Lq,
                damage_strong / scales.Lrho,
                history / scales.LH,
            )
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("scaled monolithic residual is non-finite")
        return result

    def scaled_jvp(
        self,
        z: Any,
        direction: Any,
        scales: "MonolithicNewtonScales",
        active_mask: Any,
    ) -> np.ndarray:
        """Apply the same frozen row transform to the generalized JVP."""

        _validate_model_scale_shapes(self, scales, "scaled JVP")
        jvp = self.jvp_with_active_set(z, direction, active_mask)
        q, damage_volume, history = self.unpack(jvp)
        damage_strong = damage_volume / self._damage_volumes
        result = np.concatenate(
            (
                q / scales.Lq,
                damage_strong / scales.Lrho,
                history / scales.LH,
            )
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("scaled monolithic JVP is non-finite")
        return result

    def block_scales(
        self, z: Any, gate_config: "MonolithicNewtonGateConfig"
    ) -> "MonolithicNewtonScales":
        """Compute positive row scales from the existing direct-solver gates."""

        _validate_gate_config(gate_config)
        packed = _f64(z, "monolithic scale state", (self._n_packed,))
        _, damage, history = self.unpack(packed)
        view = self._view(damage)
        rhs = np.asarray(view.residual(self._prescribed)[self._free], dtype=np.float64)
        rhs_norm = float(np.linalg.norm(rhs))
        if not np.isfinite(rhs_norm):
            raise FloatingPointError("monolithic mechanics RHS scale is non-finite")
        Lq = max(
            gate_config.mechanics_cg_atol,
            gate_config.mechanics_cg_rtol * rhs_norm,
        )

        graph = np.asarray(self._damage_matrix @ damage, dtype=np.float64)
        local = self._model.Gc / self._model.l0 * damage
        graph_term = self._model.Gc * self._model.l0 * graph
        H_used = np.array(history, dtype=np.float64, copy=True)
        driver = self._amor_driver(view, self.reconstruct_displacement(packed))
        H_next = np.maximum(self._committed_history, driver)
        history_used_damage = 2.0 * H_used * damage
        history_used_constant = 2.0 * H_used
        history_next_damage = 2.0 * H_next * damage
        history_next_constant = 2.0 * H_next
        terms = (
            local,
            graph_term,
            history_used_damage,
            history_used_constant,
            history_next_damage,
            history_next_constant,
        )
        if not all(np.all(np.isfinite(term)) for term in terms):
            raise FloatingPointError("monolithic damage scale terms are non-finite")
        damage_residual_scale = max(
            (_max_abs_array(term) for term in terms), default=0.0
        )
        Lrho = gate_config.damage_residual.limit(damage_residual_scale)

        history_scale = max(
            _max_abs_array(self._committed_history),
            _max_abs_array(H_used),
            _max_abs_array(H_next),
        )
        LH = gate_config.history.limit(history_scale)
        scales = MonolithicNewtonScales(
            mechanics_rhs_norm=rhs_norm,
            damage_residual_scale=damage_residual_scale,
            history_scale=history_scale,
            Lq=Lq,
            Lrho=Lrho,
            LH=LH,
            H_used=H_used,
            H_next=np.asarray(H_next, dtype=np.float64),
            state_digest=_state_digest(packed),
        )
        _validate_model_scale_pair(self, scales, packed, "monolithic block")
        return scales

    # ------------------------------------------------------------------
    # Exact adjoint (transpose) action
    # ------------------------------------------------------------------

    def _endpoint_directions(
        self, view: A2FixedDamageView, displacement: np.ndarray
    ) -> np.ndarray:
        """Raw per-support endpoint stress directions before any weighting.

        This is the exact ``stress -> xi K^{-T}`` contraction used by the
        existing endpoint assembly, kept so the adjoint gathers through the
        identical edge data.
        """

        stress = view._strain_stress(view._deformation_gradient(displacement))[1]
        return np.einsum(
            "eab,ebc,esc->esa",
            stress,
            np.swapaxes(self._static.shape_inverse, 1, 2),
            self._static.xi,
            optimize=True,
        )

    def _jqd_adjoint(self, z: Any, weights: np.ndarray) -> np.ndarray:
        """Exact transpose of ``Jqd`` applied to free-row weights."""

        _, d, _ = self.unpack(z)
        displacement = self.reconstruct_displacement(z)
        view = self._view(d)
        endpoint = self._endpoint_directions(view, displacement)
        weighted = endpoint * self._static.beta[:, :, None]
        full_weights = np.zeros((self._static.n_particles, 2), dtype=np.float64)
        full_weights.reshape(-1)[self._free] = weights
        support = view._safe_support()
        gathered = np.einsum("esa,esa->e", weighted, full_weights[support], optimize=True)
        source = np.einsum(
            "esa,ea->e", weighted, full_weights[self._source], optimize=True
        )
        assembled = gathered - source
        weight = -2.0 * self._material_phase_volumes * (1.0 - self._padded(d))
        edge_adjoint = self._eta * weight[self._source] * assembled
        result = np.zeros(self._static.n_particles, dtype=np.float64)
        np.add.at(result, self._source, edge_adjoint)
        return result[: self._n_material]

    def _jhq_adjoint(self, z: Any, weights: np.ndarray) -> np.ndarray:
        """Exact transpose of ``JHq`` applied to history-row weights."""

        _, d, _ = self.unpack(z)
        active = self.active_mask(z)
        if not np.any(active):
            return np.zeros(self._n_free, dtype=np.float64)
        displacement = self.reconstruct_displacement(z)
        view = self._view(d)
        strain, _ = view._strain_stress(view._deformation_gradient(displacement))
        factor = _amor_first_variation_factor(
            strain, view.lame_lambda, view.shear_modulus
        )
        edge_weight = (
            self._padded(-np.where(active, weights, 0.0))[self._source]
            * self._eta
            * self._material_source
        )
        edge_matrix = edge_weight[:, None, None] * factor
        gathered = np.einsum(
            "eac,esc->esa",
            edge_matrix,
            self._static.gradient_coefficient,
            optimize=True,
        )
        gradient = np.zeros((self._static.n_particles, 2), dtype=np.float64)
        np.add.at(gradient, view._safe_support(), gathered)
        np.add.at(gradient, self._source, -np.sum(gathered, axis=1))
        return np.asarray(gradient.reshape(-1)[self._free], dtype=np.float64)

    def vjp(self, z: Any, weights: Any) -> np.ndarray:
        """Return the exact packed adjoint action ``J^T v``.

        ``Jqq``, ``Jdd``, and ``JdH`` are symmetric; ``Jqd^T`` and ``JHq^T``
        are exact transposes of the existing assemblies.  The correctness
        evidence is an independent-oracle bilinear identity on the manufactured
        cloud, not a dense transpose assembled from production JVP columns.
        """

        vq, vd, vh = self.unpack(
            _f64(weights, "monolithic adjoint weights", (self._n_packed,))
        )
        direction_q = self.jqq_action(z, vq) + self._jhq_adjoint(z, vh)
        direction_d = self._jqd_adjoint(z, vq) + self.jdd_action(z, vd)
        direction_h = self.jdh_action(z, vd) + vh
        blocks = (direction_q, direction_d, direction_h)
        if not all(np.all(np.isfinite(block)) for block in blocks):
            raise FloatingPointError("monolithic VJP block is non-finite")
        return np.concatenate(blocks)

    # ------------------------------------------------------------------
    # Manufactured-test helper
    # ------------------------------------------------------------------

    def _dense_jacobian(self, z: Any) -> np.ndarray:
        """Assemble the full packed Jacobian from production JVP columns.

        Private manufactured-test helper.  The result is assembled from this
        class's own ``jvp`` and is therefore production self-data: it is only
        suitable for structural diagnostics of the production operator (for
        example its nonsymmetry) and must never be presented as independent
        evidence of the JVP's correctness.  Independent evidence comes from
        the test-only oracle that rebuilds every assembly from frozen static
        data.
        """

        self.unpack(z)
        columns = np.empty((self._n_packed, self._n_packed), dtype=np.float64)
        for index in range(self._n_packed):
            unit = np.zeros(self._n_packed, dtype=np.float64)
            unit[index] = 1.0
            columns[:, index] = self.jvp(z, unit)
        return columns


# ---------------------------------------------------------------------------
# Phase 2A scaled semismooth Newton/GMRES core
# ---------------------------------------------------------------------------


def _max_abs_array(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    result = float(np.max(np.abs(array)))
    if not np.isfinite(result):
        raise FloatingPointError("absolute-value scale is non-finite")
    return result


def _state_digest(values: Any) -> str:
    """Return a deterministic digest for one packed float64 solver state."""

    array = _f64(values, "packed state digest")
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _strict_integer(value: Any, name: str, *, positive: bool, zero_allowed: bool) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(raw)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and not zero_allowed and result < 1:
        raise ValueError(f"{name} must be positive")
    if not positive and zero_allowed and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class MonolithicHookstepConfig:
    """Explicit manufactured-only controls for the projected hookstep core."""

    max_krylov_dimension: int
    arnoldi_breakdown_rtol: float
    arnoldi_orthogonality_rtol: float
    metric_rank_rtol: float
    projected_rank_rtol: float
    secular_rtol: float
    prediction_consistency_rtol: float
    metric_consistency_rtol: float
    secular_max_iterations: int
    trust_radius_initial: float
    trust_radius_min: float
    trust_radius_max: float
    max_radius_rejections: int
    eta_accept: float
    eta_expand: float
    gamma_shrink: float
    gamma_expand: float
    boundary_fraction: float

    def __post_init__(self) -> None:
        max_dimension = _strict_integer(
            self.max_krylov_dimension,
            "max_krylov_dimension",
            positive=True,
            zero_allowed=False,
        )
        secular_iterations = _strict_integer(
            self.secular_max_iterations,
            "secular_max_iterations",
            positive=True,
            zero_allowed=False,
        )
        rejections = _strict_integer(
            self.max_radius_rejections,
            "max_radius_rejections",
            positive=False,
            zero_allowed=True,
        )
        positive_controls = (
            "arnoldi_breakdown_rtol",
            "arnoldi_orthogonality_rtol",
            "metric_rank_rtol",
            "projected_rank_rtol",
            "secular_rtol",
            "prediction_consistency_rtol",
            "metric_consistency_rtol",
            "trust_radius_initial",
            "trust_radius_min",
            "trust_radius_max",
        )
        values = {name: _scalar(getattr(self, name), name) for name in positive_controls}
        if any(value <= 0.0 for value in values.values()):
            raise ValueError("hookstep positive controls must be strictly positive")
        for name in (
            "arnoldi_breakdown_rtol",
            "arnoldi_orthogonality_rtol",
            "metric_rank_rtol",
            "projected_rank_rtol",
            "secular_rtol",
            "prediction_consistency_rtol",
            "metric_consistency_rtol",
        ):
            if values[name] >= 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if not (
            values["trust_radius_min"]
            <= values["trust_radius_initial"]
            <= values["trust_radius_max"]
        ):
            raise ValueError(
                "trust radii must satisfy radius_min <= radius_initial <= radius_max"
            )
        eta_accept = _scalar(self.eta_accept, "eta_accept")
        eta_expand = _scalar(self.eta_expand, "eta_expand")
        if not 0.0 < eta_accept < eta_expand < 1.0:
            raise ValueError("hookstep eta controls must satisfy 0 < eta_accept < eta_expand < 1")
        gamma_shrink = _scalar(self.gamma_shrink, "gamma_shrink")
        gamma_expand = _scalar(self.gamma_expand, "gamma_expand")
        if not 0.0 < gamma_shrink < 1.0:
            raise ValueError("gamma_shrink must lie strictly between zero and one")
        if gamma_expand <= 1.0:
            raise ValueError("gamma_expand must exceed one")
        boundary_fraction = _scalar(self.boundary_fraction, "boundary_fraction")
        if not 0.0 < boundary_fraction < 1.0:
            raise ValueError("boundary_fraction must lie strictly between zero and one")
        object.__setattr__(self, "max_krylov_dimension", max_dimension)
        object.__setattr__(self, "secular_max_iterations", secular_iterations)
        object.__setattr__(self, "max_radius_rejections", rejections)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "eta_accept", eta_accept)
        object.__setattr__(self, "eta_expand", eta_expand)
        object.__setattr__(self, "gamma_shrink", gamma_shrink)
        object.__setattr__(self, "gamma_expand", gamma_expand)
        object.__setattr__(self, "boundary_fraction", boundary_fraction)


@dataclass(frozen=True)
class MonolithicTrustMetric:
    """Fixed dimensionless quadrature metric on packed free variables."""

    diagonal: np.ndarray
    n_free: int
    n_material: int
    union_volume: float
    material_volume: float
    displacement_scale: float
    history_scale: float
    model_identity: EngineeringGripModel
    static_identity: object

    def __post_init__(self) -> None:
        n_free = _strict_integer(self.n_free, "metric n_free", positive=True, zero_allowed=False)
        n_material = _strict_integer(
            self.n_material, "metric n_material", positive=True, zero_allowed=False
        )
        diagonal = _f64(
            self.diagonal,
            "hookstep metric diagonal",
            (n_free + 2 * n_material,),
        )
        if np.any(diagonal <= 0.0):
            raise ValueError("hookstep metric diagonal must be strictly positive")
        object.__setattr__(self, "diagonal", _immutable(diagonal))
        object.__setattr__(self, "n_free", n_free)
        object.__setattr__(self, "n_material", n_material)
        if not isinstance(self.model_identity, EngineeringGripModel):
            raise TypeError("metric model_identity has the wrong type")
        if self.static_identity is not self.model_identity.static:
            raise ValueError("metric static identity is not model-paired")
        if n_free != len(self.model_identity.constraints.free_dofs):
            raise ValueError("metric free size is not model-paired")
        if n_material != self.model_identity.n_material:
            raise ValueError("metric material size is not model-paired")
        for name in (
            "union_volume",
            "material_volume",
            "displacement_scale",
            "history_scale",
        ):
            value = _scalar(getattr(self, name), f"metric {name}")
            if value <= 0.0:
                raise ValueError(f"metric {name} must be strictly positive")
            object.__setattr__(self, name, value)

    @property
    def n_packed(self) -> int:
        return int(self.diagonal.size)

    def apply(self, values: Any) -> np.ndarray:
        vector = _f64(values, "hookstep metric vector", (self.n_packed,))
        result = self.diagonal * vector
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("hookstep metric action is non-finite")
        return np.asarray(result, dtype=np.float64)

    def apply_columns(self, values: Any) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.dtype != _FLOAT64 or matrix.ndim != 2 or matrix.shape[0] != self.n_packed:
            raise ValueError(
                "hookstep metric columns must be float64 with packed leading dimension"
            )
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError("hookstep metric columns are non-finite")
        result = self.diagonal[:, None] * matrix
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("hookstep metric column action is non-finite")
        return np.asarray(result, dtype=np.float64)

    def norm(self, values: Any) -> float:
        result = float(np.linalg.norm(self.apply(values)))
        if not np.isfinite(result):
            raise FloatingPointError("hookstep metric norm is non-finite")
        return result


@dataclass(frozen=True)
class MonolithicHookstepArnoldi:
    """Immutable right-preconditioned Arnoldi evidence for one frozen model."""

    beta: float
    basis: np.ndarray
    preconditioned_basis: np.ndarray
    hessenberg: np.ndarray
    metric_basis: np.ndarray
    terminal_basis_vector: np.ndarray | None
    happy_breakdown: bool
    orthogonality_defect: float

    def __post_init__(self) -> None:
        beta = _scalar(self.beta, "hookstep Arnoldi beta")
        if beta <= 0.0:
            raise ValueError("hookstep Arnoldi beta must be strictly positive")
        basis = _f64(self.basis, "hookstep Arnoldi basis")
        preconditioned = _f64(
            self.preconditioned_basis, "hookstep Arnoldi preconditioned basis"
        )
        hessenberg = _f64(self.hessenberg, "hookstep Arnoldi Hessenberg")
        metric_basis = _f64(self.metric_basis, "hookstep Arnoldi metric basis")
        if basis.ndim != 2 or basis.shape[1] == 0:
            raise ValueError("hookstep Arnoldi basis must be a nonempty matrix")
        n, dimension = basis.shape
        if preconditioned.shape != (n, dimension):
            raise ValueError("hookstep preconditioned basis has the wrong shape")
        if hessenberg.shape != (dimension + 1, dimension):
            raise ValueError("hookstep Hessenberg has the wrong shape")
        if metric_basis.shape != (n, dimension):
            raise ValueError("hookstep metric basis has the wrong shape")
        terminal = self.terminal_basis_vector
        if terminal is not None:
            terminal = _f64(
                terminal, "hookstep Arnoldi terminal basis vector", (n,)
            )
            terminal = _immutable(terminal)
        if not isinstance(self.happy_breakdown, (bool, np.bool_)):
            raise TypeError("hookstep Arnoldi happy_breakdown must be boolean")
        happy_breakdown = bool(self.happy_breakdown)
        if not happy_breakdown and terminal is None:
            raise ValueError("non-breakdown Arnoldi evidence requires a terminal vector")
        if terminal is None and np.any(hessenberg[-1, :] != 0.0):
            raise ValueError("Arnoldi evidence omits a nonzero terminal vector")
        audit_basis = basis if terminal is None else np.column_stack((basis, terminal))
        gram = audit_basis.T @ audit_basis
        computed_defect = float(
            np.max(np.abs(gram - np.eye(audit_basis.shape[1], dtype=np.float64)))
        )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "basis", _immutable(basis))
        object.__setattr__(self, "preconditioned_basis", _immutable(preconditioned))
        object.__setattr__(self, "hessenberg", _immutable(hessenberg))
        object.__setattr__(self, "metric_basis", _immutable(metric_basis))
        object.__setattr__(self, "terminal_basis_vector", terminal)
        object.__setattr__(self, "happy_breakdown", happy_breakdown)
        defect = _nonnegative(self.orthogonality_defect, "Arnoldi orthogonality defect")
        if not np.isclose(
            defect,
            computed_defect,
            rtol=0.0,
            atol=16.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("Arnoldi orthogonality defect is not basis-paired")
        object.__setattr__(self, "orthogonality_defect", defect)

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])


@dataclass(frozen=True)
class MonolithicProjectedHookstep:
    """Immutable solution of one QR/SVD-whitened projected trust problem."""

    coefficients: np.ndarray
    trust_multiplier: float
    radius: float
    metric_norm: float
    projected_residual_norm: float
    predicted_reduction: float
    metric_rank: int
    projected_rank: int
    on_boundary: bool
    secular_iterations: int
    radius_tolerance: float

    def __post_init__(self) -> None:
        coefficients = _f64(self.coefficients, "projected hookstep coefficients")
        if coefficients.ndim != 1 or coefficients.size == 0:
            raise ValueError("projected hookstep coefficients must be nonempty")
        object.__setattr__(self, "coefficients", _immutable(coefficients))
        for name in (
            "trust_multiplier",
            "radius",
            "metric_norm",
            "projected_residual_norm",
            "predicted_reduction",
        ):
            value = _nonnegative(getattr(self, name), f"projected hookstep {name}")
            object.__setattr__(self, name, value)
        if self.radius <= 0.0:
            raise ValueError("projected hookstep radius must be strictly positive")
        if self.predicted_reduction <= 0.0:
            raise ValueError("projected hookstep predicted reduction must be positive")
        radius_tolerance = _nonnegative(
            self.radius_tolerance, "projected hookstep radius tolerance"
        )
        if self.metric_norm > self.radius + radius_tolerance:
            raise ValueError("projected hookstep violates its trust radius")
        metric_rank = _strict_integer(
            self.metric_rank, "projected hookstep metric rank", positive=True, zero_allowed=False
        )
        projected_rank = _strict_integer(
            self.projected_rank,
            "projected hookstep projected rank",
            positive=False,
            zero_allowed=True,
        )
        iterations = _strict_integer(
            self.secular_iterations,
            "projected hookstep secular iterations",
            positive=False,
            zero_allowed=True,
        )
        if metric_rank != coefficients.size or projected_rank > coefficients.size:
            raise ValueError("projected hookstep ranks are inconsistent")
        object.__setattr__(self, "metric_rank", metric_rank)
        object.__setattr__(self, "projected_rank", projected_rank)
        object.__setattr__(self, "secular_iterations", iterations)
        if not isinstance(self.on_boundary, (bool, np.bool_)):
            raise TypeError("projected hookstep on_boundary must be boolean")
        on_boundary = bool(self.on_boundary)
        if on_boundary and self.trust_multiplier <= 0.0:
            raise ValueError("boundary hookstep requires a positive multiplier")
        if not on_boundary and self.trust_multiplier != 0.0:
            raise ValueError("interior hookstep requires a zero multiplier")
        object.__setattr__(self, "on_boundary", on_boundary)
        object.__setattr__(self, "radius_tolerance", radius_tolerance)


@dataclass(frozen=True)
class MonolithicHookstepDirection:
    """Combined manufactured hookstep direction and full-model evidence."""

    step: np.ndarray
    metric_norm: float
    full_predicted_reduction: float
    projected_prediction_defect: float
    metric_consistency_defect: float
    prediction_consistency_rtol: float
    metric_consistency_rtol: float
    arnoldi: MonolithicHookstepArnoldi
    projected: MonolithicProjectedHookstep

    def __post_init__(self) -> None:
        step = _f64(self.step, "monolithic hookstep direction")
        if step.ndim != 1 or step.size == 0:
            raise ValueError("monolithic hookstep direction must be nonempty")
        if step.shape != (self.arnoldi.basis.shape[0],):
            raise ValueError("monolithic hookstep direction has the wrong shape")
        reconstructed_step = (
            self.arnoldi.preconditioned_basis @ self.projected.coefficients
        )
        if not np.array_equal(step, reconstructed_step):
            raise ValueError("monolithic hookstep direction is not projected-step-paired")
        object.__setattr__(self, "step", _immutable(step))
        object.__setattr__(
            self, "metric_norm", _nonnegative(self.metric_norm, "hookstep metric norm")
        )
        prediction = _scalar(self.full_predicted_reduction, "hookstep full prediction")
        if prediction <= 0.0:
            raise ValueError("hookstep full predicted reduction must be positive")
        object.__setattr__(self, "full_predicted_reduction", prediction)
        prediction_defect = _nonnegative(
            self.projected_prediction_defect,
            "hookstep projected prediction defect",
        )
        metric_defect = _nonnegative(
            self.metric_consistency_defect,
            "hookstep metric consistency defect",
        )
        prediction_rtol = _scalar(
            self.prediction_consistency_rtol,
            "hookstep prediction consistency rtol",
        )
        metric_rtol = _scalar(
            self.metric_consistency_rtol,
            "hookstep metric consistency rtol",
        )
        if not 0.0 < prediction_rtol < 1.0 or not 0.0 < metric_rtol < 1.0:
            raise ValueError("hookstep consistency tolerances must lie in (0, 1)")
        if prediction_defect > prediction_rtol:
            raise ValueError("hookstep projected/full prediction mismatch")
        if metric_defect > metric_rtol:
            raise ValueError("hookstep reconstructed metric norm mismatch")
        if self.metric_norm > self.projected.radius + self.projected.radius_tolerance:
            raise ValueError("hookstep direction violates the trust radius")
        projected_scale = max(
            self.full_predicted_reduction,
            self.projected.predicted_reduction,
            np.finfo(np.float64).tiny,
        )
        paired_defect = abs(
            self.full_predicted_reduction - self.projected.predicted_reduction
        ) / projected_scale
        if abs(paired_defect - prediction_defect) > 8.0 * np.finfo(np.float64).eps:
            raise ValueError("hookstep prediction defect is not result-paired")
        object.__setattr__(
            self,
            "projected_prediction_defect",
            prediction_defect,
        )
        object.__setattr__(self, "metric_consistency_defect", metric_defect)
        object.__setattr__(self, "prediction_consistency_rtol", prediction_rtol)
        object.__setattr__(self, "metric_consistency_rtol", metric_rtol)


@dataclass(frozen=True)
class MonolithicHookstepFailureAudit:
    """Immutable evidence for a manufactured hookstep numerical failure."""

    stage: str
    reason: str
    radius: float
    beta: float | None
    krylov_dimension: int
    config: MonolithicHookstepConfig
    model_identity: EngineeringGripModel | None
    current_z: np.ndarray | None
    active_mask: np.ndarray | None
    scale_state_digest: str | None
    hessenberg: np.ndarray | None
    metric_singular_values: np.ndarray | None
    projected_singular_values: np.ndarray | None

    def __post_init__(self) -> None:
        if self.stage not in {"metric", "arnoldi", "projected", "prediction", "radius"}:
            raise ValueError("unknown hookstep failure stage")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("hookstep failure reason must be nonempty")
        radius = _scalar(self.radius, "hookstep failure radius")
        if radius <= 0.0:
            raise ValueError("hookstep failure radius must be positive")
        object.__setattr__(self, "radius", radius)
        if self.beta is not None:
            object.__setattr__(self, "beta", _nonnegative(self.beta, "hookstep failure beta"))
        dimension = _strict_integer(
            self.krylov_dimension,
            "hookstep failure Krylov dimension",
            positive=False,
            zero_allowed=True,
        )
        object.__setattr__(self, "krylov_dimension", dimension)
        if not isinstance(self.config, MonolithicHookstepConfig):
            raise TypeError("hookstep failure config has the wrong type")
        if self.model_identity is not None and not isinstance(
            self.model_identity, EngineeringGripModel
        ):
            raise TypeError("hookstep failure model identity has the wrong type")
        for name in (
            "current_z",
            "hessenberg",
            "metric_singular_values",
            "projected_singular_values",
        ):
            value = getattr(self, name)
            if value is not None:
                array = _f64(value, f"hookstep failure {name}")
                object.__setattr__(self, name, _immutable(array))
        if self.current_z is not None and self.current_z.ndim != 1:
            raise ValueError("hookstep failure current_z must be one-dimensional")
        if self.hessenberg is not None and self.hessenberg.ndim != 2:
            raise ValueError("hookstep failure Hessenberg must be two-dimensional")
        for name in ("metric_singular_values", "projected_singular_values"):
            value = getattr(self, name)
            if value is not None and value.ndim != 1:
                raise ValueError(f"hookstep failure {name} must be one-dimensional")
        if self.active_mask is not None:
            active = np.asarray(self.active_mask)
            if active.dtype != np.dtype(bool) or active.ndim != 1:
                raise ValueError("hookstep failure active mask is invalid")
            object.__setattr__(self, "active_mask", _immutable(active))
        if self.model_identity is not None:
            expected_packed = (
                len(self.model_identity.constraints.free_dofs)
                + 2 * self.model_identity.n_material
            )
            if self.current_z is not None and self.current_z.shape != (expected_packed,):
                raise ValueError("hookstep failure current_z is not model-paired")
            if self.active_mask is not None and self.active_mask.shape != (
                self.model_identity.n_material,
            ):
                raise ValueError("hookstep failure active mask is not model-paired")
        if self.scale_state_digest is not None and (
            not isinstance(self.scale_state_digest, str)
            or len(self.scale_state_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.scale_state_digest)
        ):
            raise ValueError("hookstep failure scale digest is invalid")


@dataclass(frozen=True)
class MonolithicHookstepRadiusDecision:
    """Pure trust-radius decision for one finite nonlinear trial ratio."""

    accepted: bool
    expanded: bool
    next_radius: float
    rejection_count: int
    reason: str

    def __post_init__(self) -> None:
        for name in ("accepted", "expanded"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"hookstep radius {name} must be boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))
        radius = _scalar(self.next_radius, "hookstep next radius")
        if radius <= 0.0:
            raise ValueError("hookstep next radius must be positive")
        object.__setattr__(self, "next_radius", radius)
        count = _strict_integer(
            self.rejection_count,
            "hookstep rejection count",
            positive=False,
            zero_allowed=True,
        )
        object.__setattr__(self, "rejection_count", count)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("hookstep radius reason must be nonempty")
        if self.expanded and not self.accepted:
            raise ValueError("a rejected hookstep cannot expand the radius")
        if self.accepted:
            expected_reason = "accepted_expand" if self.expanded else "accepted_retain"
            if count != 0 or self.reason != expected_reason:
                raise ValueError("accepted hookstep radius decision is inconsistent")
        elif count == 0 or self.reason != "rejected_shrink":
            raise ValueError("rejected hookstep radius decision is inconsistent")


@dataclass(frozen=True)
class MonolithicHookstepTrial:
    """Immutable evidence for one nonlinear trial at a fixed radius."""

    index: int
    radius: float
    direction: MonolithicHookstepDirection
    outcome: str
    failure_reason: str | None
    candidate_z: np.ndarray | None
    trial_scaled_residual: np.ndarray | None
    actual_reduction: float | None
    rho: float | None
    radius_decision: MonolithicHookstepRadiusDecision | None

    def __post_init__(self) -> None:
        index = _strict_integer(
            self.index, "hookstep trial index", positive=False, zero_allowed=True
        )
        radius = _scalar(self.radius, "hookstep trial radius")
        if radius <= 0.0:
            raise ValueError("hookstep trial radius must be positive")
        if not isinstance(self.direction, MonolithicHookstepDirection):
            raise TypeError("hookstep trial direction has the wrong type")
        if self.direction.projected.radius != radius:
            raise ValueError("hookstep trial direction is not radius-paired")
        if self.outcome not in {
            "accepted",
            "accepted_pending_rebuild",
            "rejected_ratio",
            "rejected_nonfinite_trial",
            "failed_ratio",
        }:
            raise ValueError("unknown hookstep trial outcome")
        if self.radius_decision is not None and not isinstance(
            self.radius_decision, MonolithicHookstepRadiusDecision
        ):
            raise TypeError("hookstep trial radius decision has the wrong type")
        candidate = self.candidate_z
        residual = self.trial_scaled_residual
        actual = self.actual_reduction
        ratio = self.rho
        if candidate is not None:
            candidate = _immutable(_f64(candidate, "hookstep trial candidate"))
        if residual is not None:
            residual = _immutable(_f64(residual, "hookstep trial scaled residual"))
        if self.outcome == "rejected_nonfinite_trial":
            if actual is not None or ratio is not None:
                raise ValueError("non-finite hookstep trial carries finite reduction evidence")
            if self.radius_decision is not None and self.radius_decision.accepted:
                raise ValueError("non-finite hookstep trial cannot be accepted")
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError("non-finite hookstep trial lacks its failure reason")
        elif self.outcome == "failed_ratio":
            if candidate is None or residual is None or actual is None or ratio is not None:
                raise ValueError("failed hookstep ratio evidence is incomplete")
            actual = _scalar(actual, "hookstep trial actual reduction")
            if self.radius_decision is not None:
                raise ValueError("failed hookstep ratio cannot carry a radius decision")
            if not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError("failed hookstep ratio lacks its failure reason")
        else:
            if self.failure_reason is not None:
                raise ValueError("finite hookstep trial cannot carry a failure reason")
            if candidate is None or residual is None or actual is None or ratio is None:
                raise ValueError("finite hookstep trial evidence is incomplete")
            actual = _scalar(actual, "hookstep trial actual reduction")
            ratio = _scalar(ratio, "hookstep trial reduction ratio")
            accepted = self.outcome in {"accepted", "accepted_pending_rebuild"}
            if self.radius_decision is None:
                raise ValueError("finite hookstep trial lacks its radius decision")
            if self.radius_decision.accepted != accepted:
                raise ValueError("hookstep trial outcome and radius decision disagree")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "candidate_z", candidate)
        object.__setattr__(self, "trial_scaled_residual", residual)
        object.__setattr__(self, "actual_reduction", actual)
        object.__setattr__(self, "rho", ratio)


@dataclass(frozen=True)
class MonolithicHookstepCorrection:
    """One exact-root or accepted correction from a single frozen model."""

    status: str
    model_identity: EngineeringGripModel
    z_before: np.ndarray
    linearization_scales: "MonolithicNewtonScales"
    config: MonolithicHookstepConfig
    active_mask_before: np.ndarray
    reference_scaled_residual: np.ndarray
    metric: MonolithicTrustMetric | None
    arnoldi: MonolithicHookstepArnoldi | None
    attempts: tuple[MonolithicHookstepTrial, ...]
    z_after: np.ndarray
    active_mask_after: np.ndarray
    next_radius: float
    active_set_changed: bool
    rebuild_required: bool

    def __post_init__(self) -> None:
        if self.status not in {"exact_root", "accepted"}:
            raise ValueError("unknown hookstep correction status")
        if not isinstance(self.model_identity, EngineeringGripModel):
            raise TypeError("hookstep correction model identity has the wrong type")
        if not isinstance(self.linearization_scales, MonolithicNewtonScales):
            raise TypeError("hookstep correction scales have the wrong type")
        if not isinstance(self.config, MonolithicHookstepConfig):
            raise TypeError("hookstep correction config has the wrong type")
        n_material = self.model_identity.n_material
        n_packed = len(self.model_identity.constraints.free_dofs) + 2 * n_material
        before = _f64(self.z_before, "hookstep correction z_before", (n_packed,))
        after = _f64(self.z_after, "hookstep correction z_after", (n_packed,))
        residual = _f64(
            self.reference_scaled_residual,
            "hookstep correction reference residual",
            (n_packed,),
        )
        _validate_scale_pair(
            self.linearization_scales,
            before,
            "hookstep correction",
            expected_n_material=n_material,
            expected_packed_size=n_packed,
        )
        active_before = np.asarray(self.active_mask_before)
        active_after = np.asarray(self.active_mask_after)
        if (
            active_before.dtype != np.dtype(bool)
            or active_after.dtype != np.dtype(bool)
            or active_before.shape != (n_material,)
            or active_after.shape != (n_material,)
        ):
            raise ValueError("hookstep correction active masks are invalid")
        if not isinstance(self.attempts, tuple) or not all(
            isinstance(attempt, MonolithicHookstepTrial) for attempt in self.attempts
        ):
            raise TypeError("hookstep correction attempts must be a trial tuple")
        for expected_index, attempt in enumerate(self.attempts):
            if attempt.index != expected_index:
                raise ValueError("hookstep correction trial indices are not contiguous")
            if self.arnoldi is None or attempt.direction.arnoldi is not self.arnoldi:
                raise ValueError("hookstep correction did not reuse one Arnoldi model")
            if attempt.candidate_z is not None and not np.array_equal(
                attempt.candidate_z, before + attempt.direction.step
            ):
                raise ValueError("hookstep trial candidate is not state-paired")
            expected_radius = (
                self.config.trust_radius_initial
                if expected_index == 0
                else (
                    self.attempts[expected_index - 1].radius_decision.next_radius
                    if self.attempts[expected_index - 1].radius_decision is not None
                    else np.nan
                )
            )
            if attempt.radius != expected_radius:
                raise ValueError("hookstep correction trial radii are not contiguous")
            if attempt.radius_decision is None:
                raise ValueError("successful hookstep correction contains a terminal trial")
            if attempt.outcome == "rejected_ratio" and (
                attempt.rho is None or attempt.rho >= self.config.eta_accept
            ):
                raise ValueError("hookstep ratio rejection violates eta_accept")
            if attempt.outcome in {"accepted", "accepted_pending_rebuild"} and (
                attempt.rho is None or attempt.rho < self.config.eta_accept
            ):
                raise ValueError("hookstep accepted trial violates eta_accept")
        next_radius = _scalar(self.next_radius, "hookstep correction next radius")
        if next_radius <= 0.0:
            raise ValueError("hookstep correction next radius must be positive")
        for name in ("active_set_changed", "rebuild_required"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"hookstep correction {name} must be boolean")
        changed = bool(self.active_set_changed)
        rebuild = bool(self.rebuild_required)
        if changed != (not np.array_equal(active_before, active_after)):
            raise ValueError("hookstep active-set change flag is not mask-paired")
        if self.status == "exact_root":
            if self.metric is not None or self.arnoldi is not None or self.attempts:
                raise ValueError("exact-root hookstep correction cannot carry trial data")
            if np.any(residual != 0.0) or not np.array_equal(before, after):
                raise ValueError("exact-root hookstep correction is not state-paired")
            if changed or rebuild:
                raise ValueError("exact-root hookstep correction cannot require rebuild")
            if next_radius != self.config.trust_radius_initial:
                raise ValueError("exact-root hookstep radius is not initial-radius-paired")
        else:
            if not isinstance(self.metric, MonolithicTrustMetric) or not isinstance(
                self.arnoldi, MonolithicHookstepArnoldi
            ):
                raise TypeError("accepted hookstep correction lacks frozen model data")
            if not self.attempts or self.attempts[-1].outcome != "accepted":
                raise ValueError("accepted hookstep correction lacks an accepted trial")
            if any(attempt.outcome == "accepted" for attempt in self.attempts[:-1]):
                raise ValueError("hookstep correction continued after acceptance")
            final = self.attempts[-1]
            if final.candidate_z is None or not np.array_equal(after, final.candidate_z):
                raise ValueError("accepted hookstep correction is not trial-paired")
            if final.radius_decision.next_radius != next_radius:
                raise ValueError("accepted hookstep next radius is not decision-paired")
            if not rebuild:
                raise ValueError("accepted hookstep correction must require rebuild")
            if self.metric.model_identity is not self.model_identity:
                raise ValueError("accepted hookstep metric is not model-paired")
            reference_norm = _stable_l2_norm(
                residual, "hookstep correction reference residual"
            )
            if reference_norm != self.arnoldi.beta or not np.array_equal(
                residual / reference_norm, self.arnoldi.basis[:, 0]
            ):
                raise ValueError("accepted hookstep Arnoldi model is not residual-paired")
        object.__setattr__(self, "z_before", _immutable(before))
        object.__setattr__(self, "z_after", _immutable(after))
        object.__setattr__(self, "reference_scaled_residual", _immutable(residual))
        object.__setattr__(self, "active_mask_before", _immutable(active_before))
        object.__setattr__(self, "active_mask_after", _immutable(active_after))
        object.__setattr__(self, "next_radius", next_radius)
        object.__setattr__(self, "active_set_changed", changed)
        object.__setattr__(self, "rebuild_required", rebuild)


class MonolithicHookstepFailure(RuntimeError):
    """Manufactured hookstep numerical failure with immutable audit."""

    def __init__(self, message: str, audit: MonolithicHookstepFailureAudit):
        super().__init__(message)
        if not isinstance(audit, MonolithicHookstepFailureAudit):
            raise TypeError("hookstep failure audit has the wrong type")
        self.audit = audit


@dataclass(frozen=True)
class MonolithicHookstepControllerFailureAudit:
    """Immutable rejected-trial history for a terminal controller failure."""

    reason: str
    inner_audit: MonolithicHookstepFailureAudit
    metric: MonolithicTrustMetric
    arnoldi: MonolithicHookstepArnoldi
    attempts: tuple[MonolithicHookstepTrial, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("hookstep controller failure reason must be nonempty")
        if not isinstance(self.inner_audit, MonolithicHookstepFailureAudit):
            raise TypeError("hookstep controller inner audit has the wrong type")
        if not isinstance(self.metric, MonolithicTrustMetric):
            raise TypeError("hookstep controller metric has the wrong type")
        if not isinstance(self.arnoldi, MonolithicHookstepArnoldi):
            raise TypeError("hookstep controller Arnoldi evidence has the wrong type")
        if (
            self.inner_audit.model_identity is None
            or self.metric.model_identity is not self.inner_audit.model_identity
            or self.inner_audit.current_z is None
            or self.inner_audit.active_mask is None
            or self.inner_audit.scale_state_digest is None
        ):
            raise ValueError("hookstep controller failure lacks owner-paired evidence")
        if not isinstance(self.attempts, tuple) or not all(
            isinstance(attempt, MonolithicHookstepTrial) for attempt in self.attempts
        ):
            raise TypeError("hookstep controller attempts must be a trial tuple")
        for index, attempt in enumerate(self.attempts):
            if attempt.index != index or attempt.direction.arnoldi is not self.arnoldi:
                raise ValueError("hookstep controller failure trials are not model-paired")
            if attempt.outcome == "accepted":
                raise ValueError("hookstep controller failure cannot follow acceptance")
            if attempt.outcome == "accepted_pending_rebuild" and index != len(
                self.attempts
            ) - 1:
                raise ValueError("pending hookstep rebuild must be the terminal trial")
            if attempt.outcome == "failed_ratio" and index != len(self.attempts) - 1:
                raise ValueError("failed hookstep ratio must be the terminal trial")
            if attempt.candidate_z is not None and not np.array_equal(
                attempt.candidate_z,
                self.inner_audit.current_z + attempt.direction.step,
            ):
                raise ValueError("hookstep controller failure trial is not state-paired")
            expected_radius = (
                self.inner_audit.config.trust_radius_initial
                if index == 0
                else (
                    self.attempts[index - 1].radius_decision.next_radius
                    if self.attempts[index - 1].radius_decision is not None
                    else np.nan
                )
            )
            if attempt.radius != expected_radius:
                raise ValueError("hookstep controller failure radii are not contiguous")
            if attempt.radius_decision is None and index != len(self.attempts) - 1:
                raise ValueError("terminal hookstep trial must end controller evidence")


class MonolithicHookstepControllerFailure(RuntimeError):
    """Terminal manufactured controller failure with rejected-trial evidence."""

    def __init__(self, message: str, audit: MonolithicHookstepControllerFailureAudit):
        super().__init__(message)
        if not isinstance(audit, MonolithicHookstepControllerFailureAudit):
            raise TypeError("hookstep controller failure audit has the wrong type")
        self.audit = audit


@dataclass(frozen=True)
class MonolithicNewtonConfig:
    """Required numerical controls for one Phase-2A solve.

    There are intentionally no constructor defaults.  This object contains
    only nonlinear/linear-solver and Armijo controls; existing physical gate
    tolerances are supplied separately through
    :class:`MonolithicNewtonGateConfig`.
    """

    max_newton_iterations: int
    gmres_rtol: float
    gmres_atol: float
    gmres_restart: int
    gmres_maxiter: int
    armijo_c: float
    line_search_reduction: float
    line_search_max_backtracks: int
    line_search_alpha_min: float

    def __post_init__(self) -> None:
        max_iterations = _strict_integer(
            self.max_newton_iterations,
            "max_newton_iterations",
            positive=True,
            zero_allowed=False,
        )
        gmres_restart = _strict_integer(
            self.gmres_restart,
            "gmres_restart",
            positive=True,
            zero_allowed=False,
        )
        gmres_maxiter = _strict_integer(
            self.gmres_maxiter,
            "gmres_maxiter",
            positive=True,
            zero_allowed=False,
        )
        backtracks = _strict_integer(
            self.line_search_max_backtracks,
            "line_search_max_backtracks",
            positive=False,
            zero_allowed=True,
        )
        gmres_rtol = _nonnegative(self.gmres_rtol, "gmres_rtol")
        gmres_atol = _nonnegative(self.gmres_atol, "gmres_atol")
        if gmres_rtol == 0.0 and gmres_atol == 0.0:
            raise ValueError("gmres_rtol and gmres_atol cannot both be zero")
        armijo_c = _scalar(self.armijo_c, "armijo_c")
        if not 0.0 < armijo_c < 1.0:
            raise ValueError("armijo_c must lie strictly between zero and one")
        reduction = _scalar(self.line_search_reduction, "line_search_reduction")
        if not 0.0 < reduction < 1.0:
            raise ValueError(
                "line_search_reduction must lie strictly between zero and one"
            )
        alpha_min = _scalar(self.line_search_alpha_min, "line_search_alpha_min")
        if not 0.0 < alpha_min <= 1.0:
            raise ValueError("line_search_alpha_min must lie in (0, 1]")
        object.__setattr__(self, "max_newton_iterations", max_iterations)
        object.__setattr__(self, "gmres_rtol", gmres_rtol)
        object.__setattr__(self, "gmres_atol", gmres_atol)
        object.__setattr__(self, "gmres_restart", gmres_restart)
        object.__setattr__(self, "gmres_maxiter", gmres_maxiter)
        object.__setattr__(self, "armijo_c", armijo_c)
        object.__setattr__(self, "line_search_reduction", reduction)
        object.__setattr__(self, "line_search_max_backtracks", backtracks)
        object.__setattr__(self, "line_search_alpha_min", alpha_min)


@dataclass(frozen=True)
class MonolithicNewtonGateConfig:
    """Explicit gate data reused by the existing engineering-grip solver.

    ``mechanics_cg_rtol`` and ``mechanics_cg_atol`` define the existing
    mechanics limit ``max(atol, rtol*||rhs||)``.  The three correction gates
    retain the existing displacement/damage/history scale semantics.  The
    strong damage residual and history row scales use the corresponding
    existing `GateTolerance` objects.
    """

    mechanics_cg_rtol: float
    mechanics_cg_atol: float
    displacement: GateTolerance
    damage: GateTolerance
    history: GateTolerance
    damage_residual: GateTolerance

    def __post_init__(self) -> None:
        cg_rtol = _nonnegative(self.mechanics_cg_rtol, "mechanics_cg_rtol")
        cg_atol = _nonnegative(self.mechanics_cg_atol, "mechanics_cg_atol")
        if cg_rtol == 0.0 and cg_atol == 0.0:
            raise ValueError(
                "mechanics_cg_rtol and mechanics_cg_atol cannot both be zero"
            )
        for name in ("displacement", "damage", "history", "damage_residual"):
            if not isinstance(getattr(self, name), GateTolerance):
                raise TypeError(f"{name} must be a GateTolerance instance")
        object.__setattr__(self, "mechanics_cg_rtol", cg_rtol)
        object.__setattr__(self, "mechanics_cg_atol", cg_atol)

    @classmethod
    def from_engineering_config(cls, engineering_config: Any) -> "MonolithicNewtonGateConfig":
        """Copy gate fields from an already validated direct-solver config."""

        required = (
            "cg_rtol",
            "cg_atol",
            "displacement",
            "damage",
            "history",
            "damage_residual",
        )
        for name in required:
            if not hasattr(engineering_config, name):
                raise TypeError(
                    "engineering_config is missing the existing gate field "
                    f"{name}"
                )
        return cls(
            mechanics_cg_rtol=engineering_config.cg_rtol,
            mechanics_cg_atol=engineering_config.cg_atol,
            displacement=engineering_config.displacement,
            damage=engineering_config.damage,
            history=engineering_config.history,
            damage_residual=engineering_config.damage_residual,
        )


@dataclass(frozen=True)
class MonolithicNewtonScales:
    """Frozen gate scales used by one linearization and its merit."""

    mechanics_rhs_norm: float
    damage_residual_scale: float
    history_scale: float
    Lq: float
    Lrho: float
    LH: float
    H_used: np.ndarray
    H_next: np.ndarray
    state_digest: str

    def __post_init__(self) -> None:
        for name in (
            "mechanics_rhs_norm",
            "damage_residual_scale",
            "history_scale",
        ):
            value = _scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("Lq", "Lrho", "LH"):
            value = _scalar(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        H_used = _f64(self.H_used, "H_used scale state")
        H_next = _f64(self.H_next, "H_next scale state")
        if H_used.ndim != 1 or H_next.ndim != 1 or H_used.size == 0:
            raise ValueError("H_used and H_next scale states must be nonempty vectors")
        if H_used.shape != H_next.shape:
            raise ValueError("H_used and H_next scale states must have equal shapes")
        object.__setattr__(self, "H_used", _immutable(H_used))
        object.__setattr__(self, "H_next", _immutable(H_next))
        if not isinstance(self.state_digest, str) or len(self.state_digest) != 64:
            raise ValueError("state_digest must be a 64-character hexadecimal string")
        try:
            int(self.state_digest, 16)
        except ValueError as error:
            raise ValueError("state_digest must be hexadecimal") from error

    @property
    def mechanics_limit(self) -> float:
        return self.Lq

    @property
    def damage_limit(self) -> float:
        return self.Lrho

    @property
    def history_limit(self) -> float:
        return self.LH


def _validate_gate_config(value: Any) -> None:
    if not isinstance(value, MonolithicNewtonGateConfig):
        raise TypeError("gate_config must be a MonolithicNewtonGateConfig")


def _validate_scales(value: Any) -> None:
    if not isinstance(value, MonolithicNewtonScales):
        raise TypeError("scales must be a MonolithicNewtonScales")


def _validate_model_scale_shapes(
    monolithic: MonolithicGripResidual,
    scales: MonolithicNewtonScales,
    name: str,
) -> None:
    """Validate history-vector shapes for any model-bound scale consumer."""

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("monolithic scale owner has the wrong type")
    _validate_scales(scales)
    expected_history_shape = (monolithic.n_material,)
    if scales.H_used.shape != expected_history_shape:
        raise ValueError(
            f"{name} H_used scale state must match monolithic.n_material "
            f"{monolithic.n_material}, got shape {scales.H_used.shape}"
        )
    if scales.H_next.shape != expected_history_shape:
        raise ValueError(
            f"{name} H_next scale state must match monolithic.n_material "
            f"{monolithic.n_material}, got shape {scales.H_next.shape}"
        )


def _validate_scale_pair(
    scales: MonolithicNewtonScales,
    state: Any,
    name: str,
    *,
    expected_n_material: int | None = None,
    expected_packed_size: int | None = None,
) -> None:
    """Validate that one scale bundle belongs to the supplied packed state."""

    _validate_scales(scales)
    if expected_n_material is not None:
        if expected_n_material <= 0:
            raise ValueError("expected_n_material must be positive")
        expected_history_shape = (int(expected_n_material),)
        if scales.H_used.shape != expected_history_shape:
            raise ValueError(
                f"{name} H_used scale state must match monolithic.n_material "
                f"{expected_n_material}, got shape {scales.H_used.shape}"
            )
        if scales.H_next.shape != expected_history_shape:
            raise ValueError(
                f"{name} H_next scale state must match monolithic.n_material "
                f"{expected_n_material}, got shape {scales.H_next.shape}"
            )
    packed = _f64(state, f"{name} state")
    if expected_packed_size is not None and packed.shape != (int(expected_packed_size),):
        raise ValueError(
            f"{name} packed state must have shape ({int(expected_packed_size)},), "
            f"got {packed.shape}"
        )
    if scales.state_digest != _state_digest(packed):
        raise ValueError(f"{name} scales are paired with a different packed state")
    if packed.size < scales.H_used.size:
        raise ValueError(f"{name} scales have an incompatible packed-state shape")
    if not np.array_equal(packed[-scales.H_used.size :], scales.H_used):
        raise ValueError(f"{name} H_used scale state is not paired with the packed state")


def _validate_model_scale_pair(
    monolithic: MonolithicGripResidual,
    scales: MonolithicNewtonScales,
    state: Any,
    name: str,
) -> None:
    """Validate a scale bundle against one model-bound packed state.

    ``MonolithicNewtonScales`` intentionally remains generically constructible
    for dataclass-level tests and serialization.  Every production use through
    a :class:`MonolithicGripResidual` must, however, bind both history vectors
    and the packed-state digest to that model's exact fixed shapes.
    """

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("monolithic scale owner has the wrong type")
    _validate_model_scale_shapes(monolithic, scales, name)
    _validate_scale_pair(
        scales,
        state,
        name,
        expected_n_material=monolithic.n_material,
        expected_packed_size=monolithic.n_packed,
    )


def _scales_equal(
    first: MonolithicNewtonScales, second: MonolithicNewtonScales
) -> bool:
    return bool(
        first.mechanics_rhs_norm == second.mechanics_rhs_norm
        and first.damage_residual_scale == second.damage_residual_scale
        and first.history_scale == second.history_scale
        and first.Lq == second.Lq
        and first.Lrho == second.Lrho
        and first.LH == second.LH
        and np.array_equal(first.H_used, second.H_used)
        and np.array_equal(first.H_next, second.H_next)
        and first.state_digest == second.state_digest
    )


@dataclass(frozen=True)
class MonolithicNewtonDiagnostics:
    """Immutable audit row for one accepted Newton correction."""

    iteration: int
    z_before: np.ndarray
    z_after: np.ndarray
    delta: np.ndarray
    active_mask_before: np.ndarray
    active_mask_after: np.ndarray
    active_set_stable: bool
    linearization_scales: MonolithicNewtonScales
    accepted_trial_scales: MonolithicNewtonScales
    reference_scaled_residual: np.ndarray
    line_search_scaled_residual: np.ndarray
    accepted_trial_scaled_residual: np.ndarray
    scaled_mechanics_l2: float
    scaled_damage_inf: float
    scaled_history_inf: float
    merit_reference: float
    merit_trial: float
    gmres_info: int
    gmres_unpreconditioned_residual: float
    gmres_residual_limit: float
    gmres_callback_residuals: tuple[float, ...]
    trial_alphas: tuple[float, ...]
    line_search_rejections: tuple[str, ...]
    accepted_alpha: float
    line_search_backtracks: int
    correction_mechanics_material_inf: float
    correction_mechanics_union_inf: float
    correction_damage_inf: float
    correction_history_inf: float
    correction_displacement_pass: bool
    correction_damage_pass: bool
    correction_history_pass: bool
    correction_pass: bool

    def __post_init__(self) -> None:
        iteration = _strict_integer(
            self.iteration, "Newton diagnostic iteration", positive=True, zero_allowed=False
        )
        object.__setattr__(self, "iteration", iteration)
        for name in ("z_before", "z_after", "delta"):
            value = _f64(getattr(self, name), f"Newton diagnostic {name}")
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"Newton diagnostic {name} must be a nonempty vector")
            object.__setattr__(self, name, _immutable(value))
        for name in ("active_mask_before", "active_mask_after"):
            value = np.asarray(getattr(self, name))
            if value.dtype != np.dtype(bool) or value.ndim != 1:
                raise ValueError(
                    f"Newton diagnostic {name} must be a one-dimensional bool array"
                )
            object.__setattr__(self, name, _immutable(value))
        if self.z_before.shape != self.z_after.shape or self.delta.shape != self.z_before.shape:
            raise ValueError("Newton diagnostic packed vectors have incompatible shapes")
        if self.active_mask_before.shape != self.active_mask_after.shape:
            raise ValueError("Newton diagnostic active masks have incompatible shapes")
        for name in (
            "reference_scaled_residual",
            "line_search_scaled_residual",
            "accepted_trial_scaled_residual",
        ):
            value = _f64(
                getattr(self, name),
                f"Newton diagnostic {name}",
                self.z_before.shape,
            )
            object.__setattr__(self, name, _immutable(value))
        _validate_scale_pair(
            self.linearization_scales,
            self.z_before,
            "Newton diagnostic linearization",
        )
        _validate_scale_pair(
            self.accepted_trial_scales,
            self.z_after,
            "Newton diagnostic accepted-trial",
        )
        if self.active_mask_before.shape != self.linearization_scales.H_used.shape:
            raise ValueError(
                "Newton diagnostic active_mask_before does not match linearization scales"
            )
        if self.active_mask_after.shape != self.accepted_trial_scales.H_used.shape:
            raise ValueError(
                "Newton diagnostic active_mask_after does not match accepted-trial scales"
            )
        for name in (
            "scaled_mechanics_l2",
            "scaled_damage_inf",
            "scaled_history_inf",
            "merit_reference",
            "merit_trial",
            "gmres_unpreconditioned_residual",
            "gmres_residual_limit",
            "accepted_alpha",
            "correction_mechanics_material_inf",
            "correction_mechanics_union_inf",
            "correction_damage_inf",
            "correction_history_inf",
        ):
            value = _scalar(getattr(self, name), f"Newton diagnostic {name}")
            if value < 0.0 and name not in ():
                raise ValueError(f"Newton diagnostic {name} must be non-negative")
            object.__setattr__(self, name, value)
        # GMRES uses negative values for an illegal breakdown status.  Retain
        # that status in diagnostics rather than rejecting useful failure data.
        raw_info = np.asarray(self.gmres_info)
        if raw_info.ndim != 0 or not np.issubdtype(raw_info.dtype, np.integer):
            raise ValueError("Newton diagnostic gmres_info must be an integer")
        object.__setattr__(self, "gmres_info", int(raw_info))
        if not 0.0 < self.accepted_alpha <= 1.0:
            raise ValueError("Newton diagnostic accepted_alpha is invalid")
        backtracks = _strict_integer(
            self.line_search_backtracks,
            "Newton diagnostic line_search_backtracks",
            positive=False,
            zero_allowed=True,
        )
        object.__setattr__(self, "line_search_backtracks", backtracks)
        callbacks = tuple(float(value) for value in self.gmres_callback_residuals)
        alphas = tuple(float(value) for value in self.trial_alphas)
        rejections = tuple(str(value) for value in self.line_search_rejections)
        if len(alphas) != len(rejections):
            raise ValueError("Newton diagnostic line-search audit lengths differ")
        if not all(np.isfinite(value) and value >= 0.0 for value in callbacks):
            raise ValueError("Newton diagnostic GMRES callbacks must be finite")
        if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in alphas):
            raise ValueError("Newton diagnostic trial alphas are invalid")
        object.__setattr__(self, "gmres_callback_residuals", callbacks)
        object.__setattr__(self, "trial_alphas", alphas)
        object.__setattr__(self, "line_search_rejections", rejections)
        for name in (
            "active_set_stable",
            "correction_displacement_pass",
            "correction_damage_pass",
            "correction_history_pass",
            "correction_pass",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))

    @property
    def scaled_block_norms(self) -> tuple[float, float, float]:
        return (
            self.scaled_mechanics_l2,
            self.scaled_damage_inf,
            self.scaled_history_inf,
        )

    @property
    def gmres_callback_type(self) -> str:
        return "pr_norm"


_FAILURE_STAGES = frozenset(
    {
        "initial",
        "scales",
        "preconditioner",
        "gmres",
        "gmres_direction",
        "gmres_residual",
        "raw_trial",
        "line_search",
        "accepted_trial",
        "max_newton_iterations",
    }
)


def _optional_nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = _scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class MonolithicNewtonFailureAudit:
    """Immutable partial audit retained for every numerical Newton failure."""

    stage: str
    target_u_bar: float
    model_identity: EngineeringGripModel
    static_identity: object
    config: MonolithicNewtonConfig
    gate_config: MonolithicNewtonGateConfig
    reason: str
    iteration: int
    current_z: np.ndarray
    active_mask: np.ndarray
    linearization_scales: MonolithicNewtonScales | None
    gmres_info: int | None
    gmres_callback_residuals: tuple[float, ...]
    gmres_unpreconditioned_residual: float | None
    gmres_residual_limit: float | None
    attempted_alphas: tuple[float, ...]
    rejection_reasons: tuple[str, ...]
    completed_diagnostics: tuple[MonolithicNewtonDiagnostics, ...]

    def __post_init__(self) -> None:
        if self.stage not in _FAILURE_STAGES:
            raise ValueError(f"unknown Newton failure stage {self.stage!r}")
        target = _scalar(self.target_u_bar, "Newton failure target_u_bar")
        if not isinstance(self.model_identity, EngineeringGripModel):
            raise TypeError("Newton failure model identity has the wrong type")
        if self.static_identity is not self.model_identity.static:
            raise ValueError("Newton failure static identity is not model-paired")
        if not isinstance(self.config, MonolithicNewtonConfig):
            raise TypeError("Newton failure config has the wrong type")
        if not isinstance(self.gate_config, MonolithicNewtonGateConfig):
            raise TypeError("Newton failure gate_config has the wrong type")
        object.__setattr__(self, "target_u_bar", target)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Newton failure reason must be a nonempty string")
        iteration = _strict_integer(
            self.iteration,
            "Newton failure iteration",
            positive=False,
            zero_allowed=True,
        )
        object.__setattr__(self, "iteration", iteration)
        current = _f64(self.current_z, "Newton failure current_z")
        if current.ndim != 1 or current.size == 0:
            raise ValueError("Newton failure current_z must be a nonempty vector")
        object.__setattr__(self, "current_z", _immutable(current))
        active = np.asarray(self.active_mask)
        if active.dtype != np.dtype(bool) or active.ndim != 1 or active.size == 0:
            raise ValueError("Newton failure active_mask must be a nonempty bool vector")
        object.__setattr__(self, "active_mask", _immutable(active))
        if self.linearization_scales is not None:
            _validate_scale_pair(
                self.linearization_scales,
                current,
                "Newton failure linearization",
            )
            if active.shape != self.linearization_scales.H_used.shape:
                raise ValueError(
                    "Newton failure active_mask does not match linearization scales"
                )
        if self.gmres_info is not None:
            raw_info = np.asarray(self.gmres_info)
            if raw_info.ndim != 0 or not np.issubdtype(raw_info.dtype, np.integer):
                raise ValueError("Newton failure gmres_info must be an integer")
            object.__setattr__(self, "gmres_info", int(raw_info))
        callbacks = tuple(float(value) for value in self.gmres_callback_residuals)
        if not all(np.isfinite(value) and value >= 0.0 for value in callbacks):
            raise ValueError("Newton failure GMRES callbacks must be finite")
        object.__setattr__(self, "gmres_callback_residuals", callbacks)
        object.__setattr__(
            self,
            "gmres_unpreconditioned_residual",
            _optional_nonnegative(
                self.gmres_unpreconditioned_residual,
                "Newton failure GMRES residual",
            ),
        )
        object.__setattr__(
            self,
            "gmres_residual_limit",
            _optional_nonnegative(
                self.gmres_residual_limit,
                "Newton failure GMRES residual limit",
            ),
        )
        alphas = tuple(float(value) for value in self.attempted_alphas)
        rejections = tuple(str(value) for value in self.rejection_reasons)
        if len(alphas) != len(rejections):
            raise ValueError("Newton failure line-search audit lengths differ")
        if not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in alphas):
            raise ValueError("Newton failure attempted alphas are invalid")
        object.__setattr__(self, "attempted_alphas", alphas)
        object.__setattr__(self, "rejection_reasons", rejections)
        completed = tuple(self.completed_diagnostics)
        if not all(isinstance(row, MonolithicNewtonDiagnostics) for row in completed):
            raise TypeError("Newton failure completed diagnostics contain a wrong type")
        object.__setattr__(self, "completed_diagnostics", completed)


@dataclass(frozen=True)
class MonolithicNewtonResult:
    """Immutable successful Phase-2A result and complete solver audit."""

    target_u_bar: float
    model_identity: EngineeringGripModel
    static_identity: object
    committed_history: np.ndarray
    z: np.ndarray
    initial_z: np.ndarray
    initial_residual: np.ndarray
    initial_residual_zero: bool
    converged: bool
    convergence_status: str
    convergence_reason: str
    iterations: int
    config: MonolithicNewtonConfig
    gate_config: MonolithicNewtonGateConfig
    final_active_mask: np.ndarray
    final_accepted_trial_scales: MonolithicNewtonScales
    final_scaled_residual: np.ndarray
    final_scaled_block_norms: tuple[float, float, float]
    final_merit: float
    diagnostics: tuple[MonolithicNewtonDiagnostics, ...]

    def __post_init__(self) -> None:
        target = _scalar(self.target_u_bar, "Newton result target_u_bar")
        if not isinstance(self.model_identity, EngineeringGripModel):
            raise TypeError("Newton result model identity has the wrong type")
        if self.static_identity is not self.model_identity.static:
            raise ValueError("Newton result static identity is not model-paired")
        object.__setattr__(self, "target_u_bar", target)
        committed_history = _f64(
            self.committed_history,
            "Newton result committed_history",
            (self.model_identity.n_material,),
        )
        if np.any(committed_history < 0.0):
            raise ValueError("Newton result committed_history must be non-negative")
        object.__setattr__(self, "committed_history", _immutable(committed_history))
        z = _f64(self.z, "Newton result z")
        expected_shape = (
            len(self.model_identity.constraints.free_dofs)
            + 2 * self.model_identity.n_material,
        )
        if z.shape != expected_shape:
            raise ValueError("Newton result z has the wrong model-bound shape")
        if np.any(z[-self.model_identity.n_material :] < 0.0):
            raise ValueError("Newton result final history must satisfy H>=0")
        object.__setattr__(self, "z", _immutable(z))
        initial_z = _f64(self.initial_z, "Newton result initial_z", expected_shape)
        initial_residual = _f64(
            self.initial_residual, "Newton result initial_residual", expected_shape
        )
        object.__setattr__(self, "initial_z", _immutable(initial_z))
        object.__setattr__(self, "initial_residual", _immutable(initial_residual))
        if not isinstance(self.initial_residual_zero, (bool, np.bool_)):
            raise TypeError("Newton result initial_residual_zero must be boolean")
        object.__setattr__(
            self, "initial_residual_zero", bool(self.initial_residual_zero)
        )
        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("Newton result converged must be boolean")
        object.__setattr__(self, "converged", bool(self.converged))
        if not isinstance(self.convergence_status, str) or not self.convergence_status:
            raise ValueError("Newton result convergence_status must be nonempty")
        if not isinstance(self.convergence_reason, str) or not self.convergence_reason:
            raise ValueError("Newton result convergence_reason must be nonempty")
        iterations = _strict_integer(
            self.iterations, "Newton result iterations", positive=False, zero_allowed=True
        )
        object.__setattr__(self, "iterations", iterations)
        if not isinstance(self.config, MonolithicNewtonConfig):
            raise TypeError("Newton result config has the wrong type")
        if not isinstance(self.gate_config, MonolithicNewtonGateConfig):
            raise TypeError("Newton result gate_config has the wrong type")
        active = np.asarray(self.final_active_mask)
        if active.dtype != np.dtype(bool) or active.ndim != 1:
            raise ValueError("Newton result active mask must be a one-dimensional bool array")
        object.__setattr__(self, "final_active_mask", _immutable(active))
        _validate_scale_pair(
            self.final_accepted_trial_scales,
            z,
            "Newton result final accepted-trial",
        )
        if active.size != self.final_accepted_trial_scales.H_used.size:
            raise ValueError("Newton result active mask does not match final scales")
        scaled = _f64(self.final_scaled_residual, "Newton result scaled residual")
        if scaled.shape != z.shape:
            raise ValueError("Newton result scaled residual has the wrong shape")
        object.__setattr__(self, "final_scaled_residual", _immutable(scaled))
        raw_block_norms = tuple(self.final_scaled_block_norms)
        if len(raw_block_norms) != 3:
            raise ValueError("Newton result final scaled block norms must have length three")
        block_norms = tuple(
            _scalar(value, f"Newton result final scaled block norm {index}")
            for index, value in enumerate(raw_block_norms)
        )
        if any(value < 0.0 for value in block_norms):
            raise ValueError("Newton result final scaled block norms must be non-negative")
        object.__setattr__(self, "final_scaled_block_norms", block_norms)
        object.__setattr__(self, "final_merit", _scalar(self.final_merit, "Newton result merit"))
        if self.final_merit < 0.0:
            raise ValueError("Newton result merit must be non-negative")
        audit = tuple(self.diagnostics)
        if len(audit) != iterations:
            raise ValueError("Newton result diagnostic count does not match iterations")
        if not all(isinstance(row, MonolithicNewtonDiagnostics) for row in audit):
            raise TypeError("Newton result diagnostics contain a wrong type")
        object.__setattr__(self, "diagnostics", audit)
        self.validate()

    def _validate_owner(self, owner: MonolithicGripResidual) -> None:
        if not isinstance(owner, MonolithicGripResidual):
            raise TypeError("Newton result validation owner has the wrong type")
        if (
            engineering._engineering_model_digest(owner._model)
            != engineering._engineering_model_digest(self.model_identity)
            or owner._model.static_digest != self.model_identity.static_digest
            or owner.u_bar != self.target_u_bar
        ):
            raise ValueError("Newton result validation owner is not model/target paired")
        if not np.array_equal(owner.committed_history, self.committed_history):
            raise ValueError("Newton result validation owner history is not paired")

    def recompute_with_owner(
        self, owner: MonolithicGripResidual
    ) -> tuple[
        np.ndarray,
        tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        np.ndarray,
    ]:
        """Recompute residual vectors through one explicit in-memory owner.

        The returned row tuple contains, in order, the reference-scale residual,
        the accepted line-search residual at reference scales, and the accepted
        trial residual at accepted-trial scales.  ``validate`` also checks the
        stored vectors algebraically, so a durable/value-only audit does not
        depend on calling a possibly replaced method for every diagnostic row.
        """

        self._validate_owner(owner)
        expected_initial = _f64(
            owner.residual(self.initial_z),
            "recomputed Newton initial residual",
            (owner.n_packed,),
        )
        expected_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for row in self.diagnostics:
            expected_rows.append(
                (
                    _f64(
                        owner.scaled_residual(row.z_before, row.linearization_scales),
                        "recomputed Newton reference residual",
                        (owner.n_packed,),
                    ),
                    _f64(
                        owner.scaled_residual(row.z_after, row.linearization_scales),
                        "recomputed Newton line-search residual",
                        (owner.n_packed,),
                    ),
                    _f64(
                        owner.scaled_residual(row.z_after, row.accepted_trial_scales),
                        "recomputed Newton accepted-trial residual",
                        (owner.n_packed,),
                    ),
                )
            )
        expected_final = _f64(
            owner.scaled_residual(self.z, self.final_accepted_trial_scales),
            "recomputed Newton final scaled residual",
            (owner.n_packed,),
        )
        return expected_initial, tuple(expected_rows), expected_final

    def validate(self) -> None:
        """Validate the value/algebraic Newton audit without a mutable owner.

        The packed residual vectors retained in each diagnostic bind the
        reference, accepted line-search, and accepted-trial evaluations.  An
        explicit :meth:`validate_with_owner` call adds independent
        model/operator recomputation for an in-memory production result.
        """

        if not isinstance(self.model_identity, EngineeringGripModel):
            raise TypeError("Newton result model identity has the wrong type")
        if self.static_identity is not self.model_identity.static:
            raise ValueError("Newton result static identity is not model-paired")
        if not isinstance(self.config, MonolithicNewtonConfig):
            raise TypeError("Newton result config has the wrong type")
        if not isinstance(self.gate_config, MonolithicNewtonGateConfig):
            raise TypeError("Newton result gate_config has the wrong type")
        if not self.converged:
            raise ValueError("a returned Newton result must be converged")
        if self.convergence_status != _NEWTON_CONVERGENCE_STATUS:
            raise ValueError("Newton result convergence_status is not canonical")

        owner = MonolithicGripResidual(
            self.model_identity, self.target_u_bar, self.committed_history
        )
        audit = self.diagnostics
        if audit:
            expected_initial = _unscale_residual(
                owner,
                audit[0].reference_scaled_residual,
                audit[0].linearization_scales,
            )
        else:
            expected_initial = np.array(self.initial_residual, dtype=np.float64, copy=True)
        expected_rows = tuple(
            (
                row.reference_scaled_residual,
                row.line_search_scaled_residual,
                row.accepted_trial_scaled_residual,
            )
            for row in audit
        )
        expected_scaled = self.final_scaled_residual
        _require_roundtrip_equal(
            self.initial_residual,
            expected_initial,
            "Newton result initial residual",
        )
        expected_initial_zero = bool(np.all(self.initial_residual == 0.0))
        if self.initial_residual_zero != expected_initial_zero:
            raise ValueError("Newton result initial zero shortcut flag is not audit-derived")

        _validate_model_scale_pair(
            owner,
            self.final_accepted_trial_scales,
            self.z,
            "Newton result final accepted-trial",
        )
        if not np.array_equal(self.final_scaled_residual, expected_scaled):
            raise ValueError("Newton result final scaled residual is not recomputed")
        expected_block_norms = _scaled_block_norms(owner, expected_scaled)
        if self.final_scaled_block_norms != expected_block_norms:
            raise ValueError("Newton result final scaled block norms are not recomputed")
        expected_merit = _merit(expected_scaled)
        if self.final_merit != expected_merit:
            raise ValueError("Newton result final merit is not recomputed")
        expected_active = np.asarray(owner.active_mask(self.z), dtype=bool)
        if not np.array_equal(self.final_active_mask, expected_active):
            raise ValueError("Newton result final active mask is not recomputed")
        if np.any(self.z[-owner.n_material :] < 0.0):
            raise ValueError("Newton result final history must satisfy H>=0")

        if self.initial_residual_zero:
            if self.convergence_reason != _NEWTON_REASON_INITIAL_RESIDUAL:
                raise ValueError("zero-initial-residual result has the wrong convergence reason")
            if self.iterations != 0 or audit or not np.array_equal(self.z, self.initial_z):
                raise ValueError("zero-initial-residual shortcut is not explicitly audited")
            if not np.all(expected_scaled == 0.0):
                raise ValueError("zero-initial-residual shortcut has a nonzero final residual")
            if self.final_scaled_block_norms != (0.0, 0.0, 0.0) or self.final_merit != 0.0:
                raise ValueError("zero-initial-residual shortcut has nonzero final audit values")
            return

        if self.convergence_reason != _NEWTON_REASON_ACCEPTED_TRIAL:
            raise ValueError("Newton result has a non-canonical convergence reason")
        if self.iterations <= 0 or len(audit) != self.iterations:
            raise ValueError("Newton result iterations are not bound to ordered diagnostics")
        if not np.array_equal(audit[0].z_before, self.initial_z):
            raise ValueError("Newton result initial_z is not paired with the first diagnostic")

        previous_after = self.initial_z
        for index, row in enumerate(audit, start=1):
            if row.iteration != index:
                raise ValueError("Newton diagnostics are not ordered by iteration")
            if not np.array_equal(row.z_before, previous_after):
                raise ValueError("Newton diagnostic z chain is not ordered")
            _validate_model_scale_pair(
                owner,
                row.linearization_scales,
                row.z_before,
                "Newton diagnostic linearization",
            )
            _validate_model_scale_pair(
                owner,
                row.accepted_trial_scales,
                row.z_after,
                "Newton diagnostic accepted-trial",
            )
            expected_before_active = np.asarray(owner.active_mask(row.z_before), dtype=bool)
            expected_after_active = np.asarray(owner.active_mask(row.z_after), dtype=bool)
            if not np.array_equal(row.active_mask_before, expected_before_active):
                raise ValueError("Newton diagnostic initial active mask is not recomputed")
            if not np.array_equal(row.active_mask_after, expected_after_active):
                raise ValueError("Newton diagnostic accepted active mask is not recomputed")
            if row.active_set_stable != bool(
                np.array_equal(expected_before_active, expected_after_active)
            ):
                raise ValueError("Newton diagnostic active-set stability is not recomputed")
            if not np.array_equal(
                row.z_after, row.z_before + row.accepted_alpha * row.delta
            ):
                raise ValueError("Newton diagnostic accepted trial is not paired with delta")
            expected_reference, expected_line_search, expected_accepted = expected_rows[index - 1]
            if not np.array_equal(row.reference_scaled_residual, expected_reference):
                raise ValueError("Newton diagnostic reference residual is not recomputed")
            if not np.array_equal(row.line_search_scaled_residual, expected_line_search):
                raise ValueError("Newton diagnostic line-search residual is not recomputed")
            if not np.array_equal(row.accepted_trial_scaled_residual, expected_accepted):
                raise ValueError("Newton diagnostic accepted-trial residual is not recomputed")
            if row.merit_reference != _merit(row.reference_scaled_residual):
                raise ValueError("Newton diagnostic reference merit is not algebraically paired")
            if row.merit_trial != _merit(row.line_search_scaled_residual):
                raise ValueError("Newton diagnostic trial merit is not algebraically paired")
            expected_norms = _scaled_block_norms(owner, row.accepted_trial_scaled_residual)
            if row.scaled_block_norms != expected_norms:
                raise ValueError("Newton diagnostic block norms are not algebraically paired")
            expected_correction = _correction_metrics(
                owner, row.z_before, row.z_after, self.gate_config
            )
            actual_correction = (
                row.correction_mechanics_material_inf,
                row.correction_mechanics_union_inf,
                row.correction_damage_inf,
                row.correction_history_inf,
                row.correction_displacement_pass,
                row.correction_damage_pass,
                row.correction_history_pass,
                row.correction_pass,
            )
            if actual_correction != expected_correction:
                raise ValueError("Newton diagnostic correction audit is not recomputed")
            if row.gmres_info != 0:
                raise ValueError("accepted Newton diagnostic has nonzero GMRES status")
            if not np.isfinite(row.gmres_residual_limit):
                raise ValueError("Newton diagnostic GMRES limit is non-finite")
            if row.gmres_unpreconditioned_residual > row.gmres_residual_limit:
                raise ValueError("Newton diagnostic GMRES residual exceeds its limit")
            expected_alphas: list[float] = []
            alpha = 1.0
            for _ in range(len(row.trial_alphas)):
                expected_alphas.append(alpha)
                alpha *= self.config.line_search_reduction
            if tuple(expected_alphas) != row.trial_alphas:
                raise ValueError("Newton diagnostic line-search alpha sequence is not ordered")
            if not row.trial_alphas or row.trial_alphas[-1] != row.accepted_alpha:
                raise ValueError("Newton diagnostic accepted alpha is not the final trial")
            if row.line_search_backtracks != len(row.trial_alphas) - 1:
                raise ValueError("Newton diagnostic backtrack count is not paired")
            if row.line_search_rejections[-1] != "" or any(
                not reason for reason in row.line_search_rejections[:-1]
            ):
                raise ValueError("Newton diagnostic rejected line-search audit is not canonical")
            armijo_bound = (
                1.0 - self.config.armijo_c * row.accepted_alpha
            ) * row.merit_reference
            if row.merit_trial > armijo_bound:
                raise ValueError("Newton diagnostic final trial fails its Armijo audit")
            trial_converged = bool(
                all(value <= 1.0 for value in row.scaled_block_norms)
                and row.active_set_stable
                and row.correction_pass
            )
            if index < len(audit) and trial_converged:
                raise ValueError("a converged Newton trial was not retained as final")
            if index == len(audit) and not trial_converged:
                raise ValueError("final Newton trial does not satisfy convergence tolerance")
            previous_after = row.z_after

        final = audit[-1]
        if not np.array_equal(final.z_after, self.z):
            raise ValueError("Newton result z is not paired with its final diagnostic")
        if not np.array_equal(final.accepted_trial_scaled_residual, self.final_scaled_residual):
            raise ValueError("Newton result residual is not paired with its final diagnostic")
        if not _scales_equal(final.accepted_trial_scales, self.final_accepted_trial_scales):
            raise ValueError("Newton result final scales are not paired with its final diagnostic")
        if not np.array_equal(final.active_mask_after, self.final_active_mask):
            raise ValueError("Newton result active mask is not paired with its final diagnostic")

    def validate_with_owner(self, owner: MonolithicGripResidual) -> None:
        """Add independent residual replay through one in-memory owner.

        This method is intentionally separate from the value-only constructor
        contract.  Production solver results call it with the authoritative
        owner; failure-orchestration tests may retain their deterministic
        manufactured owner without making the durable audit depend on object
        identity or a mutable method replacement.
        """

        self.validate()
        expected_initial, expected_rows, expected_final = self.recompute_with_owner(owner)
        if not np.array_equal(self.initial_residual, expected_initial):
            raise ValueError("Newton result initial residual is not owner-recomputed")
        if not np.array_equal(self.final_scaled_residual, expected_final):
            raise ValueError("Newton result final residual is not owner-recomputed")
        for row, expected in zip(self.diagnostics, expected_rows, strict=True):
            if not np.array_equal(row.reference_scaled_residual, expected[0]):
                raise ValueError("Newton reference residual is not owner-recomputed")
            if not np.array_equal(row.line_search_scaled_residual, expected[1]):
                raise ValueError("Newton line-search residual is not owner-recomputed")
            if not np.array_equal(row.accepted_trial_scaled_residual, expected[2]):
                raise ValueError("Newton accepted-trial residual is not owner-recomputed")

    @property
    def final_z(self) -> np.ndarray:
        return self.z

    @property
    def residual(self) -> np.ndarray:
        return self.final_scaled_residual


class MonolithicNewtonFailure(RuntimeError):
    """Hard Phase-2A failure retaining one immutable partial-step audit."""

    def __init__(self, message: str, audit: MonolithicNewtonFailureAudit):
        super().__init__(message)
        if not isinstance(audit, MonolithicNewtonFailureAudit):
            raise TypeError("Newton failure audit has the wrong type")
        self.audit = audit
        # This compatibility view contains only completed prior rows.  The
        # current partial row is always retained in ``audit``.
        self.diagnostics = audit.completed_diagnostics
        self.message = message


class _MonolithicPreconditionerFailure(RuntimeError):
    """Internal marker so the solver can preserve preconditioner provenance."""


def _failure(
    monolithic: MonolithicGripResidual,
    config: MonolithicNewtonConfig,
    gate_config: MonolithicNewtonGateConfig,
    message: str,
    stage: str,
    iteration: int,
    current_z: np.ndarray,
    active_mask: np.ndarray,
    linearization_scales: MonolithicNewtonScales | None,
    gmres_info: int | None,
    gmres_callback_residuals: tuple[float, ...],
    gmres_unpreconditioned_residual: float | None,
    gmres_residual_limit: float | None,
    attempted_alphas: tuple[float, ...],
    rejection_reasons: tuple[str, ...],
    completed_diagnostics: tuple[MonolithicNewtonDiagnostics, ...],
) -> MonolithicNewtonFailure:
    audit = MonolithicNewtonFailureAudit(
        stage=stage,
        target_u_bar=monolithic.u_bar,
        model_identity=monolithic._model,
        static_identity=monolithic._model.static,
        config=config,
        gate_config=gate_config,
        reason=message,
        iteration=iteration,
        current_z=current_z,
        active_mask=active_mask,
        linearization_scales=linearization_scales,
        gmres_info=gmres_info,
        gmres_callback_residuals=gmres_callback_residuals,
        gmres_unpreconditioned_residual=gmres_unpreconditioned_residual,
        gmres_residual_limit=gmres_residual_limit,
        attempted_alphas=attempted_alphas,
        rejection_reasons=rejection_reasons,
        completed_diagnostics=completed_diagnostics,
    )
    return MonolithicNewtonFailure(message, audit)


class MonolithicBlockDiagonalPreconditioner:
    """Frozen left preconditioner for one scaled generalized Jacobian.

    The q block is an exact sparse solve of the fixed-damage mechanics tangent
    restricted to free DOFs.  The d block is an exact sparse solve of
    ``A_H = diag(Gc/l0 + 2H) + Gc*l0*D`` in strong residual coordinates.  The
    H block is the inverse of the scaled identity row.  Coupling blocks are
    deliberately omitted; there is no fallback preconditioner.
    """

    def __init__(
        self,
        monolithic: MonolithicGripResidual,
        z: Any,
        linearization_scales: MonolithicNewtonScales,
    ) -> None:
        if not isinstance(monolithic, MonolithicGripResidual):
            raise TypeError("preconditioner monolithic argument has the wrong type")
        packed = _f64(z, "preconditioner z", (monolithic.n_packed,))
        _validate_model_scale_pair(
            monolithic,
            linearization_scales,
            packed,
            "preconditioner linearization",
        )
        _, damage, history = monolithic.unpack(packed)
        view = monolithic._view(damage)
        self._n_q = monolithic.n_free
        self._n_material = monolithic.n_material
        self._n_packed = monolithic.n_packed
        self._linearization_scales = linearization_scales

        gradient = monolithic._gradient_operator
        gradient_x = gradient[0::2]
        gradient_y = gradient[1::2]
        center = view.center_factor
        extensional = center * (view.lame_lambda + 2.0 * view.shear_modulus)
        shear = center * view.shear_modulus
        volumetric = center * view.lame_lambda
        mechanics_xx = (
            gradient_x.T @ gradient_x.multiply(extensional[:, None])
            + gradient_y.T @ gradient_y.multiply(shear[:, None])
        )
        mechanics_xy = (
            gradient_x.T @ gradient_y.multiply(volumetric[:, None])
            + gradient_y.T @ gradient_x.multiply(shear[:, None])
        )
        mechanics_yy = (
            gradient_y.T @ gradient_y.multiply(extensional[:, None])
            + gradient_x.T @ gradient_x.multiply(shear[:, None])
        )
        grouped = scipy.sparse.bmat(
            [[mechanics_xx, mechanics_xy], [mechanics_xy.T, mechanics_yy]],
            format="csc",
        )
        grouped_to_full = np.concatenate(
            (
                2 * np.arange(view.n_particles, dtype=np.int64),
                2 * np.arange(view.n_particles, dtype=np.int64) + 1,
            )
        )
        full_to_grouped = np.empty(view.n_dofs, dtype=np.int64)
        full_to_grouped[grouped_to_full] = np.arange(view.n_dofs, dtype=np.int64)
        full_mechanics = grouped[full_to_grouped, :][:, full_to_grouped]
        strong_mechanics = full_mechanics[monolithic._free, :][
            :, monolithic._free
        ].tocsc()
        if not np.all(np.isfinite(strong_mechanics.data)):
            raise FloatingPointError("strong mechanics preconditioner matrix is non-finite")
        self._strong_mechanics_matrix = strong_mechanics
        try:
            self._strong_mechanics_factor = scipy.sparse.linalg.splu(strong_mechanics)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise _MonolithicPreconditionerFailure(
                f"strong mechanics preconditioner factorization failed: {error}"
            ) from error

        diagonal = self._model_damage_diagonal(monolithic, history)
        strong_matrix = scipy.sparse.diags(diagonal, offsets=0, format="csc")
        strong_matrix = strong_matrix + (
            monolithic._damage_matrix.tocsc(copy=True)
            * (monolithic._model.Gc * monolithic._model.l0)
        )
        if not np.all(np.isfinite(strong_matrix.data)):
            raise FloatingPointError("strong damage preconditioner matrix is non-finite")
        self._strong_damage_matrix = strong_matrix
        try:
            self._strong_damage_factor = scipy.sparse.linalg.splu(strong_matrix)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise _MonolithicPreconditionerFailure(
                f"strong damage preconditioner factorization failed: {error}"
            ) from error

    @staticmethod
    def _model_damage_diagonal(
        monolithic: MonolithicGripResidual, history: np.ndarray
    ) -> np.ndarray:
        diagonal = monolithic._model.Gc / monolithic._model.l0 + 2.0 * history
        if not np.all(np.isfinite(diagonal)):
            raise FloatingPointError("strong damage preconditioner diagonal is non-finite")
        return np.asarray(diagonal, dtype=np.float64)

    @property
    def n_packed(self) -> int:
        return self._n_packed

    @property
    def linearization_scales(self) -> MonolithicNewtonScales:
        return self._linearization_scales

    @property
    def strong_mechanics_matrix(self) -> scipy.sparse.csc_matrix:
        return self._strong_mechanics_matrix.copy()

    @property
    def strong_damage_matrix(self) -> scipy.sparse.csc_matrix:
        return self._strong_damage_matrix.copy()

    def apply(self, values: Any) -> np.ndarray:
        vector = _f64(values, "GMRES preconditioner vector", (self._n_packed,))
        result = np.zeros(self._n_packed, dtype=np.float64)
        q_values = vector[: self._n_q]
        try:
            solved_mechanics = self._strong_mechanics_factor.solve(q_values)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise _MonolithicPreconditionerFailure(
                f"strong mechanics preconditioner solve failed: {error}"
            ) from error
        result[: self._n_q] = self._linearization_scales.Lq * np.asarray(
            solved_mechanics, dtype=np.float64
        )
        d_start = self._n_q
        d_stop = d_start + self._n_material
        try:
            solved_damage = self._strong_damage_factor.solve(vector[d_start:d_stop])
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise _MonolithicPreconditionerFailure(
                f"strong damage preconditioner solve failed: {error}"
            ) from error
        result[d_start:d_stop] = self._linearization_scales.Lrho * np.asarray(
            solved_damage, dtype=np.float64
        )
        result[d_stop:] = self._linearization_scales.LH * vector[d_stop:]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("GMRES preconditioner output is non-finite")
        return result

    def as_linear_operator(self) -> scipy.sparse.linalg.LinearOperator:
        return scipy.sparse.linalg.LinearOperator(
            shape=(self._n_packed, self._n_packed),
            matvec=self.apply,
            dtype=np.float64,
        )


def _hookstep_failure(
    stage: str,
    reason: str,
    radius: float,
    beta: float | None,
    dimension: int,
    config: MonolithicHookstepConfig,
    *,
    model_identity: EngineeringGripModel | None = None,
    current_z: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    scale_state_digest: str | None = None,
    hessenberg: np.ndarray | None = None,
    metric_singular_values: np.ndarray | None = None,
    projected_singular_values: np.ndarray | None = None,
) -> MonolithicHookstepFailure:
    return MonolithicHookstepFailure(
        reason,
        MonolithicHookstepFailureAudit(
            stage=stage,
            reason=reason,
            radius=radius,
            beta=beta,
            krylov_dimension=dimension,
            config=config,
            model_identity=model_identity,
            current_z=current_z,
            active_mask=active_mask,
            scale_state_digest=scale_state_digest,
            hessenberg=hessenberg,
            metric_singular_values=metric_singular_values,
            projected_singular_values=projected_singular_values,
        ),
    )


def _hookstep_action(values: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Validate an operator result while separating structure from numerics."""

    array = np.asarray(values)
    if array.dtype != _FLOAT64 or array.shape != shape:
        raise ValueError(f"{name} must have dtype float64 and shape {shape}")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} is non-finite")
    return np.asarray(array, dtype=np.float64)


def build_monolithic_trust_metric(
    monolithic: MonolithicGripResidual,
) -> MonolithicTrustMetric:
    """Build the fixed normalized continuum-product metric for packed steps."""

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("hookstep metric requires MonolithicGripResidual")
    model = monolithic._model
    union_volumes = _f64(
        model.static.union_volumes,
        "hookstep union volumes",
        (model.n_union,),
    )
    material_volumes = _f64(
        model.material_operator.volumes,
        "hookstep material volumes",
        (model.n_material,),
    )
    if np.any(union_volumes <= 0.0) or np.any(material_volumes <= 0.0):
        raise ValueError("hookstep metric volumes must be strictly positive")
    union_volume = float(np.sum(union_volumes))
    material_volume = float(np.sum(material_volumes))
    displacement_scale = float(model.l0)
    history_scale = float(model.Gc / model.l0)
    if not all(
        np.isfinite(value) and value > 0.0
        for value in (
            union_volume,
            material_volume,
            displacement_scale,
            history_scale,
        )
    ):
        raise ValueError("hookstep metric normalization is invalid")
    full_q_diagonal = np.sqrt(
        np.repeat(union_volumes / union_volume, 2)
    ) / displacement_scale
    q_diagonal = full_q_diagonal[monolithic._free]
    damage_diagonal = np.sqrt(material_volumes / material_volume)
    history_diagonal = damage_diagonal / history_scale
    return MonolithicTrustMetric(
        diagonal=np.concatenate((q_diagonal, damage_diagonal, history_diagonal)),
        n_free=monolithic.n_free,
        n_material=monolithic.n_material,
        union_volume=union_volume,
        material_volume=material_volume,
        displacement_scale=displacement_scale,
        history_scale=history_scale,
        model_identity=model,
        static_identity=model.static,
    )


def build_monolithic_hookstep_arnoldi(
    monolithic: MonolithicGripResidual,
    z: Any,
    scales: MonolithicNewtonScales,
    active_mask: Any,
    config: MonolithicHookstepConfig,
    radius: Any,
    metric: MonolithicTrustMetric,
) -> MonolithicHookstepArnoldi:
    """Build one frozen right-preconditioned Arnoldi model without a trial."""

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("hookstep Arnoldi requires MonolithicGripResidual")
    if not isinstance(config, MonolithicHookstepConfig):
        raise TypeError("hookstep Arnoldi config has the wrong type")
    if not isinstance(metric, MonolithicTrustMetric):
        raise TypeError("hookstep Arnoldi metric has the wrong type")
    if (
        metric.model_identity is not monolithic._model
        or metric.static_identity is not monolithic._model.static
    ):
        raise ValueError("hookstep Arnoldi metric is not owner-paired")
    if metric.n_packed != monolithic.n_packed:
        raise ValueError("hookstep Arnoldi metric does not match the residual")
    canonical_metric = build_monolithic_trust_metric(monolithic)
    if (
        not np.array_equal(metric.diagonal, canonical_metric.diagonal)
        or metric.union_volume != canonical_metric.union_volume
        or metric.material_volume != canonical_metric.material_volume
        or metric.displacement_scale != canonical_metric.displacement_scale
        or metric.history_scale != canonical_metric.history_scale
    ):
        raise ValueError("hookstep Arnoldi metric contents are not owner-paired")
    radius_value = _scalar(radius, "hookstep Arnoldi radius")
    if not config.trust_radius_min <= radius_value <= config.trust_radius_max:
        raise ValueError("hookstep Arnoldi radius lies outside configured bounds")
    packed = _f64(z, "hookstep Arnoldi z", (monolithic.n_packed,))
    _validate_model_scale_pair(monolithic, scales, packed, "hookstep Arnoldi")
    active = np.asarray(active_mask)
    if active.dtype != np.dtype(bool) or active.shape != (monolithic.n_material,):
        raise ValueError("hookstep Arnoldi active mask has the wrong dtype or shape")
    try:
        residual = _hookstep_action(
            monolithic.scaled_residual(packed, scales),
            "hookstep initial scaled residual",
            (monolithic.n_packed,),
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise _hookstep_failure(
            "arnoldi",
            f"hookstep initial residual evaluation failed: {error}",
            radius_value,
            None,
            0,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
        ) from error
    try:
        beta = _stable_l2_norm(residual, "hookstep initial residual")
    except FloatingPointError as error:
        raise _hookstep_failure(
            "arnoldi",
            f"hookstep initial residual norm failed: {error}",
            radius_value,
            None,
            0,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
        ) from error
    if beta == 0.0:
        raise ValueError("exact root does not require a hookstep direction")
    try:
        preconditioner = MonolithicBlockDiagonalPreconditioner(monolithic, packed, scales)
    except (
        _MonolithicPreconditionerFailure,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        raise _hookstep_failure(
            "arnoldi",
            f"hookstep preconditioner construction failed: {error}",
            radius_value,
            beta,
            0,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
        ) from error

    maximum = min(config.max_krylov_dimension, monolithic.n_packed)
    hessenberg = np.zeros((maximum + 1, maximum), dtype=np.float64)
    basis_columns: list[np.ndarray] = [np.asarray(residual / beta, dtype=np.float64)]
    preconditioned_columns: list[np.ndarray] = []
    happy_breakdown = False

    for column in range(maximum):
        try:
            preconditioned = preconditioner.apply(basis_columns[column])
            action = monolithic.scaled_jvp(
                packed,
                preconditioned,
                scales,
                active,
            )
            preconditioned = _hookstep_action(
                preconditioned,
                "hookstep preconditioned Arnoldi vector",
                (monolithic.n_packed,),
            )
            action = _hookstep_action(
                action,
                "hookstep Arnoldi action",
                (monolithic.n_packed,),
            )
        except (
            _MonolithicPreconditionerFailure,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            raise _hookstep_failure(
                "arnoldi",
                f"hookstep Arnoldi action failed: {error}",
                radius_value,
                beta,
                column,
                config,
                model_identity=monolithic._model,
                current_z=packed,
                active_mask=active,
                scale_state_digest=scales.state_digest,
                hessenberg=hessenberg[: column + 2, : column + 1],
            ) from error
        preconditioned_columns.append(np.asarray(preconditioned, dtype=np.float64))
        work = np.asarray(action, dtype=np.float64).copy()
        action_norm = float(np.linalg.norm(work))
        if not np.isfinite(action_norm):
            raise _hookstep_failure(
                "arnoldi",
                "hookstep Arnoldi action norm is non-finite",
                radius_value,
                beta,
                column,
                config,
                model_identity=monolithic._model,
                current_z=packed,
                active_mask=active,
                scale_state_digest=scales.state_digest,
                hessenberg=hessenberg[: column + 2, : column + 1],
            )
        # Two MGS passes make the stored orthogonality audit meaningful without
        # changing the matrix-free operator or preconditioner.
        for _ in range(2):
            for row in range(column + 1):
                coefficient = float(np.dot(basis_columns[row], work))
                hessenberg[row, column] += coefficient
                work -= coefficient * basis_columns[row]
        next_norm = float(np.linalg.norm(work))
        hessenberg[column + 1, column] = next_norm
        threshold = config.arnoldi_breakdown_rtol * action_norm
        if action_norm == 0.0 or next_norm <= threshold:
            happy_breakdown = True
            if next_norm > 0.0:
                basis_columns.append(np.asarray(work / next_norm, dtype=np.float64))
            break
        basis_columns.append(np.asarray(work / next_norm, dtype=np.float64))

    dimension = len(preconditioned_columns)
    basis = np.column_stack(basis_columns[:dimension])
    preconditioned_basis = np.column_stack(preconditioned_columns)
    compact_hessenberg = hessenberg[: dimension + 1, :dimension]
    terminal_basis_vector = (
        basis_columns[dimension] if len(basis_columns) > dimension else None
    )
    audit_basis = (
        basis
        if terminal_basis_vector is None
        else np.column_stack((basis, terminal_basis_vector))
    )
    gram = audit_basis.T @ audit_basis
    orthogonality_defect = float(
        np.max(np.abs(gram - np.eye(audit_basis.shape[1], dtype=np.float64)))
    )
    if (
        not np.isfinite(orthogonality_defect)
        or orthogonality_defect > config.arnoldi_orthogonality_rtol
    ):
        raise _hookstep_failure(
            "arnoldi",
            "hookstep Arnoldi basis exceeds its orthogonality tolerance",
            radius_value,
            beta,
            dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=compact_hessenberg,
        )
    try:
        metric_basis = metric.apply_columns(preconditioned_basis)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise _hookstep_failure(
            "metric",
            f"hookstep metric-basis action failed: {error}",
            radius_value,
            beta,
            dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=compact_hessenberg,
        ) from error
    return MonolithicHookstepArnoldi(
        beta=beta,
        basis=basis,
        preconditioned_basis=preconditioned_basis,
        hessenberg=compact_hessenberg,
        metric_basis=metric_basis,
        terminal_basis_vector=terminal_basis_vector,
        happy_breakdown=happy_breakdown,
        orthogonality_defect=orthogonality_defect,
    )


def solve_projected_monolithic_hookstep(
    arnoldi: MonolithicHookstepArnoldi,
    radius: Any,
    config: MonolithicHookstepConfig,
) -> MonolithicProjectedHookstep:
    """Solve the SVD-whitened projected least-squares trust problem."""

    if not isinstance(arnoldi, MonolithicHookstepArnoldi):
        raise TypeError("projected hookstep requires MonolithicHookstepArnoldi")
    if not isinstance(config, MonolithicHookstepConfig):
        raise TypeError("projected hookstep config has the wrong type")
    radius_value = _scalar(radius, "projected hookstep radius")
    if not config.trust_radius_min <= radius_value <= config.trust_radius_max:
        raise ValueError("projected hookstep radius lies outside configured bounds")
    metric_basis = np.asarray(arnoldi.metric_basis, dtype=np.float64)
    try:
        _, metric_singular, metric_vt = np.linalg.svd(
            metric_basis, full_matrices=False
        )
    except np.linalg.LinAlgError as error:
        raise _hookstep_failure(
            "projected",
            f"hookstep metric-basis SVD failed: {error}",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            hessenberg=arnoldi.hessenberg,
        ) from error
    metric_threshold = config.metric_rank_rtol * float(metric_singular[0])
    metric_rank = int(np.count_nonzero(metric_singular > metric_threshold))
    if metric_rank != arnoldi.dimension:
        raise _hookstep_failure(
            "projected",
            "hookstep metric basis is rank deficient",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            hessenberg=arnoldi.hessenberg,
            metric_singular_values=metric_singular,
        )
    inverse_whitener = metric_vt.T / metric_singular[None, :]
    projected_matrix = arnoldi.hessenberg @ inverse_whitener
    rhs = np.zeros(arnoldi.dimension + 1, dtype=np.float64)
    rhs[0] = arnoldi.beta
    try:
        projected_u, projected_singular, projected_vt = np.linalg.svd(
            projected_matrix, full_matrices=False
        )
    except np.linalg.LinAlgError as error:
        raise _hookstep_failure(
            "projected",
            f"hookstep projected SVD failed: {error}",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            hessenberg=arnoldi.hessenberg,
            metric_singular_values=metric_singular,
        ) from error
    projected_threshold = config.projected_rank_rtol * float(projected_singular[0])
    projected_rank = int(np.count_nonzero(projected_singular > projected_threshold))
    projected_rhs = projected_u.T @ rhs

    def whitened_step(multiplier: float) -> np.ndarray:
        factors = np.zeros_like(projected_singular)
        retained = projected_singular > projected_threshold
        denominator = projected_singular[retained] ** 2 + multiplier
        factors[retained] = -(
            projected_singular[retained] * projected_rhs[retained] / denominator
        )
        return np.asarray(projected_vt.T @ factors, dtype=np.float64)

    whitened = whitened_step(0.0)
    whitened_norm = float(np.linalg.norm(whitened))
    multiplier = 0.0
    secular_iterations = 0
    on_boundary = False
    if whitened_norm > radius_value:
        gradient = projected_matrix.T @ rhs
        gradient_norm = float(np.linalg.norm(gradient))
        if not np.isfinite(gradient_norm) or gradient_norm == 0.0:
            raise _hookstep_failure(
                "projected",
                "projected hookstep is stationary at a nonzero residual",
                radius_value,
                arnoldi.beta,
                arnoldi.dimension,
                config,
                hessenberg=arnoldi.hessenberg,
                metric_singular_values=metric_singular,
                projected_singular_values=projected_singular,
            )
        lower = 0.0
        upper = gradient_norm / radius_value
        upper_step = whitened_step(upper)
        while float(np.linalg.norm(upper_step)) > radius_value:
            upper *= 2.0
            secular_iterations += 1
            if secular_iterations >= config.secular_max_iterations:
                raise _hookstep_failure(
                    "projected",
                    "projected hookstep could not bracket the trust multiplier",
                    radius_value,
                    arnoldi.beta,
                    arnoldi.dimension,
                    config,
                    hessenberg=arnoldi.hessenberg,
                    metric_singular_values=metric_singular,
                    projected_singular_values=projected_singular,
                )
            upper_step = whitened_step(upper)
        tolerance = config.secular_rtol * radius_value
        while secular_iterations < config.secular_max_iterations:
            multiplier = 0.5 * (lower + upper)
            candidate = whitened_step(multiplier)
            candidate_norm = float(np.linalg.norm(candidate))
            secular_iterations += 1
            if abs(candidate_norm - radius_value) <= tolerance:
                whitened = candidate
                whitened_norm = candidate_norm
                break
            if candidate_norm > radius_value:
                lower = multiplier
            else:
                upper = multiplier
        else:
            raise _hookstep_failure(
                "projected",
                "projected hookstep secular solve exhausted its iteration budget",
                radius_value,
                arnoldi.beta,
                arnoldi.dimension,
                config,
                hessenberg=arnoldi.hessenberg,
                metric_singular_values=metric_singular,
                projected_singular_values=projected_singular,
            )
        on_boundary = True
    coefficients = inverse_whitener @ whitened
    projected_action = arnoldi.hessenberg @ coefficients
    projected_residual = rhs + projected_action
    projected_residual_norm = float(np.linalg.norm(projected_residual))
    predicted_reduction = -float(np.dot(rhs, projected_action)) - 0.5 * float(
        np.dot(projected_action, projected_action)
    )
    if not np.isfinite(predicted_reduction) or predicted_reduction <= 0.0:
        raise _hookstep_failure(
            "projected",
            "projected hookstep has no positive model reduction",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            hessenberg=arnoldi.hessenberg,
            metric_singular_values=metric_singular,
            projected_singular_values=projected_singular,
        )
    return MonolithicProjectedHookstep(
        coefficients=coefficients,
        trust_multiplier=multiplier,
        radius=radius_value,
        metric_norm=whitened_norm,
        projected_residual_norm=projected_residual_norm,
        predicted_reduction=predicted_reduction,
        metric_rank=metric_rank,
        projected_rank=projected_rank,
        on_boundary=on_boundary,
        secular_iterations=secular_iterations,
        radius_tolerance=config.secular_rtol * radius_value,
    )


def _rejected_monolithic_hookstep_radius(
    radius: float,
    rejection_count: int,
    config: MonolithicHookstepConfig,
) -> MonolithicHookstepRadiusDecision | None:
    proposed = config.gamma_shrink * radius
    if not np.isfinite(proposed) or proposed <= 0.0:
        return None
    return MonolithicHookstepRadiusDecision(
        accepted=False,
        expanded=False,
        next_radius=proposed,
        rejection_count=rejection_count + 1,
        reason="rejected_shrink",
    )


def _validate_monolithic_hookstep_rejection(
    decision: MonolithicHookstepRadiusDecision,
    radius: float,
    config: MonolithicHookstepConfig,
) -> None:
    if decision.rejection_count > config.max_radius_rejections:
        raise _hookstep_failure(
            "radius",
            "hookstep radius rejection budget exhausted",
            radius,
            None,
            0,
            config,
        )
    if decision.next_radius < config.trust_radius_min:
        raise _hookstep_failure(
            "radius",
            "hookstep radius shrink would cross trust_radius_min",
            radius,
            None,
            0,
            config,
        )


def decide_monolithic_hookstep_radius(
    rho: Any,
    step_metric_norm: Any,
    radius: Any,
    rejection_count: Any,
    config: MonolithicHookstepConfig,
) -> MonolithicHookstepRadiusDecision:
    """Apply the pure accepted/rejected trust-radius update contract."""

    if not isinstance(config, MonolithicHookstepConfig):
        raise TypeError("hookstep radius config has the wrong type")
    radius_value = _scalar(radius, "hookstep current radius")
    if not config.trust_radius_min <= radius_value <= config.trust_radius_max:
        raise ValueError("hookstep current radius lies outside configured bounds")
    for value, name in (
        (rho, "hookstep reduction ratio"),
        (step_metric_norm, "hookstep radius step norm"),
    ):
        raw = np.asarray(value)
        if raw.ndim != 0:
            raise ValueError(f"{name} must be scalar, got shape {raw.shape}")
        try:
            converted = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a scalar number") from error
        if not np.isfinite(converted):
            raise _hookstep_failure(
                "radius",
                f"{name} is non-finite",
                radius_value,
                None,
                0,
                config,
            )
        if name.endswith("step norm") and converted < 0.0:
            raise ValueError(f"{name} must be non-negative")
        if name.endswith("ratio"):
            ratio = converted
        else:
            step_norm = converted
    count = _strict_integer(
        rejection_count,
        "hookstep current rejection count",
        positive=False,
        zero_allowed=True,
    )
    if step_norm > radius_value * (1.0 + config.metric_consistency_rtol):
        raise ValueError("hookstep radius decision received an infeasible step")
    if ratio < config.eta_accept:
        decision = _rejected_monolithic_hookstep_radius(
            radius_value, count, config
        )
        if decision is None:
            raise _hookstep_failure(
                "radius",
                "hookstep radius shrink underflowed below trust_radius_min",
                radius_value,
                None,
                0,
                config,
            )
        _validate_monolithic_hookstep_rejection(decision, radius_value, config)
        return decision
    expand = bool(
        ratio >= config.eta_expand
        and step_norm >= config.boundary_fraction * radius_value
    )
    next_radius = (
        min(config.gamma_expand * radius_value, config.trust_radius_max)
        if expand
        else radius_value
    )
    return MonolithicHookstepRadiusDecision(
        accepted=True,
        expanded=expand,
        next_radius=next_radius,
        rejection_count=0,
        reason="accepted_expand" if expand else "accepted_retain",
    )


def _materialize_monolithic_hookstep_direction(
    monolithic: MonolithicGripResidual,
    z: Any,
    scales: MonolithicNewtonScales,
    active_mask: Any,
    config: MonolithicHookstepConfig,
    radius: Any,
    metric: MonolithicTrustMetric,
    arnoldi: MonolithicHookstepArnoldi,
    reference_scaled_residual: Any,
) -> MonolithicHookstepDirection:
    """Resolve one radius using an already frozen Arnoldi model."""

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("monolithic hookstep requires MonolithicGripResidual")
    if not isinstance(config, MonolithicHookstepConfig):
        raise TypeError("monolithic hookstep config has the wrong type")
    radius_value = _scalar(radius, "monolithic hookstep radius")
    if not config.trust_radius_min <= radius_value <= config.trust_radius_max:
        raise ValueError("monolithic hookstep radius lies outside configured bounds")
    packed = _f64(z, "monolithic hookstep z", (monolithic.n_packed,))
    active = np.asarray(active_mask)
    if not isinstance(metric, MonolithicTrustMetric):
        raise TypeError("monolithic hookstep metric has the wrong type")
    if not isinstance(arnoldi, MonolithicHookstepArnoldi):
        raise TypeError("monolithic hookstep Arnoldi model has the wrong type")
    residual = _f64(
        reference_scaled_residual,
        "hookstep reference scaled residual",
        (monolithic.n_packed,),
    )
    residual_norm = _stable_l2_norm(residual, "hookstep reference residual")
    if residual_norm != arnoldi.beta:
        raise ValueError("hookstep reference residual is not Arnoldi-paired")
    if not np.array_equal(residual / residual_norm, arnoldi.basis[:, 0]):
        raise ValueError("hookstep reference residual vector is not Arnoldi-paired")
    try:
        projected = solve_projected_monolithic_hookstep(arnoldi, radius_value, config)
    except MonolithicHookstepFailure as error:
        audit = error.audit
        raise _hookstep_failure(
            audit.stage,
            audit.reason,
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=(
                audit.hessenberg
                if audit.hessenberg is not None
                else arnoldi.hessenberg
            ),
            metric_singular_values=audit.metric_singular_values,
            projected_singular_values=audit.projected_singular_values,
        ) from error
    step = arnoldi.preconditioned_basis @ projected.coefficients
    try:
        metric_norm = metric.norm(step)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise _hookstep_failure(
            "metric",
            f"hookstep reconstructed metric norm failed: {error}",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=arnoldi.hessenberg,
        ) from error
    try:
        full_action = monolithic.scaled_jvp(
            packed,
            step,
            scales,
            active,
        )
        full_action = _hookstep_action(
            full_action,
            "hookstep full prediction action",
            (monolithic.n_packed,),
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise _hookstep_failure(
            "prediction",
            f"hookstep full prediction action failed: {error}",
            projected.radius,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=arnoldi.hessenberg,
        ) from error
    full_prediction = -float(np.dot(residual, full_action)) - 0.5 * float(
        np.dot(full_action, full_action)
    )
    if not np.isfinite(full_prediction) or full_prediction <= 0.0:
        raise _hookstep_failure(
            "prediction",
            "hookstep full model has no positive reduction",
            projected.radius,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=arnoldi.hessenberg,
        )
    prediction_scale = max(
        abs(full_prediction), abs(projected.predicted_reduction), np.finfo(np.float64).tiny
    )
    defect = abs(full_prediction - projected.predicted_reduction) / prediction_scale
    metric_scale = max(
        abs(metric_norm), abs(projected.metric_norm), np.finfo(np.float64).tiny
    )
    metric_defect = abs(metric_norm - projected.metric_norm) / metric_scale
    if defect > config.prediction_consistency_rtol:
        raise _hookstep_failure(
            "prediction",
            "hookstep projected/full prediction exceeds its tolerance",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=arnoldi.hessenberg,
        )
    if metric_defect > config.metric_consistency_rtol:
        raise _hookstep_failure(
            "metric",
            "hookstep reconstructed metric norm exceeds its tolerance",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=arnoldi.hessenberg,
        )
    return MonolithicHookstepDirection(
        step=step,
        metric_norm=metric_norm,
        full_predicted_reduction=full_prediction,
        projected_prediction_defect=defect,
        metric_consistency_defect=metric_defect,
        prediction_consistency_rtol=config.prediction_consistency_rtol,
        metric_consistency_rtol=config.metric_consistency_rtol,
        arnoldi=arnoldi,
        projected=projected,
    )


def compute_monolithic_hookstep_direction(
    monolithic: MonolithicGripResidual,
    z: Any,
    scales: MonolithicNewtonScales,
    active_mask: Any,
    config: MonolithicHookstepConfig,
    radius: Any,
) -> MonolithicHookstepDirection:
    """Build and solve one frozen manufactured hookstep direction."""

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("monolithic hookstep requires MonolithicGripResidual")
    if not isinstance(config, MonolithicHookstepConfig):
        raise TypeError("monolithic hookstep config has the wrong type")
    radius_value = _scalar(radius, "monolithic hookstep radius")
    if not config.trust_radius_min <= radius_value <= config.trust_radius_max:
        raise ValueError("monolithic hookstep radius lies outside configured bounds")
    packed = _f64(z, "monolithic hookstep z", (monolithic.n_packed,))
    active = np.asarray(active_mask)
    metric = build_monolithic_trust_metric(monolithic)
    arnoldi = build_monolithic_hookstep_arnoldi(
        monolithic,
        packed,
        scales,
        active,
        config,
        radius_value,
        metric,
    )
    try:
        residual = _hookstep_action(
            monolithic.scaled_residual(packed, scales),
            "hookstep full prediction residual",
            (monolithic.n_packed,),
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise _hookstep_failure(
            "prediction",
            f"hookstep full prediction residual failed: {error}",
            radius_value,
            arnoldi.beta,
            arnoldi.dimension,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
            hessenberg=arnoldi.hessenberg,
        ) from error
    return _materialize_monolithic_hookstep_direction(
        monolithic,
        packed,
        scales,
        active,
        config,
        radius_value,
        metric,
        arnoldi,
        residual,
    )


def _hookstep_controller_failure(
    error: MonolithicHookstepFailure,
    monolithic: MonolithicGripResidual,
    packed: np.ndarray,
    scales: MonolithicNewtonScales,
    active: np.ndarray,
    metric: MonolithicTrustMetric,
    arnoldi: MonolithicHookstepArnoldi,
    attempts: tuple[MonolithicHookstepTrial, ...],
) -> MonolithicHookstepControllerFailure:
    source = error.audit
    inner = _hookstep_failure(
        source.stage,
        source.reason,
        source.radius,
        arnoldi.beta,
        arnoldi.dimension,
        source.config,
        model_identity=monolithic._model,
        current_z=packed,
        active_mask=active,
        scale_state_digest=scales.state_digest,
        hessenberg=(
            source.hessenberg
            if source.hessenberg is not None
            else arnoldi.hessenberg
        ),
        metric_singular_values=source.metric_singular_values,
        projected_singular_values=source.projected_singular_values,
    ).audit
    return MonolithicHookstepControllerFailure(
        str(error),
        MonolithicHookstepControllerFailureAudit(
            reason=str(error),
            inner_audit=inner,
            metric=metric,
            arnoldi=arnoldi,
            attempts=attempts,
        ),
    )


def run_monolithic_hookstep_correction(
    monolithic: MonolithicGripResidual,
    z: Any,
    gate_config: MonolithicNewtonGateConfig,
    active_mask: Any,
    config: MonolithicHookstepConfig,
) -> MonolithicHookstepCorrection:
    """Run one manufactured correction with rejected-radius projected re-solves."""

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("monolithic hookstep controller requires MonolithicGripResidual")
    if not isinstance(config, MonolithicHookstepConfig):
        raise TypeError("monolithic hookstep controller config has the wrong type")
    _validate_gate_config(gate_config)
    packed = _f64(z, "hookstep controller z", (monolithic.n_packed,))
    scales = monolithic.block_scales(packed, gate_config)
    active = np.asarray(active_mask)
    if active.dtype != np.dtype(bool) or active.shape != (monolithic.n_material,):
        raise ValueError("hookstep controller active mask has the wrong dtype or shape")
    current_active = monolithic.active_mask(packed)
    if not np.array_equal(active, current_active):
        raise ValueError("hookstep controller active mask is stale")
    radius = config.trust_radius_initial
    try:
        reference_residual = _hookstep_action(
            monolithic.scaled_residual(packed, scales),
            "hookstep controller reference residual",
            (monolithic.n_packed,),
        )
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise _hookstep_failure(
            "arnoldi",
            f"hookstep controller reference residual failed: {error}",
            radius,
            None,
            0,
            config,
            model_identity=monolithic._model,
            current_z=packed,
            active_mask=active,
            scale_state_digest=scales.state_digest,
        ) from error
    if np.all(reference_residual == 0.0):
        return MonolithicHookstepCorrection(
            status="exact_root",
            model_identity=monolithic._model,
            z_before=packed,
            linearization_scales=scales,
            config=config,
            active_mask_before=active,
            reference_scaled_residual=reference_residual,
            metric=None,
            arnoldi=None,
            attempts=(),
            z_after=packed,
            active_mask_after=active,
            next_radius=radius,
            active_set_changed=False,
            rebuild_required=False,
        )
    metric = build_monolithic_trust_metric(monolithic)
    arnoldi = build_monolithic_hookstep_arnoldi(
        monolithic,
        packed,
        scales,
        active,
        config,
        radius,
        metric,
    )
    attempts: list[MonolithicHookstepTrial] = []
    rejection_count = 0
    while True:
        try:
            direction = _materialize_monolithic_hookstep_direction(
                monolithic,
                packed,
                scales,
                active,
                config,
                radius,
                metric,
                arnoldi,
                reference_residual,
            )
        except MonolithicHookstepFailure as error:
            raise _hookstep_controller_failure(
                error,
                monolithic,
                packed,
                scales,
                active,
                metric,
                arnoldi,
                tuple(attempts),
            ) from error

        with np.errstate(over="ignore", invalid="ignore"):
            candidate_raw = packed + direction.step
        candidate_finite = bool(np.all(np.isfinite(candidate_raw)))
        candidate = np.asarray(candidate_raw, dtype=np.float64) if candidate_finite else None
        trial_residual: np.ndarray | None = None
        actual_reduction: float | None = None
        ratio: float | None = None
        trial_failure_reason: str | None = (
            None if candidate is not None else "hookstep candidate state is non-finite"
        )
        if candidate is not None:
            try:
                trial_residual = _hookstep_action(
                    monolithic.scaled_residual(candidate, scales),
                    "hookstep controller trial residual",
                    (monolithic.n_packed,),
                )
                residual_change = trial_residual - reference_residual
                actual_reduction = -float(
                    np.dot(reference_residual, residual_change)
                ) - 0.5 * float(np.dot(residual_change, residual_change))
            except FloatingPointError as error:
                trial_residual = None
                actual_reduction = None
                ratio = None
                trial_failure_reason = f"hookstep nonlinear trial is non-finite: {error}"
            except np.linalg.LinAlgError as error:
                inner = _hookstep_failure(
                    "prediction",
                    f"hookstep nonlinear trial evaluation failed: {error}",
                    radius,
                    arnoldi.beta,
                    arnoldi.dimension,
                    config,
                    model_identity=monolithic._model,
                    current_z=packed,
                    active_mask=active,
                    scale_state_digest=scales.state_digest,
                    hessenberg=arnoldi.hessenberg,
                )
                raise _hookstep_controller_failure(
                    inner,
                    monolithic,
                    packed,
                    scales,
                    active,
                    metric,
                    arnoldi,
                    tuple(attempts),
                ) from error
            if trial_residual is not None and actual_reduction is not None:
                if np.isfinite(actual_reduction):
                    ratio_scale = max(
                        abs(actual_reduction),
                        direction.full_predicted_reduction,
                        np.finfo(np.float64).tiny,
                    )
                    scaled_prediction = (
                        direction.full_predicted_reduction / ratio_scale
                    )
                    if scaled_prediction == 0.0:
                        ratio = None
                    else:
                        ratio = (actual_reduction / ratio_scale) / scaled_prediction
                    if ratio is None or not np.isfinite(ratio):
                        error = _hookstep_failure(
                            "radius",
                            "hookstep reduction ratio is non-finite",
                            radius,
                            arnoldi.beta,
                            arnoldi.dimension,
                            config,
                            model_identity=monolithic._model,
                            current_z=packed,
                            active_mask=active,
                            scale_state_digest=scales.state_digest,
                            hessenberg=arnoldi.hessenberg,
                        )
                        raise _hookstep_controller_failure(
                            error,
                            monolithic,
                            packed,
                            scales,
                            active,
                            metric,
                            arnoldi,
                            tuple(attempts),
                        ) from error
                else:
                    trial_residual = None
                    actual_reduction = None
                    trial_failure_reason = "hookstep actual reduction is non-finite"

        if trial_residual is None:
            decision = _rejected_monolithic_hookstep_radius(
                radius, rejection_count, config
            )
            trial = MonolithicHookstepTrial(
                index=len(attempts),
                radius=radius,
                direction=direction,
                outcome="rejected_nonfinite_trial",
                failure_reason=trial_failure_reason,
                candidate_z=candidate,
                trial_scaled_residual=None,
                actual_reduction=None,
                rho=None,
                radius_decision=decision,
            )
        elif ratio is not None and ratio < config.eta_accept:
            decision = _rejected_monolithic_hookstep_radius(
                radius, rejection_count, config
            )
            trial = MonolithicHookstepTrial(
                index=len(attempts),
                radius=radius,
                direction=direction,
                outcome="rejected_ratio",
                failure_reason=None,
                candidate_z=candidate,
                trial_scaled_residual=trial_residual,
                actual_reduction=actual_reduction,
                rho=ratio,
                radius_decision=decision,
            )
        else:
            if ratio is None or actual_reduction is None or trial_residual is None:
                raise RuntimeError("hookstep controller lost finite trial evidence")
            decision = decide_monolithic_hookstep_radius(
                ratio,
                direction.metric_norm,
                radius,
                rejection_count,
                config,
            )
            trial = MonolithicHookstepTrial(
                index=len(attempts),
                radius=radius,
                direction=direction,
                outcome="accepted_pending_rebuild",
                failure_reason=None,
                candidate_z=candidate,
                trial_scaled_residual=trial_residual,
                actual_reduction=actual_reduction,
                rho=ratio,
                radius_decision=decision,
            )
        if decision.accepted:
            if candidate is None:
                raise RuntimeError("accepted hookstep trial has no finite candidate")
            try:
                accepted_active = monolithic.active_mask(candidate)
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                attempts.append(trial)
                inner = _hookstep_failure(
                    "prediction",
                    f"accepted hookstep active-mask rebuild failed: {error}",
                    radius,
                    arnoldi.beta,
                    arnoldi.dimension,
                    config,
                    model_identity=monolithic._model,
                    current_z=packed,
                    active_mask=active,
                    scale_state_digest=scales.state_digest,
                    hessenberg=arnoldi.hessenberg,
                )
                raise _hookstep_controller_failure(
                    inner,
                    monolithic,
                    packed,
                    scales,
                    active,
                    metric,
                    arnoldi,
                    tuple(attempts),
                ) from error
            trial = MonolithicHookstepTrial(
                index=trial.index,
                radius=trial.radius,
                direction=trial.direction,
                outcome="accepted",
                failure_reason=None,
                candidate_z=trial.candidate_z,
                trial_scaled_residual=trial.trial_scaled_residual,
                actual_reduction=trial.actual_reduction,
                rho=trial.rho,
                radius_decision=trial.radius_decision,
            )
            attempts.append(trial)
            return MonolithicHookstepCorrection(
                status="accepted",
                model_identity=monolithic._model,
                z_before=packed,
                linearization_scales=scales,
                config=config,
                active_mask_before=active,
                reference_scaled_residual=reference_residual,
                metric=metric,
                arnoldi=arnoldi,
                attempts=tuple(attempts),
                z_after=candidate,
                active_mask_after=accepted_active,
                next_radius=decision.next_radius,
                active_set_changed=not np.array_equal(active, accepted_active),
                rebuild_required=True,
            )
        attempts.append(trial)
        try:
            _validate_monolithic_hookstep_rejection(decision, radius, config)
        except MonolithicHookstepFailure as error:
            raise _hookstep_controller_failure(
                error,
                monolithic,
                packed,
                scales,
                active,
                metric,
                arnoldi,
                tuple(attempts),
            ) from error
        radius = decision.next_radius
        rejection_count = decision.rejection_count


def _scaled_block_norms(
    monolithic: MonolithicGripResidual, scaled_residual: np.ndarray
) -> tuple[float, float, float]:
    return _scaled_block_norms_from_shape(
        monolithic.n_free, monolithic.n_material, scaled_residual
    )


def _scaled_block_norms_from_shape(
    n_free: int, n_material: int, scaled_residual: np.ndarray
) -> tuple[float, float, float]:
    packed = _f64(
        scaled_residual,
        "scaled residual block norm input",
        (n_free + 2 * n_material,),
    )
    q = packed[:n_free]
    damage = packed[n_free : n_free + n_material]
    history = packed[n_free + n_material :]
    values = (
        float(np.linalg.norm(q)),
        _max_abs_array(damage),
        _max_abs_array(history),
    )
    if not all(np.isfinite(value) for value in values):
        raise FloatingPointError("scaled residual block norm is non-finite")
    return values


def _unscale_residual(
    monolithic: MonolithicGripResidual,
    scaled_residual: np.ndarray,
    scales: MonolithicNewtonScales,
) -> np.ndarray:
    q, damage, history = monolithic.unpack(
        _f64(scaled_residual, "scaled residual to unscale", (monolithic.n_packed,))
    )
    return np.concatenate(
        (
            q * scales.Lq,
            damage * scales.Lrho * monolithic._damage_volumes,
            history * scales.LH,
        )
    )


def _merit(scaled_residual: np.ndarray) -> float:
    value = 0.5 * float(np.dot(scaled_residual, scaled_residual))
    if not np.isfinite(value):
        raise FloatingPointError("scaled Newton merit is non-finite")
    return value


def _correction_metrics(
    monolithic: MonolithicGripResidual,
    before: np.ndarray,
    after: np.ndarray,
    gate_config: MonolithicNewtonGateConfig,
) -> tuple[float, float, float, float, bool, bool, bool, bool]:
    q_before = monolithic.reconstruct_displacement(before)
    q_after = monolithic.reconstruct_displacement(after)
    q_delta = q_after - q_before
    material_dofs = 2 * monolithic.n_material
    mechanics_material = _max_abs_array(q_delta[:material_dofs])
    mechanics_union = _max_abs_array(q_delta)
    displacement_material_scale = max(
        _max_abs_array(q_before[:material_dofs]),
        _max_abs_array(q_after[:material_dofs]),
        abs(monolithic.u_bar),
    )
    displacement_union_scale = max(
        _max_abs_array(q_before),
        _max_abs_array(q_after),
        abs(monolithic.u_bar),
    )
    displacement_pass = bool(
        mechanics_material
        <= gate_config.displacement.limit(displacement_material_scale)
        and mechanics_union
        <= gate_config.displacement.limit(displacement_union_scale)
    )
    _, damage_before, history_before = monolithic.unpack(before)
    _, damage_after, history_after = monolithic.unpack(after)
    damage_error = _max_abs_array(damage_after - damage_before)
    damage_scale = max(_max_abs_array(damage_before), _max_abs_array(damage_after))
    damage_pass = bool(damage_error <= gate_config.damage.limit(damage_scale))
    history_error = _max_abs_array(history_after - history_before)
    history_scale = max(
        _max_abs_array(monolithic.committed_history),
        _max_abs_array(history_before),
        _max_abs_array(history_after),
    )
    history_pass = bool(history_error <= gate_config.history.limit(history_scale))
    return (
        mechanics_material,
        mechanics_union,
        damage_error,
        history_error,
        displacement_pass,
        damage_pass,
        history_pass,
        bool(displacement_pass and damage_pass and history_pass),
    )


def _make_result(
    monolithic: MonolithicGripResidual,
    z: np.ndarray,
    initial_z: np.ndarray,
    initial_residual: np.ndarray,
    config: MonolithicNewtonConfig,
    gate_config: MonolithicNewtonGateConfig,
    accepted_trial_scales: MonolithicNewtonScales,
    diagnostics: list[MonolithicNewtonDiagnostics],
) -> MonolithicNewtonResult:
    _, _, final_history = monolithic.unpack(z)
    if np.any(final_history < 0.0):
        raise ValueError("Newton final history must satisfy H>=0")
    _validate_model_scale_pair(
        monolithic,
        accepted_trial_scales,
        z,
        "Newton result accepted-trial",
    )
    scaled = monolithic.scaled_residual(z, accepted_trial_scales)
    result = MonolithicNewtonResult(
        target_u_bar=monolithic.u_bar,
        model_identity=monolithic._model,
        static_identity=monolithic._model.static,
        committed_history=monolithic.committed_history,
        z=np.array(z, dtype=np.float64, copy=True),
        initial_z=np.array(initial_z, dtype=np.float64, copy=True),
        initial_residual=np.array(initial_residual, dtype=np.float64, copy=True),
        initial_residual_zero=bool(np.all(initial_residual == 0.0)),
        converged=True,
        convergence_status=_NEWTON_CONVERGENCE_STATUS,
        convergence_reason=(
            _NEWTON_REASON_INITIAL_RESIDUAL
            if not diagnostics
            else _NEWTON_REASON_ACCEPTED_TRIAL
        ),
        iterations=len(diagnostics),
        config=config,
        gate_config=gate_config,
        final_active_mask=np.array(monolithic.active_mask(z), dtype=bool, copy=True),
        final_accepted_trial_scales=accepted_trial_scales,
        final_scaled_residual=scaled,
        final_scaled_block_norms=_scaled_block_norms(monolithic, scaled),
        final_merit=_merit(scaled),
        diagnostics=tuple(diagnostics),
    )
    result.validate_with_owner(monolithic)
    return result


def solve_monolithic_newton(
    monolithic: MonolithicGripResidual,
    initial_z: Any,
    config: MonolithicNewtonConfig,
    gate_config: MonolithicNewtonGateConfig,
) -> MonolithicNewtonResult:
    """Solve one fixed-load monolithic problem with scaled semismooth Newton.

    This is deliberately a core solve, not a continuation controller.  It
    does not alter the load, committed history, residual equations, or input
    arrays.  The active set and row scales are frozen for each GMRES solve and
    rebuilt only after an accepted raw trial.
    """

    if not isinstance(monolithic, MonolithicGripResidual):
        raise TypeError("monolithic must be a MonolithicGripResidual")
    if not isinstance(config, MonolithicNewtonConfig):
        raise TypeError("config must be a MonolithicNewtonConfig")
    _validate_gate_config(gate_config)
    z = np.array(_f64(initial_z, "Newton initial_z", (monolithic.n_packed,)), copy=True)
    original_initial_z = np.array(z, dtype=np.float64, copy=True)
    _, _, initial_history = monolithic.unpack(z)
    if np.any(initial_history < 0.0):
        raise ValueError("Newton initial history must satisfy H>=0")
    iteration = 0
    # Keep a valid audit mask available even when the first active-set
    # evaluation itself fails.  The zero mask is evidence that no active set
    # was established, not a fallback used for numerical progress.
    active = np.zeros(monolithic.n_material, dtype=bool)
    diagnostics: list[MonolithicNewtonDiagnostics] = []

    def fail(
        stage: str,
        reason: str,
        current_z: np.ndarray,
        current_active: np.ndarray,
        linearization_scales: MonolithicNewtonScales | None,
        gmres_info: int | None,
        callback_values: tuple[float, ...],
        gmres_residual: float | None,
        gmres_limit: float | None,
        attempted_alphas: tuple[float, ...],
        rejection_reasons: tuple[str, ...],
    ) -> MonolithicNewtonFailure:
        audit_scales = linearization_scales
        if audit_scales is not None:
            _validate_model_scale_pair(
                monolithic,
                audit_scales,
                current_z,
                "Newton failure linearization",
            )
        return _failure(
            monolithic,
            config,
            gate_config,
            reason,
            stage,
            iteration,
            current_z,
            current_active,
            audit_scales,
            gmres_info,
            callback_values,
            gmres_residual,
            gmres_limit,
            attempted_alphas,
            rejection_reasons,
            tuple(diagnostics),
        )

    try:
        candidate_active = np.asarray(monolithic.active_mask(z))
        if candidate_active.dtype != np.dtype(bool) or candidate_active.shape != (
            monolithic.n_material,
        ):
            raise ValueError(
                "initial active mask must have boolean dtype and shape "
                f"({monolithic.n_material},)"
            )
        active = np.array(candidate_active, dtype=bool, copy=True)
    except FloatingPointError as error:
        raise fail(
            "initial",
            f"initial active-set evaluation failed: {error}",
            z,
            active,
            None,
            None,
            (),
            None,
            None,
            (),
            (),
        ) from error

    try:
        raw_initial_residual = _f64(
            monolithic.residual(z),
            "initial monolithic residual",
            (monolithic.n_packed,),
        )
    except FloatingPointError as error:
        raise fail(
            "initial",
            f"initial residual evaluation failed: {error}",
            z,
            active,
            None,
            None,
            (),
            None,
            None,
            (),
            (),
        ) from error
    if np.all(raw_initial_residual == 0.0):
        iteration = 0
        try:
            exact_scales = monolithic.block_scales(z, gate_config)
            return _make_result(
                monolithic,
                z,
                original_initial_z,
                raw_initial_residual,
                config,
                gate_config,
                exact_scales,
                [],
            )
        except FloatingPointError as error:
            raise fail(
                "scales",
                f"exact-root scale construction failed: {error}",
                z,
                active,
                None,
                None,
                (),
                None,
                None,
                (),
                (),
            ) from error

    iteration = 0
    try:
        scales = monolithic.block_scales(z, gate_config)
        scaled = monolithic.scaled_residual(z, scales)
        _merit(scaled)
    except FloatingPointError as error:
        raise fail(
            "scales",
            f"initial scale/residual construction failed: {error}",
            z,
            active,
            None,
            None,
            (),
            None,
            None,
            (),
            (),
        ) from error

    for iteration in range(1, config.max_newton_iterations + 1):
        reference_z = np.array(z, dtype=np.float64, copy=True)
        reference_active = np.array(active, dtype=bool, copy=True)
        reference_scales = scales
        reference_scaled = scaled
        reference_merit = _merit(reference_scaled)
        rhs = -reference_scaled
        operator = scipy.sparse.linalg.LinearOperator(
            shape=(monolithic.n_packed, monolithic.n_packed),
            matvec=lambda values: monolithic.scaled_jvp(
                reference_z,
                values,
                reference_scales,
                reference_active,
            ),
            dtype=np.float64,
        )
        callback_values: list[float] = []

        try:
            preconditioner = MonolithicBlockDiagonalPreconditioner(
                monolithic, reference_z, reference_scales
            )
        except (
            _MonolithicPreconditionerFailure,
            FloatingPointError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            raise fail(
                "preconditioner",
                f"preconditioner construction failed: {error}",
                reference_z,
                reference_active,
                reference_scales,
                None,
                (),
                None,
                None,
                (),
                (),
            ) from error

        def callback(value: Any) -> None:
            raw = np.asarray(value)
            if raw.ndim != 0:
                raise FloatingPointError("GMRES callback residual is not scalar")
            scalar = float(raw)
            if not np.isfinite(scalar):
                raise FloatingPointError("GMRES callback residual is non-finite")
            callback_values.append(scalar)

        try:
            delta, gmres_info = scipy.sparse.linalg.gmres(
                operator,
                rhs,
                x0=np.zeros(monolithic.n_packed, dtype=np.float64),
                rtol=config.gmres_rtol,
                atol=config.gmres_atol,
                restart=config.gmres_restart,
                maxiter=config.gmres_maxiter,
                M=preconditioner.as_linear_operator(),
                callback=callback,
                callback_type="pr_norm",
            )
        except _MonolithicPreconditionerFailure as error:
            raise fail(
                "preconditioner",
                f"preconditioner application failed during GMRES: {error}",
                reference_z,
                reference_active,
                reference_scales,
                None,
                tuple(callback_values),
                None,
                None,
                (),
                (),
            ) from error
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
            raise fail(
                "gmres",
                f"GMRES failed before returning a step: {error}",
                reference_z,
                reference_active,
                reference_scales,
                None,
                tuple(callback_values),
                None,
                None,
                (),
                (),
            ) from error
        raw_info = np.asarray(gmres_info)
        if raw_info.ndim != 0 or not np.issubdtype(raw_info.dtype, np.integer):
            raise fail(
                "gmres",
                "GMRES returned a non-integer status",
                reference_z,
                reference_active,
                reference_scales,
                None,
                tuple(callback_values),
                None,
                None,
                (),
                (),
            )
        gmres_info = int(raw_info)
        delta = np.asarray(delta)
        if delta.dtype != np.dtype(np.float64) or delta.shape != (monolithic.n_packed,):
            raise fail(
                "gmres_direction",
                "GMRES returned a direction with the wrong dtype or shape",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                None,
                None,
                (),
                (),
            )
        if not np.all(np.isfinite(delta)):
            raise fail(
                "gmres_direction",
                "GMRES returned a non-finite direction",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                None,
                None,
                (),
                (),
            )
        try:
            gmres_residual = operator @ delta - rhs
            gmres_residual_norm = float(np.linalg.norm(gmres_residual))
        except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
            raise fail(
                "gmres_residual",
                f"GMRES unpreconditioned residual evaluation failed: {error}",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                None,
                None,
                (),
                (),
            ) from error
        rhs_norm = float(np.linalg.norm(rhs))
        gmres_limit = max(config.gmres_atol, config.gmres_rtol * rhs_norm)
        if not np.isfinite(gmres_residual_norm) or not np.isfinite(gmres_limit):
            raise fail(
                "gmres_residual",
                "GMRES unpreconditioned residual check is non-finite",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                None if not np.isfinite(gmres_residual_norm) else gmres_residual_norm,
                None if not np.isfinite(gmres_limit) else gmres_limit,
                (),
                (),
            )
        if gmres_info != 0:
            raise fail(
                "gmres",
                f"GMRES did not converge: info={gmres_info}",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                gmres_residual_norm,
                gmres_limit,
                (),
                (),
            )
        if gmres_residual_norm > gmres_limit:
            raise fail(
                "gmres_residual",
                "GMRES unpreconditioned residual exceeds its explicit limit: "
                f"{gmres_residual_norm} > {gmres_limit}",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                gmres_residual_norm,
                gmres_limit,
                (),
                (),
            )

        alpha = 1.0
        trial_alphas: list[float] = []
        line_search_rejections: list[str] = []
        accepted: tuple[np.ndarray, np.ndarray | None, float, float, int] | None = None
        for backtrack in range(config.line_search_max_backtracks + 1):
            if alpha < config.line_search_alpha_min:
                raise fail(
                    "line_search",
                    "line search reached alpha below line_search_alpha_min",
                    reference_z,
                    reference_active,
                    reference_scales,
                    gmres_info,
                    tuple(callback_values),
                    gmres_residual_norm,
                    gmres_limit,
                    tuple(trial_alphas),
                    tuple(line_search_rejections),
                )
            trial_alphas.append(alpha)
            with np.errstate(over="ignore", invalid="ignore"):
                trial = np.asarray(reference_z + alpha * delta, dtype=np.float64)
            reason = ""
            if not np.all(np.isfinite(trial)):
                reason = "non-finite raw trial"
            trial_scaled: np.ndarray | None = None
            trial_merit: float | None = None
            if not reason:
                try:
                    trial_scaled = monolithic.scaled_residual(trial, reference_scales)
                    trial_merit = _merit(trial_scaled)
                except FloatingPointError as error:
                    reason = f"non-finite raw trial residual or merit: {error}"
            if not reason and trial_merit is not None:
                armijo_bound = (1.0 - config.armijo_c * alpha) * reference_merit
                if trial_merit <= armijo_bound:
                    accepted = (trial, trial_scaled, trial_merit, alpha, backtrack)
                    line_search_rejections.append("")
                    break
                reason = (
                    "Armijo insufficient decrease: "
                    f"phi_trial={trial_merit:.17g} > bound={armijo_bound:.17g}"
                )
            line_search_rejections.append(reason)
            if backtrack == config.line_search_max_backtracks:
                raise fail(
                    "line_search",
                    "line search backtrack budget exhausted",
                    reference_z,
                    reference_active,
                    reference_scales,
                    gmres_info,
                    tuple(callback_values),
                    gmres_residual_norm,
                    gmres_limit,
                    tuple(trial_alphas),
                    tuple(line_search_rejections),
                )
            next_alpha = alpha * config.line_search_reduction
            if next_alpha < config.line_search_alpha_min or next_alpha >= alpha:
                raise fail(
                    "line_search",
                    "line search cannot satisfy line_search_alpha_min",
                    reference_z,
                    reference_active,
                    reference_scales,
                    gmres_info,
                    tuple(callback_values),
                    gmres_residual_norm,
                    gmres_limit,
                    tuple(trial_alphas),
                    tuple(line_search_rejections),
                )
            alpha = next_alpha
        if accepted is None:
            raise fail(
                "line_search",
                "line search returned no accepted trial",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                gmres_residual_norm,
                gmres_limit,
                tuple(trial_alphas),
                tuple(line_search_rejections),
            )
        trial, trial_scaled, trial_merit, accepted_alpha, accepted_backtracks = accepted
        if trial_scaled is None:
            raise fail(
                "line_search",
                "line search accepted a trial without a residual",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                gmres_residual_norm,
                gmres_limit,
                tuple(trial_alphas),
                tuple(line_search_rejections),
            )
        trial_active: np.ndarray | None = None
        accepted_trial_scales: MonolithicNewtonScales | None = None
        try:
            trial_active = np.array(monolithic.active_mask(trial), dtype=bool, copy=True)
            active_set_stable = bool(np.array_equal(reference_active, trial_active))
            (
                correction_mechanics_material,
                correction_mechanics_union,
                correction_damage,
                correction_history,
                correction_displacement_pass,
                correction_damage_pass,
                correction_history_pass,
                trial_correction_pass,
            ) = _correction_metrics(monolithic, reference_z, trial, gate_config)
            accepted_trial_scales = monolithic.block_scales(trial, gate_config)
            _validate_model_scale_pair(
                monolithic,
                accepted_trial_scales,
                trial,
                "accepted-trial scales",
            )
            trial_scaled_for_convergence = monolithic.scaled_residual(
                trial, accepted_trial_scales
            )
            trial_norms = _scaled_block_norms(monolithic, trial_scaled_for_convergence)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            failure_z = trial if accepted_trial_scales is not None else reference_z
            failure_active = (
                trial_active if trial_active is not None else reference_active
            )
            failure_scales = (
                accepted_trial_scales
                if accepted_trial_scales is not None
                else reference_scales
            )
            raise fail(
                "accepted_trial",
                f"accepted-trial evaluation failed: {error}",
                failure_z,
                failure_active,
                failure_scales,
                gmres_info,
                tuple(callback_values),
                gmres_residual_norm,
                gmres_limit,
                tuple(trial_alphas),
                tuple(line_search_rejections),
            ) from error
        if trial_active is None or accepted_trial_scales is None:
            raise fail(
                "accepted_trial",
                "accepted trial did not retain active-set and scale data",
                reference_z,
                reference_active,
                reference_scales,
                gmres_info,
                tuple(callback_values),
                gmres_residual_norm,
                gmres_limit,
                tuple(trial_alphas),
                tuple(line_search_rejections),
            )
        trial_converged = bool(
            all(value <= 1.0 for value in trial_norms)
            and active_set_stable
            and trial_correction_pass
        )
        row = MonolithicNewtonDiagnostics(
            iteration=iteration,
            z_before=reference_z,
            z_after=trial,
            delta=delta,
            active_mask_before=reference_active,
            active_mask_after=trial_active,
            active_set_stable=active_set_stable,
            linearization_scales=reference_scales,
            accepted_trial_scales=accepted_trial_scales,
            reference_scaled_residual=reference_scaled,
            line_search_scaled_residual=trial_scaled,
            accepted_trial_scaled_residual=trial_scaled_for_convergence,
            scaled_mechanics_l2=trial_norms[0],
            scaled_damage_inf=trial_norms[1],
            scaled_history_inf=trial_norms[2],
            merit_reference=reference_merit,
            merit_trial=trial_merit,
            gmres_info=gmres_info,
            gmres_unpreconditioned_residual=gmres_residual_norm,
            gmres_residual_limit=gmres_limit,
            gmres_callback_residuals=tuple(callback_values),
            trial_alphas=tuple(trial_alphas),
            line_search_rejections=tuple(line_search_rejections),
            accepted_alpha=accepted_alpha,
            line_search_backtracks=accepted_backtracks,
            correction_mechanics_material_inf=correction_mechanics_material,
            correction_mechanics_union_inf=correction_mechanics_union,
            correction_damage_inf=correction_damage,
            correction_history_inf=correction_history,
            correction_displacement_pass=correction_displacement_pass,
            correction_damage_pass=correction_damage_pass,
            correction_history_pass=correction_history_pass,
            correction_pass=trial_correction_pass,
        )
        diagnostics.append(row)
        if trial_converged:
            _, _, trial_history = monolithic.unpack(trial)
            if np.any(trial_history < 0.0):
                raise fail(
                    "accepted_trial",
                    "converged Newton result history must satisfy H>=0",
                    trial,
                    trial_active,
                    accepted_trial_scales,
                    gmres_info,
                    tuple(callback_values),
                    gmres_residual_norm,
                    gmres_limit,
                    tuple(trial_alphas),
                    tuple(line_search_rejections),
                )
            return _make_result(
                monolithic,
                trial,
                original_initial_z,
                raw_initial_residual,
                config,
                gate_config,
                accepted_trial_scales,
                diagnostics,
            )
        z = np.array(trial, dtype=np.float64, copy=True)
        active = trial_active
        scales = accepted_trial_scales
        scaled = trial_scaled_for_convergence

    raise fail(
        "max_newton_iterations",
        "maximum Newton iteration budget exhausted",
        z,
        active,
        scales,
        None,
        (),
        None,
        None,
        (),
        (),
    )


solve_scaled_semismooth_newton = solve_monolithic_newton


# ---------------------------------------------------------------------------
# Phase 2B single-target verified attempt
# ---------------------------------------------------------------------------


# These are the existing strict engineering-grip sweep gates, plus the final
# coupled gate owned by the verified-attempt contract.  The aliases exposed by
# ``MonolithicAuxiliarySweep.extended_gate_passes`` make the two equations that
# are often described as the damage fixed-point and damage-equation gates
# explicit without changing ``EngineeringGripDiagnostics.gate_passes``.
MONOLITHIC_AUXILIARY_GATE_NAMES = (
    "finite",
    "mechanics_before",
    "mechanics_after",
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
    "coupled",
)
_MONOLITHIC_CANONICAL_GATE_NAMES = (
    "finite",
    "monolithic_residual",
    "active_consistency",
    "history_nonnegative",
    "prescribed_constraints",
    "state_pairing",
    "auxiliary",
    "coupled",
)
_AUXILIARY_FAILURE_STAGES = frozenset(
    {
        "mechanics_before",
        "mechanics_before_gate",
        "amor_before",
        "damage",
        "rho_used",
        "damage_acceptance",
        "mechanics_after",
        "amor_after",
        "rho_next",
        "mechanics_after_gate",
        "diagnostics",
        "gates",
    }
)


def _optional_auxiliary_array(
    value: Any | None,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    if value is None:
        return None
    array = _f64(value, name, shape)
    return _immutable(array)


def _valid_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


_EPHEMERAL_DURABLE_COMPARISON_FIELDS = frozenset(
    {
        # These are Python/cache identities, not numerical or provenance data.
        "static_identity",
        "static_runtime_identity",
        "damage_runtime_identity",
    }
)


def _compare_durable_values(actual: Any, expected: Any, name: str) -> None:
    """Compare an audit tree while excluding only runtime identity fields."""

    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
            raise ValueError(f"{name} array/scalar kind differs from owner replay")
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not np.array_equal(actual, expected)
        ):
            raise ValueError(f"{name} array differs from owner replay")
        return
    if isinstance(actual, engineering.EngineeringGripModel) or isinstance(
        expected, engineering.EngineeringGripModel
    ):
        if not isinstance(actual, engineering.EngineeringGripModel) or not isinstance(
            expected, engineering.EngineeringGripModel
        ):
            raise ValueError(f"{name} model kind differs from owner replay")
        if actual.model_digest != expected.model_digest:
            raise ValueError(f"{name} model digest differs from owner replay")
        if actual.static_digest != expected.static_digest:
            raise ValueError(f"{name} static digest differs from owner replay")
        return
    if isinstance(actual, engineering.EngineeringGripConfig) or isinstance(
        expected, engineering.EngineeringGripConfig
    ):
        if not isinstance(actual, engineering.EngineeringGripConfig) or not isinstance(
            expected, engineering.EngineeringGripConfig
        ):
            raise ValueError(f"{name} config kind differs from owner replay")
        if actual.config_digest != expected.config_digest:
            raise ValueError(f"{name} config digest differs from owner replay")
        return
    if isinstance(actual, engineering.A2StaticMechanics) or isinstance(
        expected, engineering.A2StaticMechanics
    ):
        if not isinstance(actual, engineering.A2StaticMechanics) or not isinstance(
            expected, engineering.A2StaticMechanics
        ):
            raise ValueError(f"{name} static kind differs from owner replay")
        if actual.deterministic_digest != expected.deterministic_digest:
            raise ValueError(f"{name} static digest differs from owner replay")
        return
    if isinstance(actual, np.generic) or isinstance(expected, np.generic):
        if not isinstance(actual, np.generic) or not isinstance(expected, np.generic):
            raise ValueError(f"{name} scalar kind differs from owner replay")
        if actual.dtype != expected.dtype or actual.item() != expected.item():
            raise ValueError(f"{name} scalar differs from owner replay")
        return
    if is_dataclass(actual) or is_dataclass(expected):
        if not is_dataclass(actual) or not is_dataclass(expected) or type(actual) is not type(expected):
            raise ValueError(f"{name} dataclass type differs from owner replay")
        for field in fields(actual):
            if field.name in _EPHEMERAL_DURABLE_COMPARISON_FIELDS:
                continue
            _compare_durable_values(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{name}.{field.name}",
            )
        return
    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise ValueError(f"{name} mapping kind differs from owner replay")
        if actual.keys() != expected.keys():
            raise ValueError(f"{name} mapping keys differ from owner replay")
        for key in actual:
            _compare_durable_values(actual[key], expected[key], f"{name}[{key!r}]")
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"{name} sequence differs from owner replay")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare_durable_values(left, right, f"{name}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} differs from owner replay")


def _validated_gate_items(
    values: Any,
    names: tuple[str, ...],
    name: str,
) -> tuple[tuple[str, bool], ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be an ordered tuple of gate pairs")
    items = tuple(values)
    if tuple(item[0] for item in items if isinstance(item, tuple) and len(item) == 2) != names:
        raise ValueError(f"{name} must use the canonical gate order")
    if len(items) != len(names):
        raise ValueError(f"{name} has the wrong number of gates")
    result: list[tuple[str, bool]] = []
    for expected, item in zip(names, items, strict=True):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name} entries must be (name, bool) tuples")
        actual, passed = item
        if actual != expected or type(passed) is not bool:
            raise ValueError(f"{name} contains a non-canonical gate entry")
        result.append((actual, passed))
    return tuple(result)


def _a2_seed_digest(seed: Any) -> str:
    """Reproduce the authoritative A2 seed digest for provenance pairing."""

    values = _f64(seed, "A2 seed digest")
    return _a2_digest_primitives((("seed", values),))


@dataclass(frozen=True)
class MonolithicAuxiliarySweep:
    """Immutable authoritative sweep audit for one raw Newton candidate.

    ``raw_*`` are the untouched Phase-2A candidate blocks.  The remaining
    fields are produced by the existing engineering-grip ordering: mechanics
    at raw ``d``, committed-history Amor, exact fixed-history damage, mechanics
    at ``d_solved``, committed-history Amor again, and the final history
    maximum.  A complete failed sweep is retained just like a complete strict
    engineering sweep; an incomplete numerical failure retains every field
    available before its failure stage.
    """

    model: EngineeringGripModel
    config: EngineeringGripConfig
    committed_state: EngineeringGripState
    static_identity: object
    target_u_bar: float
    raw_z: np.ndarray
    raw_q: np.ndarray
    raw_damage: np.ndarray
    raw_history: np.ndarray
    complete_sweep: bool
    passed: bool
    failure_stage: str
    failure_reason: str
    exception_type: str | None
    mechanics_before: A2MechanicsSolveResult | None
    mechanics_after: A2MechanicsSolveResult | None
    q_before: np.ndarray | None
    q_after: np.ndarray | None
    psi_before: np.ndarray | None
    H_used: np.ndarray | None
    d_solved: np.ndarray | None
    psi_after: np.ndarray | None
    H_next: np.ndarray | None
    rho_used: np.ndarray | None
    rho_next: np.ndarray | None
    rho_used_volume: np.ndarray | None
    rho_next_volume: np.ndarray | None
    diagnostics: EngineeringGripDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.model, EngineeringGripModel):
            raise TypeError("auxiliary model has the wrong type")
        if not isinstance(self.config, EngineeringGripConfig):
            raise TypeError("auxiliary config has the wrong type")
        if not isinstance(self.committed_state, EngineeringGripState):
            raise TypeError("auxiliary committed_state has the wrong type")
        if self.static_identity is not self.model.static:
            raise ValueError("auxiliary static identity is not model-paired")
        target = _scalar(self.target_u_bar, "auxiliary target_u_bar")
        if target <= self.committed_state.u_bar or target > self.config.load_final:
            raise ValueError("auxiliary target_u_bar is outside the configured interval")
        object.__setattr__(self, "target_u_bar", target)

        raw_z = _f64(self.raw_z, "auxiliary raw_z")
        raw_q = _f64(self.raw_q, "auxiliary raw_q")
        raw_damage = _f64(self.raw_damage, "auxiliary raw_damage")
        raw_history = _f64(self.raw_history, "auxiliary raw_history")
        if any(array.ndim != 1 or array.size == 0 for array in (raw_z, raw_q, raw_damage, raw_history)):
            raise ValueError("auxiliary raw candidate blocks must be nonempty vectors")
        expected_z_shape = (
            len(self.model.constraints.free_dofs) + 2 * self.model.n_material,
        )
        if raw_z.shape != expected_z_shape:
            raise ValueError("auxiliary raw_z has the wrong model-bound shape")
        if raw_q.shape != (len(self.model.constraints.free_dofs),):
            raise ValueError("auxiliary raw_q has the wrong model-bound shape")
        if raw_damage.shape != (self.model.n_material,) or raw_history.shape != (
            self.model.n_material,
        ):
            raise ValueError("auxiliary raw damage/history have the wrong model-bound shape")
        if raw_damage.shape != raw_history.shape:
            raise ValueError("auxiliary raw damage/history shapes differ")
        if np.any(raw_history < 0.0):
            raise ValueError("auxiliary raw history must satisfy H>=0")
        expected_raw = np.concatenate((raw_q, raw_damage, raw_history))
        if raw_z.shape != expected_raw.shape or not np.array_equal(raw_z, expected_raw):
            raise ValueError("auxiliary raw blocks are not paired with raw_z")
        for name, value in (
            ("raw_z", raw_z),
            ("raw_q", raw_q),
            ("raw_damage", raw_damage),
            ("raw_history", raw_history),
        ):
            object.__setattr__(self, name, _immutable(value))

        if self.exception_type is not None and (
            not isinstance(self.exception_type, str) or not self.exception_type
        ):
            raise ValueError("auxiliary exception_type must be a nonempty string or None")
        if not isinstance(self.failure_stage, str) or not isinstance(
            self.failure_reason, str
        ):
            raise TypeError("auxiliary failure fields must be strings")
        complete = bool(self.complete_sweep)
        passed = bool(self.passed)
        object.__setattr__(self, "complete_sweep", complete)
        object.__setattr__(self, "passed", passed)
        if self.failure_stage and self.failure_stage not in _AUXILIARY_FAILURE_STAGES:
            raise ValueError(f"unknown auxiliary failure stage {self.failure_stage!r}")
        if complete and not self.failure_reason and not passed:
            raise ValueError("complete failed auxiliary sweep needs a failure reason")
        if not complete and (not self.failure_stage or not self.failure_reason):
            raise ValueError("incomplete auxiliary sweep needs failure provenance")
        if passed and (not complete or self.failure_stage or self.failure_reason):
            raise ValueError("passing auxiliary sweep cannot retain failure provenance")

        if self.mechanics_before is not None and not isinstance(
            self.mechanics_before, A2MechanicsSolveResult
        ):
            raise TypeError("auxiliary mechanics_before has the wrong type")
        if self.mechanics_after is not None and not isinstance(
            self.mechanics_after, A2MechanicsSolveResult
        ):
            raise TypeError("auxiliary mechanics_after has the wrong type")
        if not isinstance(self.diagnostics, EngineeringGripDiagnostics):
            raise TypeError("auxiliary diagnostics have the wrong type")
        if self.diagnostics.target_u_bar != target:
            raise ValueError("auxiliary diagnostic target is not paired")
        if not np.array_equal(
            self.diagnostics.damage_before, self.committed_state.material_damage
        ):
            raise ValueError("auxiliary diagnostic committed damage is not paired")
        if not np.array_equal(self.diagnostics.d_iter, raw_damage):
            raise ValueError("auxiliary raw_damage is not paired with diagnostics")

        diagnostic_gate_fields = (
            ("finite", "finite_pass"),
            ("mechanics_before", "mechanics_before_pass"),
            ("mechanics_after", "mechanics_after_pass"),
            ("displacement", "displacement_pass"),
            ("damage", "damage_pass"),
            ("history", "history_pass"),
            ("reaction", "reaction_pass"),
            ("damage_residual", "damage_residual_pass"),
            ("damage_range", "damage_range_pass"),
            ("healing", "healing_pass"),
            ("history_monotonic", "history_monotonic_pass"),
            ("force", "force_pass"),
            ("moment", "moment_pass"),
        )
        recomputed_diagnostic_gates = {
            name: bool(getattr(self.diagnostics, field))
            for name, field in diagnostic_gate_fields
        }
        if bool(self.diagnostics.passed) != all(recomputed_diagnostic_gates.values()):
            raise ValueError("auxiliary diagnostics passed flag is not gate-derived")
        recomputed_passed = bool(complete and all(recomputed_diagnostic_gates.values()))
        if passed != recomputed_passed:
            raise ValueError("auxiliary passed flag is not the strict gate conjunction")
        if bool(self.diagnostics.complete_sweep) != complete:
            raise ValueError("auxiliary diagnostic completion status is not paired")
        if passed:
            if self.failure_stage or self.failure_reason or self.exception_type is not None:
                raise ValueError("passing auxiliary sweep retains failure provenance")
            if self.diagnostics.failure_stage or self.diagnostics.failure_reason:
                raise ValueError("passing diagnostic retains failure provenance")
        elif complete:
            if self.failure_stage != "gates" or self.failure_reason != self.diagnostics.failure_reason:
                raise ValueError("complete auxiliary rejection has inconsistent gate provenance")
            if self.exception_type is not None or self.diagnostics.failure_stage != "gates":
                raise ValueError("complete auxiliary rejection has inconsistent exception provenance")
        else:
            if self.failure_stage != self.diagnostics.failure_stage or self.failure_reason != self.diagnostics.failure_reason:
                raise ValueError("incomplete auxiliary failure provenance is not diagnostic-paired")
        expected_available = {"damage_before", "d_iter"}
        optional_diagnostic_names = (
            "q_before",
            "q_after",
            "psi_before",
            "psi_after",
            "H_used",
            "H_next",
            "d_next",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        )
        expected_available.update(
            name
            for name in optional_diagnostic_names
            if getattr(self.diagnostics, name) is not None
        )
        if tuple(sorted(expected_available)) != self.diagnostics.available_fields:
            raise ValueError("auxiliary diagnostic available_fields are not presence-derived")

        n_material = raw_damage.size
        for name in (
            "psi_before",
            "H_used",
            "d_solved",
            "psi_after",
            "H_next",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        ):
            object.__setattr__(
                self,
                name,
                _optional_auxiliary_array(
                    getattr(self, name), f"auxiliary.{name}", (n_material,)
                ),
            )
        def pair_optional(
            audit_name: str, diagnostic_name: str | None = None
        ) -> None:
            diagnostic_field = audit_name if diagnostic_name is None else diagnostic_name
            audit_value = getattr(self, audit_name)
            diagnostic_value = getattr(self.diagnostics, diagnostic_field)
            if (audit_value is None) != (diagnostic_value is None):
                raise ValueError(
                    f"auxiliary {audit_name} has one-sided diagnostic presence"
                )
            if audit_value is not None and not np.array_equal(audit_value, diagnostic_value):
                raise ValueError(f"auxiliary {audit_name} is not paired with diagnostics")

        for name in (
            "psi_before",
            "psi_after",
            "H_used",
            "H_next",
            "d_solved",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        ):
            pair_optional(name, "d_next" if name == "d_solved" else None)
        for name in ("q_before", "q_after"):
            pair_optional(name)
        for result_name, q_name in (
            ("mechanics_before", "q_before"),
            ("mechanics_after", "q_after"),
        ):
            mechanics = getattr(self, result_name)
            q_value = getattr(self, q_name)
            diagnostic_q = getattr(self.diagnostics, q_name)
            if (mechanics is None) != (q_value is None) or (q_value is None) != (
                diagnostic_q is None
            ):
                raise ValueError(f"auxiliary {result_name}/{q_name} presence is inconsistent")
            if mechanics is not None:
                if mechanics.displacement.shape != (self.model.n_dofs,):
                    raise ValueError(f"auxiliary {result_name} displacement has the wrong shape")
                if not np.array_equal(mechanics.displacement, q_value) or not np.array_equal(
                    mechanics.displacement, diagnostic_q
                ):
                    raise ValueError(f"auxiliary {result_name} displacement is not paired")

        self._validate_mechanics_provenance(
            self.mechanics_before,
            raw_damage,
            self.reconstruct_raw_displacement(raw_z),
            "mechanics_before",
            "monolithic_newton_candidate",
            "monolithic_raw_candidate",
        )
        if self.mechanics_after is not None:
            if self.d_solved is None or self.mechanics_before is None:
                raise ValueError("auxiliary mechanics_after lacks its solved-damage seed chain")
            self._validate_mechanics_provenance(
                self.mechanics_after,
                self.d_solved,
                self.mechanics_before.displacement,
                "mechanics_after",
                "mechanics_before_solution",
                "monolithic_mechanics_before",
            )

        for name in ("q_before", "q_after"):
            value = getattr(self, name)
            if value is not None:
                array = _f64(value, f"auxiliary.{name}")
                if array.shape != (self.model.n_dofs,):
                    raise ValueError(f"auxiliary.{name} must match model union DOFs")
                object.__setattr__(self, name, _immutable(array))

        if complete:
            required = (
                "mechanics_before",
                "mechanics_after",
                "q_before",
                "q_after",
                "psi_before",
                "H_used",
                "d_solved",
                "psi_after",
                "H_next",
                "rho_used",
                "rho_next",
                "rho_used_volume",
                "rho_next_volume",
            )
            if any(getattr(self, name) is None for name in required):
                raise ValueError("complete auxiliary sweep is missing an authoritative field")

        pair_optional("d_solved", "d_next")
        for name in (
            "psi_before",
            "psi_after",
            "H_used",
            "H_next",
            "rho_used",
            "rho_next",
            "rho_used_volume",
            "rho_next_volume",
        ):
            pair_optional(name)
        for name in ("q_before", "q_after"):
            pair_optional(name)
        diagnostic_pairs = (
            ("psi_before", "psi_before"),
            ("psi_after", "psi_after"),
            ("H_used", "H_used"),
            ("H_next", "H_next"),
            ("rho_used", "rho_used"),
            ("rho_next", "rho_next"),
            ("rho_used_volume", "rho_used_volume"),
            ("rho_next_volume", "rho_next_volume"),
        )
        for diagnostic_name, audit_name in diagnostic_pairs:
            diagnostic_value = getattr(self.diagnostics, diagnostic_name)
            audit_value = getattr(self, audit_name)
            if diagnostic_value is not None and audit_value is not None:
                if not np.array_equal(diagnostic_value, audit_value):
                    raise ValueError(
                        f"auxiliary {audit_name} is not paired with diagnostics"
                    )
        if self.diagnostics.q_before is not None and self.q_before is not None:
            if not np.array_equal(self.diagnostics.q_before, self.q_before):
                raise ValueError("auxiliary q_before is not paired with diagnostics")
        if self.diagnostics.q_after is not None and self.q_after is not None:
            if not np.array_equal(self.diagnostics.q_after, self.q_after):
                raise ValueError("auxiliary q_after is not paired with diagnostics")

    def reconstruct_raw_displacement(self, raw_z: np.ndarray) -> np.ndarray:
        """Reconstruct the raw Newton full displacement for seed validation."""

        prescribed = self.model.constraints.prescribed_vector(self.target_u_bar)
        prescribed[self.model.constraints.free_dofs] += raw_z[: len(self.model.constraints.free_dofs)]
        return np.asarray(prescribed, dtype=np.float64)

    def _validate_mechanics_provenance(
        self,
        result: A2MechanicsSolveResult | None,
        target_damage: np.ndarray,
        expected_seed: np.ndarray,
        result_name: str,
        expected_kind: str,
        expected_source_state: str,
    ) -> None:
        if result is None:
            return
        if result.constraints is not self.model.constraints:
            raise ValueError(f"auxiliary {result_name} constraints are foreign")
        if result.static_runtime_identity is not self.model.static.runtime_identity:
            raise ValueError(f"auxiliary {result_name} static runtime identity is foreign")
        if result.static_digest != self.model.static_digest:
            raise ValueError(f"auxiliary {result_name} static digest is foreign")
        if result.outer_displacement != self.target_u_bar:
            raise ValueError(f"auxiliary {result_name} target is not paired")
        target_damage_digest = engineering._damage_digest(target_damage)
        raw_damage_digest = engineering._damage_digest(self.raw_damage)
        provenance = result.seed_provenance
        if provenance.kind != expected_kind or provenance.source_state != expected_source_state:
            raise ValueError(f"auxiliary {result_name} seed kind/state is not authoritative")
        if provenance.source_step != 0 or provenance.source_sweep != 0:
            raise ValueError(f"auxiliary {result_name} seed step/sweep is not target-local")
        if provenance.source_damage_digest != raw_damage_digest:
            raise ValueError(f"auxiliary {result_name} source damage digest is not raw-paired")
        if provenance.target_damage_digest != target_damage_digest:
            raise ValueError(f"auxiliary {result_name} target damage digest is not paired")
        if result.damage_digest != target_damage_digest:
            raise ValueError(f"auxiliary {result_name} result damage digest is not paired")
        if provenance.seed_digest != _a2_seed_digest(expected_seed):
            raise ValueError(f"auxiliary {result_name} seed digest is not paired")

    @property
    def gate_passes(self) -> dict[str, bool]:
        values = dict(self.diagnostics.gate_passes)
        values["coupled"] = bool(self.complete_sweep and all(values.values()))
        return values

    @property
    def extended_gate_passes(self) -> dict[str, bool]:
        """Expose explicit damage-fixed-point/equation aliases for review."""

        values = self.gate_passes
        values["damage_fixed_point"] = values["damage"]
        values["damage_equation"] = values["damage_residual"]
        values["fixed_point"] = values["damage"]
        return values

    @property
    def raw_candidate(self) -> np.ndarray:
        return self.raw_z

    @property
    def state_displacement(self) -> np.ndarray | None:
        return self.q_after


@dataclass(frozen=True)
class MonolithicAttemptFailureAudit:
    """Immutable failure provenance for one rejected Phase-2B attempt."""

    stage: str
    reason: str
    exception_type: str | None
    current_z: np.ndarray
    committed_before_digest: str
    committed_after_digest: str
    newton_audit: MonolithicNewtonFailureAudit | None
    auxiliary: MonolithicAuxiliarySweep | None
    canonical_z: np.ndarray | None
    canonical_active_mask: np.ndarray | None
    canonical_scales: MonolithicNewtonScales | None
    canonical_scaled_residual: np.ndarray | None
    canonical_gate_passes: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("attempt failure stage must be nonempty")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("attempt failure reason must be nonempty")
        if self.exception_type is not None and (
            not isinstance(self.exception_type, str) or not self.exception_type
        ):
            raise ValueError("attempt failure exception_type must be nonempty or None")
        current = _f64(self.current_z, "attempt failure current_z")
        if current.ndim != 1 or current.size == 0:
            raise ValueError("attempt failure current_z must be a nonempty vector")
        object.__setattr__(self, "current_z", _immutable(current))
        before = _valid_digest(
            self.committed_before_digest, "attempt failure committed_before_digest"
        )
        after = _valid_digest(
            self.committed_after_digest, "attempt failure committed_after_digest"
        )
        if before != after:
            raise ValueError("rejected attempt failure audit changed committed state")
        object.__setattr__(self, "committed_before_digest", before)
        object.__setattr__(self, "committed_after_digest", after)
        if self.newton_audit is not None and not isinstance(
            self.newton_audit, MonolithicNewtonFailureAudit
        ):
            raise TypeError("attempt failure newton_audit has the wrong type")
        if self.auxiliary is not None and not isinstance(
            self.auxiliary, MonolithicAuxiliarySweep
        ):
            raise TypeError("attempt failure auxiliary audit has the wrong type")
        for name in ("canonical_z", "canonical_scaled_residual"):
            value = getattr(self, name)
            if value is not None:
                array = _f64(value, f"attempt failure {name}")
                if array.ndim != 1 or array.size == 0:
                    raise ValueError(f"attempt failure {name} must be a vector")
                object.__setattr__(self, name, _immutable(array))
        if self.canonical_active_mask is not None:
            active = np.asarray(self.canonical_active_mask)
            if active.dtype != np.dtype(bool) or active.ndim != 1:
                raise ValueError("attempt failure canonical active mask is invalid")
            object.__setattr__(self, "canonical_active_mask", _immutable(active))
        if self.canonical_scales is not None and not isinstance(
            self.canonical_scales, MonolithicNewtonScales
        ):
            raise TypeError("attempt failure canonical scales have the wrong type")
        object.__setattr__(
            self,
            "canonical_gate_passes",
            _validated_gate_items(
                self.canonical_gate_passes,
                _MONOLITHIC_CANONICAL_GATE_NAMES,
                "canonical_gate_passes",
            ),
        )

    @property
    def canonical_gates(self) -> dict[str, bool]:
        return dict(self.canonical_gate_passes)


@dataclass(frozen=True)
class MonolithicVerifiedAttempt:
    """One immutable, single-target Phase-2B monolithic attempt.

    This object is an in-memory audit/result contract only.  No archive writer
    or JSON serializer is provided for rejected attempts, so raw rejected
    vectors cannot become durable physical evidence accidentally.
    """

    model: EngineeringGripModel
    config: EngineeringGripConfig
    committed_state: EngineeringGripState
    newton_config: MonolithicNewtonConfig
    newton_gate_config: MonolithicNewtonGateConfig
    target_u_bar: float
    adapter_kind: str
    initial_z: np.ndarray
    accepted: bool
    committed_before_digest: str
    committed_after_digest: str
    newton_result: MonolithicNewtonResult | None
    newton_audit: MonolithicNewtonFailureAudit | None
    auxiliary: MonolithicAuxiliarySweep | None
    canonical_z: np.ndarray | None
    canonical_active_mask: np.ndarray | None
    canonical_scales: MonolithicNewtonScales | None
    canonical_scaled_residual: np.ndarray | None
    canonical_gate_passes: tuple[tuple[str, bool], ...]
    state: EngineeringGripState | None
    failure: MonolithicAttemptFailureAudit | None

    def __post_init__(self) -> None:
        if not isinstance(self.model, EngineeringGripModel):
            raise TypeError("verified attempt model has the wrong type")
        if not isinstance(self.config, EngineeringGripConfig):
            raise TypeError("verified attempt config has the wrong type")
        if not isinstance(self.committed_state, EngineeringGripState):
            raise TypeError("verified attempt committed_state has the wrong type")
        if not isinstance(self.newton_config, MonolithicNewtonConfig):
            raise TypeError("verified attempt newton_config has the wrong type")
        if not isinstance(self.newton_gate_config, MonolithicNewtonGateConfig):
            raise TypeError("verified attempt newton_gate_config has the wrong type")
        target = _scalar(self.target_u_bar, "verified attempt target_u_bar")
        if target > self.config.load_final:
            raise ValueError("verified attempt target exceeds the configured forward interval")
        object.__setattr__(self, "target_u_bar", target)
        if not isinstance(self.adapter_kind, str) or not self.adapter_kind:
            raise ValueError("verified attempt adapter_kind must be nonempty")
        initial = _f64(
            self.initial_z,
            "verified attempt initial_z",
            (len(self.model.constraints.free_dofs) + 2 * self.model.n_material,),
        )
        object.__setattr__(self, "initial_z", _immutable(initial))
        object.__setattr__(self, "accepted", bool(self.accepted))
        before = _valid_digest(
            self.committed_before_digest, "verified attempt committed_before_digest"
        )
        after = _valid_digest(
            self.committed_after_digest, "verified attempt committed_after_digest"
        )
        if before != self.committed_state.digest:
            raise ValueError("verified attempt committed-before digest is not paired")
        object.__setattr__(self, "committed_before_digest", before)
        object.__setattr__(self, "committed_after_digest", after)

        expected_packed = len(self.model.constraints.free_dofs) + 2 * self.model.n_material
        if self.newton_result is not None:
            if not isinstance(self.newton_result, MonolithicNewtonResult):
                raise TypeError("verified attempt newton_result has the wrong type")
            if self.newton_result.z.shape != (expected_packed,):
                raise ValueError("verified attempt Newton result has the wrong packed shape")
            if self.newton_result.target_u_bar != target:
                raise ValueError("verified attempt Newton target is not paired")
            if self.newton_result.model_identity is not self.model:
                raise ValueError("verified attempt Newton model identity is foreign")
            if self.newton_result.static_identity is not self.model.static:
                raise ValueError("verified attempt Newton static identity is foreign")
            if self.newton_result.config is not self.newton_config:
                raise ValueError("verified attempt Newton config is foreign")
            if self.newton_result.gate_config is not self.newton_gate_config:
                raise ValueError("verified attempt Newton gate config is foreign")
            if np.any(self.newton_result.z[-self.model.n_material :] < 0.0):
                raise ValueError("verified attempt Newton final history must satisfy H>=0")
        if self.newton_audit is not None and not isinstance(
            self.newton_audit, MonolithicNewtonFailureAudit
        ):
            raise TypeError("verified attempt Newton audit has the wrong type")
        if self.auxiliary is not None:
            if not isinstance(self.auxiliary, MonolithicAuxiliarySweep):
                raise TypeError("verified attempt auxiliary audit has the wrong type")
            if self.auxiliary.raw_z.shape != (expected_packed,):
                raise ValueError("verified attempt auxiliary raw state has the wrong shape")
            if self.auxiliary.model is not self.model:
                raise ValueError("verified attempt auxiliary model identity is foreign")
            if self.auxiliary.config is not self.config:
                raise ValueError("verified attempt auxiliary config is foreign")
            if self.auxiliary.committed_state is not self.committed_state:
                raise ValueError("verified attempt auxiliary committed state is foreign")
            if self.auxiliary.static_identity is not self.model.static:
                raise ValueError("verified attempt auxiliary static identity is foreign")
            if self.auxiliary.target_u_bar != target:
                raise ValueError("verified attempt auxiliary target is not paired")
            if self.newton_result is not None and not np.array_equal(
                self.auxiliary.raw_z, self.newton_result.z
            ):
                raise ValueError("verified attempt auxiliary raw state is not Newton-paired")
        for name in ("canonical_z", "canonical_scaled_residual"):
            value = getattr(self, name)
            if value is not None:
                array = _f64(value, f"verified attempt {name}", (expected_packed,))
                object.__setattr__(self, name, _immutable(array))
        if self.canonical_active_mask is not None:
            active = np.asarray(self.canonical_active_mask)
            if active.dtype != np.dtype(bool) or active.shape != (self.model.n_material,):
                raise ValueError("verified attempt canonical active mask has the wrong shape")
            object.__setattr__(self, "canonical_active_mask", _immutable(active))
        if self.canonical_scales is not None:
            if not isinstance(self.canonical_scales, MonolithicNewtonScales):
                raise TypeError("verified attempt canonical scales have the wrong type")
            _validate_model_scale_shapes(
                MonolithicGripResidual(
                    self.model, target, self.committed_state.material_history
                ),
                self.canonical_scales,
                "verified attempt canonical scales",
            )
            if self.canonical_z is None:
                raise ValueError("canonical scales require canonical_z")
            if self.canonical_scales.state_digest != _state_digest(self.canonical_z):
                raise ValueError("verified attempt canonical scales are not state-paired")
            if not np.array_equal(
                self.canonical_z[-self.model.n_material :], self.canonical_scales.H_used
            ):
                raise ValueError("verified attempt canonical H_used is not state-paired")
            if self.canonical_scaled_residual is None:
                raise ValueError("canonical scales require canonical scaled residual")
            canonical_owner = MonolithicGripResidual(
                self.model, target, self.committed_state.material_history
            )
            expected_scaled = canonical_owner.scaled_residual(
                self.canonical_z, self.canonical_scales
            )
            if not np.array_equal(self.canonical_scaled_residual, expected_scaled):
                raise ValueError(
                    "verified attempt canonical scaled residual is not state-paired"
                )
            expected_active = canonical_owner.active_mask(self.canonical_z)
            if self.canonical_active_mask is None:
                raise ValueError("canonical scales require canonical active mask")
            if not np.array_equal(self.canonical_active_mask, expected_active):
                raise ValueError(
                    "verified attempt canonical active mask is not state-paired"
                )
        elif self.canonical_scaled_residual is not None:
            raise ValueError("canonical scaled residual requires canonical scales")
        object.__setattr__(
            self,
            "canonical_gate_passes",
            _validated_gate_items(
                self.canonical_gate_passes,
                _MONOLITHIC_CANONICAL_GATE_NAMES,
                "canonical_gate_passes",
            ),
        )
        canonical_gates = dict(self.canonical_gate_passes)
        expected_canonical_coupled = all(
            value for name, value in canonical_gates.items() if name != "coupled"
        )
        if canonical_gates["coupled"] != expected_canonical_coupled:
            raise ValueError("canonical coupled gate is not the strict conjunction")
        if canonical_gates["auxiliary"] != bool(
            self.auxiliary is not None and self.auxiliary.passed
        ):
            raise ValueError("canonical auxiliary gate is not auxiliary-paired")
        canonical_fields = (
            self.canonical_z,
            self.canonical_active_mask,
            self.canonical_scales,
            self.canonical_scaled_residual,
        )
        if any(value is None for value in canonical_fields) and not all(
            value is None for value in canonical_fields
        ):
            raise ValueError("canonical audit fields have one-sided presence")
        if self.failure is not None and not isinstance(
            self.failure, MonolithicAttemptFailureAudit
        ):
            raise TypeError("verified attempt failure has the wrong type")
        if self.state is not None and not isinstance(self.state, EngineeringGripState):
            raise TypeError("verified attempt state has the wrong type")
        controller_no_progress = bool(
            self.failure is not None
            and self.failure.stage == "controller"
            and not self.accepted
            and target == self.committed_state.u_bar
        )
        if target <= self.committed_state.u_bar and not controller_no_progress:
            raise ValueError("verified attempt target is outside the configured forward interval")
        expected_accepted = bool(
            self.newton_result is not None
            and self.auxiliary is not None
            and self.auxiliary.passed
            and canonical_gates["coupled"]
            and self.state is not None
            and self.failure is None
        )
        if bool(self.accepted) != expected_accepted:
            raise ValueError("verified attempt accepted status is not gate-derived")
        if self.accepted:
            if self.failure is not None:
                raise ValueError("accepted verified attempt cannot retain failure audit")
            if self.newton_audit is not None:
                raise ValueError("accepted verified attempt cannot retain Newton failure audit")
            if self.newton_result is None or self.auxiliary is None:
                raise ValueError("accepted verified attempt is missing solver/auxiliary audit")
            if self.state is None or self.canonical_z is None:
                raise ValueError("accepted verified attempt is missing canonical state")
            if (
                self.canonical_active_mask is None
                or self.canonical_scales is None
                or self.canonical_scaled_residual is None
            ):
                raise ValueError("accepted verified attempt is missing canonical audit fields")
            if not self.auxiliary.passed:
                raise ValueError("accepted verified attempt has failed auxiliary gates")
            canonical_q = self.state.union_displacement[self.model.constraints.free_dofs]
            expected = np.concatenate((canonical_q, self.state.d, self.state.H))
            if not np.array_equal(expected, self.canonical_z):
                raise ValueError("accepted canonical state is not paired with canonical_z")
            if self.state.u_bar != target:
                raise ValueError("accepted state load is not paired with target")
            engineering._check_exact_constraints(
                self.state.union_displacement,
                self.model.constraints,
                target,
                "verified accepted state",
            )
            if after != self.state.digest:
                raise ValueError("accepted committed-after digest is not state-paired")
            if not all(value for _, value in self.canonical_gate_passes):
                raise ValueError("accepted verified attempt has a failed canonical gate")
        else:
            if self.failure is None:
                raise ValueError("rejected verified attempt must retain failure audit")
            if self.state is not None:
                raise ValueError("rejected verified attempt cannot expose a committed state")
            if after != before:
                raise ValueError("rejected verified attempt changed committed state")
            if self.failure.committed_before_digest != before or self.failure.committed_after_digest != after:
                raise ValueError("verified attempt failure audit digest is not paired")
            if self.failure.auxiliary is not self.auxiliary:
                raise ValueError("verified attempt failure auxiliary is foreign")
            if self.failure.newton_audit is not self.newton_audit:
                raise ValueError("verified attempt failure Newton audit is foreign")
            if self.failure.canonical_gate_passes != self.canonical_gate_passes:
                raise ValueError("verified attempt failure canonical gates are foreign")
            if self.failure.canonical_scales is not self.canonical_scales:
                raise ValueError("verified attempt failure canonical scales are foreign")
            for name in ("canonical_z", "canonical_scaled_residual"):
                failure_value = getattr(self.failure, name)
                result_value = getattr(self, name)
                if (failure_value is None) != (result_value is None):
                    raise ValueError(f"verified attempt failure {name} has one-sided presence")
                if failure_value is not None and not np.array_equal(failure_value, result_value):
                    raise ValueError(f"verified attempt failure {name} is foreign")
            if (self.failure.canonical_active_mask is None) != (
                self.canonical_active_mask is None
            ):
                raise ValueError("verified attempt failure active mask has one-sided presence")
            if self.failure.canonical_active_mask is not None and not np.array_equal(
                self.failure.canonical_active_mask, self.canonical_active_mask
            ):
                raise ValueError("verified attempt failure active mask is foreign")
            if self.failure.stage == "newton":
                if self.newton_result is not None or self.auxiliary is not None:
                    raise ValueError("Newton failure retains a foreign later-stage audit")
                if self.newton_audit is None or self.failure.newton_audit is None:
                    raise ValueError("Newton failure is missing its Newton audit")
                audit = self.newton_audit
                if audit.target_u_bar != target or audit.model_identity is not self.model:
                    raise ValueError("Newton failure audit identity is foreign")
                if audit.static_identity is not self.model.static:
                    raise ValueError("Newton failure audit static identity is foreign")
                if audit.config is not self.newton_config or audit.gate_config is not self.newton_gate_config:
                    raise ValueError("Newton failure audit solver configuration is foreign")
                if not np.array_equal(self.failure.current_z, audit.current_z):
                    raise ValueError("Newton failure current state is not audit-paired")
                if self.failure.reason != audit.reason or self.failure.exception_type != "MonolithicNewtonFailure":
                    raise ValueError("Newton failure provenance is inconsistent")
            elif self.failure.stage == "auxiliary":
                if self.newton_result is None or self.auxiliary is None or self.newton_audit is not None:
                    raise ValueError("auxiliary failure retains an inconsistent solver audit")
                if not np.array_equal(self.failure.current_z, self.auxiliary.raw_z):
                    raise ValueError("auxiliary failure current state is not raw-paired")
                if self.failure.reason != self.auxiliary.failure_reason or self.failure.exception_type != self.auxiliary.exception_type:
                    raise ValueError("auxiliary failure provenance is inconsistent")
            elif self.failure.stage == "canonical":
                if self.newton_result is None or self.auxiliary is None or self.newton_audit is not None:
                    raise ValueError("canonical failure retains an inconsistent solver audit")
                if self.canonical_z is None or not np.array_equal(
                    self.failure.current_z, self.canonical_z
                ):
                    raise ValueError("canonical failure current state is not canonical-paired")
                failed_names = tuple(
                    name for name, value in self.canonical_gate_passes if not value
                )
                if self.failure.reason != ", ".join(failed_names) or self.failure.exception_type is not None:
                    raise ValueError("canonical failure provenance is inconsistent")
            elif self.failure.stage == "controller":
                if (
                    self.newton_result is not None
                    or self.newton_audit is not None
                    or self.auxiliary is not None
                    or self.canonical_z is not None
                    or self.canonical_active_mask is not None
                    or self.canonical_scales is not None
                    or self.canonical_scaled_residual is not None
                ):
                    raise ValueError("controller failure retains a solver audit")
                if not np.array_equal(self.failure.current_z, self.initial_z):
                    raise ValueError("controller failure current state is not predictor-paired")
                if self.failure.exception_type is not None:
                    raise ValueError("controller failure cannot retain an exception type")
            else:
                raise ValueError("verified attempt failure has an unknown stage")

    def validate(self) -> None:
        """Re-run this attempt contract and recursively validate its Newton audit."""

        if self.newton_result is not None:
            self.newton_result.validate()
        type(self)(
            model=self.model,
            config=self.config,
            committed_state=self.committed_state,
            newton_config=self.newton_config,
            newton_gate_config=self.newton_gate_config,
            target_u_bar=self.target_u_bar,
            adapter_kind=self.adapter_kind,
            initial_z=self.initial_z,
            accepted=self.accepted,
            committed_before_digest=self.committed_before_digest,
            committed_after_digest=self.committed_after_digest,
            newton_result=self.newton_result,
            newton_audit=self.newton_audit,
            auxiliary=self.auxiliary,
            canonical_z=self.canonical_z,
            canonical_active_mask=self.canonical_active_mask,
            canonical_scales=self.canonical_scales,
            canonical_scaled_residual=self.canonical_scaled_residual,
            canonical_gate_passes=self.canonical_gate_passes,
            state=self.state,
            failure=self.failure,
        )

    def validate_with_owner(
        self,
        owner: MonolithicGripResidual | EngineeringGripModel,
        config: EngineeringGripConfig | None = None,
    ) -> None:
        return self._validate_with_owner_runner(
            owner,
            config,
            _AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT,
        )

    def _validate_with_owner_runner(
        self,
        owner: MonolithicGripResidual | EngineeringGripModel,
        config: EngineeringGripConfig | None,
        runner: Any,
    ) -> None:
        """Authenticate this attempt by replaying its exact Phase-2B input.

        The structural contract is checked first.  The authoritative replay is
        then performed from the retained predictor/target/config using the
        production CPU reference entry point with no callback, mock, or caller
        supplied solver hook.  The comparison is exhaustive over durable
        values; fixed-view/cache and static Python identities are intentionally
        excluded because every replay constructs fresh runtime objects.
        """

        self.validate()
        if isinstance(owner, MonolithicGripResidual):
            owner_model = owner._model
            owner_config = self.config if config is None else config
            if owner.u_bar != self.target_u_bar:
                raise ValueError("verified attempt owner target is not paired")
            if not np.array_equal(
                owner.committed_history, self.committed_state.material_history
            ):
                raise ValueError("verified attempt owner history is not paired")
        elif isinstance(owner, EngineeringGripModel):
            if config is None:
                raise TypeError(
                    "verified attempt model owner requires EngineeringGripConfig"
                )
            owner_model = owner
            owner_config = config
        else:
            raise TypeError("verified attempt owner has the wrong type")
        if not isinstance(owner_config, EngineeringGripConfig):
            raise TypeError("verified attempt owner config has the wrong type")
        if owner_model.model_digest != self.model.model_digest:
            raise ValueError("verified attempt owner model digest differs")
        if owner_model.static_digest != self.model.static_digest:
            raise ValueError("verified attempt owner static digest differs")
        if owner_config.config_digest != self.config.config_digest:
            raise ValueError("verified attempt owner config digest differs")

        replay_owner = (
            owner
            if isinstance(owner, MonolithicGripResidual)
            else MonolithicGripResidual(
                owner_model,
                self.target_u_bar,
                self.committed_state.material_history,
            )
        )
        if self.newton_result is not None:
            self.newton_result.validate_with_owner(replay_owner)
        if self.adapter_kind != "cpu_reference":
            raise ValueError(
                "verified attempt owner replay requires the authoritative cpu_reference path"
            )
        replay = runner(
            owner_model,
            owner_config,
            self.committed_state,
            self.target_u_bar,
            None,
            self.initial_z,
            self.newton_config,
        )
        if not isinstance(replay, MonolithicVerifiedAttempt):
            raise TypeError("authoritative verified-attempt replay returned the wrong type")
        _compare_durable_values(self, replay, "verified attempt")

    @property
    def committed_before(self) -> EngineeringGripState:
        return self.committed_state

    @property
    def committed_after(self) -> EngineeringGripState | None:
        return self.state

    @property
    def final_state(self) -> EngineeringGripState | None:
        return self.state

    @property
    def failure_stage(self) -> str | None:
        return None if self.failure is None else self.failure.stage

    @property
    def failure_reason(self) -> str | None:
        return None if self.failure is None else self.failure.reason

    @property
    def exception_type(self) -> str | None:
        return None if self.failure is None else self.failure.exception_type

    @property
    def gate_passes(self) -> dict[str, bool]:
        values = {
            name: False for name in MONOLITHIC_AUXILIARY_GATE_NAMES
        }
        if self.auxiliary is not None:
            values.update(self.auxiliary.gate_passes)
        canonical = dict(self.canonical_gate_passes)
        values["coupled"] = bool(values["coupled"] and canonical.get("coupled", False))
        return values

    @property
    def extended_gate_passes(self) -> dict[str, bool]:
        values = self.gate_passes
        values["damage_fixed_point"] = values["damage"]
        values["damage_equation"] = values["damage_residual"]
        values["fixed_point"] = values["damage"]
        return values


def _auxiliary_audit(
    *,
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    target: float,
    raw_z: np.ndarray,
    raw_q: np.ndarray,
    raw_damage: np.ndarray,
    raw_history: np.ndarray,
    stage: str,
    reason: str,
    exception_type: str | None,
    complete: bool,
    before: A2MechanicsSolveResult | None,
    after: A2MechanicsSolveResult | None,
    psi_before: np.ndarray | None,
    H_used: np.ndarray | None,
    d_solved: np.ndarray | None,
    psi_after: np.ndarray | None,
    H_next: np.ndarray | None,
    rho_used: np.ndarray | None,
    rho_next: np.ndarray | None,
    rho_used_volume: np.ndarray | None,
    rho_next_volume: np.ndarray | None,
    residual_scale: float | None = None,
) -> MonolithicAuxiliarySweep:
    if complete:
        if not isinstance(before, A2MechanicsSolveResult) or not isinstance(
            after, A2MechanicsSolveResult
        ):
            raise TypeError("complete auxiliary audit requires both mechanics results")
        diagnostic = engineering._full_diagnostics(
            config,
            model,
            committed,
            raw_damage,
            d_solved,
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
            _scalar(residual_scale, "auxiliary residual_scale"),
            0,
            target,
        )
    else:
        diagnostic = engineering._incomplete_diagnostics(
            config,
            committed,
            raw_damage,
            target,
            stage,
            reason,
            before=before,
            after=after,
            psi_before=psi_before,
            psi_after=psi_after,
            H_used=H_used,
            d_next=d_solved,
            H_next=H_next,
            rho_used=rho_used,
            rho_next=rho_next,
            rho_used_volume=rho_used_volume,
            rho_next_volume=rho_next_volume,
            model=model,
            sweep_index=0,
        )
    final_stage = "" if diagnostic.passed else (stage or "gates")
    final_reason = "" if diagnostic.passed else (reason or diagnostic.failure_reason)
    return MonolithicAuxiliarySweep(
        model=model,
        config=config,
        committed_state=committed,
        static_identity=model.static,
        target_u_bar=target,
        raw_z=raw_z,
        raw_q=raw_q,
        raw_damage=raw_damage,
        raw_history=raw_history,
        complete_sweep=bool(complete),
        passed=bool(diagnostic.passed),
        failure_stage=final_stage,
        failure_reason=final_reason,
        exception_type=exception_type,
        mechanics_before=before,
        mechanics_after=after,
        q_before=None if before is None else before.displacement,
        q_after=None if after is None else after.displacement,
        psi_before=psi_before,
        H_used=H_used,
        d_solved=d_solved,
        psi_after=psi_after,
        H_next=H_next,
        rho_used=rho_used,
        rho_next=rho_next,
        rho_used_volume=rho_used_volume,
        rho_next_volume=rho_next_volume,
        diagnostics=diagnostic,
    )


def _run_monolithic_auxiliary_sweep(
    monolithic: MonolithicGripResidual,
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    target: float,
    adapter: Any,
    raw_z: np.ndarray,
) -> MonolithicAuxiliarySweep:
    """Run exactly one authoritative strict sweep from a raw Newton state."""

    raw_q, raw_damage, raw_history = monolithic.unpack(raw_z)
    raw_q = np.array(raw_q, dtype=np.float64, copy=True)
    raw_damage = np.array(raw_damage, dtype=np.float64, copy=True)
    raw_history = np.array(raw_history, dtype=np.float64, copy=True)
    raw_displacement = monolithic.reconstruct_displacement(raw_z)
    before: A2MechanicsSolveResult | None = None
    after: A2MechanicsSolveResult | None = None
    psi_before: np.ndarray | None = None
    H_used: np.ndarray | None = None
    d_solved: np.ndarray | None = None
    psi_after: np.ndarray | None = None
    H_next: np.ndarray | None = None
    rho_used: np.ndarray | None = None
    rho_next: np.ndarray | None = None
    rho_used_volume: np.ndarray | None = None
    rho_next_volume: np.ndarray | None = None
    residual_scale: float | None = None

    try:
        before = engineering._solve_direct_mechanics(
            adapter,
            model.fixed_damage_view(raw_damage),
            model.constraints,
            target,
            config.cg_rtol,
            config.cg_atol,
            config.cg_maxiter,
            initial_guess=np.array(raw_displacement, dtype=np.float64, copy=True),
            seed_kind="monolithic_newton_candidate",
            source_state="monolithic_raw_candidate",
            source_step=0,
            source_sweep=0,
            source_damage_digest=engineering._damage_digest(raw_damage),
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="mechanics_before",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=None,
            after=None,
            psi_before=None,
            H_used=None,
            d_solved=None,
            psi_after=None,
            H_next=None,
            rho_used=None,
            rho_next=None,
            rho_used_volume=None,
            rho_next_volume=None,
        )

    try:
        amor_before = compute_material_source_amor(
            model.fixed_damage_view(raw_damage),
            before.displacement,
            committed.material_history,
        )
        psi_before = _f64(
            amor_before.particle_driver,
            "auxiliary psi_before",
            (model.n_material,),
        )
        H_used = _immutable(
            np.maximum(committed.material_history, psi_before).astype(
                np.float64, copy=False
            )
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="amor_before",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=None,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=None,
            H_next=None,
            rho_used=rho_used,
            rho_next=None,
            rho_used_volume=rho_used_volume,
            rho_next_volume=None,
        )

    try:
        damage_result = engineering.solve_fixed_history_damage(
            model.material_operator, model.Gc, model.l0, H_used
        )
        if not isinstance(damage_result, engineering.FixedHistoryDamageResult):
            raise TypeError("auxiliary fixed-history solve returned the wrong result type")
        d_solved = _f64(
            damage_result.damage,
            "auxiliary d_solved",
            (model.n_material,),
        )
        if not np.array_equal(damage_result.history, H_used):
            raise ValueError("auxiliary fixed-history result is paired with a different H_used")
        rho_used = _f64(
            damage_result.strong_residual,
            "auxiliary rho_used",
            (model.n_material,),
        )
        rho_used_volume = _f64(
            damage_result.euclidean_residual,
            "auxiliary rho_used_volume",
            (model.n_material,),
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="damage",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=None,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=None,
            H_next=None,
            rho_used=rho_used,
            rho_next=None,
            rho_used_volume=rho_used_volume,
            rho_next_volume=None,
        )

    try:
        engineering._compare_damage_residual_reconstruction(
            rho_used,
            rho_used_volume,
            d_solved,
            H_used,
            model.material_operator.to_scipy().tocsr(),
            model.material_operator.volumes,
            model.Gc,
            model.l0,
            "auxiliary fixed-history residual",
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="rho_used",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=None,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=None,
            H_next=None,
            rho_used=rho_used,
            rho_next=None,
            rho_used_volume=rho_used_volume,
            rho_next_volume=None,
        )

    try:
        after = engineering._solve_direct_mechanics(
            adapter,
            model.fixed_damage_view(d_solved),
            model.constraints,
            target,
            config.cg_rtol,
            config.cg_atol,
            config.cg_maxiter,
            initial_guess=np.array(before.displacement, dtype=np.float64, copy=True),
            seed_kind="mechanics_before_solution",
            source_state="monolithic_mechanics_before",
            source_step=0,
            source_sweep=0,
            source_damage_digest=engineering._damage_digest(raw_damage),
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="mechanics_after",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=None,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=None,
            H_next=None,
            rho_used=rho_used,
            rho_next=None,
            rho_used_volume=rho_used_volume,
            rho_next_volume=None,
        )

    try:
        amor_after = compute_material_source_amor(
            model.fixed_damage_view(d_solved),
            after.displacement,
            committed.material_history,
        )
        psi_after = _f64(
            amor_after.particle_driver,
            "auxiliary psi_after",
            (model.n_material,),
        )
        H_next = _immutable(
            np.maximum(committed.material_history, psi_after).astype(
                np.float64, copy=False
            )
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="amor_after",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=after,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=psi_after,
            H_next=H_next,
            rho_used=rho_used,
            rho_next=rho_next,
            rho_used_volume=rho_used_volume,
            rho_next_volume=rho_next_volume,
        )

    try:
        _, rho_next, _, rho_next_volume, residual_scale = engineering._damage_residuals(
            model, d_solved, H_used, H_next
        )
        engineering._compare_damage_residual_reconstruction(
            rho_next,
            rho_next_volume,
            d_solved,
            H_next,
            model.material_operator.to_scipy().tocsr(),
            model.material_operator.volumes,
            model.Gc,
            model.l0,
            "auxiliary next-history residual",
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="rho_next",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=after,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=psi_after,
            H_next=H_next,
            rho_used=rho_used,
            rho_next=rho_next,
            rho_used_volume=rho_used_volume,
            rho_next_volume=rho_next_volume,
        )

    try:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="gates",
            reason="",
            exception_type=None,
            complete=True,
            before=before,
            after=after,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=psi_after,
            H_next=H_next,
            rho_used=rho_used,
            rho_next=rho_next,
            rho_used_volume=rho_used_volume,
            rho_next_volume=rho_next_volume,
            residual_scale=residual_scale,
        )
    except (RuntimeError, FloatingPointError) as error:
        return _auxiliary_audit(
            model=model,
            config=config,
            committed=committed,
            target=target,
            raw_z=raw_z,
            raw_q=raw_q,
            raw_damage=raw_damage,
            raw_history=raw_history,
            stage="diagnostics",
            reason=str(error),
            exception_type=type(error).__name__,
            complete=False,
            before=before,
            after=after,
            psi_before=psi_before,
            H_used=H_used,
            d_solved=d_solved,
            psi_after=psi_after,
            H_next=H_next,
            rho_used=rho_used,
            rho_next=rho_next,
            rho_used_volume=rho_used_volume,
            rho_next_volume=rho_next_volume,
        )


def _failure_attempt(
    *,
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    target: float,
    adapter_kind: str,
    initial_z: np.ndarray,
    newton_config: MonolithicNewtonConfig,
    newton_gate_config: MonolithicNewtonGateConfig,
    newton_result: MonolithicNewtonResult | None,
    auxiliary: MonolithicAuxiliarySweep | None,
    current_z: np.ndarray,
    stage: str,
    reason: str,
    exception_type: str | None,
    newton_audit: MonolithicNewtonFailureAudit | None,
    canonical_z: np.ndarray | None,
    canonical_active_mask: np.ndarray | None,
    canonical_scales: MonolithicNewtonScales | None,
    canonical_scaled_residual: np.ndarray | None,
    canonical_gate_passes: tuple[tuple[str, bool], ...],
) -> MonolithicVerifiedAttempt:
    before = committed.digest
    failure = MonolithicAttemptFailureAudit(
        stage=stage,
        reason=reason,
        exception_type=exception_type,
        current_z=current_z,
        committed_before_digest=before,
        committed_after_digest=before,
        newton_audit=newton_audit,
        auxiliary=auxiliary,
        canonical_z=canonical_z,
        canonical_active_mask=canonical_active_mask,
        canonical_scales=canonical_scales,
        canonical_scaled_residual=canonical_scaled_residual,
        canonical_gate_passes=canonical_gate_passes,
    )
    return MonolithicVerifiedAttempt(
        model=model,
        config=config,
        committed_state=committed,
        newton_config=newton_config,
        newton_gate_config=newton_gate_config,
        target_u_bar=target,
        adapter_kind=adapter_kind,
        initial_z=initial_z,
        accepted=False,
        committed_before_digest=before,
        committed_after_digest=before,
        newton_result=newton_result,
        newton_audit=newton_audit,
        auxiliary=auxiliary,
        canonical_z=canonical_z,
        canonical_active_mask=canonical_active_mask,
        canonical_scales=canonical_scales,
        canonical_scaled_residual=canonical_scaled_residual,
        canonical_gate_passes=canonical_gate_passes,
        state=None,
        failure=failure,
    )


def run_verified_monolithic_attempt(
    model: EngineeringGripModel,
    config: EngineeringGripConfig,
    committed: EngineeringGripState,
    target: Any,
    adapter: Any,
    initial_z: Any,
    newton_config: MonolithicNewtonConfig,
) -> MonolithicVerifiedAttempt:
    """Run and strictly verify one monolithic engineering-grip target.

    The seven arguments are deliberately required.  ``adapter`` may be the
    validated JAX CPU-replay adapter or ``None`` for the explicit NumPy/SciPy
    reference path, but it is never inferred.  The function performs no load
    stepping, warm-start construction, cutback, warning acceptance, archive
    write, or physical workflow action.

    A Newton numerical failure or a strict auxiliary/canonical gate rejection
    returns an immutable rejected :class:`MonolithicVerifiedAttempt`.  Type,
    shape, identity, and other structural contract violations propagate as
    ordinary exceptions so they cannot be mistaken for a numerical rejection.
    """

    if not isinstance(model, EngineeringGripModel):
        raise TypeError("run_verified_monolithic_attempt requires EngineeringGripModel")
    if not isinstance(config, EngineeringGripConfig):
        raise TypeError("run_verified_monolithic_attempt requires EngineeringGripConfig")
    if not isinstance(committed, EngineeringGripState):
        raise TypeError("run_verified_monolithic_attempt requires EngineeringGripState")
    if not isinstance(newton_config, MonolithicNewtonConfig):
        raise TypeError(
            "run_verified_monolithic_attempt requires MonolithicNewtonConfig"
        )
    target_value = _scalar(target, "verified monolithic target")
    if target_value <= committed.u_bar:
        raise ValueError("verified monolithic target must exceed committed.u_bar")
    if target_value > config.load_final:
        raise ValueError("verified monolithic target exceeds EngineeringGripConfig.load_final")
    if committed.union_displacement.shape != (model.n_dofs,):
        raise ValueError("committed union displacement does not match the model")
    if committed.material_damage.shape != (model.n_material,):
        raise ValueError("committed damage does not match the model")
    if committed.material_history.shape != (model.n_material,):
        raise ValueError("committed history does not match the model")
    engineering._check_exact_constraints(
        committed.union_displacement,
        model.constraints,
        committed.u_bar,
        "committed engineering-grip state",
    )
    validated_adapter = engineering._validated_mechanics_adapter(model, adapter)
    adapter_kind = (
        "cpu_reference"
        if validated_adapter is None
        else type(validated_adapter).__name__
    )
    expected_packed = len(model.constraints.free_dofs) + 2 * model.n_material
    initial = _f64(initial_z, "verified monolithic initial_z", (expected_packed,))
    if np.any(initial[-model.n_material :] < 0.0):
        raise ValueError("verified monolithic initial history must satisfy H>=0")
    initial_copy = np.array(initial, dtype=np.float64, copy=True)
    monolithic = MonolithicGripResidual(
        model, target_value, committed.material_history
    )
    gate_config = MonolithicNewtonGateConfig.from_engineering_config(config)

    try:
        newton_result = solve_monolithic_newton(
            monolithic,
            initial_copy,
            newton_config,
            gate_config,
        )
    except MonolithicNewtonFailure as error:
        return _failure_attempt(
            model=model,
            config=config,
            committed=committed,
            target=target_value,
            adapter_kind=adapter_kind,
            initial_z=initial_copy,
            newton_config=newton_config,
            newton_gate_config=gate_config,
            newton_result=None,
            auxiliary=None,
            current_z=error.audit.current_z,
            stage="newton",
            reason=str(error),
            exception_type=type(error).__name__,
            newton_audit=error.audit,
            canonical_z=None,
            canonical_active_mask=None,
            canonical_scales=None,
            canonical_scaled_residual=None,
            canonical_gate_passes=tuple((name, False) for name in _MONOLITHIC_CANONICAL_GATE_NAMES),
        )

    raw_z = np.array(newton_result.z, dtype=np.float64, copy=True)
    auxiliary = _run_monolithic_auxiliary_sweep(
        monolithic,
        model,
        config,
        committed,
        target_value,
        validated_adapter,
        raw_z,
    )
    if not auxiliary.passed:
        return _failure_attempt(
            model=model,
            config=config,
            committed=committed,
            target=target_value,
            adapter_kind=adapter_kind,
            initial_z=initial_copy,
            newton_config=newton_config,
            newton_gate_config=gate_config,
            newton_result=newton_result,
            auxiliary=auxiliary,
            current_z=raw_z,
            stage="auxiliary",
            reason=auxiliary.failure_reason,
            exception_type=auxiliary.exception_type,
            newton_audit=None,
            canonical_z=None,
            canonical_active_mask=None,
            canonical_scales=None,
            canonical_scaled_residual=None,
            canonical_gate_passes=tuple((name, False) for name in _MONOLITHIC_CANONICAL_GATE_NAMES),
        )

    assert auxiliary.q_after is not None
    assert auxiliary.d_solved is not None
    assert auxiliary.H_next is not None
    canonical_q = np.asarray(auxiliary.q_after[model.constraints.free_dofs], dtype=np.float64)
    canonical_z = monolithic.pack(canonical_q, auxiliary.d_solved, auxiliary.H_next)
    canonical_active = np.asarray(monolithic.active_mask(canonical_z), dtype=bool)
    canonical_scales = monolithic.block_scales(canonical_z, gate_config)
    canonical_scaled = monolithic.scaled_residual(canonical_z, canonical_scales)
    canonical_norms = _scaled_block_norms(monolithic, canonical_scaled)
    canonical_displacement = monolithic.reconstruct_displacement(canonical_z)
    expected_active = np.asarray(auxiliary.psi_after > committed.material_history, dtype=bool)
    active_consistency = bool(
        np.array_equal(canonical_active, expected_active)
        and np.array_equal(canonical_active, newton_result.final_active_mask)
        and np.array_equal(auxiliary.H_next, np.maximum(committed.material_history, auxiliary.psi_after))
    )
    history_nonnegative = bool(
        np.all(canonical_z[-model.n_material :] >= 0.0)
        and np.all(auxiliary.H_next >= 0.0)
    )
    finite = bool(
        np.all(np.isfinite(canonical_z))
        and np.all(np.isfinite(canonical_scaled))
        and all(np.isfinite(value) for value in canonical_norms)
    )
    monolithic_residual = bool(finite and all(value <= 1.0 for value in canonical_norms))
    prescribed = bool(
        np.array_equal(
            canonical_displacement[model.constraints.constrained_dofs],
            model.constraints.prescribed_vector(target_value)[model.constraints.constrained_dofs],
        )
        and np.array_equal(canonical_displacement, auxiliary.q_after)
    )
    expected_canonical = np.concatenate(
        (canonical_q, auxiliary.d_solved, auxiliary.H_next)
    )
    state_pairing = bool(np.array_equal(canonical_z, expected_canonical))
    canonical_gate_values = {
        "finite": finite,
        "monolithic_residual": monolithic_residual,
        "active_consistency": active_consistency,
        "history_nonnegative": history_nonnegative,
        "prescribed_constraints": prescribed,
        "state_pairing": state_pairing,
        "auxiliary": bool(auxiliary.passed),
    }
    canonical_gate_values["coupled"] = bool(all(canonical_gate_values.values()))
    canonical_gate_passes = tuple(
        (name, bool(canonical_gate_values[name]))
        for name in _MONOLITHIC_CANONICAL_GATE_NAMES
    )
    if not canonical_gate_values["coupled"]:
        reason = ", ".join(
            name for name, passed in canonical_gate_passes if not passed
        )
        return _failure_attempt(
            model=model,
            config=config,
            committed=committed,
            target=target_value,
            adapter_kind=adapter_kind,
            initial_z=initial_copy,
            newton_config=newton_config,
            newton_gate_config=gate_config,
            newton_result=newton_result,
            auxiliary=auxiliary,
            current_z=canonical_z,
            stage="canonical",
            reason=reason,
            exception_type=None,
            newton_audit=None,
            canonical_z=canonical_z,
            canonical_active_mask=canonical_active,
            canonical_scales=canonical_scales,
            canonical_scaled_residual=canonical_scaled,
            canonical_gate_passes=canonical_gate_passes,
        )

    state = EngineeringGripState(
        u_bar=target_value,
        union_displacement=auxiliary.q_after,
        material_damage=auxiliary.d_solved,
        material_history=auxiliary.H_next,
    )
    return MonolithicVerifiedAttempt(
        model=model,
        config=config,
        committed_state=committed,
        newton_config=newton_config,
        newton_gate_config=gate_config,
        target_u_bar=target_value,
        adapter_kind=adapter_kind,
        initial_z=initial_copy,
        accepted=True,
        committed_before_digest=committed.digest,
        committed_after_digest=state.digest,
        newton_result=newton_result,
        newton_audit=None,
        auxiliary=auxiliary,
        canonical_z=canonical_z,
        canonical_active_mask=canonical_active,
        canonical_scales=canonical_scales,
        canonical_scaled_residual=canonical_scaled,
        canonical_gate_passes=canonical_gate_passes,
        state=state,
        failure=None,
    )


# Owner-backed validation intentionally calls this immutable production entry
# point rather than a caller-replaceable module attribute or adapter hook.
_AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT = run_verified_monolithic_attempt


_MONOLITHIC_CONTINUATION_ACCEPTANCE_MODE = "strict"
_MONOLITHIC_CONTINUATION_COMPLETED = "completed"
_MONOLITHIC_CONTINUATION_FAILED = "failed"
_MONOLITHIC_NO_PROGRESS_REASON = "continuation made no floating-point load progress"


@dataclass(frozen=True)
class MonolithicContinuationConfig:
    """Explicit manufactured-only Phase-2C load schedule and solver binding.

    The existing :class:`EngineeringGripConfig` owns all direct-engineering
    gate and work-check values.  The six schedule fields are repeated here on
    purpose so a Phase-2C result records the schedule it actually used; they
    must equal the corresponding fields of ``engineering_config`` exactly.
    """

    load_start: float
    load_final: float
    du_initial: float
    du_min: float
    cutback_factor: float
    max_cutbacks: int
    engineering_config: EngineeringGripConfig
    newton_config: MonolithicNewtonConfig

    def __post_init__(self) -> None:
        if not isinstance(self.engineering_config, EngineeringGripConfig):
            raise TypeError("continuation engineering_config has the wrong type")
        if not isinstance(self.newton_config, MonolithicNewtonConfig):
            raise TypeError("continuation newton_config has the wrong type")
        start = _scalar(self.load_start, "continuation.load_start")
        final = _scalar(self.load_final, "continuation.load_final")
        initial = _scalar(self.du_initial, "continuation.du_initial")
        minimum = _scalar(self.du_min, "continuation.du_min")
        factor = _scalar(self.cutback_factor, "continuation.cutback_factor")
        max_cutbacks = _strict_integer(
            self.max_cutbacks,
            "continuation.max_cutbacks",
            positive=True,
            zero_allowed=False,
        )
        if start != 0.0 or final <= start:
            raise ValueError("continuation load interval must start at exact zero and increase")
        if initial <= 0.0 or minimum <= 0.0 or minimum > initial or initial > final - start:
            raise ValueError("continuation load increments are invalid")
        if not 0.0 < factor < 1.0:
            raise ValueError("continuation cutback_factor must lie strictly between zero and one")
        base = self.engineering_config
        for name, value in (
            ("load_start", start),
            ("load_final", final),
            ("du_initial", initial),
            ("du_min", minimum),
            ("cutback_factor", factor),
            ("max_cutbacks", max_cutbacks),
        ):
            if value != getattr(base, name):
                raise ValueError(
                    f"continuation {name} must equal engineering_config.{name} exactly"
                )
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_final", final)
        object.__setattr__(self, "du_initial", initial)
        object.__setattr__(self, "du_min", minimum)
        object.__setattr__(self, "cutback_factor", factor)
        object.__setattr__(self, "max_cutbacks", max_cutbacks)

    @property
    def acceptance_mode(self) -> str:
        return _MONOLITHIC_CONTINUATION_ACCEPTANCE_MODE


def _continuation_expected_initial_z(
    model: EngineeringGripModel,
    committed: EngineeringGripState,
    target: float,
) -> np.ndarray:
    """Return the exact packed form of the direct predictor contract.

    The controller itself obtains the full displacement through the
    authoritative ``engineering._direct_seed`` helper.  This side-effect-free
    reconstruction is used only by immutable-audit validation: that helper
    copies free DOFs and replaces only the affine constrained DOFs at the new
    target, while damage/history remain the committed fields.
    """

    if committed.union_displacement.shape != (model.n_dofs,):
        raise ValueError("continuation committed displacement has the wrong model shape")
    if committed.material_damage.shape != (model.n_material,):
        raise ValueError("continuation committed damage has the wrong model shape")
    if committed.material_history.shape != (model.n_material,):
        raise ValueError("continuation committed history has the wrong model shape")
    engineering._check_exact_constraints(
        committed.union_displacement,
        model.constraints,
        committed.u_bar,
        "continuation committed state",
    )
    full = np.array(committed.union_displacement, dtype=np.float64, copy=True)
    prescribed = model.constraints.prescribed_vector(target)
    full[model.constraints.constrained_dofs] = prescribed[
        model.constraints.constrained_dofs
    ]
    return _immutable(
        np.concatenate(
            (
                full[model.constraints.free_dofs],
                committed.material_damage,
                committed.material_history,
            )
        )
    )


@dataclass(frozen=True)
class MonolithicContinuationAttempt:
    """Immutable audit wrapper for exactly one Phase-2B target attempt."""

    model: EngineeringGripModel
    static_identity: object
    config: MonolithicContinuationConfig
    committed_state: EngineeringGripState
    attempt_index: int
    step_index: int
    cutback_count: int
    load_start: float
    load_target: float
    increment: float
    initial_z: np.ndarray
    phase2b: MonolithicVerifiedAttempt

    def __post_init__(self) -> None:
        if not isinstance(self.model, EngineeringGripModel):
            raise TypeError("continuation attempt model has the wrong type")
        if self.static_identity is not self.model.static:
            raise ValueError("continuation attempt static identity is not model-paired")
        if not isinstance(self.config, MonolithicContinuationConfig):
            raise TypeError("continuation attempt config has the wrong type")
        if not isinstance(self.committed_state, EngineeringGripState):
            raise TypeError("continuation attempt committed_state has the wrong type")
        if not isinstance(self.phase2b, MonolithicVerifiedAttempt):
            raise TypeError("continuation attempt Phase-2B result has the wrong type")
        attempt_index = _strict_integer(
            self.attempt_index,
            "continuation attempt_index",
            positive=False,
            zero_allowed=True,
        )
        step_index = _strict_integer(
            self.step_index,
            "continuation step_index",
            positive=False,
            zero_allowed=False,
        )
        cutbacks = _strict_integer(
            self.cutback_count,
            "continuation cutback_count",
            positive=False,
            zero_allowed=True,
        )
        start = _scalar(self.load_start, "continuation attempt load_start")
        target = _scalar(self.load_target, "continuation attempt load_target")
        increment = _scalar(self.increment, "continuation attempt increment")
        if start != self.committed_state.u_bar:
            raise ValueError("continuation attempt load_start is not committed-state paired")
        controller_no_progress = bool(
            not self.phase2b.accepted
            and self.phase2b.failure_stage == "controller"
            and target == start
        )
        if (target <= start and not controller_no_progress) or target > self.config.load_final:
            raise ValueError("continuation attempt load_target is outside the forward interval")
        if (increment <= 0.0 and not controller_no_progress) or target - start != increment:
            raise ValueError("continuation attempt increment is not target-start")
        initial = _f64(
            self.initial_z,
            "continuation attempt initial_z",
            (len(self.model.constraints.free_dofs) + 2 * self.model.n_material,),
        )
        expected_initial = _continuation_expected_initial_z(
            self.model, self.committed_state, target
        )
        if not np.array_equal(initial, expected_initial):
            raise ValueError("continuation attempt predictor is not the direct committed-state seed")
        object.__setattr__(self, "initial_z", _immutable(initial))

        phase2b = self.phase2b
        if phase2b.model is not self.model:
            raise ValueError("continuation attempt Phase-2B model is foreign")
        if phase2b.config is not self.config.engineering_config:
            raise ValueError("continuation attempt Phase-2B engineering config is foreign")
        if phase2b.committed_state is not self.committed_state:
            raise ValueError("continuation attempt Phase-2B committed state is foreign")
        if phase2b.newton_config is not self.config.newton_config:
            raise ValueError("continuation attempt Phase-2B Newton config is foreign")
        if phase2b.target_u_bar != target:
            raise ValueError("continuation attempt Phase-2B target is not paired")
        if not np.array_equal(phase2b.initial_z, initial):
            raise ValueError("continuation attempt predictor is not Phase-2B paired")
        if phase2b.committed_before_digest != self.committed_state.digest:
            raise ValueError("continuation attempt committed-before digest is not paired")
        expected_after = (
            phase2b.state.digest if phase2b.accepted and phase2b.state is not None else self.committed_state.digest
        )
        if phase2b.committed_after_digest != expected_after:
            raise ValueError("continuation attempt committed-after digest is not paired")
        if phase2b.accepted and phase2b.state is None:
            raise ValueError("accepted continuation attempt is missing its state")
        if not phase2b.accepted and phase2b.state is not None:
            raise ValueError("rejected continuation attempt exposes a committed state")
        object.__setattr__(self, "attempt_index", attempt_index)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "cutback_count", cutbacks)
        object.__setattr__(self, "load_start", start)
        object.__setattr__(self, "load_target", target)
        object.__setattr__(self, "increment", increment)

    def validate(self) -> None:
        """Re-run this wrapper and its nested Phase-2B audit."""

        self.phase2b.validate()
        type(self)(
            model=self.model,
            static_identity=self.static_identity,
            config=self.config,
            committed_state=self.committed_state,
            attempt_index=self.attempt_index,
            step_index=self.step_index,
            cutback_count=self.cutback_count,
            load_start=self.load_start,
            load_target=self.load_target,
            increment=self.increment,
            initial_z=self.initial_z,
            phase2b=self.phase2b,
        )

    def validate_with_owner(self, model: EngineeringGripModel) -> None:
        return self._validate_with_owner_runner(
            model,
            _AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT,
        )

    def _validate_with_owner_runner(
        self,
        model: EngineeringGripModel,
        runner: Any,
    ) -> None:
        """Authenticate this schedule record and its nested Phase-2B attempt."""

        self.validate()
        if not isinstance(model, EngineeringGripModel):
            raise TypeError("continuation attempt owner model has the wrong type")
        if model.model_digest != self.model.model_digest:
            raise ValueError("continuation attempt owner model digest differs")
        if model.static_digest != self.model.static_digest:
            raise ValueError("continuation attempt owner static digest differs")
        self.phase2b._validate_with_owner_runner(
            model,
            self.config.engineering_config,
            runner,
        )

    @property
    def accepted(self) -> bool:
        return self.phase2b.accepted

    @property
    def state(self) -> EngineeringGripState | None:
        return self.phase2b.state

    @property
    def failure(self) -> MonolithicAttemptFailureAudit | None:
        return self.phase2b.failure

    @property
    def failure_stage(self) -> str | None:
        return self.phase2b.failure_stage

    @property
    def failure_reason(self) -> str | None:
        return self.phase2b.failure_reason

    @property
    def exception_type(self) -> str | None:
        return self.phase2b.exception_type

    @property
    def committed_before_digest(self) -> str:
        return self.phase2b.committed_before_digest

    @property
    def committed_after_digest(self) -> str:
        return self.phase2b.committed_after_digest

    @property
    def target_u_bar(self) -> float:
        return self.load_target

    @property
    def engineering_config(self) -> EngineeringGripConfig:
        return self.config.engineering_config

    @property
    def newton_config(self) -> MonolithicNewtonConfig:
        return self.config.newton_config


@dataclass(frozen=True)
class MonolithicContinuationResult:
    """Validated immutable success or partial-failure continuation audit."""

    model: EngineeringGripModel
    static_identity: object
    config: MonolithicContinuationConfig
    work_check: Any
    initial_state: EngineeringGripState
    states: tuple[EngineeringGripState, ...]
    accepted_steps: tuple[MonolithicContinuationAttempt, ...]
    attempts: tuple[MonolithicContinuationAttempt, ...]
    completion_status: str
    failed_target: float | None
    failure_reason: str | None
    failed_attempt: MonolithicContinuationAttempt | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "accepted_steps", tuple(self.accepted_steps))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.model, EngineeringGripModel):
            raise TypeError("continuation result model has the wrong type")
        if self.static_identity is not self.model.static:
            raise ValueError("continuation result static identity is not model-paired")
        if not isinstance(self.config, MonolithicContinuationConfig):
            raise TypeError("continuation result config has the wrong type")
        if not isinstance(self.work_check, engineering.EngineeringGripWorkCheck):
            raise TypeError("continuation result work_check has the wrong type")
        base = self.config.engineering_config
        engineering._validate_work_check_against_model_config(
            self.work_check, self.model, base
        )
        for work_field, config_field in (
            ("u_bar", "work_check_load"),
            ("epsilon", "work_epsilon"),
            ("relative_limit", "work_relative_limit"),
        ):
            if getattr(self.work_check, work_field) != getattr(base, config_field):
                raise ValueError(
                    f"continuation work check {work_field} is not paired with "
                    f"engineering_config.{config_field}"
                )
        if self.work_check.passed is not True or (
            self.work_check.relative_error > base.work_relative_limit
        ):
            raise ValueError("continuation result work check did not pass")
        if self.completion_status not in (
            _MONOLITHIC_CONTINUATION_COMPLETED,
            _MONOLITHIC_CONTINUATION_FAILED,
        ):
            raise ValueError("continuation result completion_status is invalid")
        if not isinstance(self.initial_state, EngineeringGripState):
            raise TypeError("continuation result initial_state has the wrong type")
        if not self.states or self.states[0] is not self.initial_state:
            raise ValueError("continuation states must begin with the exact initial_state object")
        if len(self.states) != len(self.accepted_steps) + 1:
            raise ValueError("continuation states must contain initial plus accepted states")
        initial = self.initial_state
        if (
            initial.u_bar != self.config.load_start
            or not np.array_equal(initial.q, np.zeros(self.model.n_dofs, dtype=np.float64))
            or not np.array_equal(initial.d, np.zeros(self.model.n_material, dtype=np.float64))
            or not np.array_equal(initial.H, np.zeros(self.model.n_material, dtype=np.float64))
        ):
            raise ValueError("continuation initial state must be exact zero at load_start")
        for state in self.states:
            if not isinstance(state, EngineeringGripState):
                raise TypeError("continuation states contain a wrong type")
            if state.q.shape != (self.model.n_dofs,) or state.d.shape != (self.model.n_material,):
                raise ValueError("continuation state shape does not match the model")
            engineering._check_exact_constraints(
                state.q, self.model.constraints, state.u_bar, "continuation state"
            )
            if np.any(state.d < -self.config.engineering_config.damage_range_atol) or np.any(
                state.d > 1.0 + self.config.engineering_config.damage_range_atol
            ):
                raise ValueError("continuation committed damage is outside its strict range")
            if np.any(state.H < 0.0):
                raise ValueError("continuation committed history is negative")
        for previous, state in zip(self.states[:-1], self.states[1:], strict=True):
            if state.u_bar <= previous.u_bar:
                raise ValueError("continuation committed loads are not strictly increasing")
            if np.min(state.d - previous.d) < -self.config.engineering_config.healing_atol:
                raise ValueError("continuation committed damage is healing")
            if np.min(state.H - previous.H) < 0.0:
                raise ValueError("continuation committed history is not monotone")
        for index, step in enumerate(self.accepted_steps, start=1):
            if not isinstance(step, MonolithicContinuationAttempt):
                raise TypeError("continuation accepted_steps contain a wrong type")
            if not step.accepted or sum(candidate is step for candidate in self.attempts) != 1:
                raise ValueError("continuation accepted step is not uniquely attempt-paired")
            if self.states[index] is not step.state:
                raise ValueError("continuation accepted step is not state-paired")
            if step.load_start != self.states[index - 1].u_bar or step.load_target != self.states[index].u_bar:
                raise ValueError("continuation accepted load/state pairing is invalid")

        current = 0
        accepted_index = 0
        expected_increment = self.config.du_initial
        expected_cutbacks = 0
        expected_step = 1
        terminal_rejection = False
        for index, attempt in enumerate(self.attempts):
            if not isinstance(attempt, MonolithicContinuationAttempt):
                raise TypeError("continuation attempts contain a wrong type")
            attempt.validate()
            if attempt.attempt_index != index:
                raise ValueError("continuation attempt indices are not contiguous")
            if attempt.step_index != expected_step:
                raise ValueError("continuation attempt step index is not ordered")
            if attempt.cutback_count != expected_cutbacks:
                raise ValueError("continuation attempt cutback count is not ordered")
            if attempt.cutback_count > self.config.max_cutbacks:
                raise ValueError("continuation attempt exceeds config.max_cutbacks")
            if attempt.model is not self.model or attempt.config is not self.config:
                raise ValueError("continuation attempt model/config identity is foreign")
            if attempt.committed_state is not self.states[current]:
                raise ValueError("continuation attempt committed state is not object-paired")
            if attempt.load_start != self.states[current].u_bar:
                raise ValueError("continuation attempt load start is not current-state paired")
            if attempt.committed_before_digest != self.states[current].digest:
                raise ValueError("continuation attempt committed-before digest is not paired")
            expected_target = engineering._next_target(
                self.states[current].u_bar,
                expected_increment,
                self.config.load_final,
            )
            if expected_target <= self.states[current].u_bar:
                if (
                    attempt.accepted
                    or attempt.load_target != expected_target
                    or attempt.increment != 0.0
                    or attempt.failure_stage != "controller"
                    or index != len(self.attempts) - 1
                ):
                    raise ValueError(
                        "continuation non-progress failure is not the exact terminal controller attempt"
                    )
                terminal_rejection = True
                continue
            if attempt.load_target != expected_target:
                raise ValueError("continuation attempt target does not follow the load schedule")
            if attempt.increment != expected_target - self.states[current].u_bar:
                raise ValueError("continuation attempt increment does not follow the load schedule")
            if attempt.accepted:
                if accepted_index >= len(self.accepted_steps):
                    raise ValueError("continuation accepted attempt count exceeds accepted steps")
                if attempt is not self.accepted_steps[accepted_index]:
                    raise ValueError("continuation accepted attempt is not paired with its step")
                state = attempt.state
                if state is None or self.states[current + 1] is not state:
                    raise ValueError("continuation accepted attempt state is not paired")
                if attempt.committed_after_digest != state.digest:
                    raise ValueError("continuation accepted-after digest is not paired")
                current += 1
                accepted_index += 1
                expected_step += 1
                expected_increment = attempt.increment
                expected_cutbacks = 0
            else:
                if attempt.committed_after_digest != self.states[current].digest:
                    raise ValueError("continuation rejected attempt changed committed state")
                at_minimum = engineering._is_minimum_increment(
                    attempt.increment,
                    self.config.du_min,
                    self.states[current].u_bar,
                    attempt.load_target,
                )
                at_cutback_limit = expected_cutbacks >= self.config.max_cutbacks
                if at_minimum or at_cutback_limit:
                    terminal_rejection = True
                    if index != len(self.attempts) - 1:
                        raise ValueError("continuation has attempts after a terminal rejection")
                else:
                    if index == len(self.attempts) - 1:
                        raise ValueError(
                            "continuation ended after a rejected attempt before cutback termination"
                        )
                    expected_cutbacks += 1
                    expected_increment = max(
                        self.config.du_min,
                        attempt.increment * self.config.cutback_factor,
                    )
        if current != len(self.states) - 1 or accepted_index != len(self.accepted_steps):
            raise ValueError("continuation attempt audit does not cover accepted states")

        failed_target = None if self.failed_target is None else _scalar(
            self.failed_target, "continuation failed_target"
        )
        if self.completion_status == _MONOLITHIC_CONTINUATION_COMPLETED:
            if self.states[-1].u_bar != self.config.load_final:
                raise ValueError("completed continuation did not reach exact load_final")
            if failed_target is not None or self.failure_reason is not None or self.failed_attempt is not None:
                raise ValueError("completed continuation retains failure provenance")
            if not self.attempts and self.config.load_start != self.config.load_final:
                raise ValueError("completed continuation must retain attempts")
        else:
            if self.failure_reason is None or not isinstance(self.failure_reason, str) or not self.failure_reason:
                raise ValueError("failed continuation must retain a failure reason")
            if not self.attempts or self.failed_attempt is None:
                raise ValueError(
                    "non-complete failed continuation must retain a failed attempt"
                )
            if self.failed_attempt is not self.attempts[-1]:
                raise ValueError("failed continuation attempt is not the final audit record")
            if self.failed_attempt.accepted or not terminal_rejection:
                raise ValueError("failed continuation does not retain a terminal rejection")
            if failed_target is None or self.failed_attempt.load_target != failed_target:
                raise ValueError("failed continuation target is not paired with its attempt")
            if self.failed_attempt.failure_stage == "controller":
                if failed_target != self.states[-1].u_bar:
                    raise ValueError("controller no-progress target is not the current load")
                if self.failed_attempt.failure_reason != _MONOLITHIC_NO_PROGRESS_REASON:
                    raise ValueError("controller no-progress reason is not canonical")
                expected_reason = f"no_progress: {_MONOLITHIC_NO_PROGRESS_REASON}"
            else:
                if failed_target <= self.states[-1].u_bar or failed_target > self.config.load_final:
                    raise ValueError("failed continuation target is outside the remaining interval")
                at_minimum = engineering._is_minimum_increment(
                    self.failed_attempt.increment,
                    self.config.du_min,
                    self.states[-1].u_bar,
                    self.failed_attempt.load_target,
                )
                at_cutback_limit = self.failed_attempt.cutback_count >= self.config.max_cutbacks
                reasons = []
                if at_minimum:
                    reasons.append("du_min")
                if at_cutback_limit:
                    reasons.append("max_cutbacks")
                if not reasons:
                    raise ValueError("failed continuation attempt is not terminal by config")
                expected_reason = f"{', '.join(reasons)}: {self.failed_attempt.failure_reason}"
            if self.failure_reason != expected_reason:
                raise ValueError("failed continuation reason is not paired with its stage/attempt")
        object.__setattr__(self, "failed_target", failed_target)

    def validate_with_owner(
        self,
        model: EngineeringGripModel,
        *,
        _work_check_runner: Any | None = None,
        _attempt_runner: Any | None = None,
    ) -> None:
        if _work_check_runner is None:
            _work_check_runner = engineering._AUTHORITATIVE_RUN_ZERO_DAMAGE_WORK_CHECK
        if _attempt_runner is None:
            _attempt_runner = _AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT
        return self._validate_with_owner_runners(
            model,
            _work_check_runner,
            _attempt_runner,
        )

    def _validate_with_owner_runners(
        self,
        model: EngineeringGripModel,
        work_check_runner: Any,
        attempt_runner: Any,
    ) -> None:
        """Authenticate the complete immutable continuation by nested replay.

        Each positive-increment Phase-2B attempt is replayed from its retained
        direct predictor.  The one exception is the structurally validated
        terminal controller-only no-progress record: it represents a failed
        floating-point schedule dispatch before any PDE attempt and therefore
        receives metadata/predictor validation but no Phase-2B replay.
        """

        self.validate()
        if not isinstance(model, EngineeringGripModel):
            raise TypeError("continuation owner model has the wrong type")
        if model.model_digest != self.model.model_digest:
            raise ValueError("continuation owner model digest differs")
        if model.static_digest != self.model.static_digest:
            raise ValueError("continuation owner static digest differs")
        base = self.config.engineering_config
        self.work_check._validate_with_owner_runner(model, base, work_check_runner)

        current_state = self.initial_state
        accepted_index = 0
        expected_increment = self.config.du_initial
        expected_cutbacks = 0
        expected_step = 1
        for index, attempt in enumerate(self.attempts):
            if attempt.attempt_index != index:
                raise ValueError("continuation owner attempt index is not ordered")
            if attempt.step_index != expected_step:
                raise ValueError("continuation owner attempt step index is not ordered")
            if attempt.cutback_count != expected_cutbacks:
                raise ValueError("continuation owner attempt cutback count is not ordered")
            expected_target = engineering._next_target(
                current_state.u_bar,
                expected_increment,
                self.config.load_final,
            )
            if attempt.load_start != current_state.u_bar:
                raise ValueError("continuation owner attempt load start is not the current state")
            if attempt.load_target != expected_target:
                raise ValueError("continuation owner attempt target is not schedule-paired")
            if attempt.committed_before_digest != current_state.digest:
                raise ValueError(
                    "continuation owner attempt committed-before digest is not the prior canonical state"
                )
            if attempt.committed_state.digest != current_state.digest:
                raise ValueError(
                    "continuation owner attempt committed-before state differs from prior canonical state"
                )
            _compare_durable_values(
                attempt.committed_state,
                current_state,
                f"continuation.attempts[{index}].committed_state",
            )
            controller_stage = bool(
                isinstance(attempt, MonolithicContinuationAttempt)
                and attempt.failure_stage == "controller"
            )
            controller_only = bool(
                controller_stage
                and not attempt.accepted
                and attempt.load_target == attempt.load_start
                and attempt.increment == 0.0
            )
            if controller_stage and not controller_only:
                raise ValueError(
                    "continuation controller stage is not an exact zero-increment no-progress record"
                )
            if controller_only:
                phase2b = attempt.phase2b
                if (
                    self.completion_status != _MONOLITHIC_CONTINUATION_FAILED
                    or index != len(self.attempts) - 1
                    or self.failed_attempt is not attempt
                    or self.failed_target != attempt.load_target
                ):
                    raise ValueError(
                        "controller-only continuation record is not the terminal failed attempt"
                    )
                if attempt.load_target != current_state.u_bar or attempt.increment != 0.0:
                    raise ValueError("controller-only continuation record changed the load")
                if attempt.failure_reason != _MONOLITHIC_NO_PROGRESS_REASON:
                    raise ValueError("controller-only continuation reason is not canonical")
                if self.failure_reason != f"no_progress: {_MONOLITHIC_NO_PROGRESS_REASON}":
                    raise ValueError("controller-only continuation result reason is not canonical")
                if phase2b.adapter_kind != "cpu_reference":
                    raise ValueError("controller-only continuation record has a non-reference adapter")
                if phase2b.model.model_digest != model.model_digest:
                    raise ValueError("controller-only continuation model digest differs")
                if phase2b.config.config_digest != base.config_digest:
                    raise ValueError("controller-only continuation config digest differs")
                if phase2b.target_u_bar != attempt.load_target:
                    raise ValueError("controller-only Phase-2B target is not schedule-paired")
                if phase2b.committed_before_digest != current_state.digest:
                    raise ValueError("controller-only committed-before digest differs")
                if phase2b.committed_after_digest != current_state.digest:
                    raise ValueError("controller-only committed-after digest differs")
                if phase2b.failure is None:
                    raise ValueError("controller-only continuation record has no failure audit")
                if (
                    phase2b.failure.stage != "controller"
                    or phase2b.failure.reason != _MONOLITHIC_NO_PROGRESS_REASON
                    or phase2b.failure.exception_type is not None
                ):
                    raise ValueError("controller-only Phase-2B failure provenance is not canonical")
                if phase2b.canonical_gate_passes != tuple(
                    (name, False) for name in _MONOLITHIC_CANONICAL_GATE_NAMES
                ):
                    raise ValueError("controller-only record retains a canonical PDE gate")
                if phase2b.failure.committed_before_digest != current_state.digest:
                    raise ValueError("controller-only failure before digest differs")
                if phase2b.failure.committed_after_digest != current_state.digest:
                    raise ValueError("controller-only failure after digest differs")
                expected_initial = _continuation_expected_initial_z(
                    model, current_state, attempt.load_target
                )
                if not np.array_equal(attempt.initial_z, expected_initial):
                    raise ValueError("controller-only predictor is not owner-paired")
                if not np.array_equal(phase2b.initial_z, expected_initial):
                    raise ValueError("controller-only Phase-2B predictor is not owner-paired")
                if phase2b.failure.current_z.shape != expected_initial.shape or not np.array_equal(
                    phase2b.failure.current_z, expected_initial
                ):
                    raise ValueError("controller-only failure predictor is not owner-paired")
                if any(
                    value is not None
                    for value in (
                        phase2b.newton_result,
                        phase2b.newton_audit,
                        phase2b.auxiliary,
                        phase2b.canonical_z,
                        phase2b.canonical_active_mask,
                        phase2b.canonical_scales,
                        phase2b.canonical_scaled_residual,
                    )
                ):
                    raise ValueError("controller-only record retains a PDE attempt audit")
                if any(
                    value is not None
                    for value in (
                        phase2b.failure.newton_audit,
                        phase2b.failure.auxiliary,
                        phase2b.failure.canonical_z,
                        phase2b.failure.canonical_active_mask,
                        phase2b.failure.canonical_scales,
                        phase2b.failure.canonical_scaled_residual,
                    )
                ):
                    raise ValueError("controller-only failure retains a PDE audit")
                continue

            if attempt.increment <= 0.0:
                raise ValueError("continuation owner Phase-2B attempt has no positive increment")
            attempt._validate_with_owner_runner(model, attempt_runner)
            if attempt.accepted:
                if attempt.state is None:
                    raise ValueError("continuation owner accepted attempt has no canonical state")
                if accepted_index >= len(self.states) - 1:
                    raise ValueError("continuation owner accepted state index is out of range")
                expected_state = self.states[accepted_index + 1]
                if attempt.state.digest != expected_state.digest:
                    raise ValueError(
                        "continuation owner accepted state digest differs from canonical state"
                    )
                _compare_durable_values(
                    attempt.state,
                    expected_state,
                    f"continuation.states[{accepted_index + 1}]",
                )
                current_state = attempt.state
                accepted_index += 1
                expected_increment = attempt.increment
                expected_cutbacks = 0
                expected_step += 1
            else:
                at_minimum = engineering._is_minimum_increment(
                    attempt.increment,
                    self.config.du_min,
                    current_state.u_bar,
                    attempt.load_target,
                )
                at_cutback_limit = attempt.cutback_count >= self.config.max_cutbacks
                if not at_minimum and not at_cutback_limit:
                    expected_cutbacks += 1
                    expected_increment = max(
                        self.config.du_min,
                        attempt.increment * self.config.cutback_factor,
                    )
        if current_state.digest != self.final_state.digest:
            raise ValueError("continuation owner replay ended at a different final state")
        _compare_durable_values(current_state, self.final_state, "continuation.final_state")

    @property
    def final_state(self) -> EngineeringGripState:
        return self.states[-1]

    @property
    def acceptance_mode(self) -> str:
        return _MONOLITHIC_CONTINUATION_ACCEPTANCE_MODE

    @property
    def validated(self) -> bool:
        return self.completion_status == _MONOLITHIC_CONTINUATION_COMPLETED


class MonolithicContinuationFailure(RuntimeError):
    """Numerical minimum-increment/cutback failure with immutable partial audit."""

    def __init__(self, message: str, result: MonolithicContinuationResult):
        if not isinstance(result, MonolithicContinuationResult):
            raise TypeError("continuation failure result has the wrong type")
        if result.completion_status != _MONOLITHIC_CONTINUATION_FAILED:
            raise ValueError("continuation failure requires a failed result")
        super().__init__(message)
        self.result = result
        self.audit = result
        self.last_committed_state = result.final_state
        self.attempts = result.attempts
        self.work_check = result.work_check
        self.failed_target = result.failed_target
        self.failed_attempt = result.failed_attempt


def _controller_no_progress_attempt(
    *,
    model: EngineeringGripModel,
    config: MonolithicContinuationConfig,
    committed: EngineeringGripState,
    attempts: list[MonolithicContinuationAttempt],
    accepted_steps: list[MonolithicContinuationAttempt],
    cutbacks: int,
    target: float,
    reason: str,
) -> MonolithicContinuationAttempt:
    """Build the exact terminal audit when target rounding cannot progress."""

    initial_z = _continuation_expected_initial_z(model, committed, target)
    gate_config = MonolithicNewtonGateConfig.from_engineering_config(
        config.engineering_config
    )
    phase2b = _failure_attempt(
        model=model,
        config=config.engineering_config,
        committed=committed,
        target=target,
        adapter_kind="cpu_reference",
        initial_z=initial_z,
        newton_config=config.newton_config,
        newton_gate_config=gate_config,
        newton_result=None,
        auxiliary=None,
        current_z=initial_z,
        stage="controller",
        reason=reason,
        exception_type=None,
        newton_audit=None,
        canonical_z=None,
        canonical_active_mask=None,
        canonical_scales=None,
        canonical_scaled_residual=None,
        canonical_gate_passes=tuple(
            (name, False) for name in _MONOLITHIC_CANONICAL_GATE_NAMES
        ),
    )
    return MonolithicContinuationAttempt(
        model=model,
        static_identity=model.static,
        config=config,
        committed_state=committed,
        attempt_index=len(attempts),
        step_index=len(accepted_steps) + 1,
        cutback_count=cutbacks,
        load_start=committed.u_bar,
        load_target=target,
        increment=0.0,
        initial_z=initial_z,
        phase2b=phase2b,
    )


def _raise_monolithic_continuation_failure(
    *,
    model: EngineeringGripModel,
    config: MonolithicContinuationConfig,
    work_check: Any,
    initial: EngineeringGripState,
    states: list[EngineeringGripState],
    accepted_steps: list[MonolithicContinuationAttempt],
    attempts: list[MonolithicContinuationAttempt],
    target: float,
    reason: str,
    failed_attempt: MonolithicContinuationAttempt,
) -> NoReturn:
    if not isinstance(failed_attempt, MonolithicContinuationAttempt):
        raise TypeError("continuation failure requires its terminal failed attempt")
    result = MonolithicContinuationResult(
        model=model,
        static_identity=model.static,
        config=config,
        work_check=work_check,
        initial_state=initial,
        states=tuple(states),
        accepted_steps=tuple(accepted_steps),
        attempts=tuple(attempts),
        completion_status=_MONOLITHIC_CONTINUATION_FAILED,
        failed_target=target,
        failure_reason=reason,
        failed_attempt=failed_attempt,
    )
    raise MonolithicContinuationFailure(
        f"strict monolithic continuation failed at u_bar={target:.17g}: {reason}",
        result,
    )


def _run_monolithic_strict_continuation_impl(
    model: EngineeringGripModel,
    config: MonolithicContinuationConfig,
    adapter: Any,
    *,
    work_check_runner: Any,
    attempt_runner: Any,
) -> MonolithicContinuationResult:
    """Run the private manufactured-only adaptive Phase-2C continuation core.

    The explicit ``adapter`` is passed unchanged to the one-time existing
    zero-damage work check and to every Phase-2B attempt.  Only numerical
    Phase-2B rejection is eligible for cutback; structural exceptions remain
    ordinary exceptions and are never relabelled as load-step failure.
    """

    if not isinstance(model, EngineeringGripModel):
        raise TypeError("run_monolithic_strict_continuation requires EngineeringGripModel")
    if not isinstance(config, MonolithicContinuationConfig):
        raise TypeError("run_monolithic_strict_continuation requires MonolithicContinuationConfig")

    base = config.engineering_config
    work_check = work_check_runner(
        model,
        base.work_check_load,
        base.work_epsilon,
        base.work_relative_limit,
        base.cg_rtol,
        base.cg_atol,
        base.cg_maxiter,
        mechanics_adapter=adapter,
    )
    if not isinstance(work_check, engineering.EngineeringGripWorkCheck):
        raise TypeError("continuation work check returned the wrong type")
    if not work_check.passed:
        raise engineering.EngineeringGripWorkCheckFailure(work_check)

    initial = EngineeringGripState(
        u_bar=config.load_start,
        union_displacement=np.zeros(model.n_dofs, dtype=np.float64),
        material_damage=np.zeros(model.n_material, dtype=np.float64),
        material_history=np.zeros(model.n_material, dtype=np.float64),
    )
    states: list[EngineeringGripState] = [initial]
    accepted_steps: list[MonolithicContinuationAttempt] = []
    attempts: list[MonolithicContinuationAttempt] = []
    committed = initial
    increment = config.du_initial
    cutbacks = 0

    while committed.u_bar < config.load_final:
        target = engineering._next_target(committed.u_bar, increment, config.load_final)
        if target <= committed.u_bar:
            failed_attempt = _controller_no_progress_attempt(
                model=model,
                config=config,
                committed=committed,
                attempts=attempts,
                accepted_steps=accepted_steps,
                cutbacks=cutbacks,
                target=target,
                reason=_MONOLITHIC_NO_PROGRESS_REASON,
            )
            attempts.append(failed_attempt)
            _raise_monolithic_continuation_failure(
                model=model,
                config=config,
                work_check=work_check,
                initial=initial,
                states=states,
                accepted_steps=accepted_steps,
                attempts=attempts,
                target=target,
                reason=f"no_progress: {_MONOLITHIC_NO_PROGRESS_REASON}",
                failed_attempt=failed_attempt,
            )
        predicted_full = engineering._direct_seed(
            committed.union_displacement,
            committed.u_bar,
            target,
            model.constraints,
        )
        initial_z = _immutable(
            np.concatenate(
                (
                    predicted_full[model.constraints.free_dofs],
                    committed.material_damage,
                    committed.material_history,
                )
            )
        )
        phase2b = attempt_runner(
            model,
            base,
            committed,
            target,
            adapter,
            initial_z,
            config.newton_config,
        )
        if not isinstance(phase2b, MonolithicVerifiedAttempt):
            raise TypeError("Phase-2B continuation call returned the wrong type")
        attempt = MonolithicContinuationAttempt(
            model=model,
            static_identity=model.static,
            config=config,
            committed_state=committed,
            attempt_index=len(attempts),
            step_index=len(accepted_steps) + 1,
            cutback_count=cutbacks,
            load_start=committed.u_bar,
            load_target=target,
            increment=target - committed.u_bar,
            initial_z=initial_z,
            phase2b=phase2b,
        )
        attempts.append(attempt)
        if attempt.accepted:
            state = attempt.state
            if state is None:
                raise RuntimeError("accepted Phase-2B attempt lost its canonical state")
            accepted_steps.append(attempt)
            states.append(state)
            committed = state
            increment = attempt.increment
            cutbacks = 0
            continue

        if attempt.committed_before_digest != attempt.committed_after_digest:
            raise RuntimeError("rejected Phase-2B attempt changed the committed-state digest")
        at_minimum = engineering._is_minimum_increment(
            target - committed.u_bar,
            config.du_min,
            committed.u_bar,
            target,
        )
        at_cutback_limit = cutbacks >= config.max_cutbacks
        if at_minimum or at_cutback_limit:
            reasons: list[str] = []
            if at_minimum:
                reasons.append("du_min")
            if at_cutback_limit:
                reasons.append("max_cutbacks")
            _raise_monolithic_continuation_failure(
                model=model,
                config=config,
                work_check=work_check,
                initial=initial,
                states=states,
                accepted_steps=accepted_steps,
                attempts=attempts,
                target=target,
                reason=f"{', '.join(reasons)}: {attempt.failure_reason}",
                failed_attempt=attempt,
            )
        cutbacks += 1
        increment = max(config.du_min, (target - committed.u_bar) * config.cutback_factor)

    return MonolithicContinuationResult(
        model=model,
        static_identity=model.static,
        config=config,
        work_check=work_check,
        initial_state=initial,
        states=tuple(states),
        accepted_steps=tuple(accepted_steps),
        attempts=tuple(attempts),
        completion_status=_MONOLITHIC_CONTINUATION_COMPLETED,
        failed_target=None,
        failure_reason=None,
        failed_attempt=None,
    )


def run_monolithic_strict_continuation(
    model: EngineeringGripModel,
    config: MonolithicContinuationConfig,
    adapter: Any,
) -> MonolithicContinuationResult:
    """Run the public manufactured-only Phase-2C continuation entry point."""

    # Keep this entry point injectable for manufactured controller tests.  The
    # archive owner path below deliberately binds the captured authoritative
    # work-check and Phase-2B runners instead of consulting these replaceable
    # public module attributes.
    return _run_monolithic_strict_continuation_impl(
        model,
        config,
        adapter,
        work_check_runner=engineering.run_zero_damage_work_check,
        attempt_runner=run_verified_monolithic_attempt,
    )


def _run_monolithic_strict_continuation_authoritative(
    model: EngineeringGripModel,
    config: MonolithicContinuationConfig,
    adapter: Any,
) -> MonolithicContinuationResult:
    """Run continuation with captured owner-authenticated numerical runners."""

    return _run_monolithic_strict_continuation_impl(
        model,
        config,
        adapter,
        work_check_runner=engineering._AUTHORITATIVE_RUN_ZERO_DAMAGE_WORK_CHECK,
        attempt_runner=_AUTHORITATIVE_RUN_VERIFIED_MONOLITHIC_ATTEMPT,
    )


# Owner-backed archive replay captures this immutable controller entry point;
# it is not a caller-replaceable workflow callback.
_AUTHORITATIVE_RUN_MONOLITHIC_STRICT_CONTINUATION = (
    _run_monolithic_strict_continuation_authoritative
)


# Short spellings are aliases, not separate continuation implementations.
run_strict_monolithic_continuation = run_monolithic_strict_continuation
run_monolithic_continuation = run_monolithic_strict_continuation


# Short public spellings are aliases, not separate numerical implementations.
attempt_monolithic_target = run_verified_monolithic_attempt
solve_verified_monolithic_attempt = run_verified_monolithic_attempt
run_monolithic_attempt = run_verified_monolithic_attempt
MonolithicAuxiliaryAudit = MonolithicAuxiliarySweep
MonolithicAttempt = MonolithicVerifiedAttempt
MonolithicAttemptResult = MonolithicVerifiedAttempt


# Compatibility spellings remain aliases, not separate implementations.
ScaledSemismoothNewtonConfig = MonolithicNewtonConfig
ScaledSemismoothNewtonGateConfig = MonolithicNewtonGateConfig
ScaledSemismoothNewtonResult = MonolithicNewtonResult
