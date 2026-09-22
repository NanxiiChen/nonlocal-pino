"""Focused manufactured tests for the Phase-1 monolithic q-d-H residual.

The monolithic module is private/experimental: these tests verify the exact
code-compatible residual blocks and the analytic generalized JVP on the same
manufactured full-square engineering-grip cloud used by the engineering-grip
suite.  No Newton solver, GMRES, workflow, or physical run is covered here.
"""

from __future__ import annotations

import unittest

import jax
import numpy as np

from examples.fracture2d.nonlocal_a2_coupled import _amor_edge_energy
from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d.nonlocal_engineering_grip_monolithic import (
    MonolithicGripResidual,
    _amor_first_variation_factor,
)
from examples.fracture2d.nonlocal_grip_diagnostic import (
    build_union_geometry_from_arrays,
)


def _material_cloud():
    n_side = 5
    axis = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    coordinates = np.asarray(
        [(x, y) for x in axis for y in axis], dtype=np.float64
    )
    triangles: list[tuple[int, int, int]] = []
    for ix in range(n_side - 1):
        for iy in range(n_side - 1):
            ll = ix * n_side + iy
            lr = (ix + 1) * n_side + iy
            ul = ix * n_side + iy + 1
            ur = (ix + 1) * n_side + iy + 1
            triangles.extend(((ll, lr, ur), (ll, ur, ul)))
    triangles_array = np.asarray(triangles, dtype=np.int64)
    p0 = coordinates[triangles_array[:, 0]]
    p1 = coordinates[triangles_array[:, 1]]
    p2 = coordinates[triangles_array[:, 2]]
    areas = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    volumes = np.zeros(len(coordinates), dtype=np.float64)
    for column in range(3):
        np.add.at(volumes, triangles_array[:, column], areas / 3.0)
    bottom = coordinates[:, 1] == 0.0
    top = coordinates[:, 1] == 1.0
    delta = 0.48
    pairs: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
            if 0.0 < distance < delta:
                pairs.extend(((first, second), (second, first)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    return coordinates, triangles_array, volumes, bottom, top, edges, delta


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))), 1.0e-300)
    return float(np.max(np.abs(first - second)) / scale)


def _independent_cubic_edge_weights(
    coordinates: np.ndarray,
    volumes: np.ndarray,
    edges: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Independently rebuild the 2D cubic-kernel damage edge weights.

    ``J(r) = (120/pi)/delta^4 * (1 - r/delta)^2`` for ``r <= delta`` and zero
    otherwise; each directed edge ``(i, j)`` carries ``J(r_ij) * V_j``.  This
    is written from the declared kernel definition, not taken from the
    operator's stored values.
    """

    row = edges[:, 0]
    col = edges[:, 1]
    distance = np.linalg.norm(coordinates[col] - coordinates[row], axis=1)
    normalization = 120.0 / np.pi / delta**4
    kernel = normalization * np.where(
        distance <= delta, (1.0 - distance / delta) ** 2, 0.0
    )
    return kernel * volumes[col]


class _IndependentMonolithicOracle:
    """Test-only independent rebuild of the packed monolithic JVP.

    Every assembly is rebuilt from frozen static data (BA supports, shape
    inverses, weights, phase volumes, the affine constraint data, and the
    material operator's explicit coordinates/volumes/edges) with
    independently written algebra: explicit per-edge loops and matmuls
    instead of the production einsums/bincounts, and a from-definition
    cubic-kernel damage graph.  It never calls the production
    ``jvp``/``vjp``/``_dense_jacobian``, ``tangent_action``, the production
    endpoint assembly, or the production damage-residual helpers.
    """

    def __init__(self, model, u_bar, committed_history) -> None:
        static = model.static
        self.n_particles = int(static.n_particles)
        self.n_material = int(static.material_count)
        self.n_dofs = 2 * self.n_particles
        self.free = np.asarray(model.constraints.free_dofs, dtype=np.int64)
        self.n_free = len(self.free)
        self.n_packed = self.n_free + 2 * self.n_material
        self.u_bar = float(u_bar)
        self.committed = np.array(committed_history, dtype=np.float64)
        self.source = np.asarray(static.source, dtype=np.int64)
        self.eta = np.asarray(static.eta, dtype=np.float64)
        self.material_source = self.source < self.n_material
        self.kinv = np.asarray(static.shape_inverse, dtype=np.float64)
        self.supports = [
            np.asarray(static.support_indices[e], dtype=np.int64)[
                np.asarray(static.support_mask[e], dtype=bool)
            ]
            for e in range(static.n_edges)
        ]
        self.beta = [
            np.asarray(static.beta[e], dtype=np.float64)[
                np.asarray(static.support_mask[e], dtype=bool)
            ]
            for e in range(static.n_edges)
        ]
        self.xi = [
            np.asarray(static.xi[e], dtype=np.float64)[
                np.asarray(static.support_mask[e], dtype=bool)
            ]
            for e in range(static.n_edges)
        ]
        self.material_phase = np.asarray(
            static.union_material_phase_volumes, dtype=np.float64
        )
        self.grip_phase = np.asarray(static.grip_phase_volumes, dtype=np.float64)
        self.k_res = float(static.k_res)
        self.lam = float(static.lame_lambda)
        self.mu = float(static.shear_modulus)
        self.Gc = float(model.Gc)
        self.l0 = float(model.l0)
        self.load_direction = np.asarray(
            model.constraints.load_direction, dtype=np.float64
        )
        operator = model.material_operator
        self.damage_volumes = np.asarray(operator.volumes, dtype=np.float64)
        self.damage_edges = np.asarray(operator.directed_edges, dtype=np.int64)
        self.damage_weights = _independent_cubic_edge_weights(
            np.asarray(operator.coordinates, dtype=np.float64),
            self.damage_volumes,
            self.damage_edges,
            float(operator.delta),
        )

    # -- independent kinematics ---------------------------------------

    def _gradient(self, differences: list[np.ndarray]) -> np.ndarray:
        """Edge deformation-gradient increment from support differences."""

        result = np.empty((len(self.source), 2, 2), dtype=np.float64)
        for edge, difference in enumerate(differences):
            moment = difference.T @ (self.beta[edge][:, None] * self.xi[edge])
            result[edge] = moment @ self.kinv[edge]
        return result

    def deformation_gradient(self, displacement: np.ndarray) -> np.ndarray:
        values = displacement.reshape(self.n_particles, 2)
        differences = []
        for edge, src in enumerate(self.source):
            differences.append(
                self.xi[edge] + values[self.supports[edge]] - values[src]
            )
        return self._gradient(differences)

    def delta_gradient(self, direction: np.ndarray) -> np.ndarray:
        values = direction.reshape(self.n_particles, 2)
        differences = []
        for edge, src in enumerate(self.source):
            differences.append(values[self.supports[edge]] - values[src])
        return self._gradient(differences)

    @staticmethod
    def _strain(gradient: np.ndarray) -> np.ndarray:
        increment = gradient - np.eye(2, dtype=np.float64)
        return 0.5 * (increment + np.swapaxes(increment, 1, 2))

    @staticmethod
    def _symmetric(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + np.swapaxes(matrix, 1, 2))

    def _stress(self, strain: np.ndarray) -> np.ndarray:
        trace = np.trace(strain, axis1=1, axis2=2)
        return (
            self.lam * trace[:, None, None] * np.eye(2, dtype=np.float64)
            + 2.0 * self.mu * strain
        )

    def _assemble(self, matrices: np.ndarray, factor: np.ndarray) -> np.ndarray:
        """Independent endpoint assembly for per-edge stress-like matrices.

        For each support ``s`` of edge ``e`` the force contribution is
        ``beta_s * factor_e * (xi_s . (sigma_e K_e^{-T}))`` added to the
        support particle and subtracted in total from the source particle.
        """

        result = np.zeros((self.n_particles, 2), dtype=np.float64)
        for edge, src in enumerate(self.source):
            endpoint = self.xi[edge] @ (matrices[edge] @ self.kinv[edge].T).T
            weighted = self.beta[edge][:, None] * factor[edge] * endpoint
            np.add.at(result, self.supports[edge], weighted)
            result[src] -= weighted.sum(axis=0)
        return result.reshape(-1)

    def _padded(self, values: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (values, np.zeros(self.n_particles - self.n_material, dtype=np.float64))
        )

    def _center_weights(self, d: np.ndarray) -> np.ndarray:
        degradation = self._padded((1.0 - d) ** 2 + self.k_res)
        return (
            self.material_phase * degradation
            + self.grip_phase * (1.0 + self.k_res)
        )

    def _amor(self, strain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the Amor driver and its first-variation factor."""

        embedded = np.zeros((len(strain), 3, 3), dtype=np.float64)
        embedded[:, :2, :2] = strain
        trace = np.trace(embedded, axis1=1, axis2=2)
        deviatoric = embedded - (trace / 3.0)[:, None, None] * np.eye(
            3, dtype=np.float64
        )
        bulk = self.lam + 2.0 * self.mu / 3.0
        energy = 0.5 * bulk * np.maximum(trace, 0.0) ** 2 + self.mu * np.sum(
            deviatoric * deviatoric, axis=(1, 2)
        )
        factor = (
            bulk * np.maximum(trace, 0.0)[:, None, None] * np.eye(2, dtype=np.float64)
            + 2.0 * self.mu * deviatoric[:, :2, :2]
        )
        driver = np.zeros(self.n_material, dtype=np.float64)
        np.add.at(
            driver,
            self.source[self.material_source],
            self.eta[self.material_source] * energy[self.material_source],
        )
        return driver, factor

    def _damage_graph(self, values: np.ndarray) -> np.ndarray:
        result = np.zeros(self.n_material, dtype=np.float64)
        row = self.damage_edges[:, 0]
        col = self.damage_edges[:, 1]
        np.add.at(
            result, row, self.damage_weights * (values[row] - values[col])
        )
        return result

    def jvp(self, z: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Independent packed generalized Jacobian action."""

        nm = self.n_material
        q = z[: self.n_free]
        d = z[self.n_free : self.n_free + nm]
        history = z[self.n_free + nm :]
        dq = direction[: self.n_free]
        dd = direction[self.n_free : self.n_free + nm]
        dh = direction[self.n_free + nm :]
        displacement = self.u_bar * self.load_direction
        displacement[self.free] += q
        strain = self._strain(self.deformation_gradient(displacement))
        stress = self._stress(strain)
        factor = self._center_weights(d)[self.source] * self.eta
        lifted = np.zeros(self.n_dofs, dtype=np.float64)
        lifted[self.free] += dq
        delta_strain = self._symmetric(self.delta_gradient(lifted))
        delta_center = (
            -2.0 * self.material_phase * (1.0 - self._padded(d))
        )[self.source] * self.eta * self._padded(dd)[self.source]
        residual_q = (
            self._assemble(self._stress(delta_strain), factor)[self.free]
            + self._assemble(stress, delta_center)[self.free]
        )
        residual_d = self.damage_volumes * (
            self.Gc / self.l0 * dd
            + self.Gc * self.l0 * self._damage_graph(dd)
            + 2.0 * history * dd
            + 2.0 * (d - 1.0) * dh
        )
        driver, amor_factor = self._amor(strain)
        edge = np.sum(amor_factor * delta_strain, axis=(1, 2))
        delta_driver = np.zeros(self.n_material, dtype=np.float64)
        np.add.at(
            delta_driver,
            self.source[self.material_source],
            self.eta[self.material_source] * edge[self.material_source],
        )
        residual_h = dh - np.where(driver > self.committed, delta_driver, 0.0)
        return np.concatenate((residual_q, residual_d, residual_h))


class MonolithicGripResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        coordinates, triangles, volumes, bottom, top, edges, delta = _material_cloud()
        cls.geometry = build_union_geometry_from_arrays(
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
            cls.geometry,
            2.0,
            0.3,
            1.0e-8,
            1.0e-12,
            1.7,
            0.65,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    # ------------------------------------------------------------------
    # Shared manufactured state
    # ------------------------------------------------------------------

    def _state(
        self,
        seed: int = 11,
        u_bar: float = 1.0e-3,
        committed: np.ndarray | None = None,
        damage_scale: float = 0.4,
    ) -> tuple[MonolithicGripResidual, np.ndarray]:
        rng = np.random.default_rng(seed)
        if committed is None:
            committed = np.zeros(self.model.n_material, dtype=np.float64)
        monolithic = MonolithicGripResidual(self.model, u_bar, committed)
        q = rng.standard_normal(monolithic.n_free) * 1.0e-2
        d = rng.random(self.model.n_material) * damage_scale
        history = rng.random(self.model.n_material) * 2.0
        return monolithic, monolithic.pack(q, d, history)

    def _centered_difference(
        self,
        monolithic: MonolithicGripResidual,
        z: np.ndarray,
        direction: np.ndarray,
        step: float,
    ) -> np.ndarray:
        return (
            monolithic.residual(z + step * direction)
            - monolithic.residual(z - step * direction)
        ) / (2.0 * step)

    def _edge_traces(self, monolithic: MonolithicGripResidual, z: np.ndarray) -> np.ndarray:
        _, d, _ = monolithic.unpack(z)
        view = self.model.fixed_damage_view(d)
        gradient = view._deformation_gradient(monolithic.reconstruct_displacement(z))
        strain, _ = view._strain_stress(gradient)
        return np.trace(strain, axis1=1, axis2=2)

    # ------------------------------------------------------------------
    # 1. Residual shape/finiteness and exact Rq
    # ------------------------------------------------------------------

    def test_residual_shape_and_exact_mechanics_row(self) -> None:
        monolithic, z = self._state()
        residual = monolithic.residual(z)
        self.assertEqual(residual.shape, (monolithic.n_packed,))
        self.assertEqual(residual.dtype, np.dtype(np.float64))
        self.assertTrue(np.all(np.isfinite(residual)))
        q, d, history = monolithic.unpack(z)
        self.assertEqual(monolithic.n_packed, monolithic.n_free + 2 * monolithic.n_material)
        # Reconstruction satisfies the affine hard constraints exactly.
        displacement = monolithic.reconstruct_displacement(z)
        prescribed = self.model.constraints.prescribed_vector(monolithic.u_bar)
        constraints = self.model.constraints.constrained_dofs
        self.assertTrue(
            np.array_equal(displacement[constraints], prescribed[constraints])
        )
        self.assertTrue(np.array_equal(displacement[self.model.constraints.free_dofs], q))
        # Rq is exactly the authoritative A2 fixed-damage free residual.
        view = self.model.fixed_damage_view(d)
        authoritative = view.residual(displacement)[self.model.constraints.free_dofs]
        self.assertTrue(np.array_equal(residual[: monolithic.n_free], authoritative))

    def test_jqq_is_existing_tangent_action(self) -> None:
        monolithic, z = self._state(seed=13)
        rng = np.random.default_rng(23)
        direction = rng.standard_normal(monolithic.n_free)
        analytic = monolithic.jqq_action(z, direction)
        _, d, _ = monolithic.unpack(z)
        view = self.model.fixed_damage_view(d)
        lifted = np.zeros(self.model.n_dofs, dtype=np.float64)
        lifted[self.model.constraints.free_dofs] = direction
        authoritative = view.tangent_action(lifted)[self.model.constraints.free_dofs]
        self.assertTrue(np.array_equal(analytic, authoritative))

    # ------------------------------------------------------------------
    # 2. Rd against an independent formula and at the fixed-history solve
    # ------------------------------------------------------------------

    def test_damage_row_matches_independent_dense_formula(self) -> None:
        monolithic, z = self._state(seed=17)
        residual = monolithic.residual(z)
        _, d, history = monolithic.unpack(z)
        dense = np.asarray(self.model.material_operator.to_scipy().todense(), dtype=np.float64)
        volumes = np.asarray(self.model.material_operator.volumes, dtype=np.float64)
        rho = (
            self.model.Gc / self.model.l0 * d
            + self.model.Gc * self.model.l0 * (dense @ d)
            + 2.0 * history * (d - 1.0)
        )
        independent = volumes * rho
        block = residual[monolithic.n_free : monolithic.n_free + monolithic.n_material]
        self.assertLess(_relative_defect(block, independent), 1.0e-12)
        # The existing authoritative engineering-grip damage residual is
        # reproduced with identical operation order.
        rho_used, _, volume_used, _, _ = engineering._damage_residuals(
            self.model, d, history, history
        )
        self.assertTrue(np.array_equal(block, volume_used))
        self.assertTrue(np.array_equal(block, volumes * rho_used))

    def test_damage_row_zero_at_fixed_history_solve(self) -> None:
        monolithic, z = self._state(seed=19)
        _, _, history = monolithic.unpack(z)
        solved = engineering.solve_fixed_history_damage(
            self.model.material_operator, self.model.Gc, self.model.l0, history
        )
        q, _, _ = monolithic.unpack(z)
        solved_z = monolithic.pack(q, solved.damage, history)
        block = monolithic.residual(solved_z)[
            monolithic.n_free : monolithic.n_free + monolithic.n_material
        ]
        scale = max(
            1.0,
            float(np.max(np.abs(monolithic.residual(z)))),
            float(np.max(np.abs(solved.euclidean_residual))),
        )
        self.assertLess(float(np.max(np.abs(block))) / scale, 1.0e-10)

    # ------------------------------------------------------------------
    # 3. RH branches
    # ------------------------------------------------------------------

    def test_history_row_below_branch(self) -> None:
        monolithic, z = self._state(seed=29)
        driver = monolithic.psi_plus(z)
        committed = driver + 1.0
        below = MonolithicGripResidual(self.model, monolithic.u_bar, committed)
        below_z = below.pack(*monolithic.unpack(z))
        self.assertFalse(np.any(below.active_mask(below_z)))
        _, _, history = monolithic.unpack(z)
        residual = below.residual(below_z)
        block = residual[below.n_free + below.n_material :]
        self.assertTrue(np.array_equal(block, history - committed))

    def test_history_row_above_branch(self) -> None:
        monolithic, z = self._state(seed=31)
        driver = monolithic.psi_plus(z)
        self.assertTrue(np.any(driver > 0.0))
        active = monolithic.active_mask(z)
        self.assertTrue(np.array_equal(active, driver > monolithic.committed_history))
        _, _, history = monolithic.unpack(z)
        block = monolithic.residual(z)[monolithic.n_free + monolithic.n_material :]
        self.assertTrue(np.array_equal(block[active], (history - driver)[active]))

    def test_history_row_exact_tie_is_committed(self) -> None:
        monolithic, z = self._state(seed=37)
        driver = monolithic.psi_plus(z)
        tied = MonolithicGripResidual(self.model, monolithic.u_bar, driver.copy())
        tied_z = tied.pack(*monolithic.unpack(z))
        # psi_plus is reproduced exactly, so the tie is exact bit-for-bit.
        self.assertTrue(np.array_equal(tied.psi_plus(tied_z), tied.committed_history))
        self.assertFalse(np.any(tied.active_mask(tied_z)))
        _, _, history = tied.unpack(tied_z)
        block = tied.residual(tied_z)[tied.n_free + tied.n_material :]
        self.assertTrue(np.array_equal(block, history - tied.committed_history))

    # ------------------------------------------------------------------
    # 4. Centered FD checks of Jqq and Jqd
    # ------------------------------------------------------------------

    def test_jqq_centered_finite_difference(self) -> None:
        monolithic, z = self._state(seed=41)
        rng = np.random.default_rng(43)
        direction = rng.standard_normal(monolithic.n_free) * 1.0e-2
        packed = np.zeros(monolithic.n_packed, dtype=np.float64)
        packed[: monolithic.n_free] = direction
        step = 1.0e-7
        finite_difference = self._centered_difference(monolithic, z, packed, step)
        analytic = monolithic.jqq_action(z, direction)
        self.assertLess(
            _relative_defect(finite_difference[: monolithic.n_free], analytic), 1.0e-6
        )

    def test_jqd_centered_finite_difference(self) -> None:
        monolithic, z = self._state(seed=47)
        rng = np.random.default_rng(53)
        direction = rng.standard_normal(monolithic.n_material)
        packed = np.zeros(monolithic.n_packed, dtype=np.float64)
        packed[monolithic.n_free : monolithic.n_free + monolithic.n_material] = direction
        step = 1.0e-7
        finite_difference = self._centered_difference(monolithic, z, packed, step)
        analytic = monolithic.jqd_action(z, direction)
        self.assertLess(
            _relative_defect(finite_difference[: monolithic.n_free], analytic), 1.0e-6
        )

    # ------------------------------------------------------------------
    # 5. Centered FD checks of Jdd and JdH
    # ------------------------------------------------------------------

    def test_jdd_and_jdh_centered_finite_differences(self) -> None:
        monolithic, z = self._state(seed=59)
        rng = np.random.default_rng(61)
        start = monolithic.n_free
        stop = start + monolithic.n_material
        damage_direction = rng.standard_normal(monolithic.n_material)
        history_direction = rng.random(monolithic.n_material) + 0.5
        step = 1.0e-7
        damage_packed = np.zeros(monolithic.n_packed, dtype=np.float64)
        damage_packed[start:stop] = damage_direction
        history_packed = np.zeros(monolithic.n_packed, dtype=np.float64)
        history_packed[stop:] = history_direction
        damage_fd = self._centered_difference(monolithic, z, damage_packed, step)
        history_fd = self._centered_difference(monolithic, z, history_packed, step)
        self.assertLess(
            _relative_defect(damage_fd[start:stop], monolithic.jdd_action(z, damage_direction)),
            1.0e-6,
        )
        self.assertLess(
            _relative_defect(history_fd[start:stop], monolithic.jdh_action(z, history_direction)),
            1.0e-6,
        )

    # ------------------------------------------------------------------
    # 6. Amor first variation and JHq
    # ------------------------------------------------------------------

    def test_amor_first_variation_trace_branches(self) -> None:
        lam = self.model.static.lame_lambda
        mu = self.model.static.shear_modulus
        rng = np.random.default_rng(67)
        positive = np.array([[[2.0e-3, 1.0e-4], [1.0e-4, 5.0e-4]]], dtype=np.float64)
        negative = -positive.copy()
        # Exact zero trace: the max(tr, 0) branch point of the embedded
        # volumetric term, where the exact first variation is the deviatoric
        # part alone.
        tie = np.array([[[0.0, 3.0e-3], [3.0e-3, 0.0]]], dtype=np.float64)
        perturbation = rng.standard_normal((1, 2, 2))
        perturbation = 0.5 * (perturbation + np.swapaxes(perturbation, 1, 2))
        for name, strain, tolerance in (
            ("positive", positive, 1.0e-7),
            ("negative", negative, 1.0e-7),
            ("tie", tie, 2.0e-3),
        ):
            factor = _amor_first_variation_factor(strain, lam, mu)
            self.assertEqual(strain.shape[0], factor.shape[0])
            errors = []
            for step in (1.0e-4, 1.0e-5, 1.0e-6):
                forward = _amor_edge_energy(
                    strain + step * perturbation, lam, mu
                )
                backward = _amor_edge_energy(
                    strain - step * perturbation, lam, mu
                )
                finite_difference = (forward - backward) / (2.0 * step)
                analytic = np.sum(factor * perturbation, axis=(1, 2))
                errors.append(_relative_defect(finite_difference, analytic))
            self.assertLess(errors[-1], tolerance, msg=name)
            if name == "tie":
                # The volumetric branch is C^1 at the tie, so the centered
                # difference converges linearly; require monotone reduction.
                self.assertLess(errors[-1], errors[0])

    def test_jhq_centered_finite_difference_with_mixed_trace_edges(self) -> None:
        monolithic, z = self._state(seed=71)
        traces = self._edge_traces(monolithic, z)
        self.assertTrue(np.any(traces > 0.0))
        self.assertTrue(np.any(traces < 0.0))
        active = monolithic.active_mask(z)
        self.assertTrue(np.any(active))
        rng = np.random.default_rng(73)
        direction = rng.standard_normal(monolithic.n_free) * 1.0e-2
        packed = np.zeros(monolithic.n_packed, dtype=np.float64)
        packed[: monolithic.n_free] = direction
        step = 1.0e-7
        # Keep the FD inside a fixed active set.
        self.assertTrue(
            np.array_equal(active, monolithic.active_mask(z + step * packed))
        )
        self.assertTrue(
            np.array_equal(active, monolithic.active_mask(z - step * packed))
        )
        finite_difference = self._centered_difference(monolithic, z, packed, step)
        analytic = monolithic.jhq_action(z, direction)
        self.assertLess(
            _relative_defect(
                finite_difference[monolithic.n_free + monolithic.n_material :], analytic
            ),
            1.0e-6,
        )

    def test_jhq_exactly_zero_on_inactive_rows(self) -> None:
        monolithic, z = self._state(seed=79)
        driver = monolithic.psi_plus(z)
        inactive = MonolithicGripResidual(self.model, monolithic.u_bar, driver + 1.0)
        inactive_z = inactive.pack(*monolithic.unpack(z))
        rng = np.random.default_rng(83)
        direction = rng.standard_normal(inactive.n_free)
        result = inactive.jhq_action(inactive_z, direction)
        self.assertTrue(np.array_equal(result, np.zeros(inactive.n_material)))

    # ------------------------------------------------------------------
    # 7. Full packed JVP, adjoint, and nonsymmetry
    # ------------------------------------------------------------------

    def test_full_jvp_centered_finite_difference_fixed_active_set(self) -> None:
        monolithic, z = self._state(seed=89)
        rng = np.random.default_rng(97)
        direction = rng.standard_normal(monolithic.n_packed) * 1.0e-3
        step = 1.0e-7
        active = monolithic.active_mask(z)
        self.assertTrue(np.array_equal(active, monolithic.active_mask(z + step * direction)))
        self.assertTrue(np.array_equal(active, monolithic.active_mask(z - step * direction)))
        finite_difference = self._centered_difference(monolithic, z, direction, step)
        analytic = monolithic.jvp(z, direction)
        self.assertEqual(analytic.shape, (monolithic.n_packed,))
        self.assertLess(_relative_defect(finite_difference, analytic), 1.0e-6)

    def test_production_dense_jacobian_is_nonsymmetric(self) -> None:
        monolithic, z = self._state(seed=101)
        # Assembled from the production jvp itself: this is production
        # self-data used only for STRUCTURAL evidence (block pattern and
        # nonsymmetry of the production operator).  Correctness evidence
        # comes from the independent oracle tests below, never from this
        # helper.
        jacobian = monolithic._dense_jacobian(z)
        self.assertEqual(jacobian.shape, (monolithic.n_packed, monolithic.n_packed))
        asymmetry = np.max(np.abs(jacobian - jacobian.T))
        scale = np.max(np.abs(jacobian))
        self.assertGreater(asymmetry, 1.0e-3 * scale)
        free, material = monolithic.n_free, monolithic.n_material
        qd = jacobian[:free, free : free + material]
        dq = jacobian[free : free + material, :free]
        # The damage rows have no q dependence while the mechanics rows do:
        # explicit structural evidence of nonsymmetry.
        self.assertGreater(float(np.max(np.abs(qd))), 5.0e-3 * scale)
        self.assertEqual(float(np.max(np.abs(dq))), 0.0)
        # The self-adjoint blocks remain symmetric for consistency.
        self.assertLess(
            float(np.max(np.abs(jacobian[:free, :free] - jacobian[:free, :free].T))),
            1.0e-10 * scale,
        )
        damage_block = jacobian[free : free + material, free : free + material]
        self.assertLess(
            float(np.max(np.abs(damage_block - damage_block.T))), 1.0e-10 * scale
        )

    def test_full_jvp_matches_independent_oracle(self) -> None:
        rng = np.random.default_rng(211)
        configurations = []
        # Below/above committed mixes and different target loads, including
        # an all-inactive configuration and a partially active one.
        base = self._state(seed=11)
        base_psi = base[0].psi_plus(base[1])
        configurations.append(base)
        configurations.append(
            self._state(seed=12, u_bar=2.0e-3, committed=base_psi * 0.5)
        )
        configurations.append(
            self._state(seed=13, u_bar=0.0, committed=base_psi + 1.0)
        )
        for index, (monolithic, z) in enumerate(configurations):
            oracle = _IndependentMonolithicOracle(
                self.model, monolithic.u_bar, monolithic.committed_history
            )
            directions = []
            for block in range(3):
                packed = np.zeros(monolithic.n_packed, dtype=np.float64)
                start = (
                    0,
                    monolithic.n_free,
                    monolithic.n_free + monolithic.n_material,
                )[block]
                stop = (
                    monolithic.n_free,
                    monolithic.n_free + monolithic.n_material,
                    monolithic.n_packed,
                )[block]
                packed[start:stop] = rng.standard_normal(stop - start) * 1.0e-2
                directions.append(packed)
            directions.append(rng.standard_normal(monolithic.n_packed) * 1.0e-3)
            for direction in directions:
                production = monolithic.jvp(z, direction)
                independent = oracle.jvp(z, direction)
                self.assertLess(
                    _relative_defect(production, independent),
                    1.0e-10,
                    msg=f"configuration {index}",
                )

    def test_oracle_damage_graph_matches_frozen_operator_export(self) -> None:
        # One-time validation of the oracle's independent cubic-kernel graph
        # against the frozen operator export (operator data, not a residual
        # helper): the independent rebuild must reproduce the frozen CSR
        # action to float64 roundoff.
        rng = np.random.default_rng(223)
        oracle = _IndependentMonolithicOracle(
            self.model, 1.0e-3, np.zeros(self.model.n_material)
        )
        frozen = self.model.material_operator.to_scipy().tocsr()
        for _ in range(3):
            values = rng.random(self.model.n_material)
            independent = oracle._damage_graph(values)
            exported = np.asarray(frozen @ values, dtype=np.float64)
            self.assertLess(
                _relative_defect(independent, exported), 1.0e-12
            )

    def test_vjp_bilinear_identity_against_oracle(self) -> None:
        # The production adjoint is tied to the INDEPENDENT oracle action
        # through the bilinear identity, not to a dense matrix built from
        # the production jvp.
        monolithic, z = self._state(seed=227)
        oracle = _IndependentMonolithicOracle(
            self.model, monolithic.u_bar, monolithic.committed_history
        )
        rng = np.random.default_rng(229)
        for _ in range(4):
            first = rng.standard_normal(monolithic.n_packed) * 1.0e-2
            second = rng.standard_normal(monolithic.n_packed)
            lhs = float(np.dot(oracle.jvp(z, first), second))
            rhs = float(np.dot(first, monolithic.vjp(z, second)))
            self.assertLess(
                abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300), 1.0e-10
            )
            # Production jvp/vjp internal consistency under the same
            # identity.
            lhs_production = float(np.dot(monolithic.jvp(z, first), second))
            self.assertLess(
                abs(lhs_production - rhs)
                / max(abs(lhs_production), abs(rhs), 1.0e-300),
                1.0e-10,
            )

    # ------------------------------------------------------------------
    # Cached-static immutability
    # ------------------------------------------------------------------

    def test_cached_static_internals_are_irrevocably_read_only(self) -> None:
        monolithic, z = self._state(seed=331)
        rng = np.random.default_rng(337)
        direction = rng.standard_normal(monolithic.n_packed) * 1.0e-3
        residual_before = monolithic.residual(z)
        jvp_before = monolithic.jvp(z, direction)
        vjp_before = monolithic.vjp(z, direction)
        matrix = monolithic._damage_matrix
        graph_direction = rng.standard_normal(monolithic.n_material)
        graph_before = np.asarray(matrix @ graph_direction, dtype=np.float64)
        transpose_before = np.asarray(matrix.T @ graph_direction, dtype=np.float64)

        def ultimate_buffer_owner(array: np.ndarray) -> object:
            owner: object = array
            while isinstance(owner, np.ndarray) and owner.base is not None:
                owner = owner.base
            return owner

        for internal in ("data", "indices", "indptr"):
            array = getattr(matrix, internal)
            self.assertFalse(array.flags.writeable)
            self.assertIsInstance(ultimate_buffer_owner(array), bytes)
            with self.assertRaises(ValueError):
                array[0] = array[0]
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            self.assertFalse(array.flags.writeable)
        cached_arrays = (
            monolithic._material_source,
            monolithic._damage_volumes,
            monolithic._committed_history,
            monolithic._prescribed,
            monolithic._eta,
            monolithic._source,
            monolithic._material_phase_volumes,
            monolithic._free,
        )
        for array in cached_arrays:
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0] = array[0]
        # The public committed-history property exposes the frozen array.
        with self.assertRaises(ValueError):
            monolithic.committed_history[0] = monolithic.committed_history[0]
        # Every rejected mutation attempt leaves the SciPy graph operations and
        # all residual/JVP/VJP outputs bit-identical.
        self.assertEqual(
            graph_before.tobytes(),
            np.asarray(matrix @ graph_direction, dtype=np.float64).tobytes(),
        )
        self.assertEqual(
            transpose_before.tobytes(),
            np.asarray(matrix.T @ graph_direction, dtype=np.float64).tobytes(),
        )
        self.assertEqual(residual_before.tobytes(), monolithic.residual(z).tobytes())
        self.assertEqual(jvp_before.tobytes(), monolithic.jvp(z, direction).tobytes())
        self.assertEqual(vjp_before.tobytes(), monolithic.vjp(z, direction).tobytes())

        # The authoritative model operator was exported and copied, never
        # modified while the cached CSR was frozen.
        authoritative_before = self.model.material_operator.to_scipy().tocsr(copy=True)
        authoritative_after = self.model.material_operator.to_scipy().tocsr(copy=True)
        for internal in ("data", "indices", "indptr"):
            self.assertEqual(
                getattr(authoritative_before, internal).tobytes(),
                getattr(authoritative_after, internal).tobytes(),
            )

    # ------------------------------------------------------------------
    # Active-set switching and out-of-range inputs
    # ------------------------------------------------------------------

    def test_active_set_switching_across_committed(self) -> None:
        rng = np.random.default_rng(347)
        base, z0 = self._state(seed=349)
        psi0 = base.psi_plus(z0)
        # The median of an odd material count is an exact driver value, so
        # the committed level is tied bit-for-bit by at least one row at
        # ``t = 0`` and strictly straddled elsewhere along the line.
        level = float(np.median(psi0))
        committed = np.full(base.n_material, level, dtype=np.float64)
        monolithic = MonolithicGripResidual(self.model, base.u_bar, committed)
        z = monolithic.pack(*base.unpack(z0))
        direction = np.zeros(monolithic.n_packed, dtype=np.float64)
        direction[: monolithic.n_free] = rng.standard_normal(monolithic.n_free) * 1.0e-2
        samples = np.linspace(-1.0, 1.0, 41)
        drivers = [monolithic.psi_plus(z + t * direction) for t in samples]
        actives = [driver > committed for driver in drivers]
        self.assertTrue(np.any(actives[0] != actives[-1]))
        tie_rows = drivers[20] == committed
        self.assertTrue(np.any(tie_rows))
        self.assertFalse(np.any(actives[20] & tie_rows))
        for index in range(len(samples) - 1):
            product = (drivers[index] - committed) * (drivers[index + 1] - committed)
            flips = actives[index] != actives[index + 1]
            # Every strict straddle of the committed level flips the branch.
            self.assertTrue(np.all(flips | ~(product < 0.0)))
            # Every flip straddles or exactly ties the committed level; no
            # other discontinuous branch change exists.
            self.assertTrue(np.all(~flips | (product <= 0.0)))
            # RH = H - max(committed, psi) is 1-Lipschitz in psi, so the
            # residual cannot jump across an active-set switch.
            left = monolithic.residual(z + samples[index] * direction)
            right = monolithic.residual(z + samples[index + 1] * direction)
            history_block = slice(monolithic.n_free + monolithic.n_material, None)
            self.assertLess(
                float(
                    np.max(
                        np.abs(left[history_block] - right[history_block])
                        - np.abs(drivers[index] - drivers[index + 1])
                    )
                ),
                1.0e-12,
            )
        # Exact tie: committing the driver value bit-for-bit makes every row
        # inactive and the generalized derivative takes the committed branch,
        # so JHq contributes exactly zero on every row.
        tie_state = z + samples[20] * direction
        driver_tie = monolithic.psi_plus(tie_state)
        tie = MonolithicGripResidual(self.model, monolithic.u_bar, driver_tie.copy())
        tie_z = tie.pack(*monolithic.unpack(tie_state))
        self.assertTrue(np.array_equal(tie.psi_plus(tie_z), tie.committed_history))
        self.assertFalse(np.any(tie.active_mask(tie_z)))
        history_direction = rng.random(monolithic.n_material)
        packed = np.zeros(monolithic.n_packed, dtype=np.float64)
        packed[tie.n_free + tie.n_material :] = history_direction
        self.assertTrue(
            np.array_equal(
                tie.jvp(tie_z, packed)[tie.n_free + tie.n_material :],
                history_direction,
            )
        )

    def test_out_of_range_damage_history_evaluated_without_obstacle(self) -> None:
        # Validation contract: only dtype float64, finite values, and exact
        # packed shapes are enforced.  Out-of-range d (and likewise H) is
        # intentionally NOT rejected, because the monolithic unknown must
        # stay unconstrained: the contract forbids any d obstacle, clipping,
        # or projection.  The finite equations remain evaluable because the
        # phase center weight g(d) = (1-d)^2 + k_res stays strictly positive
        # for every finite d.
        rng = np.random.default_rng(353)
        monolithic = MonolithicGripResidual(
            self.model, 1.0e-3, np.zeros(self.model.n_material)
        )
        q = rng.standard_normal(monolithic.n_free) * 1.0e-2
        d = np.full(monolithic.n_material, 0.1, dtype=np.float64)
        d[0] = -0.25
        d[1] = 1.3
        d[2] = 2.7
        history = np.full(monolithic.n_material, 1.5, dtype=np.float64)
        history[3] = -0.5
        z = monolithic.pack(q, d, history)
        residual = monolithic.residual(z)
        self.assertTrue(np.all(np.isfinite(residual)))
        # The raw inputs pass through unmodified.
        _, d_back, history_back = monolithic.unpack(z)
        self.assertTrue(np.array_equal(d_back, d))
        self.assertTrue(np.array_equal(history_back, history))
        # Rd is the exact raw equation (no range term appears).
        operator = self.model.material_operator
        volumes = np.asarray(operator.volumes, dtype=np.float64)
        dense = np.asarray(operator.to_scipy().todense(), dtype=np.float64)
        expected_rd = volumes * (
            self.model.Gc / self.model.l0 * d
            + self.model.Gc * self.model.l0 * (dense @ d)
            + 2.0 * history * (d - 1.0)
        )
        block = residual[monolithic.n_free : monolithic.n_free + monolithic.n_material]
        self.assertLess(_relative_defect(block, expected_rd), 1.0e-12)
        # Rq is still exactly the authoritative fixed-damage residual at the
        # raw (out-of-range) damage field.
        view = self.model.fixed_damage_view(d)
        authoritative = view.residual(monolithic.reconstruct_displacement(z))[
            self.model.constraints.free_dofs
        ]
        self.assertTrue(np.array_equal(residual[: monolithic.n_free], authoritative))
        # The production JVP still matches the independent oracle at this
        # out-of-range state.
        oracle = _IndependentMonolithicOracle(
            self.model, monolithic.u_bar, monolithic.committed_history
        )
        direction = rng.standard_normal(monolithic.n_packed) * 1.0e-3
        self.assertTrue(np.all(np.isfinite(monolithic.jvp(z, direction))))
        self.assertLess(
            _relative_defect(monolithic.jvp(z, direction), oracle.jvp(z, direction)),
            1.0e-10,
        )

    # ------------------------------------------------------------------
    # 8. Input contracts
    # ------------------------------------------------------------------

    def test_inputs_are_not_mutated(self) -> None:
        monolithic, z = self._state(seed=109)
        rng = np.random.default_rng(113)
        direction = rng.standard_normal(monolithic.n_packed)
        weights = rng.standard_normal(monolithic.n_packed)
        z_before = z.copy()
        q_direction = direction[: monolithic.n_free]
        d_direction = direction[monolithic.n_free : monolithic.n_free + monolithic.n_material]
        h_direction = direction[monolithic.n_free + monolithic.n_material :]
        results = (
            monolithic.residual(z),
            monolithic.psi_plus(z),
            monolithic.active_mask(z),
            monolithic.jvp(z, direction),
            monolithic.vjp(z, weights),
            monolithic.jqq_action(z, q_direction),
            monolithic.jqd_action(z, d_direction),
            monolithic.jdd_action(z, d_direction),
            monolithic.jdh_action(z, h_direction),
            monolithic.jhq_action(z, q_direction),
        )
        for result in results:
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertFalse(np.shares_memory(result, z))
        self.assertTrue(np.array_equal(z, z_before))
        # Destructively writing a returned array must not affect any input or
        # a later evaluation.
        residual = monolithic.residual(z)
        residual[:] = 0.0
        self.assertTrue(np.array_equal(monolithic.residual(z), z_before * 0.0 + monolithic.residual(z.copy())))
        self.assertTrue(np.array_equal(z, z_before))

    def test_invalid_inputs_fail_loudly(self) -> None:
        monolithic, z = self._state(seed=127)
        with self.assertRaisesRegex(TypeError, "model"):
            MonolithicGripResidual(object(), 1.0e-3, np.zeros(self.model.n_material))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "u_bar"):
            MonolithicGripResidual(self.model, -1.0e-3, np.zeros(self.model.n_material))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MonolithicGripResidual(
                self.model, 1.0e-3, -np.ones(self.model.n_material)
            )
        bad_history = np.zeros(self.model.n_material)
        bad_history[0] = np.nan
        with self.assertRaisesRegex(ValueError, "committed history"):
            MonolithicGripResidual(self.model, 1.0e-3, bad_history)
        with self.assertRaisesRegex(ValueError, "shape"):
            monolithic.pack(
                np.zeros(monolithic.n_free + 1),
                np.zeros(monolithic.n_material),
                np.zeros(monolithic.n_material),
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            monolithic.residual(z.astype(np.float32))
        wrong_shape = np.zeros(monolithic.n_packed + 1, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "shape"):
            monolithic.residual(wrong_shape)
        nonfinite = z.copy()
        nonfinite[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            monolithic.residual(nonfinite)
        with self.assertRaisesRegex(ValueError, "finite"):
            monolithic.jvp(z, np.full(monolithic.n_packed, np.nan))
        with self.assertRaisesRegex(ValueError, "dtype"):
            monolithic.jvp(z, np.zeros(monolithic.n_packed, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            monolithic.jqq_action(z, np.zeros(monolithic.n_free + 1))
        with self.assertRaisesRegex(ValueError, "shape"):
            monolithic.jqd_action(z, np.zeros(monolithic.n_material + 1))


if __name__ == "__main__":
    unittest.main()
