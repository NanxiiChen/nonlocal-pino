"""CPU-first validation of the separate float64 fracture2d JAX backend."""

from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import numpy as np

from examples.fracture2d.nonlocal_mechanics import (
    FixedDamageBondAssociatedMechanics,
)
from examples.fracture2d.nonlocal_mechanics_jax import (
    JaxBondAssociatedMechanics,
    JaxCPUReplayResult,
)
from tests.test_fracture2d_nonlocal_continuation import _system


class Fracture2DNonlocalMechanicsJaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.initial_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        (
            cls.coordinates,
            cls.volumes,
            cls.edges,
            cls.operator,
            cls.mechanics,
            cls.constraints,
        ) = _system()
        del cls.operator
        cls.device = jax.devices("cpu")[0]
        cls.backend = JaxBondAssociatedMechanics(
            cls.mechanics, cls.constraints, cls.device
        )
        cls.damage = np.asarray(
            np.linspace(0.0, 0.85, len(cls.coordinates)), dtype=np.float64
        )
        cls.intact_damage = np.zeros(len(cls.coordinates), dtype=np.float64)

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.initial_x64)

    def _fixed(self, damage: np.ndarray) -> FixedDamageBondAssociatedMechanics:
        return FixedDamageBondAssociatedMechanics(self.mechanics, damage)

    def test_flattened_layout_is_exactly_the_valid_cpu_support_layout(self) -> None:
        mask = np.asarray(self.mechanics.support_mask)
        indices = np.asarray(self.mechanics.support_indices)
        beta = np.asarray(self.mechanics.beta)
        xi = np.asarray(self.mechanics.xi)
        valid = np.flatnonzero(mask.reshape(-1))
        expected_edge = (valid // mask.shape[1]).astype(np.int32)
        expected_particle = indices.reshape(-1)[valid].astype(np.int32)

        np.testing.assert_array_equal(
            np.asarray(self.backend.occurrence_edge), expected_edge
        )
        np.testing.assert_array_equal(
            np.asarray(self.backend.occurrence_particle), expected_particle
        )
        np.testing.assert_array_equal(
            np.asarray(self.backend.occurrence_source),
            np.asarray(self.mechanics.source)[expected_edge].astype(np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(self.backend.occurrence_beta), beta.reshape(-1)[valid]
        )
        np.testing.assert_array_equal(
            np.asarray(self.backend.occurrence_xi), xi.reshape(-1, 2)[valid]
        )
        self.assertEqual(self.backend.occurrence_edge.dtype, np.dtype(np.int32))
        self.assertEqual(self.backend.occurrence_particle.dtype, np.dtype(np.int32))
        self.assertEqual(self.backend.occurrence_beta.dtype, np.dtype(np.float64))
        self.assertEqual(
            self.backend.flattened_support_count, int(np.count_nonzero(mask))
        )

    def test_tangent_residual_conservation_and_psd_match_cpu_for_damage_fields(self) -> None:
        rng = np.random.default_rng(812)
        displacement = np.asarray(0.02 * rng.normal(size=self.mechanics.n_dofs))
        direction = np.asarray(rng.normal(size=self.mechanics.n_dofs))
        other_direction = np.asarray(rng.normal(size=self.mechanics.n_dofs))

        for damage in (self.intact_damage, self.damage):
            fixed = self._fixed(damage)
            np.testing.assert_allclose(
                np.asarray(self.backend.residual(damage, displacement)),
                fixed.residual(displacement),
                rtol=0.0,
                atol=5.0e-13,
            )
            np.testing.assert_allclose(
                np.asarray(self.backend.energy(damage, displacement)),
                fixed.energy(displacement),
                rtol=0.0,
                atol=5.0e-13,
            )
            np.testing.assert_allclose(
                np.asarray(self.backend.tangent_action(damage, direction)),
                fixed.tangent_action(direction),
                rtol=0.0,
                atol=5.0e-13,
            )

            tangent_direction = np.asarray(
                self.backend.tangent_action(damage, direction)
            )
            tangent_other = np.asarray(
                self.backend.tangent_action(damage, other_direction)
            )
            left = float(other_direction @ tangent_direction)
            right = float(direction @ tangent_other)
            self.assertLessEqual(abs(left - right), 2.0e-11)
            self.assertGreaterEqual(float(direction @ tangent_direction), -2.0e-11)

            translation_x = np.zeros(self.mechanics.n_dofs, dtype=np.float64)
            translation_x[0::2] = 0.37
            translation_y = np.zeros(self.mechanics.n_dofs, dtype=np.float64)
            translation_y[1::2] = -0.29
            rotation = (
                1.0e-4
                * self.coordinates
                @ np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64).T
            ).ravel()
            for mode in (translation_x, translation_y, rotation):
                np.testing.assert_allclose(
                    np.asarray(self.backend.residual(damage, mode)),
                    0.0,
                    rtol=0.0,
                    atol=5.0e-12,
                )

            residual = np.asarray(self.backend.residual(damage, displacement)).reshape(
                -1, 2
            )
            np.testing.assert_allclose(
                np.sum(residual, axis=0), 0.0, rtol=0.0, atol=5.0e-12
            )
            moment = np.sum(
                self.coordinates[:, 0] * residual[:, 1]
                - self.coordinates[:, 1] * residual[:, 0]
            )
            self.assertLessEqual(abs(float(moment)), 5.0e-12)

    def test_closed_form_block_jacobi_matches_cpu_and_handles_constraint_masks(self) -> None:
        fixed = self._fixed(self.damage)
        blocks = np.asarray(self.backend.tangent_block_diagonal(self.damage))
        np.testing.assert_allclose(
            blocks,
            fixed.tangent_block_diagonal,
            rtol=0.0,
            atol=5.0e-13,
        )

        inverse = np.asarray(self.backend.block_jacobi_inverse(self.damage))
        expected = np.zeros_like(inverse)
        for particle in range(self.mechanics.n_particles):
            free = np.flatnonzero(
                np.asarray(self.backend.free_mask)[particle]
            )
            if len(free) == 2:
                expected[particle][np.ix_(free, free)] = np.linalg.inv(
                    blocks[particle][np.ix_(free, free)]
                )
            elif len(free) == 1:
                expected[particle, free[0], free[0]] = 1.0 / blocks[
                    particle, free[0], free[0]
                ]
        np.testing.assert_allclose(inverse, expected, rtol=0.0, atol=5.0e-13)
        self.assertEqual(
            int(np.count_nonzero(np.sum(np.asarray(self.backend.free_mask), axis=1) == 0)),
            1,
        )
        self.assertEqual(
            int(np.count_nonzero(np.sum(np.asarray(self.backend.free_mask), axis=1) == 1)),
            7,
        )
        self.assertEqual(
            int(np.count_nonzero(np.sum(np.asarray(self.backend.free_mask), axis=1) == 2)),
            8,
        )

        vector = np.asarray(
            np.linspace(-0.4, 0.7, self.backend.n_free), dtype=np.float64
        )
        expected_vector = np.zeros_like(vector)
        position = np.full(self.backend.n_dofs, -1, dtype=np.int64)
        position[self.constraints.free_dofs] = np.arange(
            self.backend.n_free, dtype=np.int64
        )
        for particle in range(self.backend.n_particles):
            components = np.flatnonzero(np.asarray(self.backend.free_mask)[particle])
            if len(components) == 0:
                continue
            dofs = 2 * particle + components
            free_positions = position[dofs]
            expected_vector[free_positions] = np.linalg.solve(
                blocks[particle][np.ix_(components, components)],
                vector[free_positions],
            )
        np.testing.assert_allclose(
            np.asarray(self.backend.apply_block_jacobi(self.damage, vector)),
            expected_vector,
            rtol=0.0,
            atol=5.0e-13,
        )

    def test_cold_warm_early_exit_seed_policy_and_statuses(self) -> None:
        fixed = self._fixed(self.damage)
        load = 0.04
        rtol = 1.0e-11
        atol = 1.0e-13
        maxiter = 3000
        cpu = fixed.solve(self.constraints, load, rtol, atol, maxiter)
        cold = self.backend.solve(self.damage, load, rtol, atol, maxiter)
        self.assertEqual(cold.status, "converged")
        self.assertTrue(cold.converged)
        np.testing.assert_allclose(
            np.asarray(cold.displacement), cpu.displacement, rtol=0.0, atol=5.0e-12
        )
        np.testing.assert_allclose(
            np.asarray(cold.free_residual), cpu.free_residual, rtol=0.0, atol=5.0e-12
        )

        exact_warm = self.backend.solve(
            self.damage,
            load,
            rtol,
            atol,
            maxiter,
            initial_guess=np.asarray(cold.displacement),
            allow_initial_early_exit=True,
        )
        self.assertEqual(int(np.asarray(exact_warm.cg_iterations)), 0)
        self.assertTrue(bool(np.asarray(exact_warm.early_exit)))
        self.assertFalse(bool(np.asarray(exact_warm.cg_initial_guess_used)))
        np.testing.assert_array_equal(
            np.asarray(exact_warm.displacement), np.asarray(cold.displacement)
        )

        below = np.asarray(cold.displacement).copy()
        below[int(self.constraints.free_dofs[0])] += 1.0e-15
        below_limit = self.backend.solve(
            self.damage,
            load,
            rtol,
            atol,
            maxiter,
            initial_guess=below,
            allow_initial_early_exit=False,
        )
        self.assertLess(
            float(np.asarray(below_limit.initial_free_residual_norm)),
            float(np.asarray(below_limit.cg_convergence_limit)),
        )
        self.assertFalse(bool(np.asarray(below_limit.early_exit)))
        self.assertFalse(bool(np.asarray(below_limit.cg_initial_guess_used)))
        np.testing.assert_allclose(
            np.asarray(below_limit.displacement), np.asarray(cold.displacement),
            rtol=0.0, atol=5.0e-12
        )

        prescribed = self.constraints.prescribed_vector(load)
        above_limit = self.backend.solve(
            self.damage,
            load,
            rtol,
            atol,
            maxiter,
            initial_guess=prescribed,
            allow_initial_early_exit=False,
        )
        self.assertGreaterEqual(
            float(np.asarray(above_limit.initial_free_residual_norm)),
            float(np.asarray(above_limit.cg_convergence_limit)),
        )
        self.assertTrue(bool(np.asarray(above_limit.cg_initial_guess_used)))
        self.assertFalse(bool(np.asarray(above_limit.early_exit)))
        np.testing.assert_allclose(
            np.asarray(above_limit.displacement), np.asarray(cold.displacement),
            rtol=0.0, atol=5.0e-12
        )

        maxed = self.backend.solve(
            self.damage, load, 1.0e-14, 0.0, 3
        )
        self.assertEqual(maxed.status, "maxiter")
        self.assertFalse(maxed.converged)
        self.assertEqual(int(np.asarray(maxed.cg_iterations)), 3)
        self.assertEqual(int(np.asarray(maxed.cg_info)), 3)

        equality_probe = self.backend.solve(
            self.damage,
            load,
            0.0,
            float(np.asarray(cold.free_rhs_norm)),
            1,
        )
        self.assertFalse(bool(np.asarray(equality_probe.early_exit)))
        self.assertEqual(int(np.asarray(equality_probe.cg_iterations)), 1)

        overflow_damage = np.full(
            self.backend.n_particles, 1.0e308, dtype=np.float64
        )
        with self.assertRaisesRegex(FloatingPointError, "degradation"):
            self.backend.solve(
                overflow_damage, load, rtol, atol, maxiter
            )
        with self.assertRaisesRegex(FloatingPointError, "degradation"):
            self.backend.energy(
                overflow_damage,
                np.zeros(self.backend.n_dofs, dtype=np.float64),
            )
        overflow_displacement = np.zeros(self.backend.n_dofs, dtype=np.float64)
        overflow_displacement[0] = 1.0e308
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            self.backend.energy(
                self.damage,
                overflow_displacement,
            )

    def test_replay_pairs_cpu_result_and_rejects_a_cpu_gate_failure(self) -> None:
        load = 0.04
        result = self.backend.solve(
            self.damage, load, 1.0e-11, 1.0e-13, 3000
        )
        replay = self.backend.replay_cpu(
            result, self.damage, load, 1.0e-11, 1.0e-13, 3000
        )
        self.assertIsInstance(replay, JaxCPUReplayResult)
        cpu_reference = self._fixed(self.damage).solve(
            self.constraints, load, 1.0e-11, 1.0e-13, 3000
        )
        np.testing.assert_allclose(
            replay.cpu_result.displacement,
            cpu_reference.displacement,
            rtol=0.0,
            atol=5.0e-12,
        )
        self.assertAlmostEqual(
            replay.cpu_result.top_reaction, cpu_reference.top_reaction, places=11
        )
        self.assertEqual(replay.gpu_result, result)

        with self.assertRaisesRegex(ValueError, "damage does not match"):
            zero_result = self.backend.solve(
                self.intact_damage, 0.0, 1.0e-11, 1.0e-13, 3000
            )
            self.backend.replay_cpu(
                zero_result,
                np.ones_like(self.damage),
                0.0,
                1.0e-11,
                1.0e-13,
                3000,
            )
        with self.assertRaisesRegex(ValueError, "cg_rtol"):
            self.backend.replay_cpu(
                result, self.damage, load, 2.0e-11, 1.0e-13, 3000
            )

        warm = self.backend.solve(
            self.damage,
            load,
            1.0e-11,
            1.0e-13,
            3000,
            initial_guess=np.asarray(result.displacement),
            allow_initial_early_exit=True,
        )
        with self.assertRaisesRegex(ValueError, "initial_guess_supplied"):
            self.backend.replay_cpu(
                warm, self.damage, load, 1.0e-11, 1.0e-13, 3000
            )

        forged_initial = replace(
            warm,
            initial_displacement=jax.device_put(
                self.constraints.prescribed_vector(load), self.device
            ),
        )
        with self.assertRaisesRegex(ValueError, "initial_displacement"):
            self.backend.replay_cpu(
                forged_initial,
                self.damage,
                load,
                1.0e-11,
                1.0e-13,
                3000,
                initial_guess=np.asarray(warm.displacement),
                allow_initial_early_exit=True,
            )

        forged_evaluation = replace(
            result.evaluation,
            top_reaction=jax.device_put(
                np.asarray(12345.0, dtype=np.float64), self.device
            ),
        )
        forged_result = replace(result, evaluation=forged_evaluation)
        forged_replay = self.backend.replay_cpu(
            forged_result, self.damage, load, 1.0e-11, 1.0e-13, 3000
        )
        self.assertEqual(
            float(np.asarray(forged_replay.gpu_result.top_reaction)), 12345.0
        )
        self.assertAlmostEqual(
            forged_replay.cpu_result.top_reaction,
            cpu_reference.top_reaction,
            places=11,
        )

        float32_result = replace(
            result,
            displacement=jax.device_put(
                np.asarray(result.displacement, dtype=np.float32), self.device
            ),
        )
        with self.assertRaisesRegex(ValueError, "float64"):
            self.backend.replay_cpu(
                float32_result, self.damage, load, 1.0e-11, 1.0e-13, 3000
            )

        bad_displacement = np.asarray(result.displacement).copy()
        bad_displacement[int(self.constraints.free_dofs[0])] += 1.0e-3
        bad_result = replace(
            result, displacement=jax.device_put(bad_displacement, self.device)
        )
        with self.assertRaisesRegex(RuntimeError, "CPU CG gate"):
            self.backend.replay_cpu(
                bad_result, self.damage, load, 1.0e-11, 1.0e-13, 3000
            )

    def test_dynamic_values_share_one_compile_trace(self) -> None:
        backend = JaxBondAssociatedMechanics(
            self.mechanics, self.constraints, self.device
        )
        first = backend.trace_count
        backend.solve(self.intact_damage, 0.01, 1.0e-11, 1.0e-13, 3000)
        after_first = backend.trace_count
        backend.solve(
            self.damage,
            0.012,
            1.0e-10,
            1.0e-12,
            17,
            initial_guess=self.constraints.prescribed_vector(0.012),
            allow_initial_early_exit=False,
        )
        self.assertEqual(after_first, first + 1)
        self.assertEqual(backend.trace_count, after_first)

        device_damage = jax.device_put(self.damage, self.device)
        device_guess = jax.device_put(
            self.constraints.prescribed_vector(0.014), self.device
        )
        device_result = backend.solve(
            device_damage,
            0.014,
            1.0e-10,
            1.0e-12,
            3000,
            initial_guess=device_guess,
        )
        self.assertEqual(device_result.displacement.devices(), {self.device})
        self.assertEqual(backend.trace_count, after_first)

    def test_invalid_inputs_and_x64_policy_are_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "float64"):
            self.backend.solve(
                self.damage.astype(np.float32), 0.01, 1.0e-11, 1.0e-13, 10
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            self.backend.tangent_action(
                self.damage, np.zeros(self.backend.n_dofs - 1, dtype=np.float64)
            )
        with self.assertRaises(ValueError):
            self.backend.solve(self.damage, 0.01, 0.0, 0.0, 10)
        with self.assertRaises(ValueError):
            self.backend.solve(self.damage, 0.01, 1.0e-11, 1.0e-13, 0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.backend.solve(
                self.damage, 0.01, 1.0e-11, 1.0e-13, 10,
                allow_initial_early_exit=1,  # type: ignore[arg-type]
            )
        bad_guess = self.constraints.prescribed_vector(0.01)
        bad_guess[int(self.constraints.top_particles[0]) * 2 + 1] += 1.0e-15
        with self.assertRaisesRegex(ValueError, "constrained"):
            self.backend.solve(
                self.damage,
                0.01,
                1.0e-11,
                1.0e-13,
                10,
                initial_guess=bad_guess,
            )

        jax.config.update("jax_enable_x64", False)
        try:
            with self.assertRaisesRegex(RuntimeError, "jax_enable_x64"):
                self.backend.energy(self.damage, np.zeros(self.backend.n_dofs))
        finally:
            jax.config.update("jax_enable_x64", True)

    @unittest.skipUnless(
        any(device.platform == "gpu" for device in jax.devices()),
        "GPU-specific residency assertion requires a visible GPU",
    )
    def test_gpu_arrays_are_committed_to_the_explicit_gpu(self) -> None:
        gpu = next(device for device in jax.devices() if device.platform == "gpu")
        backend = JaxBondAssociatedMechanics(
            self.mechanics, self.constraints, gpu
        )
        result = backend.solve(self.intact_damage, 0.01, 1.0e-11, 1.0e-13, 3000)
        self.assertEqual(backend.source.devices(), {gpu})
        self.assertEqual(backend.occurrence_xi.devices(), {gpu})
        self.assertEqual(result.displacement.devices(), {gpu})
        self.assertEqual(result.free_residual.devices(), {gpu})


if __name__ == "__main__":
    unittest.main()
