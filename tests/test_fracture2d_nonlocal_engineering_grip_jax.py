"""Focused contract tests for the JAX A2 engineering-grip mechanics backend."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jax
import numpy as np

from examples.fracture2d import nonlocal_engineering_grip as engineering
from examples.fracture2d import nonlocal_engineering_grip_stabilized as stabilized
from examples.fracture2d.nonlocal_a2_coupled import (
    A2MechanicsSolveResult,
    _unit_system_digest,
)
from examples.fracture2d.nonlocal_engineering_grip_jax import (
    JaxA2CPUReplayAdapter,
    JaxA2CPUReplayResult,
    JaxA2SolveResult,
    JaxA2UnionMechanics,
)
from examples.fracture2d.nonlocal_grip_diagnostic import (
    build_union_geometry_from_arrays,
    build_volume_grip_constraints,
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


class EngineeringGripJaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_x64 = bool(jax.config.x64_enabled)
        jax.config.update("jax_enable_x64", True)
        coordinates, triangles, volumes, bottom, top, edges, delta = _material_cloud()
        cls.geometry = build_union_geometry_from_arrays(
            coordinates, triangles, volumes, bottom, top, 0, edges, delta, 0.2, 0.0
        )
        cls.model = engineering.build_engineering_grip_model(
            cls.geometry, 2.0, 0.3, 1.0e-8, 1.0e-12, 1.7, 0.65
        )
        gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
        if gpu_devices:
            cls.device = gpu_devices[0]
        else:
            cls.device = jax.devices("cpu")[0]
        cls.backend = JaxA2UnionMechanics(
            cls.model.static, cls.model.constraints, cls.device
        )
        cls.adapter = JaxA2CPUReplayAdapter(cls.backend)
        cls.zero_damage = np.zeros(cls.model.n_material, dtype=np.float64)
        cls.damage = np.asarray(
            np.linspace(0.05, 0.35, cls.model.n_material), dtype=np.float64
        )

    @classmethod
    def tearDownClass(cls) -> None:
        jax.config.update("jax_enable_x64", cls.old_x64)

    def test_center_weights_are_the_exact_phase_measure(self) -> None:
        phase_roundoff = float(8.0 * np.finfo(np.float64).eps)
        for damage in (self.zero_damage, self.damage):
            view = self.model.fixed_damage_view(damage)
            gpu = np.asarray(jax.device_get(self.backend.center_weights(damage)))
            # XLA may fuse the multiply/add sequence differently from NumPy
            # even with float64.  The algebraic phase measure is the contract;
            # bitwise equality is not a cross-backend floating-point contract.
            np.testing.assert_allclose(
                gpu,
                view.center_weights,
                rtol=phase_roundoff,
                atol=phase_roundoff,
            )
            static = self.model.static
            expected = (
                static.union_material_phase_volumes
                * np.pad((1.0 - damage) ** 2 + static.k_res, (0, static.n_particles - static.material_count))
                + static.grip_phase_volumes * (1.0 + static.k_res)
            )
            np.testing.assert_allclose(
                gpu,
                expected,
                rtol=phase_roundoff,
                atol=phase_roundoff,
            )

    def test_energy_residual_tangent_and_block_algebra_match_cpu(self) -> None:
        rng = np.random.default_rng(19)
        displacement = np.asarray(
            1.0e-2 * rng.normal(size=self.model.n_dofs), dtype=np.float64
        )
        direction = np.asarray(rng.normal(size=self.model.n_dofs), dtype=np.float64)
        for damage in (self.zero_damage, self.damage):
            view = self.model.fixed_damage_view(damage)
            np.testing.assert_allclose(
                np.asarray(jax.device_get(self.backend.residual(damage, displacement))),
                view.residual(displacement),
                rtol=0.0,
                atol=5.0e-13,
            )
            self.assertLessEqual(
                abs(
                    float(jax.device_get(self.backend.energy(damage, displacement)))
                    - view.energy(displacement)
                ),
                5.0e-13,
            )
            np.testing.assert_allclose(
                np.asarray(jax.device_get(self.backend.tangent_action(damage, direction))),
                view.tangent_action(direction),
                rtol=0.0,
                atol=5.0e-13,
            )
            np.testing.assert_allclose(
                np.asarray(jax.device_get(self.backend.tangent_block_diagonal(damage))),
                view.tangent_block_diagonal,
                rtol=0.0,
                atol=5.0e-13,
            )
            residual = np.asarray(
                jax.device_get(self.backend.residual(damage, displacement))
            ).reshape(-1, 2)
            np.testing.assert_allclose(
                np.sum(residual, axis=0), 0.0, rtol=0.0, atol=5.0e-12
            )

    def test_block_jacobi_inverse_and_application_match_cpu(self) -> None:
        view = self.model.fixed_damage_view(self.damage)
        blocks = np.asarray(
            jax.device_get(self.backend.tangent_block_diagonal(self.damage))
        )
        inverse = np.asarray(
            jax.device_get(self.backend.block_jacobi_inverse(self.damage))
        )
        positions = np.full(self.model.n_dofs, -1, dtype=np.int64)
        positions[self.model.constraints.free_dofs] = np.arange(
            len(self.model.constraints.free_dofs), dtype=np.int64
        )
        expected = np.zeros_like(inverse)
        for particle in range(self.model.n_union):
            components = np.flatnonzero(
                np.asarray(self.backend.free_mask)[particle]
            )
            if len(components) == 2:
                expected[particle][np.ix_(components, components)] = np.linalg.inv(
                    blocks[particle][np.ix_(components, components)]
                )
            elif len(components) == 1:
                expected[particle, components[0], components[0]] = 1.0 / blocks[
                    particle, components[0], components[0]
                ]
        np.testing.assert_allclose(inverse, expected, rtol=0.0, atol=5.0e-13)
        del view

        vector = np.asarray(
            np.linspace(-0.4, 0.7, self.backend.n_free), dtype=np.float64
        )
        expected_vector = np.zeros_like(vector)
        for particle in range(self.model.n_union):
            components = np.flatnonzero(
                np.asarray(self.backend.free_mask)[particle]
            )
            if len(components) == 0:
                continue
            dofs = 2 * particle + components
            free_positions = positions[dofs]
            expected_vector[free_positions] = np.linalg.solve(
                blocks[particle][np.ix_(components, components)],
                vector[free_positions],
            )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(self.backend.apply_block_jacobi(self.damage, vector))),
            expected_vector,
            rtol=0.0,
            atol=5.0e-13,
        )

    def test_adapter_solve_is_cpu_authoritative_with_provenance(self) -> None:
        view = self.model.fixed_damage_view(self.damage)
        cpu_reference = engineering.solve_fixed_outer_displacement(
            view,
            self.model.constraints,
            0.08,
            1.0e-10,
            1.0e-13,
            3000,
        )
        seed = engineering._direct_seed(
            cpu_reference.displacement, 0.08, 0.08, self.model.constraints
        )
        pair = self.adapter.solve_pair(
            view,
            self.model.constraints,
            0.08,
            1.0e-10,
            1.0e-13,
            3000,
            initial_guess=seed,
            seed_kind="previous_direct_solution",
            source_state="committed",
            source_step=3,
            source_sweep=1,
        )
        self.assertIsInstance(pair, JaxA2CPUReplayResult)
        self.assertIsInstance(pair.jax_result, JaxA2SolveResult)
        self.assertIsInstance(pair.cpu_result, A2MechanicsSolveResult)
        result = pair.cpu_result

        engineering._check_exact_constraints(
            result.displacement, self.model.constraints, 0.08, "adapter solve"
        )
        self.assertEqual(result.cg_info, 0)
        self.assertLess(result.free_residual_norm, result.cg_convergence_limit)
        np.testing.assert_allclose(
            result.displacement, cpu_reference.displacement, rtol=0.0, atol=1.0e-12
        )
        self.assertAlmostEqual(
            result.outer_reaction, cpu_reference.outer_reaction, places=12
        )
        self.assertAlmostEqual(
            result.bottom_reaction, cpu_reference.bottom_reaction, places=12
        )
        self.assertAlmostEqual(
            result.total_moment, cpu_reference.total_moment, places=12
        )
        np.testing.assert_allclose(
            result.total_force, 0.0, rtol=0.0, atol=2.0e-10
        )

        self.assertIs(result.static_runtime_identity, self.model.static.runtime_identity)
        self.assertIs(result.constraints, self.model.constraints)
        self.assertEqual(result.static_digest, view.static.deterministic_digest)
        self.assertEqual(result.damage_digest, view.damage_digest)
        self.assertEqual(
            result.unit_system_digest,
            _unit_system_digest(
                view.static.deterministic_digest, view.damage_digest
            ),
        )
        provenance = result.seed_provenance
        self.assertEqual(provenance.kind, "previous_direct_solution")
        self.assertEqual(provenance.source_state, "committed")
        self.assertEqual(provenance.source_step, 3)
        self.assertEqual(provenance.source_sweep, 1)
        self.assertEqual(provenance.target_damage_digest, result.damage_digest)
        self.assertEqual(provenance.unit_system_digest, result.unit_system_digest)
        self.assertFalse(provenance.early_exit_eligible)
        self.assertFalse(provenance.early_exit_used)
        fresh_view = self.model.fixed_damage_view(self.damage)
        expected_initial = float(
            np.linalg.norm(fresh_view.residual(seed)[self.model.constraints.free_dofs])
        )
        self.assertEqual(provenance.initial_residual_norm, expected_initial)
        self.assertEqual(provenance.final_residual_norm, result.free_residual_norm)
        self.assertEqual(result.cg_iterations, int(np.asarray(pair.jax_result.cg_iterations)))
        self.assertGreaterEqual(result.cg_iterations, 0)

        # The CPU evaluation is the authoritative field source.
        self.assertEqual(result.evaluation.energy, result.energy)
        np.testing.assert_array_equal(result.evaluation.residual, result.residual)

    def test_stabilized_cache_retains_only_cpu_authoritative_result(self) -> None:
        view = self.model.fixed_damage_view(self.damage)
        pair = self.adapter.solve_pair(
            view,
            self.model.constraints,
            0.08,
            1.0e-10,
            1.0e-13,
            3000,
            seed_kind="stabilized_cache_test",
            source_state="stabilized_cache_test",
            source_step=1,
            source_sweep=2,
        )
        cache = stabilized._StabilizedMechanicsReuse.from_selected(
            self.model,
            0.08,
            np.array(self.damage, copy=True),
            pair.cpu_result,
        )

        self.assertIs(cache.result, pair.cpu_result)
        self.assertIsInstance(cache.result.displacement, np.ndarray)
        self.assertIsInstance(cache.result.evaluation.residual, np.ndarray)
        self.assertFalse(cache.result.displacement.flags.writeable)
        self.assertFalse(cache.result.evaluation.residual.flags.writeable)
        self.assertFalse(hasattr(cache.result, "jax_result"))
        np.testing.assert_array_equal(
            cache.result.displacement,
            np.asarray(jax.device_get(pair.jax_result.displacement)),
        )

    def test_adapter_rejects_nonconvergence_finiteness_and_identity(self) -> None:
        view = self.model.fixed_damage_view(self.damage)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.adapter.solve_fixed_outer_displacement(
                view, self.model.constraints, 0.08, 1.0e-14, 0.0, 2
            )
        with self.assertRaisesRegex(FloatingPointError, "center weights"):
            self.backend.solve(
                np.full(self.model.n_material, 1.0e308, dtype=np.float64),
                0.08,
                1.0e-10,
                1.0e-13,
                3000,
            )
        with self.assertRaisesRegex(ValueError, "float64"):
            self.backend.solve(
                self.damage.astype(np.float32), 0.08, 1.0e-10, 1.0e-13, 3000
            )
        bad_guess = self.model.constraints.prescribed_vector(0.08)
        bad_guess[int(self.model.constraints.constrained_dofs[0])] += 1.0e-15
        with self.assertRaisesRegex(ValueError, "constrained"):
            self.adapter.solve_fixed_outer_displacement(
                view,
                self.model.constraints,
                0.08,
                1.0e-10,
                1.0e-13,
                3000,
                initial_guess=bad_guess,
            )

        other_geometry = build_union_geometry_from_arrays(
            self.geometry.material_reference_coordinates,
            self.geometry.material_reference_triangles,
            self.geometry.material_reference_volumes,
            self.geometry.material_bottom_mask[: self.model.n_material],
            self.geometry.material_top_mask[: self.model.n_material],
            0,
            self.geometry.material_reference_edges,
            self.geometry.delta,
            0.1,
            0.0,
        )
        other_model = engineering.build_engineering_grip_model(
            other_geometry, 2.0, 0.3, 1.0e-8, 1.0e-12, 1.7, 0.65
        )
        other_view = other_model.fixed_damage_view(self.damage)
        with self.assertRaisesRegex(ValueError, "static identity"):
            self.adapter.ensure_view(other_view, self.model.constraints)
        equal_constraints = build_volume_grip_constraints(self.geometry, 0)
        with self.assertRaisesRegex(ValueError, "exact bound object"):
            self.adapter.ensure_view(view, equal_constraints)
        with self.assertRaisesRegex(TypeError, "A2FixedDamageView"):
            self.adapter.ensure_view(object(), self.model.constraints)  # type: ignore[arg-type]

    def test_controller_rejects_generic_callables_and_foreign_adapters(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        for payload in (
            lambda *args, **kwargs: None,
            engineering.solve_fixed_outer_displacement,
        ):
            with self.assertRaisesRegex(TypeError, "JaxA2CPUReplayAdapter"):
                engineering.run_zero_damage_work_check(
                    self.model,
                    config.work_check_load,
                    config.work_epsilon,
                    config.work_relative_limit,
                    config.cg_rtol,
                    config.cg_atol,
                    config.cg_maxiter,
                    mechanics_adapter=payload,  # type: ignore[arg-type]
                )
            with self.assertRaisesRegex(TypeError, "JaxA2CPUReplayAdapter"):
                engineering.run_strict_engineering_grip(
                    self.model, config, mechanics_adapter=payload  # type: ignore[arg-type]
                )
        other_geometry = build_union_geometry_from_arrays(
            self.geometry.material_reference_coordinates,
            self.geometry.material_reference_triangles,
            self.geometry.material_reference_volumes,
            self.geometry.material_bottom_mask[: self.model.n_material],
            self.geometry.material_top_mask[: self.model.n_material],
            0,
            self.geometry.material_reference_edges,
            self.geometry.delta,
            0.1,
            0.0,
        )
        other_model = engineering.build_engineering_grip_model(
            other_geometry, 2.0, 0.3, 1.0e-8, 1.0e-12, 1.7, 0.65
        )
        other_adapter = JaxA2CPUReplayAdapter.from_static(
            other_model.static, other_model.constraints, self.device
        )
        with self.assertRaisesRegex(ValueError, "static identity"):
            engineering.run_zero_damage_work_check(
                self.model,
                config.work_check_load,
                config.work_epsilon,
                config.work_relative_limit,
                config.cg_rtol,
                config.cg_atol,
                config.cg_maxiter,
                mechanics_adapter=other_adapter,
            )
        with self.assertRaisesRegex(ValueError, "static identity"):
            engineering.run_strict_engineering_grip(
                self.model, config, mechanics_adapter=other_adapter
            )

    def test_work_check_uses_the_adapter_and_stays_centred(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        cpu = engineering.run_zero_damage_work_check(
            self.model,
            config.work_check_load,
            config.work_epsilon,
            config.work_relative_limit,
            config.cg_rtol,
            config.cg_atol,
            config.cg_maxiter,
        )
        gpu = engineering.run_zero_damage_work_check(
            self.model,
            config.work_check_load,
            config.work_epsilon,
            config.work_relative_limit,
            config.cg_rtol,
            config.cg_atol,
            config.cg_maxiter,
            mechanics_adapter=self.adapter,
        )
        self.assertTrue(gpu.passed)
        self.assertLessEqual(gpu.relative_error, config.work_relative_limit)
        self.assertAlmostEqual(
            gpu.reaction, cpu.reaction, delta=1.0e-12 * max(1.0, abs(cpu.reaction))
        )
        self.assertAlmostEqual(
            gpu.reaction_fd, cpu.reaction_fd, delta=1.0e-12 * max(1.0, abs(cpu.reaction_fd))
        )
        for name in ("center_result", "plus_result", "minus_result"):
            self.assertEqual(getattr(gpu, name).cg_info, 0)

    def test_strict_tiny_continuation_parity_and_archive_replay(self) -> None:
        config = engineering.EngineeringGripConfig.smoke()
        with mock.patch.object(engineering, "_localization_pass", return_value=True):
            cpu_result = engineering.run_strict_engineering_grip(self.model, config)
            gpu_result = engineering.run_strict_engineering_grip(
                self.model, config, mechanics_adapter=self.adapter
            )
        self.assertEqual(len(cpu_result.states), len(gpu_result.states))
        self.assertEqual(len(gpu_result.accepted_steps), len(cpu_result.accepted_steps))
        self.assertEqual(gpu_result.final_state.u_bar, config.load_final)
        cpu_state, gpu_state = cpu_result.final_state, gpu_result.final_state
        np.testing.assert_allclose(
            gpu_state.union_displacement,
            cpu_state.union_displacement,
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            gpu_state.material_damage, cpu_state.material_damage, rtol=0.0, atol=1.0e-14
        )
        np.testing.assert_allclose(
            gpu_state.material_history, cpu_state.material_history, rtol=0.0, atol=1.0e-14
        )
        self.assertEqual(gpu_state.digest, gpu_state.digest)
        for step in gpu_result.accepted_steps:
            self.assertTrue(all(step.diagnostics.gate_passes.values()))
            self.assertEqual(step.mechanics_before.cg_info, 0)
            self.assertEqual(step.mechanics_after.cg_info, 0)
        self.assertAlmostEqual(
            gpu_result.work_check.reaction,
            cpu_result.work_check.reaction,
            delta=1.0e-12 * max(1.0, abs(cpu_result.work_check.reaction)),
        )
        self.assertEqual(
            engineering.ARCHIVE_SCHEMA, "fracture2d-engineering-grip-v2"
        )
        with tempfile.TemporaryDirectory(prefix="engineering-grip-jax-") as directory:
            path = Path(directory) / "gpu-continuation.npz"
            with mock.patch.object(
                engineering, "_localization_pass", return_value=True
            ):
                # The manufactured full-square cloud is not a localized notch
                # solution; the write-time validation shares the same bypass
                # as the accepted fixture in the CPU engineering suite.
                engineering.write_engineering_grip_archive(path, gpu_result)
                replay = engineering.replay_engineering_grip_archive(path)
            self.assertTrue(replay["replayed"])

    def test_one_fixed_shape_compile_trace_for_changing_inputs(self) -> None:
        backend = JaxA2UnionMechanics(
            self.model.static, self.model.constraints, self.device
        )
        view = self.model.fixed_damage_view(self.damage)
        before = backend.trace_count
        backend.solve(self.zero_damage, 0.06, 1.0e-10, 1.0e-13, 3000)
        first = backend.trace_count
        self.assertEqual(first, before + 1)
        backend.solve(self.damage, 0.08, 1.0e-9, 1.0e-12, 2500)
        backend.solve(
            self.damage,
            0.09,
            1.0e-10,
            1.0e-13,
            3000,
            initial_guess=self.model.constraints.prescribed_vector(0.09),
        )
        device_damage = jax.device_put(self.damage, self.device)
        device_result = backend.solve(
            device_damage, 0.1, 1.0e-10, 1.0e-13, 3000
        )
        self.assertEqual(backend.trace_count, first)
        self.assertEqual(device_result.displacement.devices(), {self.device})

    def test_pcg_loop_performs_no_host_transfer(self) -> None:
        source = inspect.getsource(JaxA2UnionMechanics._pcg)
        self.assertNotIn("device_get", source)
        self.assertNotIn("device_put", source)
        solve_source = inspect.getsource(JaxA2UnionMechanics._traced_solve)
        self.assertNotIn("device_get", solve_source)
        self.assertNotIn("device_put", solve_source)

    def test_new_module_is_in_the_source_bundle_hash(self) -> None:
        files = list(engineering._SOURCE_BUNDLE_FILES)
        self.assertIn(
            "examples/fracture2d/nonlocal_engineering_grip_jax.py", files
        )
        current = engineering._current_source_bundle_hash()
        self.assertEqual(len(current), 64)
        try:
            int(current, 16)
        except ValueError as error:
            raise AssertionError("source bundle hash is not hexadecimal") from error

    @unittest.skipUnless(
        any(device.platform == "gpu" for device in jax.devices()),
        "GPU residency assertion requires a visible GPU",
    )
    def test_gpu_arrays_are_committed_to_the_explicit_gpu(self) -> None:
        gpu = next(device for device in jax.devices() if device.platform == "gpu")
        backend = JaxA2UnionMechanics(
            self.model.static, self.model.constraints, gpu
        )
        self.assertEqual(backend.source.devices(), {gpu})
        self.assertEqual(backend.occurrence_xi.devices(), {gpu})
        result = backend.solve(self.zero_damage, 0.05, 1.0e-10, 1.0e-13, 3000)
        self.assertEqual(result.displacement.devices(), {gpu})
        self.assertEqual(result.free_residual.devices(), {gpu})
        self.assertTrue(result.converged)
        self.assertEqual(int(np.asarray(result.cg_info)), 0)


if __name__ == "__main__":
    unittest.main()
