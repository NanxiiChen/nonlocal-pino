"""Full-cloud same-field checks against the adopted CPU direct primitives."""

import unittest
import hashlib
from tempfile import TemporaryDirectory
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.configs.train_base import get_config
from examples.fracture2d.losses import FractureResidual, update_amor_history
from examples.fracture2d.fp32 import FP32A2Backend, build_fp32_model
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.continuation import (
    _canonical, _check_endpoint_extension, _check_fork_source, _source_root_for_state, _load_accepted,
    run as run_continuation,
)
from examples.fracture2d.diagnostic_continuation import _check_warning_endpoint_extension
from examples.fracture2d.fixed_load_sweeps import _experiment_configs
from examples.fracture2d.train import (
    _build_grid, _build_network, _nonnegative_damage, _reference,
    _validate_config, _warning_region,
)
from examples.fracture2d.stagger import (
    _acceptance_gates, _damage_error_metrics, _load_ratio_scale,
    _validate_stagger, _warning_advancement_gates,
)
from examples.fracture2d.stagger_checkpoint import (
    SCHEMA_ACCEPTED, read_checkpoint, write_checkpoint,
)
from examples.fracture2d.plot_pinn_displacement import _metrics as _displacement_metrics
from examples.fracture2d.pinn_paths import physical_pinn_path


class DamageActivationTest(unittest.TestCase):
    def test_projection_is_nonnegative_and_transparent(self):
        raw = np.asarray([-1.0, -1e-3, -1e-6, 0.0, 1e-6, 1e-3, 1.0])
        floor = 1.0e-8
        out = np.asarray(_nonnegative_damage(jnp.asarray(raw), floor))
        self.assertTrue(np.all(out >= 0.0))
        positive = raw >= 1e-3
        np.testing.assert_allclose(out[positive], raw[positive], rtol=1e-6, atol=1e-12)
        self.assertLess(out[0], 1e-15)
        self.assertAlmostEqual(out[3], 0.5 * floor, places=15)

    def test_validate_config_rejects_bad_damage_floor(self):
        for bad in (0.0, -1.0e-8, 1.0, float("nan")):
            config = get_config()
            config.physical.damage_floor = bad
            with self.assertRaises(ValueError):
                _validate_config(config)


class DisplacementPlotMetricsTest(unittest.TestCase):
    def test_material_vector_norm_retains_notch_hotspot(self):
        reference = np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        predicted = reference + np.asarray([[0.0, 0.0], [0.0, 0.3], [0.1, -0.2]])
        coordinates = np.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        metrics = _displacement_metrics(predicted, reference, coordinates)
        errors = predicted - reference
        self.assertAlmostEqual(
            metrics["relative_l2_material"],
            np.linalg.norm(errors) / np.linalg.norm(reference),
        )
        self.assertAlmostEqual(metrics["max_vector_error"], 0.3)
        self.assertEqual(metrics["max_error_coordinate"], [0.5, 0.5])


class FullCloudResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        cls.config = get_config()
        cls.model = _build_grid(cls.config)
        cls.load, cls.previous, _, cls.q, cls.d, cls.H, _, cls.warning = _reference(
            cls.config, cls.model
        )
        device = jax.devices()[0]
        cls.backend = direct.JaxA2UnionMechanics(
            cls.model.static, cls.model.constraints, device
        )
        cls.residual = FractureResidual(cls.model, cls.backend)
        cls.fp32_model = build_fp32_model(cls.model)
        cls.fp32_backend = FP32A2Backend(cls.fp32_model, device)
        cls.fp32_residual = FractureResidual(cls.fp32_model, cls.fp32_backend)

    def test_reference_fields_match_direct_residual_and_history(self):
        model = self.model
        residual = self.residual
        residual.validate_fields(self.q, self.d, self.H, self.previous, self.load)
        np.testing.assert_array_equal(np.asarray(residual.lift(self.q, self.load)), self.q)
        actual = residual.evaluate(self.q, self.d, self.H, self.previous)
        view = model.fixed_damage_view(self.d)
        expected = view.evaluate(self.q)
        history = direct.compute_a2_amor_history(view, self.q, self.previous)
        damage = direct._e_damage_residuals(model, self.d, self.H, self.H)[0]
        np.testing.assert_allclose(
            np.asarray(actual.mechanics), expected.residual[model.constraints.free_dofs],
            rtol=2e-8, atol=2e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual.driver), history.particle_driver,
            rtol=2e-9, atol=2e-9,
        )
        np.testing.assert_allclose(np.asarray(actual.damage), damage, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(
            np.asarray(actual.history), self.H - history.trial_history,
            rtol=2e-9, atol=2e-9,
        )
        np.testing.assert_allclose(actual.reaction, model.constraints.outer_top_selector @ expected.residual,
                                   rtol=2e-9, atol=2e-9)

    def test_deterministic_operator_matvec_matches_and_repeats(self):
        operator = self.model.material_operator
        damage = jnp.asarray(self.d, jnp.float64)
        first = operator.deterministic_matvec(damage)
        repeated = operator.deterministic_matvec(damage)
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(repeated)))
        baseline = operator @ damage
        difference = np.asarray(first - baseline, dtype=np.float64)
        reference = np.asarray(baseline, dtype=np.float64)
        self.assertLess(
            np.linalg.norm(difference) / np.linalg.norm(reference), 1.0e-10
        )

    def test_deterministic_fp64_backend_matches_baseline_and_repeats(self):
        device = jax.devices()[0]
        first = FP64A2Backend(self.model.static, self.model.constraints, device)
        second = FP64A2Backend(self.model.static, self.model.constraints, device)
        damage = jnp.asarray(self.previous, jnp.float64)
        q = jnp.asarray(self.q, jnp.float64)
        residual = first._residual_from_factor(first._edge_factor(damage), q)
        repeated = first._residual_from_factor(first._edge_factor(damage), q)
        fresh = second._residual_from_factor(second._edge_factor(damage), q)
        self.assertTrue(np.array_equal(np.asarray(residual), np.asarray(repeated)))
        self.assertTrue(np.array_equal(np.asarray(residual), np.asarray(fresh)))
        baseline = self.residual.backend._residual_from_factor(
            self.residual.backend._edge_factor(damage), q
        )
        difference = np.asarray(residual - baseline, dtype=np.float64)
        reference = np.asarray(baseline, dtype=np.float64)
        self.assertLess(
            np.linalg.norm(difference) / np.linalg.norm(reference), 1.0e-10
        )

    def test_jitted_lift_composition_matches_eager(self):
        """Regression: the lift scatter compiled to a different field in jit."""
        model = self.fp32_model
        residual = self.fp32_residual
        net = _build_network(self.config, jax.random.PRNGKey(11), out_dim=2)
        net = jax.tree.map(
            lambda leaf: 0.05 * jax.random.normal(
                jax.random.PRNGKey(12), leaf.shape, dtype=leaf.dtype
            ),
            net,
        )
        points = jnp.asarray(model.geometry.coordinates)
        load = jnp.float32(self.load)

        def displacement(network):
            raw = jax.vmap(network)(points)
            base = jnp.stack(
                (jnp.zeros_like(points[:, 1]), load * points[:, 1]), axis=1
            )
            return residual.lift((base + load * raw).reshape(-1), load)

        eager = np.asarray(displacement(net))
        jitted = np.asarray(eqx.filter_jit(displacement)(net))
        np.testing.assert_allclose(jitted, eager, rtol=0.0, atol=1e-6)
        fixed = model.constraints.constrained_dofs
        np.testing.assert_array_equal(
            jitted[fixed],
            model.constraints.prescribed_vector(self.load)[fixed].astype(np.float32),
        )

    def test_lift_and_input_validation(self):
        q = np.zeros_like(self.q)
        lifted = np.asarray(self.residual.lift(q, self.load))
        fixed = self.model.constraints.constrained_dofs
        np.testing.assert_array_equal(
            lifted[fixed], self.model.constraints.prescribed_vector(self.load)[fixed]
        )
        with self.assertRaisesRegex(ValueError, "affine constraints"):
            self.residual.validate_fields(q, self.d, self.H, self.previous, self.load)

    def test_warning_tail_is_reported_without_relabeling(self):
        tail = _warning_region(self.config, self.model)
        self.assertEqual(len(tail), 16)
        self.assertEqual([item["warning"] for item in tail],
                         [*[True] * 9, False, False, True, *[False] * 4])
        self.assertEqual(tail[2]["drift_count"], 2)
        self.assertEqual(tail[-1]["drift_count"], 31)
        warning_config = get_config()
        warning_config.data.state_index = 60
        with self.assertRaisesRegex(ValueError, "pre-warning strict"):
            _reference(warning_config, self.model)

    def test_warning_reference_is_comparison_only_when_explicit(self):
        config = get_config()
        for index, expected_warning in ((60, True), (61, True), (69, False)):
            config.data.state_index = index
            load, _, _, q, d, H, reaction, warning = _reference(
                config, self.model, diagnostic_warning=True
            )
            self.assertAlmostEqual(load, index * 1e-4)
            self.assertEqual(warning, expected_warning)
            self.assertEqual(q.shape, (self.model.n_dofs,))
            self.assertEqual(d.shape, (self.model.n_material,))
            self.assertEqual(H.shape, (self.model.n_material,))
            self.assertTrue(np.isfinite(reaction))

    def test_float32_graph_and_hard_constraints(self):
        model = self.fp32_model
        self.assertEqual(model.geometry.coordinates.dtype, np.float32)
        self.assertEqual(model.material_operator.coordinates.dtype, jnp.float32)
        self.assertEqual(model.candidate_added, 209)
        self.assertEqual(model.candidate_removed, 76)
        self.assertEqual(model.material_added, 76)
        self.assertEqual(model.material_removed, 152)
        self.assertLess(model.max_condition, 21)
        q = np.asarray(self.fp32_residual.lift(
            jnp.zeros(model.n_dofs, dtype=jnp.float32),
            jnp.asarray(self.load, dtype=jnp.float32),
        ))
        fixed = model.constraints.constrained_dofs
        np.testing.assert_array_equal(
            q[fixed], model.constraints.prescribed_vector(self.load)[fixed].astype(np.float32),
        )
        bottom = np.asarray(model.constraints.outer_bottom_mask)
        top = np.asarray(model.constraints.outer_top_mask)
        np.testing.assert_array_equal(
            q.reshape(-1, 2)[bottom, 1], np.zeros(np.count_nonzero(bottom), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            q.reshape(-1, 2)[top, 1],
            np.full(np.count_nonzero(top), np.float32(self.load), dtype=np.float32),
        )
        self.assertTrue(np.all(fixed[fixed < 2 * model.n_material] == 0))

    def test_float32_affine_energy_and_same_field_diagnostics(self):
        model, backend = self.fp32_model, self.fp32_backend
        x = jnp.asarray(model.geometry.coordinates)
        d = jnp.zeros(model.n_material, dtype=jnp.float32)
        factor = backend._edge_factor(d)
        gradient = np.array([[0.001, -0.0002], [0.0004, 0.0008]], dtype=np.float32)
        q = (x @ jnp.asarray(gradient).T).ravel()
        F = np.asarray(backend._deformation_gradient(factor, q))
        # Rounded node values on short notch-edge supports amplify float32
        # subtraction; preserve and bound that observed affine patch defect.
        self.assertLess(float(np.max(np.abs(F - (np.eye(2) + gradient)))), 3e-5)
        energy_gradient = jax.grad(lambda u: backend._energy_from_factor(factor, u))(q)
        residual = backend._residual_from_factor(factor, q)
        np.testing.assert_allclose(energy_gradient, residual, rtol=1e-3, atol=3e-4)

        q32, d32, H32, prev32 = (
            np.asarray(value, dtype=np.float32)
            for value in (self.q, self.d, self.H, self.previous)
        )
        self.fp32_residual.validate_fields(q32, d32, H32, prev32, np.float32(self.load))
        actual = self.fp32_residual.evaluate(q32, d32, H32, prev32)
        for name in ("mechanics", "damage", "history", "reaction"):
            self.assertEqual(getattr(actual, name).dtype, jnp.float32)
        self.assertLess(float(jnp.max(jnp.abs(actual.mechanics))), 0.005)
        self.assertLess(float(jnp.max(jnp.abs(actual.damage))), 1e-6)
        self.assertLess(float(jnp.max(jnp.abs(actual.history))), 1e-5)
        self.assertLess(abs(float(actual.reaction) - 12.674946794273998), 0.15)
        exact = self.fp32_residual.evaluate(q32, d32, None, prev32)
        np.testing.assert_array_equal(np.asarray(exact.history), np.zeros_like(prev32))
        np.testing.assert_array_equal(
            np.asarray(exact.trial_history),
            np.maximum(prev32, np.asarray(exact.driver)),
        )


    def test_accepted_load_identity_and_parent_binding(self):
        config = get_config()
        config.train.dtype = "float64"
        with TemporaryDirectory(dir="/tmp/opencode") as tmp:
            directory = Path(tmp) / "state_0001"
            directory.mkdir()
            config.output.dir = str(directory)
            canonical = _canonical(config)
            (directory / "config.json").write_text(canonical, encoding="utf-8")
            q_net = _build_network(config, jax.random.PRNGKey(0), out_dim=2)
            d_net = _build_network(config, jax.random.PRNGKey(1), out_dim=1)
            q_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), q_net)
            d_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), d_net)
            opt = optax.adam(float(config.train.learning_rate))
            state = (
                q_net, opt.init(eqx.filter(q_net, eqx.is_inexact_array)),
                d_net, opt.init(eqx.filter(d_net, eqx.is_inexact_array)),
                jnp.asarray(self.q), jnp.asarray(self.d), jnp.asarray(self.H),
            )
            identity = {
                "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "cloud_sha256": config.data.cloud_sha256,
                "reference_sha256": config.data.reference_sha256,
                "direct_model_digest": self.model.model_digest,
                "precision": "float64", "state_index": 1,
            }
            write_checkpoint(
                directory / "accepted.ckpt.zip", identity, 1, "complete", 1,
                state, schema=SCHEMA_ACCEPTED,
            )
            restored, digest = _load_accepted(config, self.model, None)
            np.testing.assert_array_equal(np.asarray(restored[6]), self.H)
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "identity"):
                _load_accepted(config, self.model, "a" * 64)
            write_checkpoint(
                directory / "accepted.ckpt.zip", identity, 1, "complete",
                int(config.stagger.mechanics_steps) + 1, state,
                schema=SCHEMA_ACCEPTED,
            )
            with self.assertRaisesRegex(ValueError, "final completed sweep"):
                _load_accepted(config, self.model, None)
            config.data.end_state_index = 60
            with self.assertRaisesRegex(ValueError, "strict prefix"):
                run_continuation(config)


class StaggerSetupTest(unittest.TestCase):
    def test_pinn_path_resolver_never_redirects_reference_inputs(self):
        reference = Path("/root/autodl-tmp/data/fracture2d/engineering_grip/source/material_cloud_d048.npz")
        self.assertEqual(physical_pinn_path(reference), reference)
        scratch = Path("/tmp/opencode/independent-fracture-run")
        self.assertEqual(physical_pinn_path(scratch), scratch)
        archived = Path("/root/autodl-tmp/data/fracture2d/pinn_continuation_damage_rmse_from6_20260929/state_0059")
        self.assertTrue(physical_pinn_path(archived).is_dir())

    def test_fixed_load_sweep_experiment_keeps_parent_and_subproblem_policy(self):
        config = get_config()
        config.train.dtype = "float64"
        config.data.end_state_index = 59
        config.stagger.sweeps = 3
        config.stagger.mechanics_steps = 16000
        config.stagger.damage_steps = 12000
        config.stagger.mechanics_scale_policy = "first_load_times_load_ratio"
        before = _canonical(config)
        parent, experiment = _experiment_configs(
            config, config.output.dir, "/tmp/opencode/fixed_load_test", 12,
        )
        self.assertEqual(_canonical(config), before)
        self.assertEqual(parent.stagger.sweeps, 3)
        self.assertTrue(parent.stagger.accept_on_first_passing_report)
        self.assertEqual(experiment.stagger.sweeps, 12)
        self.assertFalse(experiment.stagger.accept_on_first_passing_report)
        self.assertEqual(experiment.train, parent.train)
        for name in ("mechanics_steps", "damage_steps", "omega", "mechanics_scale_policy"):
            self.assertEqual(experiment.stagger[name], parent.stagger[name])
        with self.assertRaisesRegex(ValueError, "increased sweeps"):
            _experiment_configs(config, config.output.dir, "/tmp/opencode/fixed_load_test", 3)

    def test_warning_advancement_excludes_untrusted_reference_gates(self):
        config = get_config()
        config.stagger.damage_rmse_limit = 0.01
        trial = {
            "mechanics_loss": 0.001, "reaction_relative": 0.8,
            "u_relative_l2": 0.8, "d_rmse": 0.8,
            "damage_min": 0.0, "damage_max": 0.5,
            "damage_healing_max": 1e-5, "history_decrease_max": 0.0,
            "drift_count": 0,
        }
        gates = _warning_advancement_gates(config, trial)
        self.assertTrue(all(gates.values()))
        for name in ("reaction", "damage_rmse", "displacement_l2"):
            self.assertNotIn(name, gates)
        trial["drift_count"] = 1
        self.assertFalse(_warning_advancement_gates(config, trial)["drift"])
        trial["drift_count"] = 0
        trial["mechanics_loss"] = 0.02
        self.assertFalse(_warning_advancement_gates(config, trial)["mechanics_loss"])

    def test_warning_endpoint_only_increases_and_keeps_numeric_identity(self):
        config = get_config()
        config.data.end_state_index = 60
        stored = _canonical(config)
        config.data.end_state_index = 61
        self.assertEqual(_check_warning_endpoint_extension(config, stored), 60)
        config.train.learning_rate = 2.0e-4
        with self.assertRaisesRegex(ValueError, "other than the endpoint"):
            _check_warning_endpoint_extension(config, stored)
        config.train.learning_rate = 1.0e-4
        config.data.end_state_index = 59
        with self.assertRaisesRegex(ValueError, "60..75"):
            _check_warning_endpoint_extension(config, stored)

    def test_damage_rmse_gate_replaces_displacement_only_when_configured(self):
        config = get_config()
        config.stagger.gate_displacement_l2 = False
        config.stagger.damage_rmse_limit = 1.0e-2
        _validate_stagger(config)
        trial = {
            "mechanics_loss": 0.0015, "u_relative_l2": 0.0523,
            "reaction_relative": 0.0069, "d_rmse": 0.009,
            "d_max_abs_error": 0.02, "damage_min": 0.0,
            "damage_max": 0.006, "damage_healing_max": 1.0e-7,
            "history_decrease_max": 0.0, "drift_count": 0,
        }
        gates = _acceptance_gates(config, trial)
        self.assertTrue(all(gates.values()))
        self.assertIn("damage_rmse", gates)
        self.assertNotIn("displacement_l2", gates)
        self.assertNotIn("d_max_abs_error", gates)
        trial["d_rmse"] = 0.011
        self.assertFalse(_acceptance_gates(config, trial)["damage_rmse"])
        config.stagger.gate_displacement_l2 = True
        self.assertFalse(_acceptance_gates(config, trial)["displacement_l2"])

    def test_nested_fork_resolves_original_accepted_prefix(self):
        with TemporaryDirectory(dir="/tmp/opencode") as tmp:
            roots = [Path(tmp) / f"root_{i}" for i in range(3)]
            for root in roots:
                root.mkdir()
            source = get_config()
            source.stagger.mechanics_scale_policy = "first_load_initial"
            source.data.end_state_index = 3
            source.output.dir = str(roots[0])
            del source.output["fork_from"]
            del source.output["fork_after_state_index"]
            del source.stagger["gate_displacement_l2"]
            del source.stagger["damage_rmse_limit"]
            (roots[0] / "config.json").write_text(_canonical(source), encoding="utf-8")
            middle = get_config()
            middle.stagger.mechanics_scale_policy = "first_load_times_load_ratio"
            middle.data.end_state_index = 7
            middle.output.dir = str(roots[1])
            middle.output.fork_from = str(roots[0])
            middle.output.fork_after_state_index = 2
            (roots[1] / "config.json").write_text(_canonical(middle), encoding="utf-8")
            target = get_config()
            target.stagger.mechanics_scale_policy = "first_load_times_load_ratio"
            target.stagger.gate_displacement_l2 = False
            target.stagger.damage_rmse_limit = 0.01
            target.data.end_state_index = 8
            target.output.dir = str(roots[2])
            target.output.fork_from = str(roots[1])
            target.output.fork_after_state_index = 6
            self.assertEqual(_source_root_for_state(target, roots[1], 1), roots[0])
            self.assertEqual(_source_root_for_state(target, roots[1], 6), roots[1])
            middle.train.learning_rate = 2e-4
            (roots[1] / "config.json").write_text(_canonical(middle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                _source_root_for_state(target, roots[1], 6)

    def test_damage_absolute_metrics_remain_defined_at_zero_reference(self):
        zero = np.zeros(4)
        actual = np.array([0.0, 0.0, 0.0, 2.0e-4])
        metrics = _damage_error_metrics(actual, zero)
        self.assertAlmostEqual(metrics["d_rmse"], 1.0e-4)
        self.assertAlmostEqual(metrics["d_max_abs_error"], 2.0e-4)
        np.testing.assert_array_equal(
            list(_damage_error_metrics(zero, zero).values()), [0.0, 0.0]
        )
        with self.assertRaisesRegex(ValueError, "material fields"):
            _damage_error_metrics(actual, zero[:2])

    def test_load_ratio_scale_matches_a_proportional_force_residual(self):
        first = 0.051029846897393503
        self.assertEqual(_load_ratio_scale(first, 0.0001, 0.0001), first)
        self.assertAlmostEqual(_load_ratio_scale(first, 0.0003, 0.0001), 3 * first)
        self.assertAlmostEqual(0.01983813141142653 / (3 * 3), 0.0022042368234918367)
        for load in (0.0, -0.0001, float("nan")):
            with self.assertRaisesRegex(ValueError, "load-ratio"):
                _load_ratio_scale(first, load, 0.0001)

    def test_fork_changes_only_explicit_scale_and_output_identity(self):
        original = get_config()
        original.train.dtype = "float64"
        source = get_config()
        source.train.dtype = "float64"
        del source.output["fork_from"]
        del source.output["fork_after_state_index"]
        original.stagger.mechanics_scale_policy = "first_load_times_load_ratio"
        original.output.fork_from = "/tmp/source"
        original.output.fork_after_state_index = 2
        original.output.dir = "/tmp/new"
        original.data.end_state_index = 3
        _check_fork_source(original, source)
        source.train.learning_rate = 2e-4
        with self.assertRaisesRegex(ValueError, "outside"):
            _check_fork_source(original, source)
        source.train.learning_rate = 1e-4
        source.stagger.mechanics_scale_policy = "per_load_warm_start"
        with self.assertRaisesRegex(ValueError, "policies"):
            _check_fork_source(original, source)

    def test_endpoint_extension_preserves_all_other_numerical_settings(self):
        config = get_config()
        config.data.end_state_index = 2
        stored = _canonical(config)
        config.data.end_state_index = 3
        self.assertEqual(_check_endpoint_extension(config, stored), 2)
        config.train.learning_rate = 2.0e-4
        with self.assertRaisesRegex(ValueError, "settings"):
            _check_endpoint_extension(config, stored)
        config.train.learning_rate = 1.0e-4
        config.data.end_state_index = 1
        with self.assertRaisesRegex(ValueError, "may only increase"):
            _check_endpoint_extension(config, stored)

    def test_healing_margin_is_independent_of_damage_range_and_mechanics(self):
        config = get_config()
        self.assertEqual(float(config.stagger.healing_atol), 1.0e-3)
        _validate_stagger(config)
        trial = {
            "mechanics_loss": 0.826,
            "u_relative_l2": 0.0294,
            "reaction_relative": 0.00234,
            "damage_min": 4.1e-8,
            "damage_max": 4.5e-4,
            "damage_healing_max": 5.97e-6,
            "history_decrease_max": 0.0,
            "drift_count": 0,
        }
        gates = _acceptance_gates(config, trial)
        self.assertTrue(gates["healing"])
        self.assertTrue(gates["damage_range"])
        self.assertTrue(gates["history_monotonic"])
        self.assertFalse(gates["mechanics_loss"])
        trial["damage_min"] = -2.0e-8
        gates = _acceptance_gates(config, trial)
        self.assertFalse(gates["damage_range"])
        self.assertTrue(gates["healing"])
        for invalid in (-1.0, 1.0, float("nan")):
            config.stagger.healing_atol = invalid
            with self.assertRaisesRegex(ValueError, "healing_atol"):
                _validate_stagger(config)
        config.stagger.healing_atol = 1.0e-3
        config.stagger.mechanics_scale_policy = "unlabelled"
        with self.assertRaisesRegex(ValueError, "mechanics_scale_policy"):
            _validate_stagger(config)

    def test_committed_history_is_detached_but_current_driver_is_differentiable(self):
        driver = jnp.array([1.0, 3.0], dtype=jnp.float32)
        committed = jnp.array([2.0, 1.0], dtype=jnp.float32)
        np.testing.assert_array_equal(update_amor_history(driver, committed), [2.0, 3.0])
        previous_grad = jax.grad(
            lambda previous: jnp.sum(update_amor_history(driver, previous))
        )(committed)
        current_grad = jax.grad(
            lambda current: jnp.sum(update_amor_history(current, committed))
        )(driver)
        np.testing.assert_array_equal(previous_grad, jnp.zeros_like(committed))
        np.testing.assert_array_equal(current_grad, jnp.array([0.0, 1.0], dtype=jnp.float32))

    def test_separate_output_shapes_and_explicit_stagger_policy(self):
        jax.config.update("jax_enable_x64", True)
        config = get_config()
        _validate_stagger(config)
        key = jax.random.PRNGKey(0)
        q_net = _build_network(config, key, out_dim=2)
        d_net = _build_network(config, jax.random.PRNGKey(1), out_dim=1)
        x = jnp.asarray([0.25, 0.5], dtype=jnp.float32)
        self.assertEqual(q_net(x).shape, (2,))
        self.assertEqual(d_net(x).shape, (1,))
        self.assertEqual(q_net(x).dtype, jnp.float32)
        self.assertEqual(d_net(x).dtype, jnp.float32)
        self.assertIsNot(q_net, d_net)
        config.stagger.omega = 0.6
        with self.assertRaisesRegex(ValueError, "adopted direct Picard"):
            _validate_stagger(config)
        config.stagger.omega = 0.5
        config.output.tensorboard_root = config.output.dir
        with self.assertRaisesRegex(ValueError, "separate from the data run"):
            _validate_stagger(config)
        config.output.tensorboard_root = ""
        with self.assertRaisesRegex(ValueError, "nonempty explicit path"):
            _validate_stagger(config)

    def test_checkpoint_restores_network_optimizer_and_tensorboard_steps(self):
        config = get_config()
        q_net = _build_network(config, jax.random.PRNGKey(2), out_dim=2)
        d_net = _build_network(config, jax.random.PRNGKey(3), out_dim=1)
        opt = optax.adam(1e-4)
        state = (
            q_net, opt.init(eqx.filter(q_net, eqx.is_inexact_array)),
            d_net, opt.init(eqx.filter(d_net, eqx.is_inexact_array)),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((8,), dtype=jnp.float32),
            jnp.ones((3,), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
        )
        identity = {"config_sha256": "a" * 64, "fp32_graph_sha256": "b" * 64}
        with TemporaryDirectory(dir="/tmp/opencode") as tmp:
            checkpoint = Path(tmp) / "latest.ckpt.zip"
            write_checkpoint(checkpoint, identity, 1, "damage", 3, state)
            metadata, restored = read_checkpoint(checkpoint, identity, state)
            self.assertEqual((metadata["sweep"], metadata["phase"], metadata["step"]),
                             (1, "damage", 3))
            np.testing.assert_array_equal(restored[0](jnp.array([0.1, 0.2], dtype=jnp.float32)),
                                          q_net(jnp.array([0.1, 0.2], dtype=jnp.float32)))
            np.testing.assert_array_equal(restored[4], state[4])
            self.assertEqual(restored[1][0].count.dtype, state[1][0].count.dtype)
            with self.assertRaisesRegex(ValueError, "identity"):
                read_checkpoint(checkpoint, {"config_sha256": "c" * 64}, state)
            event_dir = Path(tmp) / "tensorboard"
            writer = SummaryWriter(logdir=str(event_dir))
            writer.add_scalar("loss/mechanics_before", 0.5, 3)
            writer.flush()
            writer.close()
            accumulator = event_accumulator.EventAccumulator(str(event_dir))
            accumulator.Reload()
            self.assertEqual(accumulator.Scalars("loss/mechanics_before")[0].step, 3)


if __name__ == "__main__":
    unittest.main()
