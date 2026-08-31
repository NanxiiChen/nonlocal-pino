"""Unit tests for condensed fracture mechanics and damage loss."""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture1d import train
from examples.fracture1d.fixed_history_diagnostics import (
    _build_fixed_history_defect_factor,
    _lm_refine_full_network,
    _solve_output_head_least_squares,
)
from examples.fracture1d.configs.train_base import get_config
from examples.fracture1d.losses import (
    _net_d,
    compute_mechanics,
    damage_energy_values,
    damage_pseudo_time_residual,
    damage_residual_scale,
    loss_damage,
    make_loss_fn,
    project_cell_history_to_nodes,
    update_cell_history,
    update_history,
    validate_damage_inputs,
    validate_mechanics_inputs,
)
from pinn.operators import SparseD


class _ConstantRawNet(eqx.Module):
    bias: jnp.ndarray

    def __call__(self, x, t):
        del x, t
        return self.bias


class FractureLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = jnp.linspace(0.0, 1.0, 41, dtype=jnp.float32)[:, None]
        cls.h = 1.0 / 40.0
        cls.weights = jnp.full((41,), cls.h, dtype=jnp.float32)
        cls.weights = cls.weights.at[0].set(0.5 * cls.h)
        cls.weights = cls.weights.at[-1].set(0.5 * cls.h)
        cls.e = jnp.ones((41,), dtype=jnp.float32)
        cls.D = SparseD(cls.x, 0.1, kernel="cubic")

    def test_gate_mode_is_explicit_and_validated(self):
        config = get_config()
        train._validate_config(config)
        self.assertEqual(config.arch.hidden_dim, 200)
        self.assertEqual(config.arch.activation, "swish")
        self.assertEqual(config.arch.spatial_scale, 16.0)
        self.assertTrue(config.arch.use_local_hat_enrichment)
        self.assertEqual(config.arch.local_hat_count, 33)
        self.assertEqual(config.train.damage_iterations, 80000)
        self.assertEqual(config.train.dtype, "float32")
        self.assertEqual(config.loss.residual_norm, "mse")
        self.assertEqual(config.train.atol_damage_change, 2.0e-4)
        self.assertEqual(config.train.atol_damage_residual_scaled, 2.0e-5)
        self.assertEqual(config.train.atol_damage_residual_scaled_inf, 2.0e-4)
        self.assertEqual(config.train.atol_history, 5.0e-1)
        self.assertEqual(config.train.atol_mechanics, 1.0e-4)
        self.assertEqual(config.train.gate_mode, "warning")
        config.train.gate_mode = "strict"
        train._validate_config(config)
        config.train.gate_mode = "invalid"
        with self.assertRaises(ValueError):
            train._validate_config(config)

    def test_stagger_gates_use_configured_scaled_residual(self):
        config = get_config()
        values = dict(
            err_damage=1.0e-7,
            err_damage_residual=1.0,
            err_damage_residual_inf=1.0,
            err_damage_residual_scaled=1.0e-7,
            err_damage_residual_scaled_inf=1.0e-5,
            err_sigma=1.0e-7,
            err_history=1.0e-7,
            err_mechanics=1.0e-7,
        )
        self.assertTrue(train._stagger_gates_pass(config, **values))
        values["err_damage_residual_scaled_inf"] = 3.0e-4
        self.assertFalse(train._stagger_gates_pass(config, **values))

    def test_mechanics_reconstructs_exact_endpoints_and_constant_stress(self):
        d = jnp.full((41,), 0.2, dtype=jnp.float32)
        result = compute_mechanics(d, self.x, self.e, 0.01, self.weights, 1e-8)
        sigma, strain, displacement, psi_plus, compliance = result
        np.testing.assert_allclose(float(displacement[0]), 0.0, atol=1e-7)
        np.testing.assert_allclose(float(displacement[-1]), 0.01, atol=1e-7)
        np.testing.assert_allclose(
            np.asarray(strain),
            np.full(41, float(sigma) / ((1.0 - 0.2) ** 2 + 1e-8)),
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(float(compliance), 1.0 / ((1.0 - 0.2) ** 2 + 1e-8),
                                   rtol=2e-6)
        self.assertTrue(np.all(np.asarray(psi_plus) >= 0.0))

    def test_mechanics_reconstructs_interior_displacement(self):
        d = jnp.full((41,), 0.2, dtype=jnp.float32)
        _, _, displacement, _, _ = compute_mechanics(
            d, self.x, self.e, 0.01, self.weights, 1e-8
        )
        np.testing.assert_allclose(
            np.asarray(displacement),
            0.01 * np.asarray(self.x[:, 0]),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_compressive_strain_does_not_create_tensile_history(self):
        d = jnp.zeros((41,), dtype=jnp.float32)
        sigma, strain, displacement, psi_plus, _ = compute_mechanics(
            d, self.x, self.e, -0.01, self.weights, 1e-8
        )
        self.assertTrue(np.all(np.asarray(strain) < 0.0))
        np.testing.assert_allclose(psi_plus, 0.0, atol=0.0)

    def test_history_is_nonnegative_and_irreversible(self):
        strain = jnp.asarray([-2.0, -0.5, 0.0, 1.0, 2.0], dtype=jnp.float32)
        modulus = jnp.ones_like(strain)
        previous = jnp.asarray([0.3, 0.0, 0.1, 0.2, 2.1], dtype=jnp.float32)
        history = update_history(modulus, strain, previous)
        np.testing.assert_allclose(
            history,
            jnp.asarray([0.3, 0.0, 0.1, 0.5, 2.1], dtype=jnp.float32),
            atol=1e-7,
        )
        self.assertTrue(np.all(np.asarray(history) >= np.asarray(previous)))

    def test_cell_history_matches_dg0_update_and_projection(self):
        displacement = self.x[:, 0] ** 2
        e_cell = jnp.ones((len(self.x) - 1,), dtype=jnp.float32)
        previous = jnp.zeros_like(e_cell)
        history_cell = update_cell_history(
            displacement, self.x, e_cell, previous
        )
        expected_strain = np.diff(np.asarray(displacement)) / np.diff(
            np.asarray(self.x[:, 0])
        )
        np.testing.assert_allclose(
            history_cell, 0.5 * expected_strain**2, rtol=2e-6, atol=2e-6
        )
        projected = np.asarray(project_cell_history_to_nodes(history_cell))
        expected_projected = np.empty(len(self.x), dtype=np.float32)
        expected_projected[0] = np.asarray(history_cell)[0]
        expected_projected[-1] = np.asarray(history_cell)[-1]
        expected_projected[1:-1] = 0.5 * (
            np.asarray(history_cell)[:-1] + np.asarray(history_cell)[1:]
        )
        np.testing.assert_allclose(projected, expected_projected)

    def test_cell_history_mechanics_uses_cell_space_for_outer_error(self):
        mechanics_fn = train._make_mechanics_fn(
            self.x,
            self.e,
            jnp.ones((40,), dtype=jnp.float32),
            self.weights,
            1.0e-8,
            "cell_mass_lumped",
        )
        d_before = jnp.linspace(0.1, 0.3, 41, dtype=jnp.float32)
        d_next = d_before.at[20].add(1.0e-3)
        H_prev_cell = jnp.zeros((40,), dtype=jnp.float32)
        mechanics_before = mechanics_fn(d_before, 0.01, H_prev_cell)
        mechanics_next = mechanics_fn(d_next, 0.01, H_prev_cell)
        self.assertEqual(mechanics_before[5].shape, (41,))
        self.assertEqual(mechanics_before[6].shape, (40,))
        convergence = train._make_convergence_check()(
            d_next, d_before, mechanics_next, mechanics_before
        )
        expected_history_error = jnp.max(
            jnp.abs(mechanics_next[6] - mechanics_before[6])
        )
        np.testing.assert_allclose(convergence[2], expected_history_error)

    def test_projected_cell_max_is_not_replaced_by_nodal_max(self):
        previous_cell = jnp.asarray([0.0, 2.0, 0.0], dtype=jnp.float32)
        current_cell = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
        correct = project_cell_history_to_nodes(
            jnp.maximum(previous_cell, current_cell)
        )
        incorrect = jnp.maximum(
            project_cell_history_to_nodes(previous_cell),
            project_cell_history_to_nodes(current_cell),
        )
        self.assertFalse(np.allclose(np.asarray(correct), np.asarray(incorrect)))

    def test_compliance_has_finite_gradients(self):
        d = jnp.linspace(0.01, 0.4, 41, dtype=jnp.float32)

        def objective(values):
            return compute_mechanics(
                values, self.x, self.e, 0.01, self.weights, 1e-8
            )[0]

        gradient = jax.grad(objective)(d)
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))

    def test_network_output_is_unconstrained(self):
        net = _ConstantRawNet(jnp.array([1.2], dtype=jnp.float32))
        damage = _net_d(net, self.x)
        np.testing.assert_allclose(damage, 1.2, atol=1e-7)

    def test_exp_square_output_is_bounded_and_initial_bias_is_inverted(self):
        raw = _ConstantRawNet(jnp.array([2.0], dtype=jnp.float32))
        net = train._ExpSquaredDamageNet(base=raw)
        damage = train._net_d(net, self.x)
        np.testing.assert_allclose(damage, np.exp(-4.0), rtol=1e-6)
        self.assertTrue(np.all(np.asarray(damage) > 0.0))
        self.assertTrue(np.all(np.asarray(damage) <= 1.0))
        raw_bias = train._damage_raw_bias(1.0e-6, "exp_square")
        self.assertAlmostEqual(np.exp(-(raw_bias**2)), 1.0e-6, places=12)

    def test_hat_enrichment_is_local_and_trainable(self):
        base = _ConstantRawNet(jnp.array([0.2], dtype=jnp.float32))
        net = train._HatEnrichedDamageNet(
            base=base,
            coefficients=jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32),
            center_index=20,
            count=3,
            grid_spacing=1.0 / 40.0,
        )
        damage = np.asarray(_net_d(net, self.x))
        self.assertAlmostEqual(float(damage[20]), 0.0, places=6)
        self.assertAlmostEqual(float(damage[19]), 0.3, places=6)
        self.assertAlmostEqual(float(damage[21]), 0.5, places=6)
        self.assertAlmostEqual(float(damage[0]), 0.2, places=6)
        gradient = jax.grad(
            lambda coefficients: jnp.mean(
                _net_d(
                    eqx.tree_at(
                        lambda n: n.coefficients,
                        net,
                        coefficients,
                    ),
                    self.x,
                )
            )
        )(net.coefficients)
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))

    def test_hat_coefficients_can_be_replaced_without_changing_static_patch(self):
        net = train._HatEnrichedDamageNet(
            base=_ConstantRawNet(jnp.array([0.2], dtype=jnp.float32)),
            coefficients=jnp.zeros((3,), dtype=jnp.float32),
            center_index=20,
            count=3,
            grid_spacing=1.0 / 40.0,
        )
        replacement = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)
        updated = train._set_hat_coefficients(net, replacement)
        np.testing.assert_allclose(train._hat_coefficients(updated), replacement)
        self.assertEqual(updated.center_index, net.center_index)
        self.assertEqual(updated.count, net.count)

    def test_output_only_gradient_preserves_only_mlp_output(self):
        config = get_config()
        config.arch.hidden_dim = 8
        config.arch.spatial_emb_dim = 4
        config.arch.num_layers = 2
        config.sample.Nx = 41
        config.baseline.mesh_nx = 41
        config.arch.use_spatial_emb = True
        config.arch.train_spatial_embedding = True
        config.arch.use_local_hat_enrichment = False
        config.train.dtype = "float32"
        net = train._build_network(config, jax.random.PRNGKey(0))
        gradient = jax.tree.map(
            lambda leaf: jnp.ones_like(leaf) if eqx.is_array(leaf) else leaf,
            net,
        )
        output_gradient = train._zero_non_output_gradient(gradient)
        self.assertTrue(np.all(np.asarray(output_gradient.output.weight) == 1.0))
        self.assertTrue(np.all(np.asarray(output_gradient.output.bias) == 1.0))
        for layer in output_gradient.layers:
            self.assertTrue(np.all(np.asarray(layer.weight) == 0.0))
            self.assertTrue(np.all(np.asarray(layer.bias) == 0.0))
        self.assertTrue(
            np.all(np.asarray(output_gradient.spatial_emb.kernel) == 0.0)
        )

    def test_output_head_least_squares_reduces_scaled_residual(self):
        config = get_config()
        config.arch.hidden_dim = 8
        config.arch.spatial_emb_dim = 4
        config.arch.num_layers = 2
        config.sample.Nx = 41
        config.baseline.mesh_nx = 41
        config.arch.use_local_hat_enrichment = False
        config.train.dtype = "float32"
        net = train._build_network(config, jax.random.PRNGKey(0))
        H = jnp.linspace(0.0, 2.0e-4, len(self.x), dtype=jnp.float32)
        N_l = 0.02
        N_G = 1.0e-5
        before = np.asarray(
            train.damage_residual(net, self.x, self.D, N_l, N_G, H)
            / train.damage_residual_scale(self.D, N_l, N_G, H)
        )
        layers_before = jax.tree.map(np.asarray, net.layers)

        updated, rank, condition, singular_values = (
            _solve_output_head_least_squares(
                net, self.x, self.D, N_l, N_G, H
            )
        )

        after = np.asarray(
            train.damage_residual(updated, self.x, self.D, N_l, N_G, H)
            / train.damage_residual_scale(self.D, N_l, N_G, H)
        )
        self.assertLessEqual(np.mean(after**2), np.mean(before**2))
        self.assertEqual(rank, updated.output.weight.size + updated.output.bias.size)
        self.assertTrue(np.isfinite(condition))
        self.assertEqual(len(singular_values), rank)
        for before_layer, after_layer in zip(layers_before, updated.layers):
            np.testing.assert_array_equal(before_layer.weight, after_layer.weight)
            np.testing.assert_array_equal(before_layer.bias, after_layer.bias)

    def test_full_network_lm_refinement_is_damped_and_target_free(self):
        config = get_config()
        config.arch.hidden_dim = 8
        config.arch.spatial_emb_dim = 4
        config.arch.num_layers = 2
        config.sample.Nx = 41
        config.baseline.mesh_nx = 41
        config.nl.delta = 0.1
        config.arch.use_local_hat_enrichment = False
        config.train.dtype = "float32"
        net = train._build_network(config, jax.random.PRNGKey(0))
        H = jnp.linspace(0.0, 2.0e-4, len(self.x), dtype=jnp.float32)
        updated, diagnostics = _lm_refine_full_network(
            net,
            self.x,
            self.D,
            0.02,
            1.0e-5,
            H,
            "jacobi",
            "scaled_residual",
            None,
            2,
            1.0e-2,
            10.0,
            0.5,
            0.1,
            0.25,
            0.75,
            1.0e-4,
            32,
            100.0,
            2,
        )
        losses = diagnostics["loss_history"]
        self.assertTrue(np.all(np.isfinite(losses)))
        self.assertTrue(np.all(np.diff(losses) <= 1.0e-6))
        self.assertGreater(diagnostics["accepted_steps"], 0)
        layer_change = sum(
            np.linalg.norm(np.asarray(after.weight - before.weight))
            + np.linalg.norm(np.asarray(after.bias - before.bias))
            for before, after in zip(net.layers, updated.layers)
        )
        embedding_change = np.linalg.norm(
            np.asarray(updated.spatial_emb.kernel - net.spatial_emb.kernel)
        )
        self.assertGreater(layer_change + embedding_change, 0.0)

    def test_defect_lm_uses_the_fixed_history_system_factor(self):
        config = get_config()
        config.arch.hidden_dim = 8
        config.arch.spatial_emb_dim = 4
        config.arch.num_layers = 2
        config.sample.Nx = 41
        config.baseline.mesh_nx = 41
        config.nl.delta = 0.1
        config.train.dtype = "float32"
        net = train._build_network(config, jax.random.PRNGKey(2))
        H = jnp.linspace(0.0, 2.0e-4, len(self.x), dtype=jnp.float32)
        factor = _build_fixed_history_defect_factor(
            self.D.to_scipy(), 0.02, 1.0e-5, H
        )
        updated, diagnostics = _lm_refine_full_network(
            net,
            self.x,
            self.D,
            0.02,
            1.0e-5,
            H,
            "jacobi",
            "defect",
            factor,
            1,
            1.0e-2,
            10.0,
            0.5,
            0.1,
            0.25,
            0.75,
            1.0e-4,
            32,
            100.0,
            2,
        )
        self.assertTrue(np.all(np.isfinite(diagnostics["loss_history"])))
        self.assertIsNotNone(updated)

    def test_rprop_optimizer_uses_configured_step_bounds(self):
        config = get_config()
        config.train.optimizer = "rprop"
        optimizer = train._build_optimizer(config)
        net = train._build_network(config, jax.random.PRNGKey(1))
        state = optimizer.init(eqx.filter(net, eqx.is_array))
        self.assertTrue(state is not None)

    def test_hat_enrichment_config_requires_odd_centered_patch(self):
        config = get_config()
        config.arch.use_local_hat_enrichment = True
        config.arch.local_hat_count = 4
        with self.assertRaises(ValueError):
            train._validate_config(config)

    def test_local_equilibrium_context_is_bounded(self):
        H = jnp.asarray([0.0, 1.0e-4, 1.0], dtype=jnp.float32)
        previous = jnp.asarray([0.0, 0.2, 0.4], dtype=jnp.float32)
        context = train._damage_context(H, previous, 0.024, 2.7 / 210.0e3)
        beta = 2.0 * 0.024 / (2.7 / 210.0e3)
        expected_local = beta * H / (1.0 + beta * H)
        np.testing.assert_allclose(context[:, 0], expected_local, rtol=2e-6)
        np.testing.assert_allclose(context[:, 1], previous, rtol=2e-6)
        self.assertTrue(np.all(np.asarray(context[:, 0]) >= 0.0))
        self.assertTrue(np.all(np.asarray(context[:, 0]) < 1.0))

    def test_damage_residual_is_zero_for_matching_constant_solution(self):
        N_l = 0.02
        N_G = 1.0e-5
        d_value = 0.25
        history = d_value * N_G / (2.0 * N_l * (1.0 - d_value))
        net = _ConstantRawNet(
            jnp.array([d_value], dtype=jnp.float32)
        )
        H = jnp.full((41,), history, dtype=jnp.float32)
        value = loss_damage(
            net,
            self.x,
            self.D,
            N_l,
            N_G,
            H,
            "residual",
            "mae",
            "none",
            4,
            0.0,
            "dynamic",
            4.0,
            None,
        )
        self.assertLess(float(value), 2e-5)

    def test_damage_loss_supports_mean_squared_residuals(self):
        N_l = 0.02
        N_G = 1.0e-5
        d_value = 0.25
        history = d_value * N_G / (2.0 * N_l * (1.0 - d_value))
        net = _ConstantRawNet(
            jnp.array([d_value], dtype=jnp.float32)
        )
        H = jnp.full((41,), history, dtype=jnp.float32)
        value = loss_damage(
            net,
            self.x,
            self.D,
            N_l,
            N_G,
            H,
            "residual",
            "mse",
            "none",
            4,
            0.0,
            "dynamic",
            4.0,
            None,
        )
        self.assertLess(float(value), 2e-8)

    def test_hard_point_weight_adds_dynamic_top_k_residual_loss(self):
        net = _ConstantRawNet(jnp.array([0.2], dtype=jnp.float32))
        H = jnp.linspace(0.0, 0.03, len(self.x), dtype=jnp.float32)
        residual = (
            jnp.full((len(self.x),), 0.2, dtype=jnp.float32)
            - (2.0 * 0.02 / 1.0e-5) * 0.8 * H
        )
        pointwise = residual**2
        expected = jnp.mean(pointwise) + 3.0 * jnp.mean(
            jax.lax.top_k(pointwise, 5)[0]
        )
        value = loss_damage(
            net,
            self.x,
            self.D,
            0.02,
            1.0e-5,
            H,
            "residual",
            "mse",
            "none",
            5,
            3.0,
            "dynamic",
            4.0,
            None,
        )
        np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-6)

    def test_smooth_hard_point_weight_is_finite_and_differentiable(self):
        net = _ConstantRawNet(jnp.array([0.2], dtype=jnp.float32))
        H = jnp.linspace(0.0, 0.03, len(self.x), dtype=jnp.float32)
        value = loss_damage(
            net,
            self.x,
            self.D,
            0.02,
            1.0e-5,
            H,
            "residual",
            "mse",
            "none",
            5,
            0.5,
            "smooth",
            4.0,
            None,
        )
        self.assertTrue(np.isfinite(float(value)))

    def test_jacobi_scaling_uses_damage_system_diagonal(self):
        H = jnp.linspace(0.0, 0.03, len(self.x), dtype=jnp.float32)
        scale = damage_residual_scale(self.D, 0.02, 1.0e-5, H)
        expected = 1.0 + 0.02**2 * self.D.diagonal + 4000.0 * H
        np.testing.assert_allclose(scale, expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(np.all(np.asarray(scale) > 0.0))

    def test_pseudo_time_residual_adds_output_space_backward_difference(self):
        N_l = 0.02
        N_G = 1.0e-5
        tau = 0.25
        H = jnp.linspace(0.0, 0.03, len(self.x), dtype=jnp.float32)
        previous = jnp.linspace(0.1, 0.2, len(self.x), dtype=jnp.float32)
        net = _ConstantRawNet(jnp.array([0.35], dtype=jnp.float32))
        residual = damage_pseudo_time_residual(
            net,
            previous,
            self.x,
            self.D,
            N_l,
            N_G,
            H,
            tau,
            "none",
        )
        current = jnp.full_like(previous, 0.35)
        fixed_history_residual = (
            current
            + N_l**2 * (self.D @ current)
            - (2.0 * N_l / N_G) * (1.0 - current) * H
        )
        expected = (current - previous) / tau + fixed_history_residual
        np.testing.assert_allclose(residual, expected, rtol=2e-6, atol=2e-6)

    def test_damage_energy_gradient_equals_mean_raw_residual(self):
        d = jnp.linspace(0.1, 0.4, len(self.x), dtype=jnp.float32)
        H = jnp.linspace(0.0, 0.03, len(self.x), dtype=jnp.float32)
        N_l = 0.02
        N_G = 1.0e-5
        gradient = jax.grad(damage_energy_values)(d, self.D, N_l, N_G, H)
        residual = (
            d
            + N_l**2 * (self.D @ d)
            - (2.0 * N_l / N_G) * (1.0 - d) * H
        )
        np.testing.assert_allclose(
            gradient, residual / len(d), rtol=2e-5, atol=2e-5
        )

    def test_loss_factory_requires_history_and_jits_with_fixed_shapes(self):
        loss_fn = make_loss_fn(
            self.x,
            self.D,
            0.02,
            1.0e-5,
            "residual",
            "mae",
            "jacobi",
            4,
            0.0,
            "dynamic",
            4.0,
        )
        net = _ConstantRawNet(jnp.array([0.0], dtype=jnp.float32))
        with self.assertRaises(ValueError):
            loss_fn(net, jnp.ones(1), jnp.ones(1), jnp.array(False),
                    x_points=self.x, H_points=None)
        step = eqx.filter_jit(
            lambda n, H: loss_fn(
                n, jnp.ones(1), jnp.ones(1), jnp.array(False),
                x_points=self.x, H_points=H
            )[0]
        )
        value = step(net, jnp.zeros((41,), dtype=jnp.float32))
        self.assertTrue(np.isfinite(float(value)))

    def test_mechanics_input_validation_checks_trapezoidal_weights(self):
        validate_mechanics_inputs(self.x, self.e, self.weights, 1e-8)
        bad_weights = self.weights.at[1].set(self.weights[1] * 2.0)
        with self.assertRaises(ValueError):
            validate_mechanics_inputs(self.x, self.e, bad_weights, 1e-8)

    def test_damage_input_validation_checks_fixed_history_shape(self):
        validate_damage_inputs(self.x, jnp.zeros((41,), dtype=jnp.float32))
        with self.assertRaises(ValueError):
            validate_damage_inputs(self.x, jnp.zeros((40,), dtype=jnp.float32))


if __name__ == "__main__":
    unittest.main()
