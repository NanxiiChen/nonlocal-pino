"""Incremental nonlocal damage-only PINN for the 1D AT2 benchmark.

Mechanics is condensed analytically on a fixed grid. ``SparseD`` is built once
and is used only in the scalar damage residual.
"""

import json
import logging
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags
from tensorboardX import SummaryWriter

from pinn.arch import make_pinn
from pinn.operators import SparseD
from pinn.optimizers import make_optimizer
from pinn.utils import load_example_losses

_loss_module = load_example_losses(__file__)
_net_d = _loss_module._net_d
compute_mechanics = _loss_module.compute_mechanics
make_loss_fn = _loss_module.make_loss_fn
validate_mechanics_inputs = _loss_module.validate_mechanics_inputs
validate_damage_inputs = _loss_module.validate_damage_inputs
damage_residual = _loss_module.damage_residual
damage_residual_scale = _loss_module.damage_residual_scale
update_cell_history = _loss_module.update_cell_history
project_cell_history_to_nodes = _loss_module.project_cell_history_to_nodes


def _damage_context(H_points, d_previous, N_l, N_G):
    """Build immutable local-equilibrium context for one fixed-history solve."""
    beta_history = (2.0 * N_l / N_G) * H_points
    local_damage = beta_history / (1.0 + beta_history)
    return jnp.stack(
        (
            local_damage,
            jax.lax.stop_gradient(d_previous),
        ),
        axis=1,
    )


log = logging.getLogger(__name__)

_DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "train_base.py")
_CONFIG = _config_flags.DEFINE_config_file("config", _DEFAULT, "Config file")
FLAGS = _absl_flags.FLAGS


class _GaussianDamageNet(eqx.Module):
    """Add one trainable localized basis to the unconstrained damage network."""

    base: eqx.Module
    amplitude: jnp.ndarray
    center: float = eqx.field(static=True)
    width: float = eqx.field(static=True)

    def __call__(self, x, t):
        value = self.base(x, t)
        feature = jnp.exp(-((x[0] - self.center) / self.width) ** 2)
        return value.at[0].add(self.amplitude[0] * feature)


class _HatEnrichedDamageNet(eqx.Module):
    """Add trainable CG1 hat corrections around the localization center."""

    base: eqx.Module
    coefficients: jnp.ndarray
    center_index: int = eqx.field(static=True)
    count: int = eqx.field(static=True)
    grid_spacing: float = eqx.field(static=True)

    def __call__(self, x, t):
        value = self.base(x, t)
        grid_spacing = jnp.asarray(self.grid_spacing, dtype=x.dtype)
        centers = (
            self.center_index
            - self.count // 2
            + jnp.arange(self.count, dtype=x.dtype)
        ) * grid_spacing
        basis = jnp.maximum(1.0 - jnp.abs(x[0] - centers) / grid_spacing, 0.0)
        correction = jnp.dot(self.coefficients, basis)
        return value.at[0].add(correction)


def _hat_coefficients(net):
    """Return the trainable hat coefficients from a linear-output network."""
    if not isinstance(net, _HatEnrichedDamageNet):
        raise ValueError("The network does not expose a top-level hat enrichment")
    return net.coefficients


def _set_hat_coefficients(net, coefficients):
    """Replace hat coefficients without changing the network's static fields."""
    if not isinstance(net, _HatEnrichedDamageNet):
        raise ValueError("The network does not expose a top-level hat enrichment")
    return eqx.tree_at(lambda n: n.coefficients, net, coefficients)


class _ExpSquaredDamageNet(eqx.Module):
    """Map a raw scalar output to ``exp(-raw**2)`` in ``(0, 1]``."""

    base: eqx.Module

    def __call__(self, x, t):
        raw = self.base(x, t)
        return jnp.exp(-(raw**2))


class _LocalEquilibriumDamageNet(eqx.Module):
    """Predict the nonlocal correction around the pointwise local equilibrium."""

    base: eqx.Module

    def __call__(self, x, context):
        local_damage = context[0]
        correction_input = jnp.concatenate((x, context))
        correction = self.base(correction_input, jnp.array(0.0))
        return correction.at[0].add(local_damage)


def _damage_raw_bias(d_init, damage_output):
    """Convert configured initial damage to the network's raw output bias."""
    if damage_output == "linear":
        return float(d_init)
    if damage_output == "exp_square":
        if not 0.0 < d_init < 1.0:
            raise ValueError(
                "arch.d_init must lie strictly in (0, 1) for exp_square output"
            )
        return float(np.sqrt(-np.log(d_init)))
    raise ValueError(f"Unknown arch.damage_output {damage_output!r}")


def _validate_config(config):
    """Validate configuration values before creating arrays or JIT functions."""
    physical = config.physical
    geometry = config.geometry
    imperfection = config.imperfection
    nl = config.nl
    sample = config.sample
    train = config.train
    loss = config.loss

    positive_values = {
        "physical.E": physical.E,
        "physical.Gc": physical.Gc,
        "physical.l0": physical.l0,
        "geometry.L0": geometry.L0,
        "imperfection.width": imperfection.width,
        "nl.delta": nl.delta,
        "train.du": train.du,
        "train.du_min": train.du_min,
        "train.du_max": train.du_max,
        "train.du_softening": train.du_softening,
        "train.u_final": train.u_final,
        "train.max_stagger_iters": train.max_stagger_iters,
        "train.damage_iterations": train.damage_iterations,
        "train.atol_damage_change": train.atol_damage_change,
        "train.atol_damage_residual": train.atol_damage_residual,
        "train.atol_damage_residual_inf": train.atol_damage_residual_inf,
        "train.atol_damage_residual_scaled": train.atol_damage_residual_scaled,
        "train.atol_damage_residual_scaled_inf": train.atol_damage_residual_scaled_inf,
        "train.atol_sigma": train.atol_sigma,
        "train.atol_history": train.atol_history,
        "train.atol_mechanics": train.atol_mechanics,
        "train.atol_stress_constancy": train.atol_stress_constancy,
        "train.atol_healing_diagnostic": train.atol_healing_diagnostic,
        "train.lr_init": train.lr_init,
        "train.lr_decay_every": train.lr_decay_every,
        "train.lr_decay_rate": train.lr_decay_rate,
        "train.lr_min": train.lr_min,
        "train.rprop_initial_step_size": train.rprop_initial_step_size,
        "train.rprop_min_step_size": train.rprop_min_step_size,
        "train.rprop_max_step_size": train.rprop_max_step_size,
        "train.grad_clip": train.grad_clip,
        "train.log_every": train.log_every,
        "train.eval_every": train.eval_every,
        "train.checkpoint_every": train.checkpoint_every,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")

    if not np.isfinite(physical.k_res) or physical.k_res < 0.0:
        raise ValueError(
            f"physical.k_res must be non-negative and finite, got {physical.k_res}"
        )
    if not 0.0 <= imperfection.amplitude < 1.0:
        raise ValueError(
            "imperfection.amplitude must satisfy 0 <= amplitude < 1, "
            f"got {imperfection.amplitude}"
        )
    if not 0.0 <= imperfection.center <= 1.0:
        raise ValueError(
            f"imperfection.center must lie in [0, 1], got {imperfection.center}"
        )
    if int(sample.Nx) != sample.Nx or sample.Nx < 2:
        raise ValueError(f"sample.Nx must be an integer >= 2, got {sample.Nx}")
    if int(config.baseline.mesh_nx) != sample.Nx:
        raise ValueError(
            "baseline.mesh_nx must match sample.Nx for a controlled PINN baseline"
        )
    if config.baseline.history_mode not in ("nodal", "cell_mass_lumped"):
        raise ValueError(
            "baseline.history_mode must be 'nodal' or 'cell_mass_lumped'"
        )
    if train.gate_mode not in ("strict", "warning"):
        raise ValueError("train.gate_mode must be 'strict' or 'warning'")
    if not train.du_min <= train.du_softening <= train.du_max:
        raise ValueError(
            "train.du_softening must lie between train.du_min and train.du_max"
        )
    if train.du_min > train.du or train.du > train.du_max:
        raise ValueError("train.du must lie between train.du_min and train.du_max")
    if not 0.0 < train.cutback_factor < 1.0:
        raise ValueError("train.cutback_factor must lie strictly between zero and one")
    if train.growth_factor < 1.0:
        raise ValueError("train.growth_factor must be at least one")
    if int(train.max_cutbacks) != train.max_cutbacks:
        raise ValueError("train.max_cutbacks must be an integer")
    if train.max_cutbacks < 1:
        raise ValueError("train.max_cutbacks must be positive")
    if int(train.max_stagger_iters) != train.max_stagger_iters:
        raise ValueError("train.max_stagger_iters must be an integer")
    if int(train.damage_iterations) != train.damage_iterations:
        raise ValueError("train.damage_iterations must be an integer")
    if int(train.damage_check_every) != train.damage_check_every:
        raise ValueError("train.damage_check_every must be an integer")
    if train.damage_check_every < 1:
        raise ValueError("train.damage_check_every must be positive")
    if int(train.lr_decay_every) != train.lr_decay_every:
        raise ValueError("train.lr_decay_every must be an integer")
    integer_values = {
        "train.log_every": train.log_every,
        "train.eval_every": train.eval_every,
        "train.checkpoint_every": train.checkpoint_every,
    }
    for name, value in integer_values.items():
        if int(value) != value:
            raise ValueError(f"{name} must be an integer")
    if train.lr_decay_rate >= 1.0:
        raise ValueError("train.lr_decay_rate must be less than one")
    if train.optimizer == "rprop":
        if not 0.0 < train.rprop_eta_minus < 1.0:
            raise ValueError("train.rprop_eta_minus must lie in (0, 1)")
        if train.rprop_eta_plus <= 1.0:
            raise ValueError("train.rprop_eta_plus must be greater than one")
        if not (
            train.rprop_min_step_size
            <= train.rprop_initial_step_size
            <= train.rprop_max_step_size
        ):
            raise ValueError(
                "RPROP step sizes must satisfy min <= initial <= max"
            )
    elif train.optimizer not in ("adam", "soap"):
        raise ValueError(f"Unknown optimizer {train.optimizer!r}")
    if train.dtype not in ("float32", "float64"):
        raise ValueError("train.dtype must be 'float32' or 'float64'")
    if nl.method != "sparse":
        raise ValueError("fracture1d requires nl.method='sparse'")
    if nl.kernel not in ("tophat", "cubic", "quartic", "wendland"):
        raise ValueError(f"Unknown SparseD kernel {nl.kernel!r}")
    if loss.residual_norm not in ("mae", "mse"):
        raise ValueError(
            "fracture1d requires loss.residual_norm to be 'mae' or 'mse'"
        )
    if loss.objective not in ("residual", "energy"):
        raise ValueError("loss.objective must be 'residual' or 'energy'")
    if loss.residual_scaling not in ("none", "jacobi"):
        raise ValueError(
            "fracture1d requires loss.residual_scaling to be 'none' or 'jacobi'"
        )
    if (
        isinstance(loss.hard_point_count, (bool, np.bool_))
        or int(loss.hard_point_count) != loss.hard_point_count
        or not 1 <= loss.hard_point_count <= sample.Nx
    ):
        raise ValueError("loss.hard_point_count must be an integer in [1, sample.Nx]")
    if not np.isfinite(loss.hard_point_weight) or loss.hard_point_weight < 0.0:
        raise ValueError("loss.hard_point_weight must be non-negative and finite")
    if loss.hard_point_mode not in ("dynamic", "fixed", "smooth"):
        raise ValueError(
            "loss.hard_point_mode must be 'dynamic', 'fixed', or 'smooth'"
        )
    if not np.isfinite(loss.hard_point_temperature) or loss.hard_point_temperature <= 0.0:
        raise ValueError("loss.hard_point_temperature must be positive and finite")
    if loss.objective == "energy" and loss.hard_point_weight != 0.0:
        raise ValueError("loss.hard_point_weight must be zero for the energy objective")
    if config.arch.spatial_dim != 1 or config.arch.out_dim != 1:
        raise ValueError("The fracture PINN must have one spatial input and output")
    if not isinstance(config.arch.train_spatial_embedding, (bool, np.bool_)):
        raise ValueError("arch.train_spatial_embedding must be boolean")
    if not isinstance(config.arch.local_equilibrium, (bool, np.bool_)):
        raise ValueError("arch.local_equilibrium must be boolean")
    if config.arch.local_equilibrium and config.arch.damage_output != "linear":
        raise ValueError("arch.local_equilibrium requires linear damage output")
    if config.arch.local_equilibrium and (
        config.arch.use_local_gaussian or config.arch.use_local_hat_enrichment
    ):
        raise ValueError(
            "local_equilibrium cannot be combined with Gaussian or hat enrichment"
        )
    if not np.isfinite(config.arch.d_init):
        raise ValueError("arch.d_init must be finite")
    if config.arch.damage_output not in ("linear", "exp_square"):
        raise ValueError(
            "arch.damage_output must be 'linear' or 'exp_square'"
        )
    if config.arch.damage_output == "exp_square" and not (
        0.0 < config.arch.d_init < 1.0
    ):
        raise ValueError(
            "arch.d_init must lie strictly in (0, 1) for exp_square output"
        )
    if not np.isfinite(config.arch.local_gaussian_width) or (
        config.arch.local_gaussian_width <= 0.0
    ):
        raise ValueError("arch.local_gaussian_width must be positive and finite")
    if not np.isfinite(config.arch.local_gaussian_amplitude):
        raise ValueError("arch.local_gaussian_amplitude must be finite")
    local_hat_count = config.arch.local_hat_count
    if int(local_hat_count) != local_hat_count or local_hat_count < 1:
        raise ValueError("arch.local_hat_count must be a positive integer")
    if int(local_hat_count) % 2 == 0:
        raise ValueError("arch.local_hat_count must be odd")
    if not isinstance(config.arch.use_local_hat_enrichment, (bool, np.bool_)):
        raise ValueError("arch.use_local_hat_enrichment must be boolean")
    if config.arch.use_local_hat_enrichment:
        grid_spacing = 1.0 / (int(sample.Nx) - 1)
        center_index = int(round(float(imperfection.center) / grid_spacing))
        if not np.isclose(
            center_index * grid_spacing,
            float(imperfection.center),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "arch.use_local_hat_enrichment requires imperfection.center "
                "to lie on the fixed grid"
            )
        half_count = int(local_hat_count) // 2
        if center_index - half_count < 0 or center_index + half_count >= int(sample.Nx):
            raise ValueError("arch.local_hat_count patch must lie inside the grid")
    if config.arch.type != "mlp":
        raise ValueError("The first fracture implementation requires arch.type='mlp'")
    if config.arch.activation not in ("tanh", "relu", "gelu", "silu", "swish"):
        raise ValueError(f"Unknown network activation {config.arch.activation!r}")


def _build_grid(config):
    """Build the sorted fixed grid and its nodal trapezoidal weights."""
    n_points = int(config.sample.Nx)
    dtype = jnp.dtype(config.train.dtype)
    x = jnp.linspace(0.0, 1.0, n_points, dtype=dtype)[:, None]
    spacing = 1.0 / (n_points - 1)
    weights = jnp.full((n_points,), spacing, dtype=x.dtype)
    weights = weights.at[0].set(0.5 * spacing)
    weights = weights.at[-1].set(0.5 * spacing)
    return x, weights


def _build_modulus(config, x_points):
    """Build the dimensionless Young's-modulus field e(x_star)."""
    center = config.imperfection.center
    width = config.imperfection.width
    amplitude = config.imperfection.amplitude
    e_field = 1.0 - amplitude * jnp.exp(
        -(x_points[:, 0] - center) ** 2 / (2.0 * width**2)
    )
    if not np.all(np.isfinite(np.asarray(e_field))) or np.any(np.asarray(e_field) <= 0.0):
        raise ValueError("The configured modulus imperfection must give e(x) > 0")
    return e_field


def _build_network(config, key):
    """Build the configured scalar damage network with constant initial bias."""
    arch = config.arch
    network_spatial_dim = 3 if arch.local_equilibrium else arch.spatial_dim
    network_use_spatial_emb = (
        False if arch.local_equilibrium else arch.use_spatial_emb
    )
    net = make_pinn(
        spatial_dim=network_spatial_dim,
        out_dim=arch.out_dim,
        trunk_type=arch.type,
        hidden_dim=arch.hidden_dim,
        num_layers=arch.num_layers,
        activation=arch.activation,
        use_spatial_emb=network_use_spatial_emb,
        use_temporal_emb=arch.use_temporal_emb,
        spatial_emb_dim=arch.spatial_emb_dim,
        spatial_scale=arch.spatial_scale,
        temporal_emb_dim=arch.temporal_emb_dim,
        temporal_scale=arch.temporal_scale,
        static=True,
        tensor_split=arch.tensor_split,
        multi_head=arch.multi_head,
        film_hidden_dim=arch.film_hidden_dim,
        film_num_layers=arch.film_num_layers,
        key=key,
    )
    dtype = jnp.dtype(config.train.dtype)
    net = jax.tree.map(lambda array: array.astype(dtype), net)
    if not hasattr(net, "output") or net.output is None:
        raise ValueError("The damage network must expose one scalar output layer")
    raw_bias = 0.0 if arch.local_equilibrium else _damage_raw_bias(
        config.arch.d_init, config.arch.damage_output
    )
    bias = jnp.full_like(net.output.bias, raw_bias)
    net = eqx.tree_at(lambda n: n.output.bias, net, bias)
    if arch.local_equilibrium:
        zero_weight = jnp.zeros_like(net.output.weight)
        zero_bias = jnp.zeros_like(net.output.bias)
        net = eqx.tree_at(
            lambda n: (n.output.weight, n.output.bias),
            net,
            (zero_weight, zero_bias),
        )
    if config.arch.use_local_gaussian:
        net = _GaussianDamageNet(
            base=net,
            amplitude=jnp.full_like(
                bias, config.arch.local_gaussian_amplitude
            ),
            center=float(config.imperfection.center),
            width=float(config.arch.local_gaussian_width),
        )
    if config.arch.use_local_hat_enrichment:
        grid_spacing = 1.0 / (int(config.sample.Nx) - 1)
        center_index = int(round(float(config.imperfection.center) / grid_spacing))
        net = _HatEnrichedDamageNet(
            base=net,
            coefficients=jnp.zeros(
                (int(config.arch.local_hat_count),), dtype=bias.dtype
            ),
            center_index=center_index,
            count=int(config.arch.local_hat_count),
            grid_spacing=grid_spacing,
        )
    if config.arch.damage_output == "exp_square":
        net = _ExpSquaredDamageNet(base=net)
    if arch.local_equilibrium:
        net = _LocalEquilibriumDamageNet(base=net)
    return net


def _build_D(config, x_all):
    """Build the fixed SparseD graph once outside JIT."""
    return SparseD(x_all, config.nl.delta, kernel=config.nl.kernel)


def _build_optimizer(config):
    return make_optimizer(config)


def _zero_spatial_embedding_gradient(gradient):
    """Remove Fourier-kernel gradients while preserving all trunk gradients."""
    if hasattr(gradient, "spatial_emb") and gradient.spatial_emb is not None:
        return eqx.tree_at(
            lambda n: n.spatial_emb.kernel,
            gradient,
            jnp.zeros_like(gradient.spatial_emb.kernel),
        )
    if hasattr(gradient, "base"):
        base_gradient = _zero_spatial_embedding_gradient(gradient.base)
        return eqx.tree_at(lambda n: n.base, gradient, base_gradient)
    return gradient


def _zero_non_output_gradient(gradient):
    """Keep only the final field-output gradient for staged transfer training."""
    zero = jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf) if eqx.is_array(leaf) else leaf,
        gradient,
    )
    if hasattr(gradient, "base"):
        base_gradient = _zero_non_output_gradient(gradient.base)
        return eqx.tree_at(lambda n: n.base, zero, base_gradient)
    if hasattr(gradient, "output") and gradient.output is not None:
        return eqx.tree_at(lambda n: n.output, zero, gradient.output)
    if hasattr(gradient, "out_heads") and gradient.out_heads is not None:
        return eqx.tree_at(lambda n: n.out_heads, zero, gradient.out_heads)
    if hasattr(gradient, "trunk_output"):
        return eqx.tree_at(
            lambda n: n.trunk_output,
            zero,
            gradient.trunk_output,
        )
    raise ValueError("The network does not expose a supported output layer")


def _make_step(
    loss_fn,
    optim,
    freeze_hat_coefficients=False,
    freeze_spatial_embedding=False,
    output_only=False,
):
    """Create a fixed-shape JIT step with dynamic history values."""
    @eqx.filter_jit
    def step(
        net,
        opt_state,
        w_ema,
        bias,
        update_w,
        x_points,
        H_points,
        hard_point_indices=None,
        d_previous=None,
    ):
        loss, _, gradient, new_w = loss_fn(
            net,
            w_ema,
            bias,
            update_w,
            x_points=x_points,
            H_points=H_points,
            hard_point_indices=hard_point_indices,
            d_previous=d_previous,
        )
        if output_only:
            gradient = _zero_non_output_gradient(gradient)
        if freeze_hat_coefficients:
            gradient = _set_hat_coefficients(
                gradient,
                jnp.zeros_like(_hat_coefficients(gradient)),
            )
        if freeze_spatial_embedding:
            gradient = _zero_spatial_embedding_gradient(gradient)
        updates, opt_state = optim.update(gradient, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, new_w

    return step


def _make_damage_eval(x_points, local_equilibrium=False):
    """Create a fixed-shape JIT damage-network evaluation."""
    @eqx.filter_jit
    def evaluate(net, context_points=None):
        if local_equilibrium:
            if context_points is None:
                raise ValueError("local-equilibrium evaluation requires context")
            return _net_d(net, x_points, context_points)
        return _net_d(net, x_points)

    return evaluate


def _make_mechanics_fn(
    x_points,
    e_field,
    e_cell,
    quadrature_weights,
    k_res,
    history_mode,
):
    """Create condensed mechanics with an explicit history discretization."""
    if history_mode not in ("nodal", "cell_mass_lumped"):
        raise ValueError("history_mode must be 'nodal' or 'cell_mass_lumped'")
    if history_mode == "cell_mass_lumped":
        e_cell_np = np.asarray(e_cell)
        if e_cell_np.shape != (len(x_points) - 1,):
            raise ValueError("e_cell must have one value per grid cell")
        if not np.all(np.isfinite(e_cell_np)) or np.any(e_cell_np <= 0.0):
            raise ValueError("e_cell must be finite and strictly positive")

    @eqx.filter_jit
    def mechanics(d, u_bar, H_prev):
        result = compute_mechanics(
            d,
            x_points,
            e_field,
            u_bar,
            quadrature_weights,
            k_res,
        )
        sigma, strain, displacement, psi_plus, compliance = result
        if history_mode == "cell_mass_lumped":
            history_state = update_cell_history(
                displacement, x_points, e_cell, H_prev
            )
            history_damage = project_cell_history_to_nodes(history_state)
        else:
            history_state = jnp.maximum(H_prev, psi_plus)
            history_damage = history_state
        stress = e_field * ((1.0 - d) ** 2 + k_res) * strain
        finite = (
            jnp.all(jnp.isfinite(sigma))
            & jnp.all(jnp.isfinite(strain))
            & jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(psi_plus))
            & jnp.all(jnp.isfinite(compliance))
            & jnp.all(jnp.isfinite(stress))
        )
        stress_spread = jnp.max(stress) - jnp.min(stress)
        diagnostics = jnp.stack((
            sigma,
            compliance,
            displacement[0],
            displacement[-1],
            stress_spread,
            finite.astype(sigma.dtype),
        ))
        return (*result, history_damage, history_state, diagnostics)

    return mechanics


def _make_residual_check(x_points, D, N_l, N_G, local_equilibrium=False):
    """Create fixed-shape raw and Jacobi-scaled residual diagnostics."""
    @eqx.filter_jit
    def check(net, H_points, d_previous=None):
        context = None
        if local_equilibrium:
            if d_previous is None:
                raise ValueError("local-equilibrium residual check requires context")
            context = _damage_context(H_points, d_previous, N_l, N_G)
        residual = damage_residual(
            net, x_points, D, N_l, N_G, H_points, context
        )
        absolute = jnp.abs(residual)
        scaled_absolute = absolute / damage_residual_scale(
            D, N_l, N_G, H_points
        )
        return (
            jnp.mean(absolute),
            jnp.max(absolute),
            jnp.mean(scaled_absolute),
            jnp.max(scaled_absolute),
        )

    return check


def _make_convergence_check():
    """Create one fused device reduction for staggered convergence metrics."""
    @eqx.filter_jit
    def check(d_next, d_iter, mechanics_next, mechanics_before):
        sigma_next, _, u_next, _, _, _, H_next, _ = mechanics_next
        sigma_before, _, u_before, _, _, _, H_used, _ = mechanics_before
        return jnp.stack((
            jnp.max(jnp.abs(d_next - d_iter)),
            jnp.abs(sigma_next - sigma_before),
            jnp.max(jnp.abs(H_next - H_used)),
            jnp.max(jnp.abs(u_next - u_before)),
        ))

    return check


def _stagger_gates_pass(
    config,
    err_damage,
    err_damage_residual,
    err_damage_residual_inf,
    err_damage_residual_scaled,
    err_damage_residual_scaled_inf,
    err_sigma,
    err_history,
    err_mechanics,
):
    """Return whether the configured fixed-history gates pass."""
    if config.loss.residual_scaling == "jacobi":
        residual_value = err_damage_residual_scaled
        residual_inf_value = err_damage_residual_scaled_inf
        residual_atol = config.train.atol_damage_residual_scaled
        residual_inf_atol = config.train.atol_damage_residual_scaled_inf
    else:
        residual_value = err_damage_residual
        residual_inf_value = err_damage_residual_inf
        residual_atol = config.train.atol_damage_residual
        residual_inf_atol = config.train.atol_damage_residual_inf
    return (
        err_damage <= config.train.atol_damage_change
        and residual_value <= residual_atol
        and residual_inf_value <= residual_inf_atol
        and err_sigma <= config.train.atol_sigma
        and err_history <= config.train.atol_history
        and err_mechanics <= config.train.atol_mechanics
    )


def _check_mechanics_result(mechanics, u_bar, atol_stress_constancy):
    """Fail outside JIT using only compact scalar mechanics diagnostics."""
    scalar_values = np.asarray(jax.device_get(mechanics[-1]))
    sigma_value, compliance_value, displacement_start, displacement_end, spread, finite = (
        scalar_values
    )
    if not bool(finite):
        raise FloatingPointError("Condensed mechanics returned a non-finite value")
    if not np.all(np.isfinite(scalar_values)):
        raise FloatingPointError("Condensed mechanics diagnostics are non-finite")
    if compliance_value <= 0.0:
        raise FloatingPointError(f"Invalid compliance {compliance_value}")
    if not np.isclose(displacement_start, 0.0, rtol=0.0, atol=1e-6):
        raise FloatingPointError("Condensed displacement does not satisfy u[0]=0")
    if not np.isclose(displacement_end, float(u_bar), rtol=0.0, atol=1e-6):
        raise FloatingPointError("Condensed displacement does not satisfy u[-1]=u_bar")
    if spread > atol_stress_constancy:
        raise FloatingPointError(
            f"Condensed reconstructed stress is not constant: spread={spread:.3e}"
        )


def _fracture_energy(d, D, N_l, N_G, particle_volume):
    """Operator-consistent discrete non-dimensional fracture energy."""
    gradient_form = jnp.dot(d, D @ d)
    return (N_G / (2.0 * N_l)) * particle_volume * (
        jnp.sum(d**2) + N_l**2 * gradient_form
    )


def _load_reference(path, config, metadata_config):
    """Load and validate the independent local FEM reference file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FEM reference file does not exist: {path}")
    data = np.load(path)
    required = (
        "x", "x_H", "u_bar", "snap_u_bar", "u", "d", "reaction_star", "H"
    )
    missing = [name for name in required if name not in data.files]
    if missing:
        raise KeyError(f"FEM reference {path} is missing fields: {missing}")
    for name in required:
        if not np.all(np.isfinite(np.asarray(data[name]))):
            raise FloatingPointError(f"FEM reference field {name} is non-finite")
    snapshot_count = len(data["snap_u_bar"])
    if snapshot_count != len(data["d"]) or snapshot_count != len(data["u"]):
        raise ValueError("FEM snapshot coordinate, u, and d counts differ")
    if snapshot_count != len(data["H"]):
        raise ValueError("FEM snapshot coordinate and H counts differ")
    if len(data["u_bar"]) != len(data["reaction_star"]):
        raise ValueError("FEM load and reaction histories differ")
    if np.any(np.diff(np.asarray(data["u_bar"])) <= 0.0):
        raise ValueError("FEM load history must be strictly increasing")
    if np.any(np.diff(np.asarray(data["snap_u_bar"])) <= 0.0):
        raise ValueError("FEM snapshot load coordinates must be strictly increasing")
    x_ref = np.asarray(data["x"])
    if x_ref.ndim != 1 or len(x_ref) < 2 or np.any(np.diff(x_ref) <= 0.0):
        raise ValueError("FEM reference x must be a strictly increasing 1D array")
    x_history = np.asarray(data["x_H"])
    if (
        x_history.ndim != 1
        or len(x_history) < 1
        or np.any(np.diff(x_history) <= 0.0)
    ):
        raise ValueError("FEM reference x_H must be a strictly increasing 1D array")
    for name in ("u", "d"):
        if np.asarray(data[name]).shape[1] != len(x_ref):
            raise ValueError(f"FEM reference field {name} does not match x length")
    if np.asarray(data["H"]).shape[1] != len(x_history):
        raise ValueError("FEM reference field H does not match x_H length")
    if (
        hasattr(metadata_config, "history_mode")
        and metadata_config.history_mode == "cell_mass_lumped"
    ):
        if "H_cell" not in data.files:
            raise KeyError(f"Cell-history reference {path} is missing H_cell")
        H_cell = np.asarray(data["H_cell"])
        if H_cell.shape != (snapshot_count, len(x_ref) - 1):
            raise ValueError("Cell-history reference H_cell shape is invalid")
        if not np.all(np.isfinite(H_cell)):
            raise FloatingPointError("Cell-history reference H_cell is non-finite")
    if float(data["u_bar"][-1]) < float(config.train.u_final) - 1e-12:
        raise ValueError(
            "FEM reference does not cover the configured final displacement: "
            f"{float(data['u_bar'][-1])} < {float(config.train.u_final)}"
        )
    if not np.isclose(float(data["snap_u_bar"][0]), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("FEM snapshots must include the initial zero-load state")
    scalar_metadata = {
        "E0": config.physical.E,
        "Gc": config.physical.Gc,
        "l0": config.physical.l0,
        "L0": config.geometry.L0,
        "k_res": config.physical.k_res,
        "imperfection_amplitude": config.imperfection.amplitude,
        "imperfection_width": config.imperfection.width,
        "imperfection_center": config.imperfection.center,
        "loading_du_initial": config.train.du,
        "loading_du_min": config.train.du_min,
        "loading_du_max": config.train.du_max,
        "loading_du_softening": config.train.du_softening,
        "output_save_every": config.train.du,
    }
    for name, expected in scalar_metadata.items():
        if name not in data.files:
            raise KeyError(f"FEM reference {path} is missing metadata field {name}")
        actual = float(np.asarray(data[name]))
        if not np.isclose(actual, float(expected), rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"FEM reference metadata {name}={actual} does not match "
                f"the configured value {expected}"
            )
    if "mesh_nx" not in data.files:
        raise KeyError(f"FEM reference {path} is missing metadata field mesh_nx")
    if int(np.asarray(data["mesh_nx"])) != int(metadata_config.mesh_nx):
        raise ValueError(
            f"FEM reference mesh_nx={int(np.asarray(data['mesh_nx']))} does not "
            f"match configured mesh_nx={int(metadata_config.mesh_nx)}"
        )
    for name in ("history_space", "history_method"):
        if name not in data.files:
            raise KeyError(f"FEM reference {path} is missing metadata field {name}")
        actual = str(np.asarray(data[name]).item())
        expected = str(getattr(metadata_config, name))
        if actual != expected:
            raise ValueError(
                f"FEM reference {name}={actual!r} does not match configured "
                f"value {expected!r}"
            )
    if hasattr(metadata_config, "baseline_method"):
        if "baseline_method" not in data.files:
            raise KeyError(f"Reference {path} is missing metadata field baseline_method")
        actual = str(np.asarray(data["baseline_method"]).item())
        expected = str(metadata_config.baseline_method)
        if actual != expected:
            raise ValueError(
                f"Reference baseline_method={actual!r} does not match configured "
                f"value {expected!r}"
            )
    if hasattr(metadata_config, "history_mode"):
        if "history_mode" not in data.files:
            raise KeyError(f"Reference {path} is missing metadata field history_mode")
        actual = str(np.asarray(data["history_mode"]).item())
        expected = str(metadata_config.history_mode)
        if actual != expected:
            raise ValueError(
                f"Reference history_mode={actual!r} does not match configured "
                f"value {expected!r}"
            )
    return data


def _prepare_reference_fields(reference, x_eval):
    """Interpolate FEM snapshots to the PINN grid once outside the loop."""
    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_ref = np.asarray(reference["x"])
    x_history = np.asarray(reference["x_H"])
    return {
        "snap_u_bar": np.asarray(reference["snap_u_bar"]),
        "u_bar": np.asarray(reference["u_bar"]),
        "reaction_star": np.asarray(reference["reaction_star"]),
        "d": np.asarray([
            np.interp(x_eval, x_ref, row) for row in np.asarray(reference["d"])
        ]),
        "u": np.asarray([
            np.interp(x_eval, x_ref, row) for row in np.asarray(reference["u"])
        ]),
        "H": np.asarray([
            np.interp(x_eval, x_history, row) for row in np.asarray(reference["H"])
        ]),
    }


def _interpolate_load_field(snap_u_bar, snapshots, u_bar):
    """Linearly interpolate one pre-spatially-interpolated field in load."""
    if u_bar < snap_u_bar[0] - 1e-12 or u_bar > snap_u_bar[-1] + 1e-12:
        raise ValueError(
            f"Load coordinate {u_bar} lies outside FEM snapshot range "
            f"[{snap_u_bar[0]}, {snap_u_bar[-1]}]"
        )
    if np.isclose(u_bar, snap_u_bar[-1], rtol=0.0, atol=1e-12):
        return np.asarray(snapshots[-1])
    index = int(np.searchsorted(snap_u_bar, u_bar, side="right") - 1)
    index = max(0, min(index, len(snap_u_bar) - 2))
    denominator = snap_u_bar[index + 1] - snap_u_bar[index]
    fraction = (u_bar - snap_u_bar[index]) / denominator
    return (
        (1.0 - fraction) * np.asarray(snapshots[index])
        + fraction * np.asarray(snapshots[index + 1])
    )


def _add_training_figures(
    writer,
    event_index,
    x_all,
    u_bar,
    d_pred,
    u_bar_history,
    reaction_history,
    reference,
):
    """Log the current damage field and partial reaction curve to TensorBoard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x_all[:, 0], dtype=np.float64)
    damage = np.asarray(d_pred, dtype=np.float64)
    d_reference = _interpolate_load_field(
        reference["snap_u_bar"], reference["d"], u_bar
    )
    d_error = damage - d_reference
    d_l2_abs = float(np.sqrt(np.mean(d_error**2)))
    d_l2_rel = float(
        d_l2_abs / (np.sqrt(np.mean(d_reference**2)) + 1.0e-12)
    )
    if not np.all(np.isfinite((d_l2_abs, d_l2_rel))):
        raise FloatingPointError("Damage L2 diagnostic became non-finite")

    damage_figure, damage_axis = plt.subplots(figsize=(7.2, 4.4))
    damage_axis.plot(
        x,
        d_reference,
        color="#1f77b4",
        linewidth=1.5,
        label="direct nonlocal reference",
    )
    damage_axis.plot(
        x,
        damage,
        color="#d62728",
        linewidth=1.25,
        linestyle="--",
        label="PINN",
    )
    damage_axis.set_xlabel("x/L0")
    damage_axis.set_ylabel("damage d")
    damage_axis.set_title(
        f"Damage field at u_bar={u_bar:.6e}; "
        f"relative L2={d_l2_rel:.3e}"
    )
    damage_axis.grid(alpha=0.25)
    damage_axis.legend(fontsize=8)
    writer.add_figure(
        "train/damage_field",
        damage_figure,
        global_step=event_index,
        close=True,
    )

    curve_figure, curve_axis = plt.subplots(figsize=(7.2, 4.4))
    curve_axis.plot(
        np.asarray(reference["u_bar"], dtype=np.float64),
        np.asarray(reference["reaction_star"], dtype=np.float64),
        color="#1f77b4",
        linewidth=1.5,
        label="direct nonlocal reference",
    )
    curve_axis.plot(
        np.asarray(u_bar_history, dtype=np.float64),
        np.asarray(reaction_history, dtype=np.float64),
        color="#d62728",
        linewidth=1.35,
        label="PINN through current step",
    )
    curve_axis.axvline(
        float(u_bar), color="black", linestyle=":", linewidth=1.0
    )
    curve_axis.set_xlabel("average strain u_bar/L0")
    curve_axis.set_ylabel("reaction stress")
    curve_axis.set_title(f"Stress-strain curve through u_bar={u_bar:.6e}")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(fontsize=8)
    writer.add_figure(
        "train/stress_strain_curve",
        curve_figure,
        global_step=event_index,
        close=True,
    )
    return d_l2_abs, d_l2_rel


def _eval_reference(
    net,
    x_all,
    e_field,
    u_bar,
    quadrature_weights,
    k_res,
    reference,
    writer,
    step,
    config,
    H_pred_state,
    H_pred_damage,
    label,
    damage_eval,
    mechanics_fn,
    damage_context=None,
):
    """Evaluate condensed PINN fields against load-interpolated FEM fields."""
    d_pred = damage_eval(net, damage_context)
    mechanics = mechanics_fn(
        d_pred,
        jnp.asarray(u_bar, dtype=x_all.dtype),
        H_pred_state,
    )
    sigma, _, displacement, _, _, _, _, _ = mechanics
    _check_mechanics_result(
        mechanics,
        u_bar,
        config.train.atol_stress_constancy,
    )

    snap_u_bar = np.asarray(reference["snap_u_bar"])
    d_ref = _interpolate_load_field(snap_u_bar, reference["d"], u_bar)
    u_ref = _interpolate_load_field(snap_u_bar, reference["u"], u_bar)
    H_ref = _interpolate_load_field(snap_u_bar, reference["H"], u_bar)
    # The reference reaction is stored at every load step, not only snapshots.
    # Interpolate it against u_bar when the full history is available.
    ref_u_history = np.asarray(reference["u_bar"])
    ref_reaction_history = np.asarray(reference["reaction_star"])
    reaction_ref = float(np.interp(u_bar, ref_u_history, ref_reaction_history))

    d_pred_np = np.asarray(d_pred)
    u_pred_np = np.asarray(displacement)
    d_norm = np.sqrt(np.mean(d_ref**2)) + 1.0e-12
    u_norm = np.sqrt(np.mean(u_ref**2)) + 1.0e-12
    H_pred_np = np.asarray(H_pred_damage)
    H_norm = np.sqrt(np.mean(H_ref**2)) + 1.0e-12
    rel_d = float(np.sqrt(np.mean((d_pred_np - d_ref) ** 2)) / d_norm)
    rel_u = float(np.sqrt(np.mean((u_pred_np - u_ref) ** 2)) / u_norm)
    rel_H = float(np.sqrt(np.mean((H_pred_np - H_ref) ** 2)) / H_norm)
    reaction_pred = float(sigma)
    reaction_error = abs(reaction_pred - reaction_ref)
    values = (rel_d, rel_u, rel_H, reaction_error, reaction_pred, reaction_ref)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Reference evaluation produced a non-finite metric")
    writer.add_scalar(f"eval/{label}/rel_L2_d", rel_d, step)
    writer.add_scalar(f"eval/{label}/rel_L2_u", rel_u, step)
    writer.add_scalar(f"eval/{label}/rel_L2_H", rel_H, step)
    writer.add_scalar(f"eval/{label}/reaction_abs_error", reaction_error, step)
    return rel_d, rel_u, rel_H, reaction_error


def _save_results(
    config,
    output_dir,
    x_all,
    u_bar_history,
    du_history,
    stagger_iterations_history,
    cutback_history,
    gate_warning_history,
    softening_history,
    snap_u_bar,
    u_snapshots,
    d_snapshots,
    reaction_history,
    H_snapshots,
    H_state_snapshots,
    energy_history,
    damage_residual_mae_history,
    damage_residual_inf_history,
    damage_residual_scaled_history,
    damage_residual_scaled_inf_history,
    damage_residual_solve_history_mae_history,
    damage_residual_solve_history_inf_history,
    damage_residual_solve_history_scaled_mae_history,
    damage_residual_solve_history_scaled_inf_history,
    damage_increment_min_history,
    damage_optimizer_iterations_history,
    damage_fixed_point_change_history,
    d_l2_u_bar_history,
    d_l2_abs_history,
    d_l2_rel_history,
    metrics,
    N_l,
    N_G,
):
    """Save fixed-grid fields and explicit non-dimensional metadata."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, config.output.filename)
    np.savez_compressed(
        path,
        x=np.asarray(x_all[:, 0]),
        u_bar=np.asarray(u_bar_history),
        du_history=np.asarray(du_history),
        stagger_iterations=np.asarray(stagger_iterations_history),
        cutbacks=np.asarray(cutback_history),
        gate_warning=np.asarray(gate_warning_history, dtype=bool),
        softening=np.asarray(softening_history),
        snap_u_bar=np.asarray(snap_u_bar),
        u=np.asarray(u_snapshots),
        d=np.asarray(d_snapshots),
        reaction_star=np.asarray(reaction_history),
        force=np.asarray(reaction_history),
        H=np.asarray(H_snapshots),
        H_cell=(
            np.asarray(H_state_snapshots)
            if config.baseline.history_mode == "cell_mass_lumped"
            else np.empty((0, 0))
        ),
        fracture_energy=np.asarray(energy_history),
        damage_residual_mae=np.asarray(damage_residual_mae_history),
        damage_residual_inf=np.asarray(damage_residual_inf_history),
        damage_residual_scaled=np.asarray(damage_residual_scaled_history),
        damage_residual_scaled_inf=np.asarray(damage_residual_scaled_inf_history),
        damage_residual_solve_history_mae=np.asarray(
            damage_residual_solve_history_mae_history
        ),
        damage_residual_solve_history_inf=np.asarray(
            damage_residual_solve_history_inf_history
        ),
        damage_residual_solve_history_scaled_mae=np.asarray(
            damage_residual_solve_history_scaled_mae_history
        ),
        damage_residual_solve_history_scaled_inf=np.asarray(
            damage_residual_solve_history_scaled_inf_history
        ),
        damage_increment_min=np.asarray(damage_increment_min_history),
        damage_optimizer_iterations=np.asarray(damage_optimizer_iterations_history),
        damage_fixed_point_change=np.asarray(damage_fixed_point_change_history),
        d_l2_u_bar=np.asarray(d_l2_u_bar_history),
        d_l2_abs_direct_nonlocal=np.asarray(d_l2_abs_history),
        d_l2_rel_direct_nonlocal=np.asarray(d_l2_rel_history),
        reference_metrics=np.asarray(metrics),
        E0=float(config.physical.E),
        Gc=float(config.physical.Gc),
        l0=float(config.physical.l0),
        L0=float(config.geometry.L0),
        k_res=float(config.physical.k_res),
        N_l=float(N_l),
        N_G=float(N_G),
        imperfection_amplitude=float(config.imperfection.amplitude),
        imperfection_width=float(config.imperfection.width),
        imperfection_center=float(config.imperfection.center),
        delta=float(config.nl.delta),
        particle_volume=float(1.0 / (config.sample.Nx - 1)),
        loading_du_initial=float(config.train.du),
        loading_du_min=float(config.train.du_min),
        loading_du_max=float(config.train.du_max),
        loading_du_softening=float(config.train.du_softening),
        loading_cutback_factor=float(config.train.cutback_factor),
        loading_growth_factor=float(config.train.growth_factor),
        train_gate_mode=str(config.train.gate_mode),
        train_atol_damage_change=float(config.train.atol_damage_change),
        train_atol_damage_residual=float(config.train.atol_damage_residual),
        train_atol_damage_residual_inf=float(config.train.atol_damage_residual_inf),
        train_atol_damage_residual_scaled=float(
            config.train.atol_damage_residual_scaled
        ),
        train_atol_damage_residual_scaled_inf=float(
            config.train.atol_damage_residual_scaled_inf
        ),
        train_atol_healing_diagnostic=float(config.train.atol_healing_diagnostic),
        train_dtype=str(config.train.dtype),
        damage_output=str(config.arch.damage_output),
        damage_d_init=float(config.arch.d_init),
        network_activation=str(config.arch.activation),
        network_hidden_dim=int(config.arch.hidden_dim),
        local_hat_enrichment=bool(config.arch.use_local_hat_enrichment),
        local_hat_count=int(config.arch.local_hat_count),
        loss_objective=str(config.loss.objective),
        loss_residual_norm=str(config.loss.residual_norm),
        loss_residual_scaling=str(config.loss.residual_scaling),
        loss_hard_point_count=int(config.loss.hard_point_count),
        loss_hard_point_weight=float(config.loss.hard_point_weight),
        loss_hard_point_mode=str(config.loss.hard_point_mode),
        loss_hard_point_temperature=float(config.loss.hard_point_temperature),
        train_spatial_embedding=bool(config.arch.train_spatial_embedding),
        reference_mesh_nx=int(config.reference.mesh_nx),
        reference_history_space=str(config.reference.history_space),
        reference_history_method=str(config.reference.history_method),
        baseline_mesh_nx=int(config.baseline.mesh_nx),
        baseline_history_space=str(config.baseline.history_space),
        baseline_history_method=str(config.baseline.history_method),
        baseline_method=str(config.baseline.baseline_method),
        pinn_history_mode=str(config.baseline.history_mode),
        pinn_history_space=(
            "nodal_mass_lumped_from_DG0"
            if config.baseline.history_mode == "cell_mass_lumped"
            else "nodal"
        ),
    )
    return path


def run_training(config):
    """Run all load steps and return the output file path."""
    _validate_config(config)
    jax.config.update("jax_enable_x64", config.train.dtype == "float64")
    jax.config.update("jax_default_matmul_precision", config.train.matmul_precision)
    key = jr.PRNGKey(42)
    x_all, quadrature_weights = _build_grid(config)
    e_field = _build_modulus(config, x_all)
    x_cell = 0.5 * (x_all[:-1] + x_all[1:])
    e_cell = _build_modulus(config, x_cell)
    validate_mechanics_inputs(x_all, e_field, quadrature_weights,
                              config.physical.k_res)
    D = _build_D(config, x_all)
    damage_eval = _make_damage_eval(x_all, config.arch.local_equilibrium)
    mechanics_fn = _make_mechanics_fn(
        x_all,
        e_field,
        e_cell,
        quadrature_weights,
        config.physical.k_res,
        config.baseline.history_mode,
    )
    residual_check = _make_residual_check(
        x_all,
        D,
        config.physical.l0 / config.geometry.L0,
        config.physical.Gc / (config.physical.E * config.geometry.L0),
        config.arch.local_equilibrium,
    )
    convergence_check = _make_convergence_check()
    reference_local = _prepare_reference_fields(
        _load_reference(
            config.reference.data_path,
            config,
            config.reference,
        ),
        np.asarray(x_all[:, 0]),
    )
    reference_nonlocal = _prepare_reference_fields(
        _load_reference(
            config.baseline.data_path,
            config,
            config.baseline,
        ),
        np.asarray(x_all[:, 0]),
    )
    net = _build_network(config, key)
    optimizer = _build_optimizer(config)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    loss_fn = make_loss_fn(
        x_all,
        D,
        config.physical.l0 / config.geometry.L0,
        config.physical.Gc / (config.physical.E * config.geometry.L0),
        config.loss.objective,
        config.loss.residual_norm,
        config.loss.residual_scaling,
        int(config.loss.hard_point_count),
        float(config.loss.hard_point_weight),
        str(config.loss.hard_point_mode),
         float(config.loss.hard_point_temperature),
         bool(config.arch.local_equilibrium),
     )
    step_fn = _make_step(
        loss_fn,
        optimizer,
        freeze_spatial_embedding=not config.arch.train_spatial_embedding,
    )

    run_name = config.output.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(config.output.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(json.loads(config.to_json_best_effort()), indent=2))

    N_l = config.physical.l0 / config.geometry.L0
    N_G = config.physical.Gc / (config.physical.E * config.geometry.L0)
    particle_volume = 1.0 / (config.sample.Nx - 1)
    H_prev = jnp.zeros(
        (
            config.sample.Nx - 1
            if config.baseline.history_mode == "cell_mass_lumped"
            else config.sample.Nx
        ),
        dtype=x_all.dtype,
    )
    validate_damage_inputs(
        x_all, jnp.zeros((config.sample.Nx,), dtype=x_all.dtype)
    )
    d_previous = (
        jnp.zeros((config.sample.Nx,), dtype=x_all.dtype)
        if config.arch.local_equilibrium
        else damage_eval(net)
    )
    w_ema = jnp.ones((1,), dtype=x_all.dtype)
    bias = jnp.ones((1,), dtype=x_all.dtype)
    update_w = jnp.array(False)
    hard_point_indices = None

    u_bar_history = []
    du_history = []
    stagger_iterations_history = []
    cutback_history = []
    gate_warning_history = []
    reaction_history = []
    energy_history = []
    damage_residual_mae_history = []
    damage_residual_inf_history = []
    damage_residual_scaled_history = []
    damage_residual_scaled_inf_history = []
    damage_residual_solve_history_mae_history = []
    damage_residual_solve_history_inf_history = []
    damage_residual_solve_history_scaled_mae_history = []
    damage_residual_solve_history_scaled_inf_history = []
    damage_increment_min_history = []
    damage_optimizer_iterations_history = []
    damage_fixed_point_change_history = []
    d_l2_u_bar_history = []
    d_l2_abs_history = []
    d_l2_rel_history = []
    u_snapshots = []
    d_snapshots = []
    H_snapshots = []
    H_state_snapshots = []
    snap_u_bar = []
    metrics = []
    started = time.perf_counter()

    average_neighbours = len(np.asarray(D._row_idx)) / config.sample.Nx
    row_counts = np.bincount(np.asarray(D._row_idx), minlength=config.sample.Nx)
    minimum_neighbours = int(row_counts.min())
    log.info(
        "Grid=%d, h/delta=%.6e, delta/N_l=%.6e, avg neighbours=%.2f, "
        "min neighbours=%d",
        config.sample.Nx,
        particle_volume / config.nl.delta,
        config.nl.delta / N_l,
        average_neighbours,
        minimum_neighbours,
    )

    u_bar = 0.0
    step = 0
    du_current = config.train.du
    cutbacks_for_step = 0
    softening_started = False
    previous_reaction = None
    softening_history = []
    while u_bar < config.train.u_final:
        trial_du = min(du_current, config.train.u_final - u_bar)
        trial_u_bar = u_bar + trial_du
        net_committed = net
        opt_state_committed = opt_state
        w_ema_committed = w_ema
        d_iter = d_previous
        converged = False
        err_damage = np.inf
        err_sigma = np.inf
        err_history = np.inf
        err_mechanics = np.inf
        err_damage_residual = np.inf
        err_damage_residual_inf = np.inf
        err_damage_residual_scaled = np.inf
        err_damage_residual_scaled_inf = np.inf
        solve_history_residuals = (np.inf, np.inf, np.inf, np.inf)
        current_loss = np.inf
        optimizer_iterations_used = 0
        sigma = np.nan
        H_next = H_prev
        H_next_state = H_prev
        stagger_index = -1
        gate_warning = False

        for stagger_index in range(config.train.max_stagger_iters):
            d_context = d_iter
            mechanics = mechanics_fn(
                d_iter,
                jnp.asarray(trial_u_bar, dtype=x_all.dtype),
                H_prev,
            )
            sigma_before, _, _, _, _, H_used, _, _ = mechanics
            context_points = (
                _damage_context(H_used, d_context, N_l, N_G)
                if config.arch.local_equilibrium
                else None
            )
            _check_mechanics_result(
                mechanics,
                trial_u_bar,
                config.train.atol_stress_constancy,
            )
            if config.train.reset_optimizer_per_fixed_history:
                opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
            for damage_iteration in range(config.train.damage_iterations):
                if (
                    config.loss.hard_point_mode == "fixed"
                    and config.loss.hard_point_weight > 0.0
                    and (
                        hard_point_indices is None
                        or damage_iteration % config.train.damage_check_every == 0
                    )
                ):
                    residual_for_points = np.asarray(
                        damage_residual(
                            net, x_all, D, N_l, N_G, H_used, context_points
                        )
                    )
                    scale_for_points = np.asarray(
                        damage_residual_scale(D, N_l, N_G, H_used)
                    )
                    hard_point_indices = jnp.asarray(
                        np.argsort(
                            np.abs(residual_for_points / scale_for_points)
                        )[-int(config.loss.hard_point_count):],
                        dtype=jnp.int32,
                    )
                net, opt_state, loss_value, w_ema = step_fn(
                    net,
                    opt_state,
                    w_ema,
                    bias,
                    update_w,
                    x_all,
                    H_used,
                    hard_point_indices,
                    d_context,
                )
                current_loss = loss_value
                optimizer_iterations_used = damage_iteration + 1
                if (
                    optimizer_iterations_used % config.train.damage_check_every == 0
                    or optimizer_iterations_used == config.train.damage_iterations
                ):
                    (
                        check_mae,
                        check_inf,
                        check_scaled_mae,
                        check_scaled_inf,
                    ) = map(float, residual_check(net, H_used, d_context))
                    if not np.all(np.isfinite((
                        check_mae,
                        check_inf,
                        check_scaled_mae,
                        check_scaled_inf,
                    ))):
                        raise FloatingPointError("Damage residual became non-finite")
                    if config.loss.residual_scaling == "jacobi":
                        check_accept_mae = check_scaled_mae
                        check_accept_inf = check_scaled_inf
                        check_atol_mae = config.train.atol_damage_residual_scaled
                        check_atol_inf = config.train.atol_damage_residual_scaled_inf
                    else:
                        check_accept_mae = check_mae
                        check_accept_inf = check_inf
                        check_atol_mae = config.train.atol_damage_residual
                        check_atol_inf = config.train.atol_damage_residual_inf
                    if (
                        check_accept_mae <= check_atol_mae
                        and check_accept_inf <= check_atol_inf
                    ):
                        break

            (
                err_damage_residual,
                err_damage_residual_inf,
                err_damage_residual_scaled,
                err_damage_residual_scaled_inf,
            ) = map(float, residual_check(net, H_used, d_context))
            solve_history_residuals = (
                err_damage_residual,
                err_damage_residual_inf,
                err_damage_residual_scaled,
                err_damage_residual_scaled_inf,
            )
            if not np.all(np.isfinite((
                err_damage_residual,
                err_damage_residual_inf,
                err_damage_residual_scaled,
                err_damage_residual_scaled_inf,
            ))):
                raise FloatingPointError("Damage residual became non-finite")

            d_next = damage_eval(net, context_points)
            mechanics_next = mechanics_fn(
                d_next,
                jnp.asarray(trial_u_bar, dtype=x_all.dtype),
                H_prev,
            )
            sigma, _, _, _, _, H_next, H_next_state, _ = mechanics_next
            _check_mechanics_result(
                mechanics_next,
                trial_u_bar,
                config.train.atol_stress_constancy,
            )
            convergence = np.asarray(jax.device_get(
                convergence_check(d_next, d_iter, mechanics_next, mechanics)
            ))
            err_damage, err_sigma, err_history, err_mechanics = map(
                float, convergence
            )
            d_iter = d_next

            (
                err_damage_residual,
                err_damage_residual_inf,
                err_damage_residual_scaled,
                err_damage_residual_scaled_inf,
            ) = map(
                float,
                residual_check(net, H_next, d_context),
            )

            if _stagger_gates_pass(
                config,
                err_damage,
                err_damage_residual,
                err_damage_residual_inf,
                err_damage_residual_scaled,
                err_damage_residual_scaled_inf,
                err_sigma,
                err_history,
                err_mechanics,
            ):
                converged = True
                break

        if not converged:
            if config.train.gate_mode == "strict":
                net = net_committed
                opt_state = opt_state_committed
                w_ema = w_ema_committed
                at_minimum = trial_du <= config.train.du_min * (1.0 + 1e-12)
                if at_minimum or cutbacks_for_step >= config.train.max_cutbacks:
                    raise RuntimeError(
                        f"Load step at u_bar={trial_u_bar:.8e} did not converge "
                        f"after {stagger_index + 1} staggered iterations and "
                        f"{cutbacks_for_step} cutbacks: damage={err_damage:.3e}, "
                        f"damage_residual={err_damage_residual:.3e}, "
                        f"damage_residual_inf={err_damage_residual_inf:.3e}, "
                        f"damage_residual_scaled={err_damage_residual_scaled:.3e}, "
                        f"damage_residual_scaled_inf={err_damage_residual_scaled_inf:.3e}, "
                        f"sigma={err_sigma:.3e}, history={err_history:.3e}, "
                        f"mechanics={err_mechanics:.3e}, du={trial_du:.3e}"
                    )
                cutbacks_for_step += 1
                du_current = max(
                    config.train.du_min,
                    trial_du * config.train.cutback_factor,
                )
                log.info(
                    "PINN cutback before u_bar=%.8e: du=%.3e -> %.3e, "
                    "damage=%.3e, sigma=%.3e",
                    trial_u_bar,
                    trial_du,
                    du_current,
                    err_damage,
                    err_sigma,
                )
                continue
            gate_warning = True
            log.warning(
                "PINN gate warning at u_bar=%.8e after %d staggered iterations: "
                "damage=%.3e, residual_scaled=(%.3e,%.3e), sigma=%.3e, "
                "history=%.3e, mechanics=%.3e; committing finite candidate",
                trial_u_bar,
                stagger_index + 1,
                err_damage,
                err_damage_residual_scaled,
                err_damage_residual_scaled_inf,
                err_sigma,
                err_history,
                err_mechanics,
            )

        step += 1
        u_bar = trial_u_bar
        d_change = np.asarray(d_iter - d_previous)
        minimum_change = float(np.min(d_change))
        if minimum_change < -config.train.atol_healing_diagnostic:
            log.warning(
                "Damage healing diagnostic at load step %d: "
                "min(d_n-d_prev)=%.3e",
                step,
                minimum_change,
            )
        d_previous = d_iter
        H_prev = H_next_state
        H_damage_committed = H_next

        energy = _fracture_energy(d_iter, D, N_l, N_G, particle_volume)
        if not np.isfinite(float(energy)):
            raise FloatingPointError("Nonlocal fracture energy became non-finite")
        u_bar_history.append(float(u_bar))
        du_history.append(float(trial_du))
        stagger_iterations_history.append(stagger_index + 1)
        cutback_history.append(cutbacks_for_step)
        gate_warning_history.append(gate_warning)
        if (
            previous_reaction is not None
            and float(sigma) < previous_reaction - config.train.atol_sigma
        ):
            softening_started = True
        softening_history.append(bool(softening_started))
        reaction_history.append(float(sigma))
        energy_history.append(float(energy))
        damage_residual_mae_history.append(err_damage_residual)
        damage_residual_inf_history.append(err_damage_residual_inf)
        damage_residual_scaled_history.append(err_damage_residual_scaled)
        damage_residual_scaled_inf_history.append(err_damage_residual_scaled_inf)
        damage_residual_solve_history_mae_history.append(
            solve_history_residuals[0]
        )
        damage_residual_solve_history_inf_history.append(
            solve_history_residuals[1]
        )
        damage_residual_solve_history_scaled_mae_history.append(
            solve_history_residuals[2]
        )
        damage_residual_solve_history_scaled_inf_history.append(
            solve_history_residuals[3]
        )
        damage_increment_min_history.append(minimum_change)
        damage_optimizer_iterations_history.append(optimizer_iterations_used)
        damage_fixed_point_change_history.append(err_damage)
        event_index = step - 1
        writer.add_scalar("loss/damage", float(current_loss), event_index)
        writer.add_scalar("train/reaction_star", float(sigma), event_index)
        writer.add_scalar("train/fracture_energy", float(energy), event_index)
        writer.add_scalar("train/max_damage", float(jnp.max(d_iter)), event_index)
        writer.add_scalar(
            "train/damage_residual_mae", err_damage_residual, event_index
        )
        writer.add_scalar(
            "train/damage_residual_inf", err_damage_residual_inf, event_index
        )
        writer.add_scalar(
            "train/damage_residual_scaled",
            err_damage_residual_scaled,
            event_index,
        )
        writer.add_scalar(
            "train/damage_residual_scaled_inf",
            err_damage_residual_scaled_inf,
            event_index,
        )
        writer.add_scalar(
            "train/damage_increment_min", minimum_change, event_index
        )
        writer.add_scalar(
            "train/damage_optimizer_iterations",
            optimizer_iterations_used,
            event_index,
        )
        writer.add_scalar(
            "train/damage_fixed_point_change", err_damage, event_index
        )
        writer.add_scalar("train/stagger_iterations", stagger_index + 1, event_index)

        is_final = np.isclose(
            u_bar, config.train.u_final, rtol=0.0, atol=1e-14
        )
        if step % config.train.eval_every == 0 or is_final:
            evaluation_context = (
                _damage_context(H_damage_committed, d_previous, N_l, N_G)
                if config.arch.local_equilibrium
                else None
            )
            local_metrics = _eval_reference(
                net,
                x_all,
                e_field,
                u_bar,
                quadrature_weights,
                config.physical.k_res,
                reference_local,
                writer,
                event_index,
                config,
                H_prev,
                H_damage_committed,
                "local_fem",
                damage_eval,
                mechanics_fn,
                evaluation_context,
            )
            nonlocal_metrics = _eval_reference(
                net,
                x_all,
                e_field,
                u_bar,
                quadrature_weights,
                config.physical.k_res,
                reference_nonlocal,
                writer,
                event_index,
                config,
                H_prev,
                H_damage_committed,
                "direct_nonlocal",
                damage_eval,
                mechanics_fn,
                evaluation_context,
            )
            metrics.append((float(u_bar), *local_metrics, *nonlocal_metrics))
            d_l2_abs, d_l2_rel = _add_training_figures(
                writer,
                event_index,
                x_all,
                u_bar,
                d_previous,
                u_bar_history,
                reaction_history,
                reference_nonlocal,
            )
            d_l2_u_bar_history.append(float(u_bar))
            d_l2_abs_history.append(d_l2_abs)
            d_l2_rel_history.append(d_l2_rel)
            writer.add_scalar(
                "eval/direct_nonlocal/d_l2_abs",
                d_l2_abs,
                event_index,
            )
            writer.add_scalar(
                "eval/direct_nonlocal/d_l2_rel",
                d_l2_rel,
                event_index,
            )

        if step % config.train.checkpoint_every == 0 or is_final:
            checkpoint = os.path.join(
                checkpoint_dir,
                f"damage_step_{step:06d}.eqx",
            )
            eqx.tree_serialise_leaves(checkpoint, net)

        if step % config.train.log_every == 0 or is_final:
            elapsed = time.perf_counter() - started
            log.info(
                "[step %5d] u_bar=%.6e du=%.3e loss=%.3e reaction=%.6e "
                "max_d=%.6e stagger=%d cutbacks=%d softening=%s elapsed=%.1fs",
                step,
                u_bar,
                trial_du,
                float(current_loss),
                float(sigma),
                float(jnp.max(d_iter)),
                stagger_index + 1,
                cutbacks_for_step,
                softening_started,
                elapsed,
            )

        if step % config.train.eval_every == 0 or is_final:
            mechanics_save = mechanics_fn(
                d_iter,
                jnp.asarray(u_bar, dtype=x_all.dtype),
                H_prev,
            )
            u_snapshots.append(np.asarray(mechanics_save[2]))
            d_snapshots.append(np.asarray(d_iter))
            H_snapshots.append(np.asarray(H_damage_committed))
            H_state_snapshots.append(np.asarray(H_prev))
            snap_u_bar.append(float(u_bar))

        cutbacks_for_step = 0
        previous_reaction = float(sigma)
        if softening_started:
            du_current = min(
                config.train.du_softening,
                trial_du * config.train.growth_factor,
            )
        else:
            du_current = min(
                config.train.du_max,
                trial_du * config.train.growth_factor,
            )

    writer.close()
    return _save_results(
        config,
        run_dir,
        x_all,
        u_bar_history,
        du_history,
        stagger_iterations_history,
        cutback_history,
        gate_warning_history,
        softening_history,
        snap_u_bar,
        u_snapshots,
        d_snapshots,
        reaction_history,
        H_snapshots,
        H_state_snapshots,
        energy_history,
        damage_residual_mae_history,
        damage_residual_inf_history,
        damage_residual_scaled_history,
        damage_residual_scaled_inf_history,
        damage_residual_solve_history_mae_history,
        damage_residual_solve_history_inf_history,
        damage_residual_solve_history_scaled_mae_history,
        damage_residual_solve_history_scaled_inf_history,
        damage_increment_min_history,
        damage_optimizer_iterations_history,
        damage_fixed_point_change_history,
        d_l2_u_bar_history,
        d_l2_abs_history,
        d_l2_rel_history,
        metrics,
        N_l,
        N_G,
    )


def _main(_):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result_path = run_training(FLAGS.config)
    log.info("Saved nonlocal PINN results to %s", result_path)


if __name__ == "__main__":
    _absl_app.run(_main)
