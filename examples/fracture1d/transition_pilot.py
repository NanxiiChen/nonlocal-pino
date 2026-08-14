"""Replay one direct-baseline load transition with the damage PINN."""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from examples.fracture1d import train
from examples.fracture1d.configs.train_base import get_config
from examples.fracture1d.frozen_pilot import _target_fields


def _target_history_state(config, load, dtype):
    """Load the exact previous history in its configured state space."""
    reference = train._load_reference(
        config.baseline.data_path, config, config.baseline
    )
    snap_u_bar = np.asarray(reference["snap_u_bar"])
    index = int(np.argmin(np.abs(snap_u_bar - load)))
    if not np.isclose(snap_u_bar[index], load, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"No exact baseline history snapshot at {load:.8e}")
    if config.baseline.history_mode == "cell_mass_lumped":
        if "H_cell" not in reference.files:
            raise KeyError("Cell-history transition requires baseline H_cell")
        state = np.asarray(reference["H_cell"][index])
    else:
        state = np.asarray(reference["H"][index])
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("Previous history state is non-finite")
    return jnp.asarray(state, dtype=dtype)


def _history_continuation_values(H_prev, H_target, steps):
    """Return fixed-shape histories on the linear ``H_prev`` to ``H_target`` path."""
    if isinstance(steps, (bool, np.bool_)) or not isinstance(
        steps, (int, np.integer)
    ) or steps < 1:
        raise ValueError("history continuation steps must be a positive integer")
    H_prev_np = np.asarray(H_prev)
    H_target_np = np.asarray(H_target)
    if H_prev_np.ndim != 1 or H_target_np.shape != H_prev_np.shape:
        raise ValueError(
            "H_prev and H_target must be finite one-dimensional arrays with "
            "the same shape"
        )
    if not np.issubdtype(H_prev_np.dtype, np.floating):
        raise ValueError("H_prev must use a floating-point dtype")
    if not np.all(np.isfinite(H_prev_np)) or not np.all(np.isfinite(H_target_np)):
        raise ValueError("H_prev and H_target must be finite")
    H_prev_jax = jnp.asarray(H_prev)
    H_target_jax = jnp.asarray(H_target, dtype=H_prev_jax.dtype)
    fractions = jnp.arange(1, steps + 1, dtype=H_prev_jax.dtype) / steps
    histories = H_prev_jax[None, :] + fractions[:, None] * (
        H_target_jax - H_prev_jax
    )[None, :]
    return fractions, histories


def _bisect_history_interval(previous_lambda, target_lambda):
    """Return the strict midpoint used when a history substep fails."""
    if not np.isfinite(previous_lambda) or not np.isfinite(target_lambda):
        raise ValueError("history interval endpoints must be finite")
    if not 0.0 <= previous_lambda < target_lambda <= 1.0:
        raise ValueError("history interval must satisfy 0 <= previous < target <= 1")
    midpoint = 0.5 * (previous_lambda + target_lambda)
    if not previous_lambda < midpoint < target_lambda:
        raise ValueError("history interval is not splittable")
    return midpoint


def _build_objective_optimizers(
    config, residual_lr, residual_optimizer_name, rprop_step_size
):
    """Build separate energy and residual optimizers with explicit settings."""
    if residual_optimizer_name not in ("adam", "rprop"):
        raise ValueError("residual_optimizer_name must be 'adam' or 'rprop'")
    if not np.isfinite(rprop_step_size) or rprop_step_size <= 0.0:
        raise ValueError("rprop_step_size must be positive and finite")
    energy_lr = float(config.train.lr_init)
    energy_lr_min = float(config.train.lr_min)
    energy_optimizer_name = str(config.train.optimizer)
    energy_optimizer = train._build_optimizer(config)
    config.train.optimizer = residual_optimizer_name
    if residual_optimizer_name == "adam":
        config.train.lr_init = float(residual_lr)
        config.train.lr_min = min(float(config.train.lr_min), float(residual_lr))
    else:
        config.train.rprop_initial_step_size = float(rprop_step_size)
    residual_optimizer = train._build_optimizer(config)
    config.train.optimizer = energy_optimizer_name
    config.train.lr_init = energy_lr
    config.train.lr_min = energy_lr_min
    return energy_optimizer, residual_optimizer, energy_lr


def run_transition_pilot(
    config,
    previous_load,
    trial_load,
    warm_start_steps,
    damage_iterations,
    max_stagger_iters,
    history_continuation_steps,
    objective,
    switch_damage,
    residual_lr,
    residual_optimizer_name,
    rprop_step_size,
    use_local_gaussian,
    gaussian_width,
    gaussian_amplitude,
    continuation_optimizer_name,
    output_path,
    hidden_dim,
    activation,
    spatial_scale,
    damage_output,
    d_init,
    staged_output_iterations=0,
    adaptive_history=False,
    max_history_continuation_steps=None,
):
    """Warm-start at one accepted state and replay the next load transition.

    Each outer staggered iteration solves a sequence of fixed-history damage
    problems along a prescribed linear history path. Every continuation
    substep must satisfy the configured fixed-history residual gate before the
    next history value is attempted. Optional diagnostics first update only the
    output layer, and can bisect failed history intervals without changing the
    residual objective or the production configuration.
    """
    if not 0.0 <= previous_load < trial_load:
        raise ValueError("Require 0 <= previous_load < trial_load")
    for name, value in (
        ("warm_start_steps", warm_start_steps),
        ("staged_output_iterations", staged_output_iterations),
    ):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name, value in (
        ("damage_iterations", damage_iterations),
        ("max_stagger_iters", max_stagger_iters),
        ("history_continuation_steps", history_continuation_steps),
    ):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_history_continuation_steps is None:
        max_history_continuation_steps = history_continuation_steps
    if (
        int(max_history_continuation_steps) != max_history_continuation_steps
        or max_history_continuation_steps < 1
    ):
        raise ValueError("max_history_continuation_steps must be positive")
    if max_history_continuation_steps < history_continuation_steps:
        raise ValueError(
            "max_history_continuation_steps must be at least the initial count"
        )
    if objective not in ("residual", "energy", "hybrid"):
        raise ValueError("objective must be 'residual', 'energy', or 'hybrid'")
    if not 0.0 < switch_damage < 1.0:
        raise ValueError("switch_damage must lie strictly between zero and one")
    if not np.isfinite(residual_lr) or residual_lr <= 0.0:
        raise ValueError("residual_lr must be positive and finite")
    if residual_optimizer_name not in ("adam", "rprop"):
        raise ValueError("residual_optimizer_name must be 'adam' or 'rprop'")
    if not np.isfinite(rprop_step_size) or rprop_step_size <= 0.0:
        raise ValueError("rprop_step_size must be positive and finite")
    if not np.isfinite(gaussian_width) or gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be positive and finite")
    if not np.isfinite(gaussian_amplitude):
        raise ValueError("gaussian_amplitude must be finite")
    if continuation_optimizer_name not in ("residual", "lbfgs"):
        raise ValueError(
            "continuation_optimizer_name must be 'residual' or 'lbfgs'"
        )
    if int(hidden_dim) != hidden_dim or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    if activation not in ("tanh", "relu", "gelu", "silu", "swish"):
        raise ValueError(f"Unknown activation {activation!r}")
    if not np.isfinite(spatial_scale) or spatial_scale <= 0.0:
        raise ValueError("spatial_scale must be positive and finite")
    if damage_output not in ("linear", "exp_square"):
        raise ValueError("damage_output must be 'linear' or 'exp_square'")
    if not np.isfinite(d_init):
        raise ValueError("d_init must be finite")

    config.arch.hidden_dim = int(hidden_dim)
    config.arch.activation = str(activation)
    config.arch.spatial_scale = float(spatial_scale)
    config.arch.damage_output = str(damage_output)
    config.arch.d_init = float(d_init)
    config.arch.use_local_gaussian = bool(use_local_gaussian)
    config.arch.local_gaussian_width = float(gaussian_width)
    config.arch.local_gaussian_amplitude = float(gaussian_amplitude)
    config.loss.objective = "residual" if objective == "hybrid" else objective
    config.loss.residual_norm = "mse"
    config.loss.residual_scaling = "jacobi"
    config.train.damage_iterations = int(damage_iterations)
    config.train.max_stagger_iters = int(max_stagger_iters)
    if (staged_output_iterations > 0 or adaptive_history) and objective != "residual":
        raise ValueError("staged/adaptive diagnostics require objective='residual'")
    train._validate_config(config)
    jax.config.update("jax_enable_x64", config.train.dtype == "float64")
    jax.config.update("jax_default_matmul_precision", config.train.matmul_precision)

    x_all, quadrature_weights = train._build_grid(config)
    e_field = train._build_modulus(config, x_all)
    x_cell = 0.5 * (x_all[:-1] + x_all[1:])
    e_cell = train._build_modulus(config, x_cell)
    D = train._build_D(config, x_all)
    damage_eval = train._make_damage_eval(x_all)
    mechanics_fn = train._make_mechanics_fn(
        x_all,
        e_field,
        e_cell,
        quadrature_weights,
        config.physical.k_res,
        config.baseline.history_mode,
    )
    N_l = config.physical.l0 / config.geometry.L0
    N_G = config.physical.Gc / (config.physical.E * config.geometry.L0)
    residual_check = train._make_residual_check(x_all, D, N_l, N_G)
    convergence_check = train._make_convergence_check()
    energy_optimizer, residual_optimizer, energy_lr = _build_objective_optimizers(
        config, residual_lr, residual_optimizer_name, rprop_step_size
    )

    def make_step(objective_name, optimizer, output_only=False):
        loss_fn = train.make_loss_fn(
            x_all,
            D,
            N_l,
            N_G,
            objective_name,
            config.loss.residual_norm,
            config.loss.residual_scaling,
            int(config.loss.hard_point_count),
            float(config.loss.hard_point_weight),
            str(config.loss.hard_point_mode),
            float(config.loss.hard_point_temperature),
            bool(config.arch.local_equilibrium),
        )
        return train._make_step(loss_fn, optimizer, output_only=output_only)

    residual_step_fn = make_step("residual", residual_optimizer)
    residual_output_step_fn = make_step(
        "residual", residual_optimizer, output_only=True
    )
    energy_step_fn = make_step("energy", energy_optimizer)
    lbfgs_optimizer = optax.lbfgs(memory_size=10)
    lbfgs_step_fn = _make_lbfgs_step(
        x_all,
        D,
        N_l,
        N_G,
        config.loss.residual_scaling,
        int(config.loss.hard_point_count),
        float(config.loss.hard_point_weight),
        str(config.loss.hard_point_mode),
        float(config.loss.hard_point_temperature),
        bool(config.arch.local_equilibrium),
        lbfgs_optimizer,
    )

    x_eval = np.asarray(x_all[:, 0])
    hard_point_indices = None

    def select_hard_points(net, H_points):
        residual = np.asarray(
            train.damage_residual(net, x_all, D, N_l, N_G, H_points)
        )
        scale = np.asarray(
            train.damage_residual_scale(D, N_l, N_G, H_points)
        )
        return jnp.asarray(
            np.argsort(np.abs(residual / scale))[-int(
                config.loss.hard_point_count
            ):],
            dtype=jnp.int32,
        )
    d_previous_target, H_previous_np = _target_fields(
        config, x_eval, previous_load
    )
    H_previous = jnp.asarray(H_previous_np, dtype=x_all.dtype)
    H_previous_state = _target_history_state(
        config, previous_load, x_all.dtype
    )
    net = train._build_network(config, jr.PRNGKey(42))
    warm_state = residual_optimizer.init(eqx.filter(net, eqx.is_array))
    warm_loss = np.inf
    warm_residual_mae = np.inf
    warm_residual_inf = np.inf
    warm_residual_scaled_mae = np.inf
    warm_residual_scaled_inf = np.inf
    for warm_iteration in range(1, int(warm_start_steps) + 1):
        net, warm_state, warm_loss, _ = residual_step_fn(
            net,
            warm_state,
            jnp.ones((1,), dtype=x_all.dtype),
            jnp.ones((1,), dtype=x_all.dtype),
            jnp.array(False),
            x_all,
            H_previous,
            None,
        )
        if (
            warm_iteration % config.train.damage_check_every == 0
            or warm_iteration == int(warm_start_steps)
        ):
            (
                warm_residual_mae,
                warm_residual_inf,
                warm_residual_scaled_mae,
                warm_residual_scaled_inf,
            ) = map(float, residual_check(net, H_previous))
    if use_local_gaussian:
        net = eqx.tree_at(
            lambda n: n.amplitude,
            net,
            jnp.full_like(net.amplitude, gaussian_amplitude),
        )
    d_iter = damage_eval(net)
    d_warm = np.asarray(d_iter)
    warm_error = np.abs(d_warm - d_previous_target)

    stagger_damage = []
    stagger_history = []
    stagger_history_state = []
    err_damage_history = []
    err_sigma_history = []
    err_history_history = []
    err_mechanics_history = []
    residual_mae_history = []
    residual_inf_history = []
    residual_scaled_mae_history = []
    residual_scaled_inf_history = []
    history_target_history = []
    continuation_residual_mae_history = []
    continuation_residual_inf_history = []
    continuation_residual_scaled_mae_history = []
    continuation_residual_scaled_inf_history = []
    continuation_max_damage_history = []
    continuation_iterations_history = []
    continuation_objective_history = []
    continuation_converged_history = []
    continuation_lambda_history_history = []
    continuation_retry_history = []
    continuation_complete_history = []
    optimizer_iterations_history = []
    converged = False
    continuation_lambdas = np.arange(
        1, int(history_continuation_steps) + 1, dtype=np.float64
    ) / int(history_continuation_steps)
    continuation_capacity = int(max_history_continuation_steps)

    def run_damage_substep(net, opt_state, H_solve, hard_point_indices, step_fn):
        """Run full-network residual optimization for one history value."""
        w_ema = jnp.ones((1,), dtype=x_all.dtype)
        bias = jnp.ones((1,), dtype=x_all.dtype)
        update_w = jnp.array(False)
        total_iterations = 0
        residual_mae = np.inf
        residual_inf = np.inf
        residual_scaled_mae = np.inf
        residual_scaled_inf = np.inf
        substep_converged = False
        optimizer_iteration = 0
        for optimizer_iteration in range(1, int(damage_iterations) + 1):
            net, opt_state, _, _ = step_fn(
                net,
                opt_state,
                w_ema,
                bias,
                update_w,
                x_all,
                H_solve,
                hard_point_indices,
            )
            total_iterations += 1
            if (
                optimizer_iteration % config.train.damage_check_every != 0
                and optimizer_iteration != damage_iterations
            ):
                continue
            (
                residual_mae,
                residual_inf,
                residual_scaled_mae,
                residual_scaled_inf,
            ) = map(float, residual_check(net, H_solve))
            if config.loss.residual_scaling == "jacobi":
                residual_accept_mae = residual_scaled_mae
                residual_accept_inf = residual_scaled_inf
                residual_atol_mae = config.train.atol_damage_residual_scaled
                residual_atol_inf = config.train.atol_damage_residual_scaled_inf
            else:
                residual_accept_mae = residual_mae
                residual_accept_inf = residual_inf
                residual_atol_mae = config.train.atol_damage_residual
                residual_atol_inf = config.train.atol_damage_residual_inf
            if (
                residual_accept_mae <= residual_atol_mae
                and residual_accept_inf <= residual_atol_inf
            ):
                substep_converged = True
                break
            if (
                config.loss.hard_point_mode == "fixed"
                and config.loss.hard_point_weight > 0.0
            ):
                hard_point_indices = select_hard_points(net, H_solve)
        return (
            net,
            opt_state,
            hard_point_indices,
            residual_mae,
            residual_inf,
            residual_scaled_mae,
            residual_scaled_inf,
            substep_converged,
            total_iterations,
        )

    for stagger_index in range(int(max_stagger_iters)):
        d_before_stagger = d_iter
        mechanics_before = mechanics_fn(
            d_iter,
            jnp.asarray(trial_load, dtype=x_all.dtype),
            H_previous_state,
        )
        H_used = mechanics_before[5]
        history_target_history.append(np.asarray(H_used))
        continuation_residual_mae = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        continuation_residual_inf = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        continuation_residual_scaled_mae = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        continuation_residual_scaled_inf = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        continuation_max_damage = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        continuation_iterations = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        continuation_objectives = np.full(
            continuation_capacity, "not_run", dtype="U8"
        )
        continuation_converged = np.zeros(
            continuation_capacity, dtype=bool
        )
        continuation_complete = True
        total_optimizer_iterations = 0
        residual_mae = np.inf
        residual_inf = np.inf
        residual_scaled_mae = np.inf
        residual_scaled_inf = np.inf
        optimizer_iteration = 0
        H_solve = H_used
        pending_lambdas = [float(value) for value in np.asarray(continuation_lambdas)]
        accepted_lambda = 0.0
        continuation_attempts = 0
        continuation_retries = 0
        continuation_lambda_history = np.full(
            continuation_capacity, np.nan, dtype=np.float64
        )
        history_index = 0
        if staged_output_iterations > 0:
            stage_lambda = float(continuation_lambdas[0])
            stage_history = H_previous + stage_lambda * (H_used - H_previous)
            stage_state = residual_optimizer.init(eqx.filter(net, eqx.is_array))
            for _ in range(int(staged_output_iterations)):
                net, stage_state, _, _ = residual_output_step_fn(
                    net,
                    stage_state,
                    jnp.ones((1,), dtype=x_all.dtype),
                    jnp.ones((1,), dtype=x_all.dtype),
                    jnp.array(False),
                    x_all,
                    stage_history,
                    hard_point_indices,
                )
        while pending_lambdas:
            target_lambda = pending_lambdas.pop(0)
            if continuation_attempts >= continuation_capacity:
                continuation_complete = False
                break
            continuation_attempts += 1
            H_solve = H_previous + target_lambda * (H_used - H_previous)
            net_before = net
            if objective == "hybrid":
                active_objective = (
                    "energy"
                    if float(jnp.max(d_iter)) < switch_damage
                    else "residual"
                )
            else:
                active_objective = objective
            if active_objective == "energy":
                step_fn = energy_step_fn
                active_optimizer = energy_optimizer
            else:
                if continuation_optimizer_name == "lbfgs":
                    step_fn = lbfgs_step_fn
                    active_optimizer = lbfgs_optimizer
                else:
                    step_fn = residual_step_fn
                    active_optimizer = residual_optimizer
            opt_state = active_optimizer.init(eqx.filter(net, eqx.is_array))
            if (
                config.loss.hard_point_mode == "fixed"
                and config.loss.hard_point_weight > 0.0
            ):
                hard_point_indices = select_hard_points(net, H_solve)
            (
                net,
                opt_state,
                hard_point_indices,
                residual_mae,
                residual_inf,
                residual_scaled_mae,
                residual_scaled_inf,
                substep_converged,
                substep_iterations,
            ) = run_damage_substep(
                net,
                opt_state,
                H_solve,
                hard_point_indices,
                step_fn,
            )
            if not substep_converged and adaptive_history:
                if accepted_lambda >= target_lambda:
                    raise RuntimeError("History continuation target ordering failed")
                if continuation_attempts < continuation_capacity:
                    midpoint = _bisect_history_interval(
                        accepted_lambda, target_lambda
                    )
                    net = net_before
                    pending_lambdas.insert(0, target_lambda)
                    pending_lambdas.insert(0, midpoint)
                    continuation_retries += 1
                    continue

            d_iter = damage_eval(net)
            continuation_lambda_history[history_index] = target_lambda
            continuation_residual_mae[history_index] = residual_mae
            continuation_residual_inf[history_index] = residual_inf
            continuation_residual_scaled_mae[history_index] = residual_scaled_mae
            continuation_residual_scaled_inf[history_index] = residual_scaled_inf
            continuation_max_damage[history_index] = float(jnp.max(d_iter))
            continuation_iterations[history_index] = substep_iterations
            continuation_objectives[history_index] = active_objective
            continuation_converged[history_index] = substep_converged
            history_index += 1
            total_optimizer_iterations += substep_iterations
            if not substep_converged:
                continuation_complete = False
                break
            accepted_lambda = target_lambda

        if not np.isclose(accepted_lambda, 1.0, rtol=0.0, atol=1.0e-14):
            continuation_complete = False

        d_next = d_iter
        mechanics_next = mechanics_fn(
            d_next,
            jnp.asarray(trial_load, dtype=x_all.dtype),
            H_previous_state,
        )
        convergence = np.asarray(jax.device_get(
            convergence_check(
                d_next, d_before_stagger, mechanics_next, mechanics_before
            )
        ))
        err_damage, err_sigma, err_history, err_mechanics = map(float, convergence)
        stagger_damage.append(np.asarray(d_next))
        committed_history = mechanics_next[5]
        (
            residual_mae,
            residual_inf,
            residual_scaled_mae,
            residual_scaled_inf,
        ) = map(float, residual_check(net, committed_history))
        stagger_history.append(np.asarray(committed_history))
        stagger_history_state.append(np.asarray(mechanics_next[6]))
        err_damage_history.append(err_damage)
        err_sigma_history.append(err_sigma)
        err_history_history.append(err_history)
        err_mechanics_history.append(err_mechanics)
        residual_mae_history.append(residual_mae)
        residual_inf_history.append(residual_inf)
        residual_scaled_mae_history.append(residual_scaled_mae)
        residual_scaled_inf_history.append(residual_scaled_inf)
        continuation_residual_mae_history.append(continuation_residual_mae)
        continuation_residual_inf_history.append(continuation_residual_inf)
        continuation_residual_scaled_mae_history.append(
            continuation_residual_scaled_mae
        )
        continuation_residual_scaled_inf_history.append(
            continuation_residual_scaled_inf
        )
        continuation_max_damage_history.append(continuation_max_damage)
        continuation_iterations_history.append(continuation_iterations)
        continuation_objective_history.append(continuation_objectives)
        continuation_converged_history.append(continuation_converged)
        continuation_lambda_history_history.append(
            np.asarray(continuation_lambda_history)
        )
        continuation_retry_history.append(continuation_retries)
        continuation_complete_history.append(continuation_complete)
        optimizer_iterations_history.append(total_optimizer_iterations)
        print(
            f"stagger={stagger_index + 1} max_d={float(jnp.max(d_next)):.6e} "
            f"min_d={float(jnp.min(d_next)):.6e} damage={err_damage:.3e} "
            f"history={err_history:.3e} residual_mae={residual_mae:.3e} "
            f"residual_inf={residual_inf:.3e} "
            f"scaled_inf={residual_scaled_inf:.3e} "
            f"history_substeps={int(np.sum(continuation_converged))}/"
            f"{continuation_attempts} retries={continuation_retries}"
        )
        if not continuation_complete:
            break
        if config.loss.residual_scaling == "jacobi":
            residual_accept_mae = residual_scaled_mae
            residual_accept_inf = residual_scaled_inf
            residual_atol_mae = config.train.atol_damage_residual_scaled
            residual_atol_inf = config.train.atol_damage_residual_scaled_inf
        else:
            residual_accept_mae = residual_mae
            residual_accept_inf = residual_inf
            residual_atol_mae = config.train.atol_damage_residual
            residual_atol_inf = config.train.atol_damage_residual_inf
        if (
            err_damage <= config.train.atol_damage_change
            and residual_accept_mae <= residual_atol_mae
            and residual_accept_inf <= residual_atol_inf
            and err_sigma <= config.train.atol_sigma
            and err_history <= config.train.atol_history
            and err_mechanics <= config.train.atol_mechanics
            and continuation_complete
        ):
            converged = True
            break

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x_eval,
        previous_load=float(previous_load),
        trial_load=float(trial_load),
        d_previous_target=d_previous_target,
        H_previous=H_previous_np,
        H_previous_cell=(
            np.asarray(H_previous_state)
            if config.baseline.history_mode == "cell_mass_lumped"
            else np.empty(0)
        ),
        warm_start_damage=d_warm,
        warm_start_error_mae=float(np.mean(warm_error)),
        warm_start_error_inf=float(np.max(warm_error)),
        warm_start_loss=float(warm_loss),
        warm_start_residual_mae=float(warm_residual_mae),
        warm_start_residual_inf=float(warm_residual_inf),
        warm_start_residual_scaled_mae=float(warm_residual_scaled_mae),
        warm_start_residual_scaled_inf=float(warm_residual_scaled_inf),
        warm_start_objective="residual",
        warm_start_steps=int(warm_start_steps),
        d=np.asarray(stagger_damage),
        H=np.asarray(stagger_history),
        H_cell=(
            np.asarray(stagger_history_state)
            if config.baseline.history_mode == "cell_mass_lumped"
            else np.empty((0, 0))
        ),
        err_damage=np.asarray(err_damage_history),
        err_sigma=np.asarray(err_sigma_history),
        err_history=np.asarray(err_history_history),
        err_mechanics=np.asarray(err_mechanics_history),
        damage_residual_mae=np.asarray(residual_mae_history),
        damage_residual_inf=np.asarray(residual_inf_history),
        damage_residual_scaled_mae=np.asarray(residual_scaled_mae_history),
        damage_residual_scaled_inf=np.asarray(residual_scaled_inf_history),
        history_target=np.asarray(history_target_history),
        history_continuation_lambdas=continuation_lambdas,
        history_continuation_residual_mae=np.asarray(
            continuation_residual_mae_history
        ),
        history_continuation_residual_inf=np.asarray(
            continuation_residual_inf_history
        ),
        history_continuation_residual_scaled_mae=np.asarray(
            continuation_residual_scaled_mae_history
        ),
        history_continuation_residual_scaled_inf=np.asarray(
            continuation_residual_scaled_inf_history
        ),
        history_continuation_max_damage=np.asarray(
            continuation_max_damage_history
        ),
        history_continuation_iterations=np.asarray(
            continuation_iterations_history
        ),
        history_continuation_objective=np.asarray(
            continuation_objective_history
        ),
        history_continuation_converged=np.asarray(
            continuation_converged_history
        ),
        history_continuation_complete=np.asarray(
            continuation_complete_history
        ),
        history_continuation_steps=int(history_continuation_steps),
        history_continuation_max_steps=continuation_capacity,
        history_continuation_adaptive=bool(adaptive_history),
        history_continuation_lambdas_actual=np.asarray(
            continuation_lambda_history_history
        ),
        history_continuation_retries=np.asarray(continuation_retry_history),
        staged_output_iterations=int(staged_output_iterations),
        hidden_dim=int(hidden_dim),
        activation=str(activation),
        spatial_scale=float(spatial_scale),
        damage_output=str(damage_output),
        d_init=float(d_init),
        optimizer_iterations=np.asarray(optimizer_iterations_history),
        converged=converged,
        objective=objective,
        switch_damage=float(switch_damage),
        energy_lr=float(energy_lr),
        residual_lr=float(residual_lr),
        residual_optimizer=residual_optimizer_name,
        rprop_step_size=float(rprop_step_size),
        continuation_optimizer=continuation_optimizer_name,
        use_local_gaussian=bool(use_local_gaussian),
        gaussian_width=float(gaussian_width),
        gaussian_amplitude=float(gaussian_amplitude),
        dtype=config.train.dtype,
        history_mode=str(config.baseline.history_mode),
        history_state_space=(
            "DG0_cell"
            if config.baseline.history_mode == "cell_mass_lumped"
            else "nodal"
        ),
    )
    return output_path


def _make_lbfgs_step(
    x_all,
    D,
    N_l,
    N_G,
    residual_scaling,
    hard_point_count,
    hard_point_weight,
    hard_point_mode,
    hard_point_temperature,
    local_equilibrium,
    optimizer,
):
    """Create a deterministic full-grid L-BFGS step for residual refinement."""
    objective_name = "residual"
    residual_norm = "mse"

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
    ):
        del w_ema, bias, update_w, x_points

        def objective(n):
            return train._loss_module.loss_damage(
                n,
                x_all,
                D,
                N_l,
                N_G,
                H_points,
                objective_name,
                residual_norm,
                residual_scaling,
                hard_point_count,
                hard_point_weight,
                hard_point_mode,
                hard_point_temperature,
                hard_point_indices,
                None,
            )

        loss, gradient = eqx.filter_value_and_grad(objective)(net)
        updates, opt_state = optimizer.update(
            gradient,
            opt_state,
            net,
            value=loss,
            grad=gradient,
            value_fn=objective,
        )
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss, jnp.ones((1,), dtype=loss.dtype)

    return step


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-load", type=float, required=True)
    parser.add_argument("--trial-load", type=float, required=True)
    parser.add_argument("--warm-start-steps", type=int, required=True)
    parser.add_argument("--damage-iterations", type=int, required=True)
    parser.add_argument("--max-stagger-iters", type=int, required=True)
    parser.add_argument("--history-continuation-steps", type=int, required=True)
    parser.add_argument(
        "--objective", choices=("residual", "energy", "hybrid"), required=True
    )
    parser.add_argument("--switch-damage", type=float, required=True)
    parser.add_argument("--residual-lr", type=float, required=True)
    parser.add_argument(
        "--residual-optimizer", choices=("adam", "rprop"), required=True
    )
    parser.add_argument("--rprop-step-size", type=float, required=True)
    parser.add_argument("--use-local-gaussian", action="store_true")
    parser.add_argument("--gaussian-width", type=float, required=True)
    parser.add_argument("--gaussian-amplitude", type=float, required=True)
    parser.add_argument(
        "--continuation-optimizer", choices=("residual", "lbfgs"), required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument(
        "--activation",
        choices=("tanh", "relu", "gelu", "silu", "swish"),
        required=True,
    )
    parser.add_argument("--spatial-scale", type=float, required=True)
    parser.add_argument(
        "--damage-output", choices=("linear", "exp_square"), required=True
    )
    parser.add_argument("--d-init", type=float, required=True)
    parser.add_argument("--staged-output-iterations", type=int, default=0)
    parser.add_argument("--adaptive-history", action="store_true")
    parser.add_argument("--max-history-continuation-steps", type=int, default=None)
    args = parser.parse_args()
    run_transition_pilot(
        get_config(),
        args.previous_load,
        args.trial_load,
        args.warm_start_steps,
        args.damage_iterations,
        args.max_stagger_iters,
        args.history_continuation_steps,
        args.objective,
        args.switch_damage,
        args.residual_lr,
        args.residual_optimizer,
        args.rprop_step_size,
        args.use_local_gaussian,
        args.gaussian_width,
        args.gaussian_amplitude,
        args.continuation_optimizer,
        args.output,
        args.hidden_dim,
        args.activation,
        args.spatial_scale,
        args.damage_output,
        args.d_init,
        args.staged_output_iterations,
        args.adaptive_history,
        args.max_history_continuation_steps,
    )


if __name__ == "__main__":
    _main()
