"""Fixed-history damage PINN diagnostics near the nonlocal localization point.

Each case solves one direct-baseline history field from a fresh network and a
fresh optimizer state. The diagnostic deliberately does not run load continuation;
it separates fixed-history optimization/representation error from staggered
history coupling.
"""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import solve_triangular
from jax.scipy.sparse.linalg import cg
import numpy as np

from examples.fracture1d import train
from examples.fracture1d.configs.train_base import get_config
from examples.fracture1d.local_enrichment import (
    make_hat_basis,
    solve_local_damage_correction,
)
from examples.fracture1d.nonlocal_baseline import (
    build_fixed_history_system,
    solve_fixed_history_defect,
)


def _target_fields(config, x_eval, load):
    """Load one exact direct-baseline frozen history and damage field."""
    reference = train._load_reference(
        config.baseline.data_path,
        config,
        config.baseline,
    )
    x_reference = np.asarray(reference["x"])
    x_history = np.asarray(reference["x_H"])
    snap_u_bar = np.asarray(reference["snap_u_bar"])
    index = int(np.argmin(np.abs(snap_u_bar - load)))
    if not np.isclose(snap_u_bar[index], load, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"Fixed-history diagnostic load {load:.8e} has no exact baseline snapshot; "
            f"nearest is {snap_u_bar[index]:.8e}"
        )
    if len(x_eval) == len(x_reference) and np.allclose(
        x_eval, x_reference, rtol=1.0e-7, atol=1.0e-8
    ):
        d_target = np.asarray(reference["d"][index])
    else:
        d_target = np.interp(x_eval, x_reference, reference["d"][index])
    if len(x_eval) == len(x_history) and np.allclose(
        x_eval, x_history, rtol=1.0e-7, atol=1.0e-8
    ):
        H_target = np.asarray(reference["H"][index])
    else:
        H_target = np.interp(x_eval, x_history, reference["H"][index])
    return np.asarray(d_target), np.asarray(H_target)


def _select_hard_points(net, x_all, D, N_l, N_G, H_points, count):
    """Select current largest scaled residual rows without reference data."""
    residual = np.asarray(train.damage_residual(net, x_all, D, N_l, N_G, H_points))
    scale = np.asarray(train.damage_residual_scale(D, N_l, N_G, H_points))
    return jnp.asarray(
        np.argsort(np.abs(residual / scale))[-int(count):], dtype=jnp.int32
    )


def _solve_output_head_least_squares(net, x_all, D, N_l, N_G, H_points):
    """Minimize the scaled fixed-history residual over a frozen MLP trunk."""
    if not hasattr(net, "output") or net.output is None:
        raise ValueError("output-head least squares requires a plain output layer")
    if net.output.weight.shape[0] != 1 or net.output.bias.shape != (1,):
        raise ValueError("output-head least squares requires one scalar output")
    H_np = np.asarray(H_points)
    if H_np.shape != (len(x_all),) or not np.all(np.isfinite(H_np)):
        raise ValueError("H_points must be a finite vector on the full grid")

    head = jnp.concatenate((net.output.weight.reshape(-1), net.output.bias))
    weight_size = net.output.weight.size

    def evaluate(parameters):
        candidate = eqx.tree_at(
            lambda n: (n.output.weight, n.output.bias),
            net,
            (
                parameters[:weight_size].reshape(net.output.weight.shape),
                parameters[weight_size:].reshape(net.output.bias.shape),
            ),
        )
        return train._net_d(candidate, x_all)

    damage_basis = np.asarray(jax.jacfwd(evaluate)(head), dtype=np.float64)
    beta = 2.0 * float(N_l) / float(N_G)
    system_basis = (
        damage_basis
        + float(N_l) ** 2 * (D.to_scipy() @ damage_basis)
        + beta * H_np[:, None] * damage_basis
    )
    scale = np.asarray(
        train.damage_residual_scale(D, N_l, N_G, H_points), dtype=np.float64
    )
    scaled_basis = system_basis / scale[:, None]
    scaled_rhs = beta * H_np / scale
    parameters, _, rank, singular_values = np.linalg.lstsq(
        scaled_basis, scaled_rhs, rcond=None
    )
    updated = eqx.tree_at(
        lambda n: (n.output.weight, n.output.bias),
        net,
        (
            jnp.asarray(
                parameters[:weight_size].reshape(net.output.weight.shape),
                dtype=net.output.weight.dtype,
            ),
            jnp.asarray(
                parameters[weight_size:].reshape(net.output.bias.shape),
                dtype=net.output.bias.dtype,
            ),
        ),
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == scaled_basis.shape[1] and singular_values[-1] > 0.0
        else np.inf
    )
    return updated, int(rank), condition, np.asarray(singular_values)


def _validate_lm_controls(
    max_iterations,
    initial_damping,
    damping_increase,
    damping_decrease,
    rho_accept,
    rho_low,
    rho_high,
    cg_tolerance,
    cg_max_iterations,
    max_step_norm,
    max_rejections,
):
    """Validate explicit controls for the diagnostic-only LM refinement."""
    for name, value in (
        ("lm_max_iterations", max_iterations),
        ("lm_cg_max_iterations", cg_max_iterations),
        ("lm_max_rejections", max_rejections),
    ):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("lm_initial_damping", initial_damping),
        ("lm_damping_increase", damping_increase),
        ("lm_damping_decrease", damping_decrease),
        ("lm_rho_accept", rho_accept),
        ("lm_rho_low", rho_low),
        ("lm_rho_high", rho_high),
        ("lm_cg_tolerance", cg_tolerance),
        ("lm_max_step_norm", max_step_norm),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if damping_increase <= 1.0:
        raise ValueError("lm_damping_increase must be greater than one")
    if not 0.0 < damping_decrease < 1.0:
        raise ValueError("lm_damping_decrease must lie in (0, 1)")
    if not 0.0 < rho_accept <= rho_low < rho_high <= 1.0:
        raise ValueError(
            "LM rho controls must satisfy 0 < accept <= low < high <= 1"
        )


def _build_fixed_history_defect_factor(operator, N_l, N_G, H_points):
    """Build a dense Cholesky factor for the exact fixed-history defect map."""
    system, _, _ = build_fixed_history_system(operator, N_l, N_G, H_points)
    dense_system = np.asarray(system.toarray(), dtype=np.float64)
    factor = np.linalg.cholesky(dense_system)
    return jnp.asarray(factor)


def _lm_refine_full_network(
    net,
    x_all,
    D,
    N_l,
    N_G,
    H_points,
    residual_scaling,
    residual_mode,
    defect_factor,
    max_iterations,
    initial_damping,
    damping_increase,
    damping_decrease,
    rho_accept,
    rho_low,
    rho_high,
    cg_tolerance,
    cg_max_iterations,
    max_step_norm,
    max_rejections,
):
    """Run target-free matrix-free LM steps on the full network parameters."""
    _validate_lm_controls(
        max_iterations,
        initial_damping,
        damping_increase,
        damping_decrease,
        rho_accept,
        rho_low,
        rho_high,
        cg_tolerance,
        cg_max_iterations,
        max_step_norm,
        max_rejections,
    )
    if residual_scaling not in ("none", "jacobi"):
        raise ValueError("residual_scaling must be 'none' or 'jacobi'")
    if residual_mode not in ("scaled_residual", "defect"):
        raise ValueError("residual_mode must be 'scaled_residual' or 'defect'")
    if residual_mode == "defect":
        if defect_factor is None:
            raise ValueError("defect residual mode requires a Cholesky factor")
        if np.asarray(defect_factor).shape != (len(H_points), len(H_points)):
            raise ValueError("defect factor shape must match H_points")

    dynamic, static = eqx.partition(net, eqx.is_array)
    parameters, unravel = ravel_pytree(dynamic)

    def unpack(values):
        return eqx.combine(unravel(values), static)

    def residual_from_parameters(values):
        candidate = unpack(values)
        residual = train.damage_residual(
            candidate, x_all, D, N_l, N_G, H_points
        )
        if residual_mode == "defect":
            forward_solution = solve_triangular(
                defect_factor, residual, lower=True
            )
            residual = solve_triangular(
                defect_factor,
                forward_solution,
                trans=1,
                lower=True,
            )
        elif residual_scaling == "jacobi":
            residual = residual / train.damage_residual_scale(
                D, N_l, N_G, H_points
            )
        return residual

    residual_from_parameters = jax.jit(residual_from_parameters)
    residual_values = residual_from_parameters(parameters)
    current_loss = 0.5 * jnp.vdot(residual_values, residual_values)
    damping = float(initial_damping)
    accepted_steps = 0
    rejected_steps = 0
    consecutive_rejections = 0
    loss_history = [float(current_loss)]
    damping_history = [damping]
    rho_history = []
    accepted_history = []

    for _ in range(int(max_iterations)):
        residual_values, jvp_fn = jax.linearize(
            residual_from_parameters, parameters
        )
        _, vjp_fn = jax.vjp(residual_from_parameters, parameters)
        damping_value = jnp.asarray(damping, dtype=parameters.dtype)

        def normal_action(vector):
            jacobian_transpose_vector = vjp_fn(vector)[0]
            return jvp_fn(jacobian_transpose_vector) + damping_value * vector

        dual, _ = cg(
            normal_action,
            residual_values,
            tol=float(cg_tolerance),
            atol=0.0,
            maxiter=int(cg_max_iterations),
        )
        step = -vjp_fn(dual)[0]
        step_norm = float(jnp.linalg.norm(step))
        if not np.isfinite(step_norm):
            candidate_parameters = parameters
            predicted_reduction = -np.inf
            actual_reduction = -np.inf
            rho = -np.inf
        else:
            if step_norm > max_step_norm:
                step = step * (max_step_norm / step_norm)
            linearized_residual = residual_values + jvp_fn(step)
            predicted_reduction = float(
                0.5 * (
                    jnp.vdot(residual_values, residual_values)
                    - jnp.vdot(linearized_residual, linearized_residual)
                )
            )
            candidate_parameters = parameters + step
            candidate_residual = residual_from_parameters(candidate_parameters)
            candidate_loss = 0.5 * jnp.vdot(
                candidate_residual, candidate_residual
            )
            actual_reduction = float(current_loss - candidate_loss)
            rho = (
                actual_reduction / predicted_reduction
                if predicted_reduction > 0.0
                else -np.inf
            )

        accept = (
            np.isfinite(rho)
            and rho >= rho_accept
            and actual_reduction > 0.0
        )
        rho_history.append(float(rho))
        accepted_history.append(bool(accept))
        if accept:
            parameters = candidate_parameters
            residual_values = residual_from_parameters(parameters)
            current_loss = 0.5 * jnp.vdot(residual_values, residual_values)
            accepted_steps += 1
            consecutive_rejections = 0
            if rho >= rho_high:
                damping *= damping_decrease
            elif rho < rho_low:
                damping *= damping_increase
        else:
            rejected_steps += 1
            consecutive_rejections += 1
            damping *= damping_increase
            if consecutive_rejections >= int(max_rejections):
                break
        loss_history.append(float(current_loss))
        damping_history.append(float(damping))

    diagnostics = {
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "damping_history": np.asarray(damping_history, dtype=np.float64),
        "rho_history": np.asarray(rho_history, dtype=np.float64),
        "accepted_history": np.asarray(accepted_history, dtype=bool),
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "iterations": int(len(rho_history)),
        "final_damping": float(damping),
    }
    return unpack(parameters), diagnostics


def run_fixed_history_diagnostic(
    config,
    output_path,
    objective,
    residual_norm,
    residual_scaling,
    max_iterations,
    warm_start_steps,
    optimizer_mode,
    loads,
    spatial_scale,
    activation,
    damage_output,
    d_init,
    optimizer_name,
    hat_update,
    hidden_dim,
    hard_point_mode="dynamic",
    output_head_lstsq=False,
    full_network_lm=False,
    lm_residual_mode="scaled_residual",
    lm_max_iterations=None,
    lm_initial_damping=None,
    lm_damping_increase=None,
    lm_damping_decrease=None,
    lm_rho_accept=None,
    lm_rho_low=None,
    lm_rho_high=None,
    lm_cg_tolerance=None,
    lm_cg_max_iterations=None,
    lm_max_step_norm=None,
    lm_max_rejections=None,
):
    """Run the three fixed-history cases and save all diagnostic fields."""
    if objective not in ("residual", "energy"):
        raise ValueError("objective must be 'residual' or 'energy'")
    if residual_norm not in ("mae", "mse"):
        raise ValueError(f"residual_norm must be 'mae' or 'mse', got {residual_norm!r}")
    if residual_scaling not in ("none", "jacobi"):
        raise ValueError("residual_scaling must be 'none' or 'jacobi'")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if int(warm_start_steps) != warm_start_steps or warm_start_steps < 0:
        raise ValueError("warm_start_steps must be a non-negative integer")
    if optimizer_mode not in ("independent", "reset", "carry"):
        raise ValueError(
            "optimizer_mode must be 'independent', 'reset', or 'carry'"
        )
    loads = np.asarray(loads, dtype=np.float64)
    if loads.ndim != 1 or len(loads) < 1 or not np.all(np.isfinite(loads)):
        raise ValueError("loads must be a non-empty finite one-dimensional array")
    if not np.isfinite(spatial_scale) or spatial_scale <= 0.0:
        raise ValueError("spatial_scale must be positive and finite")
    if activation not in ("tanh", "relu", "gelu", "silu", "swish"):
        raise ValueError(f"Unknown activation {activation!r}")
    if damage_output not in ("linear", "exp_square"):
        raise ValueError(f"Unknown damage_output {damage_output!r}")
    if not np.isfinite(d_init):
        raise ValueError("d_init must be finite")
    if optimizer_name not in ("adam", "rprop"):
        raise ValueError("optimizer_name must be 'adam' or 'rprop'")
    if int(hidden_dim) != hidden_dim or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    if hat_update not in ("adam", "exact"):
        raise ValueError("hat_update must be 'adam' or 'exact'")
    if hard_point_mode not in ("dynamic", "fixed", "smooth"):
        raise ValueError("hard_point_mode must be 'dynamic', 'fixed', or 'smooth'")
    if output_head_lstsq and (
        objective != "residual"
        or residual_norm != "mse"
        or residual_scaling != "jacobi"
        or damage_output != "linear"
        or config.arch.type != "mlp"
        or config.arch.local_equilibrium
        or config.arch.use_local_gaussian
        or config.arch.use_local_hat_enrichment
        or config.loss.hard_point_weight != 0.0
    ):
        raise ValueError(
            "output-head least squares requires a plain MLP, linear output, "
            "Jacobi-scaled residual MSE, and zero hard-point weight"
        )
    if hat_update == "exact":
        if not config.arch.use_local_hat_enrichment:
            raise ValueError("exact hat_update requires local hat enrichment")
        if damage_output != "linear":
            raise ValueError("exact hat_update requires linear damage output")
        if config.arch.use_local_gaussian:
            raise ValueError("exact hat_update cannot wrap a local Gaussian")
    if config.arch.local_equilibrium and warm_start_steps:
        raise ValueError(
            "local_equilibrium diagnostic requires warm_start_steps=0; "
            "target supervision is not part of this diagnostic"
        )
    if full_network_lm:
        if (
            objective != "residual"
            or residual_norm != "mse"
            or not residual_scaling == "jacobi"
            or warm_start_steps != 0
            or output_head_lstsq
            or config.arch.local_equilibrium
            or config.arch.damage_output != "linear"
            or config.arch.use_local_gaussian
            or config.arch.use_local_hat_enrichment
            or config.loss.hard_point_weight != 0.0
        ):
            raise ValueError(
                "full-network LM requires target-free Jacobi-scaled residual MSE, "
                "linear output, no wrappers, and zero hard-point weight"
            )
        if lm_residual_mode not in ("scaled_residual", "defect"):
            raise ValueError(
                "lm_residual_mode must be 'scaled_residual' or 'defect'"
            )
        lm_values = (
            lm_max_iterations,
            lm_initial_damping,
            lm_damping_increase,
            lm_damping_decrease,
            lm_rho_accept,
            lm_rho_low,
            lm_rho_high,
            lm_cg_tolerance,
            lm_cg_max_iterations,
            lm_max_step_norm,
            lm_max_rejections,
        )
        if any(value is None for value in lm_values):
            raise ValueError(
                "all lm_* controls are required when full_network_lm is enabled"
            )
        _validate_lm_controls(*lm_values)

    config.loss.objective = objective
    config.loss.residual_norm = residual_norm
    config.loss.residual_scaling = residual_scaling
    config.arch.spatial_scale = float(spatial_scale)
    config.arch.activation = activation
    config.arch.damage_output = damage_output
    config.arch.d_init = float(d_init)
    config.arch.hidden_dim = int(hidden_dim)
    config.train.optimizer = optimizer_name
    config.loss.hard_point_mode = hard_point_mode
    config.arch.local_equilibrium = bool(config.arch.local_equilibrium)
    config.train.damage_iterations = int(max_iterations)
    train._validate_config(config)
    jax.config.update("jax_enable_x64", config.train.dtype == "float64")
    jax.config.update("jax_default_matmul_precision", config.train.matmul_precision)

    x_all, _ = train._build_grid(config)
    x_eval = np.asarray(x_all[:, 0])
    D = train._build_D(config, x_all)
    damage_eval = train._make_damage_eval(x_all, config.arch.local_equilibrium)
    residual_check = train._make_residual_check(
        x_all,
        D,
        config.physical.l0 / config.geometry.L0,
        config.physical.Gc / (config.physical.E * config.geometry.L0),
        config.arch.local_equilibrium,
    )
    loss_fn = train.make_loss_fn(
        x_all,
        D,
        config.physical.l0 / config.geometry.L0,
        config.physical.Gc / (config.physical.E * config.geometry.L0),
        objective,
        residual_norm,
        config.loss.residual_scaling,
        int(config.loss.hard_point_count),
        float(config.loss.hard_point_weight),
        hard_point_mode,
        float(config.loss.hard_point_temperature),
        bool(config.arch.local_equilibrium),
    )
    optimizer = train._build_optimizer(config)
    freeze_hat_coefficients = hat_update == "exact"
    step_fn = train._make_step(
        loss_fn,
        optimizer,
        freeze_hat_coefficients=freeze_hat_coefficients,
        freeze_spatial_embedding=not config.arch.train_spatial_embedding,
    )
    N_l = config.physical.l0 / config.geometry.L0
    N_G = config.physical.Gc / (config.physical.E * config.geometry.L0)

    predicted_damage = []
    target_damage = []
    target_history = []
    predicted_residual = []
    converged = []
    iterations = []
    final_loss = []
    residual_mae = []
    residual_inf = []
    residual_scaled_mae = []
    residual_scaled_inf = []
    pre_output_head_residual_mae = []
    pre_output_head_residual_inf = []
    pre_output_head_residual_scaled_mae = []
    pre_output_head_residual_scaled_inf = []
    output_head_rank = []
    output_head_condition = []
    output_head_singular_values = []
    target_residual_mae = []
    target_residual_inf = []
    fixed_history_defect = []
    fixed_history_defect_mae = []
    fixed_history_defect_inf = []
    fixed_history_defect_l2 = []
    fixed_history_linear_solve_residual_inf = []
    pre_refinement_defect_mae = []
    pre_refinement_defect_inf = []
    pre_refinement_defect_l2 = []
    lm_initial_loss = []
    lm_final_loss = []
    lm_accepted_steps = []
    lm_rejected_steps = []
    lm_iterations = []
    lm_final_damping = []
    warm_start_damage_mae = []
    warm_start_damage_inf = []
    warm_start_residual_mae = []
    warm_start_residual_inf = []
    warm_start_residual_scaled_mae = []
    warm_start_residual_scaled_inf = []
    net = None
    opt_state = None
    hat_basis = None
    hat_operator = None
    scipy_operator = D.to_scipy()
    hard_point_indices = None
    d_context = jnp.zeros((len(x_all),), dtype=x_all.dtype)
    if hat_update == "exact":
        center_index = int(
            round(float(config.imperfection.center) * (int(config.sample.Nx) - 1))
        )
        hat_basis, _ = make_hat_basis(
            x_eval,
            center_index,
            int(config.arch.local_hat_count),
        )
        hat_operator = scipy_operator

    for case_index, load in enumerate(loads):
        d_target_np, H_target_np = _target_fields(config, x_eval, load)
        H_target = jnp.asarray(H_target_np, dtype=x_all.dtype)
        if net is None or optimizer_mode == "independent":
            net = train._build_network(config, jr.PRNGKey(42 + case_index))
        if config.arch.local_equilibrium:
            d_context = jnp.zeros_like(H_target)
        warm_mae = np.inf
        warm_inf = np.inf
        warm_residual_mae = np.inf
        warm_residual_inf = np.inf
        did_warm_start = bool(warm_start_steps)
        if warm_start_steps:
            warm_step = _make_supervised_step(optimizer, x_all, d_target_np)
            warm_state = optimizer.init(eqx.filter(net, eqx.is_array))
            for _ in range(int(warm_start_steps)):
                net, warm_state, _ = warm_step(net, warm_state)
            warm_damage = np.asarray(damage_eval(net))
            warm_mae = float(np.mean(np.abs(warm_damage - d_target_np)))
            warm_inf = float(np.max(np.abs(warm_damage - d_target_np)))
            (
                warm_residual_mae,
                warm_residual_inf,
                warm_residual_scaled_mae,
                warm_residual_scaled_inf,
            ) = map(float, residual_check(net, H_target, d_context))
        else:
            warm_residual_scaled_mae = np.inf
            warm_residual_scaled_inf = np.inf
        if (
            opt_state is None
            or optimizer_mode in ("independent", "reset")
            or did_warm_start
        ):
            opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
        if config.arch.local_equilibrium:
            context_points = train._damage_context(
                H_target, d_context, N_l, N_G
            )
        else:
            context_points = None
        lm_defect_factor = None
        if full_network_lm and lm_residual_mode == "defect":
            lm_defect_factor = jnp.asarray(
                _build_fixed_history_defect_factor(
                    scipy_operator,
                    N_l,
                    N_G,
                    H_target_np,
                ),
                dtype=x_all.dtype,
            )
        if hat_update == "exact":
            d_current = np.asarray(damage_eval(net, context_points))
            _, coefficient_delta, _, _, _, _ = solve_local_damage_correction(
                hat_operator,
                float(N_l),
                float(N_G),
                H_target_np,
                d_current,
                hat_basis,
                residual_scaling,
            )
            coefficients = np.asarray(train._hat_coefficients(net))
            net = train._set_hat_coefficients(
                net,
                jnp.asarray(coefficients + coefficient_delta, dtype=x_all.dtype),
            )
        if hard_point_mode == "fixed" and config.loss.hard_point_weight > 0.0:
            hard_point_indices = _select_hard_points(
                net,
                x_all,
                D,
                N_l,
                N_G,
                H_target,
                config.loss.hard_point_count,
            )
        w_ema = jnp.ones((1,), dtype=x_all.dtype)
        bias = jnp.ones((1,), dtype=x_all.dtype)
        update_w = jnp.array(False)
        case_converged = False
        case_loss = np.inf
        case_mae = np.inf
        case_inf = np.inf
        case_scaled_mae = np.inf
        case_scaled_inf = np.inf
        iteration = 0

        for iteration in range(1, int(max_iterations) + 1):
            net, opt_state, loss_value, w_ema = step_fn(
                net,
                opt_state,
                w_ema,
                bias,
                update_w,
                x_all,
                H_target,
                hard_point_indices,
                d_context,
            )
            case_loss = float(loss_value)
            if iteration % config.train.damage_check_every != 0 and iteration != max_iterations:
                continue
            if hat_update == "exact":
                d_current = np.asarray(damage_eval(net, context_points))
                _, coefficient_delta, _, _, _, _ = solve_local_damage_correction(
                    hat_operator,
                    float(N_l),
                    float(N_G),
                    H_target_np,
                    d_current,
                    hat_basis,
                    residual_scaling,
                )
                coefficients = np.asarray(train._hat_coefficients(net))
                net = train._set_hat_coefficients(
                    net,
                    jnp.asarray(
                        coefficients + coefficient_delta,
                        dtype=x_all.dtype,
                    ),
                )
            (
                case_mae,
                case_inf,
                case_scaled_mae,
                case_scaled_inf,
                ) = map(float, residual_check(net, H_target, d_context))
            if not np.isfinite(
                case_loss + case_mae + case_inf + case_scaled_mae + case_scaled_inf
            ):
                raise FloatingPointError(
                    f"Fixed-history diagnostic became non-finite at u_bar={load:.8e}"
                )
            if residual_scaling == "jacobi":
                accept_mae = case_scaled_mae
                accept_inf = case_scaled_inf
                accept_atol_mae = config.train.atol_damage_residual_scaled
                accept_atol_inf = config.train.atol_damage_residual_scaled_inf
            else:
                accept_mae = case_mae
                accept_inf = case_inf
                accept_atol_mae = config.train.atol_damage_residual
                accept_atol_inf = config.train.atol_damage_residual_inf
            if (
                accept_mae <= accept_atol_mae
                and accept_inf <= accept_atol_inf
            ):
                case_converged = True
                break
            if hard_point_mode == "fixed" and config.loss.hard_point_weight > 0.0:
                hard_point_indices = _select_hard_points(
                    net,
                    x_all,
                    D,
                    N_l,
                    N_G,
                    H_target,
                    config.loss.hard_point_count,
                )

        pre_output_head_residual_mae.append(case_mae)
        pre_output_head_residual_inf.append(case_inf)
        pre_output_head_residual_scaled_mae.append(case_scaled_mae)
        pre_output_head_residual_scaled_inf.append(case_scaled_inf)
        d_before_refinement_np = np.asarray(damage_eval(net, context_points))
        pre_defect_np, _, _ = solve_fixed_history_defect(
            scipy_operator,
            N_l,
            N_G,
            d_before_refinement_np,
            H_target_np,
        )
        pre_refinement_defect_mae.append(float(np.mean(np.abs(pre_defect_np))))
        pre_refinement_defect_inf.append(float(np.max(np.abs(pre_defect_np))))
        pre_refinement_defect_l2.append(float(np.linalg.norm(pre_defect_np)))
        lm_initial_loss_value = np.nan
        lm_final_loss_value = np.nan
        lm_accepted_steps_value = 0
        lm_rejected_steps_value = 0
        lm_iterations_value = 0
        lm_final_damping_value = np.nan
        if output_head_lstsq:
            net, head_rank, head_condition, head_singular_values = (
                _solve_output_head_least_squares(
                    net, x_all, D, N_l, N_G, H_target
                )
            )
            (
                case_mae,
                case_inf,
                case_scaled_mae,
                case_scaled_inf,
            ) = map(float, residual_check(net, H_target, d_context))
            scaled_residual = np.asarray(
                train.damage_residual(net, x_all, D, N_l, N_G, H_target)
                / train.damage_residual_scale(D, N_l, N_G, H_target)
            )
            case_loss = float(np.mean(scaled_residual**2))
            case_converged = (
                case_scaled_mae <= config.train.atol_damage_residual_scaled
                and case_scaled_inf
                <= config.train.atol_damage_residual_scaled_inf
            )
        else:
            head_rank = 0
            head_condition = np.nan
            head_singular_values = np.empty((0,), dtype=np.float64)
        if full_network_lm:
            net, lm_diagnostics = _lm_refine_full_network(
                net,
                x_all,
                D,
                N_l,
                N_G,
                H_target,
                residual_scaling,
                lm_residual_mode,
                lm_defect_factor,
                lm_max_iterations,
                lm_initial_damping,
                lm_damping_increase,
                lm_damping_decrease,
                lm_rho_accept,
                lm_rho_low,
                lm_rho_high,
                lm_cg_tolerance,
                lm_cg_max_iterations,
                lm_max_step_norm,
                lm_max_rejections,
            )
            lm_initial_loss_value = float(lm_diagnostics["loss_history"][0])
            lm_final_loss_value = float(lm_diagnostics["loss_history"][-1])
            lm_accepted_steps_value = int(lm_diagnostics["accepted_steps"])
            lm_rejected_steps_value = int(lm_diagnostics["rejected_steps"])
            lm_iterations_value = int(lm_diagnostics["iterations"])
            lm_final_damping_value = float(lm_diagnostics["final_damping"])
            (
                case_mae,
                case_inf,
                case_scaled_mae,
                case_scaled_inf,
            ) = map(float, residual_check(net, H_target, d_context))
            case_loss = lm_final_loss_value
            case_converged = (
                case_scaled_mae <= config.train.atol_damage_residual_scaled
                and case_scaled_inf
                <= config.train.atol_damage_residual_scaled_inf
            )
        output_head_rank.append(head_rank)
        output_head_condition.append(head_condition)
        output_head_singular_values.append(head_singular_values)

        d_pred_np = np.asarray(damage_eval(net, context_points))
        residual_np = np.asarray(
            train.damage_residual(
                net, x_all, D, N_l, N_G, H_target, context_points
            )
        )
        target_residual_np = (
            d_target_np
            + N_l**2 * np.asarray(D @ jnp.asarray(d_target_np, dtype=x_all.dtype))
            - (2.0 * N_l / N_G) * (1.0 - d_target_np) * H_target_np
        )
        defect_np, _, defect_solve_residual_np = solve_fixed_history_defect(
            scipy_operator,
            N_l,
            N_G,
            d_pred_np,
            H_target_np,
        )
        predicted_damage.append(d_pred_np)
        target_damage.append(d_target_np)
        target_history.append(H_target_np)
        predicted_residual.append(residual_np)
        converged.append(case_converged)
        iterations.append(iteration)
        final_loss.append(case_loss)
        residual_mae.append(case_mae)
        residual_inf.append(case_inf)
        residual_scaled_mae.append(case_scaled_mae)
        residual_scaled_inf.append(case_scaled_inf)
        target_residual_mae.append(float(np.mean(np.abs(target_residual_np))))
        target_residual_inf.append(float(np.max(np.abs(target_residual_np))))
        fixed_history_defect.append(defect_np)
        fixed_history_defect_mae.append(float(np.mean(np.abs(defect_np))))
        fixed_history_defect_inf.append(float(np.max(np.abs(defect_np))))
        fixed_history_defect_l2.append(float(np.linalg.norm(defect_np)))
        fixed_history_linear_solve_residual_inf.append(
            float(np.max(np.abs(defect_solve_residual_np)))
        )
        lm_initial_loss.append(lm_initial_loss_value)
        lm_final_loss.append(lm_final_loss_value)
        lm_accepted_steps.append(lm_accepted_steps_value)
        lm_rejected_steps.append(lm_rejected_steps_value)
        lm_iterations.append(lm_iterations_value)
        lm_final_damping.append(lm_final_damping_value)
        warm_start_damage_mae.append(warm_mae)
        warm_start_damage_inf.append(warm_inf)
        warm_start_residual_mae.append(warm_residual_mae)
        warm_start_residual_inf.append(warm_residual_inf)
        warm_start_residual_scaled_mae.append(warm_residual_scaled_mae)
        warm_start_residual_scaled_inf.append(warm_residual_scaled_inf)
        print(
            f"norm={residual_norm} u_bar={load:.8e} "
            f"warm_mae={warm_mae:.3e} warm_inf={warm_inf:.3e} "
            f"warm_residual_mae={warm_residual_mae:.3e} "
            f"warm_residual_inf={warm_residual_inf:.3e} "
            f"converged={case_converged} iterations={iteration} "
            f"loss={case_loss:.3e} residual_mae={case_mae:.3e} "
            f"residual_inf={case_inf:.3e} "
            f"scaled_mae={case_scaled_mae:.3e} "
            f"scaled_inf={case_scaled_inf:.3e} "
            f"max_d={np.max(d_pred_np):.6e} "
            f"target_max_d={np.max(d_target_np):.6e}"
        )

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x_eval,
        u_bar=loads,
        H=np.asarray(target_history),
        d_target=np.asarray(target_damage),
        d=np.asarray(predicted_damage),
        damage_residual=np.asarray(predicted_residual),
        converged=np.asarray(converged),
        iterations=np.asarray(iterations),
        loss=np.asarray(final_loss),
        damage_residual_mae=np.asarray(residual_mae),
        damage_residual_inf=np.asarray(residual_inf),
        damage_residual_scaled_mae=np.asarray(residual_scaled_mae),
        damage_residual_scaled_inf=np.asarray(residual_scaled_inf),
        pre_output_head_residual_mae=np.asarray(pre_output_head_residual_mae),
        pre_output_head_residual_inf=np.asarray(pre_output_head_residual_inf),
        pre_output_head_residual_scaled_mae=np.asarray(
            pre_output_head_residual_scaled_mae
        ),
        pre_output_head_residual_scaled_inf=np.asarray(
            pre_output_head_residual_scaled_inf
        ),
        output_head_lstsq=bool(output_head_lstsq),
        output_head_rank=np.asarray(output_head_rank),
        output_head_condition=np.asarray(output_head_condition),
        output_head_singular_values=np.asarray(output_head_singular_values),
        target_damage_residual_mae=np.asarray(target_residual_mae),
        target_damage_residual_inf=np.asarray(target_residual_inf),
        fixed_history_defect=np.asarray(fixed_history_defect),
        fixed_history_defect_mae=np.asarray(fixed_history_defect_mae),
        fixed_history_defect_inf=np.asarray(fixed_history_defect_inf),
        fixed_history_defect_l2=np.asarray(fixed_history_defect_l2),
        fixed_history_linear_solve_residual_inf=np.asarray(
            fixed_history_linear_solve_residual_inf
        ),
        pre_refinement_defect_mae=np.asarray(pre_refinement_defect_mae),
        pre_refinement_defect_inf=np.asarray(pre_refinement_defect_inf),
        pre_refinement_defect_l2=np.asarray(pre_refinement_defect_l2),
        lm_initial_loss=np.asarray(lm_initial_loss),
        lm_final_loss=np.asarray(lm_final_loss),
        lm_accepted_steps=np.asarray(lm_accepted_steps),
        lm_rejected_steps=np.asarray(lm_rejected_steps),
        lm_iterations=np.asarray(lm_iterations),
        lm_final_damping=np.asarray(lm_final_damping),
        full_network_lm=bool(full_network_lm),
        lm_residual_mode=lm_residual_mode,
        lm_max_iterations=-1 if lm_max_iterations is None else int(lm_max_iterations),
        lm_initial_damping=np.nan
        if lm_initial_damping is None
        else float(lm_initial_damping),
        lm_damping_increase=np.nan
        if lm_damping_increase is None
        else float(lm_damping_increase),
        lm_damping_decrease=np.nan
        if lm_damping_decrease is None
        else float(lm_damping_decrease),
        lm_rho_accept=np.nan if lm_rho_accept is None else float(lm_rho_accept),
        lm_rho_low=np.nan if lm_rho_low is None else float(lm_rho_low),
        lm_rho_high=np.nan if lm_rho_high is None else float(lm_rho_high),
        lm_cg_tolerance=np.nan
        if lm_cg_tolerance is None
        else float(lm_cg_tolerance),
        lm_cg_max_iterations=-1
        if lm_cg_max_iterations is None
        else int(lm_cg_max_iterations),
        lm_max_step_norm=np.nan
        if lm_max_step_norm is None
        else float(lm_max_step_norm),
        lm_max_rejections=-1
        if lm_max_rejections is None
        else int(lm_max_rejections),
        target_supervision_used=bool(warm_start_steps > 0),
        warm_start_damage_mae=np.asarray(warm_start_damage_mae),
        warm_start_damage_inf=np.asarray(warm_start_damage_inf),
        warm_start_residual_mae=np.asarray(warm_start_residual_mae),
        warm_start_residual_inf=np.asarray(warm_start_residual_inf),
        warm_start_residual_scaled_mae=np.asarray(warm_start_residual_scaled_mae),
        warm_start_residual_scaled_inf=np.asarray(warm_start_residual_scaled_inf),
        residual_norm=residual_norm,
        objective=objective,
        residual_scaling=residual_scaling,
        max_iterations=int(max_iterations),
        warm_start_steps=int(warm_start_steps),
        optimizer_mode=optimizer_mode,
        damage_check_every=int(config.train.damage_check_every),
        atol_damage_residual=float(config.train.atol_damage_residual),
        atol_damage_residual_inf=float(config.train.atol_damage_residual_inf),
        atol_damage_residual_scaled=float(
            config.train.atol_damage_residual_scaled
        ),
        atol_damage_residual_scaled_inf=float(
            config.train.atol_damage_residual_scaled_inf
        ),
        sample_nx=int(config.sample.Nx),
        delta=float(config.nl.delta),
        spatial_scale=float(spatial_scale),
        activation=activation,
        damage_output=damage_output,
        d_init=float(d_init),
        hidden_dim=int(config.arch.hidden_dim),
        local_hat_enrichment=bool(config.arch.use_local_hat_enrichment),
        local_hat_count=int(config.arch.local_hat_count),
        hat_update=hat_update,
        optimizer=optimizer_name,
        hard_point_count=int(config.loss.hard_point_count),
        hard_point_weight=float(config.loss.hard_point_weight),
        hard_point_mode=hard_point_mode,
        hard_point_temperature=float(config.loss.hard_point_temperature),
        history_mode=str(config.baseline.history_mode),
        history_state_space=(
            "DG0_cell"
            if config.baseline.history_mode == "cell_mass_lumped"
            else "nodal"
        ),
    )
    return output_path


def _make_supervised_step(optimizer, x_all, d_target):
    """Fit the direct-baseline damage only to test network representation."""
    target = jnp.asarray(d_target, dtype=x_all.dtype)

    @eqx.filter_jit
    def step(net, opt_state):
        def objective(n):
            damage = train._net_d(n, x_all)
            return jnp.mean((damage - target) ** 2)

        loss, gradient = eqx.filter_value_and_grad(objective)(net)
        updates, opt_state = optimizer.update(gradient, opt_state, net)
        return eqx.apply_updates(net, updates), opt_state, loss

    return step


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("residual", "energy"), required=True)
    parser.add_argument("--norm", choices=("mae", "mse"), required=True)
    parser.add_argument("--scaling", choices=("none", "jacobi"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warm-start-steps", type=int, required=True)
    parser.add_argument(
        "--optimizer-mode",
        choices=("independent", "reset", "carry"),
        required=True,
    )
    parser.add_argument("--loads", nargs="+", type=float, required=True)
    parser.add_argument("--spatial-scale", type=float, required=True)
    parser.add_argument(
        "--activation",
        choices=("tanh", "relu", "gelu", "silu", "swish"),
        required=True,
    )
    parser.add_argument(
        "--damage-output",
        choices=("linear", "exp_square"),
        required=True,
    )
    parser.add_argument("--d-init", type=float, required=True)
    parser.add_argument("--optimizer", choices=("adam", "rprop"), default="adam")
    parser.add_argument("--hat-update", choices=("adam", "exact"), default="adam")
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument(
        "--hard-point-mode", choices=("dynamic", "fixed"), default="dynamic"
    )
    parser.add_argument("--output-head-lstsq", action="store_true")
    parser.add_argument("--full-network-lm", action="store_true")
    parser.add_argument(
        "--lm-residual-mode",
        choices=("scaled_residual", "defect"),
        default="scaled_residual",
    )
    parser.add_argument("--lm-max-iterations", type=int, default=None)
    parser.add_argument("--lm-initial-damping", type=float, default=None)
    parser.add_argument("--lm-damping-increase", type=float, default=None)
    parser.add_argument("--lm-damping-decrease", type=float, default=None)
    parser.add_argument("--lm-rho-accept", type=float, default=None)
    parser.add_argument("--lm-rho-low", type=float, default=None)
    parser.add_argument("--lm-rho-high", type=float, default=None)
    parser.add_argument("--lm-cg-tolerance", type=float, default=None)
    parser.add_argument("--lm-cg-max-iterations", type=int, default=None)
    parser.add_argument("--lm-max-step-norm", type=float, default=None)
    parser.add_argument("--lm-max-rejections", type=int, default=None)
    args = parser.parse_args()
    config = get_config()
    run_fixed_history_diagnostic(
        config,
        args.output,
        args.objective,
        args.norm,
        args.scaling,
        args.iterations,
        args.warm_start_steps,
        args.optimizer_mode,
        args.loads,
        args.spatial_scale,
        args.activation,
        args.damage_output,
        args.d_init,
        args.optimizer,
        args.hat_update,
        args.hidden_dim,
        args.hard_point_mode,
        args.output_head_lstsq,
        args.full_network_lm,
        args.lm_residual_mode,
        args.lm_max_iterations,
        args.lm_initial_damping,
        args.lm_damping_increase,
        args.lm_damping_decrease,
        args.lm_rho_accept,
        args.lm_rho_low,
        args.lm_rho_high,
        args.lm_cg_tolerance,
        args.lm_cg_max_iterations,
        args.lm_max_step_norm,
        args.lm_max_rejections,
    )


if __name__ == "__main__":
    _main()
