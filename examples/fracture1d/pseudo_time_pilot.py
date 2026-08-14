"""Fixed-history pseudo-time damage PINN diagnostic.

The pilot follows the implicit pseudo-time residual from arXiv:2604.23528v1
for one direct-baseline history field. It is deliberately separate from the
production load/staggered solver and uses a fixed pseudo-time step first.
"""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from examples.fracture1d import train
from examples.fracture1d.configs.train_base import get_config
from examples.fracture1d.frozen_pilot import _make_supervised_step, _target_fields
from examples.fracture1d.losses import (
    damage_residual_values,
    damage_pseudo_time_residual,
    loss_damage_pseudo_time,
)
from examples.fracture1d.nonlocal_baseline import solve_pseudo_time_damage


def _make_pseudo_time_step(
    x_all,
    D,
    N_l,
    N_G,
    residual_norm,
    residual_scaling,
    optimizer,
):
    """Create one JIT optimizer update for one implicit pseudo-time step."""
    @eqx.filter_jit
    def step(net, opt_state, d_previous, H_points, tau):
        def objective(n):
            return loss_damage_pseudo_time(
                n,
                d_previous,
                x_all,
                D,
                N_l,
                N_G,
                H_points,
                tau,
                residual_norm,
                residual_scaling,
            )

        loss, gradient = eqx.filter_value_and_grad(objective)(net)
        updates, opt_state = optimizer.update(gradient, opt_state, net)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss

    return step


def _make_pseudo_time_check(
    x_all,
    D,
    N_l,
    N_G,
    residual_scaling,
):
    """Create fixed-shape diagnostics for raw and pseudo-time residuals."""
    @eqx.filter_jit
    def check(net, d_previous, H_points, tau):
        d = train._net_d(net, x_all)
        raw_residual = train.damage_residual(
            net, x_all, D, N_l, N_G, H_points
        )
        raw_absolute = jnp.abs(raw_residual)
        scaled_absolute = raw_absolute / train.damage_residual_scale(
            D, N_l, N_G, H_points
        )
        pseudo_residual = damage_pseudo_time_residual(
            net,
            d_previous,
            x_all,
            D,
            N_l,
            N_G,
            H_points,
            tau,
            residual_scaling,
        )
        return (
            jnp.mean(raw_absolute),
            jnp.max(raw_absolute),
            jnp.mean(scaled_absolute),
            jnp.max(scaled_absolute),
            jnp.mean(pseudo_residual**2),
            jnp.max(jnp.abs(d - d_previous)),
        )

    return check


def _direct_scaled_residual(residual, D, N_l, N_G, H_points):
    """Return Jacobi-scaled direct residual diagnostics outside JIT."""
    factor = 2.0 * N_l / N_G
    scale = 1.0 + N_l**2 * np.asarray(D.diagonal) + factor * H_points
    absolute = np.abs(np.asarray(residual)) / scale
    return float(np.mean(absolute)), float(np.max(absolute))


def run_pseudo_time_pilot(
    config,
    output_path,
    initial_load,
    target_load,
    warm_start_steps,
    max_steps,
    check_every,
    record_every,
    tau,
    tau_mode,
    tau_gain,
    tau_beta,
    tau_min,
    tau_max,
    tau_update_every,
    residual_scaling,
    spatial_scale,
):
    """Run a fixed-history PINN pseudo-time solve against a direct reference."""
    if not 0.0 <= initial_load < target_load:
        raise ValueError("Require 0 <= initial_load < target_load")
    for name, value in (
        ("warm_start_steps", warm_start_steps),
        ("max_steps", max_steps),
        ("check_every", check_every),
        ("record_every", record_every),
    ):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive and finite")
    if residual_scaling not in ("none", "jacobi"):
        raise ValueError("residual_scaling must be 'none' or 'jacobi'")
    if tau_mode not in ("fixed", "adaptive"):
        raise ValueError("tau_mode must be 'fixed' or 'adaptive'")
    if tau_mode == "adaptive":
        for name, value in (
            ("tau_gain", tau_gain),
            ("tau_beta", tau_beta),
            ("tau_min", tau_min),
            ("tau_max", tau_max),
        ):
            if value is None or not np.isfinite(value):
                raise ValueError(f"{name} is required and must be finite")
        if tau_gain <= 0.0:
            raise ValueError("tau_gain must be positive")
        if not 0.0 < tau_beta <= 1.0:
            raise ValueError("tau_beta must lie in (0, 1]")
        if tau_min <= 0.0 or tau_max < tau_min:
            raise ValueError("Require 0 < tau_min <= tau_max")
        if tau < tau_min or tau > tau_max:
            raise ValueError("initial tau must lie within [tau_min, tau_max]")
        if (
            isinstance(tau_update_every, (bool, np.bool_))
            or int(tau_update_every) != tau_update_every
            or tau_update_every < 1
        ):
            raise ValueError("tau_update_every must be a positive integer")
    if not np.isfinite(spatial_scale) or spatial_scale <= 0.0:
        raise ValueError("spatial_scale must be positive and finite")

    config.arch.spatial_scale = float(spatial_scale)
    config.loss.objective = "residual"
    config.loss.residual_norm = "mse"
    config.loss.residual_scaling = residual_scaling
    train._validate_config(config)
    jax.config.update("jax_enable_x64", config.train.dtype == "float64")
    jax.config.update("jax_default_matmul_precision", config.train.matmul_precision)

    x_all, _ = train._build_grid(config)
    x_eval = np.asarray(x_all[:, 0])
    D = train._build_D(config, x_all)
    operator = D.to_scipy()
    damage_eval = train._make_damage_eval(x_all)
    residual_check = train._make_residual_check(
        x_all,
        D,
        config.physical.l0 / config.geometry.L0,
        config.physical.Gc / (config.physical.E * config.geometry.L0),
    )
    N_l = config.physical.l0 / config.geometry.L0
    N_G = config.physical.Gc / (config.physical.E * config.geometry.L0)
    d_initial_target, _ = _target_fields(config, x_eval, initial_load)
    d_target, H_target_np = _target_fields(config, x_eval, target_load)
    H_target = jnp.asarray(H_target_np, dtype=x_all.dtype)

    optimizer = train._build_optimizer(config)
    net = train._build_network(config, jr.PRNGKey(42))
    warm_step = _make_supervised_step(optimizer, x_all, d_initial_target)
    warm_state = optimizer.init(eqx.filter(net, eqx.is_array))
    for _ in range(int(warm_start_steps)):
        net, warm_state, _ = warm_step(net, warm_state)
    d_warm = np.asarray(damage_eval(net))

    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    pseudo_step = _make_pseudo_time_step(
        x_all,
        D,
        N_l,
        N_G,
        "mse",
        residual_scaling,
        optimizer,
    )
    pseudo_check = _make_pseudo_time_check(
        x_all, D, N_l, N_G, residual_scaling
    )
    tau_current = float(tau)
    d_previous = damage_eval(net)
    direct_previous = np.asarray(d_initial_target, dtype=np.float64)

    fixed_residual_mae = []
    fixed_residual_inf = []
    fixed_residual_scaled_mae = []
    fixed_residual_scaled_inf = []
    pseudo_loss = []
    pseudo_damage_change = []
    direct_residual_mae = []
    direct_residual_inf = []
    direct_residual_scaled_mae = []
    direct_residual_scaled_inf = []
    direct_damage_mae = []
    direct_damage_inf = []
    pinn_damage_mae = []
    pinn_damage_inf = []
    tau_history = []
    tau_estimate_history = []
    delta_damage_norm_history = []
    delta_residual_norm_history = []
    recorded_steps = [0]
    pinn_snapshots = [d_warm.copy()]
    direct_snapshots = [direct_previous.copy()]
    d_current = d_previous
    direct_current = direct_previous
    converged = False
    converged_step = int(max_steps)
    metrics = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    final_metrics = None

    for step_index in range(1, int(max_steps) + 1):
        tau_jax = jnp.asarray(tau_current, dtype=x_all.dtype)
        net, opt_state, _ = pseudo_step(
            net,
            opt_state,
            d_previous,
            H_target,
            tau_jax,
        )
        d_current = damage_eval(net)
        direct_current, direct_mae, direct_inf = solve_pseudo_time_damage(
            D,
            N_l,
            N_G,
            H_target_np,
            direct_previous,
            tau_current,
            operator,
        )
        direct_previous = direct_current
        metrics = tuple(
            map(float, pseudo_check(net, d_previous, H_target, tau_jax))
        )
        (
            current_mae,
            current_inf,
            current_scaled_mae,
            current_scaled_inf,
            current_pseudo_loss,
            current_damage_change,
        ) = metrics
        d_current_np = np.asarray(d_current)
        d_previous_np = np.asarray(d_previous)
        raw_current = np.asarray(
            damage_residual_values(d_current, D, N_l, N_G, H_target)
        )
        raw_previous = np.asarray(
            damage_residual_values(d_previous, D, N_l, N_G, H_target)
        )
        delta_damage_norm = float(np.linalg.norm(d_current_np - d_previous_np))
        delta_residual_norm = float(np.linalg.norm(raw_current - raw_previous))
        tau_estimate = np.nan
        tau_next = tau_current
        if tau_mode == "adaptive" and step_index % int(tau_update_every) == 0:
            # The epsilon only prevents division by zero in the finite-
            # difference inverse-Jacobian estimate.
            tau_estimate = tau_gain * delta_damage_norm / (
                delta_residual_norm + 1.0e-12
            )
            tau_next = float(np.clip(
                (1.0 - tau_beta) * tau_current + tau_beta * tau_estimate,
                tau_min,
                tau_max,
            ))
        fixed_residual_mae.append(current_mae)
        fixed_residual_inf.append(current_inf)
        fixed_residual_scaled_mae.append(current_scaled_mae)
        fixed_residual_scaled_inf.append(current_scaled_inf)
        pseudo_loss.append(current_pseudo_loss)
        pseudo_damage_change.append(current_damage_change)
        direct_residual_mae.append(direct_mae)
        direct_residual_inf.append(direct_inf)
        direct_scaled_mae, direct_scaled_inf = _direct_scaled_residual(
            direct_current
            + N_l**2 * (operator @ direct_current)
            - (2.0 * N_l / N_G) * (1.0 - direct_current) * H_target_np,
            D,
            N_l,
            N_G,
            H_target_np,
        )
        direct_residual_scaled_mae.append(direct_scaled_mae)
        direct_residual_scaled_inf.append(direct_scaled_inf)
        direct_damage_mae.append(float(np.mean(np.abs(direct_current - d_target))))
        direct_damage_inf.append(float(np.max(np.abs(direct_current - d_target))))
        pinn_damage_mae.append(float(np.mean(np.abs(d_current_np - d_target))))
        pinn_damage_inf.append(float(np.max(np.abs(d_current_np - d_target))))
        tau_history.append(tau_current)
        tau_estimate_history.append(tau_estimate)
        delta_damage_norm_history.append(delta_damage_norm)
        delta_residual_norm_history.append(delta_residual_norm)

        if step_index % int(record_every) == 0 or step_index == int(max_steps):
            recorded_steps.append(step_index)
            pinn_snapshots.append(d_current_np.copy())
            direct_snapshots.append(direct_current.copy())

        if step_index % int(check_every) == 0 or step_index == int(max_steps):
            if residual_scaling == "jacobi":
                active_mae = current_scaled_mae
                active_inf = current_scaled_inf
                atol_mae = config.train.atol_damage_residual_scaled
                atol_inf = config.train.atol_damage_residual_scaled_inf
            else:
                active_mae = current_mae
                active_inf = current_inf
                atol_mae = config.train.atol_damage_residual
                atol_inf = config.train.atol_damage_residual_inf
            if active_mae <= atol_mae and active_inf <= atol_inf:
                converged = True
                converged_step = step_index
                final_metrics = metrics
                break

        d_previous = d_current
        tau_current = tau_next
        if step_index % int(record_every) == 0 or step_index == int(max_steps):
            print(
                f"step={step_index} tau={tau_history[-1]:.3e} "
                f"fixed_scaled_inf={current_scaled_inf:.3e} "
                f"pinn_damage_inf={pinn_damage_inf[-1]:.3e} "
                f"direct_damage_inf={direct_damage_inf[-1]:.3e}"
            )

    if final_metrics is None:
        final_metrics = metrics
    d_pinn_final = np.asarray(d_current)
    d_direct_final = np.asarray(direct_previous)
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x_eval,
        initial_load=float(initial_load),
        target_load=float(target_load),
        H=H_target_np,
        d_target=d_target,
        d_warm=d_warm,
        d_pinn_final=d_pinn_final,
        d_direct_final=d_direct_final,
        d_pinn=np.asarray(pinn_snapshots),
        d_direct=np.asarray(direct_snapshots),
        recorded_steps=np.asarray(recorded_steps),
        fixed_residual_mae=np.asarray(fixed_residual_mae),
        fixed_residual_inf=np.asarray(fixed_residual_inf),
        fixed_residual_scaled_mae=np.asarray(fixed_residual_scaled_mae),
        fixed_residual_scaled_inf=np.asarray(fixed_residual_scaled_inf),
        pseudo_loss=np.asarray(pseudo_loss),
        pseudo_damage_change=np.asarray(pseudo_damage_change),
        direct_residual_mae=np.asarray(direct_residual_mae),
        direct_residual_inf=np.asarray(direct_residual_inf),
        direct_residual_scaled_mae=np.asarray(direct_residual_scaled_mae),
        direct_residual_scaled_inf=np.asarray(direct_residual_scaled_inf),
        direct_damage_mae=np.asarray(direct_damage_mae),
        direct_damage_inf=np.asarray(direct_damage_inf),
        pinn_damage_mae=np.asarray(pinn_damage_mae),
        pinn_damage_inf=np.asarray(pinn_damage_inf),
        converged=bool(converged),
        converged_step=int(converged_step),
        final_pseudo_metrics=np.asarray(final_metrics),
        tau_initial=float(tau),
        tau_final=float(tau_current),
        tau_mode=tau_mode,
        tau_gain=np.nan if tau_gain is None else float(tau_gain),
        tau_beta=np.nan if tau_beta is None else float(tau_beta),
        tau_min=np.nan if tau_min is None else float(tau_min),
        tau_max=np.nan if tau_max is None else float(tau_max),
        tau_update_every=-1 if tau_update_every is None else int(tau_update_every),
        tau_history=np.asarray(tau_history),
        tau_estimate=np.asarray(tau_estimate_history),
        delta_damage_norm=np.asarray(delta_damage_norm_history),
        delta_residual_norm=np.asarray(delta_residual_norm_history),
        residual_scaling=residual_scaling,
        warm_start_steps=int(warm_start_steps),
        max_steps=int(max_steps),
        check_every=int(check_every),
        record_every=int(record_every),
        spatial_scale=float(spatial_scale),
        dtype=config.train.dtype,
    )
    return output_path


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-load", type=float, required=True)
    parser.add_argument("--target-load", type=float, required=True)
    parser.add_argument("--warm-start-steps", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--check-every", type=int, required=True)
    parser.add_argument("--record-every", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--tau-mode", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--tau-gain", type=float)
    parser.add_argument("--tau-beta", type=float)
    parser.add_argument("--tau-min", type=float)
    parser.add_argument("--tau-max", type=float)
    parser.add_argument("--tau-update-every", type=int)
    parser.add_argument("--scaling", choices=("none", "jacobi"), required=True)
    parser.add_argument("--spatial-scale", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_pseudo_time_pilot(
        get_config(),
        args.output,
        args.initial_load,
        args.target_load,
        args.warm_start_steps,
        args.max_steps,
        args.check_every,
        args.record_every,
        args.tau,
        args.tau_mode,
        args.tau_gain,
        args.tau_beta,
        args.tau_min,
        args.tau_max,
        args.tau_update_every,
        args.scaling,
        args.spatial_scale,
    )


if __name__ == "__main__":
    _main()
