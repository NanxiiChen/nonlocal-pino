"""Pure loss and condensed-mechanics functions for the 1D fracture PINN."""

import equinox as eqx
import jax
import jax.numpy as jnp


def _net_d(net, x_points, context_points=None):
    """Evaluate the scalar damage output on a point array."""
    if context_points is None:
        raw = jax.vmap(net, in_axes=(0, None))(x_points, jnp.array(0.0))
    else:
        raw = jax.vmap(net, in_axes=(0, 0))(x_points, context_points)
    return raw[:, 0]


def update_history(e_field, strain, H_prev):
    """Update tensile history without allowing compression-driven growth."""
    psi_plus = 0.5 * e_field * jnp.maximum(strain, 0.0) ** 2
    return jnp.maximum(H_prev, psi_plus)


def update_cell_history(displacement, x_points, e_cell, H_prev_cell):
    """Update DG0 tensile history from cell strain on the fixed 1D grid."""
    x = x_points[:, 0]
    cell_strain = (displacement[1:] - displacement[:-1]) / (x[1:] - x[:-1])
    psi_plus_cell = 0.5 * e_cell * jnp.maximum(cell_strain, 0.0) ** 2
    return jnp.maximum(H_prev_cell, psi_plus_cell)


def project_cell_history_to_nodes(H_cell):
    """Apply the CG1 lumped-mass projection from DG0 cells to nodal rows."""
    interior = 0.5 * (H_cell[:-1] + H_cell[1:])
    return jnp.concatenate((H_cell[:1], interior, H_cell[-1:]))


def validate_mechanics_inputs(x_points, e_field, quadrature_weights, k_res):
    """Validate fixed-grid mechanics data outside compiled functions."""
    import numpy as np

    x = np.asarray(x_points)
    e = np.asarray(e_field)
    weights = np.asarray(quadrature_weights)
    if x.ndim != 2 or x.shape[1] != 1 or len(x) < 2:
        raise ValueError(f"x_points must have shape (N, 1), got {x.shape}")
    if e.shape != (len(x),):
        raise ValueError(f"e_field must have shape {(len(x),)}, got {e.shape}")
    if weights.shape != (len(x),):
        raise ValueError(
            f"quadrature_weights must have shape {(len(x),)}, got {weights.shape}"
        )
    if not np.all(np.isfinite(x)) or not np.all(np.diff(x[:, 0]) > 0.0):
        raise ValueError("x_points must be finite and strictly increasing")
    if not np.all(np.isfinite(e)) or np.any(e <= 0.0):
        raise ValueError("e_field must be finite and strictly positive")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("quadrature_weights must be finite and strictly positive")
    if not np.isfinite(k_res) or k_res < 0.0:
        raise ValueError(f"k_res must be non-negative and finite, got {k_res}")

    spacing = np.diff(x[:, 0])
    if not np.allclose(spacing, spacing[0], rtol=1e-5, atol=1e-7):
        raise ValueError("fracture1d requires a uniform full grid")
    expected = np.full(len(x), spacing[0], dtype=weights.dtype)
    expected[[0, -1]] *= 0.5
    if not np.allclose(weights, expected, rtol=1e-5, atol=1e-7):
        raise ValueError("quadrature_weights must be nodal trapezoidal weights")


def validate_damage_inputs(x_points, H_points):
    """Validate the fixed operator ordering and history shape outside JIT."""
    import numpy as np

    x = np.asarray(x_points)
    history = np.asarray(H_points)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"damage x_points must have shape (N, 1), got {x.shape}")
    if history.shape != (len(x),):
        raise ValueError(
            f"H_points must have shape {(len(x),)}, got {history.shape}"
        )
    if not np.all(np.isfinite(history)):
        raise ValueError("H_points must be finite")


def compute_mechanics(
    d,
    x_points,
    e_field,
    u_bar,
    quadrature_weights,
    k_res,
):
    """Condense 1D mechanics and reconstruct displacement from strain.

    The compliance denominator and displacement use the same nodal trapezoidal
    rule. The final endpoint is represented by the prescribed ``u_bar`` so the
    reconstructed field satisfies both displacement constraints exactly.
    """
    x = x_points[:, 0]
    degradation = (1.0 - d) ** 2 + k_res
    compliance_density = 1.0 / (e_field * degradation)
    compliance = jnp.sum(quadrature_weights * compliance_density)
    sigma = u_bar / compliance
    strain = sigma * compliance_density

    dx = x[1:] - x[:-1]
    increments = 0.5 * (strain[1:] + strain[:-1]) * dx
    cumulative = jnp.concatenate((jnp.zeros((1,), dtype=strain.dtype),
                                  jnp.cumsum(increments)))
    displacement = cumulative
    displacement = displacement.at[0].set(0.0)
    displacement = displacement.at[-1].set(u_bar)
    psi_plus = 0.5 * e_field * jnp.maximum(strain, 0.0) ** 2
    return sigma, strain, displacement, psi_plus, compliance


def damage_residual_values(d, D, N_l, N_G, H_points):
    """Return the fixed-history damage residual for an explicit damage vector."""
    residual = d + N_l**2 * (D @ d)
    return residual - (2.0 * N_l / N_G) * (1.0 - d) * H_points


def damage_residual(
    net,
    x_points,
    D,
    N_l,
    N_G,
    H_points,
    context_points=None,
):
    """Return the pointwise residual of the fixed-history damage equation."""
    return damage_residual_values(
        _net_d(net, x_points, context_points), D, N_l, N_G, H_points
    )


def damage_residual_scale(D, N_l, N_G, H_points):
    """Return the positive Jacobi row scale of the damage system."""
    return 1.0 + N_l**2 * D.diagonal + (2.0 * N_l / N_G) * H_points


def damage_pseudo_time_residual(
    net,
    d_previous,
    x_points,
    D,
    N_l,
    N_G,
    H_points,
    tau,
    residual_scaling,
):
    """Return the implicit pseudo-time residual at fixed history.

    ``d_previous`` is the detached output from the previous pseudo-time step.
    The artificial derivative is in output space, not network-parameter or
    physical-load space.
    """
    d = _net_d(net, x_points)
    residual = (d - jax.lax.stop_gradient(d_previous)) / tau
    residual = residual + damage_residual(net, x_points, D, N_l, N_G, H_points)
    if residual_scaling == "jacobi":
        residual = residual / damage_residual_scale(D, N_l, N_G, H_points)
    elif residual_scaling != "none":
        raise ValueError(f"Unknown damage residual scaling {residual_scaling!r}")
    return residual


def loss_damage_pseudo_time(
    net,
    d_previous,
    x_points,
    D,
    N_l,
    N_G,
    H_points,
    tau,
    residual_norm,
    residual_scaling,
):
    """Return the pseudo-time residual loss for one fixed-history step."""
    residual = damage_pseudo_time_residual(
        net,
        d_previous,
        x_points,
        D,
        N_l,
        N_G,
        H_points,
        tau,
        residual_scaling,
    )
    if residual_norm == "mae":
        return jnp.mean(jnp.abs(residual))
    if residual_norm == "mse":
        return jnp.mean(residual**2)
    raise ValueError(f"Unknown damage residual norm {residual_norm!r}")


def damage_energy_values(d, D, N_l, N_G, H_points):
    """Return the mean fixed-history nonlocal quadratic energy."""
    history_factor = 2.0 * N_l / N_G
    density_sum = (
        0.5 * jnp.dot(d, d)
        + 0.5 * N_l**2 * jnp.dot(d, D @ d)
        + 0.5 * history_factor * jnp.sum(H_points * (1.0 - d) ** 2)
    )
    return density_sum / d.size


def loss_damage(
    net,
    x_points,
    D,
    N_l,
    N_G,
    H_points,
    objective,
    residual_norm,
    residual_scaling,
    hard_point_count,
    hard_point_weight,
    hard_point_mode,
    hard_point_temperature,
    hard_point_indices,
    context_points=None,
):
    """Return the configured norm of the fixed-history damage residual."""
    if objective == "energy":
        if hard_point_weight != 0.0:
            raise ValueError("hard-point weighting requires the residual objective")
        return damage_energy_values(
            _net_d(net, x_points, context_points), D, N_l, N_G, H_points
        )
    if objective != "residual":
        raise ValueError(f"Unknown damage objective {objective!r}")
    residual = damage_residual(
        net,
        x_points,
        D,
        N_l,
        N_G,
        H_points,
        context_points,
    )
    if residual_scaling == "jacobi":
        residual = residual / damage_residual_scale(D, N_l, N_G, H_points)
    elif residual_scaling != "none":
        raise ValueError(f"Unknown damage residual scaling {residual_scaling!r}")
    if residual_norm == "mae":
        pointwise_loss = jnp.abs(residual)
    elif residual_norm == "mse":
        pointwise_loss = residual**2
    else:
        raise ValueError(f"Unknown damage residual norm {residual_norm!r}")
    loss = jnp.mean(pointwise_loss)
    if hard_point_weight > 0.0:
        if hard_point_mode == "dynamic":
            hard_values = jax.lax.top_k(pointwise_loss, hard_point_count)[0]
        elif hard_point_mode == "fixed":
            if hard_point_indices is None:
                raise ValueError("fixed hard-point mode requires indices")
            hard_values = jnp.take(pointwise_loss, hard_point_indices, axis=0)
        elif hard_point_mode == "smooth":
            magnitude = jnp.abs(residual)
            reference = jax.lax.stop_gradient(jnp.mean(magnitude)) + 1.0e-12
            score = jnp.tanh(magnitude / reference)
            logits = hard_point_temperature * score
            logits = logits - jnp.max(logits)
            smooth_weights = jnp.exp(logits)
            smooth_weights = smooth_weights / jnp.mean(smooth_weights)
            smooth_weights = jax.lax.stop_gradient(smooth_weights)
            return jnp.mean(pointwise_loss * (1.0 + hard_point_weight * smooth_weights))
        else:
            raise ValueError(f"Unknown hard-point mode {hard_point_mode!r}")
        hard_loss = jnp.mean(hard_values)
        loss = loss + hard_point_weight * hard_loss
    return loss


def make_loss_fn(
    x_all,
    D,
    N_l,
    N_G,
    objective,
    residual_norm,
    residual_scaling,
    hard_point_count,
    hard_point_weight,
    hard_point_mode,
    hard_point_temperature,
    local_equilibrium=False,
):
    """Return the fixed-history damage loss factory for one residual norm."""
    def loss_fn(
        net,
        w_ema,
        bias,
        update_w,
        x_points=None,
        H_points=None,
        hard_point_indices=None,
        d_previous=None,
    ):
        del w_ema, bias, update_w
        if x_points is None:
            x_points = x_all
        if H_points is None:
            raise ValueError("H_points is required for the fixed-history damage loss")
        context_points = None
        if local_equilibrium:
            if d_previous is None:
                raise ValueError(
                    "d_previous is required for local-equilibrium correction loss"
                )
            beta = 2.0 * N_l / N_G
            local_damage = beta * H_points / (1.0 + beta * H_points)
            context_points = jnp.stack(
                (
                    local_damage,
                    jax.lax.stop_gradient(d_previous),
                ),
                axis=1,
            )

        value, gradient = eqx.filter_value_and_grad(
            lambda n: loss_damage(
                n,
                x_points,
                D,
                N_l,
                N_G,
                H_points,
                objective,
                residual_norm,
                residual_scaling,
                hard_point_count,
                hard_point_weight,
                hard_point_mode,
                hard_point_temperature,
                hard_point_indices,
                context_points,
            )
        )(net)
        weights = jnp.ones((1,), dtype=value.dtype)
        aux = {
            "loss/damage": value,
            "weight/damage": weights[0],
        }
        return value, aux, gradient, weights

    return loss_fn
