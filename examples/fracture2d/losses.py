"""Nonlocal AT2 residual; detached context, importance sampling and weights."""
import jax
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.network import map_damage, predict_rows


def damage_residual(damage, history, diffusion, *, gc0, gc1):
    history = jax.lax.stop_gradient(history)
    return (gc0 + 2*history)*damage + gc1*diffusion - 2*history


def residual_mean(residual, history, probabilities, weights, *, population_size, norm, floor):
    """L1 mean, L2 mean-square, or stabilized squared L16 norm.

    1/(N p_i) corrects with-replacement nodal sampling. The L16 moment is
    corrected; its nonlinear root is not an unbiased full-loss estimate.
    """
    if (norm not in ('l1', 'l2', 'l16') or type(population_size) is not int
            or population_size < 1 or history.shape != (population_size,)
            or residual.ndim != 1 or probabilities.shape != residual.shape
            or weights.shape != residual.shape or not 0 < floor < 1):
        raise ValueError('invalid explicit norm/population/core shapes or loss floor')
    history, probabilities, weights = jax.tree.map(
        jax.lax.stop_gradient, (history, probabilities, weights))
    rho = residual/(2*jnp.sqrt(jnp.mean(history**2)))
    correction = weights/(population_size*probabilities)
    if norm == 'l1':
        return jnp.mean(jnp.abs(rho)*correction)
    if norm == 'l2':
        return jnp.mean(rho**2*correction)
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(abs(rho)), floor))
    return scale**2*(jnp.mean((rho/scale)**16*correction)+(floor/scale)**16)**(2/16)


def make_loss_fn(*, gc0, gc1, gain, population_size, norm, floor, head):
    if (not np.isfinite([gc0, gc1, gain]).all() or min(gc0, gc1, gain) <= 0
            or norm not in ('l1', 'l2', 'l16')
            or head not in ('sigmoid', 'exp_square', 'identity')):
        raise ValueError('invalid explicit AT2 constants, objective or output head')

    def image(net, context):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        return map_damage(head, predict_rows(net, c['features'],
            c['basis_columns'], c['basis_values'], trunk_gain=gain))

    def loss_fn(net, context, load):
        del load
        c = jax.tree.map(jax.lax.stop_gradient, context)
        halo = image(net, c)
        r = damage_residual(halo[c['core_local']], c['core_H'],
            jnp.sum(c['operator_values']*halo[c['operator_columns']], axis=1),
            gc0=gc0, gc1=gc1)
        return residual_mean(r, c['H'], c['p_core'], c['weights'],
            population_size=population_size, norm=norm, floor=floor)
    return image, loss_fn


def residual_metrics(d, H, D, *, gc0, gc1):
    parts = (gc0*d, gc1*(D @ d), 2*H*(d-1))
    r = sum(parts)
    scale = sum(np.max(abs(part)) for part in parts)
    # Exact physical zero has no forcing and zero residual.
    relative = float(np.max(abs(r))/scale) if scale > 0 else 0.
    return r, relative


def damage_weights(damage, controls):
    """Refresh once per outer; preserve weights across its inner solve."""
    d = jax.lax.stop_gradient(damage)
    if controls.weight_mode == 'uniform':
        raw = jnp.ones_like(d)
    elif controls.weight_mode == 'damage':
        raw = 1 + controls.weight_boost*jnp.clip(d, 0., 1.)**controls.weight_power
    elif controls.weight_mode == 'reciprocal':
        raw = jnp.minimum(1/(1-jnp.clip(d, 0., 1.)+controls.weight_epsilon),
                          controls.weight_cap)
    else:
        raise ValueError('unknown explicit damage weight mode')
    return jax.lax.stop_gradient(raw/jnp.mean(raw))
