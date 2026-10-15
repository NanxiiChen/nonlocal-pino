"""MAE/MSE AT2 residual controls with exact core+halo and persistent optimizers.

Uses pinned upstream SOAP_JAX through the shared optimizer factory.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.fracture2d._legacy.core_halo_damage import predict_rows
from examples.fracture2d._legacy.damage_residual_weighting import core_matvec
from examples.fracture2d._legacy.plain_damage_head import map_damage
from pinn.optimizers import soap


def residual_mean(residual, history, probabilities, *, population_size, norm):
    """Unbiased nodal MAE/MSE under positive with-replacement probabilities.

    Host callers validate positive frozen history and probabilities. There is
    no floor substitution for a zero history or zero sampling probability.
    """
    if (type(population_size) is not int or population_size < 1
            or norm not in ('l1', 'l2') or residual.ndim != 1
            or probabilities.shape != residual.shape or history.shape != (population_size,)):
        raise ValueError('low-order loss needs explicit L1/L2 and matching core/population shapes')
    history, probabilities = jax.tree.map(jax.lax.stop_gradient, (history, probabilities))
    rho = residual/(2*jnp.sqrt(jnp.mean(history**2)))
    moment = jnp.abs(rho) if norm == 'l1' else rho**2
    return jnp.mean(moment/(population_size*probabilities))


def make_loss_fn(*, gc0, gc1, gain, population_size, norm):
    """Bound every halo output, then evaluate original CSR rows at core nodes."""
    if (not np.isfinite([gc0, gc1, gain]).all() or min(gc0, gc1, gain) <= 0
            or type(population_size) is not int or population_size < 1
            or norm not in ('l1', 'l2')):
        raise ValueError('invalid explicit low-order AT2 constants/norm/population')

    def halo_image(net, context):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        return map_damage('sigmoid', predict_rows(net, c['features'],
            c['basis_columns'], c['basis_values'], trunk_gain=gain))

    def loss_fn(net, context, load):
        del load
        c = jax.tree.map(jax.lax.stop_gradient, context)
        halo = halo_image(net, c)
        d = halo[c['core_local']]
        r = (gc0+2*c['core_H'])*d+gc1*core_matvec(halo,
            c['operator_columns'], c['operator_values'])-2*c['core_H']
        return residual_mean(r, c['H'], c['p_core'],
            population_size=population_size, norm=norm)
    return halo_image, loss_fn


def make_optimizer(*, kind, learning_rate, decay_steps, decay_rate, minimum_rate,
                   grad_clip, b1, b2, eps, eps_root, shampoo_beta, correct_bias,
                   precondition_frequency, max_precond_dim, precondition_1d,
                   weight_decay):
    """All optimizer settings explicit; no optimizer restart on resampling."""
    if (kind not in ('adam', 'soap')
            or not np.isfinite([learning_rate, decay_rate, minimum_rate, grad_clip,
                b1, b2, eps, eps_root, shampoo_beta, weight_decay]).all()
            or learning_rate <= 0 or not 0 < minimum_rate <= learning_rate
            or not 0 < decay_rate <= 1 or grad_clip <= 0
            or not 0 < b1 < 1 or not 0 < b2 < 1 or eps <= 0 or eps_root < 0
            or not 0 < shampoo_beta < 1 or weight_decay != 0
            or type(correct_bias) is not bool or type(precondition_1d) is not bool
            or any(type(x) is not int or x < 1 for x in
                (decay_steps, precondition_frequency, max_precond_dim))):
        raise ValueError('invalid explicit low-order optimizer settings')
    schedule = optax.exponential_decay(init_value=learning_rate,
        transition_steps=decay_steps, decay_rate=decay_rate, end_value=minimum_rate)
    if kind == 'adam':
        inner = optax.adam(schedule, b1=b1, b2=b2, eps=eps, eps_root=eps_root)
    else:
        if eps_root != 0:
            raise ValueError('SOAP_JAX has no eps_root setting')
        inner = soap(schedule, b1=b1, b2=b2, shampoo_beta=shampoo_beta, eps=eps,
            weight_decay=weight_decay, correct_bias=correct_bias,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim, precondition_1d=precondition_1d)
    return optax.chain(optax.clip_by_global_norm(grad_clip), inner), schedule


def make_step(objective, optimizer):
    @eqx.filter_jit
    def step(net, state, context, load):
        value, gradient = eqx.filter_value_and_grad(objective)(net, context, load)
        updates, state = optimizer.update(gradient, state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), state, value, optax.global_norm(gradient)
    return step


def optimizer_count(state, *, kind):
    if kind == 'adam':
        return int(state[1][0].count)
    if kind == 'soap':
        return int(state[1][0].count)
    raise ValueError(f'unknown low-order optimizer: {kind}')
