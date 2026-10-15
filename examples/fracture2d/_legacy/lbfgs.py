"""Deterministic full-batch L-BFGS step for the fracture2d PINN stages.

The inverse-Hessian product is the standard two-loop recursion over a fixed
size history ring; the line search is a scalar halving Armijo backtracking
loop evaluated on the host. Only the objective and its gradient are jitted:
each candidate is one direct (non-batched) compiled evaluation, so no
``lax.while_loop``/``vmap`` candidate machinery, no optimizer-internal
scatter and no atomic reduction participates in a step. Combined with the
deterministic FP32 assembly in ``fp32.py``, a step is reproducible run to
run. (The ``optax.lbfgs`` zoom line search and a jitted while-loop Armijo
were both observed to take different steps under jit than eagerly on this
stiff objective; this host-controlled variant does not.)

The history is kept as flat tuples of arrays (one slot tree per memory
entry), so the serializable state contains no Equinox module statics.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


class LBFGSState(eqx.Module):
    """Flat-tuple L-BFGS history; slot 0 of every tuple is the newest pair."""

    s_memory: Any
    y_memory: Any
    rho_memory: jax.Array
    count: jax.Array


def _flat_dot(left, right):
    products = [jnp.sum(a * b) for a, b in zip(left, right)]
    total = products[0]
    for product in products[1:]:
        total = total + product
    return total


def _flat_slot(memory, index):
    return tuple(leaf[index] for leaf in memory)


def _flat_scale(values, scale):
    return tuple(scale * leaf for leaf in values)


def _flat_add(left, right, coefficient):
    return tuple(a + coefficient * b for a, b in zip(left, right))


def _update_memory(memory, pair):
    return tuple(
        jnp.roll(leaf, 1, axis=0).at[0].set(new)
        for leaf, new in zip(memory, pair)
    )


def make_lbfgs_step(
    objective, memory_size=10, armijo_c1=1e-4, max_backtracks=30
):
    """Build ``(init, step)`` for one frozen objective.

    ``init(params)`` returns the serializable history state.
    ``step(params, state)`` runs one full L-BFGS step on the host and returns
    ``(new_params, new_state, new_value)``.
    """
    for name, value in (("memory_size", memory_size), ("max_backtracks", max_backtracks)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not (0.0 < float(armijo_c1) < 1.0):
        raise ValueError("armijo_c1 must lie strictly between 0 and 1")
    memory_size = int(memory_size)
    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))
    value_only = eqx.filter_jit(objective)

    def init(params):
        flat = jax.tree.leaves(params)
        return LBFGSState(
            s_memory=tuple(
                jnp.zeros((memory_size,) + leaf.shape, dtype=leaf.dtype)
                for leaf in flat
            ),
            y_memory=tuple(
                jnp.zeros((memory_size,) + leaf.shape, dtype=leaf.dtype)
                for leaf in flat
            ),
            rho_memory=jnp.zeros((memory_size,), dtype=jnp.float32),
            count=jnp.zeros((), dtype=jnp.int32),
        )

    def step(params, state):
        value, grad = value_and_grad(params)
        flat_grad = jax.tree.leaves(grad)
        rho = state.rho_memory

        alphas = []
        vector = flat_grad
        for index in range(memory_size):
            s_pair = _flat_slot(state.s_memory, index)
            y_pair = _flat_slot(state.y_memory, index)
            alpha = rho[index] * _flat_dot(s_pair, vector)
            alphas.append(alpha)
            vector = _flat_add(vector, y_pair, -alpha)
        newest_s = _flat_slot(state.s_memory, 0)
        newest_y = _flat_slot(state.y_memory, 0)
        curvature = _flat_dot(newest_y, newest_y)
        scale = jnp.where(
            (state.count > 0) & (curvature > 0.0),
            _flat_dot(newest_s, newest_y)
            / jnp.where(curvature > 0.0, curvature, 1.0),
            1.0,
        )
        vector = _flat_scale(vector, scale)
        for index in reversed(range(memory_size)):
            y_pair = _flat_slot(state.y_memory, index)
            beta = rho[index] * _flat_dot(y_pair, vector)
            s_pair = _flat_slot(state.s_memory, index)
            vector = _flat_add(vector, s_pair, alphas[index] - beta)
        direction = _flat_scale(vector, -1.0)

        slope = _flat_dot(flat_grad, direction)
        if float(slope) >= 0.0:
            direction = _flat_scale(flat_grad, -1.0)
            slope = -_flat_dot(flat_grad, flat_grad)

        tree = jax.tree_util.tree_structure(params)
        flat_params = jax.tree.leaves(params)

        def candidate(step_size):
            return jax.tree_util.tree_unflatten(
                tree,
                [p + step_size * d for p, d in zip(flat_params, direction)],
            )

        threshold_base = float(value)
        slope_value = float(slope)
        step_size = 1.0
        for _ in range(max_backtracks):
            candidate_value = float(value_only(candidate(step_size)))
            if candidate_value <= threshold_base + armijo_c1 * step_size * slope_value:
                break
            step_size *= 0.5
        new_params = candidate(step_size)
        new_value, new_grad = value_and_grad(new_params)

        flat_new = jax.tree.leaves(new_params)
        flat_new_grad = jax.tree.leaves(new_grad)
        s_pair = tuple(new - old for new, old in zip(flat_new, flat_params))
        y_pair = tuple(new - old for new, old in zip(flat_new_grad, flat_grad))
        curvature = _flat_dot(y_pair, s_pair)
        positive = curvature > 0.0
        rho_new = jnp.where(
            positive,
            1.0 / jnp.where(positive, curvature, 1.0),
            jnp.zeros_like(curvature),
        )
        state = LBFGSState(
            s_memory=_update_memory(state.s_memory, s_pair),
            y_memory=_update_memory(state.y_memory, y_pair),
            rho_memory=jnp.roll(rho, 1).at[0].set(rho_new),
            count=jnp.minimum(state.count + 1, memory_size).astype(jnp.int32),
        )
        return new_params, state, new_value

    return init, step
