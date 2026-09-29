"""Deterministic matrix-free damped Gauss-Newton (Levenberg-Marquardt) step.

``step(params, state)`` solves the damped normal equations
``(JtJ + damping * I) delta = -Jt r`` for the least-squares residual
``r(params)`` using a truncated conjugate-gradient loop (``lax.while_loop``
with a fixed iteration cap and a relative-residual stop). JVPs/VJPs are taken
with ``eqx.filter_jvp``/``eqx.filter_vjp`` so Equinox static fields are not
differentiated. The only reductions are fixed-shape leaf sums, so a step is
reproducible run to run.

The step is robustified on the host: for each damping value an Armijo
backtracking line search runs along the Gauss-Newton direction, a successful
step decreases the damping geometrically, and a fully rejected direction
raises the damping and re-solves the damped system (bounded attempts). No
candidate evaluation happens inside a batched/while-loop jitted region,
matching the deterministic optimizer policy of ``lbfgs.py``.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


class GaussNewtonState(eqx.Module):
    """Damping trust parameter plus the last CG solve diagnostics."""

    damping: jax.Array
    cg_iterations: jax.Array
    cg_relative_residual: jax.Array
    step_size: jax.Array
    accepted: jax.Array


def _flat_dot(left, right):
    leaves_left = jax.tree.leaves(left)
    leaves_right = jax.tree.leaves(right)
    total = jnp.sum(leaves_left[0] * leaves_right[0])
    for a, b in zip(leaves_left[1:], leaves_right[1:]):
        total = total + jnp.sum(a * b)
    return total


def _flat_add(left, right, coefficient):
    return jax.tree.map(lambda a, b: a + coefficient * b, left, right)


def _flat_scale(values, coefficient):
    return jax.tree.map(lambda a: coefficient * a, values)


def _flat_zero(template):
    return jax.tree.map(jnp.zeros_like, template)


def make_gauss_newton_step(
    residual,
    initial_damping=1.0e-2,
    max_cg=200,
    cg_tolerance=1.0e-8,
    max_attempts=4,
    max_backtracks=20,
    armijo_c1=1.0e-4,
    damping_decrease=0.3,
    damping_increase=10.0,
    minimum_damping=1.0e-8,
    maximum_damping=1.0e8,
):
    """Build ``(init, step)`` for a frozen least-squares residual.

    ``residual(params)`` must return a one-dimensional array. ``init(params)``
    returns the serializable trust state and ``step(params, state)`` runs one
    damped Gauss-Newton step on the host, returning
    ``(new_params, new_state, new_value)`` with ``new_value`` the mean-square
    residual.
    """
    for name, value in (
        ("max_cg", max_cg),
        ("max_attempts", max_attempts),
        ("max_backtracks", max_backtracks),
        ("damping_decrease", damping_decrease),
        ("damping_increase", damping_increase),
        ("minimum_damping", minimum_damping),
        ("maximum_damping", maximum_damping),
        ("cg_tolerance", cg_tolerance),
        ("armijo_c1", armijo_c1),
        ("initial_damping", initial_damping),
    ):
        if isinstance(value, bool) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a positive number")
    if not (0.0 < float(damping_decrease) < 1.0 < float(damping_increase)):
        raise ValueError("damping_decrease < 1 < damping_increase is required")
    if float(minimum_damping) >= float(maximum_damping):
        raise ValueError("minimum_damping must be below maximum_damping")
    max_cg = int(max_cg)
    max_attempts = int(max_attempts)
    max_backtracks = int(max_backtracks)
    minimum_damping = jnp.float32(minimum_damping)
    maximum_damping = jnp.float32(maximum_damping)

    def objective(params):
        return jnp.mean(residual(params) ** 2)

    value_only = eqx.filter_jit(objective)
    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(objective))

    @eqx.filter_jit
    def direction(params, damping):
        r = residual(params)
        _, pullback = eqx.filter_vjp(residual, params)
        residual_gradient = pullback(r)[0]

        def matvec(vector):
            _, jv = eqx.filter_jvp(residual, (params,), (vector,))
            (normal,) = pullback(jv)
            return _flat_add(normal, vector, damping)

        rhs = _flat_scale(residual_gradient, -1.0)
        rhs_norm = jnp.sqrt(_flat_dot(rhs, rhs)) + jnp.float32(1.0e-30)
        state = (
            _flat_zero(params),
            rhs,
            rhs,
            _flat_dot(rhs, rhs),
            jnp.zeros((), jnp.int32),
        )

        def condition(carry):
            _, _, _, rs, iterations = carry
            return (iterations < max_cg) & (
                jnp.sqrt(rs) > jnp.float32(cg_tolerance) * rhs_norm
            )

        def body(carry):
            delta, residual_vector, search, rs, iterations = carry
            normal = matvec(search)
            denominator = _flat_dot(search, normal)
            safe = jnp.abs(denominator) > 0.0
            alpha = jnp.where(
                safe, rs / jnp.where(safe, denominator, jnp.float32(1.0)), 0.0
            )
            alpha = jnp.where(jnp.isfinite(alpha), alpha, 0.0)
            delta = _flat_add(delta, search, alpha)
            residual_vector = _flat_add(residual_vector, normal, -alpha)
            rs_new = _flat_dot(residual_vector, residual_vector)
            beta = jnp.where(rs > 0.0, rs_new / jnp.where(rs > 0.0, rs, 1.0), 0.0)
            search = _flat_add(residual_vector, search, beta)
            return delta, residual_vector, search, rs_new, iterations + 1

        delta, _, _, rs, iterations = jax.lax.while_loop(condition, body, state)
        relative = jnp.sqrt(rs) / rhs_norm
        return delta, iterations, relative, _flat_dot(residual_gradient, delta)

    def init(params):
        return GaussNewtonState(
            damping=jnp.asarray(initial_damping, dtype=jnp.float32),
            cg_iterations=jnp.zeros((), dtype=jnp.int32),
            cg_relative_residual=jnp.zeros((), dtype=jnp.float32),
            step_size=jnp.zeros((), dtype=jnp.float32),
            accepted=jnp.zeros((), dtype=jnp.bool_),
        )

    def step(params, state):
        base_value, base_grad = value_and_grad(params)
        base_value = float(base_value)
        damping = state.damping
        candidate = params
        iterations = state.cg_iterations
        relative = state.cg_relative_residual
        step_size = state.step_size
        accepted = False
        for _ in range(max_attempts):
            delta, iterations, relative, inner = direction(params, damping)
            slope = float(_flat_dot(base_grad, delta))
            if not (np.isfinite(slope) and slope < 0.0 and np.isfinite(float(inner))):
                damping = damping * jnp.float32(damping_increase)
                if float(damping) >= float(maximum_damping):
                    break
                continue
            step_size = 1.0
            for _ in range(max_backtracks):
                proposal = _flat_add(params, delta, step_size)
                proposal_value = float(value_only(proposal))
                if np.isfinite(proposal_value) and proposal_value <= (
                    base_value + float(armijo_c1) * step_size * slope
                ):
                    candidate = proposal
                    accepted = True
                    break
                step_size *= 0.5
            if accepted:
                break
            damping = damping * jnp.float32(damping_increase)
            if float(damping) >= float(maximum_damping):
                break
        if accepted:
            params = candidate
            damping = jnp.maximum(
                damping * jnp.float32(damping_decrease), minimum_damping
            )
        else:
            damping = jnp.minimum(damping, maximum_damping)
        value = float(value_only(params))
        state = GaussNewtonState(
            damping=jnp.asarray(damping, dtype=jnp.float32),
            cg_iterations=iterations,
            cg_relative_residual=relative,
            step_size=step_size,
            accepted=jnp.asarray(accepted, dtype=jnp.bool_),
        )
        return params, state, value

    return init, step
