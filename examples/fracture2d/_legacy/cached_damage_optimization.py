"""Reuse fixed-layout L-BFGS JIT functions; histories/loads stay dynamic.

The SciPy iteration and acceptance policy matches damage_optimization.py.
The original numerical source is preserved for its certified older run roots.
"""
import time
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from examples.fracture2d._legacy.damage_optimization import _ResidualReached


def make_cached_lbfgs(template, objective, metrics):
    arrays, static = eqx.partition(template, eqx.is_inexact_array)
    flat, unpack = ravel_pytree(arrays)
    layout = jax.tree.structure(arrays)
    specifications = [(a.shape, a.dtype) for a in jax.tree.leaves(arrays)]
    if flat.dtype != jnp.float64 or not jax.config.jax_enable_x64:
        raise ValueError('cached damage L-BFGS requires explicit x64 parameters')

    @jax.jit
    def value_grad(parameters, H, load):
        def value(p):
            return objective(eqx.combine(unpack(p), static), H, load)
        return jax.value_and_grad(value)(parameters)

    @jax.jit
    def evaluate(parameters, H, load):
        return metrics(eqx.combine(unpack(parameters), static), H, load)

    def fit(net, H, load, controls, *, progress):
        for name in ('maxiter', 'check_every', 'memory', 'max_linesearch'):
            if type(controls[name]) is not int or controls[name] < 1:
                raise ValueError(f'cached L-BFGS {name} must be explicit positive integer')
        if (not np.isfinite(controls['gtol']) or controls['gtol'] <= 0
                or not np.isfinite(controls['ftol']) or controls['ftol'] < 0
                or not 0 < controls['residual_limit'] < 1
                or not 0 <= controls['range_tolerance'] < 1):
            raise ValueError('invalid cached L-BFGS controls')
        current, current_static = eqx.partition(net, eqx.is_inexact_array)
        if (jax.tree.structure(current) != layout or not eqx.tree_equal(static, current_static)
                or [(a.shape, a.dtype) for a in jax.tree.leaves(current)] != specifications):
            raise ValueError('cached L-BFGS network structure/shapes/dtypes/static fields changed')
        parameters, _ = ravel_pytree(current)
        parameters = np.asarray(parameters, np.float64)
        rows, iteration, evaluations = [], 0, 0
        started = time.perf_counter()

        def function(p):
            nonlocal evaluations
            value, gradient = value_grad(jnp.asarray(p), H, load)
            value, gradient = float(value), np.asarray(gradient, np.float64)
            evaluations += 1
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise FloatingPointError('cached L-BFGS objective/gradient is nonfinite')
            return value, gradient

        def check(p):
            values = {k: float(v) for k, v in evaluate(jnp.asarray(p), H, load).items()}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError('cached L-BFGS metrics are nonfinite')
            ready = (values['AT2_relative'] <= controls['residual_limit']
                     and values['damage_min'] >= -controls['range_tolerance']
                     and values['damage_max'] <= 1+controls['range_tolerance'])
            rows.append(dict(iteration=iteration, evaluations=evaluations,
                seconds=time.perf_counter()-started, ready=ready, **values))
            progress(rows[-1])
            return ready

        def callback(p):
            nonlocal iteration
            iteration += 1
            if iteration % controls['check_every'] == 0 and check(p):
                raise _ResidualReached(p)

        if check(parameters):
            return net, dict(status='residual_reached', iterations=0, evaluations=0,
                             seconds=time.perf_counter()-started, rows=rows)
        try:
            result = minimize(function, parameters, method='L-BFGS-B', jac=True,
                callback=callback, options=dict(maxiter=controls['maxiter'],
                    maxcor=controls['memory'], maxls=controls['max_linesearch'],
                    gtol=controls['gtol'], ftol=controls['ftol']))
        except _ResidualReached as reached:
            parameters = reached.parameters
            status, message = 'residual_reached', 'full-field residual and range gates passed'
        else:
            parameters = result.x
            ready = check(parameters)
            status = 'residual_reached' if ready else (
                'iteration_budget_exhausted' if iteration >= controls['maxiter']
                else 'optimizer_stop_without_residual')
            message = str(result.message)
        candidate = eqx.combine(unpack(jnp.asarray(parameters)), static)
        return candidate, dict(status=status, iterations=iteration, evaluations=evaluations,
            seconds=time.perf_counter()-started, message=message, rows=rows)

    return fit, value_grad, evaluate
