"""Fixed-H AT2 neural optimization, with unchanged full-field residual gates.

For M=diag(V), reciprocal kernels give MD=(MD)^T. Therefore
E_H(d)=.5*d^T M[(gc0+2H)d+gc1 Dd]-2*d^T MH has gradient M*r_AT2.
This is a fixed-H damage functional only, not an energy for the hybrid u/H map.
"""
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags


def validate_energy_operator(operator, volumes, *, roundoff_factor):
    D = operator.to_scipy()
    if (D.shape != (len(volumes), len(volumes)) or D.dtype != np.float64
            or volumes.dtype != np.float64 or volumes.ndim != 1
            or not np.all(np.isfinite(volumes)) or np.min(volumes) <= 0
            or not np.all(np.isfinite(D.data))
            or not np.isfinite(roundoff_factor) or roundoff_factor < 1):
        raise ValueError('fixed-H energy requires finite FP64 operator/positive volumes')
    weighted = diags(volumes) @ D
    skew = weighted-weighted.T
    difference = float(np.max(abs(skew.data))) if skew.nnz else 0.
    limit = roundoff_factor*np.finfo(float).eps*np.max(abs(weighted.data))
    if difference > limit:
        raise ValueError('fixed-H damage energy requires volume symmetry')
    return dict(weighted_skew_inf=difference, weighted_skew_limit=float(limit))


def damage_energy(d, H, volumes, matvec, gc0, gc1):
    return .5*jnp.sum(volumes*d*((gc0+2*H)*d+gc1*matvec(d)))-2*jnp.sum(volumes*H*d)


def make_objectives(image, matvec, volumes, gc0, gc1):
    """Changing history/load enter compiled paths as arrays, not new closures."""
    def residual(net, H, load):
        d = image(net, load)
        rho = (gc0+2*H)*d+gc1*matvec(d)-2*H
        scale = 2*jnp.sqrt(jnp.mean(H**2))
        return jnp.mean((rho/scale)**2)

    def energy(net, H, load):
        # Positive fixed-H scalar normalization; stationary d is unchanged.
        scale = 4*jnp.sum(volumes*H**2)/gc0
        return damage_energy(image(net, load), H, volumes, matvec, gc0, gc1)/scale

    return residual, energy


class _ResidualReached(Exception):
    def __init__(self, parameters):
        self.parameters = parameters.copy()


def optimize_lbfgs(net, objective, metrics, H, load, controls, *, progress):
    """Host SciPy L-BFGS; no silent fallback when optimizer stops unready.

    metrics returns independent full-field AT2_relative and damage extrema.
    Returns an explicit residual/budget/optimizer status plus the candidate;
    caller decides whether the fixed-H result can enter an outer update.
    """
    for name in ('maxiter', 'check_every', 'memory', 'max_linesearch'):
        value = controls[name]
        if type(value) is not int or value < 1:
            raise ValueError(f'LBFGS {name} must be an explicit positive integer')
    if (not np.isfinite(controls['gtol']) or controls['gtol'] <= 0
            or not np.isfinite(controls['ftol']) or controls['ftol'] < 0
            or not 0 < controls['residual_limit'] < 1
            or not 0 <= controls['range_tolerance'] < 1):
        raise ValueError('invalid LBFGS tolerances/physical gates')
    arrays, static = eqx.partition(net, eqx.is_inexact_array)
    flat, unpack = ravel_pytree(arrays)
    if flat.dtype != jnp.float64 or not jax.config.jax_enable_x64:
        raise ValueError('fixed-H LBFGS requires x64 network parameters')

    @jax.jit
    def value_grad(parameters, history, prescribed):
        def value(p):
            return objective(eqx.combine(unpack(p), static), history, prescribed)
        return jax.value_and_grad(value)(parameters)

    @jax.jit
    def evaluate(parameters, history, prescribed):
        return metrics(eqx.combine(unpack(parameters), static), history, prescribed)

    rows, iteration, evaluations = [], 0, 0
    start = time.perf_counter()

    def function(parameters):
        nonlocal evaluations
        value, gradient = value_grad(jnp.asarray(parameters), H, load)
        value, gradient = float(value), np.asarray(gradient, np.float64)
        evaluations += 1
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError('LBFGS damage objective/gradient is nonfinite')
        return value, gradient

    def check(parameters):
        values = {k: float(v) for k, v in evaluate(jnp.asarray(parameters), H, load).items()}
        if not np.isfinite(list(values.values())).all():
            raise FloatingPointError('LBFGS damage metrics are nonfinite')
        ready = (values['AT2_relative'] <= controls['residual_limit']
                 and values['damage_min'] >= -controls['range_tolerance']
                 and values['damage_max'] <= 1+controls['range_tolerance'])
        rows.append(dict(iteration=iteration, evaluations=evaluations,
                         seconds=time.perf_counter()-start, ready=ready, **values))
        progress(rows[-1])
        return ready

    def callback(parameters):
        nonlocal iteration
        iteration += 1
        if iteration % controls['check_every'] == 0 and check(parameters):
            raise _ResidualReached(parameters)

    parameters = np.asarray(flat, np.float64)
    if check(parameters):
        return net, dict(status='residual_reached', iterations=0, evaluations=0,
                         seconds=time.perf_counter()-start, rows=rows)
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
        seconds=time.perf_counter()-start, message=message, rows=rows)
