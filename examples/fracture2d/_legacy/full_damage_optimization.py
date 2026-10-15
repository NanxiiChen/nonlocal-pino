"""Deterministic all-node Adam and cached, explicitly bounded SciPy L-BFGS."""
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from examples.fracture2d._legacy.low_order_damage import make_optimizer


def adam_from_config(t):
    keys = ('learning_rate', 'decay_steps', 'decay_rate', 'minimum_rate', 'grad_clip',
            'b1', 'b2', 'eps', 'eps_root', 'shampoo_beta', 'correct_bias',
            'precondition_frequency', 'max_precond_dim', 'precondition_1d', 'weight_decay')
    return make_optimizer(kind='adam', **{key: t[key] for key in keys})


def make_lbfgs(template, objective, controls):
    """Warm parameters, fresh curvature per frozen-H subproblem, no silent fallback."""
    for key in ('maxiter', 'maxfun', 'memory', 'max_linesearch'):
        if type(controls[key]) is not int or controls[key] < 1:
            raise ValueError(f'explicit full L-BFGS {key} must be a positive integer')
    if not np.isfinite([controls['gtol'], controls['ftol']]).all() or controls['gtol'] <= 0 or controls['ftol'] < 0:
        raise ValueError('invalid explicit full L-BFGS tolerances')
    arrays, static = eqx.partition(template, eqx.is_inexact_array)
    flat, unpack = ravel_pytree(arrays)
    layout = jax.tree.structure(arrays)
    specs = [(x.shape, x.dtype) for x in jax.tree.leaves(arrays)]
    if flat.dtype != jnp.float64 or not jax.config.jax_enable_x64:
        raise ValueError('full L-BFGS requires float64 parameters')

    @jax.jit
    def value_grad(parameters, context, load):
        return jax.value_and_grad(lambda x: objective(eqx.combine(unpack(x), static), context, load))(parameters)

    def fit(net, context, load):
        values, metadata = eqx.partition(net, eqx.is_inexact_array)
        if (jax.tree.structure(values) != layout or not eqx.tree_equal(metadata, static)
                or [(x.shape, x.dtype) for x in jax.tree.leaves(values)] != specs):
            raise ValueError('full L-BFGS parameter layout changed')
        parameters, _ = ravel_pytree(values)
        started = time.perf_counter()
        def function(x):
            loss, gradient = value_grad(jnp.asarray(x), context, load)
            loss, gradient = float(loss), np.asarray(gradient, np.float64)
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise FloatingPointError('nonfinite all-node L-BFGS loss/gradient')
            return loss, gradient
        result = minimize(function, np.asarray(parameters), method='L-BFGS-B', jac=True,
            options=dict(maxiter=controls['maxiter'], maxfun=controls['maxfun'],
                maxcor=controls['memory'], maxls=controls['max_linesearch'],
                gtol=controls['gtol'], ftol=controls['ftol']))
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise FloatingPointError('nonfinite all-node L-BFGS candidate')
        return eqx.combine(unpack(jnp.asarray(result.x)), static), dict(
            optimizer='lbfgs', iterations=int(result.nit), evaluations=int(result.nfev),
            optimizer_success=bool(result.success), status=int(result.status), message=str(result.message),
            loss=float(result.fun), seconds=time.perf_counter()-started)
    return fit, value_grad


def choose_optimizer(rows):
    """Same-objective pilot ranking; finite failure statuses remain visible."""
    if ({r['optimizer'] for r in rows} != {'adam', 'lbfgs'} or len(rows) != 2
            or not np.isfinite([r['full_loss'] for r in rows]).all()
            or min(r['full_loss'] for r in rows) < 0):
        raise ValueError('selection requires two finite same-objective pilot candidates')
    return min(rows, key=lambda row: row['full_loss'])['optimizer']
