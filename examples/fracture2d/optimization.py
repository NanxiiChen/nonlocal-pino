"""Persistent Adam/SOAP and fixed-history deterministic L-BFGS."""
import time
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax
from scipy.optimize import minimize
from pinn.optimizers import soap

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
