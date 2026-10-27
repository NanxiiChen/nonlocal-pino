"""Deterministic PDE-only L-BFGS refinement on fixed midpoint times."""
import json
import time
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize


def refine(net, loss_fn, initial_mean, config, run, callback):
    settings = config.train.refinement
    times = ((jnp.arange(settings.time_points, dtype=jnp.float64)+.5)
             *config.physical.terminal_time/settings.time_points)
    parameters, static = eqx.partition(net, eqx.is_inexact_array)
    x0, unravel = ravel_pytree(parameters)

    def objective(vector):
        model = eqx.combine(unravel(vector), static)
        return loss_fn(model, times, initial_mean)[0]
    value_grad = jax.jit(jax.value_and_grad(objective))
    counts = dict(iterations=0, function_evaluations=0)
    start = time.perf_counter()

    def evaluate(vector):
        value, grad = value_grad(jnp.asarray(vector, dtype=x0.dtype))
        value, grad = float(value), np.asarray(grad, dtype=np.float64)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('Nonfinite L-BFGS objective or gradient')
        counts['function_evaluations'] += 1
        return value, grad

    with (run/'refinement.jsonl').open('x') as log:
        def accepted(vector):
            counts['iterations'] += 1
            iteration = counts['iterations']
            if iteration == 1 or iteration % 50 == 0:
                value, grad = evaluate(vector)
                record = dict(iteration=iteration, loss=value,
                              gradient_inf_norm=float(np.max(np.abs(grad))),
                              elapsed_seconds=time.perf_counter()-start)
                log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                print(f'[L-BFGS {iteration}] loss={value:.6g}', flush=True)
            if iteration % config.log.eval_every == 0:
                callback(eqx.combine(unravel(jnp.asarray(vector, dtype=x0.dtype)), static), iteration)

        result = minimize(evaluate, np.asarray(x0), method='L-BFGS-B', jac=True,
            callback=accepted, options=dict(maxiter=settings.iterations,
                maxfun=settings.max_evaluations, maxls=settings.max_line_search,
                maxcor=settings.history_size, gtol=settings.gradient_tolerance,
                ftol=settings.function_tolerance))
    final = eqx.combine(unravel(jnp.asarray(result.x, dtype=x0.dtype)), static)
    record = dict(optimizer='L-BFGS-B', success=bool(result.success),
        status=int(result.status), message=str(result.message), iterations=int(result.nit),
        scipy_function_evaluations=int(result.nfev),
        total_function_evaluations=counts['function_evaluations'], final_loss=float(result.fun),
        final_gradient_inf_norm=float(np.max(np.abs(result.jac))),
        wall_seconds=time.perf_counter()-start, time_points=np.asarray(times).tolist(),
        includes_initial_pde_time=('include_initial_time' in config.sample and config.sample.include_initial_time),
        training_targets='PDE/phase mass only; no reference field values')
    (run/'refinement_summary.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return final, record
