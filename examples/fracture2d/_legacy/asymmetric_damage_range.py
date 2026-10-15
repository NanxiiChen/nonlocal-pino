"""Explicit asymmetric numerical range gates; original equations unchanged."""
import time
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.optimize import minimize

from examples.fracture2d._legacy.damage_optimization import _ResidualReached


def make_asymmetric_lbfgs(template, objective, metrics):
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
                or not np.isfinite(controls['negative_tolerance'])
                or not np.isfinite(controls['positive_tolerance'])
                or not 0 <= controls['negative_tolerance'] < 1
                or not 0 <= controls['positive_tolerance'] < 1):
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
                     and values['damage_min'] >= -controls['negative_tolerance']
                     and values['damage_max'] <= 1+controls['positive_tolerance'])
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


from examples.fracture2d._legacy.local_mechanics import LocalMechanics, QUADRATURE
from examples.fracture2d._legacy.outer_stopping import evaluate as original_evaluate


class AsymmetricLocalMechanics(LocalMechanics):
    def __init__(self, *args, negative_tolerance, positive_tolerance, **kwargs):
        for value in (negative_tolerance, positive_tolerance):
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError('asymmetric FEM range tolerances must be explicit finite0..1')
        super().__init__(*args, **kwargs)
        self.negative_tolerance, self.positive_tolerance = negative_tolerance, positive_tolerance

    def factors(self, damage):
        if (damage.shape != (self.n_material,) or damage.dtype != np.float64
                or not np.isfinite(damage).all()
                or damage.min() < -self.negative_tolerance
                or damage.max() > 1+self.positive_tolerance):
            raise ValueError('P1 material damage violates explicit asymmetric range')
        factors = np.full(len(self.triangles), 1+self.k_res)
        values = damage[self.triangles[:self.n_material_cells]] @ QUADRATURE.T
        factors[:self.n_material_cells] = np.mean((1-values)**2, axis=1)+self.k_res
        return factors


def evaluate_asymmetric(report, c, limits, *, sweep, previous_consecutive,
                        negative_tolerance, positive_tolerance):
    for value in (negative_tolerance, positive_tolerance):
        if not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError('asymmetric outer range tolerances must be explicit finite0..1')
    outer = original_evaluate(report,c,limits,sweep=sweep,previous_consecutive=previous_consecutive)
    checks = outer['checks']
    del checks['damage_range'], checks['image_range']
    for prefix,lo,hi in (('damage',report['damage_min'],report['damage_max']),
                         ('image',report['fixed_H_image_damage_min'],report['fixed_H_image_damage_max'])):
        for side,value,limit in (('negative',max(0.,-lo),negative_tolerance),
                                 ('positive',max(0.,hi-1),positive_tolerance)):
            checks[prefix+'_'+side+'_range'] = dict(value=value,limit=limit,**{'pass':value<=limit})
    passed = all(check['pass'] for check in checks.values())
    consecutive = previous_consecutive+1 if passed else 0
    ready = sweep>=c.min_sweeps and consecutive>=c.consecutive_passes
    outer.update(pass_this_sweep=passed,consecutive=consecutive,converged=ready,
        reason='converged' if ready else 'budget_exhausted' if sweep==c.max_sweeps else 'iterating')
    return outer
