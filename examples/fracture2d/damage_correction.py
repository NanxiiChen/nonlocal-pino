"""Explicit damped-Jacobi AT2 correction; this is a nodal numerical solve."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.damage_optimization import validate_energy_operator
from examples.fracture2d.split_pinn_gradients import make_padded_matvec


def make_corrector(operator, volumes, gc0, gc1, controls, *, roundoff_factor):
    controls = dict(controls)
    validate_energy_operator(operator, volumes, roundoff_factor=roundoff_factor)
    for name in ('maxiter', 'check_every'):
        if type(controls[name]) is not int or controls[name] < 1:
            raise ValueError(f'Jacobi {name} must be an explicit positive integer')
    if (controls['maxiter'] % controls['check_every'] != 0
            or not 0 < controls['omega'] <= 1
            or not 0 < controls['residual_limit'] < 1
            or not 0 <= controls['range_atol'] < 1
            or not np.isfinite([gc0, gc1]).all() or min(gc0, gc1) <= 0
            or not jax.config.jax_enable_x64):
        raise ValueError('invalid explicit FP64 Jacobi controls/material')
    D = operator.to_scipy()
    off = D.copy()
    off.setdiag(0.)
    off.eliminate_zeros()
    scale = np.max(abs(D.data))
    tolerance = roundoff_factor*np.finfo(float).eps*scale
    if (np.any(off.data > 0) or np.any(D.diagonal() <= 0)
            or np.max(abs(np.asarray(D.sum(axis=1)).ravel())) > tolerance):
        raise ValueError('Jacobi correction requires a positive graph Laplacian M-matrix')
    matvec, _ = make_padded_matvec(operator)
    diagonal_D = jnp.asarray(D.diagonal())

    @jax.jit
    def chunk(d, H):
        diagonal = gc0+2*H+gc1*diagonal_D
        def update(_, value):
            rho = (gc0+2*H)*value+gc1*matvec(value)-2*H
            return value-controls['omega']*rho/diagonal
        return jax.lax.fori_loop(0, controls['check_every'], update, d)

    @jax.jit
    def metrics(d, H):
        parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
        return (jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),
                jnp.min(d), jnp.max(d))

    def solve(H, initial):
        for name, value in (('history', H), ('initial', initial)):
            array = np.asarray(value)
            if (array.shape != volumes.shape or array.dtype != np.float64
                    or not np.all(np.isfinite(array))):
                raise ValueError(f'invalid Jacobi {name} field')
        if (float(jnp.min(H)) < 0 or float(jnp.max(H)) <= 0
                or float(jnp.min(initial)) < -controls['range_atol']
                or float(jnp.max(initial)) > 1+controls['range_atol']):
            raise ValueError('Jacobi requires positive history and bounded initial damage')
        start, iterations, d = time.perf_counter(), 0, jnp.asarray(initial)
        while True:
            residual, minimum, maximum = [float(v) for v in metrics(d, H)]
            if (not np.isfinite([residual, minimum, maximum]).all()
                    or minimum < -controls['range_atol'] or maximum > 1+controls['range_atol']):
                raise FloatingPointError('Jacobi correction lost finite admissibility')
            if residual <= controls['residual_limit']:
                return d, dict(iterations=iterations, AT2_relative=residual,
                    damage_min=minimum, damage_max=maximum, seconds=time.perf_counter()-start)
            if iterations >= controls['maxiter']:
                raise RuntimeError(f'Jacobi correction exhausted {iterations}: AT2 {residual} > {controls["residual_limit"]}')
            d = chunk(d, H)
            iterations += controls['check_every']
    return solve
