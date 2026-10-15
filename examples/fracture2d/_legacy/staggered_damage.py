"""Mechanics-first fixed-load sweep and exact material P1 field changes."""
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d._legacy.l1_damage_loading import committed_history


def case_configs(cases):
    """ConfigDict preserves dictionaries inside lists; normalize explicitly."""
    result = tuple(ConfigDict(case) for case in cases)
    if not result or len({case.name for case in result}) != len(result):
        raise ValueError('stagger controls require unique nonempty cases')
    for case in result:
        if (case.solver not in ['direct', 'pinn'] or not np.isfinite(case.omega)
                or not 0 < case.omega <= 1 or type(case.max_sweeps) is not int
                or case.max_sweeps < 1):
            raise ValueError('invalid explicit stagger case')
    return result


def mechanics_first_sweep(q, d, parent_H, load, carry, *, mechanics, driver,
                          damage, omega):
    """One mechanics call; returned q equilibrates d_previous, not new d.

    Damage callback returns (image, new_carry, diagnostics). Its history is
    detached and the previous load's history floor is never mutated.
    """
    if not np.isfinite(omega) or not 0 < omega <= 1:
        raise ValueError('explicit stagger relaxation must be in (0, 1]')
    q_new, reaction, mechanical = mechanics(d, load, q)
    H = committed_history(parent_H, jnp.asarray(driver(np.asarray(q_new))))
    image, carry, inner = damage(H, d, carry)
    d_new = image if omega == 1 else d + omega*(image-d)
    return dict(q=q_new, d=d_new, H=H, damage_image=image,
                q_previous=q, d_previous=d, reaction=reaction,
                mechanics=mechanical, inner=inner), carry


def p1_rms(values, triangles, areas):
    """Exact L2 norm of a scalar/vector P1 field divided by sqrt(area)."""
    values, triangles, areas = map(np.asarray, (values, triangles, areas))
    if (values.ndim not in (1, 2) or triangles.ndim != 2
            or triangles.shape[1] != 3 or areas.shape != (len(triangles),)
            or len(areas) == 0 or not np.isfinite(values).all()
            or not np.isfinite(areas).all() or np.min(areas) <= 0
            or not np.issubdtype(triangles.dtype, np.integer)
            or triangles.min() < 0 or triangles.max() >= len(values)):
        raise ValueError('P1 RMS requires finite fields and positive valid material cells')
    local = values[triangles]
    if values.ndim == 1:
        local = local[..., None]
    integrals = (np.sum(local, axis=1)**2 + np.sum(local**2, axis=1)).sum(axis=1)/12
    return float(np.sqrt(areas@integrals/areas.sum()))


def field_changes(q, previous_q, d, previous_d, triangles, areas, *, load):
    if not np.isfinite(load) or load <= 0:
        raise ValueError('displacement normalization requires positive explicit load')
    if q.shape != previous_q.shape or d.shape != previous_d.shape:
        raise ValueError('field change shapes differ')
    return dict(displacement_rms_over_load=p1_rms(
                    (np.asarray(q)-np.asarray(previous_q)).reshape(-1, 2), triangles, areas)/load,
                damage_rms=p1_rms(np.asarray(d)-np.asarray(previous_d), triangles, areas))


def field_stop(changes, *, displacement_limit, damage_limit, sweep,
               minimum_sweeps, consecutive_passes, previous_consecutive):
    if (not np.isfinite([displacement_limit, damage_limit]).all()
            or min(displacement_limit, damage_limit) <= 0
            or any(type(v) is not int or v < 1 for v in
                   (sweep, minimum_sweeps, consecutive_passes))
            or type(previous_consecutive) is not int or previous_consecutive < 0):
        raise ValueError('invalid explicit field-stopping controls')
    values = [changes['displacement_rms_over_load'], changes['damage_rms']]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('invalid field-change diagnostics')
    passed = bool(sweep >= minimum_sweeps and values[0] <= displacement_limit
                  and values[1] <= damage_limit)
    count = previous_consecutive+1 if passed else 0
    return dict(pass_this_sweep=passed, consecutive=count,
                converged=count >= consecutive_passes,
                residual_is_stopping_gate=False)
