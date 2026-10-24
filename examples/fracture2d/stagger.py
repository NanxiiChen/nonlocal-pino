"""Mechanics-first stagger and exact P1 field-change stopping."""
import jax
import jax.numpy as jnp
import numpy as np

def damage_change_limit(loading, *, damage_max):
    """Optional state-scaled tolerance, evaluated outside parameter gradients."""
    if 'damage_stop' not in loading:
        return loading.damage_tolerance
    s = loading.damage_stop
    if (s.mode != 'scaled_max' or not np.isfinite([s.offset, s.factor, damage_max]).all()
            or min(s.offset, s.factor) <= 0 or not 0 <= damage_max <= 1):
        raise ValueError('scaled damage stopping needs positive offset/factor and damage maximum in [0,1]')
    return (damage_max+s.offset)*s.factor

def committed_history(parent, current_driver):
    """Freeze the preceding load's history floor, not the preceding outer H."""
    if parent.shape != current_driver.shape:
        raise ValueError('committed history and current driver shapes differ')
    return jax.lax.stop_gradient(jnp.maximum(parent, current_driver))

def damage_driving_field(parent, current_driver, *, mode):
    """Explicit maximum-history or instantaneous tensile-driver treatment."""
    if mode == 'maximum_history':
        return committed_history(parent, current_driver)
    if mode != 'current_psi_l1' or parent.shape != current_driver.shape:
        raise ValueError('unknown damage driving mode or inconsistent nodal fields')
    return jax.lax.stop_gradient(current_driver)

def mechanics_first_sweep(q, d, parent_H, load, carry, *, mechanics, driver,
                          damage, omega, history_mode='maximum_history'):
    """One mechanics call; returned q equilibrates d_previous, not new d.

    Damage callback returns (image, new_carry, diagnostics). The detached
    driver uses the explicit history_mode; the parent field is never mutated.
    """
    if not np.isfinite(omega) or not 0 < omega <= 1:
        raise ValueError('explicit stagger relaxation must be in (0, 1]')
    q_new, reaction, mechanical = mechanics(d, load, q)
    H = damage_driving_field(parent_H, jnp.asarray(driver(np.asarray(q_new))), mode=history_mode)
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
