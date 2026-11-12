"""Explicit grid/horizon controls with a separate, immutable monitoring grid.

Training uses only online prescribed initial conditions and PDE/BC losses.
Changing the graph must not change the target used to compare final models.
The small native bank is monitoring only; the study evaluator uses local FD64.
"""
import numpy as np


def monitor_config(c):
    s = c.operator.horizon_study
    if set(s) != {'monitor_Nx', 'monitor_Ny', 'monitor_delta'}:
        raise ValueError('horizon_study requires explicit monitor_Nx/Ny/delta')
    if (c.nl.method != 'sparse' or c.operator.reference_action != 'load'
            or 'online' not in c.operator or c.arch.local_input != 'zero'
            or c.sample.spatial.mode != 'rolling_pool'):
        raise ValueError('Horizon controls require online sparse rolling training and zero query features')
    for name in ('monitor_Nx', 'monitor_Ny'):
        if isinstance(s[name], bool) or not isinstance(s[name], int) or s[name] < 3:
            raise ValueError(f'horizon_study.{name} must be an integer >=3')
    if not np.isfinite(s.monitor_delta) or s.monitor_delta <= 0:
        raise ValueError('Monitoring horizon must be positive and finite')
    hx, hy = 1/(c.sample.Nx-1), .5/(c.sample.Ny-1)
    if (not np.isclose(hx, hy, rtol=0, atol=1e-12)
            or not np.isfinite(c.nl.delta) or not max(hx, hy) < c.nl.delta < .25):
        raise ValueError('Study needs an isotropic grid and spacing < delta < .25')
    monitor = c.copy_and_resolve_references()
    with monitor.operator.unlocked():
        del monitor.operator.horizon_study
    monitor.sample.Nx, monitor.sample.Ny = s.monitor_Nx, s.monitor_Ny
    monitor.nl.delta = s.monitor_delta
    return monitor


def load_monitor(c):
    from examples.electropolish2d.operator_data import load_dataset
    # The existing loader still validates every physical coefficient, input
    # distribution, condition identity, reference graph and initial field.
    return load_dataset(monitor_config(c))


def operator_audit(op, points, delta, core_count):
    """Record actual FP32 quadrature quality; do not renormalize the operator."""
    x = np.asarray(points, np.float64)
    rows, cols = op._row_idx_np, op._col_idx_np
    w = -np.asarray(op._vals_np, np.float64)
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Expected nonnegative finite graph conductances')
    offsets = x[cols]-x[rows]
    xx, yy = (np.bincount(rows, weights=.5*w*offsets[:, k]**2, minlength=len(x))
              for k in range(2))
    interior = np.all((x > delta+1e-7) & (x < np.array([1., .5])-delta-1e-7), axis=1)
    if not np.any(interior):
        raise ValueError('No complete-neighborhood interior points')
    h = float(1/(len(np.unique(x[:, 0]))-1))
    return dict(particles=len(x), spacing=h, delta=float(delta), delta_over_h=float(delta/h),
        edges=len(rows), max_neighbors=op._k_max,
        max_nodes=min(len(x), core_count*(op._k_max+1)), max_edges=core_count*op._k_max,
        interior_quadratic_gain_x=float(xx[interior].mean()),
        interior_quadratic_gain_y=float(yy[interior].mean()),
        quadratic_gain_target=1., kernel_normalization='unchanged continuum cubic, no moment correction',
        scope='D approximates negative Laplacian; half sum(w*dx^2) should approach one')


def channel_relative_errors(prediction, reference):
    """Full point-space-time norms per channel, before channel averaging."""
    p, r = np.asarray(prediction, np.float64), np.asarray(reference, np.float64)
    if p.shape != r.shape or p.ndim != 4 or p.shape[-1] != 2:
        raise ValueError('Expected matching (time, y, x, 2) physical fields')
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Fields must be finite')
    numerator = np.sum((p-r)**2, axis=(0, 1, 2))
    denominator = np.sum(r*r, axis=(0, 1, 2))
    if np.any(denominator <= 0):
        raise ValueError('Every channel needs a positive reference norm')
    errors = np.sqrt(numerator/denominator)
    return dict(phi=float(errors[0]), c=float(errors[1]), channel_mean=float(errors.mean()),
                numerator_squared=numerator.tolist(), denominator_squared=denominator.tolist())
