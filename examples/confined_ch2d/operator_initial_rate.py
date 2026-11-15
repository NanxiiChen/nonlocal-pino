"""Input-only initial PDE rate for the opt-in conservative CH coordinate.

For D=-Laplacian and alpha=epsilon^2/beta, chi_t(0)=-M*mu0/alpha.
The exponential lift and an O(t^2) learned correction enforce this identity
without prescribing any evolved solution. This is not a time integrator.
"""
import numpy as np

from examples.confined_ch2d.operator_precision import compute_dtype


RATE_KEYS = ('initial_potential_rate', 'initial_rate_timescale')


def validate_initial_rate(c):
    settings = c.operator.initial_rate_lift
    if (set(settings) != {'kind', 'timescale'} or settings.kind != 'exact_pde_rate'
            or settings.timescale != 'stable_graph_bound'):
        raise ValueError('initial_rate_lift requires kind=exact_pde_rate, timescale=stable_graph_bound')
    if ('formulation' not in c.operator or c.operator.formulation != 'conservative_potential'
            or c.operator.metric != 'identity' or c.operator.action != 'train'
            or 'continuation' in c.train or 'augmentation' in c.operator
            or 'initial_rhs_condition' in c.operator):
        raise ValueError('Initial-rate lift requires cold unaugmented conservative-potential training')


def initial_rate_data(data, c):
    """Derive known rates on the same rounded graph used by the neural loss."""
    validate_initial_rate(c)
    dtype = compute_dtype(c)
    initial, volumes, coefficients = (
        np.asarray(data[key], dtype=dtype) for key in ('initial', 'volumes', 'coefficients'))
    neighbors = np.asarray(data['neighbors'])
    n = len(initial)
    if (initial.shape != (n,) or volumes.shape != (n,) or neighbors.ndim != 2
            or neighbors.shape[0] != n or coefficients.shape != neighbors.shape
            or neighbors.dtype.kind not in 'iu' or np.any(neighbors < 0) or np.any(neighbors >= n)
            or not all(np.isfinite(a).all() for a in (initial, volumes, coefficients))
            or np.any(coefficients < 0) or np.any(volumes < 0) or volumes.sum() <= 0):
        raise ValueError('Initial-rate lift requires a complete finite physical graph')
    p = c.physical
    if (not np.isfinite([p.beta, p.epsilon, p.mobility, p.terminal_time]).all()
            or min(p.beta, p.epsilon, p.mobility, p.terminal_time) <= 0):
        raise ValueError('Positive finite beta, epsilon, mobility and terminal time required')
    alpha = p.epsilon**2/p.beta
    lap0 = np.sum(coefficients*(initial[:, None]-initial[neighbors]), axis=-1)
    mu0 = p.beta*(initial**3-initial)+p.epsilon**2*lap0
    rate = (-p.mobility/alpha*mu0).astype(dtype)
    mean = float(volumes.astype(float)@initial/volumes.astype(float).sum())
    # Gershgorin bound uses neighbor coefficients, not the zero row sums of D.
    eigenvalue_bound = float(2*np.max(coefficients.astype(float).sum(axis=1)))
    stable_rate = max(0., p.mobility*eigenvalue_bound*(
        p.epsilon**2*eigenvalue_bound+p.beta*(3*mean**2-1)))
    tau = np.asarray(1/max(1/p.terminal_time, stable_rate), dtype=dtype)
    if not np.isfinite(rate).all() or not np.isfinite(tau) or tau <= 0:
        raise FloatingPointError('Nonfinite or unrepresentable prescribed initial lift')
    return dict(data, initial_potential_rate=rate, initial_rate_timescale=tau), dict(
        laplacian_upper_bound=eigenvalue_bound, stable_rate_upper_bound=stable_rate,
        timescale=float(tau), input_only=True, future_labels=False)
