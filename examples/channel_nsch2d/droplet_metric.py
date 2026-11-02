"""Frozen, positive minority-phase residual norm; no reference fields."""
import numpy as np
from scipy.special import expit


def weights_from_phase(phase, measure, settings):
    """Return mean-one weights with finite contrast and a positive global floor."""
    phase = np.asarray(phase, dtype=np.float64)
    measure = np.asarray(measure, dtype=np.float64)
    if (phase.ndim != 2 or measure.shape != phase.shape[1:]
            or not np.isfinite(phase).all() or not np.isfinite(measure).all()
            or np.any(measure < 0) or measure.sum() <= 0):
        raise ValueError('Frozen droplet metric needs finite time/point phases and nonnegative nonzero measure')
    alpha, temperature = settings.contrast, settings.temperature
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Finite nonnegative contrast and positive temperature required')
    occupancy = expit(-phase / temperature)
    mean = np.sum(occupancy * measure[None], axis=1) / measure.sum()
    return (1 + alpha * occupancy) / (1 + alpha * mean[:, None])


def parent_phase(model, data, nodes, c):
    """Evaluate the frozen parent at every actual BDF node, in fixed-size batches."""
    import equinox as eqx
    import jax.numpy as jnp
    from examples.channel_nsch2d.network import encode, potentials

    batch = c.train.refinement.time_batch
    if len(nodes) % batch:
        raise ValueError('Frozen metric time nodes must form complete batches')
    evaluate = eqx.filter_jit(lambda net, times: potentials(net, encode(net, data),
        data['points'], times, data['amplitudes'], c)[..., 0])
    rows = [np.asarray(evaluate(model, jnp.asarray(t, dtype=c.train.compute_dtype)))
            for t in np.asarray(nodes).reshape(-1, batch)]
    phase = np.concatenate(rows)
    if phase.shape != (len(nodes), len(data['points'])) or not np.isfinite(phase).all():
        raise ValueError('Invalid frozen parent phase field')
    return phase


def metric_summary(phase, weights, measure):
    measure = np.asarray(measure, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    mean = np.sum(weight * measure[None], axis=1) / measure.sum()
    total = np.sum(weight * measure[None])
    return dict(minimum=float(weight.min()), maximum=float(weight.max()),
        mean_normalization_max_gap=float(np.max(np.abs(mean - 1))),
        negative_phase_weight_budget=float(np.sum(weight * measure[None] * (phase < 0)) / total),
        negative_phase_unweighted_budget=float(np.sum(measure[None] * (phase < 0)) / (len(phase) * measure.sum())),
        scope='Frozen parent predictions only; no reference or changing line-search weights')
