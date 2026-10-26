"""Research-only late SOAP learning-rate decay; physical equations are unchanged."""

import numpy as np
import optax
from soap_jax import soap

from nonlocal_pinn.training.optimizers import make_optimizer


def make_tail_schedule(config):
    """Retain the original schedule up to an explicit start, then decay to end."""
    e, t = config.experiment, config.train
    start, end = e.lr_tail_start, e.lr_tail_end
    if isinstance(start, (bool, np.bool_)) or not isinstance(start, (int, np.integer)) or not 0 <= start < t.iterations:
        raise ValueError("lr_tail_start must be an integer in [0, iterations)")
    if not np.isfinite(end) or end < t.lr_min or end <= 0:
        raise ValueError("lr_tail_end must be finite, positive and >= lr_min")
    if (not np.isfinite(t.lr_init) or not np.isfinite(t.lr_min)
            or not 0 < t.lr_min <= t.lr_init or t.lr_decay_every <= 0
            or not np.isfinite(t.lr_decay_rate) or not 0 < t.lr_decay_rate <= 1):
        raise ValueError("Invalid base learning-rate schedule")
    base = optax.exponential_decay(t.lr_init, t.lr_decay_every, t.lr_decay_rate,
                                   end_value=t.lr_min)
    if start == 0:
        return base
    at_start = float(base(start))
    if end >= at_start:
        raise ValueError("lr_tail_end must be below the base rate at lr_tail_start")
    tail = optax.exponential_decay(at_start, t.iterations - start, end / at_start,
                                   end_value=end)
    return optax.join_schedules([base, tail], boundaries=[start])


def make_tail_optimizer(config):
    """Preserve optimizer state and settings; replace only the scalar LR schedule."""
    schedule = make_tail_schedule(config)
    if config.experiment.lr_tail_start == 0:
        return make_optimizer(config)
    t = config.train
    if t.optimizer != 'soap':
        raise ValueError("Late learning-rate control currently requires SOAP")
    inner = soap(learning_rate=schedule, b1=t.soap_b1, b2=t.soap_b2,
                 precondition_frequency=t.soap_precond_freq,
                 max_precond_dim=t.soap_max_precond_dim,
                 precondition_1d=t.soap_precond_1d)
    return optax.chain(optax.clip_by_global_norm(t.grad_clip), inner)
