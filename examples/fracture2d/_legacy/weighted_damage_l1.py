"""Positive detached damage-region weights for all-node L1 AT2 residuals."""
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.damage_residual_weighting import core_matvec
from examples.fracture2d._legacy.low_order_damage import make_loss_fn


def frozen_weights(damage, *, kind, epsilon, cap):
    d = np.asarray(damage)
    if (d.ndim != 1 or d.dtype != np.float64 or not np.isfinite(d).all()
            or d.min() < 0 or d.max() > 1 or kind not in ('uniform', 'reciprocal')
            or not np.isfinite(epsilon) or epsilon <= 0
            or (cap is not None and (not np.isfinite(cap) or cap < 1))):
        raise ValueError('damage weights need bounded FP64 damage and explicit positive controls')
    raw = np.ones_like(d) if kind == 'uniform' else 1/(1-d+epsilon)
    if cap is not None: raw = np.minimum(raw, cap)
    return raw/raw.mean(), dict(raw_min=float(raw.min()), raw_max=float(raw.max()),
        raw_mean=float(raw.mean()), normalization='uniform nodal mean', frozen=True)


def make_weighted_loss(*, gc0, gc1, gain, population_size):
    image, _ = make_loss_fn(gc0=gc0, gc1=gc1, gain=gain, population_size=population_size, norm='l1')
    def loss(net, context, load):
        del load
        c = jax.tree.map(jax.lax.stop_gradient, context)
        if c['weights'].shape != (population_size,) or c['core_H'].shape != (population_size,):
            raise ValueError('weighted diagnostic requires all population rows')
        d = image(net, c)
        r = (gc0+2*c['core_H'])*d[c['core_local']]+gc1*core_matvec(
            d, c['operator_columns'], c['operator_values'])-2*c['core_H']
        rho = r/(2*jnp.sqrt(jnp.mean(c['H']**2)))
        return jnp.sum(c['weights']*jnp.abs(rho))/jnp.sum(c['weights'])
    return image, loss
