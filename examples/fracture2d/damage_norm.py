"""Full-node L1 and the archived stabilized squared L16 AT2 objectives."""
import jax
import numpy as np

from examples.fracture2d.compact_damage_features import residual_power_loss
from examples.fracture2d.damage_residual_weighting import core_matvec
from examples.fracture2d.low_order_damage import make_loss_fn as low_order_loss


def make_loss_fn(*, gc0, gc1, gain, population_size, norm, power, floor):
    if norm not in ('l1', 'l16') or power != 16 or not np.isfinite(floor) or not 0 < floor < 1:
        raise ValueError('norm control requires explicit L1/L16, power16 and positive floor')
    image, ordinary = low_order_loss(gc0=gc0, gc1=gc1, gain=gain,
        population_size=population_size, norm='l1')

    def loss(net, context, load):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        if c['core_H'].shape != (population_size,) or c['core_local'].shape != (population_size,):
            raise ValueError('norm comparison requires all population rows')
        if norm == 'l1':
            return ordinary(net, c, load)
        d = image(net, c)
        residual = (gc0+2*c['core_H'])*d[c['core_local']]+gc1*core_matvec(
            d, c['operator_columns'], c['operator_values'])-2*c['core_H']
        return residual_power_loss(residual, c['H'], power=power, floor=floor)
    return image, loss


def eligible_l16(rows):
    """Common field/residual criterion, never compare unlike objective values."""
    names = {r['case'] for r in rows}
    if names != {'l1_adam', 'l16_adam', 'l16_lbfgs'} or len(rows) != 3:
        raise ValueError('selection requires the three explicit norm/optimizer controls')
    for row in rows:
        if not np.isfinite([row['same_H_damage_L2'], row['AT2_relative']]).all():
            raise ValueError('nonfinite field/residual comparison')
    baseline = next(r for r in rows if r['case'] == 'l1_adam')
    eligible = [r for r in rows if r['case'] != 'l1_adam'
        and r['same_H_damage_L2'] <= baseline['same_H_damage_L2']
        and r['AT2_relative'] <= baseline['AT2_relative']]
    return min(eligible, key=lambda r:r['same_H_damage_L2'])['case'] if eligible else None
