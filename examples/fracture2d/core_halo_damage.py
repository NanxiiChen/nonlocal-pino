"""Fixed-history NN evaluation on exact core+halo, with detached sampling."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d.damage_residual_weighting import core_matvec
from examples.fracture2d.compact_damage_features import residual_power_loss
from examples.fracture2d.range_damage_optimization import augmented_term


def pack_rows(matrix, *, width):
    """Pack fixed CSR rows without dropping any nonzero or changing coefficients."""
    a = csr_matrix(matrix, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    if (a.dtype != np.float64 or not np.isfinite(a.data).all()
            or type(width) is not int or width < 1
            or np.max(np.diff(a.indptr)) > width):
        raise ValueError('invalid f64 row packing or capacity truncates support')
    columns = np.zeros((a.shape[0], width), np.int32)
    values = np.zeros((a.shape[0], width), np.float64)
    for i in range(a.shape[0]):
        start, end = a.indptr[i:i+2]
        columns[i, :end-start] = a.indices[start:end]
        values[i, :end-start] = a.data[start:end]
    return columns, values


def predict_rows(net, features, basis_columns, basis_values, *, trunk_gain):
    """Evaluate trunk only at these rows; compact columns retain global IDs."""
    return (trunk_gain*jax.vmap(net.trunk)(features)[:, 0]
            +jnp.sum(basis_values*net.readout.coefficients[basis_columns], axis=1))


def validate_probabilities(probabilities, *, population_size):
    p = np.asarray(probabilities)
    if (type(population_size) is not int or population_size < 1
            or p.shape != (population_size,) or p.dtype != np.float64
            or not np.isfinite(p).all() or p.min() <= 0
            or not np.isclose(p.sum(), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError('sampling probabilities must be positive normalized f64')


def sampled_power_loss(residual, history, probabilities_at_core, *, population_size, power, floor):
    """Importance correction estimates the original nodal Lp moment.

    The nonlinear root is not an unbiased estimator of full Lp. Global forcing
    normalization uses full history, not the sampled-core history.
    """
    if (type(population_size) is not int or population_size < 1
            or type(power) is not int or power < 2 or power % 2 or not 0 < floor < 1
            or probabilities_at_core.shape != residual.shape):
        raise ValueError('invalid explicit sampled norm controls/shapes')
    history, p = jax.tree.map(jax.lax.stop_gradient, (history, probabilities_at_core))
    normalized = residual/(2*jnp.sqrt(jnp.mean(history**2)))
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.max(abs(normalized)), floor))
    moment = jnp.mean((normalized/scale)**power/(population_size*p))
    return scale**2*(moment+(floor/scale)**power)**(2/power)


def sampled_augmented_term(damage, history, lower, upper, penalty, p, *, population_size, gc0):
    history, lower, upper, penalty, p = jax.tree.map(jax.lax.stop_gradient,
        (history, lower, upper, penalty, p))
    scale = 2*jnp.sqrt(jnp.mean(history**2))/gc0
    low, high = -damage/scale, (damage-1)/scale
    terms = ((jnp.maximum(lower+penalty*low, 0)**2-lower**2)
             +(jnp.maximum(upper+penalty*high, 0)**2-upper**2))
    return jnp.mean(terms/(population_size*p))/(2*penalty)


def make_sampled_objective(*, gc0, gc1, power, floor, trunk_gain, population_size):
    def loss(net, context, load):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        halo = predict_rows(net, c['features'], c['basis_columns'], c['basis_values'], trunk_gain=trunk_gain)
        damage = halo[c['core_local']]
        diffusion = core_matvec(halo, c['operator_columns'], c['operator_values'])
        residual = (gc0+2*c['core_H'])*damage+gc1*diffusion-2*c['core_H']
        return (sampled_power_loss(residual, c['H'], c['p_core'],
                    population_size=population_size, power=power, floor=floor)
                +sampled_augmented_term(damage, c['H'], c['lower'], c['upper'], c['penalty'], c['p_core'],
                    population_size=population_size, gc0=gc0))
    return loss


def make_cached_objective(predict, matvec, *, gc0, gc1, power, floor):
    def loss(net, context, load):
        H, lower, upper, penalty = jax.tree.map(jax.lax.stop_gradient,
            (context['H'], context['lower'], context['upper'], context['penalty']))
        damage = predict(net)
        residual = (gc0+2*H)*damage+gc1*matvec(damage)-2*H
        return (residual_power_loss(residual, H, power=power, floor=floor)
                +augmented_term(damage, H, lower, upper, penalty, gc0=gc0))
    return loss


def crack_sampling_probabilities(points, *, start, end, width, boost, power):
    """Elasticity-style inverse-power refinement about a crack segment.

    rho >= 1 regularizes distance at the segment; a positive baseline retains
    support everywhere. Geometry is detached: this is sampling, not a PDE term.
    """
    a, b = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or points.dtype != jnp.float64 or a.shape != (2,) or b.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.sum((b-a)**2) == 0 or not np.isfinite(width) or width <= 0
            or not np.isfinite(boost) or boost < 0 or not np.isfinite(power) or power <= 0):
        raise ValueError('crack probabilities require explicit finite segment/width/boost/power and f64 points')
    x = jax.lax.stop_gradient(points)
    direction = jnp.asarray(b-a)
    t = jnp.clip((x-jnp.asarray(a))@direction/jnp.sum(direction**2), 0., 1.)
    distance2 = jnp.sum((x-(jnp.asarray(a)+t[:, None]*direction))**2, axis=1)
    raw = 1+boost*(1+distance2/width**2)**(-power/2)
    return jax.lax.stop_gradient(raw/jnp.sum(raw))
