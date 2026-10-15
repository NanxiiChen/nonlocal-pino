"""Experimental detached training weights; original AT2 and gates unchanged."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.compact_damage_features import residual_power_loss
from examples.fracture2d._legacy.range_damage_optimization import augmented_term


def make_weights(points, damage, *, mode, boost, damage_power, start, end, width):
    """Refresh outside L-BFGS; the returned array is fixed during a fit block."""
    if (points.ndim != 2 or points.shape[1] != 2 or damage.shape != (len(points),)
            or points.dtype != jnp.float64 or damage.dtype != jnp.float64):
        raise ValueError('weights require matching f64 points and damage')
    if (not np.isfinite(boost) or boost < 0 or not np.isfinite(damage_power)
            or damage_power <= 0 or not np.isfinite(width) or width <= 0):
        raise ValueError('explicit weight boost/power/width are invalid')
    a, b = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if (a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.sum((b-a)**2) == 0):
        raise ValueError('static crack segment must have finite distinct 2D ends')
    points, damage = jax.tree.map(jax.lax.stop_gradient, (points, damage))
    if mode == 'uniform':
        activity = jnp.zeros_like(damage)
    elif mode == 'static':
        direction = jnp.asarray(b-a)
        t = jnp.clip((points-jnp.asarray(a))@direction/jnp.sum(direction**2), 0., 1.)
        distance2 = jnp.sum((points-(jnp.asarray(a)+t[:, None]*direction))**2, axis=1)
        activity = jnp.exp(-distance2/(2*width**2))
    elif mode == 'damage':
        # Clipping controls weight construction only; physical damage is retained.
        activity = jnp.clip(damage, 0., 1.)**damage_power
    else:
        raise ValueError(f'unknown residual weight mode: {mode}')
    raw = 1+boost*activity
    return jax.lax.stop_gradient(raw/jnp.mean(raw))


def validate_weights(weights, history):
    """Host preflight, before JIT/optimization; no traced numerical fallback."""
    w, h = np.asarray(weights), np.asarray(history)
    if (w.shape != h.shape or w.ndim != 1 or w.size == 0
            or w.dtype != np.float64 or h.dtype != np.float64
            or not np.isfinite(w).all() or not np.isfinite(h).all()
            or w.min() <= 0 or h.min() < 0 or h.max() <= 0
            or not np.isclose(w.mean(), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError('weights/history must be matching finite f64, positive/normalized weights and positive history')


def weighted_power_loss(residual, history, weights, *, power, floor):
    """Square-root weighting of residual amplitude, preserving degree-two Lp."""
    history, weights = jax.tree.map(jax.lax.stop_gradient, (history, weights))
    return residual_power_loss(jnp.sqrt(weights)*residual, history, power=power, floor=floor)


def make_weighted_objective(image, matvec, *, gc0, gc1, power, floor):
    def objective(net, context, load):
        history, lower, upper, penalty, weights = jax.tree.map(jax.lax.stop_gradient, context)
        damage = image(net, history, load)
        residual = (gc0+2*history)*damage+gc1*matvec(damage)-2*history
        return (weighted_power_loss(residual, history, weights, power=power, floor=floor)
                +augmented_term(damage, history, lower, upper, penalty, gc0=gc0))
    return objective


def build_core_halo(operator, core, *, node_capacity, row_width):
    """Exact original rows; fixed padded halo, never rebuild kernel moments.

    Repeated core IDs are allowed for with-replacement importance sampling.
    Padding uses a zero coefficient and a mask; undersized capacities fail.
    """
    D = csr_matrix(operator, copy=True)
    D.sum_duplicates()
    D.eliminate_zeros()
    ids = np.asarray(core)
    if (D.dtype != np.float64 or D.shape[0] != D.shape[1]
            or not np.isfinite(D.data).all() or ids.ndim != 1 or len(ids) == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= D.shape[0]
            or type(node_capacity) is not int or node_capacity < 1
            or type(row_width) is not int or row_width < 1):
        raise ValueError('core+halo requires square finite f64 operator and valid explicit IDs/capacities')
    rows = D[ids]
    halo = np.unique(np.concatenate((ids, rows.indices)))
    if len(halo) > node_capacity or np.max(np.diff(rows.indptr)) > row_width:
        raise ValueError('core+halo capacity would truncate original support')
    global_ids = np.zeros(node_capacity, np.int32)
    global_ids[:len(halo)] = halo
    node_mask = np.arange(node_capacity) < len(halo)
    columns = np.zeros((len(ids), row_width), np.int32)
    values = np.zeros((len(ids), row_width), np.float64)
    for i in range(len(ids)):
        a, b = rows.indptr[i:i+2]
        columns[i, :b-a] = np.searchsorted(halo, rows.indices[a:b])
        values[i, :b-a] = rows.data[a:b]
    return dict(global_ids=global_ids, node_mask=node_mask,
                core_local=np.searchsorted(halo, ids).astype(np.int32),
                columns=columns, values=values)


def core_matvec(halo_values, columns, coefficients):
    return jnp.sum(coefficients*halo_values[columns], axis=1)


def importance_moment(sampled_values, probabilities, core, *, population_size):
    """With-replacement estimate of the full nodal mean of supplied moments.

    E[f_i/(N p_i)] = mean(f). A subsequent Lp root is nonlinear: the complete
    sampled loss is not claimed to be an unbiased estimator of that root.
    """
    p = jax.lax.stop_gradient(probabilities)
    return jnp.mean(sampled_values/(population_size*p[core]))
