"""Precomputed exact CSR core+halo pools and detached crack sampling."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

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

def validate_probabilities(probabilities, *, population_size):
    p = np.asarray(probabilities)
    if (type(population_size) is not int or population_size < 1
            or p.shape != (population_size,) or p.dtype != np.float64
            or not np.isfinite(p).all() or p.min() <= 0
            or not np.isclose(p.sum(), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError('sampling probabilities must be positive normalized f64')

def crack_sampling_probabilities(points, *, start, end, width, boost, power,
                                 tip_width, tip_boost, tip_power):
    """Elasticity-style inverse-power refinement about a crack segment.

    An optional tip bump is centered at start. A positive baseline retains
    support everywhere. Geometry is detached: this is sampling, not a PDE term.
    """
    a, b = np.asarray(start, np.float64), np.asarray(end, np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or points.dtype != jnp.float64 or a.shape != (2,) or b.shape != (2,)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.sum((b-a)**2) == 0 or not np.isfinite(width) or width <= 0
            or not np.isfinite(boost) or boost < 0 or not np.isfinite(power) or power <= 0
            or not np.isfinite([tip_width, tip_boost, tip_power]).all()
            or tip_width <= 0 or tip_boost < 0 or tip_power <= 0):
        raise ValueError('crack probabilities require explicit finite segment/width/boost/power and f64 points')
    x = jax.lax.stop_gradient(points)
    direction = jnp.asarray(b-a)
    t = jnp.clip((x-jnp.asarray(a))@direction/jnp.sum(direction**2), 0., 1.)
    distance2 = jnp.sum((x-(jnp.asarray(a)+t[:, None]*direction))**2, axis=1)
    raw = 1+boost*(1+distance2/width**2)**(-power/2)
    if tip_boost > 0:
        tip_distance2 = jnp.sum((x-jnp.asarray(a))**2, axis=1)
        raw = raw + tip_boost*(1+tip_distance2/tip_width**2)**(-tip_power/2)
    return jax.lax.stop_gradient(raw/jnp.sum(raw))

@dataclass(frozen=True)
class SamplingRows:
    columns: np.ndarray
    values: np.ndarray

def make_sampling_rows(operator, *, row_width):
    """Validate/canonicalize the full operator once, before any batch/JIT loop."""
    D = csr_matrix(operator, copy=True)
    if D.shape[0] == 0 or D.shape[0] != D.shape[1]:
        raise ValueError('sampling support requires a nonempty square operator')
    columns, values = pack_rows(D, width=row_width)
    columns.flags.writeable = False
    values.flags.writeable = False
    return SamplingRows(columns, values)

def sample_core_halo(rows, core, *, node_capacity):
    """Gather cached rows and remap the exact halo without per-core Python loops.

    Repeated cores retain their original order/multiplicity. Zero-coefficient
    padding never expands the actual halo; insufficient capacity fails loudly.
    """
    ids = np.asarray(core)
    if (not isinstance(rows, SamplingRows) or ids.ndim != 1 or ids.size == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= rows.columns.shape[0]
            or type(node_capacity) is not int or node_capacity < 1):
        raise ValueError('cached support needs valid core IDs and explicit positive halo capacity')
    neighbors, values = rows.columns[ids], rows.values[ids]
    active = values != 0
    halo, inverse = np.unique(np.concatenate((ids, neighbors[active])), return_inverse=True)
    if halo.size > node_capacity:
        raise ValueError('cached core+halo capacity would truncate original support')
    global_ids = np.zeros(node_capacity, np.int32)
    global_ids[:halo.size] = halo
    columns = np.zeros(neighbors.shape, np.int32)
    columns[active] = inverse[ids.size:]
    return dict(global_ids=global_ids, node_mask=np.arange(node_capacity) < halo.size,
                core_local=inverse[:ids.size].astype(np.int32), columns=columns, values=values)

def build_sampling_pool(rows, cores, probabilities, *, node_capacity):
    """Build a block once; callers supply fresh draws and retain their order."""
    ids = np.asarray(cores)
    if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('pool requires a nonempty integer (entries, cores) array')
    validate_probabilities(probabilities, population_size=rows.columns.shape[0])
    packed = [sample_core_halo(rows, core, node_capacity=node_capacity) for core in ids]
    pool = dict(global_ids=np.stack([p['global_ids'] for p in packed]),
        core_ids=ids.astype(np.int32), core_local=np.stack([p['core_local'] for p in packed]),
        operator_columns=np.stack([p['columns'] for p in packed]),
        operator_values=np.stack([p['values'] for p in packed]), p_core=np.asarray(probabilities)[ids])
    return pool, np.array([p['node_mask'].sum() for p in packed], np.int32)

def build_finite_sampling_pool(rows, cores, probabilities):
    """Pad a finite set to its exact largest complete support, without a guess."""
    ids = np.asarray(cores)
    if (ids.ndim != 2 or min(ids.shape) < 1
            or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= rows.columns.shape[0]):
        raise ValueError('finite pool requires valid nonempty integer core batches')
    validate_probabilities(probabilities, population_size=rows.columns.shape[0])
    counts = []
    for core in ids:
        neighbors, values = rows.columns[core], rows.values[core]
        counts.append(np.unique(np.concatenate((core, neighbors[values != 0]))).size)
    pool, actual = build_sampling_pool(rows, ids, probabilities, node_capacity=max(counts))
    if not np.array_equal(actual, counts):
        raise RuntimeError('finite pool support changed while packing')
    return pool, actual

def make_pool_objective(batch_objective, *, basis_columns, basis_values):
    """Use current frozen H/features with cached fixed support inside JIT."""
    bc, bv = jax.tree.map(jax.lax.stop_gradient, (basis_columns, basis_values))
    def objective(net, context, load):
        c = jax.tree.map(jax.lax.stop_gradient, context)
        p, slot = c['pool'], c['slot']
        ids, core = p['global_ids'][slot], p['core_ids'][slot]
        batch = dict(H=c['H'], core_H=c['H'][core], features=c['features'][ids],
            basis_columns=bc[ids], basis_values=bv[ids], core_local=p['core_local'][slot],
            operator_columns=p['operator_columns'][slot], operator_values=p['operator_values'][slot],
            p_core=p['p_core'][slot], weights=c['weights'][core])
        return batch_objective(net, batch, load)
    return objective
