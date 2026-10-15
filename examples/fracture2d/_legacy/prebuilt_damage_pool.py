"""Prebuild exact sampled supports, then select each batch inside JIT."""
import jax
import numpy as np
from examples.fracture2d._legacy.core_halo_damage import validate_probabilities
from examples.fracture2d._legacy.precomputed_damage_sampling import sample_core_halo


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
            p_core=p['p_core'][slot])
        return batch_objective(net, batch, load)
    return objective
