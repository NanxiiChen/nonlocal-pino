"""Supplement: frozen final gradients near early residual peak (no training)."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from ml_collections import ConfigDict
from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.corrosion2d.residual_experiment import boundary_values, grid, nondim
from examples.electropolish2d.operator_training import _build_network, make_loss_vector
from examples.electropolish2d.online_conditions import make_sampler

out = Path('docs/tasks/archive')
run = Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-adam-01')
c = ConfigDict(json.loads((run/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
_, _, points = grid(c); x = jnp.asarray(points, dtype=jnp.float32)
op = build_D(x, c.nl.delta, method='sparse', kernel=c.nl.kernel)
zero, one, _ = boundary_values(c, points); free = jnp.asarray(~(zero | one))
ic = np.random.default_rng(c.train.seed).choice(len(points), c.operator.ic_points, replace=False)
bc = np.r_[np.flatnonzero(zero), np.flatnonzero(one)]
ids, sub, core, _ = op.build_subgraph_pool(c.sample.K_core, c.sample.K_pool)[0]
queries = x[jnp.concatenate([ids, jnp.asarray(ic), jnp.asarray(bc)])]
mask = core & free[ids]; n_bc = int(zero.sum())
sample, geometry = make_sampler(c)
keys = jr.split(jr.PRNGKey(c.train.seed), 3)
images, local, coefficients, accepted = sample(jr.fold_in(keys[2], 0), queries, geometry(queries))
assert int(accepted) >= c.operator.condition_batch
np.testing.assert_array_equal(coefficients, np.load(run/'online_coefficients.npy')[0])
net = eqx.tree_deserialise_leaves(run/'model_final.eqx',
    _build_network(c.arch, c.operator.image_shape, keys[0]))
loss = make_loss_vector(c, nondim(c), len(ic), n_bc)
times = jnp.array([.001, .003, .005, .02], dtype=jnp.float32)


@eqx.filter_jit
def probe(model):
    values, pb = eqx.filter_vjp(lambda m:loss(m, times, (images, local, queries), sub, mask), model)
    gradients = []
    for i in range(5):
        g = pb(jax.nn.one_hot(i, 5))[0]
        gradients.append(jnp.concatenate([v.ravel() for v in jax.tree.leaves(g) if eqx.is_array(v)]))
    gradients = jnp.stack(gradients)
    return values, gradients


values, g = probe(net); g = np.asarray(g, dtype=float)
l1 = np.sum(abs(g), axis=1); l2 = np.linalg.norm(g, axis=1)
cos = (g@g.T)/l2[:, None]/l2[None, :]
norms = np.clip(l1, 1e-6, 1e6)
fresh = np.clip(norms.mean()/norms, 1e-6, c.loss.gradnorm_weight_cap)
last = json.loads((run/'loss.jsonl').read_text().splitlines()[-1])
scenarios = {}
for name, weights in [('saved_final', np.array(last['weights'])), ('fresh', fresh)]:
    weighted = g*weights[:, None]; total = weighted.sum(0)
    weighted_l2 = np.linalg.norm(weighted, axis=1)
    scenarios[name] = dict(weights=weights.tolist(), gradient_l2=weighted_l2.tolist(),
        gradient_norm_fraction=(weighted_l2/weighted_l2.sum()).tolist(),
        alignment_with_total=(g@total/l2/np.linalg.norm(total)).tolist(),
        cancellation_ratio=float(np.linalg.norm(total)/weighted_l2.sum()))
result = dict(labels=['AC','CH','top','bottom','IC'], actual_condition_batch=0,
    times=np.asarray(times).tolist(), values=np.asarray(values).tolist(),
    cosine=cos.tolist(), gradient_l1=l1.tolist(), gradient_l2=l2.tolist(), scenarios=scenarios,
    scope='Frozen final model, actual batch0 spatial/condition inputs, new explicit early times; not historical gradients.')
(out/'electro-online-balance-early-gradient.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result), flush=True)
