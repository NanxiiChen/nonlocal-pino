"""Early residual concentration and condition-spread probes, no training."""
import json
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.electropolish2d.operator_data import load_dataset
from examples.electropolish2d.operator_training import _build_network, predict_fields
from examples.corrosion2d.residual_experiment import boundary_values, nondim

out = Path('docs/tasks/archive')
run = Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-adam-01')
c = ConfigDict(json.loads((run/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
data, meta, refs = load_dataset(c)
net = eqx.tree_deserialise_leaves(run/'model_final.eqx',
    _build_network(c.arch, c.operator.image_shape, jax.random.key(0)))
with np.load(run/'prediction.npz') as f:
    indices = np.flatnonzero(f['condition_indices'] >= c.operator.train_conditions)
    pred, ref, time = f['prediction'][indices], f['target'][indices], f['time']
spread = []
for k, t in enumerate(time):
    row = dict(time=float(t))
    for i, field in enumerate(('phi', 'c')):
        p, r = pred[:, k, ..., i], ref[:, k, ..., i]
        ps, rs = np.linalg.norm(p-p.mean(0)), np.linalg.norm(r-r.mean(0))
        row[field+'_predicted_spread_over_reference'] = float(ps/rs)
    spread.append(row)

index = 16; ref = refs[meta['eval_indices'].index(index)]
x = jnp.asarray(data['points'], dtype=jnp.float32)
zero, one, _ = boundary_values(c, data['points']); free = ~(zero | one)
n = len(x)
A = sparse.diags(np.bincount(ref['graph_row'], weights=ref['graph_weight'].astype(float), minlength=n))
A -= sparse.csr_matrix((ref['graph_weight'].astype(float), (ref['graph_row'], ref['graph_col'])), shape=(n, n))
nch, n1, n2, n3, dc, cle = nondim(c)
image = jnp.asarray(data['images'][index]); known = jnp.asarray(data['initial'][index])


@eqx.filter_jit
def evaluate(model, ts):
    return jax.jvp(lambda t:predict_fields(model, image, x, t, known, c),
                   (ts,), (jnp.ones_like(ts),))


def probe(count):
    # Resolve the near-zero layer, then use a uniform late grid; no targets.
    ts = np.r_[0., np.geomspace(1e-5, .05, count), np.linspace(.05, 1., count)[1:]]
    f, dt = evaluate(net, jnp.asarray(ts, dtype=jnp.float32))
    f, dt = np.asarray(f, dtype=float), np.asarray(dt, dtype=float)
    p, conc = f[..., 0], f[..., 1]
    q = conc-dc*(3*p**2-2*p**3)-cle
    ac = dt[..., 0]+n3*(A@p.T).T+n2*(4*p**3-6*p**2+2*p)-n1*q*dc*(6*p-6*p**2)
    ch = dt[..., 1]+nch*(A@q.T).T
    entries = []
    integrals = {}
    early = np.arange(count+1); late = np.arange(count, len(ts))
    for name, residual in [('AC', ac), ('CH', ch)]:
        absolute = np.mean(abs(residual[:, free]), axis=1)
        early_int = np.trapezoid(absolute[early], ts[early])
        late_int = np.trapezoid(absolute[late], ts[late])
        integrals[name] = dict(early=float(early_int), late=float(late_int),
            early_fraction=float(early_int/(early_int+late_int)),
            early_time_average=float(early_int/.05), late_time_average=float(late_int/.95),
            peak=float(absolute.max()), peak_time=float(ts[absolute.argmax()]))
    for i, t in enumerate(ts):
        entries.append(dict(time=float(t), AC=float(np.mean(abs(ac[i, free]))), CH=float(np.mean(abs(ch[i, free])))))
    return dict(nodes=len(ts), integrals=integrals, by_time=entries)


coarse = probe(41); fine = probe(81)
gaps = {}
for name in ('AC', 'CH'):
    gaps[name] = {interval:float(abs(coarse['integrals'][name][interval]/fine['integrals'][name][interval]-1))
                  for interval in ('early', 'late')}
# This is a quadrature refinement of a frozen network, not reference refinement.
assert max(v for g in gaps.values() for v in g.values()) < .03, gaps
result = dict(run=str(run), condition_indices=[16,17,18,19], spread=spread,
    early_interval=[0., .05], residual_condition=16, coarse=coarse, fine=fine,
    quadrature_refinement_relative_gaps=gaps,
    scope='Frozen final model, positive-time labels used only in ensemble-spread evaluation; no training.')
(out/'electro-online-balance-early.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print('spread at .05/.25/1', [spread[i] for i in (1,5,20)], flush=True)
print('residual integrals', fine['integrals'], 'quadrature gaps', gaps, flush=True)
