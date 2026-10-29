"""Frozen-checkpoint loss/gradient balance and independent graph diagnostics.

Run with .venv Python from repo root; no parameter or reference updates.
"""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from ml_collections import ConfigDict
from scipy import sparse

from nonlocal_pinn.operators import build_D
from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.corrosion2d.residual_experiment import boundary_values, grid, nondim
from examples.electropolish2d.operator_data import load_dataset
from examples.electropolish2d.operator_training import _build_network, make_loss_vector, predict_fields
from examples.electropolish2d.online_conditions import make_sampler

OUT = Path('docs/tasks/archive')
RUN = Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-adam-01')
NAMES = ['AC', 'CH', 'top', 'bottom', 'IC']
c = ConfigDict(json.loads((RUN/'config.json').read_text()))
configure_gpu_policy(c.train.gpu_policy)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
start = time.perf_counter()


def save(value):
    (OUT/'electro-online-balance-diagnosis.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def relative(a, b):
    denominator = np.linalg.norm(b)
    if denominator == 0:
        raise ValueError('Undefined zero-denominator diagnostic')
    return float(np.linalg.norm(a-b)/denominator)


def stats(a):
    return dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))


log = [json.loads(line) for line in (RUN/'loss.jsonl').read_text().splitlines()]
parts = np.array([v['parts'] for v in log]); weights = np.array([v['weights'] for v in log])
steps = np.array([v['step'] for v in log]); bias = np.array(c.loss.gradnorm_bias)
weighted = parts*weights*bias
log_summary = {}
for name, select in [('all_logged', np.ones(len(log), dtype=bool)), ('last1000_logged', steps > 4000)]:
    p, w, v = parts[select], weights[select], weighted[select]
    log_summary[name] = dict(count=int(select.sum()),
        scaled_loss_median=dict(zip(NAMES, np.median(p, 0).tolist())),
        weight_median=dict(zip(NAMES, np.median(w, 0).tolist())),
        weighted_loss_median=dict(zip(NAMES, np.median(v, 0).tolist())),
        AC_to_CH_weighted_loss=stats(v[:, 0]/v[:, 1]),
        cap_logged_fraction=dict(zip(NAMES, np.mean(w >= c.loss.gradnorm_weight_cap, 0).tolist())))
parity = {}
for source in ('examples/electropolish2d/operator_training.py',
               'examples/electropolish2d/online_conditions.py',
               'examples/corrosion2d/residual_experiment.py',
               'examples/corrosion2d/residuals.py',
               'src/nonlocal_pinn/nn/conditional.py', 'src/nonlocal_pinn/operators/__init__.py'):
    parity[source] = (Path(source).read_bytes() == (RUN/'source'/source).read_bytes())
assert all(parity.values()), parity
result = dict(run=str(RUN), backend=jax.default_backend(), source_parity=parity,
    method='Frozen checkpoints; no optimizer updates; final weights last refreshed at update4976.',
    labels=NAMES, nondimensional_coefficients=list(nondim(c)),
    loss_scales=[c.loss.ac_scale, c.loss.ch_scale], log_summary=log_summary,
    final_logged=log[-1], gradients=[], fields=[])
save(result)
print('log summary', json.dumps(log_summary['last1000_logged']), flush=True)

data, metadata, refs = load_dataset(c)
xx, yy, points = grid(c); x = jnp.asarray(points, dtype=jnp.float32)
op = build_D(x, c.nl.delta, method='sparse', kernel=c.nl.kernel)
zero, one, _ = boundary_values(c, points); free = ~(zero | one)
ic_ids = np.random.default_rng(c.train.seed).choice(len(points), c.operator.ic_points, replace=False)
bc_ids = np.r_[np.flatnonzero(zero), np.flatnonzero(one)]
n_bc = int(zero.sum()); assert n_bc == one.sum()
batches = []
for ids, sub, core, _ in op.build_subgraph_pool(c.sample.K_core, c.sample.K_pool):
    node_ids = jnp.concatenate([ids, jnp.asarray(ic_ids), jnp.asarray(bc_ids)])
    batches.append((x[node_ids], sub, core & jnp.asarray(free)[ids]))
online_sample, geometry = make_sampler(c)
keys = jr.split(jr.PRNGKey(c.train.seed), 3)
template = _build_network(c.arch, c.operator.image_shape, keys[0])
initial_net = eqx.tree_deserialise_leaves(RUN/'model_initial.eqx', template)
net = eqx.tree_deserialise_leaves(RUN/'model_final.eqx', template)
actual_coefficients = np.load(RUN/'online_coefficients.npy')
loss_fn = make_loss_vector(c, nondim(c), len(ic_ids), n_bc)


@eqx.filter_jit
def gradients(model, times, payload, sub, mask):
    values, pullback = eqx.filter_vjp(lambda m: loss_fn(m, times, payload, sub, mask), model)
    vectors = []
    for i in range(5):
        grad = pullback(jax.nn.one_hot(i, 5))[0]
        vectors.append(jnp.concatenate([a.ravel() for a in jax.tree.leaves(grad) if eqx.is_array(a)]))
    return values, jnp.stack(vectors)


def gradient_record(model, iteration, checkpoint):
    queries, sub, mask = batches[iteration % len(batches)]
    images, local, coefficients, accepted = online_sample(
        jr.fold_in(keys[2], iteration), queries, geometry(queries))
    assert int(accepted) >= c.operator.condition_batch
    np.testing.assert_array_equal(coefficients, actual_coefficients[iteration])
    _, k_time = jr.split(jr.fold_in(keys[1], iteration))
    times = jr.uniform(k_time, (c.sample.Nt,), minval=c.sample.t_min, maxval=1.)
    payload = images, local, queries
    values, g = gradients(model, times, payload, sub, mask)
    values = np.asarray(values, dtype=float); g = np.asarray(g, dtype=float)
    l1 = np.sum(abs(g), 1); l2 = np.linalg.norm(g, axis=1)
    cosine = (g @ g.T)/l2[:, None]/l2[None, :]
    clipped = np.clip(l1, 1e-6, 1e6)
    fresh = np.clip(clipped.mean()/clipped, 1e-6, c.loss.gradnorm_weight_cap)
    scenarios = {}
    for name, w in [('unit', np.ones(5)), ('fresh_balance', fresh), ('saved_final_balance', weights[-1])]:
        wg = g*(w*bias)[:, None]; total = wg.sum(0)
        norms = np.linalg.norm(wg, axis=1)
        scenarios[name] = dict(weights=w.tolist(), weighted_loss=(values*w*bias).tolist(),
            gradient_l2=norms.tolist(), gradient_norm_fraction=(norms/norms.sum()).tolist(),
            total_gradient_l2=float(np.linalg.norm(total)),
            cancellation_ratio=float(np.linalg.norm(total)/norms.sum()),
            loss_directional_derivative_along_negative_total=(-g@total).tolist(),
            normalized_alignment_with_total=(g@total/l2/np.linalg.norm(total)).tolist(),
            clipping_active=bool(np.linalg.norm(total) > c.train.grad_clip))
    entry = dict(checkpoint=checkpoint, actual_online_iteration_zero_based=iteration,
        physical_time_fraction=np.asarray(times).tolist(), values=values.tolist(),
        gradient_l1=l1.tolist(), gradient_l2=l2.tolist(), cosine=cosine.tolist(), scenarios=scenarios)
    if checkpoint == 'initial':
        np.testing.assert_allclose(values, parts[0], rtol=5e-5, atol=2e-6)
        np.testing.assert_allclose(fresh, weights[0], rtol=5e-5, atol=2e-6)
        entry['initial_logged_loss_and_weight_replay_pass'] = True
    result['gradients'].append(entry); save(result)
    print('gradient', checkpoint, iteration, 'cos AC/CH', cosine[0, 1],
          'fresh norms', scenarios['fresh_balance']['gradient_l2'], flush=True)
    return times, payload, sub, mask, g


gradient_record(initial_net, 0, 'initial')
for iteration in (0, 1249, 2499, 4999):
    probe = gradient_record(net, iteration, 'final')

# Check scalar-objective gradient equals the sum of extracted gradients.
times, payload, sub, mask, g = probe
w = weights[-1]*bias
scalar_gradient = eqx.filter_jit(eqx.filter_grad(lambda m:
    jnp.dot(jnp.asarray(w, dtype=jnp.float32), loss_fn(m, times, payload, sub, mask))))(net)
flat = np.concatenate([np.asarray(a).ravel() for a in jax.tree.leaves(scalar_gradient) if eqx.is_array(a)])
gap = np.linalg.norm(flat-w@g)/np.linalg.norm(w@g)
assert gap < 2e-4, gap
result['weighted_gradient_linearity_relative_gap'] = float(gap); save(result)
print('weighted gradient linearity', gap, flush=True)

ts = jnp.asarray(refs[0]['time'], dtype=jnp.float32)


@eqx.filter_jit
def fields_dt(model, image, known):
    return jax.jvp(lambda t:predict_fields(model, image, x, t, known, c),
                   (ts,), (jnp.ones_like(ts),))


def interface_heights(phase):
    crossing = (phase[:, :-1] > .5) != (phase[:, 1:] > .5)
    if not np.all(crossing.sum(axis=1) == 1):
        raise ValueError('Roughness statistic requires exactly one vertical crossing')
    k = crossing.argmax(axis=1)
    it = np.arange(phase.shape[0])[:, None]; ix = np.arange(phase.shape[2])[None, :]
    lower, upper = phase[it, k, ix], phase[it, k+1, ix]
    return yy[k, ix] + (.5-lower)/(upper-lower)*(yy[k+1, ix]-yy[k, ix])


def physical_terms(phi, conc, A):
    nch, n1, n2, n3, dc, cle = nondim(c)
    q = conc-dc*(3*phi**2-2*phi**3)-cle
    dphi = (A@phi.T).T; dq = (A@q.T).T
    ac_terms = np.stack([n3*dphi, n2*(4*phi**3-6*phi**2+2*phi), -n1*q*dc*(6*phi-6*phi**2)], -1)
    return q, dq, ac_terms, nch*dq


with np.load(RUN/'prediction.npz') as f:
    saved_prediction = f['prediction']
field_arrays = {}
for i, (index, ref) in enumerate(zip(metadata['eval_indices'], refs)):
    if index not in metadata['test_indices']:
        continue
    f, dt = fields_dt(net, jnp.asarray(data['images'][index]), jnp.asarray(data['initial'][index]))
    f, dt = np.asarray(f, dtype=float), np.asarray(dt, dtype=float)
    saved = saved_prediction[i].reshape(f.shape)
    np.testing.assert_allclose(f, saved, rtol=0, atol=2e-5)
    target = np.stack([ref['phi'], ref['c']], -1).reshape(f.shape)
    phi, conc = f[..., 0], f[..., 1]; rp, rc = target[..., 0], target[..., 1]
    n = len(points)
    A = sparse.diags(np.bincount(ref['graph_row'], weights=ref['graph_weight'].astype(float), minlength=n))
    A -= sparse.csr_matrix((ref['graph_weight'].astype(float), (ref['graph_row'], ref['graph_col'])), shape=(n, n))
    q, dq, ac_terms, transport = physical_terms(phi, conc, A)
    rq, rdq, rt, rtransport = physical_terms(rp, rc, A)
    ac = dt[..., 0]+ac_terms.sum(-1); ch = dt[..., 1]+transport
    ph = interface_heights(phi.reshape(len(ts), c.sample.Ny, c.sample.Nx))
    rh = interface_heights(rp.reshape(len(ts), c.sample.Ny, c.sample.Nx))
    entry = dict(index=index, replay_max_gap=float(np.max(abs(f-saved))),
        q_relative_l2=relative(q, rq), q_terminal_relative_l2=relative(q[-1], rq[-1]),
        Dq_relative_l2=relative(dq[:, free], rdq[:, free]), Dq_terminal_relative_l2=relative(dq[-1, free], rdq[-1, free]),
        by_time=[], term_names=['graph_phi', 'double_well', 'concentration_drive'],
        AC_residual_mean_abs_positive=float(np.mean(abs(ac[1:, free]))),
        CH_residual_mean_abs_positive=float(np.mean(abs(ch[1:, free]))))
    for t in range(len(ts)):
        interface = free & (rp[t] > .05) & (rp[t] < .95)
        union = free & (((rp[t] > .05) & (rp[t] < .95)) | ((phi[t] > .05) & (phi[t] < .95)))
        term = ac_terms[t, free]; rterm = rt[t, free]
        entry['by_time'].append(dict(time=float(ts[t]),
            phi_l2=relative(phi[t], rp[t]), c_l2=relative(conc[t], rc[t]), q_l2=relative(q[t], rq[t]),
            Dq_l2=relative(dq[t, free], rdq[t, free]),
            roughness_prediction=float(ph[t].std()), roughness_reference=float(rh[t].std()),
            mean_height_prediction=float(ph[t].mean()), mean_height_reference=float(rh[t].mean()),
            interface_height_rmse=float(np.sqrt(np.mean((ph[t]-rh[t])**2))),
            AC_mean_abs=float(np.mean(abs(ac[t, free]))), CH_mean_abs=float(np.mean(abs(ch[t, free]))),
            AC_interface_mean_abs=float(np.mean(abs(ac[t, interface]))), CH_interface_mean_abs=float(np.mean(abs(ch[t, interface]))),
            AC_union_interface_mean_abs=float(np.mean(abs(ac[t, union]))), CH_union_interface_mean_abs=float(np.mean(abs(ch[t, union]))),
            interface_node_fraction=float(interface.sum()/free.sum()),
            AC_term_mean_abs=np.mean(abs(term), 0).tolist(), AC_reference_term_mean_abs=np.mean(abs(rterm), 0).tolist(),
            AC_reference_rhs_mean_abs=float(np.mean(abs(-rterm.sum(-1)))),
            AC_time_derivative_mean_abs=float(np.mean(abs(dt[t, free, 0]))),
            CH_reference_rhs_mean_abs=float(np.mean(abs(rtransport[t, free]))),
            CH_time_derivative_mean_abs=float(np.mean(abs(dt[t, free, 1]))),
            CH_transport_mean_abs=float(np.mean(abs(transport[t, free])))))
    result['fields'].append(entry); save(result)
    print('field', index, 'Dq terminal', entry['Dq_terminal_relative_l2'], 'roughness t*.25',
          entry['by_time'][5]['roughness_prediction'], entry['by_time'][5]['roughness_reference'], flush=True)
    if index == 16:
        field_arrays = dict(time=np.asarray(ts), points=points, X=xx, Y=yy, prediction=f, target=target,
            dt=dt, q=q, reference_q=rq, Dq=dq, reference_Dq=rdq, AC=ac, CH=ch,
            ac_terms=ac_terms, reference_ac_terms=rt, height=ph, reference_height=rh)
np.savez_compressed(OUT/'electro-online-balance-fields.npz', **field_arrays)
result['seconds'] = time.perf_counter()-start
result['scope'] = ('Recorded scalar history plus frozen initial/final gradients on four predeclared actual online batches. '
    'Saved-final weights are stale on other batches; fresh balance is diagnostic, not a trained control. '
    'Euclidean gradient conflict does not reconstruct Adam moments or prove a unique cause.')
save(result)
print('COMPLETE', result['seconds'], flush=True)
