import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from ml_collections import ConfigDict

from examples.icemelting3d.train import _build_network, _nondim_params
from examples.icemelting3d.losses import analytic_solution
from examples.icemelting3d.accuracy import _network, _field, _identity

jax.config.update('jax_default_matmul_precision', 'highest')
root = Path('/root/autodl-tmp/tf-logs/icemelting3d')
pilot = Path('reports/icemelting-weak-prior-20261024')
ref_path = pilot / 'reference.npz'
with np.load(ref_path, allow_pickle=False) as f:
    ref = {k: f[k] for k in f.files}
ref_id = json.loads(str(ref['identity']))
ref_i = int(np.flatnonzero(np.isclose(ref['time'] * ref_id['nondim']['T_c'], 4.))[0])
assert ref_i == 16
n = 128
chunk = 32768
mirror = (np.abs(np.arange(n) - (n - 1) / 2) - .5).astype(int)
truth = ref['phi'][ref_i][np.ix_(mirror, mirror, mirror)].reshape(-1)
v = jnp.linspace(0., 1., n)
x = jnp.stack(jnp.meshgrid(v, v, v, indexing='xy'), axis=-1).reshape(-1, 3)
xe = x[jr.choice(jr.PRNGKey(123), len(x), (8000,), replace=False)]
ts = jnp.asarray([0., .25, .5, .75, 1.])
report = {'physical_time_s': 4., 'grid': [n]*3,
          'reference': str(ref_path),
          'reference_snapshot_index': ref_i,
          'definition': 'norm(phi_pred-phi_ref)/norm(phi_ref), all 128^3 nodes at physical 4 s',
          'runs': {}}
specs = [('SOTA', 17600, .009576017037034035),
         ('baseline2', 19800, .010161906480789185),
         ('local_ad', 19800, .009224546141922474),
         ('ic_phase', None, None), ('ic_balanced', None, None)]
for name, step, expected in specs:
    old = step is not None
    historical_directory = 'legacy-raw-phase-soft-ic' if name == 'SOTA' else name
    directory = root/historical_directory if old else pilot/name
    c = ConfigDict(json.loads((directory/'config.json').read_text()))
    actual_activation = c.arch.activation
    if name in ('baseline2', 'local_ad'):
        assert actual_activation == 'gelu'
        actual_activation = 'tanh'  # Historical fallback, independently log-replay validated.
        c.arch.activation = actual_activation
    ckpt = directory / (f'ckpt_{step:06d}.eqx' if old else 'model_final.eqx')
    if old:
        net = _build_network(c, jr.split(jr.PRNGKey(42))[0])
    else:
        net = _network(c, jr.PRNGKey(42), True)
    net = eqx.tree_deserialise_leaves(ckpt, net)
    ident = _identity(c)
    assert ident['physical'] == ref_id['physical']
    assert ident['grid'] == ref_id['grid']
    assert ident['nondim']['L_c'] == ref_id['nondim']['L_c']
    assert ident['nondim']['xc'] == ref_id['nondim']['xc']
    if name != 'local_ad':
        assert ident['nl'] == ref_id['nl']
    N1, N2, N3, eps, den, vel, R = _nondim_params(c)
    t = 4. / c.nondim.T_c
    row = {'checkpoint': str(ckpt),
           'time_star': t, 'T_c': c.nondim.T_c, 'actual_activation': actual_activation,
           'hard_IC': not old, 'residual': 'transformed' if name == 'ic_balanced' else 'phase',
           'training_iterations_config': c.train.iterations}
    if old:
        sample_fn = eqx.filter_jit(lambda net: jax.vmap(lambda tt: jax.vmap(net, in_axes=(0,None))(xe,tt)[:,0])(ts))
        pp = sample_fn(net)
        target = jax.vmap(lambda tt: analytic_solution(xe,tt,R,vel,c.nondim.xc,den))(ts)
        replay = float(jnp.linalg.norm(pp-target)/jnp.linalg.norm(target))
        assert abs(replay-expected) < 2e-5, (name,replay,expected)
        row['logged_pooled_analytic_rel_l2'] = expected
        row['replayed_pooled_analytic_rel_l2'] = replay
        row['sample8000_terminal_analytic_rel_l2'] = float(jnp.linalg.norm(pp[-1]-target[-1])/jnp.linalg.norm(target[-1]))
        fn = eqx.filter_jit(lambda net, xx: jax.vmap(net, in_axes=(0,None))(xx,jnp.float32(t))[:,0])
    else:
        fn = eqx.filter_jit(lambda net, xx: _field(net,xx,jnp.asarray([t],dtype=jnp.float32),c,name)[0][:,0])
    pred = np.empty(n**3, dtype=np.float64)
    analytic = np.empty_like(pred)
    for start in range(0,n**3,chunk):
        ix = np.stack(np.unravel_index(np.arange(start,start+chunk),(n,n,n)),axis=1)
        xx = ix/(n-1)
        pred[start:start+chunk] = np.asarray(fn(net,jnp.asarray(xx,dtype=jnp.float32)))
        radius = np.linalg.norm(xx-np.asarray(c.nondim.xc),axis=1)
        analytic[start:start+chunk] = np.tanh((R-vel*t-radius)/den)
    row['terminal_analytic_rel_l2'] = float(np.linalg.norm(pred-analytic)/np.linalg.norm(analytic))
    if name != 'local_ad':
        row['terminal_graph_rel_l2'] = float(np.linalg.norm(pred-truth)/np.linalg.norm(truth))
        row['terminal_graph_solid_fraction_rel_l2'] = float(np.linalg.norm(pred-truth)/np.linalg.norm(truth+1.))
    report['runs'][name] = row
    print(name,json.dumps(row),flush=True)
Path('/tmp/icemelting-baseline-metric-audit-replay.json').write_text(json.dumps(report,indent=2)+'\n')
