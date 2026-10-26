"""Independent CPU final-checkpoint audit and raw cross-seed convergence plots."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np
from scipy import ndimage
from scipy import sparse

from examples.corrosion2d import residual_experiment as kks

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name('phasefield-seed-stability-results')
REPORT = ROOT/'reports/phasefield-seed-stability-20261025'
LOGS = Path('/root/autodl-tmp/tf-logs')
OLD = {
    'corrosion2d': {
        42: LOGS/'corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed42',
        7: LOGS/'corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed7'},
    'electropolish2d': {
        7: LOGS/'electropolish2d/20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed7',
        42: LOGS/'electropolish2d/20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed42'},
    'icemelting3d': {42: LOGS/'icemelting3d/20261024-5s-soap-lr1e3-seed42'},
}


def relative(a, b):
    norm = np.linalg.norm(b)
    if norm <= 0:
        raise ValueError('Positive reference norm required')
    return float(np.linalg.norm(a-b)/norm)


def read_history(run):
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    return [row for row in rows if row['step'] <= 5000]


def late_summary(rows, key):
    values = np.array([r[key] for r in rows if 4000 <= r['step'] <= 5000])
    assert len(values) == 6 and np.all(np.isfinite(values))
    return {'median': float(np.median(values)), 'min': float(values.min()),
            'max': float(values.max()), 'evaluations': 6}


def config_parity(case, config, run):
    import importlib
    base = importlib.import_module(f'examples.{case}.configs.train_base').get_config()
    def flatten(d, prefix=''):
        result = {}
        for key, value in d.items():
            path = prefix+'.'+key if prefix else key
            if isinstance(value, dict):
                result.update(flatten(value, path))
            else:
                result[path] = value
        return result
    a, b = (flatten(json.loads(c.to_json())) for c in (base, config))
    assert set(b)-set(a) == {'train.seed'} and not set(a)-set(b)
    differences = {k: {'default': a[k], 'run': b[k]} for k in a if a[k] != b[k]}
    differences['train.seed'] = {'strategy': 'not a parameter', 'run': b['train.seed']}
    allowed = {'train.seed', 'log.run_name'}
    if case == 'icemelting3d':
        allowed.add('eval.graph_reference_path')
    assert set(differences) <= allowed, (run, differences)
    assert config.train.seed == 2026 and config.train.iterations == 5000
    return differences


def audit_kks(case, config, run, endpoint):
    reference = kks.load_reference(config)
    X, Y = reference['X'], reference['Y']
    points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    model = eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',
        kks.build_model(config, jax.random.split(jax.random.PRNGKey(config.train.seed))[0]))
    x, times = jnp.asarray(points, dtype=jnp.float32), jnp.asarray(reference['time'], dtype=jnp.float32)

    @eqx.filter_jit
    def evaluate(net):
        raw = jax.vmap(lambda t: jax.vmap(net.net, in_axes=(0, None))(x, t))(times)
        fields = net.constraint(raw)
        _, ct = jax.jvp(lambda ts: jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(x, t))(ts),
                        (jnp.ones(1),), (jnp.ones(1),))
        return raw, fields, ct[0, :, 1]

    raw, fields, ct = (np.asarray(v) for v in evaluate(model))
    fields = fields.reshape(len(times), *X.shape, 2).astype(float)
    with np.load(run/'prediction.npz') as saved:
        for key in ('X', 'Y', 'time'):
            np.testing.assert_array_equal(saved[key], reference[key])
        saved_fields = np.stack([saved['phi'], saved['c']], axis=-1)
    replay = float(np.max(np.abs(fields-saved_fields)))
    np.testing.assert_allclose(fields, saved_fields, rtol=0, atol=5e-6)
    metrics = {}
    for channel, name in enumerate(('phi', 'c')):
        for scope in ('terminal', 'pooled'):
            a, b = fields[..., channel], reference[name]
            value = relative(a[-1], b[-1]) if scope == 'terminal' else relative(a, b)
            metrics[f'{scope}_rel_l2_{name}'] = value
            np.testing.assert_allclose(value, endpoint[f'{scope}_rel_l2_{name}'], atol=3e-6, rtol=0)
    p = fields[..., 0]
    zero, one, _ = kks.boundary_values(config, points)
    free = ~(zero|one)
    row, col, weight = (reference[k] for k in ('graph_row', 'graph_col', 'graph_weight'))
    D = sparse.diags(np.bincount(row, weights=weight, minlength=len(points))) - sparse.csr_matrix(
        (weight, (row, col)), shape=(len(points), len(points)))
    cle, dc = config.physical.c_le, config.physical.c_se-config.physical.c_le
    q = .5*cle*(np.tanh(raw[..., 1].astype(float))-1)
    rp = reference['phi'].reshape(len(times), -1)
    rq = reference['c'].reshape(len(times), -1)-dc*(3*rp**2-2*rp**3)-cle
    dq, refdq = D@q[-1], D@rq[-1]
    transport = {'terminal_q_rel_l2_free': relative(q[-1, free], rq[-1, free]),
                 'terminal_Dq_rel_l2_free': relative(dq[free], refdq[free]),
                 'terminal_ct_rel_l2_free': relative(ct[free], (-kks.nondim(config)[0]*refdq)[free]),
                 'stable_CH_rms_free': float(np.sqrt(np.mean((ct+kks.nondim(config)[0]*dq)[free]**2)))}
    if np.any(p == .5):
        raise ValueError('Exact phi0.5 requires explicit contour treatment')
    if case == 'electropolish2d':
        counts = ((p[:, :-1]-.5)*(p[:, 1:]-.5) < 0).sum(axis=1)
        topology = {'vertical_crossings_by_time': [{str(int(k)): int(v) for k,v in
            zip(*np.unique(a, return_counts=True))} for a in counts],
            'single_interface_every_column_every_time': bool(np.all(counts == 1))}
    else:
        topology = {'liquid_components_by_time': [int(ndimage.label(a < .5)[1]) for a in p],
                    'reference_liquid_components_by_time': [int(ndimage.label(a < .5)[1]) for a in reference['phi']],
                    'terminal_liquid_area_fraction': float(np.mean(p[-1] < .5)),
                    'reference_terminal_liquid_area_fraction': float(np.mean(reference['phi'][-1] < .5))}
    ids = [0, 5, 10, 15, 20]
    plot_fields(case, X, Y, reference['phi'][ids], p[ids], np.array([0, .25, .5, .75, 1]))
    return {'field_metrics_cpu': metrics, 'transport_cpu': transport, 'topology': topology,
            'cpu_vs_saved_gpu_fields_max_abs_difference': replay}


def audit_melting(config, run, endpoint):
    from examples.icemelting3d.train import _build_phase_network
    with np.load(config.eval.graph_reference_path, allow_pickle=False) as data:
        ref = {key: data[key] for key in data.files}
    n = config.sample.Nx
    assert n == config.sample.Ny == config.sample.Nz == 128
    identity = json.loads(str(ref['identity']))
    assert identity['physical'] == config.physical.to_dict()
    assert identity['nl'] == config.nl.to_dict() and identity['grid'] == [n, n, n]
    assert identity['nondim']['T_c'] == config.nondim.T_c == 5.
    model = eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',
        _build_phase_network(config, jax.random.split(jax.random.PRNGKey(config.train.seed))[0]))
    axis = np.asarray(jnp.linspace(0., 1., n))
    fold = (np.abs(np.arange(n)-(n-1)/2)-.5).astype(int)
    predict = eqx.filter_jit(lambda net, x, t: jax.vmap(net, in_axes=(0, None))(x, t)[:, 0])
    prediction = np.empty(n**3, dtype=np.float64)
    chunk = config.eval.chunk_size
    for start in range(0, n**3, chunk):
        ids = np.arange(start, min(start+chunk, n**3))
        ijk = np.stack(np.unravel_index(ids, (n, n, n)), axis=1)
        points = axis[ijk[:, [1, 0, 2]]]
        actual = len(points)
        if actual < chunk:
            points = np.pad(points, ((0, chunk-actual), (0, 0)), mode='edge')
        prediction[start:start+actual] = np.asarray(predict(model, jnp.asarray(points), jnp.float32(1)))[:actual]
    prediction = prediction.reshape(n, n, n)
    truth = ref['phi'][-1][np.ix_(fold, fold, fold)].swapaxes(0, 1)
    terminal = relative(prediction, truth)
    np.testing.assert_allclose(terminal, endpoint['graph_terminal_full_rel_l2'], atol=2e-7, rtol=.005)
    fractions = {'solid_volume_fraction': float(np.mean(prediction > 0)),
                 'reference_solid_volume_fraction': float(np.mean(truth > 0)),
                 'solid_fraction_relative_l2': relative(prediction+1, truth+1)}
    X, Y = np.meshgrid(axis, axis, indexing='xy')
    points = jnp.asarray(np.stack([X.ravel(), Y.ravel(), np.full(X.size, axis[n//2])], axis=-1))
    p, targets = [], []
    for time, index in zip([0, .25, .5, .75, 1], [0, 5, 10, 15, 20]):
        p.append(np.asarray(predict(model, points, jnp.float32(time))).reshape(n, n))
        targets.append(ref['phi'][index][np.ix_(fold, fold, [fold[n//2]] )][:, :, 0].T)
    plot_fields('icemelting3d', X, Y, np.array(targets), np.array(p), np.array([0, .25, .5, .75, 1]))
    return {'terminal_graph_full_phi_relative_l2_cpu': terminal,
            'phase_max_absolute_error': float(np.max(np.abs(prediction-truth))), **fractions}


def plot_fields(case, X, Y, truth, predicted, times):
    error = np.abs(predicted-truth)
    lower = -1 if case == 'icemelting3d' else 0
    fig, axes = plt.subplots(3, 5, figsize=(15, 8 if case == 'icemelting3d' else 6), layout='constrained')
    for row, values in enumerate((truth, predicted, error)):
        for column, time in enumerate(times):
            ax = axes[row, column]
            im = ax.pcolormesh(X, Y, values[column], shading='nearest', rasterized=True,
                vmin=lower if row < 2 else 0, vmax=1 if row < 2 else error.max(),
                cmap='viridis' if row < 2 else 'magma')
            ax.set(aspect='equal')
            if row == 0:
                ax.set_title(f't*={time:.2f}')
            if column == 0:
                ax.set_ylabel(('Reference', 'Seed2026 PINN', 'Absolute error')[row]+'\ny*')
            if row == 2:
                ax.set_xlabel('x*')
            if row == 1:
                ax.text(.02, .02, f'L2={relative(predicted[column], truth[column]):.2e}',
                        transform=ax.transAxes, fontsize=8,
                        bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
            if row == 1 and column == 4:
                phi_image = im
    fig.colorbar(phi_image, ax=axes[:2, :].ravel().tolist(), label='phi', shrink=.8)
    fig.colorbar(im, ax=axes[2, :].ravel().tolist(), label='Absolute error', shrink=.8)
    extra = '; z*='+str(float(X[0, X.shape[1]//2])) if case == 'icemelting3d' else ''
    fig.suptitle(case+': independent seed2026, final5000 updates'+extra)
    fig.savefig(str(OUT)+'-'+case+'-phi.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Use JAX_PLATFORMS=cpu')
    jax.config.update('jax_default_matmul_precision', 'highest')
    launch = json.loads((REPORT/'launch.json').read_text())
    assert launch['status'] == 'completed' and len(launch['runs']) == 3
    result = {'seed': 2026, 'budget': 5000, 'cases': {},
              'scope': 'One predeclared additional seed per fixed strategy; no claim of statistical robustness'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    for ax, spec in zip(axes, launch['runs']):
        assert spec['returncode'] == 0
        case, run = spec['case'], Path(spec['run'])
        config = ConfigDict(json.loads((run/'config.json').read_text()))
        summary = json.loads((run/'summary.json').read_text())
        assert summary['updates'] == 5000 and (run/'model_final.eqx').read_bytes() == (run/'ckpt_005000.eqx').read_bytes()
        history = read_history(run)
        np.testing.assert_array_equal([r['step'] for r in history], np.arange(0, 5001, 200))
        row = {'run': str(run), 'config_differences_from_strategy': config_parity(case, config, run),
               'endpoint': history[-1], 'existing_same_strategy_runs': {str(k): str(v) for k,v in OLD[case].items()}}
        endpoint = summary['final_metrics']
        if case != 'icemelting3d':
            for key in ('terminal_rel_l2_phi', 'terminal_rel_l2_c', 'terminal_rel_l2_q', 'terminal_rel_l2_Dq_free'):
                row['late_'+key] = late_summary(history, key)
                assert endpoint[key] == history[-1][key]
            row['independent_cpu'] = audit_kks(case, config, run, endpoint)
        else:
            for key in ('graph_terminal_rel_l2', 'analytic_terminal_rel_l2'):
                row['late_'+key] = late_summary(history, key)
            row['independent_cpu'] = audit_melting(config, run, endpoint)
        colors = {42: '#7c6f64', 7: '#d5791b', 2026: '#1677b8'}
        for seed, path in {**OLD[case], 2026: run}.items():
            values = read_history(path)
            steps = [r['step'] for r in values]
            if case == 'icemelting3d':
                ax.plot(steps, [r['graph_terminal_rel_l2'] for r in values],
                        color=colors[seed], label=f'seed{seed}, phi sampled', lw=1.6)
                full = [r for r in values if 'graph_terminal_full_rel_l2' in r]
                ax.scatter([r['step'] for r in full], [r['graph_terminal_full_rel_l2'] for r in full],
                           color=colors[seed], marker='o', s=20, label=f'seed{seed}, phi full')
            else:
                for name, ls in (('phi', '-'), ('c', '--')):
                    ax.plot(steps, [r['terminal_rel_l2_'+name] for r in values],
                            color=colors[seed], ls=ls, label=f'seed{seed}, {name}', lw=1.6)
            ax.set(yscale='log', xlabel='Completed updates', ylabel='Terminal physical-field relative L2', title=case)
            ax.grid(which='both', alpha=.18)
            ax.legend(fontsize=8)
        result['cases'][case] = row
        print(case, json.dumps(row['independent_cpu']), flush=True)
    fig.suptitle('Fixed strategies, new predeclared seed2026 vs previous seeds; raw200-update evaluations')
    fig.savefig(str(OUT)+'-convergence.png', dpi=180, bbox_inches='tight')
    fig.savefig(str(OUT)+'-convergence.svg', bbox_inches='tight')
    plt.close(fig)
    OUT.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
