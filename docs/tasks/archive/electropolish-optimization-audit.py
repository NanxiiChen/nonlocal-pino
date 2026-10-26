"""Independent saved-model field/topology/drive audit, CPU only."""
import argparse
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
from scipy import sparse
from examples.corrosion2d import residual_experiment as r

OUT = Path(__file__).with_suffix('')
ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')


def rel(a, b):
    if np.linalg.norm(b) <= 0:
        raise ValueError('Zero reference norm')
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def rms(a):
    return float(np.sqrt(np.mean(a*a)))


def crossings(p):
    if np.any(p == .5):
        raise ValueError('Exact phi0.5 node requires explicit crossing treatment')
    n = ((p[..., :-1, :]-.5)*(p[..., 1:, :]-.5) < 0).sum(axis=-2)
    return [{str(int(k)): int(v) for k, v in zip(*np.unique(a, return_counts=True))} for a in n]


def flatten(d, prefix=''):
    out = {}
    for key, value in d.items():
        path = prefix + '.' + key if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def main(*, partial=False):
    global OUT
    if partial:
        OUT = OUT.with_name(OUT.name+'-partial')
    if jax.default_backend() != 'cpu':
        raise ValueError('Use JAX_PLATFORMS=cpu')
    records = json.loads(Path(__file__).with_name('electropolish-optimization-controls.json').read_text())
    assert (1 <= len(records) <= 6 if partial else len(records) == 6)
    assert all(v['returncode'] == 0 for v in records)
    # Explicit adapter for the fixed six-control manifest: all use nodal data.
    for record in records:
        assert 'initial_sampling' not in record and 'boundary_sampling' not in record
        record['initial_sampling'], record['boundary_sampling'] = 'matched_graph', 'matched_graph'
    validation = Path(__file__).with_name('electropolish-optimization-validation.json')
    if validation.exists() and not partial:
        extra = json.loads(validation.read_text())
        assert len(extra) == 3 and all(v['returncode'] == 0 for v in extra)
        records += extra
    runs = {f"{v['formulation']}/{v['data_loss_mode']}/{v['initial_sampling']}/ramp{int(v['time_ramp'])}/seed{v['seed']}": Path(v['path']) for v in records}
    assert len(runs) == len(records), 'Duplicate run labels'
    runs.update({f'prior/{form}/seed42': ROOT/f'20261024-residual-{form}-seed42' for form in ('phase', 'latent')})
    result = {'scope': 'Fixed matched graph, terminal/pooled physical fields plus independent raw-KKS q/graph Dq; all final5000 checkpoints',
              'partial': partial, 'new_completed_runs': len(records), 'runs': {}}
    predictions, histories, sources, initial_by_seed = {}, {}, {}, {}
    base_config = None
    for label, run in runs.items():
        c = ConfigDict(json.loads((run/'config.json').read_text()))
        summary = json.loads((run/'summary.json').read_text())
        assert summary['updates'] == 5000 and not summary['hard_initial_condition']
        assert c.nondim.T_c == 5 and c.train.lr_init == .001 and c.train.optimizer == 'soap'
        assert c.loss.ac_scale == c.loss.ch_scale == 1. and c.loss.stagger_steps == 0
        initial = (run/'model_initial.eqx').read_bytes()
        if c.train.seed not in initial_by_seed:
            initial_by_seed[c.train.seed] = initial
        else:
            assert initial == initial_by_seed[c.train.seed], label
        if not label.startswith('prior/'):
            config = flatten(c.to_dict())
            if base_config is None:
                base_config = config
            allowed = {'log.run_name', 'experiment.formulation', 'experiment.data_loss_mode',
                       'loss.t_ramp_steps', 'loss.t_ramp_early', 'loss.t_ramp_late', 'train.seed',
                       'experiment.initial_sampling', 'experiment.boundary_sampling'}
            assert set(config) == set(base_config)
            differences = [k for k in config if config[k] != base_config[k]]
            assert set(differences) <= allowed, differences
            assert summary['initial_training_nodes'] == (5151 if c.experiment.initial_sampling == 'matched_graph' else 2000)
            boundary_count = 101 if c.experiment.boundary_sampling == 'matched_graph' else 128
            assert summary['boundary_training_nodes'] == {'pit': boundary_count, 'solid': boundary_count}
            assert summary['activation_counts'] == {'ac_only': 0, 'ch_only': 0, 'joint': 5000}
            assert summary['xla_flags'] == '--xla_gpu_deterministic_ops=true'
            assert summary['time_sampling']['last_update_max'] == 1.
            assert summary['time_sampling']['full_horizon_updates'] == (2000 if c.loss.t_ramp_steps else 5000)
            snapshot = {p: (run/'source'/p).read_bytes() for p in json.loads((run/'source/files.json').read_text())}
            if not sources:
                sources = snapshot
            else:
                assert sources == snapshot, 'Cohort source mismatch'
            assert (run/'ckpt_005000.eqx').read_bytes() == (run/'model_final.eqx').read_bytes()
        ref = r.load_reference(c)
        X, Y = ref['X'], ref['Y']
        n = X.size
        with np.load(run/'prediction.npz') as a:
            for name in ('X', 'Y', 'time'):
                np.testing.assert_array_equal(a[name], ref[name])
            p, conc = (a[name].astype(float) for name in ('phi', 'c'))
        fields = {}
        for name, a in (('phi', p), ('c', conc)):
            fields['terminal_rel_l2_'+name] = rel(a[-1], ref[name][-1])
            fields['pooled_rel_l2_'+name] = rel(a, ref[name])
            fields['rel_l2_by_time_'+name] = [rel(a[i], ref[name][i]) for i in range(len(a))]
            for key in ('terminal_rel_l2_'+name, 'pooled_rel_l2_'+name, 'rel_l2_by_time_'+name):
                np.testing.assert_allclose(fields[key], summary['final_metrics'][key], rtol=1e-12)
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        zero, one, _ = r.boundary_values(c, points)
        free = ~(zero|one)
        row, col, weight = (ref[k] for k in ('graph_row', 'graph_col', 'graph_weight'))
        matrix = sparse.diags(np.bincount(row, weights=weight, minlength=n)) - sparse.csr_matrix((weight, (row, col)), shape=(n, n))
        net = eqx.tree_deserialise_leaves(run/'model_final.eqx', r.build_model(c, jax.random.split(jax.random.PRNGKey(c.train.seed))[0]))
        x = jnp.asarray(points, dtype=jnp.float32)
        times = jnp.asarray(ref['time'], dtype=jnp.float32)

        @eqx.filter_jit
        def evaluate(m):
            raw = jax.vmap(lambda t: jax.vmap(m.net, in_axes=(0, None))(x, t))(times)
            _, ft = jax.jvp(lambda ts: jax.vmap(lambda t: jax.vmap(m, in_axes=(0, None))(x, t))(ts),
                           (jnp.ones(1),), (jnp.ones(1),))
            return raw, m.constraint(raw), ft[0, :, 1]

        raw, replay, ct = (np.asarray(a) for a in evaluate(net))
        saved = np.stack([p, conc], axis=-1).reshape(replay.shape)
        max_replay = float(np.max(np.abs(replay-saved)))
        np.testing.assert_allclose(replay, saved, atol=5e-6, rtol=0)
        dc, cle = r.nondim(c)[-2:]
        rp, rc = (ref[name].reshape(21, n) for name in ('phi', 'c'))
        rq = rc-dc*(3*rp**2-2*rp**3)-cle
        q = cle*.5*(np.tanh(raw[..., 1].astype(float))-1)
        dq, rdq = (matrix @ q[-1]), (matrix @ rq[-1])
        transport = {'terminal_q_rel_l2_free': rel(q[-1, free], rq[-1, free]),
                     'pooled_q_rel_l2_free': rel(q[:, free], rq[:, free]),
                     'terminal_Dq_rel_l2_free': rel(dq[free], rdq[free]),
                     'stable_CH_rms_free': rms((ct+r.nondim(c)[0]*dq)[free]),
                     'terminal_ct_rel_l2_free': rel(ct[free], -r.nondim(c)[0]*rdq[free]),
                     'reference_terminal_ct_rms_free': rms(-r.nondim(c)[0]*rdq[free]),
                     'initial_q_rmse_over_cle_free': rms((q[0]-rq[0])[free])/cle,
                     'top_s_rmse': rms(q[:, zero]/cle+1), 'bottom_s_rmse': rms(q[:, one]/cle),
                     'terminal_zero_cL_derivative_fraction': float(np.mean(np.asarray(jnp.tanh(jnp.asarray(raw[-1, :, 1])))**2 == 1))}
        history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        late = [v for v in history if v['step'] >= 4000]
        result['runs'][label] = {'path': str(run), 'physical_metrics': fields, 'transport': transport,
                                'configuration_differences_from_new_base': None if label.startswith('prior/') else differences,
                                'crossings_by_time': crossings(p), 'reference_crossings_by_time': crossings(ref['phi']),
                                'cpu_replay_max_abs_difference': max_replay,
                                'late_terminal_phi_median': float(np.median([v['terminal_rel_l2_phi'] for v in late])),
                                'late_terminal_c_median': float(np.median([v['terminal_rel_l2_c'] for v in late]))}
        histories[label] = history
        predictions[label] = p
        print(label, json.dumps({'fields': {k:v for k,v in fields.items() if 'by_time' not in k}, 'transport':transport,
                                 'crossings': crossings(p)[-1]}), flush=True)
    result['all_new_sources_identical'] = True
    result['reference_path'] = str(c.experiment.reference_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for label, history in histories.items():
        for ax, name in zip(axes, ('phi', 'c')):
            ax.plot([v['step'] for v in history], [v['terminal_rel_l2_'+name] for v in history], label=label)
            ax.set(xlabel='Completed updates', ylabel='Terminal relative L2', title=name, yscale='log')
            ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6)
    fig.savefig(str(OUT)+'.png', dpi=180)
    plt.close(fig)
    labels = list(runs)
    cols = 3
    rows = int(np.ceil((len(labels)+1)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.6*rows), constrained_layout=True)
    for ax, label in zip(axes.ravel(), ['Reference']+labels):
        p = ref['phi'][-1] if label == 'Reference' else predictions[label][-1]
        im = ax.pcolormesh(X, Y, p, vmin=0, vmax=1, cmap='viridis', shading='nearest')
        ax.contour(X, Y, ref['phi'][-1], levels=[.5], colors='black', linestyles='--', linewidths=.8)
        if label != 'Reference':
            ax.contour(X, Y, p, levels=[.5], colors='#ee503f', linewidths=.8)
        ax.set(title=label.replace('/matched_graph','/nodes').replace('/random_interior','/random'), xlabel='x/Lc', ylabel='y/Lc', aspect='equal')
    for ax in axes.ravel()[len(labels)+1:]:
        ax.set_visible(False)
    fig.suptitle('Electropolish: final5000 fields at5 s; black dashed reference, red PINN')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, label='phi')
    fig.savefig(str(OUT)+'-fields.png', dpi=160)
    plt.close(fig)
    OUT.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true', help='Audit only completed initial controls; never substitute this for the final report')
    args = parser.parse_args()
    main(partial=args.partial)
