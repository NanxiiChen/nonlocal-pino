"""Independent fixed/rolling input, physical-metric and timing audit/plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path('/root/nonlocal-pinn')
ROOT = Path('/root/autodl-tmp/tf-logs/icemelting3d')
PREFIX = '20261031-melting-spatial-01'
MODES = ('fixed_pool', 'rolling_pool')


def read(path):
    return json.loads(path.read_text())


def metrics(prediction, target):
    error = prediction.astype(np.float64)-target.astype(np.float64)
    reference = target.astype(np.float64)
    solid = (reference+1)/2
    return dict(pooled_phi=float(np.linalg.norm(error)/np.linalg.norm(reference)),
        terminal_phi=float(np.linalg.norm(error[-1])/np.linalg.norm(reference[-1])),
        pooled_solid=float(np.linalg.norm(error/2)/np.linalg.norm(solid)),
        terminal_solid=float(np.linalg.norm(error[-1]/2)/np.linalg.norm(solid[-1])))


def main():
    journal = read(ROOT/(PREFIX+'-pools-process.json'))
    if journal['status'] != 'training_complete':
        raise ValueError('Both5000controls must finish first')
    output = ROOT/(PREFIX+'-comparison'); output.mkdir(exist_ok=True)
    summaries, histories, streams, core_traces, results = [], [], [], [], []
    for mode in MODES:
        run = ROOT/(PREFIX+'-'+mode)
        c = read(run/'config.json'); summary = read(run/'summary.json')
        if summary['iterations'] != 5000 or not summary['cold_start']:
            raise ValueError('Incomplete or warm-started control')
        dataset = Path(c['operator']['dataset_path'])
        manifest = read(dataset/'manifest.json')
        with np.load(run/'sampling.npz') as f:
            streams.append((f['parameters'], f['times']))
            traces = {k: f[k] for k in ('draw_ids', 'entry_draw_ids', 'core_ids', 'actual_nodes')}
        core_traces.append(traces)
        if streams[-1][0].shape != (5000, 16, 10) or streams[-1][1].shape != (5000, 8):
            raise ValueError('Actual IC/time trace shape mismatch')
        if len(np.unique(streams[-1][0].reshape(-1, 10), axis=0)) != 80000:
            raise ValueError('Online IC stream unexpectedly repeats')
        for p in np.asarray(manifest['parameters'], dtype=np.float32):
            if np.any(np.all(streams[-1][0] == p, axis=-1)):
                raise ValueError('Training draw equals a held-out input')
        expected_draws = np.arange(5000) % 32 if mode == 'fixed_pool' else np.arange(5000)
        np.testing.assert_array_equal(traces['draw_ids'], expected_draws)
        covered = np.zeros(128**3, dtype=bool)
        for draw, cores in zip(traces['entry_draw_ids'], traces['core_ids'], strict=True):
            expected = np.sort(np.random.default_rng(int(draw)).choice(128**3, 256, replace=False))
            np.testing.assert_array_equal(cores, expected)
            covered[cores] = True
        spatial = read(run/'spatial_sampling.json')
        if spatial['unique_cores'] != int(covered.sum()):
            raise ValueError('Reported core coverage mismatch')
        if spatial['padded_nodes'] != 1792 or spatial['padded_edges'] != 1536:
            raise ValueError('Mismatched safe pool padding')
        per_input = []
        for entry in summary['test']['conditions']:
            index = entry['index']
            with np.load(run/f'prediction_{index:03d}.npz') as f:
                p, target = f['prediction'], f['target']
                with np.load(dataset/f'condition_{index:03d}.npz') as ref:
                    for name in ('points', 'parameters', 'time'):
                        np.testing.assert_array_equal(f[name], ref[name])
                    np.testing.assert_array_equal(target, ref['phi'])
            actual = metrics(p, target)
            for key, value in actual.items():
                np.testing.assert_allclose(value, entry[key], rtol=2e-6, atol=1e-10)
            per_input.append(dict(index=index, **actual))
        if [r['index'] for r in per_input] != manifest['test_indices']:
            raise ValueError('Changed evaluation input cohort')
        errors = {key: float(np.mean([r[key] for r in per_input])) for key in per_input[0] if key != 'index'}
        for key in ('pooled_phi', 'terminal_phi'):
            np.testing.assert_allclose(errors[key], summary['test'][key], rtol=2e-6, atol=1e-10)
        timing = read(run/'timing.json')
        update, sample, sampling = [np.asarray(timing[key]) for key in
            ('update_seconds', 'sample_seconds', 'spatial_seconds')]
        if any(len(a) != 5000 or not np.isfinite(a).all() for a in (update, sample, sampling)):
            raise ValueError('Invalid complete timing traces')
        warm = slice(100, None)
        process = next(r for r in journal['controls'] if r['mode'] == mode)
        results.append(dict(mode=mode, run=str(run), errors=errors, per_input=per_input,
            process_wall_seconds=process['process_wall_seconds'],
            setup_seconds=timing['setup_seconds'], trainer_seconds=timing['trainer_seconds'],
            first_GPU_update_seconds=float(update[0]),
            warm_GPU_ms=float(update[warm].mean()*1000),
            warm_condition_ms=float(sample[warm].mean()*1000),
            warm_spatial_ms=float(sampling[warm].mean()*1000),
            warm_effective_ms=float((update+sample+sampling)[warm].mean()*1000),
            peak_active_GiB=summary['peak_active_GiB'], spatial=spatial))
        for name in read(run/'source/files.json'):
            if (run/'source'/name).read_bytes() != (REPO/name).read_bytes():
                raise ValueError(f'Trained source changed: {name}')
        summaries.append(summary)
        histories.append([json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()
                          if json.loads(line)['split'] == 'monitor'])
    for filename in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (ROOT/(PREFIX+'-fixed_pool')/filename).read_bytes() != (ROOT/(PREFIX+'-rolling_pool')/filename).read_bytes():
            raise ValueError('Unequal initial model/Adam state')
    for first, second in zip(streams[0], streams[1], strict=True):
        np.testing.assert_array_equal(first, second)
    for key in ('core_ids', 'actual_nodes'):
        np.testing.assert_array_equal(core_traces[0][key], core_traces[1][key][:32])
    configs = [read(ROOT/(PREFIX+'-'+m)/'config.json') for m in MODES]
    configs[0]['sample']['spatial']['mode'] = configs[1]['sample']['spatial']['mode']
    configs[0]['log']['run_name'] = configs[1]['log']['run_name']
    if configs[0] != configs[1]:
        raise ValueError('Unexpected matched-control config difference')
    replay = read(REPO/'docs/tasks/melting-spatial-replay.json')
    failure = read(REPO/'docs/tasks/melting-spatial-full-residual-first-failure.json')
    if replay['status'] != 'passed' or failure['status'] != 'failed':
        raise ValueError('Preserve separate passed standard replay and failed supplemental gate')
    report = dict(status='GPU_validated_with_recorded_query_chunk_issue', controls=results,
        equality_checks=['initial model bytes', 'initial Adam bytes', '80000actual ICs',
                         '40000time draws', 'first32complete core sets', 'allactual draw IDs'],
        reference='Original8held-out native128^3solutions; L2on24^3diagnostic nodes at5times',
        scope='One common initialization, reused development inputs; no stable superiority or continuum claim',
        full_training='Canceled by user after3update performance-only probe', defaults_promoted=False,
        standard_GPU_checkpoint_replay=replay,
        supplemental_full_spatial_PDE_gate=failure,
        limitation='Same GPU/1024query evaluation only; query-partition and CPU portability not established',
        open_issue='docs/issues/melting-query-chunk-consistency.md',
        convergence=str(output/'convergence.png'), final_error_figure=str(output/'final_errors.png'))
    (output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (REPO/'docs/tasks/melting-spatial-results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), constrained_layout=True)
    labels = ('Fixed core+halo pool', 'Rolling core+halo pool')
    colors = ('#db8a2d', '#27876a')
    for h, label, color in zip(histories, labels, colors, strict=True):
        steps = np.asarray([r['step'] for r in h]); minutes = np.asarray([r['seconds'] for r in h])/60
        for row, key in enumerate(('pooled_phi', 'terminal_phi')):
            for col, x in enumerate((steps, minutes)):
                axes[row, col].plot(x, 100*np.asarray([r[key] for r in h]), label=label, color=color, lw=1.8)
                axes[row, col].set_yscale('log'); axes[row, col].grid(alpha=.2, which='both')
                axes[row, col].set_ylabel(('Pooled' if row == 0 else 'Terminal')+' phi relative L2 (%)')
    for ax in axes[:, 0]: ax.set_xlabel('Optimizer updates')
    for ax in axes[:, 1]: ax.set_xlabel('Timer incl. setup/JIT/evaluation (min)')
    axes[0, 0].legend()
    fig.suptitle('Melting 3D: cold start, 5000 Adam updates / B16 / Nt8 / unchanged 128^3 graph\n'
        'Curves: 2 fixed native monitor inputs; final 8 held-out scores reported separately')
    fig.savefig(output/'convergence.png', dpi=190); fig.savefig(output/'convergence.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, keys, title in zip(axes, (('pooled_phi', 'terminal_phi'), ('pooled_solid', 'terminal_solid')),
            ('Whole-field phase', 'Solid indicator (late shrinking-body sensitivity)'), strict=True):
        for i, (r, label, color) in enumerate(zip(results, labels, colors, strict=True)):
            bars = ax.bar(np.arange(2)+(i-.5)*.35, [100*r['errors'][k] for k in keys],
                          width=.34, color=color, label=label)
            ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
        ax.set_xticks([0, 1], ['Pooled (5 times)', 'Terminal (5 s)']); ax.set_ylabel('Mean relative L2 (%)')
        ax.set_title(title, fontsize=10); ax.margins(y=.25); ax.grid(axis='y', alpha=.2)
    fig.legend(labels, loc='upper center', bbox_to_anchor=(.5, 1.05), ncol=2, frameon=False)
    fig.suptitle('Final 5000-update checkpoints: same 8 native held-out ellipsoids', y=1.13)
    fig.savefig(output/'final_errors.png', dpi=190, bbox_inches='tight')
    fig.savefig(output/'final_errors.pdf', bbox_inches='tight'); plt.close(fig)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
