"""Independent common-error/timing comparison and scientific convergence plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
REPO = Path('/root/nonlocal-pinn')
PREFIX = '20261031-electro-spatial-01'
BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
MODES = ('full', 'fixed_pool', 'rolling_pool')
LABELS = ('All coordinates', 'Fixed core+halo pool', 'Rolling core+halo pool')
COLORS = ('#325b9b', '#db8a2d', '#27876a')


def read(path):
    return json.loads(path.read_text())


def main():
    output = ROOT/(PREFIX+'-comparison')
    output.mkdir(exist_ok=True)
    journal = read(ROOT/(PREFIX+'-process.json'))
    if journal['status'] != 'training_complete':
        raise ValueError('All full-budget controls must finish first')
    records, histories = [], []
    for mode in MODES:
        run = ROOT/(PREFIX+'-'+mode)
        evaluation = read(run/'spatial_evaluation64.json')
        timing = read(run/'timing.json')
        spatial = read(run/'spatial_sampling.json')
        memory = read(run/'memory.json')
        if evaluation['status'] != 'passed':
            raise ValueError('Unvalidated final evaluation')
        warm = slice(100, None)
        updates = np.asarray(timing['update_seconds'])
        conditions = np.asarray(timing['sample_seconds'])
        sampling = np.asarray(timing['spatial_seconds'])
        latency = updates+conditions+sampling
        peak = max(m['allocator_stats']['peak_bytes_in_use'] for m in memory)
        process = next(r for r in journal['controls'] if r['mode'] == mode)
        item = dict(mode=mode, run=str(run), iterations=5000,
            process_wall_seconds=process['process_wall_seconds'],
            setup_seconds=timing['setup_seconds'], trainer_seconds=timing['trainer_seconds'],
            first_update_compile_seconds=timing['first_update_compile_and_call_seconds'],
            warm_GPU_update_ms=float(1000*updates[warm].mean()),
            warm_condition_ms=float(1000*conditions[warm].mean()),
            warm_spatial_ms=float(1000*sampling[warm].mean()),
            warm_effective_step_ms=float(1000*latency[warm].mean()),
            warm_effective_p95_ms=float(1000*np.quantile(latency[warm], .95)),
            GPU_peak_active_GiB=peak/1024**3,
            producer_wait_seconds=spatial['producer_wait_seconds'],
            replacement_transfer_seconds=spatial['replacement_transfer_seconds'],
            free_core_coverage=spatial['coverage'], unique_free_cores=spatial['unique_free_cores'],
            padded_nodes=spatial['padded_nodes'], padded_edges=spatial['padded_edges'],
            errors={g:evaluation['groups'][g]['statistics'] for g in evaluation['groups']},
            final_monitor=read(run/'summary.json')['final'])
        records.append(item)
        histories.append([json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()])
        # Independently replay each native/local64 metric from the saved field,
        # using norm ratios rather than the evaluator's sum-of-squares helper.
        with np.load(run/'spatial_evaluation64.npz') as saved:
            predictions, times = saved['prediction'], saved['time']
        regular = np.asarray(evaluation['regular_time_indices'])
        for i, prediction in enumerate(predictions):
            with np.load(BASE/f'condition_{i:03d}/reference.npz') as f:
                native = np.stack([f['phi'], f['c']], -1)
            with np.load(LOCAL/f'condition_{i:03d}/local101.npz') as f:
                local = f['fields'][regular]
                np.testing.assert_array_equal(f['time'], times)
            for group, target in (('native64', native), ('local101_64', local)):
                for ch, field in enumerate(('phi', 'c')):
                    p, r = prediction[regular, ..., ch], target[..., ch]
                    pooled = float(np.linalg.norm(p-r)/np.linalg.norm(r))
                    final = float(np.linalg.norm(p[-1]-r[-1])/np.linalg.norm(r[-1]))
                    np.testing.assert_allclose(pooled, evaluation['records'][i][group]['pooled_'+field], rtol=2e-12)
                    np.testing.assert_allclose(final, evaluation['records'][i][group]['terminal_'+field], rtol=2e-12)
        for group in evaluation['groups']:
            chosen = [r[group] for r in evaluation['records'] if group in r]
            for name, stats in evaluation['groups'][group]['statistics'].items():
                a = np.asarray([r[name] for r in chosen])
                np.testing.assert_allclose([a.mean(), np.median(a), np.quantile(a, .9), a.max()],
                    [stats[k] for k in ('mean', 'median', 'p90', 'maximum')], rtol=2e-12)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), constrained_layout=True)
    for h, label, color in zip(histories, LABELS, COLORS, strict=True):
        steps = np.asarray([r['step'] for r in h])
        seconds = np.asarray([r['seconds'] for r in h])/60
        for row, field in enumerate(('phi', 'c')):
            values = 100*np.asarray([r['test_pooled_rel_l2_'+field] for r in h])
            for col, x in enumerate((steps, seconds)):
                axes[row, col].plot(x, values, label=label, color=color, lw=1.8)
                axes[row, col].set_yscale('log')
                axes[row, col].grid(alpha=.2, which='both')
                axes[row, col].set_ylabel(field+' relative L2 (%)')
    for ax in axes[:, 0]:
        ax.set_xlabel('Optimizer updates')
    for ax in axes[:, 1]:
        ax.set_xlabel('Training timer incl. JIT/evaluation (min)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Electropolish: cold start, 5000 updates, Adam, B8; spatial sampling only\n'
        'Curves: mean over 4 reused native validation inputs; final 64-input errors reported separately')
    fig.savefig(output/'convergence.png', dpi=190)
    fig.savefig(output/'convergence.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    keys = ('pooled_phi', 'terminal_phi', 'pooled_c', 'terminal_c')
    x = np.arange(4)
    for ax, group, title in zip(axes, ('native64', 'local101_64'),
            ('Native nonlocal reference', 'Common local FD101 reference'), strict=True):
        for j, (r, label, color) in enumerate(zip(records, LABELS, COLORS, strict=True)):
            values = [100*r['errors'][group][k]['mean'] for k in keys]
            bars = ax.bar(x+(j-1)*.25, values, width=.24, color=color, label=label)
            ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
        ax.set_xticks(x, ['phi pooled', 'phi final', 'c pooled', 'c final'])
        ax.set_ylabel('Mean relative L2 (%)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.08),
        ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Fixed 5000-update final models: same 64 development inputs')
    fig.savefig(output/'final_errors.png', dpi=190, bbox_inches='tight')
    fig.savefig(output/'final_errors.pdf', bbox_inches='tight')
    plt.close(fig)
    result = dict(status='gpu_validated_with_recorded_CPU_replay_issue', controls=records,
        final64metrics_independently_replayed=True,
        CPU_replay_status='failed; threshold retained; see separately recorded composed-attention issue',
        CPU_replay_issue='docs/issues/electropolish-attention-replay.md',
        convergence=str(output/'convergence.png'), final_errors=str(output/'final_errors.png'),
        scope='One matched initialization, same64reused native/localFD101development inputs; four refined FD201/401targets supplemental. Fixed/rolling have identical safe padding; no cross-seed superiority/default promotion.',
        timing_scope='All three5000-update processes on the same GPU; separate cold wall, initial compile, synchronized warm GPU/SDF/spatial costs. End-to-end excludes final64evaluation, common to all.')
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (REPO/'docs/tasks/electro-spatial-results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for r in records:
        print(r['mode'], 'wall_min', r['process_wall_seconds']/60, 'step_ms', r['warm_effective_step_ms'],
            'peak_GiB', r['GPU_peak_active_GiB'],
            'local_phi', r['errors']['local101_64']['pooled_phi']['mean'],
            'local_phi_final', r['errors']['local101_64']['terminal_phi']['mean'], flush=True)


if __name__ == '__main__':
    main()
