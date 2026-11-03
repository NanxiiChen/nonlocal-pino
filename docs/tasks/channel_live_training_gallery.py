"""CPU-only raw convergence and explicitly saved checkpoint field gallery."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.evaluate import plot_fields
from examples.channel_nsch2d.problem import walls
from docs.tasks.channel_droplet_metrics import contour_samples


def history(path):
    rows = [json.loads(s) for s in path.read_bytes().splitlines(keepends=True)
            if s.endswith(b'\n')]
    if not rows or any(b['step'] <= a['step'] for a, b in zip(rows, rows[1:])):
        raise ValueError(f'Nonempty ordered complete log records required: {path}')
    return rows


def main(run, control, step, output):
    run, control, output = map(Path, (run, control, output))
    if not all(p.is_absolute() for p in (run, control, output)) or step < 1:
        raise ValueError('Absolute paths and a positive saved checkpoint step required')
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((run/'config.json').read_text())
    old_config = json.loads((control/'config.json').read_text())
    if json.loads((control/'summary.json').read_text())['status'] != 'completed':
        raise ValueError('The historical comparison must be a completed run')
    for key in ('geometry', 'physical', 'initial', 'reference'):
        if config[key] != old_config[key]:
            raise ValueError(f'Trajectories use different {key}')
    replay = json.loads((run/f'partial_{step:06d}.json').read_text())
    if replay['status'] != 'partial_checkpoint_metric_replay_passed' or replay['step'] != step:
        raise ValueError('A replay-qualified explicitly saved model is required')
    path = Path(replay['prediction'])
    if path != run/f'partial_{step:06d}.npz' or not path.is_file():
        raise ValueError('The actual checkpoint field path differs')
    current, old = history(run/'metrics.jsonl'), history(control/'metrics.jsonl')
    losses = history(run/'loss.jsonl')
    for rows, c in ((current, config), (old, old_config)):
        np.testing.assert_array_equal([r['step'] for r in rows],
            np.arange(0, rows[-1]['step']+1, c['log']['eval_every']))
    if old[-1]['step'] != old_config['train']['iterations']:
        raise ValueError('The historical endpoint is missing')
    fields = ('phi', 'terminal_phi', 'velocity', 'pressure', 'dispersed_phase', 'chemical')
    titles = ('Phase field', 'Terminal phase field', 'Velocity', 'Pressure',
              'Droplet fraction', 'Chemical potential')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.4), sharex=True, layout='constrained')
    for ax, name, title in zip(axes.flat, fields, titles, strict=True):
        for rows, color, style, label in ((old, '#8b9097', '--', 'Previous 10k, Nt=40'),
                                         (current, '#0072B2', '-', 'Current 20k, Nt=36')):
            values = 100*np.array([r[name] for r in rows])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError(f'Invalid relative L2: {name}')
            ax.semilogy([r['step'] for r in rows], values, style, color=color,
                        lw=1.5, marker='o', ms=3.1, label=label)
        ax.axvline(step, color='#c28e36', ls=':', lw=1.2, label='Field checkpoint')
        ax.set(title=f'{title} | latest {100*current[-1][name]:.2f}%',
               xlim=(0, max(current[-1]['step'], old[-1]['step'])), ylabel='Relative L2 (%)')
        ax.grid(alpha=.18, which='both')
    for ax in axes[-1]:
        ax.set_xlabel('Optimizer updates')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle(f'Cold-start NS-CH PINO: raw convergence through {current[-1]["step"]:,} updates\n'
                 f'One reused local-FEM development geometry | field panels use saved {step:,} weights', fontsize=13)
    fig.savefig(output/'relative-l2.png', dpi=180)
    fig.savefig(output/'relative-l2.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.3), layout='constrained')
    groups = dict(total=lambda r: r['total'], phase=lambda r:r['parts']['phase'],
                  chemical=lambda r:r['parts']['chemical'],
                  momentum=lambda r:r['parts']['momentum_x']+r['parts']['momentum_y'],
                  boundary=lambda r:sum(r['parts'][k] for k in
                      ('wall_phase', 'wall_mu', 'outlet_traction', 'outlet_phase', 'outlet_mu')))
    for row in losses:
        np.testing.assert_allclose(row['total'], sum(row['parts'].values()), rtol=3e-6, atol=1e-7)
    for label, fn in groups.items():
        ax.semilogy([r['step'] for r in losses], [fn(r) for r in losses], label=label, lw=1.2)
    ax.set(xlabel='Optimizer updates', ylabel='Normalized sampled MSE',
           title='Current run: raw losses on fresh geometry/time batches')
    ax.grid(alpha=.18, which='both'); ax.legend(ncol=5, fontsize=9)
    fig.savefig(output/'sampled-losses.png', dpi=180); plt.close(fig)
    phase_times = (0., 1.5, 3., 4.5, 6.)
    flow_times = (.5, 1.5, 3., 4.5, 6.)
    plot_fields(path, output, selected_fields=('phi',), selected_times=phase_times)
    plot_fields(path, output, selected_fields=('speed', 'pressure'), selected_times=flow_times)
    with np.load(path, allow_pickle=False) as f:
        d = {k:f[k] for k in f.files}
    nx = len(np.unique(d['points'][:, 0])); ny = len(d['points'])//nx
    if nx*ny != len(d['points']) or d['fields'].shape != d['reference'].shape:
        raise ValueError('Aligned mapped query grid and physical fields required')
    c = ConfigDict(config)
    wall_x = np.linspace(0, c.geometry.length, 1601)
    lower, upper = walls(wall_x, d['amplitudes'], c, xp=np)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), layout='constrained')
    contour_counts = []
    for ax, time in zip(axes, phase_times, strict=True):
        ids = np.flatnonzero(np.isclose(d['time'], time, atol=1e-12, rtol=0))
        if len(ids) != 1:
            raise ValueError('Require exact saved times; no temporal interpolation')
        i = int(ids[0])
        cloud, measure, _ = contour_samples(d['points'], d['reference'][i, :, 0], (nx, ny), 1024)
        if not len(cloud):
            raise ValueError('Reference droplet contour absent')
        center = np.sum(cloud*measure[:, None], axis=0)/measure.sum()
        counts = []
        for series, color, style in ((d['reference'], '#222222', '-'),
                                     (d['fields'], '#D55E00', '--')):
            curves = contour_samples(d['points'], series[i, :, 0], (nx, ny), 1024)[2]
            counts.append(len(curves))
            for curve in curves:
                ax.plot(curve[:, 0], curve[:, 1], color=color, ls=style, lw=1.8)
        contour_counts.append(counts)
        ax.fill_between(wall_x, 0, lower, color='.88')
        ax.fill_between(wall_x, upper, 1, color='.88')
        ax.plot(wall_x, lower, color='.55', lw=.8); ax.plot(wall_x, upper, color='.55', lw=.8)
        ax.set(xlim=(center[0]-.40, center[0]+.40), ylim=(center[1]-.40, center[1]+.40),
               aspect='equal', title=f't = {time:g}', xlabel='x')
        ax.grid(alpha=.18); ax.tick_params(labelsize=8)
    axes[0].set_ylabel('y')
    fig.legend(handles=[Line2D([], [], color='#222222', label='Local FEM'),
                        Line2D([], [], color='#D55E00', ls='--', label=f'PINO checkpoint {step:,}')],
               loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Droplet zero contours: same physical zoom; positions are not aligned or corrected')
    fig.savefig(output/'droplet-contours.png', dpi=200)
    fig.savefig(output/'droplet-contours.pdf'); plt.close(fig)
    record = dict(status='live_curves_and_actual_saved_checkpoint_plotted', run=str(run),
        control=str(control), checkpoint=replay['checkpoint'], checkpoint_step=step,
        latest_curve_step=current[-1]['step'], latest_curve_metrics=current[-1],
        checkpoint_metrics=replay['metrics'], replay_metric_gaps=replay['logged_metric_absolute_gap'],
        query_shape=[nx, ny], phase_times=phase_times, flow_times=flow_times,
        contour_counts=contour_counts, smoothing=False, time_interpolation=False,
        display_interpolation='Mapped bilinear fields only; contours and metrics use native saved queries',
        scope=f'One reused development FEM geometry; saved{step} fields differ from later scalar monitor, no cohort claim')
    (output/'gallery-validation.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    (output/'curve-data.json').write_text(json.dumps(dict(current=current, old=old, losses=losses), indent=2)+'\n')
    shutil.copyfile(__file__, output/'plot-source.py')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'control', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--step', type=int, required=True)
    args = p.parse_args()
    main(args.run, args.control, args.step, args.output)
