"""Convergence,64test distributions and early side-wall physical fields."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
RUNS = [ROOT/f'20261029-electro-operator-match-{kind}-01' for kind in ('nonlocal', 'local')]
REFS = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
OUT = Path(__file__).resolve().parent
LABELS = ['Nonlocal (5 losses)', 'Local AD (6 losses)']
COLORS = ['#1764ab', '#df6b21']


def export(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_fields():
    with np.load(REFS/'condition_000/local401.npz') as stored:
        reference = stored['fields'][:, ::4, ::4, 0]
        times = stored['time']
    predictions = []
    for run in RUNS:
        with np.load(run/'evaluation64.npz') as stored:
            predictions.append(stored['prediction'][0, ..., 0])
    indices = [int(np.flatnonzero(np.isclose(times, t))[0]) for t in (0., .001, .05, .5, 1.)]
    error_max = max(float(np.max(abs(p[indices]-reference[indices]))) for p in predictions)
    fig, axes = plt.subplots(5, 5, figsize=(14, 9), constrained_layout=True)
    rows = [reference, predictions[0], abs(predictions[0]-reference), predictions[1], abs(predictions[1]-reference)]
    labels = ['Local FD401 reference', 'Nonlocal prediction', 'Nonlocal absolute error', 'Local AD prediction', 'Local AD absolute error']
    for row, (values, label) in enumerate(zip(rows, labels)):
        for col, k in enumerate(indices):
            error = row in (2, 4)
            im = axes[row, col].imshow(values[k], origin='lower', extent=(0, 1, 0, .5), aspect='equal',
                cmap='magma' if error else 'viridis', vmin=0., vmax=error_max if error else 1.)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f't*={times[k]:g} / {5*times[k]:g}s')
            if col == 0:
                axes[row, col].set_ylabel(label)
        fig.colorbar(im, ax=axes[row, :], fraction=.015, pad=.01)
    fig.suptitle('Predeclared test interface 0; original IC, fixed final checkpoints')
    export(fig, 'electro-operator-local-fields')


parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=('base', 'fields', 'all'))
stage = parser.parse_args().stage
if stage == 'fields':
    plot_fields()
    print('Physical phase fields exported', flush=True)
    raise SystemExit(0)
evaluations = [json.loads((run/'evaluation64.json').read_text()) for run in RUNS]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for run, label, color in zip(RUNS, LABELS, COLORS):
    history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    for ax, key, title in zip(axes, ('test_pooled_rel_l2_phi', 'test_terminal_rel_l2_phi'),
                              ('Pooled phase relative L2', 'Terminal phase relative L2')):
        ax.semilogy([h['step'] for h in history], [100*h[key] for h in history], label=label, color=color)
        ax.set(xlabel='Optimizer updates', ylabel='Relative L2 (%)', title=title)
        ax.grid(alpha=.25)
axes[0].legend()
fig.suptitle('Four fixed monitor interfaces, common graph target; no checkpoint selection')
export(fig, 'electro-operator-local-convergence')

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, group, title in zip(axes, ('graph64', 'local101_64'), ('Common graph target', 'Common local FD101 target')):
    for data, label, color in zip(evaluations, LABELS, COLORS):
        error = np.sort([r[group]['pooled_phi'] for r in data['records']])
        ax.step(100*error, np.arange(1, len(error)+1)/len(error), where='post', color=color, label=label)
    ax.set(xlabel='Pooled phase relative L2 (%)', ylabel='Fraction of 64 interfaces', title=title, ylim=(0, 1.02))
    ax.grid(alpha=.25)
axes[0].legend()
fig.suptitle('Final 5000-update checkpoints; identical 64 test interfaces')
export(fig, 'electro-operator-local-test64')

if stage == 'base':
    print('Convergence and64test distributions exported', flush=True)
    raise SystemExit(0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for data, label, color in zip(evaluations, LABELS, COLORS):
    ts = np.asarray(data['times']); early = (ts > 0) & (ts <= .05)
    for ax, group, title in zip(axes[:2], ('local101_64', 'local401_4'),
                                ('64 interfaces / FD101 target', '4 declared interfaces / FD401 target')):
        curve = np.asarray(data['groups'][group]['mean_curves']['side_phi_by_time'])
        ax.semilogx(5*ts[early], 100*curve[early], 'o-', color=color, label=label)
        ax.set(xlabel='Physical time (s)', ylabel='Side-strip phase relative L2 (%)', title=title)
        ax.grid(alpha=.25)
data = evaluations[0]
for name, label in (('local201_to_401', 'FD201 -> FD401'), ('native_to_local401', 'Native graph -> FD401')):
    curve = np.mean([r[name]['side_phi_by_time'] for r in data['reference_spatial_errors']], axis=0)
    axes[2].semilogx(5*ts[early], 100*curve[early], 'o-', label=label)
axes[2].set(xlabel='Physical time (s)', ylabel='Side-strip phase relative difference (%)', title='Independent reference differences (4 cases)')
axes[2].grid(alpha=.25); axes[2].legend(); axes[0].legend()
fig.suptitle('Original incompatible IC; side strip width 0.02, Dirichlet corners excluded')
export(fig, 'electro-operator-local-boundary')

plot_fields()
print('Four scientific figures exported', flush=True)
