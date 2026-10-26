"""Raw convergence curves and all three final seed2026 phi fields."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).with_suffix('')
LOGS = Path('/root/autodl-tmp/tf-logs/corrosion2d')
RUNS = {
    **{('bounded', seed): LOGS/f'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed{seed}' for seed in (42, 7)},
    ('bounded', 2026): LOGS/'20261025-independent-seed2026-default-strategy',
    **{('phase', seed): LOGS/f'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed{seed}' for seed in (42, 7)},
    ('phase', 2026): LOGS/'20261026-objective-phase-seed2026',
    **{('frozen', seed): LOGS/f'20261026-objective-bounded_frozen-seed{seed}' for seed in (42, 7, 2026)}
}


def main():
    colors = {'phase': '#1478ab', 'bounded': '#d65b36', 'frozen': '#17834b'}
    styles = {42: ':', 7: '--', 2026: '-'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for (form, seed), run in RUNS.items():
        rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        for ax, name in zip(axes.flat, ('phi', 'c', 'q', 'Dq_free')):
            ax.semilogy([v['step'] for v in rows], [v['terminal_rel_l2_'+name] for v in rows],
                        color=colors[form], ls=styles[seed], lw=1.4)
    for ax, field in zip(axes.flat, ('Phase phi, all nodes', 'Concentration c, all nodes',
                                   'Driving q, all nodes', 'Dq, free nodes')):
        ax.set(xlabel='Training updates', ylabel='Terminal relative L2', title=field, xlim=(0, 5000))
        ax.axvspan(4000, 5000, color='gray', alpha=.07)
        ax.grid(which='both', alpha=.2)
    handles = [Line2D([0], [0], color=color, label=form) for form, color in colors.items()]
    handles += [Line2D([0], [0], color='black', ls=style, label=f'seed {seed}') for seed, style in styles.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=6)
    fig.savefig(str(OUT)+'-convergence.png', dpi=180)
    plt.close(fig)

    config = json.loads((RUNS[('phase', 2026)]/'config.json').read_text())
    ref = np.load(config['experiment']['reference_path'])
    panels = [('Reference', ref['phi'])]
    for form in ('phase', 'bounded', 'frozen'):
        with np.load(RUNS[(form, 2026)]/'prediction.npz') as data:
            panels.append((form+' / seed2026', data['phi'].copy()))
    indices = (0, 5, 10, 15, 20)
    fig, axes = plt.subplots(4, 5, figsize=(14, 8), layout='constrained')
    for i, (name, fields) in enumerate(panels):
        for j, index in enumerate(indices):
            ax = axes[i, j]
            m = ax.pcolormesh(ref['X'], ref['Y'], fields[index], vmin=0, vmax=1,
                              cmap='viridis', shading='auto', rasterized=True)
            ax.contour(ref['X'], ref['Y'], ref['phi'][index], levels=[.5], colors='white', linewidths=.6)
            ax.set_aspect('equal')
            if i == 0: ax.set_title(f't* = {ref["time"][index]:.2f}')
            if j == 0: ax.set_ylabel(name)
            if i == 3: ax.set_xlabel('x*')
    fig.colorbar(m, ax=axes, label='Phase phi', shrink=.75)
    fig.savefig(str(OUT)+'-fields.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
