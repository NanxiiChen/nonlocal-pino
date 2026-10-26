"""Focused two-seed convergence plots from saved metrics, no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

OUT = Path(__file__).with_suffix('')
ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
RUNS = {
    (seed, strategy): ROOT / (f'20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed{seed}'
                             if strategy == 'Random IC/BC' else
                             f'20261025-data-s_bc-icmatched_graph-bcmatched_graph-phase-stagger0-seed{seed}')
    for seed in (42, 7) for strategy in ('Random IC/BC', 'Grid IC/BC + normalized concentration BC')
}
COLORS = {'Random IC/BC': '#d87818', 'Grid IC/BC + normalized concentration BC': '#1673b1'}


def main():
    records = {}
    for (seed, strategy), run in RUNS.items():
        rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        steps = np.array([v['step'] for v in rows])
        np.testing.assert_array_equal(steps, np.arange(0, 5001, 200))
        summary = json.loads((run/'summary.json').read_text())
        assert summary['updates'] == 5000 and not summary['hard_initial_condition']
        keys = [prefix+'_rel_l2_'+name for prefix in ('terminal', 'pooled') for name in ('phi', 'c')]
        values = {key: [v[key] for v in rows] for key in keys}
        for key, series in values.items():
            assert np.all(np.isfinite(series)) and np.all(np.array(series) > 0), (run,key)
            np.testing.assert_allclose(series[-1], summary['final_metrics'][key], rtol=1e-12)
        records[f'{seed}/{strategy}'] = {'seed': seed, 'strategy': strategy,
            'run': str(run), 'metrics_path': str(run/'metrics.jsonl'), 'steps': steps.tolist(),
            'relative_l2': values}

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'legend.fontsize': 11})
    for scope, description in (('terminal','Physical time t = 5 s; all 5151 spatial nodes'),
                               ('pooled','All 21 saved physical times, 0-5 s; all spatial nodes')):
        fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
        handles = []
        for i,seed in enumerate((42,7)):
            for j,name in enumerate(('phi','c')):
                ax = axes[i,j]
                for strategy in COLORS:
                    record = records[f'{seed}/{strategy}']
                    y = record['relative_l2'][scope+'_rel_l2_'+name]
                    line, = ax.plot(record['steps'], y, color=COLORS[strategy], label=strategy,
                                    linewidth=2, marker='o', markersize=3)
                    if i == 0 and j == 0:
                        handles.append(line)
                    other_endpoints = [records[f'{seed}/{other}']['relative_l2'][scope+'_rel_l2_'+name][-1]
                                       for other in COLORS if other != strategy]
                    label_dy = 4 if y[-1] > max(other_endpoints) else -4
                    ax.annotate(f'{y[-1]:.3e}', xy=(5000,y[-1]), xytext=(7,label_dy),
                                textcoords='offset points', va='center', color=COLORS[strategy],
                                fontsize=10, annotation_clip=False)
                ax.set_title(f'Seed {seed} | '+('Phase field phi' if name == 'phi' else 'Total concentration c'))
                ax.set_yscale('log')
                ax.set_ylim(8e-4,1.)
                ax.set_xlim(0,5700)
                ax.set_xticks(np.arange(0,5001,1000))
                ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2,5)))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.grid(which='major', alpha=.25)
                ax.grid(which='minor', axis='y', alpha=.1)
                if i == 1:
                    ax.set_xlabel('Completed training updates')
                if j == 0:
                    ax.set_ylabel('Relative L2 error')
        fig.suptitle('Electropolish training convergence\n'+description+'\nRaw logged evaluations every 200 updates; no smoothing', fontsize=15)
        fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
        path = str(OUT)+'-'+scope
        fig.savefig(path+'.png',dpi=180,bbox_inches='tight',pad_inches=.15)
        fig.savefig(path+'.svg',bbox_inches='tight',pad_inches=.15)
        plt.close(fig)
    output = {'case':'electropolish2d','metric_scope':{
        'terminal':'Euclidean relative L2 over all5151 spatial nodes at5 physical seconds',
        'pooled':'Euclidean relative L2 over all5151 nodes and all21 saved physical times0..5s'},
        'plot':{'sampling':'raw logged evaluations every200 training updates; no smoothing',
                'x':'completed optimizer updates','y':'relative L2; logarithmic; common scale across panels'},
        'runs':records}
    OUT.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Verified four 26-point histories and terminal/pooled endpoints; wrote PNG/SVG curves and raw JSON.')


if __name__ == '__main__':
    main()
