"""Plot fixed-budget phase convergence and retain raw tail diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.confined_ch2d.operator_history import training_records


def report(run, output):
    output.mkdir(parents=True, exist_ok=False)
    history = training_records(run,'metrics.jsonl')
    loss = training_records(run,'loss.jsonl')
    summary = json.loads((run/'summary.json').read_text())
    evaluation = json.loads((run/'evaluation/metrics.json').read_text())
    config = json.loads((run/'config.json').read_text())
    assert history[-1]['step'] == config['train']['iterations']
    assert config['physical']['terminal_time'] == 3.
    assert len(config['initial']['waves']) == 10
    optimizer = config['train']['optimizer'].upper()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for ax, key, title in zip(axes[:2], ('phi_pooled', 'phi_terminal'),
                             ('All five saved times', 'Terminal time t = 3'), strict=True):
        for split, label, style in (('train', '16 training monitors', '--'),
                                    ('test', '8 validation geometries', '-')):
            ax.semilogy([r['step'] for r in history],
                        [100*r[split+'_'+key] for r in history],
                        style, marker='o', markersize=2.5, label=label)
        ax.set(xlabel=optimizer+' updates', ylabel='Phase relative L2 (%)', title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].semilogy([r['step'] for r in loss], [r['loss'] for r in loss], linewidth=.8)
    axes[2].set(xlabel=optimizer+' updates', ylabel='Sampled preconditioned PDE loss',
                title='Physics only; no evolved labels')
    axes[2].grid(alpha=.2)
    if 'continuation' in config['train'] and 'learning_rate_factor' in config['train']['continuation']:
        factor=config['train']['continuation']['learning_rate_factor']
        for ax in axes:ax.axvline(summary['start_step'],color='grey',linestyle=':',linewidth=.8)
        axes[0].annotate(f'LR x {factor:g}',xy=(summary['start_step'],.97),
            xycoords=('data','axes fraction'),xytext=(4,-4),textcoords='offset points',
            va='top',fontsize=8)
    provenance='state-preserving continuation' if 'continuation' in config['train'] else 'cold start'
    fig.suptitle(f"Fixed small-scale 10-mode IC; {config['operator']['train_conditions']} training geometries; "+optimizer+' '+provenance)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'convergence.{suffix}', dpi=180)
    plt.close(fig)
    tail = [r for r in history if r['step'] >= max(0,config['train']['iterations']-5000)]
    trends = {}
    for key in ('train_phi_pooled', 'test_phi_pooled', 'test_phi_terminal', 'test_mu_pooled'):
        x = np.asarray([r['step'] for r in tail])
        y = np.asarray([r[key] for r in tail])
        trends[key] = dict(start=int(x[0]), end=int(x[-1]), final=float(y[-1]),
            minimum=float(y.min()), maximum=float(y.max()),
            linear_slope_percentage_points_per1000steps=float(np.polyfit((x-x[0])/1000, 100*y, 1)[0]))
    selected = ('step', 'train_phi_pooled', 'test_phi_pooled', 'test_phi_terminal', 'test_mu_pooled')
    result = dict(run=str(run), native_test=evaluation['test'], local_FEM=evaluation['local_FEM'],
        milestones=[{k:r[k] for k in selected} for r in history if r['step'] in (0, 5000, 10000, 15000, 20000) or r['step']==config['train']['iterations']],
        tail_trends=trends, stage_seconds=summary['wall_seconds'],
        cumulative_seconds=summary['cumulative_wall_seconds'] if 'cumulative_wall_seconds' in summary else summary['wall_seconds'],
        warm_update_seconds=summary['mean_update_seconds_after_100'],
        selection=f"Fixed total{config['train']['iterations']}-step final, not minimum held-out error",
        scope='One cold seed; eight reused development geometries. Same physical benchmark; bundled representation/optimization development, not a fresh final test.')
    (output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report(a.run, a.output)
