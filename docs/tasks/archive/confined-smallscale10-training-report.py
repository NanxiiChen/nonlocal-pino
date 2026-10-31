"""Plot fixed-budget phase convergence and retain raw tail diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(run, output):
    output.mkdir(parents=True, exist_ok=False)
    history = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    loss = [json.loads(s) for s in (run/'loss.jsonl').read_text().splitlines()]
    summary = json.loads((run/'summary.json').read_text())
    evaluation = json.loads((run/'evaluation/metrics.json').read_text())
    config = json.loads((run/'config.json').read_text())
    assert history[-1]['step'] == config['train']['iterations'] == 20000
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
    fig.suptitle('Fixed small-scale 10-mode IC; 512 training geometries; '+optimizer+' cold start')
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'convergence.{suffix}', dpi=180)
    plt.close(fig)
    tail = [r for r in history if r['step'] >= 15000]
    trends = {}
    for key in ('train_phi_pooled', 'test_phi_pooled', 'test_phi_terminal', 'test_mu_pooled'):
        x = np.asarray([r['step'] for r in tail])
        y = np.asarray([r[key] for r in tail])
        trends[key] = dict(start=int(x[0]), end=int(x[-1]), final=float(y[-1]),
            minimum=float(y.min()), maximum=float(y.max()),
            linear_slope_percentage_points_per1000steps=float(np.polyfit((x-x[0])/1000, 100*y, 1)[0]))
    selected = ('step', 'train_phi_pooled', 'test_phi_pooled', 'test_phi_terminal', 'test_mu_pooled')
    result = dict(run=str(run), native_test=evaluation['test'], local_FEM=evaluation['local_FEM'],
        milestones=[{k:r[k] for k in selected} for r in history if r['step'] in (0, 5000, 10000, 15000, 20000)],
        tail_trends=trends, cold_seconds=summary['wall_seconds'],
        warm_update_seconds=summary['mean_update_seconds_after_100'],
        selection='Fixed 20000-step final, not minimum held-out error',
        scope='One cold seed; eight reused development geometries. Changed IC/time is a new physical benchmark, not an isolated method gain.')
    (output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report(a.run, a.output)
