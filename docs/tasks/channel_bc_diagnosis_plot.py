"""Plot the saved counterfactual CH contours and sampled-loss convergence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docs.tasks.channel_droplet_metrics import contour_samples


def main(run, root):
    run, root = Path(run), Path(root)
    output = root/'figures'; output.mkdir(exist_ok=False)
    paths = [run/'prediction.npz', root/'frozen-20k-v2/CH-target_zero_neumann.npz',
             root/'frozen-20k-v2/CH-learned_continuous_neumann.npz',
             root/'reference-velocity-linear/solution.npz', root/'reference-velocity-pchip/solution.npz']
    payload = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            payload.append({k: z[k] for k in z.files})
    points, times = payload[0]['points'], payload[0]['time']
    for item in payload:
        np.testing.assert_array_equal(item['points'], points)
        np.testing.assert_allclose(item['time'], times, atol=1e-12, rtol=0)
    nx = len(np.unique(points[:, 0])); ny = len(points)//nx
    reference = payload[0]['reference'][..., 0]
    series = [('FEM', reference, 'black', '-'), ('PINO 20k', payload[0]['fields'][..., 0], '#D55E00', '-'),
              ('CH: PINO velocity, exact BC', payload[1]['fields'][..., 0], '#CC79A7', '--'),
              ('CH: PINO velocity, learned BC', payload[2]['fields'][..., 0], '#0072B2', ':'),
              ('CH: reference velocity (linear)', payload[3]['fields'][..., 0], '#009E73', '--'),
              ('CH: reference velocity (PCHIP)', payload[4]['fields'][..., 0], '#009E73', '-')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout='constrained')
    for ax, time in zip(axes, [3., 4.5, 6.], strict=True):
        ti = np.flatnonzero(np.isclose(times, time, atol=1e-12, rtol=0))
        if len(ti) != 1:
            raise ValueError('Expected exact stored figure times')
        i = int(ti[0])
        for name, field, color, style in series:
            _, _, curves = contour_samples(points, field[i], (nx, ny), 512)
            for j, curve in enumerate(curves):
                ax.plot(curve[:, 0], curve[:, 1], color=color, ls=style, lw=1.7,
                        label=name if j == 0 else None)
        center = np.mean(points[reference[i] < 0], axis=0)
        ax.set(xlim=(center[0] - .36, center[0] + .36), ylim=(center[1] - .36, center[1] + .36),
               aspect='equal', xlabel='x', title=f't={time:g}')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('y')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('Frozen-velocity CH controls: same IC/96×40 solver; reference velocity resolves the shape gap')
    fig.savefig(output/'controlled-contours.png', dpi=180); fig.savefig(output/'controlled-contours.pdf'); plt.close(fig)
    history = [json.loads(s) for s in (run/'loss.jsonl').read_text().splitlines()]
    pde = np.array([sum(r['parts'][k] for k in ('phase', 'chemical', 'momentum_x', 'momentum_y')) for r in history])
    bc = np.array([r['total'] for r in history]) - pde
    if np.any(bc <= 0):
        raise ValueError('Positive boundary loss required for logarithmic history plot')
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    step = [r['step'] for r in history]
    ax.semilogy(step, pde, label='PDE loss (actual stochastic batches)', alpha=.8)
    ax.semilogy(step, bc, label='BC loss at fixed training sites', alpha=.8)
    frozen = json.loads((root/'frozen-20k-v2/nominal-dense.json').read_text())
    ax.scatter([20000], [sum(frozen['independent_dense_boundary']['scaled_mse'])], marker='D', s=65,
               label='Independent dense BC audit (nominal20k)', color='#D55E00', zorder=5)
    ax.set(xlabel='Training updates', ylabel='Sum of scaled MSE terms')
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/'pde-bc-convergence.png', dpi=180); fig.savefig(output/'pde-bc-convergence.pdf'); plt.close(fig)
    print(str(output), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--root', required=True)
    a = p.parse_args(); main(a.run, a.root)
