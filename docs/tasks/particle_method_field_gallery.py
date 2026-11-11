"""Plot saved local-FEM, nonlocal and local-AD phase fields on common meshes."""
import argparse
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from examples.particle_corrosion2d.compare import saved_predictions
from examples.particle_corrosion2d.evaluate import load_references


def read(path):
    return json.loads(Path(path).read_text())


def render(root, seed, samples, output):
    journal = read(root/'journal.json')
    pair = [stage for stage in journal['stages']
            if stage['label'] == f'particle_corrosion2d-s{seed}-pair']
    if len(pair) != 1 or pair[0]['status'] != 'passed':
        raise ValueError('This gallery requires a completed strict particle pair')
    records = [r for r in journal['runs'] if r['case'] == 'particle_corrosion2d' and r['seed'] == seed]
    if len(records) != 2 or {r['method'] for r in records} != {'nonlocal', 'local_ad'}:
        raise ValueError('Need exactly one completed run of each method')
    runs = {r['method']: Path(r['run']) for r in records}
    configs = {name: ConfigDict(read(run/'config.json')) for name, run in runs.items()}
    references = {name: Path(read(run/'runtime.json')['reference']) for name, run in runs.items()}
    if references['nonlocal'].resolve() != references['local_ad'].resolve():
        raise ValueError('Methods must use the same reference cohort')
    c = configs['nonlocal']
    for config in configs.values():
        _, manifest = load_references(references['nonlocal'], config)
        if manifest['backend'] != 'local_p1_lumped_fem':
            raise ValueError('Expected the declared independent local-FEM reference')
    refs, manifest = load_references(references['nonlocal'], c)
    if not samples or len(set(samples)) != len(samples) or any(i < 0 or i >= len(refs) for i in samples):
        raise ValueError('Sample indices must be distinct and in the reference cohort')
    predictions, summaries = {}, {}
    for name, run in runs.items():
        if configs[name].train.iterations != 20000:
            raise ValueError('Expected the completed 20k endpoint')
        predictions[name], summaries[name] = saved_predictions(run, refs)
        saved = read(run/'summary.json')['final']
        for key, value in summaries[name].items():
            if key != 'conditions':
                np.testing.assert_allclose(value, saved[key], rtol=1e-10, atol=1e-12)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.tri import Triangulation

    output.mkdir(parents=True, exist_ok=False)
    target_times = np.array([0., .2, .5, .8, 1.])
    rendered = []
    for sample in samples:
        ref = refs[sample]
        indices = []
        for time in target_times:
            matches = np.flatnonzero(np.isclose(ref['times'], time, rtol=0, atol=1e-12))
            if len(matches) != 1:
                raise ValueError(f'No unique saved time {time} for case {sample}')
            indices.append(int(matches[0]))
        xy = ref['points']
        tri = Triangulation(xy[:, 0], xy[:, 1], ref['triangles'])
        fields = [ref['fields'], predictions['nonlocal'][sample], predictions['local_ad'][sample]]
        labels = ['Reference\n(local FEM)', 'Nonlocal PINO', 'Local-AD PINO']
        fig, axes = plt.subplots(3, 5, figsize=(14, 6.5))
        fig.subplots_adjust(left=.10, right=.985, top=.80, bottom=.17, wspace=.10, hspace=.20)
        metrics = {}
        for row, field in enumerate(fields):
            if row:
                method = ('nonlocal', 'local_ad')[row-1]
                m = summaries[method]['conditions'][sample]
                metrics[method] = dict(m)
                labels[row] += f'\nPooled L2: {100*m["phi_pooled"]:.2f}%'
                per_time = []
            for col, index in enumerate(indices):
                ax = axes[row, col]
                phi = field[index, :, 0]
                image = ax.tripcolor(tri, phi, shading='gouraud', cmap='viridis', vmin=0, vmax=1,
                                     rasterized=True)
                if phi.min() < .5 < phi.max():
                    ax.tricontour(tri, phi, levels=[.5], colors='#17242b', linewidths=1.0)
                if row:
                    reference_phi = ref['fields'][index, :, 0]
                    if reference_phi.min() < .5 < reference_phi.max():
                        ax.tricontour(tri, reference_phi, levels=[.5], colors='white',
                                      linewidths=1.1, linestyles='--')
                    error = float(np.sqrt(np.sum(ref['volumes']*(phi-reference_phi)**2)/
                                          np.sum(ref['volumes']*reference_phi**2)))
                    per_time.append(error)
                    ax.text(.97, .94, f'L2 {100*error:.2f}%', transform=ax.transAxes,
                            ha='right', va='top', fontsize=8, color='#17242b',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2))
                ax.set_facecolor('#a8b2b8')
                ax.set(xlim=(0, c.geometry.length), ylim=(c.geometry.height, 0), aspect='equal')
                ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .3, .6])
                ax.tick_params(labelsize=8, length=2, pad=2)
                if col == 0:
                    ax.set_ylabel(labels[row], fontsize=10, labelpad=7)
                else:
                    ax.set_yticklabels([])
                if row == 2:
                    ax.set_xlabel('$x^*$', fontsize=9, labelpad=1)
                else:
                    ax.set_xticklabels([])
                if row == 0:
                    ax.set_title(f'$t^*={ref["times"][index]:g}$\n{ref["physical_time"][index]:g} s', fontsize=10)
            if row:
                metrics[method]['shown_time_phi_errors'] = per_time
                metrics[method]['saved_phi_range'] = [float(field[..., 0].min()), float(field[..., 0].max())]
        fig.suptitle(f'Ceramic-particle corrosion | Geometry {sample+1}', fontsize=15, y=.98)
        fig.text(.54, .927, 'Same physical problem and query points | paired seed '+str(seed)+
                 ' | final 20,000-step models', ha='center', fontsize=10, color='#44515c')
        fig.legend(handles=[Patch(facecolor='#a8b2b8', label='Ceramic (outside corrosion domain)'),
                   Line2D([0], [0], color='#17242b', lw=1.2, label='Shown field: $\\phi=0.5$'),
                   Line2D([0], [0], color='#6a7580', lw=1.3, ls='--', label='White dashed in plots: reference interface')],
                   loc='upper center', bbox_to_anchor=(.54, .902), ncol=3, frameon=False, fontsize=9)
        bar = fig.add_axes([.32, .075, .47, .018])
        fig.colorbar(image, cax=bar, orientation='horizontal', ticks=[0, .5, 1])
        bar.set_xlabel('Phase $\\phi$: 0 = electrolyte / corroded region, 1 = metal', fontsize=10)
        bar.tick_params(labelsize=9)
        path = output/f'geometry_{sample:03d}_phi.png'
        fig.savefig(path, dpi=170, facecolor='white', pil_kwargs={'optimize': True})
        plt.close(fig)
        rendered.append(dict(sample_index=sample, reference=str(references['nonlocal']/manifest['cases'][sample]['file']),
            prediction_files={name: str(run/f'prediction_{sample:03d}.npz') for name, run in runs.items()},
            figure=str(path), time_indices=indices, nondimensional_times=ref['times'][indices].tolist(),
            physical_seconds=ref['physical_time'][indices].tolist(), metrics=metrics))
    receipt = dict(status='passed', strict_pair_stage=pair[0]['label'], seed=seed,
                   runs={name: str(run) for name, run in runs.items()}, cohort_size=len(refs),
                   all_saved_cohort_metrics_verified=summaries, selection='First three cohort geometries; no error ranking',
                   display='Original FEM triangles; shared0..1scale; gray holes; Gouraud display only; '
                   'no new triangulation, field smoothing, time interpolation or new inference.', figures=rendered)
    (output/'provenance.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'], output=str(output),
                         figures=[r['figure'] for r in rendered]), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--samples', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.root.resolve(strict=True), args.seed, args.samples, args.output.resolve())
