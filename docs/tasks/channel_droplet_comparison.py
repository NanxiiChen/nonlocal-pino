"""Compare droplet contours and diagnostics on exactly the same physical queries."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_droplet_metrics import contour_samples
import numpy as np


def compare(control, trial, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    control, trial, output = Path(control), Path(trial), Path(output)
    reports = [json.loads((p/'metrics.json').read_text()) for p in (control, trial)]
    runs = [Path(r['source']).parent for r in reports]
    provenance = [json.loads((p/'refinement.json').read_text()) for p in runs]
    summaries = [json.loads((p/'refinement_summary.json').read_text()) for p in runs]
    if provenance[0]['parent_run'] != provenance[1]['parent_run']:
        raise ValueError('Refinement parents differ')
    if provenance[0]['settings'] != provenance[1]['settings']:
        raise ValueError('Full-time optimizer budgets or settings differ')
    fields = []
    for report in reports:
        with np.load(report['source'], allow_pickle=False) as z:
            fields.append({k: z[k] for k in z.files})
    for key in ('reference', 'points', 'time', 'amplitudes'):
        np.testing.assert_array_equal(fields[0][key], fields[1][key], err_msg=f'Comparison {key} differs')
    for key in ('interface_definition', 'droplet_band_definition', 'contour_protocol', 'contour_scale'):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f'Diagnostic definition differs: {key}')
    times = fields[0]['time']
    for report in reports:
        np.testing.assert_array_equal([r['time'] for r in report['rows']], times)
    if output.exists():
        raise FileExistsError(output)
    points = fields[0]['points']
    nx = len(np.unique(points[:, 0])); shape = (nx, len(points)//nx)
    selected = [int(np.argmin(abs(times-t))) for t in (1.5, 3., 4.5, 6.)]
    if not np.allclose(times[selected], (1.5, 3., 4.5, 6.), atol=1e-8, rtol=0):
        raise ValueError('Expected common positive-time field snapshots are absent')
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.3), layout='constrained')
    colors, labels = ('#0072B2', '#D55E00'), ('Original residual', 'Droplet weighted')
    radius = reports[0]['contour_scale']
    for ax, i in zip(axes[0], selected, strict=True):
        for values, color, label in ((fields[0]['reference'], 'black', 'Local FEM'),
                (fields[0]['fields'], colors[0], labels[0]), (fields[1]['fields'], colors[1], labels[1])):
            _, _, contours = contour_samples(points, values[i, :, 0], shape, 1024)
            for j, curve in enumerate(contours):
                ax.plot(*curve.T, color=color, label=label if j == 0 else None, lw=1.7,
                        ls='--' if color == colors[0] else '-')
        cloud, measure, _ = contour_samples(points, fields[0]['reference'][i, :, 0], shape, 1024)
        if not len(cloud):
            raise ValueError('Reference contour absent from comparison')
        center = np.sum(cloud*measure[:, None], axis=0)/measure.sum()
        half = 1.5*radius
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               aspect='equal', title=f't = {times[i]:g}', xlabel='x')
        ax.grid(alpha=.15)
    axes[0, 0].set_ylabel('y'); axes[0, 0].legend(fontsize=7)
    metrics = (('minority_fraction_relative_L2', 'Droplet fraction relative L2 (%)'),
               ('interface_phi_relative_L2', 'Interface phase relative L2 (%)'),
               ('negative_phase_IoU', 'Droplet IoU (%)'),
               ('contour_mean_distance_over_initial_radius', 'Mean contour distance / R0 (%)'))
    for ax, (key, label) in zip(axes[1], metrics, strict=True):
        for report, color, name in zip(reports, colors, labels, strict=True):
            values = [r[key] for r in report['rows']]
            ax.plot(times, np.asarray(values)*100, color=color, label=name, lw=1.6)
        ax.set(xlabel='Time', ylabel=label); ax.grid(alpha=.2)
    budget = provenance[0]['settings']['iterations']
    fig.suptitle(f'Same parent, same {budget}-step full-time refinement budget; one development geometry')
    terminal = {key: dict(control=reports[0]['rows'][-1][key], trial=reports[1]['rows'][-1][key])
                for key, _ in metrics}
    output.mkdir(parents=True)
    fig.savefig(output/'comparison.png', dpi=200)
    fig.savefig(output/'comparison.pdf'); plt.close(fig)
    result = dict(status='common_query_droplet_comparison_completed',
        control=str(control/'metrics.json'), trial=str(trial/'metrics.json'), terminal=terminal,
        accepted_steps=dict(control=summaries[0]['iterations'], trial=summaries[1]['iterations']),
        same_reference_points_times_geometry=True,
        scope='One development geometry; retained completed original control, new weighted trial; masks only evaluate saved fields; losses use different norms')
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True); parser.add_argument('--trial', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); compare(args.control, args.trial, args.output)
