"""Three-way saved-field comparison; a completed trajectory is not a reference gate."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.problem import walls, identity
from docs.tasks.channel_droplet_metrics import contour_distance, contour_samples


def compare(numerical, fem, pino, output, pino_label):
    numerical, fem, pino, output = map(Path, (numerical, fem, pino, output))
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((numerical/'report.json').read_text())
    if report['status'] != 'direct_baseline_completed_NOT_refinement_validated':
        raise ValueError('Need a genuinely completed numerical trajectory, with its refinement scope retained')
    c = ConfigDict(json.loads((numerical/'config.json').read_text()))
    pc = ConfigDict(json.loads((pino.parent/'config.json').read_text()))
    if identity(c) != identity(pc):
        raise ValueError('PINO/numerical physical identity mismatch')
    with np.load(numerical/'solution.npz') as z:
        direct = {k: z[k] for k in ('fields', 'time', 'points', 'amplitudes', 'identity_json')}
    with np.load(fem) as z:
        reference = {k: z[k] for k in ('fields', 'time', 'points', 'amplitudes', 'identity_json')}
    with np.load(pino) as z:
        prediction = {k: z[k] for k in ('fields', 'time', 'points', 'amplitudes', 'reference')}
    for data in (direct, prediction):
        for key in ('points', 'time', 'amplitudes'):
            np.testing.assert_allclose(data[key], reference[key], atol=1e-12, rtol=0)
        if data['fields'].shape != reference['fields'].shape or not np.isfinite(data['fields']).all():
            raise ValueError('Finite aligned five-field arrays required')
    if json.loads(str(direct['identity_json'])) != json.loads(str(reference['identity_json'])):
        raise ValueError('FEM/numerical physical identity mismatch')
    np.testing.assert_array_equal(prediction['reference'], reference['fields'])
    points, times = reference['points'], reference['time']
    nx = len(np.unique(points[:, 0])); ny = len(points)//nx
    b, top = walls(points[:, 0], direct['amplitudes'], c, xp=np)
    measure = (top-b)*c.geometry.length/(nx*ny)
    relative = lambda a, b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
    rows = []
    for label, actual, target in (
        ('GMLS_vs_FEM', direct['fields'], reference['fields']),
        ('PINO_vs_FEM', prediction['fields'], reference['fields']),
        ('PINO_vs_GMLS', prediction['fields'], direct['fields']),
    ):
        p, r = actual[-1, :, 0], target[-1, :, 0]
        contour = contour_distance(points, p, r, (nx, ny), 1024)
        rows.append(dict(comparison=label, phase_pooled=relative(actual[..., 0], target[..., 0]),
            phase_terminal=relative(p, r),
            minority_terminal=float(np.sqrt(np.sum(measure*(p-r)**2)/np.sum(measure*(1-r)**2))),
            velocity_positive=relative(actual[1:, :, 2:4], target[1:, :, 2:4]),
            pressure_positive=relative(actual[1:, :, 4], target[1:, :, 4]),
            terminal_contour_mean_over_radius=contour['mean_distance']/c.initial.radius,
            terminal_zero_contour=contour))
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), layout='constrained')
    selected = [0, 3, 6, 9, 12]
    if len(times) != 13:
        raise ValueError('This gallery requires all 13 aligned half-second snapshots')
    for ax, i in zip(axes, selected, strict=True):
        for data, color, label in ((reference, 'black', 'Local FEM'),
            (direct, '#0072B2', f'Direct GMLS {report["grid"][0]}x{report["grid"][1]}'),
            (prediction, '#D55E00', pino_label)):
            _, _, curves = contour_samples(points, data['fields'][i, :, 0], (nx, ny), 1024)
            for j, curve in enumerate(curves):
                ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.6, label=label if j == 0 else None)
        mask = reference['fields'][i, :, 0] < 0
        center = np.average(points[mask], weights=measure[mask], axis=0)
        half = 1.6*c.initial.radius
        ax.set(aspect='equal', xlim=(center[0]-half, center[0]+half),
            ylim=(center[1]-half, center[1]+half), title=f't={times[i]:g}', xlabel='x')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('y'); axes[0].legend(fontsize=7)
    fig.suptitle('Preliminary numerical comparison: zero-phase contours; GMLS refinement not yet validated')
    fig.savefig(output/'three-way-contours.png', dpi=200)
    fig.savefig(output/'three-way-contours.pdf'); plt.close(fig)
    result = dict(status='preliminary_completed_trajectory_comparison_NOT_validated_reference',
        numerical=str(numerical), FEM=str(fem), PINO=str(pino), PINO_label=pino_label, grid=report['grid'], dt=report['dt'],
        pairs=rows, query_count=len(points), times=times.tolist(),
        physical_identity_and_all_query_frames_checked=True,
        direct_solver_residual_max=report['maximum_newton_residual'],
        direct_solver_diagnostics=report['diagnostics'],
        scope='Original unweighted phi L2; positive-time flow L2; minority/contour diagnostics. GMLS uses first derivatives and fluxes different from PINO AD/SBP. Spatial/time/pressure-boundary compatibility validation still required.')
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numerical', required=True); parser.add_argument('--fem', required=True)
    parser.add_argument('--pino', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--pino-label', required=True)
    a = parser.parse_args(); compare(a.numerical, a.fem, a.pino, a.output, a.pino_label)
