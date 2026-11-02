"""Saved FEM space/time droplet-shape audit; no new solve and no training labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel_droplet_metrics import contour_distance, contour_samples
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.problem import identity, walls


def audit(root, prediction, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root, prediction, output = Path(root), Path(prediction), Path(output)
    if output.exists():
        raise FileExistsError(output)
    c = ConfigDict(json.loads((prediction.parent/'config.json').read_text()))
    series = []
    diagnostics = {}
    for name in ('coarse', 'time', 'fine'):
        path = root/f'bdf-{name}.npz'
        with np.load(path, allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        if json.loads(str(values['identity_json'])) != json.loads(json.dumps(identity(c))):
            raise ValueError(f'Physical reference identity differs: {path}')
        series.append(values)
        raw = json.loads((root/f'bdf-{name}.json').read_text())
        diagnostics[name] = {key: raw[key] for key in ('grid', 'dt', 'dofs', 'maximum_newton_iterations', 'seconds')}
    with np.load(prediction, allow_pickle=False) as z:
        series.append({k: z[k] for k in z.files})
    fine = series[2]
    for values in series:
        for key in ('points', 'time', 'amplitudes'):
            np.testing.assert_array_equal(values[key], fine[key])
        if values['fields'].shape != fine['fields'].shape or not np.isfinite(values['fields']).all():
            raise ValueError('Aligned finite physical fields required')
    np.testing.assert_array_equal(series[3]['reference'], fine['fields'])
    points, times = fine['points'], fine['time']
    nx = len(np.unique(points[:, 0])); ny = len(points)//nx
    xx, ee = np.meshgrid((np.arange(nx)+.5)*c.geometry.length/nx,
                        (np.arange(ny)+.5)/ny, indexing='ij')
    bottom, top = walls(xx, fine['amplitudes'], c, xp=np)
    expected = np.stack((xx, bottom+ee*(top-bottom)), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(points, expected, atol=1e-10, rtol=0)
    measure = (top-bottom).ravel()*c.geometry.length/(nx*ny)
    relative = lambda a, b, w: float(np.sqrt(np.sum((a-b)**2*w)/np.sum(b*b*w)))
    rows = []
    names = ('space', 'time_step', 'fine', 'pino')
    for i, t in enumerate(times):
        ref = fine['fields'][i, :, 0]
        band = abs(ref) < .9
        entry = dict(time=float(t))
        for name, values in zip(names, series, strict=True):
            phi = values['fields'][i, :, 0]
            contours = contour_distance(points, phi, ref, (nx, ny), 1024)
            if contours['contour_missing']:
                raise ValueError(f'Missing droplet contour at {name}/{t}')
            _, _, curves = contour_samples(points, phi, (nx, ny), 1024)
            vertices = np.concatenate(curves)
            extent = np.ptp(vertices, axis=0)
            union = (phi < 0) | (ref < 0)
            entry[name] = dict(
                minority_fraction_relative_L2=relative((1-phi)/2, (1-ref)/2, measure),
                interface_phi_relative_L2=relative(phi, ref, measure*band),
                negative_phase_IoU=float(np.sum(measure*((phi < 0)&(ref < 0)))/np.sum(measure*union)),
                contour_mean_distance_over_initial_radius=contours['mean_distance']/c.initial.radius,
                zero_contour=contours, axis_aligned_width_ratio=float(extent[0]/extent[1]))
        rows.append(entry)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.25), layout='constrained')
    selected = [int(np.argmin(abs(times-t))) for t in (1.5, 3., 4.5, 6.)]
    labels = [f'FEM {d["grid"][0]}x{d["grid"][1]}, dt {d["dt"]:g}' for d in diagnostics.values()]
    labels.append('PINO checkpoint')
    for ax, i in zip(axes, selected, strict=True):
        for values, label, color, style in zip(series, labels,
                ('#009E73', '#0072B2', 'black', '#D55E00'), (':', '--', '-', '-'), strict=True):
            _, _, curves = contour_samples(points, values['fields'][i, :, 0], (nx, ny), 1024)
            for j, curve in enumerate(curves):
                ax.plot(*curve.T, color=color, ls=style, lw=1.7, label=label if j == 0 else None)
        cloud, weight, _ = contour_samples(points, fine['fields'][i, :, 0], (nx, ny), 1024)
        center = np.sum(cloud*weight[:, None], axis=0)/weight.sum()
        half = 1.5*c.initial.radius
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               aspect='equal', title=f't = {times[i]:g}', xlabel='x'); ax.grid(alpha=.15)
    axes[0].set_ylabel('y'); axes[0].legend(fontsize=7)
    fig.suptitle('FEM space/time consistency on droplet contours; saved fields only')
    output.mkdir(parents=True)
    fig.savefig(output/'contours.png', dpi=200); fig.savefig(output/'contours.pdf'); plt.close(fig)
    result = dict(status='saved_FEM_shape_refinement_completed', root=str(root), prediction=str(prediction),
        spatial_query_shape=[nx, ny], reference_identity=identity(c), FEM=diagnostics, rows=rows,
        caveat='Two meshes/timesteps from one FEM implementation: consistency, not exact error bounds or independent model certification; chemical potential was outside the old acceptance gate')
    (output/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], terminal=rows[-1]), indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True); parser.add_argument('--prediction', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); audit(args.reference_root, args.prediction, args.output)
