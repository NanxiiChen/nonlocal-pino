"""Saved-field droplet shape diagnostics; reference masks never enter training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contourpy
import numpy as np
from scipy.spatial import cKDTree
from ml_collections import ConfigDict
from examples.channel_nsch2d.problem import walls


def contour_samples(points, phi, shape, samples):
    q = points.reshape(*shape, 2)
    curves = contourpy.contour_generator(x=q[..., 0], y=q[..., 1],
        z=phi.reshape(shape), name='serial').lines(0.)
    clouds, measures = [], []
    for curve in curves:
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        keep = np.r_[True, lengths > 0]
        curve = curve[keep]
        if len(curve) < 2:
            raise ValueError('Degenerate zero contour')
        arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        nodes = (np.arange(samples) + .5) * arclength[-1] / samples
        clouds.append(np.column_stack([np.interp(nodes, arclength, curve[:, i]) for i in range(2)]))
        measures.append(np.full(samples, arclength[-1] / samples))
    if not curves:
        return np.empty((0, 2)), np.empty(0), curves
    return np.concatenate(clouds), np.concatenate(measures), curves


def contour_distance(points, prediction, reference, shape, samples):
    p, wp, pc = contour_samples(points, prediction, shape, samples)
    r, wr, rc = contour_samples(points, reference, shape, samples)
    if not len(p) or not len(r):
        return dict(predicted_components=len(pc), reference_components=len(rc),
                    mean_distance=None, sampled_hausdorff=None, contour_missing=True)
    dp = cKDTree(r).query(p)[0]
    dr = cKDTree(p).query(r)[0]
    return dict(predicted_components=len(pc), reference_components=len(rc),
        mean_distance=float(.5 * (np.sum(wp * dp) / wp.sum() + np.sum(wr * dr) / wr.sum())),
        sampled_hausdorff=float(max(dp.max(), dr.max())), contour_missing=False)


def evaluate(path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path, output = Path(path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    c = ConfigDict(json.loads((path.parent / 'config.json').read_text()))
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    prediction, reference, points = d['fields'], d['reference'], d['points']
    if prediction.shape != reference.shape or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('Finite aligned predicted/reference five-field arrays required')
    nx = len(np.unique(points[:, 0])); ny = len(points) // nx
    if nx * ny != len(points):
        raise ValueError('Expected mapped tensor queries')
    x, eta = np.meshgrid((np.arange(nx) + .5) * c.geometry.length / nx,
                         (np.arange(ny) + .5) / ny, indexing='ij')
    bottom, top = walls(x, d['amplitudes'], c, xp=np)
    expected = np.stack((x, bottom + eta * (top - bottom)), axis=-1).reshape(-1, 2)
    np.testing.assert_allclose(points, expected, atol=1e-10, rtol=0,
        err_msg='Area diagnostics require the declared midpoint query layout')
    measure = (top - bottom).ravel() * c.geometry.length / (nx * ny)
    relative = lambda a, b, w: float(np.sqrt(np.sum((a-b)**2*w) / np.sum(b*b*w)))
    rows = []
    for i, time in enumerate(d['time']):
        p, r = prediction[i, :, 0], reference[i, :, 0]
        band = np.abs(r) < .9
        droplet = r < .9
        if not np.any(band):
            raise ValueError('Reference interface absent from diagnostic queries')
        pi, ri = p < 0, r < 0
        union = np.sum(measure * (pi | ri))
        if union == 0:
            raise ValueError('Both droplets absent; IoU undefined')
        contour = contour_distance(points, p, r, (nx, ny), 1024)
        row = dict(time=float(time), phase_global_relative_L2=relative(p, r, 1.),
            minority_fraction_relative_L2=relative((1-p)/2, (1-r)/2, measure),
            interface_phi_relative_L2=relative(p, r, measure * band),
            droplet_band_phi_relative_L2=relative(p, r, measure * droplet),
            reference_negative_area_fraction=float(np.sum(measure * ri) / measure.sum()),
            negative_phase_IoU=float(np.sum(measure * (pi & ri)) / union),
            zero_contour=contour)
        if contour['mean_distance'] is not None:
            row['contour_mean_distance_over_initial_radius'] = contour['mean_distance'] / c.initial.radius
        if time > 0:
            for name, selection in (('velocity', slice(2, 4)), ('pressure', slice(4, 5))):
                for label, mask in (('global', np.ones_like(band)), ('interface', band), ('droplet_band', droplet)):
                    row[f'{name}_{label}_relative_L2'] = relative(prediction[i, :, selection],
                        reference[i, :, selection], measure[:, None] * mask[:, None])
        rows.append(row)
    output.mkdir(parents=True)
    selected = np.rint(np.linspace(0, len(rows)-1, 5)).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1), layout='constrained')
    for ax, i in zip(axes, selected, strict=True):
        for series, color, label in ((reference, 'black', 'FEM'), (prediction, '#D55E00', 'Prediction')):
            _, _, curves = contour_samples(points, series[i, :, 0], (nx, ny), 1024)
            for j, curve in enumerate(curves):
                ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.7, label=label if j == 0 else None)
        mask = reference[i, :, 0] < 0
        center = np.sum(points * (measure * mask)[:, None], axis=0) / np.sum(measure * mask)
        half = 1.6 * c.initial.radius
        ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
               aspect='equal', title=f't={d["time"][i]:g}', xlabel='x')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('y'); axes[0].legend(fontsize=8)
    fig.suptitle('Zero-phase contours; equal physical zoom scale around the reference droplet')
    fig.savefig(output/'contours.png', dpi=200); fig.savefig(output/'contours.pdf'); plt.close(fig)
    report = dict(status='saved_droplet_shape_diagnostics_completed', source=str(path),
        interface_definition='abs(reference_phi)<0.9', droplet_band_definition='reference_phi<0.9',
        contour_protocol='Linear mapped-grid contours,1024arc samples/component, arc-weighted symmetric mean',
        contour_scale=c.initial.radius, rows=rows,
        scope='Diagnostic reference masks only; original global phi is unweighted, localized metrics use physical area; no training labels')
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(source=str(path), terminal=rows[-1]), indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); evaluate(args.fields, args.output)
