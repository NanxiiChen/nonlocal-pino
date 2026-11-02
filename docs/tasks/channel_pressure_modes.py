"""Describe saved pressure errors; reference projections never modify predictions."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def decompose(error, points):
    """Orthogonal constant/x/y budgets in the original unweighted query norm."""
    error = np.asarray(error, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if error.ndim != 2 or points.shape != (error.shape[1], 2):
        raise ValueError('Pressure errors require aligned time/point and N-by-2 arrays')
    if not np.isfinite(error).all() or not np.isfinite(points).all():
        raise ValueError('Pressure mode decomposition requires finite values')
    centered = points - points.mean(axis=0)
    scale = np.ptp(points, axis=0)
    if np.any(scale <= 0):
        raise ValueError('Pressure queries must span both physical coordinates')
    basis = np.column_stack((np.ones(len(points)), centered / scale))
    q, r = np.linalg.qr(basis, mode='reduced')
    if np.linalg.cond(r) > 1e6:
        raise ValueError('Constant/x/y pressure reporting basis is poorly conditioned')
    coordinates = error @ q
    remaining = error - coordinates @ q.T
    energy = np.sum(error**2, axis=1)
    budgets = np.column_stack((coordinates**2, np.sum(remaining**2, axis=1)))
    gap = float(np.max(np.abs(budgets.sum(axis=1) - energy)))
    if gap > 1e-10 * max(1., float(energy.max())):
        raise AssertionError(f'Pressure projection energy does not close: {gap}')
    coefficient = np.linalg.solve(r, coordinates.T)
    independent = np.linalg.lstsq(basis, error.T, rcond=None)[0]
    parity = float(np.max(np.abs(coefficient - independent)))
    if parity > 1e-10 * max(1., float(np.max(np.abs(coefficient)))):
        raise AssertionError(f'QR/direct pressure reporting projection differs: {parity}')
    total = float(energy.sum())
    if total <= 0:
        raise ValueError('A zero pressure error needs no mode diagnosis')
    return dict(
        component_names=['constant', 'physical_x_after_constant',
                         'physical_y_after_constant_x', 'remaining'],
        overall_SSE_fractions=(budgets.sum(axis=0) / total).tolist(),
        per_time_SSE=energy.tolist(),
        per_time_component_SSE=budgets.tolist(),
        per_time_affine_coefficients=coefficient.T.tolist(),
        coordinate_center=points.mean(axis=0).tolist(),
        coordinate_scale=scale.tolist(),
        energy_closure_max_gap=gap,
        direct_least_squares_coefficient_max_gap=parity,
        interpretation='Descriptive reference-error projection, never a pressure correction')


def main(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    if not output.is_absolute():
        raise ValueError('Use an explicit absolute diagnostic output directory')
    payloads = []
    for name in runs:
        run = Path(name)
        if not run.is_absolute():
            raise ValueError('Use explicit absolute completed run paths')
        summary = json.loads((run / 'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError(f'Pressure diagnosis requires a completed run: {run}')
        with np.load(run / 'dense-fem-prediction.npz', allow_pickle=False) as z:
            saved = {k: z[k] for k in z.files}
        t = saved['time']
        shape = (len(t), len(saved['points']), 5)
        if (saved['fields'].shape != shape or saved['reference'].shape != shape
                or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
                or t[0] != 0 or not np.isfinite(saved['fields']).all()
                or not np.isfinite(saved['reference']).all()):
            raise ValueError(f'Invalid saved dense pressure payload: {run}')
        if payloads:
            for key in ('points', 'time', 'reference', 'amplitudes'):
                np.testing.assert_array_equal(saved[key], payloads[0][1][key])
        payloads.append((run, saved))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1), layout='constrained')
    reports = []
    for index, (run, saved) in enumerate(payloads):
        time = saved['time']
        positive = time > 0
        error = saved['fields'][positive, :, 4] - saved['reference'][positive, :, 4]
        target = saved['reference'][positive, :, 4]
        report = decompose(error, saved['points'])
        report.update(run=str(run), time=time[positive].tolist(),
                      positive_time_pressure_relative_L2=float(np.linalg.norm(error) / np.linalg.norm(target)))
        reports.append(report)
        x, inverse, count = np.unique(saved['points'][:, 0], return_inverse=True, return_counts=True)
        if len(x) < 4 or not np.all(count == count[0]):
            raise ValueError('Mean cross-section profiles require the declared structured query layout')
        for ax, fraction in zip(axes, (1 / 12, 1 / 6, .5, 1.), strict=True):
            selected = int(np.argmin(np.abs(time - fraction * time[-1])))
            if abs(time[selected] - fraction * time[-1]) > 1e-10:
                raise ValueError('Requested profile time is absent; do not interpolate silently')
            if index == 0:
                reference = np.bincount(inverse, weights=saved['reference'][selected, :, 4]) / count
                ax.plot(x, reference, color='black', linewidth=1.8, label='FEM')
            profile = np.bincount(inverse, weights=saved['fields'][selected, :, 4]) / count
            ax.plot(x, profile, linewidth=1.2, label=run.name)
            ax.set_title(f't = {time[selected]:g}')
            ax.set_xlabel('Physical x')
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean cross-section pressure')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', fontsize=7, ncol=1)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'pressure-profiles.{suffix}', dpi=200)
    plt.close(fig)
    receipt = dict(status='saved_pressure_spatial_mode_diagnostic_completed', reports=reports,
                   metric='Original unweighted query L2; positive times only',
                   scope='Reporting only; evolved reference never changes a model, gauge or training loss',
                   unique_cause_identified=False, trained_improvement_claimed=False)
    (output / 'report.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.runs, args.output)
