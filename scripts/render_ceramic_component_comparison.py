"""Three retained ceramic PINO arms, component violins and local mean trends.

CPU only. Uses saved predictions and actual development histories; no inference.
Intervals resample complete geometry trajectories, not individual time records.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from docs.tasks.ceramic_fem_comparison_figures import load, style, save, log_axis, FONT, PDFFONTS, REFERENCE
from scripts.local_trend import local_linear_matrix


FIELDS = (('phi', r'Phase $\phi$'), ('concentration', r'Concentration $c$'))
SIZE = (170, 60)
FRACTION = .30


def trajectory_interval(steps, case_errors):
    """Exact empirical four-trajectory bootstrap of the smoothed mean.

    Each bootstrap selects four complete geometry trajectories with replacement.
    All 4**4 equiprobable ordered selections are enumerated, retaining temporal
    dependence. LOWESS(log10(arithmetic case mean)) is applied to each selection.
    The pointwise percentile band is conditional on the training trajectory and
    this small development cohort; it is not cross-seed uncertainty or a
    guarantee of population coverage, and does not account for smoothing bias.
    """
    steps, values = np.asarray(steps, float), np.asarray(case_errors, float)
    if values.shape != (21, 4) or steps.shape != (21,) or np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError('Require21actual records of four positive finite case errors')
    grid = np.linspace(steps[0], steps[-1], 401)
    matrix = local_linear_matrix(steps, grid, FRACTION)
    selections = np.array(list(product(range(4), repeat=4)), dtype=np.int32)
    means = values[:, selections].mean(axis=-1).T
    mean = values.mean(axis=1)
    # Independent exact moments of an empirical iid four-draw mean.
    np.testing.assert_allclose(means.mean(axis=0), mean, rtol=2e-14, atol=2e-16)
    np.testing.assert_allclose(means.var(axis=0), values.var(axis=1)/4, rtol=2e-12, atol=2e-18)
    estimates = 10**(np.log10(means)@matrix.T)
    lower, upper = np.quantile(estimates, [.025, .975], axis=0)
    central = 10**(matrix@np.log10(mean))
    if np.any(lower > upper) or not np.isfinite(estimates).all():
        raise ValueError('Invalid bootstrap trend bounds')
    return dict(x=grid, smooth=central, lower=lower, upper=upper,
                raw_steps=steps, raw_mean=mean, raw_case_errors=values,
                bootstrap_geometry_indices=selections)


def audit_saved_fields(arms):
    """Recompute every displayed final norm directly from original saved fields."""
    reference = json.loads((REFERENCE/'manifest.json').read_text())
    records = []
    for arm in arms:
        directory = Path(arm['evaluation']).parent
        for i, case in enumerate(reference['cases']):
            prediction = directory/f'prediction_{i:03d}.npz'
            target_path = REFERENCE/case['file']
            with np.load(prediction, allow_pickle=False) as p, np.load(target_path, allow_pickle=False) as r:
                for key in ('points', 'times'):
                    np.testing.assert_array_equal(p[key], r[key])
                predicted = np.asarray(p['prediction'], dtype=np.float64)
                target = np.asarray(r['fields'], dtype=np.float64)
                if predicted.shape != target.shape or predicted.shape[0] != 11 or predicted.shape[2] != 2:
                    raise ValueError('Expected11saved times and two physical output channels')
                numerator = np.sum((predicted-target)**2, axis=(0, 1))
                denominator = np.sum(target**2, axis=(0, 1))
                actual = np.sqrt(numerator/denominator)
                np.testing.assert_allclose(actual, arm['final_values'][i][:2], rtol=2e-13, atol=1e-15)
                records.append(dict(method=arm['key'], sample=i, prediction=str(prediction),
                    reference=str(target_path), numerator=numerator.tolist(), denominator=denominator.tolist(),
                    errors=actual.tolist(), shape=list(target.shape)))
    if len(records) != 192:
        raise ValueError('All three methods and64shared samples must be present')
    return records


def panels(distribution):
    fig, axes = plt.subplots(1, 2, figsize=np.array(SIZE)/25.4)
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.75 if not distribution else .84, wspace=.27)
    for ax, (_, title) in zip(axes, FIELDS, strict=True):
        log_axis(ax)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, pad=5)
    axes[0].set_ylabel('Rel. L² error' if distribution else 'Weighted rel. L² error', labelpad=5)
    return fig, axes


def render(arms, output):
    fig, axes = panels(False)
    trends = []
    for j, (ax, (field, _)) in enumerate(zip(axes, FIELDS, strict=True)):
        for arm in arms:
            fitted = trajectory_interval(arm['steps'], np.asarray(arm['monitor_case_values'])[:, :, j])
            raw, = ax.plot(arm['steps'], fitted['raw_mean'], color=arm['color'], ls='-', lw=.55, alpha=.35)
            np.testing.assert_array_equal(raw.get_ydata(), np.asarray(arm['monitor_mean'])[:, j])
            ax.fill_between(fitted['x'], fitted['lower'], fitted['upper'], color=arm['color'], alpha=.18, lw=0)
            ax.plot(fitted['x'], fitted['smooth'], color=arm['color'], ls='-', lw=1.05, label=arm['label'])
            path = output/f"{arm['key']}_{field}_trend.npz"
            np.savez_compressed(path, **fitted)
            trends.append(dict(method=arm['key'], field=field, data=str(path)))
        ax.set_xticks([0, 5000, 10000, 15000, 20000], ['0', '5k', '10k', '15k', '20k'])
        ax.set_xlabel('Step', labelpad=4)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc='upper center',
               bbox_to_anchor=(.55, .995), handlelength=2.8, columnspacing=1.6)
    convergence = save(fig, output/'particle_corrosion2d_component_convergence', SIZE)
    convergence['trends'] = trends

    fig, axes = panels(True)
    densities = []
    for j, (ax, (field, _)) in enumerate(zip(axes, FIELDS, strict=True)):
        for i, arm in enumerate(arms):
            values = np.asarray(arm['final_values'])[:, j]
            logs = np.log10(values)
            kde = gaussian_kde(logs, bw_method='scott')
            bandwidth = float(np.sqrt(kde.covariance[0, 0]))
            support = np.linspace(logs.min()-3*bandwidth, logs.max()+3*bandwidth, 256)
            density = kde(support); width = .32*density/density.max()
            ax.fill_betweenx(10**support, i-width, i+width, color=arm['color'],
                             alpha=.85, edgecolor='#333333', linewidth=.6)
            quartiles = np.quantile(values, [.25, .5, .75])
            for q in quartiles:
                w = .32*float(kde(np.log10(q))[0])/density.max()
                ax.plot([i-w, i+w], [q, q], color='#333333', lw=.6, ls=(0, (2, 1.5)))
            densities.append(dict(method=arm['key'], field=field, count=len(values),
                log10_bandwidth=bandwidth, scott_factor=float(kde.factor), quartiles=quartiles.tolist(),
                support_log10=support.tolist(), density=density.tolist()))
        ax.set_xticks([0, 1, 2], ['Nonlocal', 'Local AD', 'FEM PINO'])
        ax.margins(x=.12)
    distribution = save(fig, output/'particle_corrosion2d_component_distribution', SIZE)
    distribution['kde'] = densities
    return dict(convergence=convergence, distribution=distribution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arms = load()
    audit = audit_saved_fields(arms)
    with matplotlib.rc_context(style()):
        figures = render(arms, output)
    record = dict(schema='ceramic_component_comparison_v1', status='passed',
        renderer=str(Path(__file__).resolve()),
        command=['env', 'PYTHONPATH=/tmp/pino-pdf-audit', 'OPENBLAS_NUM_THREADS=1', str(Path(sys.executable).absolute()),
                 '-m', 'scripts.render_ceramic_component_comparison', '--output', str(output)],
        shared_loader=str(Path('docs/tasks/ceramic_fem_comparison_figures.py').resolve()),
        smoothing_source=str(Path('scripts/local_trend.py').resolve()),
        source_selection=str(Path('artifacts/ceramic_fem_comparison_publication.json').resolve()),
        fonts=dict(regular=str(FONT), math_fontset='dejavusans',
                   mathematics_directory=str(Path(matplotlib.get_data_path())/'fonts/ttf')),
        software=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        final_metric='Each channel separately: Euclidean norms over all11times and spatial nodes,64common test samples, final20k,seed42.',
        convergence_metric='21actual0:1000:20000 records; arithmetic mean of4development geometries, physical-volume/trapezoidal-time weighted relative norms.',
        smoothing=dict(fraction=FRACTION, neighbors=int(21*FRACTION), log_space='log10', grid_points=401,
            robust_iterations=0, interval='Pointwise percentile95% empirical whole-geometry-trajectory bootstrap',
            geometries=4, all_ordered_bootstrap_selections=256,
            scope='Approximate small-cohort interval conditional on one training trajectory; no cross-seed uncertainty, simultaneous coverage or correction for smoothing/model-selection bias.'),
        density=dict(space='log10', bandwidth='scott', support_bandwidths=3, grid_points=256, normalization='equal maximum width'),
        figures=figures, arms=arms, saved_field_audit=audit,
        training_performed=False, inference_performed=False, reference_solves=False)
    (output/'provenance.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed', output=str(output), checked_channel_norms=384,
        means={a['label']:a['final_statistics'] for a in arms}), indent=2))


if __name__ == '__main__':
    main()
