"""Reproduce final channel-mean error violins and phase-field README sheets.

No checkpoint inference, training, reference solve or GPU work. Source arrays
remain external; configuration snapshots and plotted scalar values live in the
repository manifests. Run from the repo with JAX_PLATFORMS=cpu and --case all.
"""
import argparse
import colorsys
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.colors import to_rgb
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.tri import Triangulation
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, __version__ as pillow_version
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from examples.particle_corrosion2d.evaluate import metrics as particle_metrics

MANIFEST = REPO/'artifacts/pino_derivative_comparisons.json'
RECEIPT = REPO/'artifacts/pino_derivative_comparison_validation.json'


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def relative(prediction, reference, weights):
    p, r = np.asarray(prediction, np.float64), np.asarray(reference, np.float64)
    if p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Fields must be finite and identically shaped')
    denominator = np.sum(weights*r*r)
    if denominator <= 0:
        raise ValueError('Relative norm requires nonzero reference')
    return float(np.sqrt(np.sum(weights*(p-r)**2)/denominator))


def time_indices(times, wanted):
    result = []
    for time in wanted:
        match = np.flatnonzero(np.isclose(times, time, rtol=0, atol=1e-12))
        if len(match) != 1:
            raise ValueError(f'Expected one actual saved snapshot at {time}')
        result.append(int(match[0]))
    return result


def save(fig, path, settings):
    path = REPO/path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=settings['dpi'], facecolor='white')
    plt.close(fig)
    with Image.open(path) as source:
        compact = source.convert('RGB').quantize(colors=settings['palette_colors'],
            method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    compact.save(path, optimize=True, dpi=(settings['dpi'], settings['dpi']))
    size = path.stat().st_size
    if size > settings['max_image_bytes']:
        raise ValueError(f'Figure exceeds declared byte budget: {path}, {size}')
    return dict(file=str(path.relative_to(REPO)), bytes=size, pixels=list(compact.size),
                dpi=settings['dpi'], palette_colors=settings['palette_colors'])


def sample_statistics(records):
    if len(records) != 64 or [r['index'] for r in records] != list(range(64)):
        raise ValueError('Require64ordered per-input error records')
    result = {}
    for key in ('phi_pooled', 'phi_terminal', 'c_pooled', 'c_terminal'):
        values = np.array([r[key] for r in records], np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Errors must be finite and nonnegative')
        result[key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)),
            minimum=float(values.min()), maximum=float(values.max()), median=float(np.median(values)),
            p90=float(np.quantile(values, .9)), count=64, ddof=1)
    return result


def channel_relative_errors(prediction, reference):
    """Normalize each physical channel over all space-time points, then average."""
    p, r = np.asarray(prediction, np.float64), np.asarray(reference, np.float64)
    if (p.shape != r.shape or p.ndim < 3 or p.shape[-1] != 2 or
            not np.isfinite(p).all() or not np.isfinite(r).all()):
        raise ValueError('Expected finite matching space-time fields with channels(phi,c)')
    pp, rr = p.reshape(-1, 2), r.reshape(-1, 2)
    numerator = np.sqrt(np.sum((pp-rr)**2, axis=0))
    denominator = np.sqrt(np.sum(rr**2, axis=0))
    if np.any(denominator <= 0):
        raise ValueError('Every channel needs a positive full space-time reference norm')
    errors = numerator/denominator
    return dict(channel_relative_l2=dict(zip(('phi', 'c'), errors.tolist(), strict=True)),
        channel_mean_relative_l2=float(errors.mean()), numerator_norms=numerator.tolist(),
        denominator_norms=denominator.tolist(), evaluation_shape=list(p.shape),
        evaluation_points_per_channel=len(pp))


def distribution_statistics(records):
    if len(records) != 64 or [r['index'] for r in records] != list(range(64)):
        raise ValueError('Require64ordered per-input observations')
    values = np.array([r['channel_mean_relative_l2'] for r in records], np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Require finite nonnegative channel-mean errors')
    q1, median, q3 = np.quantile(values, [.25, .5, .75])
    return dict(count=64, mean=float(values.mean()), std=float(values.std(ddof=1)),
        ddof=1, minimum=float(values.min()), maximum=float(values.max()),
        q1=float(q1), median=float(median), q3=float(q3), p90=float(np.quantile(values, .9)))


def load_case(name, case, settings, journal):
    """Recompute final fields on64inputs for the first strictly audited seed only."""
    refs = [arrays(path) for path in case['reference_files']]
    if len(refs) != 64 or len(case['sample_identities']) != 64:
        raise ValueError('Expected a common64input local-reference cohort')
    selected, seed = settings['field_samples'], settings['distribution']['seed']
    if selected != [0, 1, 2] or settings['field_seed'] != 42 or seed != 42:
        raise ValueError('Use the first three gallery inputs and first strict paired seed42')
    pair = [s for s in case['pairs'] if s['label'] == f'{name}-s{seed}-pair']
    if len(pair) != 1 or pair[0]['status'] != 'passed':
        raise ValueError('Both displayed methods must pass their strict pairing audit')
    for pair_record in case['pairs']:
        if pair_record not in journal['stages']:
            raise ValueError('Declared pairing status differs from the raw journal')
    runs = [r for r in case['runs'] if r['seed'] == seed]
    if len(runs) != 2 or sorted(r['method'] for r in runs) != ['local_ad', 'nonlocal']:
        raise ValueError('Expected exactly one final run for each method at seed42')
    gallery, scores = {}, []
    reference_manifest = read(case['reference_manifest'])
    particle = name == 'particle_corrosion2d'
    if particle and read(case['reference_validation'])['status'] != 'passed':
        raise ValueError('New local FEM cohort failed reference qualification')
    indices = case['metric_evaluation_grid']['time_indices']
    for ref in refs:
        times = ref['times'] if particle else ref['time']
        expected = np.linspace(0, 1, 11 if particle else 21)
        np.testing.assert_allclose(times[indices], expected, rtol=0, atol=1e-15)
    for record in runs:
        c = read(record['config_file'])
        if c != record['config_snapshot']:
            raise ValueError(f'Saved configuration changed: {record["config_file"]}')
        for key in ('checkpoint', 'summary'):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
        source = Path(record['source_directory'])
        if (not record['frozen_source_files'] or
                sorted(str(p.relative_to(source)) for p in source.rglob('*.py')) != record['frozen_source_files']):
            raise ValueError('Frozen numerical source inventory changed')
        evaluation = read(record['evaluation64_summary'])
        final = evaluation['final']
        if (evaluation['status'] != 'passed' or evaluation['count'] != 64 or
                evaluation['run'] != record['run'] or evaluation['training_performed'] or
                final['step'] != record['steps'] or final['checkpoint'] != record['checkpoint'] or
                final['reference'] != str(Path(case['reference_manifest']).parent)):
            raise ValueError('Final checkpoint evaluation provenance mismatch')
        if sample_statistics(final['records']) != final['statistics']:
            raise ValueError('Historical metrics do not match their saved per-input values')
        final_records, metric_records, fields = [], [], []
        if particle:
            if reference_manifest['backend'] != 'local_p1_lumped_fem':
                raise ValueError('Expected independent local FEM reference')
            for key in ('geometry', 'particles', 'physical', 'nondim'):
                if c[key] != reference_manifest['config'][key]:
                    raise ValueError('Reference physics/geometry differ: '+key)
            for i, (path, ref) in enumerate(zip(record['prediction_files'], refs, strict=True)):
                data = arrays(path)
                for key in ('points', 'times'):
                    np.testing.assert_array_equal(data[key], ref[key])
                score = particle_metrics(data['prediction'], ref['fields'], ref['volumes'], ref['times'])
                for key, value in score.items():
                    np.testing.assert_allclose(value, final['records'][i][key], rtol=1e-12, atol=1e-14)
                final_records.append(dict(index=i, **score))
                metric_records.append(dict(index=i, **channel_relative_errors(
                    data['prediction'][indices], ref['fields'][indices])))
                if i in selected:
                    fields.append(data['prediction'])
        else:
            reference_config = read(reference_manifest['config'])
            for key in ('physical', 'nondim', 'surface'):
                if c[key] != reference_config[key]:
                    raise ValueError('Reference physical parameters differ: '+key)
            data = arrays(record['prediction_file'])
            np.testing.assert_array_equal(data['surface_ids'], case['sample_identities'])
            if indices != final['regular_time_indices']:
                raise ValueError('Declared uniform evaluation grid changed')
            for i, ref in enumerate(refs):
                for key in ('X', 'Y', 'time'):
                    np.testing.assert_array_equal(data[key], ref[key])
                score = {}
                for channel, field in enumerate(('phi', 'c')):
                    prediction, target = data['prediction'][i, ..., channel], ref['fields'][..., channel]
                    score[field+'_pooled'] = relative(prediction[indices], target[indices], 1.)
                    score[field+'_terminal'] = relative(prediction[-1], target[-1], 1.)
                for key, value in score.items():
                    np.testing.assert_allclose(value, final['records'][i][key], rtol=1e-12, atol=1e-14)
                final_records.append(dict(index=i, surface_id=int(case['sample_identities'][i]), **score))
                metric_records.append(dict(index=i, **channel_relative_errors(
                    data['prediction'][i, indices], ref['fields'][indices])))
            fields = [data['prediction'][i] for i in selected]
        stats = sample_statistics(final_records)
        if stats != final['statistics']:
            raise ValueError('Final-field historical metric mismatch')
        scores.append(dict(seed=seed, method=record['method'], checkpoint=record['checkpoint'],
            step=final['step'], historical_statistics=stats, conditions=final_records,
            metric_records=metric_records, distribution_statistics=distribution_statistics(metric_records),
            count=64, metric_evaluation_grid=case['metric_evaluation_grid']))
        gallery[record['method']] = fields
    return refs, gallery, scores


def publication_style(style):
    """Resolve the exact external font file; never silently substitute a face."""
    path = Path(style['font_file']).resolve(strict=True)
    with TTFont(path) as face:
        family = face['name'].getDebugName(1)
        flags = int(face['OS/2'].fsType)
        if family != style['font_family'] or 'glyf' not in face:
            raise ValueError(f'Expected {style["font_family"]} TrueType: {path}')
        if flags & (0x0002 | 0x0100 | 0x0200):
            raise ValueError(f'Font disallows this outline/subset embedding: {path}, {flags}')
        info = dict(file=str(path), source=style['font_source'], family=family,
            style=face['name'].getDebugName(2), version=face['name'].getDebugName(5),
            bytes=path.stat().st_size, fsType=flags)
    font_manager.fontManager.addfont(str(path))
    resolved = Path(font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)).resolve()
    if resolved != path:
        raise ValueError(f'Unexpected font resolution: {resolved}; expected {path}')
    sizes = [style[key] for key in ('font_size', 'label_size', 'title_size')]
    if any(not 5 <= value <= 7 for value in sizes):
        raise ValueError('Publication font sizes must be within5–7pt')
    rc = {'font.family': 'sans-serif', 'font.sans-serif': [family],
        'font.size': style['font_size'], 'axes.labelsize': style['label_size'],
        'axes.titlesize': style['title_size'], 'xtick.labelsize': style['font_size'],
        'ytick.labelsize': style['font_size'], 'legend.fontsize': style['font_size'],
        'text.color': style['text_color'], 'axes.labelcolor': style['text_color'],
        'axes.titlecolor': style['text_color'], 'xtick.color': style['text_color'],
        'ytick.color': style['text_color'], 'axes.linewidth': style['axes_linewidth'],
        'xtick.major.width': style['axes_linewidth'], 'ytick.major.width': style['axes_linewidth'],
        'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.grid': False,
        'legend.frameon': False, 'pdf.fonttype': style['pdf_fonttype'], 'ps.fonttype': 42,
        'pdf.use14corefonts': style['pdf_use14corefonts'], 'text.usetex': False,
        'mathtext.fontset': style['math_fontset'], 'mathtext.fallback': None,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'savefig.transparent': False}
    return rc, info


def save_publication(fig, case, settings, font_info, figure_mm):
    """Export the same figure as vector PDF and lossless PNG at final size."""
    style = settings['distribution']['publication']
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            item.set_fontproperties(font_manager.FontProperties(
                fname=font_info['file'], size=item.get_fontsize()))
            item.set_color(style['text_color'])
    fig.canvas.draw()
    text_records = []
    # A log locator creates off-screen tick artists that are not actually drawn.
    offscreen_ticks = set()
    for ax in fig.axes:
        lo, hi = ax.get_ylim()
        for tick in [*ax.yaxis.get_major_ticks(), *ax.yaxis.get_minor_ticks()]:
            if not lo <= tick.get_loc() <= hi:
                offscreen_ticks.update((tick.label1, tick.label2))
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text() or item in offscreen_ticks:
            continue
        size = item.get_fontsize()
        if not 5 <= size <= 7:
            raise ValueError(f'Out-of-range text size: {size}, {item.get_text()}')
        box = item.get_window_extent(fig.canvas.get_renderer())
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
            raise ValueError(f'Text outside final figure: {item.get_text()}')
        text_records.append(dict(text=item.get_text(), size_pt=size, color=item.get_color()))
    png, pdf = REPO/case['outputs']['distribution'], Path(case['publication_pdf'])
    if pdf.parent != Path(style['pdf_directory']):
        raise ValueError('PDF output directory does not match the manifest')
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, format='pdf', dpi=style['dpi'], metadata={
        'Creator': 'nonlocal-pino publication violin renderer',
        'CreationDate': style['pdf_metadata_date'], 'ModDate': style['pdf_metadata_date']})
    fig.savefig(png, format='png', dpi=style['dpi'])
    plt.close(fig)
    with Image.open(png) as source:
        rgb = source.convert('RGB')
    rgb.save(png, optimize=True, dpi=(style['dpi'], style['dpi']))
    if png.stat().st_size > settings['max_image_bytes']:
        raise ValueError(f'PNG exceeds the asset budget: {png}')
    file_record = str(png.relative_to(REPO)) if png.is_relative_to(REPO) else str(png)
    return dict(file=file_record, bytes=png.stat().st_size,
        pixels=list(rgb.size), dpi=style['dpi'], encoding=style['png_encoding'],
        figure_mm=figure_mm, font=font_info, text=text_records,
        pdf=dict(file=str(pdf), bytes=pdf.stat().st_size, fonttype=style['pdf_fonttype']))


def distribution_values(scores, rule):
    """Keep the existing physical-channel errors as unscaled sample ratios."""
    if (rule['metric'] != 'channel_mean_spacetime_relative_l2' or rule['channels'] != ['phi', 'c']
            or rule['channel_weights'] != [.5, .5] or rule['point_weights'] != 'uniform'
            or rule['sample_weights'] != 'uniform' or rule['axis'] != 'log_ratio'
            or rule['display_multiplier'] != 1 or rule['kde_coordinate'] != 'natural_log_ratio'
            or rule['density_normalization'] != 'area_in_log_error_within_case'
            or rule['density_support'] != 'log_extrema_plus_bandwidth_tails'):
        raise ValueError('Unsupported distribution metric/display specification')
    methods = ('nonlocal', 'local_ad')
    rows = [next(s for s in scores if s['method'] == m and s['seed'] == rule['seed']) for m in methods]
    values = [np.array([r['channel_mean_relative_l2'] for r in s['metric_records']]) for s in rows]
    if any(len(v) != 64 or not np.isfinite(v).all() or np.any(v <= 0) or np.std(v) == 0 for v in values):
        raise ValueError('Log KDE requires64finite positive observations with nonzero variance')
    return methods, rows, values


def draw_distribution_axis(ax, values, rule):
    """Positive-support log-error KDE; axes still show unchanged error ratios."""
    style = rule['publication']
    methods = ('nonlocal', 'local_ad')
    if style['example_titles'] is not False:
        raise ValueError('Identify examples by filenames and captions, not figure titles')
    if style['y_limits_mode'] != 'matplotlib_auto':
        raise ValueError('Distribution y limits must use Matplotlib autoscaling')
    if rule['kde_tail_bandwidths'] <= 0:
        raise ValueError('KDE tails require positive bandwidth extension')
    log_supports, densities, bandwidths = [], [], []
    for value in values:
        z = np.log(value)
        kernel = gaussian_kde(z, bw_method=rule['kde_bandwidth'])
        bandwidth = float(np.sqrt(kernel.covariance[0, 0]))
        padding = rule['kde_tail_bandwidths']*bandwidth
        support = np.linspace(z.min()-padding, z.max()+padding, rule['kde_points'])
        log_supports.append(support)
        densities.append(kernel(support))
        bandwidths.append(bandwidth)
    supports = [np.exp(z) for z in log_supports]
    maximum = max(d.max() for d in densities)
    records = []
    for i, (value, support, density, method) in enumerate(zip(values, supports, densities, methods, strict=True)):
        h, lightness, saturation = colorsys.rgb_to_hls(*to_rgb(style['colors'][method]))
        color = colorsys.hls_to_rgb(h, lightness, saturation*style['saturation'])
        width = .5*rule['violin_width']*density/maximum
        ax.fill_betweenx(support, i-width, i+width, facecolor=color,
                         edgecolor=style['outline_color'], linewidth=style['body_linewidth'],
                         alpha=style['body_alpha'])
        quartiles = np.quantile(value, [.25, .5, .75])
        for q, dash in zip(quartiles, [(1.5, 1.5), (3, 1.5), (1.5, 1.5)], strict=True):
            w = np.interp(np.log(q), log_supports[i], width)
            ax.plot([i-w, i+w], [q, q], color=style['quartile_color'],
                    linewidth=style['quartile_linewidth'], linestyle=(0, dash), solid_capstyle='butt')
        records.append(dict(method=method, support=support.tolist(), density=density.tolist(),
                            quartiles=quartiles.tolist(), facecolor=list(color),
                            log_bandwidth=bandwidths[i], density_measure='per_unit_log_error'))
    ax.set(yscale='log', xlim=(-.5, 1.5),
           xticks=[0, 1], xticklabels=['Nonlocal', 'Local AD'])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='major', length=2.5, width=style['axes_linewidth'])
    ax.tick_params(which='minor', length=1.5, width=style['axes_linewidth'])
    ax.set_ylabel(style['label'])
    return dict(methods=records, actual_y_limits=list(ax.get_ylim()),
                autoscale_y=bool(ax.get_autoscaley_on()))


def distribution(name, case, scores, settings):
    rule = settings['distribution']
    methods, rows, values = distribution_values(scores, rule)
    style = rule['publication']
    rc, font_info = publication_style(style)
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=np.asarray(style['figure_mm'])/25.4)
        fig.subplots_adjust(**style['margins'])
        densities = draw_distribution_axis(ax, values, rule)
        output = save_publication(fig, case, settings, font_info, style['figure_mm'])
    return dict(**output, seed=rule['seed'],
        metric=rule['metric'], settings=rule, evaluation_grid=case['metric_evaluation_grid'], densities=densities,
        methods={m: dict(checkpoint=s['checkpoint'], step=s['step'],
            statistics=s['distribution_statistics'], plotted_values=v.tolist(),
            records=s['metric_records']) for m, s, v in zip(methods, rows, values, strict=True)})


def fields_figure(name, case, refs, fields, scores, settings):
    samples = settings['field_samples']
    methods = ('reference', 'nonlocal', 'local_ad')
    fig, axes = plt.subplots(9, 5, figsize=(13.2, 15.7 if name == 'particle_corrosion2d' else 14.4))
    fig.subplots_adjust(left=.12, right=.985, top=.90, bottom=.07, wspace=.11, hspace=.27)
    records = []
    for group, sample in enumerate(samples):
        ref = refs[sample]
        particle = name == 'particle_corrosion2d'
        times = ref['times'] if particle else ref['time']
        indices = time_indices(times, case['field_times'])
        tri = Triangulation(ref['points'][:, 0], ref['points'][:, 1], ref['triangles']) if particle else None
        condition_record = dict(sample_index=sample, sample_identity=case['sample_identities'][sample],
            reference=case['reference_files'][sample], time_indices=indices,
            normalized_times=times[indices].tolist(), physical_seconds=ref['physical_time'][indices].tolist(), methods={})
        for local_row, method in enumerate(methods):
            row = 3*group+local_row
            field = ref['fields'] if method == 'reference' else fields[method][group]
            per_time = []
            label = 'Reference' if method == 'reference' else ('Nonlocal' if method == 'nonlocal' else 'Local AD')
            for column, index in enumerate(indices):
                ax = axes[row, column]
                phi, target = field[index, ..., 0], ref['fields'][index, ..., 0]
                if particle:
                    im = ax.tripcolor(tri, phi, shading='gouraud', cmap='viridis',
                        vmin=settings['phase_limits'][0], vmax=settings['phase_limits'][1], rasterized=True)
                    coords = (tri,)
                    contour = ax.tricontour
                else:
                    im = ax.pcolormesh(ref['X'], ref['Y'], phi, shading='gouraud', cmap='viridis',
                        vmin=settings['phase_limits'][0], vmax=settings['phase_limits'][1], rasterized=True)
                    coords = (ref['X'], ref['Y'])
                    contour = ax.contour
                if phi.min() < .5 < phi.max():
                    contour(*coords, phi, levels=[.5], colors='#17242b', linewidths=.8)
                if method != 'reference':
                    if target.min() < .5 < target.max():
                        contour(*coords, target, levels=[.5], colors='white', linestyles='--', linewidths=.95)
                    error = relative(phi, target, ref['volumes'] if particle else 1.)
                    per_time.append(error)
                    ax.text(.97, .94, f'L2 {100*error:.2f}%', transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color='#17242b', bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=1.5))
                ax.set_facecolor('#a8b2b8')
                ax.set(xlim=(0, 1), ylim=(.6 if particle else .5, 0), aspect='equal')
                ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .6 if particle else .5])
                ax.tick_params(labelsize=7, length=2, pad=2)
                if column == 0:
                    ax.set_ylabel(f'{"Geometry" if particle else "Input"} {sample+1}\n{label}', fontsize=9)
                else:
                    ax.set_yticklabels([])
                if row != 8:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('$x^*$', fontsize=8)
                if row == 0:
                    ax.set_title(f'$t^*={times[index]:g}$\n{ref["physical_time"][index]:g} s', fontsize=10)
                if local_row == 0:
                    ax.spines['top'].set_linewidth(1.5)
            if method != 'reference':
                run = next(r for r in case['runs'] if r['seed'] == settings['field_seed'] and r['method'] == method)
                summary = next(s for s in scores if s['seed'] == settings['field_seed'] and s['method'] == method)
                condition_record['methods'][method] = dict(
                    checkpoint=run['checkpoint'], prediction=(run['prediction_files'][sample] if particle else run['prediction_file']),
                    saved_phi_range=[float(field[..., 0].min()), float(field[..., 0].max())],
                    shown_time_phi_l2=per_time, full_time_metrics=summary['conditions'][sample])
        records.append(condition_record)
    title = 'Ceramic-particle corrosion' if name == 'particle_corrosion2d' else 'Electropolish'
    fig.suptitle(title+': reference / nonlocal / local AD', y=.985, fontsize=16)
    fig.text(.55, .958, 'First three evaluation inputs | paired seed42 | final '+
             ('20,000' if name == 'particle_corrosion2d' else '10,000')+'-step checkpoints', ha='center', fontsize=10)
    fig.text(.55, .939, ('Gray: ceramic | ' if name == 'particle_corrosion2d' else '')+
             'Black: shown field interface | white dashed: local-reference interface', ha='center', fontsize=9)
    bar = fig.add_axes([.33, .037, .44, .010])
    fig.colorbar(im, cax=bar, orientation='horizontal', ticks=[0, .5, 1])
    bar.set_xlabel('Phase $\\phi$: 0 = electrolyte / corroded region, 1 = metal', fontsize=10)
    bar.tick_params(labelsize=8)
    return dict(**save(fig, case['outputs']['fields'], settings), samples=records,
                display='Shared0..1scale, original nodes/triangles, Gouraud display interpolation only; '
                'no field smoothing, temporal interpolation, new mesh or altered numerical arrays')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('all', 'particle_corrosion2d', 'electropolish2d'), required=True)
    parser.add_argument('--figures', choices=('all', 'distribution'), default='all',
                        help='Redraw only distributions to preserve existing field sheets')
    args = parser.parse_args()
    manifest = read(MANIFEST)
    if manifest['schema'] != 'pino_derivative_comparison_figures_v3':
        raise ValueError('Unsupported provenance schema')
    journal = read(manifest['raw_journal'])
    if journal['status'] != 'failed' or journal['device'] != manifest['device']:
        raise ValueError('Raw journal qualification/device changed')
    receipt = read(RECEIPT) if RECEIPT.exists() else dict(cases={})
    receipt['schema'] = 'pino_derivative_comparison_validation_v3'
    receipt.update(provenance=str(MANIFEST.relative_to(REPO)), renderer=str(Path(__file__).relative_to(REPO)),
        versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__, pillow=pillow_version),
        validation='Saved configurations, actual coordinates/times, and seed42 final64case physical-field norms checked; channel-mean discrete space-time L2 recomputed; '
        'strict pair statuses preserved; no new inference, training or reference solve')
    names = list(manifest['cases']) if args.case == 'all' else [args.case]
    for name in names:
        case = manifest['cases'][name]
        refs, gallery, scores = load_case(name, case, manifest['plot'], journal)
        if args.figures == 'distribution' and name not in receipt['cases']:
            raise ValueError('Distribution-only rendering requires the existing field receipt')
        field_record = (fields_figure(name, case, refs, gallery, scores, manifest['plot'])
                        if args.figures == 'all' else receipt['cases'][name]['fields'])
        receipt['cases'][name] = dict(
            distribution=distribution(name, case, scores, manifest['plot']),
            fields=field_record,
            verified_final_metrics=scores, pairing=case['pairs'])
        print('RENDERED_AND_VERIFIED', name, flush=True)
    actual_files = [REPO/path for c in manifest['cases'].values() for path in c['outputs'].values()]
    total = sum(path.stat().st_size for path in actual_files if path.exists())
    if total > manifest['plot']['max_total_bytes']:
        raise ValueError(f'Comparison figures exceed their separate2MiBtotal budget: {total}')
    receipt.update(status='passed', total_existing_image_bytes=total)
    RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('FIGURE_BYTES', total, flush=True)


if __name__ == '__main__':
    main()
