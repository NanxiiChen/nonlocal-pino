"""CPU-only final64 distributions and recorded four-case convergence figures.

Run: PYTHONPATH=/tmp/pino-pdf-audit .venv/bin/python -m
     docs.tasks.ceramic_fem_comparison_figures --output DIR
No inference, training, reference solve, checkpoint selection or curve smoothing.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from fontTools.ttLib import TTFont
import fitz
import numpy as np
import scipy
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs')
RUNS = LOGS / 'particle_corrosion2d'
REFERENCE = Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')
FONT = Path('/root/.local/share/fonts/pino/Helvetica.ttf')
PDFFONTS = Path('/tmp/pino-poppler-audit/usr/bin/pdffonts')
ARMS = (
    ('nonlocal_replay', 'Nonlocal PINO', 'Nonlocal PINO', '#E64B35', '-',
     'ceramic-old-source-pro6000-20261113-01',
     'ceramic-old-source-pro6000-20261113-01/eval64'),
    ('local_ad', 'Local AD PINO', 'Local AD PINO', '#4DBBD5', '--',
     'ceramic-fem-comparison-20261113-02-local_ad',
     'ceramic-fem-comparison-20261113-02/local_ad-eval64'),
    ('fem_weak', 'FEM PINO', 'FEM PINO', '#00A087', '-.',
     'ceramic-fem-comparison-20261113-01-fem_weak',
     'ceramic-fem-comparison-20261113-01/fem_weak-eval64'),
)


def read(path):
    return json.loads(Path(path).read_text())


def style():
    with TTFont(FONT) as font:
        if font['name'].getDebugName(1) != 'Helvetica' or 'glyf' not in font:
            raise ValueError('An actual Helvetica TrueType file is required')
        if font['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError('Font embedding/subsetting permission is required')
    font_manager.fontManager.addfont(str(FONT))
    actual = Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve()
    if actual != FONT.resolve():
        raise ValueError(f'Unexpected font resolution: {actual}')
    return {
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'],
        'font.size': 6, 'axes.labelsize': 7, 'axes.titlesize': 7,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'legend.frameon': False, 'legend.labelcolor': '#222222',
        'text.color': '#222222', 'axes.labelcolor': '#222222',
        'xtick.color': '#222222', 'ytick.color': '#222222',
        'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.minor.width': .5, 'ytick.minor.width': .5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
        'text.usetex': False, 'mathtext.fontset': 'dejavusans',
        'mathtext.fallback': None, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
    }


def load():
    if read(LOGS / 'ceramic-fem-comparison-20261113-02/journal.json')['status'] != 'complete':
        raise ValueError('The three-arm queue must be complete')
    replay = read(LOGS / 'ceramic-old-source-pro6000-20261113-01/journal.json')
    if replay['status'] != 'complete' or not replay['original_inputs_and_cold_state_identical']:
        raise ValueError('The original-source replay must have passed its input audit')
    result = []
    common_monitor = None
    for key, label, tick, color, linestyle, run_name, eval_name in ARMS:
        run, evaluation = RUNS / run_name, LOGS / eval_name
        score = read(evaluation / 'summary.json')
        runtime, config = read(run / 'runtime.json'), read(run / 'config.json')
        if (score['status'] != 'passed' or score['count'] != 64 or score['steps'] != 20000
                or score['checkpoint'] != 'model_final.eqx' or Path(score['reference']) != REFERENCE
                or Path(score['run']) != run or runtime['seed'] != 42
                or runtime['training_uses_reference_targets']):
            raise ValueError(f'Unexpected final-model provenance: {key}')
        if read(evaluation / 'regional_metrics.json')['status'] != 'passed':
            raise ValueError(f'Independent regional replay missing: {key}')
        if common_monitor is None:
            common_monitor = runtime['reference']
        if runtime['reference'] != common_monitor:
            raise ValueError('Convergence cohorts differ')
        records = score['records']
        if [r['index'] for r in records] != list(range(64)):
            raise ValueError('Require all64ordered samples')
        values = []
        for row in records:
            error = np.sqrt(np.asarray(row['numerator']) / np.asarray(row['denominator']))
            np.testing.assert_allclose(error, [row['phi'], row['concentration']], rtol=1e-12, atol=0)
            np.testing.assert_allclose(error.mean(), row['channel_mean'], rtol=1e-12, atol=0)
            values.append(error.tolist() + [float(error.mean())])
        values = np.asarray(values)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Log plots require positive finite relative errors')
        for i, channel in enumerate(('phi', 'concentration', 'channel_mean')):
            np.testing.assert_allclose([values[:, i].mean(), values[:, i].std(ddof=1)],
                [score['statistics'][channel]['mean'], score['statistics'][channel]['std']],
                rtol=1e-12, atol=0)
        history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        steps = [r['step'] for r in history]
        if steps != list(range(0, 20001, 1000)):
            raise ValueError(f'Unexpected actual evaluation steps: {key}')
        monitors = np.array([[[c['phi_pooled'], c['c_pooled']] for c in r['conditions']]
                             for r in history])
        if monitors.shape != (21, 4, 2) or not np.isfinite(monitors).all() or np.any(monitors <= 0):
            raise ValueError('Expected21actual evaluations of4cases/2channels')
        np.testing.assert_allclose(monitors.mean(axis=1),
            [[r['phi_pooled'], r['c_pooled']] for r in history], rtol=1e-12, atol=0)
        result.append(dict(key=key, label=label, tick=tick, color=color, linestyle=linestyle,
            run=str(run), configuration=str(run / 'config.json'), saved_configuration=config,
            checkpoint=str(run / 'model_final.eqx'), seed=runtime['seed'],
            evaluation=str(evaluation / 'summary.json'), monitor=str(run / 'metrics.jsonl'),
            monitor_reference=common_monitor, final_values=values.tolist(),
            final_statistics=score['statistics'], steps=steps,
            monitor_case_values=monitors.tolist(), monitor_mean=monitors.mean(axis=1).tolist(),
            monitor_sd=monitors.std(axis=1, ddof=1).tolist(),
            training_seconds=read(run / 'summary.json')['total_seconds']))
    return result


def log_axis(ax):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='major', length=2.5)
    ax.tick_params(which='minor', length=1.5)


def save(fig, path, size_mm):
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            text.set_fontproperties(font_manager.FontProperties(fname=str(FONT), size=text.get_fontsize()))
            text.set_color('#222222')
    fig.savefig(path.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(path.with_suffix('.png'), dpi=300)
    axes = [dict(autoscale_y=ax.get_autoscaley_on(), y_limits=list(ax.get_ylim())) for ax in fig.axes]
    if not all(a['autoscale_y'] for a in axes):
        raise ValueError('Manual y limits are forbidden')
    plt.close(fig)
    pdf = path.with_suffix('.pdf')
    with fitz.open(pdf) as doc:
        page = doc[0]
        np.testing.assert_allclose([page.rect.width, page.rect.height], np.array(size_mm)*72/25.4,
                                   rtol=0, atol=1e-4)
        spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0
                 for line in b['lines'] for s in line['spans']]
        if not spans or any(not 4.99 <= s['size'] <= 7.01 for s in spans):
            raise ValueError('PDF text must remain5–7pt')
        if page.get_images():
            raise ValueError('Statistical figures must remain vector graphics')
        for span in spans:
            if not page.rect.contains(fitz.Rect(span['bbox'])):
                raise ValueError(f'Clipped PDF text: {span}')
        fonts = page.get_fonts(full=True)
        for font in fonts:
            if font[2] not in ('TrueType', 'Type0') or not doc.extract_font(font[0])[3]:
                raise ValueError(f'Non-embedded/incorrect font: {font}')
        font_report = subprocess.check_output([str(PDFFONTS), str(pdf)], text=True)
        # Edit two labels in a separate throwaway copy, preserving vector/text content.
        edit = fitz.open(stream=doc.tobytes(), filetype='pdf')
        p = edit[0]
        for text, replacement in [('Rel. L² error', 'Rel. L² score'),
                                  ('Weighted rel. L² error', 'Weighted rel. L² score'),
                                  ('Local AD', 'Local FE')]:
            matches = p.search_for(text)
            if matches:
                p.add_redact_annot(matches[0], fill=(1, 1, 1))
                p.apply_redactions(images=0, graphics=0)
                p.insert_text((5, 8 if text != 'Local AD' else 16), replacement,
                              fontsize=7, fontname='plot_helvetica', fontfile=str(FONT))
        edited_text = edit[0].get_text()
        if 'L² score' not in edited_text or 'Local FE' not in edited_text:
            raise ValueError('Independent PDF text-edit probe failed')
        edit.close()
    return dict(pdf=str(pdf), png=str(path.with_suffix('.png')), size_mm=size_mm,
                bytes={suffix:path.with_suffix(suffix).stat().st_size for suffix in ('.pdf', '.png')},
                axes=axes, text_spans=spans, pdffonts=font_report,
                text_edit_probe='Programmatic PyMuPDF temporary-copy label edits passed; no GUI editor claim.')


def distribution(arms, output):
    size = [100, 64]
    fig, ax = plt.subplots(figsize=np.array(size)/25.4)
    fig.subplots_adjust(left=.17, right=.98, bottom=.21, top=.96)
    kde_records = []
    for i, arm in enumerate(arms):
        values = np.array(arm['final_values'])[:, 2]
        kde = gaussian_kde(np.log(values), bw_method='scott')
        bw = float(np.sqrt(kde.covariance[0, 0]))
        support = np.linspace(np.log(values).min()-3*bw, np.log(values).max()+3*bw, 400)
        density = kde(support)
        width = .38*density/density.max()
        ax.fill_betweenx(np.exp(support), i-width, i+width, facecolor=arm['color'],
                         edgecolor='#555555', linewidth=.6, alpha=.82)
        quantiles = np.quantile(values, [.25, .5, .75])
        for q, dash in zip(quantiles, [(1.5, 1.5), (3, 1.5), (1.5, 1.5)], strict=True):
            w = np.interp(np.log(q), support, width)
            ax.plot([i-w, i+w], [q, q], color='#333333', linewidth=.6, linestyle=(0, dash))
        kde_records.append(dict(key=arm['key'], log_bandwidth=bw, quartiles=quantiles.tolist()))
    log_axis(ax)
    ax.set_xticks(range(len(arms)), [a['tick'] for a in arms])
    ax.set_xlim(-.55, len(arms)-.45)
    ax.set_ylabel('Rel. L² error')
    fig.text(.57, .045, 'FEM numerical reference', ha='center', fontsize=6)
    return dict(**save(fig, output/'particle_corrosion2d_fem_comparison_distribution', size),
                density_space='natural logarithm of error', bandwidth='scott', tail_bandwidths=3,
                density_normalization='equal maximum width; all64observations per arm',
                kde=kde_records)


def convergence(arms, output):
    size = [170, 68]
    fig, axes = plt.subplots(1, 2, figsize=np.array(size)/25.4)
    fig.subplots_adjust(left=.092, right=.985, bottom=.20, top=.78, wspace=.30)
    for j, ax in enumerate(axes):
        for arm in arms:
            mean, sd = np.array(arm['monitor_mean'])[:, j], np.array(arm['monitor_sd'])[:, j]
            if np.any(mean-sd <= 0):
                raise ValueError('Cannot display nonpositive mean-minus-SD on log axis')
            ax.fill_between(arm['steps'], mean-sd, mean+sd, color=arm['color'], alpha=.10, linewidth=0)
            ax.plot(arm['steps'], mean, color=arm['color'], linestyle=arm['linestyle'],
                    linewidth=1, label=arm['label'])
        log_axis(ax)
        ax.set_xticks([0, 5000, 10000, 15000, 20000], ['0', '5k', '10k', '15k', '20k'])
        ax.set_xlabel('Training step')
        ax.set_ylabel('Weighted rel. L² error')
        ax.set_title(r'Phase $\phi$' if j == 0 else r'Concentration $c$', pad=5)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=len(arms), loc='upper center',
               bbox_to_anchor=(.54, .995), columnspacing=1.7, handlelength=2.6)
    fig.text(.54, .035, 'FEM numerical reference', ha='center', fontsize=6)
    return save(fig, output/'particle_corrosion2d_fem_comparison_convergence', size)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    arms = load()
    with matplotlib.rc_context(style()):
        figures = dict(distribution=distribution(arms, output), convergence=convergence(arms, output))
    reference = read(REFERENCE/'manifest.json')
    with np.load(REFERENCE/reference['cases'][0]['file'], allow_pickle=False) as first:
        saved_times = first['times'].tolist()
        physical_times = first['physical_time'].tolist()
    receipt = dict(status='passed', renderer=str(Path(__file__).resolve()),
        command=f'PYTHONPATH=/tmp/pino-pdf-audit .venv/bin/python -m '
                f'docs.tasks.ceramic_fem_comparison_figures --output {output}',
        pdf_audit_dependency=str(Path(fitz.__file__).resolve()),
        pdffonts=dict(binary=str(PDFFONTS), package_source=
            'https://archive.ubuntu.com/ubuntu/pool/main/p/poppler/poppler-utils_22.02.0-2_amd64.deb'),
        fonts=dict(helvetica=str(FONT),
                   source='https://raw.githubusercontent.com/NanxiiChen/PI-CViT/main/helvetica/Helvetica.ttf',
                   math_fontset='dejavusans', math_font_directory=str(Path(matplotlib.get_data_path())/'fonts/ttf')),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, matplotlib=matplotlib.__version__, pymupdf=fitz.VersionBind),
        reference=dict(manifest=str(REFERENCE/'manifest.json'), backend=reference['backend'],
                       spacing=reference['actual_spacing'], boundary_points=reference['actual_boundary_points'],
                       saved_times=saved_times, physical_times_seconds=physical_times, channels=['phi', 'c'],
                       sample_ids=list(range(64)), files=[str(REFERENCE/r['file']) for r in reference['cases']]),
        final_metric='Per-sample/channel Euclidean norm over all space-time nodes; then mean(phi,c).',
        convergence_metric='Recorded4case full-space-time volume/trapezoidal-time weighted relative L2; '
                           'mean and sample SD per physical channel; no smoothing or invented evaluations.',
        qualification='Displayed Nonlocal PINO uses historical source AND launcher. '
                      'First failed run omitted from these figures at user request; '
                      'its original data and four-arm figures remain preserved. '
                      'One seed42initialization; this is not a source-only causal ablation or multi-seed ranking.',
        figures=figures, arms=arms, training_performed=False, inference_performed=False)
    (output/'provenance.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=receipt['status'], output=str(output),
                         means={a['key']:a['final_statistics']['channel_mean']['mean'] for a in arms}), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
