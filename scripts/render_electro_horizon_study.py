"""Independently audit and plot final64local-reference horizon/density controls."""
import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

from scripts.render_pino_derivative_comparisons import publication_style

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def fixed_config(c):
    c = copy.deepcopy(c)
    for key in ('Nx', 'Ny'):
        del c['sample'][key]
    for key in ('max_nodes', 'max_edges'):
        del c['sample']['spatial'][key]
    del c['nl']['delta']
    del c['log']['run_name']
    return c


def collect(root, normalize_config):
    manifest = read(root/'manifest.json')
    baseline = Path(manifest['arms'][0]['run'])
    shared = normalize_config(read(baseline/'config.json'))
    init = (baseline/'model_initial.eqx').read_bytes()
    optimizer = (baseline/'optimizer_initial.eqx').read_bytes()
    draws = np.load(baseline/'online_coefficients.npy')
    with np.load(baseline/'sampling.npz') as data:
        times = data['times']
    records = {}
    for arm in manifest['arms']:
        run = Path(arm['run'])
        if normalize_config(read(run/'config.json')) != shared:
            raise ValueError('Undeclared configuration difference: '+arm['name'])
        if ((run/'model_initial.eqx').read_bytes() != init
                or (run/'optimizer_initial.eqx').read_bytes() != optimizer):
            raise ValueError('Cold model/optimizer state mismatch: '+arm['name'])
        np.testing.assert_array_equal(np.load(run/'online_coefficients.npy'), draws)
        with np.load(run/'sampling.npz') as data:
            np.testing.assert_array_equal(data['times'], times)
        evaluation = read(run/'evaluation64.json')
        if evaluation['steps'] != 10000 or evaluation['count'] != 64:
            raise ValueError('Require final10000weights and64cases')
        with np.load(run/'evaluation64.npz') as data:
            predictions = data['prediction']
            indices = evaluation['time_indices']
            for row, p in zip(evaluation['records'], predictions, strict=True):
                i = row['index']
                if row['surface_id'] != int(data['surface_ids'][i]):
                    raise ValueError('Surface identity mismatch')
                with np.load(row['reference']) as ref:
                    np.testing.assert_array_equal(ref['time'][indices], data['time'])
                    r = ref['fields'][indices].astype(np.float64)
                p = p.astype(np.float64)
                values = [np.linalg.norm((p[..., k]-r[..., k]).ravel())/
                          np.linalg.norm(r[..., k].ravel()) for k in range(2)]
                np.testing.assert_allclose(values, [row['phi'], row['c']], rtol=2e-12, atol=1e-14)
                np.testing.assert_allclose(np.mean(values), row['channel_mean'], rtol=2e-12, atol=1e-14)
        statistics = {}
        for name in ('channel_mean', 'phi', 'c'):
            values = np.array([r[name] for r in evaluation['records']])
            actual = dict(mean=float(values.mean()), std=float(values.std(ddof=1)))
            for key in actual:
                np.testing.assert_allclose(actual[key], evaluation['statistics'][name][key], rtol=1e-13)
            statistics[name] = actual
        timing = read(run/'timing.json')
        records[arm['name']] = dict(**arm, statistics=statistics,
            warm_update_seconds=timing['warm_update_mean_seconds'],
            trainer_seconds=timing['trainer_seconds'],
            scores=[r['channel_mean'] for r in evaluation['records']])
    return manifest, records


def render(root):
    manifest, records = collect(root, fixed_config)
    settings = read(REPO/'artifacts/pino_derivative_comparisons.json')['plot']['distribution']['publication']
    rc, font = publication_style(settings)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(170/25.4, 59/25.4))
        fig.subplots_adjust(left=.072, right=.989, bottom=.225, top=.91, wspace=.36)
        definitions = [('fixed_h', 'delta', r'$\delta$', r'$h=0.01$', '#E64B35'),
                       ('fixed_delta', 'particles', 'Particles', r'$\delta=0.02$', '#4DBBD5'),
                       ('fixed_ratio', 'particles', 'Particles', r'$\delta/h=2$', '#00A087')]
        plotted = []
        for ax, (group, key, label, title, color) in zip(axes, definitions, strict=True):
            rows = sorted([records[name] for name in manifest['groups'][group]], key=lambda r: r['audit'][key])
            x = np.array([r['audit'][key] for r in rows])
            mean = np.array([r['statistics']['channel_mean']['mean'] for r in rows])
            sd = np.array([r['statistics']['channel_mean']['std'] for r in rows])
            ax.plot(x, mean, '-o', color=color, lw=.8, markersize=2.5)
            ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.22, linewidth=0)
            ax.set_xlabel(label); ax.set_title(title)
            ax.spines[['top', 'right']].set_visible(False)
            if key == 'particles':
                ax.set_xticks(x)
                ax.set_xticklabels([f'{n/1000:.2f}k' for n in x], rotation=30, ha='right')
            else:
                ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in x])
            ax.tick_params(length=2, pad=2)
            plotted.append(dict(group=group, arms=[r['name'] for r in rows], x=x.tolist(),
                                mean=mean.tolist(), std=sd.tolist(), ylim_auto=list(ax.get_ylim())))
        axes[0].set_ylabel(settings['label'])
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text() and not 5 <= item.get_fontsize() <= 7:
                raise ValueError('Publication font outside5–7pt')
        output = root/'figures'; output.mkdir(exist_ok=True)
        pdf, png = (output/f'electropolish_horizon_density.{suffix}' for suffix in ('pdf', 'png'))
        fig.savefig(pdf, dpi=300)
        fig.savefig(png, dpi=300)
        plt.close(fig)
    # Existing isolated PDF-audit installation, explicitly recorded per study.
    # Do not add a dependency to the JAX training environment or silently skip.
    sys.path.insert(0, str(Path(manifest['pdf_audit_packages']).resolve(strict=True)))
    import fitz
    with fitz.open(pdf) as doc:
        if len(doc) != 1 or abs(doc[0].rect.width*25.4/72-170) > .01:
            raise ValueError('Unexpected PDF page size')
        text_spans = [span for block in doc[0].get_text('dict')['blocks'] if 'lines' in block
                      for line in block['lines'] for span in line['spans']]
        if not text_spans or any(not 4.999 <= s['size'] <= 7.001 for s in text_spans):
            raise ValueError('PDF text must remain editable and within5–7pt')
        pdf_fonts = doc[0].get_fonts(full=True)
        for entry in pdf_fonts:
            if entry[2] == 'Type3' or not doc.extract_font(entry[0])[3]:
                raise ValueError('PDF fonts must be embedded, not Type3')
    receipt = dict(status='passed', manifest=str(root/'manifest.json'), records=records,
        cold_model_optimizer_and_physical_draws='exactly equal',
        caveat='Query grid/IC nodes and FP32 execution shapes differ; this is one initialization, not a seed study',
        norm_replay='all576sample scores independently recomputed from saved fields',
        spread='sample standard deviation ddof=1, not confidence interval',
        figure_mm=[170,59], font=font, pdf_fonts=pdf_fonts,
        plotted=plotted, outputs=[str(pdf), str(png)])
    (root/'report.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('AUDIT_AND_PLOT_COMPLETE', pdf, png, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    render(parser.parse_args().root)
