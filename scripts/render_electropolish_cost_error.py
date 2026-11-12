"""Plot audited final-checkpoint cost/error sweeps without new model inference."""
import argparse
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from scripts.render_pino_derivative_comparisons import publication_style

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def common_config(config):
    result = copy.deepcopy(config)
    for key in ('Nx', 'Ny'):
        del result['sample'][key]
    for key in ('max_nodes', 'max_edges'):
        del result['sample']['spatial'][key]
    del result['nl']['delta']
    for key in ('run_name', 'log_dir'):
        del result['log'][key]
    return result


def collect(spec):
    records, duplicates = {}, []
    shared_config = shared_evaluation = None
    for study in spec['studies']:
        report = read(study['report'])
        if report['status'] != 'passed':
            raise ValueError(f"Study has not passed its audit: {study['report']}")
        journal = [json.loads(line) for line in Path(study['journal']).read_text().splitlines()]
        for arm in report['records'].values():
            run = Path(arm['run']).resolve(strict=True)
            config = read(run / 'config.json')
            evaluation = read(run / 'evaluation64.json')
            if (evaluation['steps'], evaluation['count'], evaluation['ddof']) != (10000, 64, 1):
                raise ValueError(f'Expected final10k/64-input sample-SD evaluation: {run}')
            if Path(evaluation['checkpoint']) != run / 'model_final.eqx':
                raise ValueError(f'Expected final weights: {run}')
            identity = dict(metric=evaluation['metric'], time_indices=evaluation['time_indices'],
                            references=[(r['index'], r['surface_id'], r['reference'])
                                        for r in evaluation['records']])
            if shared_config is None:
                shared_config, shared_evaluation = common_config(config), identity
            if common_config(config) != shared_config or identity != shared_evaluation:
                raise ValueError(f'Unmatched training scheme or evaluation cohort: {run}')
            channels = np.array([[r['phi'], r['c']] for r in evaluation['records']])
            if channels.shape != (64, 2) or not np.all(np.isfinite(channels)):
                raise ValueError(f'Invalid physical-channel errors: {run}')
            scores = channels.mean(axis=1)
            np.testing.assert_allclose(scores, arm['scores'], rtol=1e-13)
            mean, sd = float(scores.mean()), float(scores.std(ddof=1))
            expected = arm['statistics']['channel_mean']
            np.testing.assert_allclose([mean, sd], [expected['mean'], expected['std']], rtol=1e-13)
            if mean <= 0:
                raise ValueError(f'Nonpositive mean error cannot use a log axis: {run}')
            count = config['sample']['Nx'] * config['sample']['Ny']
            delta = config['nl']['delta']
            if count != arm['audit']['particles'] or delta != arm['audit']['delta']:
                raise ValueError(f'Particle/horizon audit mismatch: {run}')
            if str(run) in records:
                prior = records[str(run)]
                if prior['particles'] != count or prior['delta'] != delta or prior['scores'] != scores.tolist():
                    raise ValueError(f'Duplicated run has conflicting values: {run}')
                duplicates.append(dict(run=str(run), report=study['report'], arm=arm['name']))
                continue
            label = arm['name'] + '-train'
            starts = [e for e in journal if e['event'] == 'start' and e['label'] == label]
            ends = [e for e in journal if e['event'] == 'end' and e['label'] == label]
            if len(starts) != 1 or len(ends) != 1 or ends[0]['exit_code'] != 0:
                raise ValueError(f'Expected one successful cold-training process: {label}')
            seconds = ends[0]['seconds']
            if not np.isfinite(seconds) or seconds <= arm['trainer_seconds']:
                raise ValueError(f'Invalid cold-process duration: {run}')
            records[str(run)] = dict(run=str(run), config=str(run / 'config.json'),
                checkpoint=evaluation['checkpoint'], evaluation=str(run / 'evaluation64.json'),
                report=study['report'], journal=study['journal'], journal_label=label,
                particles=count, delta=delta, ratio=arm['audit']['delta_over_h'],
                cold_process_seconds=seconds, trainer_seconds=arm['trainer_seconds'],
                warm_update_seconds=arm['warm_update_seconds'], mean=mean, sample_sd=sd,
                scores=scores.tolist(), channels=channels.tolist())
    if len(records) != spec['expected_unique_runs'] or len(duplicates) != spec['expected_duplicates']:
        raise ValueError('Unexpected unique/duplicate run count')
    rows = sorted(records.values(), key=lambda r: (r['particles'], r['delta']))
    if sorted({r['particles'] for r in rows}) != spec['particle_counts']:
        raise ValueError('Particle groups differ from the explicit figure specification')
    return rows, duplicates, shared_evaluation


def render(spec):
    rows, duplicates, evaluation = collect(spec)
    if any(row['mean'] <= row['sample_sd'] for row in rows):
        raise ValueError('Nonpositive mean-minus-SD cannot use log shading')
    style = spec['style']
    rc, font = publication_style(style['typography'])
    area = lambda delta: style['marker_area_pt2_at_reference'] * delta / style['marker_reference_delta']
    with plt.rc_context(rc):
        fig = plt.figure(figsize=np.array(style['figure_mm']) / 25.4)
        ax = fig.add_axes(style['axes_rect'])
        ax.set_xscale(style['xscale']); ax.set_yscale(style['yscale'])
        groups = []
        for count in spec['particle_counts']:
            group = [r for r in rows if r['particles'] == count]
            x = np.array([r['cold_process_seconds'] for r in group])
            y = np.array([r['mean'] for r in group])
            sd = np.array([r['sample_sd'] for r in group])
            sizes = np.array([area(r['delta']) for r in group])
            color = style['colors'][str(count)]
            if len(group) > 1:
                # Each quadrilateral joins adjacent delta settings. Actual clock
                # values are retained even if two timings differ nonmonotonically.
                for i in range(len(group) - 1):
                    sl = slice(i, i + 2)
                    ax.fill_between(x[sl], (y-sd)[sl], (y+sd)[sl], color=color,
                                    alpha=style['band_alpha'], linewidth=0, zorder=1)
                ax.plot(x, y, color=color, linewidth=style['line_width_pt'], zorder=2)
            else:
                ax.errorbar(x, y, yerr=sd, fmt='none', ecolor=color,
                            elinewidth=style['line_width_pt'], capsize=2, zorder=2)
            ax.scatter(x, y, s=sizes, color=color, edgecolor='white',
                       linewidth=style['marker_edge_width_pt'], zorder=3)
            groups.append(dict(particles=count, run_order=[r['run'] for r in group],
                               x=x.tolist(), mean=y.tolist(), sample_sd=sd.tolist(),
                               marker_areas_pt2=sizes.tolist(), color=color))
        ax.set_xlabel(style['xlabel']); ax.set_ylabel(style['ylabel'])
        ax.xaxis.set_major_locator(FixedLocator(style['xticks']))
        ax.yaxis.set_major_locator(FixedLocator(style['yticks']))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='major', length=3, pad=3)
        ax.tick_params(which='minor', length=1.5)
        particle_handles = [Line2D([], [], color=style['colors'][str(n)], linewidth=1.2,
                                   label=f'{n:,}') for n in spec['particle_counts']]
        fig.legend(handles=particle_handles, title='Particles', frameon=False,
                   loc='upper left', bbox_to_anchor=style['particle_legend_anchor'],
                   borderaxespad=0, labelspacing=.8, handlelength=2)
        size_handles = [ax.scatter([], [], s=area(delta), color=style['size_legend_color'],
                                  edgecolor='white', linewidth=style['marker_edge_width_pt'],
                                  label=f'{delta:g}') for delta in style['legend_deltas']]
        fig.legend(handles=size_handles, title=r'Horizon $\delta$', frameon=False,
                   loc='upper left', bbox_to_anchor=style['size_legend_anchor'],
                   borderaxespad=0, labelspacing=1.0, handletextpad=1.0)
        automatic_limits = dict(x=list(ax.get_xlim()), y=list(ax.get_ylim()))
        for kind in ('pdf', 'png'):
            output = Path(spec['output'][kind]); output.parent.mkdir(parents=True, exist_ok=True)
            kwargs = dict(dpi=style['dpi'])
            if kind == 'pdf':
                kwargs['metadata'] = {'CreationDate': None, 'ModDate': None}
            fig.savefig(output, **kwargs)
        plt.close(fig)
    sys.path.insert(0, str(Path(spec['pdf_audit_packages']).resolve(strict=True)))
    import fitz
    with fitz.open(spec['output']['pdf']) as doc:
        page = doc[0]
        np.testing.assert_allclose([page.rect.width*25.4/72, page.rect.height*25.4/72],
                                   style['figure_mm'], atol=.001)
        spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0
                 for line in b['lines'] for s in line['spans']]
        if not spans or any(not 4.99 <= s['size'] <= 7.01 or
                            not page.rect.contains(fitz.Rect(s['bbox'])) for s in spans):
            raise ValueError('PDF text must be editable, in-page and5–7pt')
        fonts = page.get_fonts(full=True)
        if page.get_images() or any(f[2] == 'Type3' or not doc.extract_font(f[0])[3] for f in fonts):
            raise ValueError('Expected vector-only PDF with embedded TrueType text')
    receipt = dict(specification=spec, renderer=str(Path(__file__).relative_to(REPO)),
        rows=rows, omitted_duplicate=duplicates, evaluation=evaluation, groups=groups,
        automatic_limits=automatic_limits, font=font, pdf_fonts=fonts,
        font_size_range_pt=[min(s['size'] for s in spans), max(s['size'] for s in spans)],
        output_bytes={kind:Path(spec['output'][kind]).stat().st_size for kind in ('pdf','png')},
        visual_review='pending', training=False, inference=False, reference_solves=0)
    (REPO / spec['output']['receipt']).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'unique_runs':len(rows), 'groups':len(groups), 'output':spec['output']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    render(read(parser.parse_args().spec))
