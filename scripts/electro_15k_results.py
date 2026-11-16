"""Collect completed Electropolish15k controls without selecting checkpoints.

Library and temporal panels have separate completion gates. Training, inference
and reference generation are deliberately outside this CPU-only entry point.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.pino_comparison_data import history, read, spec
from scripts.time_fusion_wall_clock import birth, recover, rows

STEPS = list(range(0, 15001, 250))
COUNTS = (32, 64, 128, 256, 512)


def stage(journal, name):
    matches = [s for s in journal['stages'] if s['name'] == name]
    if len(matches) != 1 or matches[0]['status'] != 'passed' or matches[0]['exit_code'] != 0:
        raise ValueError('Require the completed declared cold process: ' + name)
    return matches[0]


def inputs(bundle):
    current = read(bundle / 'journal.json')
    previous_path = Path(current['dependency']) / 'journal.json'
    previous = read(previous_path)
    if (previous['status'] != 'complete' or current['iterations'] != 15000
            or previous['iterations'] != 15000 or current['gpu'] != previous['gpu']
            or current['xla_flags'] != previous['xla_flags']
            or current['concurrent_training'] or previous['concurrent_training']):
        raise ValueError('Require completed baseline and matched serial15k runtime controls')
    return current, previous, previous_path


def endpoint(run, audit_name):
    report = read(run / audit_name)
    if report['status'] != 'passed' or report['steps'] != 15000:
        raise ValueError('Only passed final15k audits can enter the comparison')
    final = report['final'] if audit_name == 'suite15k_audit.json' else report
    values = np.asarray(final['channel_errors'], float)
    if final['sample_count'] != 64 or values.shape != (64, 2) or not np.isfinite(values).all():
        raise ValueError('Complete finite64case/two-channel final errors required')
    np.testing.assert_allclose([values.mean(), values.mean(1).std(ddof=1)],
                              [final['mean'], final['sample_std']], rtol=2e-13)
    np.testing.assert_allclose(values.mean(1), final['sample_errors'], rtol=2e-13)
    curves = history(run / 'local_metrics.jsonl', STEPS, list(range(64)), ('phi', 'c'), ('phi', 'c'))
    np.testing.assert_allclose(curves[-1], values, rtol=3e-6, atol=2e-8)
    return dict(run=str(run), configuration=str(run / 'config.json'),
                checkpoint=str((run / 'model_final.eqx').resolve(strict=True)),
                audit=str(run / audit_name), budget=15000, sample_count=64,
                channel_errors=values.tolist(), sample_errors=values.mean(1).tolist(),
                channel_means=values.mean(0).tolist(), mean=float(values.mean()),
                sample_std=float(values.mean(1).std(ddof=1)), reference=final['reference'])


def library_data(bundle):
    current, previous, previous_path = inputs(bundle)
    fixed = []
    for count in COUNTS:
        old = count == 32
        journal = previous if old else current
        run = Path(journal['runs'][f'fixed{count}'])
        selected = endpoint(run, 'library_extension_audit.json' if old else 'suite15k_audit.json')
        c = read(run / 'config.json')
        if c['operator']['condition_library']['count'] != count:
            raise ValueError('The selected run has a different fixed library size')
        selected.update(count=count, cold_process_seconds=stage(journal, f'train-fixed{count}')['seconds'])
        fixed.append(selected)
    online = endpoint(Path(previous['runs']['online']), 'library_extension_audit.json')
    online['cold_process_seconds'] = stage(previous, 'train-online')['seconds']
    identity = read(Path(online['run']) / 'local_monitor.json')
    for item in fixed:
        if read(Path(item['run']) / 'local_monitor.json') != identity or item['reference'] != online['reference']:
            raise ValueError('Fixed and online references/test identities differ')
    return dict(schema='electropolish_library15k_v1', status='passed', example='electropolish2d',
                budget=15000, seed=42, test_ids=list(range(64)), counts=list(COUNTS), fixed=fixed, online=online,
                journals=[str(previous_path), str(bundle / 'journal.json')], gpu=current['gpu'],
                metric='Equal-point full-space-time relativeL2 per physical channel, then arithmetic channel/sample mean',
                uncertainty='One sample standard deviation of channel-mean errors over64test inputs; not a confidence interval or cross-seed variability',
                limitation='One initialization and one nested library; no monotonicity or online-optimality guarantee',
                checkpoint_selection='Final15000update weights only; no best-checkpoint selection')


def library_figure(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    import fitz
    from docs.tasks.ceramic_fem_comparison_figures import style, log_axis, FONT, PDFFONTS
    output.mkdir(parents=True, exist_ok=False)
    size = [85, 62]
    counts = np.asarray(data['counts'])
    means = np.array([r['mean'] for r in data['fixed']])
    spread = np.array([r['sample_std'] for r in data['fixed']])
    online = data['online']
    if np.any(means - spread <= 0) or online['mean'] - online['sample_std'] <= 0:
        raise ValueError('Log-scale SD bands must remain positive; do not silently truncate')
    base = output / 'electropolish2d_library_size_error'
    with matplotlib.rc_context(style()):
        fig, ax = plt.subplots(figsize=np.asarray(size) / 25.4)
        fig.subplots_adjust(left=.20, right=.97, bottom=.20, top=.91)
        ax.plot(counts, means, 'o-', color='#E64B35', lw=1.25, ms=4, mec='white', mew=.7, label='Fixed library')
        ax.fill_between(counts, means-spread, means+spread, color='#E64B35', alpha=.18, lw=0)
        ax.axhline(online['mean'], color='#4DBBD5', lw=1.25, label='Online sampling')
        ax.axhspan(online['mean']-online['sample_std'], online['mean']+online['sample_std'],
                   color='#4DBBD5', alpha=.18, lw=0)
        ax.set_xscale('log', base=2)
        ax.set_xticks(counts, [str(x) for x in counts])
        log_axis(ax)
        ax.set_xlabel('Fixed initial-condition count')
        ax.set_ylabel('Rel. L² error')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper right', handlelength=2)
        fig.canvas.draw()
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                text.set_fontproperties(FontProperties(fname=str(FONT), size=text.get_fontsize()))
                text.set_color('#222222')
        fig.savefig(base.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(base.with_suffix('.png'), dpi=300)
        limits = list(ax.get_ylim())
        if not ax.get_autoscaley_on():
            raise ValueError('Automatic y limits are required')
        plt.close(fig)
    with fitz.open(base.with_suffix('.pdf')) as doc:
        page = doc[0]
        np.testing.assert_allclose([page.rect.width, page.rect.height], np.asarray(size)*72/25.4, atol=1e-4)
        spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0 for line in b['lines'] for s in line['spans']]
        if (not spans or page.get_images() or any(not 4.99 <= s['size'] <= 7.01 for s in spans)
                or any(not page.rect.contains(fitz.Rect(s['bbox'])) for s in spans)):
            raise ValueError('Require editable unclipped5–7pt vector figure')
        for font in page.get_fonts(full=True):
            if font[2] not in ('TrueType', 'Type0') or not doc.extract_font(font[0])[3]:
                raise ValueError('Require embedded TrueType fonts')
        edit = fitz.open(stream=doc.tobytes(), filetype='pdf')
        for i, (old, new) in enumerate((('Rel. L² error', 'Rel. L² score'), ('Fixed library', 'Fixed inputs'))):
            rectangles = edit[0].search_for(old)
            if not rectangles:
                raise ValueError('Editable label missing: ' + old)
            edit[0].add_redact_annot(rectangles[0], fill=(1, 1, 1))
            edit[0].apply_redactions(images=0, graphics=0)
            edit[0].insert_text((5, 8+i*9), new, fontsize=7, fontname='plot_helvetica', fontfile=str(FONT))
        if 'L² score' not in edit[0].get_text() or 'Fixed inputs' not in edit[0].get_text():
            raise ValueError('PDF edit probe failed')
        edit.close()
    return dict(pdf=str(base.with_suffix('.pdf')), png=str(base.with_suffix('.png')), size_mm=size,
                bytes={k: base.with_suffix('.'+k).stat().st_size for k in ('pdf', 'png')},
                y_limits=limits, automatic_y_limits=True, font=str(FONT),
                pdffonts=subprocess.check_output([str(PDFFONTS), str(base.with_suffix('.pdf'))], text=True),
                text_edit_probe='Temporary-copy label replacements passed; no GUI editor claim')


def fusion_data(bundle):
    from docs.tasks.electro_15k_suite import field_audit
    from docs.tasks.pino_time_fusion_support import timing
    current, previous, previous_path = inputs(bundle)
    if current['status'] != 'complete':
        raise ValueError('The full declared suite must finish before temporal publication')
    run_joint = Path(current['runs']['joint_query'])
    joint_audit = read(run_joint / 'suite15k_audit.json')
    if joint_audit['status'] != 'passed':
        raise ValueError('Joint-query pairing audit must pass')
    arms, norms, clocks = [], [], []
    for method, run, journal, journal_path, name, audit in (
            ('decoder_film', Path(previous['runs']['online']), previous, previous_path,
             'train-online', 'library_extension_audit.json'),
            ('joint_query', run_joint, current, bundle/'journal.json',
             'train-joint_query', 'suite15k_audit.json')):
        final = endpoint(run, audit)
        independent = field_audit(run, 15000)
        np.testing.assert_allclose(final['channel_errors'], independent['channel_errors'], rtol=2e-13)
        norms.extend(dict(run=str(run), **row) for row in independent['norms'])
        selected_stage = stage(journal, name)
        if '--config=' + journal['configs']['online' if method == 'decoder_film' else 'joint_query'] not in selected_stage['args']:
            raise ValueError('Cold timing stage belongs to another run')
        arms.append(dict(method=method, run=str(run), configuration=final['configuration'],
                         checkpoint=final['checkpoint'], monitor=str(run/'local_metrics.jsonl'),
                         monitor_errors=history(run/'local_metrics.jsonl', STEPS, list(range(64)), ('phi','c'), ('phi','c')),
                         final_errors=final['channel_errors'],
                         timing=dict(**timing('electropolish', run), cold_process_seconds=selected_stage['seconds'], gpu=current['gpu'])))
        log, metric = journal_path.parent/(name+'.log'), run/'metrics.jsonl'
        actual, local = rows(metric), rows(run/'local_metrics.jsonl')
        if [r['step'] for r in actual] != STEPS:
            raise ValueError('Every actual timed record is required')
        launched, created = birth(log), birth(metric)
        evidence = dict(launch_epoch=launched['epoch_seconds'], metric_created_epoch=created['epoch_seconds'],
                        metric_written_epoch=metric.stat().st_mtime, last_counter=actual[-1]['seconds'],
                        cold_total=selected_stage['seconds'], tolerance_seconds=.1)
        counters, durations = [r['seconds'] for r in actual], [r['evaluation_seconds'] for r in local]
        clocks.append(dict(method=method, run=str(run), parent_journal=str(journal_path), parent_stage=name,
                           launch_log=str(log), metric_file=str(metric), monitor=str(run/'local_metrics.jsonl'),
                           launch_log_birth=launched, metric_birth=created, clock_evidence=evidence,
                           trainer_counters=counters, following_local_evaluation_seconds=durations,
                           **recover(counters, durations, **evidence)))
    data = spec('electropolish2d', [('phi','Phase'), ('c','Concentration')], 15000, STEPS,
                list(range(64)), list(range(64)), arms, final['reference'],
                'All64local-FD test cases at every record;21uniform times and full101x51grid, equal-point physical channel norms',
                'whole_trajectory', .22)
    data.update(comparison_kind='time_fusion', pairing=joint_audit['pairing'], saved_field_audit=norms,
                independent_local_single_case=[], source_reports=[str(previous_path), str(bundle/'journal.json'),
                    str(run_joint/'suite15k_audit.json'), str(Path(previous['runs']['online'])/'library_extension_audit.json')],
                limitation='One paired initialization; temporal modules differ in capacity. Reused64case test cohort; no cross-seed claim.')
    comparison = bundle/'time_fusion_comparison15k.json'
    evidence = dict(schema='pino_time_fusion_clock_v1', status='passed', cases=[dict(
        example='electropolish2d', comparison=str(comparison), steps=STEPS, arms=clocks)])
    return dict(status='passed', data=data), evidence


def write_new(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--action', choices=('library', 'fusion'), required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'library':
        if args.output is None:
            parser.error('Library figures require --output')
        data = library_data(args.bundle)
        figure = library_figure(data, args.output)
        write_new(args.output/'provenance.json', dict(status='passed', data=data, figure=figure,
                  visual_review='Pending inspection', renderer=str(Path(__file__).resolve())))
    else:
        report, clocks = fusion_data(args.bundle)
        write_new(args.bundle/'time_fusion_comparison15k.json', report)
        write_new(args.bundle/'time_fusion_clock15k.json', clocks)


if __name__ == '__main__':
    main()
