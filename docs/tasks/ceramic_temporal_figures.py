"""Actual64 temporal-control curves/violins and fixedcase phase fields; CPU only."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def comparison_data(bundle):
    result = read(bundle/'comparison.json'); journal = read(bundle/'journal.json')
    if result['status'] != 'passed' or journal['status'] != 'complete':
        raise ValueError('Both completed trainers and independent paired audit are required')
    ids = list(range(64)); arms = []
    for name, method in (('film', 'decoder_film'), ('joint', 'joint_query')):
        run = Path(result['methods'][name]['run'])
        with np.load(bundle/(name+'-convergence.npz'), allow_pickle=False) as data:
            elapsed = data['cold_wall_seconds'].copy()
            elapsed[-1] = float(data['parent_seconds'])
            errors = data['errors']; final = data['final_errors']; steps = data['steps']
            # Third panel is the mean of the two channelwise norms per sample.
            curves = np.concatenate((errors, errors.mean(axis=-1, keepdims=True)), axis=-1)
            distribution = np.concatenate((final, final.mean(axis=-1, keepdims=True)), axis=-1)
        arms.append(dict(method=method, run=str(run), checkpoint=str(run/'model_final.eqx'),
            configuration=str(run/'config.json'), monitor=str(run/'metrics.jsonl'),
            monitor_errors=curves.tolist(), final_errors=distribution.tolist(),
            elapsed_seconds=elapsed.tolist(), timing=dict(cold_process_seconds=float(elapsed[-1]),
                provenance=str(bundle/'journal.json'), scope='Parent before launch through child.wait completion')))
    return dict(status='passed', comparison_kind='time_fusion', example='particle_corrosion2d',
        test_ids=ids, monitor_ids=ids, budget=20000, steps=steps.tolist(), arms=arms,
        fields=[dict(key='phi', label='Phase'), dict(key='c', label='Concentration'),
                dict(key='mean', label='Channel mean')], convergence_ylabel='Rel. L² error',
        convergence_axis=dict(kind='cold_wall_time', label='Elapsed wall time (s)',
            scope='Direct monotonic parent origin at every record; final observation includes exit/writes'),
        smoothing=dict(kind='whole_trajectory', fraction=.22, replicates=2000, seed=20261115, grid_points=240),
        interval_scope='Pointwise bootstrap interval for smoothed cohort mean, whole geometry trajectories; not cross-seed',
        metric=result['metric'], reference=journal['reference'], seed=42,
        source_audit=str(bundle/'comparison.json'), default_changed=False)


def physical_fields(bundle, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation
    import fitz
    import subprocess
    from docs.tasks.ceramic_fem_comparison_figures import style, FONT, PDFFONTS
    from scripts.render_particle_corrosion_main_figure import hole_edges
    from scripts.render_electropolish_main_figure import scientific_error_label
    journal = read(bundle/'journal.json'); reference = Path(journal['reference'])
    manifest = read(reference/'manifest.json')
    # The common64local-FEM bank stores eleven snapshots, spaced by0.1.
    # Select actual saved states; do not fabricate quarter-time interpolation.
    samples = [0, 1, 2, 3]; selected_times = [0., .2, .5, .8, 1.]
    width, height = 170., 104.
    pw, ph, left, gap = 28., 16.8, 9., 1.7
    math_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', style='oblique'),
                                     fallback_to_default=False)
    labels = dict(font_file=str(FONT), math_italic_font_file=math_path, font_size=6., superscript_size=5.,
                  superscript_rise_pt=2.5, superscript_gap_pt=.3, figure_mm=[width, height])
    reports = {}
    for arm in ('film', 'joint'):
        run = Path(journal['runs'][arm]); c = read(run/'config.json'); panels = []
        prediction_root = Path(journal['prediction_roots'][arm]) if 'prediction_roots' in journal else run
        with matplotlib.rc_context(style()):
            fig = plt.figure(figsize=(width/25.4, height/25.4))
            for row, sample in enumerate(samples):
                refpath = reference/manifest['cases'][sample]['file']; predpath = prediction_root/f'prediction_{sample:03d}.npz'
                with np.load(refpath, allow_pickle=False) as r, np.load(predpath, allow_pickle=False) as p:
                    xy, triangles, times = r['points'], r['triangles'], r['times']
                    np.testing.assert_array_equal(xy, p['points']); np.testing.assert_array_equal(times, p['times'])
                    target, pred = r['fields'][..., 0], p['prediction'][..., 0]
                if (pred.shape != target.shape or not np.isfinite(pred).all()
                        or pred.min() < 0 or pred.max() > 1): raise ValueError('Invalid physical phase fields')
                tri = Triangulation(xy[:, 0], xy[:, 1], triangles)
                edges = hole_edges(xy, triangles, (1., .6), c['particles']['count'])
                bottom = 9.+(3-row)*23.
                for col, time in enumerate(selected_times):
                    matches = np.flatnonzero(np.isclose(times, time, atol=1e-14, rtol=0))
                    if len(matches) != 1: raise ValueError('Actual saved snapshot required')
                    i = int(matches[0]); x = left+col*(pw+gap)
                    ax = fig.add_axes([x/width, bottom/height, pw/width, ph/height])
                    ax.set_facecolor('#EFE7D4')
                    image = ax.tripcolor(tri, pred[i], shading='gouraud', cmap='coolwarm', vmin=0, vmax=1, rasterized=True)
                    if not target[i].min() < .5 < target[i].max(): raise ValueError('Missing reference interface')
                    ax.tricontour(tri, target[i], levels=[.5], colors='#333333', linewidths=.55, linestyles='--')
                    ax.add_collection(LineCollection(edges, colors='#5B554C', linewidths=.35, zorder=3))
                    ax.set(xlim=(0, 1), ylim=(.6, 0), aspect='equal', xticks=[], yticks=[])
                    error = float(np.linalg.norm(pred[i].astype(float)-target[i])/np.linalg.norm(target[i]))
                    label = scientific_error_label(fig, x+pw/2, bottom-3., error, labels)
                    if label['width_mm'] > pw: raise ValueError('Physical error label too wide')
                    panels.append(dict(sample=sample, time=time, time_index=i, phase_error=error,
                        prediction=str(predpath), reference=str(refpath), label=label))
                    if row == 0:
                        fig.text((x+pw/2)/width, (bottom+ph+3.)/height, f't* = {time:.2f}',
                                 ha='center', va='center', fontsize=7)
                fig.text(5./width, (bottom+ph/2)/height, f'Sample {sample+1}', rotation=90,
                         ha='center', va='center', fontsize=6)
            center = (9.+9.+3*23.+ph)/2
            bar = fig.colorbar(image, cax=fig.add_axes([160/width, (center-20)/height, 2./width, 40./height]), ticks=[0, .5, 1])
            bar.ax.set_title('φ', fontsize=7, pad=4)
            bar.ax.tick_params(direction='in', length=2, width=.5, pad=2)
            name = output/f'particle_corrosion2d_time_{arm}_fields'
            fig.savefig(name.with_suffix('.pdf'), dpi=450, metadata={'CreationDate': None, 'ModDate': None})
            fig.savefig(name.with_suffix('.png'), dpi=300); plt.close(fig)
        with fitz.open(name.with_suffix('.pdf')) as doc:
            page = doc[0]
            np.testing.assert_allclose([page.rect.width, page.rect.height], np.array([width, height])*72/25.4, atol=1e-4)
            spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0 for line in b['lines'] for s in line['spans']]
            if len(spans) < 80 or any(not 4.99 <= s['size'] <= 7.01 for s in spans):
                raise ValueError('Physical figure text must remain editable and5–7pt')
            if any(not page.rect.contains(fitz.Rect(s['bbox'])) for s in spans): raise ValueError('Clipped physical figure text')
            for font in page.get_fonts(full=True):
                if font[2] not in ('TrueType', 'Type0') or not doc.extract_font(font[0])[3]: raise ValueError('Unembedded font')
            if not page.get_images(): raise ValueError('Dense field layer must be rasterized')
            # Editable labels are verified on a separate disposable copy.
            edited = fitz.open(stream=doc.tobytes(), filetype='pdf'); ep = edited[0]
            for original, replacement, y in [('Sample 1', 'Sample A', 8), ('Rel.', 'Error', 17)]:
                matches = ep.search_for(original)
                if not matches: raise ValueError('Missing editable field label')
                ep.add_redact_annot(matches[0], fill=(1, 1, 1)); ep.apply_redactions(images=0, graphics=0)
                ep.insert_text((5, y), replacement, fontsize=6, fontname='field_helvetica', fontfile=str(FONT))
            if 'Sample A' not in ep.get_text() or 'Error' not in ep.get_text(): raise ValueError('Field edit probe failed')
            edited.close()
        reports[arm] = dict(pdf=str(name.with_suffix('.pdf')), png=str(name.with_suffix('.png')), panels=panels,
            size_mm=[width, height], font_file=str(FONT), math_font_file=math_path,
            pdffonts=subprocess.check_output([str(PDFFONTS), str(name.with_suffix('.pdf'))], text=True),
            cmap='coolwarm', limits=[0, 1], samples=samples, normalized_times=selected_times,
            layers='Gouraud raster on original physical triangles; vector reference0.5contour, walls and text',
            interface='dashed local-P1-FEM; no cross-hole or temporal interpolation', text_spans=spans,
            default_changed=False, checkpoint=str(run/'model_final.eqx'))
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); data = comparison_data(a.bundle.resolve(strict=True))
    from scripts.pino_comparison_plot import render
    result = render(data, a.output)
    fields = physical_fields(a.bundle, a.output)
    (a.output/'physical_provenance.json').write_text(json.dumps(fields, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(statistics={key: value['png'] for key, value in result['figures'].items()},
                          fields={key: value['png'] for key, value in fields.items()})), flush=True)


if __name__ == '__main__': main()
