"""Render the Electropolish phase-only publication preview from saved arrays.

CPU-only: no checkpoint inference, training or reference solve. Raster field
colors are bilinearly interpolated for display; errors use original nodes.
"""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.textpath import TextToPath
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def phase_error(prediction, reference):
    p, r = np.asarray(prediction, np.float64), np.asarray(reference, np.float64)
    if p.shape != r.shape or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Require finite matching phase fields on original nodes')
    numerator, denominator = np.linalg.norm(p-r), np.linalg.norm(r)
    if denominator <= 0:
        raise ValueError('Relative phase error requires a positive reference norm')
    return dict(relative_l2=float(numerator/denominator),
                numerator_norm=float(numerator), denominator_norm=float(denominator),
                spatial_nodes=int(r.size))


def scientific_error_label(fig, center_mm, baseline_mm, error, style):
    """Typeset true superscripts without shrinking them below the 5 pt floor."""
    mantissa, exponent = f'{error:.2e}'.split('e')
    exponent = int(exponent)
    regular = font_manager.FontProperties(fname=style['font_file'], size=style['font_size'])
    italic = font_manager.FontProperties(fname=style['math_italic_font_file'], size=style['font_size'])
    raised = font_manager.FontProperties(fname=style['font_file'], size=style['superscript_size'])
    pieces = [('Rel. ', regular, 0, None), ('L', italic, 0, 'sans-math-italic'),
              ('2', raised, style['superscript_rise_pt'], None),
              (f' = {mantissa} × 10', regular, 0, None),
              (str(exponent).replace('-', '−'), raised, style['superscript_rise_pt'], None)]
    # Measure unhinted advances in physical points. Screen-DPI advances cause
    # overlaps when the same editable text is exported at a different DPI.
    measure = TextToPath()
    widths = [measure.get_text_width_height_descent(text, font, ismath=False)[0]
              for text, font, _, _ in pieces]
    for index in (1, 3):
        widths[index] += style['superscript_gap_pt']
    width_mm, height_mm = style['figure_mm']
    figure_width_pt = fig.get_figwidth()*72
    left = center_mm/width_mm-sum(widths)/(2*figure_width_pt)
    for (text, font, rise, gid), width in zip(pieces, widths, strict=True):
        fig.text(left, baseline_mm/height_mm+rise/(fig.get_figheight()*72), text,
                 fontproperties=font, ha='left', va='baseline', gid=gid)
        left += width/figure_width_pt
    return dict(latex=rf'\mathrm{{Rel.}}\ L^{{2}} = {mantissa} \times 10^{{{exponent}}}',
                mantissa=mantissa, exponent=exponent,
                width_mm=sum(widths)/figure_width_pt*width_mm)


def load_fields(spec):
    old = read_json(REPO/spec['source_figure_manifest'])['figures']['electropolish2d']
    if spec['run'] != old['run'] or spec['samples'] != old['samples']:
        raise ValueError('Keep the selected main-figure model and predeclared samples')
    data, report = read_arrays(spec['prediction']), read_json(spec['evaluation'])
    config = read_json(spec['config'])
    if (report['status'] != 'passed' or report['count'] != 64 or
            report['model'] != spec['checkpoint'] or not Path(spec['checkpoint']).is_file() or
            report['iterations'] != config['train']['iterations']):
        raise ValueError('Saved evaluation does not identify the selected final model')
    if spec['phase_channel'] != 0 or spec['interface_level'] != .5:
        raise ValueError('This plot requires physical phase and its midpoint interface')
    x, y = data['X'][0], data['Y'][:, 0]
    xx, yy = np.meshgrid(x, y)
    np.testing.assert_array_equal(data['X'], xx)
    np.testing.assert_array_equal(data['Y'], yy)
    np.testing.assert_allclose(np.diff(x), x[1]-x[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diff(y), y[1]-y[0], rtol=0, atol=1e-14)
    if x[0] != 0 or x[-1] != 1 or y[0] != 0 or y[-1] != .5:
        raise ValueError('Unexpected physical domain or coordinate order')
    indices = []
    for time in spec['normalized_times']:
        found = np.flatnonzero(np.isclose(data['time'], time, rtol=0, atol=1e-14))
        if len(found) != 1:
            raise ValueError(f'Need one actual saved snapshot at {time}')
        indices.append(int(found[0]))
    fields, records = [], []
    for row, sample in enumerate(spec['samples']):
        source = spec['references'][row]
        if source['index'] != sample:
            raise ValueError('Reference order differs from the selected samples')
        ref = read_arrays(source['file'])
        for key in ('X', 'Y', 'time'):
            np.testing.assert_array_equal(data[key], ref[key])
        meta, logged = json.loads(str(ref['metadata_json'])), report['records'][sample]
        if (meta['index'] != sample or logged['index'] != sample or
                meta['surface_id'] != data['surface_ids'][sample] or
                logged['surface_id'] != meta['surface_id']):
            raise ValueError(f'Physical input identity mismatch for sample {sample}')
        pred = data['prediction'][sample, ..., spec['phase_channel']]
        target = ref['fields'][..., spec['phase_channel']]
        pooled = phase_error(pred[report['regular_time_indices']],
                             target[report['regular_time_indices']])['relative_l2']
        np.testing.assert_allclose(pooled, logged['local101_64']['pooled_phi'], rtol=0, atol=2e-10)
        panels = []
        for index, time in zip(indices, spec['normalized_times'], strict=True):
            lo, hi = spec['style']['limits']
            if pred[index].min() < lo or pred[index].max() > hi:
                raise ValueError('Prediction exceeds the declared color scale; do not silently clip')
            if not target[index].min() < spec['interface_level'] < target[index].max():
                raise ValueError('Reference has no phase midpoint interface')
            panels.append(dict(time_index=index, normalized_time=time,
                physical_time_seconds=time*config['nondim']['T_c'],
                prediction_range=[float(pred[index].min()), float(pred[index].max())],
                **phase_error(pred[index], target[index])))
        fields.append((pred[indices], target[indices]))
        records.append(dict(sample_index=sample, surface_id=int(meta['surface_id']),
                            reference=source['file'], pooled_phase_check=pooled, panels=panels))
    return x, y, fields, records


def render(spec):
    x, y, fields, records = load_fields(spec)
    style = spec['style']
    font_path = Path(style['font_file']).resolve(strict=True)
    with TTFont(font_path) as face:
        if face['name'].getDebugName(1) != style['font_family'] or 'glyf' not in face:
            raise ValueError('Require the explicitly configured Helvetica TrueType file')
        if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError('Font embedding flags do not permit this export')
    font_manager.fontManager.addfont(str(font_path))
    math_path = Path(style['math_italic_font_file']).resolve(strict=True)
    with TTFont(math_path) as face:
        if face['name'].getDebugName(1) != 'DejaVu Sans' or 'glyf' not in face:
            raise ValueError('Require the declared sans-serif italic math font')
    font_manager.fontManager.addfont(str(math_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    resolved = Path(font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)).resolve()
    if resolved != font_path:
        raise ValueError('Unexpected font substitution')
    if style['example_title'] is not False:
        raise ValueError('Keep example names in filenames, not in-figure titles')
    if style['reference_legend'] is not False:
        raise ValueError('The reference-line meaning belongs in the paper caption')
    width, height = style['figure_mm']
    if style['panel_width_mm']/style['panel_height_mm'] != (x[-1]-x[0])/(y[-1]-y[0]):
        raise ValueError('Panel dimensions must preserve the physical aspect ratio')
    rc = {'font.family': family, 'font.size': style['font_size'],
          'text.color': style['text_color'], 'axes.labelcolor': style['text_color'],
          'xtick.color': style['text_color'], 'ytick.color': style['text_color'],
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'text.usetex': False, 'axes.linewidth': style['axes_linewidth']}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        dx, dy = x[1]-x[0], y[1]-y[0]
        # imshow places pixel centers at the original nodes, not half a cell inward.
        extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
        for row, (predictions, references) in enumerate(fields):
            bottom = style['row_bottoms_mm'][row]
            for col, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
                left = style['left_mm']+col*(style['panel_width_mm']+style['column_gap_mm'])
                ax = fig.add_axes([left/width, bottom/height,
                    style['panel_width_mm']/width, style['panel_height_mm']/height])
                image = ax.imshow(prediction, origin='lower', extent=extent,
                    interpolation=style['interpolation'], interpolation_stage=style['interpolation_stage'],
                    cmap=style['cmap'], vmin=style['limits'][0], vmax=style['limits'][1],
                    rasterized=True)
                contours = ax.contour(x, y, reference, levels=[spec['interface_level']],
                    colors=style['contour_color'], linewidths=style['contour_linewidth'],
                    linestyles=style['contour_linestyle'])
                if contours.get_rasterized():
                    raise ValueError('Reference contours must stay vector')
                ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), aspect='equal', xticks=[], yticks=[])
                error = records[row]['panels'][col]['relative_l2']
                label = scientific_error_label(fig, left+style['panel_width_mm']/2,
                    bottom-style['error_gap_mm'], error, style)
                if label['width_mm'] > style['panel_width_mm']:
                    raise ValueError('Scientific error annotation exceeds panel width')
                records[row]['panels'][col]['display_label'] = label
                records[row]['panels'][col]['reference_contour_segments'] = len(contours.allsegs[0])
                if row == 0:
                    fig.text((left+style['panel_width_mm']/2)/width,
                        style['time_heading_y_mm']/height, f't* = {spec["normalized_times"][col]:.2f}',
                        ha='center', va='center', fontsize=style['heading_size'])
            fig.text(style['row_label_x_mm']/width, (bottom+style['panel_height_mm']/2)/height,
                     f'Sample {row+1}', rotation=90, ha='center', va='center')
        field_bottom = min(style['row_bottoms_mm'])
        field_top = max(style['row_bottoms_mm'])+style['panel_height_mm']
        center = (field_bottom+field_top)/2
        colorbar_rect_mm = [style['colorbar_x_mm'], center-style['colorbar_height_mm']/2,
                           style['colorbar_width_mm'], style['colorbar_height_mm']]
        cb_rect = np.asarray(colorbar_rect_mm)/[width, height, width, height]
        bar = fig.colorbar(image, cax=fig.add_axes(cb_rect), ticks=[0, .5, 1])
        bar.ax.set_title('φ', fontsize=style['heading_size'], pad=style['colorbar_label_pad_pt'])
        bar.ax.tick_params(direction='in', width=style['axes_linewidth'], length=2, pad=2)
        bar.solids.set_rasterized(True)
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                selected_font = math_path if item.get_gid() == 'sans-math-italic' else font_path
                item.set_fontproperties(font_manager.FontProperties(fname=str(selected_font), size=item.get_fontsize()))
                item.set_color(style['text_color'])
        fig.canvas.draw()
        texts = []
        for item in fig.findobj(Text):
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(fig.canvas.get_renderer())
            if not 5 <= item.get_fontsize() <= 7 or not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
                raise ValueError(f'Invalid text size or clipped label: {item.get_text()}')
            texts.append(dict(text=item.get_text(), size_pt=item.get_fontsize(),
                              font_file=item.get_fontproperties().get_file()))
        pdf, png = (Path(spec['output'][key]) for key in ('pdf', 'png'))
        for path in (pdf, png):
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, dpi=style['dpi'], metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(png, dpi=style['png_dpi'])
        plt.close(fig)
    with Image.open(png) as source:
        rgb = source.convert('RGB')
    rgb.save(png, optimize=True, dpi=(style['png_dpi'], style['png_dpi']))
    result = dict(schema='electropolish_main_figure_validation_v1', status='passed',
        renderer=str(Path(__file__).relative_to(REPO)), specification=spec,
        versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        records=records, text=texts, png_pixels=list(rgb.size),
        layout=dict(field_vertical_bounds_mm=[field_bottom, field_top],
                    colorbar_rect_mm=colorbar_rect_mm, colorbar_center_mm=center,
                    colorbar_label_position='top, horizontal', reference_legend=False),
        output_bytes=dict(pdf=pdf.stat().st_size, png=png.stat().st_size),
        metric_interpolation=False, temporal_interpolation=False, inference=False,
        layers=dict(field='raster, bilinear display in data coordinates',
                    reference_contours='vector, original nodes',
                    text='editable embedded Helvetica; DejaVu Sans Oblique for L; explicit superscript sizing',
                    colorbar='raster gradient, vector ticks and text'))
    (REPO/spec['output']['receipt']).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(pdf=str(pdf), png=str(png), bytes=result['output_bytes'],
                         phase_errors=[[p['relative_l2'] for p in r['panels']] for r in records]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=REPO/'artifacts/electropolish_main_figure.json')
    args = parser.parse_args()
    spec = read_json(args.spec)
    if spec['schema'] != 'electropolish_main_figure_v1':
        raise ValueError('Unsupported figure specification')
    render(spec)


if __name__ == '__main__':
    main()
