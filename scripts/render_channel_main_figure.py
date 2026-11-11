"""Phase-only NS–CH publication gallery from the accepted saved20k fields.

Original physical mapped queries, actual saved times, no inference, numerical
solve, phase clipping or droplet position correction. CPU-only renderer.
"""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.text import Text
from matplotlib.textpath import TextToPath
from fontTools.ttLib import TTFont
from ml_collections import ConfigDict
import numpy as np
from PIL import Image

from examples.channel_nsch2d.problem import walls
from scripts.render_electropolish_main_figure import (
    phase_error, read_arrays, read_json, scientific_error_label,
)

REPO = Path(__file__).resolve().parents[1]


def load_fields(spec):
    original = read_json(REPO/spec['source_figure_manifest'])['figures']['channel_nsch2d']
    config, summary = read_json(spec['config']), read_json(spec['summary'])
    reserved = read_json(spec['reserved_evaluation'])
    if (spec['run'] != original['run'] or spec['checkpoint'] != original['checkpoint']
            or not Path(spec['checkpoint']).is_file() or summary['status'] != 'completed'
            or summary['iterations'] != spec['checkpoint_step'] or summary['metrics']['step'] != 20000
            or config['train']['iterations'] != spec['checkpoint_step']
            or config['train']['seed'] != spec['runtime_seed'] or reserved['geometries'] != 3
            or spec['phase_channel'] != 0 or spec['interface_level'] != 0
            or spec['samples'] != ['development', 'reserved_000', 'reserved_001', 'reserved_002']):
        raise ValueError('Unexpected final model, input cohort or physical phase convention')
    c = ConfigDict(config)
    length, height = c.geometry.length, c.geometry.height
    nx, ny = spec['grid_shape']
    fields, records = [], []
    for row, source in enumerate(spec['records']):
        if source['label'] != spec['samples'][row]:
            raise ValueError('Geometry order differs from declared rows')
        if row == 0:
            expected_path, expected_score = summary['metrics']['reference'], summary['metrics']['phi']
            if Path(source['prediction']) != Path(spec['run'])/'prediction.npz':
                raise ValueError('Unexpected nominal prediction file')
        else:
            entry = reserved['per_geometry'][row-1]
            expected_path, expected_score = entry['reference'], entry['metrics']['phi']
            if source['prediction'] != original['records'][row-1]['source']:
                raise ValueError('Unexpected reserved prediction file')
        if source['reference'] != expected_path:
            raise ValueError('Reference source differs from final evaluation')
        np.testing.assert_allclose(source['pooled_phase_expected'], expected_score, rtol=0, atol=2e-12)
        pred, ref = read_arrays(source['prediction']), read_arrays(source['reference'])
        for key in ('points', 'time', 'amplitudes'):
            np.testing.assert_array_equal(pred[key], ref[key])
        np.testing.assert_array_equal(pred['reference'], ref['fields'])
        identity = json.loads(str(ref['identity_json']))
        for key in ('physical', 'initial'):
            if identity[key] != config[key]:
                raise ValueError(f'Reference {key} differs from training')
        for key, value in config['geometry'].items():
            if key != 'coefficients' and identity['geometry'][key] != value:
                raise ValueError(f'Reference geometry definition mismatch: {key}')
        if row == 0:
            np.testing.assert_array_equal(pred['amplitudes'], config['geometry']['coefficients'])
        else:
            np.testing.assert_array_equal(pred['amplitudes'], reserved['per_geometry'][row-1]['amplitudes'])
        prediction, target = pred['fields'][..., 0], ref['fields'][..., 0]
        if (prediction.shape != target.shape or prediction.shape != (len(pred['time']), nx*ny)
                or not np.isfinite(prediction).all() or not np.isfinite(target).all()
                or np.any(np.diff(pred['time']) <= 0)):
            raise ValueError('Invalid saved phase/time arrays')
        lo, hi = spec['style']['limits']
        if min(prediction.min(), target.min()) < lo or max(prediction.max(), target.max()) > hi:
            raise ValueError('Phase outside shared color scale; do not clip')
        pooled = phase_error(prediction, target)['relative_l2']
        np.testing.assert_allclose(pooled, expected_score, rtol=0, atol=2e-12)
        x, y = pred['points'].reshape(nx, ny, 2).transpose(2, 0, 1)
        x_expected = (np.arange(nx)+.5)*length/nx
        np.testing.assert_allclose(x, np.broadcast_to(x_expected[:, None], (nx, ny)), rtol=0, atol=1e-13)
        bottom, top = walls(x_expected, pred['amplitudes'], c, xp=np)
        eta = (np.arange(ny)+.5)/ny
        np.testing.assert_allclose(y, bottom[:, None]+(top-bottom)[:, None]*eta, rtol=0, atol=1e-13)
        wall_x = np.linspace(0, length, spec['wall_plot_points'])
        wall_bottom, wall_top = walls(wall_x, pred['amplitudes'], c, xp=np)
        indices, panels = [], []
        for time in spec['times']:
            found = np.flatnonzero(np.isclose(pred['time'], time, rtol=0, atol=1e-14))
            if len(found) != 1:
                raise ValueError(f'Need one actual saved snapshot at {time}')
            index = int(found[0]); indices.append(index)
            if not target[index].min() < 0 < target[index].max():
                raise ValueError('No zero-phase reference interface')
            panels.append(dict(time_index=index, time=time,
                prediction_range=[float(prediction[index].min()), float(prediction[index].max())],
                reference_range=[float(target[index].min()), float(target[index].max())],
                **phase_error(prediction[index], target[index])))
        fields.append((x, y, wall_x, wall_bottom, wall_top,
                       prediction[indices].reshape(-1, nx, ny), target[indices].reshape(-1, nx, ny)))
        records.append(dict(**source, amplitudes=pred['amplitudes'].tolist(), pooled_phase_check=pooled,
                            phase_range_all_times=[float(prediction.min()), float(prediction.max())], panels=panels))
    return (length, height), fields, records


def normalized_time_heading(fig, center_mm, value, style):
    """Editable t-star heading with an explicit 5 pt superscript."""
    normal = font_manager.FontProperties(fname=style['font_file'], size=style['heading_size'])
    italic = font_manager.FontProperties(fname=style['math_italic_font_file'], size=style['heading_size'])
    raised = font_manager.FontProperties(fname=style['font_file'], size=style['superscript_size'])
    pieces = [('t', italic, 0, 'sans-math-italic'),
              ('*', raised, style['superscript_rise_pt'], None),
              (f' = {value:.2f}', normal, 0, None)]
    measure = TextToPath()
    widths = [measure.get_text_width_height_descent(text, font, ismath=False)[0]
              for text, font, _, _ in pieces]
    widths[0] += style['superscript_gap_pt']
    figure_width_pt, figure_height_pt = fig.get_figwidth()*72, fig.get_figheight()*72
    left = center_mm/style['figure_mm'][0]-sum(widths)/(2*figure_width_pt)
    baseline = style['time_heading_y_mm']/style['figure_mm'][1]-style['heading_size']*.3/figure_height_pt
    for (text, font, rise, gid), width in zip(pieces, widths, strict=True):
        fig.text(left, baseline+rise/figure_height_pt, text, fontproperties=font,
                 ha='left', va='baseline', gid=gid)
        left += width/figure_width_pt


def render(spec):
    extent, fields, records = load_fields(spec)
    style = spec['style']
    font_path = Path(style['font_file']).resolve(strict=True)
    math_path = Path(style['math_italic_font_file']).resolve(strict=True)
    for path, family in ((font_path, 'Helvetica'), (math_path, 'DejaVu Sans')):
        with TTFont(path) as face:
            if face['name'].getDebugName(1) != family or 'glyf' not in face:
                raise ValueError(f'Unexpected font: {path}')
            if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
                raise ValueError('Font flags do not permit embedding')
        font_manager.fontManager.addfont(str(path))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font_path:
        raise ValueError('Unexpected Helvetica substitution')
    width, height = style['figure_mm']
    palette = read_json(REPO/style['cmap_file'])
    colors = np.asarray(palette['colors'], dtype=np.float64)
    if (style['cmap'] not in (palette['name'], palette['name']+'_r') or colors.shape != (256, 3)
            or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1)):
        raise ValueError('Invalid or mismatched explicit colormap table')
    if style['cmap'] == palette['name']+'_r':
        colors = colors[::-1]
    cmap = ListedColormap(colors, name=style['cmap'])
    if spec['time_label'] == 't':
        time_divisor = 1.0
    elif spec['time_label'] in ('t/T', 't*'):
        time_divisor = float(read_json(spec['config'])['physical']['terminal_time'])
    else:
        raise ValueError('Time label must explicitly select t, t/T or normalized t*')
    if not np.isfinite(time_divisor) or time_divisor <= 0:
        raise ValueError('Invalid display-time divisor')
    display_times = np.asarray(spec['times'], dtype=float)/time_divisor
    np.testing.assert_allclose(style['panel_width_mm']/style['panel_height_mm'], extent[0]/extent[1])
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size'],
          'text.color': style['text_color'], 'axes.labelcolor': style['text_color'],
          'xtick.color': style['text_color'], 'ytick.color': style['text_color'],
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'text.usetex': False, 'axes.linewidth': style['axes_linewidth']}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        for row, (x, y, wall_x, wall_bottom, wall_top, predictions, references) in enumerate(fields):
            bottom = style['row_bottoms_mm'][row]
            for col, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
                left = style['left_mm']+col*(style['panel_width_mm']+style['column_gap_mm'])
                ax = fig.add_axes([left/width, bottom/height,
                    style['panel_width_mm']/width, style['panel_height_mm']/height])
                field = ax.pcolormesh(x, y, prediction, shading='gouraud', cmap=cmap,
                    vmin=style['limits'][0], vmax=style['limits'][1], rasterized=True)
                contours = ax.contour(x, y, reference, levels=[spec['interface_level']],
                    colors=style['contour_color'], linewidths=style['contour_linewidth'],
                    linestyles=style['contour_linestyle'])
                ax.fill_between(wall_x, 0, wall_bottom, color=style['wall_color'], lw=0, zorder=3)
                ax.fill_between(wall_x, wall_top, extent[1], color=style['wall_color'], lw=0, zorder=3)
                ax.plot(wall_x, wall_bottom, color=style['wall_edge_color'], lw=style['wall_linewidth'], zorder=4)
                ax.plot(wall_x, wall_top, color=style['wall_edge_color'], lw=style['wall_linewidth'], zorder=4)
                ax.set(xlim=(0, extent[0]), ylim=(0, extent[1]), aspect='equal', xticks=[], yticks=[])
                panel = records[row]['panels'][col]
                panel['display_label'] = scientific_error_label(fig, left+style['panel_width_mm']/2,
                    bottom-style['error_gap_mm'], panel['relative_l2'], style)
                panel['reference_contour_segments'] = len(contours.allsegs[0])
                if panel['display_label']['width_mm'] > style['panel_width_mm']:
                    raise ValueError('Error annotation exceeds panel width')
                if row == 0:
                    if spec['time_label'] == 't*':
                        normalized_time_heading(fig, left+style['panel_width_mm']/2, display_times[col], style)
                    else:
                        fig.text((left+style['panel_width_mm']/2)/width, style['time_heading_y_mm']/height,
                            f'{spec["time_label"]} = {display_times[col]:.2f}', ha='center', va='center',
                            fontsize=style['heading_size'])
            fig.text(style['row_label_x_mm']/width, (bottom+style['panel_height_mm']/2)/height,
                     f'Sample {row+1}', rotation=90, ha='center', va='center')
        field_bottom, field_top = min(style['row_bottoms_mm']), max(style['row_bottoms_mm'])+style['panel_height_mm']
        middle = (field_bottom+field_top)/2
        bar_rect = [style['colorbar_x_mm'], middle-style['colorbar_height_mm']/2,
                    style['colorbar_width_mm'], style['colorbar_height_mm']]
        bar = fig.colorbar(field, cax=fig.add_axes(np.array(bar_rect)/[width, height, width, height]),
                           ticks=style['colorbar_ticks'])
        bar.ax.set_title('φ', fontsize=style['heading_size'], pad=style['colorbar_label_pad_pt'])
        bar.ax.tick_params(direction='in', width=style['axes_linewidth'], length=2, pad=2)
        bar.solids.set_rasterized(True)
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                font = math_path if item.get_gid() == 'sans-math-italic' else font_path
                item.set_fontproperties(font_manager.FontProperties(fname=str(font), size=item.get_fontsize()))
                item.set_color(style['text_color'])
        fig.canvas.draw()
        texts = []
        for item in fig.findobj(Text):
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(fig.canvas.get_renderer())
            if (not 5 <= item.get_fontsize() <= 7 or not fig.bbox.contains(box.x0, box.y0)
                    or not fig.bbox.contains(box.x1, box.y1)):
                raise ValueError(f'Invalid font size or clipped label: {item.get_text()}')
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
    receipt = dict(schema='channel_main_figure_validation_v1', status='passed',
        renderer=str(Path(__file__).relative_to(REPO)), specification=spec,
        versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                      colormap_provider=palette['provider'], colormap_version=palette['version']),
        records=records, text=texts, png_pixels=list(rgb.size),
        display_time=dict(label=spec['time_label'], divisor=time_divisor, values=display_times.tolist()),
        layout=dict(field_vertical_bounds_mm=[field_bottom, field_top], colorbar_rect_mm=bar_rect,
                    colorbar_center_mm=middle, colorbar_label_position='top, horizontal', aspect='equal'),
        output_bytes=dict(pdf=pdf.stat().st_size, png=png.stat().st_size),
        metric_interpolation=False, temporal_interpolation=False, inference=False,
        layers=dict(field='raster, Gouraud on original mapped query grid',
                    reference_contours='vector, original local FEM query values',
                    walls='vector masks and analytic wall curves',
                    text='editable embedded Helvetica; DejaVu Sans Oblique L; 5pt superscripts',
                    colorbar='raster gradient, vector ticks/text'))
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(pdf=str(pdf), png=str(png), bytes=receipt['output_bytes'],
                         errors=[[p['relative_l2'] for p in r['panels']] for r in records]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    specification = read_json(args.spec)
    if specification['schema'] != 'channel_main_figure_v1':
        raise ValueError('Unsupported figure specification')
    render(specification)
