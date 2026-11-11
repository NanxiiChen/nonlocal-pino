"""Saved-cell Spinodal phase gallery; no inference or new reference solve.

Phase errors use original cell centers. Area-weighted incident-cell projection
is used only for display and contour reconstruction on the original triangles.
"""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Ellipse
from matplotlib.text import Text
from matplotlib.tri import Triangulation
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image

from scripts.render_electropolish_main_figure import (
    phase_error, read_arrays, read_json, scientific_error_label,
)

REPO = Path(__file__).resolve().parents[1]


def display_vertices(values, cells, areas, vertex_count):
    weights = np.bincount(cells.ravel(), weights=np.repeat(areas, 3), minlength=vertex_count)
    if np.any(weights <= 0):
        raise ValueError('Every display vertex needs incident physical cells')
    return np.stack([np.bincount(cells.ravel(), weights=np.repeat(v*areas, 3),
                                minlength=vertex_count)/weights for v in values])


def load_fields(spec):
    original = read_json(REPO/spec['source_figure_manifest'])['figures']['confined_ch2d']
    config, summary = read_json(spec['config']), read_json(spec['summary'])
    gallery, manifest = read_json(spec['validation']), read_json(spec['reference_manifest'])
    if (spec['run'] != original['run'] or spec['checkpoint'] != original['checkpoint']
            or not Path(spec['checkpoint']).is_file() or summary['new_updates'] != spec['checkpoint_step']
            or summary['start_step'] != 0 or summary['checkpoint'] != Path(spec['checkpoint']).name
            or gallery['step'] != spec['checkpoint_step'] or gallery['model'] != summary['checkpoint']
            or config['train']['seed'] != spec['runtime_seed'] or manifest['status'] != 'passed'
            or spec['samples'] != manifest['test_indices'][:4]
            or spec['field'] != 'phi' or spec['interface_level'] != 0):
        raise ValueError('Unexpected checkpoint, reserved inputs or phase convention')
    for key in ('physical', 'initial', 'geometry', 'nl'):
        if config[key] != manifest['specification'][key]:
            raise ValueError(f'Training/reference physical input mismatch: {key}')
    center, axes = np.array(config['geometry']['center']), np.array(config['geometry']['axes'])
    fields, records = [], []
    for sample, source in zip(spec['samples'], spec['records'], strict=True):
        if source['index'] != sample:
            raise ValueError('Input order differs from specification')
        pred, ref = read_arrays(source['prediction']), read_arrays(source['reference'])
        for key in ('points', 'volumes', 'mesh_points', 'mesh_cells', 'holes', 'time'):
            np.testing.assert_array_equal(pred[key], ref[key])
        np.testing.assert_array_equal(pred['phi_reference'], ref['phi'])
        np.testing.assert_array_equal(pred['time'], spec['dimensionless_times'])
        p, r = pred['phi'].astype(np.float64), ref['phi']
        vertices, cells, areas = pred['mesh_points'], pred['mesh_cells'], pred['volumes']
        if (p.shape != r.shape or p.shape != (5, len(cells)) or not np.isfinite(p).all()
                or not np.isfinite(r).all() or np.any(areas <= 0) or cells.shape[1] != 3
                or not np.issubdtype(cells.dtype, np.integer) or cells.min() < 0
                or cells.max() >= len(vertices) or np.any(np.diff(pred['time']) <= 0)):
            raise ValueError('Invalid cell fields, mesh or times')
        triangle_points = vertices[cells]
        np.testing.assert_allclose(triangle_points.mean(axis=1), pred['points'], rtol=0, atol=1e-12)
        a, b = triangle_points[:, 1]-triangle_points[:, 0], triangle_points[:, 2]-triangle_points[:, 0]
        np.testing.assert_allclose(abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/2, areas,
                                   rtol=1e-12, atol=1e-15)
        if min(p.min(), r.min()) < spec['style']['limits'][0] or max(p.max(), r.max()) > spec['style']['limits'][1]:
            raise ValueError('Saved phase exceeds the shared scale; do not clip')
        expected = next(item for item in gallery['verified_all8'] if item['index'] == sample)
        pooled = float(np.sqrt(np.sum(areas*(p-r)**2)/np.sum(areas*r*r)))
        np.testing.assert_allclose(pooled, expected['phi_pooled'], rtol=0, atol=2e-12)
        tri = Triangulation(vertices[:, 0], vertices[:, 1], cells)
        if np.any(tri.get_trifinder()(pred['holes'][:, 0], pred['holes'][:, 1]) != -1):
            raise ValueError('A hole center is covered by physical triangles')
        pv, rv = (display_vertices(v, cells, areas, len(vertices)) for v in (p, r))
        if pv.min() < p.min()-1e-14 or pv.max() > p.max()+1e-14:
            raise ValueError('Display reconstruction creates phase overshoot')
        panels = [dict(time_index=k, dimensionless_time=float(time),
                       prediction_range=[float(p[k].min()), float(p[k].max())],
                       **phase_error(p[k], r[k])) for k, time in enumerate(pred['time'])]
        fields.append((tri, pred['holes'], pv, rv))
        records.append(dict(index=sample, prediction=source['prediction'], reference=source['reference'],
            cells=len(cells), vertices=len(vertices), holes=pred['holes'].tolist(),
            historical_area_weighted_pooled_check=pooled, panels=panels))
    return center, axes, fields, records


def render(spec):
    center, axes, fields, records = load_fields(spec)
    style = spec['style']
    font_path = Path(style['font_file']).resolve(strict=True)
    math_path = Path(style['math_italic_font_file']).resolve(strict=True)
    for path, family in ((font_path, 'Helvetica'), (math_path, 'DejaVu Sans')):
        with TTFont(path) as face:
            if face['name'].getDebugName(1) != family or 'glyf' not in face:
                raise ValueError(f'Unexpected font: {path}')
            if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
                raise ValueError('Font embedding flags do not permit export')
        font_manager.fontManager.addfont(str(path))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font_path:
        raise ValueError('Unexpected Helvetica substitution')
    width, height = style['figure_mm']
    np.testing.assert_allclose(style['panel_width_mm']/style['panel_height_mm'], axes[0]/axes[1])
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size'],
          'text.color': style['text_color'], 'axes.labelcolor': style['text_color'],
          'xtick.color': style['text_color'], 'ytick.color': style['text_color'],
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'text.usetex': False, 'axes.linewidth': style['axes_linewidth']}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        for row, (tri, holes, predictions, references) in enumerate(fields):
            bottom = style['row_bottoms_mm'][row]
            for col, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
                left = style['left_mm']+col*(style['panel_width_mm']+style['column_gap_mm'])
                ax = fig.add_axes([left/width, bottom/height,
                    style['panel_width_mm']/width, style['panel_height_mm']/height])
                field = ax.tripcolor(tri, prediction, shading='gouraud', cmap=style['cmap'],
                    vmin=style['limits'][0], vmax=style['limits'][1], rasterized=True)
                contours = ax.tricontour(tri, reference, levels=[spec['interface_level']],
                    colors=style['contour_color'], linewidths=style['contour_linewidth'],
                    linestyles=style['contour_linestyle'])
                for x, y, radius in holes:
                    ax.add_patch(Circle((x, y), radius, facecolor=style['hole_color'],
                        edgecolor=style['hole_edge_color'], linewidth=style['hole_linewidth'], zorder=4))
                ax.add_patch(Ellipse(center, 2*axes[0], 2*axes[1], fill=False,
                    edgecolor=style['hole_edge_color'], linewidth=style['hole_linewidth'], clip_on=False))
                ax.set(xlim=(center[0]-axes[0], center[0]+axes[0]),
                       ylim=(center[1]-axes[1], center[1]+axes[1]), aspect='equal')
                ax.set_axis_off()
                panel = records[row]['panels'][col]
                panel['display_label'] = scientific_error_label(fig, left+style['panel_width_mm']/2,
                    bottom-style['error_gap_mm'], panel['relative_l2'], style)
                panel['reference_contour_segments'] = len(contours.allsegs[0])
                if panel['display_label']['width_mm'] > style['panel_width_mm']:
                    raise ValueError('Error annotation exceeds panel width')
                if col == 0:
                    fig.text(style['row_label_x_mm']/width, (bottom+style['panel_height_mm']/2)/height,
                             f'Sample {row+1}', rotation=90, ha='center', va='center')
                if row == 0:
                    fig.text((left+style['panel_width_mm']/2)/width, style['time_heading_y_mm']/height,
                        f'{spec["time_label"]} = {spec["dimensionless_times"][col]:.2f}',
                        ha='center', va='center', fontsize=style['heading_size'])
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
    receipt = dict(schema='spinodal_main_figure_validation_v1', status='passed',
        renderer=str(Path(__file__).relative_to(REPO)), specification=spec,
        versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        records=records, text=texts, png_pixels=list(rgb.size),
        layout=dict(field_vertical_bounds_mm=[field_bottom, field_top], colorbar_rect_mm=bar_rect,
                    colorbar_center_mm=middle, colorbar_label_position='top, horizontal', aspect='equal'),
        output_bytes=dict(pdf=pdf.stat().st_size, png=png.stat().st_size),
        metric_interpolation=False, temporal_interpolation=False, inference=False,
        layers=dict(field='raster, area-weighted cell-to-vertex projection and original-triangle Gouraud',
                    reference_contours='vector, same display reconstruction of native reference cells',
                    boundaries='vector ellipse/circles; white holes',
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
    if specification['schema'] != 'spinodal_main_figure_v1':
        raise ValueError('Unsupported figure specification')
    render(specification)
