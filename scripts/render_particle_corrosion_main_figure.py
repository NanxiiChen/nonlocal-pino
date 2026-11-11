"""Render saved ceramic-particle PINO phase and local-FEM interface contours.

CPU-only. Original physical triangles prevent display interpolation across
particles. Errors use unweighted raw phase nodes, not raster pixels.
"""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.text import Text
from matplotlib.tri import Triangulation
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image

from scripts.render_electropolish_main_figure import (
    phase_error, read_arrays, read_json, scientific_error_label,
)

REPO = Path(__file__).resolve().parents[1]


def hole_edges(points, triangles, extent, expected_holes):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                   triangles[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold mesh edges')
    boundary = edges[counts == 1]
    xy = points[boundary]
    outer = np.zeros(len(boundary), bool)
    for dimension in (0, 1):
        for value in (0, extent[dimension]):
            outer |= np.all(np.isclose(xy[..., dimension], value, rtol=0, atol=1e-12), axis=1)
    holes = boundary[~outer]
    adjacency = {}
    for a, b in holes:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError('Particle boundary must be a union of closed cycles')
    remaining, components = set(adjacency), 0
    while remaining:
        frontier = [remaining.pop()]
        components += 1
        while frontier:
            for neighbor in adjacency[frontier.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    if components != expected_holes:
        raise ValueError(f'Expected {expected_holes} holes, found {components}')
    return points[holes]


def load_fields(spec):
    original = read_json(REPO/spec['source_figure_manifest'])['figures']['particle_corrosion2d']
    config, summary = read_json(spec['config']), read_json(spec['summary'])
    runtime = read_json(spec['runtime'])
    manifest = read_json(spec['reference_manifest'])
    prescribed = read_json(manifest['source_geometry'])
    if (spec['run'] != original['run'] or spec['checkpoint'] != original['checkpoint']
            or not Path(spec['checkpoint']).is_file() or summary['steps'] != spec['checkpoint_step']
            or summary['final']['step'] != 20000 or spec['samples'] != [0, 1, 2, 3]
            or config['train']['iterations'] != spec['checkpoint_step']
            or runtime['seed'] != spec['runtime_seed']
            or Path(runtime['reference']) != Path(spec['reference_manifest']).parent
            or manifest['backend'] != 'local_p1_lumped_fem'
            or spec['phase_channel'] != 0 or spec['interface_level'] != .5):
        raise ValueError('Unexpected main-figure model, samples or reference')
    for key in ('geometry', 'geometry_si', 'particles', 'physical', 'nondim'):
        if config[key] != manifest['config'][key] or config[key] != prescribed['config'][key]:
            raise ValueError(f'Physical input configuration mismatch: {key}')
    extent = (config['geometry']['length'], config['geometry']['height'])
    fields, records = [], []
    for sample, source in zip(spec['samples'], spec['records'], strict=True):
        entry = manifest['cases'][sample]
        old = original['validated_local_cohort_records'][sample]
        if (source['index'] != sample or entry['index'] != sample
                or source['prediction'] != old['source'] or source['reference'] != old['reference']
                or entry['geometry'] != prescribed['samples'][sample]['geometry']):
            raise ValueError('Model/reference geometry or saved-input order mismatch')
        pred, ref = read_arrays(source['prediction']), read_arrays(source['reference'])
        for key in ('points', 'times'):
            np.testing.assert_array_equal(pred[key], ref[key])
        prediction, target = pred['prediction'][..., 0], ref['fields'][..., 0]
        xy, triangles = ref['points'], ref['triangles']
        if (prediction.shape != target.shape or prediction.shape != (len(ref['times']), len(xy))
                or not np.isfinite(prediction).all() or not np.isfinite(target).all()
                or not np.isfinite(xy).all() or triangles.ndim != 2 or triangles.shape[1] != 3
                or not np.issubdtype(triangles.dtype, np.integer)
                or triangles.min() < 0 or triangles.max() >= len(xy)
                or np.any(ref['volumes'] <= 0) or np.any(np.diff(ref['times']) <= 0)):
            raise ValueError('Invalid physical fields, mesh or times')
        np.testing.assert_allclose(xy.min(axis=0), [0, 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(xy.max(axis=0), extent, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ref['physical_time'], ref['times']*config['nondim']['T_c'],
                                   rtol=0, atol=1e-12)
        # Historical scores use quadrature; verify them without reusing that
        # different reduction as the new per-panel equal-node metric.
        tw = np.r_[np.diff(ref['times'])[0], np.diff(ref['times'])[:-1]
                   +np.diff(ref['times'])[1:], np.diff(ref['times'])[-1]]/2
        weights = tw[:, None]*ref['volumes'][None, :]
        pooled = float(np.sqrt(np.sum(weights*(prediction.astype(float)-target)**2)
                               /np.sum(weights*target**2)))
        np.testing.assert_allclose(pooled, summary['final']['conditions'][sample]['phi_pooled'],
                                   rtol=0, atol=2e-10)
        tri = Triangulation(xy[:, 0], xy[:, 1], triangles)
        edges = hole_edges(xy, triangles, extent, config['particles']['count'])
        centers = np.array([p['center'] for p in entry['geometry']['particles']])
        if np.any(tri.get_trifinder()(centers[:, 0], centers[:, 1]) != -1):
            raise ValueError('A particle interior is covered by a physical triangle')
        indices, panels = [], []
        for time in spec['normalized_times']:
            found = np.flatnonzero(np.isclose(ref['times'], time, rtol=0, atol=1e-14))
            if len(found) != 1:
                raise ValueError(f'Require an actual saved snapshot at {time}')
            index = int(found[0])
            indices.append(index)
            if (prediction[index].min() < spec['style']['limits'][0]
                    or prediction[index].max() > spec['style']['limits'][1]
                    or not target[index].min() < .5 < target[index].max()):
                raise ValueError('Phase exceeds the display range or reference has no interface')
            panels.append(dict(time_index=index, normalized_time=time,
                physical_time_seconds=time*config['nondim']['T_c'],
                prediction_range=[float(prediction[index].min()), float(prediction[index].max())],
                **phase_error(prediction[index], target[index])))
        fields.append((tri, edges, prediction[indices], target[indices]))
        records.append(dict(index=sample, **{k: source[k] for k in ('prediction', 'reference')},
                            geometry=entry['geometry'], nodes=len(xy), triangles=len(triangles),
                            hole_edges=len(edges), historical_pooled_phase_check=pooled, panels=panels))
    return extent, fields, records


def render(spec):
    extent, fields, records = load_fields(spec)
    style = spec['style']
    font_path = Path(style['font_file']).resolve(strict=True)
    math_path = Path(style['math_italic_font_file']).resolve(strict=True)
    for path, family in ((font_path, 'Helvetica'), (math_path, 'DejaVu Sans')):
        with TTFont(path) as face:
            if face['name'].getDebugName(1) != family or 'glyf' not in face:
                raise ValueError(f'Unexpected font file: {path}')
            if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
                raise ValueError('Font flags do not permit this embedding')
        font_manager.fontManager.addfont(str(path))
    resolved = Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve()
    if resolved != font_path:
        raise ValueError('Unexpected font substitution')
    width, height = style['figure_mm']
    np.testing.assert_allclose(style['panel_width_mm']/style['panel_height_mm'], extent[0]/extent[1])
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size'],
          'text.color': style['text_color'], 'axes.labelcolor': style['text_color'],
          'xtick.color': style['text_color'], 'ytick.color': style['text_color'],
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'text.usetex': False, 'axes.linewidth': style['axes_linewidth']}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        for row, (tri, edges, predictions, references) in enumerate(fields):
            bottom = style['row_bottoms_mm'][row]
            for col, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
                left = style['left_mm']+col*(style['panel_width_mm']+style['column_gap_mm'])
                ax = fig.add_axes([left/width, bottom/height,
                    style['panel_width_mm']/width, style['panel_height_mm']/height])
                ax.set_facecolor(style['hole_color'])
                field = ax.tripcolor(tri, prediction, shading='gouraud', cmap=style['cmap'],
                    vmin=style['limits'][0], vmax=style['limits'][1], rasterized=True)
                contours = ax.tricontour(tri, reference, levels=[spec['interface_level']],
                    colors=style['contour_color'], linewidths=style['contour_linewidth'],
                    linestyles=style['contour_linestyle'])
                ax.add_collection(LineCollection(edges, colors=style['hole_edge_color'],
                    linewidths=style['hole_linewidth'], rasterized=False, zorder=3))
                ax.set(xlim=(0, extent[0]), ylim=(extent[1], 0), aspect='equal', xticks=[], yticks=[])
                panel = records[row]['panels'][col]
                panel['display_label'] = scientific_error_label(fig,
                    left+style['panel_width_mm']/2, bottom-style['error_gap_mm'], panel['relative_l2'], style)
                panel['reference_contour_segments'] = len(contours.allsegs[0])
                if panel['display_label']['width_mm'] > style['panel_width_mm']:
                    raise ValueError('Error label exceeds panel width')
                if row == 0:
                    fig.text((left+style['panel_width_mm']/2)/width,
                        style['time_heading_y_mm']/height, f't* = {spec["normalized_times"][col]:.2f}',
                        ha='center', va='center', fontsize=style['heading_size'])
            fig.text(style['row_label_x_mm']/width, (bottom+style['panel_height_mm']/2)/height,
                     f'Sample {row+1}', rotation=90, ha='center', va='center')
        field_bottom = min(style['row_bottoms_mm'])
        field_top = max(style['row_bottoms_mm'])+style['panel_height_mm']
        center = (field_bottom+field_top)/2
        bar_rect = [style['colorbar_x_mm'], center-style['colorbar_height_mm']/2,
                    style['colorbar_width_mm'], style['colorbar_height_mm']]
        bar = fig.colorbar(field, cax=fig.add_axes(np.array(bar_rect)/[width, height, width, height]),
                           ticks=[0, .5, 1])
        bar.ax.set_title('φ', fontsize=style['heading_size'], pad=style['colorbar_label_pad_pt'])
        bar.ax.tick_params(direction='in', width=style['axes_linewidth'], length=2, pad=2)
        bar.solids.set_rasterized(True)
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                font = math_path if text.get_gid() == 'sans-math-italic' else font_path
                text.set_fontproperties(font_manager.FontProperties(fname=str(font), size=text.get_fontsize()))
                text.set_color(style['text_color'])
        fig.canvas.draw()
        texts = []
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(fig.canvas.get_renderer())
            if (not 5 <= text.get_fontsize() <= 7 or not fig.bbox.contains(box.x0, box.y0)
                    or not fig.bbox.contains(box.x1, box.y1)):
                raise ValueError(f'Invalid text size or clipped label: {text.get_text()}')
            texts.append(dict(text=text.get_text(), size_pt=text.get_fontsize(),
                              font_file=text.get_fontproperties().get_file()))
        pdf, png = (Path(spec['output'][key]) for key in ('pdf', 'png'))
        for path in (pdf, png):
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, dpi=style['dpi'], metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(png, dpi=style['png_dpi'])
        plt.close(fig)
    with Image.open(png) as source:
        rgb = source.convert('RGB')
    rgb.save(png, optimize=True, dpi=(style['png_dpi'], style['png_dpi']))
    receipt = dict(schema='particle_corrosion_main_figure_validation_v1', status='passed',
        renderer=str(Path(__file__).relative_to(REPO)), specification=spec,
        versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        records=records, text=texts, png_pixels=list(rgb.size),
        layout=dict(field_vertical_bounds_mm=[field_bottom, field_top], colorbar_rect_mm=bar_rect,
                    colorbar_center_mm=center, colorbar_label_position='top, horizontal'),
        output_bytes=dict(pdf=pdf.stat().st_size, png=png.stat().st_size),
        metric_interpolation=False, temporal_interpolation=False, inference=False,
        layers=dict(field='raster, Gouraud display on original physical triangles',
                    reference_contours='vector, original local FEM nodes', holes='no field; gray, vector edges',
                    text='embedded editable Helvetica; DejaVu Sans Oblique L; 5pt superscripts',
                    colorbar='raster gradient, vector ticks/text'))
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(pdf=str(pdf), png=str(png), bytes=receipt['output_bytes'],
                         errors=[[p['relative_l2'] for p in r['panels']] for r in records]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    specification = read_json(args.spec)
    if specification['schema'] != 'particle_corrosion_main_figure_v1':
        raise ValueError('Unsupported figure specification')
    render(specification)
