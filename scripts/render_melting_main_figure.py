"""Publication gallery: current Melting 3D interfaces and matched XY slices."""
import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image

from scripts.render_electropolish_main_figure import phase_error, scientific_error_label
from scripts.render_channel_main_figure import normalized_time_heading

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def surface(ax, vertices, faces, style):
    triangles = vertices[faces].astype(np.float64)
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    sizes = np.linalg.norm(normal, axis=1)
    if np.any(sizes <= 0):
        raise ValueError('Degenerate surface triangle')
    normal /= sizes[:, None]
    light = np.asarray(style['light_direction'], dtype=float)
    light /= np.linalg.norm(light)
    intensity = style['surface_ambient']+style['surface_diffuse']*np.maximum(normal@light, 0)
    colors = np.array(to_rgb(style['surface_color']))[None, :]*intensity[:, None]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors='none',
        linewidth=0, antialiased=False, zsort='average', rasterized=True, zorder=3), autolim=False)
    elev, azim = np.deg2rad([style['surface_elevation'], style['surface_azimuth']])
    view = np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), np.sin(elev)])
    front = normal@view > 0
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    np.testing.assert_array_equal(edges[::2], edges[1::2])
    boundary = vertices[edges[::2][front[owners[::2]] != front[owners[1::2]]]]
    ax.add_collection3d(Line3DCollection(boundary, colors='#385A69', linewidths=.35,
        zorder=4), autolim=False)


def render(spec):
    inference, meshes = read(REPO/spec['inference_receipt']), read(REPO/spec['mesh_receipt'])
    if (inference['status'] != 'passed' or meshes['status'] != 'passed'
            or inference['checkpoint'] != spec['checkpoint'] or inference['samples'] != spec['samples']):
        raise ValueError('Unqualified prepared fields/meshes')
    selection = read(REPO/spec['selection_record'])
    if selection['selected_indices'] != spec['samples']:
        raise ValueError('Changed predeclared random selection')
    config = read(spec['config'])
    if config['train']['iterations'] != spec['checkpoint_step'] or config['train']['seed'] != spec['runtime_seed']:
        raise ValueError('Model configuration mismatch')
    style = spec['style']; width, height = style['figure_mm']
    font_path, math_path = Path(style['font_file']).resolve(strict=True), Path(style['math_italic_font_file']).resolve(strict=True)
    for path, family in ((font_path, 'Helvetica'), (math_path, 'DejaVu Sans')):
        with TTFont(path) as face:
            if (face['name'].getDebugName(1) != family or 'glyf' not in face
                    or face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200)):
                raise ValueError('Invalid/unembeddable font')
        font_manager.fontManager.addfont(str(path))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font_path:
        raise ValueError('Unexpected Helvetica substitution')
    fields = []
    records = []
    for row, source in enumerate(spec['records']):
        sample = source['sample']
        if sample != spec['samples'][row]:
            raise ValueError('Input order mismatch')
        with np.load(Path(spec['prepared_root'])/f'case{sample:03d}-fields.npz', allow_pickle=False) as f:
            field = {k: f[k] for k in f.files}
        with np.load(source['reference'], allow_pickle=False) as f:
            for key, prepared in [('slice_points', 'slice_points'), ('slice_phi', 'slice_reference'),
                    ('parameters', 'parameters'), ('phi', 'metric_reference'), ('points', 'metric_points'), ('time', 'time')]:
                np.testing.assert_array_equal(field[prepared], f[key])
        with np.load(source['saved_prediction'], allow_pickle=False) as f:
            np.testing.assert_array_equal(field['metric_prediction'], f['prediction'])
        np.testing.assert_array_equal(field['time'], spec['times'])
        n = spec['surface_axis_points']
        x, y, z = field['slice_points'].reshape(n, n, 3).transpose(2, 0, 1)
        np.testing.assert_allclose(z, spec['slice_z'], rtol=0, atol=1e-14)
        np.testing.assert_allclose(x, np.broadcast_to(np.linspace(0, 1, n)[:, None], (n, n)), rtol=0, atol=1e-14)
        np.testing.assert_allclose(y, np.broadcast_to(np.linspace(0, 1, n)[None, :], (n, n)), rtol=0, atol=1e-14)
        pred = field['slice_prediction'].reshape(len(spec['times']), n, n)
        ref = field['slice_reference'].reshape(len(spec['times']), n, n)
        if not np.isfinite(pred).all() or min(pred.min(), ref.min()) < -1 or max(pred.max(), ref.max()) > 1:
            raise ValueError('Invalid phase range')
        field['slice_x'], field['slice_y'] = x, y
        field['slice_prediction_2d'], field['slice_reference_2d'] = pred, ref
        panels = [dict(time=t, **phase_error(field['metric_prediction'][j], field['metric_reference'][j]))
                  for j, t in enumerate(spec['times'])]
        fields.append(field)
        records.append(dict(**source, panels=panels, parameters=field['parameters'].tolist()))
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size'], 'text.color': style['text_color'],
          'axes.labelcolor': style['text_color'], 'xtick.color': style['text_color'], 'ytick.color': style['text_color'],
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False, 'text.usetex': False,
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None, 'axes.linewidth': style['axes_linewidth']}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        for row, data in enumerate(fields):
            bottom = style['slice_bottoms_mm'][row]
            top_bottom = style['surface_bottoms_mm'][row]
            for col, t in enumerate(spec['times']):
                left = style['left_mm']+col*(style['panel_width_mm']+style['column_gap_mm'])
                ax3 = fig.add_axes([left/width, top_bottom/height, style['panel_width_mm']/width,
                    style['surface_height_mm']/height], projection='3d', computed_zorder=False)
                ax3.set_proj_type('ortho')
                ax3.view_init(elev=style['surface_elevation'], azim=style['surface_azimuth'])
                ax3.set(xlim=style['surface_limits'], ylim=style['surface_limits'], zlim=style['surface_limits'])
                ax3.set_box_aspect((1, 1, 1), zoom=style['surface_zoom'])
                ax3.set_axis_off()
                if col == 0:
                    a, b = style['slice_cue_bounds']; z = spec['slice_z']
                    corners = np.array([[a, a, z], [b, a, z], [b, b, z], [a, b, z], [a, a, z]])
                    ax3.add_collection3d(Line3DCollection([corners], colors=style['slice_cue_color'],
                        linewidths=style['slice_cue_width'], linestyles=':', zorder=2), autolim=False)
                mesh = meshes['records'][row*len(spec['times'])+col]
                if mesh['sample'] != spec['samples'][row] or mesh['time'] != t:
                    raise ValueError('Mesh order mismatch')
                with np.load(mesh['file'], allow_pickle=False) as f:
                    surface(ax3, f['vertices'], f['faces'], style)
                ax = fig.add_axes([left/width, bottom/height,
                    style['panel_width_mm']/width, style['panel_width_mm']/height])
                image = ax.pcolormesh(data['slice_x'], data['slice_y'], data['slice_prediction_2d'][col],
                    shading='gouraud', cmap=style['cmap'], vmin=-1, vmax=1, rasterized=True)
                reference = data['slice_reference_2d'][col]
                if not reference.min() < 0 < reference.max():
                    raise ValueError('Selected slice lacks a reference zero contour')
                contour = ax.contour(data['slice_x'], data['slice_y'], reference, levels=[0],
                    colors=style['contour_color'], linewidths=style['contour_linewidth'],
                    linestyles=style['contour_linestyle'])
                ax.set(xlim=style['slice_limits'], ylim=style['slice_limits'], aspect='equal', xticks=[], yticks=[])
                panel = records[row]['panels'][col]
                panel['reference_contour_segments'] = len(contour.allsegs[0])
                panel['display_label'] = scientific_error_label(fig, left+style['panel_width_mm']/2,
                    bottom-style['error_gap_mm'], panel['relative_l2'], style)
                if panel['display_label']['width_mm'] > style['panel_width_mm']:
                    raise ValueError('Error annotation wider than panel')
                if row == 0:
                    normalized_time_heading(fig, left+style['panel_width_mm']/2, t, style)
            midpoint = (bottom+top_bottom+style['surface_height_mm'])/2
            fig.text(style['row_label_x_mm']/width, midpoint/height, f'Sample {row+1}',
                rotation=90, ha='center', va='center')
            for label, y_mm in [('3D', top_bottom+style['surface_height_mm']/2),
                                ('XY', bottom+style['panel_width_mm']/2)]:
                fig.text(style['row_label_x_mm']/width, y_mm/height, label, ha='right', va='center', fontsize=5)
        lower, upper = min(style['slice_bottoms_mm']), max(style['surface_bottoms_mm'])+style['surface_height_mm']
        middle = (lower+upper)/2
        bar_rect = [style['colorbar_x_mm'], middle-style['colorbar_height_mm']/2,
                    style['colorbar_width_mm'], style['colorbar_height_mm']]
        bar = fig.colorbar(image, cax=fig.add_axes(np.asarray(bar_rect)/[width, height, width, height]),
            ticks=style['colorbar_ticks'])
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
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(fig.canvas.get_renderer())
                if not 5 <= text.get_fontsize() <= 7 or not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
                    raise ValueError(f'Invalid font or clipped text: {text.get_text()}')
                texts.append(dict(text=text.get_text(), size_pt=text.get_fontsize(), font_file=text.get_fontproperties().get_file()))
        pdf, png = (Path(spec['output'][k]) for k in ('pdf', 'png'))
        fig.savefig(pdf, dpi=style['dpi'], metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(png, dpi=style['png_dpi'])
        plt.close(fig)
    with Image.open(png) as image:
        rgb = image.convert('RGB')
    rgb.save(png, optimize=True, dpi=(style['png_dpi'], style['png_dpi']))
    receipt = dict(schema='melting_main_figure_validation_v1', status='passed', specification=spec,
        renderer=str(Path(__file__).relative_to(REPO)), versions=dict(python=platform.python_version(),
            numpy=np.__version__, matplotlib=matplotlib.__version__), records=records, text=texts,
        png_pixels=list(rgb.size), output_bytes=dict(pdf=pdf.stat().st_size, png=png.stat().st_size),
        layout=dict(figure_mm=style['figure_mm'], colorbar_rect_mm=bar_rect,
            slice_aspect='equal', surface_aspect=[1, 1, 1], camera='fixed orthographic'),
        layers=dict(surfaces='raster diffuse shading; vector silhouettes',
            slices='raster physical phase', reference='vector zero contours', text='editable embedded TrueType'),
        smoothing=False, temporal_interpolation=False, metric_interpolation=False)
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(output=spec['output'], bytes=receipt['output_bytes'],
        errors=[[p['relative_l2'] for p in r['panels']] for r in records]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    if spec['schema'] != 'melting_main_figure_v1':
        raise ValueError('Unsupported Melting figure specification')
    render(spec)
