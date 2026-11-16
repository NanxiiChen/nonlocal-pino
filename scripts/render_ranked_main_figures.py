"""Best / median-near / worst phase galleries from declared test cohorts.

Ranks original evaluation arrays, never the interpolated display or a slice.
PDF text and contours remain vector objects; dense fields alone are rasterized.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Ellipse
from matplotlib.text import Text
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from fontTools.ttLib import TTFont
import numpy as np

from scripts.prepare_ranked_main_fields import read, write, error, select
from scripts.render_electropolish_main_figure import read_arrays, scientific_error_label
from scripts.render_channel_main_figure import normalized_time_heading, load_fields as channel_fields
from scripts.render_particle_corrosion_main_figure import hole_edges
from scripts.render_spinodal_main_figure import display_vertices
from scripts.render_melting_main_figure import surface

REPO = Path(__file__).resolve().parents[1]
SPECS = dict(electropolish='electropolish_main_figure.json', ceramic='particle_corrosion_main_figure.json',
             spinodal='spinodal_main_figure.json', channel='channel_main_figure_icefire_r.json',
             melting='melting_main_figure.json')
NAMES = dict(electropolish='electropolish2d', ceramic='particle_corrosion2d', spinodal='spinodal2d',
             channel='two_phase_flow2d', melting='icemelting3d')


def times_at(saved, requested):
    indices = []
    for time in requested:
        found = np.flatnonzero(np.isclose(saved, time, rtol=0, atol=1e-12))
        if len(found) != 1:
            raise ValueError(f'Need exactly one saved frame at {time}')
        indices.append(int(found[0]))
    return indices


def load(case, spec, root):
    """Return complete ranking, selected original-node scores and display fields."""
    selected_fields = []
    if case == 'electropolish':
        data, report = read_arrays(spec['prediction']), read(spec['evaluation'])
        if report['status'] != 'passed' or report['model'] != spec['checkpoint'] or report['count'] != 64:
            raise ValueError('Wrong saved Electropolish model/cohort')
        refroot = Path(spec['references'][0]['file']).parent.parent
        metric_indices = report['regular_time_indices']
        np.testing.assert_allclose(data['time'][metric_indices], np.linspace(0, 1, 21), atol=1e-12)
        records = []
        for i in range(64):
            path = refroot/f'condition_{i:03d}/local101.npz'
            r = read_arrays(path)
            for key in ('X', 'Y', 'time'):
                np.testing.assert_array_equal(data[key], r[key])
            meta = json.loads(str(r['metadata_json']))
            if meta['index'] != i or meta['surface_id'] != data['surface_ids'][i]:
                raise ValueError('Electropolish condition identity mismatch')
            score = error(data['prediction'][i, metric_indices, ..., 0], r['fields'][metric_indices, ..., 0])
            np.testing.assert_allclose(score, report['records'][i]['local101_64']['pooled_phi'], atol=2e-10, rtol=0)
            records.append(dict(index=i, prediction=spec['prediction'], reference=str(path), error=score))
        selection = select(records)
        ts = spec['normalized_times']; take = times_at(data['time'], ts)
        for item in selection:
            r = read_arrays(item['reference'])
            selected_fields.append(dict(p=data['prediction'][item['index'], take, ..., 0],
                r=r['fields'][take, ..., 0], x=data['X'][0], y=data['Y'][:, 0], extent=[1, .5]))
        metric_times = data['time'][metric_indices].tolist()
    elif case == 'ceramic':
        ranking = read(root/'ceramic_ranking.json')
        if ranking['checkpoint'] != spec['checkpoint'] or ranking['count'] != 64:
            raise ValueError('Wrong Ceramic weights or cohort')
        records, selection = ranking['records'], ranking['selected']
        c = read(spec['config']); extent = [c['geometry']['length'], c['geometry']['height']]
        ts = spec['normalized_times']
        for item in selection:
            p, r = read_arrays(item['prediction']), read_arrays(item['reference'])
            np.testing.assert_array_equal(p['points'], r['points']); np.testing.assert_array_equal(p['times'], r['times'])
            np.testing.assert_allclose(error(p['prediction'][..., 0], r['fields'][..., 0]), item['error'], atol=1e-13)
            take = times_at(r['times'], ts); points = r['points']; cells = r['triangles']
            edges = hole_edges(points, cells, extent, c['particles']['count'])
            selected_fields.append(dict(p=p['prediction'][take, ..., 0], r=r['fields'][take, ..., 0],
                tri=Triangulation(points[:, 0], points[:, 1], cells), edges=edges, extent=extent))
        metric_times = r['times'].tolist()
    elif case == 'spinodal':
        manifest = read(spec['reference_manifest']); refroot = Path(spec['reference_manifest']).parent
        c = read(spec['config']); records = []
        for i in manifest['test_indices']:
            pp, rr = Path(spec['run'])/f'prediction_{i:03d}.npz', refroot/f'case_{i:04d}/reference.npz'
            p, r = read_arrays(pp), read_arrays(rr)
            for key in ('points', 'mesh_points', 'mesh_cells', 'volumes', 'time', 'holes'):
                np.testing.assert_array_equal(p[key], r[key])
            np.testing.assert_array_equal(p['phi_reference'], r['phi'])
            records.append(dict(index=i, prediction=str(pp), reference=str(rr), error=error(p['phi'], r['phi'])))
        selection = select(records); ts = spec['dimensionless_times']
        for item in selection:
            p, r = read_arrays(item['prediction']), read_arrays(item['reference'])
            take = times_at(p['time'], ts); vertices, cells = p['mesh_points'], p['mesh_cells']
            selected_fields.append(dict(p=p['phi'][take], r=r['phi'][take],
                pd=display_vertices(p['phi'][take], cells, p['volumes'], len(vertices)),
                rd=display_vertices(r['phi'][take], cells, p['volumes'], len(vertices)),
                tri=Triangulation(vertices[:, 0], vertices[:, 1], cells), holes=p['holes'],
                center=c['geometry']['center'], axes=c['geometry']['axes']))
        metric_times = p['time'].tolist()
    elif case == 'channel':
        extent, original_fields, original_records = channel_fields(spec)
        records = [dict(index=i, prediction=r['prediction'], reference=r['reference'],
                        error=r['pooled_phase_check']) for i, r in enumerate(original_records[1:])]
        selection = select(records); ts = (np.asarray(spec['times'])/read(spec['config'])['physical']['terminal_time']).tolist()
        for item in selection:
            x, y, wx, wb, wt, p, r = original_fields[item['index']+1]
            selected_fields.append(dict(p=p, r=r, x=x, y=y, wx=wx, wb=wb, wt=wt, extent=extent))
        metric_times = read_arrays(records[0]['prediction'])['time'].tolist()
    elif case == 'melting':
        ranking = read(root/'melting_ranking.json')
        if ranking['checkpoint'] != spec['checkpoint'] or ranking['count'] != 64:
            raise ValueError('Wrong Melting weights or cohort')
        audit = read(ranking['replay_audit'])
        if ranking['backend'] != 'gpu' or not audit['original_pointwise_tolerance_passed']:
            raise ValueError('Publish only qualified original-backend Melting inference')
        records, selection = ranking['records'], ranking['selected']; ts = spec['times']
        for item in selection:
            p, r = read_arrays(item['prediction']), read_arrays(item['reference'])
            d = read_arrays(root/f"melting_dense_{item['index']:03d}.npz")
            np.testing.assert_array_equal(p['points'], r['points']); np.testing.assert_array_equal(p['time'], r['time'])
            np.testing.assert_array_equal(p['parameters'], r['parameters'])
            np.testing.assert_allclose(error(p['prediction'], r['phi']), item['error'], atol=1e-13)
            n = spec['surface_axis_points']; x, y, z = r['slice_points'].reshape(n, n, 3).transpose(2, 0, 1)
            np.testing.assert_allclose(z, spec['slice_z'])
            selected_fields.append(dict(p=p['prediction'], r=r['phi'], x=x, y=y,
                pd=d['slice_prediction'].reshape(len(ts), n, n), rd=r['slice_phi'].reshape(len(ts), n, n),
                meshes=[root/f"melting_mesh_{item['index']:03d}_{k}.npz" for k in range(len(ts))]))
        metric_times = r['time'].tolist()
    else:
        raise ValueError(case)
    for field, item in zip(selected_fields, selection, strict=True):
        item['panel_errors'] = [error(p, r) for p, r in zip(field['p'], field['r'], strict=True)]
    return records, selection, selected_fields, ts, metric_times


def render(case, root):
    spec = read(REPO/'artifacts'/SPECS[case]); style = deepcopy(spec['style'])
    if case in ('electropolish', 'ceramic'):
        style['colorbar_ticks'] = [0, .5, 1]
    records, selection, fields, times, metric_times = load(case, spec, root)
    for key in ('font_file', 'math_italic_font_file'):
        path = Path(style[key]).resolve(strict=True)
        with TTFont(path) as face:
            if 'glyf' not in face or face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
                raise ValueError('Font embedding not permitted')
        font_manager.fontManager.addfont(str(path))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != Path(style['font_file']).resolve():
        raise ValueError('Helvetica substitution')
    if case != 'melting':
        pitch = style['row_bottoms_mm'][0]-style['row_bottoms_mm'][1]
        style['figure_mm'][1] -= pitch
        style['time_heading_y_mm'] -= pitch
        style['row_bottoms_mm'] = style['row_bottoms_mm'][1:]
    width, height = style['figure_mm']
    if case == 'channel':
        colors = np.asarray(read(REPO/style['cmap_file'])['colors'])
        cmap = ListedColormap(colors[::-1], name='icefire_r')
    else:
        cmap = style['cmap']
    rc = {'font.family': 'Helvetica', 'font.size': 6, 'text.color': '#222222',
          'axes.labelcolor': '#222222', 'xtick.color': '#222222', 'ytick.color': '#222222',
          'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False,
          'text.usetex': False, 'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'axes.linewidth': style['axes_linewidth']}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        for row, (f, item) in enumerate(zip(fields, selection, strict=True)):
            bottom = style['slice_bottoms_mm'][row] if case == 'melting' else style['row_bottoms_mm'][row]
            panel_height = style['panel_width_mm'] if case == 'melting' else style['panel_height_mm']
            for col in range(5):
                left = style['left_mm']+col*(style['panel_width_mm']+style['column_gap_mm'])
                if case == 'melting':
                    top = style['surface_bottoms_mm'][row]
                    ax3 = fig.add_axes([left/width, top/height, style['panel_width_mm']/width,
                        style['surface_height_mm']/height], projection='3d', computed_zorder=False)
                    ax3.set_proj_type('ortho'); ax3.view_init(style['surface_elevation'], style['surface_azimuth'])
                    ax3.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1)); ax3.set_box_aspect((1, 1, 1), zoom=style['surface_zoom'])
                    ax3.set_axis_off(); mesh = read_arrays(f['meshes'][col])
                    if col == 0:
                        a, b = style['slice_cue_bounds']; z = spec['slice_z']
                        corners = np.array([[a,a,z], [b,a,z], [b,b,z], [a,b,z], [a,a,z]])
                        ax3.add_collection3d(Line3DCollection([corners], colors=style['slice_cue_color'],
                            linewidths=style['slice_cue_width'], linestyles=':', zorder=2), autolim=False)
                    if len(mesh['faces']):
                        surface(ax3, mesh['vertices'], mesh['faces'], style)
                    else:
                        ax3.text2D(.5, .45, 'Melted', transform=ax3.transAxes, ha='center', va='center', fontsize=6)
                ax = fig.add_axes([left/width, bottom/height, style['panel_width_mm']/width, panel_height/height])
                p = f['pd'][col] if case in ('spinodal', 'melting') else f['p'][col]
                r = f['rd'][col] if case in ('spinodal', 'melting') else f['r'][col]
                lo, hi = style['limits']
                if p.min() < lo-1e-6 or p.max() > hi+1e-6:
                    raise ValueError('Field exceeds shared colormap limits')
                contour = dict(levels=[.5 if case in ('electropolish', 'ceramic') else 0],
                               colors=style['contour_color'], linewidths=style['contour_linewidth'],
                               linestyles=style['contour_linestyle'])
                if case == 'electropolish':
                    image = ax.imshow(p, origin='lower', extent=[0, 1, 0, .5], interpolation='bilinear',
                                      cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
                    ax.contour(f['x'], f['y'], r, **contour)
                    ax.set(xlim=(0, 1), ylim=(0, .5))
                elif case in ('ceramic', 'spinodal'):
                    image = ax.tripcolor(f['tri'], p, shading='gouraud', cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
                    ax.tricontour(f['tri'], r, **contour)
                    if case == 'ceramic':
                        ax.set_facecolor(style['hole_color'])
                        ax.add_collection(LineCollection(f['edges'], colors=style['hole_edge_color'],
                            linewidths=style['hole_linewidth'], zorder=4))
                        ax.set(xlim=(0, f['extent'][0]), ylim=(f['extent'][1], 0))
                    else:
                        for x, y, radius in f['holes']:
                            ax.add_patch(Circle((x, y), radius, facecolor=style['hole_color'],
                                edgecolor=style['hole_edge_color'], lw=style['hole_linewidth'], zorder=4))
                        center, axes = np.asarray(f['center']), np.asarray(f['axes'])
                        ax.add_patch(Ellipse(center, 2*axes[0], 2*axes[1], fill=False,
                            edgecolor=style['hole_edge_color'], lw=style['hole_linewidth'], clip_on=False))
                        ax.set(xlim=(center[0]-axes[0], center[0]+axes[0]), ylim=(center[1]-axes[1], center[1]+axes[1]))
                        ax.set_axis_off()
                else:
                    image = ax.pcolormesh(f['x'], f['y'], p, shading='gouraud', cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
                    level = contour['levels'][0]
                    if r.min() < level < r.max():
                        ax.contour(f['x'], f['y'], r, **contour)
                    if case == 'channel':
                        ax.fill_between(f['wx'], 0, f['wb'], color=style['wall_color'], lw=0, zorder=3)
                        ax.fill_between(f['wx'], f['wt'], f['extent'][1], color=style['wall_color'], lw=0, zorder=3)
                        for wall in (f['wb'], f['wt']):
                            ax.plot(f['wx'], wall, color=style['wall_edge_color'], lw=style['wall_linewidth'], zorder=4)
                        ax.set(xlim=(0, f['extent'][0]), ylim=(0, f['extent'][1]))
                    else:
                        ax.set(xlim=(0, 1), ylim=(0, 1))
                ax.set(aspect='equal', xticks=[], yticks=[])
                label = scientific_error_label(fig, left+style['panel_width_mm']/2,
                    bottom-style['error_gap_mm'], item['panel_errors'][col], style)
                if label['width_mm'] > style['panel_width_mm']:
                    raise ValueError('Panel error label overflows')
                if row == 0:
                    normalized_time_heading(fig, left+style['panel_width_mm']/2, times[col], style)
            middle = (bottom+style['surface_bottoms_mm'][row]+style['surface_height_mm'])/2 if case == 'melting' else bottom+panel_height/2
            fig.text(style['row_label_x_mm']/width, middle/height, item['rank_label'], rotation=90, ha='center', va='center')
            if case == 'melting':
                for label, y in [('3D', style['surface_bottoms_mm'][row]+style['surface_height_mm']/2), ('XY', bottom+panel_height/2)]:
                    fig.text(style['row_label_x_mm']/width, y/height, label, ha='right', va='center', fontsize=5)
        low = min(style['slice_bottoms_mm']) if case == 'melting' else min(style['row_bottoms_mm'])
        high = max(style['surface_bottoms_mm'])+style['surface_height_mm'] if case == 'melting' else max(style['row_bottoms_mm'])+style['panel_height_mm']
        bh = min(style['colorbar_height_mm'], high-low-4)
        bar = fig.colorbar(image, cax=fig.add_axes([style['colorbar_x_mm']/width,
            ((low+high-bh)/2)/height, style['colorbar_width_mm']/width, bh/height]), ticks=style['colorbar_ticks'])
        bar.ax.set_title('φ', fontsize=7, pad=style['colorbar_label_pad_pt'])
        bar.ax.tick_params(direction='in', width=style['axes_linewidth'], length=2, pad=2)
        bar.solids.set_rasterized(True)
        for obj in fig.findobj(Text):
            if obj.get_visible() and obj.get_text():
                key = 'math_italic_font_file' if obj.get_gid() == 'sans-math-italic' else 'font_file'
                obj.set_fontproperties(font_manager.FontProperties(fname=style[key], size=obj.get_fontsize()))
                obj.set_color('#222222')
        fig.canvas.draw(); texts = []
        for obj in fig.findobj(Text):
            if obj.get_visible() and obj.get_text():
                box = obj.get_window_extent(fig.canvas.get_renderer())
                if not 5 <= obj.get_fontsize() <= 7 or not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
                    raise ValueError('Clipped or undersized label: '+obj.get_text())
                texts.append(dict(text=obj.get_text(), size=obj.get_fontsize(), font=obj.get_fontproperties().get_file()))
        outputs = {ext: str(root/f'{NAMES[case]}_phase_best_middle_worst.{ext}') for ext in ('pdf', 'png')}
        fig.savefig(outputs['pdf'], dpi=style['dpi'], metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(outputs['png'], dpi=style['png_dpi']); plt.close(fig)
    result = dict(case=case, original_spec=str(REPO/'artifacts'/SPECS[case]), checkpoint=spec['checkpoint'],
        config=spec['config'], count=len(records), records=records, selected=selection, outputs=outputs, style=style,
        metric='Phase only; equal-point L2 over the full declared space-time evaluation grid', metric_times=metric_times,
        selection='Sort by (phase error, sample ID); first, lower central actual sample, last',
        panel_metric='Instantaneous original spatial-node phi relative L2; Melting uses full 3D volume, not XY slice',
        text=texts, numpy_version=np.__version__, matplotlib_version=matplotlib.__version__)
    write(root/f'{case}_figure.json', result)
    print(json.dumps(dict(case=case, count=len(records), selected=[(r['index'], r['error']) for r in selection], outputs=outputs)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', choices=tuple(SPECS), required=True)
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    render(args.case, args.root)
