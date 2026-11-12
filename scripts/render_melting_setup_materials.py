"""Material-aware orthographic melting setup from a prescribed 3D ellipsoid."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from matplotlib.text import Text
from fontTools.ttLib import TTFont
import numpy as np

from examples.icemelting3d.ellipsoids import rotation, validate_parameters

REPO = Path(__file__).resolve().parents[1]


def unit_sphere(nlongitude, nlatitude):
    """Closed UV sphere with unique poles and explicit triangle fans."""
    theta = np.arange(nlongitude)*2*np.pi/nlongitude
    latitude = np.linspace(-np.pi/2, np.pi/2, nlatitude+1)[1:-1]
    tt, ll = np.meshgrid(theta, latitude)
    rings = np.stack((np.cos(ll)*np.cos(tt), np.cos(ll)*np.sin(tt), np.sin(ll)), -1)
    vertices = np.vstack(([0, 0, -1], rings.reshape(-1, 3), [0, 0, 1]))
    faces = []
    for j in range(nlongitude):
        k = (j+1) % nlongitude
        faces.append([0, 1+j, 1+k])
        for i in range(len(latitude)-1):
            a, b = 1+i*nlongitude+j, 1+i*nlongitude+k
            faces.extend([[a, a+nlongitude, b+nlongitude], [a, b+nlongitude, b]])
        offset = 1+(len(latitude)-1)*nlongitude
        faces.append([len(vertices)-1, offset+k, offset+j])
    return vertices, np.asarray(faces)


def render(spec):
    if spec['schema'] != 'melting_material_setup_v1' or spec['sample_index'] != 2:
        raise ValueError('Require the third already-prescribed illustrative ellipsoid')
    source = json.loads((REPO/spec['source']).read_text())['cases'][spec['case']]
    parameters = np.asarray(source['parameters'][spec['sample_index']])
    validate_parameters(parameters)
    initial = source['initial']
    center, axes = parameters[:3], parameters[3:6]
    if (np.any(center < initial['center_min']) or np.any(center > initial['center_max'])
            or np.any(axes < initial['axis_min']) or np.any(axes > initial['axis_max'])):
        raise ValueError('Ellipsoid outside the documented initial-condition prior')
    rot = rotation(parameters[6:], np)
    style = spec['style']
    view = np.asarray(style['view_direction'], dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross([0, 0, 1], view); right /= np.linalg.norm(right)
    up = np.cross(view, right)
    projection = np.stack((right, up))
    np.testing.assert_allclose(projection@projection.T, np.eye(2), atol=1e-14)
    scale = style['world_scale_mm']
    anchor = np.asarray(style['cube_center_mm'])

    def screen(points):
        return (np.asarray(points)-.5)@projection.T*scale+anchor

    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    edges = [(i, j) for i in range(8) for j in range(i+1, 8) if np.sum(cube[i] != cube[j]) == 1]
    unit, faces = unit_sphere(style['longitude_segments'], style['latitude_segments'])
    vertices = (unit*axes)@rot.T+center
    np.testing.assert_allclose(np.sum(((vertices-center)@rot/axes)**2, axis=1), 1, atol=1e-14)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normal, axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError('Degenerate ellipsoid surface triangle')
    normal /= lengths[:, None]
    normal *= np.where(np.sum(normal*(triangles.mean(1)-center), axis=1) > 0, 1., -1.)[:, None]
    front = normal@view > 0
    light = np.asarray(style['light_direction'], float); light /= np.linalg.norm(light)
    half = light+view; half /= np.linalg.norm(half)
    intensity = style['ambient']+style['diffuse']*np.maximum(normal@light, 0)
    highlight = style['highlight']*np.maximum(normal@half, 0)**style['highlight_power']
    colors = np.array(to_rgb(style['ice']))[None, :]*intensity[:, None]
    colors = np.clip(colors+(1-colors)*highlight[:, None], 0, 1)
    face_rgba = np.c_[colors, np.full(len(colors), style['ice_opacity'])]
    # Exact orthographic silhouette of the rotated ellipsoid.
    matrix = projection@rot@np.diag(axes)
    covariance = matrix@matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    theta = np.linspace(0, 2*np.pi, 721)
    outline = np.c_[np.cos(theta), np.sin(theta)]@np.diag(np.sqrt(eigenvalues))@eigenvectors.T
    outline = outline*scale+screen(center)
    annotations = spec['annotations']
    target_direction = np.asarray(annotations['initial_target_direction'], dtype=float)
    target_direction /= np.linalg.norm(target_direction)
    initial_target = (screen(center)+covariance@target_direction
        /np.sqrt(target_direction@covariance@target_direction)*scale
        +target_direction*annotations['gap_mm'])
    boundary_target = screen(annotations['boundary_target_xyz'])+[annotations['gap_mm'], 0]
    font = Path(style['font_file']).resolve(strict=True)
    with TTFont(font) as face:
        if face['name'].getDebugName(1) != 'Helvetica' or 'glyf' not in face:
            raise ValueError('Require TrueType Helvetica')
        if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError('Font embedding is restricted')
    for key in ('math_font_file', 'math_italic_file'):
        Path(style[key]).resolve(strict=True)
    font_manager.fontManager.addfont(str(font))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font:
        raise ValueError('Unexpected font substitution')
    width, height = style['figure_mm']
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size_pt'],
          'text.color': style['text_color'], 'pdf.fonttype': 42, 'ps.fonttype': 42,
          'pdf.use14corefonts': False, 'svg.fonttype': 'none',
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None, 'text.usetex': False}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height), aspect='equal')
        ax.axis('off')

        def label(x, y, text):
            ax.text(x, y, text, ha='center', va='center', linespacing=1.2,
                fontproperties=font_manager.FontProperties(fname=str(font), size=style['font_size_pt']),
                color=style['text_color'], zorder=9)

        def arrow(start, end):
            ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>',
                'color': style['text_color'], 'lw': .5, 'mutation_scale': 5,
                'shrinkA': .5, 'shrinkB': .5}, zorder=8)

        far_faces = [[0, 1, 3, 2], [2, 3, 7, 6], [0, 2, 6, 4]]
        for face, color in zip(far_faces, style['water_faces'], strict=True):
            ax.add_patch(Polygon(screen(cube[face]), facecolor=color, edgecolor='none', zorder=1))
        for i, j in edges:
            fixed = np.flatnonzero(cube[i] == cube[j])
            visible = any((2*cube[i, k]-1)*view[k] > 0 for k in fixed)
            ax.plot(*screen(cube[[i, j]]).T, color=style['cube_edge'],
                    lw=.65 if visible else .45, linestyle='-' if visible else (0, (3, 3)),
                    alpha=1 if visible else .5, zorder=6 if visible else 2)
        ax.add_collection(PolyCollection(screen(triangles[front]), facecolors=face_rgba[front],
            edgecolors='none', linewidths=0, antialiased=False, rasterized=True, zorder=3))
        for omitted in range(3):
            circle = np.zeros((len(theta), 3)); dims = [j for j in range(3) if j != omitted]
            circle[:, dims] = np.c_[np.cos(theta), np.sin(theta)]
            curve = (circle*axes)@rot.T+center
            normals = (circle/axes)@rot.T
            xy = screen(curve); xy[normals@view <= 0] = np.nan
            ax.plot(*xy.T, color=style['ice_curve'], linewidth=.3, alpha=.55, zorder=4)
        ax.plot(*outline.T, color=style['ice_edge'], linewidth=.65, zorder=5)
        label(*annotations['initial_label_mm'], spec['labels']['initial'])
        arrow(annotations['initial_leader_start_mm'], initial_target)
        label(*screen(center), 'Ice')
        label(30, 25, 'Water')
        label(*annotations['boundary_label_mm'], spec['labels']['boundary'])
        arrow(annotations['boundary_leader_start_mm'], boundary_target)
        triad = np.asarray(style['triad_origin_mm'])
        for axis, name in enumerate(('x', 'y', 'z')):
            direction = projection[:, axis]
            tip = triad+direction*style['triad_length_mm']
            arrow(triad, tip)
            position = tip+direction/np.linalg.norm(direction)*2.3
            label(*position, name)
        label(79, 5, '1 × 1 × 1')
        fig.canvas.draw()
        texts = []
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(fig.canvas.get_renderer())
                if (not 5 <= text.get_fontsize() <= 7 or
                        not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1)):
                    raise ValueError(f'Invalid/clipped text: {text.get_text()}')
                texts.append({'text': text.get_text(), 'size_pt': text.get_fontsize()})
        sizes = {}
        for kind in ('pdf', 'png', 'svg'):
            out = Path(spec['output'][kind]); out.parent.mkdir(parents=True, exist_ok=True)
            metadata = {'CreationDate': None, 'ModDate': None} if kind == 'pdf' else None
            fig.savefig(out, dpi=style['dpi'], metadata=metadata)
            sizes[kind] = out.stat().st_size
        plt.close(fig)
    receipt = {'schema': 'melting_material_setup_validation_v1', 'specification': spec,
        'renderer': str(Path(__file__).relative_to(REPO)), 'parameters': parameters.tolist(),
        'rotation': rot.tolist(), 'projection': projection.tolist(), 'cube_vertices_mm': screen(cube).tolist(),
        'projected_ice_center_mm': screen(center).tolist(), 'silhouette_covariance': covariance.tolist(),
        'mesh_vertices': len(vertices), 'mesh_faces': len(faces), 'visible_faces': int(front.sum()),
        'leaders': {'initial_target_mm': initial_target.tolist(),
            'boundary_target_mm': boundary_target.tolist(),
            'initial_length_mm': float(np.linalg.norm(initial_target-annotations['initial_leader_start_mm'])),
            'boundary_length_mm': float(np.linalg.norm(boundary_target-annotations['boundary_leader_start_mm']))},
        'text': texts, 'output_bytes': sizes,
        'interpretation': 'Prescribed initial phi=0 surface; decorative ice shading, no temperature or evolved field',
        'layers': {'ice_shading': 'raster', 'cube_surface_curves_and_text': 'vector'},
        'training': False, 'inference': False, 'pde_solves': 0,
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': spec['output'], 'bytes': sizes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.spec.read_text()))
