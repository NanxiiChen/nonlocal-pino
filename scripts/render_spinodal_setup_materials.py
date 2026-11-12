"""Fixed spinodal initial texture, random holes and impermeable wall schematic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse
from matplotlib.text import Text
from fontTools.ttLib import TTFont
from ml_collections import ConfigDict
import numpy as np

from examples.confined_ch2d.operator_problem import fixed_initial, valid_holes

REPO = Path(__file__).resolve().parents[1]


def render(spec):
    if spec['schema'] != 'spinodal_material_setup_v1' or spec['sample_index'] != 0:
        raise ValueError('Require the first existing prescribed two-hole geometry')
    source = json.loads((REPO/spec['source']).read_text())['cases'][spec['case']]
    c = ConfigDict({'geometry': source['geometry'], 'initial': source['initial'],
                    'operator': source['prior']})
    holes = np.asarray(source['holes'][spec['sample_index']])
    if (tuple(c.geometry.center) != (.5, .5) or tuple(c.geometry.axes) != (.5, .4)
            or len(c.initial.waves) != 10 or c.initial.kind != 'fixed_ambient'
            or not valid_holes(holes, c)
            or np.any(holes[:, :2] < np.asarray(c.operator.center_min))
            or np.any(holes[:, :2] > np.asarray(c.operator.center_max))
            or np.any(holes[:, 2] < np.asarray(c.operator.radius_min))
            or np.any(holes[:, 2] > np.asarray(c.operator.radius_max))):
        raise ValueError('Require the documented fixed ten-mode field and admissible holes')
    style = spec['style']
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
    cx, cy = style['ellipse_center_mm']
    rx, ry = style['ellipse_semiaxes_mm']
    if not np.isclose(rx/ry, .5/.4):
        raise ValueError('Preserve physical ellipse aspect ratio')
    scale = rx/.5

    def point(xy):
        return (np.asarray(xy)-np.asarray(c.geometry.center))*scale+[cx, cy]

    nx, ny = style['field_grid']
    # Cell-centered analytic evaluations; no interpolation from a solution mesh.
    x = (np.arange(nx)+.5)/nx
    y = .1+(np.arange(ny)+.5)*.8/ny
    xx, yy = np.meshgrid(x, y)
    coords = np.stack((xx, yy), axis=-1)
    field = fixed_initial(coords, c, xp=np)
    inside = ((xx-.5)/.5)**2+((yy-.5)/.4)**2 <= 1
    for hx, hy, radius in holes:
        inside &= (xx-hx)**2+(yy-hy)**2 >= radius**2
    if (not np.isfinite(field).all() or field[inside].min() < style['limits'][0]
            or field[inside].max() > style['limits'][1]):
        raise ValueError('Initial texture exceeds fixed display limits')
    cmap = LinearSegmentedColormap.from_list('initial_fluctuations', style['field_colors'])
    cmap.set_bad('white')
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size_pt'],
          'text.color': style['text_color'], 'pdf.fonttype': 42, 'ps.fonttype': 42,
          'pdf.use14corefonts': False, 'svg.fonttype': 'none',
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'hatch.linewidth': .3, 'text.usetex': False}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height), aspect='equal')
        ax.axis('off')

        def label(xpos, ypos, text):
            return ax.text(xpos, ypos, text, ha='center', va='center', linespacing=1.2,
                fontproperties=font_manager.FontProperties(fname=str(font), size=style['font_size_pt']),
                color=style['text_color'], zorder=8)

        def arrow(start, end):
            ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>',
                'color': style['text_color'], 'lw': .5, 'mutation_scale': 5,
                'shrinkA': .5, 'shrinkB': .5}, zorder=8)

        thick = style['wall_thickness_mm']
        ax.add_patch(Ellipse((cx, cy), 2*(rx+thick), 2*(ry+thick), facecolor=style['wall'],
            edgecolor=style['wall_edge'], hatch='////', linewidth=.4, zorder=1))
        ellipse = Ellipse((cx, cy), 2*rx, 2*ry, facecolor='white', edgecolor='none', zorder=2)
        ax.add_patch(ellipse)
        texture = ax.imshow(np.ma.array(field, mask=~inside), origin='lower',
            extent=(cx-rx, cx+rx, cy-ry, cy+ry), cmap=cmap,
            vmin=style['limits'][0], vmax=style['limits'][1], interpolation='bilinear', zorder=3)
        texture.set_clip_path(ellipse)
        ax.add_patch(Ellipse((cx, cy), 2*rx, 2*ry, facecolor='none',
                             edgecolor=style['interface'], linewidth=.7, zorder=4))
        for hx, hy, radius in holes:
            rr = radius*scale
            if rr <= style['hole_wall_thickness_mm']:
                raise ValueError('Hole too small for schematic wall band')
            ax.add_patch(Circle(point([hx, hy]), rr, facecolor=style['wall'],
                edgecolor=style['interface'], hatch='////', linewidth=.7, zorder=5))
            ax.add_patch(Circle(point([hx, hy]), rr-style['hole_wall_thickness_mm'],
                facecolor='white', edgecolor=style['wall_edge'], linewidth=.3, zorder=6))
        label(45, 75.5, spec['labels']['initial'])
        arrow((40, 71.5), point([.39, .82]))
        label(49, 43, spec['labels']['holes'])
        for index, start in [(0, (49, 40)), (1, (47, 46))]:
            hx, hy, radius = holes[index]
            center = point([hx, hy])
            direction = np.asarray(start)-center
            target = center+direction/np.linalg.norm(direction)*(radius*scale+.25)
            arrow(start, target)
        label(26, 7, spec['labels']['boundary'])
        angle = np.deg2rad(235)
        arrow((26, 11.8), (cx+rx*np.cos(angle), cy+ry*np.sin(angle)))
        label(78, 4, '1 × 0.8')
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
    receipt = {'schema': 'spinodal_material_setup_validation_v1', 'specification': spec,
        'renderer': str(Path(__file__).relative_to(REPO)), 'holes': holes.tolist(),
        'initial': source['initial'], 'initial_range_in_domain': [float(field[inside].min()), float(field[inside].max())],
        'initial_range_ambient': [float(field.min()), float(field.max())],
        'physical_aspect': 1.25, 'text': texts, 'output_bytes': sizes,
        'geometry_validated': True, 'layers': {'field': 'raster analytic fixed initial fluctuations; masked holes',
        'walls_and_labels': 'vector; wall bands are illustrative within excluded regions'},
        'interpretation': 'Fixed small fluctuations, not evolved pure phases; only circular holes are randomized',
        'training': False, 'inference': False, 'pde_solves': 0,
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': spec['output'], 'bytes': sizes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.spec.read_text()))
