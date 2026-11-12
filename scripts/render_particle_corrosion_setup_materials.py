"""Material, initial-pit and boundary schematic for one prescribed ceramic layout."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text
from fontTools.ttLib import TTFont
from ml_collections import ConfigDict
import numpy as np

from examples.particle_corrosion2d.geometry import boundary, validate_geometry

REPO = Path(__file__).resolve().parents[1]


def render(spec):
    if spec['schema'] != 'particle_material_setup_v1' or spec['sample_index'] != 0:
        raise ValueError('Require the first already-prescribed ceramic layout')
    source = json.loads((REPO/spec['source']).read_text())['cases'][spec['case']]
    c = ConfigDict({k: source[k] for k in ('geometry', 'particles', 'validation')})
    geometry = source['records'][spec['sample_index']]
    checked = validate_geometry(geometry, c)
    g = c.geometry
    if (c.particles.count != 6 or not np.allclose(
            [g.length, g.height, g.pit_x, g.pit_radius, g.mouth_half_width],
            [1, .6, .5, .12, .055], atol=1e-14, rtol=0)
            or source['fixed']['L_c'] != 5e-5 or source['fixed']['T_c'] != 350):
        raise ValueError('Labels require the six-particle 50x30um / 350s physical setup')
    style = spec['style']
    font = Path(style['font_file']).resolve(strict=True)
    with TTFont(font) as face:
        if face['name'].getDebugName(1) != 'Helvetica' or 'glyf' not in face:
            raise ValueError('Require TrueType Helvetica')
        if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError('Font embedding is restricted')
    for name in ('math_font_file', 'math_italic_file'):
        Path(style[name]).resolve(strict=True)
    font_manager.fontManager.addfont(str(font))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font:
        raise ValueError('Unexpected font substitution')
    width, height = style['figure_mm']
    left, bottom, panel_width, panel_height = style['main_rect_mm']
    if not np.isclose(panel_width/panel_height, g.length/g.height):
        raise ValueError('Physical domain aspect changed')
    top = bottom+panel_height

    def point(xy):
        return np.asarray(xy)*[panel_width/g.length, -panel_width/g.length]+[left, top]

    theta = np.arange(style['contour_points'])*2*np.pi/style['contour_points']
    contours = [boundary(p, theta)[0] for p in geometry['particles']]
    arc = np.linspace(0, np.pi, style['contour_points']+1)
    pit = np.c_[g.pit_x+g.pit_radius*np.cos(arc), g.pit_radius*np.sin(arc)]
    mapped = [point(v) for v in contours]
    pit_mm = point(pit)
    mouth = point([[g.pit_x-g.mouth_half_width, 0], [g.pit_x+g.mouth_half_width, 0]])
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size_pt'],
          'text.color': style['text_color'], 'pdf.fonttype': 42, 'ps.fonttype': 42,
          'pdf.use14corefonts': False, 'svg.fonttype': 'none',
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'hatch.linewidth': .3, 'text.usetex': False}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height), aspect='equal')
        ax.axis('off')

        def label(x, y, text):
            return ax.text(x, y, text, ha='center', va='center', linespacing=1.2,
                fontproperties=font_manager.FontProperties(fname=str(font), size=style['font_size_pt']),
                color=style['text_color'], zorder=8)

        def arrow(start, end):
            ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>',
                'color': style['text_color'], 'lw': .5, 'mutation_scale': 5}, zorder=8)

        ax.add_patch(Rectangle((left, bottom), panel_width, panel_height,
                              facecolor=style['metal'], edgecolor='none'))
        for y in np.arange(bottom+.7, top, .9):
            ax.plot([left, left+panel_width], [y, y], color=style['metal_grain'], lw=.17)
        ax.add_patch(Polygon(pit_mm, facecolor=style['electrolyte'], edgecolor='none', zorder=2))
        ax.plot(*pit_mm.T, color=style['interface'], lw=.7, zorder=3)
        for xy in mapped:
            ax.add_patch(Polygon(xy, facecolor=style['ceramic'], edgecolor=style['ceramic_edge'],
                                 linewidth=.7, zorder=3))
        wall_rects = [(left-1.5, bottom-1.5, 1.5, panel_height+3),
                      (left+panel_width, bottom-1.5, 1.5, panel_height+3),
                      (left, bottom-1.5, panel_width, 1.5),
                      (left, top, mouth[0, 0]-left, 1.5),
                      (mouth[1, 0], top, left+panel_width-mouth[1, 0], 1.5)]
        for x, y, w, h in wall_rects:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=style['wall'],
                edgecolor=style['wall_edge'], hatch='////', linewidth=.45, zorder=4))
        ax.plot(*mouth.T, color=style['mouth'], lw=1.5, zorder=5)
        label(50, 70.7, spec['labels']['mouth'])
        arrow((50, 68), (50, top+.25))
        label(50, 61.3, 'Electrolyte')
        label(49, 50.7, 'Initial pit')
        arrow((49.5, 52.8), point([g.pit_x, g.pit_radius]))
        label(50, 34, 'Metal')
        label(28.5, 24.5, spec['labels']['particles'])
        target = mapped[2][np.argmin(mapped[2][:, 1])]
        arrow((28.5, 28.3), target)
        label(50, 12, spec['labels']['closed_walls'])
        label(84, 2.8, '50 × 30 μm')
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
    receipt = {'schema': 'particle_material_setup_validation_v1', 'specification': spec,
        'renderer': str(Path(__file__).relative_to(REPO)), 'geometry': geometry,
        'geometry_validation': checked, 'particle_count': len(contours),
        'mouth_coordinates': [[g.pit_x-g.mouth_half_width, 0], [g.pit_x+g.mouth_half_width, 0]],
        'mouth_mm': mouth.tolist(), 'pit_radius_physical_um': 6,
        'physical_aspect': g.length/g.height, 'text': texts, 'output_bytes': sizes,
        'interpretation': 'Initial sharp contour only; inert ceramic is excluded from both physical fields',
        'boundary_equivalence': 'q_n=c_n-Delta_c*h_prime(phi)*phi_n; with phi_n=0, q_n=0 iff c_n=0',
        'training': False, 'inference': False, 'pde_solves': 0,
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': spec['output'], 'bytes': sizes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.spec.read_text()))
