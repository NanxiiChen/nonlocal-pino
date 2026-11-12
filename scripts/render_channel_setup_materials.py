"""Editable two-phase channel materials and BCs from a prescribed wall sample."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
from matplotlib.text import Text
from fontTools.ttLib import TTFont
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.geometry import (
    geometry_statistics, profiles, validate_parameters,
)

REPO = Path(__file__).resolve().parents[1]


def render(spec):
    if spec['schema'] != 'channel_material_setup_v1' or spec['sample_index'] != 0:
        raise ValueError('Require the first prescribed random-wall sample')
    source = json.loads((REPO/spec['source']).read_text())['cases'][spec['case']]
    c = ConfigDict({'geometry': source['geometry']})
    parameters = validate_parameters(source['parameters'][spec['sample_index']], c)
    initial = source['initial']
    if ((c.geometry.length, c.geometry.height) != (4., 1.)
            or initial != {'center': [.8, .5], 'radius': .22, 'transition': .05}):
        raise ValueError('Schematic requires the documented channel and initial disk')
    x = np.linspace(0, c.geometry.length, spec['style']['wall_points'])
    lower, upper = profiles(x, parameters, c, xp=np)
    center = np.asarray(initial['center'])
    if center[0]+initial['radius'] >= c.geometry.start:
        raise ValueError('Initial disk must be upstream of the variable wall segment')
    style = spec['style']
    font = Path(style['font_file']).resolve(strict=True)
    with TTFont(font) as face:
        if face['name'].getDebugName(1) != 'Helvetica' or 'glyf' not in face:
            raise ValueError('Require TrueType Helvetica')
        if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError('Font embedding is restricted')
    for key in ('math_font_file', 'math_italic_file', 'math_bold_file'):
        Path(style[key]).resolve(strict=True)
    font_manager.fontManager.addfont(str(font))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font:
        raise ValueError('Unexpected font substitution')
    width, height = style['figure_mm']
    left, bottom, panel_width, panel_height = style['main_rect_mm']
    if not np.isclose(panel_width/panel_height, 4):
        raise ValueError('Preserve physical 4:1 channel aspect')
    scale = panel_width/c.geometry.length

    def point(xy):
        return np.asarray(xy)*scale+[left, bottom]

    low = point(np.c_[x, lower])
    high = point(np.c_[x, upper])
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

        def arrow(start, end, width_pt):
            ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>',
                'color': style['text_color'], 'lw': width_pt, 'mutation_scale': 5}, zorder=8)

        ax.add_patch(Polygon(np.vstack((low, high[::-1])), facecolor=style['carrier'],
                             edgecolor='none', zorder=1))
        thick = style['wall_thickness_mm']
        for curve, back in [(low, bottom-thick), (high, bottom+panel_height+thick)]:
            poly = np.vstack(([left, back], curve, [left+panel_width, back]))
            ax.add_patch(Polygon(poly, facecolor=style['wall'], edgecolor=style['wall_edge'],
                                 hatch='////', linewidth=.4, zorder=2))
            ax.plot(*curve.T, color=style['interface'], linewidth=.7, zorder=3)
        for xpos in (left, left+panel_width):
            ax.plot([xpos, xpos], [bottom, bottom+panel_height], linestyle=(0, (2, 2)),
                    color=style['wall_edge'], linewidth=.4, zorder=3)
        ax.add_patch(Circle(point(center), initial['radius']*scale,
            facecolor=style['droplet'], edgecolor=style['interface'], linewidth=.7, zorder=4))
        # Arrows depict imposed driving direction after ramp-up, not solved velocities.
        for eta in (.15, .325, .5, .675, .85):
            yy = bottom+eta*panel_height
            length = style['inlet_arrow_max_mm']*4*eta*(1-eta)
            arrow((left-1-length, yy), (left-1, yy), .55)
        for eta in (.25, .5, .75):
            yy = bottom+eta*panel_height
            arrow((left+panel_width+1, yy), (left+panel_width+7, yy), .55)
        label(13, 56, spec['labels']['inlet_velocity'])
        label(13, 50, spec['labels']['inlet_phase'])
        label(98, 57, spec['labels']['outlet_traction'])
        label(98, 49.5, spec['labels']['no_flux'])
        label(55, 58, spec['labels']['random_walls'])
        upper_target = point([2., np.interp(2., x, upper)])
        arrow((55, 53), upper_target+[0, .3], .5)
        label(31, 40, 'Initial droplet')
        label(78, 33, 'Carrier fluid')
        label(55, 12, spec['labels']['wall_conditions'])
        lower_target = point([2., np.interp(2., x, lower)])
        arrow((55, 17), lower_target-[0, .3], .5)
        label(100, 4, '4 × 1')
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
    receipt = {'schema': 'channel_material_setup_validation_v1', 'specification': spec,
        'renderer': str(Path(__file__).relative_to(REPO)), 'parameters': parameters.tolist(),
        'geometry_statistics': geometry_statistics(parameters, c), 'initial': initial,
        'physical_aspect': 4, 'text': texts, 'output_bytes': sizes,
        'interpretation': 'Initial zero-phase contour; arrows denote driving direction, not initial or predicted velocity',
        'phase_materials': {'droplet': 'generic negative phase', 'carrier': 'generic positive phase'},
        'training': False, 'inference': False, 'pde_solves': 0,
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': spec['output'], 'bytes': sizes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.spec.read_text()))
