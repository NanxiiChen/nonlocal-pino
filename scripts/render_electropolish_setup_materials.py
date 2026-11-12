"""Editable material/IC/BC schematic from existing prescribed input profiles."""
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
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def profiles(spec):
    source = json.loads((REPO/spec['source']).read_text())['cases'][spec['case']]
    prior = source['prior']
    x = np.linspace(0, 1, 1001)
    fields = []
    for index in spec['samples']:
        coeff = np.asarray(source['coefficients'][index], dtype=float)
        if (coeff.shape != (prior['n_modes'], 3) or not np.isfinite(coeff).all()
                or np.any(coeff[:, 0] < .3*prior['amplitude'])
                or np.any(coeff[:, 0] > prior['amplitude'])
                or np.any(coeff[:, 1] < prior['freq_lo'])
                or np.any(coeff[:, 1] > prior['freq_hi'])
                or np.any(coeff[:, 2] < 0) or np.any(coeff[:, 2] > 2*np.pi)):
            raise ValueError('Invalid prescribed profile coefficients')
        angle = 2*np.pi*x[:, None]*coeff[:, 1]+coeff[:, 2]
        y = prior['y0']+np.sum(coeff[:, 0]*(np.sin(angle)+np.cos(angle+.3)), axis=-1)
        low, high = source['height_acceptance']
        if not low < y.min() <= y.max() < high:
            raise ValueError('Profile exceeds prescribed domain')
        fields.append((x, y))
    return fields


def draw_material(ax, field, rect, style, fixtures):
    left, bottom, width, height = rect
    if not np.isclose(width/height, 2):
        raise ValueError('Physical domain must retain its 2:1 aspect ratio')
    x, y = field
    xx, yy = left+width*x, bottom+width*y
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor=style['electrolyte'],
                           edgecolor='none', zorder=1))
    metal = Polygon(np.vstack(([left, bottom], np.c_[xx, yy], [left+width, bottom])),
                    facecolor=style['metal'], edgecolor='none', zorder=2)
    ax.add_patch(metal)
    # Sparse vector strokes suggest brushed metal; they carry no physical data.
    for grain_y in np.arange(bottom+.7, bottom+height, .9):
        line, = ax.plot([left, left+width], [grain_y, grain_y], color=style['metal_grain'],
                        linewidth=.17, zorder=2.1)
        line.set_clip_path(metal)
    ax.plot(xx, yy, color=style['interface'], linewidth=.7, zorder=3)
    if fixtures:
        for wall_x in (left-1.8, left+width):
            ax.add_patch(Rectangle((wall_x, bottom), 1.8, height,
                facecolor=style['wall'], edgecolor=style['wall_edge'],
                hatch='////', linewidth=.45, zorder=4))
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor='none',
                           edgecolor=style['interface'], linewidth=.65, zorder=5))


def render(spec):
    if spec['schema'] != 'electropolish_material_setup_v2' or spec['samples'] != [0]:
        raise ValueError('Unsupported schematic specification')
    style = spec['style']
    font = Path(style['font_file']).resolve(strict=True)
    with TTFont(font) as face:
        if face['name'].getDebugName(1) != 'Helvetica' or 'glyf' not in face:
            raise ValueError('Require actual TrueType Helvetica')
        if face['OS/2'].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError('Font embedding is restricted')
    for name in ('math_font_file', 'math_italic_file'):
        Path(style[name]).resolve(strict=True)
    font_manager.fontManager.addfont(str(font))
    if Path(font_manager.findfont('Helvetica', fallback_to_default=False)).resolve() != font:
        raise ValueError('Unexpected Helvetica font resolution')
    fields = profiles(spec)
    width, height = style['figure_mm']
    rc = {'font.family': 'Helvetica', 'font.size': style['font_size_pt'],
          'text.color': style['text_color'], 'pdf.fonttype': 42, 'ps.fonttype': 42,
          'pdf.use14corefonts': False, 'svg.fonttype': 'none',
          'mathtext.fontset': 'dejavusans', 'mathtext.fallback': None,
          'hatch.linewidth': .3, 'text.usetex': False}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height), aspect='equal')
        ax.axis('off')

        def label(x, y, value, size=style['font_size_pt'], **kwargs):
            return ax.text(x, y, value, ha='center', va='center', fontsize=size,
                           fontproperties=font_manager.FontProperties(fname=str(font), size=size),
                           color=style['text_color'], zorder=8, **kwargs)

        draw_material(ax, fields[0], style['main_rect_mm'], style, True)
        label(50, 54.5, spec['labels']['top'])
        label(50, 8.5, spec['labels']['bottom'])
        label(50, 46.5, 'Electrolyte')
        label(50, 22, 'Metal')
        label(6, 31.5, spec['labels']['sides'], rotation=90)
        label(94, 31.5, spec['labels']['sides'], rotation=90)
        label(85, 3.5, '100 × 50 μm', style['secondary_size_pt'])
        label(31, 25.7, spec['labels']['interface'], style['secondary_size_pt'], linespacing=1.2)
        target_x = .42
        target_y = np.interp(target_x, fields[0][0], fields[0][1])
        ax.annotate('', xy=(13+74*target_x, 13+74*target_y), xytext=(42, 29.5),
                    arrowprops={'arrowstyle': '-|>', 'color': style['text_color'],
                                'lw': .5, 'mutation_scale': 5}, zorder=8)
        texts = []
        fig.canvas.draw()
        for text in fig.findobj(Text):
            if text.get_visible() and text.get_text():
                box = text.get_window_extent(fig.canvas.get_renderer())
                if not 5 <= text.get_fontsize() <= 7 or not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
                    raise ValueError(f'Invalid or clipped label: {text.get_text()}')
                texts.append({'text': text.get_text(), 'size_pt': text.get_fontsize()})
        sizes = {}
        for kind in ('pdf', 'png', 'svg'):
            out = Path(spec['output'][kind])
            out.parent.mkdir(parents=True, exist_ok=True)
            metadata = {'CreationDate': None, 'ModDate': None} if kind == 'pdf' else None
            fig.savefig(out, dpi=style['dpi'], metadata=metadata)
            sizes[kind] = out.stat().st_size
        plt.close(fig)
    receipt = {'schema': 'electropolish_material_setup_validation_v2',
               'specification': spec, 'renderer': str(Path(__file__).relative_to(REPO)),
               'profile_ranges': [[float(y.min()), float(y.max())] for _, y in fields],
               'sample_selection': 'Keep the original main-panel input; remove both alternative panels',
               'interpretation': 'sharp outline marks the initial diffuse interface; material strokes are illustrative',
               'training': False, 'inference': False, 'pde_solves': 0,
               'physical_aspect': 2, 'text': texts, 'output_bytes': sizes,
               'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': spec['output'], 'bytes': sizes}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.spec.read_text()))
