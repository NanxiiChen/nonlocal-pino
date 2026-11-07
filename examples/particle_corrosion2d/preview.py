"""Sample and draw prescribed ceramic geometries without solving an evolution."""
import argparse
import importlib
import json
from pathlib import Path
import secrets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import numpy as np

from examples.particle_corrosion2d.geometry import boundary, sample_geometry, validate_geometry


METAL, CERAMIC, LIQUID = '#f0ddbd', '#586c78', '#8ed0e0'
INK, INTERFACE = '#273641', '#d28b32'


def draw_case(ax, record, c, index):
    """True sampled boundaries; solid colors distinguish materials at t=0."""
    g = c.geometry
    ax.set_facecolor(METAL)
    theta = np.linspace(0, 2*np.pi, 1025)
    for particle in record['particles']:
        points, _, _ = boundary(particle, theta)
        ax.add_patch(Polygon(points, closed=True, facecolor=CERAMIC,
                             edgecolor=INK, linewidth=1.15, zorder=3))
    angle = np.linspace(0, np.pi, 257)
    pit = np.column_stack((g.pit_x+g.pit_radius*np.cos(angle),
                           g.pit_radius*np.sin(angle)))
    ax.add_patch(Polygon(pit, closed=True, facecolor=LIQUID, edgecolor='none', zorder=4))
    ax.plot(pit[:, 0], pit[:, 1], color=INTERFACE, linewidth=1.6, zorder=5)
    left, right = g.pit_x-g.mouth_half_width, g.pit_x+g.mouth_half_width
    ax.plot([0, left], [0, 0], color=INK, linewidth=3.5, clip_on=False, zorder=6)
    ax.plot([right, g.length], [0, 0], color=INK, linewidth=3.5, clip_on=False, zorder=6)
    ax.plot([left, right], [0, 0], color='#268fb7', linewidth=3.5, clip_on=False, zorder=7)
    ax.annotate('', xy=(g.pit_x, .035), xytext=(g.pit_x, -.04),
                arrowprops=dict(arrowstyle='-|>', color='#268fb7', lw=1.4),
                annotation_clip=False, zorder=8)
    ax.set(xlim=(0, g.length), ylim=(g.height, 0), aspect='equal')
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .2, .4, .6])
    ax.tick_params(labelsize=8, length=3)
    ax.set_title(f'Sample {index+1} | {len(record["particles"])} ceramic particles',
                 fontsize=12, loc='left', pad=15)
    ax.set_xlabel('x', fontsize=9); ax.set_ylabel('depth y', fontsize=9)
    for spine in ax.spines.values():
        spine.set_color('#71818a')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples.particle_corrosion2d.configs.train_default')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if not 1 <= args.count <= 12:
        raise ValueError('Preview count must be between 1 and 12')
    if args.output.exists():
        raise FileExistsError(f'Use a fresh preview directory: {args.output}')
    c = importlib.import_module(args.config).get_config()
    seed = secrets.randbits(32) if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(args.count):
        record = sample_geometry(c, rng)
        metrics = validate_geometry(record, c)
        cases.append(dict(geometry=record, validation=metrics))
    args.output.mkdir(parents=True)
    columns = min(2, args.count)
    rows = (args.count+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(6.1*columns, 4.2*rows+1.15),
                             squeeze=False)
    fig.subplots_adjust(left=.06, right=.98, top=.88, bottom=.12,
                        hspace=.37, wspace=.18)
    for index, ax in enumerate(axes.flat):
        if index >= len(cases):
            ax.set_visible(False)
        else:
            draw_case(ax, cases[index]['geometry'], c, index)
    fig.suptitle('Corrosion around smooth inert ceramic aggregates', fontsize=18, y=.97)
    fig.text(.5, .93, 'Random geometry inputs only | same initial pit and uniform aluminum matrix',
             ha='center', fontsize=11, color=INK)
    legend = [Patch(facecolor=METAL, label='Aluminum matrix: can dissolve'),
              Patch(facecolor=CERAMIC, label='Ceramic: fixed and impermeable'),
              Patch(facecolor=LIQUID, label='Same initial electrolyte pit')]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .045),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .025, 'Blue opening: reservoir | dark top segments: passive film | no evolution has been solved',
             ha='center', fontsize=9, color='#55636c')
    image = args.output/'geometry_samples.png'
    fig.savefig(image, dpi=170, facecolor='white', pil_kwargs={'optimize': True})
    plt.close(fig)
    manifest = dict(schema=1, kind='prescribed_ceramic_geometry_inputs',
                    runtime_seed=seed, config=c.to_dict(), samples=cases,
                    image=str(image.resolve()), training_updates=0, pde_solves=0,
                    reference_solves=0, physical_time=0,
                    note='Sharp outlines show the initial interface location only; no future shape is prescribed.')
    path = args.output/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(image=str(image.resolve()), manifest=str(path.resolve()),
                         sample_count=len(cases), seed=seed, pde_solves=0), indent=2))


if __name__ == '__main__':
    main()
