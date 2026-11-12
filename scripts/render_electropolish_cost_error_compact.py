"""Two or three predeclared grids, four matched ratios, mean errors only."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from scripts.render_electropolish_cost_error import REPO, collect, read
from scripts.render_pino_derivative_comparisons import publication_style


def select(rows, selection):
    selected = []
    for entry in selection:
        group = []
        if len(entry['runs']) != 4 or len(entry['ratios']) != 4:
            raise ValueError('Exactly four predeclared runs per grid required')
        for run, ratio in zip(entry['runs'], entry['ratios'], strict=True):
            matched = [r for r in rows if r['run'] == run]
            if len(matched) != 1 or matched[0]['particles'] != entry['particles']:
                raise ValueError('Missing or mismatched predeclared run: ' + run)
            np.testing.assert_allclose(matched[0]['ratio'], ratio, rtol=0, atol=1e-14)
            group.append(matched[0])
        selected.extend(group)
    if len(selection) not in (2, 3) or any(
            entry['ratios'] != selection[0]['ratios'] for entry in selection):
        raise ValueError('Require two or three grids with the same four ratios')
    if len({entry['particles'] for entry in selection}) != len(selection):
        raise ValueError('Particle counts must be distinct')
    if len({r['run'] for r in selected}) != 4 * len(selection):
        raise ValueError('Four distinct final models per grid are required')
    return selected


def marker_areas(style, ratios):
    """Explicit size keys; scatter's s is the squared nominal diameter in pt."""
    if style['marker_size_mode'] == 'area_linear_ratio':
        sizes = [style['marker_area_pt2_at_reference'] * r /
                 style['marker_reference_ratio'] for r in ratios]
    elif style['marker_size_mode'] == 'diameter_levels':
        diameters = style['marker_diameters_pt']
        if (len(diameters) != len(ratios) or not np.all(np.array(diameters) > 0)
                or not np.all(np.diff(diameters) > 0)):
            raise ValueError('Each ratio needs a strictly increasing marker diameter')
        sizes = np.square(diameters).tolist()
    else:
        raise ValueError('Unknown explicit marker size mode')
    if not np.all(np.isfinite(sizes)) or not np.all(np.array(sizes) > 0):
        raise ValueError('Marker areas must be finite and positive')
    return dict(zip(ratios, sizes, strict=True))


def render(spec):
    all_rows, duplicates, evaluation = collect(spec['source_collection'])
    rows = select(all_rows, spec['selection'])
    style = spec['style']
    rc, font = publication_style(style['typography'])
    areas = marker_areas(style, spec['selection'][0]['ratios'])
    groups = []
    with plt.rc_context(rc):
        fig = plt.figure(figsize=np.array(style['figure_mm']) / 25.4)
        ax = fig.add_axes(style['axes_rect'])
        ax.set_xscale(style['xscale']); ax.set_yscale(style['yscale'])
        for entry in spec['selection']:
            count = entry['particles']
            group = [r for r in rows if r['particles'] == count]
            x = [r['cold_process_seconds'] for r in group]
            y = [r['mean'] for r in group]
            sizes = [areas[ratio] for ratio in entry['ratios']]
            color = style['colors'][str(count)]
            ax.plot(x, y, color=color, linewidth=style['line_width_pt'], zorder=2)
            ax.scatter(x, y, s=sizes, color=color, edgecolor=style['marker_edge_color'],
                       linewidth=style['marker_edge_width_pt'], zorder=3)
            groups.append(dict(particles=count, run_order=entry['runs'], ratios=entry['ratios'],
                               deltas=[r['delta'] for r in group], x=x, mean=y,
                               marker_areas_pt2=sizes, color=color))
        ax.set_xlabel(style['xlabel']); ax.set_ylabel(style['ylabel'])
        ax.xaxis.set_major_locator(FixedLocator(style['xticks']))
        ax.yaxis.set_major_locator(FixedLocator(style['yticks']))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='major', length=3, pad=3)
        ax.tick_params(which='minor', length=1.5)
        handles = [Line2D([], [], color=style['colors'][str(e['particles'])],
                          linewidth=style['legend_line_width_pt'],
                          label=f"{e['particles']:,}") for e in spec['selection']]
        fig.legend(handles=handles, title='Particles', frameon=False, loc='upper left',
                   bbox_to_anchor=style['particle_legend_anchor'], borderaxespad=0,
                   labelspacing=.8, handlelength=2)
        size_handles = [ax.scatter([], [], s=areas[ratio], color=style['size_legend_color'],
                                  edgecolor=style['marker_edge_color'],
                                  linewidth=style['marker_edge_width_pt'],
                                  label=f'{ratio:g}') for ratio in spec['selection'][0]['ratios']]
        fig.legend(handles=size_handles, title=r'$\delta/h$', frameon=False, loc='upper left',
                   bbox_to_anchor=style['size_legend_anchor'], borderaxespad=0,
                   labelspacing=style['size_legend_labelspacing'], handletextpad=1.2)
        limits = dict(x=list(ax.get_xlim()), y=list(ax.get_ylim()))
        for kind in ('pdf','png'):
            output = Path(spec['output'][kind]); output.parent.mkdir(parents=True, exist_ok=True)
            kwargs = dict(dpi=style['dpi'])
            if kind == 'pdf':
                kwargs['metadata'] = {'CreationDate':None, 'ModDate':None}
            fig.savefig(output, **kwargs)
        plt.close(fig)
    sys.path.insert(0, str(Path(spec['pdf_audit_packages']).resolve(strict=True)))
    import fitz
    with fitz.open(spec['output']['pdf']) as doc:
        page = doc[0]
        np.testing.assert_allclose([page.rect.width*25.4/72,page.rect.height*25.4/72],
                                   style['figure_mm'], atol=.001)
        spans = [s for b in page.get_text('dict')['blocks'] if b['type']==0
                 for line in b['lines'] for s in line['spans']]
        fonts = page.get_fonts(full=True)
        if not spans or any(not 4.99 <= s['size'] <= 7.01 or
                            not page.rect.contains(fitz.Rect(s['bbox'])) for s in spans):
            raise ValueError('Text must remain editable, in-page and5–7pt')
        if page.get_images() or any(f[2]=='Type3' or not doc.extract_font(f[0])[3] for f in fonts):
            raise ValueError('Expected vector-only PDF with embedded TrueType text')
    receipt = dict(specification=spec, renderer=str(Path(__file__).relative_to(REPO)),
        rows=rows, source_unique_runs=len(all_rows), source_duplicates=duplicates,
        excluded_runs=[r['run'] for r in all_rows if r['run'] not in {v['run'] for v in rows}],
        evaluation=evaluation, groups=groups, automatic_limits=limits,
        spread_display='none; mean only, no shading or error bars', font=font, pdf_fonts=fonts,
        font_size_range_pt=[min(s['size'] for s in spans),max(s['size'] for s in spans)],
        output_bytes={k:Path(spec['output'][k]).stat().st_size for k in ('pdf','png')},
        visual_review='pending')
    (REPO/spec['output']['receipt']).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'plotted_runs':len(rows), 'output':spec['output']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    render(read(parser.parse_args().spec))
