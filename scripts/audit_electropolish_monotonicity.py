"""Audit raw saved phase trajectories; no inference, solving or field clipping.

Run from the repository with .venv/bin/python. The figure manifest identifies
the exact checkpoint, evaluation, references and explicitly registered font.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def audit(phi, times, x, y):
    phi = np.asarray(phi, np.float64)
    if phi.shape != (len(times), *x.shape) or x.shape != y.shape:
        raise ValueError('Phase/time/grid shapes do not match')
    if not np.isfinite(phi).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Require finite fields and strictly increasing times')
    running_min = np.minimum.accumulate(phi, axis=0)
    excursion = phi - running_min
    end, row, col = np.unravel_index(np.argmax(excursion), phi.shape)
    start = int(np.argmin(phi[:end+1, row, col]))
    crosses = np.any((running_min[:-1] < .5) & (phi[1:] > .5), axis=0)
    area = np.trapezoid(np.trapezoid(phi, x[0], axis=2), y[:, 0], axis=1)
    return dict(maximum_upward_excursion=float(excursion[end, row, col]),
                witness=dict(x=float(x[row, col]), y=float(y[row, col]),
                             start_time=float(times[start]), end_time=float(times[end]),
                             start_phi=float(phi[start, row, col]),
                             end_phi=float(phi[end, row, col])),
                upward_half_phase_crossing_nodes=int(crosses.sum()),
                nodes_with_increase_above_001=int(np.any(excursion > .01, axis=0).sum()),
                nodes=int(x.size), solid_area=area.tolist(),
                maximum_solid_area_increment=float(np.diff(area).max()))


def ac_terms(fields, x, y, physical, scales):
    """Independent centered FD/mirror evaluation of the declared local AC RHS.

    Top and bottom are Dirichlet nodes; their computed RHS is not an ODE rate.
    The audit uses strictly interior witnesses and excludes these rows in checks.
    """
    phi, concentration = fields[..., 0], fields[..., 1]
    dx, dy = x[0, 1]-x[0, 0], y[1, 0]-y[0, 0]
    np.testing.assert_allclose(np.diff(x, axis=1), dx, atol=1e-14, rtol=0)
    np.testing.assert_allclose(np.diff(y, axis=0), dy, atol=1e-14, rtol=0)
    padded = np.pad(phi, ((0, 0), (1, 1), (1, 1)), mode='reflect')
    lap = ((padded[:, 1:-1, 2:]+padded[:, 1:-1, :-2]-2*phi)/dx**2
           +(padded[:, 2:, 1:-1]+padded[:, :-2, 1:-1]-2*phi)/dy**2)
    n1 = 2*physical['A']*physical['L']*scales['T_c']
    n2 = physical['L']*physical['omega_phi']*scales['T_c']
    n3 = physical['L']*physical['alpha_phi']*scales['T_c']/scales['L_c']**2
    dc = physical['c_se']-physical['c_le']
    q = concentration-dc*(3*phi**2-2*phi**3)-physical['c_le']
    diffusion = n3*lap
    double_well = -n2*2*phi*(1-phi)*(1-2*phi)
    chemical = n1*dc*q*6*phi*(1-phi)
    return dict(diffusion=diffusion, double_well=double_well, chemical=chemical,
                total=diffusion+double_well+chemical, q=q)


def figure(spec, traces, times, out):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    font = Path(spec['style']['font_file']).resolve(strict=True)
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    resolved = Path(font_manager.findfont(family, fallback_to_default=False)).resolve()
    if family != 'Helvetica' or resolved != font:
        raise ValueError('Require the recorded Helvetica font file')
    plt.rcParams.update({'font.family': family, 'font.size': 6,
                         'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                         'legend.fontsize': 6, 'axes.linewidth': .5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'pdf.use14corefonts': False, 'text.color': '#222222',
                         'axes.labelcolor': '#222222', 'xtick.color': '#222222',
                         'ytick.color': '#222222'})
    fig, axes = plt.subplots(1, 2, figsize=(170/25.4, 61/25.4))
    fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.81, wspace=.3)
    styles = [('PINO', '#E64B35', '-'), ('local401', '#333333', '--'),
              ('native101', '#4DBBD5', ':')]
    labels = {'PINO': 'PINO', 'local401': 'Local FD (401)', 'native101': 'Native graph'}
    for name, color, line in styles:
        record = traces[name]
        area = np.asarray(record['area'])
        axes[0].plot(times, record['phi'], color=color, ls=line, lw=1, label=labels[name])
        axes[1].plot(times, area/area[0], color=color, ls=line, lw=1)
    axes[0].axhline(.5, color='#777777', ls=':', lw=.5)
    axes[0].set_ylabel('Phase at fixed point')
    axes[1].set_ylabel('Solid area / initial area')
    for ax, letter in zip(axes, ('(a)', '(b)'), strict=True):
        ax.set_xlabel('Time (s)')
        ax.tick_params(direction='in', width=.5, length=2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0, 1.035, letter, transform=ax.transAxes, fontsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3,
               frameon=False, bbox_to_anchor=(.5, .99))
    for extension in ('pdf', 'png'):
        fig.savefig(out/f'electropolish2d_monotonicity.{extension}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.manifest.read_text())
    config = json.loads(Path(spec['config']).read_text())
    data = read_arrays(spec['prediction'])
    root = Path(spec['references'][0]['file']).parent.parent
    records = []
    for index, prediction in enumerate(data['prediction']):
        path = root/f'condition_{index:03d}'/'local101.npz'
        ref = read_arrays(path)
        meta = json.loads(str(ref['metadata_json']))
        if meta['index'] != index or meta['surface_id'] != data['surface_ids'][index]:
            raise ValueError(f'Input identity mismatch: {path}')
        for key in ('X', 'Y', 'time'):
            np.testing.assert_array_equal(data[key], ref[key])
        for key in ('physical', 'nondim'):
            if meta['spec'][key] != config[key]:
                raise ValueError(f'Mismatched {key}: {path}')
        records.append(dict(index=index, reference=str(path),
                            PINO=audit(prediction[..., 0], data['time'], data['X'], data['Y']),
                            local101=audit(ref['fields'][..., 0], ref['time'], ref['X'], ref['Y'])))
    summary = {}
    for method in ('PINO', 'local101'):
        summary[method] = dict(
            samples=len(records),
            samples_with_phase_increase_above_001=sum(
                r[method]['maximum_upward_excursion'] > .01 for r in records),
            samples_with_upward_half_phase_crossing=sum(
                r[method]['upward_half_phase_crossing_nodes'] > 0 for r in records),
            samples_with_increasing_solid_area_above_1e_9=sum(
                r[method]['maximum_solid_area_increment'] > 1e-9 for r in records))
    refinements = []
    terms_at_witness = {}
    # Interior location identified by the largest PINO excursion in figure row 1.
    witness_x, witness_y = .46, .23
    row = np.flatnonzero(np.isclose(data['Y'][:, 0], witness_y, atol=1e-14, rtol=0)).item()
    col = np.flatnonzero(np.isclose(data['X'][0], witness_x, atol=1e-14, rtol=0)).item()
    traces = {'PINO': dict(phi=data['prediction'][0, :, row, col, 0].tolist(),
                           area=records[0]['PINO']['solid_area'])}
    for index in spec['samples']:
        for method in ('local101', 'local201', 'local401', 'native101'):
            path = root/f'condition_{index:03d}'/f'{method}.npz'
            ref = read_arrays(path)
            meta = json.loads(str(ref['metadata_json']))
            np.testing.assert_array_equal(ref['time'], data['time'])
            if meta['index'] != index or meta['surface_id'] != data['surface_ids'][index]:
                raise ValueError(f'Refinement input identity mismatch: {path}')
            stats = audit(ref['fields'][..., 0], ref['time'], ref['X'], ref['Y'])
            refinements.append(dict(index=index, method=method, file=str(path), **stats,
                maximum_time_refinement_error=meta['maximum_time_refinement_error']))
            if index == 0:
                row = np.flatnonzero(np.isclose(ref['Y'][:, 0], witness_y, atol=1e-14, rtol=0)).item()
                col = np.flatnonzero(np.isclose(ref['X'][0], witness_x, atol=1e-14, rtol=0)).item()
                traces[method] = dict(phi=ref['fields'][:, row, col, 0].tolist(),
                                      area=stats['solid_area'])
                if method == 'local401':
                    terms = ac_terms(ref['fields'], ref['X'], ref['Y'],
                                     meta['spec']['physical'], meta['spec']['nondim'])
                    terms_at_witness = {key: value[:, row, col].tolist()
                                        for key, value in terms.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(manifest=str(args.manifest.resolve()), checkpoint=spec['checkpoint'],
                  prediction=spec['prediction'], config=spec['config'],
                  definition='max(phi(t_j)-phi(t_i)) over saved t_i<=t_j at each fixed node',
                  qualification='25 saved times can disprove monotonicity, not prove it between snapshots',
                  area='tensor trapezoidal integral of physical phi; not total solute mass',
                  time=data['time'].tolist(), summary=summary, records=records,
                  refinements=refinements, witness=dict(index=0, x=witness_x, y=witness_y,
                    traces=traces, local401_terms_per_normalized_time=terms_at_witness),
                  figure_font=spec['style']['font_file'])
    (args.output_dir/'electropolish2d_monotonicity.json').write_text(
        json.dumps(report, indent=2, allow_nan=False)+'\n')
    figure(spec, traces, data['time']*config['nondim']['T_c'], args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
