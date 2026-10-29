"""Physical total/native/gap errors on exactly matched local FEM query points."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

OUT = Path(__file__).resolve().parent
validation = json.loads((OUT/'melting-local-fem-validation.json').read_text())
assert validation['status'] in ('passed', 'accepted_by_user')
replay = json.loads((OUT/'melting-local-fem-checkpoint-replay.json').read_text())
assert replay['all_checks_pass']
replay_paths = {(r['optimizer'], r['index']): Path(r['path']) for r in replay['records']}
runs = dict(Adam=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01'),
            SOAP=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01'))
cases = {}
records = []


def relative(prediction, target):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    assert np.linalg.norm(t) > 0
    return float(np.linalg.norm(p-t)/np.linalg.norm(t))


def metrics(prediction, target):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    band = abs(t) < .9
    return dict(pooled_phi=relative(p, t), terminal_phi=relative(p[-1], t[-1]),
        pooled_solid=relative((p+1)/2, (t+1)/2),
        terminal_solid=relative((p[-1]+1)/2, (t[-1]+1)/2),
        phi_by_time=[relative(a, b) for a, b in zip(p, t)],
        solid_by_time=[relative((a+1)/2, (b+1)/2) for a, b in zip(p, t)],
        interface_phi_absolute_rmse=float(np.sqrt(np.mean((p[band]-t[band])**2))),
        terminal_diagnostic_solid_fraction_relative_error=relative((p[-1]+1).mean(), (t[-1]+1).mean()))


def decomposition(prediction, native, local):
    n = np.asarray(prediction, dtype=np.float64)-native
    d = native-local
    squared = np.sum(local**2)
    fitting = float(np.sum(n*n)/squared)
    gap = float(np.sum(d*d)/squared)
    cross = float(2*np.sum(n*d)/squared)
    total = float(np.sum((prediction-local)**2)/squared)
    np.testing.assert_allclose(total, fitting+gap+cross, rtol=1e-12, atol=1e-18)
    return dict(fitting_squared=fitting, reference_gap_squared=gap,
                cross_term=cross, total_squared=total,
                cosine=float(np.sum(n*d)/(np.linalg.norm(n)*np.linalg.norm(d))))


for accepted in validation['conditions']:
    index = accepted['index']
    with np.load(accepted['path']) as stored:
        local = {key: stored[key] for key in stored.files}
    info = json.loads(str(local['metadata_json']))
    assert info['range_gate_pass']
    fields = {}
    for name, run in runs.items():
        with np.load(replay_paths[(name, index)]) as stored:
            fields[name] = {key: stored[key] for key in stored.files}
        for key in ('parameters', 'points', 'slice_points', 'time'):
            np.testing.assert_array_equal(fields[name][key], local[key])
        config = json.loads((run/'config.json').read_text())
        for key in ('physical', 'nondim', 'initial'):
            assert config[key] == info['problem'][key]
    native = fields['Adam']['target']
    np.testing.assert_array_equal(native, fields['SOAP']['target'])
    item = dict(index=index, local_reference=accepted['path'],
                native_vs_local=metrics(native, local['phi']),
                native_slice_vs_local=metrics(fields['Adam']['slice_target'], local['slice_phi']), models={})
    for name, field in fields.items():
        prediction = np.asarray(field['prediction'], dtype=np.float64)
        item['models'][name] = dict(vs_local=metrics(prediction, local['phi']),
            vs_native=metrics(prediction, native),
            local_slice_pooled_phi=relative(field['slice_prediction'], local['slice_phi']),
            local_slice_terminal_phi=relative(field['slice_prediction'][-1], local['slice_phi'][-1]),
            pooled_error_decomposition=decomposition(prediction, native, local['phi']),
            terminal_error_decomposition=decomposition(prediction[-1], native[-1], local['phi'][-1]))
    records.append(item)
    cases[index] = (local, fields)

keys = ('pooled_phi', 'terminal_phi', 'pooled_solid', 'terminal_solid',
        'interface_phi_absolute_rmse', 'terminal_diagnostic_solid_fraction_relative_error')
means = {'native_vs_local': {key: float(np.mean([r['native_vs_local'][key] for r in records])) for key in keys}}
for name in runs:
    means[name] = {reference: {key: float(np.mean([r['models'][name][reference][key] for r in records]))
                              for key in keys} for reference in ('vs_local', 'vs_native')}
report = dict(reference_validation=str(OUT/'melting-local-fem-validation.json'),
              local_reference_status=validation['status'],
              strict_refinement_passed=validation['strict_refinement_passed'],
              input_count=len(records), cases=records,
              mean=means,
              decomposition_normalization='Squared fitting/gap/cross terms share the local FEM norm. Separately reported vs_native L2 uses the native norm.',
              scope='Previously random held-out input5, user requests one-case pilot; unchanged final5000checkpoints. Same24^3diagnostic nodes, not continuous/full-grid L2; slice128^2. Native reference gap includes finite-horizon/quadrature effects. No localAD training or speed/seed-stability claim.')
(OUT/'melting-local-fem-evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

# The first previously chosen random case, independent of its error ranking.
local, fields = cases[5]
physical_time_scale = json.loads(str(local['metadata_json']))['problem']['nondim']['T_c']
reference = local['slice_phi']
predictions = [fields[name]['slice_prediction'] for name in ('Adam', 'SOAP')]
vmax = float(max(abs(p-reference).max() for p in predictions))
rows = [(reference, 'Local FEM', False), (predictions[0], 'Adam', False),
        (predictions[1], 'SOAP', False),
        (abs(predictions[0]-reference), 'Adam absolute error', True),
        (abs(predictions[1]-reference), 'SOAP absolute error', True)]
fig, axes = plt.subplots(5, 5, figsize=(13, 11.5), constrained_layout=True)
for row, (values, label, error) in enumerate(rows):
    for col, t in enumerate(local['time']):
        image = axes[row, col].imshow(values[col].reshape(128, 128).T, origin='lower',
            extent=(0, 1, 0, 1), interpolation='nearest', cmap='magma' if error else 'coolwarm',
            vmin=0 if error else -1, vmax=vmax if error else 1)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(f't*={t:.2f} / {physical_time_scale*t:g}s')
        if col == 0:
            axes[row, col].set_ylabel(label)
    fig.colorbar(image, ax=axes[row, :], fraction=.017, pad=.01)
z = local['slice_points'][:, 2]
assert np.ptp(z) == 0
fig.suptitle(f'Held-out ellipsoid ID5 | local FEM comparison | z*={z[0]:.6f}')
fig.savefig(OUT/'melting-local-fem-fields.png', dpi=180, bbox_inches='tight')
plt.close(fig)
print(json.dumps(means, indent=2))

# Show actual reference-refinement checks alongside physical-field errors.
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
for ax, checks, coordinate, label in (
        (axes[0, 0], validation['conditions'][0]['space_checks'], 'mesh_subdivisions', 'Fine mesh subdivisions'),
        (axes[0, 1], validation['conditions'][0]['time_checks'], 'dt', 'Fine time step')):
    x = []
    for check in checks:
        metadata = json.loads(Path(check['fine']).with_suffix('.json').read_text())
        x.append(metadata[coordinate])
    for field, color in (('phase', 'C0'), ('solid', 'C1')):
        values = [100*max(check['phi_'+field]['maximum'], check['slice_phi_'+field]['maximum'])
                  for check in checks]
        ax.loglog(x, values, 'o-', color=color, label=field)
        tolerance = validation['phase_tolerance'] if field == 'phase' else validation['solid_tolerance']
        ax.axhline(100*tolerance, color=color, linestyle=':', alpha=.7)
    if coordinate == 'dt':
        ax.invert_xaxis()
    ax.set_xticks(x)
    ax.set_xticklabels([f'{value:g}' for value in x])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(label)
    ax.set_ylabel('Maximum adjacent-refinement difference (%)')
    ax.set_title('Spatial checks' if coordinate != 'dt' else 'Temporal checks')
    ax.grid(alpha=.2)
    ax.legend()
item = next(record for record in records if record['index'] == 5)
for ax, key, label in ((axes[1, 0], 'phi_by_time', 'Phase'),
                       (axes[1, 1], 'solid_by_time', 'Solid indicator')):
    for name in ('Adam', 'SOAP'):
        ax.plot(local['time'], 100*np.array(item['models'][name]['vs_local'][key]), 'o-', label=name)
    ax.plot(local['time'], 100*np.array(item['native_vs_local'][key]), 'o--', label='Native reference')
    ax.set_xlabel('Dimensionless time')
    ax.set_ylabel('Relative L2 against local FEM (%)')
    ax.set_title(label+' | 24^3 diagnostic points')
    ax.grid(alpha=.2)
    ax.legend()
fig.savefig(OUT/'melting-local-fem-refinement-errors.png', dpi=180, bbox_inches='tight')
plt.close(fig)


# Primary user question: independently solved native and local reference fields.
native_slice = fields['Adam']['slice_target']
fig, axes = plt.subplots(3, 5, figsize=(13, 7.2), constrained_layout=True)
error_max = float(abs(native_slice-reference).max())
for row, (values, label, error) in enumerate(((reference, 'Local FEM', False),
        (native_slice, 'Nonlocal reference', False),
        (abs(native_slice-reference), 'Absolute difference', True))):
    for col, t in enumerate(local['time']):
        values2d = values[col].reshape(128, 128).T
        im = axes[row, col].imshow(values2d, origin='lower', extent=(0, 1, 0, 1),
            interpolation='nearest', cmap='magma' if error else 'coolwarm',
            vmin=0 if error else -1, vmax=error_max if error else 1)
        if not error and values2d.min() < 0 < values2d.max():
            axes[row, col].contour(np.linspace(0, 1, 128), np.linspace(0, 1, 128),
                                  values2d, levels=[0], colors='black', linewidths=.6)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(f't*={t:.2f} / {physical_time_scale*t:g}s')
        if col == 0:
            axes[row, col].set_ylabel(label)
    fig.colorbar(im, ax=axes[row, :], fraction=.017, pad=.01)
fig.suptitle(f'Held-out ellipsoid ID5 | reference comparison | z*={z[0]:.6f}')
fig.savefig(OUT/'melting-local-fem-reference-fields.png', dpi=180, bbox_inches='tight')
plt.close(fig)
