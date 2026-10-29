"""Audit saved matched optimizer controls and plot monitor convergence."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNS = {
    'Adam': Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01'),
    'SOAP': Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01'),
}


def read(path):
    return json.loads(path.read_text())


def differences(a, b, prefix=''):
    assert set(a) == set(b), prefix
    result = []
    for key in a:
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(a[key], dict):
            result.extend(differences(a[key], b[key], name))
        elif a[key] != b[key]:
            result.append(name)
    return sorted(result)


def relative(error, target):
    norm = np.linalg.norm(np.asarray(target, dtype=np.float64))
    assert norm > 0
    return float(np.linalg.norm(np.asarray(error, dtype=np.float64)) / norm)


adam, soap = RUNS.values()
configs = {name: read(run / 'config.json') for name, run in RUNS.items()}
config_differences = differences(configs['Adam'], configs['SOAP'])
assert config_differences == ['log.run_name', 'train.optimizer']
assert configs['Adam']['train']['optimizer'] == 'adam'
assert configs['SOAP']['train']['optimizer'] == 'soap'
assert (adam / 'model_initial.eqx').read_bytes() == (soap / 'model_initial.eqx').read_bytes()
initial_arrays = 0
initial_file = soap / 'model_initial.eqx'
with initial_file.open('rb') as stream:
    while stream.tell() < initial_file.stat().st_size:
        np.load(stream, allow_pickle=False)
        initial_arrays += 1
with np.load(adam / 'sampling.npz') as a, np.load(soap / 'sampling.npz') as b:
    assert set(a.files) == set(b.files) == {'parameters', 'times'}
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
    assert a['parameters'].shape == (5000, 16, 10)
    assert a['times'].shape == (5000, 8)
    unique = len(np.unique(a['parameters'].reshape(-1, 10), axis=0))
    assert unique == 80000
assert read(adam / 'graph.json') == read(soap / 'graph.json')
sources = read(adam / 'source/files.json')
assert sources == read(soap / 'source/files.json')
for name in sources:
    assert (adam / 'source' / name).read_bytes() == (soap / 'source' / name).read_bytes(), name
    assert (soap / 'source' / name).read_bytes() == (ROOT / name).read_bytes(), name

manifest_path = Path(configs['Adam']['operator']['dataset_path']) / 'manifest.json'
manifest = read(manifest_path)
conditions = {}
performance = {}
histories = {}
for name, run in RUNS.items():
    summary = read(run / 'summary.json')
    assert summary['iterations'] == 5000 and not summary['positive_time_training_labels']
    scores = {r['index']: r for r in summary['test']['conditions']}
    assert set(scores) == set(manifest['test_indices'])
    records = []
    per_time = []
    for index in manifest['test_indices']:
        with np.load(run / f'prediction_{index:03d}.npz') as stored:
            arrays = {key: stored[key] for key in stored.files}
        p, t = arrays['prediction'].astype(np.float64), arrays['target']
        sp, st = arrays['slice_prediction'].astype(np.float64), arrays['slice_target']
        assert p.shape == t.shape == (5, 24**3)
        assert sp.shape == st.shape == (5, 128**2)
        assert np.isfinite(p).all() and np.isfinite(sp).all()
        assert p.min() >= -1 and p.max() <= 1
        with np.load(manifest_path.parent / f'condition_{index:03d}.npz') as reference:
            for output_key, reference_key in (
                ('parameters', 'parameters'), ('points', 'points'), ('time', 'time'),
                ('target', 'phi'), ('slice_target', 'slice_phi'), ('slice_points', 'slice_points')):
                np.testing.assert_array_equal(arrays[output_key], reference[reference_key])
        assert abs(p[0] - t[0]).max() < 1e-5
        solid = (t + 1) / 2
        delta = (p - t) / 2
        band = abs(t) < .9
        record = dict(index=index, pooled_phi=relative(p-t, t),
            terminal_phi=relative(p[-1]-t[-1], t[-1]),
            pooled_solid=relative(delta, solid), terminal_solid=relative(delta[-1], solid[-1]),
            interface_phi_absolute_rmse=float(np.sqrt(np.mean((p[band]-t[band])**2))),
            pooled_slice_phi=relative(sp-st, st), terminal_slice_phi=relative(sp[-1]-st[-1], st[-1]),
            terminal_solid_fraction_relative_error=relative(delta[-1].mean(), solid[-1].mean()),
            terminal_solid_fraction_absolute_error=float(abs(delta[-1].mean())),
            hard_IC_maximum_gap=float(abs(p[0]-t[0]).max()))
        for key in ('pooled_phi', 'terminal_phi', 'pooled_solid', 'terminal_solid', 'interface_phi_absolute_rmse'):
            np.testing.assert_allclose(record[key], scores[index][key], rtol=1e-12, atol=1e-15)
        time_record = dict(
            phi=[relative(e, r) for e, r in zip(p-t, t)],
            solid=[relative(e, r) for e, r in zip(delta, solid)])
        for key in ('phi', 'solid'):
            np.testing.assert_allclose(time_record[key], scores[index][key+'_by_time'], rtol=1e-12)
        per_time.append(time_record)
        records.append(record)
    mean = {key: float(np.mean([r[key] for r in records])) for key in records[0] if key != 'index'}
    maximum = {key: float(max(r[key] for r in records)) for key in mean}
    for key in ('pooled_phi', 'terminal_phi'):
        np.testing.assert_allclose(mean[key], summary['test'][key], rtol=1e-12)
    conditions[name] = dict(records=records, mean=mean, maximum=maximum,
        mean_by_time={key: np.mean([r[key] for r in per_time], axis=0).tolist()
                      for key in ('phi', 'solid')})
    process_file = OUT / ('melting-operator-process.json' if name == 'Adam'
                          else 'melting-operator-soap-process.json')
    process = read(process_file)
    assert process['returncode'] == 0
    performance[name] = dict(cold_process_seconds=process['wall_seconds'],
        trainer_seconds=summary['trainer_seconds'],
        first_update_compile_and_call_seconds=summary['first_update_compile_and_call_seconds'],
        warm_update_mean_seconds=summary['warm_update_mean_seconds'],
        warm_sampling_mean_seconds=summary['warm_sampling_mean_seconds'],
        allocator=summary['allocator'])
    histories[name] = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()
                       if json.loads(line)['split'] == 'monitor']
    assert [r['step'] for r in histories[name]] == list(range(0, 5001, 250))

assert read(OUT / 'melting-operator-process.json')['environment'] == read(OUT / 'melting-operator-soap-process.json')['environment']
for key in ('pooled_phi', 'terminal_phi'):
    assert histories['Adam'][0][key] == histories['SOAP'][0][key]

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
for name, history in histories.items():
    color = 'tab:blue' if name == 'Adam' else 'tab:orange'
    for ax, variable in zip(axes, ('phi', 'solid')):
        for scope, style in (('pooled', '-'), ('terminal', '--')):
            key = f'{scope}_{variable}'
            values = [np.mean([r[key] for r in entry['conditions']]) * 100 for entry in history]
            ax.semilogy([r['step'] for r in history], values, style, color=color,
                        label=f'{name}: {scope}')
for ax, title in zip(axes, ('Physical phase field', 'Solid indicator (phi + 1)/2')):
    ax.set(title=title, xlabel='Optimizer updates', ylabel='Mean relative L2 (%)')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
fig.suptitle('Matched cold starts: two fixed monitor inputs, native nonlocal reference')
fig.savefig(OUT / 'melting-operator-soap-convergence.png', dpi=180, bbox_inches='tight')
plt.close(fig)
report = dict(all_checks_pass=True, config_differences=config_differences,
    identical_initial_arrays=initial_arrays, identical_actual_online_conditions=unique,
    identical_actual_time_draws=5000*8, identical_source_files=len(sources),
    core_halo_parity='Same deterministic sub_sample(i,K) pool construction, coordinates, graph weights, source and cyclic pool order; raw subgraphs were not saved.',
    run_paths={name: str(run) for name, run in RUNS.items()}, conditions=conditions,
    performance=performance,
    scope='Eight fixed held-out inputs; final5000 checkpoint, 24^3 diagnostic volume/128^2 slice; native nonlocal, not local continuum or seed stability.')
(OUT / 'melting-operator-soap-comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps({name: dict(mean=conditions[name]['mean'], maximum=conditions[name]['maximum'],
                            performance=performance[name]) for name in RUNS}, indent=2))
