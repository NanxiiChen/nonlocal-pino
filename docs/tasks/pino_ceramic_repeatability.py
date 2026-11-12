"""CPU provenance and lowered-program audit for the same-seed ceramic controls.

No training, reference solve or default mutation. GPU execution nondeterminism
requires a separate same-executable replay; identical lowering alone is not proof.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def events(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        del row['wait_seconds']
        for record in row['records']:
            del record['worker_seconds']
            del record['cache_hit']
        rows.append(row)
    return rows


def audit(old, new, output):
    output.mkdir(parents=True, exist_ok=False)
    a, b = read(old/'config.json'), read(new/'config.json')
    marker = a['operator'].pop('method_comparison')
    assert marker == 'rolling_spatial_v1' and a == b, 'Unexpected numerical configuration difference'
    cold = {}
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        cold[name] = (old/name).read_bytes() == (new/name).read_bytes()
        assert cold[name], name
    arrays = {}
    for filename in ('sampling.npz', 'online_initial_inputs.npz', 'online_sampling.npz', 'online_final_inputs.npz'):
        with np.load(old/filename) as first, np.load(new/filename) as second:
            assert set(first.files) == set(second.files), filename
            for name in first.files:
                x, y = first[name], second[name]
                assert x.dtype == y.dtype and np.array_equal(x, y), (filename, name)
                arrays[filename+'/'+name] = dict(shape=list(x.shape), dtype=str(x.dtype), equal=True)
    old_events, new_events = events(old/'online_refresh.jsonl'), events(new/'online_refresh.jsonl')
    assert old_events == new_events, 'Physical geometry/spatial event mismatch'
    assert old_events[0]['completed_updates'] == 50
    runtime = read(new/'runtime.json')
    manifest = read(Path(runtime['inputs'])/'manifest.json')
    with np.load(new/'sampling.npz') as data:
        ids, pools = data['condition_ids'], data['pool_ids']
        times = data['times'][0]
    with np.load(new/'online_initial_inputs.npz') as data:
        batch = {k: data[k][ids[0]] for k in manifest['fixed_keys']}
        batch.update({k: data[k][ids[0], pools[0]] for k in manifest['graph_keys']})
        with np.load(old/'core_trace.npz') as trace:
            cores = trace['cores'][:50]
        for step in range(50):
            points = data['graph_points'][ids[step], pools[step]]
            indices = data['core_indices'][ids[step], pools[step]]
            xy = np.take_along_axis(points, indices[..., None], axis=1)
            np.testing.assert_array_equal(xy, cores[step, ..., :2])
            np.testing.assert_array_equal(data['pde_weights'][ids[step], pools[step]], cores[step, ..., 2])
    np.savez_compressed(output/'first_batch.npz', times=times, **batch)
    sources = {}
    for name in ('network.py', 'losses.py', 'online.py', 'geometry.py', 'mesh.py', 'operators.py'):
        source = 'examples/particle_corrosion2d/'+name
        sources[source] = (old/'source'/source).read_bytes() == (new/'source'/source).read_bytes()
        assert sources[source], source
    for source in ('src/nonlocal_pinn/nn/point_conditional.py', 'src/nonlocal_pinn/nn/conditional.py',
                   'src/nonlocal_pinn/training/optimizers.py', 'src/nonlocal_pinn/training/runtime.py'):
        sources[source] = (old/'source'/source).read_bytes() == (new/'source'/source).read_bytes()
        assert sources[source], source
    first_loss = [json.loads((p/'loss.jsonl').read_text().splitlines()[0]) for p in (old, new)]
    report = dict(status='passed', old=str(old), new=str(new), cold_bytes=cold,
        equal_arrays=arrays, equal_physical_event_count=len(old_events),
        actual_core_steps_replayed=50, identical_math_sources=sources,
        first_loss_parts=[r['parts'] for r in first_loss], first_loss_weights=[r['weights'] for r in first_loss],
        maximum_first_loss_difference=float(np.max(abs(np.asarray(first_loss[0]['parts'])-first_loss[1]['parts']))),
        scope='Initial/final banks, all draw IDs/times and physical refresh journals; actual cores replayed for first50steps',
        limitation='New run has no actual-core trace at intermediate refreshes; journal identity is not direct array observation')
    (output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return batch, times, b


def lower(new, output, batch, times, configuration):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.losses import make_loss_vector, make_step, LOSS_NAMES
    from nonlocal_pinn.training.optimizers import make_optimizer
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('This lowering audit requires JAX_PLATFORMS=cpu')
    c = ConfigDict(configuration)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    net = build_network(c, jax.random.PRNGKey(read(new/'runtime.json')['seed']))
    net = eqx.tree_deserialise_leaves(new/'model_initial.eqx', net)
    optimizer = make_optimizer(c)
    state = eqx.tree_deserialise_leaves(new/'optimizer_initial.eqx', optimizer.init(eqx.filter(net, eqx.is_array)))
    step = make_step(make_loss_vector(c), optimizer, c, LOSS_NAMES)
    plain = {k:jnp.asarray(v) for k,v in batch.items()}
    marked = dict(plain, pde_core_points=jax.vmap(lambda points, ids:points[ids])(
        plain['graph_points'], plain['core_indices']))
    args = (net, state, jnp.ones(len(LOSS_NAMES), dtype=jnp.float32), jnp.asarray(times))
    lowered = [step.lower(*args, payload, jnp.asarray(True)) for payload in (plain, marked)]
    texts = [value.as_text() for value in lowered]
    for label, text in zip(('plain', 'marked'), texts, strict=True):
        (output/(label+'.mlir')).write_text(text)
    report = dict(status='passed', stablehlo_identical=texts[0] == texts[1],
        stablehlo_bytes=[len(text.encode()) for text in texts],
        scatter_count=[text.count('stablehlo.scatter') for text in texts],
        scope='Canonical full-shape first optimizer update; added comparison core coordinates are never used by nonlocal loss',
        limitation='CPU lowering only; not a GPU executable/repeatability or accuracy certificate')
    (output/'lowering.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    batch, times, config = audit(args.old, args.new, args.output)
    lower(args.new, args.output, batch, times, config)


if __name__ == '__main__':
    main()
