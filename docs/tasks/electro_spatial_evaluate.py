"""Evaluate fixed-budget operator controls against identical existing 64 targets."""
import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.electropolish2d.operator_training import _build_network, predict_fields
from examples.electropolish2d.operator_data import load_dataset


BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')


def relative(prediction, target):
    a, b = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    denominator = np.sum(b*b)
    if denominator <= 0:
        raise ValueError('Relative L2 reference must have nonzero norm')
    return float(np.sqrt(np.sum((a-b)**2)/denominator))


def metrics(prediction, target):
    output = {}
    for ch, name in enumerate(('phi', 'c')):
        p, r = prediction[..., ch], target[..., ch]
        output['pooled_'+name] = relative(p, r)
        output['terminal_'+name] = relative(p[-1], r[-1])
        output['by_time_'+name] = [relative(a, b) for a, b in zip(p, r, strict=True)]
    return output


def aggregate(records, group):
    chosen = [r[group] for r in records if group in r]
    return dict(count=len(chosen), statistics={name:dict(
        mean=float(np.mean([r[name] for r in chosen])),
        median=float(np.median([r[name] for r in chosen])),
        p90=float(np.quantile([r[name] for r in chosen], .9)),
        maximum=float(np.max([r[name] for r in chosen])))
        for name in ('pooled_phi', 'terminal_phi', 'pooled_c', 'terminal_c')})


def evaluate(run, step=None):
    suffix = '' if step is None else f'_step_{step:06d}'
    model_path = run / ('model_final.eqx' if step is None else f'model{suffix}.eqx')
    monitor_path = run / f'prediction{suffix}.npz'
    path = run / f'spatial_evaluation64{suffix}.json'
    if path.exists():
        raise FileExistsError(path)
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    summary = json.loads((run/'summary.json').read_text())
    if summary['iterations'] != c.train.iterations or summary['training_uses_positive_time_labels']:
        raise ValueError('Expected complete fixed-budget physics-only run')
    if step is not None and (not 0 < step < c.train.iterations or step != c.log.checkpoint_step):
        raise ValueError('Expected the explicitly declared intermediate checkpoint step')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    manifest = json.loads((BASE/'manifest.json').read_text())
    local_manifest = json.loads((LOCAL/'manifest.json').read_text())
    source = json.loads(Path(manifest['source_config']).read_text())
    for name in ('physical', 'nondim', 'surface', 'nl'):
        if json.loads(c[name].to_json()) != source[name]:
            raise ValueError('Common reference physical identity mismatch: '+name)
    if manifest['count'] != 64:
        raise ValueError('Expected64declared input identities')
    with np.load(BASE/'prediction.npz') as stored:
        images, initial = stored['images'], stored['initial']
        xx, yy = stored['X'], stored['Y']
    times = np.asarray(local_manifest['times'])
    regular = np.searchsorted(times, np.linspace(0, 1, 21))
    np.testing.assert_array_equal(times[regular], np.linspace(0, 1, 21))
    if (images.shape != (64, *tuple(c.operator.image_shape))
            or initial.shape != (64, c.sample.Nx*c.sample.Ny, 2)
            or xx.shape != (c.sample.Ny, c.sample.Nx)):
        raise ValueError('Cached common inputs have incompatible shapes')
    points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    x, ts = jnp.asarray(points), jnp.asarray(times, dtype=jnp.float32)
    template = _build_network(c.arch, c.operator.image_shape, jax.random.key(0))
    net = eqx.tree_deserialise_leaves(model_path, template)
    if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(eqx.filter(net, eqx.is_array))):
        raise FloatingPointError('Nonfinite checkpoint')
    predict = eqx.filter_jit(lambda model, image, known: predict_fields(model, image, x, ts, known, c))
    records, predictions, inference = [], [], []
    start = time.perf_counter()
    for i, identity in enumerate(manifest['surface_ids']):
        with np.load(BASE/f'condition_{i:03d}/reference.npz') as stored:
            native = np.stack([stored['phi'], stored['c']], -1)
            np.testing.assert_array_equal(stored['time'], times[regular])
            np.testing.assert_allclose(initial[i], native[0].reshape(-1, 2), rtol=0, atol=3e-7)
        with np.load(LOCAL/f'condition_{i:03d}/local101.npz') as stored:
            local = stored['fields']
            np.testing.assert_array_equal(stored['time'], times)
            np.testing.assert_allclose(local[0], native[0], rtol=0, atol=3e-7)
        tick = time.perf_counter()
        p = np.asarray(predict(net, jnp.asarray(images[i]), jnp.asarray(initial[i])))
        inference.append(time.perf_counter()-tick)
        p = p.reshape(len(times), *xx.shape, 2)
        if not np.isfinite(p).all():
            raise FloatingPointError(f'Nonfinite prediction for input{i}')
        item = dict(index=i, surface_id=identity, native64=metrics(p[regular], native),
            local101_64=metrics(p[regular], local[regular]))
        if i in local_manifest['refinement_indices']:
            for nx in (201, 401):
                with np.load(LOCAL/f'condition_{i:03d}/local{nx}.npz') as stored:
                    np.testing.assert_array_equal(stored['time'], times)
                    target = stored['fields'][regular, ::(nx-1)//100, ::(nx-1)//100]
                item[f'local{nx}_4'] = metrics(p[regular], target)
        records.append(item)
        predictions.append(p)
        print('EVAL', run.name, i, item['native64']['pooled_phi'], item['local101_64']['pooled_phi'], flush=True)
    data, metadata, _ = load_dataset(c)
    replay = []
    with np.load(monitor_path) as stored:
        for k, index in enumerate(metadata['eval_indices']):
            actual = np.asarray(predict(net, jnp.asarray(data['images'][index]),
                jnp.asarray(data['initial'][index]))).reshape(len(times), *xx.shape, 2)[regular]
            gap = float(np.max(abs(actual-stored['prediction'][k])))
            np.testing.assert_allclose(actual, stored['prediction'][k], rtol=0, atol=2e-5)
            replay.append(gap)
    result = dict(status='passed', run=str(run), model=str(model_path),
        iterations=summary['iterations'] if step is None else step,
        training_iterations=summary['iterations'],
        spatial_mode=c.sample.spatial.mode if 'spatial' in c.sample else 'legacy_fixed_pool',
        count=64, time=times.tolist(),
        regular_time_indices=regular.tolist(), input_cache=str(BASE/'prediction.npz'),
        native_reference=str(BASE), local_reference=str(LOCAL),
        groups={g:aggregate(records, g) for g in ('native64', 'local101_64', 'local201_4', 'local401_4')},
        records=records, monitor_replay_max_gap=max(replay),
        first_inference_seconds=inference[0], warm_inference_mean_seconds=float(np.mean(inference[1:])),
        evaluation_seconds=time.perf_counter()-start,
        scope='Same 64 development inputs with independent native/local FD targets; '
              'fixed-budget checkpoint, one initialization, no fresh statistical stability claim')
    np.savez_compressed(run / f'spatial_evaluation64{suffix}.npz', prediction=np.stack(predictions),
        time=times, X=xx, Y=yy, surface_ids=np.asarray(manifest['surface_ids']))
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('EVALUATION_COMPLETE', run, json.dumps(result['groups']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, help='Evaluate the declared intermediate checkpoint')
    args = parser.parse_args()
    evaluate(args.run, args.step)
