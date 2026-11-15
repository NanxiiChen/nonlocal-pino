"""Read-only common local-FD monitoring, separate from every training payload."""
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def make_monitor(c, run):
    from examples.electropolish2d.operator_pairing import validate_reference_identity
    from examples.electropolish2d.operator_training import predict_fields
    spec = c.operator.comparison_monitor
    if set(spec) != {'inputs_root', 'reference_root', 'indices', 'time_count'}:
        raise ValueError('Common monitor requires explicit inputs/reference paths, indices and time count')
    ids = np.asarray(spec.indices)
    if (ids.shape != (64,) or not np.issubdtype(ids.dtype, np.integer)
            or not np.array_equal(ids, np.arange(64))
            or spec.time_count != 21):
        raise ValueError('Common monitor requires the complete ordered64test indices and21uniform times')
    base, reference = Path(spec.inputs_root), Path(spec.reference_root)
    manifest = json.loads((base/'manifest.json').read_text())
    validate_reference_identity(json.loads(c.to_json()), json.loads(Path(manifest['source_config']).read_text()))
    if manifest['count'] != 64:
        raise ValueError('Expected established64input cohort')
    with np.load(base/'prediction.npz', allow_pickle=False) as data:
        xx, yy, cached_times = data['X'], data['Y'], data['time']
        images, initial = data['images'][ids], data['initial'][ids]
    # Cached neural coordinates are FP32; select reference snapshots using the
    # declared FP64 grid, never searchsorted on rounded-up neural coordinates.
    times = np.linspace(0, 1, 21)
    np.testing.assert_allclose(cached_times, times, atol=np.finfo(np.float32).eps, rtol=0)
    targets, paths = [], []
    for i in ids:
        path = reference/f'condition_{i:03d}/local101.npz'
        with np.load(path, allow_pickle=False) as data:
            take = np.searchsorted(data['time'], times)
            np.testing.assert_allclose(data['time'][take], times, rtol=0, atol=1e-15)
            np.testing.assert_array_equal(xx, data['X']); np.testing.assert_array_equal(yy, data['Y'])
            targets.append(data['fields'][take])
        paths.append(str(path))
    if not np.isfinite(targets).all():
        raise ValueError('Nonfinite common monitor reference')
    targets = np.asarray(targets, dtype=np.float64)
    x = jnp.asarray(np.column_stack((xx.ravel(), yy.ravel())), dtype=jnp.float32)
    ts = jnp.asarray(times, dtype=jnp.float32)
    predict = eqx.filter_jit(lambda model, image, known: predict_fields(model, image, x, ts, known, c))
    run = Path(run)
    provenance = dict(indices=ids.tolist(), split='test', count=len(ids), source_config=manifest['source_config'],
        inputs=str(base/'prediction.npz'), references=paths, times=times.tolist(),
        metric='Per-channel unweighted complete space-time relative L2, then arithmetic sample mean',
        scope='Complete64local-FD test inputs at every evaluation; evaluation only, never training payload',
        expected_output='local_metrics.jsonl')
    (run/'local_monitor.json').write_text(json.dumps(provenance, indent=2)+'\n')

    def evaluate(net, step):
        start = time.perf_counter()
        records = []
        for j, i in enumerate(ids):
            p = np.asarray(predict(net, jnp.asarray(images[j]), jnp.asarray(initial[j])), dtype=np.float64)
            p = p.reshape(targets[j].shape)
            if not np.isfinite(p).all():
                raise FloatingPointError('Nonfinite local-FD monitor prediction')
            numerator = np.sum((p-targets[j])**2, axis=(0, 1, 2))
            denominator = np.sum(targets[j]**2, axis=(0, 1, 2))
            if np.any(denominator <= 0):
                raise ValueError('Positive reference channel norms required')
            errors = np.sqrt(numerator/denominator)
            records.append(dict(index=int(i), phi=float(errors[0]), c=float(errors[1])))
        row = dict(step=step, conditions=records,
            phi=float(np.mean([r['phi'] for r in records])), c=float(np.mean([r['c'] for r in records])),
            evaluation_seconds=time.perf_counter()-start)
        with (run/'local_metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print('LOCAL_MONITOR '+json.dumps(row), flush=True)
        return row

    return evaluate
