"""Evaluate existing paired checkpoints on common64case local references.

Statistics summarize casewise errors, never seed averages. Intermediate models
retain only three field samples; final particle fields retain all64cases. Existing
Electropolish64fields are reused by explicit paths without duplicate storage.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from ml_collections import ConfigDict

REPO = Path(__file__).resolve().parents[2]
BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def summarize(records):
    if len(records) != 64:
        raise ValueError('Require exactly64casewise error records')
    names = ('phi_pooled', 'phi_terminal', 'c_pooled', 'c_terminal')
    result = {}
    for name in names:
        values = np.asarray([row[name] for row in records], np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Finite nonnegative error values required')
        result[name] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)),
            minimum=float(values.min()), maximum=float(values.max()), median=float(np.median(values)),
            p90=float(np.quantile(values, .9)), count=64, ddof=1)
    return result


def relative(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite corresponding physical fields')
    den = np.sum(b*b)
    if den <= 0:
        raise ValueError('Reference norm must be positive')
    return float(np.sqrt(np.sum((a-b)**2)/den))


def electro_metrics(prediction, reference):
    return {name: relative(a, b) for channel, field in enumerate(('phi', 'c'))
            for name, a, b in ((field+'_pooled', prediction[..., channel], reference[..., channel]),
                               (field+'_terminal', prediction[-1, ..., channel], reference[-1, ..., channel]))}


def verify_sources(run, names):
    for name in names:
        if (run/'source'/name).read_bytes() != (REPO/name).read_bytes():
            raise ValueError('Current model/physics source differs from frozen checkpoint: '+name)


def checkpoint(run, step, final_step, case):
    if step == 0:
        path = run/'model_initial.eqx'
    elif step == final_step:
        path = run/'model_final.eqx'
    elif case == 'particle_corrosion2d':
        path = run/f'model_{step:06d}.eqx'
    else:
        path = run/f'model_step_{step:06d}.eqx'
    return path.resolve(strict=True)


def check_model(net):
    import equinox as eqx
    import jax
    if not all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(eqx.filter(net, eqx.is_array))):
        raise FloatingPointError('Checkpoint contains nonfinite weights')


def training_time(run, step):
    history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    matching = [row for row in history if row['step'] == step]
    if len(matching) != 1:
        raise ValueError('Checkpoint lacks one corresponding recorded training time')
    return matching[0]['seconds']


def save_step(output, run, case, step, final_step, records, paths, reference, seconds, extras):
    result = dict(status='passed', run=str(run), case=case, step=step,
        checkpoint=str(checkpoint(run, step, final_step, case)), count=64,
        reference=str(reference), training_elapsed_seconds=training_time(run, step),
        evaluation_seconds=seconds, records=records, statistics=summarize(records),
        saved_prediction_files=paths, training_performed=False,
        spread='Sample standard deviation across64casewise errors, ddof=1; not standard error or confidence interval',
        **extras)
    write(output/f'step_{step:06d}.json', result)
    print('EVALUATED', case, run.name, step, result['statistics']['phi_pooled'], flush=True)
    return result


def particle(run, reference, output, c):
    import equinox as eqx
    import jax
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.evaluate import load_references, make_evaluator, metrics
    verify_sources(run, ('examples/particle_corrosion2d/network.py', 'examples/particle_corrosion2d/problem.py',
        'src/nonlocal_pinn/nn/point_conditional.py', 'src/nonlocal_pinn/nn/conditional.py'))
    if read(reference/'validation.json')['status'] != 'passed':
        raise ValueError('New local reference must be qualified before inference')
    refs, manifest = load_references(reference, c)
    if len(refs) != 64 or manifest['backend'] != 'local_p1_lumped_fem':
        raise ValueError('Expected64common local-FEM geometries')
    template = build_network(c, jax.random.PRNGKey(0))
    evaluate = make_evaluator(c, refs, chunk_size=512)
    old_reference = Path(read(run/'runtime.json')['reference'])
    old_refs, _ = load_references(old_reference, c)
    results = []
    for step in (0, 5000, 10000, 15000, 20000):
        tick = time.perf_counter()
        net = eqx.tree_deserialise_leaves(checkpoint(run, step, 20000, 'particle_corrosion2d'), template)
        check_model(net)
        predictions, summary = evaluate(net)
        records, paths = [], {}
        folder = output/f'fields_{step:06d}'
        folder.mkdir()
        for i, (p, ref) in enumerate(zip(predictions, refs, strict=True)):
            # Recompute from physical fields separately from evaluator's summary.
            score = metrics(p, ref['fields'], ref['volumes'], ref['times'])
            for key, value in score.items():
                np.testing.assert_allclose(value, summary['conditions'][i][key], rtol=1e-12, atol=1e-14)
            records.append(dict(index=i, **score))
            if step == 20000 or i < 3:
                path = folder/f'prediction_{i:03d}.npz'
                np.savez_compressed(path, prediction=p, points=ref['points'], times=ref['times'])
                paths[str(i)] = str(path)
        replay = None
        if step == 20000:
            old_predictions, _ = make_evaluator(c, old_refs[:1], chunk_size=512)(net)
            saved = arrays(run/'prediction_000.npz')['prediction']
            np.testing.assert_allclose(old_predictions[0], saved, rtol=0, atol=2e-5)
            replay = float(np.max(np.abs(old_predictions[0]-saved)))
        results.append(save_step(output, run, 'particle_corrosion2d', step, 20000, records, paths,
            reference, time.perf_counter()-tick,
            dict(old_saved_field_replay_max_gap=replay, inference_chunk=512,
                 norm='Physical lumped spatial volumes; trapezoidal weights over11saved times including0')))
    return results


def electro(run, output, c):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from examples.electropolish2d.operator_training import _build_network, predict_fields
    from docs.tasks.electro_spatial_evaluate import validate_reference_identity
    verify_sources(run, ('examples/electropolish2d/operator_training.py', 'src/nonlocal_pinn/nn/conditional.py'))
    manifest = read(BASE/'manifest.json')
    validate_reference_identity(c.to_dict(), read(manifest['source_config']))
    refs = [arrays(LOCAL/f'condition_{i:03d}/local101.npz') for i in range(64)]
    cached = arrays(BASE/'prediction.npz')
    times = refs[0]['time']
    regular = np.searchsorted(times, np.linspace(0, 1, 21))
    np.testing.assert_allclose(times[regular], np.linspace(0, 1, 21), rtol=0, atol=1e-15)
    xx, yy = cached['X'], cached['Y']
    for ref in refs:
        for name, value in (('X', xx), ('Y', yy), ('time', times)):
            np.testing.assert_array_equal(ref[name], value)
    x = jnp.asarray(np.column_stack((xx.ravel(), yy.ravel())), dtype=jnp.float32)
    ts = jnp.asarray(times, dtype=jnp.float32)
    template = _build_network(c.arch, c.operator.image_shape, jax.random.key(0))
    infer = eqx.filter_jit(lambda model, image, known: predict_fields(model, image, x, ts, known, c))
    results = []
    for step in (0, 5000, 10000):
        tick = time.perf_counter()
        net = eqx.tree_deserialise_leaves(checkpoint(run, step, 10000, 'electropolish2d'), template)
        check_model(net)
        reused_file = None
        if step:
            suffix = '_step_005000' if step == 5000 else ''
            reused_file = run/f'spatial_evaluation64{suffix}.npz'
            stored = arrays(reused_file)
            previous = read(run/f'spatial_evaluation64{suffix}.json')
            if previous['status'] != 'passed' or previous['count'] != 64 or previous['iterations'] != step:
                raise ValueError('Cached evaluation checkpoint/cohort mismatch')
            for name, value in (('X', xx), ('Y', yy), ('time', times)):
                np.testing.assert_array_equal(stored[name], value)
            np.testing.assert_array_equal(stored['surface_ids'], manifest['surface_ids'])
        folder = output/f'fields_{step:06d}'
        folder.mkdir()
        records, paths = [], {}
        replay = []
        for i, ref in enumerate(refs):
            if not step or i == 0:
                p = np.asarray(jax.block_until_ready(infer(net, jnp.asarray(cached['images'][i]),
                    jnp.asarray(cached['initial'][i])))).reshape(len(times), *xx.shape, 2)
                if step:
                    np.testing.assert_allclose(p, stored['prediction'][i], rtol=0, atol=2e-5)
                    replay.append(float(np.max(np.abs(p-stored['prediction'][i]))))
            if step:
                p = stored['prediction'][i]
            score = electro_metrics(p[regular], ref['fields'][regular])
            if step:
                for key, old_key in (('phi_pooled', 'pooled_phi'), ('phi_terminal', 'terminal_phi'),
                                     ('c_pooled', 'pooled_c'), ('c_terminal', 'terminal_c')):
                    np.testing.assert_allclose(score[key], previous['records'][i]['local101_64'][old_key],
                                               rtol=1e-12, atol=1e-14)
            records.append(dict(index=i, surface_id=int(manifest['surface_ids'][i]), **score))
            if i < 3 and not step:
                path = folder/f'prediction_{i:03d}.npz'
                np.savez_compressed(path, prediction=p, X=xx, Y=yy, time=times)
                paths[str(i)] = str(path)
        results.append(save_step(output, run, 'electropolish2d', step, 10000, records, paths,
            LOCAL, time.perf_counter()-tick,
            dict(reused_prediction_file=None if reused_file is None else str(reused_file),
                 reloaded_checkpoint_replay_gaps=replay, regular_time_indices=regular.tolist(),
                 norm='Equal weights on101x51nodes and21uniform saved times including0; same existing64localFD cases')))
    return results


def run(case, run_path, reference, output):
    import jax
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    c = ConfigDict(read(run_path/'config.json'))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', c.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    if c.train.compute_dtype != 'float32':
        raise ValueError('This saved paired study uses FP32')
    if case == 'electropolish2d' and reference != LOCAL.resolve(strict=True):
        raise ValueError('Cached Electropolish inference requires its declared local64 reference directory')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'config.json', c.to_dict())
    started = time.perf_counter()
    results = particle(run_path, reference, output, c) if case == 'particle_corrosion2d' else electro(run_path, output, c)
    write(output/'summary.json', dict(status='passed', case=case, run=str(run_path),
        reference=str(reference), count=64, checkpoint_steps=[r['step'] for r in results],
        checkpoints=[str(output/f'step_{r["step"]:06d}.json') for r in results],
        final=results[-1], seconds=time.perf_counter()-started, training_performed=False))
    print('ARM_COMPLETE', case, output, time.perf_counter()-started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=('particle_corrosion2d', 'electropolish2d'))
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.case, args.run.resolve(strict=True), args.reference.resolve(strict=True), args.output.resolve())
